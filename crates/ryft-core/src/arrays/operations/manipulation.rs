//! Array IR instantiations of the shape-manipulation operation family contracts.
//!
//! Shape manipulation is where first-class runtime dimensions do their most visible work: broadcasting an array to a
//! dynamic output shape consumes one first-class dimension operand per output axis. This module supplies the array
//! universe's answers to those contracts, together with the reference backend's element conversion contracts, which
//! pair each source element category with the destination's exact conversion category.

// TODO(eaplatanios): Review this.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::arrays::addressing::{ArrayAddressing, ArraySliceAxis};
use crate::arrays::arrays::Array;
use crate::arrays::encoding::{ArrayElement, i1, i2, i4, u1, u2, u4};
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::math::{ElementAdd, ElementExtremum, ElementMul};
use crate::arrays::sharding::shardings::Sharding;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, Shape, StaticShape};
use crate::axes::Axis;
use crate::contexts::EagerContext;
use crate::operations::manipulation::broadcasting::infer_explicit_broadcast_output_type;
use crate::operations::{
    Broadcast, Concatenate, ConcatenateOperation, ConvertElementType, DimensionSize, DynamicBroadcast,
    DynamicBroadcastOperation, DynamicSlice, DynamicUpdateSlice, Gather, GatherOperation, GatherScatterMode, Pad,
    Permutation, Reshape, ReshapeParameters, Scatter, ScatterOperation, ScatterReductionKind, Slice, Transpose,
    UpdateSlice, Zero,
};
use crate::programs::{ProgramError, TypeError, Typed, Value, ValueProjection};

impl<A: Value<Type = ArrayType> + DimensionSize<usize> + Broadcast> DynamicBroadcast for ArrayIrValue<A> {
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| result.map(|dimension| Dimension::Static(dimension.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let operation = DynamicBroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, &operation)?;
        Ok(Self::Array(input.broadcast(output_type, output_axes)?))
    }
}

impl Transpose for Array {
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate the permutation and compute the output type (including sharding) via the type-level rule, so an
        // out-of-range or duplicated axis is a clean error rather than an out-of-bounds panic.
        let permutation = permutation.into();
        let output_type = self.r#type().transpose(permutation.clone())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let rank = self.r#type().rank();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }
        let mut output_index = vec![0usize; rank];
        let mut input_index = vec![0usize; rank];
        for output_flat in 0..output_addressing.element_count() {
            for (position, &input_axis) in permutation.iter().enumerate() {
                input_index[input_axis] = output_index[position];
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            output_addressing.advance_index(&mut output_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl Reshape for Array {
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError> {
        // Delegate to the type-level reshape so all element-count and sharding validation remains shared with staged
        // execution.
        let parameters = parameters.into();
        let output_type = self.r#type().reshape(parameters.clone())?;
        let transposed = parameters.dimensions().map(|dimensions| self.transpose(dimensions)).transpose()?;
        let input = transposed.as_ref().unwrap_or(self);
        let input_addressing = ArrayAddressing::new(input.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if input_addressing.is_dense_row_major() && output_addressing.is_dense_row_major() {
            bytes.copy_from_slice(input.storage_bytes());
        } else {
            for index in 0..input_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(index)]
                    .copy_from_slice(&input.storage_bytes()[input_addressing.byte_range_for_flat_index(index)]);
            }
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl Broadcast for Array {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = self.r#type().broadcast(output_type, output_axes)?;
        let Some(target_shape) = r#type.static_shape() else {
            return Err(
                TypeError::invalid(format!("cannot materialize a value of dynamically sized type {}", r#type)).into()
            );
        };
        if &r#type == self.r#type().as_ref() && output_axes.iter().copied().eq(0..r#type.rank()) {
            return Ok(self.clone());
        }
        let input_shape = self.r#type().static_shape().unwrap();
        let input_rank = input_shape.rank();
        let target_rank = target_shape.rank();
        let output_count = Self::materialized_element_count(&r#type)?;
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(r#type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_count == 0 {
            return Ok(Self::new_unchecked(r#type, Arc::new(bytes)));
        }
        let mut target_index = vec![0usize; target_rank];
        let mut input_index = vec![0usize; input_rank];
        for output_flat in 0..output_count {
            for input_axis in 0..input_rank {
                let target_axis = output_axes[input_axis];
                input_index[input_axis] = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            output_addressing.advance_index(&mut target_index);
        }
        Ok(Self::new_unchecked(r#type, Arc::new(bytes)))
    }
}

impl Array {
    /// Copies the logical block selected by `axes` into a new array of `output_type`. The caller guarantees that the
    /// selection lies in bounds and contains exactly the output's logical element count.
    fn copy_block(&self, output_type: ArrayType, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let ranges = input_addressing.ranges(axes)?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        debug_assert_eq!(ranges.element_count(), output_addressing.element_count());
        let element_byte_width = input_addressing.element_byte_width();
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let output_is_dense = output_addressing.is_dense_row_major();
        let mut output_index = 0usize;
        for range in ranges {
            let input_bytes = range.bytes();
            let element_count = range.elements().len();
            if output_is_dense {
                let output_start = output_index * element_byte_width;
                bytes[output_start..output_start + input_bytes.len()]
                    .copy_from_slice(&self.storage_bytes()[input_bytes]);
                output_index += element_count;
                continue;
            }
            for offset in 0..element_count {
                let input_start = input_bytes.start + offset * element_byte_width;
                bytes[output_addressing.byte_range_for_flat_index(output_index)]
                    .copy_from_slice(&self.storage_bytes()[input_start..input_start + element_byte_width]);
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, output_addressing.element_count());
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Overwrites the logical block of `update`'s shape starting at `start_indices` in this array with `update`. The
    /// caller guarantees that the block lies in bounds.
    fn replace_block(self, update: &Array, start_indices: &[usize]) -> Self {
        let update_shape = update.r#type().static_shape().unwrap();
        let addressing = ArrayAddressing::new(self.r#type().into_owned()).unwrap();
        let update_addressing = ArrayAddressing::new(update.r#type().into_owned()).unwrap();
        let axes = start_indices
            .iter()
            .zip(update_shape.dimensions())
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        let ranges = addressing.ranges(&axes).unwrap();
        let element_byte_width = addressing.element_byte_width();
        let mut output = self;
        let bytes = output.storage_bytes_mut();
        let update_is_dense = update_addressing.is_dense_row_major();
        let mut written = 0usize;
        for range in ranges {
            let output_bytes = range.bytes();
            let element_count = range.elements().len();
            if update_is_dense {
                let update_start = written * element_byte_width;
                bytes[output_bytes].copy_from_slice(
                    &update.storage_bytes()[update_start..update_start + element_count * element_byte_width],
                );
                written += element_count;
                continue;
            }
            for offset in 0..element_count {
                let output_start = output_bytes.start + offset * element_byte_width;
                bytes[output_start..output_start + element_byte_width]
                    .copy_from_slice(&update.storage_bytes()[update_addressing.byte_range_for_flat_index(written)]);
                written += 1;
            }
        }
        debug_assert_eq!(written, update_addressing.element_count());
        output
    }

    /// Extracts the in-band scalar start indices of a dynamic slicing operation and clamps them per StableHLO
    /// semantics: the effective start index along axis `d` is
    /// `clamp(0, start_indices[d], input_dimension[d] - block_sizes[d])`.
    fn clamped_start_indices(start_indices: &[Array], input_shape: &StaticShape, block_sizes: &[usize]) -> Vec<usize> {
        start_indices
            .iter()
            .enumerate()
            .map(|(axis, index)| {
                let addressing = ArrayAddressing::new(index.r#type().into_owned()).unwrap();
                let raw = index.index_value(&addressing, &[]);
                let maximum = (input_shape[axis] - block_sizes[axis]) as i64;
                raw.clamp(0, maximum) as usize
            })
            .collect()
    }

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
            data_type => unreachable!("cannot use an array of element data type {data_type} as indices"),
        }
    }
}

impl Pad for Array {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type = self.r#type().pad(
            padding_value.r#type().as_ref(),
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?;
        let input_shape = self.r#type().static_shape().unwrap();
        if edge_padding_low.iter().all(|padding| *padding == 0)
            && edge_padding_high.iter().all(|padding| *padding == 0)
            && input_shape
                .dimensions()
                .iter()
                .zip(interior_padding)
                .all(|(size, padding)| *padding == 0 || *size <= 1)
        {
            return Ok(self.clone());
        }
        let output_shape = output_type.static_shape().unwrap();
        let rank = input_shape.rank();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let padding_addressing = ArrayAddressing::new(padding_value.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let padding_bytes = &padding_value.storage_bytes()[padding_addressing.byte_range_for_flat_index(0)];
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if output_addressing.is_dense_row_major() && output_addressing.element_byte_width() != 0 {
            for output_bytes in bytes.chunks_exact_mut(output_addressing.element_byte_width()) {
                output_bytes.copy_from_slice(padding_bytes);
            }
        } else {
            for output_index in 0..output_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(output_index)].copy_from_slice(padding_bytes);
            }
        }
        if input_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }
        let mut input_index = vec![0usize; rank];
        let mut output_index = vec![0usize; rank];
        let mut written = 0usize;
        'elements: while written < input_addressing.element_count() {
            for axis in 0..rank {
                let input_coordinate = i128::try_from(input_index[axis])
                    .map_err(|_| TypeError::invalid(format!("'pad' input index is too large on axis {axis}")))?;
                let stride = i128::try_from(interior_padding[axis])
                    .ok()
                    .and_then(|padding| padding.checked_add(1))
                    .ok_or_else(|| TypeError::invalid(format!("'pad' stride is too large on axis {axis}")))?;
                let output_coordinate =
                    i128::from(edge_padding_low[axis])
                        .checked_add(input_coordinate.checked_mul(stride).ok_or_else(|| {
                            TypeError::invalid(format!("'pad' output index overflows on axis {axis}"))
                        })?)
                        .ok_or_else(|| TypeError::invalid(format!("'pad' output index overflows on axis {axis}")))?;
                let output_extent = i128::try_from(output_shape[axis])
                    .map_err(|_| TypeError::invalid(format!("'pad' output extent is too large on axis {axis}")))?;
                if output_coordinate < 0 || output_coordinate >= output_extent {
                    written += 1;
                    input_addressing.advance_index(&mut input_index);
                    continue 'elements;
                }
                output_index[axis] = usize::try_from(output_coordinate)
                    .map_err(|_| TypeError::invalid(format!("'pad' output index is too large on axis {axis}")))?;
            }
            bytes[output_addressing.byte_range_unchecked(&output_index)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            written += 1;
            input_addressing.advance_index(&mut input_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl Concatenate for Array {
    fn concatenate<'i, I: IntoIterator<Item = &'i Self>, A: Into<Axis>>(
        inputs: I,
        axis: A,
    ) -> Result<Self, ProgramError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(
                TypeError::invalid("'concatenate' expects at least one operand but got none".to_string()).into()
            );
        };
        if inputs.len() == 1 {
            return Ok((*first).clone());
        }
        let operation = ConcatenateOperation::new(axis, first.r#type().rank())?;
        let axis = operation.axis();
        let input_types = inputs.iter().map(|input| input.r#type()).collect::<Vec<_>>();
        let output_type = ArrayType::concatenate(input_types.iter().map(|r#type| r#type.as_ref()), axis)?;
        // Each operand owns a contiguous run of `axis` coordinates. Write its logical block at the running offset
        // along `axis` and offset zero on every other axis.
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        // Zero-initialization establishes every layout hole and tile-padding byte. Every logical element is replaced
        // below, so this does not require the element data type itself to represent zero.
        let mut output =
            Self::new_unchecked(output_type.clone(), Arc::new(vec![0; output_addressing.storage_byte_len()]));
        let mut offset = 0usize;
        for input in inputs {
            let input_axis_size = input.r#type().static_shape().unwrap()[axis];
            let mut start_indices = vec![0usize; output_type.rank()];
            start_indices[axis] = offset;
            output = output.replace_block(input, start_indices.as_slice());
            offset += input_axis_size;
        }
        Ok(output)
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
                            <Element as ElementAdd>::add(current_value, update_value)?
                        } else {
                            <Element as ElementMul>::mul(current_value, update_value)?
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
                            <Element as ElementExtremum>::minimum(current_value, update_value)
                        } else {
                            <Element as ElementExtremum>::maximum(current_value, update_value)
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

impl Slice for Array {
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().slice(start_indices, limit_indices, strides)?;
        let axes = start_indices
            .iter()
            .zip(limit_indices.iter())
            .zip(strides.iter())
            .map(|((start, limit), stride)| ArraySliceAxis::new(*start, (limit - start).div_ceil(*stride), *stride))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
    }
}

impl UpdateSlice for Array {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        self.r#type().update_slice(update.r#type().as_ref(), start_indices)?;
        Ok(self.clone().replace_block(update, start_indices))
    }
}

impl DynamicSlice for Array {
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type().into_owned()).collect();
        let output_type = self.r#type().dynamic_slice(&index_types, sizes)?;
        let input_shape = self.r#type().static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, sizes);
        let axes = starts
            .iter()
            .zip(sizes)
            .map(|(start, size)| ArraySliceAxis::new(*start, *size, 1))
            .collect::<Vec<_>>();
        self.copy_block(output_type, &axes)
    }
}

impl DynamicUpdateSlice for Array {
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type().into_owned()).collect();
        self.r#type().dynamic_update_slice(update.r#type().as_ref(), &index_types)?;
        let input_shape = self.r#type().static_shape().unwrap();
        let update_shape = update.r#type().static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, update_shape.dimensions());
        Ok(self.clone().replace_block(update, starts.as_slice()))
    }
}

impl ConvertElementType for Array {
    /// Refer to the documentation of [`Array::converted_to`] for the conversion semantics this delegates to.
    #[inline]
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        self.converted_to(data_type)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::{Array, array_type};
    use crate::arrays::batching::{ArrayIrBatch, ArrayIrBatching};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::encoding::i4;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::ShardingDimension;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, DifferentiationError};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::{
        CONCATENATE_OPERATION_NAME, ConcatenateOperation, DimensionAddOperation, DimensionMulOperation,
        DimensionSizeOperation, DynamicBroadcast, DynamicBroadcastOperation, DynamicReshapeOperation,
        DynamicShapeSliceOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, GatherDimensionNumbers,
        GatherOperation, IotaOperation, PadOperation, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind,
        SliceOperation, UpdateSliceOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_array_ir_reshape_partial_evaluation() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicReshapeOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input)),
                        (@known, first_extent),
                        (@known, second_extent),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let identity_input_type = identity_input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicReshapeOperation::new(),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input_type, replay = identity_input.clone())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_array_ir_reshape_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, first_extent, second_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0])),
            ]),
        );

        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,]))]),
        );

        // The inverse cannot recover `n` from the `[2, 2*n]` output shape without division. The reshape JVP must
        // therefore retain the original source extent as an explicit residual while it still has the source array.
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let source_extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().into_owned();
        let two = builder.add_constant(ArrayIrValue::Dimension(two_value));
        let source_type = DimensionType::new(input_type.shape().dimensions()[0].variable().unwrap().clone());
        let doubled_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&source_type, &two_type).unwrap()),
                Vec::new(),
                vec![source_extent, two],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, two, doubled_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
        for size in [0, 1, 3, 8] {
            let element_count = size * 4;
            let primal_values = (0..element_count).map(|value| value as f64).collect::<Vec<_>>();
            let tangent_values = (element_count..2 * element_count).map(|value| value as f64).collect::<Vec<_>>();
            assert_eq!(
                jvp.interpret(vec![
                    ArrayIrValue::Array(Array::matrix(size, 4, primal_values.clone())),
                    ArrayIrValue::Array(Array::matrix(size, 4, tangent_values.clone())),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, primal_values)),
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, tangent_values)),
                ]),
            );
        }
        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert_eq!(
            rendered_primal,
            "
lambda %0:f64[source, 4] .
let %1:dimension<2> = const
    %2:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
    %3:dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
    %4:f64[2, source * 2] = reshape %0 %1 %3
    %5:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
in (%4, %3, %5)
            "
            .trim(),
        );
        assert_eq!(
            rendered_tangent,
            "
lambda %0:f64[source, 4], %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)> .
let %3:dimension<2> = const
    %4:f64[2, source * 2] = linear_call [residual_count=3] %3 %1 %2 %0 [
        forward={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
%3:f64[source, 4] .
            let %4:f64[2, source * 2] = reshape %3 %0 %1
            in (%4)
        },
        transpose={
            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, \
%2:dimension<source ∈ [0, 9)>, %3:f64[2, source * 2] .
            let %4:dimension<4> = constant [value=4]
                %5:f64[source, 4] = reshape %3 %2 %4
            in (%5)
        },
    ]
in (%4)
            "
            .trim(),
        );
        assert_eq!(linearization.tangent().input_types()[0], ArrayIrType::Array(input_type.tangent()));
        assert!(
            linearization
                .tangent()
                .input_types()
                .iter()
                .skip(1)
                .all(|r#type| matches!(r#type, ArrayIrType::Dimension(_)))
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(residuals.len(), linearization.residual_count());
        assert_eq!(residuals.len(), 2);

        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs.clone()),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 6, tangent_values))]),
        );

        // The executable linear boundary remains structural when imported, including both attached regions and every
        // residual edge. Nested forward differentiation likewise treats only the array input as differentiable.
        let mut imported_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_inputs = linearization
            .tangent()
            .input_types()
            .into_iter()
            .map(|r#type| imported_builder.add_input(r#type))
            .collect::<Vec<_>>();
        let imported_outputs =
            imported_builder.splice_program(linearization.tangent(), imported_inputs.as_slice()).unwrap();
        let imported = imported_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                imported_outputs,
                vec![Placeholder; imported_inputs.len()],
                vec![Placeholder],
            )
            .unwrap();
        let [imported_call] = imported.instructions() else {
            panic!("expected one imported linear call");
        };
        assert!(matches!(imported_call.operation(), ArrayIrOperation::LinearCall(_)));
        assert_eq!(imported_call.regions().len(), 2);
        assert_eq!(imported.interpret(tangent_inputs.clone()), linearization.tangent().interpret(tangent_inputs));

        let nested_jvp = linearization.tangent().jvp().unwrap();
        let mut nested_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect()))];
        nested_inputs.extend(residuals.clone());
        nested_inputs.push(ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect())));
        assert_eq!(nested_jvp.input_ids().len(), 2 + residuals.len());
        assert_eq!(
            nested_jvp.interpret(nested_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 6, (12..24).map(|value| value as f64).collect(),)),
                ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect(),)),
            ]),
        );

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect(),))]),
        );

        // A matching explicit output-extent operand is already the authoritative SSA value for the source axis, so
        // the residual path reuses it and does not read the source array again.
        let source = DimensionVariable::new("reused_source", DimensionBounds::new(1, Some(9)).unwrap());
        let source_type = DimensionType::new(source.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)])).into(),
        );
        let source_extent = builder.add_input(source_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, source_extent, four])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(!linearization.primal().to_string().contains("dimension_size"));
        assert!(!linearization.tangent().to_string().contains("dimension_size"));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_deduplicates_repeated_permuted_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Dynamic(extent)]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                DynamicReshapeOperation::new().with_dimensions([1, 0]),
                Vec::new(),
                vec![input, extent, extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // Both output axes and both inverse axes use the same SSA extent. Partial evaluation carries it once even
        // though the linear call consumes it in multiple operand positions.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(linearization.primal().to_string().matches("dimension_size").count(), 1);
        let input = ArrayIrValue::Array(Array::matrix(3, 3, (0..9).map(|value| value as f64).collect()));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::matrix(3, 3, (9..18).map(|value| value as f64).collect()));
        let mut tangent_inputs = vec![tangent];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(
                3,
                3,
                vec![9.0, 12.0, 15.0, 10.0, 13.0, 16.0, 11.0, 14.0, 17.0],
            ))]),
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 3, (18..27).map(|value| value as f64).collect()))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(
                3,
                3,
                vec![18.0, 21.0, 24.0, 19.0, 22.0, 25.0, 20.0, 23.0, 26.0],
            ))]),
        );

        // The same compiled programs accept the lower-bound zero without inventing an extent tangent input.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()))]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_preserves_sharding_through_the_inverse() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(DimensionType::new(extent).into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(
                DynamicReshapeOperation::new().with_output_sharding(sharding),
                Vec::new(),
                vec![input, extent, four],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.tangent().output_types(), vec![input_type.tangent().into()]);
        assert_eq!(linearization.pullback().unwrap().output_types(), vec![input_type.cotangent().into()]);
    }

    #[test]
    fn test_array_ir_pad_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program
                .transpose_with_respect_to(&[0, 1])
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayIrValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(11)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
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
        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice [strides=[2]]"));

        let input = ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0]));
        let padding_value = ArrayIrValue::Array(Array::scalar(-1.0_f64));
        let output_extent = ArrayIrValue::Dimension(DimensionValue::new(result_type.clone(), 8).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, padding_value, output_extent]).unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(Array::vector(vec![-1.0_f64, 10.0, -1.0, 20.0, -1.0, 30.0, -1.0, -1.0])),
        );
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
            ArrayIrValue::Array(Array::scalar(4.0_f64)),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0_f64, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0]))]),
        );

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0])),
                ArrayIrValue::Array(Array::scalar(24.0_f64)),
            ]),
        );

        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayIrValue::Array(Array::scalar(-1.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 3).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![-1.0_f64, -1.0, -1.0])),);
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(Vec::<f64>::new())),
                ArrayIrValue::Array(Array::scalar(6.0_f64)),
            ]),
        );

        // Explicit pad geometry retains one output extent per physical axis, including statically typed axes. Keep a
        // static leading axis to verify that the pullback selects dynamic constructor operands from the right axis.
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let padded_columns = DimensionVariable::new("padded_columns", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]));
        let padded_columns_type = DimensionType::new(padded_columns);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let rows = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output_extent = builder.add_input(padded_columns_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, rows, output_extent],
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
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::scalar(-1.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(padded_columns_type, 4).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f64, 3.0, 6.0, 7.0])),
                ArrayIrValue::Array(Array::scalar(18.0_f64)),
            ]),
        );
    }

    /// A mixed pad whose operand tangent is a structural zero of an identity-bearing type cannot construct that zero
    /// from the type alone, because a dynamic zero needs one explicit extent operand per dynamic axis. The rule uses
    /// the operand primal as the runtime-geometry exemplar instead, so forward mode succeeds where an input-free
    /// dynamic `zero` would be rejected by constructor inference.
    #[test]
    fn test_array_ir_dynamic_pad_disconnected_operand_tangent_uses_a_primal_exemplar() {
        let source = DimensionVariable::new("source", DimensionBounds::new(1, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let source_type = DimensionType::new(source);
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source_extent = builder.add_input(source_type.clone().into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        // A mixed iota is a non-differentiable nullary constant, so its tangent is a structural zero of the
        // identity-bearing operand type while its primal is a non-zero exemplar and the padding-value tangent stays
        // live. The rule must still hand a concrete operand tangent to the staged pad.
        let operand = builder
            .add_instruction(
                ArrayIrOperation::<Array>::from(IotaOperation::new(input_type, 0).unwrap()),
                Vec::new(),
                vec![source_extent],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![operand, padding_value, output_extent],
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
                ArrayIrValue::Dimension(DimensionValue::new(source_type, 2).unwrap()),
                ArrayIrValue::Array(Array::scalar(-1.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 4).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0_f64)),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![-1.0_f64, 0.0, 0.0, -1.0])),
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 1.0])),
            ]),
        );
    }

    /// Mixed concatenation applies the same operand-exemplar rule: a structurally zero array tangent whose type names
    /// its extent by identity is materialized from that operand's primal rather than from the type.
    #[test]
    fn test_array_ir_dynamic_concatenate_disconnected_input_tangent_uses_a_primal_exemplar() {
        let left = DimensionVariable::new("left", DimensionBounds::new(1, Some(4)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(2, Some(5)).unwrap());
        let left_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(left.clone())]));
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]));
        let left_extent_type = DimensionType::new(left);
        let result_type = DimensionType::new(result);
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left_type.clone().into(), right_type.clone().into(), result_type.clone().into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left_extent = builder.add_input(left_extent_type.clone().into());
        let right = builder.add_input(right_type.into());
        let extent = builder.add_input(result_type.clone().into());
        let left = builder
            .add_instruction(
                ArrayIrOperation::<Array>::from(IotaOperation::new(left_type, 0).unwrap()),
                Vec::new(),
                vec![left_extent],
            )
            .unwrap()[0];
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right, extent]).unwrap()[0];
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
                ArrayIrValue::Dimension(DimensionValue::new(left_extent_type, 2).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![30.0_f64])),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0_f64])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 30.0])),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 1.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2]))),
                Vec::new(),
                vec![input, start],
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

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_update_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );

        let extent = DimensionVariable::new("strided_extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(
                    SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap(),
                )),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 0.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_ir_gather_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
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

        assert_eq!(linearization.residual_count(), 2);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        let indices = ArrayIrValue::Array(Array::from_f64s(indices_type, vec![1.0, 1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])), indices])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 2.0, 4.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]))]),
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
        let indices = ArrayIrValue::Array(Array::from_f64s(indices_type, vec![1.0, 3.0]));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                indices,
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0])),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 12.0, 3.0, 24.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 7.0, 3.0, 10.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 40.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![3]))),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 10.0, 11.0, 12.0]))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 11.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![5.0_f64, 7.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 7.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_ir_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(3, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::UpdateSlice(UpdateSliceOperation::new(vec![1]))),
                Vec::new(),
                vec![input, update],
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

        assert_eq!(linearization.residual_count(), 0);
        let primal = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
            ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0])),
        ];
        assert_eq!(
            linearization.primal().interpret(primal),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0]))]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
                ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        assert_eq!(
            linearization
                .pullback()
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0,]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_update_slice_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let update = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation)),
                Vec::new(),
                vec![input, update, start],
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
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64, 8.0])),
                ArrayIrValue::Array(Array::scalar(1_i32)),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 9.0, 8.0, 4.0])));
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0])),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 5.0, 6.0, 40.0]))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 0.0, 0.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_reshape_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let source_dimension_type = DimensionType::new(source.clone());
        let source_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(source_array_type.clone().into());
        let extent = builder.add_input(source_dimension_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![array, extent, four])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.output_types(),
            vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]),)
                    .into()
            ],
        );

        let target = DimensionVariable::new("target", bounds);
        let target_dimension_type = DimensionType::new(target.clone());
        let target_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]));
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                target_dimension_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]),
                )
                .into()
            ],
        );

        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = destination.add_input(target_array_type.into());
        let extent = destination.add_input(target_dimension_type.into());
        let outputs = destination.splice_program(&instantiated, &[array, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported reshape instruction");
        };
        assert_eq!(instruction.inputs()[..2], [array, extent]);
        assert_eq!(instruction.outputs(), outputs.as_slice());
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(4)]),
            )),
        );
    }

    #[test]
    fn test_array_ir_broadcast() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected_output = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                &[input.clone(), first_extent.clone(), second_extent.clone()],
            ),
            Ok(vec![expected_output.clone()]),
        );
        let eager_dynamic_type =
            DimensionType::new(DimensionVariable::new("eager_extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            context.bind(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![7.0_f64])),
                    ArrayIrValue::Dimension(DimensionValue::new(eager_dynamic_type, 3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 1, vec![7.0_f64, 7.0, 7.0]))]),
        );

        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![1]),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, expected_output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input.clone())),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@residual, expected_output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicBroadcastOperation::new(vec![0]),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input.r#type().into_owned(), replay = identity_input.clone())),
                    (@known, second_extent.clone()),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let first_extent = builder.add_constant(first_extent);
        let second_extent = builder.add_constant(second_extent);
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![1]),
                Vec::new(),
                vec![input, first_extent, second_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(program.instructions()[0].operation(), ArrayIrOperation::Broadcast(_)));
        assert_eq!(program.instructions()[0].inputs(), &[input, first_extent, second_extent]);
        assert!(program.to_string().contains("broadcast [output_axes=[1]]"));

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0])),
            ]),
            Ok(
                vec![expected_output, ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0_f64, 4.0, 3.0, 4.0, 3.0, 4.0])),]
            ),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![9.0_f64, 12.0]))]),
        );

        let dynamic_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_extent = DimensionType::new(dynamic_variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dynamic_variable)])).into());
        let extent = builder.add_input(dynamic_extent.clone().into());
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(vec![0]), Vec::new(), vec![input, extent])
            .unwrap()[0];
        let dynamic_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(dynamic_program.jvp().is_ok());
        let linearization = dynamic_program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dynamic_extent, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]));
        let mut tangent_inputs = vec![tangent.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![tangent.clone()]));
        let mut pullback_inputs = vec![tangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangent]));

        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = builder.add_input(DimensionType::new(source.clone()).into());
        let one = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(vec![1]), Vec::new(), vec![input, extent, one])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let target = DimensionVariable::new("target", bounds);
        let instantiated = program
            .with_instantiated_type_identities(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionType::new(target.clone()).into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(1)]),
                )
                .into()
            ],
        );
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = destination.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = destination.add_input(DimensionType::new(target.clone()).into());
        let outputs = destination.splice_program(&instantiated, &[input, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported broadcast instruction");
        };
        assert_eq!(instruction.inputs()[..2], [input, extent]);
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(1)]),
            )),
        );
    }

    #[test]
    fn test_array_ir_dynamic_shape_slice() -> Result<(), ProgramError> {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::matrix(3, 4, (0..12).map(|value| value as f64).collect::<Vec<_>>()));
        let dimension = |extent| Ok::<_, ProgramError>(ArrayIrValue::Dimension(DimensionValue::constant(extent)?));
        let output = context.bind(
            DynamicShapeSliceOperation::new(2),
            Vec::new(),
            &[input, dimension(1)?, dimension(1)?, dimension(2)?, dimension(2)?],
        )?;
        assert_eq!(output, vec![ArrayIrValue::Array(Array::matrix(2, 2, vec![5.0, 6.0, 9.0, 10.0]))]);

        // The slice geometry is discrete, but the array operand remains linear: JVP applies the same runtime slice to
        // the primal and tangent instead of treating the complete mixed operation as a constant.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)])).into());
        let start = builder.add_constant(dimension(1)?);
        let size = builder.add_constant(dimension(2)?);
        let output =
            builder.add_instruction(DynamicShapeSliceOperation::new(1), Vec::new(), vec![input, start, size])?[0];
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0])),
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 30.0])),
            ]),
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation 'dynamic_shape_slice' does not yet support reverse-mode differentiation",
        ));

        Ok(())
    }

    #[test]
    fn test_array_ir_broadcast_to_first_class_dimensions() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let eager = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        assert_eq!(
            eager.dynamic_broadcast_leading_sizes(&[2]),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],))),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(value.dynamic_broadcast(&[extent], &[0]).unwrap().atom_id(), value.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());

        // A shape-preserving axis permutation is still a real broadcast. Eager execution transposes the payload and
        // tracing retains the operation even though its input and output types are equal.
        let square_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let square = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]));
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let expected = ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]));
        assert_eq!(square.dynamic_broadcast(&[two.clone(), two.clone()], &[1, 0]), Ok(expected.clone()));

        let context = TestContext::new();
        let value = context.input(square_type.into());
        let extent = context.constant(two);
        let output = value.dynamic_broadcast(&[extent.clone(), extent], &[1, 0]).unwrap();
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected one shape-preserving broadcast instruction");
            };
            let ArrayIrOperation::Broadcast(operation) = instruction.operation() else {
                panic!("expected a broadcast instruction");
            };
            assert_eq!(operation.output_axes(), &[1, 0]);
        }
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(vec![square]), Ok(vec![expected]));

        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let context = TestContext::new();
        let scalar = context.input(ArrayType::scalar(DataType::F64).into());
        let extent = context.input(extent_type.clone().into());
        let output = scalar.dynamic_broadcast_to(std::slice::from_ref(&extent)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<extent \u{2208} [1, 5)> .
                let %2:f64[extent] = broadcast [output_axes=[]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::scalar(2.5_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5]))]),
        );
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0_f64))]),
        );

        let context = TestContext::new();
        let value = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).into());
        let rows = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let columns = context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = value.dynamic_broadcast_to(&[rows, columns]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0_f64]))]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0_f64; 6]))]),
        );

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let context = TestContext::new();
        let value = context.input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let output = value.dynamic_broadcast_leading_sizes(&[2]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("dimension_size").count(), 1);
        assert!(rendered.contains("broadcast [output_axes=[1, 2]]"));
    }

    #[test]
    fn test_array_ir_concatenate() {
        let left = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let right = ArrayIrValue::Array(Array::vector(vec![3.0_f32]));
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left.r#type().into_owned(), right.r#type().into_owned(), extent.r#type().into_owned()],
        )
        .unwrap();

        // Eager execution consumes the explicit extent without copying either array during member projection.
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[left.clone(), right.clone(), extent.clone()],),
            Ok(vec![output.clone()]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[left.clone(), ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),],
            ),
            Ok(vec![left.clone()]),
        );

        let observed_extent_type =
            DimensionType::new(DimensionVariable::new("observed", DimensionBounds::new(1, Some(9)).unwrap()));
        let checked_operation = ConcatenateOperation::<ArrayIrType>::from(ConcatenateOperation::new(0, 1).unwrap());
        assert_eq!(
            ArrayIrOperation::<Array>::from(checked_operation).interpret(
                &context,
                &EmptyRegionDriver,
                &[
                    left.clone(),
                    right.clone(),
                    ArrayIrValue::Dimension(DimensionValue::new(observed_extent_type, 4).unwrap()),
                ],
            ),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "'{}' result extent must equal the sum of input axis 0 extents; expected 3 but got 4",
                    CONCATENATE_OPERATION_NAME,
                ),
            }),
        );

        // Partial evaluation folds a fully known concatenate and otherwise retains exactly one operation with the
        // explicit extent edge, including when only that extent is unknown.
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, left.clone()), (@known, right.clone()), (@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = left.r#type().into_owned(), replay = left.clone())),
                        (@known, right.clone()),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, left.clone()),
                        (@known, right.clone()),
                        (@unknown(type = extent.r#type().into_owned(), replay = extent.clone())),
                    ],
                    outputs = [(@residual, output.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        // A stored dynamic program computes the trailing extent through ordinary dimension SSA and records every
        // dependency explicitly on the concatenate instruction.
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(5)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(6)).unwrap());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(left_variable.clone())]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(right_variable.clone())]));
        let left_size_operation = DimensionSizeOperation::new(&left_type, 0).unwrap();
        let right_size_operation = DimensionSizeOperation::new(&right_type, 0).unwrap();
        let left_size_type = left_size_operation.result_type().clone();
        let right_size_type = right_size_operation.result_type().clone();
        let add_operation = DimensionAddOperation::new(&left_size_type, &right_size_type).unwrap();
        let result_extent_type =
            DimensionType::new(DimensionVariable::new(add_operation.result_name(), add_operation.result_bounds()));
        let dynamic_operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left_type.clone().into(), right_type.clone().into(), result_extent_type.into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left_input = builder.add_input(left_type.into());
        let right_input = builder.add_input(right_type.into());
        let left_size = builder.add_instruction(left_size_operation, Vec::new(), vec![left_input]).unwrap()[0];
        let right_size = builder.add_instruction(right_size_operation, Vec::new(), vec![right_input]).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![left_size, right_size])
            .unwrap()[0];
        let concatenated = builder
            .add_instruction(dynamic_operation, Vec::new(), vec![left_input, right_input, result_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![concatenated],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let [left_size_instruction, right_size_instruction, add_instruction, concatenate_instruction] =
            program.instructions()
        else {
            panic!("expected two dimension reads, one dimension addition, and one concatenate");
        };
        assert_eq!(left_size_instruction.inputs(), &[left_input]);
        assert_eq!(right_size_instruction.inputs(), &[right_input]);
        assert_eq!(add_instruction.inputs(), &[left_size, right_size]);
        assert_eq!(concatenate_instruction.inputs(), &[left_input, right_input, result_extent]);
        assert!(matches!(concatenate_instruction.operation(), ArrayIrOperation::Concatenate(_),));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[left], %1:f32[right] .
                let %2:dimension<left ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<right ∈ [1, 6)> = dimension_size [axis=0] %1
                    %4:dimension<left + right ∈ [2, 10)> = dimension_add %2 %3
                    %5:f32[left + right] = concatenate [axis=0] %0 %1 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(vec![left, right]), Ok(vec![output.clone()]));

        // The same stored dynamic program composes dimension arithmetic with both forward differentiation and
        // batching. Its tangent retains the result extent and both operand extents as ordinary residual SSA edges.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 4);
        let transformed_result_extent = jvp
            .instructions()
            .iter()
            .find_map(|instruction| {
                matches!(instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Add(_)),)
                    .then_some(instruction.outputs()[0])
            })
            .unwrap();
        assert_eq!(
            jvp.instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayIrOperation::Concatenate(_) => instruction.inputs().last().copied(),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![transformed_result_extent],
        );
        let tangent_call = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)))
            .unwrap();
        assert_eq!(tangent_call.inputs()[0], transformed_result_extent);
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f32])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f32])),
            ]),
            Ok(vec![output, ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0])),]),
        );

        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("items".to_string())
        .with_axis_sharding(ShardingDimension::Unconstrained);
        let batched_outputs = program
            .interpret_in_context(
                &batching_context,
                vec![
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 4.0, 5.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        batching_context.clone(),
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 1, vec![3.0_f32, 6.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(batched_outputs.len(), 1);
        assert_eq!(batched_outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            batched_outputs[0].batch().value(),
            &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 3);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=3]"));
        assert!(linearization.pullback().unwrap().to_string().contains("dynamic_shape_slice"));
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f32])),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f32])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_concatenate_differentiation() {
        let left_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]));
        let extent_value = DimensionValue::constant(3).unwrap();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left_type.clone().into(), right_type.clone().into(), extent_value.r#type().into_owned().into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = builder.add_input(left_type.into());
        let right = builder.add_input(right_type.into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(extent_value));
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right, extent]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![3.0_f64])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f64])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0])),
            ]),
        );
        assert_eq!(
            program
                .transpose_with_respect_to(&[0, 1])
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0_f64, 8.0, 9.0]))]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![7.0_f64, 8.0])),
                ArrayIrValue::Array(Array::vector(vec![9.0_f64])),
            ]),
        );
    }

    #[test]
    fn test_array_ir_concatenate_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let result = DimensionVariable::new("result", DimensionBounds::new(2, Some(12)).unwrap());
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let fixed_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let result_extent_type = DimensionType::new(result.clone());
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[source_array_type.clone().into(), fixed_array_type.clone().into(), result_extent_type.clone().into()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source_array = builder.add_input(source_array_type.into());
        let fixed_array = builder.add_input(fixed_array_type.clone().into());
        let result_extent = builder.add_input(result_extent_type.into());
        let output = builder
            .add_instruction(operation, Vec::new(), vec![source_array, fixed_array, result_extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let target = DimensionVariable::new("target", bounds);
        let target_result = DimensionVariable::new("target_result", DimensionBounds::new(2, Some(12)).unwrap());
        let target_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let target_result_type = DimensionType::new(target_result.clone());
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                fixed_array_type.clone().into(),
                target_result_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_result.clone())])).into()],
        );

        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_source = destination.add_input(target_array_type.into());
        let imported_fixed = destination.add_input(fixed_array_type.into());
        let imported_extent = destination.add_input(target_result_type.into());
        let imported_outputs = destination
            .splice_program(&instantiated, &[imported_source, imported_fixed, imported_extent])
            .unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported concatenate instruction");
        };
        assert_eq!(instruction.inputs(), &[imported_source, imported_fixed, imported_extent]);
        assert_eq!(instruction.outputs(), imported_outputs.as_slice());
        assert_eq!(
            destination.atoms()[imported_outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_result)]),)),
        );
    }

    #[test]
    fn test_array_broadcast() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let output_type = array_type(DataType::F64, &[3, 2]);
        let broadcast = Broadcast::broadcast(&vector, output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().into_owned(), output_type);
        assert_eq!(broadcast.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Broadcasting reads a reversed input layout and writes the output's requested physical layout, retaining
        // zero in its holes.
        let input_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type, &[0x1122u16, 0x3344]).unwrap();
        let output_type =
            array_type(DataType::U16, &[2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![6, 2])));
        let broadcast = input.broadcast(output_type.clone(), &[1]).unwrap();
        assert_eq!(broadcast.r#type().as_ref(), &output_type);
        assert_eq!(broadcast.elements::<u16>(), Ok(vec![0x1122, 0x3344, 0x1122, 0x3344]));
        assert_eq!(broadcast.storage_bytes(), [0x22, 0x11, 0x44, 0x33, 0, 0, 0x22, 0x11, 0x44, 0x33]);

        // Deliberately malformed concrete values with dynamic types fail through the structured materialization
        // diagnostic before either the identity fast path or static-shape payload logic can accept or panic on them.
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(dynamic.clone()), Dimension::Dynamic(dynamic)]),
        );
        let dynamic = Array::with_unchecked_type(
            dynamic_type.clone(),
            [1.0f64, 2.0, 3.0, 4.0].into_iter().flat_map(f64::to_le_bytes).collect(),
        );
        for output_axes in [vec![0, 1], vec![1, 0]] {
            assert!(matches!(
                dynamic.broadcast(dynamic_type.clone(), output_axes.as_slice()),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == "cannot materialize a value of dynamically sized type f64[dynamic, dynamic]",
            ));
        }
    }

    #[test]
    fn test_array_transpose() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(transposed.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(matrix.transpose([0, 0]).is_err());

        // Transposition traverses the input's physical layout while producing the canonical layout-free output type.
        let input_type =
            array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.r#type().into_owned(), array_type(DataType::U16, &[3, 2]));
        assert_eq!(transposed.elements::<u16>(), Ok(vec![1, 4, 2, 5, 3, 6]));
        assert_eq!(transposed.storage_bytes(), [1, 0, 4, 0, 2, 0, 5, 0, 3, 0, 6, 0]);

        // Empty arrays transpose without calculating strides that may overflow for otherwise irrelevant dimensions.
        let empty = Array::new(array_type(DataType::F64, &[0, usize::MAX, usize::MAX]), Vec::new()).unwrap();
        assert_eq!(
            empty.transpose([1, 2, 0]).unwrap(),
            Array::new(array_type(DataType::F64, &[usize::MAX, usize::MAX, 0]), Vec::new()).unwrap(),
        );
    }

    #[test]
    fn test_array_reshape() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.r#type().into_owned(), array_type(DataType::F64, &[3, 2]));
        assert_eq!(reshaped.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix.reshape(Shape::new(vec![Dimension::Static(4)])).is_err());

        // Reshaping preserves logical order independently of the input's physical placement.
        let input_type =
            array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.elements::<u16>(), Ok(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(reshaped.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn test_array_slicing() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap(), Array::vector(vec![2.0, 4.0]));
        assert_eq!(
            vector.update_slice(&Array::vector(vec![10.0, 20.0]), &[1]).unwrap(),
            Array::vector(vec![1.0, 10.0, 20.0, 4.0, 5.0]),
        );
        // Dynamic start indices clamp so the block stays in bounds.
        let start = [Array::scalar(4i64)];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![4.0, 5.0]));
        assert_eq!(
            vector.dynamic_update_slice(&Array::vector(vec![10.0, 20.0]), &start).unwrap(),
            Array::vector(vec![1.0, 2.0, 3.0, 10.0, 20.0]),
        );
        // Index decoding is typed and supports sub-byte integers directly; a negative start still clamps to zero.
        let start = [Array::scalar(i4::new(-1).unwrap())];
        assert_eq!(vector.dynamic_slice(&start, &[2]).unwrap(), Array::vector(vec![1.0, 2.0]));

        // Static slicing and updating traverse arbitrary source and update layouts while preserving the destination
        // layout for updates.
        let input_type = array_type(DataType::U16, &[5]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type.clone(), &[1u16, 2, 3, 4, 5]).unwrap();
        assert_eq!(vector.slice(&[1], &[5], &[2]).unwrap().elements::<u16>(), Ok(vec![2, 4]));
        let update_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let update = Array::from_elements(update_type, &[10u16, 20]).unwrap();
        let updated = vector.update_slice(&update, &[1]).unwrap();
        assert_eq!(updated.r#type().as_ref(), &input_type);
        assert_eq!(updated.elements::<u16>(), Ok(vec![1, 10, 20, 4, 5]));
        assert_eq!(updated.storage_bytes(), [5, 0, 4, 0, 20, 0, 10, 0, 1, 0]);
    }

    #[test]
    fn test_array_pad() {
        let vector = Array::vector(vec![1.0, 2.0]);
        let padded = vector.pad(&Array::scalar(0.5), &[1], &[2], &[1]).unwrap();
        assert_eq!(padded, Array::vector(vec![0.5, 1.0, 0.5, 2.0, 0.5, 0.5]));

        // Padding copies both the reversed input layout and the rank-zero padding element by their exact bytes.
        let input_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type, &[1u16, 2]).unwrap();
        let padded = vector.pad(&Array::scalar(9u16), &[1], &[1], &[1]).unwrap();
        assert_eq!(padded.r#type().into_owned(), array_type(DataType::U16, &[5]));
        assert_eq!(padded.elements::<u16>(), Ok(vec![9, 1, 9, 2, 9]));
        assert_eq!(padded.storage_bytes(), [9, 0, 1, 0, 9, 0, 2, 0, 9, 0]);
    }

    #[test]
    fn test_array_concatenate() {
        // Three operands joined along axis 0 preserve their order.
        let concatenated = Array::concatenate(
            [&Array::vector(vec![1.0]), &Array::vector(vec![2.0, 3.0]), &Array::vector(vec![4.0])],
            0,
        )
        .unwrap();
        assert_eq!(concatenated, Array::vector(vec![1.0, 2.0, 3.0, 4.0]));

        // A rank-3 middle-axis concatenation exercises the row-major block odometer.
        let first = Array::from_f64s(array_type(DataType::F64, &[2, 1, 2]), vec![1.0, 2.0, 3.0, 4.0]);
        let second =
            Array::from_f64s(array_type(DataType::F64, &[2, 2, 2]), vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let concatenated = Array::concatenate([&first, &second], 1).unwrap();
        assert_eq!(concatenated.r#type().into_owned(), array_type(DataType::F64, &[2, 3, 2]));
        assert_eq!(concatenated.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],);

        // Concatenation traverses each input's physical layout and emits the canonical layout-free result.
        let first_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let second_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let first = Array::from_elements(first_type, &[1u16, 2]).unwrap();
        let second = Array::from_elements(second_type, &[3u16, 4]).unwrap();
        let concatenated = Array::concatenate([&first, &second], 0).unwrap();
        assert_eq!(concatenated.r#type().into_owned(), array_type(DataType::U16, &[4]));
        assert_eq!(concatenated.elements::<u16>(), Ok(vec![1, 2, 3, 4]));
        assert_eq!(concatenated.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0]);

        // Concatenation does not require an artificial additive zero, including when the output itself is empty.
        let element_type = array_type(DataType::F8E8M0FNU, &[1]);
        let first = Array::new(element_type.clone(), vec![1]).unwrap();
        let second = Array::new(element_type, vec![2]).unwrap();
        assert_eq!(
            Array::concatenate([&first, &second], 0),
            Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![1, 2]),
        );
        let empty_type = array_type(DataType::F8E8M0FNU, &[0]);
        let empty = Array::new(empty_type.clone(), Vec::new()).unwrap();
        assert_eq!(Array::concatenate([&empty, &empty], 0), Array::new(empty_type, Vec::new()));
    }

    #[test]
    fn test_array_gather() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = Array::matrix(2, 1, vec![2i64, 0]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.r#type().into_owned(), array_type(DataType::F64, &[2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);

        // In-bounds and clipping modes do not materialize an unused zero fill, so they work for formats that cannot
        // represent zero.
        let operand = Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![0x7f, 0x80]).unwrap();
        let indices = Array::matrix(1, 1, vec![1i64]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(operand.gather(&indices, &operation), Array::new(array_type(DataType::F8E8M0FNU, &[1]), vec![0x80]));

        // Gather reads both a reversed operand and reversed sub-byte indices through their physical addressing. An
        // out-of-bounds query in fill-or-drop mode writes the element type's zero encoding into the dense result.
        let operand_type = array_type(DataType::U16, &[3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type, &[10u16, 20, 30]).unwrap();
        let indices_type =
            array_type(DataType::I4, &[3, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
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
        let operand = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let indices = Array::from_f64s(array_type(DataType::I64, &[2, 1]), vec![3.0, 0.0]);
        let updates = Array::vector(vec![10.0, 20.0]);
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        let scattered = operand.scatter(&indices, &updates, &operation).unwrap();
        assert_eq!(scattered, Array::vector(vec![21.0, 2.0, 3.0, 14.0]));

        // Scatter decodes sub-byte indices through their physical layout without materializing a scalar index vector.
        let indices_type =
            array_type(DataType::I4, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(indices_type, &[i4::new(3).unwrap(), i4::new(0).unwrap()]).unwrap();
        assert_eq!(operand.scatter(&indices, &updates, &operation).unwrap(), Array::vector(vec![21.0, 2.0, 3.0, 14.0]),);

        // Operand and update payloads are decoded and written through their independent physical layouts.
        let operand_type = array_type(DataType::U16, &[4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type.clone(), &[1u16, 2, 3, 4]).unwrap();
        let updates_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let updates = Array::from_elements(updates_type, &[10u16, 20]).unwrap();
        assert_eq!(
            operand.scatter(&indices, &updates, &operation),
            Array::from_elements(operand_type, &[21u16, 2, 3, 14]),
        );

        // Sub-byte arithmetic wraps in the declared bit width, including repeated modular addition.
        let operand = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]);
        let indices = Array::matrix(2, 1, vec![0i32, 1]);
        let updates = Array::vector(vec![i4::new(2).unwrap(), i4::new(-3).unwrap()]);
        assert_eq!(
            operand.scatter(&indices, &updates, &operation).unwrap().elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );

        // Overwrite moves encodings without requiring arithmetic identities, including for formats without zero.
        let operand = Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![0x7f, 0x80]).unwrap();
        let updates = Array::new(array_type(DataType::F8E8M0FNU, &[1]), vec![0x81]).unwrap();
        let indices = Array::matrix(1, 1, vec![0i32]);
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
            ScatterReductionKind::Overwrite,
        );
        assert_eq!(
            operand.scatter(&indices, &updates, &operation),
            Array::new(array_type(DataType::F8E8M0FNU, &[2]), vec![0x81, 0x80]),
        );

        // Extrema follow JAX for floating-point NaNs and signed zero and for lexicographically ordered complex values.
        let indices = Array::matrix(2, 1, vec![0i32, 1]);
        let operand = Array::vector(vec![f32::NAN, -0.0]);
        let updates = Array::vector(vec![1.0f32, 0.0]);
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

        let operand = Array::vector(vec![ComplexNumber::new(1.0f32, 9.0), ComplexNumber::new(2.0, -1.0)]);
        let updates = Array::vector(vec![ComplexNumber::new(1.0f32, 10.0), ComplexNumber::new(1.0, 100.0)]);
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
