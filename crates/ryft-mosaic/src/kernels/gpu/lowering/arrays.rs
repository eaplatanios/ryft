//! Scalar-loop implementations of canonical array operations over cooperative flat storage.

use ryft_core::{Array, ArrayOperation, ArrayType, ComparisonDirection, DataType, Operation, ReductionKind};
use ryft_mlir::dialects::{arith, math, scf};
use ryft_mlir::{Block, DetachedBlock, Type, Value};

use crate::kernels::gpu::Error;
use crate::kernels::gpu::lowering::module::element_type;
use crate::kernels::gpu::lowering::{Buffer, KernelValue, Lowering, append, shape};

/// Checks the exact array lowering subset before any native operations are constructed.
pub(super) fn validate(
    operation: &ArrayOperation<Array>,
    inputs: &[ArrayType],
    output: &ArrayType,
) -> Result<(), Error> {
    for r#type in inputs.iter().chain(std::iter::once(output)) {
        if !matches!(
            r#type.data_type(),
            DataType::Boolean
                | DataType::U8
                | DataType::F8E4M3FN
                | DataType::F8E8M0FNU
                | DataType::F16
                | DataType::BF16
                | DataType::I32
                | DataType::U32
                | DataType::I64
                | DataType::U64
                | DataType::F32
                | DataType::F64
        ) {
            return Err(unsupported(operation, "element type has no baseline scalar implementation"));
        }
        checked_count(operation, &shape(r#type)?)?;
    }
    if inputs
        .iter()
        .chain(std::iter::once(output))
        .any(|r#type| matches!(r#type.data_type(), DataType::F16 | DataType::BF16))
        && !matches!(
            operation,
            ArrayOperation::Constant(_)
                | ArrayOperation::ConvertElementType(_)
                | ArrayOperation::Transpose(_)
                | ArrayOperation::Broadcast(_)
                | ArrayOperation::Reshape(_)
                | ArrayOperation::StopGradient(_)
                | ArrayOperation::Select(_)
        )
    {
        return Err(unsupported(
            operation,
            "half-precision storage supports only bit-preserving operations and numeric conversion",
        ));
    }
    if inputs
        .iter()
        .chain(std::iter::once(output))
        .any(|r#type| matches!(r#type.data_type(), DataType::U8 | DataType::F8E4M3FN | DataType::F8E8M0FNU))
        && !matches!(
            operation,
            ArrayOperation::Constant(_)
                | ArrayOperation::Transpose(_)
                | ArrayOperation::Broadcast(_)
                | ArrayOperation::Reshape(_)
                | ArrayOperation::StopGradient(_)
                | ArrayOperation::Select(_)
        )
    {
        return Err(unsupported(
            operation,
            "packed operand and scale storage supports only bit-preserving array operations",
        ));
    }
    let expected = operation.infer_output_types(inputs, &[])?;
    if expected.as_slice() != [output.clone()] {
        return Err(unsupported(operation, "result type differs from canonical array inference"));
    }
    let data_inputs = if matches!(operation, ArrayOperation::Select(_)) { &inputs[1..] } else { inputs };
    if data_inputs.windows(2).any(|pair| pair[0].data_type() != pair[1].data_type()) {
        return Err(unsupported(operation, "arithmetic operand element types must agree"));
    }
    match operation {
        ArrayOperation::Zero(_)
        | ArrayOperation::ZeroLike(_)
        | ArrayOperation::One(_)
        | ArrayOperation::OneLike(_)
        | ArrayOperation::Constant(_)
        | ArrayOperation::Transpose(_)
        | ArrayOperation::Broadcast(_)
        | ArrayOperation::Add(_)
        | ArrayOperation::Sub(_)
        | ArrayOperation::Mul(_)
        | ArrayOperation::Neg(_)
        | ArrayOperation::And(_)
        | ArrayOperation::Or(_)
        | ArrayOperation::Xor(_)
        | ArrayOperation::Not(_)
        | ArrayOperation::Select(_)
        | ArrayOperation::Compare(_)
        | ArrayOperation::StopGradient(_) => Ok(()),
        ArrayOperation::Iota(_) if output.data_type() != DataType::Boolean => Ok(()),
        ArrayOperation::Div(_) if is_float(output.data_type()) => Ok(()),
        ArrayOperation::Exp(_) if matches!(output.data_type(), DataType::F32 | DataType::F64) => Ok(()),
        ArrayOperation::ConvertElementType(operation)
            if !operation.bitcast() && is_float(inputs[0].data_type()) && is_float(output.data_type()) =>
        {
            Ok(())
        }
        ArrayOperation::Reduce(reduction)
            if reduction.output_sharding().is_none()
                && (reduction.kind() == ReductionKind::Sum
                    || reduction.kind() == ReductionKind::Max
                        && matches!(output.data_type(), DataType::F32 | DataType::F64)) =>
        {
            let input_shape = shape(&inputs[0])?;
            checked_count(operation, &reduction.axes().iter().map(|&axis| input_shape[axis]).collect::<Vec<_>>())?;
            Ok(())
        }
        ArrayOperation::Dot(dot)
            if dot.accumulation_type().is_none()
                && dot.output_sharding().is_none()
                && output.data_type() != DataType::Boolean =>
        {
            let input_shape = shape(&inputs[0])?;
            checked_count(
                operation,
                &dot.dimensions()
                    .lhs_contracting_dimensions()
                    .iter()
                    .map(|&axis| input_shape[axis])
                    .collect::<Vec<_>>(),
            )?;
            Ok(())
        }
        ArrayOperation::Reshape(reshape) if reshape.output_sharding().is_none() => Ok(()),
        _ => Err(unsupported(operation, "operation or its metadata has no baseline scalar implementation")),
    }
}

/// Checks products used as native loop bounds, including contractions of otherwise empty arrays.
fn checked_count(operation: &ArrayOperation<Array>, dimensions: &[usize]) -> Result<usize, Error> {
    dimensions.iter().try_fold(1usize, |count, &extent| {
        (extent <= i64::MAX as usize)
            .then_some(count)
            .and_then(|count| count.checked_mul(extent))
            .filter(|&count| count <= i64::MAX as usize)
            .ok_or_else(|| unsupported(operation, "array loop extent exceeds signed native index bounds"))
    })
}

impl<'c, 't> Lowering<'c, 't> {
    /// Emits one ordinary array operation; every result element has exactly one owning lane.
    pub(super) fn array(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &ArrayOperation<Array>,
        inputs: &[Buffer<'c, 't>],
        output: &Buffer<'c, 't>,
    ) -> Result<(), Error> {
        validate(operation, &inputs.iter().map(|input| input.r#type.clone()).collect::<Vec<_>>(), &output.r#type)?;
        let output_shape = shape(&output.r#type)?;
        let input_shapes = inputs.iter().map(|input| shape(&input.r#type)).collect::<Result<Vec<_>, _>>()?;
        let data_type = output.r#type.data_type();
        let count = output_shape.iter().product();
        self.distributed(block, count, |lowering, body, index| {
            let coordinates = lowering.coordinates(body, index, &output_shape)?;
            let result = match operation {
                ArrayOperation::Zero(_) | ArrayOperation::ZeroLike(_) => lowering.literal(body, data_type, 0)?,
                ArrayOperation::One(_) | ArrayOperation::OneLike(_) => {
                    lowering.literal(body, data_type, one_bits(data_type))?
                }
                ArrayOperation::Constant(constant) => {
                    let bytes = constant.value().logical_bytes();
                    let width = match data_type {
                        DataType::Boolean => 1,
                        DataType::F16 | DataType::BF16 => 16,
                        DataType::F32 | DataType::I32 | DataType::U32 => 4,
                        _ => 8,
                    };
                    let mut selected = lowering.literal(body, data_type, 0)?;
                    for (element, bytes) in bytes.chunks_exact(width).enumerate() {
                        let mut encoding = [0u8; 8];
                        encoding[..width].copy_from_slice(bytes);
                        let value = lowering.literal(body, data_type, u64::from_le_bytes(encoding))?;
                        let position = lowering.index(body, element)?;
                        let matches = append(
                            body,
                            arith::cmpi(index, position, arith::IntegerComparisonPredicate::Equal, lowering.location)?,
                        )?;
                        selected = append(body, arith::select(matches, value, selected, lowering.location)?)?;
                    }
                    selected
                }
                ArrayOperation::Iota(iota) => {
                    let value = coordinates[iota.dimension()];
                    let integer_type = lowering.context.signless_integer_type(64);
                    let value = append(body, arith::index_cast(value, integer_type, lowering.location)?)?;
                    match data_type {
                        DataType::F32 | DataType::F64 => append(
                            body,
                            arith::sitofp(value, element_type(lowering.context, data_type)?, lowering.location)?,
                        )?,
                        DataType::I64 | DataType::U64 => value,
                        DataType::I32 | DataType::U32 => append(
                            body,
                            arith::trunci(value, element_type(lowering.context, data_type)?, lowering.location)?,
                        )?,
                        _ => return Err(unsupported(operation, "iota requires a supported numeric element type")),
                    }
                }
                ArrayOperation::Reshape(_) => lowering.load(body, &inputs[0], index)?,
                ArrayOperation::Transpose(transpose) => {
                    let mut source_coordinates = coordinates.clone();
                    for (axis, source_axis) in
                        transpose.permutation().normalize(input_shapes[0].len())?.into_iter().enumerate()
                    {
                        source_coordinates[source_axis] = coordinates[axis];
                    }
                    let source = lowering.flat_index(body, &source_coordinates, &input_shapes[0])?;
                    lowering.load(body, &inputs[0], source)?
                }
                ArrayOperation::Broadcast(broadcast) => {
                    let source_coordinates = broadcast
                        .output_axes()
                        .iter()
                        .enumerate()
                        .map(|(axis, &output_axis)| {
                            if input_shapes[0][axis] == 1 {
                                lowering.index(body, 0)
                            } else {
                                Ok(coordinates[output_axis])
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let source = lowering.flat_index(body, &source_coordinates, &input_shapes[0])?;
                    lowering.load(body, &inputs[0], source)?
                }
                ArrayOperation::Reduce(reduction) if reduction.axes().is_empty() => {
                    lowering.load(body, &inputs[0], index)?
                }
                ArrayOperation::Reduce(reduction) => {
                    let axes = reduction.axes();
                    let reduction_shape = axes.iter().map(|&axis| input_shapes[0][axis]).collect::<Vec<_>>();
                    lowering.accumulate(
                        body,
                        reduction_shape.iter().product(),
                        data_type,
                        match (reduction.kind(), data_type) {
                            (ReductionKind::Max, DataType::F32) => f32::NEG_INFINITY.to_bits() as u64,
                            (ReductionKind::Max, DataType::F64) => f64::NEG_INFINITY.to_bits(),
                            _ => 0,
                        },
                        |lowering, body, reduction_index, accumulator| {
                            let reduction_coordinates =
                                lowering.coordinates(body, reduction_index, &reduction_shape)?;
                            let mut source_coordinates = vec![index; input_shapes[0].len()];
                            let mut result_axis = 0;
                            for (axis, coordinate) in source_coordinates.iter_mut().enumerate() {
                                if let Some(position) = axes.iter().position(|&reduction_axis| reduction_axis == axis) {
                                    *coordinate = reduction_coordinates[position];
                                } else {
                                    *coordinate = coordinates[result_axis];
                                    result_axis += 1;
                                }
                            }
                            let source = lowering.flat_index(body, &source_coordinates, &input_shapes[0])?;
                            let value = lowering.load(body, &inputs[0], source)?;
                            if reduction.kind() == ReductionKind::Max {
                                append(body, arith::maximumf(accumulator, value, lowering.location)?)
                            } else {
                                lowering.add(body, data_type, accumulator, value)
                            }
                        },
                    )?
                }
                ArrayOperation::Dot(dot) => {
                    let dimensions = dot.dimensions();
                    let left_contracting = dimensions.lhs_contracting_dimensions();
                    let right_contracting = dimensions.rhs_contracting_dimensions();
                    let left_batching = dimensions.lhs_batching_dimensions();
                    let right_batching = dimensions.rhs_batching_dimensions();
                    let contraction_shape =
                        left_contracting.iter().map(|&axis| input_shapes[0][axis]).collect::<Vec<_>>();
                    lowering.accumulate(
                        body,
                        contraction_shape.iter().product(),
                        data_type,
                        0,
                        |lowering, body, contraction_index, accumulator| {
                            let contraction_coordinates =
                                lowering.coordinates(body, contraction_index, &contraction_shape)?;
                            let mut left = vec![index; input_shapes[0].len()];
                            let mut right = vec![index; input_shapes[1].len()];
                            for (position, (&left_axis, &right_axis)) in
                                left_batching.iter().zip(right_batching).enumerate()
                            {
                                left[left_axis] = coordinates[position];
                                right[right_axis] = coordinates[position];
                            }
                            for (position, (&left_axis, &right_axis)) in
                                left_contracting.iter().zip(right_contracting).enumerate()
                            {
                                left[left_axis] = contraction_coordinates[position];
                                right[right_axis] = contraction_coordinates[position];
                            }
                            let mut output_axis = left_batching.len();
                            for (axis, coordinate) in left.iter_mut().enumerate() {
                                if !left_batching.contains(&axis) && !left_contracting.contains(&axis) {
                                    *coordinate = coordinates[output_axis];
                                    output_axis += 1;
                                }
                            }
                            for (axis, coordinate) in right.iter_mut().enumerate() {
                                if !right_batching.contains(&axis) && !right_contracting.contains(&axis) {
                                    *coordinate = coordinates[output_axis];
                                    output_axis += 1;
                                }
                            }
                            let left_index = lowering.flat_index(body, &left, &input_shapes[0])?;
                            let right_index = lowering.flat_index(body, &right, &input_shapes[1])?;
                            let left = lowering.load(body, &inputs[0], left_index)?;
                            let right = lowering.load(body, &inputs[1], right_index)?;
                            let product = if is_float(data_type) {
                                append(body, arith::mulf(left, right, lowering.location)?)?
                            } else {
                                append(body, arith::muli(left, right, lowering.location)?)?
                            };
                            lowering.add(body, data_type, accumulator, product)
                        },
                    )?
                }
                _ => {
                    let values = inputs
                        .iter()
                        .zip(&input_shapes)
                        .map(|(input, input_shape)| {
                            let offset = output_shape.len().checked_sub(input_shape.len()).ok_or_else(|| {
                                unsupported(operation, "operand rank exceeds the pointwise result rank")
                            })?;
                            let source_coordinates =
                                input_shape
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, &extent)| {
                                        if extent == 1 {
                                            lowering.index(body, 0)
                                        } else {
                                            Ok(coordinates[offset + axis])
                                        }
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                            let source = lowering.flat_index(body, &source_coordinates, input_shape)?;
                            lowering.load(body, input, source)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    lowering.pointwise(body, operation, inputs, &values, data_type)?
                }
            };
            lowering.store(body, output, index, result)
        })?;
        self.barrier(block)
    }

    /// Constructs row-major coordinates. Empty dimensions occur only inside loops whose trip count is zero.
    fn coordinates(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        mut index: KernelValue<'c, 't>,
        dimensions: &[usize],
    ) -> Result<Vec<KernelValue<'c, 't>>, Error> {
        let mut coordinates = Vec::with_capacity(dimensions.len());
        for &extent in dimensions.iter().rev() {
            let divisor = self.index(block, extent.max(1))?;
            coordinates.push(append(block, arith::remui(index, divisor, self.location)?)?);
            index = append(block, arith::divui(index, divisor, self.location)?)?;
        }
        coordinates.reverse();
        Ok(coordinates)
    }

    /// Converts logical coordinates into a flat physical element index.
    fn flat_index(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        coordinates: &[KernelValue<'c, 't>],
        dimensions: &[usize],
    ) -> Result<KernelValue<'c, 't>, Error> {
        let mut index = self.index(block, 0)?;
        for (&coordinate, &extent) in coordinates.iter().zip(dimensions) {
            let scale = self.index(block, extent)?;
            let scaled = append(block, arith::muli(index, scale, self.location)?)?;
            index = append(block, arith::addi(scaled, coordinate, self.location)?)?;
        }
        Ok(index)
    }

    /// Emits an exact scalar encoding, including NaN payloads and signed zeros.
    pub(super) fn literal(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        data_type: DataType,
        bits: u64,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let width = match data_type {
            DataType::Boolean => 1,
            DataType::U8 | DataType::F8E4M3FN | DataType::F8E8M0FNU => 8,
            DataType::F16 | DataType::BF16 => 16,
            DataType::F32 | DataType::I32 | DataType::U32 => 32,
            _ => 64,
        };
        let integer = append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(width), bits as i64),
                self.location,
            )?,
        )?;
        if is_float(data_type) {
            append(block, arith::bitcast(integer, element_type(self.context, data_type)?, self.location)?)
        } else {
            Ok(integer)
        }
    }

    /// Accumulates a static Cartesian contraction in a runtime SCF loop with a scalar carried value.
    fn accumulate(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        count: usize,
        data_type: DataType,
        identity: u64,
        function: impl FnOnce(
            &mut Self,
            &mut DetachedBlock<'c, 't>,
            KernelValue<'c, 't>,
            KernelValue<'c, 't>,
        ) -> Result<KernelValue<'c, 't>, Error>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let lower = self.index(block, 0)?;
        let upper = self.index(block, count)?;
        let step = self.index(block, 1)?;
        let identity = self.literal(block, data_type, identity)?;
        let mut body = self.context.block(&[
            (self.context.index_type().as_ref(), self.location),
            (element_type(self.context, data_type)?, self.location),
        ]);
        let index = body.argument(0)?.as_ref();
        let accumulator = body.argument(1)?.as_ref();
        let result = function(self, &mut body, index, accumulator)?;
        body.append_operation(scf::r#yield(&[result], self.location)?)?;
        append(block, scf::r#for(lower, upper, step, &[identity], false, body.try_into()?, self.location)?)
    }

    /// Adds values using the admitted scalar element type.
    fn add(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        data_type: DataType,
        left: KernelValue<'c, 't>,
        right: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        if is_float(data_type) {
            append(block, arith::addf(left, right, self.location)?)
        } else {
            append(block, arith::addi(left, right, self.location)?)
        }
    }

    /// Selects the scalar operation after broadcast indices have been made explicit.
    fn pointwise(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &ArrayOperation<Array>,
        inputs: &[Buffer<'c, 't>],
        values: &[KernelValue<'c, 't>],
        data_type: DataType,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let location = self.location;
        Ok(match operation {
            ArrayOperation::Exp(_) => append(block, math::exp(values[0], location)?)?,
            ArrayOperation::ConvertElementType(_) => {
                let source = inputs[0].r#type.data_type();
                if source == data_type {
                    values[0]
                } else {
                    let source_width = match source {
                        DataType::F64 => 64,
                        DataType::F32 => 32,
                        _ => 16,
                    };
                    let target_width = match data_type {
                        DataType::F64 => 64,
                        DataType::F32 => 32,
                        _ => 16,
                    };
                    let target_type = element_type(self.context, data_type)?;
                    if source_width < target_width {
                        append(block, arith::extf(values[0], target_type, location)?)?
                    } else if source_width > target_width {
                        append(block, arith::truncf(values[0], target_type, location)?)?
                    } else {
                        // F16 and BF16 share a storage width but have different exponent and significand ranges.
                        let widened = append(block, arith::extf(values[0], self.context.float32_type(), location)?)?;
                        append(block, arith::truncf(widened, target_type, location)?)?
                    }
                }
            }
            ArrayOperation::Add(_) => self.add(block, data_type, values[0], values[1])?,
            ArrayOperation::Sub(_) if is_float(data_type) => {
                append(block, arith::subf(values[0], values[1], location)?)?
            }
            ArrayOperation::Sub(_) => append(block, arith::subi(values[0], values[1], location)?)?,
            ArrayOperation::Mul(_) if is_float(data_type) => {
                append(block, arith::mulf(values[0], values[1], location)?)?
            }
            ArrayOperation::Mul(_) => append(block, arith::muli(values[0], values[1], location)?)?,
            ArrayOperation::Div(_) if is_float(data_type) => {
                append(block, arith::divf(values[0], values[1], location)?)?
            }
            ArrayOperation::Neg(_) if is_float(data_type) => append(block, arith::negf(values[0], location)?)?,
            ArrayOperation::Neg(_) => {
                let zero = self.literal(block, data_type, 0)?;
                append(block, arith::subi(zero, values[0], location)?)?
            }
            ArrayOperation::And(_) => append(block, arith::andi(values[0], values[1], location)?)?,
            ArrayOperation::Or(_) => append(block, arith::ori(values[0], values[1], location)?)?,
            ArrayOperation::Xor(_) => append(block, arith::xori(values[0], values[1], location)?)?,
            ArrayOperation::Not(_) => {
                let ones = self.literal(block, data_type, u64::MAX)?;
                append(block, arith::xori(values[0], ones, location)?)?
            }
            ArrayOperation::Select(_) => append(block, arith::select(values[0], values[1], values[2], location)?)?,
            ArrayOperation::StopGradient(_) => values[0],
            ArrayOperation::Compare(compare) => {
                let direction = compare.direction();
                if is_float(inputs[0].r#type.data_type()) {
                    let predicate = match direction {
                        ComparisonDirection::Equal => arith::FloatingPointComparisonPredicate::Equal,
                        ComparisonDirection::NotEqual => arith::FloatingPointComparisonPredicate::NotEqual,
                        ComparisonDirection::LessThan => arith::FloatingPointComparisonPredicate::LessThan,
                        ComparisonDirection::LessThanOrEqual => {
                            arith::FloatingPointComparisonPredicate::LessThanOrEqual
                        }
                        ComparisonDirection::GreaterThan => arith::FloatingPointComparisonPredicate::GreaterThan,
                        ComparisonDirection::GreaterThanOrEqual => {
                            arith::FloatingPointComparisonPredicate::GreaterThanOrEqual
                        }
                    };
                    append(block, arith::cmpf(values[0], values[1], predicate, location)?)?
                } else {
                    let unsigned =
                        matches!(inputs[0].r#type.data_type(), DataType::U32 | DataType::U64 | DataType::Boolean);
                    let predicate = match (direction, unsigned) {
                        (ComparisonDirection::Equal, _) => arith::IntegerComparisonPredicate::Equal,
                        (ComparisonDirection::NotEqual, _) => arith::IntegerComparisonPredicate::NotEqual,
                        (ComparisonDirection::LessThan, false) => arith::IntegerComparisonPredicate::SignedLessThan,
                        (ComparisonDirection::LessThanOrEqual, false) => {
                            arith::IntegerComparisonPredicate::SignedLessThanOrEqual
                        }
                        (ComparisonDirection::GreaterThan, false) => {
                            arith::IntegerComparisonPredicate::SignedGreaterThan
                        }
                        (ComparisonDirection::GreaterThanOrEqual, false) => {
                            arith::IntegerComparisonPredicate::SignedGreaterThanOrEqual
                        }
                        (ComparisonDirection::LessThan, true) => arith::IntegerComparisonPredicate::UnsignedLessThan,
                        (ComparisonDirection::LessThanOrEqual, true) => {
                            arith::IntegerComparisonPredicate::UnsignedLessThanOrEqual
                        }
                        (ComparisonDirection::GreaterThan, true) => {
                            arith::IntegerComparisonPredicate::UnsignedGreaterThan
                        }
                        (ComparisonDirection::GreaterThanOrEqual, true) => {
                            arith::IntegerComparisonPredicate::UnsignedGreaterThanOrEqual
                        }
                    };
                    append(block, arith::cmpi(values[0], values[1], predicate, location)?)?
                }
            }
            _ => return Err(unsupported(operation, "operation has no baseline scalar implementation")),
        })
    }
}

/// Identifies scalar floating-point types admitted by the baseline.
fn is_float(data_type: DataType) -> bool {
    matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32 | DataType::F64)
}

/// Returns the exact encoding of one at the admitted scalar type.
fn one_bits(data_type: DataType) -> u64 {
    match data_type {
        DataType::F32 => 1.0f32.to_bits().into(),
        DataType::F64 => 1.0f64.to_bits(),
        _ => 1,
    }
}

/// Reports an unsupported semantic contract without discarding operation metadata.
fn unsupported(operation: &ArrayOperation<Array>, reason: &str) -> Error {
    Error::Unsupported { operation: operation.name(), reason: reason.to_owned() }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, KernelSchedule, VerifiedKernel,
        whole_array_parameter,
    };
    use ryft_core::{
        AddOperation, ArrayIrOperation, Context, ConvertElementTypeOperation, DotDimensionNumbers, DotOperation,
        ExpOperation, ReduceOperation, ReferenceRead, ReferenceWrite,
    };
    use ryft_mlir::{Context as MlirContext, Operation as MlirOperation, WalkOrder, WalkResult};

    use crate::kernels::gpu::{Compiler, Options, Target};

    use super::*;

    /// Lowers a canonically traced operation inside a real GPU module and records its floating arithmetic.
    fn lowered_arithmetic(operation: ArrayOperation<Array>, input_types: Vec<ArrayType>) -> Vec<String> {
        let output_type = operation.infer_output_types(&input_types, &[]).unwrap().remove(0);
        let mut parameters = input_types
            .into_iter()
            .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::ReadOnly).unwrap())
            .collect::<Vec<_>>();
        parameters.push(whole_array_parameter(output_type, KernelParameterAccess::WriteOnly).unwrap());
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            let input_count = references.len() - 1;
            let inputs = references[..input_count].iter().map(ReferenceRead::read).collect::<Result<Vec<_>, _>>()?;
            let value = references[input_count]
                .context()
                .bind(ArrayIrOperation::Array(operation), vec![], &inputs)?
                .remove(0);
            references[input_count].write(&value)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let context = MlirContext::new();
        let module = Compiler
            .module(&context, &kernel, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert!(module.verify().unwrap());
        let mut operations = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().to_string();
            if matches!(
                name.as_str(),
                "arith.addf" | "arith.mulf" | "arith.maximumf" | "math.exp" | "arith.extf" | "arith.truncf"
            ) {
                operations.push(name);
            }
            WalkResult::Advance
        });
        operations
    }

    #[test]
    fn test_validate() {
        let operation =
            ArrayOperation::Dot(DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0])));
        let inputs = [ArrayType::new_static(DataType::F32, [2, 3, 4]), ArrayType::new_static(DataType::F32, [2, 4, 5])];
        assert!(validate(&operation, &inputs, &ArrayType::new_static(DataType::F32, [2, 3, 5])).is_ok());
        let reduction = ArrayOperation::Reduce(ReduceOperation::new(vec![0, 2], ReductionKind::Sum));
        assert!(validate(&reduction, &inputs[..1], &ArrayType::new_static(DataType::F32, [3])).is_ok());
    }

    #[test]
    fn test_validate_mixed_element_types() {
        let operation = ArrayOperation::Add(AddOperation::new());
        assert!(matches!(
            validate(
                &operation,
                &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
                &ArrayType::scalar(DataType::F64),
            ),
            Err(Error::Unsupported { operation: "add", reason })
                if reason == "arithmetic operand element types must agree",
        ));
    }

    #[test]
    fn test_validate_empty_contraction() {
        let dot = ArrayOperation::Dot(DotOperation::matmul());
        let inputs = [ArrayType::new_static(DataType::F32, [2, 0]), ArrayType::new_static(DataType::F32, [0, 3])];
        assert!(validate(&dot, &inputs, &ArrayType::new_static(DataType::F32, [2, 3])).is_ok());
        let sum = ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
        assert!(
            validate(&sum, &[ArrayType::new_static(DataType::F32, [0])], &ArrayType::scalar(DataType::F32)).is_ok()
        );
    }

    #[test]
    fn test_validate_unsupported_reduction() {
        let operation = ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Mean));
        assert!(matches!(
            validate(&operation, &[ArrayType::new_static(DataType::F32, [2])], &ArrayType::scalar(DataType::F32)),
            Err(Error::Unsupported { operation: "reduce_mean", reason })
                if reason == "operation or its metadata has no baseline scalar implementation",
        ));
    }

    #[test]
    fn test_validate_inconsistent_result() {
        let operation = ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
        assert!(matches!(
            validate(
                &operation,
                &[ArrayType::new_static(DataType::F32, [2])],
                &ArrayType::new_static(DataType::F32, [1]),
            ),
            Err(Error::Unsupported { operation: "reduce_sum", reason })
                if reason == "result type differs from canonical array inference",
        ));
    }

    #[test]
    fn test_validate_invalid_axis() {
        let operation = ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum));
        let error =
            validate(&operation, &[ArrayType::new_static(DataType::F32, [2])], &ArrayType::scalar(DataType::F32))
                .unwrap_err();
        assert!(matches!(error, Error::Type(_)));
        assert_eq!(error.to_string(), "`reduce_sum` axis 1 is out of bounds for rank 1");
    }

    #[test]
    fn test_array() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Add(AddOperation::new()),
                vec![ArrayType::new_static(DataType::F32, [2, 1]), ArrayType::new_static(DataType::F32, [1, 3])],
            ),
            vec!["arith.addf"],
        );
    }

    #[test]
    fn test_array_reduce() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0, 2], ReductionKind::Sum)),
                vec![ArrayType::new_static(DataType::F32, [2, 3, 4])],
            ),
            vec!["arith.addf"],
        );
    }

    #[test]
    fn test_array_reduce_max() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Max)),
                vec![ArrayType::new_static(DataType::F32, [2, 3])],
            ),
            vec!["arith.maximumf"]
        );
    }

    #[test]
    fn test_array_exp() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Exp(ExpOperation::new()),
                vec![ArrayType::new_static(DataType::F32, [2, 3])]
            ),
            vec!["math.exp"]
        );
    }

    #[test]
    fn test_array_convert_element_type() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::F16, false)),
                vec![ArrayType::new_static(DataType::F32, [2, 3])],
            ),
            vec!["arith.truncf"]
        );
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::F32, false)),
                vec![ArrayType::new_static(DataType::BF16, [2, 3])],
            ),
            vec!["arith.extf"]
        );
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::BF16, false)),
                vec![ArrayType::new_static(DataType::F16, [2, 3])],
            ),
            vec!["arith.extf", "arith.truncf"]
        );
    }

    #[test]
    fn test_array_reduce_identity() {
        // An empty axis list copies bits; adding zero would change negative zeros and signaling NaNs.
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Reduce(ReduceOperation::new(vec![], ReductionKind::Sum)),
                vec![ArrayType::new_static(DataType::F32, [2])],
            ),
            Vec::<String>::new(),
        );
    }

    #[test]
    fn test_array_dot() {
        assert_eq!(
            lowered_arithmetic(
                ArrayOperation::Dot(DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))),
                vec![ArrayType::new_static(DataType::F32, [2, 3, 4]), ArrayType::new_static(DataType::F32, [2, 4, 5])],
            ),
            vec!["arith.mulf", "arith.addf"],
        );
    }

    #[test]
    fn test_checked_count() {
        let operation = ArrayOperation::Dot(DotOperation::matmul());
        assert_eq!(checked_count(&operation, &[2, 3]).unwrap(), 6);
        assert_eq!(checked_count(&operation, &[2, 0]).unwrap(), 0);
        assert!(matches!(checked_count(&operation, &[i64::MAX as usize, 2]), Err(Error::Unsupported { .. })));
        assert!(matches!(checked_count(&operation, &[0, usize::MAX]), Err(Error::Unsupported { .. })));
    }

    #[test]
    fn test_one_bits() {
        assert_eq!(one_bits(DataType::F32), u64::from(1.0f32.to_bits()));
        assert_eq!(one_bits(DataType::F64), 1.0f64.to_bits());
        assert_eq!(one_bits(DataType::I64), 1);
        assert_eq!(one_bits(DataType::Boolean), 1);
    }
}
