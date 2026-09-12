//! Reference [`Array`] kernels for the mathematics operation family contracts.
//!
//! Kernels use the scalar arithmetic contracts in [`crate::arrays::elements`], decode operands through their
//! physical addressing, and materialize owned results. Element data type promotion and broadcasting follow the
//! corresponding operations' type-inference rules.

use std::sync::Arc;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::elements::{
    ArrayElement, FloatingPointArrayElement, NumericArrayElement, RealArrayElement, RealFloatingPointArrayElement,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::collectives::decode_nonnegative_integer_metadata;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::macros::impl_array_elementwise_operation;
use crate::operations::math::log_sum_exp::{log_sum_exp_abstract, validate_log_sum_exp_data_type};
use crate::operations::{
    Add, Atan2, Ceil, ConvertElementType, Dot, DotDimensionNumbers, DotOperation, Erf, Floor, Log, Log1p, LogAddExp,
    LogSumExp, Logistic, Pow, RAGGED_DOT_OPERATION_NAME, RaggedDot, RaggedDotDimensionNumbers, RaggedDotMode,
    RaggedDotOperation, Rem, Reshape, Round, Rsqrt, Sign, Slice, Sqrt,
};
use crate::programs::{Operation, ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this.

impl Array {
    /// Computes one numerically stable `log(sum(exp(x)))` directly over typed elements, following the guarded
    /// construction that [`LogSumExpOperation`](crate::operations::math::LogSumExpOperation) documents: a maximum
    /// reduction whose identity is the element type's own lowest value, that maximum replaced by zero wherever it is
    /// not finite, and then `log(sum(exp(x - safe_maximum))) + safe_maximum`. Every intermediate is held in the
    /// element's own encoding, so the result matches what the equivalent staged program computes.
    fn log_sum_exp_elements<T: FloatingPointArrayElement>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(output_type.data_type(), T::data_type());
        let zero = T::zero()?;
        let mut maximums = self.reduce_elements::<T>(output_type.clone(), axes, T::max_identity(), |left, right| {
            Ok(ArrayElement::max(&left, &right))
        })?;
        maximums.map_elements_in_place::<T>(|value| {
            Ok(if value.convert_to::<f64>()?.is_finite() { value } else { zero })
        })?;

        let input_shape = self.r#type().static_shape().unwrap();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut reduce_mask = vec![false; input_shape.rank()];
        axes.iter().for_each(|axis| reduce_mask[*axis] = true);

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        for output in 0..output_addressing.element_count() {
            zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output)]);
        }

        let mut input_index = vec![0usize; input_shape.rank()];
        let mut output_index = vec![0usize; output_type.rank()];
        for _ in 0..input_addressing.element_count() {
            let mut output_axis = 0usize;
            for axis in 0..input_shape.rank() {
                if !reduce_mask[axis] {
                    output_index[output_axis] = input_index[axis];
                    output_axis += 1;
                }
            }
            let input_value = T::decode(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            let output_range = output_addressing.byte_range_unchecked(&output_index);
            let maximum = T::decode(&maximums.storage_bytes()[output_range.clone()]);
            let shifted = input_value.sub(maximum)?.exp()?;
            let sum = T::decode(&bytes[output_range.clone()]).add(shifted)?;
            sum.encode(&mut bytes[output_range]);
            input_addressing.advance_index(&mut input_index);
        }

        for output in 0..output_addressing.element_count() {
            let range = output_addressing.byte_range_for_flat_index(output);
            let maximum = T::decode(&maximums.storage_bytes()[range.clone()]);
            let value = T::decode(&bytes[range.clone()]).log()?.add(maximum)?;
            value.encode(&mut bytes[range]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Allocates an array whose logical elements are initialized to the additive identity.
    fn zeroed<T: ArrayElement>(output_type: ArrayType) -> Result<Self, ProgramError> {
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let zero = T::zero()?;
        for element in 0..output_addressing.element_count() {
            zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(element)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Evaluates grouped generalized dot extent-exactly. Each concrete group's raw cumulative interval is clipped to
    /// the physical ragged extent, the resulting pair of operand slices is contracted by the ordinary generalized-dot
    /// kernel, and the result is written into its output window. This keeps temporary storage proportional to one
    /// group rather than the whole operand times the group count.
    fn ragged_dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let mut output_types = RaggedDotOperation::new(dimensions.clone()).infer_output_types(
            &[self.r#type().into_owned(), rhs.r#type().into_owned(), group_sizes.r#type().into_owned()],
            &[],
        )?;
        let output_type = output_types.remove(0);
        let dot_dimensions = dimensions.dot_dimensions();
        let ragged_axis = dimensions.lhs_ragged_dimensions()[0];
        let mode = dimensions.mode(self.r#type().rank())?;
        if mode == RaggedDotMode::Batch {
            return self.dot_elements::<T>(rhs, dot_dimensions);
        }
        let prefix_axes = dimensions.group_sizes_prefix_dimensions(self.r#type().rank())?;
        let prefix_shape = prefix_axes
            .iter()
            .map(|axis| self.r#type().shape().dimensions()[*axis].value().unwrap())
            .collect::<Vec<_>>();
        let prefix_count = prefix_shape.iter().product::<usize>();
        let group_count = group_sizes.r#type().shape().dimensions().last().unwrap().value().ok_or_else(|| {
            ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` requires a static group count for eager evaluation"),
            }
        })?;
        let sizes = decode_nonnegative_integer_metadata(group_sizes, RAGGED_DOT_OPERATION_NAME, "group_sizes")?;
        let expected_size_count = if group_sizes.r#type().rank() == 1 {
            group_count
        } else {
            prefix_count.checked_mul(group_count).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes element count does not fit in `usize`"),
            })?
        };
        if sizes.len() != expected_size_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes storage does not match its shape"),
            });
        }
        let ragged_extent = self.r#type().shape().dimensions()[ragged_axis].value().unwrap();
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let lhs_strides = vec![1; lhs_shape.rank()];
        let rhs_strides = vec![1; rhs_shape.rank()];
        let output_strides = vec![1; output_type.rank()];
        let lhs_result = crate::operations::dot::lhs_result_axes(dot_dimensions, self.r#type().rank());
        let non_contracting_metadata = (mode == RaggedDotMode::NonContracting).then(|| {
            let rhs_group_axis = dimensions.rhs_group_dimensions()[0];
            let rhs_slice_shape = Shape::new(
                rhs_shape
                    .dimensions()
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, dimension)| {
                        (axis != rhs_group_axis).then(|| {
                            let is_prefix_axis = dot_dimensions
                                .lhs_batching_dimensions()
                                .iter()
                                .zip(dot_dimensions.rhs_batching_dimensions())
                                .any(|(lhs_axis, rhs_axis)| *rhs_axis == axis && prefix_axes.contains(lhs_axis));
                            Dimension::Static(if is_prefix_axis { 1 } else { *dimension })
                        })
                    })
                    .collect(),
            );
            let remap_rhs_axis = |axis: usize| if axis < rhs_group_axis { axis } else { axis - 1 };
            let dense_dimensions = DotDimensionNumbers::new(
                dot_dimensions.lhs_contracting_dimensions().to_vec(),
                dot_dimensions.rhs_contracting_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
                dot_dimensions.lhs_batching_dimensions().to_vec(),
                dot_dimensions.rhs_batching_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
            );
            let ragged_position = lhs_result.iter().position(|axis| *axis == ragged_axis).unwrap();
            let ragged_output_axis = dot_dimensions.lhs_batching_dimensions().len() + ragged_position;
            (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis)
        });
        let contracting_rhs_ragged_axis = (mode == RaggedDotMode::Contracting).then(|| {
            let contracting_position =
                dot_dimensions.lhs_contracting_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
            dot_dimensions.rhs_contracting_dimensions()[contracting_position]
        });
        let mut output = Self::zeroed::<T>(output_type)?;
        for prefix in 0..prefix_count {
            let mut remainder = prefix;
            let mut prefix_coordinates = vec![0; prefix_axes.len()];
            for (coordinate, extent) in prefix_coordinates.iter_mut().zip(prefix_shape.iter()).rev() {
                *coordinate = remainder % extent;
                remainder /= extent;
            }
            let metadata_prefix = if group_sizes.r#type().rank() == 1 { 0 } else { prefix };
            let group_range = metadata_prefix * group_count..(metadata_prefix + 1) * group_count;
            let mut lhs_starts = vec![0; lhs_shape.rank()];
            let mut lhs_limits = lhs_shape.dimensions().to_vec();
            for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                lhs_starts[axis] = coordinate;
                lhs_limits[axis] = coordinate + 1;
            }
            let mut rhs_starts = vec![0; rhs_shape.rank()];
            let mut rhs_limits = rhs_shape.dimensions().to_vec();
            for (&lhs_axis, &rhs_axis) in
                dot_dimensions.lhs_batching_dimensions().iter().zip(dot_dimensions.rhs_batching_dimensions())
            {
                if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                    let coordinate = prefix_coordinates[prefix_position];
                    rhs_starts[rhs_axis] = coordinate;
                    rhs_limits[rhs_axis] = coordinate + 1;
                }
            }
            if mode == RaggedDotMode::Contracting {
                for (&lhs_axis, &rhs_axis) in
                    dot_dimensions.lhs_contracting_dimensions().iter().zip(dot_dimensions.rhs_contracting_dimensions())
                {
                    if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                        let coordinate = prefix_coordinates[prefix_position];
                        rhs_starts[rhs_axis] = coordinate;
                        rhs_limits[rhs_axis] = coordinate + 1;
                    }
                }
            }
            let mut output_starts = vec![0; output.r#type().rank()];
            let mut output_limits = output.r#type().static_shape().unwrap().dimensions().to_vec();
            match mode {
                RaggedDotMode::NonContracting => {
                    for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                        if let Some(position) =
                            dot_dimensions.lhs_batching_dimensions().iter().position(|candidate| *candidate == axis)
                        {
                            output_starts[position] = coordinate;
                        } else {
                            let position = lhs_result.iter().position(|candidate| *candidate == axis).unwrap();
                            let position = dot_dimensions.lhs_batching_dimensions().len() + position;
                            output_starts[position] = coordinate;
                        }
                    }
                }
                RaggedDotMode::Contracting => {
                    for (position, lhs_axis) in dot_dimensions.lhs_batching_dimensions().iter().enumerate() {
                        if let Some(prefix_position) = prefix_axes.iter().position(|axis| axis == lhs_axis) {
                            output_starts[position + 1] = prefix_coordinates[prefix_position];
                            output_limits[position + 1] = prefix_coordinates[prefix_position] + 1;
                        }
                    }
                }
                RaggedDotMode::Batch => unreachable!(),
            }
            let mut raw_ragged_start = 0usize;
            for (group, &group_size) in sizes[group_range].iter().enumerate() {
                if raw_ragged_start >= ragged_extent {
                    break;
                }
                let ragged_start = raw_ragged_start;
                let ragged_limit = raw_ragged_start.saturating_add(group_size).min(ragged_extent);
                raw_ragged_start = ragged_limit;
                if ragged_start == ragged_limit {
                    continue;
                }
                lhs_starts[ragged_axis] = ragged_start;
                lhs_limits[ragged_axis] = ragged_limit;
                let lhs_slice = self.slice(&lhs_starts, &lhs_limits, &lhs_strides)?;
                let dot = match mode {
                    RaggedDotMode::NonContracting => {
                        let (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis) =
                            non_contracting_metadata.as_ref().unwrap();
                        rhs_starts[*rhs_group_axis] = group;
                        rhs_limits[*rhs_group_axis] = group + 1;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        let rhs_slice = rhs_slice.reshape(rhs_slice_shape.clone())?;
                        output_starts[*ragged_output_axis] = ragged_start;
                        lhs_slice.dot_elements::<T>(&rhs_slice, dense_dimensions)?
                    }
                    RaggedDotMode::Contracting => {
                        let rhs_ragged_axis = contracting_rhs_ragged_axis.unwrap();
                        rhs_starts[rhs_ragged_axis] = ragged_start;
                        rhs_limits[rhs_ragged_axis] = ragged_limit;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        output_starts[0] = group;
                        output_limits[0] = group + 1;
                        let dot = lhs_slice.dot_elements::<T>(&rhs_slice, dot_dimensions)?;
                        let mut dimensions = vec![Dimension::Static(1)];
                        dimensions.extend_from_slice(dot.r#type().shape().dimensions());
                        let dot = dot.reshape(Shape::new(dimensions))?;
                        let current = output.slice(&output_starts, &output_limits, &output_strides)?;
                        Add::add(&current, &dot)?
                    }
                    RaggedDotMode::Batch => unreachable!(),
                };
                output = output.replace_block(&dot, &output_starts);
            }
        }
        Ok(output)
    }

    fn dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(rhs.r#type().data_type(), T::data_type());
        let mut output_types = DotOperation::new(dimensions.clone())
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(0);
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let lhs_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let rhs_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;

        let lhs_batching = dimensions.lhs_batching_dimensions();
        let rhs_batching = dimensions.rhs_batching_dimensions();
        let lhs_contracting = dimensions.lhs_contracting_dimensions();
        let rhs_contracting = dimensions.rhs_contracting_dimensions();
        let lhs_result = (0..lhs_shape.rank())
            .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let rhs_result = (0..rhs_shape.rank())
            .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let contracting_shape =
            StaticShape::new(lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect::<Vec<_>>());
        let contracting_strides = contracting_shape.row_major_strides();
        let contracting_count = contracting_shape.dimensions().iter().product();

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut lhs_index = vec![0usize; lhs_shape.rank()];
        let mut rhs_index = vec![0usize; rhs_shape.rank()];
        for output_flat in 0..output_addressing.element_count() {
            // Decode the result coordinate directly into the corresponding batch and non-contracting operand axes.
            let mut output_axis = 0usize;
            for (&lhs_axis, &rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
                let coordinate = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                lhs_index[lhs_axis] = coordinate;
                rhs_index[rhs_axis] = coordinate;
                output_axis += 1;
            }
            for &lhs_axis in &lhs_result {
                lhs_index[lhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }
            for &rhs_axis in &rhs_result {
                rhs_index[rhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }

            let mut accumulator = if T::data_type() == DataType::F8E8M0FNU { None } else { Some(T::zero()?) };
            for contracting_flat in 0..contracting_count {
                for (contracting_axis, (&lhs_axis, &rhs_axis)) in
                    lhs_contracting.iter().zip(rhs_contracting).enumerate()
                {
                    let coordinate = (contracting_flat / contracting_strides[contracting_axis])
                        % contracting_shape[contracting_axis];
                    lhs_index[lhs_axis] = coordinate;
                    rhs_index[rhs_axis] = coordinate;
                }
                let lhs_value = T::decode(&self.storage_bytes()[lhs_addressing.byte_range_unchecked(&lhs_index)]);
                let rhs_value = T::decode(&rhs.storage_bytes()[rhs_addressing.byte_range_unchecked(&rhs_index)]);
                let product = lhs_value.mul(rhs_value)?;
                accumulator = Some(match accumulator {
                    Some(accumulator) => accumulator.add(product)?,
                    None => product,
                });
            }
            let accumulator = match accumulator {
                Some(accumulator) => accumulator,
                None => T::zero()?,
            };
            accumulator.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output_flat)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl_array_elementwise_operation!(
    @binary
    Atan2, atan2,
    operation = "atan2",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| FloatingPointArrayElement::atan2(lhs, rhs),
);

impl_array_elementwise_operation!(
    @unary
    Log, log,
    operation = "log",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::log(input),
);

impl_array_elementwise_operation!(
    @unary
    Log1p, log1p,
    operation = "log1p",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::log1p(input),
);

impl_array_elementwise_operation!(
    @binary
    LogAddExp, log_add_exp,
    operation = "log_add_exp",
    inputs = @float @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| RealFloatingPointArrayElement::log_add_exp(lhs, rhs),
);

impl_array_elementwise_operation!(
    @unary
    Sqrt, sqrt,
    operation = "sqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::sqrt(input),
);

impl_array_elementwise_operation!(
    @unary
    Rsqrt, rsqrt,
    operation = "rsqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::rsqrt(input),
);

impl_array_elementwise_operation!(
    @unary
    Logistic, logistic,
    operation = "logistic",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::logistic(input),
);

impl_array_elementwise_operation!(
    @unary
    Erf, erf,
    operation = "erf",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::erf(input),
);

impl_array_elementwise_operation!(
    @binary
    Pow, pow,
    operation = "pow",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| FloatingPointArrayElement::pow(lhs, rhs),
);

impl Sign for Array {
    fn sign(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_signed() && !data_type.is_floating_point() && !data_type.is_complex() {
            return Err(TypeError::invalid(format!(
                "cannot compute the sign of a scalar of data type `{}`",
                data_type,
            ))
            .into());
        }
        if data_type.is_signed() {
            dispatch_on_array_element_type!(@signed data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else if data_type.is_complex() {
            dispatch_on_array_element_type!(@complex data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else {
            dispatch_on_array_element_type!(@float data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        }
    }
}

impl_array_elementwise_operation!(
    @unary
    Floor, floor,
    operation = "floor",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::floor(input),
);

impl_array_elementwise_operation!(
    @unary
    Ceil, ceil,
    operation = "ceil",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::ceil(input),
);

impl_array_elementwise_operation!(
    @unary
    Round, round,
    operation = "round",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::round(input),
);

impl_array_elementwise_operation!(
    @binary
    Rem, rem,
    operation = "rem",
    inputs = @numeric @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| RealArrayElement::rem(lhs, rhs),
);

impl Dot for Array {
    /// Computes an accumulation-typed dot by upcasting both operands to `accumulation_type` and delegating to the
    /// ordinary evaluator, which is exactly the upcast-then-accumulate contract of
    /// [`DotOperation::with_accumulation_type`].
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self {
        let lhs = self.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        let rhs = rhs.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        lhs.dot(&rhs, dimensions)
    }

    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        // TODO(eaplatanios): What about the accumulation type?
        let data_type = self.r#type().data_type();
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.dot_elements::<Element>(rhs, dimensions)
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl RaggedDot for Array {
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.ragged_dot_elements::<Element>(rhs, group_sizes, dimensions)
        })
    }
}

impl LogSumExp for Array {
    fn log_sum_exp(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        // Reducing along no axes is the identity, but only for the operands this primitive accepts at all, so the
        // element data type is validated before the shortcut is taken.
        if axes.is_empty() {
            validate_log_sum_exp_data_type(self.r#type().data_type())?;
            return Ok(self.clone());
        }
        // Reuse the abstract rule for validation and for the complete result metadata. The concrete kernel below then
        // decodes directly from the input's physical layout into the result's addressed storage.
        let output_type = log_sum_exp_abstract(self.r#type().as_ref(), axes)?;
        let data_type = output_type.data_type();
        dispatch_on_array_element_type!(@float data_type, |Element| {
            self.log_sum_exp_elements::<Element>(output_type, axes)
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::f16;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::elements::{i2, i4};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_low_precision_float_arithmetic() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![1.0, 2.0]).unwrap();
        let right = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![0.5, 0.25]).unwrap();
        assert_eq!(left.rem(&right).unwrap().to_f64s(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_array_math() {
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, 4.0]).unwrap().sqrt().unwrap(),
            Array::vector(vec![1.0, 2.0]).unwrap(),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, std::f64::consts::E]).unwrap().log().unwrap(),
            Array::vector(vec![0.0, 1.0]).unwrap(),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0]).unwrap().atan2(&Array::vector(vec![1.0]).unwrap()).unwrap(),
            Array::vector(vec![std::f64::consts::FRAC_PI_4]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_transcendental_math_uses_typed_storage() {
        // Binary kernels perform complete broadcasting after promoting both physical inputs to their common type.
        let left_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[0.0f32, 1.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[1.0f64, 1.0, -1.0]).unwrap();
        let angles = left.atan2(&right).unwrap();
        assert_eq!(angles.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
        assert_abs_diff_eq!(
            angles,
            Array::matrix(
                2,
                3,
                vec![
                    0.0,
                    0.0,
                    std::f64::consts::PI,
                    std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                    3.0 * std::f64::consts::FRAC_PI_4
                ],
            )
            .unwrap(),
            epsilon = 1e-12,
        );
        let bases = Array::matrix(2, 1, vec![2.0f32, 3.0]).unwrap();
        let exponents = Array::matrix(1, 3, vec![1.0f64, 2.0, 3.0]).unwrap();
        assert_eq!(
            bases.pow(&exponents).unwrap(),
            Array::matrix(2, 3, vec![2.0f64, 4.0, 8.0, 3.0, 9.0, 27.0]).unwrap(),
        );
        assert!(matches!(
            Array::scalar(1i32).unwrap().atan2(&Array::scalar(1.0f64).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`atan2` does not support input data type `i32`",
        ));
        assert!(matches!(
            Array::scalar(2.0f64).unwrap().pow(&Array::scalar(3i32).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`pow` does not support input data type `i32`",
        ));
    }

    #[test]
    fn test_array_real_float_math_uses_typed_storage() {
        let input = Array::vector(vec![-1.5f64, -0.0, 2.5, 3.5]).unwrap();
        assert_eq!(input.floor().unwrap(), Array::vector(vec![-2.0, -0.0, 2.0, 3.0]).unwrap());
        assert_eq!(input.ceil().unwrap(), Array::vector(vec![-1.0, -0.0, 3.0, 4.0]).unwrap());
        assert_eq!(input.round().unwrap(), Array::vector(vec![-2.0, -0.0, 2.0, 4.0]).unwrap());
        assert_eq!(Array::vector(vec![1.0f64, 4.0]).unwrap().rsqrt().unwrap(), Array::vector(vec![1.0, 0.5]).unwrap());
        assert_abs_diff_eq!(
            Array::vector(vec![-1.0f64, 0.0, 1.0]).unwrap().erf().unwrap(),
            Array::vector(vec![-0.8427007929497149, 0.0, 0.8427007929497149]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_dot() {
        // Ordinary matrix multiplication uses the generalized contraction order.
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        let product = lhs.dot(&rhs, &dimensions);
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(product.to_f64s(), vec![58.0, 64.0, 139.0, 154.0]);

        // Both operands are decoded through their physical layouts rather than through dense logical payload copies.
        let lhs_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let rhs_type =
            ArrayType::new_static(DataType::U16, [3, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, -2])));
        let lhs = Array::from_elements(lhs_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let rhs = Array::from_elements(rhs_type, &[7u16, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<u16>(), Ok(vec![58, 64, 139, 154]));

        // Batched generalized contraction places batch axes before both operands' non-contracting axes.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 2]), &[1i32, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2i32, 3, 4, 5]).unwrap();
        let batched = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let product = lhs.dot(&rhs, &batched);
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::I32, [2, 2, 1]));
        assert_eq!(product.elements::<i32>(), Ok(vec![8, 18, 50, 68]));

        // Narrow integer products and sums wrap at the declared element width, and complex accumulation retains both
        // components.
        let lhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [1, 2]),
            &[i4::new(7).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let rhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [2, 1]),
            &[i4::new(2).unwrap(), i4::new(2).unwrap()],
        )
        .unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<i4>(), Ok(vec![i4::new(-4).unwrap()]));
        let lhs = Array::matrix(1, 2, vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(3.0, -1.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![ComplexNumber::new(2.0f32, -1.0), ComplexNumber::new(0.5, 4.0)]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &dimensions).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(9.5, 14.5)]),
        );

        // Preferred accumulation first promotes both inputs and then runs the same typed contraction at the wider
        // element data type.
        let lhs = Array::matrix(1, 2, vec![f16::from_f32(1.5), f16::from_f32(2.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![f16::from_f32(2.0), f16::from_f32(3.0)]).unwrap();
        let product = lhs.dot_with_accumulation_type(&rhs, &dimensions, DataType::F32);
        assert_eq!(product.r#type().data_type(), DataType::F32);
        assert_eq!(product.elements::<f32>(), Ok(vec![9.0]));

        // An empty contracting dimension materializes one additive identity for every result coordinate.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<f32>(), Ok(vec![0.0; 6]));
    }

    #[test]
    fn test_array_complex_math() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let expect = |values: [ComplexNumber<f64>; 2]| Array::vector(values.to_vec()).unwrap();
        assert_abs_diff_eq!(left.log().unwrap(), expect([left_values[0].ln(), left_values[1].ln()]), epsilon = 1e-12);
        assert_abs_diff_eq!(
            left.sqrt().unwrap(),
            expect([left_values[0].sqrt(), left_values[1].sqrt()]),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_integer_semantics() {
        // Remainder by zero returns a structured error.
        assert!(matches!(
            Array::vector(vec![1u8]).unwrap().rem(&Array::vector(vec![0u8]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "cannot compute the remainder of an integer scalar of data type `u8` with a zero divisor",
        ));
    }

    #[test]
    fn test_array_sign() {
        // Sign preserves IEEE signed zero and NaN behavior and also covers signed sub-byte integers.
        let signs = Array::from_elements(
            ArrayType::new_static(DataType::F64, [4]),
            &[-2.0f64, -0.0, 0.0, f64::from_bits(0x7ff8_0000_0000_1234)],
        )
        .unwrap()
        .sign()
        .unwrap()
        .elements::<f64>()
        .unwrap();
        assert_eq!(signs[0], -1.0);
        assert_eq!(signs[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(signs[2].to_bits(), 0.0f64.to_bits());
        assert_eq!(signs[3].to_bits(), 0x7ff8_0000_0000_1234);
        let narrow =
            Array::from_elements(ArrayType::new_static(DataType::I2, [3]), &[i2::MIN, i2::new(0).unwrap(), i2::MAX])
                .unwrap();
        assert_eq!(
            narrow.sign().unwrap().elements::<i2>(),
            Ok(vec![i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]),
        );
        assert!(matches!(
            Array::scalar(1u8).unwrap().sign(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot compute the sign of a scalar of data type `u8`",
        ));
    }
}
