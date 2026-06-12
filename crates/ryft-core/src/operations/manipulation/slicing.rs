use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Shape, Size, Type, TypeError};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`SliceOperation`].
pub const SLICE_OPERATION_NAME: &'static str = "slice";

/// Canonical operation name for [`UpdateSliceOperation`].
pub const UPDATE_SLICE_OPERATION_NAME: &'static str = "update_slice";

/// Canonical operation name for [`DynamicSliceOperation`].
pub const DYNAMIC_SLICE_OPERATION_NAME: &'static str = "dynamic_slice";

/// Canonical operation name for [`DynamicUpdateSliceOperation`].
pub const DYNAMIC_UPDATE_SLICE_OPERATION_NAME: &'static str = "dynamic_update_slice";

/// Returns `true` when `data_type` is a signed or unsigned integer type and can therefore carry a slice start index.
fn is_integer(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64
            | DataType::U1
            | DataType::U2
            | DataType::U4
            | DataType::U8
            | DataType::U16
            | DataType::U32
            | DataType::U64,
    )
}

/// Validates the scalar integer start-index operand types of a dynamic slicing operation. Each index type must be a
/// rank-0 integer type, and all indices must share one integer type. The `operation_name` parameter selects the
/// reported operation name because this helper serves both [`DynamicSliceOperation`] and
/// [`DynamicUpdateSliceOperation`].
fn validate_start_index_types(operation_name: &'static str, index_types: &[&ArrayType]) -> Result<(), ProgramError> {
    for (index, index_type) in index_types.iter().enumerate() {
        if index_type.rank() != 0 || !is_integer(index_type.data_type()) {
            return Err(TypeError {
                message: format!(
                    "{operation_name} start index {index} must be a scalar integer but has type {index_type}",
                ),
            }
            .into());
        }
        if index_type.data_type() != index_types[0].data_type() {
            return Err(TypeError {
                message: format!(
                    "{operation_name} start indices must share one integer type but index {index} has type \
                    {index_type} and index 0 has type {first}",
                    first = index_types[0],
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// [`Operation`] that extracts a (possibly strided) sub-array from its input using static start, limit, and stride
/// values. Refer to the documentation of [`Slice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceOperation {
    /// Inclusive start index for each input axis.
    start_indices: Vec<usize>,

    /// Exclusive limit index for each input axis.
    limit_indices: Vec<usize>,

    /// Stride for each input axis (every stride is at least `1`).
    strides: Vec<usize>,
}

impl SliceOperation {
    /// Creates a new [`SliceOperation`] with the provided start and limit indices and unit strides. Use
    /// [`with_strides`](Self::with_strides) to attach non-unit strides.
    #[inline]
    pub fn new(start_indices: Vec<usize>, limit_indices: Vec<usize>) -> Self {
        let strides = vec![1; start_indices.len()];
        Self { start_indices, limit_indices, strides }
    }

    /// Replaces the strides of this [`SliceOperation`] with `strides`. There must be one stride per start index and
    /// every stride must be at least `1`.
    pub fn with_strides(mut self, strides: Vec<usize>) -> Result<Self, ProgramError> {
        if strides.len() != self.start_indices.len() {
            return Err(TypeError {
                message: format!(
                    "slice strides has length {} but start_indices has length {}",
                    strides.len(),
                    self.start_indices.len(),
                ),
            }
            .into());
        }
        if let Some(axis) = strides.iter().position(|stride| *stride == 0) {
            return Err(TypeError {
                message: format!("slice strides must be at least 1 but axis {axis} has stride 0"),
            }
            .into());
        }
        self.strides = strides;
        Ok(self)
    }

    /// Returns the inclusive start indices of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[usize] {
        self.start_indices.as_slice()
    }

    /// Returns the exclusive limit indices of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn limit_indices(&self) -> &[usize] {
        self.limit_indices.as_slice()
    }

    /// Returns the strides of this [`SliceOperation`], one per input axis.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }
}

impl Display for SliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for SliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("start_indices", format_args!("{:?}", self.start_indices))?;
            operation.field("limit_indices", format_args!("{:?}", self.limit_indices))?;
            if self.strides.iter().any(|stride| *stride != 1) {
                operation.field("strides", format_args!("{:?}", self.strides))?;
            }
            Ok(())
        })
    }
}

impl<V: Value<ArrayType> + Slice<Output = V>> InterpretableOperation<ArrayType, V> for SliceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().slice(
            self.start_indices.as_slice(),
            self.limit_indices.as_slice(),
            self.strides.as_slice(),
        )?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`SliceOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`SliceOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsSlice<T: Type> {
    /// Constructs an instance of [`SliceOperation`] for this [`Operation`] type with the provided start indices,
    /// limit indices, and strides.
    fn slice_operation(start_indices: Vec<usize>, limit_indices: Vec<usize>, strides: Vec<usize>) -> Self;
}

/// Represents the ability to extract a (possibly strided) sub-array using static start, limit, and stride values.
/// This is the direct analogue of the StableHLO [`slice`](https://openxla.org/stablehlo/spec#slice) operation and
/// JAX's [`lax.slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html).
///
/// `t.slice(start_indices, limit_indices, strides)` returns the sub-array whose element at index `i` is the input
/// element at index `start_indices + i * strides`, with output dimension
/// `ceil((limit_indices[d] - start_indices[d]) / strides[d])` along each axis `d` (an axis with
/// `start_indices[d] == limit_indices[d]` is empty). All three slices must have length equal to the input rank, and
/// each axis must satisfy `start_indices[d] <= limit_indices[d] <= input_dimension[d]` and `strides[d] >= 1`. Slicing
/// requires static input extents: inputs with dynamic dimensions are rejected because the bounds cannot be validated
/// against an unknown extent.
///
/// # Example
///
/// The following example shows how to use [`Slice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Slice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Slice the middle 1x2 block out of a 2x3 matrix. This is equivalent to
/// // `jax.lax.slice(x, start_indices=(1, 1), limit_indices=(2, 3))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.slice(&[1, 1], &[2, 3], &[1, 1])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.values, vec![5.0, 6.0]);
///
/// // A non-unit stride keeps every other element, like `x[0:6:2]` in NumPy.
/// let x = Array::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = x.slice(&[1], &[6], &[2])?;
/// assert_eq!(y.values, vec![1.0, 3.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub trait Slice {
    /// Output type of the slice operation.
    type Output;

    /// Slices `self` between `start_indices` and `limit_indices` with `strides`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: Inclusive start index for each input axis.
    ///   - `limit_indices`: Exclusive limit index for each input axis.
    ///   - `strides`: Stride for each input axis (every stride must be at least `1`).
    fn slice(
        self,
        start_indices: &[usize],
        limit_indices: &[usize],
        strides: &[usize],
    ) -> Result<Self::Output, ProgramError>;
}

impl Slice for &ArrayType {
    type Output = ArrayType;

    fn slice(
        self,
        start_indices: &[usize],
        limit_indices: &[usize],
        strides: &[usize],
    ) -> Result<ArrayType, ProgramError> {
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!("slice start_indices has length {} but input has rank {rank}", start_indices.len(),),
            }
            .into());
        }
        if limit_indices.len() != rank {
            return Err(TypeError {
                message: format!("slice limit_indices has length {} but input has rank {rank}", limit_indices.len(),),
            }
            .into());
        }
        if strides.len() != rank {
            return Err(TypeError {
                message: format!("slice strides has length {} but input has rank {rank}", strides.len()),
            }
            .into());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for (axis, ((&start, &limit), &stride)) in
            start_indices.iter().zip(limit_indices.iter()).zip(strides.iter()).enumerate()
        {
            let dimension = self.dimension(axis as isize);
            let Size::Static(size) = dimension else {
                return Err(TypeError {
                    message: format!(
                        "slice does not support dynamic input axis {axis} with size {dimension}; slice bounds \
                        cannot be validated against an unknown extent",
                    ),
                }
                .into());
            };
            if stride == 0 {
                return Err(TypeError {
                    message: format!("slice strides must be at least 1 but axis {axis} has stride 0"),
                }
                .into());
            }
            if start > limit {
                return Err(TypeError {
                    message: format!("slice start index {start} is greater than limit index {limit} at axis {axis}"),
                }
                .into());
            }
            if limit > size {
                return Err(TypeError {
                    message: format!("slice limit index {limit} is out of bounds for axis {axis} with size {size}"),
                }
                .into());
            }
            output_dimensions.push(Size::Static((limit - start).div_ceil(stride)));
        }
        Ok(ArrayType::new(self.data_type(), Shape::new(output_dimensions)))
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsSlice<ArrayType>>> Slice for Tracer<C> {
    type Output = Self;

    fn slice(self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let mut outputs = self.context().stage_operation(
            C::Operation::slice_operation(start_indices.to_vec(), limit_indices.to_vec(), strides.to_vec()),
            &[&self],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + Slice<Output = V>> Slice for Tangent<ArrayType, V> {
    type Output = Self;

    fn slice(self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        match self {
            Self::Zero(r#type) => Ok(Self::Zero((&r#type).slice(start_indices, limit_indices, strides)?)),
            Self::Value(value) => Ok(Self::Value(value.slice(start_indices, limit_indices, strides)?)),
        }
    }
}

/// [`Operation`] that overwrites a contiguous sub-array of its first operand with its second operand at static start
/// indices. Refer to the documentation of [`UpdateSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UpdateSliceOperation {
    /// Inclusive start index for each input axis at which the update is written.
    start_indices: Vec<usize>,
}

impl UpdateSliceOperation {
    /// Creates a new [`UpdateSliceOperation`] with the provided start indices.
    #[inline]
    pub fn new(start_indices: Vec<usize>) -> Self {
        Self { start_indices }
    }

    /// Returns the inclusive start indices of this [`UpdateSliceOperation`], one per input axis.
    #[inline]
    pub fn start_indices(&self) -> &[usize] {
        self.start_indices.as_slice()
    }
}

impl Display for UpdateSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for UpdateSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match (&input_types[0]).update_slice(&input_types[1], self.start_indices.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("start_indices", format_args!("{:?}", self.start_indices)))
    }
}

impl<V: Value<ArrayType> + UpdateSlice<Output = V>> InterpretableOperation<ArrayType, V> for UpdateSliceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone().update_slice(inputs[1].clone(), self.start_indices.as_slice())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`UpdateSliceOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`UpdateSliceOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsUpdateSlice<T: Type> {
    /// Constructs an instance of [`UpdateSliceOperation`] for this [`Operation`] type with the provided start
    /// indices.
    fn update_slice_operation(start_indices: Vec<usize>) -> Self;
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at static start indices. This is
/// the statically indexed sibling of [`DynamicUpdateSlice`] and the transpose partner of [`Slice`]: writing a
/// cotangent block into a zero array at the slice offsets is exactly an update-slice of a zero input. StableHLO has
/// no statically indexed update operation, so backends lower this operation to
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) with constant start indices.
///
/// `t.update_slice(update, start_indices)` returns a value equal to `t` except that the block starting at
/// `start_indices` is replaced by `update`. The update must have the same data type and rank as the input, all of
/// its dimensions must be static, and each axis must satisfy
/// `start_indices[d] + update_dimension[d] <= input_dimension[d]`. Unlike [`DynamicUpdateSlice`], the start indices
/// are validated when the operation is constructed, so no index clamping occurs.
///
/// # Example
///
/// The following example shows how to use [`UpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::UpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let y = x.update_slice(update, &[0, 1])?;
/// assert_eq!(y.values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait UpdateSlice: Sized {
    /// Output type of the update-slice operation.
    type Output;

    /// Overwrites the block of `self` starting at `start_indices` with `update`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `update`: Value written into `self`. Must have the same data type and rank as `self`, static dimensions,
    ///     and fit within `self` at the provided start indices.
    ///   - `start_indices`: Inclusive start index for each input axis at which `update` is written.
    fn update_slice(self, update: Self, start_indices: &[usize]) -> Result<Self::Output, ProgramError>;
}

impl UpdateSlice for &ArrayType {
    type Output = ArrayType;

    fn update_slice(self, update: Self, start_indices: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != update.data_type() {
            return Err(TypeError {
                message: format!(
                    "update_slice input data type {} does not match update data type {}",
                    self.data_type(),
                    update.data_type(),
                ),
            }
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError {
                message: format!("update_slice update has rank {} but input has rank {rank}", update.rank()),
            }
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "update_slice start_indices has length {} but input has rank {rank}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        for (axis, &start) in start_indices.iter().enumerate() {
            let update_dimension = update.dimension(axis as isize);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError {
                    message: format!(
                        "update_slice does not support dynamic update axis {axis} with size {update_dimension}; \
                        update shapes must be static",
                    ),
                }
                .into());
            };
            let input_dimension = self.dimension(axis as isize);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError {
                    message: format!(
                        "update_slice cannot prove that the update fits along dynamic input axis {axis} with \
                        size {input_dimension}",
                    ),
                }
                .into());
            };
            if start + update_size > input_size {
                return Err(TypeError {
                    message: format!(
                        "update_slice update axis {axis} with start index {start} and size {update_size} does not \
                        fit in input size {input_size}",
                    ),
                }
                .into());
            }
        }
        Ok(self.clone())
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsUpdateSlice<ArrayType>>> UpdateSlice for Tracer<C> {
    type Output = Self;

    fn update_slice(self, update: Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        let mut outputs = self
            .context()
            .stage_operation(C::Operation::update_slice_operation(start_indices.to_vec()), &[&self, &update])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + UpdateSlice<Output = V> + crate::operations::constants::Zero<ArrayType>> UpdateSlice
    for Tangent<ArrayType, V>
{
    type Output = Self;

    fn update_slice(self, update: Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        match (self, update) {
            (Self::Zero(input_type), Self::Zero(update_type)) => {
                // Writing a symbolic zero into a symbolic zero stays symbolically zero; the type-level capability
                // still validates the update window.
                Ok(Self::Zero((&input_type).update_slice(&update_type, start_indices)?))
            }
            (Self::Zero(input_type), Self::Value(update)) => {
                Ok(Self::Value(V::zero(&input_type)?.update_slice(update, start_indices)?))
            }
            (Self::Value(input), Self::Zero(update_type)) => {
                Ok(Self::Value(input.update_slice(V::zero(&update_type)?, start_indices)?))
            }
            (Self::Value(input), Self::Value(update)) => Ok(Self::Value(input.update_slice(update, start_indices)?)),
        }
    }
}

/// [`Operation`] that extracts a statically shaped sub-array from its input at start indices that are computed at
/// run time. Refer to the documentation of [`DynamicSlice`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicSliceOperation {
    /// Size of the extracted slice along each input axis.
    sizes: Vec<usize>,
}

impl DynamicSliceOperation {
    /// Creates a new [`DynamicSliceOperation`] with the provided slice sizes.
    #[inline]
    pub fn new(sizes: Vec<usize>) -> Self {
        Self { sizes }
    }

    /// Returns the slice sizes of this [`DynamicSliceOperation`], one per input axis.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.sizes.as_slice()
    }
}

impl Display for DynamicSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DynamicSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.is_empty() {
            return Err(TypeError {
                message: "dynamic_slice expects an input operand followed by its start index operands but got no \
                    inputs"
                    .to_string(),
            });
        }
        match input_types[0].dynamic_slice(input_types[1..].iter().collect(), self.sizes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sizes", format_args!("{:?}", self.sizes)))
    }
}

impl<V: Value<ArrayType> + DynamicSlice<Output = V>> InterpretableOperation<ArrayType, V> for DynamicSliceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let [input, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes.len(), actual: 0 });
        };
        Ok(vec![input.clone().dynamic_slice(start_indices.to_vec(), self.sizes.as_slice())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`DynamicSliceOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`DynamicSliceOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsDynamicSlice<T: Type> {
    /// Constructs an instance of [`DynamicSliceOperation`] for this [`Operation`] type with the provided slice
    /// sizes.
    fn dynamic_slice_operation(sizes: Vec<usize>) -> Self;
}

/// Represents the ability to extract a statically shaped sub-array at start indices that are computed at run time.
/// This is the direct analogue of the StableHLO
/// [`dynamic_slice`](https://openxla.org/stablehlo/spec#dynamic_slice) operation and JAX's
/// [`lax.dynamic_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html).
///
/// `t.dynamic_slice(start_indices, sizes)` extracts the block of shape `sizes` whose origin is given by the scalar
/// integer values in `start_indices` (one per input axis). Start indices are clamped per StableHLO semantics so the
/// extracted block always lies in bounds: the effective start index along axis `d` is
/// `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`. The output shape is exactly `sizes` and is therefore
/// fully static even though the slice origin is not. Each size must satisfy `sizes[d] <= input_dimension[d]`, and
/// inputs with dynamic dimensions are rejected because that bound cannot be proven against an unknown extent.
///
/// # Example
///
/// The following example shows how to use [`DynamicSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Extract a 1x2 block starting at row 1, column 1 of a 2x3 matrix. This is equivalent to
/// // `jax.lax.dynamic_slice(x, (i, j), slice_sizes=(1, 2))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let i = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let j = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_slice(vec![i, j], &[1, 2])?;
/// // `y` has shape [1, 2] with values [[5.0, 6.0]].
/// assert_eq!(y.values, vec![5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait DynamicSlice: Sized {
    /// Output type of the dynamic-slice operation.
    type Output;

    /// Extracts the block of shape `sizes` starting at `start_indices` from `self`. Refer to the documentation of
    /// this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `start_indices`: Scalar integer start index values, one per input axis, clamped to keep the extracted
    ///     block in bounds.
    ///   - `sizes`: Size of the extracted slice along each input axis.
    fn dynamic_slice(self, start_indices: Vec<Self>, sizes: &[usize]) -> Result<Self::Output, ProgramError>;
}

impl DynamicSlice for &ArrayType {
    type Output = ArrayType;

    fn dynamic_slice(self, start_indices: Vec<Self>, sizes: &[usize]) -> Result<ArrayType, ProgramError> {
        let rank = self.rank();
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "dynamic_slice expects one start index per input axis ({rank}) but got {}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        if sizes.len() != rank {
            return Err(TypeError {
                message: format!("dynamic_slice sizes has length {} but input has rank {rank}", sizes.len()),
            }
            .into());
        }
        validate_start_index_types(DYNAMIC_SLICE_OPERATION_NAME, start_indices.as_slice())?;
        for (axis, &size) in sizes.iter().enumerate() {
            let dimension = self.dimension(axis as isize);
            let Size::Static(input_size) = dimension else {
                return Err(TypeError {
                    message: format!(
                        "dynamic_slice cannot prove that size {size} fits along dynamic input axis {axis} with \
                        size {dimension}",
                    ),
                }
                .into());
            };
            if size > input_size {
                return Err(TypeError {
                    message: format!(
                        "dynamic_slice size {size} is out of bounds for axis {axis} with size {input_size}",
                    ),
                }
                .into());
            }
        }
        Ok(ArrayType::new(self.data_type(), Shape::new(sizes.iter().map(|size| Size::Static(*size)).collect())))
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsDynamicSlice<ArrayType>>> DynamicSlice for Tracer<C> {
    type Output = Self;

    fn dynamic_slice(self, start_indices: Vec<Self>, sizes: &[usize]) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(1 + start_indices.len());
        inputs.push(&self);
        inputs.extend(start_indices.iter());
        let mut outputs = self
            .context()
            .stage_operation(C::Operation::dynamic_slice_operation(sizes.to_vec()), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// [`Operation`] that overwrites a contiguous sub-array of its first operand with its second operand at start
/// indices that are computed at run time. Refer to the documentation of [`DynamicUpdateSlice`] for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicUpdateSliceOperation;

impl Display for DynamicUpdateSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DynamicUpdateSliceOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_UPDATE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() < 2 {
            return Err(TypeError {
                message: format!(
                    "dynamic_update_slice expects an input operand and an update operand followed by start index \
                    operands but got {} inputs",
                    input_types.len(),
                ),
            });
        }
        match (&input_types[0]).dynamic_update_slice(&input_types[1], input_types[2..].iter().collect()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name()).map(|_| ())
    }
}

impl<V: Value<ArrayType> + DynamicUpdateSlice<Output = V>> InterpretableOperation<ArrayType, V>
    for DynamicUpdateSliceOperation
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let [input, update, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        };
        Ok(vec![input.clone().dynamic_update_slice(update.clone(), start_indices.to_vec())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`DynamicUpdateSliceOperation`]. Backend-owned
/// closed [`Operation`] types implement this trait so that generic transform code can stage
/// [`DynamicUpdateSliceOperation`]s without knowing which operation type is in use.
pub trait SupportsDynamicUpdateSlice<T: Type> {
    /// Constructs an instance of [`DynamicUpdateSliceOperation`] for this [`Operation`] type.
    fn dynamic_update_slice_operation() -> Self;
}

/// Represents the ability to overwrite a contiguous sub-array with an update value at start indices that are
/// computed at run time. This is the direct analogue of the StableHLO
/// [`dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) operation and JAX's
/// [`lax.dynamic_update_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html).
///
/// `t.dynamic_update_slice(update, start_indices)` returns a value equal to `t` except that the block whose origin
/// is given by the scalar integer values in `start_indices` (one per input axis) is replaced by `update`. Start
/// indices are clamped per StableHLO semantics so the updated block always lies in bounds: the effective start index
/// along axis `d` is `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`. The update must have
/// the same data type and rank as the input, all of its dimensions must be static, and each axis must satisfy
/// `update_dimension[d] <= input_dimension[d]`; inputs with dynamic dimensions are rejected because that bound
/// cannot be proven against an unknown extent.
///
/// # Example
///
/// The following example shows how to use [`DynamicUpdateSlice`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::DynamicUpdateSlice;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Overwrite the last two elements of the first row of a 2x3 matrix. This is equivalent to
/// // `jax.lax.dynamic_update_slice(x, update, (i, j))` in JAX.
/// let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let update = Array::matrix(1, 2, vec![8.0, 9.0]);
/// let i = Array::new(ArrayType::scalar(DataType::I32), vec![0.0]);
/// let j = Array::new(ArrayType::scalar(DataType::I32), vec![1.0]);
/// let y = x.dynamic_update_slice(update, vec![i, j])?;
/// assert_eq!(y.values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait DynamicUpdateSlice: Sized {
    /// Output type of the dynamic-update-slice operation.
    type Output;

    /// Overwrites the block of `self` starting at `start_indices` with `update`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `update`: Value written into `self`. Must have the same data type and rank as `self`, static dimensions,
    ///     and dimensions that do not exceed those of `self`.
    ///   - `start_indices`: Scalar integer start index values, one per input axis, clamped to keep the updated block
    ///     in bounds.
    fn dynamic_update_slice(self, update: Self, start_indices: Vec<Self>) -> Result<Self::Output, ProgramError>;
}

impl DynamicUpdateSlice for &ArrayType {
    type Output = ArrayType;

    fn dynamic_update_slice(self, update: Self, start_indices: Vec<Self>) -> Result<ArrayType, ProgramError> {
        if self.data_type() != update.data_type() {
            return Err(TypeError {
                message: format!(
                    "dynamic_update_slice input data type {} does not match update data type {}",
                    self.data_type(),
                    update.data_type(),
                ),
            }
            .into());
        }
        let rank = self.rank();
        if update.rank() != rank {
            return Err(TypeError {
                message: format!("dynamic_update_slice update has rank {} but input has rank {rank}", update.rank()),
            }
            .into());
        }
        if start_indices.len() != rank {
            return Err(TypeError {
                message: format!(
                    "dynamic_update_slice expects one start index per input axis ({rank}) but got {}",
                    start_indices.len(),
                ),
            }
            .into());
        }
        validate_start_index_types(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, start_indices.as_slice())?;
        for axis in 0..rank {
            let update_dimension = update.dimension(axis as isize);
            let Size::Static(update_size) = update_dimension else {
                return Err(TypeError {
                    message: format!(
                        "dynamic_update_slice does not support dynamic update axis {axis} with size \
                        {update_dimension}; update shapes must be static",
                    ),
                }
                .into());
            };
            let input_dimension = self.dimension(axis as isize);
            let Size::Static(input_size) = input_dimension else {
                return Err(TypeError {
                    message: format!(
                        "dynamic_update_slice cannot prove that the update fits along dynamic input axis {axis} \
                        with size {input_dimension}",
                    ),
                }
                .into());
            };
            if update_size > input_size {
                return Err(TypeError {
                    message: format!(
                        "dynamic_update_slice update axis {axis} has size {update_size} which exceeds input size \
                        {input_size}",
                    ),
                }
                .into());
            }
        }
        Ok(self.clone())
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsDynamicUpdateSlice<ArrayType>>> DynamicUpdateSlice
    for Tracer<C>
{
    type Output = Self;

    fn dynamic_update_slice(self, update: Self, start_indices: Vec<Self>) -> Result<Self, ProgramError> {
        let mut inputs = Vec::with_capacity(2 + start_indices.len());
        inputs.push(&self);
        inputs.push(&update);
        inputs.extend(start_indices.iter());
        let mut outputs =
            self.context().stage_operation(C::Operation::dynamic_update_slice_operation(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::Typed;

    use super::*;

    /// Returns a scalar integer-typed test array carrying `value` as its in-band payload.
    fn index(value: f64) -> TestArray {
        TestArray::new(ArrayType::scalar(DataType::I32), vec![value])
    }

    #[test]
    fn test_slice() {
        let operation = SliceOperation::new(vec![1, 1], vec![2, 3]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "slice [start_indices=[1, 1], limit_indices=[2, 3]]");
        assert_eq!(operation.start_indices(), &[1, 1]);
        assert_eq!(operation.limit_indices(), &[2, 3]);

        // Type inference validates the slice bounds and returns the sliced type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![output_type.clone()]));
        assert_eq!(input_type.slice(&[1, 1], &[2, 3], &[1, 1]), Ok(output_type.clone()));

        // Interpretation copies the selected block out of the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation.interpret(std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![5.0, 6.0]);

        // Empty slices produce empty payloads and rank-0 slices pass through.
        let empty = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .slice(&[1, 1], &[1, 3], &[1, 1])
            .unwrap();
        assert_eq!(empty.values, Vec::<f64>::new());
        let scalar = TestArray::scalar(42.0).slice(&[], &[], &[]).unwrap();
        assert_eq!(scalar.values, vec![42.0]);

        // Strided operations carry their strides through the builder, accessors, rendering, and inference: the
        // output dimension per axis is `ceil((limit - start) / stride)`.
        let strided = SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap();
        assert_eq!(strided.strides(), &[2]);
        assert_eq!(format!("{strided}"), "slice [start_indices=[1], limit_indices=[6], strides=[2]]");
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        assert_eq!(
            strided.infer_output_types(std::slice::from_ref(&vector_type)),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))]),
        );

        // Strided interpretation keeps the elements at `start + i * stride`.
        let vector = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let strided_output = strided.interpret(std::slice::from_ref(&vector)).unwrap();
        assert_eq!(strided_output[0].values, vec![1.0, 3.0, 5.0]);

        // A stride larger than the sliced extent keeps a single element, and `start == limit` keeps none.
        let single = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[1], &[4], &[5]).unwrap();
        assert_eq!(single.values, vec![1.0]);
        let strided_empty = TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]).slice(&[2], &[2], &[2]).unwrap();
        assert_eq!(*strided_empty.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)])));
        assert_eq!(strided_empty.values, Vec::<f64>::new());

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError { message: "slice start_indices has length 2 but input has rank 1".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "slice limit_indices has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![2, 0], vec![1, 3]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "slice start index 2 is greater than limit index 1 at axis 0".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 4]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "slice limit index 4 is out of bounds for axis 1 with size 3".to_string() }),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![2]),
            Err(ProgramError::Type(TypeError {
                message: "slice strides has length 1 but start_indices has length 2".to_string(),
            })),
        );
        assert_eq!(
            SliceOperation::new(vec![0, 0], vec![2, 3]).with_strides(vec![1, 0]),
            Err(ProgramError::Type(TypeError {
                message: "slice strides must be at least 1 but axis 1 has stride 0".to_string(),
            })),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1]),
            Err(ProgramError::Type(TypeError {
                message: "slice strides has length 1 but input has rank 2".to_string(),
            })),
        );
        assert_eq!(
            input_type.slice(&[0, 0], &[2, 3], &[1, 0]),
            Err(ProgramError::Type(TypeError {
                message: "slice strides must be at least 1 but axis 1 has stride 0".to_string(),
            })),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Dynamic(None), Size::Static(3)]),
            )]),
            Err(TypeError {
                message: "slice does not support dynamic input axis 0 with size *; slice bounds cannot be \
                    validated against an unknown extent"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, SliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[1, 2] = slice [start_indices=[1, 1], limit_indices=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_update_slice() {
        let operation = UpdateSliceOperation::new(vec![0, 1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "update_slice [start_indices=[0, 1]]");
        assert_eq!(operation.start_indices(), &[0, 1]);

        // Type inference validates that the update fits and returns the input type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), update_type.clone()]),
            Ok(vec![input_type.clone()]),
        );
        assert_eq!((&input_type).update_slice(&update_type, &[0, 1]), Ok(input_type.clone()));

        // Interpretation overwrites the selected block of the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = TestArray::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation.interpret(&[input, update]).unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Rank-0 updates replace the input entirely.
        let scalar = TestArray::scalar(1.0).update_slice(TestArray::scalar(7.0), &[]).unwrap();
        assert_eq!(scalar.values, vec![7.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "update_slice input data type f64 does not match update data type f32".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
            ]),
            Err(TypeError { message: "update_slice update has rank 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0]).infer_output_types(&[input_type.clone(), update_type.clone()]),
            Err(TypeError { message: "update_slice start_indices has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "update_slice does not support dynamic update axis 0 with size *; update shapes must be \
                    static"
                    .to_string(),
            }),
        );
        assert_eq!(
            UpdateSliceOperation::new(vec![0, 2]).infer_output_types(&[input_type.clone(), update_type.clone()]),
            Err(TypeError {
                message: "update_slice update axis 1 with start index 2 and size 2 does not fit in input size 3"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4)), Size::Static(3)])),
                update_type.clone(),
            ]),
            Err(TypeError {
                message: "update_slice cannot prove that the update fits along dynamic input axis 0 with size <4"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured indices.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, UpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_output = builder.add_instruction(operation, vec![program_input, program_update]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 2] .
                let %2:f64[2, 3] = update_slice [start_indices=[0, 1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_slice() {
        let operation = DynamicSliceOperation::new(vec![1, 2]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), DYNAMIC_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_slice [sizes=[1, 2]]");
        assert_eq!(operation.sizes(), &[1, 2]);

        // Type inference validates the sizes and index operand types and returns the statically shaped output.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone(), index_type.clone()]),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(input_type.dynamic_slice(vec![&index_type, &index_type], &[1, 2]), Ok(output_type.clone()));

        // Interpretation extracts the block at the in-band start indices.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation.interpret(&[input.clone(), index(1.0), index(1.0)]).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - sizes[d])`.
        let clamped = operation.interpret(&[input.clone(), index(5.0), index(-2.0)]).unwrap();
        assert_eq!(clamped[0].values, vec![4.0, 5.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError {
                message: "dynamic_slice expects an input operand followed by its start index operands but got no \
                    inputs"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone()]),
            Err(TypeError {
                message: "dynamic_slice expects one start index per input axis (2) but got 1".to_string()
            }),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![1]).infer_output_types(&[
                input_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "dynamic_slice sizes has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            DynamicSliceOperation::new(vec![1, 4]).infer_output_types(&[
                input_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "dynamic_slice size 4 is out of bounds for axis 1 with size 3".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_slice cannot prove that size 1 fits along dynamic input axis 0 with size *"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), ArrayType::scalar(DataType::F64), index_type.clone()]),
            Err(TypeError {
                message: "dynamic_slice start index 0 must be a scalar integer but has type f64[]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_slice start index 0 must be a scalar integer but has type i32[2]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), index_type.clone(), ArrayType::scalar(DataType::I64)]),
            Err(TypeError {
                message: "dynamic_slice start indices must share one integer type but index 1 has type i64[] and \
                    index 0 has type i32[]"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured sizes.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, DynamicSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output =
            builder.add_instruction(operation, vec![program_input, program_index_0, program_index_1]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(
                vec![program_output],
                vec![Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:i32[], %2:i32[] .
                let %3:f64[1, 2] = dynamic_slice [sizes=[1, 2]] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_update_slice() {
        let operation = DynamicUpdateSliceOperation;

        // Operation identity.
        assert_eq!(operation.name(), DYNAMIC_UPDATE_SLICE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_update_slice");

        // Type inference validates the update and index operand types and returns the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        let index_type = ArrayType::scalar(DataType::I32);
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                update_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Ok(vec![input_type.clone()]),
        );
        assert_eq!(
            (&input_type).dynamic_update_slice(&update_type, vec![&index_type, &index_type]),
            Ok(input_type.clone()),
        );

        // Interpretation overwrites the block at the in-band start indices.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let update = TestArray::matrix(1, 2, vec![8.0, 9.0]);
        let output = operation.interpret(&[input.clone(), update.clone(), index(0.0), index(1.0)]).unwrap();
        assert_eq!(*output[0].r#type(), input_type);
        assert_eq!(output[0].values, vec![1.0, 8.0, 9.0, 4.0, 5.0, 6.0]);

        // Out-of-bounds start indices clamp per StableHLO semantics: the effective start index along axis `d` is
        // `clamp(0, start_indices[d], input_dimension[d] - update_dimension[d])`.
        let clamped = operation.interpret(&[input.clone(), update.clone(), index(5.0), index(-3.0)]).unwrap();
        assert_eq!(clamped[0].values, vec![1.0, 2.0, 3.0, 8.0, 9.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[input_type.clone()]),
            Err(TypeError {
                message: "dynamic_update_slice expects an input operand and an update operand followed by start \
                    index operands but got 1 inputs"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_update_slice input data type f64 does not match update data type f32".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError { message: "dynamic_update_slice update has rank 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), update_type.clone(), index_type.clone()]),
            Err(TypeError {
                message: "dynamic_update_slice expects one start index per input axis (2) but got 1".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_update_slice does not support dynamic update axis 0 with size *; update shapes \
                    must be static"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(4)])),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_update_slice update axis 1 has size 4 which exceeds input size 3".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                update_type.clone(),
                index_type.clone(),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_update_slice cannot prove that the update fits along dynamic input axis 0 with \
                    size *"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                input_type.clone(),
                update_type.clone(),
                ArrayType::scalar(DataType::F64),
                index_type.clone(),
            ]),
            Err(TypeError {
                message: "dynamic_update_slice start index 0 must be a scalar integer but has type f64[]".to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, DynamicUpdateSliceOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_update = builder.add_input(update_type);
        let program_index_0 = builder.add_input(index_type.clone());
        let program_index_1 = builder.add_input(index_type);
        let program_output = builder
            .add_instruction(operation, vec![program_input, program_update, program_index_0, program_index_1])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(
                vec![program_output],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[1, 2], %2:i32[], %3:i32[] .
                let %4:f64[2, 3] = dynamic_update_slice %0 %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_slicing_test_array_kernels() {
        // Rank-3 slice exercises the row-major odometer across non-contiguous blocks.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type.clone(), values.clone())
            .slice(&[0, 1, 2], &[2, 3, 4], &[1, 1, 1])
            .unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(output.values, vec![6.0, 7.0, 10.0, 11.0, 18.0, 19.0, 22.0, 23.0]);

        // The matching update-slice writes the block back into place.
        let update = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
            vec![-6.0, -7.0, -10.0, -11.0, -18.0, -19.0, -22.0, -23.0],
        );
        let updated = TestArray::new(input_type, values).update_slice(update, &[0, 1, 2]).unwrap();
        assert_eq!(
            updated.values,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0, 9.0, -10.0, -11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                -18.0, -19.0, 20.0, 21.0, -22.0, -23.0,
            ],
        );

        // Strided slicing walks the row-major odometer with per-axis steps: rows with stride 2 and columns with
        // stride 3 keep elements at indices (0, 0), (0, 3), (1, 0), and (1, 3) of a 2x3x4 input's last two axes.
        let strided = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            (0..24).map(|value| value as f64).collect(),
        )
        .slice(&[0, 0, 0], &[2, 3, 4], &[2, 2, 3])
        .unwrap();
        assert_eq!(
            *strided.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(strided.values, vec![0.0, 3.0, 8.0, 11.0]);

        // The dynamic kernels validate their index operand shapes eagerly.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            input.clone().dynamic_slice(vec![index(0.0), TestArray::vector(vec![1.0, 2.0])], &[1, 2]),
            Err(ProgramError::Type(TypeError {
                message: "dynamic_slice start index 1 must be a scalar integer but has type f64[2]".to_string(),
            })),
        );
        assert_eq!(
            input.dynamic_update_slice(TestArray::matrix(1, 2, vec![8.0, 9.0]), vec![index(0.0)]),
            Err(ProgramError::Type(TypeError {
                message: "dynamic_update_slice expects one start index per input axis (2) but got 1".to_string(),
            })),
        );
    }
}
