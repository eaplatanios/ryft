use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Shape, Size, StaticShape, Type, TypeError, Typed};

/// Canonical operation name for [`BroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &'static str = "broadcast";

/// [`Operation`] that performs general N-dimensional broadcasting. Refer to the documentation of [`Broadcast`]
/// for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastOperation {
    /// Target output [`ArrayType`].
    target_type: ArrayType,

    /// Vector that contains, for each input axis `i`, the output axis that it maps to.
    output_axes: Vec<usize>,
}

impl BroadcastOperation {
    /// Creates a new [`BroadcastOperation`] with the supplied target type and output axes.
    #[inline]
    pub fn new(target_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { target_type, output_axes }
    }

    /// Returns the target output [`ArrayType`] of this [`BroadcastOperation`].
    #[inline]
    pub fn target_type(&self) -> &ArrayType {
        &self.target_type
    }

    /// Returns the output axes of this [`BroadcastOperation`]. The resulting slice contains, for each input axis,
    /// the output axis that it maps to.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

impl Display for BroadcastOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(BROADCAST_OPERATION_NAME)
    }
}

impl Operation<ArrayType> for BroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].broadcast(self.target_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("target_type", &self.target_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<V: Value<ArrayType> + Broadcast<Output = V>> InterpretableOperation<ArrayType, V> for BroadcastOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().broadcast(self.target_type.clone(), self.output_axes.as_slice())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`BroadcastOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`BroadcastOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsBroadcast<T: Type> {
    /// Constructs an instance of [`BroadcastOperation`] for this [`Operation`] type with the provided target type
    /// and output axes.
    fn broadcast_operation(target_type: T, output_axes: Vec<usize>) -> Self;
}

/// Represents the ability to perform general N-dimensional broadcasting. This is the direct analogue of JAX's
/// [`lax.broadcast_in_dim`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html) and the lowering
/// target of StableHLO's [`broadcast_in_dim`](https://openxla.org/stablehlo/spec#broadcast_in_dim) in the XLA backend.
///
/// `t.broadcast(target_type, output_axes)` expands `t` to `target_type` by mapping each input axis `i` to output axis
/// `output_axes[i]`, replicating the value along the axes of `target_type` that are not named in `output_axes`. For
/// each `i`, the input dimension at axis `i` must either equal the corresponding output dimension or be `1` (in which
/// case it is replicated to match).
///
/// # Examples
///
/// The following examples show how to use [`Broadcast`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::{Broadcast, broadcast_evaluate};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # struct Array {
/// #     r#type: ArrayType,
/// #     values: Vec<f64>,
/// # }
/// #
/// # impl Array {
/// #     fn vector(values: Vec<f64>) -> Self {
/// #         let shape = Shape::new(vec![Size::Static(values.len())]);
/// #         Self { r#type: ArrayType::new(DataType::F64, shape), values }
/// #     }
/// #
/// #     fn matrix(rows: usize, columns: usize, values: Vec<f64>) -> Self {
/// #         let shape = Shape::new(vec![Size::Static(rows), Size::Static(columns)]);
/// #         Self { r#type: ArrayType::new(DataType::F64, shape), values }
/// #     }
/// # }
/// #
/// # impl Broadcast for Array {
/// #     type Output = Self;
/// #
/// #     fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
/// #         let input_shape = self.r#type.static_shape().unwrap();
/// #         let target_shape = target_type.static_shape().unwrap();
/// #         let values = broadcast_evaluate(&self.values, &input_shape, &target_shape, output_axes);
/// #         Ok(Self { r#type: target_type, values })
/// #     }
/// # }
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Broadcast a length-3 vector to a `[2, 3]` matrix by mapping its single axis to output axis `1`. This is
/// // equivalent to `jax.lax.broadcast_in_dim(jnp.array([1, 2, 3]), shape=(2, 3), broadcast_dimensions=(1,))` in JAX.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
/// let y = x.broadcast(target, &[1])?;
/// // `y` has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// assert_eq!(y.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
///
/// // Broadcast a `[2, 2]` matrix over a new middle dimension of size `3` by mapping its axes to output axes `0` and
/// // `2`. This is equivalent to
/// // `jax.lax.broadcast_in_dim(jnp.array([[1, 2], [3, 4]]), shape=(2, 3, 2), broadcast_dimensions=(0, 2))` in JAX.
/// let x = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let target = ArrayType::new(
///     DataType::F64,
///     Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]),
/// );
/// let y = x.broadcast(target, &[0, 2])?;
/// // `y` has shape [2, 3, 2] with values:
/// //   [[[1.0, 2.0],
/// //     [1.0, 2.0],
/// //     [1.0, 2.0]],
/// //    [[3.0, 4.0],
/// //     [3.0, 4.0],
/// //     [3.0, 4.0]]]
/// assert_eq!(y.values, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub trait Broadcast {
    /// Output type of the broadcasting operation.
    type Output;

    /// Broadcasts `self` to `target_type` using `output_axes`. Refer to the documentation of this trait for more
    /// information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `target_type`: [`ArrayType`] of the output array.
    ///   - `output_axes`: Slice that contains, for each axis `i` of the input, the output axis that it maps to. This
    ///     slice must have length equal to the input's rank and contain distinct values in `0..target_type.rank()`.
    fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self::Output, ProgramError>;
}

impl Broadcast for &ArrayType {
    type Output = ArrayType;

    fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != target_type.data_type() {
            return Err(TypeError {
                message: format!(
                    "broadcasting input data type {} does not match target data type {}",
                    self.data_type(),
                    target_type.data_type(),
                ),
            }
            .into());
        }
        let input_rank = self.rank();
        let target_rank = target_type.rank();
        if output_axes.len() != input_rank {
            return Err(TypeError {
                message: format!(
                    "broadcasting output axes has length {} but input has rank {input_rank}",
                    output_axes.len(),
                ),
            }
            .into());
        }
        let mut seen = vec![false; target_rank];
        for (input_axis, &output_axis) in output_axes.iter().enumerate() {
            if output_axis >= target_rank {
                return Err(TypeError {
                    message: format!(
                        "broadcasting `output_axes[{input_axis}] = {output_axis}` is out of bounds for \
                        target rank {target_rank}",
                    ),
                }
                .into());
            }
            if seen[output_axis] {
                return Err(TypeError {
                    message: format!("broadcasting output axes map two input axes to output axis {output_axis}",),
                }
                .into());
            }
            seen[output_axis] = true;
            let input_dimension = self.dimension(input_axis as isize);
            let target_dimension = target_type.dimension(output_axis as isize);
            match (input_dimension.value(), target_dimension.value()) {
                (Some(input_size), Some(target_size)) if input_size != target_size && input_size != 1 => {
                    return Err(TypeError {
                        message: format!(
                            "broadcasting input axis {input_axis} has size {input_size}, which is \
                             neither {target_size} nor 1",
                        ),
                    }
                    .into());
                }
                _ => {}
            }
        }
        Ok(target_type)
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsBroadcast<ArrayType>>> Broadcast for Tracer<C> {
    type Output = Self;

    fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let mut outputs = self
            .context()
            .stage_operation(C::Operation::broadcast_operation(target_type, output_axes.to_vec()), &[&self])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + Broadcast<Output = V>> Broadcast for Tangent<ArrayType, V> {
    type Output = Self;

    fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        match self {
            Self::Zero(r#type) => Ok(Self::Zero(r#type.broadcast(target_type, output_axes)?)),
            Self::Value(value) => Ok(Self::Value(value.broadcast(target_type, output_axes)?)),
        }
    }
}

/// Represents the ability to prepend leading dimensions of specific sizes to an array by replicating it along those
/// dimensions. This is the direct analogue of JAX's
/// [`lax.broadcast`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast.html).
///
/// `t.broadcast_leading([s0, s1, ...])` produces a value whose shape is `[s0, s1, ..., t.shape...]`, with the original
/// value replicated across the new leading axes. This is equivalent to `t.broadcast(target_type, output_axes)` with
/// `target_type` having shape `[s0, s1, ..., t.shape...]` and `output_axes` mapping each input axis `i` to output axis
/// `i + sizes.len()`.
///
/// # Example
///
/// The following example shows how to use [`BroadcastLeading`] in practice:
///
/// ```rust
/// # use std::borrow::Cow;
/// #
/// # use ryft_core::operations::manipulation::{Broadcast, BroadcastLeading, broadcast_evaluate};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size, Typed};
/// #
/// # struct Array {
/// #     r#type: ArrayType,
/// #     values: Vec<f64>,
/// # }
/// #
/// # impl Array {
/// #     fn vector(values: Vec<f64>) -> Self {
/// #         let shape = Shape::new(vec![Size::Static(values.len())]);
/// #         Self { r#type: ArrayType::new(DataType::F64, shape), values }
/// #     }
/// # }
/// #
/// # impl Typed<ArrayType> for Array {
/// #     fn r#type(&self) -> Cow<'_, ArrayType> {
/// #         Cow::Borrowed(&self.r#type)
/// #     }
/// # }
/// #
/// # impl Broadcast for Array {
/// #     type Output = Self;
/// #
/// #     fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
/// #         let input_shape = self.r#type.static_shape().unwrap();
/// #         let target_shape = target_type.static_shape().unwrap();
/// #         let values = broadcast_evaluate(&self.values, &input_shape, &target_shape, output_axes);
/// #         Ok(Self { r#type: target_type, values })
/// #     }
/// # }
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Broadcast a length-3 vector to a `[2, 3]` matrix by prepending one leading axis of size `2`. This is
/// // equivalent to `jax.lax.broadcast(jnp.array([1, 2, 3]), sizes=(2,))` in JAX.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.broadcast_leading(vec![2])?;
/// // `y` has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// assert_eq!(y.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub trait BroadcastLeading: Sized {
    /// Broadcasts `self` by prepending leading dimensions of the provided sizes. Refer to the documentation of this
    /// trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `sizes`: Sizes of the new leading dimensions to prepend, in order.
    fn broadcast_leading(self, sizes: Vec<usize>) -> Result<Self, ProgramError>;
}

impl<T: Typed<ArrayType> + Broadcast<Output = T>> BroadcastLeading for T {
    fn broadcast_leading(self, sizes: Vec<usize>) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let mut output_dimensions: Vec<Size> = sizes.iter().map(|size| Size::Static(*size)).collect();
        output_dimensions.extend(input_type.shape().dimensions().iter().copied());
        let target_type = ArrayType::new(input_type.data_type(), Shape::new(output_dimensions));
        let output_axes = (0..input_type.rank()).map(|axis| axis + sizes.len()).collect::<Vec<_>>();
        self.broadcast(target_type, output_axes.as_slice())
    }
}

/// Represents the ability to broadcast an array to a target shape using the broadcasting semantics of
/// [`Broadcastable`](crate::Broadcastable). This is the direct analogue of JAX's
/// [`jnp.broadcast_to`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html), which itself mirrors
/// NumPy's [`numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html).
///
/// `t.broadcast_to(target_shape)` right-aligns the input shape with `target_shape`: input axis `i` corresponds to
/// output axis `target_shape.rank() - input.rank() + i`. Each corresponding input dimension must equal the output
/// dimension or be `1`, in which case it is replicated. Missing leading input dimensions are treated as size `1`,
/// and so a smaller-rank array can be broadcast to a larger-rank target shape. This is equivalent to
/// `t.broadcast(target_type, output_axes)` with `output_axes` computed as the trailing range of indices.
///
/// # Example
///
/// The following example shows how to use [`BroadcastTo`] in practice:
///
/// ```rust
/// # use std::borrow::Cow;
/// #
/// # use ryft_core::operations::manipulation::{Broadcast, BroadcastTo, broadcast_evaluate};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size, Typed};
/// #
/// # struct Array {
/// #     r#type: ArrayType,
/// #     values: Vec<f64>,
/// # }
/// #
/// # impl Array {
/// #     fn vector(values: Vec<f64>) -> Self {
/// #         let shape = Shape::new(vec![Size::Static(values.len())]);
/// #         Self { r#type: ArrayType::new(DataType::F64, shape), values }
/// #     }
/// # }
/// #
/// # impl Typed<ArrayType> for Array {
/// #     fn r#type(&self) -> Cow<'_, ArrayType> {
/// #         Cow::Borrowed(&self.r#type)
/// #     }
/// # }
/// #
/// # impl Broadcast for Array {
/// #     type Output = Self;
/// #
/// #     fn broadcast(self, target_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
/// #         let input_shape = self.r#type.static_shape().unwrap();
/// #         let target_shape = target_type.static_shape().unwrap();
/// #         let values = broadcast_evaluate(&self.values, &input_shape, &target_shape, output_axes);
/// #         Ok(Self { r#type: target_type, values })
/// #     }
/// # }
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Broadcast a length-3 vector to a `[3, 3]` matrix by replicating the input across the leading axis. This is
/// // equivalent to `jnp.broadcast_to(jnp.array([1, 2, 3]), (3, 3))` in JAX.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.broadcast_to(Shape::new(vec![Size::Static(3), Size::Static(3)]))?;
/// // `y` has shape [3, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// assert_eq!(y.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub trait BroadcastTo: Sized {
    /// Broadcasts `self` to `target_shape` using the broadcasting semantics of [`Broadcastable`](crate::Broadcastable).
    /// Refer to the documentation of this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `target_shape`: [`Shape`] to broadcast `self` to. This shape must have rank at least equal to the input's
    ///     rank and must be compatible with the shape of the input in terms of broadcasting semantics.
    fn broadcast_to(self, target_shape: Shape) -> Result<Self, ProgramError>;
}

impl<T: Typed<ArrayType> + Broadcast<Output = T>> BroadcastTo for T {
    fn broadcast_to(self, target_shape: Shape) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let input_rank = input_type.rank();
        let offset = target_shape.rank().saturating_sub(input_rank);
        let target_type = ArrayType::new(input_type.data_type(), target_shape);
        let output_axes = (0..input_rank).map(|axis| axis + offset).collect::<Vec<_>>();
        self.broadcast(target_type, output_axes.as_slice())
    }
}

// TODO(eaplatanios): Review from here onwards.

/// N-D broadcast helper that operates on a flat row-major payload and shape.
///
/// Returns the broadcasted values in row-major order over `target_shape`. Each input axis `i` maps to output axis
/// `output_axes[i]`; output axes not named in `output_axes` are added (the input value is replicated along them).
/// When the input's dimension at axis `i` is `1`, that axis is also replicated along the corresponding output axis.
pub fn broadcast_evaluate<T: Clone>(
    values: &[T],
    input_shape: &StaticShape,
    target_shape: &StaticShape,
    output_axes: &[usize],
) -> Vec<T> {
    let input_rank = input_shape.rank();
    let target_rank = target_shape.rank();
    let output_count: usize = target_shape.dimensions().iter().product();
    if output_count == 0 {
        return Vec::new();
    }
    let input_strides = input_shape.row_major_strides();
    let mut output = Vec::with_capacity(output_count);
    let mut target_index = vec![0usize; target_rank];
    loop {
        let mut input_flat = 0usize;
        for input_axis in 0..input_rank {
            let target_axis = output_axes[input_axis];
            let coordinate = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
            input_flat += coordinate * input_strides[input_axis];
        }
        output.push(values[input_flat].clone());

        let mut position = target_rank;
        while position > 0 {
            position -= 1;
            target_index[position] += 1;
            if target_index[position] < target_shape[position] {
                break;
            }
            target_index[position] = 0;
            if position == 0 {
                return output;
            }
        }
        if target_rank == 0 {
            return output;
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_broadcast() {
        let target_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(target_type.clone(), vec![1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), BROADCAST_OPERATION_NAME);
        assert_eq!(*operation.target_type(), target_type);
        assert_eq!(operation.output_axes(), &[1]);

        // Type inference validates the axis mapping and returns the target type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![target_type.clone()]));

        // Type-level (abstract) broadcasting validates the axis mapping and returns the target type without
        // consuming the borrowed input type.
        assert_eq!(input_type.broadcast(target_type.clone(), &[1]), Ok(target_type.clone()));

        // Interpretation replicates the payload along the added axis.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = operation.interpret(std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].array_type(), target_type);
        assert_eq!(output[0].values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // The convenience capabilities delegate to `broadcast`.
        let broadcast_leading = TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast_leading(vec![2]).unwrap();
        assert_eq!(*broadcast_leading.array_type(), target_type);
        assert_eq!(broadcast_leading.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let broadcast_to = TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast_to(target_type.shape().clone()).unwrap();
        assert_eq!(*broadcast_to.array_type(), target_type);
        assert_eq!(broadcast_to.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]))]),
            Err(TypeError {
                message: "broadcasting input data type f32 does not match target data type f64".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&target_type)),
            Err(TypeError { message: "broadcasting output axes has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            BroadcastOperation::new(target_type.clone(), vec![2]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for target rank 2".to_string()
            }),
        );
        assert_eq!(
            BroadcastOperation::new(target_type.clone(), vec![1, 1]).infer_output_types(&[ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Static(3), Size::Static(3)]),
            )]),
            Err(TypeError { message: "broadcasting output axes map two input axes to output axis 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError { message: "broadcasting input axis 0 has size 2, which is neither 3 nor 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast(target_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for target rank 2".to_string(),
            })),
        );

        // Program rendering uses the canonical operation name and includes the captured metadata.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, BroadcastOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[2, 3] = broadcast [target_type=f64[2, 3], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_broadcast_evaluate() {
        // Mapping a vector's single axis to output axis 1 replicates it across the added leading axis.
        assert_eq!(
            broadcast_evaluate(&[1.0, 2.0, 3.0], &StaticShape::new(vec![3]), &StaticShape::new(vec![2, 3]), &[1]),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        );

        // Mapping a [2, 2] matrix to output axes 0 and 2 replicates it along the added middle axis.
        assert_eq!(
            broadcast_evaluate(
                &[1.0, 2.0, 3.0, 4.0],
                &StaticShape::new(vec![2, 2]),
                &StaticShape::new(vec![2, 3, 2]),
                &[0, 2]
            ),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        );

        // Static unit axes are stretched to the corresponding target extent.
        assert_eq!(
            broadcast_evaluate(&[1.0, 2.0, 3.0], &StaticShape::new(vec![1, 3]), &StaticShape::new(vec![2, 3]), &[0, 1]),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        );

        // Empty targets produce empty payloads.
        assert_eq!(
            broadcast_evaluate(&[1.0, 2.0], &StaticShape::new(vec![2]), &StaticShape::new(vec![0, 2]), &[1]),
            Vec::<f64>::new()
        );
    }
}
