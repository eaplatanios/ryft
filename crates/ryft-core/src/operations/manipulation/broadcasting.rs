use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Shape, Size, Type, TypeError, Typed};

/// Canonical operation name for [`BroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &'static str = "broadcast";

/// [`Operation`] that performs general N-dimensional broadcasting. Refer to the documentation of [`Broadcast`]
/// for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastOperation {
    /// Output [`ArrayType`].
    output_type: ArrayType,

    /// Vector that contains, for each input axis `i`, the output axis that it maps to.
    output_axes: Vec<usize>,
}

impl BroadcastOperation {
    /// Creates a new [`BroadcastOperation`] with the supplied output type and output axes.
    #[inline]
    pub fn new(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { output_type, output_axes }
    }

    /// Returns the output [`ArrayType`] of this [`BroadcastOperation`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
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
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for BroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].broadcast(self.output_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<V: Value<ArrayType> + Broadcast<Output = V>> InterpretableOperation<ArrayType, V> for BroadcastOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().broadcast(self.output_type.clone(), self.output_axes.as_slice())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`BroadcastOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`BroadcastOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsBroadcast<T: Type> {
    /// Constructs an instance of [`BroadcastOperation`] for this [`Operation`] type with the provided output type
    /// and output axes.
    fn broadcast_operation(output_type: T, output_axes: Vec<usize>) -> Self;
}

/// Represents the ability to perform general N-dimensional broadcasting. This is the direct analogue of JAX's
/// [`lax.broadcast_in_dim`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html).
///
/// `t.broadcast(output_type, output_axes)` expands `t` to `output_type` by mapping each input axis `i` to output axis
/// `output_axes[i]`, replicating the value along the axes of `output_type` that are not named in `output_axes`. For
/// each `i`, the input dimension at axis `i` must either equal the corresponding output dimension or be `1` (in which
/// case it is replicated to match). A [`Size::Dynamic`] input dimension only maps to an identical dynamic output
/// dimension, and every replicated axis (i.e., a static-1 input dimension or an unmapped output axis) must have a
/// static output extent because replication requires a known count.
///
/// # Examples
///
/// The following examples show how to use [`Broadcast`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Broadcast;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Broadcast a length-3 vector to a `[2, 3]` matrix by mapping its single axis to output axis `1`. This is
/// // equivalent to `jax.lax.broadcast_in_dim(jnp.array([1, 2, 3]), shape=(2, 3), broadcast_dimensions=(1,))` in JAX.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
/// let y = x.broadcast(output_type, &[1])?;
/// // `y` has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// assert_eq!(y.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
///
/// // Broadcast a `[2, 2]` matrix over a new middle dimension of size `3` by mapping its axes to output axes `0` and
/// // `2`. This is equivalent to
/// // `jax.lax.broadcast_in_dim(jnp.array([[1, 2], [3, 4]]), shape=(2, 3, 2), broadcast_dimensions=(0, 2))` in JAX.
/// let x = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let output_type = ArrayType::new(
///     DataType::F64,
///     Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]),
/// );
/// let y = x.broadcast(output_type, &[0, 2])?;
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
    /// Output type of the broadcast operation.
    type Output;

    /// Broadcasts `self` to `output_type` using `output_axes`. Refer to the documentation of this trait for more
    /// information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: [`ArrayType`] of the output array.
    ///   - `output_axes`: Slice that contains, for each axis `i` of the input, the output axis that it maps to. This
    ///     slice must have length equal to the input's rank and contain distinct values in `0..output_type.rank()`.
    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self::Output, ProgramError>;
}

impl Broadcast for &ArrayType {
    type Output = ArrayType;

    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != output_type.data_type() {
            return Err(TypeError {
                message: format!(
                    "broadcasting input data type {} does not match output data type {}",
                    self.data_type(),
                    output_type.data_type(),
                ),
            }
            .into());
        }

        let input_rank = self.rank();
        let output_rank = output_type.rank();
        if output_axes.len() != input_rank {
            return Err(TypeError {
                message: format!(
                    "broadcasting output axes has length {} but input has rank {input_rank}",
                    output_axes.len(),
                ),
            }
            .into());
        }

        let mut seen = vec![false; output_rank];
        for (input_axis, &output_axis) in output_axes.iter().enumerate() {
            if output_axis >= output_rank {
                return Err(TypeError {
                    message: format!(
                        "broadcasting `output_axes[{input_axis}] = {output_axis}` is out of bounds for \
                        output rank {output_rank}",
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
            let output_dimension = output_type.dimension(output_axis as isize);
            match (input_dimension, output_dimension) {
                // Identical sizes always map through, including identical dynamic sizes.
                (input_dimension, output_dimension) if input_dimension == output_dimension => {}
                // A static size-1 input dimension is replicated to match any static output extent. Expanding it
                // into a dynamic output dimension is unsupported because the replication count is unknown.
                (Size::Static(1), Size::Static(_)) => {}
                (Size::Static(1), Size::Dynamic(_)) => {
                    return Err(TypeError {
                        message: format!(
                            "broadcasting cannot expand input axis {input_axis} of size 1 into dynamic \
                            output size {output_dimension}",
                        ),
                    }
                    .into());
                }
                (Size::Static(input_size), Size::Static(output_size)) => {
                    return Err(TypeError {
                        message: format!(
                            "broadcasting input axis {input_axis} has size {input_size}, which is \
                             neither {output_size} nor 1",
                        ),
                    }
                    .into());
                }
                // All remaining combinations pair a dynamic size with a mismatched size on the other side.
                (input_dimension, output_dimension) => {
                    return Err(TypeError {
                        message: format!(
                            "broadcasting input axis {input_axis} has size {input_dimension} but the output has \
                            size {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic \
                            dimension",
                        ),
                    }
                    .into());
                }
            }
        }

        // Output axes that no input axis maps to replicate the input along that axis, which requires a known
        // replication count and is therefore unsupported for dynamic output dimensions.
        for (output_axis, mapped) in seen.iter().enumerate() {
            let output_dimension = output_type.dimension(output_axis as isize);
            if !mapped && matches!(output_dimension, Size::Dynamic(_)) {
                return Err(TypeError {
                    message: format!(
                        "broadcasting cannot replicate the input into unmapped dynamic output axis {output_axis} \
                        of size {output_dimension}",
                    ),
                }
                .into());
            }
        }
        Ok(output_type)
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsBroadcast<ArrayType>>> Broadcast for Tracer<C> {
    type Output = Self;

    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let mut outputs = self
            .context()
            .stage_operation(C::Operation::broadcast_operation(output_type, output_axes.to_vec()), &[&self])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + Broadcast<Output = V>> Broadcast for Tangent<ArrayType, V> {
    type Output = Self;

    fn broadcast(self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        match self {
            Self::Zero(r#type) => Ok(Self::Zero(r#type.broadcast(output_type, output_axes)?)),
            Self::Value(value) => Ok(Self::Value(value.broadcast(output_type, output_axes)?)),
        }
    }
}

/// Represents the ability to prepend leading dimensions of specific sizes to an array by replicating it along those
/// dimensions. This is the direct analogue of JAX's
/// [`lax.broadcast`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast.html).
///
/// `t.broadcast_leading([s0, s1, ...])` produces a value whose shape is `[s0, s1, ..., t.shape...]`, with the original
/// value replicated across the new leading axes. This is equivalent to `t.broadcast(output_type, output_axes)` with
/// `output_type` having shape `[s0, s1, ..., t.shape...]` and `output_axes` mapping each input axis `i` to output axis
/// `i + sizes.len()`.
///
/// # Example
///
/// The following example shows how to use [`BroadcastLeading`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::BroadcastLeading;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
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
        let output_type = ArrayType::new(input_type.data_type(), Shape::new(output_dimensions));
        let output_axes = (0..input_type.rank()).map(|axis| axis + sizes.len()).collect::<Vec<_>>();
        self.broadcast(output_type, output_axes.as_slice())
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
/// `t.broadcast(output_type, output_axes)` with `output_axes` computed as the trailing range of indices.
///
/// # Example
///
/// The following example shows how to use [`BroadcastTo`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::BroadcastTo;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{Shape, Size};
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
        let output_type = ArrayType::new(input_type.data_type(), target_shape);
        let output_axes = (0..input_rank).map(|axis| axis + offset).collect::<Vec<_>>();
        self.broadcast(output_type, output_axes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_broadcast() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "broadcast [output_type=f64[2, 3], output_axes=[1]]");
        assert_eq!(*operation.output_type(), output_type);
        assert_eq!(operation.output_axes(), &[1]);

        // Type inference validates the axis mapping and returns the target type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![output_type.clone()]));

        // Type-level (abstract) broadcasting validates the axis mapping and returns the target type without
        // consuming the borrowed input type.
        assert_eq!(input_type.broadcast(output_type.clone(), &[1]), Ok(output_type.clone()));

        // Interpretation replicates the payload along the added axis.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = operation.interpret(std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // The convenience capabilities delegate to `broadcast`.
        let broadcast_leading = TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast_leading(vec![2]).unwrap();
        assert_eq!(*broadcast_leading.r#type(), output_type);
        assert_eq!(broadcast_leading.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let broadcast_to = TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast_to(output_type.shape().clone()).unwrap();
        assert_eq!(*broadcast_to.r#type(), output_type);
        assert_eq!(broadcast_to.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]))]),
            Err(TypeError {
                message: "broadcasting input data type f32 does not match output data type f64".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&output_type)),
            Err(TypeError { message: "broadcasting output axes has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            BroadcastOperation::new(output_type.clone(), vec![2]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string()
            }),
        );
        assert_eq!(
            BroadcastOperation::new(output_type.clone(), vec![1, 1]).infer_output_types(&[ArrayType::new(
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
            TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast(output_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string(),
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
                let %1:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_broadcast_with_dynamic_dimensions() {
        // A dynamic input dimension maps through to an identical dynamic output dimension, while replicated axes
        // (unmapped output axes) keep requiring static extents.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2), Size::Static(3)]));
        assert_eq!(input_type.broadcast(output_type.clone(), &[0, 2]), Ok(output_type));

        // A dynamic input dimension does not broadcast to a different dynamic output dimension or to a static one.
        let unbounded = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        assert_eq!(
            unbounded.broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4))])), &[0]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting input axis 0 has size * but the output has size <4; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            })),
        );
        assert_eq!(
            unbounded.broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])), &[0]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting input axis 0 has size * but the output has size 3; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            })),
        );

        // Static input dimensions do not expand into dynamic output dimensions, whether they are 1 or not.
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1)])).broadcast(unbounded.clone(), &[0]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting cannot expand input axis 0 of size 1 into dynamic output size *".to_string(),
            })),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])).broadcast(unbounded, &[0]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting input axis 0 has size 3 but the output has size *; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            })),
        );

        // Replicating the input along an unmapped dynamic output axis is rejected because the replication count
        // is unknown.
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])).broadcast(
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)])),
                &[1],
            ),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting cannot replicate the input into unmapped dynamic output axis 0 of size *"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_broadcast_test_array() {
        // Mapping a vector's single axis to output axis 1 replicates it across the added leading axis.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output = TestArray::vector(vec![1.0, 2.0, 3.0]).broadcast(target, &[1]).unwrap();
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Mapping a [2, 2] matrix to output axes 0 and 2 replicates it along the added middle axis.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]));
        let output = TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).broadcast(target, &[0, 2]).unwrap();
        assert_eq!(output.values, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);

        // Static unit axes are stretched to the corresponding target extent.
        let input = TestArray::matrix(1, 3, vec![1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output = input.broadcast(target, &[0, 1]).unwrap();
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Empty targets produce empty payloads.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0), Size::Static(2)]));
        let output = TestArray::vector(vec![1.0, 2.0]).broadcast(target, &[1]).unwrap();
        assert_eq!(output.values, Vec::<f64>::new());
    }
}
