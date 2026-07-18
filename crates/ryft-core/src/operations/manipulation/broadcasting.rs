use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::elementwise::BroadcastDerivativeAlignment;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::sharding::ReshardOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Shape, Size};

use super::{ConvertElementTypeOperation, ReshapeOperation, TransposeOperation};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`BroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &str = "broadcast";

/// [`Operation`] that performs general N-dimensional broadcasting.
/// Refer to the documentation of [`Broadcast`] for more information.
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for BroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].broadcast(self.output_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Broadcast>> InterpretableOperation<C> for BroadcastOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().broadcast(self.output_type.clone(), self.output_axes.as_slice())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<BroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for BroadcastOperation
{
}

/// Forward-mode rule for [`BroadcastOperation`]: `broadcast` is structural-linear, so the tangent is the same
/// broadcast applied to the operand tangent. The shared all-zero fast path handles a zero operand tangent before this
/// rule is consulted, so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for BroadcastOperation
where
    C::Operation: From<BroadcastOperation>,
    C::Value: Broadcast,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().broadcast(self.output_type().clone(), self.output_axes())?;
        let tangent_type = primal.r#type().tangent();
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.broadcast(tangent_type, self.output_axes())?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for a [`BroadcastOperation`].
///
/// The pullback of a broadcast is a sum-reduction over every output axis the input was replicated along: the axes of
/// the target type that are not named in `output_axes`, plus the mapped axes whose input extent is `1` stretched to a
/// larger target extent. After the reduction, the surviving axes are reordered into input-axis order when
/// `output_axes` is not monotonically increasing, and stretched unit axes are restored with a reshape so the cotangent
/// matches the input type exactly. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for BroadcastOperation
where
    O: Operation<ArrayType>
        + From<BroadcastOperation>
        + From<ConvertElementTypeOperation>
        + From<crate::tracing_v2::operations::reduce::ReduceOperation>
        + From<TransposeOperation>
        + From<ReshapeOperation>
        + From<ReshardOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let input_cotangent_type = inputs[0].r#type().cotangent();
        if input_cotangent_type.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: "'broadcast' input 0 has no cotangent space".to_string(),
            }
            .into());
        }
        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
        };
        Ok(vec![MaybeZero::Value(cotangent.unalign_cotangent_along(&input_cotangent_type, self.output_axes())?)])
    }
}

/// Lifts a broadcast's `output_axes` and target shape through one batching level.
///
/// When the input is batched at axis `k` (in the input's unbatched logical shape), the lifted broadcast inserts a batch
/// dimension of size `axis_size` at position `k` in the target shape, places a corresponding batch axis at output
/// position `k_out = k`, and shifts the existing `output_axes` so each previously mapped output axis `>= k_out` is
/// shifted by one. The new batch input axis itself maps to `k_out`.
pub fn lift_broadcast(
    output_axes: &[usize],
    output_type: &ArrayType,
    input_batch_axis: usize,
    axis_size: usize,
) -> Result<(Vec<usize>, ArrayType, usize), TypeError> {
    let target_batch_axis = input_batch_axis;
    let lifted_target = output_type.with_inserted_dimension(target_batch_axis, Size::Static(axis_size))?;

    let mut lifted_dimensions = Vec::with_capacity(output_axes.len() + 1);
    for &output_axis in output_axes.iter() {
        let shifted_output_axis = if output_axis >= target_batch_axis { output_axis + 1 } else { output_axis };
        lifted_dimensions.push(shifted_output_axis);
    }
    lifted_dimensions.insert(input_batch_axis, target_batch_axis);

    Ok((lifted_dimensions, lifted_target, target_batch_axis))
}

impl<C: Context<Type = ArrayType, Value: Broadcast>> BatchableOperation<C> for BroadcastOperation {
    fn batch<D: BatchingDriver<C>>(
        &self,
        _context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        match batch_axes[0] {
            None => {
                // Replicated input: the broadcast itself does not change. Pass through.
                let output_value =
                    inputs[0].value().clone().broadcast(self.output_type().clone(), self.output_axes())?;
                Ok(vec![ArrayBatch::new(self.output_type().clone(), output_value, None)?])
            }
            Some(batch_axis) => {
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let (lifted_dimensions, mut lifted_target, target_batch_axis) =
                    lift_broadcast(self.output_axes(), self.output_type(), batch_axis, axis_size)?;
                if let Some(sharding) = self.output_type().sharding() {
                    lifted_target.sharding = Some(
                        sharding
                            .with_inserted_dimension(target_batch_axis, ArrayBatch::sharding_for_inputs(inputs)?)
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                let output_value =
                    inputs[0].value().clone().broadcast(lifted_target.clone(), lifted_dimensions.as_slice())?;
                Ok(vec![ArrayBatch::new(lifted_target, output_value, BatchAxis::from_position(target_batch_axis))?])
            }
        }
    }
}

/// Represents the ability to perform general N-dimensional broadcasting. This is the direct analogue of JAX's
/// [`lax.broadcast_in_dim`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html).
/// `t.broadcast(output_type, output_axes)` expands `t` to `output_type` by mapping each input axis `i` to output axis
/// `output_axes[i]`, replicating the value along the axes of `output_type` that are not named in `output_axes`. For
/// each `i`, the input dimension at axis `i` must either equal the corresponding output dimension or be `1` (in which
/// case it is replicated to match). A [`Size::Dynamic`] input dimension only maps to an identical dynamic output
/// dimension, and every replicated axis (i.e., a static-1 input dimension or an unmapped output axis) must have a
/// static output extent because replication requires a known count.
///
/// [`broadcast_leading`](Broadcast::broadcast_leading) and [`broadcast_to`](Broadcast::broadcast_to) are convenience
/// methods, implemented in terms of [`broadcast`](Broadcast::broadcast), covering two common cases: prepending new
/// replicated leading axes, and broadcasting to a target [`Shape`] using NumPy-style right alignment. Both require the
/// implementer to be [`Typed`] against [`ArrayType`] so they can read the input's own type.
///
/// # Examples
///
/// The following examples show how to use [`Broadcast`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Broadcast;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Broadcast a length-3 vector to a `[2, 3]` matrix by mapping its single axis to output axis 1. This is
/// // equivalent to `broadcast_in_dim(jnp.array([1, 2, 3]), shape=(2, 3), broadcast_dimensions=(1,))` in JAX.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
/// let y = x.broadcast(output_type, &[1])?;
/// // `y` has shape [2, 3] with values:
/// //   [[1.0, 2.0, 3.0],
/// //    [1.0, 2.0, 3.0]]
/// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
///
/// // Broadcast a `[2, 2]` matrix over a new dimension of size 3 by mapping its axes to output axes 0 and 2. This is
/// // equivalent to `broadcast_in_dim(jnp.array([[1, 2], [3, 4]]), shape=(2, 3, 2), broadcast_dimensions=(0, 2))`
/// // in JAX.
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
/// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub trait Broadcast: Sized {
    /// Broadcasts `self` to `output_type` using `output_axes`. Refer to the documentation of this trait for more
    /// information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: [`ArrayType`] of the output array.
    ///   - `output_axes`: Slice that contains, for each axis `i` of the input, the output axis that it maps to. This
    ///     slice must have length equal to the input's rank and contain distinct values in `0..output_type.rank()`.
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError>;

    /// Broadcasts `self` by prepending leading dimensions of the provided sizes, replicating it along those new
    /// dimensions. This is the direct analogue of JAX's
    /// [`lax.broadcast`]( https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast.html).
    /// `t.broadcast_leading([s0, s1, ...])` produces a value whose shape is `[s0, s1, ..., t.shape...]`, with the
    /// original value replicated across the new leading axes. This is equivalent to
    /// `t.broadcast(output_type, output_axes)` with `output_type` having shape `[s0, s1, ..., t.shape...]`
    /// and `output_axes` mapping each input axis `i` to output axis `i + sizes.len()`.
    ///
    /// # Parameters
    ///
    ///   - `sizes`: Sizes of the new leading dimensions to prepend, in order.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::operations::manipulation::Broadcast;
    /// # use ryft_core::programs::ProgramError;
    /// # use ryft_core::backends::arrays::Array;
    /// #
    /// # fn main() -> Result<(), ProgramError> {
    /// // Broadcast a length-3 vector to a `[2, 3]` matrix by prepending one leading axis of size 2.
    /// // This is equivalent to `jax.lax.broadcast(jnp.array([1, 2, 3]), sizes=(2,))` in JAX.
    /// let x = Array::vector(vec![1.0, 2.0, 3.0]);
    /// let y = x.broadcast_leading(vec![2])?;
    /// // `y` has shape [2, 3] with values:
    /// //   [[1.0, 2.0, 3.0],
    /// //    [1.0, 2.0, 3.0]]
    /// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn broadcast_leading(&self, sizes: Vec<usize>) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type().into_owned();
        let mut output_dimensions: Vec<Size> = sizes.iter().map(|size| Size::Static(*size)).collect();
        output_dimensions.extend(input_type.shape().dimensions().iter().copied());
        let output_type = ArrayType::new(input_type.data_type(), Shape::new(output_dimensions));
        let output_axes = (0..input_type.rank()).map(|axis| axis + sizes.len()).collect::<Vec<_>>();
        self.broadcast(output_type, output_axes.as_slice())
    }

    /// Broadcasts `self` to `shape` using the broadcasting semantics of [`Broadcastable`](crate::Broadcastable).
    /// This is the direct analogue of JAX's
    /// [`jnp.broadcast_to`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html),
    /// which itself mirrors NumPy's
    /// [`numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html).
    /// `t.broadcast_to(shape)` right-aligns the input shape with `shape`: input axis `i` corresponds to output axis
    /// `shape.rank() - input.rank() + i`. Each corresponding input dimension must equal the output dimension or be `1`,
    /// in which case it is replicated. Missing leading input dimensions are treated as size `1`, and so a smaller-rank
    /// array can be broadcast to a larger-rank target shape. This is equivalent to
    /// `t.broadcast(output_type, output_axes)` with `output_axes` computed as the trailing range of indices.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`Shape`] to broadcast `self` to. This shape must have rank at least equal to the input's rank and
    ///     must be compatible with the shape of the input in terms of broadcasting semantics.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::operations::manipulation::Broadcast;
    /// # use ryft_core::programs::ProgramError;
    /// # use ryft_core::backends::arrays::Array;
    /// # use ryft_core::types::{Shape, Size};
    /// #
    /// # fn main() -> Result<(), ProgramError> {
    /// // Broadcast a length-3 vector to a `[3, 3]` matrix by replicating the input across the leading axis.
    /// // This is equivalent to `jnp.broadcast_to(jnp.array([1, 2, 3]), (3, 3))` in JAX.
    /// let x = Array::vector(vec![1.0, 2.0, 3.0]);
    /// let y = x.broadcast_to(Shape::new(vec![Size::Static(3), Size::Static(3)]))?;
    /// // `y` has shape [3, 3] with values:
    /// //   [[1.0, 2.0, 3.0],
    /// //    [1.0, 2.0, 3.0],
    /// //    [1.0, 2.0, 3.0]]
    /// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn broadcast_to(&self, shape: Shape) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type().into_owned();
        let input_rank = input_type.rank();
        let offset = shape.rank().saturating_sub(input_rank);
        let output_type = ArrayType::new(input_type.data_type(), shape);
        let output_axes = (0..input_rank).map(|axis| axis + offset).collect::<Vec<_>>();
        self.broadcast(output_type, output_axes.as_slice())
    }
}

impl Broadcast for ArrayType {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
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
                    message: format!("broadcasting output axes map two input axes to output axis {output_axis}"),
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

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<BroadcastOperation>>>>
    Broadcast for V
{
    #[inline]
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(BroadcastOperation::new(output_type, output_axes.to_vec()), Vec::new(), &[self.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::programs::{Program, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::TracingContext;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{DataType, Memory};

    use super::*;

    /// Runs `operation`'s transpose rule against a staged cotangent of the target type and returns the built pullback
    /// program mapping output cotangents to input cotangents.
    fn transposed_broadcast_program(
        operation: &BroadcastOperation,
        input_type: &ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Array, Array> {
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let builder = context.builder().clone();
        let cotangent_atom = builder.borrow_mut().add_input(operation.output_type().clone());
        let cotangent = context.tracer(cotangent_atom, None);
        let contribution = operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type.clone())],
                &[MaybeZero::Value(cotangent)],
            )
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let MaybeZero::Value(contribution) = contribution else {
            panic!("transpose should produce one staged cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);
        let builder =
            Rc::try_unwrap(builder).expect("transpose builder should not have outstanding terms").into_inner();
        builder.build::<Array, Array>(vec![contribution_atom], Placeholder, Placeholder).unwrap()
    }

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
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type), &[]), Ok(vec![output_type.clone()]));

        // Type-level (abstract) broadcasting validates the axis mapping and returns the target type without
        // consuming the borrowed input type.
        assert_eq!(input_type.broadcast(output_type.clone(), &[1]), Ok(output_type.clone()));

        // Interpretation replicates the payload along the added axis.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // The convenience capabilities delegate to `broadcast`.
        let broadcast_leading = Array::vector(vec![1.0, 2.0, 3.0]).broadcast_leading(vec![2]).unwrap();
        assert_eq!(*broadcast_leading.r#type(), output_type);
        assert_eq!(broadcast_leading.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let broadcast_to = Array::vector(vec![1.0, 2.0, 3.0]).broadcast_to(output_type.shape().clone()).unwrap();
        assert_eq!(*broadcast_to.r#type(), output_type);
        assert_eq!(broadcast_to.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]))], &[]),
            Err(TypeError {
                message: "broadcasting input data type f32 does not match output data type f64".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&output_type), &[]),
            Err(TypeError { message: "broadcasting output axes has length 1 but input has rank 2".to_string() }),
        );
        assert_eq!(
            BroadcastOperation::new(output_type.clone(), vec![2])
                .infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string()
            }),
        );
        assert_eq!(
            BroadcastOperation::new(output_type.clone(), vec![1, 1]).infer_output_types(
                &[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(3)]),)],
                &[]
            ),
            Err(TypeError { message: "broadcasting output axes map two input axes to output axis 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))], &[]),
            Err(TypeError { message: "broadcasting input axis 0 has size 2, which is neither 3 nor 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0]).broadcast(output_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError {
                message: "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string(),
            })),
        );

        // Program rendering uses the canonical operation name and includes the captured metadata.
        let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
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
    fn test_broadcast_array() {
        // Mapping a vector's single axis to output axis 1 replicates it across the added leading axis.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output = Array::vector(vec![1.0, 2.0, 3.0]).broadcast(target, &[1]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Mapping a [2, 2] matrix to output axes 0 and 2 replicates it along the added middle axis.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)]));
        let output = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).broadcast(target, &[0, 2]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);

        // Static unit axes are stretched to the corresponding target extent.
        let input = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output = input.broadcast(target, &[0, 1]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Empty targets produce empty payloads.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0), Size::Static(2)]));
        let output = Array::vector(vec![1.0, 2.0]).broadcast(target, &[1]).unwrap();
        assert_eq!(output.to_f64s(), Vec::<f64>::new());
    }

    #[test]
    fn test_lift_broadcast_preserves_memory_and_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap()
            .with_memory(Memory::Host { pinned: true });

        let (output_axes, lifted, batch_axis) = lift_broadcast(&[0], &output_type, 0, 2).unwrap();

        assert_eq!(output_axes, vec![0, 1]);
        assert_eq!(lifted, output_type.with_inserted_dimension(0, Size::Static(2)).unwrap());
        assert_eq!(batch_axis, 0);
    }

    #[test]
    fn test_broadcast_batching_preserves_explicit_mapped_axis_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let physical_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let logical_output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.clone(), 2))
            .unwrap();
        let expected_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
        let input = ArrayBatch::new(
            physical_input_type.clone(),
            Array::from_f64s(physical_input_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        let outputs = BroadcastOperation::new(logical_output_type, vec![0])
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap();

        assert_eq!(outputs[0].r#type().as_ref(), &expected_output_type);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    }

    #[test]
    fn test_broadcast_differentiation_uses_the_primal_output_tangent_type() {
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2)]));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        let primal_output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let tangent_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]));

        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |value| value.broadcast(primal_output_type.clone(), &[1]),
                Array::from_f64s(primal_type, vec![2.0, 4.0]),
                Array::from_f64s(tangent_type, vec![1.0, 3.0]),
            )
            .unwrap();

        assert_eq!(primal.r#type().as_ref(), &primal_output_type);
        assert_eq!(tangent.r#type().as_ref(), &tangent_output_type);
        assert_eq!(tangent.to_f64s(), vec![1.0, 3.0, 1.0, 3.0]);
    }

    #[test]
    fn test_broadcast_transpose_sums_over_added_axes() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type, vec![1]);

        let program = transposed_broadcast_program(&operation, &input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[3] = reduce_sum [axes=[0]] %0
                in (%1)
            "}
            .trim_end(),
        );
        let cotangent = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(contribution.to_f64s(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_transpose_restores_permuted_dimensions() {
        // Input axis 0 (size 2) maps to output axis 2 and input axis 1 (size 3) maps to output axis 0, so the pullback
        // must sum over output axis 1 and swap the surviving axes back into input order.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)]));
        let operation = BroadcastOperation::new(output_type, vec![2, 0]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent_values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let cotangent = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)])),
            cotangent_values.clone(),
        );
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.r#type(), input_type);
        for input_0 in 0..2 {
            for input_1 in 0..3 {
                let expected: f64 = (0..4).map(|reduced| cotangent_values[input_1 * 8 + reduced * 2 + input_0]).sum();
                assert_abs_diff_eq!(contribution.to_f64s()[input_0 * 3 + input_1], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_broadcast_transpose_sums_stretched_unit_axes() {
        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it and restores the unit
        // axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type, vec![0, 1]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.r#type(), input_type);
        assert_eq!(contribution.to_f64s(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_transpose_propagates_symbolic_zero() {
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let input_cotangent_type = input_type.cotangent();
        let output_cotangent_type = output_type.cotangent();

        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Zero(output_cotangent_type)],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_cotangent_type);
    }

    #[test]
    fn test_broadcast_value_and_gradient() {
        // f(x) = sum(broadcast(x, [2, 3], [1])): every input coordinate is replicated twice, so the gradient is 2 at
        // every coordinate.
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| x.broadcast(output_type.clone(), &[1]).unwrap().reduce(&[0, 1], ReductionKind::Sum),
                Array::vector(vec![1.0, 2.0, 3.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 12.0, epsilon = 1e-9);
        assert_eq!(gradient.to_f64s(), vec![2.0, 2.0, 2.0]);
    }
}
