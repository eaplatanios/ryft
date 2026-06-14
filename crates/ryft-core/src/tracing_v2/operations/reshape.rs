use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::manipulation::{Reshape, ReshapeOperation, SupportsReshape};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Shape, Size};

/// Convenience trait for values that support reshape.
pub trait ReshapeOps: Reshape + Sized {}

impl<T: Reshape> ReshapeOps for T {}

/// Convenience trait for traceable leaves that can serve as the concrete values of a staged reshape.
///
/// This is the trait bound most reshape-aware transforms use when they need both the abstract leaf
/// contract and the value-level reshape operation.
pub trait ReshapeValue: Value<ArrayType> + ReshapeOps {}

impl<T: Value<ArrayType> + ReshapeOps> ReshapeValue for T {}

/// Lifts a reshape's per-lane `input_shape` / `output_shape` pair through one batching level by
/// inserting a new dimension of size `axis_size` at the supplied input position and finding the
/// matching output position.
///
/// The lifted reshape preserves per-lane semantics in row-major order, which requires that the
/// element count to the left of the batch dimension is the same on both sides:
/// `product(input_shape[..k_in]) == product(output_shape[..k_out])`. When such a `k_out` exists,
/// the helper inserts `axis_size` at position `k_in` in the input shape and at position `k_out`
/// in the output shape, and returns `Some((lifted_input_shape, lifted_output_shape, k_out))`. If
/// no matching position can be found (for example, the batch axis falls in the middle of a
/// reshape that mixes dimensions on both sides), the helper returns `None` and the caller should
/// surface a [`BatchingError::UnsupportedOperation`](crate::batching::BatchingError::UnsupportedOperation)
/// pointing at a future fix that emits an explicit transpose before the reshape.
///
/// Dynamic dimensions in `input_shape[..k_in]` or in any candidate `output_shape[..k_out]` are
/// rejected (they make the prefix product undefined).
///
/// # Parameters
///
///   - `input_shape`: Per-lane shape of the reshape's input.
///   - `output_shape`: Per-lane shape produced by [`ReshapeOperation::output_shape`].
///   - `k_in`: Position of the batched axis in the parent-physical input.
///   - `axis_size`: Size of the batched lane this level introduces.
pub fn lift_reshape_shapes(
    input_shape: &Shape,
    output_shape: &Shape,
    k_in: usize,
    axis_size: usize,
) -> Option<(Shape, Shape, usize)> {
    if k_in > input_shape.rank() {
        return None;
    }
    let mut prefix_product = 1usize;
    for dim in &input_shape.dimensions()[..k_in] {
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        prefix_product = prefix_product.checked_mul(value)?;
    }

    let target_prefix_product = prefix_product;
    let mut output_prefix_product = 1usize;
    let mut k_out = None;
    for (index, dim) in output_shape.dimensions().iter().enumerate() {
        if output_prefix_product == target_prefix_product {
            k_out = Some(index);
            break;
        }
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        output_prefix_product = output_prefix_product.checked_mul(value)?;
    }
    if k_out.is_none() && output_prefix_product == target_prefix_product {
        k_out = Some(output_shape.rank());
    }
    let k_out = k_out?;

    let mut lifted_input_dimensions = input_shape.dimensions().to_vec();
    lifted_input_dimensions.insert(k_in, Size::Static(axis_size));
    let mut lifted_output_dimensions = output_shape.dimensions().to_vec();
    lifted_output_dimensions.insert(k_out, Size::Static(axis_size));

    Some((Shape::new(lifted_input_dimensions), Shape::new(lifted_output_dimensions), k_out))
}

impl<V, O> TransposableOperation<ArrayType, V, O> for ReshapeOperation
where
    V: ReshapeValue,
    O: Operation<ArrayType> + SupportsReshape<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_types[0].shape().clone())?)])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D> DifferentiableOperation<D> for ReshapeOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: ReshapeValue,
    LinearOperationOf<D>: SupportsReshape<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().reshape(self.output_shape().clone())?;
        let tangent = inputs[0].tangent().clone().reshape(self.output_shape().clone())?;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast
        + crate::operations::manipulation::Transpose,
    C,
> crate::tracing_v2::batching::BatchableOperation<V, C> for ReshapeOperation
where
    ReshapeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let Some(k_in) = input_axes[0] else {
            // Lane-uniform: reshape is the same elementwise op (no axis arithmetic needed).
            return crate::tracing_v2::batching::apply_elementwise_batch(self, inputs);
        };
        let input_shape = inputs[0].logical_type()?.shape().clone();
        let Some((_, lifted_output_shape, k_out)) =
            lift_reshape_shapes(&input_shape, self.output_shape(), k_in, axis_size)
        else {
            return Err(crate::batching::BatchingError::UnsupportedOperation {
                message: format!(
                    "missing batching rule for ReshapeOperation with batch axis {k_in} crossing reshape group \
                    boundaries in {input_shape} -> {}",
                    self.output_shape(),
                ),
            }
            .into());
        };
        let lifted_op = ReshapeOperation::new(lifted_output_shape);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[Some(k_out)])
    }
}
