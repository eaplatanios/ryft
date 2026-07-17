use crate::batching::{BatchAxis, BatchingContext, BatchingDriver, InterpretableBatchableOperation};
use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::manipulation::{Reshape, ReshapeOperation};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Shape, Size};
use crate::{ArrayBatch, BatchableOperation, BatchingError, Broadcast, Transpose};

/// Convenience trait for values that support reshape.
pub trait ReshapeOps: Reshape + Sized {}

impl<T: Reshape> ReshapeOps for T {}

/// Convenience trait for traceable leaves that can serve as the concrete values of a staged reshape.
///
/// This is the trait bound most reshape-aware transforms use when they need both the abstract leaf
/// contract and the value-level reshape operation.
pub trait ReshapeValue: Value<Type = ArrayType> + ReshapeOps {}

impl<T: Value<Type = ArrayType> + ReshapeOps> ReshapeValue for T {}

/// Lifts a reshape's per-item `input_shape` / `output_shape` pair through one batching level by
/// inserting a new dimension of size `axis_size` at the supplied input position and finding the
/// matching output position.
///
/// The lifted reshape preserves per-item semantics in row-major order, which requires that the
/// element count to the left of the batch dimension is the same on both sides:
/// `product(input_shape[..k_in]) == product(output_shape[..k_out])`. When such a `k_out` exists,
/// the helper inserts `axis_size` at position `k_in` in the input shape and at position `k_out`
/// in the output shape, and returns `Some((lifted_input_shape, lifted_output_shape, k_out))`. If
/// no matching position can be found (for example, the batch axis falls in the middle of a
/// reshape that mixes dimensions on both sides), the helper returns `None` and the caller should
/// surface a [`BatchingError::UnsupportedOperation`]
/// pointing at a future fix that emits an explicit transpose before the reshape.
///
/// Dynamic dimensions in `input_shape[..k_in]` or in any candidate `output_shape[..k_out]` are
/// rejected (they make the prefix product undefined).
///
/// # Parameters
///
///   - `input_shape`: Per-item shape of the reshape's input.
///   - `output_shape`: Per-item shape produced by [`ReshapeOperation::output_shape`].
///   - `k_in`: Position of the batched axis in the parent-physical input.
///   - `axis_size`: Size of the batched item this level introduces.
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

impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ReshapeOperation
where
    O: Operation<ArrayType> + From<ReshapeOperation>,
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
        match &outputs[0] {
            MaybeZero::Value(cotangent) => {
                Ok(vec![MaybeZero::Value(cotangent.reshape(inputs[0].r#type().shape().clone())?)])
            }
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
        }
    }
}

/// Forward-mode rule for [`ReshapeOperation`]: `reshape` is structural-linear, so the tangent is the same reshape
/// applied to the operand tangent. The shared all-zero fast path handles a zero operand tangent before this rule is
/// consulted, so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ReshapeOperation
where
    C::Operation: From<ReshapeOperation>,
    C::Value: Reshape,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reshape(self.output_shape().clone())?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshape(self.output_shape().clone())?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ReshapeOperation
where
    C::Value: Broadcast + Transpose,
    ReshapeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let Some(k_in) = inputs[0].batch_axis_position() else {
            // Replicated input: there is no batch axis to thread through the reshape, so interpret it as given and
            // report the output replicated.
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let input_shape = inputs[0].unbatched_type().shape().clone();
        let Some((_, lifted_output_shape, k_out)) =
            lift_reshape_shapes(&input_shape, self.output_shape(), k_in, axis_size)
        else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "missing batching rule for ReshapeOperation with batch axis {k_in} crossing reshape group \
                    boundaries in {input_shape} -> {}",
                    self.output_shape(),
                ),
            });
        };
        let lifted_op = ReshapeOperation::new(lifted_output_shape);
        lifted_op.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(k_out)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::EagerContext;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_reshape_batching_lifts_input_and_output_shapes() {
        // x has shape [2, 6]; outer batch over axis 0 yields per-item rank-1 vectors of size 6,
        // which we reshape to per-item [2, 3]. The combined effect should be a [2, 2, 3] tensor
        // whose leading axis is the original batch dimension.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(2, 6, x_data.clone());

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| row.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)])),
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])),
        );
        // Row-major reshape preserves payload ordering; the lifted op only repositions strides.
        assert_eq!(output.values, x_data);
    }
}
