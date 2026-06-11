use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, SupportsBroadcast};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Shape, Size, TypeError};

/// Transpose (vector-Jacobian product) for a [`BroadcastOperation`].
///
/// The pullback of a broadcast is a sum-reduction over every output axis the input was replicated
/// along: the axes of the target type that are not named in `output_axes`, plus the
/// mapped axes whose input extent is `1` stretched to a larger target extent. After the
/// reduction, the surviving axes are reordered into input-axis order when `output_axes`
/// is not monotonically increasing, and stretched unit axes are restored with a reshape so the
/// cotangent matches the input type exactly. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for BroadcastOperation
where
    O: Operation<ArrayType>
        + crate::tracing_v2::operations::reduce::SupportsReduce<ArrayType>
        + crate::operations::manipulation::SupportsTranspose<ArrayType>
        + crate::tracing_v2::operations::reshape::SupportsReshape<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        use crate::operations::manipulation::Transpose;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
        use crate::tracing_v2::operations::reshape::Reshape;

        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        let Cotangent::Staged(cotangent) = &output_cotangents[0] else {
            return Ok(vec![Cotangent::Zero]);
        };
        let input_type = input_types[0];

        // Mapped input axes whose extent matches the target are kept; mapped axes with a static
        // unit extent stretched to a larger target extent are summed like the added axes and
        // restored via the final reshape.
        let mut kept_axes = Vec::with_capacity(self.output_axes().len());
        let mut has_stretched_axes = false;
        for (input_axis, &output_axis) in self.output_axes().iter().enumerate() {
            let input_extent = input_type.dimension(input_axis as isize);
            let target_extent = self.target_type().dimension(output_axis as isize);
            if input_extent == Size::Static(1) && target_extent != Size::Static(1) {
                has_stretched_axes = true;
            } else {
                kept_axes.push((input_axis, output_axis));
            }
        }
        let reduce_axes: Vec<usize> = (0..self.target_type().rank())
            .filter(|axis| kept_axes.iter().all(|(_, output_axis)| output_axis != axis))
            .collect();

        let mut contribution = cotangent.clone();
        if !reduce_axes.is_empty() {
            contribution = contribution.reduce(reduce_axes.as_slice(), ReductionKind::Sum);
        }
        // After the reduction the surviving axes appear in ascending output-axis order; reorder
        // them into input-axis order when the mapped axes were permuted.
        let mut kept_axes_by_output = kept_axes.clone();
        kept_axes_by_output.sort_by_key(|(_, output_axis)| *output_axis);
        let permutation: Vec<usize> = kept_axes
            .iter()
            .map(|kept| kept_axes_by_output.iter().position(|candidate| candidate == kept).unwrap())
            .collect();
        if permutation.iter().enumerate().any(|(index, &position)| index != position) {
            contribution = contribution.transpose(permutation);
        }
        if has_stretched_axes {
            contribution = contribution.reshape(input_type.shape().clone())?;
        }
        Ok(vec![Cotangent::Staged(contribution)])
    }
}

impl<D> DifferentiableOperation<D> for BroadcastOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Broadcast<Output = D::Value>,
    D::Tangent: Broadcast<Output = D::Tangent>,
    LinearOperationOf<D>: SupportsBroadcast<ArrayType>,
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
        let primal = inputs[0].primal().clone().broadcast(self.target_type().clone(), self.output_axes())?;
        let tangent = inputs[0].tangent().clone().broadcast(self.target_type().clone(), self.output_axes())?;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Lifts a broadcast's `output_axes` and target shape through one batching level.
///
/// When the input is batched at axis `k` (in the input's per-lane logical shape), the lifted
/// broadcast inserts a batch dimension of size `axis_size` at position `k` in the target shape,
/// places a corresponding batch axis at output position `k_out = k`, and shifts the existing
/// `output_axes` so each previously-mapped output axis `>= k_out` is shifted by one.
/// The new batch input axis itself maps to `k_out`.
pub fn lift_broadcast(
    output_axes: &[usize],
    target_type: &ArrayType,
    input_batch_axis: usize,
    axis_size: usize,
) -> Result<(Vec<usize>, ArrayType, usize), TypeError> {
    let target_batch_axis = input_batch_axis;
    let mut lifted_target_dimensions: Vec<Size> = target_type.shape().dimensions().to_vec();
    lifted_target_dimensions.insert(target_batch_axis, Size::Static(axis_size));
    let lifted_target = ArrayType::new(target_type.data_type(), Shape::new(lifted_target_dimensions));

    let mut lifted_dimensions = Vec::with_capacity(output_axes.len() + 1);
    for &output_axis in output_axes.iter() {
        let shifted_output_axis = if output_axis >= target_batch_axis { output_axis + 1 } else { output_axis };
        lifted_dimensions.push(shifted_output_axis);
    }
    lifted_dimensions.insert(input_batch_axis, target_batch_axis);

    Ok((lifted_dimensions, lifted_target, target_batch_axis))
}

impl<V: Value<ArrayType> + Broadcast<Output = V>, C> crate::tracing_v2::batching::BatchableOperation<V, C>
    for BroadcastOperation
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        match input_axes[0] {
            None => {
                // Lane-uniform input: the broadcast itself does not change. Pass through.
                let output_value =
                    inputs[0].value().clone().broadcast(self.target_type().clone(), self.output_axes())?;
                Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(self.target_type().clone(), output_value, None)?])
            }
            Some(batch_axis) => {
                let (lifted_dimensions, lifted_target, target_batch_axis) =
                    lift_broadcast(self.output_axes(), self.target_type(), batch_axis, axis_size)?;
                let output_value =
                    inputs[0].value().clone().broadcast(lifted_target.clone(), lifted_dimensions.as_slice())?;
                Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(
                    lifted_target,
                    output_value,
                    Some(target_batch_axis),
                )?])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::differentiation::Cotangent;
    use crate::domains::AbstractDomain;
    use crate::parameters::Placeholder;
    use crate::programs::Program;
    use crate::programs::ProgramBuilder;
    use crate::tracing::AbstractTracingContext;
    use crate::tracing_v2::LinearArrayOperation;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::{TestArray, TestArrayDomain, assert_close};
    use crate::types::DataType;

    use super::*;

    /// Runs `operation`'s transpose rule against a staged cotangent of the target type and
    /// returns the built pullback program mapping output cotangents to input cotangents.
    fn transposed_broadcast_program(
        operation: &BroadcastOperation,
        input_type: &ArrayType,
    ) -> Program<ArrayType, TestArray, LinearArrayOperation<TestArray, TestArray, ArrayType>, TestArray, TestArray>
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType>,
        >::new()));
        let cotangent_atom = builder.borrow_mut().add_input(operation.target_type().clone());
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder.clone());
        let cotangent = context.tracer(cotangent_atom, None);
        let contribution = operation
            .transpose(&mut context, &[input_type], &[Cotangent::Staged(cotangent)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one staged cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);
        let builder =
            Rc::try_unwrap(builder).expect("transpose builder should not have outstanding terms").into_inner();
        builder.build::<TestArray, TestArray>(vec![contribution_atom], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn test_broadcast_transpose_sums_over_added_axes() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let target_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(target_type, vec![1]);

        let program = transposed_broadcast_program(&operation, &input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[3] = reduce_sum %0
                in (%1)
            "}
            .trim_end(),
        );
        let cotangent = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(contribution.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_transpose_restores_permuted_dimensions() {
        // Input axis 0 (size 2) maps to output axis 2 and input axis 1 (size 3) maps to output
        // axis 0, so the pullback must sum over output axis 1 and swap the surviving axes back
        // into input order.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let target_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)]));
        let operation = BroadcastOperation::new(target_type, vec![2, 0]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent_values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let cotangent = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)])),
            cotangent_values.clone(),
        );
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.array_type(), input_type);
        for input_0 in 0..2 {
            for input_1 in 0..3 {
                let expected: f64 = (0..4).map(|reduced| cotangent_values[input_1 * 8 + reduced * 2 + input_0]).sum();
                assert_close(contribution.values[input_0 * 3 + input_1], expected);
            }
        }
    }

    #[test]
    fn test_broadcast_transpose_sums_stretched_unit_axes() {
        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it
        // and restores the unit axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(3)]));
        let target_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(target_type, vec![0, 1]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.array_type(), input_type);
        assert_eq!(contribution.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_transpose_propagates_symbolic_zero() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let target_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(target_type, vec![1]);

        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType>,
        >::new()));
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder);
        let contributions = operation.transpose(&mut context, &[&input_type], &[Cotangent::Zero]).unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
    }

    #[test]
    fn test_value_and_grad_through_broadcast() {
        // f(x) = sum(broadcast(x, [2, 3], [1])): every input coordinate is replicated
        // twice, so the gradient is 2 at every coordinate.
        let target_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let (value, gradient) = crate::tracing_v2::value_and_grad(
            &TestArrayDomain,
            |x| x.broadcast(target_type.clone(), &[1]).unwrap().reduce(&[0, 1], ReductionKind::Sum),
            TestArray::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_close(value.values[0], 12.0);
        assert_eq!(gradient.values, vec![2.0, 2.0, 2.0]);
    }
}
