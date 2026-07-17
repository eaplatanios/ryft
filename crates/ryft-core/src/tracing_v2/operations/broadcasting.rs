use crate::batching::BatchAxis;
use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::manipulation::ConvertElementType;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, ReshapeOperation, TransposeOperation};
use crate::operations::sharding::Reshard;
use crate::operations::sharding::ReshardOperation;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::batching::{BatchingContext, BatchingDriver};
use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::programs::types::{TypeError, Typed};
use crate::types::{ArrayType, Size};

/// Value whose elementwise derivative contributions can be mapped between an operand descriptor and the common
/// descriptor inferred for an implicitly broadcasting arithmetic result.
pub(crate) trait ElementwiseDifferentiableValue<T: DifferentiableType>: Value<Type = T> {
    /// Converts and broadcasts this live tangent to `target`.
    fn normalize_elementwise_tangent(&self, target: &T) -> Result<Self, DifferentiationError>;

    /// Applies the adjoint of the implicit conversion and broadcast from `target` to this value's descriptor.
    fn unbroadcast_elementwise_cotangent(&self, target: &T) -> Result<Self, DifferentiationError>;
}

impl<V: Value<Type = crate::types::DataType> + ConvertElementType>
    ElementwiseDifferentiableValue<crate::types::DataType> for V
{
    fn normalize_elementwise_tangent(&self, target: &crate::types::DataType) -> Result<Self, DifferentiationError> {
        if self.r#type().as_ref() == target { Ok(self.clone()) } else { Ok(self.convert_element_type(*target)?) }
    }

    fn unbroadcast_elementwise_cotangent(&self, target: &crate::types::DataType) -> Result<Self, DifferentiationError> {
        if self.r#type().as_ref() == target { Ok(self.clone()) } else { Ok(self.convert_element_type(*target)?) }
    }
}

impl<V> ElementwiseDifferentiableValue<ArrayType> for V
where
    V: Value<Type = ArrayType>
        + Broadcast
        + ConvertElementType
        + crate::operations::manipulation::Reshape
        + crate::operations::manipulation::Transpose
        + Reshard
        + crate::tracing_v2::operations::reduce::Reduce,
{
    fn normalize_elementwise_tangent(&self, target: &ArrayType) -> Result<Self, DifferentiationError> {
        normalize_elementwise_tangent(self, target)
    }

    fn unbroadcast_elementwise_cotangent(&self, target: &ArrayType) -> Result<Self, DifferentiationError> {
        let rank = self.r#type().rank();
        let offset = rank.checked_sub(target.rank()).ok_or_else(|| TypeError {
            message: format!("cannot unbroadcast cotangent type {} to input cotangent type {target}", self.r#type()),
        })?;
        let output_axes = (0..target.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        unbroadcast_elementwise_cotangent(self, target, output_axes.as_slice())
    }
}

/// Converts and broadcasts one live tangent contribution to the exact descriptor of its primal output.
pub(crate) fn normalize_elementwise_tangent<V>(value: &V, target: &ArrayType) -> Result<V, DifferentiationError>
where
    V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Reshard,
{
    let mut value = if value.r#type().data_type() == target.data_type() {
        value.clone()
    } else {
        value.convert_element_type(target.data_type())?
    };
    if value.r#type().as_ref() == target {
        return Ok(value);
    }
    let requires_reshard = value.r#type().sharding() != target.sharding();
    let rank = value.r#type().rank();
    if rank > target.rank() {
        return Err(TypeError {
            message: format!("cannot normalize tangent type {} to output type {target}", value.r#type()),
        }
        .into());
    }
    let offset = target.rank() - rank;
    let output_axes = (0..rank).map(|axis| axis + offset).collect::<Vec<_>>();
    value = value.broadcast(target.clone(), output_axes.as_slice())?;
    // `BroadcastOperation` carries the requested output descriptor, but changing an explicit/manual sharding is a
    // semantic redistribution rather than a metadata-only broadcast. Stage that transition explicitly so backend
    // lowering cannot silently relabel the tangent when the primal result is placed differently from this operand.
    if requires_reshard && let Some(sharding) = target.sharding() {
        value = value.reshard(sharding);
    }
    Ok(value)
}

/// Applies the adjoint of an implicit elementwise broadcast and promotion, returning a contribution with the exact
/// cotangent descriptor of the corresponding primal input.
pub(crate) fn unbroadcast_elementwise_cotangent<V>(
    value: &V,
    target: &ArrayType,
    output_axes: &[usize],
) -> Result<V, DifferentiationError>
where
    V: Value<Type = ArrayType>
        + Broadcast
        + ConvertElementType
        + crate::operations::manipulation::Reshape
        + crate::operations::manipulation::Transpose
        + Reshard
        + crate::tracing_v2::operations::reduce::Reduce,
{
    use crate::tracing_v2::operations::reduce::ReductionKind;

    let value_type = value.r#type();
    if output_axes.len() != target.rank() || output_axes.iter().any(|axis| *axis >= value_type.rank()) {
        return Err(TypeError {
            message: format!(
                "cannot unbroadcast cotangent type {value_type} to input cotangent type {target} using output axes \
                 {output_axes:?}",
            ),
        }
        .into());
    }
    let mut kept_axes = Vec::with_capacity(target.rank());
    for target_axis in 0..target.rank() {
        let value_axis = output_axes[target_axis];
        let target_dimension = target.dimension(target_axis as isize);
        let value_dimension = value_type.dimension(value_axis as isize);
        if target_dimension != value_dimension {
            if target_dimension != Size::Static(1) {
                return Err(TypeError {
                    message: format!(
                        "cannot unbroadcast cotangent axis {value_axis} of size {value_dimension} to input axis \
                         {target_axis} of size {target_dimension}",
                    ),
                }
                .into());
            }
        } else {
            kept_axes.push((target_axis, value_axis));
        }
    }
    let reduce_axes = (0..value_type.rank())
        .filter(|axis| kept_axes.iter().all(|(_, value_axis)| value_axis != axis))
        .collect::<Vec<_>>();
    let mut contribution =
        if reduce_axes.is_empty() { value.clone() } else { value.reduce(reduce_axes.as_slice(), ReductionKind::Sum) };
    let mut kept_axes_by_value = kept_axes.clone();
    kept_axes_by_value.sort_by_key(|(_, value_axis)| *value_axis);
    let permutation = kept_axes
        .iter()
        .map(|kept| kept_axes_by_value.iter().position(|candidate| candidate == kept).unwrap())
        .collect::<Vec<_>>();
    if permutation.iter().enumerate().any(|(axis, position)| axis != *position) {
        contribution = crate::operations::manipulation::Transpose::transpose(&contribution, permutation)?;
    }
    if contribution.r#type().shape() != target.shape() {
        contribution = contribution.reshape(target.shape().clone())?;
    }
    if contribution.r#type().data_type() != target.data_type() {
        contribution = contribution.convert_element_type(target.data_type())?;
    }
    if contribution.r#type().sharding() != target.sharding()
        && let Some(sharding) = target.sharding()
    {
        contribution = contribution.reshard(sharding);
    }
    if contribution.r#type().as_ref() != target {
        let output_axes = (0..target.rank()).collect::<Vec<_>>();
        contribution = contribution.broadcast(target.clone(), output_axes.as_slice())?;
    }
    if contribution.r#type().as_ref() != target {
        return Err(TypeError {
            message: format!(
                "unbroadcasted cotangent type {} does not match required input cotangent type {target}",
                contribution.r#type(),
            ),
        }
        .into());
    }
    Ok(contribution)
}

/// Transpose (vector-Jacobian product) for a [`BroadcastOperation`].
///
/// The pullback of a broadcast is a sum-reduction over every output axis the input was replicated
/// along: the axes of the target type that are not named in `output_axes`, plus the
/// mapped axes whose input extent is `1` stretched to a larger target extent. After the
/// reduction, the surviving axes are reordered into input-axis order when `output_axes`
/// is not monotonically increasing, and stretched unit axes are restored with a reshape so the
/// cotangent matches the input type exactly. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for BroadcastOperation
where
    O: Operation<ArrayType>
        + From<BroadcastOperation>
        + From<crate::operations::manipulation::ConvertElementTypeOperation>
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
        Ok(vec![MaybeZero::Value(unbroadcast_elementwise_cotangent(
            cotangent,
            &input_cotangent_type,
            self.output_axes(),
        )?)])
    }
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

/// Lifts a broadcast's `output_axes` and target shape through one batching level.
///
/// When the input is batched at axis `k` (in the input's unbatched logical shape), the lifted
/// broadcast inserts a batch dimension of size `axis_size` at position `k` in the target shape,
/// places a corresponding batch axis at output position `k_out = k`, and shifts the existing
/// `output_axes` so each previously-mapped output axis `>= k_out` is shifted by one.
/// The new batch input axis itself maps to `k_out`.
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

impl<C: Context<Type = ArrayType, Value: Broadcast>> crate::batching::BatchableOperation<C> for BroadcastOperation {
    fn batch<D: BatchingDriver<C>>(
        &self,
        _context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[crate::batching::ArrayBatch<C::Value>],
    ) -> Result<Vec<crate::batching::ArrayBatch<C::Value>>, crate::batching::BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        match batch_axes[0] {
            None => {
                // Replicated input: the broadcast itself does not change. Pass through.
                let output_value =
                    inputs[0].value().clone().broadcast(self.output_type().clone(), self.output_axes())?;
                Ok(vec![crate::batching::ArrayBatch::new(self.output_type().clone(), output_value, None)?])
            }
            Some(batch_axis) => {
                let axis_size = crate::batching::ArrayBatch::common_batch_size(inputs)?
                    .expect("a mapped input pins the batch size");
                let (lifted_dimensions, mut lifted_target, target_batch_axis) =
                    lift_broadcast(self.output_axes(), self.output_type(), batch_axis, axis_size)?;
                if let Some(sharding) = self.output_type().sharding() {
                    lifted_target.sharding = Some(
                        sharding
                            .with_inserted_dimension(
                                target_batch_axis,
                                crate::batching::ArrayBatch::sharding_for_inputs(inputs)?,
                            )
                            .map_err(|error| crate::batching::BatchingError::MisalignedBatchAxes {
                                message: error.to_string(),
                            })?,
                    );
                }
                let output_value =
                    inputs[0].value().clone().broadcast(lifted_target.clone(), lifted_dimensions.as_slice())?;
                Ok(vec![crate::batching::ArrayBatch::new(
                    lifted_target,
                    output_value,
                    BatchAxis::from_position(target_batch_axis),
                )?])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::parameters::Placeholder;
    use crate::programs::Program;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::tracing::TracingContext;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{DataType, Memory, Shape};

    use super::*;

    /// Runs `operation`'s transpose rule against a staged cotangent of the target type and
    /// returns the built pullback program mapping output cotangents to input cotangents.
    fn transposed_broadcast_program(
        operation: &BroadcastOperation,
        input_type: &ArrayType,
    ) -> Program<TestArray, ArrayOperation<TestArray>, TestArray, TestArray> {
        let mut context = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let builder = context.builder().clone();
        let cotangent_atom = builder.borrow_mut().add_input(operation.output_type().clone());
        let cotangent = context.tracer(cotangent_atom, None);
        let contribution = operation
            .transpose(
                &mut context,
                &crate::programs::regions::EmptyRegionDriver,
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
        builder.build::<TestArray, TestArray>(vec![contribution_atom], Placeholder, Placeholder).unwrap()
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
        let input = crate::batching::ArrayBatch::new(
            physical_input_type.clone(),
            TestArray::new(physical_input_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let context = BatchingContext::new(EagerContext::<TestArray, ArrayOperation<TestArray>>::new(), 2);

        let outputs = BroadcastOperation::new(logical_output_type, vec![0])
            .batch(&context, &crate::programs::regions::EmptyRegionDriver, &[input])
            .unwrap();

        assert_eq!(outputs[0].r#type().as_ref(), &expected_output_type);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    }

    #[test]
    fn test_broadcast_jvp_uses_the_primal_output_tangent_type() {
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2)]));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        let primal_output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let tangent_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]));

        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |value| value.broadcast(primal_output_type.clone(), &[1]),
                TestArray::new(primal_type, vec![2.0, 4.0]),
                TestArray::new(tangent_type, vec![1.0, 3.0]),
            )
            .unwrap();

        assert_eq!(primal.r#type().as_ref(), &primal_output_type);
        assert_eq!(tangent.r#type().as_ref(), &tangent_output_type);
        assert_eq!(tangent.values(), &[1.0, 3.0, 1.0, 3.0]);
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
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)]));
        let operation = BroadcastOperation::new(output_type, vec![2, 0]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent_values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let cotangent = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)])),
            cotangent_values.clone(),
        );
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.r#type(), input_type);
        for input_0 in 0..2 {
            for input_1 in 0..3 {
                let expected: f64 = (0..4).map(|reduced| cotangent_values[input_1 * 8 + reduced * 2 + input_0]).sum();
                assert_abs_diff_eq!(contribution.values[input_0 * 3 + input_1], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_broadcast_transpose_sums_stretched_unit_axes() {
        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it
        // and restores the unit axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type, vec![0, 1]);

        let program = transposed_broadcast_program(&operation, &input_type);
        let cotangent = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let contribution = program.interpret(cotangent).unwrap();
        assert_eq!(*contribution.r#type(), input_type);
        assert_eq!(contribution.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_transpose_propagates_symbolic_zero() {
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let input_cotangent_type = input_type.cotangent();
        let output_cotangent_type = output_type.cotangent();

        let mut context = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let contributions = operation
            .transpose(
                &mut context,
                &crate::programs::regions::EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Zero(output_cotangent_type)],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_cotangent_type);
    }

    #[test]
    fn test_value_and_grad_through_broadcast() {
        // f(x) = sum(broadcast(x, [2, 3], [1])): every input coordinate is replicated
        // twice, so the gradient is 2 at every coordinate.
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| x.broadcast(output_type.clone(), &[1]).unwrap().reduce(&[0, 1], ReductionKind::Sum),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 12.0, epsilon = 1e-9);
        assert_eq!(gradient.values, vec![2.0, 2.0, 2.0]);
    }
}
