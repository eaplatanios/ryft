use super::*;

// Alignment delegates mapped-axis materialization to `ArrayExtentBatchingPolicy::match_axis`, so a dynamic mapped
// extent remains a first-class value. The lifted contraction keeps each item independent across the inserted axis.
impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        // Two mapped inputs must describe the same mapped extent. Comparing the mapped dimensions validates static
        // extents exactly as `ArrayBatch::common_batch_size` does, and it additionally admits a dynamic mapped extent,
        // which two inputs share exactly when it is the same dimension variable.
        let mapped_dimension = |index: usize| batch_axes[index].map(|axis| inputs[index].r#type().dimension(axis));
        if let (Some(left), Some(right)) = (mapped_dimension(0), mapped_dimension(1))
            && left != right
        {
            return Err(match (left.value(), right.value()) {
                (Some(expected), Some(actual)) => BatchingError::MismatchedBatchSizes { expected, actual },
                _ => BatchingError::MisalignedBatchAxes {
                    message: format!(
                        // TODO(eaplatanios): Are backticks conventional in Rust for these kinds of error messages?
                        //  If so, can we use them conssitently in the codebase (e.g., replacing single quotes where
                        //  this same convention would apply)?
                        "`{DOT_OPERATION_NAME}` inputs map different batch extents `{left}` and `{right}`"
                    ),
                },
            });
        }
        // Mixed batched/unbatched: materialize a batch axis on the replicated input at position 0 (JAX's `matchaxis(0)`
        // convention), then fall through to the both-batched arm of `lift_dot_dimensions`. The active policy owns that
        // materialization, so the mapped extent never has to be a statically known host size: under a dimension-valued
        // policy it stays a first-class extent value grounding the staged broadcast.
        let aligned_inputs: Vec<ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => vec![inputs[0].clone(), P::match_axis(context, &inputs[1], Axis::from(0))?],
            (None, Some(_)) => vec![P::match_axis(context, &inputs[0], Axis::from(0))?, inputs[1].clone()],
        };
        let aligned_axes: Vec<Option<usize>> = aligned_inputs.iter().map(|input| input.batch_axis_position()).collect();
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(self.dimensions(), aligned_axes[0], aligned_axes[1])
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: format!(
                    "`{DOT_OPERATION_NAME}` batching failed to lift its dimension numbers for the aligned batch axes",
                ),
            })?;
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions)
            .with_accumulation_type(self.accumulation_type())
            .with_output_sharding(lift_output_sharding(self.output_sharding(), output_axis, axis_sharding)?);
        let output_batch_axes = [BatchAxis::from_optional_position(output_axis)];
        Ok(lifted_op.interpret_with_batch_axes(context, &aligned_inputs, &output_batch_axes)?)
    }
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for RaggedDotOperation
where
    RaggedDotOperation: InterpretableOperation<C>,
{
    // The explicit group metadata and both data inputs must be mapped over their leading axis, matching the exact
    // leading-axis restriction of the grouped-dot contract. Explicit `group_sizes` specify the ragged groups.
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        let batch_axes = inputs.iter().map(ArrayBatch::batch_axis_position).collect::<Vec<_>>();
        if batch_axes.iter().all(Option::is_none) {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?);
        }
        if batch_axes.iter().any(|axis| *axis != Some(0)) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` batching requires every input to be mapped over leading axis 0",
                ),
            });
        }
        ArrayBatch::common_batch_size(inputs)?;

        let lhs_rank = inputs[0].r#type().rank() - 1;
        let mode = self.dimensions().mode(lhs_rank)?;
        let (dot_dimensions, _) = lift_dot_dimensions(self.dimensions().dot_dimensions(), Some(0), Some(0)).unwrap();
        let dimensions = RaggedDotDimensionNumbers::new(
            dot_dimensions,
            self.dimensions().lhs_ragged_dimensions().iter().map(|axis| axis + 1).collect(),
            self.dimensions().rhs_group_dimensions().iter().map(|axis| axis + 1).collect(),
        );
        let operation = RaggedDotOperation::new(dimensions);
        let output_axis = if mode == RaggedDotMode::Contracting { 1 } else { 0 };
        Ok(operation.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(output_axis)])?)
    }
}
