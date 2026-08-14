use super::*;

impl<C: Context<Type = ArrayType>, P: RaggedArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        // A replicated ragged operand is rejected before alignment, because the broadcast that materializes its batch
        // axis carries no per-item extents and would drop the metadata that records them.
        for (input, batch_axis) in inputs.iter().zip(batch_axes.iter()) {
            if batch_axis.is_none()
                && let Some(ragged_axis) = input.ragged_axes().first()
            {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' does not support bounded ragged dimension `{}` on a replicated operand",
                        ragged_axis.dimension(),
                    ),
                });
            }
        }
        // Two mapped operands must describe the same mapped extent. Comparing the mapped dimensions validates static
        // extents exactly as `ArrayBatch::common_batch_size` does, and it additionally admits a dynamic mapped extent,
        // which two operands share exactly when it is the same dimension variable.
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
                        "'{DOT_OPERATION_NAME}' operands map different batch extents `{left}` and `{right}`"
                    ),
                },
            });
        }
        // Mixed batched/unbatched: materialize a batch axis on the replicated operand at position 0 (JAX's
        // `matchaxis(0)` convention), then fall through to the both-batched arm of `lift_dot_dimensions`. The active
        // policy owns that materialization, so the mapped extent never has to be a statically known host size: under a
        // dimension-valued policy it stays a first-class extent value grounding the staged broadcast.
        let aligned_inputs: Vec<ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => vec![inputs[0].clone(), P::match_axis(context, &inputs[1], Axis::from(0))?],
            (None, Some(_)) => vec![P::match_axis(context, &inputs[0], Axis::from(0))?, inputs[1].clone()],
        };
        let aligned_axes: Vec<Option<usize>> = aligned_inputs.iter().map(|input| input.batch_axis_position()).collect();
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(self.dimensions(), aligned_axes[0], aligned_axes[1])
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: "'dot' batching failed to lift its dimension numbers for the aligned batch axes".to_string(),
            })?;
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions)
            .with_accumulation_type(self.accumulation_type())
            .with_output_sharding(lift_output_sharding(self.output_sharding(), output_axis, axis_sharding)?);
        let output_batch_axes = [BatchAxis::from_optional_position(output_axis)];
        if aligned_inputs.iter().all(|input| input.ragged_axes().is_empty()) {
            return Ok(lifted_op.interpret_with_batch_axes(context, &aligned_inputs, &output_batch_axes)?.into());
        }

        // A generalized dot lays its result out as the batching dimensions, then the LHS free axes, then the RHS free
        // axes, so each operand axis either lands at a known result axis or is contracted away.
        let dimensions = lifted_op.dimensions();
        let batching_count = dimensions.lhs_batching_dimensions().len();
        let lhs_result = lhs_result_axes(dimensions, aligned_inputs[0].r#type().rank());
        let rhs_result = rhs_result_axes(dimensions, aligned_inputs[1].r#type().rank());
        let operand_output_axes = |rank: usize, batching: &[usize], result: &[usize], offset: usize| {
            (0..rank)
                .map(|axis| {
                    batching.iter().position(|batching_axis| *batching_axis == axis).or_else(|| {
                        result
                            .iter()
                            .position(|result_axis| *result_axis == axis)
                            .map(|index| batching_count + offset + index)
                    })
                })
                .collect::<Vec<_>>()
        };
        let output_axes = [
            operand_output_axes(
                aligned_inputs[0].r#type().rank(),
                dimensions.lhs_batching_dimensions(),
                lhs_result.as_slice(),
                0,
            ),
            operand_output_axes(
                aligned_inputs[1].r#type().rank(),
                dimensions.rhs_batching_dimensions(),
                rhs_result.as_slice(),
                lhs_result.len(),
            ),
        ];

        let contracting_axes = [dimensions.lhs_contracting_dimensions(), dimensions.rhs_contracting_dimensions()];
        let batching_axes = [dimensions.lhs_batching_dimensions(), dimensions.rhs_batching_dimensions()];
        let mut contracted_dimensions = Vec::new();
        let mut output_ragged_axes = Vec::new();
        let mut contraction_inputs = Vec::with_capacity(aligned_inputs.len());
        for (index, input) in aligned_inputs.iter().enumerate() {
            if let Some(ragged_axis) =
                input.ragged_axes().iter().find(|ragged_axis| batching_axes[index].contains(&ragged_axis.axis()))
            {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' does not support bounded ragged dimension `{}` on a batching dimension",
                        ragged_axis.dimension(),
                    ),
                });
            }
            let contracted = input
                .ragged_axes()
                .iter()
                .filter(|ragged_axis| contracting_axes[index].contains(&ragged_axis.axis()))
                .map(|ragged_axis| ragged_axis.dimension().clone())
                .collect::<Vec<_>>();
            for ragged_axis in input.ragged_axes() {
                // A contracted ragged axis is consumed by the zeroing below and reported as evidence; every other one
                // must reach the result together with the axes its per-item extents index.
                if output_axes[index][ragged_axis.axis()].is_none() {
                    continue;
                }
                output_ragged_axes.push(ragged_axis.clone().relocated(output_axes[index].as_slice()).ok_or_else(
                    || BatchingError::UnsupportedOperation {
                        message: format!(
                            "'{DOT_OPERATION_NAME}' contracts an axis carrying the per-item extents of bounded ragged \
                             dimension `{}`",
                            ragged_axis.dimension(),
                        ),
                    },
                )?);
            }
            contraction_inputs.push(if contracted.is_empty() {
                input.clone()
            } else {
                P::pad_contraction_input(context, input, contracting_axes[index])?
            });
            contracted_dimensions.extend(contracted);
        }

        let mut outputs = lifted_op.interpret_with_batch_axes(context, &contraction_inputs, &output_batch_axes)?;
        check_count!("output", outputs, 1, ProgramError);
        let output = ArrayBatch::new(outputs.remove(0).into_value(), output_batch_axes[0])?
            .with_ragged_axes(output_ragged_axes)?;
        Ok(BatchedOutputs::new(vec![output], contracted_dimensions))
    }
}
