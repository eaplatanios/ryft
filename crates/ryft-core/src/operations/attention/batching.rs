use super::*;

/// Semantic role of one attention batching operand.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchInput {
    /// Query, key, value, output, or output cotangent in `TNH`/`BTNH` form.
    Tensor,

    /// Broadcastable bias or Boolean mask, right-aligned to `B N T S`.
    Score,

    /// Per-batch sequence lengths.
    Length,

    /// Log-sum-exp statistic in `TN`/`BTN` form.
    Statistic,
}

/// Semantic role of one attention batching result.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchOutput {
    /// Tensor whose logical rank follows the input at this index.
    Tensor(usize),

    /// Log-sum-exp statistic whose logical rank follows the query rank.
    Statistic,

    /// Bias cotangent restored to the logical bias shape at this input index.
    Score(usize),
}

/// Returns the canonical forward operand and result roles for `signature` and `configuration`.
fn attention_forward_batch_roles(
    signature: AttentionOperandSignature,
    configuration: AttentionConfiguration,
) -> (Vec<AttentionBatchInput>, Vec<AttentionBatchOutput>) {
    let mut inputs = vec![AttentionBatchInput::Tensor; 3];
    inputs.extend(signature.has_bias().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_mask().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_query_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend(signature.has_key_value_sequence_lengths().then_some(AttentionBatchInput::Length));
    let mut outputs = vec![AttentionBatchOutput::Tensor(0)];
    outputs.extend(configuration.return_residual().then_some(AttentionBatchOutput::Statistic));
    (inputs, outputs)
}

/// Returns the canonical backward operand and result roles for `signature`.
fn attention_backward_batch_roles(
    signature: AttentionOperandSignature,
) -> (Vec<AttentionBatchInput>, Vec<AttentionBatchOutput>, Option<usize>) {
    let mut inputs = vec![AttentionBatchInput::Tensor; 3];
    let bias_index = signature.has_bias().then_some(inputs.len());
    inputs.extend(signature.has_bias().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_mask().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_query_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend(signature.has_key_value_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend([AttentionBatchInput::Tensor, AttentionBatchInput::Statistic, AttentionBatchInput::Tensor]);
    let outputs =
        vec![AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Tensor(1), AttentionBatchOutput::Tensor(2)];
    (inputs, outputs, bias_index)
}

/// Static-extent normalization adapter shared by the forward and backward fused attention boundaries.
///
/// A mapped level is normalized to a leading prefix and folded into attention's batch axis. Rank-three attention has
/// an implicit logical batch of one, so `[v, T, N, H]` is already the primitive's canonical rank-four form. Rank-four
/// attention instead folds `[v, B, T, N, H]` to `[v * B, T, N, H]`. Biases and masks are first materialized to
/// `[v, B, N, T, S]`, which handles every broadcastable rank and preserves mapped score operands. Results reverse the
/// normalization, and a bias cotangent is reduced over precisely the axes broadcast by its logical operand.
fn batch_attention_static<C, O>(
    operation: &O,
    context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
    inputs: &[ArrayBatch<C::Value>],
    input_roles: &[AttentionBatchInput],
    output_roles: &[AttentionBatchOutput],
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>,
    O: Operation<Type = ArrayType> + InterpretableBatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>>,
{
    check_count!("input", input_roles, inputs.len(), ProgramError);
    let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_count = operation.infer_output_types(input_types.as_slice(), &[])?.len();
        let axes = vec![BatchAxis::replicated(); output_count];
        return operation.interpret_with_batch_axes(context, inputs, axes.as_slice());
    };
    let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
    let static_dimensions = |value_type: &ArrayType| -> Result<Vec<usize>, BatchingError> {
        match value_type.static_shape() {
            Some(shape) => Ok(shape.dimensions().to_vec()),
            None => Err(ProgramError::from(TypeError::invalid(format!(
                "`{}` batching requires statically shaped operands",
                operation.name()
            )))
            .into()),
        }
    };
    let aligned_inputs = inputs
        .iter()
        .map(|input| input.match_axis(0, axis_size, axis_sharding.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let query_type = inputs[0].unbatched_type();
    let query_rank = query_type.rank();
    let key_type = inputs[1].unbatched_type();
    let batch_size = if query_rank == 4 { query_type.shape()[0].value().unwrap() } else { 1 };
    let query_sequence_size = query_type.shape()[query_rank - 3].value().unwrap();
    let head_count = query_type.shape()[query_rank - 2].value().unwrap();
    let key_sequence_size = key_type.shape()[key_type.rank() - 3].value().unwrap();
    let merged_inputs = aligned_inputs
        .iter()
        .zip(input_roles)
        .map(|(aligned, role)| {
            let logical_type = aligned.unbatched_type();
            let aligned_dimensions = static_dimensions(aligned.r#type().as_ref())?;
            let value = match role {
                AttentionBatchInput::Tensor => {
                    if logical_type.rank() == 4 {
                        aligned.value().reshape(static_shape(
                            std::iter::once(axis_size * aligned_dimensions[1])
                                .chain(aligned_dimensions[2..].iter().copied())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        ))?
                    } else {
                        aligned.value().clone()
                    }
                }
                AttentionBatchInput::Score => {
                    let target_dimensions = [axis_size, batch_size, head_count, query_sequence_size, key_sequence_size];
                    let rank = logical_type.rank();
                    let output_axes =
                        std::iter::once(0).chain((0..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    aligned
                        .value()
                        .clone()
                        .broadcast(
                            ArrayType::new(logical_type.data_type(), static_shape(&target_dimensions)),
                            output_axes.as_slice(),
                        )?
                        .reshape(static_shape(&[
                            axis_size * batch_size,
                            head_count,
                            query_sequence_size,
                            key_sequence_size,
                        ]))?
                }
                AttentionBatchInput::Length => aligned.value().reshape(static_shape(&[axis_size * batch_size]))?,
                AttentionBatchInput::Statistic => {
                    if query_rank == 4 {
                        aligned.value().reshape(static_shape(&[
                            axis_size * aligned_dimensions[1],
                            aligned_dimensions[2],
                            aligned_dimensions[3],
                        ]))?
                    } else {
                        aligned.value().clone()
                    }
                }
            };
            Ok(ArrayBatch::replicated(value))
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let merged_types = merged_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_count = operation.infer_output_types(merged_types.as_slice(), &[])?.len();
    check_count!("output", output_roles, output_count, ProgramError);
    let axes = vec![BatchAxis::replicated(); output_count];
    let outputs = operation.interpret_with_batch_axes(context, merged_inputs.as_slice(), axes.as_slice())?;
    outputs
        .into_iter()
        .zip(output_roles)
        .map(|(output, role)| {
            let output_dimensions = static_dimensions(&output.r#type())?;
            let value = match role {
                AttentionBatchOutput::Tensor(input_index) => {
                    let logical_type = inputs[*input_index].unbatched_type();
                    if logical_type.rank() == 4 {
                        output.value().reshape(static_shape(
                            std::iter::once(axis_size)
                                .chain(std::iter::once(logical_type.shape()[0].value().unwrap()))
                                .chain(output_dimensions[1..].iter().copied())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        ))?
                    } else {
                        output.value().clone()
                    }
                }
                AttentionBatchOutput::Statistic => {
                    if query_rank == 4 {
                        output.value().reshape(static_shape(&[
                            axis_size,
                            query_type.shape()[0].value().unwrap(),
                            output_dimensions[1],
                            output_dimensions[2],
                        ]))?
                    } else {
                        output.value().clone()
                    }
                }
                AttentionBatchOutput::Score(input_index) => {
                    let logical_type = inputs[*input_index].unbatched_type();
                    let normalized = output.value().reshape(static_shape(&[
                        axis_size,
                        batch_size,
                        head_count,
                        query_sequence_size,
                        key_sequence_size,
                    ]))?;
                    let offset = 4 - logical_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(logical_type.shape()[axis - offset], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let target = inputs[*input_index].match_axis(0, axis_size, axis_sharding.clone())?;
                    normalized
                        .reduce(reduction_axes.as_slice(), ReductionKind::Sum)
                        .reshape(target.r#type().shape().clone())?
                }
            };
            ArrayBatch::new(value, BatchAxis::new(0))
        })
        .collect()
}

/// Binds a dynamic reshape in the enclosing array IR context.
fn reshape_attention_array<C>(
    context: &C,
    value: C::Value,
    dimensions: Vec<C::Value>,
) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>,
{
    let inputs = std::iter::once(value).chain(dimensions).collect::<Vec<_>>();
    Ok(context.bind(DynamicReshapeOperation::new(), Vec::new(), inputs.as_slice())?.remove(0))
}

/// Multiplies two first-class dimensions in the enclosing array IR context.
fn multiply_attention_dimensions<C>(context: &C, left: &C::Value, right: &C::Value) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DimensionMulOperation>>,
{
    let left_type = left.r#type();
    let left_type = <&DimensionType>::try_from(left_type.as_ref())?;
    let right_type = right.r#type();
    let right_type = <&DimensionType>::try_from(right_type.as_ref())?;
    Ok(context
        .bind(
            DimensionMulOperation::new(left_type, right_type).map_err(ProgramError::from)?,
            Vec::new(),
            &[left.clone(), right.clone()],
        )?
        .remove(0))
}

/// First-class-extent normalization adapter shared by the forward and backward fused attention boundaries.
///
/// This is the dynamic counterpart of [`batch_attention_static`]. It stages the same prefix normalization using
/// mixed `broadcast` and `reshape` operations whose result dimensions are ordinary SSA operands. Consequently a
/// dynamic mapped extent, logical batch, or sequence length never becomes host metadata or a specialization key.
fn batch_attention_dynamic<C, O>(
    operation: &O,
    context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
    inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    input_roles: &[AttentionBatchInput],
    output_roles: &[AttentionBatchOutput],
) -> Result<Vec<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>>, BatchingError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<O> + From<ReduceOperation>,
    O: Operation<Type = ArrayType> + Clone,
{
    check_count!("input", input_roles, inputs.len(), ProgramError);
    let outer_context = context.parent().parent();
    if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
        let values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        return context
            .parent()
            .bind(operation.clone(), Vec::new(), values.as_slice())?
            .into_iter()
            .map(|output| Ok(ArrayBatch::replicated(output)))
            .collect();
    }

    let aligned_inputs = inputs
        .iter()
        .map(|input| DynamicArrayBatchingPolicy::match_axis(context, input, 0.into()))
        .collect::<Result<Vec<_>, _>>()?;
    let query_type = inputs[0].unbatched_type();
    let query_rank = query_type.rank();
    let key_type = inputs[1].unbatched_type();
    let query = C::Value::from_projected(aligned_inputs[0].value().clone());
    let key = C::Value::from_projected(aligned_inputs[1].value().clone());
    let mapped_extent = context.axis_extent().clone();
    let batch_extent = if query_rank == 4 {
        folded_array_dimension(outer_context, &query, 1)?
    } else {
        dimension_constant(outer_context, 1)?
    };
    let query_sequence_extent = folded_array_dimension(outer_context, &query, 1 + query_rank - 3)?;
    let head_extent = folded_array_dimension(outer_context, &query, 1 + query_rank - 2)?;
    let key_sequence_extent = folded_array_dimension(outer_context, &key, 1 + key_type.rank() - 3)?;
    let merged_batch_extent = multiply_attention_dimensions(outer_context, &mapped_extent, &batch_extent)?;

    let merged_values = aligned_inputs
        .iter()
        .zip(input_roles)
        .map(|(aligned, role)| {
            let logical_type = aligned.unbatched_type();
            let value = C::Value::from_projected(aligned.value().clone());
            match role {
                AttentionBatchInput::Tensor => {
                    if logical_type.rank() == 4 {
                        let dimensions = std::iter::once(merged_batch_extent.clone())
                            .chain(
                                (2..aligned.r#type().rank())
                                    .map(|axis| folded_array_dimension(outer_context, &value, axis))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                            .collect();
                        reshape_attention_array(outer_context, value, dimensions)
                    } else {
                        Ok(value)
                    }
                }
                AttentionBatchInput::Score => {
                    let rank = logical_type.rank();
                    let output_axes =
                        std::iter::once(0).chain((0..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    let normalized = broadcast_array(
                        outer_context,
                        value,
                        vec![
                            mapped_extent.clone(),
                            batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                        output_axes,
                        None,
                    )?;
                    reshape_attention_array(
                        outer_context,
                        normalized,
                        vec![
                            merged_batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                    )
                }
                AttentionBatchInput::Length => {
                    reshape_attention_array(outer_context, value, vec![merged_batch_extent.clone()])
                }
                AttentionBatchInput::Statistic => {
                    if query_rank == 4 {
                        let dimensions = std::iter::once(merged_batch_extent.clone())
                            .chain(
                                (2..aligned.r#type().rank())
                                    .map(|axis| folded_array_dimension(outer_context, &value, axis))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                            .collect();
                        reshape_attention_array(outer_context, value, dimensions)
                    } else {
                        Ok(value)
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let merged_values = merged_values.into_iter().map(C::Value::into_projected).collect::<Result<Vec<_>, _>>()?;
    let mut outputs = context.parent().bind(operation.clone(), Vec::new(), merged_values.as_slice())?;
    check_count!("output", output_roles, outputs.len(), ProgramError);
    outputs
        .drain(..)
        .zip(output_roles)
        .map(|(output, role)| {
            let output = C::Value::from_projected(output);
            let value = match role {
                AttentionBatchOutput::Tensor(input_index) => {
                    if inputs[*input_index].unbatched_type().rank() == 4 {
                        let aligned = &aligned_inputs[*input_index];
                        let aligned_value = C::Value::from_projected(aligned.value().clone());
                        let dimensions = (0..aligned.r#type().rank())
                            .map(|axis| folded_array_dimension(outer_context, &aligned_value, axis))
                            .collect::<Result<Vec<_>, _>>()?;
                        reshape_attention_array(outer_context, output, dimensions)?
                    } else {
                        output
                    }
                }
                AttentionBatchOutput::Statistic => {
                    if query_rank == 4 {
                        let dimensions = (0..aligned_inputs[0].r#type().rank() - 1)
                            .map(|axis| folded_array_dimension(outer_context, &query, axis))
                            .collect::<Result<Vec<_>, _>>()?;
                        reshape_attention_array(outer_context, output, dimensions)?
                    } else {
                        output
                    }
                }
                AttentionBatchOutput::Score(input_index) => {
                    let normalized = reshape_attention_array(
                        outer_context,
                        output,
                        vec![
                            mapped_extent.clone(),
                            batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                    )?;
                    let logical_type = inputs[*input_index].unbatched_type();
                    let offset = 4 - logical_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(logical_type.shape()[axis - offset], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let reduced = if reduction_axes.is_empty() {
                        normalized
                    } else {
                        let normalized = C::Value::into_projected(normalized)?;
                        let reduced = context
                            .parent()
                            .bind(ReduceOperation::new(reduction_axes, ReductionKind::Sum), Vec::new(), &[normalized])?
                            .remove(0);
                        C::Value::from_projected(reduced)
                    };
                    let aligned = &aligned_inputs[*input_index];
                    let aligned_value = C::Value::from_projected(aligned.value().clone());
                    let dimensions = (0..aligned.r#type().rank())
                        .map(|axis| folded_array_dimension(outer_context, &aligned_value, axis))
                        .collect::<Result<Vec<_>, _>>()?;
                    reshape_attention_array(outer_context, reduced, dimensions)?
                }
            };
            ArrayBatch::new(C::Value::into_projected(value)?, BatchAxis::new(0))
        })
        .collect()
}

/// Batching rule for [`DotProductAttentionOperation`]: one mapped batch level folds into the operation's own batch
/// dimension through the shared static-extent normalization adapter.
impl<C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>>
    BatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>> for DotProductAttentionOperation
where
    DotProductAttentionOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<StaticArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<StaticArrayBatchingPolicy>>, BatchingError> {
        let (input_roles, output_roles) = attention_forward_batch_roles(self.signature(), self.configuration());
        Ok(batch_attention_static(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// Batching rule for [`DotProductAttentionBackwardOperation`]: the same static-extent normalization as the forward
/// operation, additionally restoring a broadcast bias-cotangent batch dimension.
impl<C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>>
    BatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>> for DotProductAttentionBackwardOperation
where
    DotProductAttentionBackwardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<StaticArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<StaticArrayBatchingPolicy>>, BatchingError> {
        let (input_roles, mut output_roles, bias_index) = attention_backward_batch_roles(self.signature());
        output_roles.extend(
            bias_index
                .filter(|&index| !inputs[index].unbatched_type().cotangent().is_zero_space())
                .map(AttentionBatchOutput::Score),
        );
        Ok(batch_attention_static(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// First-class-extent batching rule for [`DotProductAttentionOperation`].
impl<C> BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>
    for DotProductAttentionOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<DotProductAttentionOperation> + From<ReduceOperation>,
{
    fn batch<D: BatchingDriver<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    ) -> Result<BatchedOutputs<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>, BatchingError>
    {
        let (input_roles, output_roles) = attention_forward_batch_roles(self.signature(), self.configuration());
        Ok(batch_attention_dynamic(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// First-class-extent batching rule for [`DotProductAttentionBackwardOperation`].
impl<C> BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>
    for DotProductAttentionBackwardOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<DotProductAttentionBackwardOperation> + From<ReduceOperation>,
{
    fn batch<D: BatchingDriver<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    ) -> Result<BatchedOutputs<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>, BatchingError>
    {
        let (input_roles, mut output_roles, bias_index) = attention_backward_batch_roles(self.signature());
        output_roles.extend(
            bias_index
                .filter(|&index| !inputs[index].unbatched_type().cotangent().is_zero_space())
                .map(AttentionBatchOutput::Score),
        );
        Ok(batch_attention_dynamic(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}
