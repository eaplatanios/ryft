use super::*;

/// Borrowed canonical view of attention input types.
pub(super) struct AttentionInputTypes<'a> {
    /// Query type.
    pub(super) query: &'a ArrayType,

    /// Key type.
    pub(super) key: &'a ArrayType,

    /// Value type.
    pub(super) value: &'a ArrayType,

    /// Optional bias type.
    pub(super) bias: Option<&'a ArrayType>,

    /// Optional arbitrary Boolean mask type.
    pub(super) mask: Option<&'a ArrayType>,

    /// Optional query-length type.
    pub(super) query_sequence_lengths: Option<&'a ArrayType>,

    /// Optional key/value-length type.
    pub(super) key_value_sequence_lengths: Option<&'a ArrayType>,

    /// Optional dropout random state type.
    pub(super) dropout_state: Option<&'a ArrayType>,

    /// Forward output type in a backward boundary.
    pub(super) output: Option<&'a ArrayType>,

    /// Forward residual type in a backward boundary.
    pub(super) activation: Option<&'a ArrayType>,

    /// Incoming output cotangent type in a backward boundary.
    pub(super) output_cotangent: Option<&'a ArrayType>,
}

impl<'a> AttentionInputTypes<'a> {
    /// Parses one forward operation boundary.
    pub(super) fn forward(signature: AttentionInputSignature, input_types: &'a [ArrayType]) -> Result<Self, TypeError> {
        let [query, key, value, optional @ ..] = input_types else {
            return Err(TypeError::invalid("attention requires query, key, and value inputs"));
        };
        let mut index = 0;
        let bias = signature.has_bias().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let mask = signature.has_mask().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let query_sequence_lengths = signature.has_query_sequence_lengths().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let key_value_sequence_lengths = signature.has_key_value_sequence_lengths().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let dropout_state = signature.has_dropout_state().then(|| &optional[index]);
        Ok(Self {
            query,
            key,
            value,
            bias,
            mask,
            query_sequence_lengths,
            key_value_sequence_lengths,
            dropout_state,
            output: None,
            activation: None,
            output_cotangent: None,
        })
    }

    /// Parses one backward operation boundary.
    pub(super) fn backward(
        signature: AttentionInputSignature,
        input_types: &'a [ArrayType],
    ) -> Result<Self, TypeError> {
        let optional_count = signature.count();
        let mut types = Self::forward(signature, &input_types[..3 + optional_count])?;
        types.output = Some(&input_types[3 + optional_count]);
        types.activation = Some(&input_types[4 + optional_count]);
        types.output_cotangent = Some(&input_types[5 + optional_count]);
        Ok(types)
    }
}

/// Returns canonical `[batch, sequence, heads, head_dimension]` dimensions for a `TNH` or `BTNH` input.
fn attention_dimensions(
    operation_name: &str,
    descriptor: &str,
    value_type: &ArrayType,
) -> Result<[Dimension; 4], TypeError> {
    match value_type.shape().dimensions() {
        [sequence, heads, head_dimension] => {
            Ok([Dimension::Static(1), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        [batch, sequence, heads, head_dimension] => {
            Ok([batch.clone(), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        dimensions => Err(TypeError::invalid(format!(
            "`{operation_name}` {descriptor} must have rank 3 or 4 but got rank {}",
            dimensions.len(),
        ))),
    }
}

/// Validated dimensions shared by the attention operations' input contracts.
pub(super) struct AttentionDimensions {
    /// Shared batch dimension of the query, key, and value.
    pub(super) batch: Dimension,

    /// Query sequence length.
    pub(super) query_sequence: Dimension,

    /// Number of query heads.
    pub(super) query_heads: usize,

    /// Number of key/value heads; divides `query_heads`, with grouped-query attention when strictly smaller.
    pub(super) key_value_heads: usize,

    /// Head (feature) dimension of the query, key, and value.
    pub(super) head_dimension: usize,

    /// Shared floating-point data type of the query, key, and value.
    pub(super) data_type: DataType,
}

/// Validates the shared input contract of the attention operations — the `BTNH` query/key/value shapes and data
/// types (including the grouped-query heads divisibility), the optional broadcastable bias, and the sliding-window
/// attribute — and returns the validated [`AttentionDimensions`]. Refer to the documentation of
/// [`DotProductAttention`] for the contract itself.
pub(super) fn validated_attention_inputs(
    operation_name: &str,
    query_type: &ArrayType,
    key_type: &ArrayType,
    value_type: &ArrayType,
    bias_type: Option<&ArrayType>,
    mask_type: Option<&ArrayType>,
) -> Result<AttentionDimensions, TypeError> {
    let query = attention_dimensions(operation_name, "query", query_type)?;
    let key = attention_dimensions(operation_name, "key", key_type)?;
    let value = attention_dimensions(operation_name, "value", value_type)?;
    let data_type = query_type.data_type();
    if !data_type.is_floating_point() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` requires floating-point inputs but got data type `{data_type}`"
        )));
    }
    for (descriptor, dimensions, input_type) in [("key", &key, key_type), ("value", &value, value_type)] {
        if input_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} data type `{}` does not match the query data type `{data_type}`",
                input_type.data_type(),
            )));
        }
        if dimensions[0] != query[0] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} batch dimension ({}) does not match the query batch dimension \
                     ({})",
                dimensions[0], query[0],
            )));
        }
        if dimensions[3] != query[3] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} head dimension ({}) does not match the query head dimension \
                     ({})",
                dimensions[3], query[3],
            )));
        }
    }
    if value[1] != key[1] {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` value sequence dimension ({}) does not match the key sequence dimension ({})",
            value[1], key[1],
        )));
    }
    if value[2] != key[2] {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` value heads dimension ({}) does not match the key heads dimension ({})",
            value[2], key[2],
        )));
    }
    let Dimension::Static(query_heads) = &query[2] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` query heads dimension must be static but got {}",
            query[2],
        )));
    };
    let Dimension::Static(key_value_heads) = &key[2] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` key/value heads dimension must be static but got {}",
            key[2],
        )));
    };
    let Dimension::Static(head_dimension) = &query[3] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` head dimension must be static but got {}",
            query[3],
        )));
    };
    if *key_value_heads == 0 || *query_heads % *key_value_heads != 0 {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` key/value heads dimension ({}) must divide the query heads dimension ({})",
            key_value_heads, query_heads,
        )));
    }
    for (descriptor, score_type, expected_data_type) in
        [("bias", bias_type, None), ("mask", mask_type, Some(DataType::Boolean))]
    {
        let Some(score_type) = score_type else {
            continue;
        };
        let score_dimensions = score_type.shape().dimensions();
        if score_dimensions.len() > 4 {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} must have rank at most 4 but got rank {}",
                score_dimensions.len(),
            )));
        }
        let dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - score_dimensions.len())
            .chain(score_dimensions.iter().cloned())
            .collect::<Vec<_>>();
        let [score_batch, score_heads, score_rows, score_columns] = dimensions.as_slice() else {
            unreachable!("attention bias and mask shapes are normalized to exactly four dimensions")
        };
        if let Some(expected_data_type) = expected_data_type {
            if score_type.data_type() != expected_data_type {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` {descriptor} must have data type `{expected_data_type}` but got `{}`",
                    score_type.data_type(),
                )));
            }
        } else if !score_type.data_type().is_numeric() && !score_type.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` bias must have a numeric or Boolean data type but got `{}`",
                score_type.data_type(),
            )));
        }
        if score_batch != &Dimension::Static(1) && score_batch != &query[0] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} batch dimension ({score_batch}) must be 1 or match the query \
                 batch dimension ({})",
                query[0],
            )));
        }
        if score_heads != &Dimension::Static(1) && score_heads != &query[2] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} heads dimension ({score_heads}) must be 1 or match the query \
                 heads dimension ({})",
                query[2],
            )));
        }
        if score_rows != &Dimension::Static(1) && score_rows != &query[1] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} query-sequence dimension ({score_rows}) must be 1 or match the \
                 query sequence dimension ({})",
                query[1],
            )));
        }
        if score_columns != &Dimension::Static(1) && score_columns != &key[1] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} key/value-sequence dimension ({score_columns}) must be 1 or \
                 match the key sequence dimension ({})",
                key[1],
            )));
        }
    }
    Ok(AttentionDimensions {
        batch: query[0].clone(),
        query_sequence: query[1].clone(),
        query_heads: *query_heads,
        key_value_heads: *key_value_heads,
        head_dimension: *head_dimension,
        data_type,
    })
}

/// Validates the optional trailing pair of `i32[batch]` sequence-length inputs shared by the attention operations:
/// each sequence-length input must be a rank-1 `i32` vector whose dimension exactly matches the shared batch
/// dimension. Refer to the documentation of [`DotProductAttention`] for the padding semantics.
pub(super) fn validated_sequence_length_inputs(
    operation_name: &str,
    query_lengths_type: Option<&ArrayType>,
    key_value_lengths_type: Option<&ArrayType>,
    batch: &Dimension,
) -> Result<(), TypeError> {
    for (descriptor, value_type) in
        [("query sequence lengths", query_lengths_type), ("key/value sequence lengths", key_value_lengths_type)]
    {
        let Some(value_type) = value_type else {
            continue;
        };
        if value_type.data_type() != DataType::I32 {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} must have data type `i32` but got `{}`",
                value_type.data_type(),
            )));
        }
        match value_type.shape().dimensions() {
            [size] if size == batch => {}
            [size] => {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` {descriptor} size ({size}) does not match the batch dimension ({batch})",
                )));
            }
            ref dimensions => {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` {descriptor} must have rank 1 but got rank {}",
                    dimensions.len(),
                )));
            }
        }
    }
    Ok(())
}

/// Validates the optional dropout rate and random state shared by the attention definitions. The rate must lie in the
/// open interval `(0, 1)` (a zero rate is expressed by omitting dropout altogether), a random state is only accepted
/// together with a rate, and it must have the state type of a [`RandomAlgorithm`].
pub(super) fn validated_dropout(
    operation_name: &str,
    dropout: Option<f64>,
    dropout_state_type: Option<&ArrayType>,
) -> Result<(), TypeError> {
    if let Some(rate) = dropout
        && !(rate > 0.0 && rate < 1.0)
    {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` dropout rate must lie in the open interval (0, 1) but got {rate}",
        )));
    }
    if let Some(state_type) = dropout_state_type {
        if dropout.is_none() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` received a dropout random state without a dropout rate",
            )));
        }
        RandomAlgorithm::from_state_type(state_type)?;
    }
    Ok(())
}

/// Returns the static [`Shape`] with the provided dimensions.
pub(super) fn static_shape(dimensions: &[usize]) -> Shape {
    Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect())
}

/// Returns the query-data-type `[batch, query_sequence, heads]` or `[query_sequence, heads]` log-sum-exp residual
/// type. The residual inherits the query's sharding with only the trailing head dimension removed.
pub(super) fn attention_activation_type(
    dimensions: &AttentionDimensions,
    query_type: &ArrayType,
) -> Result<ArrayType, TypeError> {
    let dimensions = if query_type.rank() == 3 {
        vec![dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    } else {
        vec![dimensions.batch.clone(), dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    };
    let activation_type = ArrayType::new(query_type.data_type(), Shape::new(dimensions));
    let Some(query_sharding) = query_type.sharding() else {
        return Ok(activation_type);
    };
    let sharding = query_sharding
        .with_dimensions(query_sharding.dimensions()[..query_type.rank() - 1].to_vec())
        .and_then(|sharding| activation_type.with_sharding(sharding))
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(sharding)
}

/// Validates one forward attention boundary (i.e., the [`AttentionInputs`] of `signature` in canonical order) and
/// returns the types of its outputs: the attended output, which is query-shaped at the query data type and propagates
/// the query's metadata (e.g., its sharding), followed by the log-sum-exp residual when `configuration` requests it.
/// Every input must have matching manual variation, and reduced and unreduced inputs are rejected, because softmax and
/// the log-sum-exp residual do not preserve partial-sum or zero-filled replica representations.
pub(super) fn attention_forward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != 3 + signature.count() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expects {} inputs for its optional-input signature but got {}",
            3 + signature.count(),
            input_types.len(),
        )));
    }
    let types = AttentionInputTypes::forward(signature, input_types)?;
    let dimensions = validated_attention_boundary(operation_name, configuration, &types, input_types)?;
    let mut output_types = vec![types.query.clone()];
    if configuration.return_residual() {
        output_types.push(attention_activation_type(&dimensions, types.query)?);
    }
    Ok(output_types)
}

/// Validates one backward attention boundary (i.e., the [`AttentionInputs`] of `signature` in canonical order,
/// followed by the forward output, the log-sum-exp residual, and the output cotangent) and returns the types of its
/// outputs: the cotangents of the query, key, and value, followed by that of a present bias whose cotangent space is
/// not zero-dimensional. The forward output, residual, and output cotangent are compared by data type and shape only,
/// so that per-input metadata such as sharding never fails the structural contract.
pub(super) fn attention_backward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != 6 + signature.count() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expects {} inputs for its optional-input signature but got {}",
            6 + signature.count(),
            input_types.len(),
        )));
    }
    let types = AttentionInputTypes::backward(signature, input_types)?;
    let dimensions = validated_attention_boundary(operation_name, configuration, &types, input_types)?;
    let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
        actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
    };
    if !matches_expected(types.output.unwrap(), types.query) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` output type {} does not match the expected forward output type {}",
            types.output.unwrap(),
            types.query,
        )));
    }
    let expected_output_cotangent_type = types.query.cotangent()?;
    if !matches_expected(types.output_cotangent.unwrap(), &expected_output_cotangent_type) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` output cotangent type {} does not match the expected cotangent type \
             {expected_output_cotangent_type}",
            types.output_cotangent.unwrap(),
        )));
    }
    let expected_activation_type = attention_activation_type(&dimensions, types.query)?;
    if !matches_expected(types.activation.unwrap(), &expected_activation_type) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` activation type {} does not match the expected activation type \
             {expected_activation_type}",
            types.activation.unwrap(),
        )));
    }
    let mut output_types = vec![types.query.cotangent()?, types.key.cotangent()?, types.value.cotangent()?];
    if let Some(bias) = types.bias {
        let bias_cotangent = bias.cotangent()?;
        if !bias_cotangent.is_zero_space() {
            output_types.push(bias_cotangent);
        }
    }
    Ok(output_types)
}

/// Validates the contract shared by the forward and backward attention boundaries (i.e., matching manual variation,
/// the query, key, value, bias, and mask shapes and data types, the sequence lengths, the dropout rate and random
/// state, and the absence of reduced and unreduced inputs) and returns the validated [`AttentionDimensions`].
fn validated_attention_boundary(
    operation_name: &str,
    configuration: AttentionConfiguration,
    types: &AttentionInputTypes<'_>,
    input_types: &[ArrayType],
) -> Result<AttentionDimensions, TypeError> {
    ArrayType::check_matching_manual_variation(operation_name, &input_types.iter().collect::<Vec<_>>())?;
    let dimensions =
        validated_attention_inputs(operation_name, types.query, types.key, types.value, types.bias, types.mask)?;
    validated_sequence_length_inputs(
        operation_name,
        types.query_sequence_lengths,
        types.key_value_sequence_lengths,
        &dimensions.batch,
    )?;
    validated_dropout(operation_name, configuration.dropout(), types.dropout_state)?;
    for input_type in input_types {
        if !input_type.reduced_axes().is_empty() {
            return Err(TypeError::invalid(format!("`{operation_name}` does not support reduced inputs")));
        }
        if !input_type.unreduced_axes().is_empty() {
            return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced inputs")));
        }
    }
    Ok(dimensions)
}
