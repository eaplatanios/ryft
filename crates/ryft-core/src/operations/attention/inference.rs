use super::*;

/// Borrowed canonical view of attention operand types.
pub(super) struct AttentionOperandTypes<'a> {
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

    /// Forward output type in a backward boundary.
    pub(super) output: Option<&'a ArrayType>,

    /// Forward residual type in a backward boundary.
    pub(super) activation: Option<&'a ArrayType>,

    /// Incoming output cotangent type in a backward boundary.
    pub(super) output_cotangent: Option<&'a ArrayType>,
}

impl<'a> AttentionOperandTypes<'a> {
    /// Parses one forward operation boundary.
    pub(super) fn forward(
        signature: AttentionOperandSignature,
        input_types: &'a [ArrayType],
    ) -> Result<Self, TypeError> {
        let [query, key, value, optional @ ..] = input_types else {
            return Err(TypeError::invalid("attention requires query, key, and value operands"));
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
        let key_value_sequence_lengths = signature.has_key_value_sequence_lengths().then(|| &optional[index]);
        Ok(Self {
            query,
            key,
            value,
            bias,
            mask,
            query_sequence_lengths,
            key_value_sequence_lengths,
            output: None,
            activation: None,
            output_cotangent: None,
        })
    }

    /// Parses one backward operation boundary.
    pub(super) fn backward(
        signature: AttentionOperandSignature,
        input_types: &'a [ArrayType],
    ) -> Result<Self, TypeError> {
        let optional_count = signature.count();
        let mut operands = Self::forward(signature, &input_types[..3 + optional_count])?;
        operands.output = Some(&input_types[3 + optional_count]);
        operands.activation = Some(&input_types[4 + optional_count]);
        operands.output_cotangent = Some(&input_types[5 + optional_count]);
        Ok(operands)
    }
}

/// Returns canonical `[batch, sequence, heads, head_dimension]` dimensions for a `TNH` or `BTNH` operand.
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
            "'{operation_name}' {descriptor} must have rank 3 or 4 but got rank {}",
            dimensions.len(),
        ))),
    }
}

/// Validated dimensions shared by the attention operations' operand contracts.
pub(super) struct AttentionDimensions {
    /// Shared batch dimension of every operand.
    pub(super) batch: Dimension,

    /// Query sequence length.
    pub(super) query_sequence: Dimension,

    /// Number of query heads.
    pub(super) query_heads: usize,

    /// Number of key/value heads; divides `query_heads`, with grouped-query attention when strictly smaller.
    pub(super) key_value_heads: usize,

    /// Head (feature) dimension of every operand.
    pub(super) head_dimension: usize,

    /// Shared floating-point operand data type.
    pub(super) data_type: DataType,
}

/// Validates the shared operand contract of the attention operations — the `BTNH` query/key/value shapes and data
/// types (including the grouped-query heads divisibility), the optional broadcastable bias, and the sliding-window
/// attribute — and returns the validated [`AttentionDimensions`]. Refer to the documentation of
/// [`DotProductAttentionOperation`] for the contract itself.
pub(super) fn validated_attention_operands(
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
            "'{operation_name}' requires floating-point operands but got data type {data_type}"
        )));
    }
    for (descriptor, dimensions, input_type) in [("key", &key, key_type), ("value", &value, value_type)] {
        if input_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} data type {} does not match the query data type {data_type}",
                input_type.data_type(),
            )));
        }
        if dimensions[0] != query[0] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} batch dimension ({}) does not match the query batch dimension \
                     ({})",
                dimensions[0], query[0],
            )));
        }
        if dimensions[3] != query[3] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} head dimension ({}) does not match the query head dimension \
                     ({})",
                dimensions[3], query[3],
            )));
        }
    }
    if value[1] != key[1] {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' value sequence dimension ({}) does not match the key sequence dimension ({})",
            value[1], key[1],
        )));
    }
    if value[2] != key[2] {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' value heads dimension ({}) does not match the key heads dimension ({})",
            value[2], key[2],
        )));
    }
    let Dimension::Static(query_heads) = &query[2] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' query heads dimension must be static but got {}",
            query[2],
        )));
    };
    let Dimension::Static(key_value_heads) = &key[2] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' key/value heads dimension must be static but got {}",
            key[2],
        )));
    };
    let Dimension::Static(head_dimension) = &query[3] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' head dimension must be static but got {}",
            query[3],
        )));
    };
    if *key_value_heads == 0 || *query_heads % *key_value_heads != 0 {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' key/value heads dimension ({}) must divide the query heads dimension ({})",
            key_value_heads, query_heads,
        )));
    }
    for (descriptor, operand_type, expected_data_type) in
        [("bias", bias_type, None), ("mask", mask_type, Some(DataType::Boolean))]
    {
        let Some(operand_type) = operand_type else {
            continue;
        };
        let operand_dimensions = operand_type.shape().dimensions();
        if operand_dimensions.len() > 4 {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} must have rank at most 4 but got rank {}",
                operand_dimensions.len(),
            )));
        }
        let dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - operand_dimensions.len())
            .chain(operand_dimensions.iter().cloned())
            .collect::<Vec<_>>();
        let [operand_batch, operand_heads, operand_rows, operand_columns] = dimensions.as_slice() else {
            unreachable!("attention score operands are normalized to exactly four dimensions")
        };
        if let Some(expected_data_type) = expected_data_type {
            if operand_type.data_type() != expected_data_type {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} must have data type {expected_data_type} but got {}",
                    operand_type.data_type(),
                )));
            }
        } else if !operand_type.data_type().is_numeric() && !operand_type.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias must have a numeric or Boolean data type but got {}",
                operand_type.data_type(),
            )));
        }
        if operand_batch != &Dimension::Static(1) && operand_batch != &query[0] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} batch dimension ({operand_batch}) must be 1 or match the query \
                 batch dimension ({})",
                query[0],
            )));
        }
        if operand_heads != &Dimension::Static(1) && operand_heads != &query[2] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} heads dimension ({operand_heads}) must be 1 or match the query \
                 heads dimension ({})",
                query[2],
            )));
        }
        if operand_rows != &Dimension::Static(1) && operand_rows != &query[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} query-sequence dimension ({operand_rows}) must be 1 or match the \
                 query sequence dimension ({})",
                query[1],
            )));
        }
        if operand_columns != &Dimension::Static(1) && operand_columns != &key[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} key/value-sequence dimension ({operand_columns}) must be 1 or \
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

/// Validates the optional trailing pair of `i32[batch]` sequence-length operands shared by the attention
/// operations: each operand must be a rank-1 `i32` vector whose dimension exactly matches the shared batch dimension.
/// Refer to the documentation of [`DotProductAttentionOperation`] for the padding semantics.
pub(super) fn validated_sequence_length_operands(
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
                "'{operation_name}' {descriptor} must have data type i32 but got {}",
                value_type.data_type(),
            )));
        }
        match value_type.shape().dimensions() {
            [size] if size == batch => {}
            [size] => {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} size ({size}) does not match the batch dimension ({batch})",
                )));
            }
            ref dimensions => {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} must have rank 1 but got rank {}",
                    dimensions.len(),
                )));
            }
        }
    }
    Ok(())
}

/// Validates the optional `(rate, seed)` dropout attribute shared by the attention operations: the rate must lie in
/// the open interval `(0, 1)` (a zero rate is expressed by omitting dropout altogether).
pub(super) fn validated_dropout(operation_name: &str, dropout: Option<(f64, u64)>) -> Result<(), TypeError> {
    if let Some((rate, _)) = dropout {
        if !(rate > 0.0 && rate < 1.0) {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' dropout rate must lie in the open interval (0, 1) but got {rate}",
            )));
        }
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
    let sharding =
        Sharding::new(query_sharding.mesh().clone(), query_sharding.dimensions()[..query_type.rank() - 1].to_vec())
            .and_then(|sharding| activation_type.with_sharding(sharding))
            .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(sharding)
}
