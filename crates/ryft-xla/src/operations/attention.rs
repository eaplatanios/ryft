//! cuDNN fused-attention adaptation for canonical Ryft attention boundaries.

use ryft_core::macros::check_count;
use ryft_core::operations::attention::{
    AttentionConfiguration, AttentionImplementation, AttentionInputs, DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
    DOT_PRODUCT_ATTENTION_OPERATION_NAME, DotProductAttentionBackwardOperation, DotProductAttentionOperation,
    dot_product_attention_backward_ir_composition, dot_product_attention_ir_composition,
};
use ryft_core::{
    ArrayType, DataType, Dimension, DomainTracingContext, Operation, ProgramError, ProjectedValue, ReductionKind, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type,
    Value as MlirValue, ValueRef,
};

use crate::experimental::domains::XlaTracer;
use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::compositions::{
    NamedCompositionAttribute, NamedCompositionKey, lower_decomposition_call, named_composition_input_types,
};
use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_tensor_type,
    physical_bound_type, static_dimensions,
};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaDomain, XlaLowerer};

/// Traces the canonical portable attention composition for one validated operation boundary.
pub(crate) fn trace_attention_composition(
    operation: &DotProductAttentionOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = named_composition_input_types(input_types)?;
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let signature = operation.signature();
            let attention_inputs = AttentionInputs::from_values(signature, &inputs)?;
            let (output, activation) =
                dot_product_attention_ir_composition(&attention_inputs, operation.configuration())?;
            Ok(std::iter::once(output.into_value())
                .chain(activation.map(ProjectedValue::into_value))
                .collect::<Vec<_>>())
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Traces the canonical portable attention-backward composition for one validated operation boundary.
pub(crate) fn trace_attention_backward_composition(
    operation: &DotProductAttentionBackwardOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = named_composition_input_types(input_types)?;
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let signature = operation.signature();
            let optional_count = signature.count();
            let attention_inputs = AttentionInputs::from_values(signature, &inputs[..3 + optional_count])?;
            dot_product_attention_backward_ir_composition(
                &attention_inputs,
                &inputs[3 + optional_count],
                &inputs[4 + optional_count],
                &inputs[5 + optional_count],
                operation.configuration(),
            )
            .map(|outputs| outputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Builds the canonical identity of one portable attention composition.
pub(crate) fn attention_composition_key(
    operation: &DotProductAttentionOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> NamedCompositionKey {
    let configuration = operation.configuration();
    let signature = operation.signature();
    NamedCompositionKey {
        name: "ryft.dot_product_attention",
        version: 0,
        decomposition: "ryft.dot_product_attention.v0",
        attributes: vec![
            ("scale", NamedCompositionAttribute::OptionalFloat64(configuration.scale().map(f64::to_bits))),
            ("causal", NamedCompositionAttribute::Boolean(configuration.causal())),
            (
                "local_window_left",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.0)),
            ),
            (
                "local_window_right",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.1)),
            ),
            (
                "dropout",
                NamedCompositionAttribute::OptionalFloat64Unsigned(
                    configuration.dropout().map(|(rate, seed)| (rate.to_bits(), seed)),
                ),
            ),
            ("residual", NamedCompositionAttribute::Boolean(configuration.return_residual())),
            ("bias", NamedCompositionAttribute::Boolean(signature.has_bias())),
            ("mask", NamedCompositionAttribute::Boolean(signature.has_mask())),
            ("query_lengths", NamedCompositionAttribute::Boolean(signature.has_query_sequence_lengths())),
            ("key_value_lengths", NamedCompositionAttribute::Boolean(signature.has_key_value_sequence_lengths())),
        ],
        input_types: input_types.to_vec(),
        output_types: output_types.to_vec(),
    }
}

/// Builds the canonical identity of one portable attention-backward composition.
pub(crate) fn attention_backward_composition_key(
    operation: &DotProductAttentionBackwardOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> NamedCompositionKey {
    let configuration = operation.configuration();
    let signature = operation.signature();
    NamedCompositionKey {
        name: "ryft.dot_product_attention_backward",
        version: 0,
        decomposition: "ryft.dot_product_attention_backward.v0",
        attributes: vec![
            ("scale", NamedCompositionAttribute::OptionalFloat64(configuration.scale().map(f64::to_bits))),
            ("causal", NamedCompositionAttribute::Boolean(configuration.causal())),
            (
                "local_window_left",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.0)),
            ),
            (
                "local_window_right",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.1)),
            ),
            (
                "dropout",
                NamedCompositionAttribute::OptionalFloat64Unsigned(
                    configuration.dropout().map(|(rate, seed)| (rate.to_bits(), seed)),
                ),
            ),
            ("bias", NamedCompositionAttribute::Boolean(signature.has_bias())),
            ("mask", NamedCompositionAttribute::Boolean(signature.has_mask())),
            ("query_lengths", NamedCompositionAttribute::Boolean(signature.has_query_sequence_lengths())),
            ("key_value_lengths", NamedCompositionAttribute::Boolean(signature.has_key_value_sequence_lengths())),
        ],
        input_types: input_types.to_vec(),
        output_types: output_types.to_vec(),
    }
}

/// Canonical `[batch, sequence, heads, head_dimension]` dimensions of one rank-three or rank-four attention type.
fn attention_dimensions(input_type: &ArrayType) -> Result<[Dimension; 4], LoweringError> {
    match input_type.shape().dimensions() {
        [sequence, heads, head_dimension] => {
            Ok([Dimension::Static(1), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        [batch, sequence, heads, head_dimension] => {
            Ok([batch.clone(), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        _ => Err(LoweringError::UnsupportedOp {
            op: format!(
                "{} operand must have rank 3 or 4 but got rank {}",
                DOT_PRODUCT_ATTENTION_OPERATION_NAME,
                input_type.rank(),
            ),
        }),
    }
}

/// Returns the physical maximum extent of one attention dimension.
fn attention_physical_extent(dimension: &Dimension) -> Result<usize, LoweringError> {
    match dimension {
        Dimension::Static(extent) => Ok(*extent),
        Dimension::Dynamic(variable) => {
            variable.bounds().upper().and_then(|upper| upper.checked_sub(1)).ok_or_else(|| {
                LoweringError::UnsupportedOp {
                    op: format!(
                        "{DOT_PRODUCT_ATTENTION_OPERATION_NAME} dimension {variable} needs a finite positive physical \
                         bound",
                    ),
                }
            })
        }
    }
}

/// Returns an [`ArrayType`] with the provided data type and dimensions used by the attention lowerings.
pub(super) fn attention_array_type<D: Clone + Into<Dimension>>(data_type: DataType, dimensions: &[D]) -> ArrayType {
    ArrayType::new(data_type, Shape::new(dimensions.iter().cloned().map(Into::into).collect()))
}

/// The cuDNN fMHA proto-JSON dot-dimension-number block of the two forward attention matrix products under the
/// `BTNH` operand convention (`bmm1 = Q·Kᵀ` contracting the head axis, `bmm2 = P·V` contracting the key/value
/// sequence axis), exactly as validated on hardware.
const FMHA_FORWARD_DOT_DIMENSION_NUMBERS: &str = "\"bmm1_dot_dimension_numbers\":\
     {\"lhs_contracting_dimensions\":[\"3\"],\"rhs_contracting_dimensions\":[\"3\"],\
     \"lhs_batch_dimensions\":[\"0\",\"2\"],\"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]}";

/// The cuDNN fMHA proto-JSON dot-dimension-number block of the four backward gradient matrix products
/// (`bmm1_grad_gemm1 = dS·K`, `bmm1_grad_gemm2 = dSᵀ·Q`, `bmm2_grad_gemm1 = dO·Vᵀ`, `bmm2_grad_gemm2 = Pᵀ·dO`),
/// exactly as validated on hardware. These replace the forward `bmm1`/`bmm2` blocks in the backward call's
/// configuration.
const FMHA_BACKWARD_DOT_DIMENSION_NUMBERS: &str = "\"bmm1_grad_gemm1_dot_dimension_numbers\":\
     {\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\
     \"lhs_batch_dimensions\":[\"0\",\"1\"],\"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm1_grad_gemm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_grad_gemm1_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_grad_gemm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"3\"],\"lhs_batch_dimensions\":[\"0\",\"2\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]}";

/// Renders the `GpuBackendConfig` proto-JSON string of one cuDNN fMHA custom call — the raw backend-config form
/// (not typed-FFI dictionaries) that XLA's cuDNN custom-call compiler parses — with the score scale, the
/// intermediate `[batch, q_heads, q_seq, kv_seq]` score shape at the operand element type (load-bearing on the
/// backward path, where the `P` descriptor and statistic strides derive from it), the mask kind, the
/// forward or backward dot-dimension-number block, the dropout rate and seed (the hardware-validated defaults
/// `0.0`/`42` when dropout is off), and the sliding-window length.
fn fmha_backend_config(
    element_type: &str,
    intermediate_dimensions: [usize; 4],
    scale: f64,
    mask_type: &str,
    dot_dimension_numbers: &str,
    dropout: Option<(f64, u64)>,
    sliding_window_length: usize,
) -> String {
    let [batch, heads, query_sequence, key_value_sequence] = intermediate_dimensions;
    // The `{:?}` rate formatting keeps the validated `0.0` spelling for the off state while rendering set rates
    // with their shortest round-trip form.
    let dropout_rate = dropout.map_or(0.0, |(rate, _)| rate);
    let seed = dropout.map_or(42, |(_, seed)| seed);
    format!(
        "{{\"operation_queue_id\":\"0\",\"cudnn_fmha_backend_config\":{{\
         \"algorithm\":{{\"algo_id\":\"0\",\"math_type\":\"TENSOR_OP_MATH\",\"tuning_knobs\":\
         {{\"17\":\"1\",\"24\":\"0\"}},\"is_cudnn_frontend\":true,\"workspace_size\":\"0\"}},\
         \"fmha_scale\":{scale},\"intermediate_tensor_shape\":{{\"element_type\":\"{element_type}\",\
         \"dimensions\":[\"{batch}\",\"{heads}\",\"{query_sequence}\",\"{key_value_sequence}\"],\
         \"tuple_shapes\":[],\"layout\":{{\"dim_level_types\":[],\"dim_unique\":[],\"dim_ordered\":[],\
         \"minor_to_major\":[\"3\",\"2\",\"1\",\"0\"],\"tiles\":[],\"element_size_in_bits\":\"0\",\
         \"memory_space\":\"0\",\"index_primitive_type\":\"PRIMITIVE_TYPE_INVALID\",\
         \"pointer_primitive_type\":\"PRIMITIVE_TYPE_INVALID\",\
         \"dynamic_shape_metadata_prefix_bytes\":\"0\"}},\
         \"is_dynamic_dimension\":[false,false,false,false]}},\
         \"is_flash_attention\":true,\"mask_type\":\"{mask_type}\",\
         {dot_dimension_numbers},\
         \"dropout_rate\":{dropout_rate:?},\"seed\":{seed},\"sliding_window_length\":{sliding_window_length},\
         \"max_seg_per_batch\":1,\"is_paged_attention\":false}}}}",
    )
}

/// Returns the cuDNN fMHA custom-call target name for the provided feature set: the name varies only with the bias
/// operand and dropout (`__cudnn$fmha[ScaleBias]Softmax[Dropout][Backward]`) — padding, sliding windows, and
/// grouped-query head counts are carried by the backend configuration and the operand shapes instead.
fn fmha_target_name(has_bias: bool, has_dropout: bool, backward: bool) -> String {
    format!(
        "__cudnn$fmha{}Softmax{}{}",
        if has_bias { "ScaleBias" } else { "" },
        if has_dropout { "Dropout" } else { "" },
        if backward { "Backward" } else { "" },
    )
}

/// Returns the cuDNN fMHA `mask_type` configuration value for the provided built-in mask and sequence-length
/// presence: the padding mask kinds compose the built-in mask with the appended `i32[batch]` sequence-length
/// operands.
fn fmha_mask_type(causal: bool, has_sequence_lengths: bool) -> &'static str {
    match (causal, has_sequence_lengths) {
        (false, false) => "NO_MASK",
        (true, false) => "CAUSAL",
        (false, true) => "PADDING",
        (true, true) => "PADDING_CAUSAL",
    }
}

/// Result of checking one attention boundary against the complete cuDNN fMHA contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FmhaEligibility {
    /// The boundary can use the fused custom call without changing its semantics.
    Eligible,

    /// The boundary must use the portable decomposition, with the reason used by forced-fused diagnostics.
    Ineligible(&'static str),
}

/// Checks one attention boundary against the semantic and physical requirements of the cuDNN fMHA adapter.
fn fmha_eligibility(
    configuration: AttentionConfiguration,
    signature: ryft_core::operations::attention::AttentionInputSignature,
    input_types: &[ArrayType],
    collective_state: &CollectiveLoweringState,
) -> Result<FmhaEligibility, LoweringError> {
    if configuration.local_window().is_some_and(|(_, right)| {
        right != 0
            && (!configuration.causal()
                || signature.has_query_sequence_lengths()
                || signature.has_key_value_sequence_lengths())
    }) {
        return Ok(FmhaEligibility::Ineligible(
            "a nonzero right local-window radius requires causal masking without sequence-length padding",
        ));
    }
    if !collective_state.target_platform().is_some_and(|platform| platform == "cuda") {
        return Ok(FmhaEligibility::Ineligible("the target platform is not CUDA"));
    }
    if !matches!(input_types[0].data_type(), DataType::BF16 | DataType::F16) {
        return Ok(FmhaEligibility::Ineligible("the operand data type is not `f16` or `bf16`"));
    }
    let [_, _, _, head_dimension] = attention_dimensions(&input_types[0])?;
    let Some(head_dimension) = head_dimension.value() else {
        return Ok(FmhaEligibility::Ineligible("the head dimension is not static"));
    };
    if head_dimension % 8 != 0 {
        return Ok(FmhaEligibility::Ineligible("the head dimension is not divisible by eight"));
    }
    if input_types[..3].iter().any(|r#type| r#type.static_shape().is_none())
        && !signature.has_query_sequence_lengths()
        && !signature.has_key_value_sequence_lengths()
    {
        return Ok(FmhaEligibility::Ineligible("bounded-dynamic operands require query or key/value sequence lengths"));
    }
    Ok(FmhaEligibility::Eligible)
}

/// Canonical operand indices of one attention boundary.
fn attention_input_indices(
    signature: ryft_core::operations::attention::AttentionInputSignature,
) -> AttentionInputs<usize> {
    let indices = (0..3 + signature.count()).collect::<Vec<_>>();
    AttentionInputs::from_values(signature, indices.as_slice())
        .expect("the generated index list exactly matches the attention operand signature")
}

/// Adds the implicit batch dimension required by the cuDNN `BTNH` ABI to one physical rank-three operand.
fn normalize_fmha_operand<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if r#type.rank() == 4 {
        return Ok(value);
    }
    let mut dimensions = vec![1];
    dimensions.extend(static_dimensions(r#type)?);
    let normalized = block.append_operation(stable_hlo::reshape(value, dimensions.as_slice(), location)?)?;
    Ok(normalized.result(0).expect("stablehlo.reshape should return one result").as_ref())
}

/// Broadcasts one physical bias or mask from its normalized rank-four shape to the complete fMHA score shape.
fn broadcast_fmha_score_operand<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    score_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let mut dimensions = vec![1; 4 - r#type.rank()];
    dimensions.extend(static_dimensions(r#type)?);
    let value = if r#type.rank() == 4 {
        value
    } else {
        let reshaped = block.append_operation(stable_hlo::reshape(value, dimensions.as_slice(), location)?)?;
        reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref()
    };
    let broadcast = block.append_operation(stable_hlo::broadcast(
        value,
        lower_tensor_type(score_type, context, location)?,
        &[0, 1, 2, 3],
        location,
    )?)?;
    Ok(broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref())
}

/// Converts the optional Boolean mask to additive bias and combines it with the optional user bias at the physical
/// `[batch, heads, query_sequence, key_value_sequence]` score shape expected by cuDNN.
#[allow(clippy::too_many_arguments)]
fn prepare_fmha_bias<'b, 'c: 'b, 't: 'c>(
    inputs: &AttentionInputs<usize>,
    physical_inputs: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    score_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Option<ValueRef<'b, 'c, 't>>, LoweringError> {
    let bias = inputs
        .bias
        .map(|index| {
            let bias = broadcast_fmha_score_operand(
                physical_inputs[index],
                &physical_bound_type(&input_types[index])?,
                score_type,
                block,
                context,
                location,
            )?;
            if input_types[index].data_type() == score_type.data_type() {
                return Ok::<_, LoweringError>(bias);
            }
            let converted = block.append_operation(stable_hlo::convert(
                bias,
                lower_tensor_type(score_type, context, location)?,
                location,
            )?)?;
            Ok::<_, LoweringError>(converted.result(0).expect("stablehlo.convert should return one result").as_ref())
        })
        .transpose()?;
    let mask = inputs
        .mask
        .map(|index| {
            let mask = broadcast_fmha_score_operand(
                physical_inputs[index],
                &physical_bound_type(&input_types[index])?,
                &score_type.clone().with_data_type(DataType::Boolean),
                block,
                context,
                location,
            )?;
            let tensor_type = lower_tensor_type(score_type, context, location)?;
            let zero = lower_f64_constant_splat(0.0, score_type, tensor_type, block, context, location)?;
            // These are the pinned JAX/cuDNN wrapper's deliberately conservative finite mask sentinels. They avoid
            // a cuDNN subtraction bug that can surface when two values sit at the element type's extreme finite
            // magnitude.
            let masked_value = if score_type.data_type() == DataType::F16 { -32_768.0 } else { -2_199_023_255_552.0 };
            let masked = lower_f64_constant_splat(masked_value, score_type, tensor_type, block, context, location)?;
            let selected = block.append_operation(stable_hlo::select(mask, zero, masked, location)?)?;
            Ok::<_, LoweringError>(selected.result(0).expect("stablehlo.select should return one result").as_ref())
        })
        .transpose()?;
    match (bias, mask) {
        (Some(bias), Some(mask)) => {
            let combined = block.append_operation(stable_hlo::add(bias, mask, location)?)?;
            Ok(Some(combined.result(0).expect("stablehlo.add should return one result").as_ref()))
        }
        (Some(bias), None) => Ok(Some(bias)),
        (None, Some(mask)) => Ok(Some(mask)),
        (None, None) => Ok(None),
    }
}

/// Synthesizes one full per-batch sequence-length vector for the sequence axis of `source`.
#[allow(clippy::too_many_arguments)]
fn synthesize_fmha_sequence_lengths<'b, 'c: 'b, 't: 'c>(
    source: ValueRef<'b, 'c, 't>,
    source_axis: usize,
    batch: &Dimension,
    physical_batch: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let size = block.append_operation(stable_hlo::get_dimension_size(source, source_axis, location)?)?;
    let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
    let physical_type = attention_array_type(DataType::I32, &[physical_batch]);
    let lengths = block.append_operation(stable_hlo::broadcast(
        size,
        lower_tensor_type(&physical_type, context, location)?,
        &[],
        location,
    )?)?;
    let mut lengths = lengths.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    if matches!(batch, Dimension::Dynamic(_)) {
        let batch_size = block.append_operation(stable_hlo::get_dimension_size(source, 0, location)?)?;
        let batch_size = batch_size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
        let logical_type = attention_array_type(DataType::I32, std::slice::from_ref(batch));
        let dynamic = block.append_operation(stable_hlo::set_dimension_size(
            lengths,
            batch_size,
            lower_tensor_type(&logical_type, context, location)?,
            0,
            location,
        )?)?;
        lengths = dynamic.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
        lengths = lower_physical_bound_value(lengths, &logical_type, 0.0, block, context, location)?;
    }
    Ok(lengths)
}

/// Lowers one [`DotProductAttentionOperation`] through either its canonical typed decomposition or the cuDNN fused ABI.
///
/// Portable mode always calls the private decomposition registered for the operation. Automatic mode does the same
/// when [`fmha_eligibility`] rejects the fused ABI. Forced fused mode instead reports the eligibility reason, so it
/// never silently changes semantics. An eligible `bf16`/`f16` CUDA operation emits
/// `__cudnn$fmha[ScaleBias]Softmax[Dropout]` with physical `BNTH` inputs and restores the operation's logical rank and
/// dynamic dimensions on its results. Optional bias and sequence-length operands retain the canonical
/// [`AttentionInputs`] order until the final ABI assembly.
///
/// This function owns only fused-kernel eligibility, physical-bound materialization, ABI configuration, and result
/// refinement. Grouped-head expansion, visibility, softmax, and portable attention mathematics remain exclusively in
/// the core decomposition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_dot_product_attention_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &DotProductAttentionOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let signature = operation.signature();
    if input_values.len() != 3 + signature.count() || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: 3, actual: input_values.len() }.into());
    }
    let configuration = operation.configuration();
    let expected_output_count = if configuration.return_residual() { 2 } else { 1 };
    check_count!("output", output_types, expected_output_count, ProgramError);
    let indices = attention_input_indices(signature);
    let uses_sequence_lengths = signature.has_query_sequence_lengths() || signature.has_key_value_sequence_lengths();
    let eligibility = fmha_eligibility(configuration, signature, input_types, collective_state)?;
    if configuration.implementation() == AttentionImplementation::Portable
        || matches!(eligibility, FmhaEligibility::Ineligible(_))
    {
        if configuration.implementation() == AttentionImplementation::Fused {
            if configuration.dropout().is_some() {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "`{DOT_PRODUCT_ATTENTION_OPERATION_NAME}` dropout is only supported by the fused CUDA lowering",
                    ),
                });
            }
            let FmhaEligibility::Ineligible(reason) = eligibility else { unreachable!() };
            return Err(LoweringError::UnsupportedOp {
                op: format!("the fused attention implementation is unavailable because {reason}"),
            });
        }
        let decomposition = collective_state
            .named_compositions
            .as_ref()
            .and_then(|functions| functions.get_attention(operation, input_types, output_types))
            .ok_or_else(|| LoweringError::UnsupportedOp {
                op: format!("missing typed decomposition for `{}`", operation.name()),
            })?;
        return lower_decomposition_call(decomposition, input_values, output_types, block, context, location);
    }
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_dimensions(&input_types[0])?;
    let [_, key_value_sequence, _, _] = attention_dimensions(&input_types[1])?;
    let heads = heads.value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DOT_PRODUCT_ATTENTION_OPERATION_NAME} heads dimension must be static"),
    })?;
    let head_dimension = head_dimension.value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DOT_PRODUCT_ATTENTION_OPERATION_NAME} head dimension must be static"),
    })?;
    let physical_batch = attention_physical_extent(&batch)?;
    let physical_query_sequence = attention_physical_extent(&query_sequence)?;
    let physical_key_value_sequence = attention_physical_extent(&key_value_sequence)?;
    {
        let mut physical_inputs = input_values
            .iter()
            .zip(input_types)
            .map(|(value, r#type)| lower_physical_bound_value(*value, r#type, 0.0, block, context, location))
            .collect::<Result<Vec<_>, _>>()?;
        for index in 0..3 {
            let physical_type = physical_bound_type(&input_types[index])?;
            physical_inputs[index] = normalize_fmha_operand(physical_inputs[index], &physical_type, block, location)?;
        }
        let score_type = attention_array_type(
            data_type,
            &[physical_batch, heads, physical_query_sequence, physical_key_value_sequence],
        );
        let bias = prepare_fmha_bias(
            &indices,
            physical_inputs.as_slice(),
            input_types,
            &score_type,
            block,
            context,
            location,
        )?;
        let sequence_lengths = if uses_sequence_lengths {
            let query_lengths = match indices.query_sequence_lengths {
                Some(index) => physical_inputs[index],
                None => synthesize_fmha_sequence_lengths(
                    input_values[0],
                    input_types[0].rank() - 3,
                    &batch,
                    physical_batch,
                    block,
                    context,
                    location,
                )?,
            };
            let key_value_lengths = match indices.key_value_sequence_lengths {
                Some(index) => physical_inputs[index],
                None => synthesize_fmha_sequence_lengths(
                    input_values[1],
                    input_types[1].rank() - 3,
                    &batch,
                    physical_batch,
                    block,
                    context,
                    location,
                )?,
            };
            Some((query_lengths, key_value_lengths))
        } else {
            None
        };
        let element_type = if data_type == DataType::BF16 { "BF16" } else { "F16" };
        let backend_config = fmha_backend_config(
            element_type,
            [physical_batch, heads, physical_query_sequence, physical_key_value_sequence],
            configuration.scale().unwrap_or(1.0 / (head_dimension as f64).sqrt()),
            fmha_mask_type(configuration.causal(), sequence_lengths.is_some()),
            FMHA_FORWARD_DOT_DIMENSION_NUMBERS,
            configuration.dropout(),
            configuration.local_window().map_or(0, |(left, _)| left + 1),
        );
        // The operand order matches the traced operand order exactly: the bias sits after the value and the
        // `i32[batch]` sequence lengths trail. The workspace result is declared at size zero (the compiler resizes
        // it as needed), and the training form inserts the `f32[b, n, t]` activation statistic before it.
        let mut operands = physical_inputs[..3].to_vec();
        if let Some(bias) = bias {
            operands.push(bias);
        }
        if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
            operands.extend([query_lengths, key_value_lengths]);
        }
        let mut operand_layouts = vec![vec![3, 2, 1, 0]; if bias.is_some() { 4 } else { 3 }];
        if sequence_lengths.is_some() {
            operand_layouts.extend([vec![0], vec![0]]);
        }
        let attended_type =
            attention_array_type(data_type, &[physical_batch, heads, physical_query_sequence, head_dimension]);
        let mut custom_call_output_types = vec![lower_tensor_type(&attended_type, context, location)?];
        let mut result_layouts = vec![vec![3, 1, 2, 0]];
        if configuration.return_residual() {
            let activation_type =
                attention_array_type(DataType::F32, &[physical_batch, heads, physical_query_sequence]);
            custom_call_output_types.push(lower_tensor_type(&activation_type, context, location)?);
            result_layouts.push(vec![2, 1, 0]);
        }
        custom_call_output_types.push(lower_tensor_type(&attention_array_type(DataType::U8, &[0]), context, location)?);
        result_layouts.push(vec![0]);
        let custom_call = block.append_operation(stable_hlo::custom_call(
            operands.as_slice(),
            fmha_target_name(bias.is_some(), configuration.dropout().is_some(), false).as_str(),
            false,
            Some(context.string_attribute(backend_config.as_str()).as_ref()),
            CustomCallApiVersion::StatusReturning,
            &[],
            Some(CustomCallMemoryLayouts { operands: operand_layouts, results: result_layouts }),
            &[],
            None,
            &custom_call_output_types,
            location,
        )?)?;
        let attended =
            custom_call.result(0).expect("stablehlo.custom_call should return the attention output").as_ref();
        let result = block.append_operation(stable_hlo::transpose(attended, &[0, 2, 1, 3], location)?)?;
        let mut result = result.result(0).expect("stablehlo.transpose should return one result").as_ref();
        if input_types[0].rank() == 3 {
            let reshaped = block.append_operation(stable_hlo::reshape(
                result,
                &[physical_query_sequence, heads, head_dimension],
                location,
            )?)?;
            result = reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref();
        }
        let output_sources = (0..input_types[0].rank()).map(|axis| (input_values[0], axis)).collect::<Vec<_>>();
        let result = lower_restore_dynamic_dimensions(
            result,
            &output_types[0],
            output_sources.as_slice(),
            block,
            context,
            location,
        )?;
        let mut results = vec![result];
        if configuration.return_residual() {
            let activation = custom_call
                .result(1)
                .expect("stablehlo.custom_call should return the activation statistic")
                .as_ref();
            let activation = block.append_operation(stable_hlo::transpose(activation, &[0, 2, 1], location)?)?;
            let mut activation = activation.result(0).expect("stablehlo.transpose should return one result").as_ref();
            let physical_activation_type = if input_types[0].rank() == 3 {
                attention_array_type(data_type, &[physical_batch, physical_query_sequence, heads])
            } else {
                physical_bound_type(&output_types[1])?
            };
            let converted = block.append_operation(stable_hlo::convert(
                activation,
                lower_tensor_type(&physical_activation_type, context, location)?,
                location,
            )?)?;
            activation = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            if input_types[0].rank() == 3 {
                let reshaped = block.append_operation(stable_hlo::reshape(
                    activation,
                    &[physical_query_sequence, heads],
                    location,
                )?)?;
                activation = reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref();
            }
            let residual_sources = (0..output_types[1].rank()).map(|axis| (input_values[0], axis)).collect::<Vec<_>>();
            results.push(lower_restore_dynamic_dimensions(
                activation,
                &output_types[1],
                residual_sources.as_slice(),
                block,
                context,
                location,
            )?);
        }
        return Ok(results);
    }
}

/// Lowers one [`DotProductAttentionBackwardOperation`] through its canonical typed decomposition or the cuDNN fused
/// backward ABI.
///
/// Portable and ineligible automatic configurations call the registered private decomposition. Eligible fused
/// configurations reorder the canonical input boundary into cuDNN's `(Q, K, V, residual, dO[, bias], O[, query
/// lengths, key/value lengths])` ABI, emit `__cudnn$fmha[ScaleBias]Softmax[Dropout]Backward`, and restore each
/// cotangent's logical layout, data type, and dynamic dimensions. Forced fused mode reports a precise eligibility
/// failure instead of falling back.
///
/// This function does not reproduce the analytical backward. It owns only the fused adapter's physical padding,
/// operand ordering, backend configuration, and result refinement; the portable mathematics remain exclusively in
/// the core backward decomposition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_dot_product_attention_backward_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &DotProductAttentionBackwardOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let signature = operation.signature();
    if input_values.len() != 6 + signature.count() || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: 6, actual: input_values.len() }.into());
    }
    let configuration = operation.configuration();
    let indices = attention_input_indices(signature);
    let uses_sequence_lengths = signature.has_query_sequence_lengths() || signature.has_key_value_sequence_lengths();
    if output_types.len() < 3 || output_types.len() > 3 + usize::from(signature.has_bias()) {
        return Err(ProgramError::InvalidOutputCount {
            expected: 3 + usize::from(signature.has_bias()),
            actual: output_types.len(),
        }
        .into());
    }
    let eligibility = fmha_eligibility(configuration, signature, input_types, collective_state)?;
    if configuration.implementation() == AttentionImplementation::Portable
        || matches!(eligibility, FmhaEligibility::Ineligible(_))
    {
        if configuration.implementation() == AttentionImplementation::Fused {
            if configuration.dropout().is_some() {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "`{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}` dropout is only supported by the fused CUDA \
                         lowering",
                    ),
                });
            }
            let FmhaEligibility::Ineligible(reason) = eligibility else { unreachable!() };
            return Err(LoweringError::UnsupportedOp {
                op: format!("the fused attention backward implementation is unavailable because {reason}"),
            });
        }
        let decomposition = collective_state
            .named_compositions
            .as_ref()
            .and_then(|functions| functions.get_attention_backward(operation, input_types, output_types))
            .ok_or_else(|| LoweringError::UnsupportedOp {
                op: format!("missing typed decomposition for `{}`", operation.name()),
            })?;
        return lower_decomposition_call(decomposition, input_values, output_types, block, context, location);
    }
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_dimensions(&input_types[0])?;
    let [_, key_value_sequence, key_value_heads, _] = attention_dimensions(&input_types[1])?;
    let heads = heads.value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME} heads dimension must be static"),
    })?;
    let key_value_heads = key_value_heads.value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME} key/value heads dimension must be static"),
    })?;
    let head_dimension = head_dimension.value().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME} head dimension must be static"),
    })?;
    let physical_batch = attention_physical_extent(&batch)?;
    let physical_query_sequence = attention_physical_extent(&query_sequence)?;
    let physical_key_value_sequence = attention_physical_extent(&key_value_sequence)?;
    let offset = 3 + signature.count();
    {
        let mut physical_inputs = input_values
            .iter()
            .zip(input_types)
            .map(|(value, r#type)| lower_physical_bound_value(*value, r#type, 0.0, block, context, location))
            .collect::<Result<Vec<_>, _>>()?;
        for index in [0, 1, 2, offset, offset + 2] {
            let physical_type = physical_bound_type(&input_types[index])?;
            physical_inputs[index] = normalize_fmha_operand(physical_inputs[index], &physical_type, block, location)?;
        }
        let score_type = attention_array_type(
            data_type,
            &[physical_batch, heads, physical_query_sequence, physical_key_value_sequence],
        );
        let bias = prepare_fmha_bias(
            &indices,
            physical_inputs.as_slice(),
            input_types,
            &score_type,
            block,
            context,
            location,
        )?;
        let sequence_lengths = if uses_sequence_lengths {
            let query_lengths = match indices.query_sequence_lengths {
                Some(index) => physical_inputs[index],
                None => synthesize_fmha_sequence_lengths(
                    input_values[0],
                    input_types[0].rank() - 3,
                    &batch,
                    physical_batch,
                    block,
                    context,
                    location,
                )?,
            };
            let key_value_lengths = match indices.key_value_sequence_lengths {
                Some(index) => physical_inputs[index],
                None => synthesize_fmha_sequence_lengths(
                    input_values[1],
                    input_types[1].rank() - 3,
                    &batch,
                    physical_batch,
                    block,
                    context,
                    location,
                )?,
            };
            Some((query_lengths, key_value_lengths))
        } else {
            None
        };
        let element_type = if data_type == DataType::BF16 { "BF16" } else { "F16" };
        let backend_config = fmha_backend_config(
            element_type,
            [physical_batch, heads, physical_query_sequence, physical_key_value_sequence],
            configuration.scale().unwrap_or(1.0 / (head_dimension as f64).sqrt()),
            fmha_mask_type(configuration.causal(), sequence_lengths.is_some()),
            FMHA_BACKWARD_DOT_DIMENSION_NUMBERS,
            configuration.dropout(),
            configuration.local_window().map_or(0, |(left, _)| left + 1),
        );
        let activation_input = if input_types[0].rank() == 3 {
            let reshaped = block.append_operation(stable_hlo::reshape(
                physical_inputs[offset + 1],
                &[1, physical_query_sequence, heads],
                location,
            )?)?;
            reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref()
        } else {
            physical_inputs[offset + 1]
        };
        let activation_type = attention_array_type(DataType::F32, &[physical_batch, physical_query_sequence, heads]);
        let activation = block.append_operation(stable_hlo::convert(
            activation_input,
            lower_tensor_type(&activation_type, context, location)?.as_ref(),
            location,
        )?)?;
        let activation = block.append_operation(stable_hlo::transpose(
            activation.result(0).expect("stablehlo.convert should return one result").as_ref(),
            &[0, 2, 1],
            location,
        )?)?;
        let activation = activation.result(0).expect("stablehlo.transpose should return one result").as_ref();
        // The kernel call order differs from the traced operand order: `(Q, K, V, activation, dO[, bias], O
        // [, q_seqlen, kv_seqlen])`, with the bias between the output cotangent and the forward output.
        let mut operands =
            vec![physical_inputs[0], physical_inputs[1], physical_inputs[2], activation, physical_inputs[offset + 2]];
        let mut operand_layouts =
            vec![vec![3, 2, 1, 0], vec![3, 2, 1, 0], vec![3, 2, 1, 0], vec![2, 1, 0], vec![3, 2, 1, 0]];
        if let Some(bias) = bias {
            operands.push(bias);
            operand_layouts.push(vec![3, 2, 1, 0]);
        }
        operands.push(physical_inputs[offset]);
        operand_layouts.push(vec![3, 2, 1, 0]);
        if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
            operands.extend([query_lengths, key_value_lengths]);
            operand_layouts.extend([vec![0], vec![0]]);
        }
        let query_gradient_type =
            attention_array_type(data_type, &[physical_batch, heads, physical_query_sequence, head_dimension]);
        let key_value_gradient_type = attention_array_type(
            data_type,
            &[physical_batch, key_value_heads, physical_key_value_sequence, head_dimension],
        );
        let mut custom_call_output_types = vec![
            lower_tensor_type(&query_gradient_type, context, location)?,
            lower_tensor_type(&key_value_gradient_type, context, location)?,
            lower_tensor_type(&key_value_gradient_type, context, location)?,
        ];
        let mut result_layouts = vec![vec![3, 1, 2, 0], vec![3, 1, 2, 0], vec![3, 1, 2, 0]];
        if bias.is_some() {
            custom_call_output_types.push(lower_tensor_type(&score_type, context, location)?);
            result_layouts.push(vec![3, 2, 1, 0]);
        }
        custom_call_output_types.push(lower_tensor_type(&attention_array_type(DataType::U8, &[0]), context, location)?);
        result_layouts.push(vec![0]);
        let custom_call = block.append_operation(stable_hlo::custom_call(
            operands.as_slice(),
            fmha_target_name(bias.is_some(), configuration.dropout().is_some(), true).as_str(),
            false,
            Some(context.string_attribute(backend_config.as_str()).as_ref()),
            CustomCallApiVersion::StatusReturning,
            &[],
            Some(CustomCallMemoryLayouts { operands: operand_layouts, results: result_layouts }),
            &[],
            None,
            &custom_call_output_types,
            location,
        )?)?;
        // Each gradient comes back in the physical `[b, n, seq, h]` layout and transposes to the logical
        // `BTNH`/`BSNH` layout (a pure bitcast given the declared result layouts); the bias cotangent keeps its
        // own shape.
        let mut results = Vec::with_capacity(output_types.len());
        for index in 0..3 {
            let gradient = custom_call.result(index).expect("stablehlo.custom_call should return the gradient");
            let transposed =
                block.append_operation(stable_hlo::transpose(gradient.as_ref(), &[0, 2, 1, 3], location)?)?;
            let transposed = transposed.result(0).expect("stablehlo.transpose should return one result").as_ref();
            let mut transposed = transposed;
            if input_types[index].rank() == 3 {
                let dimensions = static_dimensions(&physical_bound_type(&output_types[index])?)?;
                let reshaped =
                    block.append_operation(stable_hlo::reshape(transposed, dimensions.as_slice(), location)?)?;
                transposed = reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref();
            }
            let sources = (0..input_types[index].rank()).map(|axis| (input_values[index], axis)).collect::<Vec<_>>();
            results.push(lower_restore_dynamic_dimensions(
                transposed,
                &output_types[index],
                sources.as_slice(),
                block,
                context,
                location,
            )?);
        }
        if let Some(bias_index) = indices.bias
            && output_types.len() == 4
        {
            let bias_type = &input_types[bias_index];
            let physical_bias_type = physical_bound_type(bias_type)?;
            let mut normalized_bias_dimensions = static_dimensions(&physical_bias_type)?;
            normalized_bias_dimensions.splice(0..0, std::iter::repeat_n(1, 4 - bias_type.rank()));
            let score_dimensions = static_dimensions(&score_type)?;
            let reduced_axes = normalized_bias_dimensions
                .iter()
                .zip(&score_dimensions)
                .enumerate()
                .filter_map(|(axis, (bias, score))| (*bias == 1 && score != bias).then_some(axis))
                .collect::<Vec<_>>();
            let mut bias_cotangent =
                custom_call.result(3).expect("stablehlo.custom_call should return the bias cotangent").as_ref();
            if !reduced_axes.is_empty() {
                let reduced_dimensions = score_dimensions
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, dimension)| (!reduced_axes.contains(&axis)).then_some(*dimension))
                    .collect::<Vec<_>>();
                let reduced_type = attention_array_type(data_type, reduced_dimensions.as_slice());
                bias_cotangent = lower_reduce_to_mlir(
                    ReductionKind::Sum,
                    reduced_axes.as_slice(),
                    bias_cotangent,
                    &reduced_type,
                    block,
                    context,
                    location,
                )?;
            }
            if bias_type.data_type() != data_type {
                let converted = block.append_operation(stable_hlo::convert(
                    bias_cotangent,
                    lower_tensor_type(&physical_bias_type, context, location)?,
                    location,
                )?)?;
                bias_cotangent = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            }
            bias_cotangent = block
                .append_operation(stable_hlo::reshape(
                    bias_cotangent,
                    static_dimensions(&physical_bias_type)?.as_slice(),
                    location,
                )?)?
                .result(0)
                .expect("stablehlo.reshape should return one result")
                .as_ref();
            let sources = (0..bias_type.rank()).map(|axis| (input_values[bias_index], axis)).collect::<Vec<_>>();
            results.push(lower_restore_dynamic_dimensions(
                bias_cotangent,
                output_types.last().unwrap(),
                sources.as_slice(),
                block,
                context,
                location,
            )?);
        }
        return Ok(results);
    }
}

impl LowerableOperation for DotProductAttentionOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let collective_state = lowerer.collective_state.clone();
        lower_dot_product_attention_to_mlir(
            self,
            &collective_state,
            input_values,
            input_types,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl LowerableOperation for DotProductAttentionBackwardOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let collective_state = lowerer.collective_state.clone();
        lower_dot_product_attention_backward_to_mlir(
            self,
            &collective_state,
            input_values,
            input_types,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::operations::attention::AttentionInputSignature;
    use ryft_core::{AtomId, DimensionBounds, DimensionVariable, Placeholder, Typed};

    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::lowering::tests::{XlaProgramBuilder, unproject_plain_program};
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgram};
    use crate::to_mlir_module_for_program;

    use super::*;

    fn attention_operation(configuration: AttentionConfiguration) -> DotProductAttentionOperation {
        DotProductAttentionOperation::new(configuration, AttentionInputSignature::default())
    }

    fn attention_backward_operation(configuration: AttentionConfiguration) -> DotProductAttentionBackwardOperation {
        DotProductAttentionBackwardOperation::new(configuration, AttentionInputSignature::default())
    }

    /// Builds the attention program shared by the platform-gated lowering fixtures below.
    fn dot_product_attention_fixture_program(
        data_type: DataType,
        head_dimension: usize,
        causal: bool,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let operand_type = attention_array_type(data_type, &[1, 4, 2, head_dimension]);
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
        ];
        let output = builder
            .add_instruction(
                attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(causal)),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![operand_type.clone(), operand_type.clone(), operand_type.clone()];
        (unproject_plain_program(program), input_types, vec![operand_type])
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_on_cuda() {
        // Qualifying causal BF16 attention uses the cuDNN fMHA ABI and preserves the causal mask.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::BF16, 8, true);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("CAUSAL"));
        assert!(module.contains("BF16"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_without_mask_on_cuda() {
        // The unmasked F16 form uses the same fMHA target with NO_MASK and the F16 element contract.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F16, 8, false);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("NO_MASK"));
        assert!(module.contains("F16"));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_without_target_platform() {
        // Without target information, the entry function calls the one private typed decomposition containing the
        // score dot, causal mask, stable softmax, value dot, and final layout restoration.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F32, 8, true);
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("stablehlo.dot_general").count(), 2);
        assert!(module.contains("stablehlo.compare LE"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dot_product_attention_composition_for_f32_on_cuda() {
        // `f32` operands do not qualify for the cuDNN flash-attention call, so even a CUDA target lowers the
        // portable composition.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F32, 8, true);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmhaSoftmax"));
        assert!(module.contains("stablehlo.dot_general"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dot_product_attention_composition_for_unaligned_head_dim_on_cuda() {
        // A head dimension that is not a multiple of 8 fails cuDNN's compile-time gate, so a CUDA target with `bf16`
        // operands still lowers the portable composition (which upcasts its softmax to `f32`).
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::BF16, 4, true);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmhaSoftmax"));
        assert!(module.contains("stablehlo.dot_general"));
        assert!(module.contains("stablehlo.convert"));
    }

    /// Returns the inferred output [`ArrayType`]s of the provided staged output atoms.
    fn builder_output_types(builder: &XlaProgramBuilder, outputs: &[AtomId]) -> Vec<ArrayType> {
        outputs.iter().map(|output| builder.atoms()[output.index()].r#type().into_owned()).collect()
    }

    /// Builds an extended dot-product attention fixture program: `query [2, 4, query_heads, 8]` over
    /// `key`/`value [2, 4, key_value_heads, 8]` at the provided data type, with an optional broadcast-batch bias
    /// `[1, query_heads, 4, 4]` and an optional trailing `i32[2]` sequence-length pair.
    fn dot_product_attention_extended_fixture_program(
        data_type: DataType,
        query_heads: usize,
        key_value_heads: usize,
        bias: bool,
        sequence_lengths: bool,
        operation: DotProductAttentionOperation,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let query_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(query_heads),
                Dimension::Static(8),
            ]),
        );
        let key_value_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(key_value_heads),
                Dimension::Static(8),
            ]),
        );
        let mut builder = XlaProgramBuilder::new();
        let mut input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type];
        if bias {
            input_types.push(ArrayType::new(
                data_type,
                Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Static(query_heads),
                    Dimension::Static(4),
                    Dimension::Static(4),
                ]),
            ));
        }
        if sequence_lengths {
            let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
            input_types.push(lengths_type.clone());
            input_types.push(lengths_type);
        }
        let operation = DotProductAttentionOperation::new(
            operation.configuration(),
            AttentionInputSignature::new(bias, false, sequence_lengths, sequence_lengths),
        );
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (unproject_plain_program(program), input_types, output_types)
    }

    /// Builds the matching backward fixture program for [`dot_product_attention_extended_fixture_program`]'s
    /// canonical operand order: `(query, key, value[, bias][, query lengths, key/value lengths], output, residual,
    /// output cotangent)`.
    fn dot_product_attention_backward_fixture_program(
        data_type: DataType,
        query_heads: usize,
        key_value_heads: usize,
        bias: bool,
        sequence_lengths: bool,
        operation: DotProductAttentionBackwardOperation,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let query_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(query_heads),
                Dimension::Static(8),
            ]),
        );
        let key_value_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(key_value_heads),
                Dimension::Static(8),
            ]),
        );
        let activation_type = ArrayType::new(
            data_type,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(query_heads)]),
        );
        let mut builder = XlaProgramBuilder::new();
        let mut input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type];
        if bias {
            input_types.push(ArrayType::new(
                data_type,
                Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Static(query_heads),
                    Dimension::Static(4),
                    Dimension::Static(4),
                ]),
            ));
        }
        if sequence_lengths {
            let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
            input_types.push(lengths_type.clone());
            input_types.push(lengths_type);
        }
        input_types.extend([query_type.clone(), activation_type, query_type]);
        let operation = DotProductAttentionBackwardOperation::new(
            operation.configuration(),
            AttentionInputSignature::new(bias, false, sequence_lengths, sequence_lengths),
        );
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (unproject_plain_program(program), input_types, output_types)
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_scale_bias_softmax_on_cuda() {
        // A broadcastable user bias is expanded to the physical score shape before entering the ScaleBias ABI.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.broadcast_in_dim %arg3"));
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax"));
        assert!(module.contains("tensor<2x2x4x4xbf16>"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_training_forward_with_padding_on_cuda() {
        // Training attention carries both length vectors and converts the f32 kernel statistic to the public residual.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            true,
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_residual(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("PADDING_CAUSAL"));
        assert!(module.contains("tensor<2x2x4xf32>"));
        assert!(module.contains("stablehlo.convert"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_dropout_with_sliding_window_on_cuda() {
        // Dropout selects its target and records the deterministic seed and exclusive left-window length.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            false,
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_local_window((1, 0))
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmaxDropout"));
        assert!(module.contains("dropout_rate"));
        assert!(module.contains("123"));
        assert!(module.contains("sliding_window_length"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_backward_with_bias_on_cuda() {
        // The fused backward restores operand gradients and sums the score-shaped gradient to the user bias shape.
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            attention_backward_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward"));
        assert!(module.matches("stablehlo.transpose").count() >= 3);
        assert!(module.contains("stablehlo.reduce"));
        assert!(module.contains("tensor<1x2x4x4xbf16>"));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_with_extensions() {
        // GQA expansion, broadcast bias, causal/local masking, and residual production all live in one private typed
        // decomposition rather than in the backend adapter.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_local_window((1, 0))
                    .with_residual(true),
            ),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention").count(), 1);
        assert!(module.contains("stablehlo.broadcast_in_dim"));
        assert!(module.contains("stablehlo.reshape"));
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.log"));
        assert!(module.contains("-> (tensor<2x4x4x8xf32>, tensor<2x4x4xf32>)"));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_backward_composition() {
        // The private typed backward decomposition recomputes normalized probabilities and emits the query, grouped
        // key/value, and broadcast-bias cotangent reductions.
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            attention_backward_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention_backward").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention_backward").count(), 1);
        assert!(module.matches("stablehlo.dot_general").count() >= 4);
        assert!(module.contains("stablehlo.exponential"));
        assert!(module.contains("stablehlo.reduce"));
        assert!(module.contains("tensor<1x4x4x4xf32>"));
        assert!(!module.contains("__cudnn$fmha"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_dot_product_attention_dropout_off_the_fast_path() {
        // No composition implements dropout, so a dropout-carrying operation that misses the fused fast-path gate
        // (here: `f32` operands on CUDA, and any operands without a target platform) reports an explicit error
        // instead of silently computing dropout-free attention, for both the forward and the backward operations.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F32,
            2,
            2,
            false,
            false,
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        for platform in [Some("cuda"), None] {
            let result =
                lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, platform);
            assert!(matches!(
                result,
                Err(error) if error.to_string().contains(
                    "`dot_product_attention` dropout is only supported by the fused CUDA lowering",
                ),
            ));
        }
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            2,
            2,
            false,
            false,
            attention_backward_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        let result =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"));
        assert!(matches!(
            result,
            Err(error) if error.to_string().contains(
                "`dot_product_attention_backward` dropout is only supported by the fused CUDA lowering",
            ),
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_padded_fmha_with_a_right_local_window() {
        // The pinned cuDNN ABI supports a nonzero right radius only for the unpadded causal mask kind. Forced fused
        // selection therefore reports this semantic limitation instead of silently dropping the right radius.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            true,
            attention_operation(
                AttentionConfiguration::new()
                    .with_causal(true)
                    .with_local_window((2, 1))
                    .with_implementation(AttentionImplementation::Fused),
            ),
        );

        let result =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"));

        assert!(matches!(
            result,
            Err(error) if error.to_string().contains(
                "a nonzero right local-window radius requires causal masking without sequence-length padding",
            ),
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dynamically_shaped_attention_compositions() {
        // Portable attention obtains every logical extent from the mixed array IR, so bounded-dynamic programs do
        // not need to expose physical padding as sequence-length operands and support the same GQA and window
        // configurations as static programs. Explicit lengths remain an independent semantic mask when provided.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4)).unwrap());
        let operand_type = |heads: usize| {
            ArrayType::new(
                DataType::BF16,
                Shape::new(vec![
                    Dimension::Dynamic(batch.clone()),
                    Dimension::Static(4),
                    Dimension::Static(heads),
                    Dimension::Static(8),
                ]),
            )
        };
        let activation_type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(4), Dimension::Static(2)]),
        );
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(batch.clone())]));
        let build_forward = |key_value_heads: usize,
                             sequence_lengths: bool,
                             operation: DotProductAttentionOperation| {
            let mut input_types = vec![operand_type(2), operand_type(key_value_heads), operand_type(key_value_heads)];
            if sequence_lengths {
                input_types.push(lengths_type.clone());
                input_types.push(lengths_type.clone());
            }
            let operation = DotProductAttentionOperation::new(
                operation.configuration(),
                AttentionInputSignature::new(false, false, sequence_lengths, sequence_lengths),
            );
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
            let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
            let output_count = outputs.len();
            let output_types = builder_output_types(&builder, outputs.as_slice());
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    outputs,
                    vec![Placeholder; input_types.len()],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            (unproject_plain_program(program), input_types, output_types)
        };
        let build_backward = |key_value_heads: usize,
                              sequence_lengths: bool,
                              operation: DotProductAttentionBackwardOperation| {
            let mut input_types = vec![operand_type(2), operand_type(key_value_heads), operand_type(key_value_heads)];
            if sequence_lengths {
                input_types.push(lengths_type.clone());
                input_types.push(lengths_type.clone());
            }
            input_types.extend([operand_type(2), activation_type.clone(), operand_type(2)]);
            let operation = DotProductAttentionBackwardOperation::new(
                operation.configuration(),
                AttentionInputSignature::new(false, false, sequence_lengths, sequence_lengths),
            );
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
            let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
            let output_count = outputs.len();
            let output_types = builder_output_types(&builder, outputs.as_slice());
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    outputs,
                    vec![Placeholder; input_types.len()],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            (unproject_plain_program(program), input_types, output_types)
        };

        // Dynamic forward attention without explicit sequence lengths uses the tensor's logical dimensions.
        let (program, input_types, output_types) =
            build_forward(2, false, attention_operation(AttentionConfiguration::new().with_scale(0.125)));
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();

        // Sliding-window masking and grouped-query head expansion are ordinary semantic operations.
        for (key_value_heads, operation) in [
            (
                2,
                attention_operation(
                    AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_local_window((1, 0)),
                ),
            ),
            (1, attention_operation(AttentionConfiguration::new().with_scale(0.125))),
        ] {
            let (program, input_types, output_types) = build_forward(key_value_heads, true, operation);
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None)
                .unwrap();
        }

        // The analytical backward uses the same first-class dimensions and mask construction.
        let (program, input_types, output_types) =
            build_backward(2, false, attention_backward_operation(AttentionConfiguration::new().with_scale(0.125)));
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();

        let (program, input_types, output_types) = build_backward(
            2,
            true,
            attention_backward_operation(
                AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_local_window((1, 0)),
            ),
        );
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_jax_attention_surface() {
        use ryft_core::operations::attention::AttentionConfiguration;

        // Rank-three attention, an arbitrary Boolean mask, one independently present sequence-length operand, an
        // asymmetric local window, default scaling, and a residual all lower through the typed portable composition.
        let query_type = attention_array_type(DataType::F32, &[2, 1, 2]);
        let key_value_type = attention_array_type(DataType::F32, &[3, 1, 2]);
        let mask_type = attention_array_type(DataType::Boolean, &[1, 2, 3]);
        let lengths_type = attention_array_type(DataType::I32, &[1]);
        let signature = AttentionInputSignature::new(false, true, true, false);
        let operation = DotProductAttentionOperation::new(
            AttentionConfiguration::new()
                .with_local_window(Some((2, 1)))
                .with_implementation(AttentionImplementation::Portable)
                .with_residual(true),
            signature,
        );
        let input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type, mask_type, lengths_type];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &input_types,
            &output_types,
            "main",
            None,
            None,
            None,
        )
        .unwrap()
        .stable_hlo;

        assert!(module.contains("func.func private @ryft.dot_product_attention"));
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));
        assert_eq!(output_types[0], query_type);
        assert_eq!(output_types[1].shape(), &Shape::new(vec![2.into(), 1.into()]));
    }

    #[test]
    fn test_lower_mlir_module_for_program_fuses_jax_attention_surface_on_cuda() {
        // The cuDNN adapter normalizes TNH inputs, turns an arbitrary Boolean mask into additive bias, and synthesizes
        // the absent key/value length vector while preserving a supported causal left window.
        let query_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let key_value_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let mask_type = attention_array_type(DataType::Boolean, &[1, 4, 4]);
        let lengths_type = attention_array_type(DataType::I32, &[1]);
        let residual_type = attention_array_type(DataType::F16, &[4, 2]);
        let signature = AttentionInputSignature::new(false, true, true, false);
        let configuration = AttentionConfiguration::new()
            .with_causal(true)
            .with_local_window((2, 0))
            .with_implementation(AttentionImplementation::Fused)
            .with_residual(true);
        let forward_input_types = vec![
            query_type.clone(),
            key_value_type.clone(),
            key_value_type.clone(),
            mask_type.clone(),
            lengths_type.clone(),
        ];
        let mut builder = XlaProgramBuilder::new();
        let inputs = forward_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(DotProductAttentionOperation::new(configuration, signature), Vec::new(), inputs, None)
            .unwrap()
            .to_vec();
        let forward_output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; forward_input_types.len()],
                vec![Placeholder; 2],
            )
            .unwrap();
        let forward = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &forward_input_types,
            &forward_output_types,
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert!(forward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax"));
        assert!(forward.contains("PADDING_CAUSAL"));
        assert!(forward.contains("stablehlo.select"));
        assert!(forward.contains("stablehlo.get_dimension_size %arg1, dim = 0"));
        assert!(forward.contains("tensor<4x2x8xf16>"));
        assert_eq!(forward_output_types, vec![query_type.clone(), residual_type.clone()]);

        // The matching backward consumes the same canonical optional prefix and discards the synthetic mask-bias
        // cotangent while preserving the three differentiable operand cotangents.
        let backward_input_types = forward_input_types
            .iter()
            .cloned()
            .chain([query_type.clone(), residual_type, query_type.clone()])
            .collect::<Vec<_>>();
        let mut builder = XlaProgramBuilder::new();
        let inputs = backward_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(
                DotProductAttentionBackwardOperation::new(configuration, signature),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let backward_output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; backward_input_types.len()],
                vec![Placeholder; 3],
            )
            .unwrap();
        let backward = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &backward_input_types,
            &backward_output_types,
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert!(backward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward"));
        assert_eq!(backward_output_types, vec![query_type.clone(), key_value_type.clone(), key_value_type]);
    }

    #[test]
    fn test_lower_mlir_module_for_program_never_emits_fmha_for_f8_attention() {
        // F8 attention is a validated NO-GO on the pinned cuDNN (the fused kernels reject F8 I/O at compile time),
        // so `f8` operands never qualify for the fMHA fast path and take the ordinary composition route even on
        // CUDA targets.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F8E4M3FN,
            2,
            2,
            false,
            false,
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmha"));
        assert!(module.contains("stablehlo.dot_general"));
        assert!(module.contains("stablehlo.convert"));
    }
}
