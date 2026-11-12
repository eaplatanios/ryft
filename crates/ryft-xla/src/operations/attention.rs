//! cuDNN fused-attention implementations of the named attention definitions.
//!
//! The named attention definitions (refer to [`AttentionKey`]) carry their portable bodies, which every backend can
//! lower. On CUDA targets, this module replaces eligible calls with cuDNN's fused multi-head attention custom calls,
//! selected by [`select_attention_implementation`] from the decoded key and the boundary of each call. This module
//! owns only the fused kernels' eligibility, physical-bound materialization, ABI configuration, and output refinement.
//! The attention mathematics (e.g., grouped-head expansion, masking, and softmax) remain exclusively in the core
//! bodies.

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, AttentionConfiguration, AttentionImplementation, AttentionInputs, AttentionKey,
    CompositeKey, DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DataType,
    Dimension as CoreDimension, Parameterized, Placeholder, ProgramError, ReductionKind, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type,
    Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_tensor_type,
    physical_bound_type, static_dimensions,
};
use crate::operations::composites::NamedCompositeImplementation;
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{Error, LoweringError};

/// Seed of the dropout masks of the fused attention kernels. Calls with dropout but without a dropout state have an
/// implementation-defined mask (refer to [`AttentionInputs::dropout_state`]), and this fixed seed makes the forward and
/// backward kernels of one boundary apply the same mask.
const FMHA_DROPOUT_SEED: u64 = 42;

/// Returns the implementation that replaces the calls of the named attention definition `key` with the provided
/// input and output types when lowering for `target_platform`, or [`None`] when the calls lower their portable body.
///
/// Keys that do not name an attention definition, malformed attention keys, and boundaries that are not array-typed
/// are declined. [`AttentionImplementation::Portable`] always lowers the body, and so does a call with a dropout state,
/// because the fused kernels cannot reproduce the mask of a caller-provided random state.
/// [`AttentionImplementation::Automatic`] selects the fused kernels when [`fmha_eligibility`] accepts the boundary and
/// the body otherwise. [`AttentionImplementation::Fused`] selects the fused kernels when they are eligible and an
/// implementation that reports why they are not otherwise, so that it never silently changes the requested numerics.
pub(crate) fn select_attention_implementation(
    key: &CompositeKey,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    target_platform: Option<&str>,
) -> Option<NamedCompositeImplementation> {
    let key = AttentionKey::from_composite_key(key).ok()??;
    let configuration = key.configuration();
    let input_structure = key.input_structure();
    let implementation = configuration.implementation();
    if implementation == AttentionImplementation::Portable {
        return None;
    }
    let array_types = |types: &[ArrayIrType]| {
        types.iter().map(|r#type| <&ArrayType>::try_from(r#type).ok().cloned()).collect::<Option<Vec<_>>>()
    };
    let input_types = array_types(input_types)?;
    array_types(output_types)?;
    let name = match key {
        AttentionKey::Forward { .. } => DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
        AttentionKey::Backward { .. } => DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
    };
    let unavailable = |reason: String| match implementation {
        AttentionImplementation::Fused => {
            Some(NamedCompositeImplementation::FusedAttentionUnavailable { name, reason })
        }
        AttentionImplementation::Automatic | AttentionImplementation::Portable => None,
    };
    if input_structure.dropout_state.is_some() {
        return unavailable(
            "the fused kernels cannot reproduce the dropout mask of a dropout state; omit the dropout state to use \
             the fused kernels' own mask"
                .to_string(),
        );
    }
    let expected_input_count =
        input_structure.parameter_count() + 3 * usize::from(matches!(key, AttentionKey::Backward { .. }));
    if input_types.len() != expected_input_count {
        return None;
    }
    match fmha_eligibility(configuration, input_structure, input_types.as_slice(), target_platform) {
        Ok(FmhaEligibility::Eligible) => Some(match key {
            AttentionKey::Forward { return_residual, .. } => {
                NamedCompositeImplementation::Attention { configuration, input_structure, return_residual }
            }
            AttentionKey::Backward { .. } => {
                NamedCompositeImplementation::AttentionBackward { configuration, input_structure }
            }
        }),
        Ok(FmhaEligibility::Ineligible(reason)) => unavailable(reason.to_string()),
        Err(error) => unavailable(error.to_string()),
    }
}

/// Canonical `[batch, sequence, heads, head_dimension]` dimensions of one rank-three or rank-four attention type.
fn attention_dimensions(input_type: &ArrayType) -> Result<[CoreDimension; 4], Error> {
    match input_type.shape().dimensions() {
        [sequence, heads, head_dimension] => {
            Ok([CoreDimension::Static(1), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        [batch, sequence, heads, head_dimension] => {
            Ok([batch.clone(), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        _ => Err(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME.to_string(),
            reason: format!(
                "{} operand must have rank 3 or 4 but got rank {}",
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                input_type.rank(),
            ),
        })),
    }
}

/// Returns the physical maximum extent of one attention dimension.
fn attention_physical_extent(dimension: &CoreDimension) -> Result<usize, Error> {
    match dimension {
        CoreDimension::Static(extent) => Ok(*extent),
        CoreDimension::Dynamic(variable) => {
            variable.bounds().upper().and_then(|upper| upper.checked_sub(1)).ok_or_else(|| {
                Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME.to_string(),
                    reason: format!(
                        "{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME} dimension {variable} needs a finite positive physical \
                         bound",
                    ),
                })
            })
        }
    }
}

/// Returns an [`ArrayType`] with the provided data type and dimensions used by the attention lowerings.
pub(super) fn attention_array_type<D: Clone + Into<CoreDimension>>(data_type: DataType, dimensions: &[D]) -> ArrayType {
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
/// backward path, where the `P` descriptor and residual strides derive from it), the mask kind, the
/// forward or backward dot-dimension-number block, the dropout rate and the fixed seed [`FMHA_DROPOUT_SEED`] (the
/// rate is the hardware-validated default `0.0` when dropout is off), and the sliding-window length.
fn fmha_backend_config(
    element_type: &str,
    intermediate_dimensions: [usize; 4],
    scale: f64,
    mask_type: &str,
    dot_dimension_numbers: &str,
    dropout: Option<f64>,
    sliding_window_length: usize,
) -> String {
    let [batch, heads, query_sequence, key_value_sequence] = intermediate_dimensions;
    // The `{:?}` rate formatting keeps the validated `0.0` spelling for the off state while rendering set rates
    // with their shortest round-trip form.
    let dropout_rate = dropout.unwrap_or(0.0);
    let seed = FMHA_DROPOUT_SEED;
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
    input_structure: AttentionInputs<Placeholder>,
    input_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<FmhaEligibility, Error> {
    // The right radius of a configuration with causal masking is always zero.
    if configuration.local_window().is_some_and(|(_, right)| right != 0) {
        return Ok(FmhaEligibility::Ineligible("a nonzero right local-window radius requires causal masking"));
    }
    if target_platform != Some("cuda") {
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
        && input_structure.query_sequence_lengths.is_none()
        && input_structure.key_value_sequence_lengths.is_none()
    {
        return Ok(FmhaEligibility::Ineligible("bounded-dynamic operands require query or key/value sequence lengths"));
    }
    Ok(FmhaEligibility::Eligible)
}

/// Canonical operand indices of one attention boundary.
fn attention_input_indices(input_structure: AttentionInputs<Placeholder>) -> AttentionInputs<usize> {
    AttentionInputs::from_parameters(input_structure, 0..input_structure.parameter_count())
        .expect("the generated indices match the attention input structure")
}

/// Adds the implicit batch dimension required by the cuDNN `BTNH` ABI to one physical rank-three operand.
fn normalize_fmha_operand<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
) -> Result<Option<ValueRef<'b, 'c, 't>>, Error> {
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
                return Ok::<_, Error>(bias);
            }
            let converted = block.append_operation(stable_hlo::convert(
                bias,
                lower_tensor_type(score_type, context, location)?,
                location,
            )?)?;
            Ok::<_, Error>(converted.result(0).expect("stablehlo.convert should return one result").as_ref())
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
            Ok::<_, Error>(selected.result(0).expect("stablehlo.select should return one result").as_ref())
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
    batch: &CoreDimension,
    physical_batch: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
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
    if matches!(batch, CoreDimension::Dynamic(_)) {
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

/// Lowers one call of the forward attention definition with the provided configuration, input structure, and
/// residual request to cuDNN's fused ABI, for a boundary that [`fmha_eligibility`] accepts.
///
/// An eligible `bf16`/`f16` CUDA call emits `__cudnn$fmha[ScaleBias]Softmax[Dropout]` with physical `BNTH` inputs and
/// restores the logical rank and dynamic dimensions of its outputs. Optional bias and sequence-length inputs retain the
/// canonical [`AttentionInputs`] order until the final ABI assembly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_fused_attention<'b, 'c: 'b, 't: 'c>(
    configuration: AttentionConfiguration,
    input_structure: AttentionInputs<Placeholder>,
    return_residual: bool,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let input_count = input_structure.parameter_count();
    if input_values.len() != input_count || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: input_count, actual: input_values.len() }.into());
    }
    let expected_output_count = if return_residual { 2 } else { 1 };
    check_count!("output", output_types, expected_output_count, ProgramError);
    let indices = attention_input_indices(input_structure);
    let uses_sequence_lengths =
        input_structure.query_sequence_lengths.is_some() || input_structure.key_value_sequence_lengths.is_some();
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_dimensions(&input_types[0])?;
    let [_, key_value_sequence, _, _] = attention_dimensions(&input_types[1])?;
    let heads = heads.value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME.to_string(),
            reason: "the heads dimension must be static".to_string(),
        })
    })?;
    let head_dimension = head_dimension.value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME.to_string(),
            reason: "the head dimension must be static".to_string(),
        })
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
        // it as needed), and the training form inserts the `f32[b, n, t]` log-sum-exp residual before it.
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
        if return_residual {
            let residual_type = attention_array_type(DataType::F32, &[physical_batch, heads, physical_query_sequence]);
            custom_call_output_types.push(lower_tensor_type(&residual_type, context, location)?);
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
        if return_residual {
            let residual = custom_call
                .result(1)
                .expect("stablehlo.custom_call should return the log-sum-exp residual")
                .as_ref();
            let residual = block.append_operation(stable_hlo::transpose(residual, &[0, 2, 1], location)?)?;
            let mut residual = residual.result(0).expect("stablehlo.transpose should return one result").as_ref();
            let physical_residual_type = if input_types[0].rank() == 3 {
                attention_array_type(data_type, &[physical_batch, physical_query_sequence, heads])
            } else {
                physical_bound_type(&output_types[1])?
            };
            let converted = block.append_operation(stable_hlo::convert(
                residual,
                lower_tensor_type(&physical_residual_type, context, location)?,
                location,
            )?)?;
            residual = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            if input_types[0].rank() == 3 {
                let reshaped = block.append_operation(stable_hlo::reshape(
                    residual,
                    &[physical_query_sequence, heads],
                    location,
                )?)?;
                residual = reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref();
            }
            let residual_sources = (0..output_types[1].rank()).map(|axis| (input_values[0], axis)).collect::<Vec<_>>();
            results.push(lower_restore_dynamic_dimensions(
                residual,
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

/// Lowers one call of the backward attention definition with the provided configuration and input structure
/// to cuDNN's fused backward ABI, for a boundary that [`fmha_eligibility`] accepts.
///
/// The canonical input boundary is reordered into cuDNN's `(Q, K, V, residual, dO[, bias], O[, query lengths,
/// key/value lengths])` ABI, the call emits `__cudnn$fmha[ScaleBias]Softmax[Dropout]Backward`, and each cotangent's
/// logical layout, data type, and dynamic dimensions are restored. This function does not reproduce the analytical
/// backward pass. It owns only the fused kernel's physical padding, input ordering, backend configuration, and output
/// refinement, while the portable mathematics remain exclusively in the core backward body.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_fused_attention_backward<'b, 'c: 'b, 't: 'c>(
    configuration: AttentionConfiguration,
    input_structure: AttentionInputs<Placeholder>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let input_count = input_structure.parameter_count() + 3;
    if input_values.len() != input_count || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: input_count, actual: input_values.len() }.into());
    }
    let indices = attention_input_indices(input_structure);
    let uses_sequence_lengths =
        input_structure.query_sequence_lengths.is_some() || input_structure.key_value_sequence_lengths.is_some();
    if output_types.len() < 3 || output_types.len() > 3 + usize::from(input_structure.bias.is_some()) {
        return Err(ProgramError::InvalidOutputCount {
            expected: 3 + usize::from(input_structure.bias.is_some()),
            actual: output_types.len(),
        }
        .into());
    }
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_dimensions(&input_types[0])?;
    let [_, key_value_sequence, key_value_heads, _] = attention_dimensions(&input_types[1])?;
    let heads = heads.value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME.to_string(),
            reason: "the heads dimension must be static".to_string(),
        })
    })?;
    let key_value_heads = key_value_heads.value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME.to_string(),
            reason: "the key/value heads dimension must be static".to_string(),
        })
    })?;
    let head_dimension = head_dimension.value().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME.to_string(),
            reason: "the head dimension must be static".to_string(),
        })
    })?;
    let physical_batch = attention_physical_extent(&batch)?;
    let physical_query_sequence = attention_physical_extent(&query_sequence)?;
    let physical_key_value_sequence = attention_physical_extent(&key_value_sequence)?;
    let offset = input_structure.parameter_count();
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
        let residual_input = if input_types[0].rank() == 3 {
            let reshaped = block.append_operation(stable_hlo::reshape(
                physical_inputs[offset + 1],
                &[1, physical_query_sequence, heads],
                location,
            )?)?;
            reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref()
        } else {
            physical_inputs[offset + 1]
        };
        let residual_type = attention_array_type(DataType::F32, &[physical_batch, physical_query_sequence, heads]);
        let residual = block.append_operation(stable_hlo::convert(
            residual_input,
            lower_tensor_type(&residual_type, context, location)?.as_ref(),
            location,
        )?)?;
        let residual = block.append_operation(stable_hlo::transpose(
            residual.result(0).expect("stablehlo.convert should return one result").as_ref(),
            &[0, 2, 1],
            location,
        )?)?;
        let residual = residual.result(0).expect("stablehlo.transpose should return one result").as_ref();
        // The kernel call order differs from the traced operand order: `(Q, K, V, residual, dO[, bias], O
        // [, q_seqlen, kv_seqlen])`, with the bias between the output cotangent and the forward output.
        let mut operands =
            vec![physical_inputs[0], physical_inputs[1], physical_inputs[2], residual, physical_inputs[offset + 2]];
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION, DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME, Differentiate,
        DimensionBounds, DimensionVariable, DomainTracingContext, Typed, Value as CoreValue,
        differentiable_dot_product_attention, dot_product_attention,
    };

    use crate::domains::Tracer;
    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::ops::FlatProgram;
    use crate::{Domain, to_mlir_module_for_program};

    use super::*;

    /// Staged program together with its input and output types.
    type StagedProgram = (FlatProgram, Vec<ArrayType>, Vec<ArrayType>);

    /// Returns the array types of `types`.
    fn array_types(types: &[ArrayIrType]) -> Vec<ArrayType> {
        types.iter().map(|r#type| <&ArrayType>::try_from(r#type).unwrap().clone()).collect()
    }

    /// Stages `calls` consecutive calls of one inference function with the provided configuration over inputs with
    /// the provided input structure and types, where each call attends the output of the previous one.
    fn forward_program(
        configuration: AttentionConfiguration,
        input_structure: AttentionInputs<Placeholder>,
        return_residual: bool,
        input_types: Vec<ArrayType>,
        calls: usize,
    ) -> StagedProgram {
        let attention = dot_product_attention::<Domain<'static>>(configuration);
        let (output_types, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let mut inputs = AttentionInputs::from_parameters(input_structure, inputs)?;
                let mut outputs = Vec::new();
                for _ in 0..calls {
                    outputs = if return_residual {
                        let (output, residual) = attention.call_with_residual(inputs.clone())?;
                        vec![output, residual]
                    } else {
                        vec![attention.call(inputs.clone())?]
                    };
                    inputs.query = outputs[0].clone();
                }
                Ok(outputs)
            },
            input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>(),
        )
        .unwrap();
        (program.into_flat_program(), input_types, array_types(&output_types))
    }

    /// Stages the reverse mode differentiation of one call of a training function with the provided configuration
    /// over inputs with the provided input structure and types, followed by the output cotangent. The program
    /// calls the forward definition with its residual and the backward definition, and its outputs are the attended
    /// output followed by the cotangents of the query, key, and value, and of a floating-point bias.
    fn training_program(
        configuration: AttentionConfiguration,
        input_structure: AttentionInputs<Placeholder>,
        input_types: Vec<ArrayType>,
    ) -> StagedProgram {
        let attention = differentiable_dot_product_attention::<Domain<'static>>(configuration);
        let input_types = input_types.iter().cloned().chain([input_types[0].clone()]).collect::<Vec<_>>();
        let (output_types, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| {
                let (output_cotangent, inputs) = inputs.split_last().unwrap();
                let inputs = AttentionInputs::from_parameters(input_structure, inputs.iter().cloned())?;
                let has_bias = inputs.bias.as_ref().is_some_and(|bias| {
                    <&ArrayType>::try_from(bias.r#type().as_ref()).unwrap().data_type().is_floating_point()
                });
                let (output, pullback) =
                    output_cotangent.domain().differentiate_at(inputs).vjp(|inputs| attention.call(inputs))?;
                let cotangents = pullback.apply_materialized(output_cotangent.clone())?;
                Ok([output, cotangents.query, cotangents.key, cotangents.value]
                    .into_iter()
                    .chain(cotangents.bias.filter(|_| has_bias))
                    .collect::<Vec<_>>())
            },
            input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>(),
        )
        .unwrap();
        (program.into_flat_program(), input_types, array_types(&output_types))
    }

    /// Returns the `[2, 4, heads, 8]` input types of attention over `data_type` arrays with `query_heads` query heads
    /// and `key_value_heads` key/value heads, followed by an optional broadcast-batch bias `[1, query_heads, 4, 4]`
    /// and an optional pair of `i32[2]` sequence lengths.
    fn extended_input_types(
        data_type: DataType,
        query_heads: usize,
        key_value_heads: usize,
        bias: bool,
        sequence_lengths: bool,
    ) -> (AttentionInputs<Placeholder>, Vec<ArrayType>) {
        let mut input_types = vec![
            attention_array_type(data_type, &[2, 4, query_heads, 8]),
            attention_array_type(data_type, &[2, 4, key_value_heads, 8]),
            attention_array_type(data_type, &[2, 4, key_value_heads, 8]),
        ];
        if bias {
            input_types.push(attention_array_type(data_type, &[1, query_heads, 4, 4]));
        }
        if sequence_lengths {
            input_types.extend([attention_array_type(DataType::I32, &[2]), attention_array_type(DataType::I32, &[2])]);
        }
        let input_structure = AttentionInputs {
            bias: bias.then_some(Placeholder),
            query_sequence_lengths: sequence_lengths.then_some(Placeholder),
            key_value_sequence_lengths: sequence_lengths.then_some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        (input_structure, input_types)
    }

    /// Lowers `program` for `target_platform`, returning its StableHLO module or the lowering error.
    fn lower(
        (program, input_types, output_types): &StagedProgram,
        target_platform: Option<&str>,
    ) -> Result<String, Error> {
        lower_mlir_module_for_program(program, &[], input_types, output_types, "main", None, None, target_platform)
            .map(|module| module.stable_hlo)
    }

    #[test]
    fn test_select_attention_implementation() {
        let operand_type = ArrayIrType::from(attention_array_type(DataType::BF16, &[1, 4, 2, 8]));
        let f32_operand_type = ArrayIrType::from(attention_array_type(DataType::F32, &[1, 4, 2, 8]));
        let state_type = ArrayIrType::from(attention_array_type(DataType::U64, &[2]));
        let forward_types = vec![operand_type.clone(); 3];
        let select = |implementation: AttentionImplementation,
                      configuration: AttentionConfiguration,
                      input_structure: AttentionInputs<Placeholder>,
                      input_types: &[ArrayIrType],
                      target_platform: Option<&str>| {
            let key = AttentionKey::Forward {
                configuration: configuration.with_implementation(implementation),
                input_structure,
                return_residual: false,
            };
            select_attention_implementation(
                &key.to_composite_key(),
                input_types,
                std::slice::from_ref(&operand_type),
                target_platform,
            )
        };
        let configuration = AttentionConfiguration::new().with_causal(true);
        let input_structure = AttentionInputs::new(Placeholder, Placeholder, Placeholder);

        // Eligible CUDA calls use the fused kernels unless the portable implementation is requested.
        for implementation in [AttentionImplementation::Automatic, AttentionImplementation::Fused] {
            assert!(matches!(
                select(implementation, configuration, input_structure, &forward_types, Some("cuda")),
                Some(NamedCompositeImplementation::Attention {
                    configuration: selected,
                    input_structure: selected_input_structure,
                    return_residual: false,
                }) if selected == configuration.with_implementation(implementation)
                    && selected_input_structure == input_structure,
            ));
        }
        assert!(
            select(AttentionImplementation::Portable, configuration, input_structure, &forward_types, Some("cuda"))
                .is_none()
        );

        // The residual request of a forward key reaches the selected implementation.
        let residual_key = AttentionKey::Forward { configuration, input_structure, return_residual: true };
        assert!(matches!(
            select_attention_implementation(
                &residual_key.to_composite_key(),
                &forward_types,
                &[operand_type.clone(), ArrayIrType::from(attention_array_type(DataType::BF16, &[1, 4, 2]))],
                Some("cuda"),
            ),
            Some(NamedCompositeImplementation::Attention { return_residual: true, .. }),
        ));

        // Ineligible calls lower their body unless the fused kernels are requested, which reports why they are not.
        let f32_types = vec![f32_operand_type; 3];
        for (input_types, target_platform, reason) in [
            (&forward_types, None, "the target platform is not CUDA"),
            (&forward_types, Some("cpu"), "the target platform is not CUDA"),
            (&f32_types, Some("cuda"), "the operand data type is not `f16` or `bf16`"),
        ] {
            assert!(
                select(
                    AttentionImplementation::Automatic,
                    configuration,
                    input_structure,
                    input_types,
                    target_platform
                )
                .is_none()
            );
            assert!(matches!(
                select(AttentionImplementation::Fused, configuration, input_structure, input_types, target_platform),
                Some(NamedCompositeImplementation::FusedAttentionUnavailable { name, reason: selected_reason })
                    if name == DOT_PRODUCT_ATTENTION_COMPOSITE_NAME && selected_reason == reason,
            ));
        }

        // Calls with a dropout state always lower their body, because the fused kernels cannot reproduce its mask.
        let dropout = configuration.with_dropout(0.5);
        let stateful = AttentionInputs { dropout_state: Some(Placeholder), ..input_structure };
        let stateful_types = forward_types.iter().cloned().chain([state_type]).collect::<Vec<_>>();
        assert!(select(AttentionImplementation::Automatic, dropout, stateful, &stateful_types, Some("cuda")).is_none());
        assert!(matches!(
            select(AttentionImplementation::Fused, dropout, stateful, &stateful_types, Some("cuda")),
            Some(NamedCompositeImplementation::FusedAttentionUnavailable { reason, .. })
                if reason.starts_with("the fused kernels cannot reproduce the dropout mask of a dropout state"),
        ));
        assert!(matches!(
            select(AttentionImplementation::Automatic, dropout, input_structure, &forward_types, Some("cuda")),
            Some(NamedCompositeImplementation::Attention { .. }),
        ));

        // Backward definitions select the fused backward kernels.
        let backward_types = forward_types
            .iter()
            .cloned()
            .chain([
                operand_type.clone(),
                ArrayIrType::from(attention_array_type(DataType::BF16, &[1, 4, 2])),
                operand_type.clone(),
            ])
            .collect::<Vec<_>>();
        let backward_key = AttentionKey::Backward { configuration, input_structure }.to_composite_key();
        assert!(matches!(
            select_attention_implementation(&backward_key, &backward_types, &forward_types, Some("cuda")),
            Some(NamedCompositeImplementation::AttentionBackward { .. }),
        ));

        // Other composites, malformed attention keys, and boundaries that do not match their key are declined.
        assert!(
            select_attention_implementation(
                &CompositeKey::new("ryft.other", 1),
                &forward_types,
                &forward_types,
                Some("cuda")
            )
            .is_none()
        );
        let malformed =
            CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION);
        assert!(select_attention_implementation(&malformed, &forward_types, &forward_types, Some("cuda")).is_none());
        assert!(
            select(
                AttentionImplementation::Automatic,
                configuration,
                input_structure,
                &forward_types[..2],
                Some("cuda")
            )
            .is_none()
        );
        let training_key = CompositeKey::new(DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME, 1);
        assert!(select_attention_implementation(&training_key, &forward_types, &forward_types, Some("cuda")).is_none());
    }

    #[test]
    fn test_fused_attention_on_cuda() {
        // Qualifying causal BF16 attention uses the cuDNN fMHA ABI and preserves the causal mask, while the unmasked
        // F16 form uses the same target with `NO_MASK`.
        for (data_type, causal, mask_type, element_type) in
            [(DataType::BF16, true, "CAUSAL", "BF16"), (DataType::F16, false, "NO_MASK", "F16")]
        {
            let configuration = AttentionConfiguration::new().with_scale(0.125).with_causal(causal);
            let input_types = vec![attention_array_type(data_type, &[1, 4, 2, 8]); 3];
            let module = lower(
                &forward_program(
                    configuration,
                    AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                    false,
                    input_types,
                    1,
                ),
                Some("cuda"),
            )
            .unwrap();
            assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax("));
            assert!(module.contains(mask_type));
            assert!(module.contains(element_type));
            assert!(module.contains("\\22dropout_rate\\22:0.0,\\22seed\\22:42"));
            assert!(!module.contains("stablehlo.exponential"));
        }

        // A broadcastable bias is expanded to the physical score shape before entering the `ScaleBias` ABI.
        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, true, false);
        let configuration = AttentionConfiguration::new().with_scale(0.125).with_causal(true);
        let module =
            lower(&forward_program(configuration, input_structure, false, input_types, 1), Some("cuda")).unwrap();
        assert!(module.contains("stablehlo.broadcast_in_dim %arg3"));
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax("));
        assert!(module.contains("tensor<2x2x4x4xbf16>"));

        // Residual attention carries both length vectors and converts the `f32` kernel residual to the query data type.
        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, false, true);
        let module =
            lower(&forward_program(configuration, input_structure, true, input_types, 1), Some("cuda")).unwrap();
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax("));
        assert!(module.contains("PADDING_CAUSAL"));
        assert!(module.contains("tensor<2x2x4xf32>"));
        assert!(module.contains("stablehlo.convert"));

        // Dropout without a dropout state selects the dropout target with the fixed seed, and a left local window
        // records its exclusive length.
        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, false, false);
        let dropout = configuration.with_local_window((1, 0)).with_dropout(0.5);
        let module = lower(&forward_program(dropout, input_structure, false, input_types, 1), Some("cuda")).unwrap();
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmaxDropout("));
        assert!(module.contains("\\22dropout_rate\\22:0.5,\\22seed\\22:42,\\22sliding_window_length\\22:2"));
    }

    #[test]
    fn test_fused_attention_backward_on_cuda() {
        // Fused training stages the forward kernels with their residual and the backward kernels, which restore the
        // input cotangents and sum the score-shaped bias cotangent to the bias shape.
        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, true, false);
        let configuration = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_implementation(AttentionImplementation::Fused);
        let program = training_program(configuration, input_structure, input_types.clone());
        assert_eq!(
            program.2,
            vec![
                input_types[0].clone(),
                input_types[0].clone(),
                input_types[1].clone(),
                input_types[2].clone(),
                input_types[3].clone(),
            ],
        );
        let module = lower(&program, Some("cuda")).unwrap();
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax("));
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward("));
        assert!(module.contains("stablehlo.reduce"));
        assert!(module.contains("tensor<1x2x4x4xbf16>"));
        assert!(!module.contains("stablehlo.exponential"));

        // The automatic training implementation is portable, so it never uses the fused kernels.
        let configuration = configuration.with_implementation(AttentionImplementation::Automatic);
        let module = lower(&training_program(configuration, input_structure, input_types), Some("cuda")).unwrap();
        assert!(!module.contains("__cudnn$fmha"));
        assert!(module.contains("stablehlo.exponential"));
    }

    #[test]
    fn test_fused_attention_unavailable() {
        // Requesting the fused kernels for calls that they cannot implement reports why instead of silently lowering
        // the portable body (e.g., `f32` operands on CUDA, any operands without a CUDA target, a dropout state, or a
        // padded right local window).
        let fused = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_implementation(AttentionImplementation::Fused);
        let (input_structure, input_types) = extended_input_types(DataType::F32, 2, 2, false, false);
        let program = forward_program(fused, input_structure, false, input_types.clone(), 1);
        for (target_platform, reason) in
            [(Some("cuda"), "the operand data type is not `f16` or `bf16`"), (None, "the target platform is not CUDA")]
        {
            assert!(matches!(
                lower(&program, target_platform),
                Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason: message }))
                    if operation == DOT_PRODUCT_ATTENTION_COMPOSITE_NAME
                        && message == format!("the fused attention implementation is unavailable because {reason}"),
            ));
        }
        assert!(matches!(
            lower(&training_program(fused, input_structure, input_types), Some("cuda")),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason.contains("the operand data type is not `f16` or `bf16`"),
        ));

        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, false, false);
        let state_type = attention_array_type(DataType::U64, &[2]);
        let stateful_types = input_types.into_iter().chain([state_type]).collect::<Vec<_>>();
        let stateful = AttentionInputs { dropout_state: Some(Placeholder), ..input_structure };
        let program = forward_program(fused.with_dropout(0.5), stateful, false, stateful_types, 1);
        assert!(matches!(
            lower(&program, Some("cuda")),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason.contains("the fused kernels cannot reproduce the dropout mask of a dropout state"),
        ));

        // The fused kernels have no right local-window radius. Causal masking makes that radius zero, so only a
        // configuration without it can request one.
        let (input_structure, input_types) = extended_input_types(DataType::BF16, 2, 2, false, true);
        let windowed = fused.with_causal(false).with_local_window((2, 1));
        let program = forward_program(windowed, input_structure, false, input_types, 1);
        assert!(matches!(
            lower(&program, Some("cuda")),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason.contains("a nonzero right local-window radius requires causal masking"),
        ));
    }

    #[test]
    fn test_portable_attention() {
        // Without target information, repeated calls share one function that contains the portable body: the score
        // dot, causal mask, stable softmax, value dot, and final layout restoration.
        let configuration = AttentionConfiguration::new().with_scale(0.125).with_causal(true);
        let input_types = vec![attention_array_type(DataType::F32, &[1, 4, 2, 8]); 3];
        let program = forward_program(
            configuration,
            AttentionInputs::new(Placeholder, Placeholder, Placeholder),
            false,
            input_types.clone(),
            2,
        );
        let module = to_mlir_module_for_program(&program.0, &[], &program.1, &program.2, "main", None, None).unwrap();
        assert_eq!(module.matches("func.func private @ryft.dot_product_attention.v1(").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention.v1(").count(), 2);
        assert_eq!(module.matches("stablehlo.dot_general").count(), 2);
        assert!(module.contains("stablehlo.compare LE"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));

        // `f32` operands, unaligned head dimensions, and `f8` operands do not qualify for the fused kernels, so even
        // CUDA targets lower the portable body, which converts narrow operands for its `f32` softmax.
        for (data_type, head_dimension) in [(DataType::F32, 8), (DataType::BF16, 4), (DataType::F8E4M3FN, 8)] {
            let input_types = vec![attention_array_type(data_type, &[1, 4, 2, head_dimension]); 3];
            let program = forward_program(
                configuration,
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                input_types,
                1,
            );
            let module = lower(&program, Some("cuda")).unwrap();
            assert!(!module.contains("fmha"));
            assert!(module.contains("stablehlo.dot_general"));
        }

        // Grouped-query expansion, a broadcast bias, causal and local masking, and the residual all live in the body.
        let (input_structure, input_types) = extended_input_types(DataType::F32, 4, 2, true, false);
        let program = forward_program(configuration.with_local_window((1, 0)), input_structure, true, input_types, 1);
        let module = lower(&program, None).unwrap();
        assert!(module.contains("stablehlo.broadcast_in_dim"));
        assert!(module.contains("stablehlo.reshape"));
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.log"));
        assert_eq!(
            program.2,
            vec![attention_array_type(DataType::F32, &[2, 4, 4, 8]), attention_array_type(DataType::F32, &[2, 4, 4])],
        );

        // The backward body recomputes the normalized probabilities and reduces the grouped key and value cotangents
        // and the broadcast bias cotangent.
        let (input_structure, input_types) = extended_input_types(DataType::F32, 4, 2, true, false);
        let module = lower(&training_program(configuration, input_structure, input_types), None).unwrap();
        assert!(module.matches("stablehlo.dot_general").count() >= 6);
        assert!(module.contains("stablehlo.exponential"));
        assert!(module.contains("tensor<1x4x4x4xf32>"));
        assert!(!module.contains("__cudnn$fmha"));
    }

    #[test]
    fn test_portable_attention_with_dynamic_extents() {
        // Portable attention obtains every logical extent from the mixed array IR, so bounded-dynamic programs do not
        // need to expose physical padding as sequence-length inputs and support the same grouped-query and window
        // configurations as static programs. Explicit lengths remain an independent semantic mask when provided.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4)).unwrap());
        let operand_type = |heads: usize| {
            ArrayType::new(
                DataType::BF16,
                Shape::new(vec![
                    CoreDimension::Dynamic(batch.clone()),
                    CoreDimension::Static(4),
                    CoreDimension::Static(heads),
                    CoreDimension::Static(8),
                ]),
            )
        };
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Dynamic(batch.clone())]));
        let boundary = |key_value_heads: usize, sequence_lengths: bool| {
            let mut input_types = vec![operand_type(2), operand_type(key_value_heads), operand_type(key_value_heads)];
            if sequence_lengths {
                input_types.extend([lengths_type.clone(), lengths_type.clone()]);
            }
            let input_structure = AttentionInputs {
                query_sequence_lengths: sequence_lengths.then_some(Placeholder),
                key_value_sequence_lengths: sequence_lengths.then_some(Placeholder),
                ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
            };
            (input_structure, input_types)
        };
        let configuration = AttentionConfiguration::new().with_scale(0.125);
        let windowed = configuration.with_causal(true).with_local_window((1, 0));
        for (key_value_heads, sequence_lengths, configuration) in
            [(2, false, configuration), (2, true, windowed), (1, true, configuration)]
        {
            let (input_structure, input_types) = boundary(key_value_heads, sequence_lengths);
            let program = forward_program(configuration, input_structure, false, input_types, 1);
            assert_eq!(program.2, vec![operand_type(2)]);
            lower(&program, None).unwrap();
        }

        // The analytical backward uses the same first-class dimensions and mask construction.
        for (sequence_lengths, configuration) in [(false, configuration), (true, windowed)] {
            let (input_structure, input_types) = boundary(2, sequence_lengths);
            lower(&training_program(configuration, input_structure, input_types), None).unwrap();
        }

        // Bounded-dynamic fused attention requires sequence lengths, which mask the physical padding.
        let fused = configuration.with_implementation(AttentionImplementation::Fused);
        let (input_structure, input_types) = boundary(2, true);
        let module = lower(&forward_program(fused, input_structure, false, input_types, 1), Some("cuda")).unwrap();
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax("));
        assert!(module.contains("PADDING"));
        let (input_structure, input_types) = boundary(2, false);
        assert!(matches!(
            lower(&forward_program(fused, input_structure, false, input_types, 1), Some("cuda")),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { reason, .. }))
                if reason.contains("bounded-dynamic operands require query or key/value sequence lengths"),
        ));
    }

    #[test]
    fn test_attention_jax_surface() {
        // Rank-three attention, an arbitrary Boolean mask, one independently present sequence-length input, an
        // asymmetric local window, default scaling, and a residual all lower through the portable body.
        let query_type = attention_array_type(DataType::F32, &[2, 1, 2]);
        let key_value_type = attention_array_type(DataType::F32, &[3, 1, 2]);
        let mask_type = attention_array_type(DataType::Boolean, &[1, 2, 3]);
        let lengths_type = attention_array_type(DataType::I32, &[1]);
        let input_structure = AttentionInputs {
            mask: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        let configuration = AttentionConfiguration::new()
            .with_local_window(Some((2, 1)))
            .with_implementation(AttentionImplementation::Portable);
        let input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type, mask_type, lengths_type];
        let program = forward_program(configuration, input_structure, true, input_types, 1);
        let module = lower(&program, None).unwrap();
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));
        assert_eq!(program.2, vec![query_type, attention_array_type(DataType::F32, &[2, 1])]);

        // The cuDNN adapter normalizes `TNH` inputs, turns an arbitrary Boolean mask into an additive bias, and
        // synthesizes the absent key/value length vector while preserving a supported causal left window. The
        // matching backward discards the synthetic mask-bias cotangent and returns the three input cotangents.
        let query_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let key_value_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let input_types = vec![
            query_type.clone(),
            key_value_type.clone(),
            key_value_type.clone(),
            attention_array_type(DataType::Boolean, &[1, 4, 4]),
            attention_array_type(DataType::I32, &[1]),
        ];
        let configuration = AttentionConfiguration::new()
            .with_causal(true)
            .with_local_window((2, 0))
            .with_implementation(AttentionImplementation::Fused);
        let program = forward_program(configuration, input_structure, true, input_types.clone(), 1);
        let forward = lower(&program, Some("cuda")).unwrap();
        assert!(forward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax("));
        assert!(forward.contains("PADDING_CAUSAL"));
        assert!(forward.contains("stablehlo.select"));
        assert!(forward.contains("stablehlo.get_dimension_size %arg1, dim = 0"));
        assert!(forward.contains("tensor<4x2x8xf16>"));
        assert_eq!(program.2, vec![query_type.clone(), attention_array_type(DataType::F16, &[4, 2])]);

        let program = training_program(configuration, input_structure, input_types);
        let backward = lower(&program, Some("cuda")).unwrap();
        assert!(backward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward("));
        assert_eq!(program.2, vec![query_type.clone(), query_type, key_value_type.clone(), key_value_type]);
    }
}
