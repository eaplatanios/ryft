use std::collections::HashMap;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, DataType, Dimension, DomainTracingContext, EmptyRegionDriver, MemberInterpretableOperation, Operation,
    ProgramError, ScaledDotOperation, Shape, Value,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::{CustomCallApiVersion, CustomCallMemoryLayouts};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, StringRef, Type,
    Value as MlirValue, ValueRef,
};

use crate::experimental::domains::XlaTracer;
use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::compositions::{
    NamedCompositionAttribute, NamedCompositionKey, lower_decomposition_call, named_composition_input_types,
};
use crate::experimental::lowering::types::{
    lower_element_type, lower_f64_constant_splat, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_tensor_type, physical_bound_type,
};
use crate::experimental::ops::FlatXlaProgram;
use crate::{Error, LowerableOperation, LowerableOperationFamily, XlaDomain, XlaLowerer};

/// Traces the canonical scaled-dot decomposition for one validated operation boundary.
pub(crate) fn trace_scaled_dot_composition(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, Error> {
    let input_types = named_composition_input_types(&scaled_dot_composite_input_types(operation, input_types)?)?;
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let [lhs, rhs, lhs_scale, rhs_scale] = inputs.as_slice() else {
                return Err(ProgramError::InvalidInputCount { expected: 4, actual: inputs.len() });
            };
            let mut inputs = vec![lhs.clone(), rhs.clone()];
            inputs.extend(operation.has_lhs_scale().then(|| lhs_scale.clone()));
            inputs.extend(operation.has_rhs_scale().then(|| rhs_scale.clone()));
            operation.interpret_in_parent(&lhs.domain(), &EmptyRegionDriver, &inputs)
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Returns the dummy scale type used by JAX's four-operand `xla.scaled_dot` boundary for one absent scale.
fn scaled_dot_dummy_scale_type(elements: &ArrayType, contracting_dimensions: &[usize]) -> ArrayType {
    ArrayType::new(
        DataType::F8E8M0FNU,
        Shape::new(
            elements
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(
                    |(axis, dimension)| {
                        if contracting_dimensions.contains(&axis) { Dimension::Static(1) } else { dimension.clone() }
                    },
                )
                .collect(),
        ),
    )
}

/// Expands one public scaled-dot boundary to the exact four inputs consumed by the StableHLO composite.
fn scaled_dot_composite_input_types(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, Error> {
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_types, expected_input_count, ProgramError);
    let mut scale_index = 2;
    let lhs_scale = if operation.has_lhs_scale() {
        let r#type = input_types[scale_index].clone();
        scale_index += 1;
        r#type
    } else {
        scaled_dot_dummy_scale_type(&input_types[0], operation.dimensions().lhs_contracting_dimensions())
    };
    let rhs_scale = if operation.has_rhs_scale() {
        input_types[scale_index].clone()
    } else {
        scaled_dot_dummy_scale_type(&input_types[1], operation.dimensions().rhs_contracting_dimensions())
    };
    Ok(vec![input_types[0].clone(), input_types[1].clone(), lhs_scale, rhs_scale])
}

/// Returns whether scaled dot can use a physical CUDA composite boundary without losing a runtime contracting-axis
/// requirement. Dynamic contracting element or scale dimensions retain the logical portable decomposition, whose
/// ordinary dimension SSA proves their block ratio before computing the result.
fn scaled_dot_uses_cuda_physical_boundary(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<bool, Error> {
    if target_platform != Some("cuda") {
        return Ok(false);
    }
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_types, expected_input_count, ProgramError);
    let mut scale_index = 2;
    for (element_index, has_scale, contracting_dimensions) in [
        (0, operation.has_lhs_scale(), operation.dimensions().lhs_contracting_dimensions()),
        (1, operation.has_rhs_scale(), operation.dimensions().rhs_contracting_dimensions()),
    ] {
        if contracting_dimensions
            .iter()
            .any(|axis| matches!(input_types[element_index].dimension(*axis), Dimension::Dynamic(_)))
        {
            return Ok(false);
        }
        if has_scale {
            if contracting_dimensions
                .iter()
                .any(|axis| matches!(input_types[scale_index].dimension(*axis), Dimension::Dynamic(_)))
            {
                return Ok(false);
            }
            scale_index += 1;
        }
    }
    Ok(true)
}

/// Returns the input permutations that bring one scaled dot to the `[batch, free, contracting]` layout (without
/// `batch` when there is no batching dimension) of the `__op$block_scaled_dot` custom call, or [`None`] when the call
/// does not qualify for it. On CUDA, XLA's `BlockScalingRewriter` expands that custom call to cuDNN's block-scaled
/// matrix product when the device has compute capability 10 or higher and cuDNN is at least 9.10, and to a
/// dequantized `dot_general` otherwise. Unlike the `xla.scaled_dot` composite, which the pinned XLA replaces only for
/// `f8e8m0fnu` scales, this also covers NVFP4.
///
/// A call qualifies on CUDA when both scales are present, it accumulates at `f32` or `bf16`, its contracting extents
/// are static (see [`scaled_dot_uses_cuda_physical_boundary`]), it has one contracting dimension and at most one
/// batching dimension, and each side has exactly one free dimension. The pair of formats must also be one that cuDNN
/// accepts, with the block size given by the ratio of the element and scale contracting extents. That is NVFP4 on both
/// sides (i.e., `f4e2m1fn` elements with `f8e4m3fn` scales over blocks of 16), or MXFP8 (i.e., `f8e4m3fn` or `f8e5m2`
/// elements with `f8e8m0fnu` scales over blocks of 32) unless both sides use `f8e5m2`. A call that fails these checks
/// keeps the composite, so XLA never receives a custom call that it cannot hand to cuDNN on a capable device.
fn block_scaled_dot_custom_call_permutations(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<Option<[Vec<usize>; 2]>, Error> {
    let dimensions = operation.dimensions();
    let batching_dimension_count = dimensions.lhs_batching_dimensions().len();
    if !operation.has_lhs_scale()
        || !operation.has_rhs_scale()
        || !matches!(operation.accumulation_type(), DataType::F32 | DataType::BF16)
        || dimensions.lhs_contracting_dimensions().len() != 1
        || batching_dimension_count > 1
        || !scaled_dot_uses_cuda_physical_boundary(operation, input_types, target_platform)?
    {
        return Ok(None);
    }
    let mut formats = Vec::with_capacity(2);
    let mut permutations = Vec::with_capacity(2);
    for (element_index, scale_index, batching_dimensions, contracting_dimension) in [
        (0, 2, dimensions.lhs_batching_dimensions(), dimensions.lhs_contracting_dimensions()[0]),
        (1, 3, dimensions.rhs_batching_dimensions(), dimensions.rhs_contracting_dimensions()[0]),
    ] {
        let (elements, scales) = (&input_types[element_index], &input_types[scale_index]);
        let rank = elements.rank();
        if rank != batching_dimension_count + 2 {
            return Ok(None);
        }
        let (Dimension::Static(element_extent), Dimension::Static(scale_extent)) =
            (elements.dimension(contracting_dimension), scales.dimension(contracting_dimension))
        else {
            return Ok(None);
        };
        if scale_extent == 0 || element_extent % scale_extent != 0 {
            return Ok(None);
        }
        formats.push((elements.data_type(), scales.data_type(), element_extent / scale_extent));
        let free_dimension = (0..rank)
            .find(|axis| *axis != contracting_dimension && !batching_dimensions.contains(axis))
            .unwrap();
        permutations.push(
            batching_dimensions
                .iter()
                .copied()
                .chain([free_dimension, contracting_dimension])
                .collect::<Vec<_>>(),
        );
    }
    let supported = matches!(
        (formats[0], formats[1]),
        ((DataType::F4E2M1FN, DataType::F8E4M3FN, 16), (DataType::F4E2M1FN, DataType::F8E4M3FN, 16))
            | ((DataType::F8E4M3FN, DataType::F8E8M0FNU, 32), (DataType::F8E4M3FN, DataType::F8E8M0FNU, 32))
            | ((DataType::F8E4M3FN, DataType::F8E8M0FNU, 32), (DataType::F8E5M2, DataType::F8E8M0FNU, 32))
            | ((DataType::F8E5M2, DataType::F8E8M0FNU, 32), (DataType::F8E4M3FN, DataType::F8E8M0FNU, 32))
    );
    Ok(supported.then(|| permutations.try_into().unwrap()))
}

/// Returns the typed scaled-dot composition boundary used for the target platform, or [`None`] when the call lowers to
/// the `__op$block_scaled_dot` custom call (see [`block_scaled_dot_custom_call_permutations`]) and needs no
/// composition. XLA's CUDA block-scaled-dot replacement cannot propagate dynamic-dimension annotations through its
/// fused HLO, so eligible CUDA calls use the bounded static carriers while the caller masks physical suffix lanes and
/// restores the logical result dimensions. Other calls retain the logical boundary consumed by the portable
/// decomposition.
pub(crate) fn scaled_dot_composition_types(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<Option<(Vec<ArrayType>, Vec<ArrayType>)>, Error> {
    if block_scaled_dot_custom_call_permutations(operation, input_types, target_platform)?.is_some() {
        return Ok(None);
    }
    if !scaled_dot_uses_cuda_physical_boundary(operation, input_types, target_platform)? {
        return Ok(Some((input_types.to_vec(), output_types.to_vec())));
    }
    Ok(Some((
        input_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
        output_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
    )))
}

/// Lowers one scaled dot. On CUDA, calls whose formats cuDNN accepts lower to the `__op$block_scaled_dot` custom call
/// (see [`block_scaled_dot_custom_call_permutations`]), which XLA hands to cuDNN's block-scaled matrix product on
/// capable devices. Every other call lowers to the exact four-operand `stablehlo.composite "xla.scaled_dot"` boundary
/// consumed by XLA's block-scaling replacement pass. Either scale omitted by the public Ryft call is represented by the
/// same JAX-compatible identity scale used by the pinned composite contract. The typed decomposition remains the
/// portable fallback when XLA does not replace the composite.
pub(crate) fn lower_scaled_dot_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ScaledDotOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_values, expected_input_count, ProgramError);
    check_count!("input", input_types, input_values.len(), ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let input_values = &lower_scaled_dot_input_barrier(input_values, input_types, block, location)?;
    if let Some(permutations) =
        block_scaled_dot_custom_call_permutations(operation, input_types, collective_state.target_platform())?
    {
        return lower_block_scaled_dot_custom_call(
            operation,
            &permutations,
            input_values,
            input_types,
            output_types,
            block,
            context,
            location,
        );
    }
    let uses_physical_boundary =
        scaled_dot_uses_cuda_physical_boundary(operation, input_types, collective_state.target_platform())?;
    // The composition types are present because the call does not lower to the custom call.
    let (composition_input_types, composition_output_types) =
        scaled_dot_composition_types(operation, input_types, output_types, collective_state.target_platform())?
            .unwrap();
    let decomposition = match collective_state.named_compositions.as_ref() {
        Some(functions) => functions.get(operation, &composition_input_types, &composition_output_types)?,
        None => None,
    };
    if let Some(function) = decomposition {
        if collective_state.target_platform() == Some("cuda") && !uses_physical_boundary {
            return lower_decomposition_call(function, input_values, output_types, block, context, location);
        }
        let composition_input_values = if uses_physical_boundary {
            input_values
                .iter()
                .zip(input_types)
                .map(|(value, r#type)| lower_physical_bound_value(*value, r#type, 0.0, block, context, location))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            input_values.to_vec()
        };
        let mut composite_values = vec![composition_input_values[0], composition_input_values[1]];
        let mut scale_index = 2;
        for (present, element_index, contracting_dimensions) in [
            (operation.has_lhs_scale(), 0, operation.dimensions().lhs_contracting_dimensions()),
            (operation.has_rhs_scale(), 1, operation.dimensions().rhs_contracting_dimensions()),
        ] {
            if present {
                composite_values.push(composition_input_values[scale_index]);
                scale_index += 1;
                continue;
            }
            let dummy_type =
                scaled_dot_dummy_scale_type(&composition_input_types[element_index], contracting_dimensions);
            let physical_type = physical_bound_type(&dummy_type)?;
            let physical_tensor_type = lower_tensor_type(&physical_type, context, location)?;
            let dummy = lower_f64_constant_splat(1.0, &physical_type, physical_tensor_type, block, context, location)?;
            if uses_physical_boundary {
                composite_values.push(dummy);
            } else {
                let sources =
                    (0..dummy_type.rank()).map(|axis| (input_values[element_index], axis)).collect::<Vec<_>>();
                composite_values.push(lower_restore_dynamic_dimensions(
                    dummy,
                    &dummy_type,
                    sources.as_slice(),
                    block,
                    context,
                    location,
                )?);
            }
        }
        let render_axes = |axes: &[usize]| axes.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
        let dimensions = format!(
            "[[[{}], [{}]], [[{}], [{}]]]",
            render_axes(operation.dimensions().lhs_contracting_dimensions()),
            render_axes(operation.dimensions().rhs_contracting_dimensions()),
            render_axes(operation.dimensions().lhs_batching_dimensions()),
            render_axes(operation.dimensions().rhs_batching_dimensions()),
        );
        let attributes = HashMap::from([
            (StringRef::from("dimension_numbers"), context.parse_attribute(dimensions.as_str())?),
            (
                StringRef::from("preferred_element_type"),
                context.type_attribute(lower_element_type(operation.accumulation_type(), context)?).as_ref(),
            ),
        ]);
        let result_types = composition_output_types
            .iter()
            .map(|r#type| lower_tensor_type(r#type, context, location).map(|r#type| r#type.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let composite = block.append_operation(stable_hlo::composite(
            "xla.scaled_dot",
            0,
            Some(&attributes),
            composite_values.as_slice(),
            function.symbol.as_str(),
            Vec::new(),
            result_types.as_slice(),
            location,
        )?)?;
        let results =
            (0..output_types.len()).map(|index| composite.result(index).unwrap().as_ref()).collect::<Vec<_>>();
        if !uses_physical_boundary {
            return Ok(results);
        }
        return Ok(vec![lower_restore_dynamic_dimensions(
            results[0],
            &output_types[0],
            scaled_dot_output_sources(operation, input_values, input_types).as_slice(),
            block,
            context,
            location,
        )?]);
    }

    Err(Error::UnsupportedOperation { op: format!("missing typed decomposition for `{}`", operation.name()) })
}

/// Places an optimization barrier on the scaled-dot inputs whose element types are narrow floating-point types. The
/// decomposition converts these inputs to `bf16` before contracting them, and when no native kernel replaces the
/// composite (e.g., on CPUs and on GPUs without block-scaled matrix units), XLA's floating-point conversion
/// simplification can fold a producer's conversion to the narrow type into that conversion (e.g., `f32 → f4e2m1fn →
/// bf16` into `f32 → bf16`), so the contraction would silently consume unquantized values. The barrier keeps both
/// conversions, as the `optimization_barrier` around the outputs of JAX's `quantize` helper does.
fn lower_scaled_dot_input_barrier<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let narrow = input_types
        .iter()
        .map(|r#type| r#type.data_type().is_floating_point() && r#type.data_type().bit_width() <= 8)
        .collect::<Vec<_>>();
    let barrier_inputs = input_values
        .iter()
        .zip(&narrow)
        .filter(|(_, narrow)| **narrow)
        .map(|(value, _)| *value)
        .collect::<Vec<_>>();
    if barrier_inputs.is_empty() {
        return Ok(input_values.to_vec());
    }
    let barrier = block.append_operation(stable_hlo::optimization_barrier(barrier_inputs.as_slice(), location)?)?;
    let mut barrier_index = 0;
    Ok(input_values
        .iter()
        .zip(narrow)
        .map(|(value, narrow)| {
            if !narrow {
                return *value;
            }
            barrier_index += 1;
            barrier.result(barrier_index - 1).unwrap().as_ref()
        })
        .collect())
}

/// Lowers one qualifying scaled dot (see [`block_scaled_dot_custom_call_permutations`]) to
/// `stablehlo.custom_call @__op$block_scaled_dot(lhs, rhs, lhs_scale, rhs_scale)`. The inputs are first exposed as
/// their bounded physical carriers, because XLA's block-scaling expansion cannot propagate dynamic dimension
/// annotations. Elements are padded with zeros, so padded lanes contribute nothing to the result, and scales are
/// padded with ones, which every scale type represents (unlike zero for `f8e8m0fnu`). Each element array and its scale
/// array are then transposed by `permutations` to the `[batch, free, contracting]` layout that the custom call
/// requires. XLA gives a transpose the permuted memory layout of its input, which makes it a bitcast, but
/// `BlockScalingRewriter` hands only inputs with the default major-to-minor layout to cuDNN. The custom call therefore
/// declares default input and result layouts, which makes XLA materialize every transposed input. The custom call
/// returns its result in the `[batch, lhs free, rhs free]` order of `dot_general`, so only its logical dynamic extents
/// need restoring.
#[allow(clippy::too_many_arguments)]
fn lower_block_scaled_dot_custom_call<'b, 'c: 'b, 't: 'c>(
    operation: &ScaledDotOperation,
    permutations: &[Vec<usize>; 2],
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let mut operands = Vec::with_capacity(input_values.len());
    for (index, (value, r#type)) in input_values.iter().zip(input_types).enumerate() {
        let padding_value = if index < 2 { 0.0 } else { 1.0 };
        let value = lower_physical_bound_value(*value, r#type, padding_value, block, context, location)?;
        let permutation = &permutations[index % 2];
        if permutation.iter().enumerate().all(|(axis, source)| axis == *source) {
            operands.push(value);
            continue;
        }
        let transposed = block.append_operation(stable_hlo::transpose(value, permutation, location)?)?;
        operands.push(transposed.result(0).expect("stablehlo.transpose should return one result").as_ref());
    }
    let output_type = lower_tensor_type(&physical_bound_type(&output_types[0])?, context, location)?;
    let default_layout = |rank: usize| (0..rank).rev().collect::<Vec<_>>();
    let custom_call = block.append_operation(stable_hlo::custom_call(
        operands.as_slice(),
        "__op$block_scaled_dot",
        false,
        None,
        CustomCallApiVersion::Original,
        &[],
        Some(CustomCallMemoryLayouts {
            operands: input_types.iter().map(|r#type| default_layout(r#type.rank())).collect(),
            results: vec![default_layout(output_types[0].rank())],
        }),
        &[],
        None,
        &[output_type],
        location,
    )?)?;
    let result = custom_call.result(0).expect("stablehlo.custom_call should return one result").as_ref();
    Ok(vec![lower_restore_dynamic_dimensions(
        result,
        &output_types[0],
        scaled_dot_output_sources(operation, input_values, input_types).as_slice(),
        block,
        context,
        location,
    )?])
}

/// Returns the input value and axis that carry the runtime extent of each scaled-dot output dimension, in the
/// `[batch, lhs free, rhs free]` output order of `dot_general`.
fn scaled_dot_output_sources<'b, 'c: 'b, 't: 'c>(
    operation: &ScaledDotOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
) -> Vec<(ValueRef<'b, 'c, 't>, usize)> {
    let dimensions = operation.dimensions();
    dimensions
        .lhs_batching_dimensions()
        .iter()
        .copied()
        .chain(dimensions.lhs_result_dimensions(input_types[0].rank()))
        .map(|axis| (input_values[0], axis))
        .chain(
            dimensions
                .rhs_result_dimensions(input_types[1].rank())
                .into_iter()
                .map(|axis| (input_values[1], axis)),
        )
        .collect()
}

/// Builds the canonical named-composition identity for one scaled-dot boundary.
pub(crate) fn scaled_dot_composition_key(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> Result<NamedCompositionKey, Error> {
    let dimensions = operation.dimensions();
    Ok(NamedCompositionKey {
        name: "xla.scaled_dot",
        version: 0,
        decomposition: "ryft.scaled_dot.v0",
        attributes: vec![
            (
                "dimension_numbers",
                NamedCompositionAttribute::AxisLists(vec![
                    vec![
                        dimensions.lhs_contracting_dimensions().to_vec(),
                        dimensions.rhs_contracting_dimensions().to_vec(),
                    ],
                    vec![dimensions.lhs_batching_dimensions().to_vec(), dimensions.rhs_batching_dimensions().to_vec()],
                ]),
            ),
            ("preferred_element_type", NamedCompositionAttribute::DataType(operation.accumulation_type())),
        ],
        input_types: scaled_dot_composite_input_types(operation, input_types)?,
        output_types: output_types.to_vec(),
    })
}

impl LowerableOperation for ScaledDotOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let collective_state = lowerer.collective_state.clone();
        lower_scaled_dot_to_mlir(
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array, ArrayIrType, ArrayOperation, BlockQuantize, BlockQuantizedValue,
        BlockScaledDotConfiguration, ConvertElementType, Device, DeviceMesh, DimensionBounds, DimensionVariable,
        DotDimensionNumbers, EagerContext, LogicalMesh, MeshAxis, MeshAxisType, Placeholder, ProjectedContext,
        RandomAlgorithm, Sharding, block_scaled_dot, differentiate_at,
    };

    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::lowering::tests::{
        MixedValue, XlaProgramBuilder, add_xla_jit_call, execute_mixed_program, unproject_plain_program,
    };
    use crate::experimental::ops::{
        XlaArrayConstant, XlaConstant, XlaProgram, XlaProgramBuilder as ArrayIrXlaProgramBuilder,
    };
    use crate::jit::{CompiledXlaFunction, compile};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, XlaArray, XlaDomain, XlaSession, to_mlir_module_for_program};

    use super::*;

    /// Builds the NVFP4 scaled-dot program shared by the platform-gated lowering fixtures below:
    /// `f4e2m1fn [2, 16]` operands with `f8e4m3fn [2, 1]` scales over blocks of 16.
    fn scaled_dot_fixture_program() -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>)
    {
        use ryft_core::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        (unproject_plain_program(program), input_types, vec![output_type])
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scaled_dot_composition_without_target_platform() {
        // Target-independent StableHLO contains the exact named composite plus its portable typed decomposition;
        // the eventual XLA target decides whether to replace that boundary. The narrow inputs pass through an
        // optimization barrier, so their producers' conversions cannot fold into the decomposition's conversions.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @xla.scaled_dot(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x16xf4E2M1FN>, %arg2: tensor<2x1xf8E4M3FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xbf16>
                    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x1xf8E4M3FN>) -> tensor<2x1x16xf8E4M3FN>
                    %2 = stablehlo.reshape %1 : (tensor<2x1x16xf8E4M3FN>) -> tensor<2x16xf8E4M3FN>
                    %3 = stablehlo.convert %2 : (tensor<2x16xf8E4M3FN>) -> tensor<2x16xbf16>
                    %4 = stablehlo.multiply %0, %3 : tensor<2x16xbf16>
                    %5 = stablehlo.convert %arg1 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xbf16>
                    %6 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<2x1xf8E4M3FN>) -> tensor<2x1x16xf8E4M3FN>
                    %7 = stablehlo.reshape %6 : (tensor<2x1x16xf8E4M3FN>) -> tensor<2x16xf8E4M3FN>
                    %8 = stablehlo.convert %7 : (tensor<2x16xf8E4M3FN>) -> tensor<2x16xbf16>
                    %9 = stablehlo.multiply %5, %8 : tensor<2x16xbf16>
                    %10 = stablehlo.dot_general %4, %9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16xbf16>, tensor<2x16xbf16>) -> tensor<2x2xf32>
                    return %10 : tensor<2x2xf32>
                  }
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x16xf4E2M1FN>, %arg2: tensor<2x1xf8E4M3FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0:4 = stablehlo.optimization_barrier %arg0, %arg1, %arg2, %arg3 : tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>
                    %1 = stablehlo.composite "xla.scaled_dot" %0#0, %0#1, %0#2, %0#3 {composite_attributes = {dimension_numbers = [[[1], [1]], [[], []]], preferred_element_type = f32}, decomposition = @xla.scaled_dot} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_materializes_absent_scaled_dot_scales() {
        // The public optional-scale API still lowers to JAX's exact four-operand composite. Contracting dummy-scale
        // axes have size one, noncontracting axes retain the corresponding element extents, and the decomposition
        // ignores the identity operands rather than imposing block-ratio requirements on an absent scale.
        let element_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![builder.add_input(element_type.clone()), builder.add_input(element_type.clone())];
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                    DataType::F32,
                    false,
                    false,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[element_type.clone(), element_type],
            &[output_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(module.contains(
            "func.func private @xla.scaled_dot(%arg0: tensor<2x2xf8E4M3FN>, %arg1: tensor<2x2xf8E4M3FN>, \
             %arg2: tensor<2x1xf8E8M0FNU>, %arg3: tensor<1x2xf8E8M0FNU>)",
        ));
        assert_eq!(module.matches("dense<1.000000e+00> : tensor<f8E8M0FNU>").count(), 2);
        assert!(module.contains("stablehlo.optimization_barrier %arg0, %arg1 :"));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %0#0, %0#1"));
        assert!(!module.contains("func.call @xla.scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_block_scaled_dot_custom_call_on_cuda() {
        // On CUDA, NVFP4 inputs lower to the `__op$block_scaled_dot` custom call, which XLA hands to cuDNN's
        // block-scaled matrix product on capable devices, instead of the composite. The narrow inputs still pass
        // through the optimization barrier, because XLA's dequantized fallback for other devices would otherwise
        // expose foldable conversion pairs.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x16xf4E2M1FN>, %arg2: tensor<2x1xf8E4M3FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0:4 = stablehlo.optimization_barrier %arg0, %arg1, %arg2, %arg3 : tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>
                    %1 = stablehlo.custom_call @__op$block_scaled_dot(%0#0, %0#1, %0#2, %0#3) {operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_transposes_block_scaled_dot_custom_call_inputs_on_cuda() {
        // A product that contracts the middle right dimension (e.g., the input gradient of a block-scaled dot)
        // transposes the right elements and scales to the `[batch, free, contracting]` layout of the custom call. The
        // left inputs already have that layout and stay untransposed.
        let element_type = |shape: [usize; 3]| ArrayType::new_static(DataType::F4E2M1FN, shape);
        let scale_type = |shape: [usize; 3]| ArrayType::new_static(DataType::F8E4M3FN, shape);
        let input_types =
            vec![element_type([1, 2, 32]), element_type([1, 32, 4]), scale_type([1, 2, 2]), scale_type([1, 2, 4])];
        let output_types = vec![ArrayType::new_static(DataType::BF16, [1, 2, 4])];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
                    DataType::BF16,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 4],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x2x32xf4E2M1FN>, %arg1: tensor<1x32x4xf4E2M1FN>, %arg2: tensor<1x2x2xf8E4M3FN>, %arg3: tensor<1x2x4xf8E4M3FN>) -> tensor<1x2x4xbf16> {
                    %0:4 = stablehlo.optimization_barrier %arg0, %arg1, %arg2, %arg3 : tensor<1x2x32xf4E2M1FN>, tensor<1x32x4xf4E2M1FN>, tensor<1x2x2xf8E4M3FN>, tensor<1x2x4xf8E4M3FN>
                    %1 = stablehlo.transpose %0#1, dims = [0, 2, 1] : (tensor<1x32x4xf4E2M1FN>) -> tensor<1x4x32xf4E2M1FN>
                    %2 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<1x2x4xf8E4M3FN>) -> tensor<1x4x2xf8E4M3FN>
                    %3 = stablehlo.custom_call @__op$block_scaled_dot(%0#0, %1, %0#2, %2) {operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<1x2x32xf4E2M1FN>, tensor<1x4x32xf4E2M1FN>, tensor<1x2x2xf8E4M3FN>, tensor<1x4x2xf8E4M3FN>) -> tensor<1x2x4xbf16>
                    return %3 : tensor<1x2x4xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_preserves_scaled_dot_composite_for_unsupported_formats_on_cuda() {
        // MXFP4 (i.e., `f4e2m1fn` elements with `f8e8m0fnu` scales over blocks of 32) is not a format pair that cuDNN
        // accepts, so CUDA lowering keeps the named composite and its private decomposition.
        let element_type = ArrayType::new_static(DataType::F4E2M1FN, [2, 32]);
        let scale_type = ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]);
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        let output_types = vec![ArrayType::new_static(DataType::F32, [2, 2])];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 4],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("func.func private @xla.scaled_dot"));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %0#0, %0#1, %0#2, %0#3"));
        assert!(!module.contains("stablehlo.custom_call @__op$block_scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_physicalizes_dynamic_scaled_dot_on_cuda() {
        // XLA's block-scaling expansion requires a static boundary. Ryft masks the bounded physical inputs of the
        // MXFP8 custom call, padding elements with zeros and scales with ones, and restores the logical row extent on
        // the result.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let lhs_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(32)]),
        );
        let rhs_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(1), Dimension::Static(32)]));
        let lhs_scale_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(1)]),
        );
        let rhs_scale_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(1)]));
        let input_types = vec![lhs_type, rhs_type, lhs_scale_type, rhs_scale_type];
        let output_types = vec![output_type];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 4],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4x32xf8E4M3FN>, %arg1: tensor<1x32xf8E4M3FN>, %arg2: tensor<4x1xf8E8M0FNU>, %arg3: tensor<1x1xf8E8M0FNU>, %arg4: tensor<i32>, %arg5: tensor<i32>) -> (tensor<?x1xf32, #stablehlo.bounds<4, ?>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg4, dim = 0 : (tensor<4x32xf8E4M3FN>, tensor<i32>) -> tensor<?x32xf8E4M3FN, #stablehlo.bounds<4, ?>>
                    %1 = stablehlo.set_dimension_size %arg2, %arg5, dim = 0 : (tensor<4x1xf8E8M0FNU>, tensor<i32>) -> tensor<?x1xf8E8M0FNU, #stablehlo.bounds<4, ?>>
                    %2:4 = stablehlo.optimization_barrier %0, %arg1, %1, %arg3 : tensor<?x32xf8E4M3FN, #stablehlo.bounds<4, ?>>, tensor<1x32xf8E4M3FN>, tensor<?x1xf8E8M0FNU, #stablehlo.bounds<4, ?>>, tensor<1x1xf8E8M0FNU>
                    %3 = stablehlo.get_dimension_size %2#0, dim = 0 : (tensor<?x32xf8E4M3FN, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %c = stablehlo.constant dense<4> : tensor<i32>
                    %4 = stablehlo.set_dimension_size %2#0, %c, dim = 0 : (tensor<?x32xf8E4M3FN, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x32xf8E4M3FN>
                    %5 = stablehlo.iota dim = 0 : tensor<4x32xi32>
                    %6 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<4x32xi32>
                    %7 = stablehlo.compare LT, %5, %6, SIGNED : (tensor<4x32xi32>, tensor<4x32xi32>) -> tensor<4x32xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f8E4M3FN>
                    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f8E4M3FN>) -> tensor<4x32xf8E4M3FN>
                    %9 = stablehlo.select %7, %4, %8 : tensor<4x32xi1>, tensor<4x32xf8E4M3FN>
                    %10 = stablehlo.get_dimension_size %2#2, dim = 0 : (tensor<?x1xf8E8M0FNU, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %c_0 = stablehlo.constant dense<4> : tensor<i32>
                    %11 = stablehlo.set_dimension_size %2#2, %c_0, dim = 0 : (tensor<?x1xf8E8M0FNU, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x1xf8E8M0FNU>
                    %12 = stablehlo.iota dim = 0 : tensor<4x1xi32>
                    %13 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<4x1xi32>
                    %14 = stablehlo.compare LT, %12, %13, SIGNED : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi1>
                    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f8E8M0FNU>
                    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f8E8M0FNU>) -> tensor<4x1xf8E8M0FNU>
                    %16 = stablehlo.select %14, %11, %15 : tensor<4x1xi1>, tensor<4x1xf8E8M0FNU>
                    %17 = stablehlo.custom_call @__op$block_scaled_dot(%9, %2#1, %16, %2#3) {operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x32xf8E4M3FN>, tensor<1x32xf8E4M3FN>, tensor<4x1xf8E8M0FNU>, tensor<1x1xf8E8M0FNU>) -> tensor<4x1xf32>
                    %18 = stablehlo.get_dimension_size %2#0, dim = 0 : (tensor<?x32xf8E4M3FN, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %19 = stablehlo.set_dimension_size %17, %18, dim = 0 : (tensor<4x1xf32>, tensor<i32>) -> tensor<?x1xf32, #stablehlo.bounds<4, ?>>
                    %20 = stablehlo.get_dimension_size %19, dim = 0 : (tensor<?x1xf32, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %21 = stablehlo.convert %20 : (tensor<i32>) -> tensor<i64>
                    return %19, %21 : tensor<?x1xf32, #stablehlo.bounds<4, ?>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dynamic_scaled_dot_composition_on_cuda() {
        // Dynamic contracting dimensions stay on the logical decomposition rather than CUDA's static fused
        // boundary. A single block makes divisibility trivial; folding the dimension arithmetic also preserves
        // the reshape extent identity without an equality assertion.
        for block_bound in [2, 3] {
            let elements = DimensionVariable::new("elements", DimensionBounds::new(32, Some(65)).unwrap());
            // A single block keeps reshape capacities equal, and the element bounds prove the minimum ratio.
            // Multiple possible blocks still fail when their physical reshape capacities differ.
            let blocks = DimensionVariable::new("blocks", DimensionBounds::new(1, Some(block_bound)).unwrap());
            let element_type = ArrayType::new(
                DataType::F8E4M3FN,
                Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(elements)]),
            );
            let scale_type = ArrayType::new(
                DataType::F8E8M0FNU,
                Shape::new(vec![
                    Dimension::Static(1),
                    if block_bound == 2 { Dimension::Static(1) } else { Dimension::Dynamic(blocks) },
                ]),
            );
            let output_type =
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]));
            let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let output = builder
                .add_instruction(
                    ScaledDotOperation::new(
                        DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                        DataType::F32,
                        true,
                        true,
                    ),
                    Vec::new(),
                    inputs,
                    None,
                )
                .unwrap()[0];
            let program = unproject_plain_program(
                builder
                    .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                        vec![output],
                        vec![Placeholder; 4],
                        vec![Placeholder],
                    )
                    .unwrap(),
            );
            let module = lower_mlir_module_for_program(
                &program,
                &[],
                &input_types,
                &[output_type],
                "main",
                None,
                None,
                Some("cuda"),
            );
            if block_bound == 3 {
                assert!(matches!(
                    module.err().unwrap(),
                    Error::UnsupportedOperation { op }
                        if op == "reshape with unequal physical capacities (input 128, output 64); dynamic bounds must \
                           preserve the physical element count",
                ));
                continue;
            }
            let module = module.unwrap().stable_hlo;

            assert!(module.contains("call @xla.scaled_dot"));
            assert!(!module.contains("stablehlo.custom_call @ryft.assert"));
            assert!(!module.contains("stablehlo.composite \"xla.scaled_dot\""));
        }
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_block_scaled_dot_custom_call_in_shared_jit_call_callee() {
        // A block-scaled-dot custom call nested inside a shared `jit_call` callee is emitted once, because the CUDA
        // target platform reaches the callee body, while both outer call sites reuse the same function.
        use ryft_core::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut callee_builder = XlaProgramBuilder::new();
        let callee_inputs = vec![
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(scale_type.clone()),
            callee_builder.add_input(scale_type.clone()),
        ];
        let callee_output = callee_builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                callee_inputs,
                None,
            )
            .unwrap()[0];
        let callee = Arc::new(unproject_plain_program(
            callee_builder.build(vec![callee_output], vec![Placeholder; 4], vec![Placeholder]).unwrap(),
        ));

        let mut builder = ArrayIrXlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone().into()),
            builder.add_input(element_type.clone().into()),
            builder.add_input(scale_type.clone().into()),
            builder.add_input(scale_type.clone().into()),
        ];
        let first = add_xla_jit_call(&mut builder, &callee, inputs.clone());
        let second = add_xla_jit_call(&mut builder, &callee, inputs);
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        let module = lower_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &[output_type],
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert_eq!(module.matches("func.func private @jit_call_0").count(), 1);
        assert_eq!(module.matches("stablehlo.custom_call @__op$block_scaled_dot").count(), 1);
        assert_eq!(module.matches("stablehlo.composite \"xla.scaled_dot\"").count(), 0);
        assert_eq!(module.matches("call @jit_call_0").count(), 2);
    }

    #[test]
    fn test_compile_block_quantize_dynamic_extents() {
        let client = execution_client();
        for (element_type, scale_type) in
            [(DataType::F8E4M3FN, DataType::F8E8M0FNU), (DataType::F4E2M1FN, DataType::F8E4M3FN)]
        {
            let extent = DimensionVariable::new("n", DimensionBounds::new(0, Some(9)).unwrap());
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
            let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
                |inputs: Vec<XlaTracer<'static>>| {
                    let BlockQuantizedValue { elements, scales, .. } =
                        inputs[0].block_quantize(4, element_type, scale_type)?;
                    // Return ordinary floating-point arrays so that the mixed executable fixture can decode them.
                    Ok(vec![elements.convert_element_type(DataType::F64)?, scales.convert_element_type(DataType::F64)?])
                },
                vec![ArrayIrType::Array(input_type)],
            )
            .unwrap();
            let program = program.into_flat_program().simplified().unwrap();
            let input = Array::vector(vec![3.0f64, 1.5, 0.5, 6.0]).unwrap();
            let BlockQuantizedValue { elements, scales, .. } =
                input.block_quantize(4, element_type, scale_type).unwrap();
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Array(vec![3.0, 1.5, 0.5, 6.0, 0.0, 0.0, 0.0, 0.0], vec![8])],
                    &[4],
                )
                .unwrap(),
                vec![
                    MixedValue::Array(elements.to_f64s(), vec![4]),
                    MixedValue::Array(scales.to_f64s(), vec![1]),
                    MixedValue::Dimension(4),
                    MixedValue::Dimension(1),
                ],
            );
        }
    }

    #[test]
    fn test_compile_block_scaled_dot_gradients_match_reference_backend() {
        // The compiled NVIDIA NVFP4 training recipe matches the reference backend on values off the NVFP4 grid. This
        // also shows that the compiled program keeps every quantization step rather than folding the conversions to
        // and from the narrow types away, and that random draws agree across backends.
        let client = execution_client();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let domain = XlaSession::new(&client).domain();
        let placed = |data_type: DataType, shape: &[usize]| {
            ArrayType::new_static(data_type, shape.to_vec())
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), shape.len()))
                .unwrap()
        };
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let configuration = BlockScaledDotConfiguration::nvfp4_training(dimensions);
        let function = block_scaled_dot::<ProjectedContext<XlaDomain<'_>, ArrayType>>(configuration.clone()).unwrap();
        let compiled: CompiledXlaFunction<
            '_,
            (ArrayType, ArrayType, ArrayType, ArrayType),
            (ArrayType, ArrayType, ArrayType),
        > = compile(
            move |(lhs, rhs, seed, state)| {
                let (output, pullback) = differentiate_at((lhs, rhs))
                    .with_captures(state)
                    .vjp(|(lhs, rhs), state| function.call(lhs, rhs, None, None, None, Some(state)))
                    .unwrap();
                let (lhs_cotangent, rhs_cotangent) = pullback.apply_materialized(seed).unwrap();
                (output, lhs_cotangent, rhs_cotangent)
            },
            (
                placed(DataType::F32, &[32, 32]),
                placed(DataType::F32, &[16, 32]),
                placed(DataType::F32, &[32, 16]),
                placed(DataType::U64, &[2]),
            ),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let values = |rows: usize, columns: usize, phase: f64| {
            (0..rows * columns).map(|index| (index as f64 * 0.37 + phase).sin() as f32).collect::<Vec<_>>()
        };
        let (lhs, rhs, seed) = (values(32, 32, 0.1), values(16, 32, 0.7), values(32, 16, 1.3));
        let device = |data_type: DataType, shape: &[usize], bytes: Vec<u8>| {
            XlaArray::from_host_buffer(&domain, placed(data_type, shape), mesh.clone(), bytes.as_slice()).unwrap()
        };
        let outputs = domain
            .interpret(
                &compiled.executable_function(),
                (
                    device(DataType::F32, &[32, 32], values_to_bytes(&lhs)),
                    device(DataType::F32, &[16, 32], values_to_bytes(&rhs)),
                    device(DataType::F32, &[32, 16], values_to_bytes(&seed)),
                    device(DataType::U64, &[2], values_to_bytes(&[7u64, 0])),
                ),
            )
            .unwrap();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let read = |array: &XlaArray<'_>| {
            let bytes = array.device_shard(device_id).unwrap().buffer().unwrap().copy_to_host(None).unwrap();
            values_from_bytes::<f32>(bytes.r#await().unwrap().as_slice())
        };

        let reference_function = block_scaled_dot::<EagerContext<Array, ArrayOperation<Array>>>(configuration).unwrap();
        let host = |shape: [usize; 2], values: &[f32]| {
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, shape), values).unwrap()
        };
        let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[7u64, 0]).unwrap();
        let (reference_output, pullback) = differentiate_at((host([32, 32], &lhs), host([16, 32], &rhs)))
            .with_captures(state)
            .vjp(|(lhs, rhs), state| reference_function.call(lhs, rhs, None, None, None, Some(state)))
            .unwrap();
        let (reference_lhs, reference_rhs) = pullback.apply_materialized(host([32, 16], &seed)).unwrap();
        for (actual, expected) in
            [(read(&outputs.0), reference_output), (read(&outputs.1), reference_lhs), (read(&outputs.2), reference_rhs)]
        {
            let expected = expected.to_f64s();
            assert_eq!(actual.len(), expected.len());
            let magnitude = expected.iter().fold(0.0f64, |maximum, &value| maximum.max(f64::abs(value)));
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!(
                    (f64::from(actual) - expected).abs() <= 1e-5 * magnitude,
                    "expected {expected} but got {actual}",
                );
            }
        }
    }
}
