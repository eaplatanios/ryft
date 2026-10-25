use std::collections::HashMap;

use ryft_core::macros::check_count;
use ryft_core::operations::dot::{lhs_result_axes, rhs_result_axes};
use ryft_core::operations::quantization::scaled_dot_ir_composition;
use ryft_core::{
    ArrayType, DataType, Dimension, DomainTracingContext, Operation, ProgramError, ScaledDotOperation, Shape,
};
use ryft_mlir::dialects::stable_hlo;
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
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaDomain, XlaLowerer};

/// Traces the canonical scaled-dot decomposition for one validated operation boundary.
pub(crate) fn trace_scaled_dot_composition(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = named_composition_input_types(&scaled_dot_composite_input_types(operation, input_types)?)?;
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let [lhs, rhs, lhs_scale, rhs_scale] = inputs.as_slice() else {
                return Err(ProgramError::InvalidInputCount { expected: 4, actual: inputs.len() });
            };
            Ok(vec![
                scaled_dot_ir_composition(
                    lhs,
                    rhs,
                    operation.has_lhs_scale().then_some(lhs_scale),
                    operation.has_rhs_scale().then_some(rhs_scale),
                    operation.dimensions(),
                    operation.preferred_element_type(),
                )?
                .into_value(),
            ])
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
) -> Result<Vec<ArrayType>, LoweringError> {
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
) -> Result<bool, LoweringError> {
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

/// Returns the typed scaled-dot composition boundary used for the target platform. XLA's CUDA block-scaled-dot
/// replacement cannot propagate dynamic-dimension annotations through its fused HLO, so eligible CUDA calls use the
/// bounded static carriers while the caller masks physical suffix lanes and restores the logical result dimensions.
/// Other calls retain the logical boundary consumed by the portable decomposition.
pub(crate) fn scaled_dot_composition_types(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<(Vec<ArrayType>, Vec<ArrayType>), LoweringError> {
    if !scaled_dot_uses_cuda_physical_boundary(operation, input_types, target_platform)? {
        return Ok((input_types.to_vec(), output_types.to_vec()));
    }
    Ok((
        input_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
        output_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
    ))
}

/// Lowers one scaled dot as the exact four-operand `stablehlo.composite "xla.scaled_dot"` boundary consumed by XLA's
/// block-scaling replacement pass. Either scale omitted by the public Ryft call is represented by the same
/// JAX-compatible identity scale used by the pinned composite contract. The typed decomposition remains the portable
/// fallback when XLA does not replace the composite.
pub(crate) fn lower_scaled_dot_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ScaledDotOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_values, expected_input_count, ProgramError);
    check_count!("input", input_types, input_values.len(), ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let uses_physical_boundary =
        scaled_dot_uses_cuda_physical_boundary(operation, input_types, collective_state.target_platform())?;
    let (composition_input_types, composition_output_types) =
        scaled_dot_composition_types(operation, input_types, output_types, collective_state.target_platform())?;
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
                context.type_attribute(lower_element_type(operation.preferred_element_type(), context)?).as_ref(),
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
        let dimensions = operation.dimensions();
        let output_sources = dimensions
            .lhs_batching_dimensions()
            .iter()
            .copied()
            .map(|axis| (input_values[0], axis))
            .chain(lhs_result_axes(dimensions, input_types[0].rank()).into_iter().map(|axis| (input_values[0], axis)))
            .chain(rhs_result_axes(dimensions, input_types[1].rank()).into_iter().map(|axis| (input_values[1], axis)))
            .collect::<Vec<_>>();
        return Ok(vec![lower_restore_dynamic_dimensions(
            results[0],
            &output_types[0],
            output_sources.as_slice(),
            block,
            context,
            location,
        )?]);
    }

    Err(LoweringError::UnsupportedOp { op: format!("missing typed decomposition for `{}`", operation.name()) })
}

/// Builds the canonical named-composition identity for one scaled-dot boundary.
pub(crate) fn scaled_dot_composition_key(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> Result<NamedCompositionKey, LoweringError> {
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
            ("preferred_element_type", NamedCompositionAttribute::DataType(operation.preferred_element_type())),
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
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
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
    use ryft_core::{AddOperation, DimensionBounds, DimensionVariable, DotDimensionNumbers, Placeholder};

    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::lowering::tests::{XlaProgramBuilder, add_xla_jit_call, unproject_plain_program};
    use crate::experimental::ops::{
        XlaArrayConstant, XlaConstant, XlaProgram, XlaProgramBuilder as ArrayIrXlaProgramBuilder,
    };
    use crate::to_mlir_module_for_program;

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
        // the eventual XLA target decides whether to replace that boundary.
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
                    %0 = stablehlo.composite "xla.scaled_dot" %arg0, %arg1, %arg2, %arg3 {composite_attributes = {dimension_numbers = [[[1], [1]], [[], []]], preferred_element_type = f32}, decomposition = @xla.scaled_dot} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
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
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %arg0, %arg1"));
        assert!(!module.contains("func.call @xla.scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_preserves_scaled_dot_composite_on_cuda() {
        // CUDA lowering preserves the exact named-composite boundary and its private semantic fallback. XLA's
        // block-scaling replacement pass, rather than Ryft's StableHLO emitter, decides whether to select a native
        // implementation for the target device.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("func.func private @xla.scaled_dot"));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %arg0, %arg1, %arg2, %arg3"));
        assert!(!module.contains("stablehlo.custom_call @__op$block_scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_physicalizes_dynamic_scaled_dot_on_cuda() {
        // CUDA's fused scaled-dot HLO requires a static boundary. Ryft masks the bounded physical operands, keeps the
        // named composite and its decomposition static, and restores the logical row extent on the result.
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

        assert!(module.contains(
            "func.func private @xla.scaled_dot(%arg0: tensor<4x32xf8E4M3FN>, %arg1: tensor<1x32xf8E4M3FN>, \
             %arg2: tensor<4x1xf8E8M0FNU>, %arg3: tensor<1x1xf8E8M0FNU>) -> tensor<4x1xf32>",
        ));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\"",));
        assert!(module.contains(
            ": (tensor<4x32xf8E4M3FN>, tensor<1x32xf8E4M3FN>, tensor<4x1xf8E8M0FNU>, tensor<1x1xf8E8M0FNU>) \
             -> tensor<4x1xf32>",
        ));
        assert!(module.contains("stablehlo.set_dimension_size"));
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
                assert_eq!(
                    module.err().unwrap(),
                    LoweringError::UnsupportedOp {
                        op: "reshape with unequal physical capacities (input 128, output 64); dynamic bounds must \
                             preserve the physical element count"
                            .to_string(),
                    },
                );
                continue;
            }
            let module = module.unwrap().stable_hlo;

            assert!(module.contains("call @xla.scaled_dot"));
            assert!(!module.contains("stablehlo.custom_call @ryft.assert"));
            assert!(!module.contains("stablehlo.composite \"xla.scaled_dot\""));
        }
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_scaled_dot_composite_in_shared_jit_call_callee() {
        // A scaled-dot composite nested inside a shared `jit_call` callee is emitted once, while both outer call sites
        // reuse the same function and XLA retains the opportunity to replace the composite after inlining.
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
        assert_eq!(module.matches("stablehlo.composite \"xla.scaled_dot\"").count(), 1);
        assert_eq!(module.matches("call @jit_call_0").count(), 2);
    }
}
