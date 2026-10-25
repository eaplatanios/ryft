use ryft_core::{
    ArrayIrType, ArrayType, ConcatenateOperation, DataType, Dimension, DimensionAddOperation, DimensionSizeOperation,
    DimensionType, EffectClass, Operation, ProgramError, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::types::{
    lower_constant_output, lower_physical_bound_value, lower_restore_dynamic_dimensions, lower_static_index_constants,
    physical_bound_type, reshape_dimension_i32, reshape_dimension_i64,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_CONCATENATE_KIND, ASSERT_DETAIL_ATTRIBUTE, ASSERT_KIND_ATTRIBUTE,
    lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::operations::lower_tensor_type;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Joins inputs using a bounded fan-in tree so very wide concatenations do not create pathological compiler work.
pub(crate) fn lower_concatenate_tree<'b, 'c: 'b, 't: 'c>(
    inputs: &[ValueRef<'b, 'c, 't>],
    axis: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    }
    let mut values = inputs.to_vec();
    while values.len() > 1 {
        values = values
            .chunks(16)
            .map(|chunk| {
                if chunk.len() == 1 {
                    return Ok(chunk[0]);
                }
                let joined = block.append_operation(stable_hlo::concatenate(chunk, axis, location)?)?;
                Ok(joined.result(0).unwrap().as_ref())
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
    }
    Ok(values[0])
}

/// Lowers the dynamic explicit-result-extent check owned by array IR concatenation.
pub(crate) fn lower_concatenate_extent_assertion<'b, 'c: 'b, 't: 'c>(
    axis: usize,
    actual: ValueRef<'b, 'c, 't>,
    input_extents: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let predicate = lower_deliberately_false_assertion_predicate(actual, block, context, location)?;
    let axis = axis.to_string();
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute("concatenate")),
        context.named_attribute(
            context.identifier(ASSERT_KIND_ATTRIBUTE),
            context.string_attribute(ASSERT_CONCATENATE_KIND),
        ),
        context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(axis.as_str())),
    ]);
    let observed = std::iter::once(actual).chain(input_extents.iter().copied()).collect::<Vec<_>>();
    lower_assertion_custom_call(predicate, observed.as_slice(), backend_config, effect_tokens, block, context, location)
}

impl LowerableOperation for ConcatenateOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Ok(vec![lower_concatenate_tree(input_values, self.axis(), &mut lowerer.block, lowerer.location)?])
    }
}

impl LowerableOperation for ConcatenateOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        let Some((result_extent, array_inputs)) = input_values.split_last() else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }.into());
        };
        let Some((result_extent_type, array_input_types)) = input_types.split_last() else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
        };
        if array_inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
        }
        <&DimensionType>::try_from(result_extent_type).map_err(|error| LoweringError::Tracing(error.into()))?;

        if self.effects().classes().contains(EffectClass::OrderedAssertion) {
            // The callback receives every concrete logical input extent and computes their checked sum on the
            // host. This avoids both overflow in a speculative StableHLO sum and false rejection from conservative
            // declared maxima. A type-derived proof omits this entire assertion path for static signatures.
            let i64_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
            let mut input_extents = Vec::with_capacity(array_inputs.len());
            for (input, r#type) in array_inputs.iter().zip(array_input_types) {
                let r#type = <&ArrayType>::try_from(r#type).map_err(|error| LoweringError::Tracing(error.into()))?;
                let extent = match r#type.shape().dimensions()[self.axis()] {
                    Dimension::Static(extent) => lower_static_index_constants(&[extent], block, context, location)?[0],
                    Dimension::Dynamic(_) => {
                        let extent =
                            block.append_operation(stable_hlo::get_dimension_size(*input, self.axis(), location)?)?;
                        let extent =
                            extent.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
                        let extent = block.append_operation(stable_hlo::convert(extent, i64_type, location)?)?;
                        extent.result(0).expect("stablehlo.convert should return one result").as_ref()
                    }
                };
                input_extents.push(extent);
            }
            lower_concatenate_extent_assertion(
                self.axis(),
                *result_extent,
                input_extents.as_slice(),
                effect_tokens,
                block,
                context,
                location,
            )?;
        }

        // The explicit extent may refine the sum's bound or even prove a static result. Native concatenate
        // infers its own bound from its inputs, so reconcile that physical type with the declared result.
        let mut result = lower_concatenate_tree(array_inputs, self.axis(), block, location)?;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let expected_type = lower_tensor_type(output_type, context, location)?;
        if result.r#type()? != expected_type.as_ref() {
            let inputs = array_input_types
                .iter()
                .map(<&ArrayType>::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| LoweringError::Tracing(error.into()))?;
            let mut inferred_extent = DimensionSizeOperation::new(inputs[0], self.axis())
                .map_err(ProgramError::from)?
                .output_type()
                .clone();
            for input in &inputs[1..] {
                let extent = DimensionSizeOperation::new(input, self.axis()).map_err(ProgramError::from)?;
                let sum =
                    DimensionAddOperation::new(&inferred_extent, extent.output_type()).map_err(ProgramError::from)?;
                inferred_extent = DimensionType::new(sum.output_name(), sum.output_bounds());
            }
            let mut inferred_dimensions = output_type.shape().dimensions().to_vec();
            // Native concatenation keeps the axis dynamic even when its bounds prove a singleton extent.
            inferred_dimensions[self.axis()] = Dimension::Dynamic(inferred_extent.to_variable());
            let inferred_type = output_type.clone().with_shape(Shape::new(inferred_dimensions));
            let physical_input_type = physical_bound_type(&inferred_type)?;
            let physical_output_type = physical_bound_type(output_type)?;
            reshape_dimension_i32(physical_output_type.dimension(self.axis()).value().unwrap())?;
            let logical_result = result;
            // Concatenate has already packed logical input extents contiguously. Expose its bounded storage,
            // resize only its trailing padding, then restore the logical dimensions. Reshape cannot change an
            // allocation bound in XLA, even when its runtime element count stays unchanged.
            let padding_value = i64::from(output_type.data_type() == DataType::F8E8M0FNU);
            result =
                lower_physical_bound_value(result, &inferred_type, padding_value as f64, block, context, location)?;
            let mut high_padding = vec![0; output_type.rank()];
            high_padding[self.axis()] =
                reshape_dimension_i64(physical_output_type.dimension(self.axis()).value().unwrap())?
                    - reshape_dimension_i64(physical_input_type.dimension(self.axis()).value().unwrap())?;
            if high_padding[self.axis()] != 0 {
                // Padding is outside the logical extent; choose a representable value even for the exponent-only
                // float format, which has no zero encoding.
                let padding = lower_constant_output(
                    &[ArrayType::scalar(output_type.data_type())],
                    padding_value,
                    block,
                    context,
                    location,
                )?[0];
                let resized = block.append_operation(stable_hlo::pad(
                    result,
                    padding,
                    &vec![0; output_type.rank()],
                    &high_padding,
                    &vec![0; output_type.rank()],
                    location,
                )?)?;
                result = resized.result(0).unwrap().as_ref();
            }
            result = lower_restore_dynamic_dimensions(
                result,
                output_type,
                &(0..output_type.rank()).map(|axis| (logical_result, axis)).collect::<Vec<_>>(),
                block,
                context,
                location,
            )?;
        }
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, ArrayType, Concatenate, ConcatenateOperation, ConstantOperation, DataType, Device, DeviceMesh,
        Dimension, DimensionAddOperation, DimensionBounds, DimensionOperation, DimensionSizeOperation, DimensionType,
        DimensionValue, DimensionVariable, EffectClasses, LogicalMesh, MeshAxis, MeshAxisType, Placeholder, Shape,
        Sharding, ShardingDimension, TypeError, Typed,
    };
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options, test_matrix_type, test_vector_type,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::lowering::types::physical_bound_type;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{CompiledXlaFunction, FromPjrt, ToPjrt, XlaArray, XlaSession, compile, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_concatenate() {
        // A static-shaped concatenate along axis 0 lowers to a single `stablehlo.concatenate` joining the operands.
        let first_type = test_matrix_type(1, 2);
        let second_type = test_matrix_type(3, 2);
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(first_type);
        let second = builder.add_input(second_type);
        let joined = builder
            .add_instruction(
                ConcatenateOperation::<ArrayType>::new(0, 2).unwrap(),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, XlaArrayConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<4x2xf32> {
                    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1x2xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // Dynamic concatenation requires its derived result extent as an explicit operand rather than manufacturing
        // an unstable result identity.
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(dynamic_type.clone());
        let second = builder.add_input(dynamic_type);
        assert_eq!(
            builder.add_instruction(
                ConcatenateOperation::<ArrayType>::new(0, 2).unwrap(),
                Vec::new(),
                vec![first, second],
                None,
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`concatenate` dynamic axis 0 requires an explicit result-dimension input".to_string(),
            ))),
        );

        // A mixed concatenate whose static operand types prove its explicit result extent is pure and lowers without
        // assertion callback or dimension-size machinery.
        let first_type = test_matrix_type(1, 2);
        let second_type = test_matrix_type(3, 2);
        let result_extent = DimensionValue::constant(4).unwrap();
        let operation = ConcatenateOperation::<ArrayIrType>::new(
            0,
            &[first_type.clone().into(), second_type.clone().into(), result_extent.r#type().into_owned().into()],
        )
        .unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let first = builder.add_input(first_type.clone().into());
        let second = builder.add_input(second_type.clone().into());
        let result_extent = builder
            .add_instruction(
                DimensionOperation::from(ConstantOperation::new(result_extent)),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let joined =
            builder.add_instruction(operation, Vec::new(), vec![first, second, result_extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![joined],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.effects().classes(), EffectClasses::NONE);
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![first_type, second_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 0, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.get_dimension_size").count(), 0, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.concatenate").count(), 1, "{stablehlo}");

        // Once that extent is explicit, lowering reads the concrete logical input sizes, checks their sum on the
        // ordered-assertion chain, and passes only the arrays to StableHLO's concatenate operation.
        let first_variable =
            DimensionVariable::new("first", DimensionBounds::new(0, Some(5)).expect("valid test bounds"));
        let second_variable =
            DimensionVariable::new("second", DimensionBounds::new(0, Some(5)).expect("valid test bounds"));
        let first_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(first_variable), Dimension::Static(2)]));
        let second_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(second_variable), Dimension::Static(2)]));
        let first_size_operation = DimensionSizeOperation::new(&first_type, 0).unwrap();
        let second_size_operation = DimensionSizeOperation::new(&second_type, 0).unwrap();
        let add_operation =
            DimensionAddOperation::new(first_size_operation.output_type(), second_size_operation.output_type())
                .unwrap();
        let result_extent_type = DimensionType::new(add_operation.output_name(), add_operation.output_bounds());
        let concatenate_operation = ConcatenateOperation::<ArrayIrType>::new(
            0,
            &[first_type.clone().into(), second_type.clone().into(), result_extent_type.into()],
        )
        .unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let first = builder.add_input(first_type.clone().into());
        let second = builder.add_input(second_type.clone().into());
        let first_size = builder.add_instruction(first_size_operation, Vec::new(), vec![first], None).unwrap()[0];
        let second_size = builder.add_instruction(second_size_operation, Vec::new(), vec![second], None).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![first_size, second_size], None)
            .unwrap()[0];
        let joined = builder
            .add_instruction(concatenate_operation, Vec::new(), vec![first, second, result_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![joined],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![first_type, second_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(stablehlo.matches("stablehlo.get_dimension_size").count(), 5, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 1, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.concatenate"), "{stablehlo}");

        // A dynamic non-concatenated dimension is shared by the output cotangent and every input, so the transpose
        // slices it whole through a dynamic slice limit instead of looking up a hidden runtime size.
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), columns.clone().into()]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), columns.into()]));
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let joined = builder
            .add_instruction(ConcatenateOperation::<ArrayType>::new(0, 2).unwrap(), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, XlaArrayConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(program.transpose().unwrap().output_types(), vec![left_type, right_type]);
    }

    #[test]
    fn test_lower_concatenate_refined_result_extents() {
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        // The actual sum is three, while the input allocation bounds sum to eight. Both tighter and wider declared
        // output bounds must preserve the logical concatenation, including a fully static result refinement.
        for data_type in [DataType::I64, DataType::C64] {
            let element = if data_type == DataType::I64 { "i64" } else { "complex<f32>" };
            for extent in [
                DimensionValue::constant(3).unwrap(),
                DimensionValue::new(DimensionType::new("result", DimensionBounds::new(0, Some(5)).unwrap()), 3)
                    .unwrap(),
                DimensionValue::new(DimensionType::new("result", DimensionBounds::new(0, Some(11)).unwrap()), 3)
                    .unwrap(),
            ] {
                let first_type = ArrayType::new(data_type, Shape::new(vec![dynamic_dimension("first", Some(5))]));
                let second_type = ArrayType::new(data_type, Shape::new(vec![dynamic_dimension("second", Some(5))]));
                let extent_type = extent.r#type().into_owned();
                let operation = ConcatenateOperation::<ArrayIrType>::new(
                    0,
                    &[first_type.clone().into(), second_type.clone().into(), extent_type.into()],
                )
                .unwrap();
                let mut builder = ArrayIrXlaProgramBuilder::new();
                let first = builder.add_input(first_type.clone().into());
                let second = builder.add_input(second_type.clone().into());
                let extent = builder
                    .add_instruction(
                        DimensionOperation::from(ConstantOperation::new(extent)),
                        Vec::new(),
                        Vec::new(),
                        None,
                    )
                    .unwrap()[0];
                let result =
                    builder.add_instruction(operation, Vec::new(), vec![first, second, extent], None).unwrap()[0];
                let program = builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap();
                let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
                let module = to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![first_type, second_type],
                    &vec![output_type.clone()],
                    "concatenate_dynamic",
                    None,
                    None,
                )
                .unwrap();
                let bound = physical_bound_type(&output_type).unwrap().dimension(0).value().unwrap();
                // Inputs carry static storage plus hidden i32 extents; dynamic outputs also return hidden i64 extents.
                let storage_type = format!("tensor<{bound}x{element}>");
                let body = if output_type.static_shape().is_some() {
                    format!(
                        indoc! {r#"
                        %result = func.call @concatenate_dynamic(%left, %right, %left_size, %right_size)
                            : (tensor<4x{element}>, tensor<4x{element}>, tensor<i32>, tensor<i32>) -> {storage_type}
                        return %result : {storage_type}
                    "#},
                        element = element,
                        storage_type = storage_type,
                    )
                } else {
                    let result_type = format!("tensor<?x{element}, #stablehlo.bounds<{bound}>>");
                    format!(
                        indoc! {r#"
                        %result, %extent = func.call @concatenate_dynamic(%left, %right, %left_size, %right_size)
                            : (tensor<4x{element}>, tensor<4x{element}>, tensor<i32>, tensor<i32>)
                                -> ({result_type}, tensor<i64>)
                        %bound = stablehlo.constant dense<{bound}> : tensor<i32>
                        %storage = stablehlo.set_dimension_size %result, %bound, dim = 0
                            : ({result_type}, tensor<i32>) -> {storage_type}
                        return %storage : {storage_type}
                    "#},
                        element = element,
                        result_type = result_type,
                        bound = bound,
                        storage_type = storage_type,
                    )
                };
                let prefix = module
                    .strip_suffix("}\n")
                    .unwrap()
                    .replace("func.func @concatenate_dynamic", "func.func private @concatenate_dynamic");
                let module = format!(
                    indoc! {r#"
                    {prefix}
                    func.func @main(%left: tensor<4x{element}>, %right: tensor<4x{element}>,
                                    %left_size: tensor<i32>, %right_size: tensor<i32>) -> {storage_type} {{
                      {body}
                    }}
                    }}
                "#},
                    prefix = prefix,
                    element = element,
                    storage_type = storage_type,
                    body = body,
                );
                let executable = client
                    .compile(
                        &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                        &ragged_dot_cpu_compilation_options(),
                    )
                    .unwrap();
                let device = executable.addressable_devices().unwrap().remove(0);
                // Decode complex fixture bytes as opaque words so assertions compare exact signed-zero and imaginary
                // bits.
                let left = if data_type == DataType::I64 {
                    vec![i64::MAX, 13, 14, 15]
                } else {
                    values_from_bytes::<i64>(&values_to_bytes(&[1_f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0]))
                };
                let right = if data_type == DataType::I64 {
                    vec![i64::MIN, 9_007_199_254_740_993, 17, 18]
                } else {
                    values_from_bytes::<i64>(&values_to_bytes(&[-0_f32, 0.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0]))
                };
                for (left_size, right_size) in [(1_i32, 2_i32), (0, 3), (2, 2)] {
                    let inputs = vec![
                        ExecutionInput {
                            buffer: Arc::new(
                                client
                                    .buffer(
                                        values_to_bytes(&left).as_slice(),
                                        data_type.to_pjrt(),
                                        [4],
                                        None,
                                        device.clone(),
                                        None,
                                    )
                                    .unwrap(),
                            ),
                            donatable: false,
                        },
                        ExecutionInput {
                            buffer: Arc::new(
                                client
                                    .buffer(
                                        values_to_bytes(&right).as_slice(),
                                        data_type.to_pjrt(),
                                        [4],
                                        None,
                                        device.clone(),
                                        None,
                                    )
                                    .unwrap(),
                            ),
                            donatable: false,
                        },
                        ExecutionInput {
                            buffer: Arc::new(
                                client
                                    .buffer(
                                        values_to_bytes(&[left_size]).as_slice(),
                                        BufferType::I32,
                                        [],
                                        None,
                                        device.clone(),
                                        None,
                                    )
                                    .unwrap(),
                            ),
                            donatable: false,
                        },
                        ExecutionInput {
                            buffer: Arc::new(
                                client
                                    .buffer(
                                        values_to_bytes(&[right_size]).as_slice(),
                                        BufferType::I32,
                                        [],
                                        None,
                                        device.clone(),
                                        None,
                                    )
                                    .unwrap(),
                            ),
                            donatable: false,
                        },
                    ];
                    let execution = executable
                        .execute(
                            vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                            Vec::new(),
                            0,
                            None,
                            Some(file!()),
                            None,
                            None,
                        )
                        .and_then(|execution| execution.block_until_ready());
                    if left_size + right_size != 3 {
                        assert!(matches!(execution, Err(error)
                            if error.to_string().contains("expected 4 but got 3")));
                        continue;
                    }
                    let execution = execution.unwrap().remove(0);
                    let observed = execution.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                    let expected = left[..left_size as usize]
                        .iter()
                        .chain(&right[..right_size as usize])
                        .copied()
                        .collect::<Vec<_>>();
                    assert_eq!(&values_from_bytes::<i64>(observed.as_slice())[..3], expected.as_slice());
                }
            }
        }
    }

    #[test]
    fn test_lower_concatenate_tree() {
        let mut builder = XlaProgramBuilder::new();
        let inputs = (0..33).map(|_| builder.add_input(test_vector_type(1))).collect::<Vec<_>>();
        let joined = builder
            .add_instruction(ConcatenateOperation::<ArrayType>::new(0, 1).unwrap(), Vec::new(), inputs, None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, XlaArrayConstant>(vec![joined], vec![Placeholder; 33], Placeholder)
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.concatenate").count(), 3);
        assert!(module.contains("-> tensor<33xf32>"), "{module}");
    }

    #[test]
    fn test_lower_concatenate_explicit_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::I64, vec![4]).with_sharding(sharding.clone()).unwrap();
        let domain = XlaSession::new(&client).domain();
        let compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), ArrayType> = compile(
            |(first, second)| first.concatenate_with([&second], 0).unwrap(),
            (input_type.clone(), input_type.clone()),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let first =
            XlaArray::from_host_buffer(&domain, input_type.clone(), mesh.clone(), values_to_bytes(&[1_i64, 2, 3, 4]))
                .unwrap();
        let second = XlaArray::from_host_buffer(&domain, input_type, mesh, values_to_bytes(&[5_i64, 6, 7, 8])).unwrap();
        let output = domain.interpret(&compiled.executable_function(), (first, second)).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&sharding));
        for (index, device) in devices.iter().enumerate() {
            let bytes = output
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let expected = if index == 0 { vec![1_i64, 2, 3, 4] } else { vec![5_i64, 6, 7, 8] };
            assert_eq!(values_from_bytes::<i64>(bytes.as_slice()), expected);
        }
    }
}
