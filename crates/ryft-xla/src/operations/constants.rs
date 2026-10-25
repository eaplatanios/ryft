use ryft_core::macros::check_count;
use ryft_core::{
    Array, ArrayIrType, ArrayType, ConstantOperation, DataType, Dimension, IotaOperation, Layout, OneLikeOperation,
    OneOperation, Operation, ProgramError, Shape, TypeError, ZeroLikeOperation, ZeroOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::MlirLowerableValue;
use crate::experimental::lowering::types::{
    lower_constant_output, lower_tensor_type, stable_hlo_dynamic_dimension_bound,
};
use crate::operations::assertions::lower_output_extent_validation;
use crate::operations::manipulation::memory::annotate_output_memory;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, MemberLowerableOperation, XlaLowerer};

impl LowerableOperation for IotaOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let result = lowerer.block.append_operation(stable_hlo::iota(
            output_tensor_type,
            self.dimension(),
            lowerer.location,
        )?)?;
        let result = result.result(0).expect("stablehlo.iota should return one result").as_ref();
        Ok(vec![annotate_output_memory(
            result,
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

impl LowerableOperation for ConstantOperation<Array> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let constant_value =
            self.value().lower_constant_value(&[], &mut lowerer.block, lowerer.context, lowerer.location)?;
        Ok(vec![constant_value])
    }
}

/// Lowers a zero or one with its exemplar's runtime shape.
pub(crate) fn lower_like_constant<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    integer_value: i64,
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if input_values.len() != 1 {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
    }
    check_count!("output", output_types, 1, ProgramError);
    let data_type = output_types[0].data_type();
    match (integer_value, data_type) {
        (0, DataType::Token | DataType::F8E8M0FNU) => {
            return Err(ProgramError::Type(TypeError::invalid(format!(
                "data type `{data_type}` cannot represent zero",
            )))
            .into());
        }
        (1, DataType::Token | DataType::Zero) => {
            return Err(ProgramError::Type(TypeError::invalid(format!(
                "data type `{data_type}` cannot represent one",
            )))
            .into());
        }
        _ => {}
    }
    let output_type = &output_types[0];
    let result = if output_type.static_shape().is_some() {
        lower_constant_output(output_types, integer_value, block, context, location)?
    } else {
        // Allocation uses the declared bounds; logical sizes come from the exemplar, including zero extents.
        let mut dimensions = Vec::new();
        for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Dynamic(_)) {
                let size = block.append_operation(stable_hlo::get_dimension_size(input_values[0], axis, location)?)?;
                dimensions.push(size.result(0).unwrap().as_ref());
            }
        }
        let name = if integer_value == 0 { "zero_like" } else { "one_like" };
        lower_dynamic_constructor(
            name,
            integer_value,
            &dimensions,
            &[output_type.clone().into()],
            block,
            context,
            location,
        )?
    };
    if let Some(sharding) = output_type.sharding() {
        lower_sharding_constraint(&result, sharding, bound_manual_axes, block, location)
    } else {
        Ok(result)
    }
}

/// Constrains the layout of a statically shaped physical buffer before attaching its runtime dimensions.
pub(crate) fn lower_layout_constraint<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let Some(layout) = r#type.layout() else {
        return Ok(value);
    };
    let Layout::Tiled(tiled) = layout else {
        return Err(LoweringError::UnsupportedOp { op: format!("strided output layout `{layout}`") });
    };
    if !tiled.tiles().is_empty() {
        return Err(LoweringError::UnsupportedOp { op: format!("tiled output layout `{layout}`") });
    }
    let axes = tiled.minor_to_major();
    if axes.len() != r#type.rank()
        || axes.iter().any(|axis| *axis >= r#type.rank())
        || axes.iter().collect::<std::collections::HashSet<_>>().len() != r#type.rank()
    {
        return Err(LoweringError::UnsupportedOp { op: format!("invalid output layout `{layout}` for type `{type}`") });
    }
    // XLA recognizes this custom call as a physical layout constraint. Apply it to the allocation's static shape:
    // runtime extents are attached afterward, so the call needs no separate dynamic result-shape arguments.
    let constrained = block.append_operation(stable_hlo::custom_call(
        &[value],
        "LayoutConstraint",
        false,
        None,
        stable_hlo::CustomCallApiVersion::Original,
        &[],
        Some(stable_hlo::CustomCallMemoryLayouts { operands: vec![axes.to_vec()], results: vec![axes.to_vec()] }),
        &[],
        None,
        &[value.r#type()?],
        location,
    )?)?;
    Ok(constrained.result(0).unwrap().as_ref())
}

/// Derives the declared and statically bounded physical types shared by dynamic array constructors.
pub(crate) fn dynamic_constructor_types(
    name: &str,
    input_count: usize,
    output_types: &[ArrayIrType],
) -> Result<(ArrayType, ArrayType), LoweringError> {
    let [output_type] = output_types else {
        return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
    };
    let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
    let dynamic_count = output_type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count();
    if input_count != dynamic_count {
        return Err(ProgramError::InvalidInputCount { expected: dynamic_count, actual: input_count }.into());
    }

    let physical_dimensions = output_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                let bound = stable_hlo_dynamic_dimension_bound(dimension).ok_or_else(|| LoweringError::UnsupportedOp {
                    op: format!(
                        "{name} output dimension `{variable}` needs a finite upper bound for physical buffer allocation",
                    ),
                })?;
                i32::try_from(bound).map_err(|_| LoweringError::DynamicDimensionOutOfRange { value: bound })?;
                Ok(Dimension::Static(bound))
            }
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let physical_type = output_type.clone().with_shape(Shape::new(physical_dimensions));
    Ok((output_type.clone(), physical_type))
}

/// Refines one statically bounded constructor result from its compact first-class dimension operands.
pub(crate) fn refine_dynamic_constructor_result<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    mut refined_type: ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    result = lower_layout_constraint(result, &refined_type, block, location)?;
    let i32_scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    // The dimension operands are compact (one per dynamic axis, in axis order), so advance the operand iterator only
    // when a dynamic axis is encountered rather than indexing by the physical axis number.
    let mut operands = input_values.iter();
    for (axis, dimension) in output_type.shape().dimensions().iter().cloned().enumerate() {
        if !matches!(dimension, Dimension::Dynamic(_)) {
            continue;
        }
        let operand = *operands.next().unwrap();
        let extent = block.append_operation(stable_hlo::convert(operand, i32_scalar_type, location)?)?;
        let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
        let mut dimensions = refined_type.shape().dimensions().to_vec();
        dimensions[axis] = dimension;
        refined_type = refined_type.with_shape(Shape::new(dimensions));
        let refined_tensor_type = lower_tensor_type(&refined_type, context, location)?;
        let operation = block.append_operation(stable_hlo::set_dimension_size(
            result,
            extent,
            refined_tensor_type,
            axis,
            location,
        )?)?;
        result = operation.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(vec![result])
}

/// Selects the compact first-class dimension operands of a dynamic array constructor whose output axes remain
/// dynamic. The constructor consumes one operand per dynamic axis of its declared payload type, but type inference
/// refines every axis of a dimension identity to a static extent when one of its operands has that exact extent (e.g.,
/// an eager dimension value). This function omits the operands of refined axes after validating them: it emits
/// runtime assertions, through [`lower_output_extent_validation`], that every symbolic operand of a refined axis
/// equals the refined extent.
pub(crate) fn dynamic_constructor_inputs<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    name: &str,
    declared_type: &ArrayType,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let [output_type] = output_types else {
        return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
    };
    let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
    let declared_dimensions = declared_type.shape().dimensions();
    let dynamic_count =
        declared_dimensions.iter().filter(|dimension| matches!(dimension, Dimension::Dynamic(_))).count();
    if input_values.len() != dynamic_count {
        return Err(ProgramError::InvalidInputCount { expected: dynamic_count, actual: input_values.len() }.into());
    }
    if declared_dimensions.len() != output_type.rank() {
        return Err(ProgramError::MalformedProgram(format!(
            "`{name}` output type `{output_type}` does not have the rank of its declared type `{declared_type}`",
        ))
        .into());
    }
    lower_output_extent_validation(
        name,
        std::slice::from_ref(declared_type),
        input_values,
        input_types,
        std::slice::from_ref(output_type),
        &mut lowerer.effect_tokens,
        &mut lowerer.block,
        lowerer.context,
        lowerer.location,
    )?;
    let mut inputs = input_values.iter();
    Ok(declared_dimensions
        .iter()
        .zip(output_type.shape().dimensions())
        .filter_map(|(declared, output)| match declared {
            Dimension::Static(_) => None,
            Dimension::Dynamic(_) => {
                let input = *inputs.next().unwrap();
                matches!(output, Dimension::Dynamic(_)).then_some(input)
            }
        })
        .collect())
}

/// Lowers one bounded dynamic integer-valued array constructor from its compact first-class dimension operands.
pub(crate) fn lower_dynamic_constructor<'b, 'c: 'b, 't: 'c>(
    name: &str,
    integer_value: i64,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayIrType],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (output_type, physical_type) = dynamic_constructor_types(name, input_values.len(), output_types)?;
    let result =
        lower_constant_output(std::slice::from_ref(&physical_type), integer_value, block, context, location)?.remove(0);
    refine_dynamic_constructor_result(result, input_values, &output_type, physical_type, block, context, location)
}

impl LowerableOperation for ZeroOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if !input_values.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
        }
        lower_constant_output(output_types, 0, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

impl LowerableOperation for OneOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if !input_values.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
        }
        lower_constant_output(output_types, 1, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

impl LowerableOperation for ZeroLikeOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_like_constant(
            input_values,
            output_types,
            0,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl LowerableOperation for OneLikeOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_like_constant(
            input_values,
            output_types,
            1,
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for ZeroOperation<ArrayType> {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let input_values =
            dynamic_constructor_inputs(self.name(), self.r#type(), input_values, input_types, output_types, lowerer)?;
        let block = &mut lowerer.block;
        lower_dynamic_constructor(self.name(), 0, &input_values, output_types, block, context, location)
    }
}

impl MemberLowerableOperation<ArrayIrType> for OneOperation<ArrayType> {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let input_values =
            dynamic_constructor_inputs(self.name(), self.r#type(), input_values, input_types, output_types, lowerer)?;
        let block = &mut lowerer.block;
        lower_dynamic_constructor(self.name(), 1, &input_values, output_types, block, context, location)
    }
}

impl MemberLowerableOperation<ArrayIrType> for IotaOperation<ArrayType> {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let input_values =
            dynamic_constructor_inputs(self.name(), self.r#type(), input_values, input_types, output_types, lowerer)?;
        let block = &mut lowerer.block;
        let (output_type, physical_type) = dynamic_constructor_types(self.name(), input_values.len(), output_types)?;
        let tensor_type = lower_tensor_type(&physical_type, context, location)?;
        let iota = block.append_operation(stable_hlo::iota(tensor_type, self.dimension(), location)?)?;
        let result = iota.result(0).expect("stablehlo.iota should return one result").as_ref();
        refine_dynamic_constructor_result(result, &input_values, &output_type, physical_type, block, context, location)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrOperation, ArrayOperation, Context, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        Fill, LogicalMesh, Memory, MeshAxis, MeshAxisType, OneLikeOperation, OneOperation, Placeholder, ProgramBuilder,
        Sharding, ShardingDimension, TiledLayout, TracingContext, Typed, ZeroLikeOperation, ZeroOperation, i1, i2, i4,
        u1, u2, u4,
    };
    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;
    use ryft_mlir::{ElementsAttribute, Size as MlirSize};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use crate::experimental::lowering::tests::{
        MixedValue, XlaProgramBuilder, dynamic_dimension, execute_mixed_program, ragged_dot_cpu_compilation_options,
        test_vector_type,
    };
    use crate::experimental::lowering::{MlirLowerableValue, to_mlir_module_for_plain_program};
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::to_mlir_module_for_program;

    use super::*;

    #[cfg(test)]
    use ryft_core::Complex as ComplexNumber;

    /// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
    fn test_literal_dense_bytes(literal: &Array, byte_count: usize) -> Vec<u8> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
        let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
        unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
    }

    #[test]
    fn test_zero_space_type_and_constant_lower_to_a_false_i1_carrier() {
        let output_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let output = builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("func.func @main() -> tensor<3xi1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<false> : tensor<i1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.broadcast_in_dim"), "{stablehlo}");
    }

    #[test]
    fn test_lower_dynamic_constructor_singleton_dimension() {
        // A symbolic dimension whose bounds admit a single extent is an input identity rather than a static extent
        // (which a dimension constant would supply instead), so the constructors keep it dynamic.
        let size = DimensionVariable::new("size", DimensionBounds::new(3, Some(4)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let dimension = builder.add_input(DimensionType::from(size).into());
        let zero = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::Zero(ZeroOperation::new(output_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let one = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::One(OneOperation::new(output_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let iota = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::Iota(IotaOperation::new(output_type.clone(), 0).unwrap()),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![zero, one, iota], vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone()); 3]);
        let input_types = vec![ArrayType::scalar(DataType::I64)];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &vec![output_type; 3], "main", None, None).unwrap();
        // Even a singleton dimension retains its declared identity and logical extent in the lowered signature.
        assert_eq!(module.matches("stablehlo.broadcast_in_dim").count(), 2, "{module}");
        assert_eq!(module.matches("stablehlo.iota").count(), 1, "{module}");
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 3, "{module}");
        // Each dynamic output is followed by its trailing hidden runtime extent.
        assert_eq!(
            execute_mixed_program(&execution_client(), &program, &[MixedValue::Dimension(3)], &[]),
            Ok(vec![
                MixedValue::Array(vec![0.0, 0.0, 0.0], vec![3]),
                MixedValue::Array(vec![1.0, 1.0, 1.0], vec![3]),
                MixedValue::Array(vec![0.0, 1.0, 2.0], vec![3]),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
            ]),
        );
    }

    #[test]
    fn test_lower_dynamic_constructor_distinct_symbolic_peers() {
        // Distinct operands that share a symbolic identity are equal by construction because entry refinements
        // validate identities shared by boundary inputs. Both axes therefore stay dynamic without any assertion.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let first = builder.add_input(DimensionType::from(size.clone()).into());
        let second = builder.add_input(DimensionType::from(size).into());
        let output = builder
            .add_instruction(ZeroOperation::new(output_type.clone()), Vec::new(), vec![first, second], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64); 2],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<?x?xf64, #stablehlo.bounds<8, 8>>, tensor<i64>, tensor<i64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
                    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
                    %2 = stablehlo.set_dimension_size %0, %1, dim = 0 : (tensor<8x8xf64>, tensor<i32>) -> tensor<?x8xf64, #stablehlo.bounds<8, ?>>
                    %3 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %4 = stablehlo.set_dimension_size %2, %3, dim = 1 : (tensor<?x8xf64, #stablehlo.bounds<8, ?>>, tensor<i32>) -> tensor<?x?xf64, #stablehlo.bounds<8, 8>>
                    %5 = stablehlo.get_dimension_size %4, dim = 0 : (tensor<?x?xf64, #stablehlo.bounds<8, 8>>) -> tensor<i32>
                    %6 = stablehlo.convert %5 : (tensor<i32>) -> tensor<i64>
                    %7 = stablehlo.get_dimension_size %4, dim = 1 : (tensor<?x?xf64, #stablehlo.bounds<8, 8>>) -> tensor<i32>
                    %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                    return %4, %6, %8 : tensor<?x?xf64, #stablehlo.bounds<8, 8>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Dimension(3), MixedValue::Dimension(3)],
                &[],
            ),
            Ok(vec![MixedValue::Array(vec![0.0; 9], vec![3, 3]), MixedValue::Dimension(3), MixedValue::Dimension(3)]),
        );
    }

    #[test]
    fn test_lower_dynamic_constructor_shared_symbolic_operand() {
        // Reusing one operand for every axis of a symbolic identity needs no assertion either.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(DimensionType::from(size).into());
        let output = builder
            .add_instruction(ZeroOperation::new(output_type.clone()), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64)],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>) -> (tensor<?x?xf64, #stablehlo.bounds<8, 8>>, tensor<i64>, tensor<i64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
                    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
                    %2 = stablehlo.set_dimension_size %0, %1, dim = 0 : (tensor<8x8xf64>, tensor<i32>) -> tensor<?x8xf64, #stablehlo.bounds<8, ?>>
                    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
                    %4 = stablehlo.set_dimension_size %2, %3, dim = 1 : (tensor<?x8xf64, #stablehlo.bounds<8, ?>>, tensor<i32>) -> tensor<?x?xf64, #stablehlo.bounds<8, 8>>
                    %5 = stablehlo.get_dimension_size %4, dim = 0 : (tensor<?x?xf64, #stablehlo.bounds<8, 8>>) -> tensor<i32>
                    %6 = stablehlo.convert %5 : (tensor<i32>) -> tensor<i64>
                    %7 = stablehlo.get_dimension_size %4, dim = 1 : (tensor<?x?xf64, #stablehlo.bounds<8, 8>>) -> tensor<i32>
                    %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                    return %4, %6, %8 : tensor<?x?xf64, #stablehlo.bounds<8, 8>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
        assert_eq!(
            execute_mixed_program(&execution_client(), &program, &[MixedValue::Dimension(3)], &[]),
            Ok(vec![MixedValue::Array(vec![0.0; 9], vec![3, 3]), MixedValue::Dimension(3), MixedValue::Dimension(3)]),
        );
    }

    #[test]
    fn test_lower_dynamic_constructor_exact_extents() {
        // Exact operands refine every axis to their static extent and prove it, so they need no assertion and do not
        // reach the lowered constructor.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let first = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
        let second = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(ZeroOperation::new(output_type), Vec::new(), vec![first, second], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![], vec![Placeholder])
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [3, 3]);
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone())]);
        let module =
            to_mlir_module_for_program(&program, &[], &Vec::<ArrayType>::new(), &vec![output_type], "main", None, None)
                .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<3x3xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                    return %0 : tensor<3x3xf64>
                  }
                }
            "#},
        );
        assert_eq!(
            execute_mixed_program(&execution_client(), &program, &[], &[]),
            Ok(vec![MixedValue::Array(vec![0.0; 9], vec![3, 3])]),
        );
    }

    #[test]
    fn test_lower_dynamic_constructor_mixed_repeated_extents() {
        // An exact operand makes both repeated axes static, but the symbolic peer must still equal that extent. The
        // constructors share the mixed boundary and cover both operand orders.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
        let zero = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::Zero(ZeroOperation::new(output_type.clone())),
                Vec::new(),
                vec![symbolic, exact],
                None,
            )
            .unwrap()[0];
        let one = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::One(OneOperation::new(output_type.clone())),
                Vec::new(),
                vec![exact, symbolic],
                None,
            )
            .unwrap()[0];
        let iota = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::Iota(IotaOperation::new(output_type, 0).unwrap()),
                Vec::new(),
                vec![symbolic, exact],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![zero, one, iota], vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [3, 3]);
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone()); 3]);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64)],
            &vec![output_type; 3],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>) -> (tensor<3x3xf64>, tensor<3x3xf64>, tensor<3x3xf64>) {
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "zero", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                    %c_0 = stablehlo.constant dense<3> : tensor<i64>
                    %4 = stablehlo.compare EQ, %arg0, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %5 = stablehlo.custom_call @ryft.assert(%4, %arg0, %2) {api_version = 4 : i32, backend_config = {actor = "one", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
                    %c_2 = stablehlo.constant dense<3> : tensor<i64>
                    %7 = stablehlo.compare EQ, %arg0, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %8 = stablehlo.custom_call @ryft.assert(%7, %arg0, %5) {api_version = 4 : i32, backend_config = {actor = "iota", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %9 = stablehlo.iota dim = 0 : tensor<3x3xf64>
                    return %3, %6, %9 : tensor<3x3xf64>, tensor<3x3xf64>, tensor<3x3xf64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]),
            Ok(vec![
                MixedValue::Array(vec![0.0; 9], vec![3, 3]),
                MixedValue::Array(vec![1.0; 9], vec![3, 3]),
                MixedValue::Array(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![3, 3]),
            ]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(4)], &[]).unwrap_err().to_string();
        assert!(error.contains("input dimension `size` expected 3"), "{error}");
    }

    #[test]
    fn test_lower_dynamic_constructor_mixed_repeated_extents_shared_symbolic_operand() {
        // A symbolic operand repeated across axes refined to the same extent needs only one assertion.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![size.clone().into(), size.clone().into(), size.clone().into()]),
        );
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(2).unwrap()));
        let output = builder
            .add_instruction(ZeroOperation::new(output_type), Vec::new(), vec![symbolic, symbolic, exact], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2, 2]);
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type.clone())]);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64)],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>) -> tensor<2x2x2xf64> {
                    %c = stablehlo.constant dense<2> : tensor<i64>
                    %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "zero", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 2"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x2x2xf64>
                    return %3 : tensor<2x2x2xf64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(2)], &[]),
            Ok(vec![MixedValue::Array(vec![0.0; 8], vec![2, 2, 2])]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]).unwrap_err().to_string();
        assert!(error.contains("input dimension `size` expected 2"), "{error}");
    }

    #[test]
    fn test_lower_like_constant_executes_with_constructed_exemplar() {
        // The exemplar's runtime extent comes from another dynamic constructor rather than from an entry argument.
        // `zero_like(zero)` also equals the `zero` output, which XLA merges into one bounded dynamic buffer. Each
        // output must nevertheless report its own logical extent.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let dimension = builder.add_input(DimensionType::from(size).into());
        let zero = builder
            .add_instruction(
                ArrayIrOperation::<XlaArrayConstant>::Zero(ZeroOperation::new(output_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let one_like = ArrayIrOperation::<XlaArrayConstant>::from(OneLikeOperation::<ArrayIrType>::new());
        let one_like = builder.add_instruction(one_like, Vec::new(), vec![zero], None).unwrap()[0];
        let zero_like = ArrayIrOperation::<XlaArrayConstant>::from(ZeroLikeOperation::<ArrayIrType>::new());
        let zero_like = builder.add_instruction(zero_like, Vec::new(), vec![zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![zero, one_like, zero_like],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(output_type); 3]);
        for extent in [3, 0] {
            let length = extent as usize;
            assert_eq!(
                execute_mixed_program(&execution_client(), &program, &[MixedValue::Dimension(extent)], &[]),
                Ok(vec![
                    MixedValue::Array(vec![0.0; length], vec![length as u64]),
                    MixedValue::Array(vec![1.0; length], vec![length as u64]),
                    MixedValue::Array(vec![0.0; length], vec![length as u64]),
                    MixedValue::Dimension(extent),
                    MixedValue::Dimension(extent),
                    MixedValue::Dimension(extent),
                ]),
            );
        }
    }

    #[test]
    fn test_lower_like_constant_dynamic_shape() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("size", Some(5))]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![zero, one],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xf32, #stablehlo.bounds<4>>) -> (tensor<?xf32, #stablehlo.bounds<4>>, tensor<?xf32, #stablehlo.bounds<4>>) {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %2 = stablehlo.convert %0 : tensor<i32>
                    %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<4>>
                    %4 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<i32>
                    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %6 = stablehlo.convert %4 : tensor<i32>
                    %7 = stablehlo.set_dimension_size %5, %6, dim = 0 : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<4>>
                    return %3, %7 : tensor<?xf32, #stablehlo.bounds<4>>, tensor<?xf32, #stablehlo.bounds<4>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_like_constant_dynamic_metadata() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::C128, Shape::new(vec![dynamic_dimension("size", Some(5))]))
            .with_memory(Memory::Host { pinned: true })
            .with_layout(Some(TiledLayout::new(vec![0], Vec::new()).into()))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![zero, one],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(program.output_types(), vec![input_type.clone(), input_type]);
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(module.matches("stablehlo.get_dimension_size").count(), 2, "{module}");
        assert_eq!(module.matches("stablehlo.set_dimension_size").count(), 2, "{module}");
        assert_eq!(module.matches("_xla_buffer_placement = \"pinned_host\"").count(), 2, "{module}");
        assert_eq!(module.matches("sdy.sharding_constraint").count(), 2, "{module}");
    }

    #[test]
    fn test_lower_like_constant_dynamic_shape_requires_supported_bound() {
        for (operation, name) in [
            (ArrayOperation::ZeroLike(ZeroLikeOperation::new()), "zero_like"),
            (ArrayOperation::OneLike(OneLikeOperation::new()), "one_like"),
        ] {
            let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("size", None)]));
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(
                to_mlir_module_for_plain_program(&program, "main"),
                Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "{name} output dimension `size` needs a finite upper bound for physical buffer allocation",
                    ),
                }),
            );
        }

        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("size", Some(i32::MAX as usize + 2))]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::DynamicDimensionOutOfRange { value: i32::MAX as usize + 1 }),
        );
    }

    #[test]
    fn test_lower_like_constant_executes_dynamic_shape() {
        let input_type = ArrayType::new(DataType::C64, Shape::new(vec![dynamic_dimension("size", Some(5))]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], Vec::new())));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![zero, one],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "like_dynamic").unwrap();
        assert_eq!(module.matches("stablehlo.custom_call @LayoutConstraint").count(), 2, "{module}");
        // The wrapper supplies an input extent at execution time. Its static physical boundary avoids CPU PJRT's
        // unsupported dynamic buffer boundary; one compiled executable handles both size two and size zero.
        let module = format!(
            "{}{}",
            module
                .strip_suffix("}\n")
                .unwrap()
                .replace("func.func @like_dynamic", "func.func private @like_dynamic"),
            indoc! {r#"
              func.func @main(%arg0: tensor<4xcomplex<f32>>, %arg1: tensor<i32>) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<i32>, tensor<i32>) {
                %input = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<4xcomplex<f32>>, tensor<i32>) -> tensor<?xcomplex<f32>, #stablehlo.bounds<4>>
                %zero, %one = func.call @like_dynamic(%input) : (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>) -> (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>, tensor<?xcomplex<f32>, #stablehlo.bounds<4>>)
                %zero_size = stablehlo.get_dimension_size %zero, dim = 0 : (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>) -> tensor<i32>
                %one_size = stablehlo.get_dimension_size %one, dim = 0 : (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>) -> tensor<i32>
                %bound = stablehlo.constant dense<4> : tensor<i32>
                %zero_storage = stablehlo.set_dimension_size %zero, %bound, dim = 0 : (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>, tensor<i32>) -> tensor<4xcomplex<f32>>
                %one_storage = stablehlo.set_dimension_size %one, %bound, dim = 0 : (tensor<?xcomplex<f32>, #stablehlo.bounds<4>>, tensor<i32>) -> tensor<4xcomplex<f32>>
                return %zero_storage, %one_storage, %zero_size, %one_size : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<i32>, tensor<i32>
              }
            }
            "#},
        );
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        for extent in [2_i32, 0] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&[3_f32, 1.0, 5.0, 2.0, 7.0, 3.0, 9.0, 4.0]).as_slice(),
                                BufferType::C64,
                                &[4],
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
                                values_to_bytes(&[extent]).as_slice(),
                                BufferType::I32,
                                &[],
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
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0);
            let outputs = execution
                .outputs
                .iter()
                .map(|output| output.copy_to_host(None).unwrap().r#await().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                &values_from_bytes::<f32>(outputs[0].as_slice())[..2 * extent as usize],
                vec![0.0; 2 * extent as usize],
            );
            assert_eq!(
                &values_from_bytes::<f32>(outputs[1].as_slice())[..2 * extent as usize],
                [1.0, 0.0].repeat(extent as usize),
            );
            assert_eq!(values_from_bytes::<i32>(outputs[2].as_slice()), vec![extent]);
            assert_eq!(values_from_bytes::<i32>(outputs[3].as_slice()), vec![extent]);
        }
    }

    #[test]
    fn test_zero_space_one_like_is_rejected() {
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        // Unsupported identities are rejected during inference before invalid IR can reach lowering.
        assert_eq!(
            builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![input], None),
            Err(ProgramError::Type(TypeError::invalid("data type `zero` cannot represent one"))),
        );
    }

    #[test]
    fn test_f8e8m0fnu_zero_like_is_rejected() {
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        // Unsupported identities are rejected during inference before invalid IR can reach lowering.
        assert_eq!(
            builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None),
            Err(ProgramError::Type(TypeError::invalid("data type `f8e8m0fnu` cannot represent zero"))),
        );
    }

    #[test]
    fn test_literal_constant_operation_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let value =
                Array::from_elements::<f32>(test_vector_type(4).with_memory(memory), &[1.0, 2.0, 3.0, 4.0]).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let output = builder
                .add_instruction(ArrayOperation::Constant(ConstantOperation::new(value)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_fill_lowers_as_scalar_constant_plus_broadcast() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let output_type = test_vector_type(4).with_memory(memory);
            let output = context.fill(&output_type, 2.5f64).unwrap();
            assert_eq!(*output.r#type(), output_type);
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
                .unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
            assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 1, "{stablehlo}");
            assert!(stablehlo.contains("stablehlo.constant dense<2.500000e+00> : tensor<f32>"), "{stablehlo}");
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }

            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let output_type = ArrayType::scalar(DataType::F32).with_memory(memory);
            let output = context.fill(&output_type, 2.5f64).unwrap();
            assert_eq!(*output.r#type(), output_type);
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
                .unwrap();
            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
            assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_rank_positive_literal_constant_preserves_signed_zero_elements() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let literal = Array::from_elements(test_vector_type(2), &[-0.0_f32, 0.0]).unwrap();
        let output = context.bind(ConstantOperation::new(literal), Vec::new(), &[]).unwrap().remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
        assert!(stablehlo.contains("dense<[-0.000000e+00, 0.000000e+00]>"), "{stablehlo}");

        // A rank-positive one-element literal remains an ordinary shaped dense constant rather than being rewritten
        // as a rank-zero constant plus broadcast.
        let literal = Array::from_elements(test_vector_type(1), &[2.5_f32]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output = builder
            .add_instruction(ArrayOperation::Constant(ConstantOperation::new(literal)), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<2.500000e+00> : tensor<1xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_rank_positive_literal_dense_attributes_preserve_exact_payloads() {
        // Boolean values retain their logical order through MLIR's bit-packed dense representation.
        let boolean = Array::vector(vec![true, false, true]).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(boolean.r#type().as_ref(), &context, location).unwrap();
        let attribute = boolean.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(
            unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() },
            vec![true, false, true],
        );

        // One-bit integers use the same packed MLIR representation as Boolean while retaining their Ryft data type.
        for (literal, expected) in [
            (
                Array::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
            (
                Array::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
        ] {
            let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
            let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
            assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, expected);
        }

        // Wider sub-byte integers occupy one MLIR raw-buffer byte per element and preserve only their declared bits.
        for (literal, expected) in [
            (Array::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
            (Array::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
            (Array::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
            (Array::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
        ] {
            assert_eq!(test_literal_dense_bytes(&literal, expected.len()), expected);
        }

        // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
        // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
        let integer_cases = vec![
            (Array::vector(vec![-127_i8, 126]).unwrap(), values_to_bytes(&[-127_i8, 126])),
            (Array::vector(vec![-0x1234_i16, 0x2345]).unwrap(), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (
                Array::vector(vec![-0x1234_567_i32, 0x2345_678]).unwrap(),
                values_to_bytes(&[-0x1234_567_i32, 0x2345_678]),
            ),
            (
                Array::vector(vec![-0x1234_5678_9abc_def_i64, 0x2345_6789_abcd_ef0]).unwrap(),
                values_to_bytes(&[-0x1234_5678_9abc_def_i64, 0x2345_6789_abcd_ef0]),
            ),
            (Array::vector(vec![0x12_u8, 0xfe]).unwrap(), values_to_bytes(&[0x12_u8, 0xfe])),
            (Array::vector(vec![0x1234_u16, 0xfedc]).unwrap(), values_to_bytes(&[0x1234_u16, 0xfedc])),
            (
                Array::vector(vec![0x1234_5678_u32, 0xfedc_ba98]).unwrap(),
                values_to_bytes(&[0x1234_5678_u32, 0xfedc_ba98]),
            ),
            (
                Array::vector(vec![(1_u64 << 53) + 1, u64::MAX - 1]).unwrap(),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
        ];
        for (literal, expected) in integer_cases {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "integer literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Sub-byte and eight-bit floating-point formats retain their exact encodings, including NaN payloads.
        for (data_type, bits, rendered_values) in [
            (DataType::F4E2M1FN, [0x01, 0x0f], "dense<[5.000000e-01, -6.000000e+00]>"),
            (DataType::F6E2M3FN, [0x01, 0x3f], "dense<[1.250000e-01, -7.500000e+00]>"),
            (DataType::F6E3M2FN, [0x02, 0x3e], "dense<[1.250000e-01, -2.400000e+01]>"),
            (DataType::F8E3M4, [0x01, 0xff], "dense<[1.562500e-02, 0xFF]>"),
            (DataType::F8E4M3, [0x02, 0xfe], "dense<[3.906250e-03, 0xFE]>"),
            (DataType::F8E4M3FN, [0x03, 0x7f], "dense<[5.859380e-03, 0x7F]>"),
            (DataType::F8E4M3FNUZ, [0x04, 0x80], "dense<[3.906250e-03, 0x80]>"),
            (DataType::F8E4M3B11FNUZ, [0x05, 0x80], "dense<[6.103520e-04, 0x80]>"),
            (DataType::F8E5M2, [0x06, 0x7f], "dense<[9.155270e-05, 0x7F]>"),
            (DataType::F8E5M2FNUZ, [0x07, 0x80], "dense<[5.340580e-05, 0x80]>"),
            (DataType::F8E8M0FNU, [0x08, 0xff], "dense<[1.504630e-36, 0xFF]>"),
        ] {
            let literal = Array::from_logical_bytes(
                ArrayType::new(data_type, Shape::new(vec![Dimension::Static(bits.len())])),
                &bits,
            )
            .unwrap();
            assert_eq!(test_literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let output = builder
                .add_instruction(
                    ArrayOperation::Constant(ConstantOperation::new(literal)),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert!(stablehlo.contains(rendered_values), "low-precision literal {data_type}: {stablehlo}");
        }

        // Standard floating-point families preserve signed zero, infinities, and NaN payload bits.
        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let f64_values =
            [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
        for (literal, expected) in [
            (Array::vector(bf16_values.to_vec()).unwrap(), values_to_bytes(&bf16_values)),
            (Array::vector(f16_values.to_vec()).unwrap(), values_to_bytes(&f16_values)),
            (Array::vector(f32_values.to_vec()).unwrap(), values_to_bytes(&f32_values)),
            (Array::vector(f64_values.to_vec()).unwrap(), values_to_bytes(&f64_values)),
        ] {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "floating-point literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Complex storage interleaves independently exact real and imaginary components in source order.
        let c64_components =
            [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
        let c64 = Array::vector(vec![
            ComplexNumber::new(c64_components[0], c64_components[1]),
            ComplexNumber::new(c64_components[2], c64_components[3]),
        ])
        .unwrap();
        assert_eq!(test_literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components),);
        let c128_components = [
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let c128 = Array::vector(vec![
            ComplexNumber::new(c128_components[0], c128_components[1]),
            ComplexNumber::new(c128_components[2], c128_components[3]),
        ])
        .unwrap();
        assert_eq!(test_literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components),);

        // Explicit physical layouts are traversed in logical row-major order before constructing the literal.
        let layout_type = ArrayType::new(DataType::I16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
            .with_layout(Some(ryft_core::arrays::StridedLayout::new(vec![2, 4]).into()));
        let layout_literal = Array::from_elements(layout_type, &[1_i16, 2, 3, 4]).unwrap();
        assert_eq!(layout_literal.storage_bytes(), values_to_bytes(&[1_i16, 3, 2, 4]));
        assert_eq!(test_literal_dense_bytes(&layout_literal, 8), values_to_bytes(&[1_i16, 2, 3, 4]));

        // Empty tensors remain valid dense constants and carry no raw payload bytes.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        let tensor_type = lower_tensor_type(empty.r#type().as_ref(), &context, location).unwrap();
        let attribute = empty.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(attribute.elements_count(), 0);

        // Zero literals use the established false i1 physical carrier without introducing logical payload bytes.
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
        assert_eq!(
            zero.to_dense_elements_attribute(tensor_type, &context).unwrap().to_string(),
            "dense<false> : tensor<2xi1>"
        );

        // Tokens remain effect values and have no dense literal representation.
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [2]), &[]).unwrap();
        assert_eq!(
            token.to_dense_elements_attribute(tensor_type, &context),
            Err(LoweringError::UnsupportedDataType { data_type: DataType::Token }),
        );
    }

    #[test]
    fn test_rank_positive_literal_constants_execute_exactly_on_cpu() {
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            ClientOptions, CpuClientOptions, ExecutionDeviceInputs, Program as PjrtProgram, load_cpu_plugin,
        };

        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let f64_values = [f64::from_bits(0x8000_0000_0000_0000), f64::from_bits(0x7ff8_0000_0000_1234)];
        let c64_components = [1.5_f32, -2.0, f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let c128_components =
            [1.5_f64, -2.0, f64::from_bits(0x8000_0000_0000_0000), f64::from_bits(0x7ff8_0000_0000_1234)];
        let cases = vec![
            (Array::vector(vec![true, false, true]).unwrap(), vec![1_u8, 0, 1]),
            (Array::vector(vec![-0x1234_i16, 0x2345]).unwrap(), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (
                Array::vector(vec![(1_u64 << 53) + 1, u64::MAX - 1]).unwrap(),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
            (Array::vector(bf16_values.to_vec()).unwrap(), values_to_bytes(&bf16_values)),
            (Array::vector(f16_values.to_vec()).unwrap(), values_to_bytes(&f16_values)),
            (Array::vector(f32_values.to_vec()).unwrap(), values_to_bytes(&f32_values)),
            (Array::vector(f64_values.to_vec()).unwrap(), values_to_bytes(&f64_values)),
            (
                Array::vector(vec![
                    ComplexNumber::new(c64_components[0], c64_components[1]),
                    ComplexNumber::new(c64_components[2], c64_components[3]),
                ])
                .unwrap(),
                values_to_bytes(&c64_components),
            ),
            (
                Array::vector(vec![
                    ComplexNumber::new(c128_components[0], c128_components[1]),
                    ComplexNumber::new(c128_components[2], c128_components[3]),
                ])
                .unwrap(),
                values_to_bytes(&c128_components),
            ),
            (Array::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap()]).unwrap(), vec![0x01, 0x00]),
            (Array::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
            (Array::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
            (Array::vector(vec![u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap(), vec![0x00, 0x01]),
            (Array::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
            (Array::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
            (Array::vector(Vec::<i32>::new()).unwrap(), Vec::new()),
        ];

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let outputs = cases
            .iter()
            .map(|(literal, _)| {
                builder
                    .add_instruction(
                        ArrayOperation::Constant(ConstantOperation::new(literal.clone())),
                        Vec::new(),
                        Vec::new(),
                        None,
                    )
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, Vec::new(), vec![Placeholder; cases.len()])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        for literal in [
            "dense<[true, false, true]> : tensor<3xi1>",
            "dense<[9007199254740993, 18446744073709551614]> : tensor<2xui64>",
            "dense<[-0.000000e+00, 0x7FC1]> : tensor<2xbf16>",
            "dense<[-0.000000e+00, 0x7E01]> : tensor<2xf16>",
            "dense<[-0.000000e+00, 0x7FC01234]> : tensor<2xf32>",
            "dense<[-0.000000e+00, 0x7FF8000000001234]> : tensor<2xf64>",
            "dense<[(1.500000e+00,-2.000000e+00), (-0.000000e+00,0x7FC01234)]> : tensor<2xcomplex<f32>>",
            "dense<[(1.500000e+00,-2.000000e+00), (-0.000000e+00,0x7FF8000000001234)]> : tensor<2xcomplex<f64>>",
            "dense<[true, false]> : tensor<2xi1>",
            "dense<[-2, 1]> : tensor<2xi2>",
            "dense<[-8, 7]> : tensor<2xi4>",
            "dense<[false, true]> : tensor<2xi1>",
            "dense<[0, 3]> : tensor<2xui2>",
            "dense<[1, 15]> : tensor<2xui4>",
            "dense<> : tensor<0xi32>",
        ] {
            assert!(module.contains(literal), "{module}");
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &CompilationOptions {
                    argument_layouts: Vec::new(),
                    parameter_is_tupled_arguments: false,
                    executable_build_options: Some(ExecutableCompilationOptions {
                        device_ordinal: -1,
                        replica_count: 1,
                        partition_count: 1,
                        ..Default::default()
                    }),
                    compile_portable_executable: false,
                    profile_version: 0,
                    individually_defined_output_indices: Vec::new(),
                    serialized_multi_slice_configuration: Vec::new(),
                    environment_option_overrides: std::collections::HashMap::new(),
                    target_config: None,
                    allow_in_place_mlir_modification: false,
                    matrix_unit_operand_precision: Precision::Default as i32,
                },
            )
            .unwrap();
        let mut executions = executable
            .execute(vec![ExecutionDeviceInputs::default()], Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let outputs = executions.remove(0).outputs;
        assert_eq!(outputs.len(), cases.len());
        for (output, (literal, expected)) in outputs.into_iter().zip(cases) {
            let actual = output.copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(actual, expected, "executed literal type {}", literal.r#type());
        }
    }

    #[test]
    fn test_literal_constant_atom_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let value =
                Array::from_elements::<f32>(test_vector_type(4).with_memory(memory), &[1.0, 2.0, 3.0, 4.0]).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let output = builder.add_constant(value);
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }
}
