use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, ConstantOperation, DataType, Dimension, DimensionAddOperation,
    DimensionDivOperation, DimensionFromScalarOperation, DimensionMaxOperation, DimensionMinOperation,
    DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRemOperation,
    DimensionSaturatingSubOperation, DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation,
    DimensionType, DimensionValue, EffectClass, Operation, ProgramError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::types::{
    lower_constant_elements_attribute, lower_dimension_extent, lower_static_index_constants,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_ADD_KIND, ASSERT_BOUNDS_KIND, ASSERT_DETAIL_ATTRIBUTE, ASSERT_DIV_KIND,
    ASSERT_KIND_ATTRIBUTE, ASSERT_LEFT_ATTRIBUTE, ASSERT_MUL_KIND, ASSERT_POW_KIND, ASSERT_REM_KIND,
    ASSERT_RIGHT_ATTRIBUTE, ASSERT_SUB_KIND, lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::lower_tensor_type;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one runtime safety check for dimension arithmetic whose declared bounds do not prove totality.
pub(crate) fn lower_dimension_arithmetic_assertion<'b, 'c: 'b, 't: 'c>(
    operation: &DimensionOperation<DimensionValue>,
    left_type: &DimensionType,
    right_type: &DimensionType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let kind = match operation {
        DimensionOperation::Add(_) => ASSERT_ADD_KIND,
        DimensionOperation::Sub(_) => ASSERT_SUB_KIND,
        DimensionOperation::Mul(_) => ASSERT_MUL_KIND,
        DimensionOperation::Pow(_) => ASSERT_POW_KIND,
        DimensionOperation::Div(_) => ASSERT_DIV_KIND,
        DimensionOperation::Rem(_) => ASSERT_REM_KIND,
        _ => return Ok(()),
    };
    let predicate = lower_deliberately_false_assertion_predicate(left, block, context, location)?;
    let left_name = left_type.dimension().to_string();
    let right_name = right_type.dimension().to_string();
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute(operation.name())),
        context.named_attribute(context.identifier(ASSERT_KIND_ATTRIBUTE), context.string_attribute(kind)),
        context
            .named_attribute(context.identifier(ASSERT_LEFT_ATTRIBUTE), context.string_attribute(left_name.as_str())),
        context
            .named_attribute(context.identifier(ASSERT_RIGHT_ATTRIBUTE), context.string_attribute(right_name.as_str())),
    ]);
    lower_assertion_custom_call(predicate, &[left, right], backend_config, effect_tokens, block, context, location)
}

impl LowerableOperation for DimensionSizeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let block = &mut lowerer.block;
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        let [input_type] = input_types else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_types.len() }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        match input_type.shape().dimensions()[self.axis()] {
            Dimension::Static(extent) => {
                let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                let extent = i64::try_from(extent).map_err(|_| LoweringError::UnsupportedOp {
                    op: format!("dimension extent {extent} does not fit StableHLO i64"),
                })?;
                let elements = lower_constant_elements_attribute(DataType::I64, output_type, extent, context)?;
                let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
                Ok(vec![constant.result(0).expect("stablehlo.constant should return one result").as_ref()])
            }
            Dimension::Dynamic(_) => {
                let size = block.append_operation(stable_hlo::get_dimension_size(*input, self.axis(), location)?)?;
                let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
                let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                let converted = block.append_operation(stable_hlo::convert(size, output_type, location)?)?;
                Ok(vec![converted.result(0).expect("stablehlo.convert should return one result").as_ref()])
            }
        }
    }
}

impl LowerableOperation for DimensionFromScalarOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        let i64_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
        let converted = block.append_operation(stable_hlo::convert(*input, i64_type, location)?)?;
        let converted = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
        let bounds = self.output_type().bounds();
        let lower = lower_static_index_constants(&[bounds.lower()], block, context, location)?[0];
        let at_least_lower =
            lower_compare_to_mlir(ComparisonDirection::GreaterThanOrEqual, converted, lower, block, location)?;
        // Do not encode an exclusive maximum-plus-one in the signed runtime extent representation.
        let predicate = match bounds.upper().filter(|upper| *upper <= ryft_core::MAX_DIMENSION_EXTENT) {
            Some(upper) => {
                let upper = lower_static_index_constants(&[upper], block, context, location)?[0];
                let below_upper =
                    lower_compare_to_mlir(ComparisonDirection::LessThan, converted, upper, block, location)?;
                block
                    .append_operation(stable_hlo::and(at_least_lower, below_upper, location)?)?
                    .result(0)
                    .unwrap()
                    .as_ref()
            }
            None => at_least_lower,
        };
        let backend_config = context.dictionary_attribute(&[
            context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute(self.name())),
            context.named_attribute(
                context.identifier(ASSERT_KIND_ATTRIBUTE),
                context.string_attribute(ASSERT_BOUNDS_KIND),
            ),
            context.named_attribute(
                context.identifier(ASSERT_LEFT_ATTRIBUTE),
                context.string_attribute(self.output_type().dimension().to_string().as_str()),
            ),
            context.named_attribute(
                context.identifier(ASSERT_DETAIL_ATTRIBUTE),
                context.string_attribute(bounds.to_string().as_str()),
            ),
        ]);
        lower_assertion_custom_call(predicate, &[converted], backend_config, effect_tokens, block, context, location)?;
        Ok(vec![converted])
    }
}

impl LowerableOperation for DimensionToScalarOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        Ok(vec![*input])
    }
}

impl LowerableOperation for ConstantOperation<DimensionValue> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if !input_values.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
        }
        Ok(vec![lower_dimension_extent(self.value(), &mut lowerer.block, lowerer.context, lowerer.location)?])
    }
}

impl LowerableOperation for DimensionMaxOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Max(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionMinOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Min(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionAddOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Add(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionSubOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Sub(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionSaturatingSubOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::SaturatingSub(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionMulOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Mul(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionDivOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Div(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionRemOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Rem(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionPowOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        _output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_dimension_arithmetic(&DimensionOperation::Pow(self.clone()), lowerer, input_values, input_types)
    }
}

/// Lowers one first-class dimension arithmetic operation on scalar `i64` extents. Arithmetic that the operand bounds
/// do not prove safe keeps the checked eager semantics through a runtime assertion on the ordered-assertion chain.
fn lower_dimension_arithmetic<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    operation: &DimensionOperation<DimensionValue>,
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[DimensionType],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let context = lowerer.context;
    let location = lowerer.location;
    let effect_tokens = &mut lowerer.effect_tokens;
    let block = &mut lowerer.block;
    let [left, right] = input_values else {
        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
    };
    let [left_type, right_type] = input_types else {
        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
    };
    // Bounds-proven arithmetic lowers without assertion overhead. Otherwise, preserve eager checked
    // semantics with one diagnostic runtime check on the ordered-assertion chain.
    let requires_runtime_assertion = operation.effects().classes().contains(EffectClass::OrderedAssertion);
    if requires_runtime_assertion {
        lower_dimension_arithmetic_assertion(
            operation,
            left_type,
            right_type,
            *left,
            *right,
            effect_tokens,
            block,
            context,
            location,
        )?;
    }
    // A runtime assertion and the arithmetic operation do not have a StableHLO data dependency.
    // Select a valid divisor for the data operation so speculative or reordered execution cannot
    // evaluate division by zero before the host callback reports the original operands.
    let safe_right = if requires_runtime_assertion
        && matches!(operation, DimensionOperation::Div(_) | DimensionOperation::Rem(_))
    {
        let constants = lower_static_index_constants(&[0, 1], block, context, location)?;
        let positive = lower_compare_to_mlir(ComparisonDirection::GreaterThan, *right, constants[0], block, location)?;
        let selected = block.append_operation(stable_hlo::select(positive, *right, constants[1], location)?)?;
        selected.result(0).expect("stablehlo.select should return one result").as_ref()
    } else {
        *right
    };
    let result = match operation {
        DimensionOperation::Add(_) => block.append_operation(stable_hlo::add(*left, *right, location)?)?,
        DimensionOperation::Sub(_) => block.append_operation(stable_hlo::subtract(*left, *right, location)?)?,
        DimensionOperation::SaturatingSub(_) => {
            let difference = block.append_operation(stable_hlo::subtract(*left, *right, location)?)?;
            let difference = difference.result(0).unwrap().as_ref();
            let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, 0_i64, context)?;
            let zero = block.append_operation(stable_hlo::constant(elements, location)?)?;
            let zero = zero.result(0).unwrap().as_ref();
            block.append_operation(stable_hlo::maximum(difference, zero, location)?)?
        }
        DimensionOperation::Mul(_) => block.append_operation(stable_hlo::multiply(*left, *right, location)?)?,
        DimensionOperation::Pow(_) => block.append_operation(stable_hlo::power(*left, *right, location)?)?,
        DimensionOperation::Div(_) => block.append_operation(stable_hlo::divide(*left, safe_right, location)?)?,
        DimensionOperation::Rem(_) => block.append_operation(stable_hlo::remainder(*left, safe_right, location)?)?,
        DimensionOperation::Min(_) => block.append_operation(stable_hlo::minimum(*left, *right, location)?)?,
        DimensionOperation::Max(_) => block.append_operation(stable_hlo::maximum(*left, *right, location)?)?,
        _ => unreachable!(),
    };
    // For the same no-data-dependency reason, an unproven subtraction can hand a negative extent to
    // consumers such as `set_dimension_size`, and unproven addition/multiplication/power can hand
    // them a wrapped negative, failing inside XLA before the host callback reports the original
    // operands. Clamping the unproven data path to zero keeps it well-defined so the assertion owns
    // the diagnostic.
    let result = if requires_runtime_assertion
        && matches!(
            operation,
            DimensionOperation::Add(_)
                | DimensionOperation::Sub(_)
                | DimensionOperation::Mul(_)
                | DimensionOperation::Pow(_)
        ) {
        let constants = lower_static_index_constants(&[0], block, context, location)?;
        let raw = result.result(0).unwrap().as_ref();
        block.append_operation(stable_hlo::maximum(raw, constants[0], location)?)?
    } else {
        result
    };
    Ok(vec![result.result(0).unwrap().as_ref()])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType, DimensionBounds, DimensionVariable, Placeholder};

    use crate::experimental::ops::{XlaConstant, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_clamps_unproven_dimension_arithmetic_data_paths() {
        use ryft_core::{
            DimensionFromScalarOperation, DimensionOperation, DimensionSubOperation, DimensionToScalarOperation,
        };

        let scalar_type = ArrayType::scalar(DataType::I64);
        let build = |left_bounds: DimensionBounds, right_bounds: DimensionBounds| {
            let left_variable = DimensionVariable::new("left", left_bounds);
            let right_variable = DimensionVariable::new("right", right_bounds);
            let left_type = DimensionType::from(left_variable.clone());
            let right_type = DimensionType::from(right_variable.clone());
            let mut builder = CompositeXlaProgramBuilder::new();
            let left = builder.add_input(scalar_type.clone().into());
            let right = builder.add_input(scalar_type.clone().into());
            let left = builder
                .add_instruction(DimensionFromScalarOperation::new(left_variable), Vec::new(), vec![left], None)
                .unwrap()[0];
            let right = builder
                .add_instruction(DimensionFromScalarOperation::new(right_variable), Vec::new(), vec![right], None)
                .unwrap()[0];
            let difference = builder
                .add_instruction(
                    DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
                    Vec::new(),
                    vec![left, right],
                    None,
                )
                .unwrap()[0];
            let output =
                builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![difference], None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![scalar_type.clone(), scalar_type.clone()],
                &vec![scalar_type.clone()],
                "main",
                None,
                None,
            )
            .unwrap()
        };

        // An unproven subtraction keeps its diagnostic assertion (two gateway checks plus one arithmetic check), and
        // because the assertion custom call has no StableHLO data dependency on the subtraction, the data path is
        // clamped to zero so downstream extent consumers cannot fail inside XLA before the host callback reports the
        // original operands.
        let unproven = build(DimensionBounds::new(1, Some(9)).unwrap(), DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(unproven.matches("stablehlo.subtract").count(), 1, "{unproven}");
        assert_eq!(unproven.matches("stablehlo.maximum").count(), 1, "{unproven}");
        assert_eq!(unproven.matches("@ryft.assert").count(), 3, "{unproven}");

        // A bounds-proven subtraction lowers without an arithmetic assertion and without a clamp.
        let proven = build(DimensionBounds::new(5, Some(9)).unwrap(), DimensionBounds::new(1, Some(3)).unwrap());
        assert_eq!(proven.matches("stablehlo.subtract").count(), 1, "{proven}");
        assert_eq!(proven.matches("stablehlo.maximum").count(), 0, "{proven}");
        assert_eq!(proven.matches("@ryft.assert").count(), 2, "{proven}");
    }
}
