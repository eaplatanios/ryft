use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, ConstantOperation, DataType, Dimension as CoreDimension,
    DimensionAddOperation, DimensionDivOperation, DimensionFromScalarOperation, DimensionMaxOperation,
    DimensionMinOperation, DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRemOperation,
    DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation, DimensionType, DimensionValue,
    EffectClass, Operation as CoreOperation, ProgramError, SaturatingSubOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::EffectTokens;
use crate::lowering::types::{
    lower_constant_elements_attribute, lower_dimension_extent, lower_static_index_constants, lower_tensor_type,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_ADD_KIND, ASSERT_BOUNDS_KIND, ASSERT_DETAIL_ATTRIBUTE, ASSERT_DIV_KIND,
    ASSERT_KIND_ATTRIBUTE, ASSERT_LEFT_ATTRIBUTE, ASSERT_MUL_KIND, ASSERT_POW_KIND, ASSERT_REM_KIND,
    ASSERT_RIGHT_ATTRIBUTE, ASSERT_SUB_KIND, lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for DimensionSizeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        let context = lowerer.context();
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        let [input_type] = input_types else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_types.len() }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        match input_type.shape().dimensions()[self.axis()] {
            CoreDimension::Static(extent) => {
                let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                let extent = i64::try_from(extent).map_err(|_| {
                    ryft_core::DimensionError::ExtentExceedsBackendWidth { value: extent, maximum: i64::MAX as usize }
                })?;
                let elements = lower_constant_elements_attribute(DataType::I64, output_type, extent, context)?;
                let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
                Ok(vec![constant.result(0).unwrap().as_ref()])
            }
            CoreDimension::Dynamic(_) => {
                let size = block.append_operation(stable_hlo::get_dimension_size(*input, self.axis(), location)?)?;
                let size = size.result(0).unwrap().as_ref();
                let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                let converted = block.append_operation(stable_hlo::convert(size, output_type, location)?)?;
                Ok(vec![converted.result(0).unwrap().as_ref()])
            }
        }
    }
}

impl LowerableOperation for DimensionFromScalarOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let context = lowerer.context();
        let location = lowerer.location();
        let block = &mut lowerer.block();
        let effect_tokens = lowerer.effect_tokens_mut();
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        let i64_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
        let converted = block.append_operation(stable_hlo::convert(*input, i64_type, location)?)?;
        let converted = converted.result(0).unwrap().as_ref();
        let bounds = self.output_type().bounds();
        let lower = lower_static_index_constants(&[bounds.lower()], block, context, location)?[0];
        let at_least_lower =
            lower_compare_to_mlir(ComparisonDirection::GreaterThanOrEqual, converted, lower, block, location)?;
        // Do not encode an exclusive upper bound, one past the maximum, in the signed runtime extent representation.
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
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_types, output_types);
        let [input] = input_values else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        };
        Ok(vec![*input])
    }
}

impl LowerableOperation for ConstantOperation<DimensionValue> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        if !input_values.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
        }
        Ok(vec![lower_dimension_extent(self.value(), &mut lowerer.block(), lowerer.context(), lowerer.location())?])
    }
}

impl LowerableOperation for DimensionMaxOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Max(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionMinOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Min(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionAddOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Add(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionSubOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Sub(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for SaturatingSubOperation<DimensionType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::SaturatingSub(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionMulOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Mul(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionDivOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Div(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionRemOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Rem(self.clone()), lowerer, input_values, input_types)
    }
}

impl LowerableOperation for DimensionPowOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        lower_dimension_arithmetic(&DimensionOperation::Pow(self.clone()), lowerer, input_values, input_types)
    }
}

/// Lowers one first-class dimension arithmetic operation on scalar `i64` extents. Arithmetic that the operand bounds
/// do not prove safe keeps the checked eager semantics through a runtime assertion on the ordered assertion chain.
fn lower_dimension_arithmetic<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    operation: &DimensionOperation<DimensionValue>,
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[DimensionType],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let context = lowerer.context();
    let location = lowerer.location();
    let block = &mut lowerer.block();
    let effect_tokens = lowerer.effect_tokens_mut();
    let [left, right] = input_values else {
        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into());
    };
    let [left_type, right_type] = input_types else {
        return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_types.len() }.into());
    };
    // Arithmetic that the bounds prove safe lowers without assertion overhead. Otherwise, preserve the checked eager
    // semantics with one diagnostic runtime check on the ordered assertion chain.
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
        selected.result(0).unwrap().as_ref()
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
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, 0i64, context)?;
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
    // Because of the same missing data dependency, an unproven subtraction can hand a negative extent to consumers
    // such as `set_dimension_size`, and an unproven addition, multiplication, or power can hand them a wrapped
    // negative, failing inside XLA before the host callback reports the original operands. Clamping the unproven data
    // path to zero keeps it well defined, so that the assertion owns the diagnostic.
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

/// Lowers one runtime safety check for dimension arithmetic whose declared bounds do not prove totality.
#[allow(clippy::too_many_arguments)]
fn lower_dimension_arithmetic_assertion<'b, 'c: 'b, 't: 'c>(
    operation: &DimensionOperation<DimensionValue>,
    left_type: &DimensionType,
    right_type: &DimensionType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{CompilationCacheDomain, DimensionBounds, DimensionVariable, Placeholder, Shape};

    use crate::domains::tests::{domain_mesh, read_i64s, replicated_scalar_type};
    use crate::lowering::tests::{MixedValue, execute_mixed_program};
    use crate::operations::ProgramBuilder as ArrayIrProgramBuilder;
    use crate::tests::{cpu_client, execution_client};
    use crate::{Array, Constant, Options, Session, to_mlir_module_for_program};

    use super::*;

    /// Lowers one binary dimension arithmetic operation over two dimension inputs, whose boundary values are `i64`
    /// scalars, to a rendered StableHLO module.
    fn lowered_binary_dimension_module(
        operation: DimensionOperation<DimensionValue>,
        left_type: DimensionType,
        right_type: DimensionType,
    ) -> String {
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(left_type.into());
        let right = builder.add_input(right_type.into());
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let boundary_type = ArrayType::scalar(DataType::I64);
        to_mlir_module_for_program(
            &program,
            &[],
            &[boundary_type.clone(), boundary_type.clone()],
            &[boundary_type],
            "main",
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_dimension_size_operation_lower() {
        // Static extents lower to constants, and dynamic extents read the runtime size of their dimension.
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                CoreDimension::Static(3),
                CoreDimension::Dynamic(DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap())),
            ]),
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let static_size = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let dynamic_size = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 1).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![static_size, dynamic_size],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let boundary_type = ArrayType::scalar(DataType::I64);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[input_type],
            &[boundary_type.clone(), boundary_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<i32>) -> (tensor<i64>, tensor<i64>) {
                %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<3x4xf32>, tensor<i32>) -> tensor<3x?xf32, #stablehlo.bounds<?, 4>>
                %c = stablehlo.constant dense<3> : tensor<i64>
                %1 = stablehlo.get_dimension_size %0, dim = 1 : (tensor<3x?xf32, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                return %c, %2 : tensor<i64>, tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_from_scalar_operation_lower_reports_bounds_failures() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client).with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());

        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 1, "{}", lowered.stable_hlo());
        let compiled = session.compile_program(&lowered).unwrap();
        assert!(compiled.requires_assertion_handler());
        // Some PJRT plugins do not serialize executables containing host callbacks. When serialization is available,
        // exercise the path of cache hits and prove that its metadata about runtime features survives restoration.
        let compiled = match session.serialize_program(&compiled).unwrap() {
            Some(bytes) => session.deserialize_program(bytes.as_slice()).unwrap().unwrap(),
            None => compiled,
        };
        assert!(compiled.requires_assertion_handler());

        let valid =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), 4i64.to_ne_bytes().as_slice()).unwrap();
        let valid = session.execute_program(&compiled, vec![valid]).unwrap();
        assert_eq!(read_i64s(&client, &valid[0]), vec![4]);

        let invalid = Array::from_host_buffer(&session, input_type, mesh, 9i64.to_ne_bytes().as_slice()).unwrap();
        let error = session.execute_program(&compiled, vec![invalid]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "`dimension_from_scalar` failed: input dimension `extent` = 9 is outside its declared bounds [1, 9)",
        );
    }

    #[test]
    fn test_dimension_max_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Max(DimensionMaxOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.maximum %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_min_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Min(DimensionMinOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.minimum %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_add_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_sub_operation_lower_clamps_unproven_data_path() {
        let scalar_type = ArrayType::scalar(DataType::I64);
        let build = |left_bounds: DimensionBounds, right_bounds: DimensionBounds| {
            let left_variable = DimensionVariable::new("left", left_bounds);
            let right_variable = DimensionVariable::new("right", right_bounds);
            let left_type = DimensionType::from(left_variable.clone());
            let right_type = DimensionType::from(right_variable.clone());
            let mut builder = ArrayIrProgramBuilder::new();
            let left = builder.add_input(scalar_type.clone().into());
            let right = builder.add_input(scalar_type.clone().into());
            let left = builder
                .add_instruction(DimensionFromScalarOperation::new(left_variable.into()), Vec::new(), vec![left], None)
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_variable.into()),
                    Vec::new(),
                    vec![right],
                    None,
                )
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
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
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

        // An unproven subtraction keeps its diagnostic assertions, which are two gateway checks and one arithmetic
        // check. Because the assertion custom call has no StableHLO data dependency on the subtraction, the data path
        // is clamped to zero so that downstream extent consumers cannot fail inside XLA before the host callback
        // reports the original operands.
        let unproven = build(DimensionBounds::new(1, Some(9)).unwrap(), DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(
            unproven,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.convert %arg0 : tensor<i64>
                %c = stablehlo.constant dense<1> : tensor<i64>
                %1 = stablehlo.compare GE, %0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %c_0 = stablehlo.constant dense<9> : tensor<i64>
                %2 = stablehlo.compare LT, %0, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %3 = stablehlo.and %1, %2 : tensor<i1>
                %4 = stablehlo.after_all  : !stablehlo.token
                %5 = stablehlo.custom_call @ryft.assert(%3, %0, %4) {api_version = 4 : i32, backend_config = {actor = "dimension_from_scalar", detail = "[1, 9)", kind = "bounds", left = "left"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                %6 = stablehlo.convert %arg1 : tensor<i64>
                %c_1 = stablehlo.constant dense<1> : tensor<i64>
                %7 = stablehlo.compare GE, %6, %c_1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %c_2 = stablehlo.constant dense<9> : tensor<i64>
                %8 = stablehlo.compare LT, %6, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %9 = stablehlo.and %7, %8 : tensor<i1>
                %10 = stablehlo.custom_call @ryft.assert(%9, %6, %5) {api_version = 4 : i32, backend_config = {actor = "dimension_from_scalar", detail = "[1, 9)", kind = "bounds", left = "right"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                %c_3 = stablehlo.constant dense<0> : tensor<i64>
                %11 = stablehlo.compare LT, %0, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %12 = stablehlo.custom_call @ryft.assert(%11, %0, %6, %10) {api_version = 4 : i32, backend_config = {actor = "dimension_sub", kind = "sub", left = "left", right = "right"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                %13 = stablehlo.subtract %0, %6 : tensor<i64>
                %c_4 = stablehlo.constant dense<0> : tensor<i64>
                %14 = stablehlo.maximum %13, %c_4 : tensor<i64>
                return %14 : tensor<i64>
              }
            }
            "#},
        );

        // A subtraction that the bounds prove safe lowers without an arithmetic assertion and without a clamp.
        let proven = build(DimensionBounds::new(5, Some(9)).unwrap(), DimensionBounds::new(1, Some(3)).unwrap());
        assert_eq!(
            proven,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.convert %arg0 : tensor<i64>
                %c = stablehlo.constant dense<5> : tensor<i64>
                %1 = stablehlo.compare GE, %0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %c_0 = stablehlo.constant dense<9> : tensor<i64>
                %2 = stablehlo.compare LT, %0, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %3 = stablehlo.and %1, %2 : tensor<i1>
                %4 = stablehlo.after_all  : !stablehlo.token
                %5 = stablehlo.custom_call @ryft.assert(%3, %0, %4) {api_version = 4 : i32, backend_config = {actor = "dimension_from_scalar", detail = "[5, 9)", kind = "bounds", left = "left"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                %6 = stablehlo.convert %arg1 : tensor<i64>
                %c_1 = stablehlo.constant dense<1> : tensor<i64>
                %7 = stablehlo.compare GE, %6, %c_1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %c_2 = stablehlo.constant dense<3> : tensor<i64>
                %8 = stablehlo.compare LT, %6, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                %9 = stablehlo.and %7, %8 : tensor<i1>
                %10 = stablehlo.custom_call @ryft.assert(%9, %6, %5) {api_version = 4 : i32, backend_config = {actor = "dimension_from_scalar", detail = "[1, 3)", kind = "bounds", left = "right"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                %11 = stablehlo.subtract %0, %6 : tensor<i64>
                return %11 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_saturating_sub_operation_lower() {
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(DimensionType::new("left", DimensionBounds::non_negative(None).unwrap()).into());
        let right = builder.add_input(DimensionType::new("right", DimensionBounds::non_negative(None).unwrap()).into());
        let output = builder
            .add_instruction(SaturatingSubOperation::<DimensionType>::new(), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let boundary_type = ArrayType::scalar(DataType::I64);
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &[boundary_type.clone(), boundary_type.clone()],
                &[boundary_type],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i64>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %1 = stablehlo.maximum %0, %c : tensor<i64>
                    return %1 : tensor<i64>
                  }
                }
            "#},
        );
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(2), MixedValue::Dimension(5)], &[])
                .unwrap(),
            vec![MixedValue::Dimension(0)],
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Dimension(i64::MAX), MixedValue::Dimension(0)],
                &[],
            ).unwrap(),
            vec![MixedValue::Dimension(i64::MAX)],
        );
    }

    #[test]
    fn test_dimension_mul_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Mul(DimensionMulOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_div_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Div(DimensionDivOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.divide %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_rem_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.remainder %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_dimension_pow_operation_lower() {
        let left_type = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(3)).unwrap());
        let module = lowered_binary_dimension_module(
            DimensionOperation::Pow(DimensionPowOperation::new(&left_type, &right_type).unwrap()),
            left_type,
            right_type,
        );
        assert_eq!(
            module,
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.power %arg0, %arg1 : tensor<i64>
                return %0 : tensor<i64>
              }
            }
            "#},
        );
    }

    #[test]
    fn test_lower_dimension_arithmetic_preserves_checked_diagnostics() {
        let client = cpu_client(1);
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::with_client(&client).with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let check = |operation: DimensionOperation<DimensionValue>,
                     valid_operands: (i64, i64),
                     expected_output: i64,
                     invalid_operands: (i64, i64),
                     expected_error: &str| {
            let mut builder = ArrayIrProgramBuilder::new();
            let left_input = builder.add_input(input_type.clone().into());
            let right_input = builder.add_input(input_type.clone().into());
            let left = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(left_type.clone()),
                    Vec::new(),
                    vec![left_input],
                    None,
                )
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_type.clone()),
                    Vec::new(),
                    vec![right_input],
                    None,
                )
                .unwrap()[0];
            let result = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
            let output =
                builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![result], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap();
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 3, "{}", lowered.stable_hlo());
            let compiled = session.compile_program(&lowered).unwrap();
            let input = |value: i64| {
                Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap()
            };

            let output =
                session.execute_program(&compiled, vec![input(valid_operands.0), input(valid_operands.1)]).unwrap();
            assert_eq!(read_i64s(&client, &output[0]), vec![expected_output]);
            let error = session
                .execute_program(&compiled, vec![input(invalid_operands.0), input(invalid_operands.1)])
                .unwrap_err();
            assert_eq!(error.to_string(), expected_error);
        };

        check(
            DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
            (5, 2),
            3,
            (2, 5),
            "`dimension_sub` requires left >= right; got left=2, right=5",
        );
        check(
            DimensionOperation::Div(DimensionDivOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            2,
            (7, 0),
            "`dimension_div` requires right > 0; got left=7, right=0",
        );
        check(
            DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            1,
            (7, 0),
            "`dimension_rem` requires right > 0; got left=7, right=0",
        );
    }
}
