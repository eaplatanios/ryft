use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, Dimension, ProgramError, RNG_BIT_GENERATOR_OPERATION_NAME, RandomAlgorithm,
    RngBitGeneratorOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::lower_tensor_type;
use crate::operations::assertions::lower_output_extent_validation;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one traced random bit generation with `algorithm` to a `stablehlo.rng_bit_generator` that produces bits of
/// type `bits_type`, mapping the algorithm to the corresponding StableHLO algorithm attribute. The two results are the
/// advanced generator state and the bits. The bits type is the instruction's (possibly refined) output type rather
/// than the operation's declared one, whose dynamic axes exact extent operands may have refined to static axes.
pub(crate) fn lower_rng_bit_generator_to_mlir<'b, 'c: 'b, 't: 'c>(
    algorithm: RandomAlgorithm,
    bits_type: &ArrayType,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    let algorithm = match algorithm {
        RandomAlgorithm::ThreeFry => stable_hlo::RngAlgorithm::ThreeFry,
        RandomAlgorithm::Philox => stable_hlo::RngAlgorithm::Philox,
    };
    let output_type = lower_tensor_type(bits_type, context, location)?;
    let generated =
        block.append_operation(stable_hlo::rng_bit_generator(input_values[0], algorithm, output_type, location)?)?;
    Ok(vec![
        generated.result(0).expect("stablehlo.rng_bit_generator should return a state result").as_ref(),
        generated.result(1).expect("stablehlo.rng_bit_generator should return a bits result").as_ref(),
    ])
}

impl LowerableOperation for RngBitGeneratorOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_rng_bit_generator_to_mlir(
            self.algorithm(),
            self.output_type(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl LowerableOperation for RngBitGeneratorOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let block = &mut lowerer.block;
        // Type inference refines the declared dynamic output axes whose extents are exact (e.g., those of eager
        // dimension values) to static axes, so only the refined bits type decides whether the generator can be
        // lowered. Their extent operands can be omitted after validating symbolic operands against the refined extents.
        let [_, bits_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 2, actual: output_types.len() }.into());
        };
        let bits_type = <&ArrayType>::try_from(bits_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        if bits_type.static_shape().is_none() {
            return Err(LoweringError::UnsupportedOp {
                op: "rng-bit-generator with dynamic output extents cannot lower by generating the physical \
                     upper-bound shape because that would advance its functional state by the physical rather \
                     than logical element count"
                    .to_string(),
            });
        }
        let expected_input_count = 1 + self
            .output_type()
            .shape()
            .dimensions()
            .iter()
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count();
        check_count!("input", input_values, expected_input_count, ProgramError);
        check_count!("input", input_types, expected_input_count, ProgramError);
        lower_output_extent_validation(
            RNG_BIT_GENERATOR_OPERATION_NAME,
            std::slice::from_ref(self.output_type()),
            &input_values[1..],
            &input_types[1..],
            std::slice::from_ref(bits_type),
            &mut lowerer.effect_tokens,
            block,
            context,
            location,
        )?;
        lower_rng_bit_generator_to_mlir(self.algorithm(), bits_type, &input_values[..1], block, context, location)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType, Dimension, Placeholder, Shape};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, unproject_plain_program};
    use crate::experimental::ops::XlaArrayConstant;
    use crate::to_mlir_module_for_program;

    #[test]
    fn test_to_mlir_module_for_program_lowers_rng_bit_generator() {
        use ryft_core::{RandomAlgorithm, RngBitGeneratorOperation};

        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type.clone());
        let outputs = builder
            .add_instruction(
                RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type.clone()),
                Vec::new(),
                vec![state],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type, bits_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>) {
                    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
                    return %output_state, %output : tensor<2xui64>, tensor<4xui32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_rng_bit_generator_with_exact_output_extent() {
        use ryft_core::{
            ArrayIrType, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, RandomAlgorithm,
            RngBitGeneratorOperation,
        };

        use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};

        // An exact extent refines the declared dynamic bits axis to a static one, so the generator lowers at that
        // static shape and its redundant extent operand does not reach `stablehlo.rng_bit_generator`.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let mut builder = CompositeXlaProgramBuilder::new();
        let state = builder.add_input(state_type.clone().into());
        let extent =
            builder.add_constant(XlaConstant::Dimension(DimensionValue::new(DimensionType::from(extent), 3).unwrap()));
        let outputs = builder
            .add_instruction(
                XlaOperation::RngBitGenerator(RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type)),
                Vec::new(),
                vec![state, extent],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let bits_type = ArrayType::new_static(DataType::U32, [3]);
        assert_eq!(program.output_types()[1], ArrayIrType::Array(bits_type.clone()));
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type, bits_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3xui32>) {
                    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3xui32>)
                    return %output_state, %output : tensor<2xui64>, tensor<3xui32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_rng_bit_generator_mixed_repeated_extents() {
        use ryft_core::{
            ConvertElementTypeOperation, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
            RandomAlgorithm, RngBitGeneratorOperation, ZeroOperation,
        };

        use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program};
        use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
        use crate::operations::assertions::ensure_assertion_handler_registered;
        use crate::tests::execution_client;

        // An exact operand makes both repeated bits axes static, but the symbolic peer must still equal that extent.
        // The two generators cover both operand orders and consume the same state, so they produce the same bits.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
        let state = builder
            .add_instruction(ZeroOperation::new(RandomAlgorithm::ThreeFry.state_type()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let mut outputs = Vec::new();
        for inputs in [vec![state, symbolic, exact], vec![state, exact, symbolic]] {
            let bits = builder
                .add_instruction(
                    XlaOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                        RandomAlgorithm::ThreeFry,
                        bits_type.clone(),
                    )),
                    Vec::new(),
                    inputs,
                    None,
                )
                .unwrap()[1];
            outputs.push(
                builder
                    .add_instruction(
                        ConvertElementTypeOperation::<ArrayType>::new(DataType::F64, false),
                        Vec::new(),
                        vec![bits],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::I64)],
            &vec![ArrayType::new_static(DataType::F64, [3, 3]); 2],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>) -> (tensor<3x3xf64>, tensor<3x3xf64>) {
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<2xui64>
                    %c_0 = stablehlo.constant dense<3> : tensor<i64>
                    %1 = stablehlo.compare EQ, %arg0, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %2 = stablehlo.after_all  : !stablehlo.token
                    %3 = stablehlo.custom_call @ryft.assert(%1, %arg0, %2) {api_version = 4 : i32, backend_config = {actor = "rng_bit_generator", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %output_state, %output = stablehlo.rng_bit_generator %0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xui32>)
                    %4 = stablehlo.convert %output : (tensor<3x3xui32>) -> tensor<3x3xf64>
                    %c_1 = stablehlo.constant dense<3> : tensor<i64>
                    %5 = stablehlo.compare EQ, %arg0, %c_1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %6 = stablehlo.custom_call @ryft.assert(%5, %arg0, %3) {api_version = 4 : i32, backend_config = {actor = "rng_bit_generator", kind = "generic", label_0 = "size", label_count = "1", message = "input dimension `size` expected 3"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %output_state_2, %output_3 = stablehlo.rng_bit_generator %0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xui32>)
                    %7 = stablehlo.convert %output_3 : (tensor<3x3xui32>) -> tensor<3x3xf64>
                    return %4, %7 : tensor<3x3xf64>, tensor<3x3xf64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        // The bits are deterministic for the fixed all-zero generator state.
        let bits = vec![
            1797259609.0,
            1351547692.0,
            1688610540.0,
            2579123966.0,
            3235790642.0,
            4229293427.0,
            3098264785.0,
            2892874427.0,
            1447157908.0,
        ];
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]),
            Ok(vec![MixedValue::Array(bits.clone(), vec![3, 3]), MixedValue::Array(bits, vec![3, 3])]),
        );
        let error = execute_mixed_program(&client, &program, &[MixedValue::Dimension(4)], &[]).unwrap_err().to_string();
        assert!(error.contains("input dimension `size` expected 3"), "{error}");
    }

    #[test]
    fn test_lower_rng_bit_generator_rejects_missing_input_types() {
        use ryft_core::{
            ArrayIrType, DimensionBounds, DimensionVariable, ProgramError, RandomAlgorithm, RngBitGeneratorOperation,
        };
        use ryft_mlir::{Block, Context as MlirContext, Location, Value as MlirValue};

        use crate::experimental::lowering::types::lower_tensor_type;
        use crate::experimental::ops::XlaOperation;
        use crate::{LowerableOperation, LoweringError, XlaLowerer};

        // Input types are validated before they are split into the state type and the extent operand types.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let operation = RngBitGeneratorOperation::<ArrayIrType>::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![extent.into()])),
        );
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[
            (lower_tensor_type(&state_type, &context, location).unwrap(), location),
            (lower_tensor_type(&ArrayType::scalar(DataType::I64), &context, location).unwrap(), location),
        ]);
        let inputs = vec![block.argument(0).unwrap().as_ref(), block.argument(1).unwrap().as_ref()];
        let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> =
            XlaLowerer::new(block.as_ref(), &context, location.as_ref());
        let output_types = vec![state_type.into(), ArrayType::new_static(DataType::U32, [3]).into()];
        assert!(matches!(
            operation.lower(&mut lowerer, &inputs, &[], &output_types),
            Err(LoweringError::Tracing(ProgramError::InvalidInputCount { expected: 2, actual: 0 })),
        ));
    }
}
