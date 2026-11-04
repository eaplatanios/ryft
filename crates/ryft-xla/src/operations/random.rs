use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, DROPOUT_OPERATION_NAME, DataType, Dimension, DropoutOperation, Operation, ProgramError,
    RNG_BIT_GENERATOR_OPERATION_NAME, RandomAlgorithm, RngBitGeneratorOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::operations::assertions::lower_output_extent_validation;
use crate::{Error, LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

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
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
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
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
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
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let context = lowerer.context;
        let location = lowerer.location;
        let block = &mut lowerer.block;
        // Type inference refines the declared dynamic output axes whose extents are exact (e.g., those of eager
        // dimension values) to static axes, so only the refined bits type decides whether the generator can be
        // lowered. Their extent operands can be omitted after validating symbolic operands against the refined extents.
        let [_, bits_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 2, actual: output_types.len() }.into());
        };
        let bits_type = <&ArrayType>::try_from(bits_type)?;
        if bits_type.static_shape().is_none() {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: self.name().to_string(),
                reason: "rng-bit-generator with dynamic output extents cannot lower by generating the physical \
                     upper-bound shape because that would advance its functional state by the physical rather \
                     than logical element count"
                    .to_string(),
            }));
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

impl LowerableOperation for DropoutOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // This is the portable lowering that `DropoutOperation::drop_threshold` documents, which draws the same mask as
        // the eager `Array` implementation. It draws one `u32` word per mask element from the provided state, or from
        // the fixed `[0, 0]` ThreeFry state, keeps the elements whose words reach the drop threshold, broadcasts that
        // mask along the broadcast axes, and selects between the scaled input and zero, which keeps dropped non-finite
        // elements at zero.
        check_count!("input", input_types, input_values.len(), ProgramError);
        check_count!("output", output_types, input_values.len(), ProgramError);
        let (input, input_type) = match (input_values, input_types) {
            ([input, ..], [input_type, ..]) if input_values.len() <= 2 => (*input, input_type),
            _ => return Err(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() }.into()),
        };
        if input_type.static_shape().is_none() {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: self.name().to_string(),
                reason: format!("`{DROPOUT_OPERATION_NAME}` with dynamic input extents"),
            }));
        }
        let context = lowerer.context;
        let location = lowerer.location;
        let block = &mut lowerer.block;
        let (algorithm, state) = match input_values.get(1) {
            Some(state) => (RandomAlgorithm::from_state_type(&input_types[1])?, *state),
            None => {
                let state_type = RandomAlgorithm::ThreeFry.state_type();
                let state_tensor_type = lower_tensor_type(&state_type, context, location)?;
                let state = lower_f64_constant_splat(0.0, &state_type, state_tensor_type, block, context, location)?;
                (RandomAlgorithm::ThreeFry, state)
            }
        };
        let mask_type = self.mask_type(input_type)?;
        let generated = lower_rng_bit_generator_to_mlir(algorithm, &mask_type, &[state], block, context, location)?;
        let mask_tensor_type = lower_tensor_type(&mask_type, context, location)?;
        let threshold = lower_f64_constant_splat(
            f64::from(self.drop_threshold()),
            &mask_type,
            mask_tensor_type,
            block,
            context,
            location,
        )?;
        let keep = block
            .append_operation(stable_hlo::compare(
                generated[1],
                threshold,
                stable_hlo::ComparisonDirection::GreaterThanOrEqual,
                stable_hlo::ComparisonType::Unsigned,
                location,
            )?)?
            .result(0)
            .expect("stablehlo.compare should return one result")
            .as_ref();
        let input_tensor_type = lower_tensor_type(input_type, context, location)?;
        let keep = if mask_type.shape() == input_type.shape() {
            keep
        } else {
            let keep_type = input_type.clone().with_data_type(DataType::Boolean).with_layout(None);
            let keep_tensor_type = lower_tensor_type(&keep_type, context, location)?;
            let dimensions = (0..input_type.rank()).collect::<Vec<_>>();
            block
                .append_operation(stable_hlo::broadcast(keep, keep_tensor_type, &dimensions, location)?)?
                .result(0)
                .expect("stablehlo.broadcast_in_dim should return one result")
                .as_ref()
        };

        // Complex scales and zeros combine real parts with one shared zero imaginary part.
        let part_data_type = match input_type.data_type() {
            DataType::C64 => Some(DataType::F32),
            DataType::C128 => Some(DataType::F64),
            _ => None,
        };
        let (scale, zero) = match part_data_type {
            None => (
                lower_f64_constant_splat(self.scale(), input_type, input_tensor_type, block, context, location)?,
                lower_f64_constant_splat(0.0, input_type, input_tensor_type, block, context, location)?,
            ),
            Some(part_data_type) => {
                let part_type = input_type.clone().with_data_type(part_data_type).with_layout(None);
                let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
                let part_zero = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
                let part_scale =
                    lower_f64_constant_splat(self.scale(), &part_type, part_tensor_type, block, context, location)?;
                let mut complex = |real: ValueRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, Error> {
                    Ok(block
                        .append_operation(stable_hlo::complex(real, part_zero, location)?)?
                        .result(0)
                        .expect("stablehlo.complex should return one result")
                        .as_ref())
                };
                (complex(part_scale)?, complex(part_zero)?)
            }
        };
        // A rate of one drops every element, after drawing the words that advance the state as for any other rate.
        if self.drops_everything() {
            return Ok(std::iter::once(zero).chain(input_values.get(1).map(|_| generated[0])).collect());
        }
        let scaled = block
            .append_operation(stable_hlo::multiply(input, scale, location)?)?
            .result(0)
            .expect("stablehlo.multiply should return one result")
            .as_ref();
        let output = block
            .append_operation(stable_hlo::select(keep, scaled, zero, location)?)?
            .result(0)
            .expect("stablehlo.select should return one result")
            .as_ref();
        Ok(std::iter::once(output).chain(input_values.get(1).map(|_| generated[0])).collect())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayType, Axes, ConvertElementTypeOperation, DataType, Dimension, Dropout, DropoutOperation,
        Placeholder, Shape, TypedProgram, ZeroOperation,
    };

    use crate::experimental::lowering::tests::{
        MixedValue, XlaProgramBuilder, execute_mixed_program, unproject_plain_program,
    };
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
    use crate::tests::execution_client;
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

        use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder};

        // An exact extent refines the declared dynamic bits axis to a static one, so the generator lowers at that
        // static shape and its redundant extent operand does not reach `stablehlo.rng_bit_generator`.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let mut builder = ArrayIrXlaProgramBuilder::new();
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
        use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
        use crate::operations::assertions::ensure_assertion_handler_registered;
        use crate::tests::execution_client;

        // An exact operand makes both repeated bits axes static, but the symbolic peer must still equal that extent.
        // The two generators cover both operand orders and consume the same state, so they produce the same bits.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
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
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]).unwrap(),
            vec![MixedValue::Array(bits.clone(), vec![3, 3]), MixedValue::Array(bits, vec![3, 3])],
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
        use crate::{Error, LowerableOperation, XlaLowerer};

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
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }))),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dropout() {
        // The mask is drawn at the input shape with the broadcast axes replaced by one, compared against the drop
        // threshold, and broadcast to the input shape before it selects between the scaled input and zero.
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let state_type = ryft_core::RandomAlgorithm::Philox.state_type();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let state = builder.add_input(state_type.clone());
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.25, [0]), Vec::new(), vec![input, state], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![input_type.clone(), state_type.clone()];
        let output_types = vec![input_type, state_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3xui64>) -> (tensor<2x3xf32>, tensor<3xui64>) {
                    %output_state, %output = stablehlo.rng_bit_generator %arg1, algorithm = PHILOX : (tensor<3xui64>) -> (tensor<3xui64>, tensor<1x3xui32>)
                    %c = stablehlo.constant dense<1073741824> : tensor<ui32>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1x3xui32>
                    %1 = stablehlo.compare GE, %output, %0, UNSIGNED : (tensor<1x3xui32>, tensor<1x3xui32>) -> tensor<1x3xi1>
                    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x3xi1>) -> tensor<2x3xi1>
                    %cst = stablehlo.constant dense<1.33333337> : tensor<f32>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
                    %5 = stablehlo.multiply %arg0, %3 : tensor<2x3xf32>
                    %6 = stablehlo.select %2, %5, %4 : tensor<2x3xi1>, tensor<2x3xf32>
                    return %6, %output_state : tensor<2x3xf32>, tensor<3xui64>
                  }
                }
            "#},
        );

        // Without a state, the mask is drawn from the fixed all-zero ThreeFry state, and complex scales and zeros
        // combine real parts with zero imaginary parts.
        let input_type = ArrayType::new_static(DataType::C64, [4]);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.5, Axes::default()), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        let types = vec![input_type];
        let module = to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
                    %c = stablehlo.constant dense<0> : tensor<ui64>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<2xui64>
                    %output_state, %output = stablehlo.rng_bit_generator %0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
                    %c_0 = stablehlo.constant dense<2147483648> : tensor<ui32>
                    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<4xui32>
                    %2 = stablehlo.compare GE, %output, %1, UNSIGNED : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xi1>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %5 = stablehlo.complex %4, %3 : tensor<4xcomplex<f32>>
                    %6 = stablehlo.complex %3, %3 : tensor<4xcomplex<f32>>
                    %7 = stablehlo.multiply %arg0, %5 : tensor<4xcomplex<f32>>
                    %8 = stablehlo.select %2, %7, %6 : tensor<4xi1>, tensor<4xcomplex<f32>>
                    return %8 : tensor<4xcomplex<f32>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_dropout_matches_eager_arrays() {
        // Compiled dropout draws exactly the masks and advanced states of the eager `Array` implementation, with and
        // without a state, with broadcast axes, and for a mask shape without even extents, for which ThreeFry splits
        // its largest axis.
        let input_type = ArrayType::new_static(DataType::F64, [4, 8]);
        let state_type = ryft_core::RandomAlgorithm::Philox.state_type();
        let values = (0..32).map(|value| f64::from(value) - 7.5).collect::<Vec<_>>();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let odd_input = builder.add_input(ArrayType::new_static(DataType::F64, [3, 5]).into());
        let state = builder
            .add_instruction(ZeroOperation::new(state_type.clone()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.5, Axes::default()), Vec::new(), vec![input, state], None)
            .unwrap()
            .to_vec();
        let advanced_state = builder
            .add_instruction(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::F64, false),
                Vec::new(),
                vec![outputs[1]],
                None,
            )
            .unwrap()[0];
        let broadcast =
            builder.add_instruction(DropoutOperation::new(0.25, [0]), Vec::new(), vec![input], None).unwrap()[0];
        let odd = builder
            .add_instruction(DropoutOperation::new(0.5, Axes::default()), Vec::new(), vec![odd_input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![outputs[0], advanced_state, broadcast, odd],
                vec![Placeholder; 2],
                vec![Placeholder; 4],
            )
            .unwrap();
        let odd_values = (0..15).map(|value| f64::from(value) + 0.5).collect::<Vec<_>>();
        let odd_output = Array::from_elements(ArrayType::new_static(DataType::F64, [3, 5]), &odd_values)
            .unwrap()
            .dropout(0.5, None)
            .unwrap()
            .0;
        let input = Array::from_elements(input_type, &values).unwrap();
        let (output, advanced_state) =
            input.dropout(0.5, Some(&Array::from_elements(state_type, &[0u64; 3]).unwrap())).unwrap();
        let (broadcast, _) = input.dropout_with_broadcast_axes(0.25, [0], None).unwrap();
        let advanced_state =
            advanced_state.unwrap().elements::<u64>().unwrap().into_iter().map(|value| value as f64).collect();
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Array(values, vec![4, 8]), MixedValue::Array(odd_values, vec![3, 5])],
                &[],
            )
            .unwrap(),
            vec![
                MixedValue::Array(output.to_f64s(), vec![4, 8]),
                MixedValue::Array(advanced_state, vec![3]),
                MixedValue::Array(broadcast.to_f64s(), vec![4, 8]),
                MixedValue::Array(odd_output.to_f64s(), vec![3, 5]),
            ],
        );
    }
}
