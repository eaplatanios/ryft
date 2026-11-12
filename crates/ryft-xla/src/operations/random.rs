use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, DROPOUT_OPERATION_NAME, DataType, Dimension as CoreDimension, DropoutOperation,
    Operation as CoreOperation, ProgramError, RNG_BIT_GENERATOR_OPERATION_NAME, RandomAlgorithm,
    RngBitGeneratorOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::operations::assertions::lower_output_extent_validation;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError};

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
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
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
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
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
            .filter(|dimension| matches!(dimension, CoreDimension::Dynamic(_)))
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
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
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
        Array as CoreArray, ArrayType, Axes, CategoricalSamplingMode, ConvertElementType, ConvertElementTypeOperation,
        DataType, DeviceMesh, Dimension as CoreDimension, Dropout, DropoutOperation, Placeholder, Random,
        RandomAlgorithm, RngBitGenerator, Shape, TypedProgram, ZeroOperation,
    };

    use crate::domains::tests::domain_mesh;
    use crate::experimental::lowering::tests::{
        MixedValue, ProgramBuilder, execute_mixed_program, unproject_plain_program,
    };
    use crate::experimental::ops::{ArrayConstant, Constant, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::{Array, Domain, Session, to_mlir_module_for_program};

    // Uploads one generator state to the device and builds the same state on the reference backend.
    fn rng_states<'c>(
        domain: &Domain<'c>,
        mesh: &DeviceMesh,
        algorithm: RandomAlgorithm,
        state_values: &[u64],
    ) -> (Array<'c>, CoreArray) {
        let bytes = values_to_bytes(state_values);
        let state = replicated_array(domain, mesh, DataType::U64, &[state_values.len()], bytes.as_slice());
        (state, CoreArray::from_elements(algorithm.state_type(), state_values).unwrap())
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_rng_bit_generator() {
        use ryft_core::{RandomAlgorithm, RngBitGeneratorOperation};

        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![CoreDimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![CoreDimension::Static(4)]));
        let mut builder = ProgramBuilder::new();
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
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
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

        use crate::experimental::ops::{Constant, Operation, ProgramBuilder as ArrayIrProgramBuilder};

        // An exact extent refines the declared dynamic bits axis to a static one, so the generator lowers at that
        // static shape and its redundant extent operand does not reach `stablehlo.rng_bit_generator`.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let mut builder = ArrayIrProgramBuilder::new();
        let state = builder.add_input(state_type.clone().into());
        let extent =
            builder.add_constant(Constant::Dimension(DimensionValue::new(DimensionType::from(extent), 3).unwrap()));
        let outputs = builder
            .add_instruction(
                Operation::RngBitGenerator(RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type)),
                Vec::new(),
                vec![state, extent],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
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
        use crate::experimental::ops::{Constant, Operation, ProgramBuilder as ArrayIrProgramBuilder};
        use crate::operations::assertions::ensure_assertion_handler_registered;
        use crate::tests::execution_client;

        // An exact operand makes both repeated bits axes static, but the symbolic peer must still equal that extent.
        // The two generators cover both operand orders and consume the same state, so they produce the same bits.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(9)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![size.clone().into(), size.clone().into()]));
        let mut builder = ArrayIrProgramBuilder::new();
        let symbolic = builder.add_input(DimensionType::from(size).into());
        let exact = builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let state = builder
            .add_instruction(ZeroOperation::new(RandomAlgorithm::ThreeFry.state_type()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let mut outputs = Vec::new();
        for inputs in [vec![state, symbolic, exact], vec![state, exact, symbolic]] {
            let bits = builder
                .add_instruction(
                    Operation::RngBitGenerator(RngBitGeneratorOperation::new(
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
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
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
        use crate::experimental::ops::Operation;
        use crate::{Error, LowerableOperation, Lowerer};

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
        let mut lowerer: Lowerer<'_, '_, '_, Operation> = Lowerer::new(block.as_ref(), &context, location.as_ref());
        let output_types = vec![state_type.into(), ArrayType::new_static(DataType::U32, [3]).into()];
        assert!(matches!(
            operation.lower(&mut lowerer, &inputs, &[], &output_types),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
            ),
        ));
    }

    #[test]
    fn test_lower_rng_bit_generator_matches_reference_backend() {
        // Bits and advanced states are byte-identical on both backends. Five 32-bit words exercise the padded counter
        // blocks and the truncated word layout. For outputs narrower than 64 bits, XLA's ThreeFry expansion splits the
        // first even-sized axis of the output (or else its first largest axis) and advances the counter by the size
        // of the resulting half shape, which the last group covers at several shapes and every narrow width.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        for (algorithm, state_values, outputs) in [
            (RandomAlgorithm::ThreeFry, vec![42u64, 7], vec![(DataType::U32, vec![5]), (DataType::U64, vec![3])]),
            (RandomAlgorithm::Philox, vec![42u64, 7, 9], vec![(DataType::U32, vec![5]), (DataType::U64, vec![3])]),
            (
                RandomAlgorithm::ThreeFry,
                vec![(1u64 << 32) | 42, u64::from(u32::MAX)],
                vec![
                    (DataType::U32, vec![]),
                    (DataType::U32, vec![2, 3]),
                    (DataType::U32, vec![3, 4]),
                    (DataType::U32, vec![3, 5]),
                    (DataType::U32, vec![3, 3]),
                    (DataType::U32, vec![3, 2, 5]),
                    (DataType::U16, vec![2, 3]),
                    (DataType::U8, vec![3, 3]),
                ],
            ),
        ] {
            let (state, reference_state) = rng_states(&domain, &mesh, algorithm, &state_values);
            for (data_type, dimensions) in outputs {
                let output_type = ArrayType::new_static(data_type, dimensions);
                let (device_advanced_state, device_bits) = state.rng_bit_generator(algorithm, &output_type).unwrap();
                let (reference_advanced_state, reference_bits) =
                    reference_state.rng_bit_generator(algorithm, &output_type).unwrap();
                assert_eq!(
                    device_advanced_state.copy_to_host().unwrap(),
                    reference_advanced_state.logical_bytes(),
                    "{algorithm} state for {output_type}",
                );
                assert_eq!(
                    device_bits.copy_to_host().unwrap(),
                    reference_bits.logical_bytes(),
                    "{algorithm} bits for {output_type}",
                );
            }
        }
    }

    #[test]
    fn test_lower_rng_bit_generator_matches_reference_backend_at_counter_boundaries() {
        // Empty outputs preserve the state, and counter advancement wraps at the counter width of the algorithm while
        // the bits stay byte-identical with the reference backend for every output width.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        for (algorithm, state_values) in [
            (RandomAlgorithm::ThreeFry, vec![42u64, u64::MAX - 1]),
            (RandomAlgorithm::Philox, vec![42u64, u64::MAX - 1, 7]),
            (RandomAlgorithm::Philox, vec![42u64, u64::MAX - 1, u64::MAX]),
        ] {
            let (state, reference_state) = rng_states(&domain, &mesh, algorithm, &state_values);
            for data_type in [DataType::U8, DataType::U16, DataType::U32, DataType::U64] {
                for dimensions in [vec![2, 0, 3], vec![5]] {
                    // The empty shape runs no cipher invocations. Five elements run enough invocations to cross the
                    // low counter boundary, and Philox carries into the high half, which can itself wrap to zero.
                    let element_count = dimensions.iter().product::<usize>();
                    let invocation_count = match (algorithm, data_type) {
                        (RandomAlgorithm::ThreeFry, DataType::U64) => element_count,
                        (RandomAlgorithm::ThreeFry, _) => element_count.div_ceil(2),
                        (RandomAlgorithm::Philox, DataType::U64) => element_count.div_ceil(2),
                        (RandomAlgorithm::Philox, _) => element_count.div_ceil(4),
                    };
                    let (low_counter, carry) = state_values[1].overflowing_add(invocation_count as u64);
                    let mut expected_state = vec![state_values[0], low_counter];
                    if algorithm == RandomAlgorithm::Philox {
                        expected_state.push(state_values[2].wrapping_add(u64::from(carry)));
                    }

                    let output_type = ArrayType::new_static(data_type, dimensions);
                    let (device_advanced_state, device_bits) =
                        state.rng_bit_generator(algorithm, &output_type).unwrap();
                    let (_, reference_bits) = reference_state.rng_bit_generator(algorithm, &output_type).unwrap();
                    assert_eq!(
                        values_from_bytes::<u64>(&device_advanced_state.copy_to_host().unwrap()),
                        expected_state,
                        "{algorithm} state for {output_type}",
                    );
                    assert_eq!(
                        device_bits.copy_to_host().unwrap(),
                        reference_bits.logical_bytes(),
                        "{algorithm} bits for {output_type}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_split_rng_key_matches_reference_backend() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let (state, reference_state) = rng_states(&domain, &mesh, RandomAlgorithm::ThreeFry, &[11u64, 3]);
        let (device_advanced_state, device_keys) = state.split_rng_key(2).unwrap();
        let (reference_advanced_state, reference_keys) = reference_state.split_rng_key(2).unwrap();
        assert_eq!(device_advanced_state.copy_to_host().unwrap(), reference_advanced_state.logical_bytes());
        assert_eq!(
            device_keys.iter().map(|key| key.copy_to_host().unwrap()).collect::<Vec<_>>(),
            reference_keys.iter().map(CoreArray::logical_bytes).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_random_uniform_matches_reference_backend() {
        // Every step of the composition is exact arithmetic over byte-identical ThreeFry bits, so the samples keep
        // their exact encodings on both backends, including the narrow floating-point types.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let (state, reference_state) = rng_states(&domain, &mesh, RandomAlgorithm::ThreeFry, &[11u64, 3]);
        for data_type in [DataType::F32, DataType::BF16, DataType::F16] {
            let sample_type = ArrayType::new_static(data_type, [8]);
            let (device_advanced_state, device_samples) = state.random_uniform(&sample_type).unwrap();
            let (reference_advanced_state, reference_samples) = reference_state.random_uniform(&sample_type).unwrap();
            assert_eq!(device_samples.data_type(), data_type);
            assert_eq!(device_samples.copy_to_host().unwrap(), reference_samples.logical_bytes(), "{data_type}");
            assert_eq!(
                device_advanced_state.copy_to_host().unwrap(),
                reference_advanced_state.logical_bytes(),
                "{data_type}",
            );
        }
    }

    #[test]
    fn test_random_normal_matches_reference_backend() {
        // The Box-Muller transform goes through transcendental functions, whose last bits may differ between the
        // backends. Complex samples consist of two independently drawn components.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let (state, reference_state) = rng_states(&domain, &mesh, RandomAlgorithm::ThreeFry, &[11u64, 3]);
        for (data_type, component_count, tolerance) in
            [(DataType::F32, 8, 1e-5f64), (DataType::C64, 16, 1e-5), (DataType::C128, 16, 1e-12)]
        {
            let sample_type = ArrayType::new_static(data_type, [8]);
            let (device_advanced_state, device_samples) = state.random_normal(&sample_type).unwrap();
            let (reference_advanced_state, reference_samples) = reference_state.random_normal(&sample_type).unwrap();
            assert_eq!(device_samples.data_type(), data_type);
            assert_eq!(
                device_advanced_state.copy_to_host().unwrap(),
                reference_advanced_state.logical_bytes(),
                "{data_type}",
            );
            let device_bytes = device_samples.copy_to_host().unwrap();
            let reference_bytes = reference_samples.logical_bytes();
            let (device_components, reference_components) = if data_type == DataType::C128 {
                (values_from_bytes::<f64>(&device_bytes), values_from_bytes::<f64>(&reference_bytes))
            } else {
                (
                    values_from_bytes::<f32>(&device_bytes).into_iter().map(f64::from).collect::<Vec<_>>(),
                    values_from_bytes::<f32>(&reference_bytes).into_iter().map(f64::from).collect::<Vec<_>>(),
                )
            };
            assert_eq!(device_components.len(), component_count);
            assert_eq!(reference_components.len(), component_count);
            for (index, (device_component, reference_component)) in
                device_components.iter().zip(&reference_components).enumerate()
            {
                assert!(
                    (device_component - reference_component).abs() < tolerance,
                    "{data_type} component {index} is {device_component} but the reference backend computed \
                     {reference_component}",
                );
            }
        }
    }

    #[test]
    fn test_random_categorical_matches_reference_backend() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let (state, reference_state) = rng_states(&domain, &mesh, RandomAlgorithm::ThreeFry, &[11u64, 3]);
        let logit_values = [0f32, 10.0, -3.0, 2.0];
        let logits = replicated_array(&domain, &mesh, DataType::F32, &[4], values_to_bytes(&logit_values).as_slice());
        let reference_logits =
            CoreArray::from_elements(ArrayType::new_static(DataType::F32, [4]), &logit_values).unwrap();
        let (device_advanced_state, device_samples) = state.random_categorical(&logits, 0).unwrap();
        let (reference_advanced_state, reference_samples) =
            reference_state.random_categorical(&reference_logits, 0).unwrap();
        assert_eq!(values_from_bytes::<i32>(&device_samples.copy_to_host().unwrap()), vec![1]);
        assert_eq!(device_samples.copy_to_host().unwrap(), reference_samples.logical_bytes());
        assert_eq!(device_advanced_state.copy_to_host().unwrap(), reference_advanced_state.logical_bytes());
    }

    #[test]
    fn test_random_categorical_with_mode_matches_reference_backend() {
        // High-precision sampling never selects a category masked with `-inf` and advances the same state on both
        // backends.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        for (algorithm, state_values) in
            [(RandomAlgorithm::ThreeFry, vec![11u64, 3]), (RandomAlgorithm::Philox, vec![11u64, 3, 0])]
        {
            let (state, reference_state) = rng_states(&domain, &mesh, algorithm, &state_values);
            for data_type in [DataType::F32, DataType::F64] {
                let reference_logits =
                    CoreArray::matrix(2, 3, vec![f64::NEG_INFINITY, 2000.0, 0.0, f64::NEG_INFINITY, 0.0, 2000.0])
                        .unwrap()
                        .convert_element_type(data_type)
                        .unwrap();
                let logits =
                    replicated_array(&domain, &mesh, data_type, &[2, 3], reference_logits.logical_bytes().as_slice());
                let (device_advanced_state, device_samples) =
                    state.random_categorical_with_mode(&logits, -1, CategoricalSamplingMode::HighPrecision).unwrap();
                let (reference_advanced_state, _) = reference_state
                    .random_categorical_with_mode(&reference_logits, -1, CategoricalSamplingMode::HighPrecision)
                    .unwrap();
                assert_eq!(
                    values_from_bytes::<i32>(&device_samples.copy_to_host().unwrap()),
                    vec![1, 2],
                    "{algorithm} {data_type}",
                );
                assert_eq!(
                    device_advanced_state.copy_to_host().unwrap(),
                    reference_advanced_state.logical_bytes(),
                    "{algorithm} {data_type}",
                );
            }
        }

        // A zero first uniform exercises the rare tail where an ordinary `log(1 - x)` loses the distinction from zero.
        let (state, _) = rng_states(&domain, &mesh, RandomAlgorithm::ThreeFry, &[0u64, 0x1badd8a64e5859ae]);
        let logits = replicated_array(&domain, &mesh, DataType::F32, &[2], values_to_bytes(&[-16f32, 0.0]).as_slice());
        for (mode, expected) in
            [(CategoricalSamplingMode::LowPrecision, 1i32), (CategoricalSamplingMode::HighPrecision, 0)]
        {
            let (_, samples) = state.random_categorical_with_mode(&logits, 0, mode).unwrap();
            assert_eq!(values_from_bytes::<i32>(&samples.copy_to_host().unwrap()), vec![expected], "{mode:?}");
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dropout() {
        // The mask is drawn at the input shape with the broadcast axes replaced by one, compared against the drop
        // threshold, and broadcast to the input shape before it selects between the scaled input and zero.
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let state_type = ryft_core::RandomAlgorithm::Philox.state_type();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let state = builder.add_input(state_type.clone());
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.25, [0]), Vec::new(), vec![input, state], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
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
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.5, Axes::default()), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder])
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
        let mut builder = ArrayIrProgramBuilder::new();
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
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![outputs[0], advanced_state, broadcast, odd],
                vec![Placeholder; 2],
                vec![Placeholder; 4],
            )
            .unwrap();
        let odd_values = (0..15).map(|value| f64::from(value) + 0.5).collect::<Vec<_>>();
        let odd_output = CoreArray::from_elements(ArrayType::new_static(DataType::F64, [3, 5]), &odd_values)
            .unwrap()
            .dropout(0.5, None)
            .unwrap()
            .0;
        let input = CoreArray::from_elements(input_type, &values).unwrap();
        let (output, advanced_state) =
            input.dropout(0.5, Some(&CoreArray::from_elements(state_type, &[0u64; 3]).unwrap())).unwrap();
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
