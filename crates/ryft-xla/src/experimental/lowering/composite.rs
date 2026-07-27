//! StableHLO lowering for programs that mix arrays with first-class dimensions.

use ryft_core::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
use ryft_core::parameters::Parameterized;
use ryft_core::programs::{Operation as CoreOperation, Program, ProgramError, Typed};
use ryft_core::sharding::LogicalMesh;
use ryft_core::types::{ArrayProgramType, ArrayType, DataType, Dimension};
use ryft_mlir::dialects::{func, stable_hlo};
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, Region, Type as MlirType, TypeAndAttributes,
    Value as MlirValue, ValueRef,
};

use super::{
    LowerableXlaOperation, LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode,
    lower_constant_elements_attribute, lower_constant_output, lower_tensor_type, merge_logical_meshes,
    normalize_function_name, replay_region_ref_into_block,
};
use crate::experimental::ops::XlaConstant;
use crate::mlir::ToMlir;

/// Error returned while lowering a stored program that mixes arrays and first-class dimensions.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArrayProgramLoweringError {
    /// Program validation or replay failure.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// StableHLO construction, validation, or backend-specific lowering failure.
    #[error("{message}")]
    Lowering { message: String },
}

impl From<LoweringError> for ArrayProgramLoweringError {
    fn from(error: LoweringError) -> Self {
        match error {
            LoweringError::Tracing(error) => Self::Program(error),
            error => Self::Lowering { message: error.to_string() },
        }
    }
}

/// Lowers a composite array-program type to its physical StableHLO tensor type.
fn lower_array_program_type<'c, 't, L: Location<'c, 't>>(
    r#type: &ArrayProgramType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    match r#type {
        ArrayProgramType::Array(r#type) => lower_tensor_type(r#type, context, location),
        ArrayProgramType::Dimension(_) => lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location),
    }
}

/// Lowers one composite constant without converting first-class dimensions into array-program authority.
fn lower_array_program_constant<'b, 'c: 'b, 't: 'c, A: MlirLowerableValue>(
    value: &ArrayProgramValue<A>,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    match value {
        ArrayProgramValue::Array(value) => value.lower_constant_value(&[], block, context, location),
        ArrayProgramValue::Dimension(value) => {
            let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
            let extent = i64::try_from(value.extent()).map_err(|_| LoweringError::UnsupportedOp {
                op: format!("dimension extent {} does not fit StableHLO i64", value.extent()),
            })?;
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
            let operation = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(operation.result(0).expect("stablehlo.constant should return one result").as_ref())
        }
    }
}

/// Lowers one array-program instruction.
fn lower_array_program_operation<'b, 'c: 'b, 't: 'c, A>(
    operation: &ArrayProgramOperation<A>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayProgramType],
    output_types: &[ArrayProgramType],
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    A: MlirLowerableValue,
{
    match operation {
        ArrayProgramOperation::Zero(_) => {
            let output_types = output_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            lower_constant_output(output_types.as_slice(), 0, block, context, location)
        }
        ArrayProgramOperation::Array(operation) => {
            let input_types = input_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = output_types
                .iter()
                .map(|r#type| {
                    <&ArrayType>::try_from(r#type).cloned().map_err(|error| LoweringError::Tracing(error.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut lowerer = PlainMlirLowerer::new(*block, context, location).with_input_types(input_types);
            operation.lower_to_mlir(
                input_values,
                &[],
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )
        }
        ArrayProgramOperation::Dimension(operation) => Err(LoweringError::UnsupportedOp {
            op: format!("first-class dimension operation `{}` has not been lowered yet", operation.name()),
        }),
        ArrayProgramOperation::DimensionSize(operation) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            let [input_type] = input_types else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_types.len() }.into());
            };
            let input_type =
                <&ArrayType>::try_from(input_type).map_err(|error| LoweringError::Tracing(error.into()))?;
            match input_type.shape().dimensions()[operation.axis()] {
                Dimension::Static(extent) => {
                    let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                    let extent = i64::try_from(extent).map_err(|_| LoweringError::UnsupportedOp {
                        op: format!("dimension extent {extent} does not fit StableHLO i64"),
                    })?;
                    let elements = lower_constant_elements_attribute(DataType::I64, output_type, extent, context)?;
                    let operation = block.append_operation(stable_hlo::constant(elements, location)?)?;
                    Ok(vec![operation.result(0).expect("stablehlo.constant should return one result").as_ref()])
                }
                Dimension::Dynamic(_) => {
                    let size =
                        block.append_operation(stable_hlo::get_dimension_size(*input, operation.axis(), location)?)?;
                    let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
                    let output_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
                    let converted = block.append_operation(stable_hlo::convert(size, output_type, location)?)?;
                    Ok(vec![converted.result(0).expect("stablehlo.convert should return one result").as_ref()])
                }
            }
        }
        ArrayProgramOperation::DimensionToScalar(_) => {
            let [input] = input_values else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            };
            Ok(vec![*input])
        }
    }
}

/// Lowers a stored array program containing ordinary arrays and first-class dimensions to verified StableHLO.
///
/// Array atoms lower to their tensor representation and dimension atoms lower to scalar `i64` tensors. Mixed
/// operations retain their explicit SSA data flow: in particular, a
/// [`DimensionToScalarOperation`](ryft_core::DimensionToScalarOperation) lowers to the identity because both sides of
/// that logical boundary use the same physical scalar representation.
pub fn lower_array_program_to_stable_hlo<
    Input: Parameterized<ArrayProgramValue<XlaConstant>>,
    Output: Parameterized<ArrayProgramValue<XlaConstant>>,
>(
    program: &Program<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>, Input, Output>,
    function_name: &str,
) -> Result<String, ArrayProgramLoweringError> {
    to_mlir_module_for_array_program(program, function_name).map_err(Into::into)
}

/// Generic implementation of [`lower_array_program_to_stable_hlo`] shared with the reference-backend tests.
fn to_mlir_module_for_array_program<
    A,
    Input: Parameterized<ArrayProgramValue<A>>,
    Output: Parameterized<ArrayProgramValue<A>>,
>(
    program: &Program<ArrayProgramValue<A>, ArrayProgramOperation<A>, Input, Output>,
    function_name: &str,
) -> Result<String, LoweringError>
where
    A: MlirLowerableValue,
{
    let function_name = normalize_function_name(function_name)?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    let mut mesh: Option<LogicalMesh> = None;
    for region in program.regions().iter() {
        for atom in region.atoms() {
            let atom_type = atom.r#type();
            let ArrayProgramType::Array(r#type) = atom_type.as_ref() else {
                continue;
            };
            let Some(sharding) = r#type.sharding() else {
                continue;
            };
            mesh = Some(match mesh.take() {
                Some(existing) => merge_logical_meshes(&existing, sharding.mesh())?,
                None => sharding.mesh().clone(),
            });
        }
    }
    if let Some(mesh) = mesh {
        module.body()?.append_operation(mesh.to_mlir(location)?)?;
    }

    let input_types = program
        .input_ids()
        .iter()
        .map(|id| lower_array_program_type(program.atoms()[id.index()].r#type().as_ref(), &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let output_types = program
        .output_ids()
        .iter()
        .map(|id| lower_array_program_type(program.atoms()[id.index()].r#type().as_ref(), &context, location))
        .collect::<Result<Vec<_>, _>>()?;

    module.body()?.append_operation({
        let function_block =
            context.block(input_types.iter().map(|r#type| (r#type.as_ref(), location)).collect::<Vec<_>>().as_slice());
        {
            let mut block = function_block.as_ref();
            let arguments = (0..input_types.len())
                .map(|index| block.argument(index).expect("function block arguments should exist").as_ref())
                .collect::<Vec<_>>();
            let outputs = replay_region_ref_into_block(
                program.entry_region_ref(),
                arguments,
                &mut block,
                &context,
                location.as_ref(),
                |_, value, block, context, location| lower_array_program_constant(value, block, context, location),
                |instruction, inputs, block, context, location| {
                    if !instruction.regions().is_empty() {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "composite operation `{}` with attached regions is not supported",
                                instruction.operation().name(),
                            ),
                        });
                    }
                    let input_types = instruction
                        .inputs()
                        .iter()
                        .map(|id| program.atoms()[id.index()].r#type().into_owned())
                        .collect::<Vec<_>>();
                    let output_types = instruction
                        .outputs()
                        .iter()
                        .map(|id| program.atoms()[id.index()].r#type().into_owned())
                        .collect::<Vec<_>>();
                    lower_array_program_operation(
                        instruction.operation(),
                        inputs,
                        input_types.as_slice(),
                        output_types.as_slice(),
                        block,
                        context,
                        location,
                    )
                },
            )?;
            block.append_operation(func::r#return(outputs.as_slice(), location)?)?;
        }
        let mut region = context.region();
        region.append_block(function_block)?;
        func::func(
            function_name.as_str(),
            func::FuncAttributes {
                arguments: input_types
                    .iter()
                    .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                    .collect(),
                results: output_types
                    .iter()
                    .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                    .collect(),
                ..Default::default()
            },
            region,
            location,
        )?
    })?;

    if !module.verify()? {
        return Err(LoweringError::MlirVerificationFailure);
    }
    Ok(module.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use ryft_core::contexts::StagingContext;
    use ryft_core::operations::dimensions::{DimensionSize, DimensionToScalar};
    use ryft_core::parameters::Placeholder;
    use ryft_core::tracing::TracingContext;
    use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
    use ryft_core::types::{DataType, Shape};
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use super::*;
    use crate::tests::{values_from_bytes, values_to_bytes};

    #[test]
    fn test_array_program_lowering() {
        type TestContext = TracingContext<ArrayProgramValue<XlaConstant>, ArrayProgramOperation<XlaConstant>>;

        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(17)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let context = TestContext::new();
        let input = context.input(input_type.into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let dynamic_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(dynamic_module.matches("stablehlo.get_dimension_size").count(), 1);
        assert_eq!(dynamic_module.matches("stablehlo.convert").count(), 1);
        assert_eq!(dynamic_module.matches("dimension_to_scalar").count(), 0);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(7)]));
        let context = TestContext::new();
        let input = context.input(input_type.into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<XlaConstant>>, Vec<ArrayProgramValue<XlaConstant>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let static_module = lower_array_program_to_stable_hlo(&program, "main").unwrap();
        assert_eq!(static_module.matches("stablehlo.constant dense<7> : tensor<i64>").count(), 1);
        assert_eq!(static_module.matches("stablehlo.get_dimension_size").count(), 0);
        assert_eq!(static_module.matches("stablehlo.convert").count(), 0);

        // Compile and execute the static path on the real CPU plugin. The bounded-dynamic module above is verified as
        // StableHLO, but the CPU XLA plugin currently lowers bounded arguments through its unavailable `PadToStatic`
        // custom call. Static execution still proves that the first-class dimension remains scalar SSA through
        // `dimension_to_scalar`, without a host readback or reconstruction step.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let options = CompilationOptions {
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
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: static_module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_values = [0.0_f32; 7];
        let input_bytes = values_to_bytes(input_values.as_slice());
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F32, &[7], None, device, None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<i64>(output_bytes.as_slice()), vec![7]);
    }
}
