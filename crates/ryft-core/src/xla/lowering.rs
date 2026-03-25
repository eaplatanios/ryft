//! StableHLO/Shardy lowering helpers for traced XLA programs.

use std::collections::HashMap;

use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, Context as MlirContext, DenseElementsAttributeRef, Location, Operation, Region, Size as MlirSize,
    Type, TypeAndAttributes, TypeRef, Value, ValueRef,
};

use crate::parameters::Parameterized;
use crate::tracing_v2::{AtomSource, Graph, StagedOpRef};
use crate::types::{ArrayType, DataType, Size};

use super::LogicalMesh;
use super::shard_map::{
    LinearTensorShardMapOp, ShardMap, ShardMapConstantKind, ShardMapError, ShardMapTensor, StagedShardMapOp,
};
use super::sharding::{ShardingContext, ShardingError, normalize_mesh_symbol_name};

/// Error type for StableHLO/Shardy lowering.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum LoweringError {
    /// Underlying shard-map error returned while building manual-computation attributes.
    #[error("{0}")]
    ShardMapError(#[from] ShardMapError),

    /// Underlying sharding error returned while building mesh or sharding attributes.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a lowered function name is empty or contains whitespace.
    #[error("invalid function name '{function_name}' used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// Error returned when lowering encounters a traced tensor type that MLIR rejects.
    #[error("invalid tensor type '{array_type}' used during XLA lowering")]
    InvalidTensorType { array_type: ArrayType },

    /// Error returned when lowering encounters a staged op that produces an unsupported number of outputs.
    #[error("unsupported output count {output_count} for staged op '{op}' during XLA lowering")]
    UnsupportedOutputCount { op: String, output_count: usize },

    /// Error returned when lowering encounters a staged op that does not yet have StableHLO support.
    #[error("unsupported staged op '{op}' during XLA lowering")]
    UnsupportedOp { op: String },

    /// Error returned when lowering encounters a constant value that it does not know how to build.
    #[error("unsupported traced constant at atom %{atom_id} during XLA lowering")]
    UnsupportedConstant { atom_id: usize },

    /// Error returned when lowering encounters a type that does not have StableHLO support yet.
    #[error("unsupported data type '{data_type}' during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

    /// Error returned when lowering needs a staged value that was never assigned.
    #[error("missing lowered value for staged atom %{atom_id}")]
    MissingAtomValue { atom_id: usize },

    /// Error returned when MLIR rejects the constructed dense-elements attribute.
    #[error("invalid dense elements attribute for data type '{data_type}' during XLA lowering")]
    InvalidDenseElementsAttribute { data_type: DataType },

    /// Error returned when the constructed MLIR module fails verification.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    MlirVerificationFailure,

    /// Error returned when one traced XLA graph mixes shard maps from incompatible meshes.
    #[error("traced XLA lowering requires all nested shard maps to use the same logical mesh")]
    IncompatibleNestedMeshes,

    /// Error returned when simplifying a staged program prior to lowering fails.
    #[error("failed to simplify staged XLA program before lowering: {message}")]
    SimplificationFailure { message: String },
}

/// Lowers a traced shard-map program to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module<Input, Output, GraphInput, GraphOutput, S, M>(
    shard_map: &ShardMap,
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    global_input_types: &Input,
    local_input_types: &Input,
    global_output_types: &Output,
    _local_output_types: &Output,
    function_name: S,
    mesh_symbol_name: M,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
    M: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let local_input_types = local_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh_attribute = shard_map.mesh().to_shardy_mesh_attribute(&context);
    module.body().append_operation(shardy::mesh(mesh_symbol_name.as_str(), mesh_attribute, location));

    let function_arguments = global_input_tensor_types
        .iter()
        .zip(shard_map.in_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_shardy_tensor_sharding(
                mesh_symbol_name.as_str(),
                &context,
                ShardingContext::ExplicitSharding,
            )?;
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let function_results = global_output_tensor_types
        .iter()
        .zip(shard_map.out_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_shardy_tensor_sharding(
                mesh_symbol_name.as_str(),
                &context,
                ShardingContext::ExplicitSharding,
            )?;
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;

    module.body().append_operation({
        let mut function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let outer_inputs = (0..global_input_tensor_types.len())
            .map(|index| function_block.argument(index).expect("function block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        let manual_results = lower_manual_computation(
            &mut function_block,
            outer_inputs.as_slice(),
            shard_map,
            graph,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            mesh_symbol_name.as_str(),
            &context,
            location,
        )?;
        function_block.append_operation(func::r#return(manual_results.as_slice(), location));

        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

/// Lowers an arbitrary traced XLA graph to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module_for_graph<Input, Output, GraphInput, GraphOutput, S, M>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    mesh_symbol_name: M,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
    M: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    if let Some(mesh) = collect_nested_shard_map_mesh(graph, None)? {
        let mesh_attribute = mesh.to_shardy_mesh_attribute(&context);
        module.body().append_operation(shardy::mesh(mesh_symbol_name.as_str(), mesh_attribute, location));
    }

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let function_arguments = global_input_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();
    let function_results = global_output_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();

    module.body().append_operation({
        let mut function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let outputs = lower_graph_outputs(graph, &mut function_block, mesh_symbol_name.as_str(), &context, location)?;
        function_block.append_operation(func::r#return(outputs.as_slice(), location));
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }
    Ok(module.to_string())
}

fn collect_nested_shard_map_mesh<GraphInput, GraphOutput>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError>
where
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
{
    let mut mesh = existing;
    for equation in graph.equations() {
        if let Some(shard_map_op) = equation.op.as_any().downcast_ref::<StagedShardMapOp>() {
            match &mesh {
                Some(existing_mesh) if existing_mesh != shard_map_op.body.shard_map.mesh() => {
                    return Err(LoweringError::IncompatibleNestedMeshes);
                }
                Some(_) => {}
                None => mesh = Some(shard_map_op.body.shard_map.mesh().clone()),
            }
            mesh = collect_nested_shard_map_mesh(shard_map_op.body.compiled.graph(), mesh)?;
            continue;
        }
        if let Some(linear_shard_map_op) = equation.op.as_any().downcast_ref::<LinearTensorShardMapOp>() {
            match &mesh {
                Some(existing_mesh) if existing_mesh != linear_shard_map_op.eval_body.shard_map.mesh() => {
                    return Err(LoweringError::IncompatibleNestedMeshes);
                }
                Some(_) => {}
                None => mesh = Some(linear_shard_map_op.eval_body.shard_map.mesh().clone()),
            }
            mesh = collect_nested_shard_map_mesh(linear_shard_map_op.eval_body.compiled.graph(), mesh)?;
        }
    }
    Ok(mesh)
}

/// Lowers one traced graph to values inside a block.
fn lower_graph_outputs<'b, 'c: 'b, 't: 'c, B, GraphInput, GraphOutput, L>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    block: &mut B,
    mesh_symbol_name: &str,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    L: Location<'c, 't> + Copy,
{
    let mut atom_values = vec![None; graph.atom_count()];
    for (input_index, atom_id) in graph.input_atoms().iter().copied().enumerate() {
        atom_values[atom_id] = Some(block.argument(input_index).expect("body block arguments should exist").as_ref());
    }

    let mut equation_by_first_output = vec![None; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        if let Some(first_output) = equation.outputs.first() {
            equation_by_first_output[*first_output] = Some(equation_index);
        }
    }

    for atom_id in 0..graph.atom_count() {
        let atom = graph.atom(atom_id).expect("atom IDs should be dense");
        let lowered_value = match &atom.source {
            AtomSource::Input => None,
            AtomSource::Constant => Some(lower_constant(atom_id, &atom.example_value, block, context, location)?),
            AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                if equation.outputs.len() != 1 {
                    return Err(LoweringError::UnsupportedOutputCount {
                        op: equation.op.name().to_string(),
                        output_count: equation.outputs.len(),
                    });
                }
                let inputs = equation
                    .inputs
                    .iter()
                    .map(|input| atom_values[*input].ok_or(LoweringError::MissingAtomValue { atom_id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                Some(lower_equation(
                    graph,
                    equation_index,
                    inputs.as_slice(),
                    block,
                    mesh_symbol_name,
                    context,
                    location,
                )?)
            }
        };
        if let Some(lowered_value) = lowered_value {
            atom_values[atom_id] = Some(lowered_value);
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| atom_values[*output].ok_or(LoweringError::MissingAtomValue { atom_id: *output }))
        .collect::<Result<Vec<_>, _>>()
}

/// Lowers one `sdy.manual_computation` operation, including its nested body graph.
fn lower_manual_computation<'b, 'c: 'b, 't: 'c, B, GraphInput, GraphOutput, L>(
    block: &mut B,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    shard_map: &ShardMap,
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    mesh_symbol_name: &str,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    L: Location<'c, 't> + Copy,
{
    let local_input_tensor_types = local_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;

    let mut body_region = context.region();
    let mut body_block = context.block(
        local_input_tensor_types
            .iter()
            .map(|tensor_type| (*tensor_type, location))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let body_outputs = lower_graph_outputs(graph, &mut body_block, mesh_symbol_name, context, location)?;
    body_block.append_operation(shardy::r#return(body_outputs.as_slice(), location));
    body_region.append_block(body_block);

    let manual_computation = block.append_operation(shardy::manual_computation(
        outer_inputs,
        global_output_tensor_types.as_slice(),
        shard_map.to_shardy_in_shardings(mesh_symbol_name, context)?,
        shard_map.to_shardy_out_shardings(mesh_symbol_name, context)?,
        shard_map.to_shardy_manual_axes(context),
        body_region,
        location,
    ));
    Ok(manual_computation.results().map(|result| result.as_ref()).collect::<Vec<_>>())
}

/// Lowers a traced constant atom to a StableHLO constant operation and returns its result value.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    atom_id: usize,
    value: &ShardMapTensor,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let constant_kind = value.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id })?;
    let tensor_type = lower_tensor_type(value.r#type(), context, location)?;
    let elements = lower_constant_elements_attribute(value.r#type().data_type, tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers one traced equation to the corresponding StableHLO operation and returns its result value.
fn lower_equation<'b, 'c: 'b, 't: 'c, B, GraphInput, GraphOutput, L>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    equation_index: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    mesh_symbol_name: &str,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    L: Location<'c, 't> + Copy,
{
    let equation = &graph.equations()[equation_index];
    if let Some(shard_map_op) = equation.op.as_any().downcast_ref::<StagedShardMapOp>() {
        let simplified_body = shard_map_op
            .body
            .simplified()
            .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
        let results = lower_manual_computation(
            block,
            input_values,
            &simplified_body.shard_map,
            simplified_body.compiled.graph(),
            simplified_body.local_input_types.as_slice(),
            simplified_body.global_output_types.as_slice(),
            mesh_symbol_name,
            context,
            location,
        )?;
        if results.len() != 1 {
            return Err(LoweringError::UnsupportedOutputCount {
                op: equation.op.name().to_string(),
                output_count: results.len(),
            });
        }
        return Ok(results[0]);
    }
    if let Some(linear_shard_map_op) = equation.op.as_any().downcast_ref::<LinearTensorShardMapOp>() {
        let simplified_body = linear_shard_map_op
            .eval_body
            .simplified()
            .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
        let results = lower_manual_computation(
            block,
            input_values,
            &simplified_body.shard_map,
            simplified_body.compiled.graph(),
            simplified_body.local_input_types.as_slice(),
            simplified_body.global_output_types.as_slice(),
            mesh_symbol_name,
            context,
            location,
        )?;
        if results.len() != 1 {
            return Err(LoweringError::UnsupportedOutputCount {
                op: equation.op.name().to_string(),
                output_count: results.len(),
            });
        }
        return Ok(results[0]);
    }

    let output_type = lower_tensor_type(
        &graph.atom(equation.outputs[0]).expect("equation output should exist").abstract_value,
        context,
        location,
    )?;
    let operation = match equation.op.name() {
        "add" => block.append_operation(stable_hlo::add(input_values[0], input_values[1], location)),
        "mul" => block.append_operation(stable_hlo::multiply(input_values[0], input_values[1], location)),
        "neg" => block.append_operation(stable_hlo::negate(input_values[0], location)),
        "sin" => block.append_operation(stable_hlo::sine(input_values[0], stable_hlo::Accuracy::Default, location)),
        "cos" => block.append_operation(stable_hlo::cosine(input_values[0], stable_hlo::Accuracy::Default, location)),
        "matmul" => block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            None,
            None,
            output_type,
            location,
        )),
        "matrix_transpose" => block.append_operation(stable_hlo::transpose(input_values[0], &[1, 0], location)),
        op => return Err(LoweringError::UnsupportedOp { op: op.to_string() }),
    };
    Ok(operation.result(0).expect("supported staged ops should return one result").as_ref())
}

/// Normalizes a user-provided MLIR symbol name.
fn normalize_function_name(function_name: &str) -> Result<String, LoweringError> {
    let function_name = function_name.trim();
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(LoweringError::InvalidFunctionName { function_name: function_name.to_string() });
    }
    Ok(function_name.strip_prefix('@').unwrap_or(function_name).to_string())
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    let element_type = lower_element_type(array_type.data_type, context)?;
    let dimensions = array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => MlirSize::Static(*value),
            Size::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    context
        .tensor_type(element_type, dimensions.as_slice(), None, location)
        .ok_or_else(|| LoweringError::InvalidTensorType { array_type: array_type.clone() })
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
fn lower_element_type<'c, 't>(
    data_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<TypeRef<'c, 't>, LoweringError> {
    Ok(match data_type {
        DataType::Token => return Err(LoweringError::UnsupportedDataType { data_type }),
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::I1 => context.signless_integer_type(1).as_ref(),
        DataType::I2 => context.signless_integer_type(2).as_ref(),
        DataType::I4 => context.signless_integer_type(4).as_ref(),
        DataType::I8 => context.signless_integer_type(8).as_ref(),
        DataType::I16 => context.signless_integer_type(16).as_ref(),
        DataType::I32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 => context.signless_integer_type(64).as_ref(),
        DataType::U1 => context.unsigned_integer_type(1).as_ref(),
        DataType::U2 => context.unsigned_integer_type(2).as_ref(),
        DataType::U4 => context.unsigned_integer_type(4).as_ref(),
        DataType::U8 => context.unsigned_integer_type(8).as_ref(),
        DataType::U16 => context.unsigned_integer_type(16).as_ref(),
        DataType::U32 => context.unsigned_integer_type(32).as_ref(),
        DataType::U64 => context.unsigned_integer_type(64).as_ref(),
        DataType::BF16 => context.bfloat16_type().as_ref(),
        DataType::F16 => context.float16_type().as_ref(),
        DataType::F32 => context.float32_type().as_ref(),
        DataType::F64 => context.float64_type().as_ref(),
        DataType::F4E2M1FN => context.float4e2m1fn_type().as_ref(),
        DataType::F8E3M4 => context.float8e3m4_type().as_ref(),
        DataType::F8E4M3 => context.float8e4m3_type().as_ref(),
        DataType::F8E4M3FN => context.float8e4m3fn_type().as_ref(),
        DataType::F8E4M3FNUZ => context.float8e4m3fnuz_type().as_ref(),
        DataType::F8E4M3B11FNUZ => context.float8e4m3b11fnuz_type().as_ref(),
        DataType::F8E5M2 => context.float8e5m2_type().as_ref(),
        DataType::F8E5M2FNUZ => context.float8e5m2fnuz_type().as_ref(),
        DataType::F8E8M0FNU => context.float8e8m0fnu_type().as_ref(),
        DataType::C64 => context.complex_type(context.float32_type()).as_ref(),
        DataType::C128 => context.complex_type(context.float64_type()).as_ref(),
    })
}

/// Builds the dense-elements attribute for one traced splat constant.
fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    constant_kind: ShardMapConstantKind,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    let integer_value = match constant_kind {
        ShardMapConstantKind::Zero => 0,
        ShardMapConstantKind::One => 1,
    };
    let float_value = integer_value as f64;

    match data_type {
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(integer_value != 0))
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.unsigned_integer_type(unsigned_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.bfloat16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F32 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float32_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F64 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float64_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F4E2M1FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float4e2m1fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E3M4 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e3m4_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3B11FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3b11fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E8M0FNU => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e8m0fnu_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Returns the bit width of a signed integer [`DataType`].
fn signed_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::I1 => 1,
        DataType::I2 => 2,
        DataType::I4 => 4,
        DataType::I8 => 8,
        DataType::I16 => 16,
        DataType::I32 => 32,
        DataType::I64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

/// Returns the bit width of an unsigned integer [`DataType`].
fn unsigned_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::U1 => 1,
        DataType::U2 => 2,
        DataType::U4 => 4,
        DataType::U8 => 8,
        DataType::U16 => 16,
        DataType::U32 => 32,
        DataType::U64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::{FloatExt, MatrixOps, OneLike, ZeroLike};
    use crate::types::{MeshAxisType, Shape};

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use super::super::sharding::{LogicalMesh, MeshAxis, PartitionDimension, PartitionSpec};
    use super::*;

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::with_type(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]), None)
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None)
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
        mesh_symbol_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name, mesh_symbol_name)
    }

    #[test]
    fn test_to_mlir_module_renders_a_full_add_module() {
        let global_input_type = test_vector_type(8);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| x.clone() + x,
            global_input_type,
            test_manual_mesh("x", 4),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "main", "mesh").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| {
                let product = x.clone().transpose_matrix().matmul(x);
                let waveform = (-product).cos().sin();
                (waveform.clone() * waveform.one_like()) + waveform.zero_like()
            },
            global_input_type,
            test_manual_mesh("x", 2),
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "kernel", "mesh").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @kernel(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
                      %6 = stablehlo.multiply %5, %cst : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
                      %7 = stablehlo.add %6, %cst_0 : tensor<4x4xf32>
                      sdy.return %7 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#}
        );
    }
}
