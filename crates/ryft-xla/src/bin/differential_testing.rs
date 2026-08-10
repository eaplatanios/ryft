//! Emits Ryft observations for the pinned JAX differential-testing harness.
//!
//! This binary is deliberately a test tool rather than library API. It executes a fixed case registry and writes one
//! versioned JSON record per case. `python/scripts/compare_behavior_with_jax.py` builds matching JAX records and
//! compares values, staging capabilities, and the semantic collective subset of each emitted StableHLO module.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;

use serde::Serialize;

use ryft_core::operations::collectives::{
    AllGather, AllGatherOutputVariance, AllToAll, CollectiveOptions, PSumScatter, PSwapAxes, Pshuffle,
};
use ryft_core::{
    Array as CpuArray, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType,
    BatchAxis, BatchingContext, BatchingTracer, ConvertElementTypeOperation, DataType, Device, DeviceMesh, Dimension,
    DimensionBounds, DimensionFromScalarOperation, DimensionValue, DimensionVariable, DynamicShapeSliceOperation,
    EagerContext, LogicalMesh, MeshAxis, MeshAxisType, Placeholder, ProgramBuilder, ProgramError, ReduceOperation,
    ReductionKind, Shape, Sharding, ShardingDimension,
};
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{BufferType, Client, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};
use ryft_xla::experimental::{ShardMapTracer, TracedXlaProgram, shard_map, trace};
use ryft_xla::{Array, FromPjrt};

/// Schema version emitted by this binary and accepted by the Python comparison harness.
const SCHEMA: &str = "ryft-jax-differential-v1";

/// One framework's staging result for a differential case.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
enum StagingObservation {
    /// The program staged successfully with the rendered type of its first result.
    Supported { output_type: String },
}

/// One Ryft-side case record consumed by the Python comparison harness.
#[derive(Clone, Debug, Serialize, PartialEq)]
struct DifferentialObservation {
    /// Versioned schema identifier.
    schema: &'static str,

    /// Stable case identifier shared with the JAX registry.
    case_id: &'static str,

    /// Named outputs, represented as one flattened logical value vector per participating device or eager execution.
    observations: BTreeMap<&'static str, Vec<Vec<f32>>>,

    /// Staging result when the case compares staging capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    staging: Option<StagingObservation>,

    /// Raw StableHLO module. The Python harness compares only its semantic collective projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    stablehlo: Option<String>,
}

/// Descriptor for one fixed Ryft observation case.
#[derive(Copy, Clone)]
struct DifferentialCase {
    /// Stable case identifier shared with the JAX registry.
    case_id: &'static str,

    /// Callback that executes and records the case.
    emit: fn() -> Result<DifferentialObservation, Box<dyn Error>>,
}

/// Returns the fixed case registry and verifies that case IDs are unique.
fn registry() -> Vec<DifferentialCase> {
    let cases = vec![
        DifferentialCase { case_id: "grouped_shape_changing_collectives", emit: emit_grouped_collectives },
        DifferentialCase { case_id: "pshuffle", emit: emit_pshuffle },
        DifferentialCase { case_id: "pswapaxes", emit: emit_pswapaxes },
        DifferentialCase { case_id: "data_dependent_prefix_take", emit: emit_data_dependent_prefix_take },
    ];
    for (index, case) in cases.iter().enumerate() {
        assert!(
            cases[..index].iter().all(|previous| previous.case_id != case.case_id),
            "duplicate differential-testing case ID '{}'",
            case.case_id,
        );
    }
    cases
}

/// Converts a slice of plain-old-data values to native-endian bytes for PJRT host transfers.
fn values_to_bytes<V: Copy>(values: &[V]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        // SAFETY: `value` points at one live, properly aligned `V` owned by `values`. This private helper is only
        // used with `f32`, whose object representation is plain old data with no padding, so all `size_of::<V>()`
        // bytes are initialized. The byte slice is copied by `extend_from_slice` and never retained.
        let value_bytes = unsafe { std::slice::from_raw_parts(value as *const V as *const u8, size_of::<V>()) };
        bytes.extend_from_slice(value_bytes);
    }
    bytes
}

/// Decodes native-endian `f32` values copied from a PJRT buffer.
fn f32_values_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Returns the four-device manual mesh shared by the collective cases.
fn collective_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
}

/// Returns XLA SPMD compilation options for one four-partition program.
fn collective_compilation_options() -> CompilationOptions {
    CompilationOptions {
        argument_layouts: Vec::new(),
        parameter_is_tupled_arguments: false,
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: 4,
            use_spmd_partitioning: true,
            use_shardy_partitioner: true,
            ..Default::default()
        }),
        compile_portable_executable: false,
        profile_version: 0,
        serialized_multi_slice_configuration: Vec::new(),
        environment_option_overrides: HashMap::new(),
        target_config: None,
        allow_in_place_mlir_modification: false,
        matrix_unit_operand_precision: Precision::Default as i32,
    }
}

/// Executes one already-lowered four-device collective module and returns flattened outputs in device order.
///
/// # Parameters
///
///   - `client`: Four-device CPU PJRT client.
///   - `device_mesh`: Physical devices arranged according to [`collective_mesh`].
///   - `sharding`: Global input sharding over the manual mesh axis.
///   - `module`: StableHLO/Shardy module to compile.
///   - `global_shape`: Global logical input shape.
///   - `local_shape`: Per-device input-buffer shape.
///   - `values`: One flattened local input vector per device.
fn execute_collective_module(
    client: &Client<'_>,
    device_mesh: DeviceMesh,
    sharding: Sharding,
    module: &str,
    global_shape: &[usize],
    local_shape: &[u64],
    values: &[Vec<f32>],
) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn Error>> {
    let client_devices = client.addressable_devices()?;
    if values.len() != client_devices.len() {
        return Err(format!("expected {} per-device inputs but got {}", client_devices.len(), values.len()).into());
    }
    let buffers = client_devices
        .iter()
        .zip(values)
        .map(|(device, values)| {
            client.buffer(values_to_bytes(values).as_slice(), BufferType::F32, local_shape, None, device.clone(), None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_type =
        ArrayType::new(DataType::F32, Shape::new(global_shape.iter().copied().map(Dimension::Static).collect()))
            .with_sharding(sharding)?;
    let input = Array::from_addressable_buffers(client, input_type, device_mesh, buffers)?;
    let executable =
        client.compile(&Program::Mlir { bytecode: module.as_bytes().to_vec() }, &collective_compilation_options())?;
    let execution_device_ids =
        executable.addressable_devices()?.iter().map(|device| device.id()).collect::<Result<Vec<_>, _>>()?;
    let arguments = Array::into_execute_arguments(vec![input], execution_device_ids.as_slice())?;
    let outputs = executable
        .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)?
        .block_until_ready()?;
    outputs
        .into_iter()
        .map(|output| {
            output
                .outputs
                .into_iter()
                .map(|buffer| buffer.copy_to_host(None)?.r#await().map(|bytes| f32_values_from_bytes(bytes.as_slice())))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Creates the CPU client, logical-to-physical mesh, and sharding shared by collective emitters.
fn collective_runtime() -> Result<(Client<'static>, DeviceMesh, Sharding), Box<dyn Error>> {
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))?;
    let client_devices = client.addressable_devices()?;
    let devices = client_devices.iter().map(Device::from_pjrt).collect::<Result<Vec<_>, _>>()?;
    let mesh = collective_mesh();
    let device_mesh = DeviceMesh::new(mesh.clone(), devices)?;
    let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])?;
    Ok((client, device_mesh, sharding))
}

/// Emits grouped all-gather, sum-scatter, and all-to-all behavior plus their StableHLO module.
fn emit_grouped_collectives() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, sharding) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let traced: TracedXlaProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
        {
            let sharding = sharding.clone();
            move |input: ShardMapTracer| {
                shard_map::<_, _, (ArrayType, ArrayType, ArrayType), _>(
                    |local_input: ShardMapTracer| {
                        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
                        (
                            local_input
                                .all_gather_with_options("x", 0, options.clone(), AllGatherOutputVariance::Varying)
                                .unwrap(),
                            local_input.clone().psum_scatter_with_options("x", 0, options.clone()).unwrap(),
                            local_input.all_to_all_with_options("x", 0, 0, options).unwrap(),
                        )
                    },
                    input,
                    mesh.clone(),
                    sharding.clone(),
                    (sharding.clone(), sharding.clone(), sharding.clone()),
                )
                .unwrap()
            }
        },
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)])),
    )?;
    let stablehlo = traced.to_mlir_module("main")?;
    let input_values = (0..4)
        .map(|device| (0..4).map(|offset| (device * 4 + offset) as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let outputs = execute_collective_module(
        &client,
        device_mesh,
        sharding,
        stablehlo.as_str(),
        &[16],
        &[4],
        input_values.as_slice(),
    )?;
    let observations = BTreeMap::from([
        ("all_gather", outputs.iter().map(|device| device[0].clone()).collect()),
        ("psum_scatter", outputs.iter().map(|device| device[1].clone()).collect()),
        ("all_to_all", outputs.iter().map(|device| device[2].clone()).collect()),
    ]);
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "grouped_shape_changing_collectives",
        observations,
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Emits `pshuffle` behavior plus its canonical `collective_permute` StableHLO module.
fn emit_pshuffle() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, sharding) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let sharding = sharding.clone();
            move |input: ShardMapTracer| {
                shard_map::<_, _, ArrayType, _>(
                    |local_input: ShardMapTracer| local_input.pshuffle("x", &[2, 0, 3, 1]).unwrap(),
                    input,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap()
            }
        },
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
    )?;
    let stablehlo = traced.to_mlir_module("main")?;
    let input_values = (0..4)
        .map(|device| (0..2).map(|offset| (device * 2 + offset) as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let outputs = execute_collective_module(
        &client,
        device_mesh,
        sharding,
        stablehlo.as_str(),
        &[8],
        &[2],
        input_values.as_slice(),
    )?;
    type Parent = EagerContext<ArrayIrValue<CpuArray>, ArrayIrOperation<CpuArray>>;
    let context = BatchingContext::<_, ArrayIrBatching>::new(
        Parent::new(),
        ArrayIrValue::Dimension(DimensionValue::constant(4)?),
    )
    .with_axis_name("x".to_string());
    let input = ArrayIrBatch::new(
        ArrayIrValue::Array(CpuArray::matrix(4, 2, (0..8).map(|value| value as f32).collect())),
        BatchAxis::new(0),
    )?;
    let output = BatchingTracer::new(context, input).pshuffle("x", &[2, 0, 3, 1])?.into_batch();
    let ArrayIrValue::Array(output) = output.into_value() else {
        unreachable!("pshuffle preserves the array member kind")
    };
    let surface_output = output
        .to_f64s()
        .chunks_exact(2)
        .map(|values| values.iter().map(|value| *value as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let lowered_output = outputs.iter().map(|device| device[0].clone()).collect::<Vec<_>>();
    if surface_output != lowered_output {
        return Err("pshuffle composition disagrees with its canonical ppermute lowering".into());
    }
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "pshuffle",
        observations: BTreeMap::from([("output", surface_output)]),
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Emits `pswapaxes` behavior plus its canonical `all_to_all` StableHLO module.
fn emit_pswapaxes() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, _) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let sharding =
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])?;
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let sharding = sharding.clone();
            move |input: ShardMapTracer| {
                shard_map::<_, _, ArrayType, _>(
                    |local_input: ShardMapTracer| local_input.pswapaxes("x", 0).unwrap(),
                    input,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap()
            }
        },
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16), Dimension::Static(2)])),
    )?;
    let stablehlo = traced.to_mlir_module("main")?;
    let input_values = (0..4)
        .map(|device| (0..8).map(|offset| (device * 8 + offset) as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let outputs = execute_collective_module(
        &client,
        device_mesh,
        sharding,
        stablehlo.as_str(),
        &[16, 2],
        &[4, 2],
        input_values.as_slice(),
    )?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "pswapaxes",
        observations: BTreeMap::from([("output", outputs.into_iter().map(|device| device[0].clone()).collect())]),
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Builds the bounded data-dependent prefix program shared by its eager and staged observations.
fn data_dependent_prefix_program() -> Result<
    ryft_core::Program<
        ArrayIrValue<CpuArray>,
        ArrayIrOperation<CpuArray>,
        Vec<ArrayIrValue<CpuArray>>,
        Vec<ArrayIrValue<CpuArray>>,
    >,
    ProgramError,
> {
    let mut builder = ProgramBuilder::<ArrayIrValue<CpuArray>, ArrayIrOperation<CpuArray>>::new();
    let mask = builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])).into());
    let values = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])).into());
    let mask = builder.add_instruction(
        ArrayIrOperation::Array(ArrayOperation::from(ConvertElementTypeOperation::<ArrayType>::new(DataType::I64))),
        Vec::new(),
        vec![mask],
    )?[0];
    let count = builder.add_instruction(
        ArrayIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
        Vec::new(),
        vec![mask],
    )?[0];
    let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5))?);
    let count = builder.add_instruction(DimensionFromScalarOperation::new(count_variable), Vec::new(), vec![count])?[0];
    let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0)?));
    let output =
        builder.add_instruction(DynamicShapeSliceOperation::new(1), Vec::new(), vec![values, start, count])?[0];
    builder.build::<Vec<ArrayIrValue<CpuArray>>, Vec<ArrayIrValue<CpuArray>>>(
        vec![output],
        vec![Placeholder, Placeholder],
        vec![Placeholder],
    )
}

/// Emits Ryft's eager and staged behavior for `n = count(mask); take(values, n)`.
fn emit_data_dependent_prefix_take() -> Result<DifferentialObservation, Box<dyn Error>> {
    let program = data_dependent_prefix_program()?;
    let execute = |mask| -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let [ArrayIrValue::Array(output)]: [ArrayIrValue<CpuArray>; 1] = program
            .interpret(vec![
                ArrayIrValue::Array(CpuArray::vector(mask)),
                ArrayIrValue::Array(CpuArray::vector(vec![10.0_f32, 20.0, 30.0, 40.0])),
            ])?
            .try_into()
            .unwrap()
        else {
            unreachable!("the prefix program has one array output")
        };
        Ok(vec![output.to_f64s().into_iter().map(|value| value as f32).collect()])
    };
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "data_dependent_prefix_take",
        observations: BTreeMap::from([
            ("two_matches", execute(vec![true, false, true, false])?),
            ("zero_matches", execute(vec![false, false, false, false])?),
        ]),
        staging: Some(StagingObservation::Supported { output_type: program.output_types()[0].to_string() }),
        stablehlo: None,
    })
}

/// Parses selected case IDs, emits deterministic JSON records, and returns an error for an unknown case.
fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let cases = registry();
    let mut requested = Vec::new();
    let mut list = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--list" => {
                list = true;
                index += 1;
            }
            "--case" if index + 1 < arguments.len() => {
                requested.push(arguments[index + 1].as_str());
                index += 2;
            }
            argument => return Err(format!("expected '--list' or '--case CASE_ID' but got '{argument}'").into()),
        }
    }
    if list {
        if !requested.is_empty() {
            return Err("'--list' cannot be combined with '--case'".into());
        }
        for case in cases {
            println!("{}", case.case_id);
        }
        return Ok(());
    }
    let selected = if requested.is_empty() {
        cases
    } else {
        requested
            .into_iter()
            .map(|case_id| {
                cases
                    .iter()
                    .copied()
                    .find(|case| case.case_id == case_id)
                    .ok_or_else(|| format!("unknown differential-testing case '{case_id}'"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut records = selected.into_iter().map(|case| (case.emit)()).collect::<Result<Vec<_>, _>>()?;
    records.sort_by_key(|record| record.case_id);
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

fn main() {
    if let Err(error) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_registry() {
        assert_eq!(
            registry().into_iter().map(|case| case.case_id).collect::<Vec<_>>(),
            vec!["grouped_shape_changing_collectives", "pshuffle", "pswapaxes", "data_dependent_prefix_take",],
        );
    }

    #[test]
    fn test_data_dependent_prefix_take() {
        assert_eq!(
            emit_data_dependent_prefix_take().unwrap(),
            DifferentialObservation {
                schema: SCHEMA,
                case_id: "data_dependent_prefix_take",
                observations: BTreeMap::from(
                    [("two_matches", vec![vec![10.0, 20.0]]), ("zero_matches", vec![vec![]]),]
                ),
                staging: Some(StagingObservation::Supported { output_type: "f32[count]".to_string() }),
                stablehlo: None,
            },
        );
    }
}
