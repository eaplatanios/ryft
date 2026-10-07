//! Emits Ryft observations for the pinned JAX differential-testing harness.
//!
//! This binary is deliberately a test tool rather than library API. It executes a fixed case registry and writes one
//! versioned JSON record per case. `python/scripts/compare_behavior_with_jax.py` builds matching JAX records and
//! compares values, staging capabilities, and declared semantic contracts in each emitted StableHLO module.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;

use serde::Serialize;

use ryft_core::operations::attention::{
    AttentionConfiguration, AttentionImplementation, AttentionInputs, DotProductAttention,
};
use ryft_core::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType,
    BatchAxis, BatchingContext, BatchingTracer, CollectiveOptions, Compare, ComparisonDirection,
    ConvertElementTypeOperation, DataType, Device, DeviceMesh, Differentiate, Dimension, DimensionBounds,
    DimensionFromScalarOperation, DimensionValue, DimensionVariable, DotDimensionNumbers, DynamicSlice,
    DynamicSliceOperation, DynamicUpdateSlice, EagerContext, LogicalMesh, MeshAxis, MeshAxisType, ParallelAllGather,
    ParallelAllGatherOutputVariance, ParallelAllToAll, ParallelPermute, ParallelReduce, ParallelSumScatter,
    Placeholder, ProgramBuilder, ProgramError, Reduce, ReduceOperation, ReductionKind, ScaledDot, Shape, Sharding,
    ShardingDimension, Value, ValueProjection, ZeroLike, condition,
};
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{BufferType, Client, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};
use ryft_xla::experimental::{ShardMapTracer, TracedXlaProgram, shard_map, trace};
use ryft_xla::{FromPjrt, XlaArray, XlaSession};

#[path = "differential_testing/collectives.rs"]
mod collectives;

#[path = "differential_testing/ragged.rs"]
mod ragged;

#[path = "differential_testing/distributed_collectives.rs"]
mod distributed_collectives;

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
    case_id: String,

    /// Named outputs, represented as one flattened logical value vector per participating device or eager execution.
    observations: BTreeMap<&'static str, Vec<Vec<f32>>>,

    /// Staging result when the case compares staging capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    staging: Option<StagingObservation>,

    /// Raw StableHLO module interpreted through the semantic contract declared by the shared case registry.
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

    /// Whether this legacy case belongs to the collective suite.
    collective: bool,
}

/// Returns the fixed case registry and verifies that case IDs are unique.
fn registry() -> Vec<DifferentialCase> {
    let cases = vec![
        DifferentialCase {
            case_id: "grouped_shape_changing_collectives",
            emit: emit_grouped_collectives,
            collective: true,
        },
        DifferentialCase { case_id: "pshuffle", emit: emit_parallel_shuffle, collective: true },
        DifferentialCase { case_id: "pswapaxes", emit: emit_parallel_swap_axes, collective: true },
        DifferentialCase {
            case_id: "data_dependent_prefix_take",
            emit: emit_data_dependent_prefix_take,
            collective: false,
        },
        DifferentialCase { case_id: "scaled_dot_and_matmul", emit: emit_scaled_dot_and_matmul, collective: false },
        DifferentialCase { case_id: "dot_product_attention", emit: emit_dot_product_attention, collective: false },
        DifferentialCase { case_id: "negative_dynamic_slice", emit: emit_negative_dynamic_slice, collective: false },
        DifferentialCase {
            case_id: "condition_varying_predicate_gradient",
            emit: emit_condition_varying_predicate_gradient,
            collective: false,
        },
    ];
    for (index, case) in cases.iter().enumerate() {
        assert!(
            cases[..index].iter().all(|previous| previous.case_id != case.case_id),
            "duplicate differential-testing case ID `{}`",
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

/// Returns XLA SPMD compilation options for the requested participant count.
fn collective_compilation_options(partition_count: usize) -> CompilationOptions {
    CompilationOptions {
        argument_layouts: Vec::new(),
        parameter_is_tupled_arguments: false,
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: partition_count as i64,
            use_spmd_partitioning: true,
            use_shardy_partitioner: true,
            ..Default::default()
        }),
        compile_portable_executable: false,
        profile_version: 0,
        individually_defined_output_indices: Vec::new(),
        serialized_multi_slice_configuration: Vec::new(),
        environment_option_overrides: HashMap::new(),
        target_config: None,
        allow_in_place_mlir_modification: false,
        matrix_unit_operand_precision: Precision::Default as i32,
    }
}

/// Executes one already-lowered collective module and returns flattened outputs in device order.
///
/// # Parameters
///
///   - `client`: CPU PJRT client with one device per participant.
///   - `device_mesh`: Physical devices arranged according to [`collective_mesh`].
///   - `sharding`: Global input sharding over the manual mesh axis.
///   - `module`: StableHLO/Shardy module to compile.
///   - `global_shape`: Global logical input shape.
///   - `local_shape`: Per-device input-buffer shape.
///   - `values`: One flattened local input vector per device.
///   - `expected_output_shapes`: Physical output shapes checked against returned PJRT buffers before decoding.
fn execute_collective_module(
    client: &Client<'_>,
    device_mesh: DeviceMesh,
    sharding: Sharding,
    module: &str,
    global_shape: &[usize],
    local_shape: &[u64],
    values: &[Vec<f32>],
    expected_output_shapes: &[Vec<usize>],
) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn Error>> {
    execute_collective_inputs(
        client,
        device_mesh,
        module,
        &[(sharding, global_shape, local_shape, values)],
        expected_output_shapes,
    )
}

/// Executes a lowered collective module with one independently shaped buffer family per logical input.
fn execute_collective_inputs(
    client: &Client<'_>,
    device_mesh: DeviceMesh,
    module: &str,
    inputs: &[(Sharding, &[usize], &[u64], &[Vec<f32>])],
    expected_output_shapes: &[Vec<usize>],
) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn Error>> {
    let client_devices = client.addressable_devices()?;
    let mut arrays = Vec::new();
    for (sharding, global_shape, local_shape, values) in inputs {
        if values.len() != client_devices.len() {
            return Err(format!("expected {} per-device inputs but got {}", client_devices.len(), values.len()).into());
        }
        let buffers = client_devices
            .iter()
            .zip(*values)
            .map(|(device, values)| {
                client.buffer(
                    values_to_bytes(values).as_slice(),
                    BufferType::F32,
                    *local_shape,
                    None,
                    device.clone(),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_type = ArrayType::new_static(DataType::F32, global_shape.to_vec()).with_sharding(sharding.clone())?;
        arrays.push(XlaArray::from_addressable_buffers(
            &XlaSession::new(client).domain(),
            input_type,
            device_mesh.clone(),
            buffers,
        )?);
    }
    let executable = client.compile(
        &Program::Mlir { bytecode: module.as_bytes().to_vec() },
        &collective_compilation_options(client_devices.len()),
    )?;
    let execution_device_ids =
        executable.addressable_devices()?.iter().map(|device| device.id()).collect::<Result<Vec<_>, _>>()?;
    let arguments = XlaArray::into_execute_arguments(arrays, execution_device_ids.as_slice())?;
    let outputs = executable
        .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)?
        .block_until_ready()?;
    let mut observations = Vec::new();
    for (participant, output) in outputs.into_iter().enumerate() {
        if output.outputs.len() != expected_output_shapes.len() {
            return Err(format!(
                "participant {participant} returned {} outputs, expected {}",
                output.outputs.len(),
                expected_output_shapes.len()
            )
            .into());
        }
        let mut values = Vec::new();
        for (index, buffer) in output.outputs.into_iter().enumerate() {
            if buffer.element_type()? != BufferType::F32 {
                return Err(format!(
                    "participant {participant} output {index} has element type {:?}, expected `F32`",
                    buffer.element_type()?
                )
                .into());
            }
            let actual = buffer.dimensions()?.iter().map(|extent| *extent as usize).collect::<Vec<_>>();
            if actual != expected_output_shapes[index] {
                return Err(format!(
                    "participant {participant} output {index} has shape {actual:?}, expected {:?}",
                    expected_output_shapes[index]
                )
                .into());
            }
            values.push(f32_values_from_bytes(&buffer.copy_to_host(None)?.r#await()?));
        }
        observations.push(values);
    }
    Ok(observations)
}

/// Creates the CPU client, logical-to-physical mesh, and sharding shared by collective emitters.
fn collective_runtime() -> Result<(Client<'static>, DeviceMesh, Sharding), Box<dyn Error>> {
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))?;
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
                                .parallel_all_gather_with_options(
                                    "x",
                                    0,
                                    options.clone(),
                                    ParallelAllGatherOutputVariance::Varying,
                                )
                                .unwrap(),
                            local_input.clone().parallel_sum_scatter_with_options("x", 0, options.clone()).unwrap(),
                            local_input.parallel_all_to_all_with_options("x", 0, 0, options).unwrap(),
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
        &[vec![8], vec![2], vec![4]],
    )?;
    let observations = BTreeMap::from([
        ("all_gather", outputs.iter().map(|device| device[0].clone()).collect()),
        ("psum_scatter", outputs.iter().map(|device| device[1].clone()).collect()),
        ("all_to_all", outputs.iter().map(|device| device[2].clone()).collect()),
    ]);
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "grouped_shape_changing_collectives".into(),
        observations,
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Emits `parallel_shuffle` behavior plus its canonical `collective_permute` StableHLO module.
fn emit_parallel_shuffle() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, sharding) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let sharding = sharding.clone();
            move |input: ShardMapTracer| {
                shard_map::<_, _, ArrayType, _>(
                    |local_input: ShardMapTracer| local_input.parallel_shuffle("x", &[2, 0, 3, 1]).unwrap(),
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
        &[vec![2]],
    )?;
    type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
    let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
        Parent::new(),
        ArrayIrValue::Dimension(DimensionValue::constant(4)?),
    )
    .with_axis_name("x".to_string());
    let input = ArrayIrBatch::new(
        ArrayIrValue::Array(Array::matrix(4, 2, (0..8).map(|value| value as f32).collect())?),
        BatchAxis::new(0),
    )?;
    let output = BatchingTracer::new(context, input).parallel_shuffle("x", &[2, 0, 3, 1])?.into_batch();
    let ArrayIrValue::Array(output) = output.into_value() else {
        unreachable!("parallel_shuffle preserves the array member kind")
    };
    let surface_output = output
        .to_f64s()
        .chunks_exact(2)
        .map(|values| values.iter().map(|value| *value as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let lowered_output = outputs.iter().map(|device| device[0].clone()).collect::<Vec<_>>();
    if surface_output != lowered_output {
        return Err("`parallel_shuffle` composition disagrees with its canonical `parallel_permute` lowering".into());
    }
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "pshuffle".into(),
        observations: BTreeMap::from([("output", surface_output)]),
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Emits `parallel_swap_axes` behavior plus its canonical `parallel_all_to_all` StableHLO module.
fn emit_parallel_swap_axes() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, _) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let sharding =
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])?;
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let sharding = sharding.clone();
            move |input: ShardMapTracer| {
                shard_map::<_, _, ArrayType, _>(
                    |local_input: ShardMapTracer| local_input.parallel_swap_axes("x", 0).unwrap(),
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
        &[vec![4, 2]],
    )?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "pswapaxes".into(),
        observations: BTreeMap::from([("output", outputs.into_iter().map(|device| device[0].clone()).collect())]),
        staging: None,
        stablehlo: Some(stablehlo),
    })
}

/// Builds the bounded data-dependent prefix program shared by its eager and staged observations.
fn data_dependent_prefix_program() -> Result<
    ryft_core::Program<
        ArrayIrValue<Array>,
        ArrayIrOperation<Array>,
        Vec<ArrayIrValue<Array>>,
        Vec<ArrayIrValue<Array>>,
    >,
    ProgramError,
> {
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
    let mask = builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])).into());
    let values = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])).into());
    let mask = builder.add_instruction(
        ArrayIrOperation::Array(ArrayOperation::from(ConvertElementTypeOperation::<ArrayType>::new(
            DataType::I64,
            false,
        ))),
        Vec::new(),
        vec![mask],
        None,
    )?[0];
    let count = builder.add_instruction(
        ArrayIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
        Vec::new(),
        vec![mask],
        None,
    )?[0];
    let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5))?);
    let count =
        builder.add_instruction(DimensionFromScalarOperation::new(count_variable), Vec::new(), vec![count], None)?[0];
    let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0)?));
    let output = builder.add_instruction(
        DynamicSliceOperation::<ArrayIrType>::from_rank(1),
        Vec::new(),
        vec![values, start, count],
        None,
    )?[0];
    builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
        vec![output],
        vec![Placeholder, Placeholder],
        vec![Placeholder],
    )
}

/// Emits Ryft's eager and staged behavior for `n = count(mask); take(values, n)`.
fn emit_data_dependent_prefix_take() -> Result<DifferentialObservation, Box<dyn Error>> {
    let program = data_dependent_prefix_program()?;
    let execute = |mask| -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let [ArrayIrValue::Array(output)]: [ArrayIrValue<Array>; 1] = program
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(mask)?),
                ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0, 40.0])?),
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
        case_id: "data_dependent_prefix_take".into(),
        observations: BTreeMap::from([
            ("two_matches", execute(vec![true, false, true, false])?),
            ("zero_matches", execute(vec![false, false, false, false])?),
        ]),
        staging: Some(StagingObservation::Supported { output_type: program.output_types()[0].to_string() }),
        stablehlo: None,
    })
}

/// Emits generalized scaled-dot and rank-three scaled-matmul values plus the named-composite StableHLO contract.
fn emit_scaled_dot_and_matmul() -> Result<DifferentialObservation, Box<dyn Error>> {
    let lhs = Array::from_elements::<f32>(
        ArrayType::new_static(DataType::F32, [2, 4]),
        &(1..=8).map(|value| value as f32).collect::<Vec<_>>(),
    )?;
    let rhs = Array::from_elements::<f32>(
        ArrayType::new_static(DataType::F32, [4, 3]),
        &(1..=12).map(|value| value as f32).collect::<Vec<_>>(),
    )?;
    let lhs_scale = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 2]), &[1.0, 2.0, 0.5, 1.0])?;
    let rhs_scale =
        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 3]), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])?;
    let dimensions = DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new());
    let values = |value: Array| value.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();

    let matmul_lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 4]), &[1.0; 4])?;
    let matmul_rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 4]), &[1.0; 8])?;
    let matmul_lhs_scale = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[1.0; 2])?;
    let matmul_rhs_scale = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 2]), &[1.0; 4])?;
    let observations = BTreeMap::from([
        (
            "both_scales",
            vec![values(lhs.scaled_dot(
                &rhs,
                Some(&lhs_scale),
                Some(&rhs_scale),
                Some(&dimensions),
                Some(DataType::F32),
            )?)],
        ),
        (
            "lhs_scale",
            vec![values(lhs.scaled_dot(&rhs, Some(&lhs_scale), None, Some(&dimensions), Some(DataType::F32))?)],
        ),
        (
            "rhs_scale",
            vec![values(lhs.scaled_dot(&rhs, None, Some(&rhs_scale), Some(&dimensions), Some(DataType::F32))?)],
        ),
        ("unscaled", vec![values(lhs.scaled_dot(&rhs, None, None, Some(&dimensions), Some(DataType::F32))?)]),
        (
            "scaled_matmul",
            vec![values(matmul_lhs.scaled_matmul(&matmul_rhs, &matmul_lhs_scale, &matmul_rhs_scale, None)?)],
        ),
    ]);
    let traced: TracedXlaProgram<(ArrayType, ArrayType, ArrayType, ArrayType), ArrayType> = trace(
        {
            let dimensions = dimensions.clone();
            move |(lhs, rhs, lhs_scale, rhs_scale): (ShardMapTracer, ShardMapTracer, ShardMapTracer, ShardMapTracer)| {
                lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), Some(&dimensions), Some(DataType::F32))
                    .unwrap()
            }
        },
        (
            ArrayType::new_static(DataType::F32, [2, 4]),
            ArrayType::new_static(DataType::F32, [4, 3]),
            ArrayType::new_static(DataType::F32, [2, 2]),
            ArrayType::new_static(DataType::F32, [2, 3]),
        ),
    )?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "scaled_dot_and_matmul".into(),
        observations,
        staging: None,
        stablehlo: Some(traced.to_mlir_module("main")?),
    })
}

/// Emits the portable rank-three MQA attention surface plus its semantic StableHLO composition.
fn emit_dot_product_attention() -> Result<DifferentialObservation, Box<dyn Error>> {
    let query_type = ArrayType::new_static(DataType::F32, [2, 2, 1]);
    let key_value_type = ArrayType::new_static(DataType::F32, [2, 1, 1]);
    let bias_type = ArrayType::scalar(DataType::F32);
    let mask_type = ArrayType::new_static(DataType::Boolean, [2, 2]);
    let lengths_type = ArrayType::new_static(DataType::I32, [1]);
    let configuration = AttentionConfiguration::new()
        .with_scale(1.0)
        .with_causal(true)
        .with_local_window((1, 0))
        .with_residual(true);
    let inputs = AttentionInputs {
        query: Array::from_elements::<f32>(query_type.clone(), &[0.0; 4])?,
        key: Array::from_elements::<f32>(key_value_type.clone(), &[0.0; 2])?,
        value: Array::from_elements::<f32>(key_value_type.clone(), &[3.0, 9.0])?,
        bias: Some(Array::from_elements::<f32>(bias_type.clone(), &[0.0])?),
        mask: Some(Array::from_elements(mask_type.clone(), &[true, false, false, true])?),
        query_sequence_lengths: Some(Array::from_elements(lengths_type.clone(), &[2_i32])?),
        key_value_sequence_lengths: Some(Array::from_elements(lengths_type.clone(), &[2_i32])?),
    };
    let (output, residual) = Array::dot_product_attention(inputs, configuration)?;
    let values = |value: Array| vec![value.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>()];
    let gqa_query_type = ArrayType::new_static(DataType::F32, [1, 2, 4, 1]);
    let gqa_key_value_type = ArrayType::new_static(DataType::F32, [1, 3, 2, 1]);
    let (gqa_output, _) = Array::dot_product_attention(
        AttentionInputs {
            query: Array::from_elements::<f32>(gqa_query_type, &[0.0; 8])?,
            key: Array::from_elements::<f32>(gqa_key_value_type.clone(), &[0.0; 6])?,
            value: Array::from_elements::<f32>(gqa_key_value_type, &[1.0, 10.0, 2.0, 20.0, 4.0, 40.0])?,
            bias: None,
            mask: None,
            query_sequence_lengths: None,
            key_value_sequence_lengths: Some(Array::from_elements(lengths_type.clone(), &[2_i32])?),
        },
        AttentionConfiguration::new()
            .with_local_window((1, 1))
            .with_implementation(AttentionImplementation::Portable),
    )?;
    let observations = BTreeMap::from([
        ("output", values(output)),
        ("residual", values(residual.expect("the configuration requests a residual"))),
        ("rank_four_gqa", values(gqa_output)),
    ]);
    let traced: TracedXlaProgram<
        (ArrayType, ArrayType, ArrayType, ArrayType, ArrayType, ArrayType, ArrayType),
        (ArrayType, ArrayType),
    > = trace(
        move |(query, key, value, bias, mask, query_sequence_lengths, key_value_sequence_lengths): (
            ShardMapTracer,
            ShardMapTracer,
            ShardMapTracer,
            ShardMapTracer,
            ShardMapTracer,
            ShardMapTracer,
            ShardMapTracer,
        )| {
            let (output, residual) = ShardMapTracer::dot_product_attention(
                AttentionInputs {
                    query,
                    key,
                    value,
                    bias: Some(bias),
                    mask: Some(mask),
                    query_sequence_lengths: Some(query_sequence_lengths),
                    key_value_sequence_lengths: Some(key_value_sequence_lengths),
                },
                configuration,
            )
            .unwrap();
            (output, residual.expect("the configuration requests a residual"))
        },
        (query_type, key_value_type.clone(), key_value_type, bias_type, mask_type, lengths_type.clone(), lengths_type),
    )?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "dot_product_attention".into(),
        observations,
        staging: None,
        stablehlo: Some(traced.to_mlir_module("main")?),
    })
}

/// Emits dynamic slicing values at negative and out-of-range starts under both negative-index policies, plus the
/// StableHLO of a traced dynamic slice whose signed start wraps before the native clamp.
fn emit_negative_dynamic_slice() -> Result<DifferentialObservation, Box<dyn Error>> {
    let vector = Array::vector(vec![10.0_f32, 20.0, 30.0, 40.0])?;
    let update = Array::vector(vec![1.0_f32, 2.0])?;
    let start = |value: i32| -> Result<Array, ProgramError> { Array::scalar(value) };
    let values = |value: Array| vec![value.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>()];
    let observations = BTreeMap::from([
        ("slice_minus_one", values(vector.dynamic_slice(&[start(-1)?], &[2])?)),
        ("slice_minus_nine", values(vector.dynamic_slice(&[start(-9)?], &[2])?)),
        ("slice_past_end", values(vector.dynamic_slice(&[start(3)?], &[2])?)),
        ("slice_minus_one_clamp_only", values(vector.dynamic_slice_with_negative_indices(&[start(-1)?], &[2], false)?)),
        ("update_minus_one", values(vector.dynamic_update_slice(&update, &[start(-1)?])?)),
        ("update_minus_nine", values(vector.dynamic_update_slice(&update, &[start(-9)?])?)),
        (
            "update_minus_one_clamp_only",
            values(vector.dynamic_update_slice_with_negative_indices(&update, &[start(-1)?], false)?),
        ),
    ]);
    let traced: TracedXlaProgram<(ArrayType, ArrayType), ArrayType> = trace(
        |(input, start): (ShardMapTracer, ShardMapTracer)| input.dynamic_slice(&[start], &[2]).unwrap(),
        (ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)),
    )?;
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "negative_dynamic_slice".into(),
        observations,
        staging: None,
        stablehlo: Some(traced.to_mlir_module("main")?),
    })
}

/// Emits the value and gradients of a `condition` whose predicate varies across the devices of a manual region.
///
/// Each device of a four-device `x` mesh holds one element of `values = [1, -2, 3, -4]` and the replicated `weight
/// = [2]`. Each device takes the `true` branch, `weight * values`, when its element is positive and the `false` branch,
/// `values`, otherwise, and the result is summed over the mesh. [`ryft_core::condition`] varies the invariant weight
/// before the branches, so the gradient of the weight is summed across devices after the transposed condition rather
/// than inside the branch that only some devices take. The value is `2 - 2 + 6 - 4 = 2`, the weight gradient is the sum
/// of the positive elements, `4`, and the values gradient is `[2, 1, 2, 1]`.
fn emit_condition_varying_predicate_gradient() -> Result<DifferentialObservation, Box<dyn Error>> {
    let (client, device_mesh, sharded) = collective_runtime()?;
    let mesh = device_mesh.logical_mesh().clone();
    let replicated = Sharding::replicated(mesh.clone(), 1);
    let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
        {
            let replicated = replicated.clone();
            let sharded = sharded.clone();
            move |inputs: Vec<ShardMapTracer>| {
                let (value, gradients) = inputs[0]
                    .clone()
                    .into_value()
                    .domain()
                    .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                    .value_and_gradient(|(weight, values)| {
                        let weight = ValueProjection::<ArrayType>::into_projected(weight)?;
                        let values = ValueProjection::<ArrayType>::into_projected(values)?;
                        Ok(shard_map::<_, _, ArrayType, _>(
                            |(weight, values): (ShardMapTracer, ShardMapTracer)| {
                                let sum = values.reduce(&[0], ReductionKind::Sum).unwrap();
                                let predicate =
                                    sum.compare(&sum.zero_like().unwrap(), ComparisonDirection::GreaterThan).unwrap();
                                let mut outputs = condition(
                                    &predicate.into_value(),
                                    vec![weight.into_value(), values.into_value()],
                                    |inputs| {
                                        let weight = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                                        let values = ValueProjection::<ArrayType>::into_projected(inputs[1].clone())?;
                                        Ok(vec![(weight * values).into_value()])
                                    },
                                    |inputs| Ok(vec![inputs[1].clone()]),
                                )
                                .unwrap();
                                ValueProjection::<ArrayType>::into_projected(outputs.remove(0))
                                    .unwrap()
                                    .reduce(&[0], ReductionKind::Sum)
                                    .unwrap()
                                    .parallel_reduce(ReductionKind::Sum, "x")
                                    .unwrap()
                            },
                            (weight, values),
                            mesh.clone(),
                            (replicated.clone(), sharded.clone()),
                            Sharding::replicated(mesh.clone(), 0),
                        )
                        .unwrap()
                        .into_value())
                    })
                    .unwrap();
                vec![
                    ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(gradients.0).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(gradients.1).unwrap(),
                ]
            }
        },
        vec![ArrayType::new_static(DataType::F32, [1]), ArrayType::new_static(DataType::F32, [4])],
    )?;
    let stablehlo = traced.to_mlir_module("main")?;
    let outputs = execute_collective_inputs(
        &client,
        device_mesh,
        stablehlo.as_str(),
        &[
            (replicated, &[1], &[1], &vec![vec![2.0f32]; 4]),
            (sharded, &[4], &[1], &[vec![1.0f32], vec![-2.0], vec![3.0], vec![-4.0]]),
        ],
        &[vec![], vec![1], vec![1]],
    )?;
    // The value and the weight gradient are replicated, so every participant must hold the same one.
    for (participant, output) in outputs.iter().enumerate() {
        if output[0] != outputs[0][0] || output[1] != outputs[0][1] {
            return Err(format!("participant {participant} disagrees on a replicated output: {output:?}").into());
        }
    }
    Ok(DifferentialObservation {
        schema: SCHEMA,
        case_id: "condition_varying_predicate_gradient".into(),
        observations: BTreeMap::from([
            ("value", vec![outputs[0][0].clone()]),
            ("weight_gradient", vec![outputs[0][1].clone()]),
            ("values_gradient", vec![outputs.iter().flat_map(|output| output[2].clone()).collect()]),
        ]),
        staging: Some(StagingObservation::Supported { output_type: "f32[1]".to_string() }),
        stablehlo: Some(stablehlo),
    })
}

/// Parses selected case IDs, emits deterministic JSON records, and returns an error for an unknown case.
fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let cases = registry();
    let mut requested = Vec::new();
    let mut list = false;
    let mut suite = None;
    let mut worker_rank = None;
    let mut coordinator = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--list" => {
                list = true;
                index += 1;
            }
            "--suite"
                if index + 1 < arguments.len()
                    && matches!(
                        arguments[index + 1].as_str(),
                        "collectives" | "cuda-collectives" | "cuda-distributed-collectives"
                    ) =>
            {
                suite = Some(arguments[index + 1].as_str());
                index += 2;
            }
            "--distributed-worker" if index + 1 < arguments.len() => {
                worker_rank = Some(arguments[index + 1].parse::<usize>()?);
                index += 2;
            }
            "--coordinator" if index + 1 < arguments.len() => {
                coordinator = Some(arguments[index + 1].as_str());
                index += 2;
            }
            "--case" if index + 1 < arguments.len() => {
                requested.push(arguments[index + 1].as_str());
                index += 2;
            }
            argument => {
                return Err(format!(
                    "expected `--list`, `--case CASE_ID`, or `--suite SUITE`, `--distributed-worker RANK`, or `--coordinator ADDRESS` but got `{argument}`"
                )
                .into());
            }
        }
    }
    if list {
        if !requested.is_empty() || worker_rank.is_some() || coordinator.is_some() {
            return Err("`--list` cannot be combined with `--case` or distributed worker options".into());
        }
        for case in cases.iter().filter(|case| suite.is_none() || (suite == Some("collectives") && case.collective)) {
            println!("{}", case.case_id);
        }
        if suite.is_none() || suite == Some("collectives") {
            for case in collectives::registry()? {
                println!("{}", case.id);
            }
        }
        if suite == Some("cuda-collectives") {
            for case in ragged::registry()? {
                println!("{}", case.id);
            }
        }
        if suite == Some("cuda-distributed-collectives") {
            for case in distributed_collectives::registry()? {
                println!("{}", case.id);
            }
        }
        return Ok(());
    }
    if let Some(rank) = worker_rank {
        if suite.is_some_and(|suite| suite != "cuda-distributed-collectives") {
            return Err("distributed workers require suite `cuda-distributed-collectives`".into());
        }
        let coordinator = coordinator.ok_or("distributed workers require `--coordinator ADDRESS`")?;
        let mut records = distributed_collectives::run_worker(rank, coordinator, &requested)?;
        records.sort_by(|left, right| left.case_id.cmp(&right.case_id));
        println!("{}", serde_json::to_string_pretty(&records)?);
        return Ok(());
    }
    if coordinator.is_some() {
        return Err("`--coordinator` requires `--distributed-worker RANK`".into());
    }
    let distributed_cases = distributed_collectives::registry()?;
    if suite == Some("cuda-distributed-collectives")
        || requested.iter().any(|id| distributed_cases.iter().any(|case| case.id == *id))
    {
        return Err("suite `cuda-distributed-collectives` requires the two-process launcher or `--distributed-worker RANK --coordinator ADDRESS`".into());
    }
    let collective_cases = collectives::registry()?;
    let ragged_cases = ragged::registry()?;
    let requested = if requested.is_empty() {
        cases
            .iter()
            .filter(|case| suite.is_none() || (suite == Some("collectives") && case.collective))
            .map(|case| case.case_id)
            .chain(collective_cases.iter().filter(|_| suite != Some("cuda-collectives")).map(|case| case.id.as_str()))
            .chain(ragged_cases.iter().filter(|_| suite == Some("cuda-collectives")).map(|case| case.id.as_str()))
            .collect()
    } else {
        requested
    };
    let mut records = Vec::new();
    for case_id in requested {
        if let Some(case) = cases.iter().find(|case| case.case_id == case_id) {
            if suite == Some("cuda-collectives") || (suite == Some("collectives") && !case.collective) {
                return Err(format!("case `{case_id}` is not in suite `{}`", suite.unwrap()).into());
            }
            records.push((case.emit)()?);
        } else if let Some(case) = collective_cases.iter().find(|case| case.id == case_id) {
            if suite == Some("cuda-collectives") {
                return Err(format!("case `{case_id}` is not in suite `cuda-collectives`").into());
            }
            records.push(case.emit().map_err(|error| format!("collective case `{case_id}` failed: {error}"))?);
        } else if let Some(case) = ragged_cases.iter().find(|case| case.id == case_id) {
            if suite == Some("collectives") {
                return Err(format!("case `{case_id}` is not in suite `collectives`").into());
            }
            records.push(case.emit().map_err(|error| format!("CUDA collective case `{case_id}` failed: {error}"))?);
        } else {
            return Err(format!("unknown differential-testing case `{case_id}`").into());
        }
    }
    records.sort_by(|left, right| left.case_id.cmp(&right.case_id));
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
            vec![
                "grouped_shape_changing_collectives",
                "pshuffle",
                "pswapaxes",
                "data_dependent_prefix_take",
                "scaled_dot_and_matmul",
                "dot_product_attention",
                "negative_dynamic_slice",
                "condition_varying_predicate_gradient",
            ],
        );
    }

    #[test]
    fn test_run_rejects_case_outside_suite() {
        assert_eq!(
            run(&["--suite".into(), "collectives".into(), "--case".into(), "dot_product_attention".into()])
                .unwrap_err()
                .to_string(),
            "case `dot_product_attention` is not in suite `collectives`",
        );
    }

    #[test]
    fn test_run_rejects_cuda_case_in_cpu_suite() {
        assert_eq!(
            run(&["--suite".into(), "collectives".into(), "--case".into(), "cuda_ragged_seed_holes_i32".into()])
                .unwrap_err()
                .to_string(),
            "case `cuda_ragged_seed_holes_i32` is not in suite `collectives`",
        );
    }

    #[test]
    fn test_run_rejects_cpu_case_in_cuda_suite() {
        assert_eq!(
            run(&["--suite".into(), "cuda-collectives".into(), "--case".into(), "collective_gather_1_untiled".into()])
                .unwrap_err()
                .to_string(),
            "case `collective_gather_1_untiled` is not in suite `cuda-collectives`",
        );
    }

    #[cfg(not(feature = "cuda-13"))]
    #[test]
    fn test_run_requires_cuda_feature() {
        assert_eq!(
            run(&["--suite".into(), "cuda-collectives".into()]).unwrap_err().to_string(),
            "CUDA collective case `cuda_ragged_seed_holes_i32` failed: suite `cuda-collectives` requires the `cuda-13` Cargo feature",
        );
    }

    #[test]
    fn test_run_requires_distributed_worker() {
        assert_eq!(
            run(&["--suite".into(), "cuda-distributed-collectives".into()]).unwrap_err().to_string(),
            "suite `cuda-distributed-collectives` requires the two-process launcher or `--distributed-worker RANK --coordinator ADDRESS`",
        );
    }

    #[test]
    fn test_run_requires_distributed_coordinator() {
        assert_eq!(
            run(&["--distributed-worker".into(), "0".into()]).unwrap_err().to_string(),
            "distributed workers require `--coordinator ADDRESS`",
        );
    }

    #[test]
    fn test_run_rejects_distributed_options_with_cpu_suite() {
        assert_eq!(
            run(&["--suite".into(), "collectives".into(), "--distributed-worker".into(), "0".into()])
                .unwrap_err()
                .to_string(),
            "distributed workers require suite `cuda-distributed-collectives`",
        );
    }

    #[test]
    fn test_data_dependent_prefix_take() {
        assert_eq!(
            emit_data_dependent_prefix_take().unwrap(),
            DifferentialObservation {
                schema: SCHEMA,
                case_id: "data_dependent_prefix_take".into(),
                observations: BTreeMap::from(
                    [("two_matches", vec![vec![10.0, 20.0]]), ("zero_matches", vec![vec![]]),]
                ),
                staging: Some(StagingObservation::Supported { output_type: "f32[count]".to_string() }),
                stablehlo: None,
            },
        );
    }
}
