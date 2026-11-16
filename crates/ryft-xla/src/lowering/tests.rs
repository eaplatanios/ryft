use std::collections::HashMap;
use std::sync::Arc;

use indoc::indoc;
use pretty_assertions::assert_eq;
use ryft_core::{
    AddOperation, Array as CoreArray, AtomId, CaptureReference, CastOperation, Complex as ComplexNumber,
    ConditionOperation, ConstantOperation, Context, Cos, Differentiate, Dimension as CoreDimension, DimensionBounds,
    DimensionToScalarOperation, DimensionType, DimensionValue, DimensionVariable, DomainTracingContext, Dot,
    DotDimensionNumbers, EagerContext, EffectClasses, Layout, MeshAxis, MeshAxisType, NegOperation, OneLike,
    ParallelReduce, ParameterizedFamily, Placeholder, PrintOperation, Program as CoreProgram,
    ProgramBuilder as CoreProgramBuilder, ProjectedContext, ProvenanceScope, ReduceOperation, ReductionKind,
    ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceType,
    ReferenceWriteOperation, ReshapeOperation, ReverseModeDifferentiate, ShardMap, ShardMapOperation, ShardMapOptions,
    Sin, StridedLayout, TiledLayout, Transpose, TypedProgram, ZeroLike, i1, i2, i4, shard_map, u1, u2, u4,
};
use ryft_kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;
use ryft_mlir::{Attribute, ElementsAttribute, Operation as MlirOperation, Region, Size as MlirSize};
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

use crate::lowering::LoweringError;
use crate::operations::assertions::ensure_assertion_handler_registered;
use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
use crate::{
    ArrayTracer, FlatProgram, JitCallOperation, ProgramBuilder as ArrayIrProgramBuilder, Session, TracedProgram, trace,
    trace_with_context,
};

use super::*;

// TODO(eaplatanios): Review this module.

/// Builder of homogeneous array programs that operation tests across the crate use to stage fixtures for the test-only
/// plain program lowering (i.e., [`to_mlir_module_for_plain_program`]) and for [`unproject_plain_program`].
pub(crate) type ProgramBuilder = CoreProgramBuilder<ArrayConstant, ArrayOperation<ArrayConstant>>;

/// Homogeneous array program that [`ProgramBuilder`] builds, which operation tests across the crate lower through the
/// test-only plain program lowering or lift into the array IR through [`unproject_plain_program`].
pub(crate) type PlainProgram =
    CoreProgram<ArrayConstant, ArrayOperation<ArrayConstant>, Vec<ArrayConstant>, Vec<ArrayConstant>>;

/// Logical mesh with one manual axis named `axis_name` of size `axis_size`.
pub(crate) fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
}

/// Static `f32` vector type of `length` elements.
pub(crate) fn test_vector_type(length: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(length)]))
}

/// Static `f32` matrix type with `rows` rows and `columns` columns.
pub(crate) fn test_matrix_type(rows: usize, columns: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(rows), CoreDimension::Static(columns)]))
}

/// Global array type of `data_type` with the static `global_shape`, placed by `sharding`.
pub(crate) fn static_sharded_array_type(data_type: DataType, global_shape: &[usize], sharding: Sharding) -> ArrayType {
    ArrayType::new(data_type, Shape::new(global_shape.iter().copied().map(CoreDimension::Static).collect()))
        .with_sharding(sharding)
        .unwrap()
}

/// Sharding over `mesh` with the provided dimension shardings and unreduced axes.
pub(crate) fn test_sharding(
    mesh: &LogicalMesh,
    dimensions: Vec<ShardingDimension>,
    unreduced_axes: Vec<String>,
) -> Sharding {
    Sharding::new(mesh.clone(), dimensions).unwrap().with_unreduced_axes(unreduced_axes).unwrap()
}

/// Traces `function` as the body of one top-level `shard_map` (refer to [`ShardMap`]) over global inputs of
/// `global_input_types` placed by `input_shardings` without their auto axes (i.e., as the boundary places them), and
/// lowers the traced program to a module whose `main` function signature carries the shardings of its global input and
/// output types, which are the boundary shardings.
pub(crate) fn shard_map_module<F, Input, Output>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    input_shardings: Input::To<Sharding>,
    output_shardings: Output::To<Sharding>,
    options: ShardMapOptions,
) -> Result<String, Error>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer> + ParameterizedFamily<Sharding>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer> + ParameterizedFamily<Sharding>,
    F: FnOnce(
        &ProjectedContext<DomainTracingContext<Session<'static>>, ArrayType>,
        Input::To<ArrayTracer>,
    ) -> Result<Output::To<ArrayTracer>, ProgramError>,
{
    let input_structure = global_input_types.parameter_structure();
    let output_structure = output_shardings.parameter_structure();
    let input_shardings = input_shardings.into_parameters().collect::<Vec<_>>();
    let output_shardings = output_shardings.into_parameters().collect::<Vec<_>>();
    let global_input_types = Input::from_parameters(
        input_structure.clone(),
        global_input_types
            .into_parameters()
            .zip(&input_shardings)
            .map(|(r#type, sharding)| r#type.with_sharding(sharding.without_auto_axes()))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let traced: TracedProgram<Input, Output> = trace_with_context(
        |context, inputs| {
            let outputs = context.shard_map(
                |body_context, local_inputs: Vec<ArrayTracer>| {
                    let local_inputs = Input::To::<ArrayTracer>::from_parameters(input_structure, local_inputs)?;
                    Ok(function(body_context, local_inputs)?.into_parameters().collect::<Vec<_>>())
                },
                inputs.into_parameters().collect::<Vec<_>>(),
                mesh,
                input_shardings,
                output_shardings,
                options,
            )?;
            Ok(Output::To::<ArrayTracer>::from_parameters(output_structure, outputs)?)
        },
        global_input_types,
    )?;
    traced.to_mlir_module("main")
}

/// Compilation options for an SPMD executable with `partition_count` partitions, partitioned by Shardy.
pub(crate) fn test_spmd_compilation_options(partition_count: usize) -> ryft_pjrt::protos::CompilationOptions {
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
        ..Default::default()
    }
}

/// Dynamic dimension of a fresh non-negative variable named `name`, bounded above by `exclusive_upper_bound` when it
/// is provided and unbounded otherwise.
pub(crate) fn dynamic_dimension(name: &str, exclusive_upper_bound: Option<usize>) -> CoreDimension {
    DimensionVariable::new(name, DimensionBounds::non_negative(exclusive_upper_bound).unwrap()).into()
}

/// One physical value of the mixed executable ABI used by the reshape execution tests. It is either `f64` or `i32`
/// array storage, with its physical dimensions on input and the row-major elements inside its logical dimensions on
/// output, or one `i64` dimension, which on output also carries the hidden runtime extents that follow the logical
/// outputs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MixedValue {
    /// `f64` storage of one array together with its dimensions.
    Array(Vec<f64>, Vec<u64>),

    /// `i32` storage of one array together with its dimensions (e.g., explicit group sizes or transfer lengths).
    Integers(Vec<i32>, Vec<u64>),

    /// One `i64` first-class dimension or hidden runtime extent.
    Dimension(i64),
}

/// Compiles one array IR program for the mixed executable ABI and executes it once on `client`. Each logical input
/// receives the matching entry of `inputs` (i.e., array storage sized to the allocation bound, or one dimension), and
/// every bounded dynamic input axis then receives its runtime extent from `extents` in boundary order.
pub(crate) fn execute_mixed_program(
    client: &ryft_pjrt::Client<'_>,
    program: &FlatProgram,
    inputs: &[MixedValue],
    extents: &[i32],
) -> Result<Vec<MixedValue>, Error> {
    // First-class dimensions cross the executable boundary as `i64` scalars.
    let boundary_type = |r#type: &ArrayIrType| match r#type {
        ArrayIrType::Array(r#type) => r#type.clone(),
        ArrayIrType::Dimension(_) => ArrayType::scalar(DataType::I64),
        ArrayIrType::Reference(_) => panic!("reshape execution fixtures have no reference values"),
    };
    let input_types = program.input_types().iter().map(boundary_type).collect::<Vec<_>>();
    let output_types = program.output_types().iter().map(boundary_type).collect::<Vec<_>>();
    let module = to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap();
    // Lowering may synthesize runtime checks (e.g., entry guards of mixed region entries), so register their handler.
    ensure_assertion_handler_registered(client)?;
    let executable =
        client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())?;
    let device = executable.addressable_devices()?.remove(0);
    let mut buffers = Vec::with_capacity(inputs.len() + extents.len());
    for input in inputs {
        buffers.push(match input {
            MixedValue::Array(values, dimensions) => client.buffer(
                values_to_bytes(values).as_slice(),
                BufferType::F64,
                dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )?,
            MixedValue::Integers(values, dimensions) => client.buffer(
                values_to_bytes(values).as_slice(),
                BufferType::I32,
                dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )?,
            MixedValue::Dimension(value) => {
                client.buffer(values_to_bytes(&[*value]).as_slice(), BufferType::I64, [], None, device.clone(), None)?
            }
        });
    }
    for extent in extents {
        buffers.push(client.buffer(
            values_to_bytes(&[*extent]).as_slice(),
            BufferType::I32,
            [],
            None,
            device.clone(),
            None,
        )?);
    }
    let inputs = buffers
        .into_iter()
        .map(|buffer| ExecutionInput { buffer: Arc::new(buffer), donatable: false })
        .collect::<Vec<_>>();
    let outputs = executable
        .execute(
            vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
            Vec::new(),
            0,
            None,
            Some(file!()),
            None,
            None,
            &[],
        )?
        .block_until_ready()?
        .remove(0)
        .outputs;
    outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            let bytes = output
                .copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(output.dimensions()?.len())))?
                .r#await()?;
            let data_type = output_types.get(index).map(|output_type| output_type.data_type());
            if data_type.is_none_or(|data_type| data_type != DataType::F64 && data_type != DataType::I32) {
                // Dimension outputs and the trailing hidden runtime extents are `i64` scalars.
                return Ok(MixedValue::Dimension(values_from_bytes::<i64>(bytes.as_slice())[0]));
            }
            // A populated dynamic result is copied back as its compact logical elements, while an empty logical
            // result may still carry its backing allocation and trailing size metadata, which are ignored.
            let logical = output.unpadded_dimensions()?.to_vec();
            let element_count = logical.iter().product::<u64>() as usize;
            if data_type == Some(DataType::I32) {
                let elements = values_from_bytes::<i32>(&bytes[..element_count * size_of::<i32>()]);
                return Ok(MixedValue::Integers(elements, logical));
            }
            let elements = values_from_bytes::<f64>(&bytes[..element_count * size_of::<f64>()]);
            Ok(MixedValue::Array(elements, logical))
        })
        .collect()
}

/// Plain program that returns its single input of `input_type` unchanged.
pub(crate) fn identity_branch(input_type: ArrayType) -> PlainProgram {
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type);
    builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
}

/// Plain program that returns the negation of its single input of `input_type`.
pub(crate) fn neg_branch(input_type: ArrayType) -> PlainProgram {
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type);
    let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

/// Lifts an array-only test fixture into the production array IR XLA program representation.
pub(crate) fn unproject_plain_program(program: PlainProgram) -> FlatProgram {
    program.into_unprojected::<Constant, Operation>().unwrap()
}

/// Stages one `jit_call` to `callee`, which is interned as a shared callee root region, over `inputs` in `builder`.
pub(crate) fn add_jit_call(
    builder: &mut ArrayIrProgramBuilder,
    callee: &Arc<FlatProgram>,
    inputs: Vec<AtomId>,
) -> AtomId {
    let callee_region = builder.intern_callee(callee, None).unwrap();
    builder
        .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], inputs, None)
        .unwrap()[0]
}

/// Wraps one flat vector program in a replicated manual `shard_map` and lowers the enclosing program.
pub(crate) fn lower_replicated_shard_map_body(body_program: FlatProgram, device_count: usize) -> Result<String, Error> {
    let vector_type = test_vector_type(4);
    let mesh = test_manual_mesh("x", device_count);
    let sharding = Sharding::replicated(mesh.clone(), 1);
    let operation = ShardMapOperation::from_program(
        &body_program,
        vec![vector_type.clone().into()],
        ShardMapBoundary::new(mesh, vec![sharding.clone()], vec![sharding], vec!["x".to_string()]).unwrap(),
    )
    .unwrap();
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(vector_type.clone().into());
    let body = builder.import_program(body_program);
    let output = builder.add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![input], None)?[0];
    let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])?;
    let input_types = vec![vector_type.clone()];
    let output_types = vec![vector_type];
    to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
}

/// Builds a one-instruction program applying the given single-operand operation to an input of the given data
/// type and dimensions, and returns the result of lowering it to a rendered StableHLO module.
pub(crate) fn lowered_unary_module(
    operation: ArrayOperation<ArrayConstant>,
    data_type: DataType,
    dimensions: Vec<usize>,
) -> Result<String, Error> {
    let input_type = ArrayType::new(data_type, Shape::new(dimensions.into_iter().map(CoreDimension::Static).collect()));
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type);
    let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
    let program = builder
        .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    to_mlir_module_for_plain_program(&program, "main")
}

/// Compilation options for a single-device CPU executable with one replica and one partition, shared by ragged dot
/// execution and rejection tests and by other single-device execution helpers (e.g., [`execute_mixed_program`]).
pub(crate) fn ragged_dot_cpu_compilation_options() -> ryft_pjrt::protos::CompilationOptions {
    CompilationOptions {
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
        environment_option_overrides: Default::default(),
        target_config: None,
        allow_in_place_mlir_modification: false,
        matrix_unit_operand_precision: Precision::Default as i32,
        ..Default::default()
    }
}

/// Builds a one-instruction program reducing an input of the given data type and dimensions, and returns the
/// result of lowering it to a rendered StableHLO module.
pub(crate) fn lowered_reduce_module(
    data_type: DataType,
    kind: ReductionKind,
    axes: Vec<usize>,
    dimensions: Vec<usize>,
) -> Result<String, Error> {
    lowered_unary_module(ArrayOperation::Reduce(ReduceOperation::new(axes, kind)), data_type, dimensions)
}

/// Computes `f(x, y) = x * y + sin(x)` for the pair `inputs` of `(x, y)`.
fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + ryft_core::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
}

/// Computes `f(x) = x^4 + sin(x)`.
fn scalar_quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + ryft_core::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
}

/// Eager context over reference backend arrays that stages the scalar programs of the plain program lowering tests.
static TEST_ARRAY_DOMAIN: EagerContext<CoreArray, ArrayOperation<CoreArray>> =
    EagerContext::<CoreArray, ArrayOperation<CoreArray>>::new();

/// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
fn literal_dense_bytes(literal: &CoreArray, byte_count: usize) -> Vec<u8> {
    let context = MlirContext::new();
    let location = context.unknown_location();
    let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
    let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
    unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
}

#[test]
fn test_validate_local_output_placement() {
    // An array IR `Operation::Cast` (i.e., a `CastOperation<ArrayIrType>` rather than a projected
    // `ArrayOperation::Cast`) lowers without passing through `ArrayOperation::lower`, so only the validation of the
    // family in `Operation::lower` sees its output placement. Its first output requests a mesh that redefines the bound
    // manual axis `x` as explicit and shards a local dimension over it, which only the region boundary may do.
    let vector_type = test_vector_type(4);
    let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let explicit_type = vector_type
        .clone()
        .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
        .unwrap();
    let cast = |output_type: &ArrayType, input_type: &ArrayType| {
        Operation::Cast(CastOperation::unchecked(output_type.clone().into(), vec![input_type.clone().into()]).unwrap())
    };
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(vector_type.clone().into());
    let placed = builder.add_instruction(cast(&explicit_type, &vector_type), Vec::new(), vec![input], None).unwrap()[0];
    let output =
        builder.add_instruction(cast(&vector_type, &explicit_type), Vec::new(), vec![placed], None).unwrap()[0];
    let body = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert!(matches!(
        lower_replicated_shard_map_body(body, 2),
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                if *message == "`cast` cannot assign bound manual mesh axes to local output dimensions",
        ),
    ));
}

#[test]
fn test_collective_lowering_state_next_channel_id() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
    let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

    // Two manual regions in one module each emit a channeled `all_reduce`. The channel allocator of the module
    // hands out distinct handles (i.e., 1 and 2), which XLA requires across a module.
    let traced: TracedProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            let sharding = sharding.clone();
            move |x: ArrayTracer| {
                let first = shard_map(
                    |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap();
                shard_map(
                    |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                    first,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap()
            }
        },
        global_input_type,
    )
    .unwrap();

    assert_eq!(
        traced.to_mlir_module("main").unwrap(),
        indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=4]>
              func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                  %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                    %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                    stablehlo.return %3 : tensor<f32>
                  }) : (tensor<2xf32>) -> tensor<2xf32>
                  sdy.return %2 : tensor<2xf32>
                } : (tensor<8xf32>) -> tensor<8xf32>
                %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                  %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                    %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                    stablehlo.return %3 : tensor<f32>
                  }) : (tensor<2xf32>) -> tensor<2xf32>
                  sdy.return %2 : tensor<2xf32>
                } : (tensor<8xf32>) -> tensor<8xf32>
                return %1 : tensor<8xf32>
              }
            }
        "#},
    );
}

#[test]
fn test_collective_lowering_state_checked_collective_mesh() {
    let outer_mesh = test_manual_mesh("x", 2);
    let inner_mesh = test_manual_mesh("x", 3);
    let outer = ShardMapBoundary::new(outer_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
    let inner = ShardMapBoundary::new(inner_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
    let state = CollectiveLoweringState::new().enter_manual_region(outer).enter_manual_region(inner);
    assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 3, None).unwrap(), &inner_mesh);
    assert!(matches!(
        state.checked_collective_mesh("parallel_permute", "x", 3, Some(&outer_mesh)),
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
        ),
    ));
    assert!(matches!(
        state.checked_collective_mesh("parallel_permute", "x", 2, Some(&outer_mesh)),
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "collective over axis `x` records size 2, but the enclosing mesh axis has size 3",
        ),
    ));

    // A payload mesh that retypes only a free axis of the binder mesh (e.g., inside `auto_axes`) is accepted and
    // returned, while a payload mesh that retypes a manual axis or has another topology is still rejected.
    let binder_mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
    ])
    .unwrap();
    let binder = ShardMapBoundary::new(binder_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
    let state = CollectiveLoweringState::new().enter_manual_region(binder);
    let free_axis_retyped_mesh = binder_mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
    let manual_axis_retyped_mesh = binder_mesh.with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
    let other_topology_mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        MeshAxis::new("y", 4, MeshAxisType::Auto).unwrap(),
    ])
    .unwrap();
    assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 2, None).unwrap(), &binder_mesh);
    assert_eq!(
        state.checked_collective_mesh("parallel_permute", "x", 2, Some(&free_axis_retyped_mesh)).unwrap(),
        &free_axis_retyped_mesh,
    );
    for payload_mesh in [&manual_axis_retyped_mesh, &other_topology_mesh] {
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(payload_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
            ),
        ));
    }
}

#[test]
fn test_collective_lowering_state_validate_manual_collective_uses_owner_policy() {
    let mesh = test_manual_mesh("x", 2);
    let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
    let checked = CollectiveLoweringState::new().enter_manual_region(boundary.clone());
    checked
        .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, Some(&mesh));
            Ok(vec![1])
        })
        .unwrap();
    let unchecked = CollectiveLoweringState::new().enter_manual_region(boundary.with_check_manual_variation(false));
    unchecked
        .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, None);
            Ok(vec![1])
        })
        .unwrap();
}

#[test]
fn test_collective_lowering_state_validate_manual_collective_uses_each_axis_owner() {
    let mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
    ])
    .unwrap();
    let outer = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
    let inner = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["y".to_string()]).unwrap();
    let state = CollectiveLoweringState::new()
        .enter_manual_region(outer.clone())
        .enter_manual_region(inner.clone().with_check_manual_variation(false));
    state
        .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, Some(&mesh));
            Ok(vec![1])
        })
        .unwrap();
    state
        .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, None);
            Ok(vec![1])
        })
        .unwrap();

    let state = CollectiveLoweringState::new()
        .enter_manual_region(outer.with_check_manual_variation(false))
        .enter_manual_region(inner);
    state
        .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, None);
            Ok(vec![1])
        })
        .unwrap();
    state
        .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
            assert_eq!(inference_mesh, Some(&mesh));
            Ok(vec![1])
        })
        .unwrap();
}

#[test]
fn test_collective_lowering_state_validate_manual_collective_rejects_tracked_unchecked_payload() {
    let mesh = test_manual_mesh("x", 2);
    let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
        .unwrap()
        .with_check_manual_variation(false);
    let state = CollectiveLoweringState::new().enter_manual_region(boundary);
    let result = state.validate_manual_collective("parallel_permute", "x", 2, Some(&mesh), &[1usize], |_| {
        panic!("tracked payload must be rejected before ordinary inference");
    });
    assert!(matches!(
        result,
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "`parallel_permute` cannot use tracked mesh metadata inside an unchecked manual binding",
        ),
    ));
}

#[test]
fn test_collective_lowering_state_validate_manual_collective_retains_unchecked_validation() {
    let mesh = test_manual_mesh("x", 2);
    let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
        .unwrap()
        .with_check_manual_variation(false);
    let state = CollectiveLoweringState::new().enter_manual_region(boundary);
    let wrong_size = state.validate_manual_collective("parallel_permute", "x", 3, None, &[1usize], |_| {
        panic!("axis size must be validated before ordinary inference");
    });
    assert!(matches!(
        wrong_size,
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "collective over axis `x` records size 3, but the enclosing mesh axis has size 2",
        ),
    ));

    let other_mesh = test_manual_mesh("x", 3);
    let wrong_mesh = state.validate_manual_collective("parallel_permute", "x", 2, Some(&other_mesh), &[1usize], |_| {
        panic!("payload mesh must be validated before ordinary inference");
    });
    assert!(matches!(
        wrong_mesh,
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
        ),
    ));

    let wrong_outputs =
        state.validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |_| Ok(vec![2usize]));
    assert!(matches!(
        wrong_outputs,
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                if *message == "`parallel_permute` output types do not match the enclosing manual mesh contract",
        ),
    ));
}

#[test]
fn test_to_mlir_module_for_program() {
    let global_input_type = test_vector_type(8);
    let mesh = test_manual_mesh("x", 4);
    let module = shard_map_module::<_, ArrayType, ArrayType>(
        |_, x| Ok(x.clone() + x),
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ShardMapOptions::new(),
    )
    .unwrap();

    assert_eq!(
        module,
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
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_renders_provenance_locations() {
    let global_input_type = test_vector_type(8);
    let mesh = test_manual_mesh("x", 4);
    let module = shard_map_module::<_, ArrayType, ArrayType>(
        |_, x| {
            let context = x.value().context().clone();
            // One instruction staged under nested scopes, one under a fused origin, and one with unknown
            // provenance, covering every provenance shape in one lowered module.
            let scoped = context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), || {
                context.invoke_with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
            });
            let fused = Provenance::fused([
                Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
            ]);
            let product = context.invoke_with_provenance_origin(fused, || scoped.clone() * scoped);
            Ok(product + x)
        },
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ShardMapOptions::new(),
    )
    .unwrap();

    // The scoped addition carries the nested chain of name locations, the fused multiplication carries a fused location
    // without metadata, the addition with unknown provenance keeps the unknown base location, and the module and
    // function scaffolding keeps the base location. Locations render because modules that carry provenance serialize
    // with debug information enabled, in the parsable form that is not pretty printed.
    assert_eq!(
        module,
        indoc! {r#"
            #loc = loc(unknown)
            module {
              sdy.mesh @mesh = <["x"=4]> loc(#loc)
              func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>} loc(unknown)) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32> loc(unknown)) {
                  %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32> loc(#loc4)
                  %2 = stablehlo.multiply %1, %1 : tensor<2xf32> loc(#loc5)
                  %3 = stablehlo.add %2, %arg1 : tensor<2xf32> loc(#loc)
                  sdy.return %3 : tensor<2xf32> loc(#loc)
                } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc)
                return %0 : tensor<8xf32> loc(#loc)
              } loc(#loc)
            } loc(#loc)
            #loc1 = loc("inner")
            #loc2 = loc("a")
            #loc3 = loc("b")
            #loc4 = loc("outer"(#loc1))
            #loc5 = loc(fused[#loc2, #loc3])
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_renders_constants_and_supported_ops() {
    let global_input_type = test_matrix_type(4, 4);
    let mesh = test_manual_mesh("x", 2);
    let module = shard_map_module::<_, ArrayType, ArrayType>(
        |_, x| {
            let product = x.transpose(vec![1, 0])?.dot(&x, &DotDimensionNumbers::matmul())?;
            let waveform = (-product).cos()?.sin()?;
            Ok((waveform.clone() * waveform.one_like()?) + waveform.zero_like()?)
        },
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
        ShardMapOptions::new(),
    )
    .unwrap();

    assert_eq!(
        module,
        indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=2]>
              func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                  %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                  %3 = stablehlo.negate %2 : tensor<4x4xf32>
                  %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                  %5 = stablehlo.sine %4 : tensor<4x4xf32>
                  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                  %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                  %7 = sdy.sharding_constraint %6 <@mesh, [{}, {}]> : tensor<4x4xf32>
                  %8 = stablehlo.multiply %5, %7 : tensor<4x4xf32>
                  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                  %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                  %10 = sdy.sharding_constraint %9 <@mesh, [{}, {}]> : tensor<4x4xf32>
                  %11 = stablehlo.add %8, %10 : tensor<4x4xf32>
                  sdy.return %11 : tensor<4x4xf32>
                } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                return %0 : tensor<8x4xf32>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_preserves_boundary_layouts() {
    let array_type =
        ArrayType::new_static(DataType::F32, [2, 3]).with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
    let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(array_type.clone().into());
    let zero = builder.add_input(zero_type.clone().into());
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
        .unwrap();
    assert_eq!(
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![array_type.clone(), zero_type.clone()],
            &vec![zero_type, array_type],
            "main",
            None,
            None,
        )
        .unwrap(),
        indoc! {"
            module {
              func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) \
                  -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) {
                %c = stablehlo.constant dense<false> : tensor<i1>
                %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x3xi1>
                return %arg0 : tensor<2x3xf32>
              }
            }
        "},
    );
}

#[test]
fn test_to_mlir_module_for_program_lowers_captures_as_hidden_arguments() {
    let array_type = test_vector_type(4);
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(array_type.clone().into());
    let capture = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
    let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let capture_types = vec![array_type.clone()];
    let input_types = vec![array_type.clone()];
    let output_types = vec![array_type];
    let stablehlo =
        to_mlir_module_for_program(&program, capture_types.as_slice(), &input_types, &output_types, "main", None, None)
            .unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_lowers_immediate_dimension_constants_in_place() {
    // An immediate dimension constant carries its own extent from the host, so unlike a capture reference, it
    // materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden capture
    // argument. That is exactly what makes first-class extents usable where no capture table is reachable, such as
    // inside a `shard_map` manual region.
    let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
    let mut builder = ArrayIrProgramBuilder::new();
    let extent = builder.add_constant(Constant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
    let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
    let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
    let input_types: [ArrayType; 0] = [];
    let stablehlo = to_mlir_module_for_program(
        &program,
        &[],
        &input_types,
        &vec![ArrayType::scalar(DataType::I64)],
        "main",
        None,
        None,
    )
    .unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main() -> tensor<i64> {
                %c = stablehlo.constant dense<4> : tensor<i64>
                return %c : tensor<i64>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
    let array_type = test_vector_type(4);
    let mut builder = ArrayIrProgramBuilder::new();
    let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
    let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
    let input_types: [ArrayType; 0] = [];
    let output_types = vec![array_type.clone()];
    let stablehlo = to_mlir_module_for_program(
        &program,
        std::slice::from_ref(&array_type),
        &input_types,
        &output_types,
        "main",
        None,
        None,
    )
    .unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                return %arg0 : tensor<4xf32>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_forwards_capture_into_nested_regions() {
    let array_type = test_vector_type(4);
    let branch = || {
        let mut builder = ArrayIrProgramBuilder::new();
        let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
    };
    let mut builder = ArrayIrProgramBuilder::new();
    let true_region = builder.import_program(branch());
    let false_region = builder.import_program(branch());
    let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
    let output = builder
        .add_instruction(
            Operation::Condition(ConditionOperation::new()),
            vec![true_region, false_region],
            vec![predicate],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let output_types = vec![array_type.clone()];
    let stablehlo = to_mlir_module_for_program(
        &program,
        std::slice::from_ref(&array_type),
        &[ArrayType::scalar(DataType::Boolean)],
        &output_types,
        "main",
        None,
        None,
    )
    .unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
                %0 = "stablehlo.if"(%arg1) ({
                  stablehlo.return %arg0 : tensor<4xf32>
                }, {
                  stablehlo.return %arg0 : tensor<4xf32>
                }) : (tensor<i1>) -> tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
    let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(zero_type.clone().into());
    let program: FlatProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

    let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main() {
                %c = stablehlo.constant dense<false> : tensor<i1>
                %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                return
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_preserves_effects_that_produce_an_erased_output() {
    let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(zero_type.clone().into());
    let output = builder
        .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
        .unwrap()[0];
    let program: FlatProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

    let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main() {
                %c = stablehlo.constant dense<false> : tensor<i1>
                %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                %1 = stablehlo.after_all  : !stablehlo.token
                %2 = stablehlo.custom_call @ryft.print(%0, %1) {api_version = 4 : i32, backend_config = {label = "zero"}, has_side_effect = true} : (tensor<3xi1>, !stablehlo.token) -> !stablehlo.token
                return
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
    let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
    let mut builder = ArrayIrProgramBuilder::new();
    let input = builder.add_input(zero_type.clone().into());
    let program: FlatProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

    let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<i32>) -> (tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>) {
                %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<i32>) -> tensor<?xi1, #stablehlo.bounds<2>>
                %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<i32>
                %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                return %0, %2 : tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>
              }
            }
        "#},
    );
}

#[test]
fn test_to_mlir_module_for_program_isolates_bounded_dynamic_results_behind_one_barrier() {
    let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
    let static_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
    let mut builder = ArrayIrProgramBuilder::new();
    let dynamic = builder.add_input(dynamic_type.clone().into());
    let r#static = builder.add_input(static_type.clone().into());
    let program: FlatProgram =
        builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();

    let stablehlo = to_mlir_module_for_program(
        &program,
        &[],
        &vec![dynamic_type.clone(), static_type.clone()],
        &vec![dynamic_type.clone(), static_type, dynamic_type],
        "main",
        None,
        None,
    )
    .unwrap();

    // Both dynamic results pass through the barrier, while the static result and the hidden extents bypass it.
    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>) {
                %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                %1:2 = stablehlo.optimization_barrier %0, %0 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<?xf32, #stablehlo.bounds<3>>
                %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                %4 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                return %1#0, %arg1, %1#1, %3, %5 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>
              }
            }
        "#},
    );
}

#[test]
fn test_lower_mlir_module_for_program_rejects_unresolved_reference_state() {
    assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

    // The reference scan over the whole artifact catches the reference atoms and intrinsic semantics before token
    // threading, so two representative reference operations are enough to pin the diagnostic.
    let array_type = ArrayType::scalar(DataType::F32);
    let mut builder = crate::operations::ProgramBuilder::new();
    let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
    let reference = builder
        .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
        .unwrap()[0];
    builder
        .add_instruction(
            Operation::ReferenceWrite(ReferenceWriteOperation::new()),
            Vec::new(),
            vec![reference, input],
            None,
        )
        .unwrap();
    let output = builder
        .add_instruction(Operation::ReferenceFreeze(ReferenceFreezeOperation::new()), Vec::new(), vec![reference], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();

    // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
    // from reference operations receives the dedicated reference diagnostic. The state error remains behind it for
    // future state that does not come from references, and `token_threaded_effects` above pins that `OrderedState`
    // has no token slot either way.
    assert!(matches!(
        lower_mlir_module_for_program(
            &program,
            &[],
            &vec![array_type.clone()],
            &vec![array_type],
            "main",
            None,
            None,
            None,
        ),
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
    ));
}

#[test]
fn test_lower_mlir_module_for_program_rejects_preserved_reference_entries() {
    let mut builder = ArrayIrProgramBuilder::new();
    let scalar_type = ArrayType::scalar(DataType::F32);
    let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
    let snapshot =
        builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
    validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
    assert!(matches!(
        lower_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
            None,
        ),
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
    ));
}

#[test]
fn test_lower_mlir_module_for_program_with_reference_state() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
    let state_type = ArrayType::new_static(DataType::F32, [4]);
    let mut builder = crate::operations::ProgramBuilder::new();
    let _zero = builder.add_input(ArrayIrType::Array(zero_type.clone()));
    let mutated = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let _read_only = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let program: FlatProgram = builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
    let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
    let output_types = vec![state_type.clone()];
    let argument_shardings = vec![
        Sharding::replicated(mesh.clone(), 1),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
    ];
    let result_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
    let states = [
        ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
        ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, None),
    ];

    let lowered = lower_mlir_module_for_program_with_reference_state(
        &program,
        &[],
        &input_types,
        &output_types,
        "main",
        Some(argument_shardings.as_slice()),
        Some(result_shardings.as_slice()),
        None,
        &states,
        RaggedDotLoweringStrategy::default(),
        &[],
    )
    .unwrap();
    assert_eq!(lowered.signature.input_mapping(), &[None, Some(0), Some(1)]);
    assert_eq!(lowered.signature.output_mapping(), &[Some(0)]);
    let expected_signature = concat!(
        "  func.func @main(",
        "%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>, ",
        "tf.aliasing_output = 0 : i64}, ",
        "%arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> ",
        "(tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {",
    );
    assert_eq!(
        lowered.stable_hlo,
        indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=2]>
            @SIGNATURE@
                %c = stablehlo.constant dense<false> : tensor<i1>
                %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                return %arg0 : tensor<4xf32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature),
    );

    let invalid_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
    assert!(matches!(
        lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(argument_shardings.as_slice()),
            Some(result_shardings.as_slice()),
            None,
            std::slice::from_ref(&invalid_state),
            RaggedDotLoweringStrategy::default(),
            &[],
        ),
        Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
            if message == "logical state input 0 is erased from the executable boundary",
    ));

    let asymmetric_sharding_message =
        "reference-state aliases require both argument and result sharding metadata or neither";
    for (argument_shardings, result_shardings) in
        [(Some(argument_shardings.as_slice()), None), (None, Some(result_shardings.as_slice()))]
    {
        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                argument_shardings,
                result_shardings,
                None,
                &states,
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == asymmetric_sharding_message,
        ));
    }

    let mismatched_result_shardings = vec![Sharding::replicated(mesh, 1)];
    assert!(matches!(
        lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(argument_shardings.as_slice()),
            Some(mismatched_result_shardings.as_slice()),
            None,
            &states,
            RaggedDotLoweringStrategy::default(),
            &[],
        ),
        Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
            if message == "state input 1 and output 0 must use the same sharding",
    ));
}

#[test]
fn test_lower_mlir_module_for_program_with_reference_state_rejects_bounded_dynamic_aliases() {
    let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
    let state_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]));
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
    let sharding = Sharding::replicated(mesh, 1);
    let mut builder = crate::operations::ProgramBuilder::new();
    let state = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let program: FlatProgram = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
    let reference_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0));

    assert!(matches!(
        lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &vec![state_type.clone()],
            &vec![state_type],
            "main",
            Some(std::slice::from_ref(&sharding)),
            Some(std::slice::from_ref(&sharding)),
            None,
            std::slice::from_ref(&reference_state),
            RaggedDotLoweringStrategy::default(),
            &[],
        ),
        Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
            if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                           compatibility is unsupported",
    ));
}

#[test]
fn test_entry_alias_attribute_executes_in_place_on_cpu() {
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location).unwrap();
    let tensor_type = context.tensor_type(context.float32_type(), &[MlirSize::Static(4)], None, location).unwrap();
    module
        .body()
        .unwrap()
        .append_operation({
            let function_block = context.block(&[(tensor_type, location)]);
            {
                let input = function_block.argument(0).unwrap();
                let mut function_block_ref = function_block.as_ref();
                let doubled =
                    function_block_ref.append_operation(stable_hlo::add(input, input, location).unwrap()).unwrap();
                function_block_ref
                    .append_operation(func::r#return(&[doubled.result(0).unwrap()], location).unwrap())
                    .unwrap();
            }
            let mut function_region = context.region();
            function_region.append_block(function_block).unwrap();
            func::func(
                "main",
                func::FuncAttributes {
                    arguments: vec![TypeAndAttributes {
                        r#type: tensor_type.as_ref(),
                        attributes: Some(HashMap::from([
                            ("mhlo.sharding".into(), context.string_attribute("{replicated}").as_ref()),
                            (
                                "tf.aliasing_output".into(),
                                context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                            ),
                        ])),
                    }],
                    results: vec![tensor_type.into()],
                    ..Default::default()
                },
                function_region,
                location,
            )
            .unwrap()
        })
        .unwrap();
    assert!(module.verify().unwrap());
    let module = module.to_string();
    assert_eq!(
        module,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<4xf32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i64}) -> tensor<4xf32> {
                %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
    );

    let client = cpu_client(1);
    let executable = client
        .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
        .unwrap();
    let device = executable.addressable_devices().unwrap().remove(0);
    let input = client
        .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, [4], None, device, None)
        .unwrap();
    input.ready().unwrap().r#await().unwrap();
    // The input is ready and remains alive in `inputs` until execution completes, so comparing its opaque address
    // with the synchronized output address does not dereference either device pointer.
    let input_pointer = unsafe { input.unsafe_pointer().unwrap() };
    let inputs = [ExecutionInput { buffer: Arc::new(input), donatable: true }];
    let execution = executable
        .execute(
            vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
            Vec::new(),
            0,
            None,
            Some(file!()),
            None,
            None,
            &[],
        )
        .unwrap();
    let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
    let output = device_outputs.outputs.remove(0);
    let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
    // `tf.aliasing_output` is a hint rather than a guarantee, because the copy insertion of XLA may materialize a fresh
    // output. This pointer equality is therefore a deliberate canary pinned to the CPU plugin version that the
    // repository pins. If a plugin upgrade breaks it while the numeric assertion below still passes, re-evaluate
    // whether the alias hint is still honored rather than assuming a correctness bug.
    assert_eq!(output_pointer, input_pointer);
    let output = output.copy_to_host(None).unwrap().r#await().unwrap();
    assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_mlir_lowerable_value_to_dense_elements_attribute() {
    // Boolean values retain their logical order through MLIR's bit-packed dense representation.
    let boolean = CoreArray::vector(vec![true, false, true]).unwrap();
    let context = MlirContext::new();
    let location = context.unknown_location();
    let tensor_type = lower_tensor_type(boolean.r#type().as_ref(), &context, location).unwrap();
    let attribute = boolean.to_dense_elements_attribute(tensor_type, &context).unwrap();
    assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, vec![true, false, true],);

    // One-bit integers use the same packed MLIR representation as Boolean while retaining their Ryft data type.
    for (literal, expected) in [
        (
            CoreArray::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]).unwrap(),
            vec![true, false, true],
        ),
        (
            CoreArray::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap(),
            vec![true, false, true],
        ),
    ] {
        let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
        let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, expected);
    }

    // Wider sub-byte integers occupy one MLIR raw-buffer byte per element and preserve only their declared bits.
    for (literal, expected) in [
        (CoreArray::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
        (CoreArray::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
        (CoreArray::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
        (CoreArray::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
    ] {
        assert_eq!(literal_dense_bytes(&literal, expected.len()), expected);
    }

    // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
    // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
    let integer_cases = vec![
        (CoreArray::vector(vec![-127i8, 126]).unwrap(), values_to_bytes(&[-127i8, 126])),
        (CoreArray::vector(vec![-0x1234i16, 0x2345]).unwrap(), values_to_bytes(&[-0x1234i16, 0x2345])),
        (
            CoreArray::vector(vec![-0x0123_4567i32, 0x0234_5678]).unwrap(),
            values_to_bytes(&[-0x0123_4567i32, 0x0234_5678]),
        ),
        (
            CoreArray::vector(vec![-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]).unwrap(),
            values_to_bytes(&[-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]),
        ),
        (CoreArray::vector(vec![0x12u8, 0xfe]).unwrap(), values_to_bytes(&[0x12u8, 0xfe])),
        (CoreArray::vector(vec![0x1234u16, 0xfedc]).unwrap(), values_to_bytes(&[0x1234u16, 0xfedc])),
        (
            CoreArray::vector(vec![0x1234_5678u32, 0xfedc_ba98]).unwrap(),
            values_to_bytes(&[0x1234_5678u32, 0xfedc_ba98]),
        ),
        (
            CoreArray::vector(vec![(1u64 << 53) + 1, u64::MAX - 1]).unwrap(),
            values_to_bytes(&[(1u64 << 53) + 1, u64::MAX - 1]),
        ),
    ];
    for (literal, expected) in integer_cases {
        assert_eq!(
            literal_dense_bytes(&literal, expected.len()),
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
        let literal = CoreArray::from_logical_bytes(
            ArrayType::new(data_type, Shape::new(vec![CoreDimension::Static(bits.len())])),
            &bits,
        )
        .unwrap();
        assert_eq!(literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
        let mut builder = CoreProgramBuilder::<CoreArray, ArrayOperation<CoreArray>>::new();
        let output = builder
            .add_instruction(ArrayOperation::Constant(ConstantOperation::new(literal)), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains(rendered_values), "low-precision literal {data_type}: {stablehlo}");
    }

    // Standard floating-point families preserve signed zero, infinities, and NaN payload bits.
    let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
    let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
    let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY, f32::from_bits(0x7fc0_1234)];
    let f64_values = [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
    for (literal, expected) in [
        (CoreArray::vector(bf16_values.to_vec()).unwrap(), values_to_bytes(&bf16_values)),
        (CoreArray::vector(f16_values.to_vec()).unwrap(), values_to_bytes(&f16_values)),
        (CoreArray::vector(f32_values.to_vec()).unwrap(), values_to_bytes(&f32_values)),
        (CoreArray::vector(f64_values.to_vec()).unwrap(), values_to_bytes(&f64_values)),
    ] {
        assert_eq!(
            literal_dense_bytes(&literal, expected.len()),
            expected,
            "floating-point literal type {}",
            literal.r#type().as_ref(),
        );
    }

    // Complex storage interleaves independently exact real and imaginary components in source order.
    let c64_components = [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
    let c64 = CoreArray::vector(vec![
        ComplexNumber::new(c64_components[0], c64_components[1]),
        ComplexNumber::new(c64_components[2], c64_components[3]),
    ])
    .unwrap();
    assert_eq!(literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components));
    let c128_components = [
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let c128 = CoreArray::vector(vec![
        ComplexNumber::new(c128_components[0], c128_components[1]),
        ComplexNumber::new(c128_components[2], c128_components[3]),
    ])
    .unwrap();
    assert_eq!(literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components));

    // Explicit physical layouts are traversed in logical row-major order before constructing the literal.
    let layout_type =
        ArrayType::new(DataType::I16, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(2)]))
            .with_layout(Some(StridedLayout::new(vec![2, 4]).into()));
    let layout_literal = CoreArray::from_elements(layout_type, &[1i16, 2, 3, 4]).unwrap();
    assert_eq!(layout_literal.storage_bytes(), values_to_bytes(&[1i16, 3, 2, 4]));
    assert_eq!(literal_dense_bytes(&layout_literal, 8), values_to_bytes(&[1i16, 2, 3, 4]));

    // Empty tensors remain valid dense constants and carry no raw payload bytes.
    let empty = CoreArray::vector(Vec::<i32>::new()).unwrap();
    let tensor_type = lower_tensor_type(empty.r#type().as_ref(), &context, location).unwrap();
    let attribute = empty.to_dense_elements_attribute(tensor_type, &context).unwrap();
    assert_eq!(attribute.elements_count(), 0);

    // Zero literals use the established false i1 physical carrier without introducing logical payload bytes.
    let tensor_type = context
        .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
        .unwrap();
    let zero = CoreArray::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
    assert_eq!(
        zero.to_dense_elements_attribute(tensor_type, &context).unwrap().to_string(),
        "dense<false> : tensor<2xi1>",
    );

    // Tokens remain effect values and have no dense literal representation.
    let token = CoreArray::from_logical_bytes(ArrayType::new_static(DataType::Token, [2]), &[]).unwrap();
    assert!(matches!(
        token.to_dense_elements_attribute(tensor_type, &context),
        Err(Error::Lowering(LoweringError::UnsupportedElementType { data_type: DataType::Token, .. })),
    ));
}

#[test]
fn test_mlir_lowerable_value_to_dense_elements_attribute_rejects_symbolic_values() {
    let context = MlirContext::new();
    let array_type = ArrayType::scalar(DataType::F32);
    let tensor_type = lower_tensor_type(&array_type, &context, context.unknown_location()).unwrap();
    assert!(matches!(
        array_type.to_dense_elements_attribute(tensor_type, &context),
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::Concretization { message })
                if *message == "abstract XLA value `f32[]` cannot be materialized as a StableHLO literal",
        ),
    ));
    let captured = ArrayConstant::Captured(CaptureReference::new(3, array_type));
    assert!(matches!(
        captured.to_dense_elements_attribute(tensor_type, &context),
        Err(Error::Core(core_error)) if matches!(
            core_error.as_ref(),
            ryft_core::Error::Program(ProgramError::Concretization { message })
                if *message == "captured constant #3 cannot be materialized as a literal without capture bindings",
        ),
    ));
}

// ---------------------------------------------------------------------------
// Plain program StableHLO lowering tests for scalar programs
// ---------------------------------------------------------------------------

#[test]
fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
    let (_, compiled): (
        CoreArray,
        ryft_core::programs::Program<
            CoreArray,
            ryft_core::ArrayOperation<CoreArray>,
            (CoreArray, CoreArray),
            CoreArray,
        >,
    ) = TEST_ARRAY_DOMAIN
        .interpret_and_trace(
            |inputs| Ok(scalar_bilinear_sin(inputs)),
            (CoreArray::scalar(2.0).unwrap(), CoreArray::scalar(3.0).unwrap()),
        )
        .unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                %1 = stablehlo.sine %arg0 : tensor<f64>
                %2 = stablehlo.add %0, %1 : tensor<f64>
                return %2 : tensor<f64>
              }
            }
        "#},
    );
}

#[test]
fn test_plain_scalar_quartic_plus_sin_grad_stablehlo() {
    let (_, compiled): (
        CoreArray,
        ryft_core::programs::Program<CoreArray, ryft_core::ArrayOperation<CoreArray>, CoreArray, CoreArray>,
    ) = TEST_ARRAY_DOMAIN
        .interpret_and_trace(
            |x| {
                let context = x.context().clone();
                Ok(context.differentiate_at(x).gradient(scalar_quartic_plus_sin).unwrap())
            },
            CoreArray::scalar(2.0).unwrap(),
        )
        .unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();

    // The gradient computes one cosine for the derivative of `sin(x)`, expands the `x^4` chain rule into
    // multiplications, accumulates the cotangent contributions with additions, and contains no sine.
    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                %0 = stablehlo.cosine %arg0 : tensor<f64>
                %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                %1 = stablehlo.multiply %0, %cst : tensor<f64>
                %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
                %4 = stablehlo.multiply %3, %cst : tensor<f64>
                %5 = stablehlo.add %1, %4 : tensor<f64>
                %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
                %7 = stablehlo.multiply %2, %6 : tensor<f64>
                %8 = stablehlo.add %5, %7 : tensor<f64>
                %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
                %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
                %11 = stablehlo.add %8, %10 : tensor<f64>
                %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
                %13 = stablehlo.add %11, %12 : tensor<f64>
                return %13 : tensor<f64>
              }
            }
        "#},
    );
}

#[test]
fn test_plain_scalar_bilinear_sin_vjp_pullback_standalone_stablehlo() {
    // Standalone pullback over the primal operation family, produced by the reverse path that is aware of partitions.
    // The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`, where the residuals
    // are the factors `cos(x)`, `x`, and `y` at the linearization point. It lowers the backward pass, weighted by those
    // residuals, as `stablehlo.multiply`s of the cotangent against the residual inputs rather than baking the primal
    // point in as constants. This is the analogue of the standalone `vjp_fn` of JAX, with the residuals threaded as
    // explicit arguments.
    let (_, pullback): (CoreArray, _) = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::new()
        .vjp(
            |inputs, ()| Ok(scalar_bilinear_sin(inputs)),
            (CoreArray::scalar(2.0).unwrap(), CoreArray::scalar(3.0).unwrap()),
            (),
        )
        .unwrap();
    let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
    let pullback = pullback.into_program();

    let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
    // The pullback takes the output cotangent plus three residual inputs and returns one cotangent output for each of
    // `x` and `y`. It multiplies the cotangent by the residual inputs rather than baking the primal point in as
    // constants, so the lowering consists of multiplications and additions with no `stablehlo.constant`.
    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
                %1 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                %2 = stablehlo.multiply %arg1, %arg0 : tensor<f64>
                %3 = stablehlo.add %0, %2 : tensor<f64>
                return %3, %1 : tensor<f64>, tensor<f64>
              }
            }
        "#},
    );
}

#[test]
fn test_plain_scalar_bilinear_sin_grad_jit_stablehlo() {
    // The gradient of `f` staged for JIT compilation stays symbolic, like `jit(grad(f))` in JAX. It uses the traced
    // gradient path, which traces through the VJP and its pullback.
    let (_, compiled): (
        (CoreArray, CoreArray),
        ryft_core::programs::Program<
            CoreArray,
            ryft_core::ArrayOperation<CoreArray>,
            (CoreArray, CoreArray),
            (CoreArray, CoreArray),
        >,
    ) = TEST_ARRAY_DOMAIN
        .interpret_and_trace(
            |inputs| {
                let context = inputs.0.context().clone();
                Ok(context.differentiate_at(inputs).gradient(scalar_bilinear_sin).unwrap())
            },
            (CoreArray::scalar(2.0).unwrap(), CoreArray::scalar(3.0).unwrap()),
        )
        .unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
    // `cos(x)` is computed symbolically from `%arg0` rather than baked in as a constant, both inputs are referenced, and
    // the gradient contains no sine, because the derivative of sine is cosine.
    assert_eq!(
        stablehlo,
        indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                %0 = stablehlo.cosine %arg0 : tensor<f64>
                %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                %1 = stablehlo.multiply %0, %cst : tensor<f64>
                %2 = stablehlo.multiply %arg1, %cst : tensor<f64>
                %3 = stablehlo.add %1, %2 : tensor<f64>
                %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
                return %3, %4 : tensor<f64>, tensor<f64>
              }
            }
        "#},
    );
}

#[test]
fn test_collect_nested_sharding_mesh() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let sharding = Sharding::replicated(mesh.clone(), 2);
    let input_type = test_vector_type(4);
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type.clone());
    builder
        .add_instruction(
            ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(sharding),
            Vec::new(),
            vec![input],
            None,
        )
        .unwrap();
    let program = builder
        .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
        .unwrap()
        .into_unprojected::<Constant, Operation>()
        .unwrap();
    assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
    // The mesh exists only on an internal value. Lowering must still declare the symbol referenced by its
    // reshape constraint, even though neither the signature nor an explicit sharding operation supplies it.
    to_mlir_module_for_program(&program, &[], &vec![input_type.clone()], &Vec::<ArrayType>::new(), "main", None, None)
        .unwrap();
    let mut builder = ProgramBuilder::new();
    builder.add_input(input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
    let metadata_only = builder
        .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
        .unwrap()
        .into_unprojected::<Constant, Operation>()
        .unwrap();
    assert_eq!(collect_nested_sharding_mesh(&metadata_only, None).unwrap(), None);
    let incompatible = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
    assert!(matches!(
        collect_nested_sharding_mesh(&program, Some(incompatible.clone())),
        Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
            if expected == incompatible && actual == mesh,
    ));
    assert!(matches!(
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type],
            &Vec::<ArrayType>::new(),
            "main",
            Some(&[Sharding::replicated(incompatible.clone(), 1)]),
            None,
        ),
        Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
            if expected == incompatible && actual == mesh,
    ));

    // Cast placement can be the module's only mesh use, including when its result is internal.
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let input_type = test_vector_type(4);
    let output_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
    for operation in [
        CastOperation::<ArrayIrType>::checked(output_type.clone().into(), vec![input_type.clone().into()])
            .unwrap()
            .into(),
        Operation::Array(ArrayOperation::Cast(CastOperation::checked(output_type, vec![input_type.clone()]).unwrap())),
    ] {
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        builder.add_instruction(operation, vec![], vec![input], None).unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![]).unwrap();
        assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone()],
                &Vec::<ArrayType>::new(),
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return
                  }
                }
            "#},
        );
    }
}

#[test]
fn test_collect_nested_sharding_mesh_requires_one_topology() {
    let mesh_2x2 = |names: [&str; 2], r#type: MeshAxisType| {
        LogicalMesh::new(names.map(|name| MeshAxis::new(name, 2, r#type).unwrap()).to_vec()).unwrap()
    };
    let mesh = mesh_2x2(["x", "y"], MeshAxisType::Explicit);
    let input_type = test_vector_type(4);
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type.clone());
    builder
        .add_instruction(
            ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(Sharding::replicated(mesh.clone(), 2)),
            Vec::new(),
            vec![input],
            None,
        )
        .unwrap();
    let program = builder
        .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
        .unwrap()
        .into_unprojected::<Constant, Operation>()
        .unwrap();

    // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology.
    let reordered = mesh_2x2(["y", "x"], MeshAxisType::Explicit);
    assert!(matches!(
        collect_nested_sharding_mesh(&program, Some(reordered.clone())),
        Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
            if expected == reordered && actual == mesh,
    ));
    assert!(matches!(
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone()],
            &Vec::<ArrayType>::new(),
            "main",
            Some(&[Sharding::replicated(reordered.clone(), 1)]),
            None,
        ),
        Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
            if expected == reordered && actual == mesh,
    ));
    assert_eq!(
        LoweringError::MismatchedMeshTopology { expected: reordered, actual: mesh.clone() }.to_string(),
        "mesh `['x'=2:explicit, 'y'=2:explicit]` does not have the topology of the module mesh \
         `['y'=2:explicit, 'x'=2:explicit]`",
    );

    // Meshes with other axes are not merged into a union.
    let other_axes = mesh_2x2(["x", "z"], MeshAxisType::Explicit);
    assert!(matches!(
        collect_nested_sharding_mesh(&program, Some(other_axes.clone())),
        Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
            if expected == other_axes && actual == mesh,
    ));

    // Meshes that differ only in axis types share the module mesh, which keeps the axis types of its first mesh.
    let manual = mesh_2x2(["x", "y"], MeshAxisType::Manual);
    assert_eq!(collect_nested_sharding_mesh(&program, Some(manual.clone())).unwrap(), Some(manual.clone()));
    assert_eq!(
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type],
            &Vec::<ArrayType>::new(),
            "main",
            Some(&[Sharding::replicated(manual, 1)]),
            None,
        )
        .unwrap(),
        indoc! {"
            module {
              sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
              func.func @main(%arg0: tensor<4xf32> \
                  {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\", \"y\"}>}) {
                %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}], replicated={\"x\", \"y\"}> : tensor<2x2xf32>
                return
              }
            }
        "},
    );
}

#[test]
fn test_normalize_function_name() {
    assert_eq!(normalize_function_name(" main ").unwrap(), "main");
    assert_eq!(normalize_function_name(" @main ").unwrap(), "main");
    assert!(matches!(
        normalize_function_name(" @ "),
        Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name.is_empty(),
    ));
    assert!(matches!(
        normalize_function_name("main function"),
        Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name == "main function",
    ));
}
