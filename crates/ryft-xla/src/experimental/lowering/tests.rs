use std::sync::Arc;

use indoc::indoc;
use pretty_assertions::assert_eq;
use ryft_core::{
    AddOperation, AtomId, CaptureReference, CastOperation, ConditionOperation, Context, Cos, Differentiate, Dimension,
    DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Dot, DotDimensionNumbers, EagerContext,
    EffectClasses, MeshAxis, MeshAxisType, NegOperation, OneLike, Placeholder, Program, ProgramBuilder,
    ProvenanceScope, ReduceOperation, ReductionKind, ReshapeOperation, ReverseModeDifferentiate, Sin, TracedShardMap,
    Transpose, ZeroLike, trace_shard_map,
};
use ryft_mlir::{Attribute, Operation as MlirOperation, Region, Size as MlirSize};
use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

use crate::experimental::domains::XlaDomain;
use crate::experimental::lowering::types::lower_array_ir_type;
use crate::experimental::ops::{FlatXlaProgram, XlaProgramBuilder as CompositeXlaProgramBuilder};
use crate::operations::assertions::ensure_assertion_handler_registered;
use crate::operations::jit_call::JitCallOperation;
use crate::operations::sharding::shard_map::to_mlir_module;
use crate::tests::{values_from_bytes, values_to_bytes};

use super::*;

/// Homogeneous builder retained only for tests of the test-only plain-program lowering helper.
pub(crate) type XlaProgramBuilder = ProgramBuilder<XlaArrayConstant, ArrayOperation<XlaArrayConstant>>;

/// Homogeneous program retained only for tests of the test-only plain-program lowering helper.
pub(crate) type PlainXlaProgram =
    Program<XlaArrayConstant, ArrayOperation<XlaArrayConstant>, Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>;

pub(crate) fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
}

pub(crate) fn test_vector_type(length: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
}

pub(crate) fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(cols)]))
}

/// Global array type of `data_type` with the static `global_shape`, placed by `sharding`.
pub(crate) fn static_sharded_array_type(data_type: DataType, global_shape: &[usize], sharding: Sharding) -> ArrayType {
    ArrayType::new(data_type, Shape::new(global_shape.iter().copied().map(Dimension::Static).collect()))
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

/// Compilation options for an SPMD executable with `partition_count` partitions, partitioned by Shardy.
pub(crate) fn test_spmd_compilation_options(partition_count: usize) -> ryft_pjrt::protos::CompilationOptions {
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};

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

pub(crate) fn dynamic_dimension(name: &str, exclusive_upper_bound: Option<usize>) -> Dimension {
    DimensionVariable::new(name, DimensionBounds::non_negative(exclusive_upper_bound).unwrap()).into()
}

/// One physical value of the mixed executable ABI used by the reshape execution tests: `f64` or `i32` array
/// storage with its physical dimensions on input and the row-major elements inside its logical dimensions on
/// output, or one `i64` dimension, which on output also carries the hidden runtime extents that follow the logical
/// outputs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MixedValue {
    /// `f64` storage of one array together with its dimensions.
    Array(Vec<f64>, Vec<u64>),

    /// `i32` storage of one array together with its dimensions (e.g., the per-item extents of a ragged batch).
    Integers(Vec<i32>, Vec<u64>),

    /// One `i64` first-class dimension or hidden runtime extent.
    Dimension(i64),
}

/// Compiles one composite program for the mixed executable ABI and executes it once on `client`. Each logical
/// input receives the matching entry of `inputs` (array storage sized to the allocation bound, or one dimension),
/// and every bounded dynamic input axis then receives its runtime extent from `extents` in boundary order.
pub(crate) fn execute_mixed_program(
    client: &ryft_pjrt::Client<'_>,
    program: &FlatXlaProgram,
    inputs: &[MixedValue],
    extents: &[i32],
) -> Result<Vec<MixedValue>, ryft_pjrt::Error> {
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
            MixedValue::Dimension(value) => client.buffer(
                values_to_bytes(&[*value]).as_slice(),
                BufferType::I64,
                &[],
                None,
                device.clone(),
                None,
            )?,
        });
    }
    for extent in extents {
        buffers.push(client.buffer(
            values_to_bytes(&[*extent]).as_slice(),
            BufferType::I32,
            &[],
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

pub(crate) fn xla_identity_branch(input_type: ArrayType) -> PlainXlaProgram {
    let mut builder = XlaProgramBuilder::new();
    let input = builder.add_input(input_type);
    builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
}

pub(crate) fn xla_neg_branch(input_type: ArrayType) -> PlainXlaProgram {
    let mut builder = XlaProgramBuilder::new();
    let input = builder.add_input(input_type);
    let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

/// Lifts an array-only test fixture into the production composite XLA program representation.
pub(crate) fn unproject_plain_program(program: PlainXlaProgram) -> FlatXlaProgram {
    program.into_unprojected::<XlaConstant, XlaOperation>().unwrap()
}

#[test]
fn test_collect_nested_sharding_mesh() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let sharding = Sharding::replicated(mesh.clone(), 2);
    let input_type = test_vector_type(4);
    let mut builder = XlaProgramBuilder::new();
    let input = builder.add_input(input_type.clone());
    builder
        .add_instruction(
            ReshapeOperation::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                .with_output_sharding(sharding),
            Vec::new(),
            vec![input],
            None,
        )
        .unwrap();
    let program = builder
        .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![], vec![Placeholder], vec![])
        .unwrap()
        .into_unprojected::<XlaConstant, XlaOperation>()
        .unwrap();
    assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
    // The mesh exists only on an internal value. Lowering must still declare the symbol referenced by its
    // reshape constraint, even though neither the signature nor an explicit sharding operation supplies it.
    to_mlir_module_for_program(&program, &[], &vec![input_type.clone()], &Vec::<ArrayType>::new(), "main", None, None)
        .unwrap();
    let mut builder = XlaProgramBuilder::new();
    builder.add_input(input_type.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap());
    let metadata_only = builder
        .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![], vec![Placeholder], vec![])
        .unwrap()
        .into_unprojected::<XlaConstant, XlaOperation>()
        .unwrap();
    assert_eq!(collect_nested_sharding_mesh(&metadata_only, None).unwrap(), None);
    let incompatible = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
    assert!(matches!(
        collect_nested_sharding_mesh(&program, Some(incompatible.clone())),
        Err(LoweringError::IncompatibleNestedMeshes)
    ));
    assert!(matches!(
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type],
            &Vec::<ArrayType>::new(),
            "main",
            Some(&[Sharding::replicated(incompatible, 1)]),
            None,
        ),
        Err(LoweringError::IncompatibleNestedMeshes)
    ));

    // Cast placement can be the module's only mesh use, including when its result is internal.
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let input_type = test_vector_type(4);
    let output_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
    for operation in [
        CastOperation::<ArrayIrType>::checked(output_type.clone().into(), vec![input_type.clone().into()])
            .unwrap()
            .into(),
        XlaOperation::Array(ArrayOperation::Cast(
            CastOperation::checked(output_type, vec![input_type.clone()]).unwrap(),
        )),
    ] {
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        builder.add_instruction(operation, vec![], vec![input], None).unwrap();
        let program = builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder], vec![]).unwrap();
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
fn test_composite_lowering_rejects_unresolved_reference_type() {
    use ryft_core::ReferenceType;

    let context = MlirContext::new();
    let location = context.unknown_location();
    let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
    assert!(matches!(
        lower_array_ir_type(&reference_type, &context, location),
        Err(LoweringError::UnresolvedReference { construct }) if construct == reference_type.to_string(),
    ));
}

#[test]
fn test_composite_lowering_rejects_bound_manual_output_placement() {
    // A composite `XlaOperation::Cast` (i.e., a `CastOperation<ArrayIrType>` rather than a projected
    // `ArrayOperation::Cast`) lowers without passing through `ArrayOperation::lower`, so only the family-level validation
    // of `XlaOperation::lower` sees its output placement. Its first output requests a mesh that redefines the bound manual
    // axis `x` as explicit and shards a local dimension over it, which only the region boundary may do.
    let vector_type = test_vector_type(4);
    let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let explicit_type = vector_type
        .clone()
        .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
        .unwrap();
    let cast = |output_type: &ArrayType, input_type: &ArrayType| {
        XlaOperation::Cast(
            CastOperation::unchecked(output_type.clone().into(), vec![input_type.clone().into()]).unwrap(),
        )
    };
    let mut builder = CompositeXlaProgramBuilder::new();
    let input = builder.add_input(vector_type.clone().into());
    let placed = builder.add_instruction(cast(&explicit_type, &vector_type), Vec::new(), vec![input], None).unwrap()[0];
    let output =
        builder.add_instruction(cast(&vector_type, &explicit_type), Vec::new(), vec![placed], None).unwrap()[0];
    let body = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert!(matches!(
        lower_replicated_shard_map_body(body, 2),
        Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message }))
            if message == "`cast` cannot assign bound manual mesh axes to local output dimensions",
    ));
}

#[test]
fn test_xla_lowering_rejects_unresolved_reference_state_before_token_threading() {
    use ryft_core::{ReferenceFreezeOperation, ReferenceNewOperation, ReferenceWriteOperation};

    assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

    // The artifact-wide reference scan catches the reference atoms and intrinsic semantics before token threading,
    // so two representative reference operations are enough to pin the diagnostic.
    let array_type = ArrayType::scalar(DataType::F32);
    let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
    let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
    let reference = builder
        .add_instruction(XlaOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
        .unwrap()[0];
    builder
        .add_instruction(
            XlaOperation::ReferenceWrite(ReferenceWriteOperation::new()),
            Vec::new(),
            vec![reference, input],
            None,
        )
        .unwrap();
    let output = builder
        .add_instruction(
            XlaOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
            Vec::new(),
            vec![reference],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();

    // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
    // from reference operations receives the dedicated reference diagnostic; the state error remains behind it
    // for future non-reference state (`token_threaded_effects` above pins that `OrderedState` has no token slot
    // either way).
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
        Err(LoweringError::UnresolvedReference { construct }) if construct == "program",
    ));
}

#[test]
fn test_reference_state_aliases_use_physical_indices_and_preserve_shardings() {
    use ryft_core::ReferenceSource;

    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
    let state_type = ArrayType::new_static(DataType::F32, [4]);
    let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
    let _zero = builder.add_input(ArrayIrType::Array(zero_type.clone()));
    let mutated = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let _read_only = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let program: FlatXlaProgram = builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
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
        ),
        Err(LoweringError::InvalidReferenceStateAbi { message })
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
            ),
            Err(LoweringError::InvalidReferenceStateAbi { message })
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
        ),
        Err(LoweringError::InvalidReferenceStateAbi { message })
            if message == "state input 1 and output 0 must use the same sharding",
    ));
}

#[test]
fn test_bounded_dynamic_reference_state_alias_is_rejected_before_lowering() {
    use ryft_core::ReferenceSource;

    let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
    let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
    let sharding = Sharding::replicated(mesh, 1);
    let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
    let state = builder.add_input(ArrayIrType::Array(state_type.clone()));
    let program: FlatXlaProgram = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
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
        ),
        Err(LoweringError::InvalidReferenceStateAbi { message })
            if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                           compatibility is unsupported",
    ));
}

#[test]
fn test_entry_alias_attribute_executes_in_place_on_cpu() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use crate::tests::{values_from_bytes, values_to_bytes};

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
    assert!(
        module.contains("%arg0: tensor<4xf32> {mhlo.sharding = \"{replicated}\", tf.aliasing_output = 0 : i64}",),
        "{module}",
    );

    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let executable = client
        .compile(
            &PjrtProgram::Mlir { bytecode: module.into_bytes() },
            &CompilationOptions {
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
                environment_option_overrides: HashMap::new(),
                target_config: None,
                allow_in_place_mlir_modification: false,
                matrix_unit_operand_precision: Precision::Default as i32,
            },
        )
        .unwrap();
    let device = executable.addressable_devices().unwrap().remove(0);
    let input = client
        .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, &[4], None, device, None)
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
        )
        .unwrap();
    let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
    let output = device_outputs.outputs.remove(0);
    let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
    // `tf.aliasing_output` is a hint rather than a guarantee (XLA's copy insertion may materialize a fresh
    // output), so this pointer equality is a deliberate canary pinned to the repository's pinned CPU plugin: if a
    // plugin upgrade breaks it while the numeric assertion below still passes, re-evaluate whether the alias hint
    // is still honored rather than assuming a correctness bug.
    assert_eq!(output_pointer, input_pointer);
    let output = output.copy_to_host(None).unwrap().r#await().unwrap();
    assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_to_mlir_module_for_program_preserves_boundary_layouts() {
    use ryft_core::{Layout, TiledLayout};

    let array_type =
        ArrayType::new_static(DataType::F32, [2, 3]).with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
    let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
    let mut builder = CompositeXlaProgramBuilder::new();
    let input = builder.add_input(array_type.clone().into());
    let zero = builder.add_input(zero_type.clone().into());
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
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
    let mut builder = CompositeXlaProgramBuilder::new();
    let input = builder.add_input(array_type.clone().into());
    let capture = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
    let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let capture_types = vec![array_type.clone()];
    let input_types = vec![array_type.clone()];
    let output_types = vec![array_type];
    let stablehlo =
        to_mlir_module_for_program(&program, capture_types.as_slice(), &input_types, &output_types, "main", None, None)
            .unwrap();

    assert!(
        stablehlo.contains("func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>"),
        "{stablehlo}",
    );
    assert!(stablehlo.contains("stablehlo.add %arg1, %arg0 : tensor<4xf32>"), "{stablehlo}");
}

#[test]
fn test_to_mlir_module_for_program_lowers_immediate_dimension_constants_in_place() {
    use ryft_core::DimensionToScalarOperation;

    // An immediate dimension constant carries its own host-sized extent, and so — unlike a capture reference —
    // it materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden
    // capture argument. That is exactly what makes first-class extents usable where no capture table is
    // reachable, such as inside a `shard_map` manual region.
    let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
    let mut builder = CompositeXlaProgramBuilder::new();
    let extent = builder.add_constant(XlaConstant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
    let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
        .unwrap();
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

    assert!(stablehlo.contains("func.func @main() -> tensor<i64>"), "{stablehlo}");
    assert!(stablehlo.contains("stablehlo.constant dense<4> : tensor<i64>"), "{stablehlo}");
}

#[test]
fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
    let array_type = test_vector_type(4);
    let mut builder = CompositeXlaProgramBuilder::new();
    let output = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
        .unwrap();
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

    assert!(stablehlo.contains("func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32>"), "{stablehlo}");
    assert!(stablehlo.contains("return %arg0 : tensor<4xf32>"), "{stablehlo}");
    assert!(!stablehlo.contains("stablehlo.constant"), "{stablehlo}");
}

#[test]
fn test_to_mlir_module_for_program_forwards_capture_into_nested_regions() {
    let array_type = test_vector_type(4);
    let branch = || {
        let mut builder = CompositeXlaProgramBuilder::new();
        let output = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap()
    };
    let mut builder = CompositeXlaProgramBuilder::new();
    let true_region = builder.import_program(branch());
    let false_region = builder.import_program(branch());
    let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
    let output = builder
        .add_instruction(
            XlaOperation::Condition(ConditionOperation::new()),
            vec![true_region, false_region],
            vec![predicate],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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

    assert!(
        stablehlo.contains("func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32>",),
        "{stablehlo}",
    );
    assert_eq!(stablehlo.matches("stablehlo.return %arg0 : tensor<4xf32>").count(), 2, "{stablehlo}");
    assert!(!stablehlo.contains("stablehlo.constant"), "{stablehlo}");
}

/// Stages one `jit_call` to `callee` (interned as a shared callee root region) over `inputs` in `builder`.
pub(crate) fn add_xla_jit_call(
    builder: &mut CompositeXlaProgramBuilder,
    callee: &Arc<FlatXlaProgram>,
    inputs: Vec<AtomId>,
) -> AtomId {
    let callee_region = builder.intern_callee(callee, None).unwrap();
    builder
        .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee_region], inputs, None)
        .unwrap()[0]
}

/// Wraps one flat vector program in a replicated manual `shard_map` and lowers the enclosing program.
pub(crate) fn lower_replicated_shard_map_body(
    body_program: FlatXlaProgram,
    device_count: usize,
) -> Result<String, LoweringError> {
    use ryft_core::ShardMapOperation;

    let vector_type = test_vector_type(4);
    let mesh = test_manual_mesh("x", device_count);
    let sharding = Sharding::replicated(mesh.clone(), 1);
    let operation = ShardMapOperation::from_program(
        &body_program,
        vec![vector_type.clone().into()],
        ShardMap::new(mesh, vec![sharding.clone()], vec![sharding], vec!["x".to_string()]).unwrap(),
    )
    .unwrap();
    let mut builder = CompositeXlaProgramBuilder::new();
    let input = builder.add_input(vector_type.clone().into());
    let body = builder.import_program(body_program);
    let output =
        builder.add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body], vec![input], None)?[0];
    let program =
        builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])?;
    let input_types = vec![vector_type.clone()];
    let output_types = vec![vector_type];
    to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
}

#[test]
fn test_to_mlir_module_renders_a_full_add_module() {
    let global_input_type = test_vector_type(8);
    let mesh = test_manual_mesh("x", 4);
    let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
        |x| x.clone() + x,
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
    )
    .unwrap();

    assert_eq!(
        to_mlir_module(&traced, "main").unwrap(),
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
fn test_to_mlir_module_renders_provenance_locations() {
    let global_input_type = test_vector_type(8);
    let mesh = test_manual_mesh("x", 4);
    let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
        |x| {
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
            product + x
        },
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
    )
    .unwrap();

    // The scoped addition carries the nested name-location chain, the fused multiplication carries a
    // metadata-free fused location, the unknown-provenance addition keeps the base (unknown) location, and
    // module/function scaffolding keeps the base location. Locations render because provenance-carrying modules
    // serialize with debug information enabled (and the non-pretty, parsable form).
    assert_eq!(
        to_mlir_module(&traced, "main").unwrap(),
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
fn test_to_mlir_module_renders_constants_and_supported_ops() {
    let global_input_type = test_matrix_type(4, 4);
    let mesh = test_manual_mesh("x", 2);
    let traced: TracedShardMap<XlaDomain<'static>, ArrayType, ArrayType> = trace_shard_map(
        |x| {
            let product = x.transpose(vec![1, 0]).unwrap().dot(&x, &DotDimensionNumbers::matmul()).unwrap();
            let waveform = (-product).cos().unwrap().sin().unwrap();
            (waveform.clone() * waveform.one_like().unwrap()) + waveform.zero_like().unwrap()
        },
        global_input_type,
        mesh.clone(),
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
        Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
    )
    .unwrap();

    assert_eq!(
        to_mlir_module(&traced, "kernel").unwrap(),
        indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=2]>
              func.func @kernel(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
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

/// Builds a one-instruction program applying the given single-operand operation to an input of the given data
/// type and dimensions, and returns the result of lowering it to a rendered StableHLO module.
pub(crate) fn lowered_unary_module(
    operation: ArrayOperation<XlaArrayConstant>,
    data_type: DataType,
    dimensions: Vec<usize>,
) -> Result<String, LoweringError> {
    let input_type = ArrayType::new(data_type, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()));
    let mut builder = XlaProgramBuilder::new();
    let input = builder.add_input(input_type);
    let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
    let program = builder
        .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    to_mlir_module_for_plain_program(&program, "main")
}

// Returns the single-device CPU compilation contract shared by grouped-dot execution and rejection coverage.
pub(crate) fn ragged_dot_cpu_compilation_options() -> ryft_pjrt::protos::CompilationOptions {
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};

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
    }
}

/// Builds a one-instruction program reducing an input of the given data type and dimensions, and returns the
/// result of lowering it to a rendered StableHLO module.
pub(crate) fn lowered_reduce_module(
    data_type: DataType,
    kind: ReductionKind,
    axes: Vec<usize>,
    dimensions: Vec<usize>,
) -> Result<String, LoweringError> {
    lowered_unary_module(ArrayOperation::Reduce(ReduceOperation::new(axes, kind)), data_type, dimensions)
}

// ---------------------------------------------------------------------------
// Plain-program StableHLO lowering tests for scalar programs
// ---------------------------------------------------------------------------

fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + ryft_core::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
}

fn scalar_quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + ryft_core::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
}

static TEST_ARRAY_DOMAIN: EagerContext<Array, ArrayOperation<Array>> =
    EagerContext::<Array, ArrayOperation<Array>>::new();

#[test]
fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
    let (_, compiled): (
        Array,
        ryft_core::programs::Program<Array, ryft_core::ArrayOperation<Array>, (Array, Array), Array>,
    ) = TEST_ARRAY_DOMAIN
        .interpret_and_trace(
            |inputs| Ok(scalar_bilinear_sin(inputs)),
            (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
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
    let (_, compiled): (Array, ryft_core::programs::Program<Array, ryft_core::ArrayOperation<Array>, Array, Array>) =
        TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context
                        .differentiate_at(x)
                        .gradient(scalar_quartic_plus_sin)
                        .expect("scalar gradient should succeed"))
                },
                Array::scalar(2.0).unwrap(),
            )
            .unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
    println!("=== ryft grad(x^4 + sin(x)) StableHLO ===\n{stablehlo}");

    // Verify key structural properties matching JAX's output:
    // 1. Single cosine for the sin(x) derivative
    assert_eq!(stablehlo.matches("stablehlo.cosine").count(), 1, "should have exactly one cosine");
    // 2. Multiple multiplies for the x^4 chain rule
    assert!(stablehlo.matches("stablehlo.multiply").count() >= 5, "should have several multiplies for x^4 chain");
    // 3. Multiple adds accumulating cotangent contributions
    assert!(stablehlo.matches("stablehlo.add").count() >= 3, "should have adds for cotangent accumulation");
    // 4. No sine in the gradient (it's consumed in forward, derivative is cosine)
    assert_eq!(stablehlo.matches("stablehlo.sine").count(), 0, "gradient should not contain sine");
}

#[test]
fn test_plain_scalar_bilinear_sin_vjp_pullback_standalone_stablehlo() {
    // Standalone pullback over the primal operation family, produced by the partition-aware reverse path.
    // The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`, where the
    // residuals are the linearization-point factors `cos(x)`, `x`, and `y`, and lowers the residual-weighted
    // backward pass as `stablehlo.multiply`s of the cotangent against those residual inputs rather than baking the
    // primal point in as constants — the analogue of JAX's standalone `vjp_fn`, with the residuals threaded as
    // explicit arguments.
    let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
        .vjp(
            |inputs, ()| Ok(scalar_bilinear_sin(inputs)),
            (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
            (),
        )
        .unwrap();
    let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
    println!("=== ryft standalone vjp_pullback(x*y + sin(x)) StableHLO ===\n{stablehlo}");

    // The pullback takes the output cotangent plus three residual inputs and returns two cotangent outputs (for x
    // and y).
    assert!(
        stablehlo.contains(
            "func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> \
             (tensor<f64>, tensor<f64>)",
        ),
        "pullback should take a cotangent plus residual inputs and return two outputs, but got:\n{stablehlo}",
    );
    // The residual-weighted reverse multiplies the cotangent by the residual inputs rather than baking the primal
    // point in as constants, so the lowering is multiplies-and-adds with no `stablehlo.constant`.
    assert!(stablehlo.contains("stablehlo.multiply"), "residual-weighted reverse should multiply: \n{stablehlo}");
    assert!(
        !stablehlo.contains("stablehlo.constant"),
        "residuals are runtime inputs, not baked-in constants, but got:\n{stablehlo}",
    );
}

#[test]
fn test_plain_scalar_bilinear_sin_grad_jitted_stablehlo() {
    // grad(f) wrapped in JIT â€” symbolic, like JAX's jit(grad(f)).
    // Uses the traced value-and-gradient path that traces through vjp+pullback.
    let (_, compiled): (
        (Array, Array),
        ryft_core::programs::Program<Array, ryft_core::ArrayOperation<Array>, (Array, Array), (Array, Array)>,
    ) = TEST_ARRAY_DOMAIN
        .interpret_and_trace(
            |inputs| {
                let context = inputs.0.context().clone();
                Ok(context
                    .differentiate_at(inputs)
                    .gradient(scalar_bilinear_sin)
                    .expect("scalar gradient should succeed"))
            },
            (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
        )
        .unwrap();

    let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
    println!("=== ryft jit(grad(bilinear_sin)) StableHLO ===\n{stablehlo}");

    // cos(x) should be computed symbolically from %arg0, NOT as a baked-in constant.
    assert!(stablehlo.contains("stablehlo.cosine %arg0"), "cos(x) should be computed from input");
    // Should reference both inputs.
    assert!(stablehlo.contains("%arg0") && stablehlo.contains("%arg1"), "should reference both inputs");
    // No sine (sin derivative = cosine, not sine).
    assert!(!stablehlo.contains("stablehlo.sine"), "gradient should not contain sine");
}

#[test]
fn test_xla_lowering_rejects_preserved_reference_entries() {
    use ryft_core::kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
    use ryft_core::{ReferenceReadOperation, ReferenceType};

    let mut builder = CompositeXlaProgramBuilder::new();
    let scalar_type = ArrayType::scalar(DataType::F32);
    let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
    let snapshot =
        builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
    assert!(validate_kernel_body(program.entry_region_ref(), &contract).is_ok());
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
        Err(LoweringError::UnresolvedReference { construct }) if construct == "program",
    ));
}
