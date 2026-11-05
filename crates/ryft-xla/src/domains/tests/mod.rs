//! Shared XLA domain fixtures and domain configuration coverage.

use std::sync::Arc;

use indoc::indoc;
use pretty_assertions::assert_eq;
#[cfg(feature = "cuda-13")]
use ryft_core::operations::attention::dot_product_attention;
use ryft_core::operations::attention::{
    AttentionConfiguration, AttentionImplementation, AttentionInputs, differentiable_dot_product_attention,
};
use ryft_core::operations::manipulation::indexing::index;
use ryft_core::{
    AddOperation, AndOperation, ArgMaxOperation, ArgMinOperation, Array as CoreArray, ArrayBatch, ArrayBatchingPolicy,
    ArrayIrBatch, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArrayType, Assert, AssertOperation, AtomId, AxisIndexOperation, BatchAxis,
    BatchableOperation, BatchingContext, CollectiveOptions, CompareOperation, ComparisonDirection,
    CompilationCacheDomain, CompilationTracer, Compile, CompiledFunctionDispatcher, ConcatenateOperation,
    ConstantOperation, Context, ConvertElementTypeOperation, CumulativeKind, CumulativeOperation, DataType, Device,
    DeviceMesh, Differentiate, Dimension as CoreDimension, DimensionAddOperation, DimensionBounds,
    DimensionDivOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionMulOperation,
    DimensionOperation, DimensionRemOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation,
    DimensionToScalarOperation, DimensionType, DimensionValue, DimensionVariable, DomainTracingContext,
    DotDimensionNumbers, DotOperation, DynamicArrayExtentBatchingPolicy, DynamicBroadcastOperation, DynamicGather,
    DynamicReshape, DynamicReshapeOperation, DynamicScatter, DynamicSlice, DynamicSliceOperation,
    DynamicSliceWithDimensions, DynamicUpdateSlice, EmptyRegionDriver, Gather, GatherDimensionNumbers, GatherMode,
    GatherOperation, GatherOptions, Indexing, IotaOperation, Linearization, LogicalMesh, MeshAxis, MeshAxisType,
    NegOperation, OneOperation, Operation as CoreOperation, OperationPayloadProjection, ParallelAllGatherOperation,
    ParallelAllGatherOutputVariance, ParallelAllToAllOperation, ParallelPermuteOperation, ParallelReduceOperation,
    ParallelVaryOperation, Placeholder, PrintOperation, Program as CoreProgram, ProgramBuilder as CoreProgramBuilder,
    ProgramError, ProjectedContext, ReduceOperation, ReductionKind, ReferenceAddUpdateOperation,
    ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceType, Reshape, ReshapeOperation,
    ScaledDotOperation, Scatter, ScatterDimensionNumbers, ScatterMode, ScatterOperation, ScatterOptions,
    ScatterReductionKind, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, SortDirection,
    SortOperation, StagingContext, StaticShape, SubOperation, TracingContext, Type, Typed, TypedProgram,
    Value as CoreValue, ValueProjection, batch,
};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
#[cfg(feature = "cuda-13")]
use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

use crate::arrays_v0::host::materialize_dense_array_bytes;
use crate::experimental::ops::{Constant, FlatProgram, Operation, ProgramBuilder};
use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
use crate::{Array, Error, FromPjrt, Value};

use super::*;

/// Checks compiled basic and broadcast advanced reads and repeated indexed updates on the provided backend.
fn assert_compiled_indexing(client: &Client<'_>) {
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_types = vec![
        ArrayType::new_static(DataType::F64, [3, 4]),
        ArrayType::new_static(DataType::I64, [2, 1]),
        ArrayType::new_static(DataType::I64, [3]),
        ArrayType::new_static(DataType::F64, [2, 3]),
    ];
    let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| {
            let selection = index![&inputs[1], &inputs[2]];
            vec![
                inputs[0].at(&index![.. by -1, 1..4 by 2]).get(&GatherOptions::new()).unwrap(),
                inputs[0].at(&selection).get(&GatherOptions::new()).unwrap(),
                inputs[0].at(&selection).add(&inputs[3], &ScatterOptions::new()).unwrap(),
                inputs[0].at(&index![.. by i128::MIN, ...]).get(&GatherOptions::new()).unwrap(),
            ]
        },
        input_types.clone(),
        &domain,
        mesh.clone(),
    )
    .unwrap();
    let bytes = [
        values_to_bytes(&[0_f64, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]),
        values_to_bytes(&[2_i64, 0]),
        values_to_bytes(&[1_i64, 1, 3]),
        values_to_bytes(&[1_f64; 6]),
    ];
    let inputs = input_types
        .into_iter()
        .zip(bytes)
        .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
        .collect::<Vec<_>>();
    let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
    assert_eq!(outputs[0].shape().as_slice(), &[3, 2]);
    assert_eq!(read_f64s(&client, &outputs[0]), vec![9., 11., 5., 7., 1., 3.]);
    assert_eq!(outputs[1].shape().as_slice(), &[2, 3]);
    assert_eq!(read_f64s(&client, &outputs[1]), vec![9., 9., 11., 1., 1., 3.]);
    // A huge negative stride selects only the last row and must not wrap when lowered to signed attributes.
    assert_eq!(outputs[3].shape().as_slice(), &[1, 4]);
    assert_eq!(read_f64s(&client, &outputs[3]), vec![8., 9., 10., 11.]);
    // Duplicate advanced coordinates contribute twice; the original input remains a value, not a mutable view.
    assert_eq!(read_f64s(&client, &outputs[2]), vec![0., 3., 2., 4., 4., 5., 6., 7., 8., 11., 10., 12.]);
}

/// Replays one symbolic indexing graph at distinct concrete extents on the provided backend.
fn assert_compiled_indexing_symbolic_extent(client: &Client<'_>) {
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(4, Some(6)).unwrap()));
    let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| {
            // Runtime dimension operations live in the mixed IR family. Explicitly retain that family while
            // indexing symbolic extents, and project the array results back to the public JIT signature.
            let inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
            let selection = index![&inputs[1]];
            vec![
                inputs[0].at(&selection).get(&GatherOptions::new()).unwrap().into_projected().unwrap(),
                inputs[0].at(&selection).add(&inputs[2], &ScatterOptions::new()).unwrap().into_projected().unwrap(),
            ]
        },
        vec![
            ArrayType::new(DataType::F64, Shape::new(vec![extent])),
            ArrayType::new_static(DataType::I64, [3]),
            ArrayType::new_static(DataType::F64, [3]),
        ],
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let program = staged.source_program().program().clone();
    let linearization = program.linearize_with_respect_to(&[2]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    // Reuse one symbolic source graph at distinct concrete extents, including negative index normalization.
    let execute = |values: &[f64]| {
        let input_types = vec![
            ArrayType::new_static(DataType::F64, [values.len()]),
            ArrayType::new_static(DataType::I64, [3]),
            ArrayType::new_static(DataType::F64, [3]),
            ArrayType::new_static(DataType::F64, [3]),
            ArrayType::new_static(DataType::F64, [values.len()]),
        ];
        let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                let mut cotangents = inputs.split_off(3);
                let context = inputs[0].context().clone();
                let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
                cotangents.extend(outputs.split_off(2));
                outputs.extend(pullback.interpret_in_context(&context, cotangents).unwrap());
                outputs
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let bytes = [
            values_to_bytes(values),
            values_to_bytes(&[1_i64, 1, -1]),
            values_to_bytes(&[10_f64, 20., 30.]),
            values_to_bytes(&[0_f64; 3]),
            values_to_bytes(values),
        ];
        let inputs = input_types
            .into_iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect::<Vec<_>>();
        domain
            .interpret(&compiled.executable_function(), inputs)
            .unwrap()
            .iter()
            .map(|output| read_f64s(&client, output))
            .collect::<Vec<_>>()
    };
    assert_eq!(execute(&[1., 2., 3., 4.]), vec![vec![2., 2., 4.], vec![1., 32., 3., 34.], vec![2., 2., 4.]]);
    assert_eq!(execute(&[1., 2., 3., 4., 5.]), vec![vec![2., 2., 5.], vec![1., 32., 3., 4., 35.], vec![2., 2., 5.]]);
}

pub(super) fn domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .into_iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    DeviceMesh::new(logical_mesh, devices).unwrap()
}

pub(super) fn external_reference_read_program(referent_type: ArrayType) -> FlatProgram {
    let mut builder = ProgramBuilder::new();
    let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(referent_type)));
    let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
    builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap()
}

pub(super) fn array_domain<'c>(client: &'c Client<'c>) -> ProjectedContext<Domain<'c>, ArrayType> {
    ProjectedContext::new(Session::new(client).domain())
}

pub(super) fn program_array<'a, 'c>(value: &'a Value<'c>) -> &'a Array<'c> {
    let Value::Array(array) = value else {
        panic!("expected an array IR value");
    };
    array
}

pub(super) fn replicated_vector_type(mesh: &DeviceMesh, size: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(size)]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap()
}

pub(super) fn replicated_scalar_type(mesh: &DeviceMesh, data_type: DataType) -> ArrayType {
    ArrayType::scalar(data_type)
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
        .unwrap()
}

pub(super) fn f32_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[f32]) -> Array<'c> {
    let r#type = replicated_vector_type(mesh, values.len());
    Array::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
}

pub(super) fn f32_scalar<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, value: f32) -> Array<'c> {
    let r#type = replicated_scalar_type(mesh, DataType::F32);
    Array::from_host_buffer(domain, r#type, mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
}

pub(super) fn f64_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[f64]) -> Array<'c> {
    let r#type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(values.len())]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
    Array::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes::<f64>(values).as_slice()).unwrap()
}

pub(super) fn boolean_vector<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, values: &[bool]) -> Array<'c> {
    let r#type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(values.len())]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
    let values = values.iter().copied().map(u8::from).collect::<Vec<_>>();
    Array::from_host_buffer(domain, r#type, mesh.clone(), values.as_slice()).unwrap()
}

pub(super) fn boolean_scalar<'c>(domain: &Domain<'c>, mesh: &DeviceMesh, value: bool) -> Array<'c> {
    let r#type = replicated_scalar_type(mesh, DataType::Boolean);
    Array::from_host_buffer(domain, r#type, mesh.clone(), &[u8::from(value)]).unwrap()
}

pub(super) fn read_f32s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<f32>(bytes.as_slice())
}

pub(super) fn read_bf16s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<u16>(bytes.as_slice())
        .into_iter()
        .map(|bits| half::bf16::from_bits(bits).to_f32())
        .collect()
}

pub(super) fn read_f64s(client: &Client<'_>, array: &Array<'_>) -> Vec<f64> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<f64>(bytes.as_slice())
}

pub(super) fn read_i32s(client: &Client<'_>, array: &Array<'_>) -> Vec<i32> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<i32>(bytes.as_slice())
}

pub(super) fn read_u64s(client: &Client<'_>, array: &Array<'_>) -> Vec<u64> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<u64>(bytes.as_slice())
}

pub(super) fn read_i64s(client: &Client<'_>, array: &Array<'_>) -> Vec<i64> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<i64>(bytes.as_slice())
}

pub(super) fn read_booleans(client: &Client<'_>, array: &Array<'_>) -> Vec<bool> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap()
        .into_iter()
        .map(|value| value != 0)
        .collect()
}

/// Builds the kind-sensitive bounded-dynamic padding fixture shared by CPU and CUDA execution tests.
pub(super) fn data_derived_padding_fixture(size_type: ArrayType, extent: DimensionVariable) -> FlatProgram {
    let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
    let index_type = dynamic_type.clone().with_data_type(DataType::U64);
    let matrix_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(4)]),
    );

    // The iota and its negation make physical suffix lanes numerically distinguishable from every admitted live
    // prefix. This catches both reduction identities and the opposite sentinels needed by ascending and descending
    // sort. The boolean reductions cover their own false/true identities, and dot covers zero-padded contraction.
    let mut builder = ProgramBuilder::new();
    let size = builder.add_input(size_type.into());
    let dimension = builder
        .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
        .unwrap()[0];
    let iota = builder
        .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
        .unwrap()[0];
    let indices = builder
        .add_instruction(IotaOperation::new(index_type, 0).unwrap(), Vec::new(), vec![dimension], None)
        .unwrap()[0];
    let negative = builder.add_instruction(NegOperation::new(), Vec::new(), vec![iota], None).unwrap()[0];
    let sorted_ascending = builder
        .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative], None)
        .unwrap()[0];
    let sorted_descending = builder
        .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota], None)
        .unwrap()[0];
    let argmin_indices = builder
        .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative, indices], None)
        .unwrap()[1];
    let argmax_indices = builder
        .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota, indices], None)
        .unwrap()[1];
    let sum = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![iota], None)
        .unwrap()[0];
    // Shift the iota to nonzero factors so including physical padding changes the product.
    let ones = builder.add_instruction(OneOperation::new(dynamic_type), Vec::new(), vec![dimension], None).unwrap()[0];
    let factors = builder.add_instruction(AddOperation::new(), Vec::new(), vec![iota, ones], None).unwrap()[0];
    let product = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Product), Vec::new(), vec![factors], None)
        .unwrap()[0];
    let maximum = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Max), Vec::new(), vec![iota], None)
        .unwrap()[0];
    let minimum = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Min), Vec::new(), vec![negative], None)
        .unwrap()[0];
    let any_input = builder
        .add_instruction(
            ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan)),
            Vec::new(),
            vec![negative, iota],
            None,
        )
        .unwrap()[0];
    let any = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Any), Vec::new(), vec![any_input], None)
        .unwrap()[0];
    let all_input = builder
        .add_instruction(
            ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThanOrEqual)),
            Vec::new(),
            vec![iota, negative],
            None,
        )
        .unwrap()[0];
    let all = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::All), Vec::new(), vec![all_input], None)
        .unwrap()[0];
    let dot = builder
        .add_instruction(DotOperation::new(DotDimensionNumbers::inner_product()), Vec::new(), vec![iota, iota], None)
        .unwrap()[0];

    // The rank-2 `i + j` matrix carries the data-derived extent on an axis that neither the reduction nor the scan
    // touches, which is the configuration those two disciplines actually admit: a log-sum-exp `reduce` consumes
    // the static axis and still produces a dynamically shaped output, and the prefix scan consumes the static
    // axis while every row it emits keeps the dynamic extent.
    let rows = builder
        .add_instruction(IotaOperation::new(matrix_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
        .unwrap()[0];
    let columns = builder
        .add_instruction(IotaOperation::new(matrix_type, 1).unwrap(), Vec::new(), vec![dimension], None)
        .unwrap()[0];
    let matrix = builder.add_instruction(AddOperation::new(), Vec::new(), vec![rows, columns], None).unwrap()[0];
    let log_sum_exp = builder
        .add_instruction(ReduceOperation::new(vec![1], ReductionKind::LogSumExp), Vec::new(), vec![matrix], None)
        .unwrap()[0];
    let cumulative_sum = builder
        .add_instruction(CumulativeOperation::new(1, CumulativeKind::Sum), Vec::new(), vec![matrix], None)
        .unwrap()[0];

    // Index reductions over the data-derived axis would select a physical suffix lane if its padding were not
    // masked, because the suffix lanes of the increasing iota are larger than every live lane and those of its
    // negation are smaller. The matrix covers both a reduced data-derived axis and a data-derived axis that the
    // reduction keeps.
    let argmax = builder
        .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![iota], None)
        .unwrap()[0];
    let argmin = builder
        .add_instruction(ArgMinOperation::new(0, DataType::I32), Vec::new(), vec![negative], None)
        .unwrap()[0];
    let row_argmax = builder
        .add_instruction(ArgMaxOperation::new(1, DataType::I32), Vec::new(), vec![matrix], None)
        .unwrap()[0];
    let column_argmax = builder
        .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![matrix], None)
        .unwrap()[0];
    builder
        .build::<Vec<Constant>, Vec<Constant>>(
            vec![
                sorted_ascending,
                sorted_descending,
                argmin_indices,
                argmax_indices,
                sum,
                maximum,
                minimum,
                any,
                all,
                dot,
                log_sum_exp,
                cumulative_sum,
                product,
                argmax,
                argmin,
                row_argmax,
                column_argmax,
            ],
            vec![Placeholder],
            vec![Placeholder; 17],
        )
        .unwrap()
}

/// Builds the bounded-dynamic scaled-dot and attention proof fixture. The attention values vary along the
/// key/value sequence axis, so including any physical suffix lane changes every valid output row. Attention is
/// staged by the training function with the provided implementation, whose reverse mode differentiation calls the
/// forward attention definition with its residual and the backward attention definition.
pub(super) fn data_derived_scaled_dot_attention_fixture(
    size_type: ArrayType,
    extent: DimensionVariable,
    scaled_element_type: DataType,
    scaled_scale_type: DataType,
    attention_implementation: AttentionImplementation,
) -> FlatProgram {
    /// Binds `operation` to `inputs` in the context of `value` and returns its only output.
    fn bind<O: Into<Operation>>(value: &Tracer<'static>, operation: O, inputs: &[Tracer<'static>]) -> Tracer<'static> {
        value.domain().bind(operation, Vec::<FlatProgram>::new(), inputs).unwrap().remove(0)
    }

    let scaled_block_size = match scaled_element_type {
        DataType::F8E4M3FN | DataType::F8E5M2 => 32,
        _ => 16,
    };
    let attention_batch = DimensionVariable::new("attention_batch", extent.bounds().clone());
    let scaled_lhs_type = ArrayType::new(
        scaled_element_type,
        Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(scaled_block_size)]),
    );
    let scaled_lhs_scale_type = ArrayType::new(
        scaled_scale_type,
        Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(1)]),
    );
    let scaled_rhs_type = ArrayType::new(
        scaled_element_type,
        Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(scaled_block_size)]),
    );
    let scaled_rhs_scale_type =
        ArrayType::new(scaled_scale_type, Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(1)]));
    let attention_type = ArrayType::new(
        DataType::BF16,
        Shape::new(vec![
            CoreDimension::Dynamic(attention_batch.clone()),
            CoreDimension::Dynamic(extent.clone()),
            CoreDimension::Static(1),
            CoreDimension::Static(8),
        ]),
    );
    let attention = differentiable_dot_product_attention::<Domain<'static>>(
        AttentionConfiguration::new().with_scale(1.0).with_implementation(attention_implementation),
    );
    let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
        |inputs: Vec<Tracer<'static>>| {
            let size = &inputs[0];
            let dimension = bind(size, DimensionFromScalarOperation::new(extent.clone().into()), &[size.clone()]);
            let batch_dimension =
                bind(size, DimensionFromScalarOperation::new(attention_batch.clone().into()), &[size.clone()]);

            // `scaled_dot`: the runtime row extent is non-contracting. Both dynamic element/scale operands share
            // it, while the static right operand fixes one output column and the static block shape fixes
            // `k = 16`.
            let scaled_lhs = bind(size, OneOperation::new(scaled_lhs_type.clone()), &[dimension.clone()]);
            let scaled_lhs_scales = bind(size, OneOperation::new(scaled_lhs_scale_type.clone()), &[dimension.clone()]);
            let scaled_rhs = bind(size, OneOperation::new(scaled_rhs_type.clone()), &[]);
            let scaled_rhs_scales = bind(size, OneOperation::new(scaled_rhs_scale_type.clone()), &[]);
            let scaled_dot = bind(
                size,
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                &[scaled_lhs, scaled_rhs, scaled_lhs_scales, scaled_rhs_scales],
            );

            // Attention: equal query/key scores make the live softmax uniform, while values equal their sequence
            // index. Explicit lengths of `extent - 1` exercise the supported further-restriction contract: visible
            // query rows return the mean `(extent - 2) / 2`, while the final logical row and all physical suffix
            // lanes stay zero. Backward has exact `dQ = 0`, `dK[j] = 8 * (j - mean)`, and `dV = 1` on visible
            // rows and zeros elsewhere.
            let extents = [batch_dimension.clone(), dimension.clone()];
            let query = bind(size, OneOperation::new(attention_type.clone()), &extents);
            let key = bind(size, OneOperation::new(attention_type.clone()), &extents);
            let value = bind(size, IotaOperation::new(attention_type.clone(), 1).unwrap(), &extents);
            let output_cotangent = bind(size, OneOperation::new(attention_type.clone()), &extents);
            let logical_length = bind(size, DimensionToScalarOperation, &[dimension.clone()]);
            let logical_length =
                bind(size, ConvertElementTypeOperation::<ArrayType>::new(DataType::I32, false), &[logical_length]);
            let one = CoreArray::from_elements(ArrayType::scalar(DataType::I32), &[1_i32]).unwrap();
            let one = bind(size, ConstantOperation::new(one), &[]);
            let length = bind(size, SubOperation::new(), &[logical_length, one]);
            let lengths = bind(size, DynamicBroadcastOperation::new(Vec::new()), &[length, batch_dimension.clone()]);
            let inputs = AttentionInputs {
                query_sequence_lengths: Some(lengths.clone()),
                key_value_sequence_lengths: Some(lengths),
                ..AttentionInputs::new(query, key, value)
            };
            let (output, pullback) = size.domain().differentiate_at(inputs).vjp(|inputs| attention.call(inputs))?;
            let cotangents = pullback.apply_materialized(output_cotangent)?;
            Ok(vec![scaled_dot, output, cotangents.query, cotangents.key, cotangents.value])
        },
        vec![ArrayIrType::from(size_type)],
    )
    .unwrap();
    program.into_flat_program()
}

/// Kind of `shard_map` program that [`shard_map_differential_program`] builds to compare the reference backend's
/// emulation of `shard_map` with its execution through XLA.
#[derive(Copy, Clone, Debug)]
enum ShardMapDifferentialCase {
    /// Tiled identity of an input sharded along the manual axis `x`, together with its sum across `x`.
    IdentityAndSum,

    /// Coordinate of every device along the manual axis `x`.
    AxisIndex,

    /// Invariant tiled all-gather across the manual axis `x`.
    AllGather,

    /// Tiled all-to-all across the manual axis `x` that moves a row-sharded matrix to a column-sharded one.
    AllToAll,

    /// Permutation across the manual axis `x` that sends the shard of device 0 to device 1 and zeros to device 0.
    Permute,

    /// Map over the manual axis `x` whose body maps over the manual axis `y` and sums across the enclosing axis
    /// `x`.
    Nested,

    /// Map that adds its shard of an update into its shard of a caller-allocated reference and reads it back.
    Reference,

    /// Map that reads a reference replicated along the manual axis `x` at the coordinate of every device along `x`.
    VaryingIndexReference,

    /// Map that doubles a reference replicated along the manual axis `x` on every device and reads it back.
    ReplicatedReference,
}

/// Builds a program over `input_types` whose outputs are the atoms that `build` adds to it.
fn build_differential_program<V: CoreValue<Type = ArrayIrType>, O: CoreOperation<Type = ArrayIrType>>(
    input_types: Vec<ArrayIrType>,
    build: impl FnOnce(&mut CoreProgramBuilder<V, O>, &[AtomId]) -> Vec<AtomId>,
) -> CoreProgram<V, O, Vec<V>, Vec<V>> {
    let mut builder = CoreProgramBuilder::<V, O>::new();
    let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let outputs = build(&mut builder, &inputs);
    let output_count = outputs.len();
    builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap()
}

/// Builds the program of `case` over any operation family `O` that lifts the array IR family over `A`, so that the
/// reference backend's emulation (over [`ArrayIrOperation<Array>`]) and XLA execution (over [`Operation`]) run
/// the same program. Every program input and output is an array: the reference of
/// [`ShardMapDifferentialCase::Reference`] is allocated from a program input, and its final value is frozen into a
/// trailing program output.
fn shard_map_differential_program<A, V, O>(case: ShardMapDifferentialCase) -> CoreProgram<V, O, Vec<V>, Vec<V>>
where
    A: CoreValue<Type = ArrayType>,
    V: CoreValue<Type = ArrayIrType>,
    O: CoreOperation<Type = ArrayIrType> + OperationPayloadProjection + From<ArrayIrOperation<A>>,
{
    let array_operation = ArrayIrOperation::<A>::from;
    let mesh = match case {
        ShardMapDifferentialCase::Nested => LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ]),
        _ => LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]),
    }
    .unwrap();
    let sharded = |axis: &str| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded([axis])]).unwrap();
    let replicated = Sharding::replicated(mesh.clone(), 1);
    let f32_type = |shape: &[usize], sharding: &Sharding| {
        ArrayType::new_static(DataType::F32, shape.to_vec()).with_sharding(sharding.clone()).unwrap()
    };
    let sum_over_x = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
    let (boundary, global_input_types, body) = match case {
        ShardMapDifferentialCase::IdentityAndSum => {
            let boundary = ShardMapBoundary::new(
                mesh.clone(),
                vec![sharded("x")],
                vec![sharded("x"), replicated],
                vec!["x".into()],
            )
            .unwrap();
            let global_type = f32_type(&[4], &sharded("x"));
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                vec![inputs[0], builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap()[0]]
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::AxisIndex => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()]).unwrap();
            let global_type = f32_type(&[2], &sharded("x"));
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, _| {
                let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                let index = array_operation(ArrayOperation::AxisIndex(index));
                let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                let reshape = ReshapeOperation::new(Shape::new(vec![CoreDimension::Static(1)]));
                let reshape = array_operation(ArrayOperation::Reshape(reshape));
                builder.add_instruction(reshape, Vec::new(), vec![index], None).unwrap().to_vec()
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::AllGather => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()]).unwrap();
            let global_type = f32_type(&[4], &sharded("x"));
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                let gather = ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Invariant,
                );
                let gather = array_operation(ArrayOperation::ParallelAllGather(gather.with_mesh(mesh.clone())));
                builder.add_instruction(gather, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::AllToAll => {
            let rows =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
            let columns =
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]);
            let (rows, columns) = (rows.unwrap(), columns.unwrap());
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![rows.clone()], vec![columns], vec!["x".into()]).unwrap();
            let global_type = f32_type(&[2, 4], &rows);
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                let exchange = ParallelAllToAllOperation::new("x".to_string(), 2, 1, 0, CollectiveOptions::tiled());
                let exchange = array_operation(ArrayOperation::ParallelAllToAll(exchange.with_mesh(mesh.clone())));
                builder.add_instruction(exchange, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::Permute => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![sharded("x")], vec!["x".into()]).unwrap();
            let global_type = f32_type(&[4], &sharded("x"));
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let body = build_differential_program::<V, O>(vec![local_type.into()], |builder, inputs| {
                let permute = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
                let permute = array_operation(ArrayOperation::ParallelPermute(permute.with_mesh(mesh.clone())));
                builder.add_instruction(permute, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::Nested => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("x")], vec![replicated], vec!["x".into()]).unwrap();
            let global_type = f32_type(&[4], &sharded("x"));
            let outer_local_type = boundary.local_input_type(0, &global_type).unwrap();
            let inner_boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("y")], vec![sharded("y")], vec!["y".into()]).unwrap();
            let inner_local_type = inner_boundary.local_input_type(0, &outer_local_type).unwrap();
            let inner_body = build_differential_program::<V, O>(vec![inner_local_type.into()], |builder, inputs| {
                let sum = array_operation(ArrayOperation::ParallelReduce(sum_over_x));
                builder.add_instruction(sum, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            let inner_operation =
                ShardMapOperation::from_program(&inner_body, vec![outer_local_type.clone().into()], inner_boundary)
                    .unwrap();
            let body = build_differential_program::<V, O>(vec![outer_local_type.into()], |builder, inputs| {
                let inner_body = builder.import_program(inner_body);
                let inner_operation = ArrayIrOperation::<A>::from(inner_operation);
                builder.add_instruction(inner_operation, vec![inner_body], inputs.to_vec(), None).unwrap().to_vec()
            });
            (boundary, vec![global_type.into()], body)
        }
        ShardMapDifferentialCase::Reference => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![sharded("x"); 2], vec![sharded("x")], vec!["x".into()])
                    .unwrap();
            let global_type = f32_type(&[4], &sharded("x"));
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let input_types = vec![ReferenceType::new(local_type.clone()).into(), local_type.into()];
            let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                let add_update = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                let add_update = ArrayIrOperation::<A>::from(add_update);
                builder.add_instruction(add_update, Vec::new(), inputs.to_vec(), None).unwrap();
                let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                let read = ArrayIrOperation::<A>::from(read);
                builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            (boundary, vec![ReferenceType::new(global_type.clone()).into(), global_type.into()], body)
        }
        ShardMapDifferentialCase::VaryingIndexReference => {
            let table_sharding = Sharding::replicated(mesh.clone(), 2);
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![table_sharding.clone()], vec![sharded("x")], vec!["x".into()])
                    .unwrap();
            let global_type = f32_type(&[2, 3], &table_sharding);
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let input_types = vec![ReferenceType::new(local_type).into()];
            let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                let index = array_operation(ArrayOperation::AxisIndex(index));
                let index = builder.add_instruction(index, Vec::new(), Vec::new(), None).unwrap()[0];
                let read =
                    ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new().with_transforms(
                        vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
                    );
                let read = ArrayIrOperation::<A>::from(read);
                builder.add_instruction(read, Vec::new(), vec![inputs[0], index], None).unwrap().to_vec()
            });
            (boundary, vec![ReferenceType::new(global_type).into()], body)
        }
        ShardMapDifferentialCase::ReplicatedReference => {
            let boundary =
                ShardMapBoundary::new(mesh.clone(), vec![replicated.clone()], vec![sharded("x")], vec!["x".into()])
                    .unwrap();
            let global_type = f32_type(&[2], &replicated);
            let local_type = boundary.local_input_type(0, &global_type).unwrap();
            let input_types = vec![ReferenceType::new(local_type).into()];
            let body = build_differential_program::<V, O>(input_types, |builder, inputs| {
                let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                let read = ArrayIrOperation::<A>::from(read);
                let state = builder.add_instruction(read.clone(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                let add_update = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                let add_update = ArrayIrOperation::<A>::from(add_update);
                builder.add_instruction(add_update, Vec::new(), vec![inputs[0], state], None).unwrap();
                let state = builder.add_instruction(read, Vec::new(), vec![inputs[0]], None).unwrap()[0];
                let vary = array_operation(ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".into())));
                builder.add_instruction(vary, Vec::new(), vec![state], None).unwrap().to_vec()
            });
            (boundary, vec![ReferenceType::new(global_type).into()], body)
        }
    };
    let operation = ShardMapOperation::from_program(&body, global_input_types, boundary).unwrap();

    // Reference inputs of the map are allocated from the corresponding program inputs and frozen after the map.
    let program_input_types = operation
        .global_input_types()
        .iter()
        .map(|r#type| match r#type {
            ArrayIrType::Reference(reference) => ArrayIrType::Array(reference.referent().clone()),
            r#type => r#type.clone(),
        })
        .collect();
    build_differential_program::<V, O>(program_input_types, |builder, inputs| {
        let mut map_inputs = inputs.to_vec();
        let mut references = Vec::new();
        for (input, r#type) in map_inputs.iter_mut().zip(operation.global_input_types()) {
            if r#type.is_reference() {
                let new = ArrayIrOperation::<A>::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new());
                *input = builder.add_instruction(new, Vec::new(), vec![*input], None).unwrap()[0];
                references.push(*input);
            }
        }
        let body = builder.import_program(body);
        let operation = ArrayIrOperation::<A>::from(operation);
        let mut outputs = builder.add_instruction(operation, vec![body], map_inputs, None).unwrap().to_vec();
        for reference in references {
            let freeze = ArrayIrOperation::<A>::from(ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new());
            outputs.push(builder.add_instruction(freeze, Vec::new(), vec![reference], None).unwrap()[0]);
        }
        outputs
    })
}

/// Runs the program of `case` (refer to [`shard_map_differential_program`]) through the reference backend's
/// emulation and through XLA execution on all devices of `client`, which must match the devices of the mesh of
/// `case`, over the same integer-valued inputs, and checks that both produce exactly the same outputs, including
/// their types.
fn assert_shard_map_emulation_matches_execution(client: &Client<'_>, case: ShardMapDifferentialCase) {
    type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
    let emulated_program =
        shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>(case);
    let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
    let input_types = xla_program
        .input_types()
        .into_owned()
        .into_iter()
        .map(|r#type| match r#type {
            ArrayIrType::Array(r#type) => r#type,
            r#type => panic!("differential programs have array inputs only but got `{type}`"),
        })
        .collect::<Vec<_>>();
    let input_bytes = input_types
        .iter()
        .enumerate()
        .map(|(index, r#type)| {
            let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
            let elements = (0..element_count).map(|element| (100 * index + element + 1) as f32).collect::<Vec<_>>();
            values_to_bytes::<f32>(&elements)
        })
        .collect::<Vec<_>>();

    // The reference backend emulates every device of the map on the host.
    let emulated_inputs = input_types
        .iter()
        .zip(&input_bytes)
        .map(|(r#type, bytes)| ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap()))
        .collect();
    let emulated_outputs = emulated_program.interpret(emulated_inputs).unwrap();

    // XLA compiles the map into one manual computation and executes it on the devices of `client`.
    let devices = client.addressable_devices().unwrap();
    let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
    let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
    let domain = Session::new(client).domain();
    let inputs = input_types
        .into_iter()
        .zip(input_bytes)
        .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
        .collect();
    let lowered = domain.lower_program(&xla_program, 0, &Options::new(mesh)).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let outputs = domain.execute_program(&compiled, inputs).unwrap();
    let outputs = outputs
        .iter()
        .map(|output| {
            let bytes = materialize_dense_array_bytes(output).unwrap();
            ArrayIrValue::Array(CoreArray::from_logical_bytes(output.r#type().into_owned(), &bytes).unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs, emulated_outputs, "`{case:?}` outputs differ between XLA and the emulation");
}

/// Linearizes the program of `case` (refer to [`shard_map_differential_program`]) with respect to its first input
/// and runs its primal program and then its pullback through the reference backend's emulation and through XLA
/// execution on all devices of `client`, which must match the devices of the mesh of `case`, over the same
/// integer-valued inputs and output seeds, and checks that both produce exactly the same input cotangent, including
/// its type. The pullback consumes the residuals of the primal program of the same backend.
fn assert_shard_map_pullback_emulation_matches_execution(client: &Client<'_>, case: ShardMapDifferentialCase) {
    type ArrayMember = <Constant as ValueProjection<ArrayType>>::Projected;
    let emulated_program =
        shard_map_differential_program::<CoreArray, ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>(case);
    let xla_program = shard_map_differential_program::<ArrayMember, Constant, Operation>(case);
    let emulated = emulated_program.entry_region_ref().linearize(&[0]).unwrap();
    let xla = xla_program.entry_region_ref().linearize(&[0]).unwrap();
    let array_types = |types: Vec<ArrayIrType>| {
        types
            .into_iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type,
                r#type => panic!("differential programs have array inputs and outputs only but got `{type}`"),
            })
            .collect::<Vec<_>>()
    };
    let input_types = array_types(xla_program.input_types().into_owned());
    let output_types = array_types(xla_program.output_types().into_owned());
    let bytes = |offset: usize, r#type: &ArrayType| {
        let element_count = r#type.static_shape().unwrap().dimensions().iter().product::<usize>();
        let elements = (0..element_count).map(|element| (offset + element + 1) as f32).collect::<Vec<_>>();
        values_to_bytes::<f32>(&elements)
    };
    let input_bytes =
        input_types.iter().enumerate().map(|(index, r#type)| bytes(100 * index, r#type)).collect::<Vec<_>>();
    let seed_bytes = output_types
        .iter()
        .enumerate()
        .map(|(index, r#type)| bytes(1000 + 100 * index, r#type))
        .collect::<Vec<_>>();

    // The reference backend emulates every device of both maps on the host.
    let emulated_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
        types
            .iter()
            .zip(bytes)
            .map(|(r#type, bytes)| ArrayIrValue::Array(CoreArray::from_logical_bytes(r#type.clone(), bytes).unwrap()))
            .collect::<Vec<_>>()
    };
    let mut primal_outputs = emulated.primal().interpret(emulated_values(&input_types, &input_bytes)).unwrap();
    let residuals = primal_outputs.split_off(primal_outputs.len() - emulated.residual_count());
    let mut pullback_inputs = emulated_values(&output_types, &seed_bytes);
    pullback_inputs.extend(residuals);
    let emulated_cotangents = emulated.pullback().unwrap().program().interpret(pullback_inputs).unwrap();

    // XLA compiles each map into one manual computation and executes it on the devices of `client`.
    let devices = client.addressable_devices().unwrap();
    let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
    let mesh = DeviceMesh::new(input_types[0].sharding().unwrap().mesh().clone(), devices).unwrap();
    let domain = Session::new(client).domain();
    let xla_values = |types: &[ArrayType], bytes: &[Vec<u8>]| {
        types
            .iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), bytes).unwrap())
            .collect::<Vec<_>>()
    };
    let execute = |program: &FlatProgram, inputs: Vec<_>| {
        let lowered = domain.lower_program(program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        domain.execute_program(&compiled, inputs).unwrap()
    };
    let mut primal_outputs = execute(xla.primal(), xla_values(&input_types, &input_bytes));
    let residuals = primal_outputs.split_off(primal_outputs.len() - xla.residual_count());
    let mut pullback_inputs = xla_values(&output_types, &seed_bytes);
    pullback_inputs.extend(residuals);
    let cotangents = execute(&xla.pullback().unwrap().into_program(), pullback_inputs)
        .iter()
        .map(|cotangent| {
            let bytes = materialize_dense_array_bytes(cotangent).unwrap();
            ArrayIrValue::Array(CoreArray::from_logical_bytes(cotangent.r#type().into_owned(), &bytes).unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(cotangents, emulated_cotangents, "`{case:?}` cotangents differ between XLA and the emulation");
}

#[test]
fn test_domain_accessors_return_constructor_arguments() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let cloned = domain.clone();

    assert_eq!(domain.mesh().unwrap(), &mesh);
    assert_eq!(domain.compilation_options(), &CompilationOptions::default());
    assert!(Arc::ptr_eq(&domain.compilation_options, &cloned.compilation_options));
    assert!(size_of::<Domain<'_>>() < size_of::<CompilationOptions>());
}

#[test]
fn test_domain_with_compilation_options() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
    let domain = Session::new(&client).domain();
    let mut compilation_options = CompilationOptions::default();
    compilation_options.compile_portable_executable = true;
    let sibling = domain.with_compilation_options(compilation_options.clone());
    assert!(Arc::ptr_eq(domain.session(), sibling.session()));
    assert!(Arc::ptr_eq(&domain.effect_scope, &sibling.effect_scope));
    assert_eq!(sibling.compilation_options(), &compilation_options);
    assert_eq!(domain.compilation_options(), &CompilationOptions::default());
}

mod assertions;
mod dynamic;
mod pipeline;
