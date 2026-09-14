//! Macro-authored portable kernels exercised independently of the handwritten JAX ABI probes.

use pretty_assertions::assert_eq;

use ryft_core::kernels::{
    AsyncCopyOperation, Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, ScratchOperation,
    WaitOperation, whole_array_parameter,
};
use ryft_core::{Array, ArrayType, Context, DataType, ReferenceRead, ReferenceWrite};

/// Adds independently loaded tiles, including a partially populated final output tile.
#[ryft_core::kernels::kernel(requires = left.shape()[0] == right.shape()[0])]
fn vector_add(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]], tile = [256], boundary = masked)] output: &mut Array,
) {
    let [block] = output.tile_index();
    let left_tiles = left.tiles([256]).pad(0.0);
    let right_tiles = right.tiles([256]).pad(0.0);
    output.store(left_tiles.load([block]) + right_tiles.load([block]));
}

/// Reduces a complete vector to one scalar through the canonical reduction operation.
#[ryft_core::kernels::kernel]
fn sum(#[input(data_type = F32, rank = 1)] input: &Array, #[output(data_type = F32, shape = [])] output: &mut Array) {
    output.store(input.load().sum([0]));
}

/// Carries tiled dot products through a staged reduction loop with partial tiles on every axis.
#[ryft_core::kernels::kernel(requires = left.shape()[1] == right.shape()[0])]
fn matmul(
    #[input(data_type = F32, rank = 2)] left: &Array,
    #[input(data_type = F32, rank = 2)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], tile = [32, 32], boundary = masked)]
    output: &mut Array,
) {
    let [row, column] = output.tile_index();
    let left_tiles = left.tiles([32, 32]).pad(0.0);
    let right_tiles = right.tiles([32, 32]).pad(0.0);
    let mut accumulator = zeros::<f32>([32, 32]);
    for depth in 0..left.shape()[1].div_ceil(32) {
        let left_tile = left_tiles.load([row, depth]);
        let right_tile = right_tiles.load([depth, column]);
        accumulator += left_tile.dot(right_tile);
    }
    output.store(accumulator);
}

/// One immutable source definition and independent scalar oracle shared by host and GPU verification.
struct KernelCase {
    /// Human-readable diagnostic name.
    name: &'static str,

    /// The canonical traced source.
    definition: KernelDefinition,

    /// Checked full-array argument types.
    input_types: Vec<ArrayType>,

    /// Host inputs uploaded without changing their element representation.
    inputs: Vec<Vec<f32>>,

    /// Scalar-loop oracle for the sole logical output.
    expected: Vec<f32>,
}

/// Exercises actual asynchronous global-to-scratch communication through canonical explicit builders.
/// This case does not claim asynchronous syntax in the kernel macro.
fn async_copy_case() -> KernelCase {
    let input_type = ArrayType::new_static(DataType::F32, [67]);
    let scratch = ScratchOperation::new(input_type.clone(), 16).unwrap();
    let operation = KernelCallOperation::new(
        Grid::new(vec![]).unwrap(),
        vec![
            whole_array_parameter(input_type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
            whole_array_parameter(input_type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
        ],
    )
    .unwrap();
    let definition = KernelDefinition::trace(operation, |(references, _)| {
        let context = references[0].context();
        let destination = context.bind(scratch, vec![], &[])?.remove(0);
        let token = context.bind(AsyncCopyOperation, vec![], &[references[0].clone(), destination.clone()])?.remove(0);
        context.bind(WaitOperation, vec![], &[token])?;
        references[1].write(&destination.read()?)
    })
    .unwrap();
    let values = (0..67).map(|index| index as f32 - 20.0).collect::<Vec<_>>();
    KernelCase {
        name: "async_copy",
        definition,
        input_types: vec![input_type],
        inputs: vec![values.clone()],
        expected: values,
    }
}

/// Constructs deterministic exact-integer floating-point cases so reduction ordering cannot obscure a wrong result.
fn kernel_cases() -> Vec<KernelCase> {
    let vector_type = ArrayType::new_static(DataType::F32, [1003]);
    let left = (0..1003).map(|index| (index % 11) as f32 - 5.0).collect::<Vec<_>>();
    let right = (0..1003).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
    let expected = left.iter().zip(&right).map(|(left, right)| left + right).collect();
    let vector = KernelCase {
        name: "vector_add",
        definition: vector_add::definition(&vector_type, &vector_type).unwrap(),
        input_types: vec![vector_type.clone(), vector_type],
        inputs: vec![left, right],
        expected,
    };
    let reduction_type = ArrayType::new_static(DataType::F32, [257]);
    let values = (0..257).map(|index| (index % 13) as f32 - 6.0).collect::<Vec<_>>();
    let expected = vec![values.iter().copied().sum()];
    let reduction = KernelCase {
        name: "sum",
        definition: sum::definition(&reduction_type).unwrap(),
        input_types: vec![reduction_type],
        inputs: vec![values],
        expected,
    };
    let left_type = ArrayType::new_static(DataType::F32, [33, 35]);
    let right_type = ArrayType::new_static(DataType::F32, [35, 34]);
    let left = (0..33 * 35).map(|index| (index % 5) as f32 - 2.0).collect::<Vec<_>>();
    let right = (0..35 * 34).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
    let mut expected = vec![0.0; 33 * 34];
    for row in 0..33 {
        for column in 0..34 {
            for depth in 0..35 {
                expected[row * 34 + column] += left[row * 35 + depth] * right[depth * 34 + column];
            }
        }
    }
    vec![
        vector,
        reduction,
        KernelCase {
            name: "matmul",
            definition: matmul::definition(&left_type, &right_type).unwrap(),
            input_types: vec![left_type, right_type],
            inputs: vec![left, right],
            expected,
        },
        async_copy_case(),
    ]
}

/// Checks the portable reference interpreter against the independent scalar oracle.
fn interpret_case(case: &KernelCase) {
    let inputs = case
        .input_types
        .iter()
        .zip(&case.inputs)
        .map(|(r#type, values)| Array::from_elements(r#type.clone(), values).unwrap())
        .collect::<Vec<_>>();
    let output_type = case.definition.operation().output_types()[0].clone();
    let ryft_core::ArrayIrType::Array(output_type) = output_type else { unreachable!() };
    assert_eq!(
        case.definition.interpret(inputs, 1024).unwrap(),
        vec![Array::from_elements(output_type, &case.expected).unwrap()],
        "{}",
        case.name,
    );
}

#[test]
fn test_macro_kernels_interpretation() {
    for case in kernel_cases() {
        interpret_case(&case);
    }
}

#[cfg(feature = "mosaic-gpu")]
mod gpu {
    use std::env;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use ryft_core::kernels::KernelSchedule;
    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProjectedValue, Typed};
    use ryft_mosaic::kernels::gpu::{Compiler, Options, Target};
    use ryft_xla::experimental::XlaDomainError;
    use ryft_xla::kernels::{MosaicGpuEmbedding, XlaKernelCompilerBinding, stage_kernel};
    use ryft_xla::{CompiledXlaFunction, FromPjrt, XlaCompileTracer, XlaOptions, XlaSession, compile_with_options};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// Compiles and executes the canonical kernel definition through the selected Mosaic adapter and XLA GPU runtime.
    fn execute_case<'c>(client: &'c crate::Client<'c>, case: KernelCase) -> Result<Vec<f32>, XlaDomainError> {
        let device = client.addressable_devices().unwrap().remove(0);
        let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
            panic!("the CUDA PJRT device must report its compute capability");
        };
        let (major, minor) = capability.split_once('.').unwrap();
        let target = Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap();
        let binding = XlaKernelCompilerBinding::new(
            Compiler,
            target,
            Options::default(),
            KernelSchedule::default(),
            MosaicGpuEmbedding,
            1024,
        )
        .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let session = Arc::new(XlaSession::new(client));
        let domain = session.domain();
        let compiled: CompiledXlaFunction<'_, Vec<ArrayType>, Vec<ArrayType>> = compile_with_options(
            |inputs: Vec<XlaCompileTracer<'_>>| {
                let values = inputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>();
                stage_kernel(values[0].context(), &case.definition, &values)
                    .unwrap()
                    .into_iter()
                    .map(|value| {
                        let ryft_core::ArrayIrType::Array(r#type) = value.r#type().into_owned() else {
                            panic!("kernel output must be an array");
                        };
                        ProjectedValue::new(value, r#type)
                    })
                    .collect()
            },
            case.input_types.clone(),
            &domain,
            XlaOptions::new(mesh.clone()).with_kernel_compiler(binding),
        )
        .unwrap();
        let inputs = case
            .input_types
            .into_iter()
            .zip(case.inputs)
            .map(|(r#type, values)| {
                let bytes = values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
                session.array(r#type, mesh.clone(), bytes).unwrap()
            })
            .collect::<Vec<_>>();
        let outputs = domain.interpret(&compiled.executable_function(), inputs)?;
        assert_eq!(outputs.len(), 1);
        let bytes = outputs[0]
            .device_shard(device.id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)?
            .r#await()?;
        let values = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok(values)
    }

    /// Runs one independent case on every explicitly enabled CUDA plugin.
    fn execute_on_cuda(name: &str) {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let case = kernel_cases().into_iter().find(|case| case.name == name).unwrap();
                interpret_case(&case);
                let expected = case.expected.clone();
                assert_eq!(execute_case(&client, case).unwrap(), expected, "{name}");
            }
        });
    }

    #[test]
    fn test_vector_add_on_cuda() {
        execute_on_cuda("vector_add");
    }

    #[test]
    fn test_sum_on_cuda() {
        execute_on_cuda("sum");
    }

    #[test]
    fn test_matmul_on_cuda() {
        execute_on_cuda("matmul");
    }

    #[test]
    fn test_async_copy_on_cuda() {
        execute_on_cuda("async_copy");
    }

    #[test]
    #[ignore = "device assertion failure must run in a separate process after positive GPU tests"]
    fn test_assertion_failure_on_cuda() {
        use ryft_core::{
            ArrayIrOperation, ArrayIrValue, DimensionBounds, DimensionFromScalarOperation, DimensionVariable,
        };

        assert_eq!(env::var("RYFT_PJRT_RUN_MOSAIC_GPU_ASSERTION_FAILURE").as_deref(), Ok("1"));
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let r#type = ArrayType::new_static(DataType::F32, [1]);
                let operation = KernelCallOperation::new(
                    Grid::new(vec![]).unwrap(),
                    vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadWrite).unwrap()],
                )
                .unwrap();
                let definition = KernelDefinition::trace(operation, |(references, _)| {
                    let context = references[0].context();
                    let invalid = context.lift(ArrayIrValue::Array(Array::scalar(-1i64)?))?;
                    context.bind(
                        ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(
                            DimensionVariable::new("checked", DimensionBounds::new(0, Some(2))?),
                        )),
                        vec![],
                        &[invalid],
                    )?;
                    references[0].write(&references[0].read()?)
                })
                .unwrap();
                let case = KernelCase {
                    name: "assertion_failure",
                    definition,
                    input_types: vec![r#type],
                    inputs: vec![vec![1.0]],
                    expected: vec![],
                };
                let error = execute_case(&client, case).unwrap_err().to_string();
                assert!(
                    error.contains("CUDA_ERROR_LAUNCH_FAILED") || error.contains("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
                    "{error}",
                );
            }
        });
    }
}
