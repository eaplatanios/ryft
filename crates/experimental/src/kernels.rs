//! Portable and explicit GPU kernel experiments with independent numerical oracles.

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

/// Stages scaled dot-product attention with a stable row-wise softmax. Half operands request FP32 accumulation and
/// round normalized probabilities back to the operand type before the second dot, making both contractions eligible
/// for Hopper WGMMA. The FP32 variant exercises the same softmax and numerical oracle on the baseline adapter.
fn attention_definition(data_type: DataType, rows: usize, columns: usize, depth: usize) -> KernelDefinition {
    use ryft_core::{
        ArrayIrOperation, ArrayOperation, BroadcastOperation, ConstantOperation, ConvertElementTypeOperation,
        DivOperation, DotOperation, ExpOperation, MulOperation, ReduceOperation, ReductionKind, SubOperation,
    };

    let output_type = ArrayType::new_static(DataType::F32, [rows, depth]);
    let scores_type = ArrayType::new_static(DataType::F32, [rows, columns]);
    let call = KernelCallOperation::new(
        Grid::new(vec![]).unwrap(),
        vec![
            whole_array_parameter(ArrayType::new_static(data_type, [rows, depth]), KernelParameterAccess::ReadOnly)
                .unwrap(),
            whole_array_parameter(ArrayType::new_static(data_type, [depth, columns]), KernelParameterAccess::ReadOnly)
                .unwrap(),
            whole_array_parameter(ArrayType::new_static(data_type, [columns, depth]), KernelParameterAccess::ReadOnly)
                .unwrap(),
            whole_array_parameter(output_type, KernelParameterAccess::WriteOnly).unwrap(),
        ],
    )
    .unwrap();
    KernelDefinition::trace(call, |(references, _)| {
        let context = references[0].context();
        let bind = |operation, inputs: &[_]| {
            context.bind(ArrayIrOperation::Array(operation), vec![], inputs).map(|mut values| values.remove(0))
        };
        let dot = if data_type == DataType::F32 {
            DotOperation::matmul()
        } else {
            DotOperation::matmul().with_accumulation_type(DataType::F32)
        };
        let scores = bind(ArrayOperation::Dot(dot.clone()), &[references[0].read()?, references[1].read()?])?;
        let scale =
            bind(ArrayOperation::Constant(ConstantOperation::new(Array::scalar((depth as f32).sqrt().recip())?)), &[])?;
        let scores = bind(ArrayOperation::Mul(MulOperation::new()), &[scores, scale])?;
        let maximum =
            bind(ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Max)), &[scores.clone()])?;
        let maximum =
            bind(ArrayOperation::Broadcast(BroadcastOperation::new(scores_type.clone(), vec![0])), &[maximum])?;
        let centered = bind(ArrayOperation::Sub(SubOperation::new()), &[scores, maximum])?;
        let exponentials = bind(ArrayOperation::Exp(ExpOperation::new()), &[centered])?;
        let sum =
            bind(ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)), &[exponentials.clone()])?;
        let sum = bind(ArrayOperation::Broadcast(BroadcastOperation::new(scores_type, vec![0])), &[sum])?;
        let probabilities = bind(ArrayOperation::Div(DivOperation::new()), &[exponentials, sum])?;
        let probabilities = if data_type == DataType::F32 {
            probabilities
        } else {
            bind(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(data_type, false)),
                &[probabilities],
            )?
        };
        let output = bind(ArrayOperation::Dot(dot), &[probabilities, references[2].read()?])?;
        references[3].write(&output)
    })
    .unwrap()
}

/// Creates a small FP32 attention case with an independent scalar softmax and contraction oracle.
fn attention_case() -> KernelCase {
    let rows = 8;
    let columns = 8;
    let depth = 16;
    let query = (0..rows * depth).map(|index| (index % 7) as f32 * 0.125 - 0.375).collect::<Vec<_>>();
    let key = (0..depth * columns).map(|index| (index % 11) as f32 * 0.125 - 0.625).collect::<Vec<_>>();
    let value = (0..columns * depth).map(|index| (index % 13) as f32 * 0.25 - 1.5).collect::<Vec<_>>();
    let mut expected = vec![0.0; rows * depth];
    for row in 0..rows {
        let mut scores = vec![0.0f64; columns];
        for column in 0..columns {
            for contraction in 0..depth {
                scores[column] += query[row * depth + contraction] as f64 * key[contraction * columns + column] as f64;
            }
            scores[column] /= (depth as f64).sqrt();
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        for column in 0..depth {
            let mut result = 0.0;
            for contraction in 0..columns {
                result += weights[contraction] / total * value[contraction * depth + column] as f64;
            }
            expected[row * depth + column] = result as f32;
        }
    }
    KernelCase {
        name: "attention",
        definition: attention_definition(DataType::F32, rows, columns, depth),
        input_types: vec![
            ArrayType::new_static(DataType::F32, [rows, depth]),
            ArrayType::new_static(DataType::F32, [depth, columns]),
            ArrayType::new_static(DataType::F32, [columns, depth]),
        ],
        inputs: vec![query, key, value],
        expected,
    }
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
fn async_copy_case(shape: &[usize]) -> KernelCase {
    let elements = shape.iter().product();
    let input_type = ArrayType::new_static(DataType::F32, shape.to_vec());
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
    let values = (0..elements).map(|index| index as f32 - 20.0).collect::<Vec<_>>();
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
        async_copy_case(&[67]),
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

/// Checks FP32 attention against the independently accumulated FP64 oracle.
fn check_attention(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() <= 2e-6, "attention element {index}: {actual} differs from {expected}");
    }
}

#[test]
fn test_attention_interpretation() {
    let case = attention_case();
    let inputs = case
        .input_types
        .iter()
        .zip(&case.inputs)
        .map(|(r#type, values)| Array::from_elements(r#type.clone(), values).unwrap())
        .collect();
    let outputs = case.definition.interpret(inputs, 1024).unwrap();
    check_attention(&outputs[0].elements::<f32>().unwrap(), &case.expected);
}

#[cfg(feature = "mosaic-gpu")]
mod gpu {
    use std::env;
    use std::sync::Arc;
    use std::time::Instant;

    use pretty_assertions::assert_eq;

    use ryft_core::kernels::{KernelExtension, KernelOperation, KernelSchedule};
    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProgramError, ProjectedValue, Typed};
    use ryft_mosaic::kernels::gpu::{Compiler, GpuOperation, Options, Target};
    use ryft_xla::experimental::XlaDomainError;
    use ryft_xla::kernels::{MosaicGpuEmbedding, XlaKernelCompilerBinding, XlaKernelExtension, stage_kernel};
    use ryft_xla::{CompiledXlaFunction, FromPjrt, XlaCompileTracer, XlaOptions, XlaSession, compile_with_options};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// Builds an explicit target from the actual CUDA device's reported compute capability.
    fn device_target(client: &crate::Client<'_>) -> Target {
        let device = client.addressable_devices().unwrap().remove(0);
        let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
            panic!("the CUDA PJRT device must report its compute capability");
        };
        let (major, minor) = capability.split_once('.').unwrap();
        Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap()
    }

    /// Compiles and executes the canonical kernel definition through the selected Mosaic adapter and XLA GPU runtime.
    fn execute_case<'c>(
        client: &'c crate::Client<'c>,
        case: KernelCase,
        options: Options,
    ) -> Result<Vec<f32>, XlaDomainError> {
        let bytes = case
            .inputs
            .into_iter()
            .map(|values| values.iter().flat_map(|value| value.to_ne_bytes()).collect())
            .collect();
        execute_definition(client, &case.definition, case.input_types, bytes, device_target(client), options, 1)
    }

    /// Executes explicit typed source and physical bytes using the selected extension-family binding. More than one
    /// iteration measures cached invocation plus awaited host readback after one warmup; compilation and input upload
    /// are excluded. Benchmarks deliberately include dispatch and transfer costs rather than claiming GPU-only timing.
    fn execute_definition<'c, Extension>(
        client: &'c crate::Client<'c>,
        definition: &KernelDefinition<Extension>,
        input_types: Vec<ArrayType>,
        input_bytes: Vec<Vec<u8>>,
        target: Target,
        options: Options,
        iterations: usize,
    ) -> Result<Vec<f32>, XlaDomainError>
    where
        Extension: 'static
            + Send
            + Sync
            + KernelExtension
            + Into<GpuOperation>
            + Into<XlaKernelExtension>
            + TryFrom<XlaKernelExtension, Error = ProgramError>,
    {
        assert!((1..=128).contains(&iterations));
        let description = format!("{options:?}");
        let device = client.addressable_devices().unwrap().remove(0);
        assert_eq!(target.compute_capability(), device_target(client).compute_capability());
        let binding = XlaKernelCompilerBinding::new_with_extensions::<_, _, Extension>(
            Compiler,
            target,
            options,
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
                stage_kernel(values[0].context(), definition, &values)
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
            input_types.clone(),
            &domain,
            XlaOptions::new(mesh.clone()).with_kernel_compiler(binding),
        )
        .unwrap();
        let inputs = input_types
            .into_iter()
            .zip(input_bytes)
            .map(|(r#type, bytes)| session.array(r#type, mesh.clone(), bytes).unwrap())
            .collect::<Vec<_>>();
        let warmup = usize::from(iterations > 1);
        let mut start = Instant::now();
        let mut bytes = Vec::new();
        for iteration in 0..iterations + warmup {
            if iteration == warmup {
                start = Instant::now();
            }
            let outputs = domain.interpret(&compiled.executable_function(), inputs.clone())?;
            assert_eq!(outputs.len(), 1);
            bytes = outputs[0]
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)?
                .r#await()?;
        }
        if iterations > 1 {
            eprintln!(
                "cached invocation and host readback: {description}; {iterations} iterations; mean {:?}",
                start.elapsed() / iterations as u32
            );
        }
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
                assert_eq!(execute_case(&client, case, Options::default()).unwrap(), expected, "{name}");
            }
        });
    }

    /// Exercises dense and sparse packing with an independent scalar expansion oracle.
    fn run_nvfp4(sparse: bool, invalid_metadata: bool) {
        if !invalid_metadata && env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                for (rows, columns, contraction, tensor_scale, nan_scale) in [
                    (16, 8, 64, true, false),
                    (32, 16, 128, true, false),
                    (16, 8, 64, false, false),
                    (16, 8, 64, true, true),
                ] {
                    if invalid_metadata
                        && (rows, columns, contraction, tensor_scale, nan_scale) != (16, 8, 64, true, false)
                    {
                        continue;
                    }
                    let contraction = if sparse { contraction * 2 } else { contraction };
                    let scale_span = if sparse { 32 } else { 16 };
                    let mut input_types = vec![
                        ArrayType::new_static(DataType::U8, [rows, contraction / if sparse { 4 } else { 2 }]),
                        ArrayType::new_static(DataType::U8, [columns, contraction / 2]),
                        ArrayType::new_static(DataType::F8E4M3FN, [rows, contraction / scale_span]),
                        ArrayType::new_static(DataType::F8E4M3FN, [columns, contraction / scale_span]),
                        ArrayType::new_static(DataType::F32, [rows, columns]),
                    ];
                    if sparse {
                        input_types.push(ArrayType::new_static(DataType::U8, [rows, contraction / 8]));
                    }
                    if tensor_scale {
                        input_types.push(ArrayType::scalar(DataType::F32));
                    }
                    let input_count = input_types.len();
                    let mut parameters = input_types
                        .iter()
                        .cloned()
                        .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::ReadOnly).unwrap())
                        .collect::<Vec<_>>();
                    parameters.push(
                        whole_array_parameter(
                            ArrayType::new_static(DataType::F32, [rows, columns]),
                            KernelParameterAccess::WriteOnly,
                        )
                        .unwrap(),
                    );
                    let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
                    let definition: KernelDefinition<GpuOperation> =
                        KernelDefinition::trace(call, |(references, _)| {
                            let values = references[..input_count]
                                .iter()
                                .map(|reference| reference.read())
                                .collect::<Result<Vec<_>, _>>()?;
                            let result = references[0]
                                .context()
                                .bind(
                                    KernelOperation::Extension(if sparse {
                                        GpuOperation::Nvfp4Sparse { tensor_scale }
                                    } else {
                                        GpuOperation::Nvfp4 { tensor_scale }
                                    }),
                                    vec![],
                                    &values,
                                )?
                                .remove(0);
                            references[input_count].write(&result)
                        })
                        .unwrap();
                    let pack = |count: usize, row_factor: usize, contraction_factor: usize, bias: usize| {
                        (0..count)
                            .flat_map(|row| {
                                (0..contraction / 2).map(move |byte| {
                                    let nibble = |element: usize| {
                                        let row = (row * row_factor + 1) as u32;
                                        let element = (element * contraction_factor + bias + 1) as u32;
                                        ((row.wrapping_mul(0x9e37_79b1) ^ element.wrapping_mul(0x85eb_ca77)) >> 13) & 15
                                    };
                                    let low = nibble(byte * 2);
                                    let high = nibble(byte * 2 + 1);
                                    (low | (high << 4)) as u8
                                })
                            })
                            .collect::<Vec<_>>()
                    };
                    let left = pack(rows, 3, 5, 0);
                    let right = pack(columns, 7, 3, 1);
                    let left_encodings = [0x30u8, 0xb8, 0x40, 0xc4];
                    let right_encodings = [0x38u8, 0xb0, 0x44, 0xc0];
                    let mut left_scales = (0..rows)
                        .flat_map(|row| {
                            (0..contraction / scale_span)
                                .map(move |group| left_encodings[(row + row / 8 + group + group / 4) % 4])
                        })
                        .collect::<Vec<_>>();
                    let right_scales = (0..columns)
                        .flat_map(|row| {
                            (0..contraction / scale_span)
                                .map(move |group| right_encodings[(row * 3 + row / 4 + group * 3 + group / 4) % 4])
                        })
                        .collect::<Vec<_>>();
                    let accumulator = (0..rows * columns).map(|index| (index % 5) as f32 - 2.0).collect::<Vec<_>>();
                    let decode = |bytes: &[u8], row: usize, column: usize| {
                        let nibble = (bytes[row * contraction / 2 + column / 2] >> (column % 2 * 4)) & 15;
                        let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize];
                        if nibble & 8 == 0 { magnitude } else { -magnitude }
                    };
                    let mut metadata = (0..rows)
                        .flat_map(|row| {
                            (0..contraction / 8)
                                .map(move |chunk| [0x4u8, 0x8, 0x9, 0xc, 0xd, 0xe][(row * 3 + chunk) % 6])
                        })
                        .collect::<Vec<_>>();
                    let mut compressed = Vec::new();
                    if sparse {
                        for row in 0..rows {
                            for chunk in 0..contraction / 8 {
                                let code = metadata[row * contraction / 8 + chunk];
                                for pair in [code & 3, code >> 2] {
                                    compressed.push(left[row * contraction / 2 + chunk * 4 + pair as usize]);
                                }
                            }
                        }
                    }
                    let mut expected = accumulator.clone();
                    let decode_scale = |byte: u8| {
                        let exponent = (byte >> 3) & 15;
                        let mantissa = (byte & 7) as f32;
                        let magnitude = if exponent == 0 {
                            mantissa * 2.0f32.powi(-9)
                        } else {
                            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent as i32 - 7)
                        };
                        if byte & 128 == 0 { magnitude } else { -magnitude }
                    };
                    for row in 0..rows {
                        for column in 0..columns {
                            let mut product = 0.0f32;
                            for index in 0..contraction {
                                if sparse {
                                    let code = metadata[row * contraction / 8 + index / 8];
                                    let pair = (index % 8 / 2) as u8;
                                    if pair != code & 3 && pair != code >> 2 {
                                        continue;
                                    }
                                }
                                let left = decode(&left, row, index)
                                    * decode_scale(left_scales[row * contraction / scale_span + index / scale_span]);
                                let right = decode(&right, column, index)
                                    * decode_scale(
                                        right_scales[column * contraction / scale_span + index / scale_span],
                                    );
                                product += left * right;
                            }
                            expected[row * columns + column] += product * if tensor_scale { 0.5 } else { 1.0 };
                        }
                    }
                    if nan_scale {
                        left_scales[0] = 0x7f;
                    }
                    let mut input_bytes = vec![
                        if sparse { compressed } else { left },
                        right,
                        left_scales,
                        right_scales,
                        accumulator.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                    ];
                    if sparse {
                        if invalid_metadata {
                            metadata[0] = 0x5;
                        }
                        input_bytes.push(metadata);
                    }
                    if tensor_scale {
                        input_bytes.push(0.5f32.to_ne_bytes().to_vec());
                    }
                    let result = execute_definition(
                        &client,
                        &definition,
                        input_types,
                        input_bytes,
                        device_target(&client),
                        Options::default(),
                        1,
                    );
                    if invalid_metadata {
                        let error = result.unwrap_err().to_string();
                        assert!(
                            error.contains("CUDA_ERROR_ILLEGAL_INSTRUCTION")
                                || error.contains("CUDA_ERROR_LAUNCH_FAILED"),
                            "{error}"
                        );
                        continue;
                    }
                    let result = result.unwrap();
                    if nan_scale {
                        assert!(result[..columns].iter().all(|value| value.is_nan()));
                        assert_eq!(result[columns..], expected[columns..]);
                    } else {
                        assert_eq!(result, expected);
                    }
                }
            }
        });
    }

    #[test]
    fn test_tma_copy_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let case = async_copy_case(&[128]);
                interpret_case(&case);
                let expected = case.expected.clone();
                assert_eq!(execute_case(&client, case, Options::default().with_tma(true)).unwrap(), expected);
            }
        });
    }

    #[test]
    fn test_tma_benchmark_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_BENCHMARKS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                for tma in [false, true] {
                    let case = async_copy_case(&[16, 256]);
                    let bytes = case
                        .inputs
                        .iter()
                        .map(|values| values.iter().flat_map(|value| value.to_ne_bytes()).collect())
                        .collect();
                    let result = execute_definition(
                        &client,
                        &case.definition,
                        case.input_types,
                        bytes,
                        device_target(&client),
                        Options::default().with_tma(tma),
                        32,
                    )
                    .unwrap();
                    assert_eq!(result, case.expected);
                }
            }
        });
    }

    #[test]
    fn test_cluster_on_cuda() {
        use ryft_core::{AddOperation, ArrayIrOperation, ArrayIrValue, ArrayOperation};

        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let r#type = ArrayType::new_static(DataType::F32, [128]);
                let call = KernelCallOperation::new(
                    Grid::new(vec![]).unwrap(),
                    vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadWrite).unwrap()],
                )
                .unwrap();
                let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
                    let context = references[0].context();
                    let one =
                        context.lift(ArrayIrValue::Array(Array::from_elements(r#type.clone(), &[1.0f32; 128])?))?;
                    for _ in 0..2 {
                        let input = references[0].read()?;
                        let result = context
                            .bind(
                                ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                                vec![],
                                &[input, one.clone()],
                            )?
                            .remove(0);
                        references[0].write(&result)?;
                    }
                    Ok(())
                })
                .unwrap();
                let input = (0..128).map(|index| index as f32 - 64.0).collect::<Vec<_>>();
                let bytes = input.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
                let expected = input.iter().map(|value| value + 2.0).collect::<Vec<_>>();
                for blocks in [1, 2] {
                    let target = device_target(&client).with_blocks_per_cluster(blocks).unwrap();
                    let actual = execute_definition(
                        &client,
                        &definition,
                        vec![r#type.clone()],
                        vec![bytes.clone()],
                        target,
                        Options::default(),
                        1,
                    )
                    .unwrap();
                    assert_eq!(actual, expected);
                }
                let case = kernel_cases().into_iter().find(|case| case.name == "vector_add").unwrap();
                let bytes = case
                    .inputs
                    .iter()
                    .map(|values| values.iter().flat_map(|value| value.to_ne_bytes()).collect())
                    .collect();
                let actual = execute_definition(
                    &client,
                    &case.definition,
                    case.input_types,
                    bytes,
                    device_target(&client).with_blocks_per_cluster(2).unwrap(),
                    Options::default(),
                    1,
                )
                .unwrap();
                assert_eq!(actual, case.expected);
            }
        });
    }

    #[test]
    fn test_nvfp4_on_cuda() {
        run_nvfp4(false, false);
    }

    #[test]
    fn test_nvfp4_sparse_on_cuda() {
        run_nvfp4(true, false);
    }

    #[test]
    #[ignore = "device assertion failure must run in a separate process after positive GPU tests"]
    fn test_nvfp4_sparse_invalid_metadata_on_cuda() {
        assert_eq!(env::var("RYFT_PJRT_RUN_MOSAIC_GPU_ASSERTION_FAILURE").as_deref(), Ok("1"));
        run_nvfp4(true, true);
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
    fn test_attention_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(any(feature = "cuda-12", feature = "cuda-13")), "GPU execution requires a CUDA feature");
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let case = attention_case();
                let expected = case.expected.clone();
                let actual = execute_case(&client, case, Options::default()).unwrap();
                check_attention(&actual, &expected);
            }
        });
    }

    #[test]
    fn test_attention_wgmma_module() {
        use ryft_core::kernels::VerifiedKernel;
        use ryft_mlir::{Operation, WalkOrder, WalkResult};
        use ryft_mosaic::kernels::gpu::Mma;

        let definition = attention_definition(DataType::F16, 64, 64, 16);
        let kernel = VerifiedKernel::new(&definition, 1024).unwrap();
        let context = ryft_mlir::Context::new();
        let target = Target::new(9, 0)
            .unwrap()
            .with_threads_per_block(128)
            .unwrap()
            .with_maximum_shared_memory_bytes(192 * 1024)
            .unwrap();
        let module = Compiler
            .module(&context, &kernel, &target, &Options::default().with_mma(Mma::Wgmma), &KernelSchedule::default())
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        let mut matrix_instructions = 0;
        let mut exponentials = 0;
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            match operation.name().as_str().unwrap() {
                "nvvm.wgmma.mma_async" => matrix_instructions += 1,
                "math.exp" => exponentials += 1,
                _ => {}
            }
            WalkResult::Advance
        });
        assert_eq!(matrix_instructions, 5);
        assert_eq!(exponentials, 1);
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
                let error = execute_case(&client, case, Options::default()).unwrap_err().to_string();
                assert!(
                    error.contains("CUDA_ERROR_LAUNCH_FAILED") || error.contains("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
                    "{error}",
                );
            }
        });
    }
}
