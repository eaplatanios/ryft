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

/// Generates eight fixed compiler cases around warp and tile boundaries. Partial tiles exercise the macro's
/// generated valid-lane masks; this is bounded compiler qualification, not a sustained fuzz campaign. Integer-valued
/// inputs keep the independent scalar oracle exact in FP32, including cancellation, negative values, and zeros.
fn generated_vector_cases() -> Vec<KernelCase> {
    [
        (1, "generated_vector_1"),
        (31, "generated_vector_31"),
        (32, "generated_vector_32"),
        (33, "generated_vector_33"),
        (255, "generated_vector_255"),
        (256, "generated_vector_256"),
        (257, "generated_vector_257"),
        (1003, "generated_vector_1003"),
    ]
    .into_iter()
    .map(|(extent, name)| {
        let r#type = ArrayType::new_static(DataType::F32, [extent]);
        let left = (0..extent).map(|index| ((index * 17) % 37) as f32 - 18.0).collect::<Vec<_>>();
        let right = (0..extent).map(|index| ((index * 13 + 5) % 29) as f32 - 14.0).collect::<Vec<_>>();
        let expected = left.iter().zip(&right).map(|(left, right)| left + right).collect();
        KernelCase {
            name,
            definition: vector_add::definition(&r#type, &r#type).unwrap(),
            input_types: vec![r#type.clone(), r#type],
            inputs: vec![left, right],
            expected,
        }
    })
    .collect()
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

/// Uses fractional operands that IEEE FP32 retains but TF32 rounds, with a single exact oracle product.
fn precision_case() -> KernelCase {
    let left_type = ArrayType::new_static(DataType::F32, [33, 1]);
    let right_type = ArrayType::new_static(DataType::F32, [1, 34]);
    let left = 1.0f32 + 1.0 / 4096.0;
    let right = 1.0f32 + 1.0 / 8192.0;
    let expected = (f64::from(left) * f64::from(right)) as f32;
    assert_ne!(expected, 1.0);
    KernelCase {
        name: "matmul_precision",
        definition: matmul::definition(&left_type, &right_type).unwrap(),
        input_types: vec![left_type, right_type],
        inputs: vec![vec![left; 33], vec![right; 34]],
        expected: vec![expected; 33 * 34],
    }
}

/// Batches the unchanged vector macro through the canonical reference-index transform.
fn batched_case() -> KernelCase {
    let mut case = kernel_cases().into_iter().find(|case| case.name == "vector_add").unwrap();
    case.name = "batched_vector_add";
    case.definition = case.definition.batched(2, &[Some(0), Some(0), Some(0)], 1024).unwrap();
    case.input_types = vec![ArrayType::new_static(DataType::F32, [2, 1003]); 2];
    for (index, values) in case.inputs.iter_mut().enumerate() {
        let offset = if index == 0 { 3.0 } else { -1.0 };
        values.extend(values.clone().into_iter().map(|value| value + offset));
    }
    case.expected.extend(case.expected.clone().into_iter().map(|value| value + 2.0));
    case
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
fn test_generated_vector_cases() {
    let cases = generated_vector_cases();
    assert_eq!(
        cases.iter().map(|case| case.expected.len()).collect::<Vec<_>>(),
        vec![1, 31, 32, 33, 255, 256, 257, 1003],
    );
    for case in cases {
        interpret_case(&case);
    }
}

#[test]
fn test_macro_kernels_interpretation() {
    for case in kernel_cases().into_iter().chain([precision_case(), batched_case()]) {
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

/// Exports a macro-authored whole-array kernel, reloads it in a fresh session, and awaits its native output.
#[cfg(any(feature = "mosaic-gpu", feature = "cutile"))]
fn execute_aot_case<'c>(
    plugin: &crate::Plugin,
    client: &'c crate::Client<'c>,
    binding: ryft_xla::kernels::XlaKernelCompilerBinding,
    backend: &str,
) {
    use std::io::Read;
    use std::sync::Arc;

    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};
    use sha2::{Digest, Sha256};

    use ryft_xla::kernels::{KernelAotBundle, XlaKernelExecutionFacts};
    use ryft_xla::{FromPjrt, XlaOptions, XlaSession};

    /// Uses the portable source codec's whole-array subset with no hidden captures or runtime references.
    #[ryft_core::kernels::kernel]
    fn add(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
    ) {
        output.store(left.load() + right.load());
    }

    let r#type = ArrayType::new_static(DataType::F32, [64]);
    let definition = add::definition(&r#type, &r#type).unwrap();
    let device = client.addressable_devices().unwrap().remove(0);
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::from_pjrt(&device).unwrap()],
    )
    .unwrap();
    let producer = Arc::new(XlaSession::new(client));
    let bundle = KernelAotBundle::compile(
        &definition,
        &producer.domain(),
        XlaOptions::new(mesh.clone()).with_kernel_compiler(binding.clone()),
        1024,
    )
    .unwrap();
    assert!(bundle.stable_hlo().contains("ryft.kernel.semantic"));
    assert_eq!(bundle.report()["semantic_digest"].as_str().unwrap().len(), 64);
    let report = bundle.report().clone();
    let bytes = bundle.to_bytes().unwrap();
    drop(bundle);
    drop(producer);
    let restored = KernelAotBundle::from_bytes(&bytes, 1024).unwrap();
    let runtime = Arc::new(XlaSession::new(client));
    let loaded = restored.load(&runtime.domain(), &binding, &mesh).unwrap();
    let inputs: Vec<_> = [1.0f32, 2.0]
        .into_iter()
        .map(|value| runtime.array(r#type.clone(), mesh.clone(), value.to_ne_bytes().repeat(64)).unwrap())
        .collect();
    let iterations = std::env::var("RYFT_KERNEL_AOT_ITERATIONS")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(128);
    assert!((1..=8192).contains(&iterations), "AOT stress iterations must be within `1..=8192`");
    // One completed warmup drains uploads and first-invocation setup before recorded latency samples.
    loaded.call(inputs.clone()).unwrap().block_until_ready().unwrap();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let outputs = loaded.call(inputs.clone()).unwrap().block_until_ready().unwrap();
        let output = outputs[0]
            .device_shard(device.id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(output, 3.0f32.to_ne_bytes().repeat(64));
        samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap());
    }
    if let Some(directory) = std::env::var_os("RYFT_KERNEL_PROFILE_DIRECTORY") {
        let facts = XlaKernelExecutionFacts::from_client(client, &mesh).unwrap();
        let lowering: std::time::Duration = serde_json::from_value(report["lowering_duration"].clone()).unwrap();
        let compilation: std::time::Duration = serde_json::from_value(report["compilation_duration"].clone()).unwrap();
        let mut executable = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
        let executable_bytes = executable.metadata().unwrap().len();
        let mut executable_digest = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let count = executable.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            executable_digest.update(&buffer[..count]);
        }
        let executable_digest = format!("{:x}", executable_digest.finalize());
        let source = std::env::var("RYFT_KERNEL_SOURCE_SHA256").unwrap();
        let environment = std::env::var("RYFT_KERNEL_PROFILE_ENVIRONMENT").unwrap();
        assert!(!environment.trim().is_empty());
        assert!(source.len() == 64 && source.bytes().all(|byte| byte.is_ascii_hexdigit()));
        // Independent process runs contribute one robust latency summary; retain all observations separately.
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{backend}-samples.json")),
            serde_json::to_vec_pretty(&samples).unwrap(),
        )
        .unwrap();
        samples.sort_unstable();
        let profile = serde_json::json!({
            "schema": 1,
            "identity": {
                "semantic_digest": report["semantic_digest"],
                "configuration_digest": report["configuration_digest"],
                "execution_digest": format!("{:x}", Sha256::digest(facts.configuration_key().unwrap())),
                "environment": format!("{backend}; {environment}; host={}/{}; fixed-f32-add-64; ordinary-cache; iterations={iterations}",
                    std::env::consts::OS, std::env::consts::ARCH),
                "methodology": concat!(
                    "one completed warmup; per-process upper median of host completion/readback/oracle samples; ",
                    "one lowering and compile-or-cache sample per process",
                ),
            },
            "provenance": [{ "source_sha256": source, "binary_sha256": executable_digest }],
            "metrics": {
                "host_completion_readback": { "unit": "ns", "samples": [samples[samples.len() / 2]] },
                "lowering": { "unit": "ns", "samples": [u64::try_from(lowering.as_nanos()).unwrap()] },
                "compile_or_cache": { "unit": "ns", "samples": [u64::try_from(compilation.as_nanos()).unwrap()] },
                "aot_bundle": { "unit": "bytes", "samples": [bytes.len()] },
                "test_binary": { "unit": "bytes", "samples": [executable_bytes] },
            },
        });
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{backend}.json")),
            serde_json::to_vec_pretty(&profile).unwrap(),
        )
        .unwrap();
    }
    if let Ok(address) = std::env::var("RYFT_KERNEL_DISTRIBUTED_ADDRESS") {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        use ryft_xla::DistributedRuntime;

        use ryft_xla::kernels::{DistributedKernel, DistributedKernelOptions};

        use crate::KeyValueStore;

        let process = std::env::var("RYFT_KERNEL_DISTRIBUTED_PROCESS").unwrap().parse::<usize>().unwrap();
        assert!(process < 2);
        let coordination = DistributedRuntime::initialize(plugin, &address, 2, process as u32).unwrap();
        let options =
            DistributedKernelOptions::new(1024, Duration::from_secs(30)).unwrap().with_chunk_bytes(63).unwrap();
        let mut distributed = DistributedKernel::new(&coordination, &loaded, options).unwrap();
        let inputs = [10.0 + process as f32, 20.0 + process as f32]
            .into_iter()
            .map(|value| runtime.array(r#type.clone(), mesh.clone(), value.to_ne_bytes().repeat(64)).unwrap())
            .collect::<Vec<_>>();
        let cancelled = Arc::new(AtomicBool::new(false));
        // Cross-process and local routes must differ numerically; partial chunks exercise exact reassembly.
        for (sources, expected) in
            [([1 - process, process], 31.0f32), ([process, process], 30.0 + 2.0 * process as f32)]
        {
            let pending = distributed.call_async(inputs.clone(), &sources, Arc::clone(&cancelled)).unwrap();
            let outputs = pending.r#await().unwrap();
            let actual = outputs[0]
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(actual, expected.to_ne_bytes().repeat(64));
        }
        for (input, expected) in inputs.iter().zip([10.0 + process as f32, 20.0 + process as f32]) {
            let actual = input
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(actual, expected.to_ne_bytes().repeat(64));
        }
        coordination.key_value_store().put(format!("gpu-finished-{process}").as_bytes(), b"done").unwrap();
        assert_eq!(
            coordination
                .key_value_store()
                .get(format!("gpu-finished-{}", 1 - process).as_bytes(), Duration::from_secs(30))
                .unwrap(),
            b"done"
        );
        eprintln!("distributed GPU passed: backend={backend} process={process} rounds=2");
    }
    eprintln!("AOT: {} bytes; fresh-session reload and {iterations} awaited native outputs verified", bytes.len());
}

#[cfg(feature = "mosaic-gpu")]
mod gpu {
    use std::env;
    use std::sync::Arc;
    use std::time::Instant;

    use pretty_assertions::assert_eq;
    use sha2::{Digest, Sha256};

    use ryft_core::kernels::{KernelExtension, KernelOperation, KernelSchedule};
    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProgramError, ProjectedValue, Typed};
    use ryft_mosaic::kernels::gpu::{Compiler, GpuOperation, Options, Target};
    use ryft_xla::experimental::XlaDomainError;
    use ryft_xla::kernels::{MosaicGpuEmbedding, XlaKernelCompilerBinding, XlaKernelExtension, stage_kernel};
    use ryft_xla::{CompiledXlaFunction, FromPjrt, XlaCompileTracer, XlaOptions, XlaSession, compile_with_options};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    #[test]
    fn test_aot_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let binding = XlaKernelCompilerBinding::new(
                    Compiler,
                    device_target(&client),
                    Options::default(),
                    KernelSchedule::default(),
                    MosaicGpuEmbedding,
                    1024,
                )
                .unwrap();
                execute_aot_case(&_plugin, &client, binding, "mosaic");
                executed = true;
            }
        });
        assert!(executed, "enabled Mosaic AOT qualification did not execute a CUDA platform");
    }

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
    pub(super) fn execute_case<'c>(
        client: &'c crate::Client<'c>,
        case: KernelCase,
        options: Options,
    ) -> Result<Vec<f32>, XlaDomainError> {
        eprintln!("Mosaic {}: semantic={:x}", case.name, Sha256::digest(case.definition.semantic_key().unwrap()));
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
                let case = match name {
                    "matmul_precision" => precision_case(),
                    "batched_vector_add" => batched_case(),
                    _ => kernel_cases().into_iter().find(|case| case.name == name).unwrap(),
                };
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

    /// Executes the same eight generated definitions used by cuTile, checking every valid output lane.
    #[test]
    fn test_generated_vectors_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                for case in generated_vector_cases() {
                    let expected = case.expected.clone();
                    let name = case.name;
                    let actual = execute_case(&client, case, Options::default()).unwrap();
                    assert_eq!(actual, expected, "{name}");
                }
                executed = true;
            }
        });
        assert!(executed, "enabled generated Mosaic qualification did not execute CUDA");
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
    fn test_matmul_precision_on_cuda() {
        execute_on_cuda("matmul_precision");
    }

    #[test]
    fn test_batched_vector_add_on_cuda() {
        execute_on_cuda("batched_vector_add");
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

#[cfg(feature = "cutile")]
mod cutile {
    use std::env;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use pretty_assertions::assert_eq;
    use sha2::{Digest, Sha256};

    use ryft_core::kernels::{KernelSchedule, VerifiedKernel};
    use ryft_core::{
        AddOperation, ArrayIrType, ArrayIrValue, ArrayOperation, CompilationCacheDomain, CompilationDomain,
        CompilationStagingRequest, ConstantOperation, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType,
        call_function,
    };
    use ryft_cuda::kernels::cutile::{CompiledKernel, Compiler, Options, Target};
    use ryft_xla::experimental::XlaDomainError;
    use ryft_xla::kernels::{CuTileEmbedding, XlaKernelCompilerBinding, stage_kernel};
    use ryft_xla::{FromPjrt, XlaDomain, XlaSession};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    #[test]
    fn test_aot_on_cuda() {
        if env::var("RYFT_PJRT_RUN_CUTILE_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let device = client.addressable_devices().unwrap().remove(0);
                let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
                    panic!("missing CUDA compute capability")
                };
                let (major, minor) = capability.split_once('.').unwrap();
                let python = PathBuf::from(env::var_os("RYFT_CUTILE_PYTHON").unwrap());
                let binding = XlaKernelCompilerBinding::new(
                    Compiler::new(python),
                    Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap(),
                    Options::default(),
                    KernelSchedule::default(),
                    CuTileEmbedding,
                    1024,
                )
                .unwrap();
                execute_aot_case(&_plugin, &client, binding, "cutile");
                executed = true;
            }
        });
        assert!(executed, "enabled cuTile AOT qualification did not execute a CUDA platform");
    }

    /// Constructs two ordered updates to one read-write root, retaining the declared input/output alias.
    fn alias_case() -> KernelCase {
        let r#type = ArrayType::new_static(DataType::F32, [67]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let ones = Array::from_elements(r#type.clone(), &[1.0f32; 67]).unwrap();
        let definition = KernelDefinition::trace(call, |(references, _)| {
            let context = references[0].context();
            let one = context.bind(ArrayOperation::Constant(ConstantOperation::new(ones)), vec![], &[])?.remove(0);
            let first = context
                .bind(ArrayOperation::Add(AddOperation::new()), vec![], &[references[0].read()?, one.clone()])?
                .remove(0);
            references[0].write(&first)?;
            let second = context
                .bind(ArrayOperation::Add(AddOperation::new()), vec![], &[references[0].read()?, one])?
                .remove(0);
            references[0].write(&second)
        })
        .unwrap();
        let values = (0..67).map(|index| index as f32 - 20.0).collect::<Vec<_>>();
        KernelCase {
            name: "alias",
            definition,
            input_types: vec![r#type],
            inputs: vec![values.clone()],
            expected: values.into_iter().map(|value| value + 2.0).collect(),
        }
    }

    /// Uses the selected adapter, then disables compilation before manifest and executable restoration.
    pub(super) fn execute_case<'c>(
        client: &'c crate::Client<'c>,
        case: KernelCase,
        repeated_input: bool,
    ) -> Result<(), XlaDomainError> {
        if case.name != "assertion_failure" {
            interpret_case(&case);
        }
        let device = client.addressable_devices().unwrap().remove(0);
        let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
            panic!("missing CUDA compute capability")
        };
        let (major, minor) = capability.split_once('.').unwrap();
        let target = Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap();
        let cancellation = Arc::new(AtomicBool::new(false));
        let python = PathBuf::from(
            env::var_os("RYFT_CUTILE_PYTHON").expect("`RYFT_CUTILE_PYTHON` must name the pinned compiler Python"),
        );
        let compiler = Compiler::new(python).with_cancellation(Arc::clone(&cancellation));
        let schedule = KernelSchedule::default();
        let options = Options::default();
        eprintln!("cuTile {}: semantic={:x}", case.name, Sha256::digest(case.definition.semantic_key().unwrap()));
        let verified = VerifiedKernel::new(&case.definition, 1024).unwrap();
        let output = verified.compile(&compiler, &target, &options, &schedule).unwrap();
        let binding =
            XlaKernelCompilerBinding::new(compiler, target, options, schedule, CuTileEmbedding, 1024).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let producer = Arc::new(XlaSession::new(client));
        let domain = producer.domain();
        let staged = domain
            .stage(CompilationStagingRequest::<XlaDomain<'c>, _, Vec<ArrayIrType>, Vec<ArrayIrType>>::new(
                |_, _, inputs: Vec<ryft_core::CompilationTracer<XlaDomain<'c>>>| {
                    Ok(stage_kernel(inputs[0].context(), &case.definition, &inputs)?)
                },
                vec![],
                case.input_types.iter().cloned().map(ArrayIrType::Array).collect(),
                ryft_xla::XlaOptions::new(mesh.clone()).with_kernel_compiler(binding),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let executable_bytes = domain.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        cancellation.store(true, Ordering::Release);
        let restored_output =
            CompiledKernel::from_manifest(&verified, output.manifest(), output.artifact().bytes().to_vec()).unwrap();
        assert_eq!(restored_output.artifact().bytes(), output.artifact().bytes());
        assert_eq!(restored_output.arguments(), output.arguments());
        let runtime_session = Arc::new(XlaSession::new(client));
        let runtime = runtime_session.domain();
        let restored = runtime.deserialize_program(&executable_bytes).unwrap().unwrap();
        let executable = compiled
            .executable_function()
            .with_compiled_program(Arc::new(restored), compiled.executable_function().output_types().to_vec());
        let mut arrays = case
            .input_types
            .iter()
            .zip(&case.inputs)
            .map(|(r#type, values)| {
                runtime_session
                    .array(
                        r#type.clone(),
                        mesh.clone(),
                        values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        if repeated_input {
            arrays[1] = arrays[0].clone();
        }
        let original = arrays[0].clone();
        let results = call_function(&runtime, &executable, arrays.into_iter().map(ArrayIrValue::Array).collect())?;
        let ArrayIrValue::Array(result) = &results[0] else { panic!("kernel output must be an array") };
        let bytes =
            result.device_shard(device.id().unwrap()).unwrap().buffer().unwrap().copy_to_host(None)?.r#await()?;
        let values = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, case.expected, "{}", case.name);
        let original_bytes = original
            .device_shard(device.id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let original_values = original_bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(original_values, case.inputs[0], "live input for {}", case.name);
        eprintln!(
            "cuTile {}: cubin={:x}; manifest={:x}; executable={:x}; Python disabled before reload",
            case.name,
            Sha256::digest(output.artifact().bytes()),
            Sha256::digest(output.manifest()),
            Sha256::digest(&executable_bytes)
        );
        if let Some(directory) = env::var_os("RYFT_CUTILE_ARTIFACT_DIRECTORY") {
            let directory = PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            let name = if repeated_input { "repeated_input" } else { case.name };
            std::fs::write(directory.join(format!("{name}.cubin")), output.artifact().bytes()).unwrap();
            std::fs::write(directory.join(format!("{name}.json")), output.manifest()).unwrap();
            std::fs::write(directory.join(format!("{name}.executable")), executable_bytes).unwrap();
        }
        Ok(())
    }

    /// Selects the pinned CUDA platform explicitly; an enabled GPU request cannot pass without executing it.
    fn run(name: &str, repeated_input: bool) {
        if env::var("RYFT_PJRT_RUN_CUTILE_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(feature = "cuda-13"), "cuTile qualification requires the `cuda-13` feature");
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let mut case = if name == "alias" {
                    alias_case()
                } else if name == "batched_vector_add" {
                    batched_case()
                } else if name == "matmul_precision" {
                    precision_case()
                } else {
                    kernel_cases().into_iter().find(|case| case.name == name).unwrap()
                };
                if repeated_input {
                    case.inputs[1] = case.inputs[0].clone();
                    case.expected = case.inputs[0].iter().map(|value| value * 2.0).collect();
                }
                execute_case(&client, case, repeated_input).unwrap();
                executed = true;
            }
        });
        assert!(executed, "enabled cuTile qualification did not execute a CUDA platform");
    }

    /// Reuses the generated Mosaic definitions and the existing compile, reload, completion, and oracle checks.
    #[test]
    fn test_generated_vectors_on_cuda() {
        if env::var("RYFT_PJRT_RUN_CUTILE_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                for case in generated_vector_cases() {
                    execute_case(&client, case, false).unwrap();
                }
                executed = true;
            }
        });
        assert!(executed, "enabled generated cuTile qualification did not execute CUDA");
    }

    #[test]
    fn test_vector_add_on_cuda() {
        run("vector_add", false);
    }

    #[test]
    fn test_sum_on_cuda() {
        run("sum", false);
    }

    #[test]
    fn test_matmul_on_cuda() {
        run("matmul", false);
    }

    #[test]
    fn test_matmul_precision_on_cuda() {
        run("matmul_precision", false);
    }

    #[test]
    fn test_batched_vector_add_on_cuda() {
        run("batched_vector_add", false);
    }

    #[test]
    fn test_alias_on_cuda() {
        run("alias", false);
    }

    #[test]
    fn test_repeated_input_on_cuda() {
        run("vector_add", true);
    }

    #[test]
    fn test_shard_map_on_cuda() {
        use ryft_core::{ProjectedValue, Sharding, ShardingDimension, Typed};
        use ryft_xla::experimental::{ShardMapTracer, shard_map};
        use ryft_xla::{CompiledXlaFunction, XlaCompileTracer, XlaOptions, compile_with_options};

        if env::var("RYFT_PJRT_RUN_CUTILE_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        assert!(cfg!(feature = "cuda-13"), "cuTile qualification requires the `cuda-13` feature");
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let device = client.addressable_devices().unwrap().remove(0);
                let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
                    panic!("missing CUDA compute capability")
                };
                let (major, minor) = capability.split_once('.').unwrap();
                let target = Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap();
                let compiler = Compiler::new(PathBuf::from(env::var_os("RYFT_CUTILE_PYTHON").unwrap()));
                let binding = XlaKernelCompilerBinding::new(
                    compiler,
                    target,
                    Options::default(),
                    KernelSchedule::default(),
                    CuTileEmbedding,
                    1024,
                )
                .unwrap();
                let logical_mesh =
                    LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Manual).unwrap()]).unwrap();
                let sharding =
                    Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["device"])]).unwrap();
                let r#type = ArrayType::new_static(DataType::F32, [17]).with_sharding(sharding.clone()).unwrap();
                let mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::from_pjrt(&device).unwrap()]).unwrap();
                let session = Arc::new(XlaSession::new(&client));
                let domain = session.domain();
                let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_options(
                    |input: XlaCompileTracer<'_>| {
                        shard_map::<_, _, ArrayType, _>(
                            |local: ShardMapTracer| {
                                let local_type = local.r#type().into_owned();
                                let definition: KernelDefinition = KernelDefinition::trace(
                                    KernelCallOperation::new(
                                        Grid::new(vec![]).unwrap(),
                                        vec![
                                            whole_array_parameter(local_type.clone(), KernelParameterAccess::ReadWrite)
                                                .unwrap(),
                                        ],
                                    )
                                    .unwrap(),
                                    |(references, _)| references[0].write(&references[0].read()?),
                                )
                                .unwrap();
                                let value = local.into_value();
                                let output =
                                    stage_kernel(value.context(), &definition, &[value.clone()]).unwrap().remove(0);
                                ProjectedValue::new(output, local_type)
                            },
                            input,
                            logical_mesh.clone(),
                            sharding.clone(),
                            sharding.clone(),
                        )
                        .unwrap()
                    },
                    r#type.clone(),
                    &domain,
                    XlaOptions::new(mesh.clone()).with_kernel_compiler(binding),
                )
                .unwrap();
                let expected = (0..17).map(|value| value as f32 - 8.0).collect::<Vec<_>>();
                let input = session
                    .array(r#type, mesh, expected.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let output = domain.interpret(&compiled.executable_function(), input).unwrap();
                let bytes = output
                    .device_shard(device.id().unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                let actual = bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
                executed = true;
            }
        });
        assert!(executed, "enabled cuTile shard-map qualification did not execute CUDA");
    }

    #[test]
    #[ignore = "device assertion failure must run in a separate process after positive GPU tests"]
    fn test_assertion_failure_on_cuda() {
        use ryft_core::{
            ArrayIrOperation, ConvertElementTypeOperation, DimensionBounds, DimensionFromScalarOperation,
            DimensionVariable,
        };

        assert_eq!(env::var("RYFT_PJRT_RUN_CUTILE_ASSERTION_FAILURE").as_deref(), Ok("1"));
        assert!(cfg!(feature = "cuda-13"), "cuTile qualification requires the `cuda-13` feature");
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let r#type = ArrayType::scalar(DataType::F32);
                let call = KernelCallOperation::new(
                    Grid::new(vec![]).unwrap(),
                    vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadWrite).unwrap()],
                )
                .unwrap();
                let definition = KernelDefinition::trace(call, |(references, _)| {
                    let context = references[0].context();
                    let value = references[0].read()?;
                    let integer = context
                        .bind(
                            ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false)),
                            vec![],
                            &[value.clone()],
                        )?
                        .remove(0);
                    context.bind(
                        ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(
                            DimensionVariable::new("checked", DimensionBounds::new(0, Some(2))?),
                        )),
                        vec![],
                        &[integer],
                    )?;
                    references[0].write(&value)
                })
                .unwrap();
                let error = execute_case(
                    &client,
                    KernelCase {
                        name: "assertion_failure",
                        definition,
                        input_types: vec![r#type],
                        inputs: vec![vec![-1.0]],
                        expected: vec![],
                    },
                    false,
                )
                .unwrap_err()
                .to_string();
                assert!(
                    error.contains("CUDA_ERROR_ASSERT")
                        || error.contains("CUDA_ERROR_LAUNCH_FAILED")
                        || error.contains("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
                    "{error}",
                );
                executed = true;
            }
        });
        assert!(executed, "enabled cuTile assertion qualification did not execute CUDA");
    }
}

/// Bounded real-device schedule measurements; host readback and numerical validation are part of each sample.
#[cfg(any(feature = "mosaic-gpu", feature = "cutile"))]
mod tuning {
    use std::env;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{KernelCompiler, KernelSchedule, VerifiedKernel};
    use ryft_core::{
        ArrayIrType, ArrayIrValue, CompilationCall, CompilationDomain, CompilationStagingRequest, CompilationTracer,
        Device, DeviceMesh, ExecutableFunction, LogicalMesh, MeshAxis, MeshAxisType, ReferenceExecution,
        StatefulCompilationDomain,
    };
    use ryft_xla::kernels::{
        KernelOutputEmbedding, KernelTuner, KernelTuningBudget, KernelTuningError, KernelTuningRequest,
        KernelTuningRunner, XlaKernelCompilerBinding, XlaKernelExecutionFacts, XlaKernelTarget, stage_kernel,
    };
    use ryft_xla::{Array as XlaArray, FromPjrt, XlaDomain, XlaOptions, XlaSession};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// One immutable workload with ordinary compiled-function ownership; no alternate execution engine is introduced.
    struct Runner<'c> {
        /// Existing XLA execution domain and session owner.
        domain: XlaDomain<'c>,

        /// Fixed independent oracle and canonical input signature.
        case: KernelCase,

        /// Exact physical device placement used by compilation and invocation.
        mesh: DeviceMesh,

        /// Ordinary typed bindings for the finite requested schedule list.
        bindings: Vec<(KernelSchedule, XlaKernelCompilerBinding)>,

        /// Prepared ordinary executable, replaced only between candidates.
        executable: Option<ExecutableFunction<XlaDomain<'c>, Vec<ArrayIrType>, Vec<ArrayIrType>>>,

        /// Fixed functional inputs reused across all samples.
        inputs: Vec<ArrayIrValue<XlaArray<'c>>>,
    }

    impl<'c> KernelTuningRunner for Runner<'c> {
        fn prepare(&mut self, schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
            for input in &self.inputs {
                let ArrayIrValue::Array(input) = input else {
                    unreachable!();
                };
                input.block_until_ready().map_err(error)?;
            }
            let binding = self.bindings.iter().find(|(candidate, _)| candidate == schedule).unwrap().1.clone();
            let staged = self
                .domain
                .stage(CompilationStagingRequest::<_, _, Vec<ArrayIrType>, Vec<ArrayIrType>>::new(
                    |_, _, inputs: Vec<CompilationTracer<XlaDomain<'c>>>| {
                        Ok(stage_kernel(inputs[0].context(), &self.case.definition, &inputs)?)
                    },
                    vec![],
                    self.case.input_types.iter().cloned().map(ArrayIrType::Array).collect(),
                    XlaOptions::new(self.mesh.clone()).with_kernel_compiler(binding),
                ))
                .map_err(error)?;
            // The existing lower/compile path performs real adapter admission before artifact lookup.
            let compiled = self.domain.compile(self.domain.lower(staged).map_err(error)?).map_err(error)?;
            self.executable = Some(compiled.executable_function().clone());
            Ok(())
        }

        fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
            let result = (|| {
                let outputs = self
                    .domain
                    .call_statefully_async(CompilationCall::new(self.executable.as_ref().unwrap(), self.inputs.clone()))
                    .r#await()
                    .map_err(error)?;
                let ArrayIrValue::Array(output) = &outputs[0] else {
                    return Err(KernelTuningError::Invalid { message: "expected an ordinary array output".into() });
                };
                let bytes = output
                    .device_shard(self.mesh.devices()[0].id())
                    .ok_or_else(|| error("missing output device shard"))?
                    .buffer()
                    .ok_or_else(|| error("missing output device buffer"))?
                    .copy_to_host(None)
                    .map_err(error)?
                    .r#await()
                    .map_err(error)?;
                let actual = bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                if actual != self.case.expected {
                    return Err(KernelTuningError::Invalid {
                        message: "native tuning output differs from the independent oracle".into(),
                    });
                }
                Ok(())
            })();
            // Every native invocation and readback has really completed before this ready wrapper is returned.
            ReferenceExecution::ready(result)
        }
    }

    /// Preserves concrete integration diagnostics in the tuner-owned error family.
    fn error(error: impl std::fmt::Display) -> KernelTuningError {
        KernelTuningError::Compiler { message: error.to_string() }
    }

    /// Measures two explicit schedule choices through the same real runtime and functional workload.
    fn measure<'c, Compiler, Embedding>(
        client: &'c crate::Client<'c>,
        compiler: Compiler,
        target: Compiler::Target,
        options: Compiler::Options,
        embedding: Embedding,
        cancellation: &AtomicBool,
    ) where
        Compiler: 'static + Clone + Send + Sync + KernelCompiler<Error: 'static + Send + Sync>,
        Compiler::Target: 'static + Clone + Send + Sync + XlaKernelTarget,
        Compiler::Options: 'static + Clone + Send + Sync,
        Embedding: 'static + Clone + Send + Sync + KernelOutputEmbedding<Compiler::Output>,
    {
        let case = kernel_cases().into_iter().find(|case| case.name == "vector_add").unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let facts = XlaKernelExecutionFacts::from_client(client, &mesh).unwrap();
        let candidates = vec![
            KernelSchedule::default(),
            KernelSchedule::default().with_pipeline_stages(NonZeroUsize::new(2).unwrap()),
        ];
        let environment = format!(
            concat!(
                "vector-add-1003-integer-f32-v1; fixed-inputs; exclusive-test-process; host={}/{}; ",
                "includes-host-readback-and-oracle",
            ),
            env::consts::OS,
            env::consts::ARCH,
        );
        let request = KernelTuningRequest::new(
            &VerifiedKernel::new(&case.definition, 1024).unwrap(),
            &compiler,
            &target,
            &options,
            &embedding,
            &facts,
            environment.as_bytes(),
            candidates.clone(),
            KernelTuningBudget::new(2, 1, 2, Duration::from_secs(120)).unwrap(),
        )
        .unwrap();
        let bindings = candidates
            .into_iter()
            .map(|schedule| {
                let binding = XlaKernelCompilerBinding::new(
                    compiler.clone(),
                    target.clone(),
                    options.clone(),
                    schedule.clone(),
                    embedding.clone(),
                    1024,
                )
                .unwrap();
                (schedule, binding)
            })
            .collect();
        let session = Arc::new(XlaSession::new(client));
        let inputs = case
            .input_types
            .iter()
            .zip(&case.inputs)
            .map(|(r#type, values)| {
                ArrayIrValue::Array(
                    session
                        .array(
                            r#type.clone(),
                            mesh.clone(),
                            values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>(),
                        )
                        .unwrap(),
                )
            })
            .collect();
        let mut runner = Runner { domain: session.domain(), case, mesh, bindings, executable: None, inputs };
        let result = KernelTuner::new(None).run(&request, &mut runner, cancellation).unwrap();
        assert_eq!(result.samples().iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
        assert!(result.best_candidate() < request.candidates().len());
        eprintln!(
            "tuning: candidate={} host-completion-readback-samples={:?}",
            result.best_candidate(),
            result.samples()
        );
    }

    #[cfg(feature = "mosaic-gpu")]
    #[test]
    fn test_mosaic_on_cuda() {
        if env::var("RYFT_PJRT_RUN_MOSAIC_GPU_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda12 | TestPlatform::Cuda13) {
                let device = client.addressable_devices().unwrap().remove(0);
                let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
                    panic!("missing CUDA compute capability");
                };
                let (major, minor) = capability.split_once('.').unwrap();
                measure(
                    &client,
                    ryft_mosaic::kernels::gpu::Compiler,
                    ryft_mosaic::kernels::gpu::Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap(),
                    ryft_mosaic::kernels::gpu::Options::default(),
                    ryft_xla::kernels::MosaicGpuEmbedding,
                    &AtomicBool::new(false),
                );
                executed = true;
            }
        });
        assert!(executed, "enabled Mosaic tuning did not execute CUDA");
    }

    #[cfg(feature = "cutile")]
    #[test]
    fn test_cutile_on_cuda() {
        if env::var("RYFT_PJRT_RUN_CUTILE_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let device = client.addressable_devices().unwrap().remove(0);
                let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
                    panic!("missing CUDA compute capability");
                };
                let (major, minor) = capability.split_once('.').unwrap();
                let cancellation = Arc::new(AtomicBool::new(false));
                let compiler = ryft_cuda::kernels::cutile::Compiler::new(std::path::PathBuf::from(
                    env::var_os("RYFT_CUTILE_PYTHON").unwrap(),
                ))
                .with_cancellation(cancellation.clone());
                measure(
                    &client,
                    compiler,
                    ryft_cuda::kernels::cutile::Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap(),
                    ryft_cuda::kernels::cutile::Options::default(),
                    ryft_xla::kernels::CuTileEmbedding,
                    &cancellation,
                );
                executed = true;
            }
        });
        assert!(executed, "enabled cuTile tuning did not execute CUDA");
    }
}

/// Seeded compiler stress driven in isolated processes by `tools/kernel_stress.py`.
mod stress {
    #[cfg(any(feature = "mosaic-gpu", feature = "cutile"))]
    use std::env;
    use std::path::Path;

    use pretty_assertions::assert_eq;
    use serde::Serialize;

    use super::*;

    // Defines another tile width using the existing portable macro and its canonical valid-lane masking.
    macro_rules! tiled_add {
        ($name:ident, $width:literal) => {
            #[ryft_core::kernels::kernel]
            fn $name(
                #[input(data_type = F32, rank = 1)] left: &Array,
                #[input(data_type = F32, rank = 1)] right: &Array,
                #[output(data_type = F32, shape = [left.shape()[0]], tile = [$width], boundary = masked)]
                output: &mut Array,
            ) {
                let [block] = output.tile_index();
                let left_tiles = left.tiles([$width]).pad(0.0);
                let right_tiles = right.tiles([$width]).pad(0.0);
                output.store(left_tiles.load([block]) + right_tiles.load([block]));
            }
        };
    }

    tiled_add!(add32, 32);
    tiled_add!(add128, 128);

    /// A codec-supported mutation source, independent of tile-load codec eligibility.
    #[ryft_core::kernels::kernel]
    fn codec_add(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
    ) {
        output.store(left.load() + right.load());
    }

    /// Reconstructible generator state; no compiler-generated expected values enter the oracle.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct StressCase {
        /// User-selected deterministic campaign seed.
        seed: u64,

        /// Independently replayable mutation index.
        iteration: u64,

        /// Vector length or matrix row count.
        rows: usize,

        /// Matrix output width; zero selects vector addition.
        columns: usize,

        /// Contraction depth for matrix multiplication.
        depth: usize,

        /// Vector tile width, selected from the three compiled macro forms.
        tile: usize,
    }

    impl StressCase {
        /// Produces bounded edge-biased shapes and reproducible pseudo-random interior dimensions.
        fn new(seed: u64, iteration: u64) -> Self {
            let mut state = seed.wrapping_add(iteration.wrapping_mul(0x9e3779b97f4a7c15));
            let edges = [1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1003, 2049];
            let rows = if iteration % 2 == 0 {
                edges[(next(&mut state) % edges.len() as u64) as usize]
            } else {
                (next(&mut state) % 2049) as usize + 1
            };
            if iteration % 4 == 3 {
                Self {
                    seed,
                    iteration,
                    rows: rows.min(35),
                    columns: (next(&mut state) % 35) as usize + 1,
                    depth: (next(&mut state) % 65) as usize + 1,
                    tile: 32,
                }
            } else {
                Self {
                    seed,
                    iteration,
                    rows,
                    columns: 0,
                    depth: 0,
                    tile: [32, 128, 256][(next(&mut state) % 3) as usize],
                }
            }
        }

        /// Builds existing macro definitions and an independent exact-integer scalar oracle.
        fn kernel(&self) -> KernelCase {
            let mut state = self.seed ^ self.iteration.rotate_left(17);
            if self.columns == 0 {
                let r#type = ArrayType::new_static(DataType::F32, [self.rows]);
                let left = (0..self.rows).map(|_| (next(&mut state) % 17) as f32 - 8.0).collect::<Vec<_>>();
                let right = (0..self.rows).map(|_| (next(&mut state) % 17) as f32 - 8.0).collect::<Vec<_>>();
                let expected = left.iter().zip(&right).map(|(left, right)| left + right).collect();
                let definition = match self.tile {
                    32 => add32::definition(&r#type, &r#type),
                    128 => add128::definition(&r#type, &r#type),
                    256 => vector_add::definition(&r#type, &r#type),
                    _ => unreachable!(),
                }
                .unwrap();
                KernelCase {
                    name: "stress_vector",
                    definition,
                    input_types: vec![r#type.clone(), r#type],
                    inputs: vec![left, right],
                    expected,
                }
            } else {
                let left_type = ArrayType::new_static(DataType::F32, [self.rows, self.depth]);
                let right_type = ArrayType::new_static(DataType::F32, [self.depth, self.columns]);
                let left = (0..self.rows * self.depth).map(|_| (next(&mut state) % 5) as f32 - 2.0).collect::<Vec<_>>();
                let right =
                    (0..self.depth * self.columns).map(|_| (next(&mut state) % 5) as f32 - 2.0).collect::<Vec<_>>();
                let mut expected = vec![0.0; self.rows * self.columns];
                for row in 0..self.rows {
                    for column in 0..self.columns {
                        for depth in 0..self.depth {
                            expected[row * self.columns + column] +=
                                left[row * self.depth + depth] * right[depth * self.columns + column];
                        }
                    }
                }
                KernelCase {
                    name: "stress_matmul",
                    definition: matmul::definition(&left_type, &right_type).unwrap(),
                    input_types: vec![left_type, right_type],
                    inputs: vec![left, right],
                    expected,
                }
            }
        }

        /// Checks canonical roundtrip plus four rejected mutations, saving every reproducer before decode.
        fn check_codec(&self, directory: Option<&Path>) {
            let r#type = ArrayType::new_static(DataType::F32, [self.rows]);
            let definition = codec_add::definition(&r#type, &r#type).unwrap();
            let bytes = serde_json::to_vec(&definition).unwrap();
            assert!(bytes.len() <= 1024 * 1024);
            let restored: KernelDefinition = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(restored.semantic_key().unwrap(), definition.semantic_key().unwrap());
            for mutation in 0..4 {
                let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                match mutation {
                    0 => wire["version"] = serde_json::json!(999),
                    1 => wire["body"]["entry"] = serde_json::json!(u64::MAX),
                    2 => wire["unexpected"] = serde_json::json!(self.seed),
                    _ => {}
                }
                let mut malformed = serde_json::to_vec(&wire).unwrap();
                if mutation == 3 {
                    malformed.truncate((self.iteration % (malformed.len() - 1) as u64) as usize + 1);
                }
                if let Some(directory) = directory {
                    std::fs::write(directory.join(format!("malformed-{mutation}.json")), &malformed).unwrap();
                }
                let error = serde_json::from_slice::<KernelDefinition>(&malformed).unwrap_err();
                match mutation {
                    0 => assert_eq!(error.to_string(), "unsupported kernel source schema version 999"),
                    1 => {
                        assert_eq!(error.to_string(), "invalid serialized kernel source: entry region is out of bounds")
                    }
                    2 => assert_eq!(error.classify(), serde_json::error::Category::Data),
                    _ => assert!(matches!(
                        error.classify(),
                        serde_json::error::Category::Eof | serde_json::error::Category::Syntax
                    )),
                }
            }
        }
    }

    /// Advances a deterministic wrapping generator; no statistical or cryptographic randomness claim is made.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    #[test]
    fn test_stress_case_new() {
        assert_eq!(StressCase::new(7, 13), StressCase::new(7, 13));
        assert_ne!(StressCase::new(7, 13), StressCase::new(8, 13));
        for iteration in 0..32 {
            let case = StressCase::new(7, iteration);
            assert!((1..=2049).contains(&case.rows));
            assert!([32, 128, 256].contains(&case.tile));
            assert!(case.columns <= 35 && case.depth <= 65);
        }
    }

    #[test]
    fn test_stress_case_kernel() {
        for iteration in 0..8 {
            interpret_case(&StressCase::new(7, iteration).kernel());
        }
    }

    #[test]
    fn test_stress_case_check_codec() {
        for iteration in 0..8 {
            StressCase::new(7, iteration).check_codec(None);
        }
    }

    #[cfg(any(feature = "mosaic-gpu", feature = "cutile"))]
    #[test]
    #[ignore = "seeded native compiler stress is launched by tools/kernel_stress.py"]
    fn test_compiler_case_on_cuda() {
        let seed = env::var("RYFT_KERNEL_STRESS_SEED").unwrap().parse().unwrap();
        let iteration = env::var("RYFT_KERNEL_STRESS_ITERATION").unwrap().parse().unwrap();
        let backend = env::var("RYFT_KERNEL_STRESS_BACKEND").unwrap();
        assert!(["mosaic", "cutile", "both"].contains(&backend.as_str()));
        let directory = std::path::PathBuf::from(env::var_os("RYFT_KERNEL_STRESS_CASE_DIRECTORY").unwrap());
        let descriptor = StressCase::new(seed, iteration);
        std::fs::write(directory.join("case.json"), serde_json::to_vec_pretty(&descriptor).unwrap()).unwrap();
        let case = descriptor.kernel();
        std::fs::write(directory.join("source.txt"), case.definition.body().to_string()).unwrap();
        std::fs::write(directory.join("semantic.txt"), case.definition.semantic_key().unwrap()).unwrap();
        descriptor.check_codec(Some(&directory));
        interpret_case(&case);
        let mut executed = false;
        crate::tests::test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, crate::tests::TestPlatform::Cuda13) {
                if backend == "mosaic" || backend == "both" {
                    #[cfg(feature = "mosaic-gpu")]
                    {
                        let actual = super::gpu::execute_case(
                            &client,
                            descriptor.kernel(),
                            ryft_mosaic::kernels::gpu::Options::default(),
                        )
                        .unwrap();
                        assert_eq!(actual, case.expected);
                    }
                    #[cfg(not(feature = "mosaic-gpu"))]
                    panic!("compiler stress requires the `mosaic-gpu` feature");
                }
                if backend == "cutile" || backend == "both" {
                    #[cfg(feature = "cutile")]
                    super::cutile::execute_case(&client, descriptor.kernel(), false).unwrap();
                    #[cfg(not(feature = "cutile"))]
                    panic!("compiler stress requires the `cutile` feature");
                }
                executed = true;
            }
        });
        assert!(executed, "compiler stress must execute the CUDA 13 platform");
        eprintln!("compiler stress passed: seed={seed} iteration={iteration} backend={backend}");
    }
}
