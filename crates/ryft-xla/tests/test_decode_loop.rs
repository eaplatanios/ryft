//! Compiles a tiny language model decode loop (i.e., a `While` loop with key and value caches, gated attention,
//! greedy or top-k sampling, and an optional XLA FFI custom call) through the public frontends and compares it with an
//! eager run of the same model over the reference backend.

use std::mem::MaybeUninit;
use std::sync::OnceLock;

use pretty_assertions::assert_eq;
use ryft_core::{
    Add, ArgMax, Array as CoreArray, ArrayIrType, ArrayOperation, ArrayType, Broadcast, Compare, ComparisonDirection,
    Context, CustomCall, CustomCallOperation, DataType, Device, DeviceMesh, Dimension as CoreDimension, Div,
    DomainTracingContext, Dot, DotDimensionNumbers, DynamicSlice, DynamicUpdateSlice, EagerContext, Exp, Fill, Iota,
    LogicalMesh, Logistic, MeshAxis, MeshAxisType, Mul, OneLike, ProgramError, ProjectedValue, Random, Reduce,
    ReductionKind, Reshape, Select, Shape, Sharding, Sub, Tanh, TopK, Typed, Value as CoreValue, ValueProjection,
    WhileOperation, ZeroLike,
};
use ryft_pjrt::extensions::ffi::{
    FfiBufferType, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput,
    FfiTypeId, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::{
    Array, CompileTracer, CompiledFunction, FromPjrt, Operation, Session, StatefulCompileTracer, Value, compile,
};

/// Shape and sampling hyperparameters of the tiny decode loop demo model.
#[derive(Copy, Clone)]
struct DecodeConfiguration {
    /// Vocabulary size (i.e., the number of token embeddings and output logits).
    vocabulary: usize,

    /// Model dimension shared by the embeddings, the attention cache rows, and the gated MLP.
    dimension: usize,

    /// Number of decode steps performed by the loop, which is also the capacity of the key and value caches.
    steps: usize,

    /// Number of logits with the highest probabilities that [`DecodeSampling::TopK`] sampling retains.
    top_k: usize,
}

/// Token selection strategy of one decode step.
#[derive(Copy, Clone, PartialEq)]
enum DecodeSampling {
    /// Deterministically selects the logit with the highest score.
    Greedy,

    /// Restricts the logits to their top-k entries and draws one categorically using the threaded ThreeFry
    /// generator state.
    TopK,
}

/// Attention implementation used by one decode step.
#[derive(Copy, Clone, PartialEq)]
enum DecodeAttention {
    /// Masked scaled dot-product attention composed from `ryft` operations.
    Composed,

    /// A registered [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] XLA FFI kernel computing the same attention.
    CustomCall,
}

/// Creates a CPU client with `device_count` devices.
fn cpu_client(device_count: usize) -> Client<'static> {
    load_cpu_plugin()
        .unwrap()
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(device_count), ..Default::default() }))
        .unwrap()
}

/// Creates a mesh with one auto axis named `axis` of size `axis_size` over the addressable devices of `client`.
fn domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .into_iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    DeviceMesh::new(logical_mesh, devices).unwrap()
}

/// Encodes `values` as consecutive native-endian bytes.
fn values_to_bytes<V: Copy>(values: &[V]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        // SAFETY: `value` is a valid reference to a `Copy` value of `size_of::<V>()` bytes, and the byte view does not
        // outlive it. The tests only encode primitive integer and floating-point values, which have no padding.
        let value_bytes = unsafe { std::slice::from_raw_parts(value as *const V as *const u8, size_of::<V>()) };
        bytes.extend_from_slice(value_bytes);
    }
    bytes
}

/// Decodes `bytes` as consecutive native-endian values of type `V`. Panics if the length of `bytes` is not a multiple
/// of the size of `V`.
fn values_from_bytes<V: Copy>(bytes: &[u8]) -> Vec<V> {
    assert_eq!(bytes.len() % size_of::<V>(), 0);
    bytes
        .chunks_exact(size_of::<V>())
        .map(|chunk| {
            let mut value = MaybeUninit::<V>::uninit();
            // SAFETY: `chunk` holds exactly `size_of::<V>()` bytes, which fully initialize `value`. The tests only
            // decode primitive integer and floating-point values, for which every bit pattern is valid.
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), value.as_mut_ptr() as *mut u8, size_of::<V>());
                value.assume_init()
            }
        })
        .collect()
}

/// Reads a replicated `f32` array back from the single test device.
fn read_f32s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
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

/// Returns a static [`Shape`] with the provided dimensions.
fn static_shape(dimensions: &[usize]) -> Shape {
    Shape::new(dimensions.iter().map(|&dimension| CoreDimension::Static(dimension)).collect())
}

/// Returns `count` deterministic pseudo-random weight values in `[-0.5, 0.5)` derived from `seed` with an
/// xorshift generator, so the demo model is reproducible without carrying literal weight tables in the test.
fn decode_weight_values(seed: u32, count: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// One decode step of the tiny language model with gated attention that the compiled decode loop demo and its
/// eager reference loop share. The step is written once against the value capability traits of `ryft`, so the
/// same function executes eagerly over reference [`CoreArray`](ryft_core::Array)s and stages symbolically over
/// [`CompileTracer`]s inside the compiled `While` body.
///
/// The decode state is a vector of thirteen values. It holds the current position and token as `i32` scalars, the
/// `[steps, dimension]` key and value caches, the `[steps]` record of decoded tokens, the `ui64[2]` ThreeFry
/// generator state, and the seven weight matrices that stay invariant across the loop (i.e., the embeddings, the
/// query, key, and value projections, the hidden and gate MLP projections, and the output projection). One step
/// embeds the current token, appends its projected key and value rows to the caches at `position`, attends over
/// the visible cache prefix, applies the MLP gated by `tanh` and `logistic` and the output projection, selects the
/// next token according to `sampling`, records it, and advances the position.
fn decode_step<C, V>(
    context: &C,
    state: &[V],
    configuration: &DecodeConfiguration,
    sampling: DecodeSampling,
    attention: DecodeAttention,
) -> Result<Vec<V>, ProgramError>
where
    V: Clone
        + CoreValue<Type = ArrayType>
        + Add
        + Sub
        + Mul
        + Div
        + Exp
        + Tanh
        + Logistic
        + Dot
        + Reduce
        + Compare<V>
        + Select
        + TopK
        + ArgMax
        + DynamicSlice
        + DynamicUpdateSlice
        + Reshape
        + Broadcast
        + Random
        + CustomCall
        + OneLike
        + ZeroLike,
    C: Fill<f32, V> + Fill<i32, V> + Iota<V>,
{
    let dimension = configuration.dimension;
    let position = state[0].clone();
    let token = state[1].clone();
    let tokens = state[4].clone();
    let generator = state[5].clone();
    let embeddings = &state[6];
    let query_weights = &state[7];
    let key_weights = &state[8];
    let value_weights = &state[9];
    let hidden_weights = &state[10];
    let gate_weights = &state[11];
    let output_weights = &state[12];

    // Embed the current token and append its projected key and value rows to the caches at `position`.
    let zero_index = position.zero_like()?;
    let embedding = embeddings
        .dynamic_slice(&[token, zero_index.clone()], &[1, dimension])?
        .reshape(static_shape(&[dimension]))?;
    let vector_times_matrix = DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new());
    let query = embedding.dot(query_weights, &vector_times_matrix).unwrap();
    let key = embedding.dot(key_weights, &vector_times_matrix).unwrap();
    let value = embedding.dot(value_weights, &vector_times_matrix).unwrap();
    let cache_keys = state[2]
        .dynamic_update_slice(&key.reshape(static_shape(&[1, dimension]))?, &[position.clone(), zero_index.clone()])?;
    let cache_values = state[3]
        .dynamic_update_slice(&value.reshape(static_shape(&[1, dimension]))?, &[position.clone(), zero_index])?;

    // Masked scaled dot-product attention over the visible cache prefix `[0, position]`.
    let attended = match attention {
        DecodeAttention::Composed => {
            let scores =
                cache_keys.dot(&query, &DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new())).unwrap();
            let scores_type = scores.r#type().into_owned();
            let scale = context.fill(&scores_type, 1.0 / (dimension as f32).sqrt())?;
            let scores = scores.mul(&scale)?;
            let positions_type = tokens.r#type().into_owned();
            let positions = context.iota(&positions_type, 0)?;
            let visible =
                positions.compare(&position.broadcast(positions_type, &[])?, ComparisonDirection::LessThanOrEqual)?;
            let masked = visible.select(&scores, &context.fill(&scores_type, -1.0e30f32)?)?;
            let stabilized =
                masked.sub(&masked.reduce(&[0], ReductionKind::Max).unwrap().broadcast(scores_type.clone(), &[])?)?;
            let exponentials = stabilized.exp()?;
            let weights = exponentials
                .div(&exponentials.reduce(&[0], ReductionKind::Sum).unwrap().broadcast(scores_type, &[])?)?;
            weights
                .dot(&cache_values, &DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()))
                .unwrap()
        }
        DecodeAttention::CustomCall => {
            let operation =
                CustomCallOperation::new(DECODE_ATTENTION_CUSTOM_CALL_TARGET, vec![query.r#type().into_owned()]);
            let inputs = [cache_keys.clone(), cache_values.clone(), query, position.clone()];
            V::custom_call(&operation, &inputs)?.remove(0)
        }
    };

    // Gated multilayer perceptron, output projection, and selection of the next token.
    let hidden = attended.dot(hidden_weights, &vector_times_matrix).unwrap().tanh()?;
    let gate = attended.dot(gate_weights, &vector_times_matrix).unwrap().logistic()?;
    let logits = hidden.mul(&gate)?.dot(output_weights, &vector_times_matrix).unwrap();
    let (generator, next_token) = match sampling {
        DecodeSampling::Greedy => (generator, logits.argmax(0)?),
        DecodeSampling::TopK => {
            let (top_logits, top_indices) = logits.top_k(configuration.top_k, 0)?;
            let (generator, choice) = generator.random_categorical(&top_logits, 0)?;
            let next_token = top_indices.dynamic_slice(&[choice], &[1])?.reshape(static_shape(&[]))?;
            (generator, next_token)
        }
    };
    let tokens =
        tokens.dynamic_update_slice(&next_token.reshape(static_shape(&[1]))?, std::slice::from_ref(&position))?;
    let position = position.add(&position.one_like()?)?;
    Ok(vec![
        position,
        next_token,
        cache_keys,
        cache_values,
        tokens,
        generator,
        embeddings.clone(),
        query_weights.clone(),
        key_weights.clone(),
        value_weights.clone(),
        hidden_weights.clone(),
        gate_weights.clone(),
        output_weights.clone(),
    ])
}

/// Stages the decode loop demo as a `While` operation over the decode state of thirteen values. Both regions are
/// traced with [`decode_step`] and a `position < steps` condition through the public tracing API, and the staged
/// loop returns the decoded tokens together with the final key and value caches.
fn build_decode_loop<'c>(
    inputs: Vec<CompileTracer<'c>>,
    configuration: &DecodeConfiguration,
    sampling: DecodeSampling,
    attention: DecodeAttention,
) -> Vec<CompileTracer<'c>> {
    let context = inputs[0].value().context().clone();
    let carry_types = inputs.iter().map(|input| input.value().r#type().into_owned()).collect::<Vec<ArrayIrType>>();
    let steps = configuration.steps;

    // The `While` condition and body region programs are staged in the XLA domain universe. Their tracers are
    // `StatefulCompileTracer`s, whose `ArrayType` projections are `CompileTracer`s, so the shared `decode_step`
    // model runs unchanged inside the staged regions and in the eager reference loop.
    let (_, condition) = DomainTracingContext::<Session<'c>, Value<'c>>::trace(
        |state: Vec<StatefulCompileTracer<'c>>| {
            let state = state
                .into_iter()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).unwrap())
                .collect::<Vec<_>>();
            let position = &state[0];
            let limit = position.domain().fill(&position.r#type(), steps as i32)?;
            Ok(vec![position.compare(&limit, ComparisonDirection::LessThan)?.into_value()])
        },
        carry_types.clone(),
    )
    .unwrap();
    let (_, body) = DomainTracingContext::<Session<'c>, Value<'c>>::trace(
        |state: Vec<StatefulCompileTracer<'c>>| {
            let state = state
                .into_iter()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).unwrap())
                .collect::<Vec<_>>();
            let context = state[0].domain();
            Ok(decode_step(&context, &state, configuration, sampling, attention)?
                .into_iter()
                .map(ProjectedValue::into_value)
                .collect::<Vec<_>>())
        },
        carry_types,
    )
    .unwrap();
    let inputs = inputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>();
    let outputs = context
        .bind(Operation::While(WhileOperation::new()), vec![condition, body], inputs.as_slice())
        .unwrap();
    let outputs = outputs
        .into_iter()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).unwrap())
        .collect::<Vec<_>>();
    vec![outputs[4].clone(), outputs[2].clone(), outputs[3].clone()]
}

/// Runs the eager reference of the decode loop demo, which executes the same [`decode_step`] over reference
/// [`CoreArray`](ryft_core::Array) values in a plain Rust loop, without `While` staging, compilation, or XLA.
fn reference_decode_loop(
    initial_state: Vec<CoreArray>,
    configuration: &DecodeConfiguration,
    sampling: DecodeSampling,
) -> Vec<CoreArray> {
    let context = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::new();
    let mut state = initial_state;
    for _ in 0..configuration.steps {
        state = decode_step(&context, &state, configuration, sampling, DecodeAttention::Composed).unwrap();
    }
    state
}

/// Name of the XLA custom call target registered by [`ensure_decode_attention_handler_registered`]. The target is
/// a masked scaled dot-product attention kernel over `(keys [steps, dimension], values [steps, dimension],
/// query [dimension], position i32[])` that produces the attended `[dimension]` vector, and it exercises a foreign
/// FFI kernel inside a compiled decode loop.
const DECODE_ATTENTION_CUSTOM_CALL_TARGET: &str = "ryft.test.decode_attention";

/// Registers the [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided
/// client. Registration is idempotent and global to the process, because the XLA runtime rejects registering the
/// same target name again, so the outcome is cached in a [`OnceLock`](std::sync::OnceLock).
fn ensure_decode_attention_handler_registered(client: &Client<'_>) -> Result<(), ryft_pjrt::Error> {
    static DECODE_ATTENTION_HANDLER_REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();
    DECODE_ATTENTION_HANDLER_REGISTRATION
        .get_or_init(|| {
            let platform_name = client.platform_name()?.into_owned();
            client.register_ffi_handler(
                DECODE_ATTENTION_CUSTOM_CALL_TARGET,
                platform_name,
                FfiHandler::from(decode_attention_handler as XLA_FFI_Handler),
                FfiHandlerTraits::NONE,
            )
        })
        .clone()
}

/// XLA FFI handler for [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] custom calls. It computes masked scaled
/// dot-product attention over the cache prefix `[0, position]`, matching the composed attention staged by
/// [`decode_step`].
unsafe extern "C" fn decode_attention_handler(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and all
    // further unsafe access to it is localized in the safe `FfiCallFrame` wrapper and
    // `handle_decode_attention_call_frame`.
    unsafe {
        match FfiCallFrame::from_c_api(call_frame) {
            Err(_) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
            Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
            Ok(call_frame) => match call_frame.api() {
                Err(_) => std::ptr::null_mut(),
                Ok(api) => match handle_decode_attention_call_frame(&call_frame) {
                    Ok(()) => std::ptr::null_mut(),
                    Err(error) => error.to_c_api(api),
                },
            },
        }
    }
}

/// Decodes a [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] call frame and fills its `[dimension]` output buffer with
/// the masked scaled dot-product attention of the query over the `[0, position]` cache prefix, using the same
/// `-1e30` mask value and the same softmax stabilized by the maximum as the composed attention in [`decode_step`].
fn handle_decode_attention_call_frame(call_frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    let mut inputs = call_frame.inputs();
    let mut next_buffer = |name: &str| match inputs.next() {
        Some(Ok(FfiInput::Buffer { buffer })) => Ok(buffer),
        _ => Err(FfiError::invalid_argument(format!(
            "expected the `{name}` input buffer of the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call"
        ))),
    };
    let keys = next_buffer("keys")?;
    let values = next_buffer("values")?;
    let query = next_buffer("query")?;
    let position = next_buffer("position")?;
    let mut outputs = call_frame.outputs();
    let Some(Ok(FfiOutput::Buffer { buffer: output })) = outputs.next() else {
        return Err(FfiError::invalid_argument(format!(
            "expected the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call to have one output buffer"
        )));
    };
    let &[steps, dimension] = keys.dimensions() else {
        return Err(FfiError::invalid_argument(format!(
            "expected the `keys` input of the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call to have rank 2"
        )));
    };
    if keys.element_type() != FfiBufferType::F32
        || values.element_type() != FfiBufferType::F32
        || query.element_type() != FfiBufferType::F32
        || position.element_type() != FfiBufferType::I32
        || values.dimensions() != [steps, dimension]
        || query.dimensions() != [dimension]
        || !position.dimensions().is_empty()
        || output.dimensions() != [dimension]
    {
        return Err(FfiError::invalid_argument(format!(
            "unexpected `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call buffer types or shapes"
        )));
    }
    let steps = steps.max(0) as usize;
    let dimension = dimension.max(0) as usize;
    // SAFETY: All data pointers are provided by the XLA runtime, are valid for the duration of the handler
    // invocation, and, according to the element type and shape checks above, are backed by allocations of the
    // checked sizes. The runtime allocates inputs and outputs separately so they do not overlap.
    unsafe {
        let keys = std::slice::from_raw_parts(keys.data() as *const f32, steps * dimension);
        let values = std::slice::from_raw_parts(values.data() as *const f32, steps * dimension);
        let query = std::slice::from_raw_parts(query.data() as *const f32, dimension);
        let position = *(position.data() as *const i32);
        let output = std::slice::from_raw_parts_mut(output.data() as *mut f32, dimension);

        let scale = 1.0f32 / (dimension as f32).sqrt();
        let mut scores = vec![0.0f32; steps];
        for step in 0..steps {
            let row = &keys[step * dimension..(step + 1) * dimension];
            let score = row.iter().zip(query).map(|(key, query)| key * query).sum::<f32>() * scale;
            scores[step] = if step as i32 <= position { score } else { -1.0e30 };
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f32 = exponentials.iter().sum();
        output.fill(0.0);
        for step in 0..steps {
            let weight = exponentials[step] / denominator;
            let row = &values[step * dimension..(step + 1) * dimension];
            for (accumulator, value) in output.iter_mut().zip(row) {
                *accumulator += weight * value;
            }
        }
    }
    Ok(())
}

/// Reads a replicated `i32` array back from the single test device.
fn read_i32_array(client: &Client<'_>, array: &Array<'_>) -> Vec<i32> {
    let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
    let shard_bytes = array
        .device_shard(device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    values_from_bytes::<i32>(shard_bytes.as_slice())
}

/// Shared driver of the decode loop demo tests. It traces the decode loop through the public API, compiles and
/// executes it on the CPU plugin, runs the same model eagerly over the reference backend, and compares the decoded
/// token sequence exactly, because the ThreeFry draws are bit-exact across backends, and the final key and value
/// caches within floating-point tolerance.
fn run_decode_loop_demo(sampling: DecodeSampling, attention: DecodeAttention) {
    let client = cpu_client(1);
    let mesh = domain_mesh(&client, "x", 1);
    let engine = Session::builder().client(&client).build().unwrap();
    if attention == DecodeAttention::CustomCall {
        ensure_decode_attention_handler_registered(&client).unwrap();
    }

    let configuration = DecodeConfiguration { vocabulary: 16, dimension: 8, steps: 6, top_k: 4 };
    let DecodeConfiguration { vocabulary, dimension, steps, .. } = configuration;
    let weight_dimensions: [&[usize]; 7] = [
        &[vocabulary, dimension],
        &[dimension, dimension],
        &[dimension, dimension],
        &[dimension, dimension],
        &[dimension, dimension],
        &[dimension, dimension],
        &[dimension, vocabulary],
    ];
    let weights: Vec<Vec<f32>> = weight_dimensions
        .iter()
        .enumerate()
        .map(|(index, dimensions)| decode_weight_values(index as u32 + 1, dimensions.iter().product()))
        .collect();

    let replicated = |data_type: DataType, dimensions: &[usize]| {
        ArrayType::new_static(data_type, dimensions)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))
            .unwrap()
    };
    let mut input_types = vec![
        replicated(DataType::I32, &[]),
        replicated(DataType::I32, &[]),
        replicated(DataType::F32, &[steps, dimension]),
        replicated(DataType::F32, &[steps, dimension]),
        replicated(DataType::I32, &[steps]),
        replicated(DataType::U64, &[2]),
    ];
    input_types.extend(weight_dimensions.iter().map(|dimensions| replicated(DataType::F32, dimensions)));

    let compiled: CompiledFunction<'_, Vec<ArrayType>, Vec<ArrayType>> = compile(
        |inputs| build_decode_loop(inputs, &configuration, sampling, attention),
        input_types.clone(),
        &engine,
        mesh.clone(),
    )
    .unwrap();

    let device_input = |index: usize, bytes: Vec<u8>| {
        Array::from_host_buffer(&engine, input_types[index].clone(), mesh.clone(), bytes.as_slice()).unwrap()
    };
    let mut device_inputs = vec![
        device_input(0, values_to_bytes::<i32>(&[0])),
        device_input(1, values_to_bytes::<i32>(&[3])),
        device_input(2, values_to_bytes::<f32>(&vec![0.0; steps * dimension])),
        device_input(3, values_to_bytes::<f32>(&vec![0.0; steps * dimension])),
        device_input(4, values_to_bytes::<i32>(&vec![0; steps])),
        device_input(5, values_to_bytes::<u64>(&[42, 0])),
    ];
    device_inputs
        .extend(weights.iter().enumerate().map(|(index, values)| device_input(index + 6, values_to_bytes(values))));
    let outputs = compiled.executable().execute(&engine, device_inputs).unwrap();
    assert_eq!(outputs.len(), 3);
    let device_tokens = read_i32_array(&client, &outputs[0]);
    let device_cache_keys = read_f32s(&client, &outputs[1]);
    let device_cache_values = read_f32s(&client, &outputs[2]);

    let mut reference_state = vec![
        CoreArray::from_elements(ArrayType::new_static(DataType::I32, []), &[0i32]).unwrap(),
        CoreArray::from_elements(ArrayType::new_static(DataType::I32, []), &[3i32]).unwrap(),
        CoreArray::from_elements(
            ArrayType::new_static(DataType::F32, [steps, dimension]),
            &vec![0.0f32; steps * dimension],
        )
        .unwrap(),
        CoreArray::from_elements(
            ArrayType::new_static(DataType::F32, [steps, dimension]),
            &vec![0.0f32; steps * dimension],
        )
        .unwrap(),
        CoreArray::from_elements(ArrayType::new_static(DataType::I32, [steps]), &vec![0i32; steps]).unwrap(),
        CoreArray::from_elements(ArrayType::new_static(DataType::U64, [2]), &[42u64, 0]).unwrap(),
    ];
    reference_state.extend(weight_dimensions.iter().zip(&weights).map(|(dimensions, values)| {
        CoreArray::from_elements(ArrayType::new_static(DataType::F32, *dimensions), values).unwrap()
    }));
    let reference_state = reference_decode_loop(reference_state, &configuration, sampling);
    let reference_tokens = reference_state[4].elements::<i32>().unwrap();

    // The decoded token sequences must agree exactly, because greedy selection and the categorical draws over
    // bit-identical ThreeFry bits are only sensitive to floating-point differences at exact logit ties, which the
    // deterministic pseudo-random weights avoid.
    assert_eq!(device_tokens, reference_tokens);
    assert!(reference_tokens.iter().all(|&token| (0..vocabulary as i32).contains(&token)));
    // The decoded sequences are pinned as fixtures. They are deterministic functions of the pseudo-random weights
    // and the ThreeFry seed, and pinning them guards against both backends drifting together. The
    // greedy and sampled sequences differ, demonstrating that the categorical draws actually steer decoding.
    let expected_tokens: &[i32] = match sampling {
        DecodeSampling::Greedy => &[12, 8, 12, 8, 12, 8],
        DecodeSampling::TopK => &[13, 12, 10, 10, 2, 15],
    };
    assert_eq!(device_tokens, expected_tokens);
    for (state_index, device_cache) in [(2, &device_cache_keys), (3, &device_cache_values)] {
        let reference_cache = reference_state[state_index].to_f64s();
        assert_eq!(device_cache.len(), reference_cache.len());
        for (index, (&device_value, &reference_value)) in device_cache.iter().zip(reference_cache.iter()).enumerate() {
            assert!(
                (device_value as f64 - reference_value).abs() < 1e-4,
                "cache element {index}: device {device_value} vs reference {reference_value}",
            );
        }
    }
}

#[test]
fn test_compile_decode_loop_greedy_matches_reference_backend() {
    // A compiled `While` loop maintains `DynamicUpdateSlice` key and value caches with composed masked attention,
    // an MLP gated by `tanh` and `logistic`, and greedy `argmax` sampling. It is traced through the public API and
    // compared against the eager reference backend.
    run_decode_loop_demo(DecodeSampling::Greedy, DecodeAttention::Composed);
}

#[test]
fn test_compile_decode_loop_top_k_sampling_matches_reference_backend() {
    // Like the greedy demo, but tokens are selected with `top_k` and a categorical draw from the ThreeFry generator
    // state threaded through the loop carry. This exercises `RngBitGenerator` inside a compiled `While` region
    // with bit-exact token parity across backends.
    run_decode_loop_demo(DecodeSampling::TopK, DecodeAttention::Composed);
}

#[test]
fn test_compile_decode_loop_custom_call_attention_matches_reference_backend() {
    // The attention body is swapped for a registered XLA FFI custom call kernel, which exercises `custom_call`
    // inside a compiled `While` region against the composed eager reference.
    run_decode_loop_demo(DecodeSampling::Greedy, DecodeAttention::CustomCall);
}
