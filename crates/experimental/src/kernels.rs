//! Portable and explicit GPU kernel experiments with independent numerical oracles.

use pretty_assertions::assert_eq;

use ryft::kernel;
use ryft_core::{Array, ArrayType, Context, DataType, ReferenceRead, ReferenceWrite};
use ryft_kernels::{
    AsyncCopyOperation, Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, ScratchOperation,
    WaitOperation, whole_array_parameter,
};

/// Adds independently loaded tiles, including a partially populated final output tile.
#[kernel(requires = left.shape()[0] == right.shape()[0])]
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
#[kernel]
fn sum(#[input(data_type = F32, rank = 1)] input: &Array, #[output(data_type = F32, shape = [])] output: &mut Array) {
    output.store(input.load().sum([0]));
}

/// Carries tiled dot products through a staged reduction loop with partial tiles on every axis.
#[kernel(requires = left.shape()[1] == right.shape()[0])]
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

/// Multiplies two matrices whose extents are multiples of 32 on a grid of blocks. The program `(i, j, k)` adds the
/// product of the input blocks `(i, k)` and `(k, j)` into the output block `(i, j)`, which it zeroes at the first step
/// of the sequential axis `k`, so each output block is initialized once and then accumulates the whole contraction.
#[kernel(
    grid(
        i = left.shape()[0].div_ceil(32),
        j = right.shape()[1].div_ceil(32),
        k = sequential(left.shape()[1].div_ceil(32)),
    ),
    requires = left.shape()[1] == right.shape()[0],
)]
fn blocked_matmul(
    #[input(data_type = F32, rank = 2, block = [32, 32], index = [i, k])] left: &Array,
    #[input(data_type = F32, rank = 2, block = [32, 32], index = [k, j])] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], block = [32, 32], index = [i, j])]
    output: &mut Array,
) {
    if k == 0 {
        output.store(zeros::<f32>([32, 32]));
    }
    output.accumulate(left.load().dot(right.load()));
}

/// Multiplies two matrices of any shape like [`blocked_matmul`]. The edge blocks are masked, so their padding is
/// loaded as zero, contributes nothing to the products, and is never stored.
#[kernel(
    grid(
        i = left.shape()[0].div_ceil(32),
        j = right.shape()[1].div_ceil(32),
        k = sequential(left.shape()[1].div_ceil(32)),
    ),
    requires = left.shape()[1] == right.shape()[0],
)]
fn masked_blocked_matmul(
    #[input(data_type = F32, rank = 2, block = [32, 32], index = [i, k], boundary = masked)] left: &Array,
    #[input(data_type = F32, rank = 2, block = [32, 32], index = [k, j], boundary = masked)] right: &Array,
    #[output(
        data_type = F32,
        shape = [left.shape()[0], right.shape()[1]],
        block = [32, 32],
        index = [i, j],
        boundary = masked,
    )]
    output: &mut Array,
) {
    if k == 0 {
        output.store(zeros::<f32>([32, 32]));
    }
    output.accumulate(left.load_or(0.0).dot(right.load_or(0.0)));
}

/// Computes the elementwise sum and difference of two vectors whose length is a multiple of 128 in one pass. The
/// callable returns both results as a tuple in declaration order.
#[kernel(grid(i = left.shape()[0].div_ceil(128)), requires = left.shape()[0] == right.shape()[0])]
fn sum_and_difference(
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] left: &Array,
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]], block = [128], index = [i])] sum: &mut Array,
    #[output(data_type = F32, shape = [left.shape()[0]], block = [128], index = [i])] difference: &mut Array,
) {
    let first = left.load();
    let second = right.load();
    sum.store(first + second);
    difference.store(first - second);
}

/// Adds a vector whose length is a multiple of 128 into another one in place. The `inout` parameter is read and
/// written by the same program, and the callable returns its updated value.
#[kernel(grid(i = input.shape()[0].div_ceil(128)), requires = input.shape()[0] == accumulator.shape()[0])]
fn add_in_place(
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] input: &Array,
    #[inout(data_type = F32, rank = 1, block = [128], index = [i])] accumulator: &mut Array,
) {
    accumulator.store(accumulator.load() + input.load());
}

/// Replaces the negative elements of a vector whose length is a multiple of 128 with zero in place. The store is
/// masked by a mask that the body computes, so the other elements are never written.
#[kernel(grid(i = values.shape()[0].div_ceil(128)))]
fn relu_in_place(#[inout(data_type = F32, rank = 1, block = [128], index = [i])] values: &mut Array) {
    let value = values.load();
    values.store_where(zeros::<f32>([128]), value < 0.0f32);
}

/// Doubles a vector of any length in place. The blocks are masked, and the store is masked by the window mask of the
/// block, which selects exactly the elements of the last block that lie within the vector.
#[kernel(grid(i = values.shape()[0].div_ceil(128)))]
fn double_in_place(
    #[inout(data_type = F32, rank = 1, block = [128], index = [i], boundary = masked)] values: &mut Array,
) {
    let value = values.load_or(0.0);
    values.store_where(value + value, values.window_mask());
}

/// Keeps the elements of `values` whose weight is positive and zeroes the others, for vectors whose length is a
/// multiple of 128. The masked load reads only the selected elements of `values` and fills the rest with zero.
#[kernel(grid(i = values.shape()[0].div_ceil(128)), requires = values.shape()[0] == weights.shape()[0])]
fn keep_weighted(
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] values: &Array,
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] weights: &Array,
    #[output(data_type = F32, shape = [values.shape()[0]], block = [128], index = [i])] output: &mut Array,
) {
    output.store(values.load_where(weights.load() > 0.0f32, 0.0));
}

/// Replaces the negative elements of a vector whose length is a multiple of 128 with zero in place, and returns the
/// replaced elements, with zero at the positions that were kept. The masked swap writes and reads in one access.
#[kernel(grid(i = values.shape()[0].div_ceil(128)))]
fn extract_negatives(
    #[inout(data_type = F32, rank = 1, block = [128], index = [i])] values: &mut Array,
    #[output(data_type = F32, shape = [values.shape()[0]], block = [128], index = [i])] negatives: &mut Array,
) {
    let value = values.load();
    negatives.store(values.swap_where(zeros::<f32>([128]), value < 0.0f32, 0.0));
}

/// Copies a vector through uninitialized scratch storage that the kernel writes before it reads.
#[kernel]
fn scratch_copy(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let staging = scratch::<f32>([input.shape()[0]]);
    staging.store(input.load());
    output.store(staging.load());
}

/// Copies a vector asynchronously into scratch storage aligned to 16 bytes and waits for the copy to complete before
/// reading the scratch storage.
#[kernel]
fn async_copy(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let staging = scratch_aligned::<f32>([input.shape()[0]], 16);
    let token = input.copy_to(staging);
    token.wait();
    output.store(staging.load());
}

/// Copies the block of 128 elements of `table` whose block index is the prefetched scalar `block`, which the window
/// mapping of `table` reads before the kernel runs.
#[kernel(grid(i = 1))]
fn gather_block(
    #[input(data_type = F32, rank = 1, block = [128], index = [block])] table: &Array,
    #[prefetch(data_type = I32)] block: &Array,
    #[output(data_type = F32, shape = [128])] output: &mut Array,
) {
    output.store(table.load());
}

/// Reverses the order of the blocks of 128 elements of a vector whose length is a multiple of 128. The input windows
/// start at element offsets that run backwards, which no separable tiling describes, so the verifier enumerates the
/// grid to prove that the windows lie within the input.
#[kernel(grid(i = input.shape()[0].div_ceil(128)))]
fn reverse_blocks(
    #[input(data_type = F32, rank = 1, block = [128], start = [input.shape()[0] - 128 - i * 128])] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]], block = [128], index = [i])] output: &mut Array,
) {
    output.store(input.load());
}

/// Sums a vector whose length is a multiple of 128 into `total` on a parallel grid. Every program reduces its own
/// block and adds the partial sum to `total` atomically, so the programs need no ordering.
#[kernel(grid(i = input.shape()[0].div_ceil(128)))]
fn parallel_sum(
    #[input(data_type = F32, rank = 1, block = [128], index = [i])] input: &Array,
    #[inout(data_type = F32, rank = 0)] total: &mut Array,
) {
    total.atomic_accumulate(input.load().reduce_sum([0]));
}

/// Copies the first two columns of a matrix through a static slice of its window.
#[kernel(requires = input.shape()[1] >= 2)]
fn leading_columns(
    #[input(data_type = F32, rank = 2)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0], 2])] output: &mut Array,
) {
    output.store(input.slice([.., 0..2]).load());
}

/// Sums the exponentials of the elements of a vector that are not positive, which is the denominator of a softmax
/// restricted to those elements. A comparison selects the exponential or zero per element, and the selected values are
/// summed in F64 and rounded back to F32.
#[kernel]
fn nonpositive_exp_sum(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
    let value = input.load();
    let exponentials = select(value <= 0.0f32, value.exp(), zeros::<f32>([input.shape()[0]]));
    output.store((exponentials as f64).reduce_sum([0]) as f32);
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
        let maximum = bind(
            ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Max)),
            std::slice::from_ref(&scores),
        )?;
        let maximum =
            bind(ArrayOperation::Broadcast(BroadcastOperation::new(scores_type.clone(), vec![0])), &[maximum])?;
        let centered = bind(ArrayOperation::Sub(SubOperation::new()), &[scores, maximum])?;
        let exponentials = bind(ArrayOperation::Exp(ExpOperation::new()), &[centered])?;
        let sum = bind(
            ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
            std::slice::from_ref(&exponentials),
        )?;
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
        let token = context
            .bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), destination.clone()])?
            .remove(0);
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
    for case in kernel_cases()
        .into_iter()
        .filter(|case| case.name != "async_copy")
        .chain([precision_case(), batched_case()])
    {
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

/// Returns the operands of a `rows` by `depth` by `columns` matrix multiplication and their product. The elements are
/// small integers, so every FP32 product and sum is exact regardless of the accumulation order.
fn matmul_operands(rows: usize, depth: usize, columns: usize) -> (Array, Array, Array) {
    let left = (0..rows * depth).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
    let right = (0..depth * columns).map(|index| (index % 5) as f32 - 2.0).collect::<Vec<_>>();
    let mut product = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            for inner in 0..depth {
                product[row * columns + column] += left[row * depth + inner] * right[inner * columns + column];
            }
        }
    }
    (
        Array::matrix(rows, depth, left).unwrap(),
        Array::matrix(depth, columns, right).unwrap(),
        Array::matrix(rows, columns, product).unwrap(),
    )
}

#[test]
fn test_blocked_matmul() {
    let (left, right, product) = matmul_operands(64, 96, 32);
    assert_eq!(blocked_matmul(&left, &right), Ok(product));
}

#[test]
fn test_masked_blocked_matmul() {
    // The shapes cover a single partial block, partial edge blocks on every axis, and complete blocks only.
    for (rows, depth, columns) in [(1, 1, 1), (33, 35, 34), (64, 96, 32)] {
        let (left, right, product) = matmul_operands(rows, depth, columns);
        assert_eq!(masked_blocked_matmul(&left, &right), Ok(product));
    }
}

#[test]
fn test_sum_and_difference() {
    let left = (0..256).map(|index| index as f32 - 100.0).collect::<Vec<_>>();
    let right = (0..256).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
    let sum = left.iter().zip(&right).map(|(left, right)| left + right).collect::<Vec<_>>();
    let difference = left.iter().zip(&right).map(|(left, right)| left - right).collect::<Vec<_>>();
    assert_eq!(
        sum_and_difference(&Array::vector(left).unwrap(), &Array::vector(right).unwrap()),
        Ok((Array::vector(sum).unwrap(), Array::vector(difference).unwrap())),
    );
}

#[test]
fn test_add_in_place() {
    let input = (0..256).map(|index| (index % 11) as f32 - 5.0).collect::<Vec<_>>();
    let accumulator = (0..256).map(|index| index as f32).collect::<Vec<_>>();
    let expected = input.iter().zip(&accumulator).map(|(input, accumulator)| input + accumulator).collect::<Vec<_>>();
    assert_eq!(
        add_in_place(&Array::vector(input).unwrap(), &Array::vector(accumulator).unwrap()),
        Ok(Array::vector(expected).unwrap()),
    );
}

#[test]
fn test_relu_in_place() {
    let values = (0..256).map(|index| (index % 9) as f32 - 4.0).collect::<Vec<_>>();
    let expected = values.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
    assert_eq!(relu_in_place(&Array::vector(values).unwrap()), Ok(Array::vector(expected).unwrap()));
}

#[test]
fn test_double_in_place() {
    // The last of the three blocks has 44 valid elements.
    let values = (0..300).map(|index| index as f32 - 150.0).collect::<Vec<_>>();
    let expected = values.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
    assert_eq!(double_in_place(&Array::vector(values).unwrap()), Ok(Array::vector(expected).unwrap()));
}

#[test]
fn test_keep_weighted() {
    let values = (0..256).map(|index| index as f32 + 1.0).collect::<Vec<_>>();
    let weights = (0..256).map(|index| (index % 3) as f32 - 1.0).collect::<Vec<_>>();
    let expected = values
        .iter()
        .zip(&weights)
        .map(|(value, weight)| if *weight > 0.0 { *value } else { 0.0 })
        .collect::<Vec<_>>();
    assert_eq!(
        keep_weighted(&Array::vector(values).unwrap(), &Array::vector(weights).unwrap()),
        Ok(Array::vector(expected).unwrap()),
    );
}

#[test]
fn test_extract_negatives() {
    let values = (0..256).map(|index| (index % 9) as f32 - 4.0).collect::<Vec<_>>();
    let kept = values.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
    let negatives = values.iter().map(|value| value.min(0.0)).collect::<Vec<_>>();
    assert_eq!(
        extract_negatives(&Array::vector(values).unwrap()),
        Ok((Array::vector(kept).unwrap(), Array::vector(negatives).unwrap())),
    );
}

#[test]
fn test_scratch_copy() {
    let input = Array::vector((0..67).map(|index| index as f32 - 20.0).collect::<Vec<_>>()).unwrap();
    assert_eq!(scratch_copy(&input), Ok(input));
}

#[test]
fn test_async_copy() {
    let input = Array::vector((0..67).map(|index| index as f32 - 20.0).collect::<Vec<_>>()).unwrap();
    assert_eq!(async_copy(&input), Ok(input));
}

#[test]
fn test_gather_block() {
    let table = Array::vector((0..512).map(|index| index as f32).collect::<Vec<_>>()).unwrap();
    assert_eq!(
        gather_block(&table, &Array::scalar(2i32).unwrap()),
        Ok(Array::vector((256..384).map(|index| index as f32).collect::<Vec<_>>()).unwrap()),
    );
}

#[test]
fn test_reverse_blocks() {
    let input = (0..384).map(|index| index as f32).collect::<Vec<_>>();
    let expected = input.chunks(128).rev().flatten().copied().collect::<Vec<_>>();
    assert_eq!(reverse_blocks(&Array::vector(input).unwrap()), Ok(Array::vector(expected).unwrap()));
}

#[test]
fn test_parallel_sum() {
    let input = (0..384).map(|index| (index % 13) as f32 - 6.0).collect::<Vec<_>>();
    let expected = 10.0 + input.iter().sum::<f32>();
    assert_eq!(
        parallel_sum(&Array::vector(input).unwrap(), &Array::scalar(10.0f32).unwrap()),
        Ok(Array::scalar(expected).unwrap()),
    );
}

#[test]
fn test_leading_columns() {
    let input = Array::matrix(3, 4, (0..12).map(|index| index as f32).collect()).unwrap();
    assert_eq!(leading_columns(&input), Ok(Array::matrix(3, 2, vec![0.0f32, 1.0, 4.0, 5.0, 8.0, 9.0]).unwrap()));
}

#[test]
fn test_nonpositive_exp_sum() {
    // The exponentials of zero and negative infinity are exact, so the expected sum is exact.
    let input = Array::vector(vec![0.0f32, 1.5, f32::NEG_INFINITY, -0.0, 2.0]).unwrap();
    assert_eq!(nonpositive_exp_sum(&input), Ok(Array::scalar(2.0f32).unwrap()));
}

/// Exports a macro-authored whole-array kernel, reloads it in a fresh session, and awaits its native output.
#[cfg(any(feature = "mosaic-gpu", feature = "cutile", feature = "triton"))]
fn execute_aot_case(
    plugin: &crate::Plugin,
    client: &'static crate::Client<'static>,
    binding: ryft_xla::kernels::KernelCompilerBinding,
    deployment_binding: ryft_xla::kernels::KernelCompilerBinding,
    backend: &str,
) {
    use std::io::Read;
    use std::sync::Arc;

    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};
    use sha2::{Digest, Sha256};

    use ryft_xla::kernels::{KernelAotBundle, KernelExecutionFacts};
    use ryft_xla::{FromPjrt, Options, Session};

    /// Uses the portable source codec's whole-array subset with no hidden captures or runtime references.
    #[kernel]
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
    let producer = Session::builder().client(client).build().unwrap();
    let bundle = KernelAotBundle::compile(
        &definition,
        &producer,
        Options::new(mesh.clone()).with_kernel_compiler(binding.clone()),
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
    let runtime = Session::builder().client(client).build().unwrap();
    let loaded = restored.load(&runtime, &deployment_binding, &mesh).unwrap();
    let inputs: Vec<_> = [1.0f32, 2.0]
        .into_iter()
        .map(|value| {
            runtime
                .with_device_mesh(mesh.clone())
                .array(r#type.clone(), value.to_ne_bytes().repeat(64))
                .unwrap()
        })
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
        let facts = KernelExecutionFacts::from_target(runtime.target().unwrap(), &mesh).unwrap();
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
        let executable_digest = hex::encode(executable_digest.finalize());
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
                "execution_digest": hex::encode(Sha256::digest(facts.configuration_key().unwrap())),
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
            .map(|value| {
                runtime
                    .with_device_mesh(mesh.clone())
                    .array(r#type.clone(), value.to_ne_bytes().repeat(64))
                    .unwrap()
            })
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
    use std::time::Instant;

    use pretty_assertions::assert_eq;
    use sha2::{Digest, Sha256};

    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProgramError, ProjectedValue, Typed};
    use ryft_kernels::{KernelExtension as PortableKernelExtension, KernelOperation, KernelSchedule};
    use ryft_mosaic::kernels::gpu::{Compiler, GpuOperation, Options as MosaicOptions, Target};
    use ryft_xla::kernels::{KernelCompilerBinding, KernelExtension, MosaicGpuEmbedding, stage_kernel};
    use ryft_xla::{CompileTracer, CompiledFunction, Error, FromPjrt, Options, Session, compile_with_options};

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
                let binding = KernelCompilerBinding::new(
                    Compiler,
                    device_target(client),
                    MosaicOptions::default(),
                    KernelSchedule::default(),
                    MosaicGpuEmbedding,
                    1024,
                )
                .unwrap();
                execute_aot_case(&_plugin, client, binding.clone(), binding, "mosaic");
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
    pub(super) fn execute_case(
        client: &'static crate::Client<'static>,
        case: KernelCase,
        options: MosaicOptions,
    ) -> Result<Vec<f32>, Error> {
        eprintln!(
            "Mosaic {}: semantic={}",
            case.name,
            hex::encode(Sha256::digest(case.definition.semantic_key().unwrap()))
        );
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
    fn execute_definition<Extension>(
        client: &'static crate::Client<'static>,
        definition: &KernelDefinition<Extension>,
        input_types: Vec<ArrayType>,
        input_bytes: Vec<Vec<u8>>,
        target: Target,
        options: MosaicOptions,
        iterations: usize,
    ) -> Result<Vec<f32>, Error>
    where
        Extension: 'static
            + Send
            + Sync
            + PortableKernelExtension
            + Into<GpuOperation>
            + Into<KernelExtension>
            + TryFrom<KernelExtension, Error = ProgramError>,
    {
        assert!((1..=128).contains(&iterations));
        let description = format!("{options:?}");
        let device = client.addressable_devices().unwrap().remove(0);
        assert_eq!(target.compute_capability(), device_target(client).compute_capability());
        let binding = KernelCompilerBinding::new_with_extensions::<_, _, Extension>(
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
        let session = Session::builder().client(client).build()?;
        let compiled: CompiledFunction<Vec<ArrayType>, Vec<ArrayType>> = compile_with_options(
            |inputs: Vec<CompileTracer>| {
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
            &session,
            Options::new(mesh.clone()).with_kernel_compiler(binding),
        )
        .unwrap();
        let inputs = input_types
            .into_iter()
            .zip(input_bytes)
            .map(|(r#type, bytes)| session.with_device_mesh(mesh.clone()).array(r#type, bytes).unwrap())
            .collect::<Vec<_>>();
        let warmup = usize::from(iterations > 1);
        let mut start = Instant::now();
        let mut bytes = Vec::new();
        for iteration in 0..iterations + warmup {
            if iteration == warmup {
                start = Instant::now();
            }
            let outputs = compiled.executable().execute(&session, inputs.clone())?;
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
                assert_eq!(execute_case(client, case, MosaicOptions::default()).unwrap(), expected, "{name}");
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
                        client,
                        &definition,
                        input_types,
                        input_bytes,
                        device_target(client),
                        MosaicOptions::default(),
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
                assert_eq!(execute_case(client, case, MosaicOptions::default().with_tma(true)).unwrap(), expected);
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
                        client,
                        &case.definition,
                        case.input_types,
                        bytes,
                        device_target(client),
                        MosaicOptions::default().with_tma(tma),
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
                    let target = device_target(client).with_blocks_per_cluster(blocks).unwrap();
                    let actual = execute_definition(
                        client,
                        &definition,
                        vec![r#type.clone()],
                        vec![bytes.clone()],
                        target,
                        MosaicOptions::default(),
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
                    client,
                    &case.definition,
                    case.input_types,
                    bytes,
                    device_target(client).with_blocks_per_cluster(2).unwrap(),
                    MosaicOptions::default(),
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
                    let actual = execute_case(client, case, MosaicOptions::default()).unwrap();
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
                let actual = execute_case(client, case, MosaicOptions::default()).unwrap();
                check_attention(&actual, &expected);
            }
        });
    }

    #[test]
    fn test_attention_wgmma_module() {
        use ryft_kernels::VerifiedKernel;
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
            .module(
                &context,
                &kernel,
                &target,
                &MosaicOptions::default().with_mma(Mma::Wgmma),
                &KernelSchedule::default(),
            )
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
                            DimensionVariable::new("checked", DimensionBounds::new(0, Some(2))?).into(),
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
                let error = execute_case(client, case, MosaicOptions::default()).unwrap_err().to_string();
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

    use ryft_core::{
        AddOperation, ArrayIrType, ArrayOperation, CompilationCacheDomain, ConstantOperation, Device, DeviceMesh,
        LogicalMesh, MeshAxis, MeshAxisType, StagedFunction as CoreStagedFunction,
    };
    use ryft_cuda::kernels::cutile::{CompiledKernel, Compiler, Options as CuTileOptions, Target};
    use ryft_kernels::{KernelSchedule, VerifiedKernel};
    use ryft_xla::kernels::{CuTileEmbedding, KernelCompilerBinding, stage_kernel};
    use ryft_xla::{Error, FromPjrt, Session, Value};

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
                let binding = KernelCompilerBinding::new(
                    Compiler::new(python),
                    Target::new(major.parse().unwrap(), minor.parse().unwrap()).unwrap(),
                    CuTileOptions::default(),
                    KernelSchedule::default(),
                    CuTileEmbedding,
                    1024,
                )
                .unwrap();
                execute_aot_case(&_plugin, client, binding.clone(), binding, "cutile");
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
    pub(super) fn execute_case(
        client: &'static crate::Client<'static>,
        case: KernelCase,
        repeated_input: bool,
    ) -> Result<(), Error> {
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
        let options = CuTileOptions::default();
        eprintln!(
            "cuTile {}: semantic={}",
            case.name,
            hex::encode(Sha256::digest(case.definition.semantic_key().unwrap()))
        );
        let verified = VerifiedKernel::new(&case.definition, 1024).unwrap();
        let output = verified.compile(&compiler, &target, &options, &schedule).unwrap();
        let binding = KernelCompilerBinding::new(compiler, target, options, schedule, CuTileEmbedding, 1024).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let producer = Session::builder().client(client).build()?;
        let staged = CoreStagedFunction::<Session, Vec<ArrayIrType>, Vec<ArrayIrType>>::stage(
            |_, _, inputs: Vec<ryft_core::CompilationTracer<Session>>| {
                Ok(stage_kernel(inputs[0].context(), &case.definition, &inputs)?)
            },
            vec![],
            case.input_types.iter().cloned().map(ArrayIrType::Array).collect(),
            ryft_xla::Options::new(mesh.clone()).with_kernel_compiler(binding),
        )
        .unwrap();
        let compiled = staged.lower(&producer).unwrap().compile(&producer).unwrap();
        let executable_bytes = producer.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        cancellation.store(true, Ordering::Release);
        let restored_output =
            CompiledKernel::from_manifest(&verified, output.manifest(), output.artifact().bytes().to_vec()).unwrap();
        assert_eq!(restored_output.artifact().bytes(), output.artifact().bytes());
        assert_eq!(restored_output.arguments(), output.arguments());
        let runtime_session = Session::builder().client(client).build()?;
        let runtime = runtime_session.clone();
        let restored = runtime.deserialize_program(&executable_bytes).unwrap().unwrap();
        let executable = compiled.executable().with_compiled_program(Arc::new(restored));
        let mut arrays = case
            .input_types
            .iter()
            .zip(&case.inputs)
            .map(|(r#type, values)| {
                runtime_session
                    .with_device_mesh(mesh.clone())
                    .array(r#type.clone(), values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        if repeated_input {
            arrays[1] = arrays[0].clone();
        }
        let original = arrays[0].clone();
        let results = executable.execute(&runtime, arrays.into_iter().map(Value::Array).collect())?;
        let Value::Array(result) = &results[0] else { panic!("kernel output must be an array") };
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
            "cuTile {}: cubin={}; manifest={}; executable={}; Python disabled before reload",
            case.name,
            hex::encode(Sha256::digest(output.artifact().bytes())),
            hex::encode(Sha256::digest(output.manifest())),
            hex::encode(Sha256::digest(&executable_bytes)),
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
                execute_case(client, case, repeated_input).unwrap();
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
                    execute_case(client, case, false).unwrap();
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
        use ryft_core::{ProjectedValue, ShardMapOptions, Sharding, ShardingDimension, Typed, shard_map};
        use ryft_xla::tracing::ArrayTracer;
        use ryft_xla::{CompileTracer, CompiledFunction, Options, compile_with_options};

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
                let binding = KernelCompilerBinding::new(
                    compiler,
                    target,
                    CuTileOptions::default(),
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
                let session = Session::builder().client(client).build().unwrap();
                let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
                    |input: CompileTracer| {
                        shard_map(
                            |local: ArrayTracer| {
                                // The kernel is physical, and its call accepts the local shard as it is placed.
                                let physical = local.r#type().into_owned().with_sharding(None::<Sharding>).unwrap();
                                let definition: KernelDefinition = KernelDefinition::trace(
                                    KernelCallOperation::new(
                                        Grid::new(vec![]).unwrap(),
                                        vec![
                                            whole_array_parameter(physical, KernelParameterAccess::ReadWrite).unwrap(),
                                        ],
                                    )
                                    .unwrap(),
                                    |(references, _)| references[0].write(&references[0].read()?),
                                )
                                .unwrap();
                                let value = local.into_value();
                                let output = stage_kernel(value.context(), &definition, &[value.clone()])?.remove(0);
                                let ArrayIrType::Array(output_type) = output.r#type().into_owned() else {
                                    unreachable!("kernel results are arrays");
                                };
                                Ok(ProjectedValue::new(output, output_type))
                            },
                            input,
                            logical_mesh.clone(),
                            sharding.clone(),
                            sharding.clone(),
                            ShardMapOptions::new(),
                        )
                        .unwrap()
                    },
                    r#type.clone(),
                    &session,
                    Options::new(mesh.clone()).with_kernel_compiler(binding),
                )
                .unwrap();
                let expected = (0..17).map(|value| value as f32 - 8.0).collect::<Vec<_>>();
                let input = session
                    .with_device_mesh(mesh)
                    .array(r#type, expected.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let output = compiled.executable().execute(&session, input).unwrap();
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
                            DimensionVariable::new("checked", DimensionBounds::new(0, Some(2))?).into(),
                        )),
                        vec![],
                        &[integer],
                    )?;
                    references[0].write(&value)
                })
                .unwrap();
                let error = execute_case(
                    client,
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
#[cfg(any(feature = "mosaic-gpu", feature = "cutile", feature = "triton"))]
mod tuning {
    use std::env;
    use std::num::NonZeroUsize;
    #[cfg(feature = "cutile")]
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, CompilationTracer, Device, DeviceMesh, ExecutableFunction as CoreExecutableFunction, LogicalMesh,
        MeshAxis, MeshAxisType, StagedFunction as CoreStagedFunction, StatefulExecution,
    };
    use ryft_kernels::{KernelCompiler, KernelSchedule, VerifiedKernel};
    use ryft_xla::kernels::{
        KernelCompilerBinding, KernelExecutionFacts, KernelOutputEmbedding, KernelTarget, KernelTuner,
        KernelTuningBudget, KernelTuningError, KernelTuningRequest, KernelTuningRunner, stage_kernel,
    };
    use ryft_xla::{FromPjrt, Options, Session, Value};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// One immutable workload with ordinary compiled-function ownership; no alternate execution engine is introduced.
    struct Runner {
        /// Existing XLA session.
        session: Session,

        /// Fixed independent oracle and canonical input signature.
        case: KernelCase,

        /// Exact physical device placement used by compilation and invocation.
        mesh: DeviceMesh,

        /// Ordinary typed bindings for the finite requested schedule list.
        bindings: Vec<(KernelSchedule, KernelCompilerBinding)>,

        /// Prepared ordinary executable, replaced only between candidates.
        executable: Option<CoreExecutableFunction<Session, Vec<ArrayIrType>, Vec<ArrayIrType>>>,

        /// Fixed functional inputs reused across all samples.
        inputs: Vec<Value>,
    }

    impl KernelTuningRunner for Runner {
        fn prepare(&mut self, schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
            for input in &self.inputs {
                let Value::Array(input) = input else {
                    unreachable!();
                };
                input.block_until_ready().map_err(error)?;
            }
            let binding = self.bindings.iter().find(|(candidate, _)| candidate == schedule).unwrap().1.clone();
            let staged = CoreStagedFunction::<Session, Vec<ArrayIrType>, Vec<ArrayIrType>>::stage(
                |_, _, inputs: Vec<CompilationTracer<Session>>| {
                    Ok(stage_kernel(inputs[0].context(), &self.case.definition, &inputs)?)
                },
                vec![],
                self.case.input_types.iter().cloned().map(ArrayIrType::Array).collect(),
                Options::new(self.mesh.clone()).with_kernel_compiler(binding),
            )
            .map_err(error)?;
            // The existing lower/compile path performs real adapter admission before artifact lookup.
            let compiled = staged.lower(&self.session).map_err(error)?.compile(&self.session).map_err(error)?;
            self.executable = Some(compiled.executable().clone());
            Ok(())
        }

        fn execute(&mut self) -> StatefulExecution<(), KernelTuningError> {
            let result = (|| {
                let outputs = self
                    .executable
                    .as_ref()
                    .unwrap()
                    .execute_statefully_and_wait(&self.session, self.inputs.clone())
                    .map_err(error)?;
                let Value::Array(output) = &outputs[0] else {
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
            StatefulExecution::ready(result)
        }
    }

    /// Preserves concrete integration diagnostics in the tuner-owned error family.
    fn error(error: impl std::fmt::Display) -> KernelTuningError {
        KernelTuningError::Compiler { message: error.to_string() }
    }

    /// Measures two explicit schedule choices through the same real runtime and functional workload.
    fn measure<Compiler, Embedding>(
        client: &'static crate::Client<'static>,
        compiler: Compiler,
        target: Compiler::Target,
        options: Compiler::Options,
        embedding: Embedding,
        cancellation: &AtomicBool,
    ) where
        Compiler: 'static + Clone + Send + Sync + KernelCompiler<Error: 'static + Send + Sync>,
        Compiler::Target: 'static + Clone + Send + Sync + KernelTarget,
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
        let session = Session::builder().client(client).build().unwrap();
        let facts = KernelExecutionFacts::from_target(session.target().unwrap(), &mesh).unwrap();
        let candidates = vec![
            KernelSchedule::default().with_pipeline_stages(NonZeroUsize::new(1).unwrap()),
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
                let binding = KernelCompilerBinding::new(
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
        let inputs = case
            .input_types
            .iter()
            .zip(&case.inputs)
            .map(|(r#type, values)| {
                Value::Array(
                    session
                        .with_device_mesh(mesh.clone())
                        .array(r#type.clone(), values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
                        .unwrap(),
                )
            })
            .collect();
        let mut runner = Runner { session: session.clone(), case, mesh, bindings, executable: None, inputs };
        let result = KernelTuner::new(None).run(&request, &mut runner, cancellation).unwrap();
        assert_eq!(result.samples().iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
        assert!(result.best_candidate() < request.candidates().len());
        eprintln!(
            "tuning: candidate={} host-completion-readback-samples={:?}",
            result.best_candidate(),
            result.samples()
        );
    }

    #[cfg(feature = "triton")]
    #[test]
    fn test_triton_on_cuda() {
        if env::var("RYFT_PJRT_RUN_TRITON_KERNELS").ok().as_deref() != Some("1") {
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
                let compiler = ryft_triton::kernels::Compiler::new();
                measure(
                    client,
                    compiler,
                    ryft_triton::kernels::Target::Cuda { major: major.parse().unwrap(), minor: minor.parse().unwrap() },
                    ryft_triton::kernels::Options::default(),
                    ryft_xla::kernels::TritonEmbedding,
                    &AtomicBool::new(false),
                );
                executed = true;
            }
        });
        assert!(executed, "enabled Triton tuning did not execute CUDA");
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
                    client,
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
                    client,
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
            #[kernel]
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
    #[kernel]
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
            let rows = if iteration.is_multiple_of(2) {
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
                            client,
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
                    super::cutile::execute_case(client, descriptor.kernel(), false).unwrap();
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

#[cfg(feature = "triton")]
mod triton {
    use std::env;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use pretty_assertions::assert_eq;

    use ryft_core::{
        ArrayIrType, CompilationCacheDomain, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType,
        StagedFunction as CoreStagedFunction,
    };
    use ryft_kernels::{KernelCompiler, KernelSchedule};
    use ryft_triton::kernels::{Compiler, Options as TritonOptions, Target};
    use ryft_xla::kernels::{KernelCompilerBinding, TritonEmbedding, stage_kernel};
    use ryft_xla::{FromPjrt, Options, Session, Value};

    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// Reads the actual CUDA architecture for explicit compiler selection.
    fn device_target(client: &crate::Client<'_>) -> Target {
        let device = client.addressable_devices().unwrap().remove(0);
        let crate::Value::String(capability) = device.attribute("compute_capability").unwrap() else {
            panic!("missing CUDA compute capability")
        };
        let (major, minor) = capability.split_once('.').unwrap();
        Target::Cuda { major: major.parse().unwrap(), minor: minor.parse().unwrap() }
    }

    /// Compiles the unchanged portable definition, disables its compiler, and executes a fresh-session reload.
    fn execute_case(client: &'static crate::Client<'static>, case: KernelCase) {
        interpret_case(&case);
        let semantic = case.definition.semantic_key().unwrap();
        let cancellation = Arc::new(AtomicBool::new(false));
        let compiler = Compiler::new().with_cancellation(cancellation.clone());
        let binding = KernelCompilerBinding::new(
            compiler,
            device_target(client),
            TritonOptions::default(),
            KernelSchedule::default(),
            TritonEmbedding,
            1024,
        )
        .unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let producer = Session::builder().client(client).build().unwrap();
        let staged = CoreStagedFunction::<Session, Vec<ArrayIrType>, Vec<ArrayIrType>>::stage(
            |_, _, inputs: Vec<ryft_core::CompilationTracer<Session>>| {
                Ok(stage_kernel(inputs[0].context(), &case.definition, &inputs)?)
            },
            vec![],
            case.input_types.iter().cloned().map(ArrayIrType::Array).collect(),
            Options::new(mesh.clone()).with_kernel_compiler(binding),
        )
        .unwrap();
        let compiled = staged.lower(&producer).unwrap().compile(&producer).unwrap();
        let bytes = producer.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        cancellation.store(true, Ordering::Release);
        let session = Session::builder().client(client).build().unwrap();
        let runtime = session.clone();
        let restored = runtime.deserialize_program(&bytes).unwrap().unwrap();
        let executable = compiled.executable().with_compiled_program(Arc::new(restored));
        let inputs = case
            .input_types
            .iter()
            .zip(&case.inputs)
            .map(|(r#type, values)| {
                session
                    .with_device_mesh(mesh.clone())
                    .array(r#type.clone(), values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for _ in 0..3 {
            let outputs = executable.execute(&runtime, inputs.iter().cloned().map(Value::Array).collect()).unwrap();
            let Value::Array(output) = &outputs[0] else { panic!("expected array output") };
            let bytes = output
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let values = bytes
                .chunks_exact(size_of::<f32>())
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, case.expected, "{}", case.name);
        }
        for (input, expected) in inputs.iter().zip(&case.inputs) {
            let bytes = input
                .device_shard(device.id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(bytes, expected.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>());
        }
        assert_eq!(case.definition.semantic_key().unwrap(), semantic);
        eprintln!("Triton {}: exact output, unchanged inputs, compiler-disabled reload", case.name);
    }

    /// Exercises the production adapter's complete HSACO path without loading HIP or requiring an AMD device.
    #[test]
    fn test_compiler_on_rocm() {
        if env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        /// Uses a whole-array contraction without checked loop dimensions or assertion hostcalls.
        #[kernel]
        fn dot(
            #[input(data_type = F32, rank = 2)] left: &Array,
            #[input(data_type = F32, rank = 2)] right: &Array,
            #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]])] output: &mut Array,
        ) {
            output.store(left.load().dot(right.load()));
        }
        let compiler = Compiler::new();
        let vector = kernel_cases().remove(0).definition;
        let matrix = ArrayType::new_static(DataType::F32, [32, 32]);
        let dot = dot::definition(&matrix, &matrix).unwrap();
        for (architecture, definition) in
            [("gfx908", &vector), ("gfx90a", &vector), ("gfx942", &vector), ("gfx942", &dot)]
        {
            let verified = ryft_kernels::VerifiedKernel::new(definition, 1024).unwrap();
            let output = verified
                .compile(
                    &compiler,
                    &Target::Rocm { architecture: architecture.into() },
                    &TritonOptions::default(),
                    &KernelSchedule::default(),
                )
                .unwrap();
            let ryft_triton::kernels::Artifact::Rocm(artifact) = output.artifact() else {
                panic!("ROCm compiler must return HSACO")
            };
            assert_eq!(artifact.target(), architecture);
            assert_eq!(artifact.entry_name(), "ryft_kernel");
            assert_eq!(artifact.parameter_count(), 3);
            assert_eq!(artifact.launch_dimensions().block(), [256, 1, 1]);
            assert_eq!(output.semantic_key(), definition.semantic_key().unwrap());
            eprintln!("ROCm {architecture}: validated {} HSACO bytes; no hardware execution", artifact.image().len());
        }
    }

    #[test]
    fn test_unsupported_scratch() {
        let case = async_copy_case(&[67]);
        let verified = ryft_kernels::VerifiedKernel::new(&case.definition, 1024).unwrap();
        let compiler = Compiler::new();
        assert!(matches!(
            compiler.admit(&verified, &Target::Cuda { major: 12, minor: 1 }, &TritonOptions::default(),
                &KernelSchedule::default()),
            Err(ryft_kernels::KernelCompilationError::Unsupported { operation: "scratch", requested, .. })
                if requested == "operation is outside the portable Triton subset",
        ));
    }

    #[test]
    fn test_portable_kernels_on_cuda() {
        if env::var("RYFT_PJRT_RUN_TRITON_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                for case in kernel_cases()
                    .into_iter()
                    .filter(|case| case.name != "async_copy")
                    .chain([precision_case(), batched_case()])
                {
                    execute_case(client, case);
                }
                executed = true;
            }
        });
        assert!(executed, "enabled Triton qualification did not execute CUDA");
    }

    #[test]
    fn test_aot_on_cuda() {
        if env::var("RYFT_PJRT_RUN_TRITON_KERNELS").ok().as_deref() != Some("1") {
            return;
        }
        let mut executed = false;
        test_for_each_platform!(|_plugin, client, platform| {
            if matches!(platform, TestPlatform::Cuda13) {
                let compiler = Compiler::new();
                let target = device_target(client);
                let options = TritonOptions::default();
                let schedule = KernelSchedule::default();
                let configuration = compiler.configuration_key(&target, &options, &schedule).unwrap();
                let deployment = Compiler::from_configuration(&configuration).unwrap();
                let binding = KernelCompilerBinding::new(
                    compiler,
                    target.clone(),
                    options.clone(),
                    schedule.clone(),
                    TritonEmbedding,
                    1024,
                )
                .unwrap();
                let deployment_binding =
                    KernelCompilerBinding::new(deployment, target, options, schedule, TritonEmbedding, 1024).unwrap();
                assert_eq!(binding.configuration(), deployment_binding.configuration());
                execute_aot_case(&_plugin, client, binding, deployment_binding, "triton");
                executed = true;
            }
        });
        assert!(executed, "enabled Triton AOT qualification did not execute CUDA");
    }
}

/// Records which macro kernels each compiler adapter admits and lowers for a CUDA 12.1 target, as far as this is
/// possible without a GPU, a native compiler invocation, or the cuTile Python compiler.
#[cfg(all(feature = "cutile", feature = "mosaic-gpu", feature = "triton"))]
mod adapters {
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;

    use ryft_cuda::kernels::cutile::{Compiler as CuTileCompiler, Options as CuTileOptions, Target as CuTileTarget};
    use ryft_kernels::{
        KernelCompilationError, KernelCompiler, KernelInitializationError, KernelSchedule, VerifiedKernel,
    };
    use ryft_mosaic::kernels::gpu::{Compiler as MosaicCompiler, Options as MosaicOptions, Target as MosaicTarget};
    use ryft_triton::kernels::{Compiler as TritonCompiler, Options as TritonOptions, Target as TritonTarget};

    use super::*;

    /// Absolute path of a Python executable that does not exist. cuTile admission lowers the kernel to Python source
    /// before it probes the compiler installation, so failing to launch this executable means that lowering succeeded.
    const MISSING_PYTHON: &str = "/nonexistent/ryft/python";

    /// Admits a kernel through Mosaic GPU, builds and verifies its MLIR module, and serializes the module into the
    /// bytes that XLA compiles. Returns `lowers` when every step succeeds and the debug rendering of the first error
    /// otherwise.
    fn mosaic(kernel: &VerifiedKernel<'_>) -> String {
        let target = MosaicTarget::new(12, 1).unwrap();
        let options = MosaicOptions::default();
        let schedule = KernelSchedule::default();
        if let Err(error) = MosaicCompiler.admit(kernel, &target, &options, &schedule) {
            return format!("{error:?}");
        }
        let context = ryft_mlir::Context::new();
        match MosaicCompiler.module(&context, kernel, &target, &options, &schedule) {
            Ok(module) => assert_eq!(module.verify(), Ok(true)),
            Err(error) => return format!("module: {error:?}"),
        }
        match MosaicCompiler.compile(kernel, &target, &options, &schedule) {
            Ok(_) => "lowers".into(),
            Err(error) => format!("compile: {error:?}"),
        }
    }

    /// Admits a kernel through Triton, which lowers it to a verified Triton MLIR module and only then checks that the
    /// linked native archive includes the CUDA compiler. Returns `lowers` when admission succeeds or fails only because
    /// that compiler is not linked (e.g., on hosts without CUDA), and the debug rendering of its error otherwise.
    fn triton(kernel: &VerifiedKernel<'_>) -> String {
        let target = TritonTarget::Cuda { major: 12, minor: 1 };
        match TritonCompiler::new().admit(kernel, &target, &TritonOptions::default(), &KernelSchedule::default()) {
            Ok(()) => "lowers".into(),
            Err(KernelCompilationError::Unavailable { message })
                if message == "linked native archive does not include the requested Triton compiler target" =>
            {
                "lowers".into()
            }
            Err(error) => format!("{error:?}"),
        }
    }

    /// Admits a kernel through cuTile with [`MISSING_PYTHON`]. Returns `lowers` when admission fails only because that
    /// executable cannot be launched, and the debug rendering of its error otherwise.
    fn cutile(kernel: &VerifiedKernel<'_>) -> String {
        let target = CuTileTarget::new(12, 1).unwrap();
        let compiler = CuTileCompiler::new(PathBuf::from(MISSING_PYTHON));
        match compiler.admit(kernel, &target, &CuTileOptions::default(), &KernelSchedule::default()) {
            Ok(()) => "lowers".into(),
            Err(KernelCompilationError::Unavailable { message })
                if message == format!("`{MISSING_PYTHON}`: No such file or directory (os error 2)") =>
            {
                "lowers".into()
            }
            Err(error) => format!("{error:?}"),
        }
    }

    /// Builds the definition of every macro kernel of this file that is reachable from this module, for small
    /// argument types that the host interpretation tests also use.
    fn definitions() -> Vec<(&'static str, KernelDefinition)> {
        let vector = |extent: usize| ArrayType::new_static(DataType::F32, [extent]);
        let matrix = |rows: usize, columns: usize| ArrayType::new_static(DataType::F32, [rows, columns]);
        vec![
            ("vector_add", vector_add::definition(&vector(1003), &vector(1003)).unwrap()),
            ("sum", sum::definition(&vector(257)).unwrap()),
            ("matmul", matmul::definition(&matrix(33, 35), &matrix(35, 34)).unwrap()),
            ("blocked_matmul", blocked_matmul::definition(&matrix(64, 96), &matrix(96, 32)).unwrap()),
            ("masked_blocked_matmul", masked_blocked_matmul::definition(&matrix(33, 35), &matrix(35, 34)).unwrap()),
            ("sum_and_difference", sum_and_difference::definition(&vector(256), &vector(256)).unwrap()),
            ("add_in_place", add_in_place::definition(&vector(256), &vector(256)).unwrap()),
            ("relu_in_place", relu_in_place::definition(&vector(256)).unwrap()),
            ("double_in_place", double_in_place::definition(&vector(300)).unwrap()),
            ("keep_weighted", keep_weighted::definition(&vector(256), &vector(256)).unwrap()),
            ("extract_negatives", extract_negatives::definition(&vector(256)).unwrap()),
            ("scratch_copy", scratch_copy::definition(&vector(67)).unwrap()),
            ("async_copy", async_copy::definition(&vector(67)).unwrap()),
            (
                "gather_block",
                gather_block::definition(&vector(512), &ArrayType::scalar(DataType::I32))
                    .unwrap()
                    .specialize_prefetch(&[Array::scalar(2i32).unwrap()])
                    .unwrap(),
            ),
            ("reverse_blocks", reverse_blocks::definition(&vector(384)).unwrap()),
            ("parallel_sum", parallel_sum::definition(&vector(384), &ArrayType::scalar(DataType::F32)).unwrap()),
            ("leading_columns", leading_columns::definition(&matrix(3, 4)).unwrap()),
            ("nonpositive_exp_sum", nonpositive_exp_sum::definition(&vector(5)).unwrap()),
        ]
    }

    #[test]
    fn test_adapter_admission() {
        // Verification requires every prefetched scalar to be specialized into a constant first, so no adapter
        // receives a runtime prefetch, and `definitions` specializes `gather_block` to the block index 2.
        let definition =
            gather_block::definition(&ArrayType::new_static(DataType::F32, [512]), &ArrayType::scalar(DataType::I32))
                .unwrap();
        assert!(matches!(
            VerifiedKernel::new(&definition, 1024),
            Err(KernelInitializationError::UnsupportedLaunch { boundary: "unspecialized scalar prefetch" }),
        ));

        // Columns are Mosaic GPU, Triton, and cuTile, and every rejection is the debug rendering of its error.
        let lowers = || "lowers".to_owned();
        let mosaic_unsupported = |operation: &str| {
            format!(
                "Unsupported {{ owner: \"ryft_mosaic::kernels::gpu\", operation: {operation:?}, \
                 requested: \"operation has no baseline GPU lowering\", capability: \"Mosaic GPU lowering\" }}",
            )
        };
        let triton_unsupported = |operation: &str, requested: &str| {
            format!(
                "Unsupported {{ owner: \"ryft_triton::kernels\", operation: {operation:?}, requested: {requested:?}, \
                 capability: \"direct Triton lowering\" }}",
            )
        };
        let cutile_unsupported = |operation: &str, requested: &str| {
            format!(
                "Unsupported {{ owner: \"ryft_cuda::kernels::cutile\", operation: {operation:?}, \
                 requested: {requested:?}, capability: \"cuTile source lowering\" }}",
            )
        };
        // Triton orders no global memory access across the threads of one program without a CTA barrier, which
        // ryft-mlir does not expose. It therefore keeps the carried output blocks of the blocked products in SSA
        // values that it stores once, and it rejects the old values of the swap of `extract_negatives` because they
        // are read from a read-write parameter and reach the store of another one.
        let expected = vec![
            ("vector_add", [lowers(), lowers(), lowers()]),
            ("sum", [lowers(), lowers(), lowers()]),
            ("matmul", [lowers(), lowers(), lowers()]),
            ("blocked_matmul", [lowers(), lowers(), lowers()]),
            ("masked_blocked_matmul", [lowers(), lowers(), lowers()]),
            ("sum_and_difference", [lowers(), lowers(), lowers()]),
            ("add_in_place", [lowers(), lowers(), lowers()]),
            ("relu_in_place", [lowers(), lowers(), lowers()]),
            ("double_in_place", [lowers(), lowers(), lowers()]),
            ("keep_weighted", [lowers(), lowers(), lowers()]),
            (
                "extract_negatives",
                [
                    lowers(),
                    triton_unsupported(
                        "reference_write",
                        "values read from a read-write parameter may reach memory only elementwise through its own \
                         write",
                    ),
                    lowers(),
                ],
            ),
            (
                "scratch_copy",
                [
                    lowers(),
                    triton_unsupported("scratch", "operation is outside the portable Triton subset"),
                    cutile_unsupported("scratch", "operation is outside the portable cuTile subset"),
                ],
            ),
            (
                "async_copy",
                [
                    lowers(),
                    triton_unsupported("scratch", "operation is outside the portable Triton subset"),
                    cutile_unsupported("scratch", "operation is outside the portable cuTile subset"),
                ],
            ),
            ("gather_block", [lowers(), triton_unsupported("constant", "unsupported constant element type"), lowers()]),
            ("reverse_blocks", [lowers(), lowers(), lowers()]),
            (
                "parallel_sum",
                [
                    mosaic_unsupported("reference_atomic_add_update"),
                    lowers(),
                    cutile_unsupported(
                        "reference_atomic_add_update",
                        "operation is outside the portable cuTile subset",
                    ),
                ],
            ),
            ("leading_columns", [lowers(), lowers(), lowers()]),
            (
                "nonpositive_exp_sum",
                [
                    lowers(),
                    triton_unsupported("exp", "array operation or geometry is outside the portable Triton subset"),
                    lowers(),
                ],
            ),
        ];
        let actual = definitions()
            .iter()
            .map(|(name, definition)| {
                let kernel = VerifiedKernel::new(definition, 1024).unwrap();
                (*name, [mosaic(&kernel), triton(&kernel), cutile(&kernel)])
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
