//! Tests for the `#[kernel]` attribute through a renamed `ryft-kernels` dependency and through the `ryft` facade,
//! covering the generated callables, their definitions, and the diagnostics of unsupported kernels.

extern crate ryft_core as core_alias;
extern crate ryft_kernels as kernels_alias;

use indoc::indoc;
use pretty_assertions::assert_eq;

use core_alias::{
    Abs, AddOperation, Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, Atan2, Ceil,
    ComparisonDirection, Condition, Context, ConvertElementType, Cos, DataType, Erf, Exp, ExpM1, Floor, Ln1p, Log,
    LogAddExp, LogicalMesh, Logistic, Max, MeshAxis, MeshAxisType, Min, Neg, NestedTracingContext, Pow, ProjectedValue,
    Reduce, ReductionKind, ReferenceAtomicAddUpdate, ReferenceRead, ReferenceWrite, Rem, Round, Rsqrt, Sharding, Sign,
    Sin, Sqrt, Tan, Tanh, Tracer, TracingContext, TypeError,
};
use kernels_alias::{
    BoundaryPolicy, Grid, GridDimension, GridExecution, KernelDefinition, KernelError, KernelOperation,
    KernelParameterAccess, MappingExpression, ParameterWindow, async_copy, compare, compare_dimension, dot, for_loop,
    full, mapped_call, masked_load_or, ones, scratch, select, tile_load, tile_store, wait, window_mask, zeros,
};

/// Adds two vectors. This is the only kernel that names its namespace with a leading `::`.
#[kernels_alias::kernel(crate = "::kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn add(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
) {
    let first = left.load();
    let second = right.load();
    output.store(first + second);
}

/// Multiplies two matrices using a whole-array kernel body.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[1] == right.shape()[0])]
fn matmul(
    #[input(data_type = F32, rank = 2)] left: &Array,
    #[input(data_type = F32, rank = 2)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]])] output: &mut Array,
) {
    output.store(left.load().dot(right.load()));
}

/// Multiplies two matrices using padded tile loads on a parallel grid over the output tiles.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[1] == right.shape()[0])]
fn tiled_matmul(
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

/// Subtracts the tiles of two vectors, with the tiled output declared before both of its inputs.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn tiled_difference_output_first(
    #[output(data_type = F32, shape = [left.shape()[0]], tile = [2], boundary = masked)] output: &mut Array,
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
) {
    let [block] = output.tile_index();
    let left_tiles = left.tiles([2]).pad(0.0);
    let right_tiles = right.tiles([2]).pad(0.0);
    output.store(left_tiles.load([block]) - right_tiles.load([block]));
}

/// Subtracts the tiles of two vectors, with the tiled output declared between its inputs.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn tiled_difference_output_middle(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]], tile = [2], boundary = masked)] output: &mut Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
) {
    let [block] = output.tile_index();
    let left_tiles = left.tiles([2]).pad(0.0);
    let right_tiles = right.tiles([2]).pad(0.0);
    output.store(left_tiles.load([block]) - right_tiles.load([block]));
}

/// Subtracts the tiles of two vectors, with the tiled output declared after both of its inputs.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn tiled_difference_output_last(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]], tile = [2], boundary = masked)] output: &mut Array,
) {
    let [block] = output.tile_index();
    let left_tiles = left.tiles([2]).pad(0.0);
    let right_tiles = right.tiles([2]).pad(0.0);
    output.store(left_tiles.load([block]) - right_tiles.load([block]));
}

/// Constructs a vector without inputs. Its `#[cfg_attr]` attribute applies only to the generated callable.
#[kernels_alias::kernel(crate = "kernels_alias")]
#[cfg_attr(all(), inline)]
fn zero(#[output(data_type = F32, shape = [2])] output: &mut Array) {
    let value = zeros::<f32>([2]);
    output.store(value);
}

/// Reduces a vector through the canonical sum operation.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn sum(#[input(data_type = F32, rank = 1)] input: &Array, #[output(data_type = F32, shape = [])] output: &mut Array) {
    output.store(input.load().sum([0]));
}

/// Carries an accumulator through a loop whose bound depends on the input shape.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn accumulate(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let mut accumulator = zeros::<f32>([input.shape()[0]]);
    for depth in 0..input.shape()[0].div_ceil(1) {
        accumulator += input.load();
    }
    output.store(accumulator);
}

/// Selects additive or subtractive updates inside a staged loop.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn conditional_accumulate(
    #[input(data_type = Boolean, rank = 0)] flag: &Array,
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let mut accumulator = zeros::<f32>([input.shape()[0]]);
    for depth in 0..3 {
        if flag.load() {
            accumulator += input.load();
        } else {
            accumulator -= input.load();
        }
    }
    output.store(accumulator);
}

/// Multiplies a vector by twice itself through a compound assignment whose right operand is a binary expression.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn compound_assignment(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let mut value = input.load();
    value *= input.load() + input.load();
    output.store(value);
}

/// Kernel whose paths resolve relative to its module: its namespace is a `use` alias and its element type is a type
/// alias, neither of which is visible from the generated definition module unless that module imports its parent.
mod relative {
    use kernels_alias as kernels;

    type Element = f32;

    /// Constructs a vector of zeros with the element type `Element`.
    #[kernels::kernel(crate = "kernels")]
    pub(super) fn zero(#[output(data_type = F32, shape = [3])] output: &mut Array) {
        output.store(zeros::<Element>([3]));
    }
}

const LOCATED_SOURCE_LINE: u32 = line!() + 7;
/// Copies a vector, recording the source location of its store.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn located(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    output.store(input.load());
}

/// Multiplies two matrices whose extents are multiples of 2, accumulating each output block over the sequential grid
/// axis `k` after zeroing it at the first step.
#[kernels_alias::kernel(
    crate = "kernels_alias",
    grid(
        i = left.shape()[0].div_ceil(2),
        j = right.shape()[1].div_ceil(2),
        k = sequential(left.shape()[1].div_ceil(2)),
    ),
    requires = left.shape()[1] == right.shape()[0],
)]
fn carried_matmul(
    #[input(data_type = F32, rank = 2, block = [2, 2], index = [i, k])] left: &Array,
    #[input(data_type = F32, rank = 2, block = [2, 2], index = [k, j])] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], block = [2, 2], index = [i, j])]
    output: &mut Array,
) {
    if k == 0 {
        output.store(zeros::<f32>([2, 2]));
    }
    output.accumulate(left.load().dot(right.load()));
}

/// Multiplies two matrices of any shape like [`carried_matmul`], with masked edge blocks whose padding is zero.
#[kernels_alias::kernel(
    crate = "kernels_alias",
    grid(
        i = parallel(left.shape()[0].div_ceil(2)),
        j = right.shape()[1].div_ceil(2),
        k = sequential(left.shape()[1].div_ceil(2)),
    ),
    requires = left.shape()[1] == right.shape()[0],
)]
fn masked_carried_matmul(
    #[input(data_type = F32, rank = 2, block = [2, 2], index = [i, k], boundary = masked)] left: &Array,
    #[input(data_type = F32, rank = 2, block = [2, 2], index = [k, j], boundary = masked)] right: &Array,
    #[output(
        data_type = F32,
        shape = [left.shape()[0], right.shape()[1]],
        block = [2, 2],
        index = [i, j],
        boundary = masked,
    )]
    output: &mut Array,
) {
    if k == 0 {
        output.store(zeros::<f32>([2, 2]));
    }
    output.accumulate(left.load_or(0.0).dot(right.load_or(0.0)));
}

/// Converts the elements of a vector to integers and keeps the elements above `2.5`, in two outputs.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = input.shape()[0].div_ceil(2)))]
fn split(
    #[input(data_type = F32, rank = 1, block = [2], index = [i])] input: &Array,
    #[output(data_type = I32, shape = [input.shape()[0]], block = [2], index = [i])] converted: &mut Array,
    #[output(data_type = F32, shape = [input.shape()[0]], block = [2], index = [i])] thresholded: &mut Array,
) {
    let value = input.load();
    converted.store(value as i32);
    let threshold = full([2], 2.5f32);
    let zero = zeros::<f32>([2]);
    thresholded.store(select(value > threshold, value, zero));
}

/// Increments each element of `counts` in place and atomically adds the number of elements to `total`.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = counts.shape()[0]))]
fn count(
    #[inout(data_type = I32, rank = 1, block = [1], index = [i])] counts: &mut Array,
    #[inout(data_type = I32, rank = 0)] total: &mut Array,
) {
    counts.accumulate(ones::<i32>([1]));
    total.atomic_accumulate(ones::<i32>([]));
}

/// Negates the negative elements of a vector in place, in masked blocks whose padding is never written.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = values.shape()[0].div_ceil(2)))]
fn absolute(#[inout(data_type = I32, rank = 1, block = [2], index = [i], boundary = masked)] values: &mut Array) {
    let value = values.load_or(0);
    values.store_where(-value, value < 0);
}

/// Returns which elements of the masked blocks of a vector of 3 elements are valid.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = 2))]
fn validity(
    #[input(data_type = I32, rank = 1, block = [2], index = [i], boundary = masked)] input: &Array,
    #[output(data_type = Boolean, shape = [4], block = [2], index = [i])] output: &mut Array,
) {
    output.store(input.window_mask());
}

/// Loads the elements of a vector of 2 elements that a computed mask selects into a vector of 4 elements, filling the
/// other elements and the padding of the masked window with `7`.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = 1))]
fn select_elements(
    #[input(data_type = I32, rank = 1, block = [4], index = [i], boundary = masked)] input: &Array,
    #[output(data_type = I32, shape = [4])] output: &mut Array,
) {
    output.store(input.load_where(iota::<i32>(4) != 1, 7));
}

/// Swaps the elements of the masked blocks of `values` that a computed mask selects with `9 + index`, and returns the
/// previous elements with `-1` for the elements that were not swapped.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = values.shape()[0].div_ceil(4)))]
fn swap(
    #[inout(data_type = I32, rank = 1, block = [4], index = [i], boundary = masked)] values: &mut Array,
    #[output(data_type = I32, shape = [values.shape()[0]], block = [4], index = [i], boundary = masked)]
    previous: &mut Array,
) {
    let indices = iota::<i32>(4);
    let mask = indices != 1;
    previous.store(values.swap_where(indices + full([4], 9), mask, -1));
}

/// Copies a scalar through scratch storage.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn scratch_copy(
    #[input(data_type = I32, rank = 0)] input: &Array,
    #[output(data_type = I32, shape = [])] output: &mut Array,
) {
    let storage = scratch::<i32>([]);
    storage.store(input.load());
    output.store(storage.load());
}

/// Copies a vector asynchronously through aligned scratch storage.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn staged_copy(
    #[input(data_type = I32, rank = 1)] input: &Array,
    #[output(data_type = I32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let staging = scratch_aligned::<i32>([input.shape()[0]], 16);
    let token = input.copy_to(staging);
    token.wait();
    output.store(staging.load());
}

/// Gathers the block of 2 elements of `input` whose block index is the prefetched `offset`.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = 1))]
fn gather(
    #[input(data_type = I32, rank = 1, block = [2], index = [offset])] input: &Array,
    #[prefetch(data_type = I32)] offset: &Array,
    #[output(data_type = I32, shape = [2])] output: &mut Array,
) {
    output.store(input.load());
}

/// Reverses the blocks of 2 elements of a vector of 6 elements through element starts that run backwards.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = 3))]
fn reverse_blocks(
    #[input(data_type = I32, rank = 1, block = [2], start = [4 - i * 2])] input: &Array,
    #[output(data_type = I32, shape = [6], block = [2], index = [i])] output: &mut Array,
) {
    output.store(input.load());
}

/// Copies the last two columns of a matrix through a static slice.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn columns(
    #[input(data_type = I32, rank = 2)] input: &Array,
    #[output(data_type = I32, shape = [input.shape()[0], 2])] output: &mut Array,
) {
    let tail = input.slice([.., 1..input.shape()[1]]);
    output.store(tail.load());
}

/// Adds a vector into the first elements of an `inout` vector through a slice.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn add_prefix(
    #[input(data_type = I32, rank = 1)] input: &Array,
    #[inout(data_type = I32, rank = 1)] output: &mut Array,
) {
    output.slice([0..input.shape()[0]]).accumulate(input.load());
}

/// Loads the second element of `first` through a slice of a local that is rebound to `second` after the slice is
/// declared, followed by the second element of `second` through the rebound local.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn rebound_slice_load(
    #[input(data_type = I32, rank = 1)] first: &Array,
    #[input(data_type = I32, rank = 1)] second: &Array,
    #[output(data_type = I32, shape = [2])] output: &mut Array,
) {
    let mut source = first;
    let element = source.slice([1..2]);
    source = second;
    output.slice([0..1]).store(element.load());
    output.slice([1..2]).store(source.slice([1..2]).load());
}

/// Stores `7` into the first element of `values` through a slice of a local that is rebound to scratch storage after
/// the slice is declared, then increments that element through `values`. `observed` records the element through
/// `values` after the store and through the slice after the increment.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn rebound_slice_store(
    #[inout(data_type = I32, rank = 1)] values: &mut Array,
    #[output(data_type = I32, shape = [2])] observed: &mut Array,
) {
    let mut target = values;
    let head = target.slice([0..1]);
    target = scratch::<i32>([2]);
    target.store(zeros::<i32>([2]));
    head.store(full([1], 7i32));
    observed.slice([0..1]).store(values.slice([0..1]).load());
    values.slice([0..1]).accumulate(ones::<i32>([1]));
    observed.slice([1..2]).store(head.load());
}

/// Sums the second elements of `first` and the first elements of `second` twice in a staged loop, through a slice of
/// a local that is rebound to `second` after the slice is declared and through a slice of the rebound local that the
/// loop body declares. After the loop, loads through the first slice and through the rebound local.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn rebound_slice_loop(
    #[input(data_type = I32, rank = 1)] first: &Array,
    #[input(data_type = I32, rank = 1)] second: &Array,
    #[output(data_type = I32, shape = [3])] output: &mut Array,
) {
    let mut source = first;
    let element = source.slice([1..2]);
    source = second;
    let mut total = zeros::<i32>([1]);
    for step in 0..2 {
        let inner = source.slice([0..1]);
        total += element.load() + inner.load();
    }
    output.slice([0..1]).store(total);
    output.slice([1..2]).store(element.load());
    output.slice([2..3]).store(source.slice([1..2]).load());
}

/// Loads the second element of `first` through a slice of a local that is rebound to `second` after the slice is
/// declared, inside the `true` branch of a staged condition and after it. The `false` branch loads the second element
/// of `second` through the rebound local instead.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn rebound_slice_condition(
    #[input(data_type = Boolean, rank = 0)] flag: &Array,
    #[input(data_type = I32, rank = 1)] first: &Array,
    #[input(data_type = I32, rank = 1)] second: &Array,
    #[output(data_type = I32, shape = [3])] output: &mut Array,
) {
    let mut source = first;
    let element = source.slice([1..2]);
    source = second;
    let mut inside = zeros::<i32>([1]);
    if flag.load() {
        inside = element.load();
    } else {
        inside = source.slice([1..2]).load();
    }
    output.slice([0..1]).store(inside);
    output.slice([1..2]).store(element.load());
    output.slice([2..3]).store(source.slice([1..2]).load());
}

/// Loads the first element of `first` through a slice of a local that is declared with a raw identifier and sliced
/// through its ordinary spelling, and through a slice of a mutable local that is declared with an ordinary identifier,
/// sliced through its raw spelling, and rebound to `second` through its raw spelling after the slice is declared,
/// followed by the first element of `second` through the rebound local.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn raw_identifier_slices(
    #[input(data_type = I32, rank = 1)] first: &Array,
    #[input(data_type = I32, rank = 1)] second: &Array,
    #[output(data_type = I32, shape = [3])] output: &mut Array,
) {
    let r#source = first;
    let element = source.slice([0..1]);
    let mut target = first;
    let saved = r#target.slice([0..1]);
    r#target = second;
    output.slice([0..1]).store(element.load());
    output.slice([1..2]).store(saved.load());
    output.slice([2..3]).store(target.slice([0..1]).load());
}

/// Copies a vector.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn copy(
    #[input(data_type = I32, rank = 1)] input: &Array,
    #[output(data_type = I32, shape = [input.shape()[0]])] output: &mut Array,
) {
    output.store(input.load());
}

/// Combines constant vectors of 3 elements.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn constants(#[output(data_type = I32, shape = [3])] output: &mut Array) {
    output.store(iota::<i32>(3) + ones::<i32>([3]) * full::<i32>([3], 2));
}

/// Applies elementwise logic to two Boolean vectors.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn logic(
    #[input(data_type = Boolean, rank = 1)] left: &Array,
    #[input(data_type = Boolean, rank = 1)] right: &Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] output: &mut Array,
) {
    output.store((left.load() & right.load()) ^ !(left.load() | right.load()));
}

/// Negates the remainder of two integer vectors.
#[kernels_alias::kernel(crate = "kernels_alias")]
fn negated_remainder(
    #[input(data_type = I32, rank = 1)] left: &Array,
    #[input(data_type = I32, rank = 1)] right: &Array,
    #[output(data_type = I32, shape = [left.shape()[0]])] output: &mut Array,
) {
    output.store(-(left.load() % right.load()));
}

/// Compares two vectors with each comparison operator, comparing with an integer literal on either side in two cases.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn comparisons(
    #[input(data_type = I32, rank = 1)] left: &Array,
    #[input(data_type = I32, rank = 1)] right: &Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] equal: &mut Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] not_equal: &mut Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] less: &mut Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] less_or_equal: &mut Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] greater: &mut Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] greater_or_equal: &mut Array,
) {
    equal.store(left.load() == right.load());
    not_equal.store(left.load() != right.load());
    less.store(left.load() < right.load());
    less_or_equal.store(left.load() <= 2);
    greater.store(left.load() > right.load());
    greater_or_equal.store(2 >= right.load());
}

/// Keeps the elements of `left` that are less than those of `right` and replaces the others with the literal `9`.
#[kernels_alias::kernel(crate = "kernels_alias", requires = left.shape()[0] == right.shape()[0])]
fn select_less(
    #[input(data_type = I32, rank = 1)] left: &Array,
    #[input(data_type = I32, rank = 1)] right: &Array,
    #[output(data_type = I32, shape = [left.shape()[0]])] output: &mut Array,
) {
    output.store(select(left.load() < right.load(), left.load(), 9));
}

/// Compares the grid coordinate `i` with constants, with the coordinate on either side, and records each comparison
/// of every program as `1` or `0` in its element of one output per comparison operator.
#[kernels_alias::kernel(crate = "kernels_alias", grid(i = 4))]
fn coordinate_comparisons(
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] equal: &mut Array,
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] not_equal: &mut Array,
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] less: &mut Array,
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] less_or_equal: &mut Array,
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] greater: &mut Array,
    #[output(data_type = I32, shape = [4], block = [1], index = [i])] greater_or_equal: &mut Array,
) {
    equal.store(select(i == 1, ones::<i32>([1]), zeros::<i32>([1])));
    not_equal.store(select(1 != i, ones::<i32>([1]), zeros::<i32>([1])));
    less.store(select(i < 2, ones::<i32>([1]), zeros::<i32>([1])));
    less_or_equal.store(select(2 <= i, ones::<i32>([1]), zeros::<i32>([1])));
    greater.store(select(i > 2, ones::<i32>([1]), zeros::<i32>([1])));
    greater_or_equal.store(select(2 >= i, ones::<i32>([1]), zeros::<i32>([1])));
}

/// Kernels that apply one capability function of kernel body values to vectors, each named after that function.
mod capabilities {
    /// Defines one kernel per unary elementwise capability function.
    macro_rules! unary {
        ($($function:ident),* $(,)?) => {
            $(
                /// Applies one unary elementwise capability function.
                #[kernels_alias::kernel(crate = "kernels_alias")]
                pub(super) fn $function(
                    #[input(data_type = F32, rank = 1)] input: &Array,
                    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
                ) {
                    output.store(input.load().$function());
                }
            )*
        };
    }

    /// Defines one kernel per binary elementwise capability function.
    macro_rules! binary {
        ($($function:ident),* $(,)?) => {
            $(
                /// Applies one binary elementwise capability function.
                #[kernels_alias::kernel(crate = "kernels_alias")]
                pub(super) fn $function(
                    #[input(data_type = F32, rank = 1)] left: &Array,
                    #[input(data_type = F32, rank = 1)] right: &Array,
                    #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
                ) {
                    output.store(left.load().$function(right.load()));
                }
            )*
        };
    }

    /// Defines one kernel per reduction capability function over a vector of the provided data type.
    macro_rules! reduction {
        ($data_type:ident: $($function:ident),* $(,)?) => {
            $(
                /// Reduces a vector through one reduction capability function.
                #[kernels_alias::kernel(crate = "kernels_alias")]
                pub(super) fn $function(
                    #[input(data_type = $data_type, rank = 1)] input: &Array,
                    #[output(data_type = $data_type, shape = [])] output: &mut Array,
                ) {
                    output.store(input.load().$function([0]));
                }
            )*
        };
    }

    unary!(abs, ceil, cos, erf, exp, exp_m1, floor, ln_1p, log, logistic, round, rsqrt, sign, sin, sqrt, tan, tanh);
    binary!(atan2, log_add_exp, max, min, pow);
    reduction!(F32: reduce_sum, reduce_product, reduce_mean, reduce_log_sum_exp, reduce_max, reduce_min);
    reduction!(Boolean: reduce_any, reduce_all);
}

/// Returns the operands of a `rows` by `depth` by `columns` matrix multiplication and its expected result.
fn matmul_case(rows: usize, depth: usize, columns: usize) -> (Array, Array, Array) {
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
        Array::from_elements(ArrayType::new_static(DataType::F32, [rows, depth]), &left).unwrap(),
        Array::from_elements(ArrayType::new_static(DataType::F32, [depth, columns]), &right).unwrap(),
        Array::from_elements(ArrayType::new_static(DataType::F32, [rows, columns]), &product).unwrap(),
    )
}

/// Builds the definition of a matrix product of `f32[m, k]` and `f32[k, n]` inputs with blocks of 2 by 2 elements over
/// the grid `(i, j, k)`, where only `k` is sequential, through `mapped_call` and the authoring functions, in the form
/// that [`carried_matmul`] and [`masked_carried_matmul`] generate. Like every generated condition, the first step
/// condition carries the references and the grid coordinates.
fn carried_matmul_definition(shape: [usize; 3], boundary: BoundaryPolicy) -> KernelDefinition {
    type BranchTracer = Tracer<NestedTracingContext<TracingContext<ArrayIrValue<Array>, KernelOperation>>>;
    let [m, k, n] = shape;
    let window = |axes: [usize; 2]| ParameterWindow::BlockIndices {
        block_shape: vec![2, 2],
        boundary,
        indices: axes.map(MappingExpression::GridCoordinate).to_vec(),
    };
    let grid = Grid::new(vec![
        GridDimension::new(m.div_ceil(2).into(), GridExecution::Parallel).with_name("i").unwrap(),
        GridDimension::new(n.div_ceil(2).into(), GridExecution::Parallel).with_name("j").unwrap(),
        GridDimension::new(k.div_ceil(2).into(), GridExecution::Sequential).with_name("k").unwrap(),
    ])
    .unwrap();
    let operation = mapped_call(
        grid,
        vec![
            (ArrayType::new_static(DataType::F32, [m, k]), KernelParameterAccess::ReadOnly, window([0, 2])),
            (ArrayType::new_static(DataType::F32, [k, n]), KernelParameterAccess::ReadOnly, window([2, 1])),
            (ArrayType::new_static(DataType::F32, [m, n]), KernelParameterAccess::WriteOnly, window([0, 1])),
        ],
        vec![],
    )
    .unwrap();
    KernelDefinition::trace(operation, |(references, coordinates)| {
        let context = references[0].context();
        let first = compare_dimension(context, &coordinates[2], ComparisonDirection::Equal, 0)?;
        first.condition(
            |values: Vec<BranchTracer>| {
                values[2].write(&zeros::<f32, _>(values[0].context(), [2, 2])?)?;
                Ok(values)
            },
            |values: Vec<BranchTracer>| Ok(values),
            references.iter().chain(&coordinates).cloned().collect::<Vec<_>>(),
        )?;
        let tiles = references[..2]
            .iter()
            .map(|reference| match boundary {
                BoundaryPolicy::InBounds => reference.read(),
                BoundaryPolicy::Masked => masked_load_or(context, reference, 0f32),
            })
            .collect::<Result<Vec<_>, _>>()?;
        kernels_alias::accumulate(context, &references[2], &dot(&tiles[0], &tiles[1])?, boundary)
    })
    .unwrap()
}

#[test]
fn test_kernel_callable() {
    let left = Array::vector(vec![1.0f32, 2.0]).unwrap();
    let right = Array::vector(vec![3.0f32, 5.0]).unwrap();
    assert_eq!(add(&left, &right), Ok(Array::vector(vec![4.0f32, 7.0]).unwrap()));
    assert_eq!(zero::<Array>(), Ok(Array::vector(vec![0.0f32, 0.0]).unwrap()));
    assert_eq!(relative::zero::<Array>(), Ok(Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap()));
    assert_eq!(sum(&Array::vector(vec![1.0f32, -2.0, 4.0]).unwrap()), Ok(Array::scalar(3.0f32).unwrap()));
    assert_eq!(sum(&Array::vector(Vec::<f32>::new()).unwrap()), Ok(Array::scalar(0.0f32).unwrap()));
}

#[test]
fn test_kernel_callable_stages() {
    // Projected staged values bind the kernel call into their context instead of interpreting it.
    let r#type = ArrayType::new_static(DataType::F32, [2]);
    let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
        |inputs: Vec<_>| {
            let left = ProjectedValue::new(inputs[0].clone(), r#type.clone());
            let right = ProjectedValue::new(inputs[1].clone(), r#type.clone());
            Ok(vec![add(&left, &right)?.into_value()])
        },
        vec![ArrayIrType::Array(r#type.clone()); 2],
    )
    .unwrap();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f32[2], %1:f32[2] .
            let %2:f32[2] = kernel_call [
                grid=[],
                parameter_0=read-only f32[2] [2] InBounds,
                mapping_0={
                    lambda  .
                    let %0:dimension<0> = const 0
                    in (%0)
                },
                parameter_1=read-only f32[2] [2] InBounds,
                mapping_1={
                    lambda  .
                    let %0:dimension<0> = const 0
                    in (%0)
                },
                parameter_2=write-only f32[2] [2] InBounds,
                mapping_2={
                    lambda  .
                    let %0:dimension<0> = const 0
                    in (%0)
                },
            ] %0 %1 [
                body={
                    lambda %0:ref<f32[2]>, %1:ref<f32[2]>, %2:ref<f32[2]> .
                    let %3:f32[2] = reference_read %0
                        %4:f32[2] = reference_read %1
                        %5:f32[2] = add %3 %4
                        () = reference_write %2 %5
                    in ()
                },
            ]
            in (%2)"
        },
    );
    assert_eq!(
        program.interpret(vec![
            ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![3.0f32, 5.0]).unwrap()),
        ]),
        Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 7.0]).unwrap())]),
    );
}

#[test]
fn test_kernel_facade_namespace() {
    // Facade callers rely on the default `ryft::kernels` namespace at both re-exported attribute paths.
    #[ryft::kernels::kernel(requires = left.shape()[0] == right.shape()[0])]
    fn facade_add(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
    ) {
        let first = left.load();
        let second = right.load();
        output.store(first + second);
    }

    #[ryft::kernel(requires = left.shape()[0] == right.shape()[0])]
    fn top_level_add(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
    ) {
        let first = left.load();
        let second = right.load();
        output.store(first + second);
    }

    let left = Array::vector(vec![1.0f32, 2.0]).unwrap();
    let right = Array::vector(vec![3.0f32, 5.0]).unwrap();
    let expected = add(&left, &right);
    assert_eq!(facade_add(&left, &right), expected);
    assert_eq!(top_level_add(&left, &right), expected);
    let r#type = ArrayType::new_static(DataType::F32, [2]);
    let key = add::definition(&r#type, &r#type).unwrap().semantic_key().unwrap();
    assert_eq!(facade_add::definition(&r#type, &r#type).unwrap().semantic_key().unwrap(), key);
    assert_eq!(top_level_add::definition(&r#type, &r#type).unwrap().semantic_key().unwrap(), key);
}

#[test]
fn test_kernel_definition() {
    // Generated definitions stage the same programs as explicitly traced definitions.
    let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
    let generated = matmul::definition(&r#type, &r#type).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        references[2].write(&dot(&references[0].read()?, &references[1].read()?)?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());

    // A reduction capability stages one canonical reduction.
    let definition = sum::definition(&ArrayType::new_static(DataType::F32, [3])).unwrap();
    assert_eq!(
        definition.body().to_string(),
        indoc! {"
            lambda %0:ref<f32[3]>, %1:ref<f32[]> .
            let %2:f32[3] = reference_read %0
                %3:f32[] = reduce [kind=sum, axes=[0]] %2
                () = reference_write %1 %3
            in ()"
        },
    );
}

#[test]
fn test_kernel_definition_validation() {
    // Input types are checked against the annotations before the `requires` constraint is evaluated.
    assert_eq!(
        matmul::definition(&ArrayType::scalar(DataType::F32), &ArrayType::new_static(DataType::F32, [2, 2]))
            .map(|_| ()),
        Err(KernelError::Type(TypeError::invalid("kernel input `left` requires data type `F32` and rank `2`"))),
    );
    assert_eq!(
        matmul::definition(
            &ArrayType::new_static(DataType::F32, [2, 3]),
            &ArrayType::new_static(DataType::F32, [4, 2])
        )
        .map(|_| ()),
        Err(KernelError::Type(TypeError::invalid("kernel shape requirement is not satisfied"))),
    );
}

#[test]
fn test_kernel_matmul() {
    for (rows, depth, columns) in [(0, 0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1), (2, 3, 4), (3, 2, 5)] {
        let (left, right, product) = matmul_case(rows, depth, columns);
        assert_eq!(matmul(&left, &right), Ok(product));
    }
}

#[test]
fn test_kernel_tiled_matmul() {
    // The shapes cover empty outputs, partial edge tiles, and reductions over several tiles.
    for (rows, depth, columns) in
        [(0, 0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1), (2, 3, 4), (3, 2, 5), (1, 33, 2), (33, 35, 34)]
    {
        let (left, right, product) = matmul_case(rows, depth, columns);
        assert_eq!(tiled_matmul(&left, &right), Ok(product));
    }

    let generated = tiled_matmul::definition(
        &ArrayType::new_static(DataType::F32, [3, 5]),
        &ArrayType::new_static(DataType::F32, [5, 4]),
    )
    .unwrap();
    let explicit: KernelDefinition =
        KernelDefinition::trace(generated.operation().clone(), |(references, coordinates)| {
            let context = references[0].context();
            let accumulator = zeros::<f32, _>(context, [32, 32])?;
            let state = vec![
                references[0].clone(),
                references[1].clone(),
                references[2].clone(),
                coordinates[0].clone(),
                coordinates[1].clone(),
                accumulator,
            ];
            let values = for_loop(context, 0..1, state, |depth, mut values| {
                let context = depth.context();
                let left = tile_load(context, &values[0], vec![32, 32], &[values[3].clone(), depth.clone()], 0.0f32)?;
                let right = tile_load(context, &values[1], vec![32, 32], &[depth.clone(), values[4].clone()], 0.0f32)?;
                let product = dot(&left, &right)?;
                let operation = ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new()));
                values[5] = context.bind(operation, vec![], &[values[5].clone(), product])?.remove(0);
                Ok(values)
            })?;
            tile_store(context, &references[2], &values[5])
        })
        .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_tiled_output_position() {
    // The tiled output keeps its declared position among the body references. The subtraction and the partial edge
    // tile make a swapped binding of the inputs or of the output visible.
    let left = Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap();
    let right = Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap();
    let difference = Array::vector(vec![9.0f32, 18.0, 27.0]).unwrap();
    assert_eq!(tiled_difference_output_first(&left, &right), Ok(difference.clone()));
    assert_eq!(tiled_difference_output_middle(&left, &right), Ok(difference.clone()));
    assert_eq!(tiled_difference_output_last(&left, &right), Ok(difference.clone()));

    let r#type = ArrayType::new_static(DataType::F32, [3]);
    let read = KernelParameterAccess::ReadOnly;
    let write = KernelParameterAccess::WriteOnly;
    for (definition, accesses) in [
        (tiled_difference_output_first::definition(&r#type, &r#type).unwrap(), [write, read, read]),
        (tiled_difference_output_middle::definition(&r#type, &r#type).unwrap(), [read, write, read]),
        (tiled_difference_output_last::definition(&r#type, &r#type).unwrap(), [read, read, write]),
    ] {
        let parameters = definition.operation().parameters();
        assert_eq!(parameters.iter().map(|parameter| parameter.access()).collect::<Vec<_>>(), accesses);
        assert_eq!(definition.interpret(vec![left.clone(), right.clone()], 16), Ok(vec![difference.clone()]));
    }
}

#[test]
fn test_kernel_loops() {
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(accumulate(&input), Ok(Array::vector(vec![2.0f32, 4.0]).unwrap()));
    let empty = Array::vector(Vec::<f32>::new()).unwrap();
    assert_eq!(accumulate(&empty), Ok(empty));

    // Loops stage one `while` operation regardless of their trip count, and they carry every visible binding.
    let definition = accumulate::definition(&ArrayType::new_static(DataType::F32, [1000])).unwrap();
    assert_eq!(
        definition.body().to_string(),
        indoc! {"
            lambda %0:ref<f32[1000]>, %1:ref<f32[1000]> .
            let %2:f32[1000] = zero [type=f32[1000]]
                %3:i64[] = const 0
                %4:i64[], %5:ref<f32[1000]>, %6:ref<f32[1000]>, %7:f32[1000] = \
                    while [iteration_bound=1000] %3 %0 %1 %2 [
                    condition={
                        lambda %0:i64[], %1:ref<f32[1000]>, %2:ref<f32[1000]>, %3:f32[1000] .
                        let %4:i64[] = const 1000
                            %5:bool[] = compare [direction=LessThan] %0 %4
                        in (%5)
                    },
                    body={
                        lambda %0:i64[], %1:ref<f32[1000]>, %2:ref<f32[1000]>, %3:f32[1000] .
                        let %4:f32[1000] = reference_read %1
                            %5:f32[1000] = add %3 %4
                            %6:i64[] = const 1
                            %7:i64[] = add %0 %6
                        in (%7, %1, %2, %5)
                    },
                ]
                () = reference_write %1 %7
            in ()"
        },
    );
    let generated = accumulate::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let context = references[0].context();
        let accumulator = zeros::<f32, _>(context, [2])?;
        let state = vec![references[0].clone(), references[1].clone(), accumulator];
        let values = for_loop(context, 0..2, state, |_, mut values| {
            let input = values[0].read()?;
            let operation = ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new()));
            values[2] = values[0].context().bind(operation, vec![], &[values[2].clone(), input])?.remove(0);
            Ok(values)
        })?;
        references[1].write(&values[2])
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_conditions() {
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(
        conditional_accumulate(&Array::scalar(true).unwrap(), &input),
        Ok(Array::vector(vec![3.0f32, 6.0]).unwrap()),
    );
    assert_eq!(
        conditional_accumulate(&Array::scalar(false).unwrap(), &input),
        Ok(Array::vector(vec![-3.0f32, -6.0]).unwrap()),
    );
}

#[test]
fn test_kernel_compound_assignment_grouping() {
    // The right operand is evaluated as a whole, so each element `x` becomes `x * (x + x)` and not `x * x + x`.
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(compound_assignment(&input), Ok(Array::vector(vec![2.0f32, 8.0]).unwrap()));
}

#[test]
fn test_kernel_source_provenance() {
    let definition = located::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
    let write = definition.body().instructions().last().unwrap();
    let (scope, origin) = write.provenance().as_scope().unwrap();
    assert_eq!(scope.name(), format!("{}:{}:5", file!(), LOCATED_SOURCE_LINE));
    assert!(origin.is_unknown());
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(located(&input), Ok(input));
}

#[test]
fn test_kernel_declarative_macro_fragments() {
    // Literals and expressions that declarative macros forward into kernels arrive wrapped in invisible groups.
    macro_rules! tiled_scale {
        ($width:literal, $scale:expr) => {
            /// Scales a vector tile by tile, with a partially populated final tile.
            #[kernels_alias::kernel(crate = "kernels_alias")]
            fn scale(
                #[input(data_type = F32, rank = 1)] input: &Array,
                #[output(data_type = F32, shape = [input.shape()[0]], tile = [$width], boundary = masked)]
                output: &mut Array,
            ) {
                let [block] = output.tile_index();
                let tiles = input.tiles([$width]).pad(0.0);
                output.store(tiles.load([block]) * $scale);
            }
        };
    }
    tiled_scale!(32, (2.0f32));
    let input = Array::vector((0..33).map(|index| index as f32 - 16.0).collect::<Vec<_>>()).unwrap();
    let expected = Array::vector((0..33).map(|index| 2.0 * (index as f32 - 16.0)).collect::<Vec<_>>()).unwrap();
    assert_eq!(scale(&input), Ok(expected));
}

#[test]
fn test_kernel_carried_matmul() {
    // Every output block is zeroed at `k == 0` and then accumulates along the sequential axis `k`.
    let left = Array::matrix(4, 4, (1..=16).map(|value| value as f32).collect()).unwrap();
    let right = Array::matrix(4, 4, (1..=16).rev().map(|value| value as f32).collect()).unwrap();
    let expected = Array::matrix(
        4,
        4,
        vec![80f32, 70., 60., 50., 240., 214., 188., 162., 400., 358., 316., 274., 560., 502., 444., 386.],
    )
    .unwrap();
    assert_eq!(carried_matmul(&left, &right), Ok(expected));

    let r#type = ArrayType::new_static(DataType::F32, [4, 4]);
    let generated = carried_matmul::definition(&r#type, &r#type).unwrap();
    let explicit = carried_matmul_definition([4, 4, 4], BoundaryPolicy::InBounds);
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_carried_matmul_masked_edges() {
    for (rows, depth, columns) in [(3, 4, 3), (1, 1, 1), (2, 3, 4), (3, 5, 2)] {
        let (left, right, product) = matmul_case(rows, depth, columns);
        assert_eq!(masked_carried_matmul(&left, &right), Ok(product));
    }

    let generated = masked_carried_matmul::definition(
        &ArrayType::new_static(DataType::F32, [3, 4]),
        &ArrayType::new_static(DataType::F32, [4, 3]),
    )
    .unwrap();
    let explicit = carried_matmul_definition([3, 4, 3], BoundaryPolicy::Masked);
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_two_outputs() {
    // Several results are returned as a tuple in declaration order.
    assert_eq!(
        split(&Array::vector(vec![1.5f32, 2.0, 3.5, 4.0]).unwrap()),
        Ok((Array::vector(vec![1i32, 2, 3, 4]).unwrap(), Array::vector(vec![0f32, 0.0, 3.5, 4.0]).unwrap())),
    );

    let generated = split::definition(&ArrayType::new_static(DataType::F32, [4])).unwrap();
    let window = || ParameterWindow::BlockIndices {
        block_shape: vec![2],
        boundary: BoundaryPolicy::InBounds,
        indices: vec![MappingExpression::GridCoordinate(0)],
    };
    let operation = mapped_call(
        Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel).with_name("i").unwrap()]).unwrap(),
        vec![
            (ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::ReadOnly, window()),
            (ArrayType::new_static(DataType::I32, [4]), KernelParameterAccess::WriteOnly, window()),
            (ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::WriteOnly, window()),
        ],
        vec![],
    )
    .unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
        let context = references[0].context();
        let input = references[0].read()?;
        references[1].write(&input.convert_element_type(DataType::I32)?)?;
        let threshold = full::<f32, _>(context, [2], 2.5f32)?;
        let zero = zeros::<f32, _>(context, [2])?;
        let mask = compare(context, &input, &threshold, ComparisonDirection::GreaterThan)?;
        references[2].write(&select(context, &mask, &input, &zero)?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_inout() {
    // Each grid point increments its own element of `counts` and atomically increments the whole of `total`.
    assert_eq!(
        count(&Array::vector(vec![1i32, 2, 3]).unwrap(), &Array::scalar(10i32).unwrap()),
        Ok((Array::vector(vec![2i32, 3, 4]).unwrap(), Array::scalar(13i32).unwrap())),
    );

    let generated =
        count::definition(&ArrayType::new_static(DataType::I32, [3]), &ArrayType::scalar(DataType::I32)).unwrap();
    assert_eq!(generated.operation().aliases(), vec![(0, 0), (1, 1)]);
    let operation = mapped_call(
        Grid::new(vec![GridDimension::new(3usize.into(), GridExecution::Parallel).with_name("i").unwrap()]).unwrap(),
        vec![
            (
                ArrayType::new_static(DataType::I32, [3]),
                KernelParameterAccess::ReadWrite,
                ParameterWindow::BlockIndices {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    indices: vec![MappingExpression::GridCoordinate(0)],
                },
            ),
            (ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, ParameterWindow::WholeArray),
        ],
        vec![],
    )
    .unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
        let context = references[0].context();
        kernels_alias::accumulate(context, &references[0], &ones::<i32, _>(context, [1])?, BoundaryPolicy::InBounds)?;
        references[1].atomic_add_update(&ones::<i32, _>(context, [])?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_masked_store() {
    // The padding of the last block is loaded as zero and never stored.
    assert_eq!(
        absolute(&Array::vector(vec![3i32, -1, 4, -5, -2]).unwrap()),
        Ok(Array::vector(vec![3i32, 1, 4, 5, 2]).unwrap()),
    );
}

#[test]
fn test_kernel_window_mask() {
    // The window of the second grid point has one valid element.
    assert_eq!(
        validity(&Array::vector(vec![1i32, 2, 3]).unwrap()),
        Ok(Array::vector(vec![true, true, true, false]).unwrap()),
    );

    let generated = validity::definition(&ArrayType::new_static(DataType::I32, [3])).unwrap();
    let window = |boundary| ParameterWindow::BlockIndices {
        block_shape: vec![2],
        boundary,
        indices: vec![MappingExpression::GridCoordinate(0)],
    };
    let operation = mapped_call(
        Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel).with_name("i").unwrap()]).unwrap(),
        vec![
            (
                ArrayType::new_static(DataType::I32, [3]),
                KernelParameterAccess::ReadOnly,
                window(BoundaryPolicy::Masked),
            ),
            (
                ArrayType::new_static(DataType::Boolean, [4]),
                KernelParameterAccess::WriteOnly,
                window(BoundaryPolicy::InBounds),
            ),
        ],
        vec![],
    )
    .unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
        references[1].write(&window_mask(references[0].context(), &references[0])?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_masked_load_and_swap() {
    // The mask is computed in the body, and the padding of masked windows takes the fill value.
    assert_eq!(
        select_elements(&Array::vector(vec![1i32, 2]).unwrap()),
        Ok(Array::vector(vec![1i32, 7, 7, 7]).unwrap()),
    );
    assert_eq!(
        swap(&Array::vector(vec![1i32, 2, 3, 4, 5]).unwrap()),
        Ok((Array::vector(vec![9i32, 2, 11, 12, 9]).unwrap(), Array::vector(vec![1i32, -1, 3, 4, 5]).unwrap())),
    );
}

#[test]
fn test_kernel_scratch() {
    assert_eq!(scratch_copy(&Array::scalar(23i32).unwrap()), Ok(Array::scalar(23i32).unwrap()));

    let generated = scratch_copy::definition(&ArrayType::scalar(DataType::I32)).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let storage = scratch::<i32, _>(references[0].context(), [], 4)?;
        storage.write(&references[0].read()?)?;
        references[1].write(&storage.read()?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_async_copy() {
    let input = Array::vector(vec![17i32, 29]).unwrap();
    assert_eq!(staged_copy(&input), Ok(input));

    let generated = staged_copy::definition(&ArrayType::new_static(DataType::I32, [2])).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let context = references[0].context();
        let staging = scratch::<i32, _>(context, [2], 16)?;
        let token = async_copy(context, &references[0], &staging)?;
        wait(context, &token)?;
        references[1].write(&staging.read()?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_prefetch() {
    // The input window starts at the block whose index is the prefetched scalar, which follows the other inputs.
    let input = Array::vector((0..8i32).collect::<Vec<_>>()).unwrap();
    assert_eq!(gather(&input, &Array::scalar(1i32).unwrap()), Ok(Array::vector(vec![2i32, 3]).unwrap()));
    assert_eq!(gather(&input, &Array::scalar(3i32).unwrap()), Ok(Array::vector(vec![6i32, 7]).unwrap()));

    let generated =
        gather::definition(&ArrayType::new_static(DataType::I32, [8]), &ArrayType::scalar(DataType::I32)).unwrap();
    let operation = mapped_call(
        Grid::new(vec![GridDimension::new(1usize.into(), GridExecution::Parallel).with_name("i").unwrap()]).unwrap(),
        vec![
            (
                ArrayType::new_static(DataType::I32, [8]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::BlockIndices {
                    block_shape: vec![2],
                    boundary: BoundaryPolicy::InBounds,
                    indices: vec![MappingExpression::Prefetch(0)],
                },
            ),
            (ArrayType::new_static(DataType::I32, [2]), KernelParameterAccess::WriteOnly, ParameterWindow::WholeArray),
        ],
        vec![ArrayType::scalar(DataType::I32)],
    )
    .unwrap();
    let explicit: KernelDefinition =
        KernelDefinition::trace(operation, |(references, _)| references[1].write(&references[0].read()?)).unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_element_starts() {
    // The input windows run backwards, which no separable tiling proof describes, so the grid is enumerated.
    assert_eq!(
        reverse_blocks(&Array::vector(vec![0i32, 1, 2, 3, 4, 5]).unwrap()),
        Ok(Array::vector(vec![4i32, 5, 2, 3, 0, 1]).unwrap()),
    );

    let generated = reverse_blocks::definition(&ArrayType::new_static(DataType::I32, [6])).unwrap();
    let operation = mapped_call(
        Grid::new(vec![GridDimension::new(3usize.into(), GridExecution::Parallel).with_name("i").unwrap()]).unwrap(),
        vec![
            (
                ArrayType::new_static(DataType::I32, [6]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![2],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![
                        MappingExpression::Constant(4)
                            - MappingExpression::GridCoordinate(0) * MappingExpression::Constant(2),
                    ],
                },
            ),
            (
                ArrayType::new_static(DataType::I32, [6]),
                KernelParameterAccess::WriteOnly,
                ParameterWindow::BlockIndices {
                    block_shape: vec![2],
                    boundary: BoundaryPolicy::InBounds,
                    indices: vec![MappingExpression::GridCoordinate(0)],
                },
            ),
        ],
        vec![],
    )
    .unwrap();
    let explicit: KernelDefinition =
        KernelDefinition::trace(operation, |(references, _)| references[1].write(&references[0].read()?)).unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_slices() {
    assert_eq!(
        columns(&Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap()),
        Ok(Array::matrix(2, 2, vec![2i32, 3, 5, 6]).unwrap()),
    );
    assert_eq!(
        add_prefix(&Array::vector(vec![10i32, 20]).unwrap(), &Array::vector(vec![1i32, 2, 3]).unwrap()),
        Ok(Array::vector(vec![11i32, 22, 3]).unwrap()),
    );
}

#[test]
fn test_kernel_slices_rebound_source() {
    // A slice stays attached to the allocation that its source named where the slice was declared, so rebinding the
    // source afterwards changes neither its loads nor its stores, while writes to that allocation stay visible.
    let first = Array::vector(vec![0i32, 8]).unwrap();
    let second = Array::vector(vec![1i32, 9]).unwrap();
    assert_eq!(rebound_slice_load(&first, &second), Ok(Array::vector(vec![8i32, 9]).unwrap()));
    assert_eq!(
        rebound_slice_store(&Array::vector(vec![8i32, 9]).unwrap()),
        Ok((Array::vector(vec![8i32, 9]).unwrap(), Array::vector(vec![7i32, 8]).unwrap())),
    );

    // Staged loops and conditions carry the slice, so it keeps its allocation inside and after them. Staging rejects
    // a reference that a loop or condition rebinds, so the source is rebound before them.
    assert_eq!(rebound_slice_loop(&first, &second), Ok(Array::vector(vec![18i32, 8, 9]).unwrap()));
    assert_eq!(
        rebound_slice_condition(&Array::scalar(true).unwrap(), &first, &second),
        Ok(Array::vector(vec![8i32, 8, 9]).unwrap()),
    );
    assert_eq!(
        rebound_slice_condition(&Array::scalar(false).unwrap(), &first, &second),
        Ok(Array::vector(vec![9i32, 8, 9]).unwrap()),
    );

    // Raw and ordinary spellings of an identifier name the same binding, so both reach the declared source, and
    // shadowing a binding through the other spelling is rejected like any other shadowing.
    let first = Array::vector(vec![8i32]).unwrap();
    let second = Array::vector(vec![9i32]).unwrap();
    assert_eq!(raw_identifier_slices(&first, &second), Ok(Array::vector(vec![8i32, 8, 9]).unwrap()));
}

#[test]
fn test_kernel_callable_placed_inputs() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
    let r#type = ArrayType::new_static(DataType::I32, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
    let placed = Array::from_elements(r#type, &[3i32, 5]).unwrap();
    let unplaced = Array::vector(vec![3i32, 5]).unwrap();

    // The definition is built for the physical parameter types, and the call receives the placed argument, so the
    // callable returns what calling the physical definition with that argument returns.
    let definition = copy::definition(&ArrayType::new_static(DataType::I32, [2])).unwrap();
    assert_eq!(definition.interpret(vec![placed.clone()], 16), Ok(vec![unplaced.clone()]));
    assert_eq!(copy(&placed), Ok(unplaced.clone()));

    // Unplaced calls are unchanged.
    assert_eq!(copy(&unplaced), Ok(unplaced.clone()));
}

#[test]
fn test_kernel_constants() {
    assert_eq!(constants::<Array>(), Ok(Array::vector(vec![2i32, 3, 4]).unwrap()));
}

#[test]
fn test_kernel_logic_and_negation() {
    let left = Array::vector(vec![true, true, false, false]).unwrap();
    let right = Array::vector(vec![true, false, true, false]).unwrap();
    assert_eq!(logic(&left, &right), Ok(Array::vector(vec![true, false, false, true]).unwrap()));

    let left = Array::vector(vec![7i32, -7, 9]).unwrap();
    let right = Array::vector(vec![3i32, 3, 4]).unwrap();
    assert_eq!(negated_remainder(&left, &right), Rem::rem(&left, &right).and_then(|remainder| remainder.neg()));
}

#[test]
fn test_kernel_comparisons() {
    // Each comparison operator stages a `compare` with its direction, and a literal operand broadcasts as a scalar.
    let left = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
    let right = Array::vector(vec![2i32, 2, 1, 5]).unwrap();
    let flags = |elements: [bool; 4]| Array::vector(elements.to_vec()).unwrap();
    assert_eq!(
        comparisons(&left, &right),
        Ok((
            flags([false, true, false, false]),
            flags([true, false, true, true]),
            flags([true, false, false, true]),
            flags([true, true, false, false]),
            flags([false, false, true, false]),
            flags([true, true, true, false]),
        )),
    );

    let r#type = ArrayType::new_static(DataType::I32, [4]);
    let generated = comparisons::definition(&r#type, &r#type).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let context = references[0].context();
        let two = || context.lift(ArrayIrValue::Array(Array::scalar(2i32).unwrap()));
        let equal = compare(context, &references[0].read()?, &references[1].read()?, ComparisonDirection::Equal)?;
        references[2].write(&equal)?;
        let not_equal =
            compare(context, &references[0].read()?, &references[1].read()?, ComparisonDirection::NotEqual)?;
        references[3].write(&not_equal)?;
        let less = compare(context, &references[0].read()?, &references[1].read()?, ComparisonDirection::LessThan)?;
        references[4].write(&less)?;
        let less_or_equal = compare(context, &references[0].read()?, &two()?, ComparisonDirection::LessThanOrEqual)?;
        references[5].write(&less_or_equal)?;
        let greater =
            compare(context, &references[0].read()?, &references[1].read()?, ComparisonDirection::GreaterThan)?;
        references[6].write(&greater)?;
        let greater_or_equal =
            compare(context, &two()?, &references[1].read()?, ComparisonDirection::GreaterThanOrEqual)?;
        references[7].write(&greater_or_equal)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_select() {
    // The literal branch broadcasts to the shape of the mask and of the other branch.
    let left = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
    let right = Array::vector(vec![2i32, 2, 1, 5]).unwrap();
    assert_eq!(select_less(&left, &right), Ok(Array::vector(vec![1i32, 9, 9, 4]).unwrap()));

    let r#type = ArrayType::new_static(DataType::I32, [4]);
    let generated = select_less::definition(&r#type, &r#type).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let context = references[0].context();
        let mask = compare(context, &references[0].read()?, &references[1].read()?, ComparisonDirection::LessThan)?;
        let kept = references[0].read()?;
        let other = context.lift(ArrayIrValue::Array(Array::scalar(9i32)?))?;
        references[2].write(&select(context, &mask, &kept, &other)?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_coordinate_comparisons() {
    // Comparisons of a grid coordinate with a constant stage `compare_dimension`, which mirrors the direction when
    // the constant is the left-hand side input, and their rank-0 results broadcast through `select`.
    let values = |elements: [i32; 4]| Array::vector(elements.to_vec()).unwrap();
    assert_eq!(
        coordinate_comparisons::<Array>(),
        Ok((
            values([0, 1, 0, 0]),
            values([1, 0, 1, 1]),
            values([1, 1, 0, 0]),
            values([0, 0, 1, 1]),
            values([0, 0, 0, 1]),
            values([1, 1, 1, 0]),
        )),
    );

    let generated = coordinate_comparisons::definition().unwrap();
    let explicit: KernelDefinition =
        KernelDefinition::trace(generated.operation().clone(), |(references, coordinates)| {
            let context = references[0].context();
            let flag = |direction, constant| {
                let mask = compare_dimension(context, &coordinates[0], direction, constant)?;
                select(context, &mask, &ones::<i32, _>(context, [1])?, &zeros::<i32, _>(context, [1])?)
            };
            references[0].write(&flag(ComparisonDirection::Equal, 1)?)?;
            references[1].write(&flag(ComparisonDirection::NotEqual, 1)?)?;
            references[2].write(&flag(ComparisonDirection::LessThan, 2)?)?;
            references[3].write(&flag(ComparisonDirection::GreaterThanOrEqual, 2)?)?;
            references[4].write(&flag(ComparisonDirection::GreaterThan, 2)?)?;
            references[5].write(&flag(ComparisonDirection::LessThanOrEqual, 2)?)
        })
        .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_capabilities() {
    // Each kernel agrees with the capability function of the same name applied to host arrays.
    let input = Array::vector(vec![0.25f32, 0.5, 0.75]).unwrap();
    let other = Array::vector(vec![0.5f32, 0.25, 2.0]).unwrap();
    assert_eq!(capabilities::abs(&input), input.abs());
    assert_eq!(capabilities::ceil(&input), input.ceil());
    assert_eq!(capabilities::cos(&input), input.cos());
    assert_eq!(capabilities::erf(&input), input.erf());
    assert_eq!(capabilities::exp(&input), input.exp());
    assert_eq!(capabilities::exp_m1(&input), input.exp_m1());
    assert_eq!(capabilities::floor(&input), input.floor());
    assert_eq!(capabilities::ln_1p(&input), input.ln_1p());
    assert_eq!(capabilities::log(&input), input.log());
    assert_eq!(capabilities::logistic(&input), input.logistic());
    assert_eq!(capabilities::round(&input), input.round());
    assert_eq!(capabilities::rsqrt(&input), input.rsqrt());
    assert_eq!(capabilities::sign(&input), input.sign());
    assert_eq!(capabilities::sin(&input), input.sin());
    assert_eq!(capabilities::sqrt(&input), input.sqrt());
    assert_eq!(capabilities::tan(&input), input.tan());
    assert_eq!(capabilities::tanh(&input), input.tanh());
    assert_eq!(capabilities::atan2(&input, &other), input.atan2(&other));
    assert_eq!(capabilities::log_add_exp(&input, &other), input.log_add_exp(&other));
    assert_eq!(capabilities::max(&input, &other), Max::max(&input, &other));
    assert_eq!(capabilities::min(&input, &other), Min::min(&input, &other));
    assert_eq!(capabilities::pow(&input, &other), input.pow(&other));
    assert_eq!(capabilities::reduce_sum(&input), input.reduce(&[0], ReductionKind::Sum));
    assert_eq!(capabilities::reduce_product(&input), input.reduce(&[0], ReductionKind::Product));
    assert_eq!(capabilities::reduce_mean(&input), input.reduce(&[0], ReductionKind::Mean));
    assert_eq!(capabilities::reduce_log_sum_exp(&input), input.reduce(&[0], ReductionKind::LogSumExp));
    assert_eq!(capabilities::reduce_max(&input), Ok(Array::scalar(0.75f32).unwrap()));
    assert_eq!(capabilities::reduce_min(&input), Ok(Array::scalar(0.25f32).unwrap()));
    let flags = Array::vector(vec![true, false, true]).unwrap();
    assert_eq!(capabilities::reduce_any(&flags), Ok(Array::scalar(true).unwrap()));
    assert_eq!(capabilities::reduce_all(&flags), Ok(Array::scalar(false).unwrap()));
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/kernels/error_signatures.rs");
    test_cases.compile_fail("tests/kernels/error_metadata.rs");
    test_cases.compile_fail("tests/kernels/error_bodies.rs");
}
