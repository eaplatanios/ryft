use ryft_kernels::kernel;

#[kernel(crate = "ryft_kernels")]
fn host_loop(#[output(data_type = F32, shape = [])] output: &mut Array) {
    loop {
        output.store(0.0f32);
    }
}

#[kernel(crate = "ryft_kernels")]
fn attribute(#[output(data_type = F32, shape = [])] output: &mut Array) {
    #[cfg(any())]
    output.store(0.0f32);
}

#[kernel(crate = "ryft_kernels")]
fn shadowing(
    #[input(data_type = F32, rank = 0)] input: &Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
    let input = input.load();
    output.store(input);
}

#[kernel(crate = "ryft_kernels")]
fn dynamic_axes(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
    output.store(input.load().sum([input.shape()[0]]));
}

#[kernel(crate = "ryft_kernels")]
fn helper(#[output(data_type = F32, shape = [2])] output: &mut Array) {
    output.store(helpers::zeros::<f32>([2]));
}

#[kernel(crate = "ryft_kernels")]
fn untiled_coordinates(#[output(data_type = F32, shape = [2])] output: &mut Array) {
    let [block] = output.tile_index();
}

#[kernel(crate = "ryft_kernels", grid(i = 2))]
fn masked_load(
    #[input(data_type = F32, rank = 1, block = [2], index = [i], boundary = masked)] input: &Array,
    #[output(data_type = F32, shape = [4], block = [2], index = [i])] output: &mut Array,
) {
    output.store(input.load());
}

#[kernel(crate = "ryft_kernels")]
fn short_circuit(
    #[input(data_type = Boolean, rank = 1)] left: &Array,
    #[input(data_type = Boolean, rank = 1)] right: &Array,
    #[output(data_type = Boolean, shape = [left.shape()[0]])] output: &mut Array,
) {
    output.store(left.load() && right.load());
}

#[kernel(crate = "ryft_kernels")]
fn unknown_capability(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    output.store(input.load().softplus());
}

#[kernel(crate = "ryft_kernels")]
fn capability_arity(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    output.store(input.load_or());
}

#[kernel(crate = "ryft_kernels")]
fn slice_window_mask(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = Boolean, shape = [1])] output: &mut Array,
) {
    output.store(input.slice([0..1]).window_mask());
}

fn main() {}
