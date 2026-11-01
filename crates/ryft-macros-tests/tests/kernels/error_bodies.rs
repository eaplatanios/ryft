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

fn main() {}
