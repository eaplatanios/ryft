use ryft_kernels::kernel;

#[kernel(crate = "ryft_kernels", schedule = [1])]
fn arguments(#[output(data_type = F32, shape = [])] output: &mut Array) {}

#[kernel(crate = "ryft_kernels")]
fn escaping_reference(#[output(data_type = F32, shape = [])] output: &mut Array) -> &mut Array {
    output
}

#[kernel(crate = "ryft_kernels")]
fn generic_array(#[output(data_type = F32, shape = [])] output: &mut Array<f32>) {}

#[kernel(crate = "ryft_kernels")]
fn unannotated(input: &Array, #[output(data_type = F32, shape = [])] output: &mut Array) {}

#[kernel(crate = "ryft_kernels")]
fn mutable_input(
    #[input(data_type = F32, rank = 1)] input: &mut Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
}

#[kernel(crate = "ryft_kernels")]
fn missing_output(#[input(data_type = F32, rank = 1)] input: &Array) {}

#[kernel(crate = "ryft_kernels", grid(i = 2, i = 3))]
fn duplicate_axis(#[output(data_type = F32, shape = [2])] output: &mut Array) {}

#[kernel(crate = "ryft_kernels", grid(output = 2))]
fn colliding_axis(#[output(data_type = F32, shape = [2])] output: &mut Array) {}

#[kernel(crate = "ryft_kernels", grid(i = 2))]
fn grid_and_tile(#[output(data_type = F32, shape = [4], tile = [2], boundary = masked)] output: &mut Array) {}

#[kernel(crate = "ryft_kernels")]
fn tile_and_inout(
    #[inout(data_type = F32, rank = 1)] state: &mut Array,
    #[output(data_type = F32, shape = [4], tile = [2], boundary = masked)] output: &mut Array,
) {
}

#[kernel(crate = "ryft_kernels")]
fn immutable_inout(#[inout(data_type = F32, rank = 1)] state: &Array) {}

#[kernel(crate = "ryft_kernels")]
fn mutable_prefetch(
    #[prefetch(data_type = I32)] offset: &mut Array,
    #[output(data_type = F32, shape = [2])] output: &mut Array,
) {
}

fn main() {}
