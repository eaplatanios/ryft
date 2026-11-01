use ryft_kernels::kernel;

#[kernel(crate = "ryft_kernels")]
fn duplicate_rank(
    #[input(data_type = F32, rank = 1, rank = 2)] input: &Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
}

#[kernel(crate = "ryft_kernels")]
fn qualified_data_type(
    #[input(data_type = DataType::F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [])] output: &mut Array,
) {
}

#[kernel(crate = "ryft_kernels")]
fn host_shape(#[output(data_type = F32, shape = host_callback())] output: &mut Array) {}

#[kernel(crate = "ryft_kernels")]
fn out_of_bounds_axis(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[1]])] output: &mut Array,
) {
}

#[kernel(crate = "ryft_kernels")]
fn boundary(#[output(data_type = F32, shape = [4], tile = [2], boundary = clamped)] output: &mut Array) {}

fn main() {}
