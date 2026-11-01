use ryft_kernels::kernel;

#[kernel(crate = "ryft_kernels", grid = [1])]
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

fn main() {}
