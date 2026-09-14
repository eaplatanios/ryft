use ryft_core::kernels::kernel;

#[kernel]
fn invalid(#[output(data_type = F32, shape = [])] output: &mut Array) -> &mut Array {
    output
}

fn main() {}
