use ryft_core::kernels::kernel;

#[kernel]
fn invalid(#[output(data_type = F32, shape = host_callback())] output: &mut Array) {}

fn main() {}
