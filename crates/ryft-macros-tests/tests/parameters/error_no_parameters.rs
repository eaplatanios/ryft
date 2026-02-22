use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
struct NestedVecOfTuples<P> {
    parameter: Vec<P>,
}

fn main() {}
