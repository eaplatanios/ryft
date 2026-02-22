use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
struct NestedVecOfTuples<P: Parameter> {
    parameter: Vec<(P, usize)>,
}

fn main() {}
