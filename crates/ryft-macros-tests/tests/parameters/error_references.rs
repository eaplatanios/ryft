use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
struct StructWithReferences<'p, P: 'p + Parameter> {
    metadata_0: usize,
    metadata_1: u32,
    parameter_0: &'p P,
    parameter_1: &'p P,
}

#[derive(Parameterized)]
struct StructWithNestedReferences<'p, P: 'p + Parameter> {
    parameter_0: P,
    parameter_1: (P, (P, (usize, &'p P))),
    metadata_0: &'static str,
    metadata_1: String,
}

fn main() {}
