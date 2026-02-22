use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
struct StructWithMultipleParameters<P: Parameter, V: Parameter> {
    parameter_0: P,
    parameter_1: (P, V),
}

#[derive(Parameterized)]
struct StructWithUnusedParameterType<P: Parameter> {
    parameter_0: usize,
    parameter_1: (u32, u64),
}

#[derive(Parameterized)]
struct StructWithInvalidTypeParam<__P: Parameter> {
    parameter_0: __P,
    parameter_1: (__P, usize),
}

#[derive(Parameterized)]
struct StructWithInvalidLifetimeParam<'__p, P: Parameter> {
    parameter_0: P,
    parameter_1: (&'__p str, usize),
}

fn main() {}
