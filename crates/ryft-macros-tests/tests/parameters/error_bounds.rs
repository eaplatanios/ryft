use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
struct ParameterWithCloneBound<P: Clone + Parameter> {
    parameter_0: P,
    parameter_1: (P, P),
    non_parameter_0: usize,
    non_parameter_1: (usize, usize),
}

#[derive(Parameterized)]
struct ParameterWithLifetimeBound<'p, P: 'p + Clone + Parameter> {
    parameter_0: P,
    parameter_1: (P, &'p P),
    non_parameter_0: usize,
    non_parameter_1: (usize, usize),
}

fn main() {}
