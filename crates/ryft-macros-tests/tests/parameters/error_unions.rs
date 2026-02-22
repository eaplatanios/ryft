use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
union UnionWithParameters<P: Parameter> {
    i: P,
    f: (u64, P),
}

#[derive(Parameterized)]
union UnionWithMultipleParameters<P: Copy + Parameter, V: Copy + Parameter> {
    i: P,
    f: (P, V),
}

#[derive(Parameterized)]
union UnionWithoutParameters<P: Copy, V: Copy> {
    i: P,
    f: (P, V),
}

fn main() {}
