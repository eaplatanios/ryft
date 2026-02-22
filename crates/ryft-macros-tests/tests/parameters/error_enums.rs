use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
enum EnumWithMultipleParameters<P: Parameter, V: Parameter> {
    Variant0(P),
    Variant1(P, V),
}

#[derive(Parameterized)]
enum EnumWithUnusedParameterType<P: Parameter> {
    Variant0(usize),
    Variant1(u32, u64),
}

#[derive(Parameterized)]
enum EnumWithVeryNestedParameterReference<P: Parameter> {
    Variant0(usize),
    Variant1 { field0: u32, field1: (u64, (i8, P, &'static P), u64) },
}

fn main() {}
