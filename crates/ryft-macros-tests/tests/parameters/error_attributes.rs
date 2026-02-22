use ryft::{Parameter, Parameterized};

#[derive(Parameterized)]
#[ryft(crate = "ryft")]
struct StructWithBadFieldAttribute<P: Parameter> {
    weights: P,

    #[ryft(crate = "_ryft")]
    bias: P,
}

#[derive(Parameterized)]
#[ryft(crate = "ryft")]
#[ryft(crate = "ryft")]
struct StructWithDuplicateAttribute<P: Parameter> {
    weights: P,
    bias: P,
}

#[derive(Parameterized)]
#[ryft(crate = "_ryft")]
struct StructWithBadRyftCrate<P: Parameter> {
    weights: P,
    bias: P,
}

#[derive(Parameterized)]
#[ryft(crate = "ryft", parameter_type = "P")]
struct StructWithManyBadAttributes<P: Parameter> {
    weights: P,

    #[ryft(crate = "_ryft")]
    bias: P,
}

#[derive(Parameterized)]
#[ryft(crate = "ryft")]
enum EnumWithBadVariantAttributes<P: Parameter> {
    #[ryft(crate = "_ryft")]
    Variant1(P),

    #[ryft(crate = "_ryft")]
    Variant2(P),
}

#[derive(Parameterized)]
#[ryft(ryft = "ryft")]
enum EnumWithBadAttributes<P: Parameter> {
    #[ryft(crate = "_ryft")]
    Variant1,
    Variant2(),
    Variant3(#[ryft(crate = "_ryft")] P),
    Variant4(P, #[ryft(crate = "_ryft")] (P, usize)),
    Variant5 {
        x: P,
        #[ryft(crate = "_ryft")]
        y: (P, usize),
    },
}

fn main() {}
