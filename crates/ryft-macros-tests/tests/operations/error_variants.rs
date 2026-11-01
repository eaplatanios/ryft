use ryft::{ArrayType, ZeroOperation};

#[derive(ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>))]
enum BadVariantAttributesOperation {
    #[ryft(family_member(ArrayType))]
    UnknownAttribute(ZeroOperation<ArrayType>),
    #[ryft(structural(ArrayType))]
    RemovedStructuralClass(ZeroOperation<ArrayType>),
    #[ryft(skip_from, skip_from)]
    DuplicateSkipFrom(ZeroOperation<ArrayType>),
    FieldAttribute(#[ryft(projected(ArrayType))] ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>))]
enum BadVariantClassesOperation {
    #[ryft(projected(ArrayType), projected(ArrayType))]
    DuplicateClass(ZeroOperation<ArrayType>),
    #[ryft(projected(ArrayType), mixed(ArrayType))]
    ContradictoryClasses(ZeroOperation<ArrayType>),
    #[ryft(projected)]
    MissingMemberType(ZeroOperation<ArrayType>),
    #[ryft(projected(ArrayType, replicated))]
    UnknownRole(ZeroOperation<ArrayType>),
    #[ryft(mixed(ArrayType, structural, DimensionType))]
    TooManyArguments(ZeroOperation<ArrayType>),
    #[ryft(mixed)]
    UndeclaredMembers(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>), members(ArrayType, DimensionType))]
enum BadDeclaredMemberTypesOperation {
    #[ryft(mixed(ScalarType))]
    UndeclaredMemberType(ZeroOperation<ArrayType>),
    #[ryft(mixed(structural))]
    AmbiguousDefaultMemberType(ZeroOperation<ArrayType>),
}

fn main() {}
