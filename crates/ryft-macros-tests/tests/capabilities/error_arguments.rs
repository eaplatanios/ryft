use ryft::{Capability, ProgramError, capability};

#[capability(projections(ArrayIrType => ArrayType))]
trait UnknownArgument<T = <Self as Capability>::Universe>: Capability + Sized {
    fn double(&self) -> Result<Self, ProgramError>;
}

#[capability(crate = "ryft", crate = "ryft")]
trait DuplicateCrate<T = <Self as Capability>::Universe>: Capability + Sized {
    fn double(&self) -> Result<Self, ProgramError>;
}

#[capability(projection(ArrayIrType => ArrayType), projection(ArrayIrType => ArrayType))]
trait DuplicateProjection<T = <Self as Capability>::Universe>: Capability + Sized {
    fn double(&self) -> Result<Self, ProgramError>;
}

#[capability(projection(ArrayIrType => ArrayType, ArrayType))]
trait MalformedProjection<T = <Self as Capability>::Universe>: Capability + Sized {
    fn double(&self) -> Result<Self, ProgramError>;
}

fn main() {}
