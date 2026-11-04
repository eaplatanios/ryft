use ryft::{Array, Capability, ProgramError, Type, capability};

#[capability]
trait MissingUniverse: Capability {}

#[capability]
trait BoundedUniverse<T: Type = <Self as Capability>::Universe>: Capability {}

#[capability]
trait MissingCapabilitySupertrait<T = <Self as Capability>::Universe>: Sized {}

#[capability(projection(ArrayIrType => ArrayType))]
trait UnsupportedParameter<Stored = Array, T = <Self as Capability>::Universe>: Capability + Sized {
    fn double(&self) -> Result<Self, ProgramError>;
}

#[capability(projection(ArrayIrType => ArrayType))]
trait UnsupportedFunctions<T = <Self as Capability>::Universe>: Capability + Sized {
    type Output;

    fn by_value(self) -> Result<Self, ProgramError>;

    fn by_vector(&self, others: Vec<Self>) -> Result<Self, ProgramError>;

    fn infallible(&self) -> Self;

    fn boxed(&self) -> Result<Box<Self>, ProgramError>;
}

/// Uses an invalid capability, which reports no additional errors because the capability is still emitted.
fn double<V: UnsupportedParameter>(value: &V) -> Result<V, ProgramError> {
    value.double()
}

fn main() {}
