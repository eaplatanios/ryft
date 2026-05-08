/// Represents types that can be converted to a PJRT representation (e.g., a PJRT buffer type or layout).
pub trait ToPjrt {
    /// PJRT type that instances of this type can be converted to.
    type Output;

    /// Converts `self` into its PJRT representation.
    fn to_pjrt(&self) -> Self::Output;
}

/// Represents types that can be constructed from a PJRT representation (e.g., a PJRT buffer type or layout).
pub trait FromPjrt<T> {
    /// Type that this PJRT type instance can be converted to.
    type Output;

    /// Constructs an instance of this type from the provided PJRT instance.
    fn from_pjrt(value: T) -> Self::Output;
}
