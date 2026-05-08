use ryft_mlir::Location;

/// Represents types that can be converted to an MLIR representation (e.g., an MLIR [`Attribute`](ryft_mlir::Attribute),
/// [`Type`](ryft_mlir::Type), or [`Operation`](ryft_mlir::Operation)).
pub trait ToMlir {
    /// MLIR type that instances of this type can be converted to.
    type Output<'c, 't: 'c>;

    /// Converts `self` into its MLIR representation in the [`Context`](ryft_mlir::Context) associated
    /// with the provided `location`.
    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> Result<Self::Output<'c, 't>, ryft_mlir::Error>;
}
