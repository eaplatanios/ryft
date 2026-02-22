use ryft_xla_sys::bindings::{MlirAttribute, stablehloResultAccuracyAttrGet, stablehloResultAccuracyAttrGetMode};

use crate::{
    Attribute, Context, DenseIntegerElementsAttributeRef, DialectHandle, Location, Operation, Size, StringRef,
    TensorTypeRef, Type, mlir_attribute_field, mlir_enum_attribute, mlir_subtype_trait_impls,
};

pub mod arithmetic;
pub mod batch_normalization;
pub mod bit_manipulation;
pub mod casting;
pub mod communication;
pub mod comparison;
pub mod complex;
pub mod control_flow;
pub mod exponential;
pub mod functional;
pub mod linear_algebra;
pub mod logical;
pub mod manipulation;
pub mod miscellaneous;
pub mod random;
pub mod rounding;
pub mod trigonometric;
pub mod tuples;

pub use arithmetic::*;
pub use batch_normalization::*;
pub use bit_manipulation::*;
pub use casting::*;
pub use communication::*;
pub use comparison::*;
pub use complex::*;
pub use control_flow::*;
pub use exponential::*;
#[allow(deprecated)]
pub use functional::*;
pub use linear_algebra::*;
pub use logical::*;
pub use manipulation::*;
pub use miscellaneous::*;
#[allow(deprecated)]
pub use random::*;
pub use rounding::*;
pub use trigonometric::*;
#[allow(deprecated)]
pub use tuples::*;

mlir_enum_attribute!(
    rust_name = AccuracyMode,
    mlir_name = ResultAccuracyMode,
    description = "StableHLO result accuracy mode",
    variants = {
        Default => "DEFAULT",
        Highest => "HIGHEST",
        Tolerance => "TOLERANCE",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// StableHLO [`Attribute`] that represents the requested accuracy for [`HasAccuracy`]s in StableHLO.
/// For information on how to use this attribute and how to interpret its properties, refer to this
/// [official StableHLO RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20241015-result-accuracy.md).
#[derive(Copy, Clone)]
pub struct AccuracyAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> AccuracyAttributeRef<'c, 't> {
    mlir_attribute_field!(absolute_tolerance, ResultAccuracyAttrGetAtol, f64, mlir_prefix = stablehlo);
    mlir_attribute_field!(relative_tolerance, ResultAccuracyAttrGetRtol, f64, mlir_prefix = stablehlo);
    mlir_attribute_field!(units_of_least_precision, ResultAccuracyAttrGetUlps, usize, mlir_prefix = stablehlo);

    /// Returns the [`AccuracyMode`] stored in this [`AccuracyAttributeRef`].
    pub fn mode(&self) -> AccuracyMode {
        unsafe {
            AccuracyModeAttributeRef::from_c_api(stablehloResultAccuracyAttrGetMode(self.handle), self.context)
                .unwrap()
                .value()
        }
    }
}

mlir_subtype_trait_impls!(
    AccuracyAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = ResultAccuracyAttr,
    mlir_prefix = stablehlo,
);

/// Represents the requested accuracy for [`HasAccuracy`]s in StableHLO. Refer to the documentation
/// of [`AccuracyAttributeRef`] and [`Context::stable_hlo_accuracy`] for more information.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[derive(Default)]
pub enum Accuracy {
    #[default]
    Default,
    Highest,
    Tolerance { absolute_tolerance: f64, relative_tolerance: f64, units_of_least_precision: usize },
}

impl<'c, 't> From<AccuracyAttributeRef<'c, 't>> for Accuracy {
    fn from(value: AccuracyAttributeRef<'c, 't>) -> Self {
        match value.mode() {
            AccuracyMode::Default => Accuracy::Default,
            AccuracyMode::Highest => Accuracy::Highest,
            AccuracyMode::Tolerance => Accuracy::Tolerance {
                absolute_tolerance: value.absolute_tolerance(),
                relative_tolerance: value.relative_tolerance(),
                units_of_least_precision: value.units_of_least_precision(),
            },
        }
    }
}

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`AccuracyAttributeRef`] based on the provided [`Accuracy`]. This function delegates to
    /// one of the following functions depending on the value of `accuracy`:
    ///   - [`Context::stable_hlo_default_accuracy`]
    ///   - [`Context::stable_hlo_highest_accuracy`]
    ///   - [`Context::stable_hlo_tolerance_accuracy`]
    pub fn stable_hlo_accuracy<'c>(&'c self, accuracy: Accuracy) -> AccuracyAttributeRef<'c, 't> {
        match accuracy {
            Accuracy::Default => self.stable_hlo_default_accuracy(),
            Accuracy::Highest => self.stable_hlo_highest_accuracy(),
            Accuracy::Tolerance { absolute_tolerance, relative_tolerance, units_of_least_precision } => {
                self.stable_hlo_tolerance_accuracy(absolute_tolerance, relative_tolerance, units_of_least_precision)
            }
        }
    }

    /// Creates a new StableHLO [`AccuracyAttributeRef`] that represents the _default_ result accuracy. This
    /// configuration will result in the fastest implementation of the underlying [`Operation`] with potentially
    /// less accuracy than other configurations.
    pub fn stable_hlo_default_accuracy<'c>(&'c self) -> AccuracyAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            AccuracyAttributeRef::from_c_api(
                stablehloResultAccuracyAttrGet(
                    *self.handle.borrow(),
                    0f64,
                    0f64,
                    0,
                    StringRef::from("DEFAULT").to_c_api(),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`AccuracyAttributeRef`] that represents the _highest_ result accuracy. This
    /// configuration will result in the most accurate implementation of the underlying [`Operation`] at the cost
    /// of potentially slower execution.
    pub fn stable_hlo_highest_accuracy<'c>(&'c self) -> AccuracyAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            AccuracyAttributeRef::from_c_api(
                stablehloResultAccuracyAttrGet(
                    *self.handle.borrow(),
                    0f64,
                    0f64,
                    0,
                    StringRef::from("HIGHEST").to_c_api(),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new StableHLO [`AccuracyAttributeRef`] that represents a _custom_ result accuracy based on the
    /// provided absolute tolerance, relative tolerance, and units of least precision (represented as a factor of the
    /// smallest representable value, ε, of the underlying data type). When using this configuration, the numerical
    /// tolerances will be compared against compiler errors according to the following inequality:
    ///
    /// ```text
    /// abs(expected(x) - actual(x)) <= max(
    ///   abs(expected(x)) * max(relative_tolerance, units_of_least_precision * ε),
    ///   absolute_tolerance,
    /// )
    /// ```
    /// This inequality will be checked against the errors of each implementation of the underlying [`Operation`] and
    /// the one that can satisfy the constraint will be returned. If multiple implementations satisfy the inequality,
    /// the faster implementation will be used. If none of the implementations can meet the requested tolerance, the
    /// compiler will return a compile time error.
    ///
    /// Note that if either the provided `absolute_tolerance` or the provided `relative_tolerance` is negative,
    /// [`Operation`]s that have this attribute will fail verification (i.e., [`Operation::verify`] will fail for them).
    ///
    /// For information refer to this
    /// [official StableHLO RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20241015-result-accuracy.md).
    pub fn stable_hlo_tolerance_accuracy<'c>(
        &'c self,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        units_of_least_precision: usize,
    ) -> AccuracyAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            AccuracyAttributeRef::from_c_api(
                stablehloResultAccuracyAttrGet(
                    *self.handle.borrow(),
                    absolute_tolerance,
                    relative_tolerance,
                    units_of_least_precision as i64,
                    StringRef::from("TOLERANCE").to_c_api(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Name of the [`Attribute`] that is used to store [`HasAccuracy::accuracy`].
pub const RESULT_ACCURACY_ATTRIBUTE: &str = "result_accuracy";

/// Trait used to represent StableHLO [`Operation`]s which can have an associated [`Accuracy`] specified as part of
/// their attributes. This is used by StableHLO transcendental elementwise unary [`Operation`]s.
pub trait HasAccuracy<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the numerical [`Accuracy`] of this [`Operation`]. If no specific numerical accuracy is specified,
    /// this function will return [`Accuracy::Default`].
    fn accuracy(&self) -> Accuracy {
        self.attribute(RESULT_ACCURACY_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AccuracyAttributeRef>())
            .map(|attribute| attribute.into())
            .unwrap_or(Accuracy::Default)
    }
}

/// Name of the [`Attribute`] that is used to store [`HasPadding::padding`].
pub const PADDING_ATTRIBUTE: &str = "padding";

pub trait HasPadding<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the padding of this [`Operation`], if specified. The padding consists of a pair of numbers for each
    /// dimension that is being padded. The first number specifies the amount of padding inserted _before_ the values of
    /// the tensor on that dimension and the second number specifies the amount of padding inserted _after_ the values
    /// of the tensor on that dimension. All padding values default to zero when not specified.
    fn padding(&self) -> Option<Vec<(usize, usize)>> {
        self.attribute(PADDING_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseIntegerElementsAttributeRef>())
            .and_then(|attribute| {
                let attribute_type = attribute.r#type().cast::<TensorTypeRef>();
                let mut padding_values = unsafe { attribute.i64_elements() };
                attribute_type.and_then(|tensor_type| {
                    if let Size::Static(padding_row_count) = tensor_type.dimension(0) {
                        Some(
                            (0..padding_row_count)
                                .flat_map(|_| {
                                    padding_values
                                        .next()
                                        .zip(padding_values.next())
                                        .map(|(before, after)| (before as usize, after as usize))
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            })
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`DenseIntegerElementsAttributeRef`] that represents padding used in some StableHLO operations.
    /// The resulting attribute is compatible with StableHLO [`Operation`]s that are [`HasPadding`].
    pub fn stable_hlo_padding<'c, L: Location<'c, 't>>(
        &'c self,
        padding: &[(usize, usize)],
        location: L,
    ) -> DenseIntegerElementsAttributeRef<'c, 't> {
        let padding_type = self
            .tensor_type(
                self.signless_integer_type(64),
                &[Size::Static(padding.len()), Size::Static(2)],
                None,
                location,
            )
            .unwrap();
        let padding = padding.iter().flat_map(|(before, after)| [*before as i64, *after as i64]).collect::<Vec<_>>();
        self.dense_i64_elements_attribute(padding_type, padding.as_slice()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_accuracy_mode_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), AccuracyMode::Tolerance);
    }

    #[test]
    fn test_accuracy_mode_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        let attribute_2 = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_accuracy_mode(AccuracyMode::Default);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_accuracy_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        test_attribute_display_and_debug(attribute, "#stablehlo.result_accuracy_mode<TOLERANCE>");
    }

    #[test]
    fn test_accuracy_mode_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_accuracy_mode(AccuracyMode::Tolerance);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_accuracy_attribute() {
        let context = Context::new();

        // Test default result accuracy.
        let attribute = context.stable_hlo_default_accuracy();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.mode(), AccuracyMode::Default);
        assert_eq!(attribute.absolute_tolerance(), 0f64);
        assert_eq!(attribute.relative_tolerance(), 0f64);
        assert_eq!(attribute.units_of_least_precision(), 0);

        // Test highest result accuracy.
        let attribute = context.stable_hlo_highest_accuracy();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.mode(), AccuracyMode::Highest);
        assert_eq!(attribute.absolute_tolerance(), 0f64);
        assert_eq!(attribute.relative_tolerance(), 0f64);
        assert_eq!(attribute.units_of_least_precision(), 0);

        // Test tolerance result accuracy.
        let attribute = context.stable_hlo_tolerance_accuracy(1e-5, 1e-3, 2);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.mode(), AccuracyMode::Tolerance);
        assert_eq!(attribute.absolute_tolerance(), 1e-5);
        assert_eq!(attribute.relative_tolerance(), 1e-3);
        assert_eq!(attribute.units_of_least_precision(), 2);
    }

    #[test]
    fn test_accuracy_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_tolerance_accuracy(1e-5, 1e-3, 2);
        let attribute_2 = context.stable_hlo_accuracy(Accuracy::Tolerance {
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-3,
            units_of_least_precision: 2,
        });
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_highest_accuracy();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_tolerance_accuracy(1e-5, 1e-3, 2);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_accuracy_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_tolerance_accuracy(1e-5, 1e-3, 2);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.result_accuracy<\
              atol = 1.000000e-05, \
              rtol = 1.000000e-03, \
              ulps = 2, \
              mode = #stablehlo.result_accuracy_mode<TOLERANCE>\
            >",
        );
    }

    #[test]
    fn test_accuracy_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_tolerance_accuracy(1e-5, 1e-3, 2);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_padding_attribute() {
        let context = Context::new();

        // Test empty padding.
        let padding = vec![];
        let attribute = context.stable_hlo_padding(&padding, context.unknown_location());
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.r#type().cast::<TensorTypeRef>().unwrap().rank(), 2);

        // Test single dimension padding.
        let padding = vec![(1, 2)];
        let attribute = context.stable_hlo_padding(&padding, context.unknown_location());
        assert_eq!(&context, attribute.context());
        let tensor_type = attribute.r#type().cast::<TensorTypeRef>().unwrap();
        assert_eq!(tensor_type.rank(), 2);
        assert_eq!(tensor_type.dimension(0), Size::Static(1));
        assert_eq!(tensor_type.dimension(1), Size::Static(2));

        // Test multi-dimension padding.
        let padding = vec![(1, 2), (3, 4), (5, 6)];
        let attribute = context.stable_hlo_padding(&padding, context.unknown_location());
        assert_eq!(&context, attribute.context());
        let tensor_type = attribute.r#type().cast::<TensorTypeRef>().unwrap();
        assert_eq!(tensor_type.rank(), 2);
        assert_eq!(tensor_type.dimension(0), Size::Static(3));
        assert_eq!(tensor_type.dimension(1), Size::Static(2));
    }

    #[test]
    fn test_padding_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_padding(&[(1, 2), (3, 4)], context.unknown_location());
        let attribute_2 = context.stable_hlo_padding(&[(1, 2), (3, 4)], context.unknown_location());
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_padding(&[(1, 2), (5, 6)], context.unknown_location());
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_padding(&[(1, 2), (3, 4)], context.unknown_location());
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_padding_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_padding(&[(1, 2), (3, 4)], context.unknown_location());
        test_attribute_display_and_debug(attribute, "dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>");
    }

    #[test]
    fn test_padding_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_padding(&[(1, 2), (3, 4)], context.unknown_location());
        test_attribute_casting(attribute);
    }
}
