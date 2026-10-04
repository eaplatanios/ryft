use ryft_xla_sys::bindings::{MlirAttribute, chloRaggedDotDimensionNumbersGet};

use crate::macros::{mlir_attribute_field, mlir_enum_attribute, mlir_subtype_trait_impls};
use crate::{Attribute, Context, DialectHandle, Error};

mlir_enum_attribute!(
    rust_name = Precision,
    mlir_name = Precision,
    description = "CHLO precision for grouped dot products",
    variants = {
        Default => "DEFAULT",
        High => "HIGH",
        Highest => "HIGHEST",
    },
    rust_prefix = chlo,
    mlir_prefix = chlo,
    mlir_dialect_handle_constructor = chlo,
);

/// CHLO [`Attribute`] that models batching, contracting, ragged, and group dimensions for
/// [`RaggedDotOperation`](super::operations::RaggedDotOperation).
#[derive(Copy, Clone)]
pub struct RaggedDotDimensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> RaggedDotDimensionsAttributeRef<'c, 't> {
    mlir_attribute_field!(
        lhs_batching_dimensions,
        RaggedDotDimensionNumbersGetLhsBatchingDimensions,
        [usize],
        mlir_prefix = chlo,
    );

    mlir_attribute_field!(
        rhs_batching_dimensions,
        RaggedDotDimensionNumbersGetRhsBatchingDimensions,
        [usize],
        mlir_prefix = chlo,
    );

    mlir_attribute_field!(
        lhs_contracting_dimensions,
        RaggedDotDimensionNumbersGetLhsContractingDimensions,
        [usize],
        mlir_prefix = chlo,
    );

    mlir_attribute_field!(
        rhs_contracting_dimensions,
        RaggedDotDimensionNumbersGetRhsContractingDimensions,
        [usize],
        mlir_prefix = chlo,
    );

    mlir_attribute_field!(
        lhs_ragged_dimensions,
        RaggedDotDimensionNumbersGetLhsRaggedDimensions,
        [usize],
        mlir_prefix = chlo,
    );

    mlir_attribute_field!(
        rhs_group_dimensions,
        RaggedDotDimensionNumbersGetRhsGroupDimensions,
        [usize],
        mlir_prefix = chlo,
    );
}

mlir_subtype_trait_impls!(
    RaggedDotDimensionsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = RaggedDotDimensionNumbers,
    mlir_prefix = chlo,
);

impl<'t> Context<'t> {
    /// Creates a CHLO [`RaggedDotDimensionsAttributeRef`] owned by this context. Refer to the documentation of
    /// [`RaggedDotOperation`](super::operations::RaggedDotOperation) for the supported ragged-dot modes and dimension
    /// constraints.
    ///
    /// # Parameters
    ///
    ///   - `lhs_batching_dimensions`: Left-Hand Side (LHS) dimensions paired with `rhs_batching_dimensions`
    ///     as batch dimensions.
    ///   - `rhs_batching_dimensions`: Right-Hand Side (RHS) dimensions paired with `lhs_batching_dimensions`
    ///     as batch dimensions.
    ///   - `lhs_contracting_dimensions`: Left-Hand Side (LHS) dimensions paired with `rhs_contracting_dimensions`
    ///     for contraction.
    ///   - `rhs_contracting_dimensions`: Right-Hand Side (RHS) dimensions paired with `lhs_contracting_dimensions`
    ///     for contraction.
    ///   - `lhs_ragged_dimensions`: Left-Hand Side (LHS) dimension that is partitioned according to the `group_sizes`
    ///     operand.
    ///   - `rhs_group_dimensions`: Optional Right-Hand Side (RHS) dimension that indexes ragged groups.
    pub fn chlo_ragged_dot_dimensions<'c>(
        &'c self,
        lhs_batching_dimensions: &[usize],
        rhs_batching_dimensions: &[usize],
        lhs_contracting_dimensions: &[usize],
        rhs_contracting_dimensions: &[usize],
        lhs_ragged_dimensions: &[usize],
        rhs_group_dimensions: &[usize],
    ) -> Result<RaggedDotDimensionsAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::chlo()?)?;
        let convert = |dimensions: &[usize]| dimensions.iter().map(|dimension| *dimension as i64).collect::<Vec<_>>();
        let lhs_batching_dimensions = convert(lhs_batching_dimensions);
        let rhs_batching_dimensions = convert(rhs_batching_dimensions);
        let lhs_contracting_dimensions = convert(lhs_contracting_dimensions);
        let rhs_contracting_dimensions = convert(rhs_contracting_dimensions);
        let lhs_ragged_dimensions = convert(lhs_ragged_dimensions);
        let rhs_group_dimensions = convert(rhs_group_dimensions);
        // This constructor can mutate the context by adding an entry to its attribute-uniquing table. We use an
        // immutable borrow because MLIR contexts are not thread-safe and callers need to keep using this context
        // while assembling the surrounding operation, matching the other MLIR attribute constructors in this crate.
        unsafe {
            RaggedDotDimensionsAttributeRef::from_c_api(
                chloRaggedDotDimensionNumbersGet(
                    *self.handle.borrow(),
                    lhs_batching_dimensions.len().cast_signed(),
                    lhs_batching_dimensions.as_ptr(),
                    rhs_batching_dimensions.len().cast_signed(),
                    rhs_batching_dimensions.as_ptr(),
                    lhs_contracting_dimensions.len().cast_signed(),
                    lhs_contracting_dimensions.as_ptr(),
                    rhs_contracting_dimensions.len().cast_signed(),
                    rhs_contracting_dimensions.as_ptr(),
                    lhs_ragged_dimensions.len().cast_signed(),
                    lhs_ragged_dimensions.as_ptr(),
                    rhs_group_dimensions.len().cast_signed(),
                    rhs_group_dimensions.as_ptr(),
                ),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::chlo_ragged_dot_dimensions`"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_precision_attribute() {
        let context = Context::new();
        let attribute = context.chlo_precision(Precision::Highest).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value().unwrap(), Precision::Highest);
    }

    #[test]
    fn test_precision_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.chlo_precision(Precision::Highest).unwrap();
        let attribute_2 = context.chlo_precision(Precision::Highest).unwrap();
        assert_eq!(attribute_1, attribute_2);
        let attribute_2 = context.chlo_precision(Precision::High).unwrap();
        assert_ne!(attribute_1, attribute_2);
        let other_context = Context::new();
        let attribute_2 = other_context.chlo_precision(Precision::Highest).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_precision_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.chlo_precision(Precision::Highest).unwrap();
        test_attribute_display_and_debug(attribute, "#chlo<precision HIGHEST>");
    }

    #[test]
    fn test_precision_attribute_casting() {
        let context = Context::new();
        let attribute = context.chlo_precision(Precision::Highest).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_ragged_dot_dimensions_attribute() {
        let context = Context::new();
        let attribute = context.chlo_ragged_dot_dimensions(&[0, 1], &[2, 3], &[4], &[5], &[6], &[7]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.lhs_batching_dimensions(), vec![0, 1]);
        assert_eq!(attribute.rhs_batching_dimensions(), vec![2, 3]);
        assert_eq!(attribute.lhs_contracting_dimensions(), vec![4]);
        assert_eq!(attribute.rhs_contracting_dimensions(), vec![5]);
        assert_eq!(attribute.lhs_ragged_dimensions(), vec![6]);
        assert_eq!(attribute.rhs_group_dimensions(), vec![7]);
    }

    #[test]
    fn test_ragged_dot_dimensions_attribute_equality() {
        let context = Context::new();

        // Identical attributes in one context are uniqued.
        let attribute_1 = context.chlo_ragged_dot_dimensions(&[0], &[1], &[2], &[3], &[4], &[5]).unwrap();
        let attribute_2 = context.chlo_ragged_dot_dimensions(&[0], &[1], &[2], &[3], &[4], &[5]).unwrap();
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.chlo_ragged_dot_dimensions(&[1], &[0], &[3], &[2], &[5], &[4]).unwrap();
        assert_ne!(attribute_1, attribute_2);

        let other_context = Context::new();
        let attribute_2 = other_context.chlo_ragged_dot_dimensions(&[0], &[1], &[2], &[3], &[4], &[5]).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_ragged_dot_dimensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.chlo_ragged_dot_dimensions(&[0], &[1], &[2], &[3], &[4], &[5]).unwrap();
        test_attribute_display_and_debug(
            attribute,
            "#chlo.ragged_dot<\
              lhs_batching_dimensions = [0], \
              rhs_batching_dimensions = [1], \
              lhs_contracting_dimensions = [2], \
              rhs_contracting_dimensions = [3], \
              lhs_ragged_dimensions = [4], \
              rhs_group_dimensions = [5]\
            >",
        );
    }

    #[test]
    fn test_ragged_dot_dimensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.chlo_ragged_dot_dimensions(&[0], &[1], &[2], &[3], &[4], &[5]).unwrap();
        test_attribute_casting(attribute);
    }
}
