use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, DataType};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::programs::{ProgramError, TypeError, Typed};

/// Canonical operation name for [`NotOperation`].
pub const NOT_OPERATION_NAME: &str = "not";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise negation (i.e., `!input`) of one value while
    /// preserving its type metadata. This operation covers both logical (i.e., Boolean) and bitwise negation. The two
    /// semantics coincide on Boolean element types, and StableHLO's [`not`](https://openxla.org/stablehlo/spec#not)
    /// operation likewise serves both.
    NotOperation,
    NOT_OPERATION_NAME,
    Not,
    not,
);

impl_differentiable_elementwise_operation!(@non_differentiable NotOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise negation capability. [`Not`] is the fallible Ryft counterpart to [`std::ops::Not`]
    /// that [`NotOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something
    /// goes wrong (e.g., when a value's data type does not support negation), instead of panicking. Value types
    /// additionally provide [`std::ops::Not`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Not,
    /// Computes [`NotOperation`] elementwise for this value.
    not,
    NotOperation,
);

define_tracer_operator!(@unary std::ops::Not, not, NotOperation, "`not` operation failed");

/// Implements [`Not`] for one host primitive type as logical not for `bool` and bitwise not for integers, matching the
/// reference backends and StableHLO.
macro_rules! impl_capability_for_primitive {
    // The `!` operator is logical for `bool` and bitwise for integer primitives, and cannot fail for either.
    ($type:ty) => {
        impl Not for $type {
            fn not(&self) -> Result<Self, ProgramError> {
                Ok(!*self)
            }
        }
    };
}

impl_capability_for_primitive!(bool);
impl_capability_for_primitive!(i8);
impl_capability_for_primitive!(i16);
impl_capability_for_primitive!(i32);
impl_capability_for_primitive!(i64);
impl_capability_for_primitive!(i128);
impl_capability_for_primitive!(isize);
impl_capability_for_primitive!(u8);
impl_capability_for_primitive!(u16);
impl_capability_for_primitive!(u32);
impl_capability_for_primitive!(u64);
impl_capability_for_primitive!(u128);
impl_capability_for_primitive!(usize);

impl Not for Array {
    fn not(&self) -> Result<Self, ProgramError> {
        let mask = match self.r#type().data_type() {
            DataType::Boolean | DataType::I1 | DataType::U1 => 0b1,
            DataType::I2 | DataType::U2 => 0b11,
            DataType::I4 | DataType::U4 => 0b1111,
            data_type if data_type.is_integer() => u8::MAX,
            data_type => {
                return Err(TypeError::invalid(format!(
                    "cannot apply `{NOT_OPERATION_NAME}` to an array of element data type `{data_type}`"
                ))
                .into());
            }
        };
        let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let input_bytes = self.storage_bytes();
        let mut bytes = vec![0; addressing.storage_byte_len()];
        for element in 0..addressing.element_count() {
            for byte in addressing.byte_range_for_flat_index(element) {
                bytes[byte] = !input_bytes[byte] & mask;
            }
        }
        // Masking retains valid Boolean and sub-byte encodings; full-width integers admit every bit pattern, and
        // zero-initialization preserves all layout holes and padding.
        Ok(Self::new_unchecked(self.r#type().into_owned(), Arc::new(bytes)))
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    fn not(self) -> Self::Output {
        Not::not(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Layout, StridedLayout, i2};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        TransposableOperation, TranspositionContext,
    };
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero, ProgramError};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_not() {
        assert_eq!(NotOperation::<ArrayType>::new().to_string(), "not");
    }

    #[test]
    fn test_not_type_inference() {
        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @unary,
            operation = NotOperation,
            cases = [{
                input_data_types = [DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_not_interpretation() {
        // Check the operation-specific eager value semantics.
        assert_eq!((!Array::vector(vec![true, false, true]).unwrap()).elements::<bool>(), Ok(vec![false, true, false]));
    }

    #[test]
    fn test_not_partial_evaluation() {
        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = NotOperation::new(),
            inputs = [Array::scalar(true).unwrap()],
            expected = Array::scalar(false).unwrap(),
        );
    }

    #[test]
    fn test_not_batching() {
        // Check mapped and replicated batching behavior.
        check_operation_batching!(
            @exact,
            operation = NotOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::vector(vec![true, false]).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]).unwrap())],
                },
                {
                    inputs = [(@replicated, Array::scalar(true).unwrap())],
                    outputs = [(@replicated, Array::scalar(false).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_not_differentiation() {
        let outputs = NotOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(Array::scalar(true).unwrap()).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(false).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );
    }

    #[test]
    fn test_not_transposition() {
        // Program transposition elides zero-space Boolean cotangents, so check the primitive's rejection directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            NotOperation::<ArrayType>::new().transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                ],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `not` is not transposable",
        ));
    }

    #[test]
    fn test_not_for_primitives() {
        assert_eq!(Not::not(&true), Ok(false));
        assert_eq!(Not::not(&0b1100_u8), Ok(0b1111_0011));
    }

    #[test]
    fn test_not_for_array() {
        let left = Array::vector(vec![true, true, false, false]).unwrap();
        assert_eq!(left.not().unwrap(), Array::vector(vec![false, false, true, true]).unwrap());
        assert_eq!(Array::vector(vec![0x00ff_i16, -1]).unwrap().not().unwrap().elements::<i16>(), Ok(vec![-256, 0]));
        // Sub-byte negation complements only the declared low bits, retaining a valid sign-extended encoding.
        let signed_sub_byte = Array::vector(vec![i2::MIN, i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::MAX]).unwrap();
        assert_eq!(
            signed_sub_byte.not().unwrap().elements::<i2>(),
            Ok(vec![i2::MAX, i2::new(0).unwrap(), i2::new(-1).unwrap(), i2::MIN]),
        );
        // Physical layouts are traversed through addressing, so holes stay zero rather than being complemented.
        let strided_type =
            ArrayType::new_static(DataType::Boolean, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let strided = Array::new(strided_type.clone(), vec![1, 0, 0]).unwrap().not().unwrap();
        assert_eq!(strided.r#type().as_ref(), &strided_type);
        assert_eq!(strided.storage_bytes(), [0, 0, 1]);
        assert_eq!(strided.elements::<bool>(), Ok(vec![false, true]));
        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(!left.clone(), Array::vector(vec![false, false, true, true]).unwrap());
    }
}
