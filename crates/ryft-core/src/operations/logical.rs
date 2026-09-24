//! Operations that compute logical and bitwise functions of Boolean and integer values. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`AndOperation`]) together with a value capability trait (e.g., [`And`])
//! whose functions apply it to eager [`Array`]s and traced values alike, so the same code executes immediately or
//! records into a program depending on the value it runs on. Each capability is the fallible counterpart of an
//! [`std::ops`] operator, which values additionally implement as panicking sugar:
//!
//!   - [`Not`] computes the elementwise negation of one value (i.e., `!input`).
//!   - [`And`], [`Or`], and [`Xor`] compute the elementwise conjunction, disjunction, and exclusive disjunction of two
//!     values (i.e., `left & right`, `left | right`, and `left ^ right`), broadcasting them to a common shape.
//!
//! Every operation is logical on Boolean elements and bitwise on integer elements, since the two semantics
//! coincide on Booleans, as they do for StableHLO's [`not`](https://openxla.org/stablehlo/spec#not),
//! [`and`](https://openxla.org/stablehlo/spec#and), [`or`](https://openxla.org/stablehlo/spec#or),
//! and [`xor`](https://openxla.org/stablehlo/spec#xor). Sub-byte integers only ever involve their low
//! [`bit_width`](DataType::bit_width) bits. None of the operations is differentiable; their derivatives
//! are structural zeros, and cannot be transposed.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{And, Array, Not, Or, ProgramError, Xor};
//! # fn main() -> Result<(), ProgramError> {
//! let left = Array::vector(vec![true, true, false, false])?;
//! let right = Array::vector(vec![true, false, true, false])?;
//! assert_eq!(left.and(&right)?, Array::vector(vec![true, false, false, false])?);
//! assert_eq!(left.xor(&right)?.not()?, Array::vector(vec![true, false, false, true])?);
//! assert_eq!(Array::scalar(0b1100u8)?.or(&Array::scalar(0b1010u8)?)?, Array::scalar(0b1110u8)?);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, ArrayType, Broadcastable, DataType};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::programs::{ProgramError, TypeError, Typed};

/// Implements one logical capability for one host primitive type through the corresponding Rust operator, which is
/// logical for `bool` and bitwise for integers, matching the reference backends and StableHLO.
macro_rules! impl_capability_for_primitive {
    // Implements a unary capability through a prefix operator, which cannot fail for any primitive.
    (@unary $capability:ident, $function:ident, $operator:tt, $type:ty) => {
        impl $capability for $type {
            fn $function(&self) -> Result<Self, ProgramError> {
                Ok($operator *self)
            }
        }
    };

    // Implements a binary capability through an infix operator, which cannot fail for any primitive.
    (@binary $capability:ident, $function:ident, $operator:tt, $type:ty) => {
        impl $capability for $type {
            fn $function(&self, rhs: &Self) -> Result<Self, ProgramError> {
                Ok(*self $operator *rhs)
            }
        }
    };
}

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
    /// Value-level elementwise negation capability. [`Not`] is the fallible Ryft counterpart to [`std::ops::Not`] that
    /// [`NotOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong (e.g., when a
    /// value's data type does not support negation), instead of panicking. Value types additionally provide
    /// [`std::ops::Not`] as ergonomic (albeit panicking) sugar layered on top of this capability.
    Not,
    /// Computes [`NotOperation`] elementwise for this value.
    not,
    NotOperation,
);

define_tracer_operator!(@unary std::ops::Not, not, NotOperation, "`not` operation failed");

impl_capability_for_primitive!(@unary Not, not, !, bool);
impl_capability_for_primitive!(@unary Not, not, !, i8);
impl_capability_for_primitive!(@unary Not, not, !, i16);
impl_capability_for_primitive!(@unary Not, not, !, i32);
impl_capability_for_primitive!(@unary Not, not, !, i64);
impl_capability_for_primitive!(@unary Not, not, !, i128);
impl_capability_for_primitive!(@unary Not, not, !, isize);
impl_capability_for_primitive!(@unary Not, not, !, u8);
impl_capability_for_primitive!(@unary Not, not, !, u16);
impl_capability_for_primitive!(@unary Not, not, !, u32);
impl_capability_for_primitive!(@unary Not, not, !, u64);
impl_capability_for_primitive!(@unary Not, not, !, u128);
impl_capability_for_primitive!(@unary Not, not, !, usize);

impl Not for Array {
    fn not(&self) -> Result<Self, ProgramError> {
        let mask = match self.r#type().data_type() {
            DataType::Boolean => {
                // A Boolean occupies a whole byte, but only its lowest bit encodes its value.
                0b1
            }
            data_type if data_type.is_integer() => {
                // An integer's value occupies the low `bit_width` bits of each of its bytes, which is all of them
                // for any integer that is at least one byte wide.
                u8::MAX >> (8 - data_type.bit_width().min(8))
            }
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

        // Masking retains valid Boolean and sub-byte encodings; full-width integers admit every bit pattern,
        // and zero-initialization preserves all layout holes and padding.
        Ok(Self::new_unchecked(self.r#type().into_owned(), Arc::new(bytes)))
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Not::not(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Canonical operation name for [`AndOperation`].
pub const AND_OPERATION_NAME: &str = "and";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise conjunction (i.e., `left & right`) of two
    /// values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise conjunction. The two semantics coincide on Boolean element types, and StableHLO's
    /// [`and`](https://openxla.org/stablehlo/spec#and) operation likewise serves both.
    AndOperation,
    AND_OPERATION_NAME,
    And,
    and,
);

impl_differentiable_elementwise_operation!(@non_differentiable AndOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise conjunction capability. [`And`] is the fallible Ryft counterpart to [`std::ops::BitAnd`]
    /// that [`AndOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong (e.g., when a
    /// value's data type does not support conjunction), instead of panicking. Value types additionally provide
    /// [`std::ops::BitAnd`] as ergonomic (albeit panicking) sugar layered on top of this capability.
    And,
    /// Computes [`AndOperation`] elementwise for this value and `rhs`.
    and(rhs),
    AndOperation,
);

define_tracer_operator!(@binary std::ops::BitAnd, bitand, capability = And, method = and);

impl_capability_for_primitive!(@binary And, and, &, bool);
impl_capability_for_primitive!(@binary And, and, &, i8);
impl_capability_for_primitive!(@binary And, and, &, i16);
impl_capability_for_primitive!(@binary And, and, &, i32);
impl_capability_for_primitive!(@binary And, and, &, i64);
impl_capability_for_primitive!(@binary And, and, &, i128);
impl_capability_for_primitive!(@binary And, and, &, isize);
impl_capability_for_primitive!(@binary And, and, &, u8);
impl_capability_for_primitive!(@binary And, and, &, u16);
impl_capability_for_primitive!(@binary And, and, &, u32);
impl_capability_for_primitive!(@binary And, and, &, u64);
impl_capability_for_primitive!(@binary And, and, &, u128);
impl_capability_for_primitive!(@binary And, and, &, usize);

impl And for Array {
    #[inline]
    fn and(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, AND_OPERATION_NAME, |left, right| left & right)
    }
}

impl std::ops::BitAnd for Array {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        And::and(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Canonical operation name for [`OrOperation`].
pub const OR_OPERATION_NAME: &str = "or";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise disjunction (i.e., `left | right`) of two values
    /// and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e., Boolean) and
    /// bitwise disjunction. The two semantics coincide on Boolean element types, and StableHLO's
    /// [`or`](https://openxla.org/stablehlo/spec#or) operation likewise serves both.
    OrOperation,
    OR_OPERATION_NAME,
    Or,
    or,
);

impl_differentiable_elementwise_operation!(@non_differentiable OrOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise disjunction capability. [`Or`] is the fallible Ryft counterpart to [`std::ops::BitOr`]
    /// that [`OrOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong (e.g., when a
    /// value's data type does not support disjunction), instead of panicking. Value types additionally provide
    /// [`std::ops::BitOr`] as ergonomic (albeit panicking) sugar layered on top of this capability.
    Or,
    /// Computes [`OrOperation`] elementwise for this value and `rhs`.
    or(rhs),
    OrOperation,
);

define_tracer_operator!(@binary std::ops::BitOr, bitor, capability = Or, method = or);

impl_capability_for_primitive!(@binary Or, or, |, bool);
impl_capability_for_primitive!(@binary Or, or, |, i8);
impl_capability_for_primitive!(@binary Or, or, |, i16);
impl_capability_for_primitive!(@binary Or, or, |, i32);
impl_capability_for_primitive!(@binary Or, or, |, i64);
impl_capability_for_primitive!(@binary Or, or, |, i128);
impl_capability_for_primitive!(@binary Or, or, |, isize);
impl_capability_for_primitive!(@binary Or, or, |, u8);
impl_capability_for_primitive!(@binary Or, or, |, u16);
impl_capability_for_primitive!(@binary Or, or, |, u32);
impl_capability_for_primitive!(@binary Or, or, |, u64);
impl_capability_for_primitive!(@binary Or, or, |, u128);
impl_capability_for_primitive!(@binary Or, or, |, usize);

impl Or for Array {
    #[inline]
    fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, OR_OPERATION_NAME, |left, right| left | right)
    }
}

impl std::ops::BitOr for Array {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Or::or(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

/// Canonical operation name for [`XorOperation`].
pub const XOR_OPERATION_NAME: &str = "xor";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise exclusive disjunction (i.e., `left ^ right`) of
    /// two values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise exclusive disjunction: the two semantics coincide on Boolean element types, and
    /// StableHLO's [`xor`](https://openxla.org/stablehlo/spec#xor) operation likewise serves both.
    XorOperation,
    XOR_OPERATION_NAME,
    Xor,
    xor,
);

impl_differentiable_elementwise_operation!(@non_differentiable XorOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise exclusive-disjunction capability. [`Xor`] is the fallible Ryft counterpart to
    /// [`std::ops::BitXor`] that [`XorOperation`] interprets through, surfacing a [`ProgramError`] when something goes
    /// wrong (e.g., when a value's data type does not support exclusive disjunction), instead of panicking. Value types
    /// additionally provide [`std::ops::BitXor`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Xor,
    /// Computes [`XorOperation`] elementwise for this value and `rhs`.
    xor(rhs),
    XorOperation,
);

define_tracer_operator!(@binary std::ops::BitXor, bitxor, capability = Xor, method = xor);

impl_capability_for_primitive!(@binary Xor, xor, ^, bool);
impl_capability_for_primitive!(@binary Xor, xor, ^, i8);
impl_capability_for_primitive!(@binary Xor, xor, ^, i16);
impl_capability_for_primitive!(@binary Xor, xor, ^, i32);
impl_capability_for_primitive!(@binary Xor, xor, ^, i64);
impl_capability_for_primitive!(@binary Xor, xor, ^, i128);
impl_capability_for_primitive!(@binary Xor, xor, ^, isize);
impl_capability_for_primitive!(@binary Xor, xor, ^, u8);
impl_capability_for_primitive!(@binary Xor, xor, ^, u16);
impl_capability_for_primitive!(@binary Xor, xor, ^, u32);
impl_capability_for_primitive!(@binary Xor, xor, ^, u64);
impl_capability_for_primitive!(@binary Xor, xor, ^, u128);
impl_capability_for_primitive!(@binary Xor, xor, ^, usize);

impl Xor for Array {
    #[inline]
    fn xor(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, XOR_OPERATION_NAME, |left, right| left ^ right)
    }
}

impl std::ops::BitXor for Array {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Xor::xor(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Array {
    /// Applies a binary logical or bitwise operation specified by `function` directly to validated Boolean or integer
    /// element bytes. Since bitwise operations act independently on every bit, their result is independent of integer
    /// signedness and host endianness. Logical Boolean encodings use the same `0` and `1` bitwise truth tables.
    pub(crate) fn binary_logical(
        &self,
        rhs: &Self,
        operation: &str,
        function: impl Fn(u8, u8) -> u8,
    ) -> Result<Self, ProgramError> {
        let left_data_type = self.r#type().data_type();
        let right_data_type = rhs.r#type().data_type();
        if left_data_type != right_data_type || !(left_data_type.is_boolean() || left_data_type.is_integer()) {
            return Err(TypeError::invalid(format!(
                "cannot apply `{operation}` to arrays of element data types `{left_data_type}` and `{right_data_type}`",
            ))
            .into());
        }

        ArrayType::check_matching_manual_variation(operation, &[self.r#type().as_ref(), rhs.r#type().as_ref()])?;

        let output_type = Broadcastable::broadcast(self.r#type().as_ref(), rhs.r#type().as_ref())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_shape = output_type.static_shape().unwrap();
        let left_shape = self.r#type().static_shape().unwrap();
        let right_shape = rhs.r#type().static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let left_strides = left_shape.row_major_strides();
        let right_strides = right_shape.row_major_strides();
        let left_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let right_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let left_bytes = self.storage_bytes();
        let right_bytes = rhs.storage_bytes();
        let element_byte_width = output_addressing.element_byte_width();
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let left_range = left_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &left_shape,
                &left_strides,
            ));
            let right_range = right_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &right_shape,
                &right_strides,
            ));
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            for byte in 0..element_byte_width {
                output_bytes[output_range.start + byte] =
                    function(left_bytes[left_range.start + byte], right_bytes[right_range.start + byte]);
            }
        }

        // Valid inputs, bitwise closure outputs, and zero-initialized unoccupied storage preserve every `Array`
        // encoding invariant without a second validation traversal.
        Ok(Self::new_unchecked(output_type, Arc::new(output_bytes)))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Layout, LogicalMesh, MeshAxis, MeshAxisType, Sharding,
        StridedLayout, i2, u4,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        TransposableOperation, TranspositionContext,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero};
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
        assert_eq!(
            NotOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![true, false]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![false, true]).unwrap()]),
        );
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
        // Negation is non-differentiable, so its tangent is a structural zero in the zero tangent space.
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
                &[PartialValue::Unknown(ArrayType::scalar(DataType::Boolean))],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `not` is not transposable",
        ));
    }

    #[test]
    fn test_not_for_primitives() {
        // The operator is logical for `bool` and bitwise for integer primitives.
        assert_eq!(Not::not(&true), Ok(false));
        assert_eq!(Not::not(&0b1100u8), Ok(0b1111_0011));
    }

    #[test]
    fn test_not_for_array() {
        let left = Array::vector(vec![true, true, false, false]).unwrap();
        assert_eq!(left.not().unwrap(), Array::vector(vec![false, false, true, true]).unwrap());
        assert_eq!(Array::vector(vec![0x00ffi16, -1]).unwrap().not().unwrap().elements::<i16>(), Ok(vec![-256, 0]));

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

        // Floating-point elements have no negation.
        assert!(matches!(
            Array::vector(vec![1.0]).unwrap().not(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot apply `not` to an array of element data type `f64`",
        ));

        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(!left.clone(), left.not().unwrap());
    }

    #[test]
    fn test_and() {
        assert_eq!(AndOperation::<ArrayType>::new().to_string(), "and");
    }

    #[test]
    fn test_and_type_inference() {
        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @binary,
            operation = AndOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_and_interpretation() {
        assert_eq!(
            AndOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::vector(vec![true, true, false, false]).unwrap(),
                    Array::vector(vec![true, false, true, false]).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![true, false, false, false]).unwrap()]),
        );
    }

    #[test]
    fn test_and_partial_evaluation() {
        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = AndOperation::new(),
            inputs = [Array::scalar(true).unwrap(), Array::scalar(false).unwrap()],
            expected = Array::scalar(false).unwrap(),
        );
    }

    #[test]
    fn test_and_batching() {
        // Check both mixed mapped/replicated operand orderings.
        check_operation_batching!(
            @exact,
            operation = AndOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                        (@replicated, Array::scalar(true).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(false).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, false]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_and_differentiation() {
        // Conjunction is non-differentiable, so its tangent is a structural zero in the zero tangent space.
        let outputs = AndOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(true).unwrap()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(false).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(false).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );
    }

    #[test]
    fn test_and_transposition() {
        // Program transposition elides zero-space Boolean cotangents, so check the primitive's rejection directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            AndOperation::<ArrayType>::new().transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                ],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `and` is not transposable",
        ));
    }

    #[test]
    fn test_and_for_primitives() {
        // The operator is logical for `bool` and bitwise for integer primitives.
        assert_eq!(And::and(&true, &false), Ok(false));
        assert_eq!(And::and(&0b1100u8, &0b1010), Ok(0b1000));
    }

    #[test]
    fn test_and_for_array() {
        // Boolean elements follow the logical truth table and integer elements combine bitwise.
        let left = Array::vector(vec![true, true, false, false]).unwrap();
        let right = Array::vector(vec![true, false, true, false]).unwrap();
        assert_eq!(left.and(&right), Array::vector(vec![true, false, false, false]));
        assert_eq!(Array::scalar(0b1100u8).unwrap().and(&Array::scalar(0b1010u8).unwrap()), Array::scalar(0b1000u8));

        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(left.clone() & right.clone(), left.and(&right).unwrap());
    }

    #[test]
    fn test_or() {
        assert_eq!(OrOperation::<ArrayType>::new().to_string(), "or");
    }

    #[test]
    fn test_or_type_inference() {
        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @binary,
            operation = OrOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_or_interpretation() {
        assert_eq!(
            OrOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::vector(vec![true, true, false, false]).unwrap(),
                    Array::vector(vec![true, false, true, false]).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![true, true, true, false]).unwrap()]),
        );
    }

    #[test]
    fn test_or_partial_evaluation() {
        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = OrOperation::new(),
            inputs = [Array::scalar(true).unwrap(), Array::scalar(false).unwrap()],
            expected = Array::scalar(true).unwrap(),
        );
    }

    #[test]
    fn test_or_batching() {
        // Check mixed mapped/replicated batching.
        check_operation_batching!(
            @exact,
            operation = OrOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                    (@replicated, Array::scalar(false).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_or_differentiation() {
        // Disjunction is non-differentiable, so its tangent is a structural zero in the zero tangent space.
        let outputs = OrOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(true).unwrap()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(false).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(true).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );
    }

    #[test]
    fn test_or_transposition() {
        // Program transposition elides zero-space Boolean cotangents, so check the primitive's rejection directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            OrOperation::<ArrayType>::new().transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                ],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `or` is not transposable",
        ));
    }

    #[test]
    fn test_or_for_primitives() {
        // The operator is logical for `bool` and bitwise for integer primitives.
        assert_eq!(Or::or(&true, &false), Ok(true));
        assert_eq!(Or::or(&0b1100u8, &0b1010), Ok(0b1110));
    }

    #[test]
    fn test_or_for_array() {
        // Boolean elements follow the logical truth table and integer elements combine bitwise.
        let left = Array::vector(vec![true, true, false, false]).unwrap();
        let right = Array::vector(vec![true, false, true, false]).unwrap();
        assert_eq!(left.or(&right), Array::vector(vec![true, true, true, false]));
        assert_eq!(Array::scalar(0b1100u8).unwrap().or(&Array::scalar(0b1010u8).unwrap()), Array::scalar(0b1110u8));

        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(left.clone() | right.clone(), left.or(&right).unwrap());
    }

    #[test]
    fn test_xor() {
        assert_eq!(XorOperation::<ArrayType>::new().to_string(), "xor");
    }

    #[test]
    fn test_xor_type_inference() {
        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @binary,
            operation = XorOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_xor_interpretation() {
        assert_eq!(
            XorOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::vector(vec![true, true, false, false]).unwrap(),
                    Array::vector(vec![true, false, true, false]).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![false, true, true, false]).unwrap()]),
        );
    }

    #[test]
    fn test_xor_partial_evaluation() {
        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = XorOperation::new(),
            inputs = [Array::scalar(true).unwrap(), Array::scalar(false).unwrap()],
            expected = Array::scalar(true).unwrap(),
        );
    }

    #[test]
    fn test_xor_batching() {
        // Check mixed mapped/replicated batching.
        check_operation_batching!(
            @exact,
            operation = XorOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                    (@replicated, Array::scalar(true).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_xor_differentiation() {
        // Exclusive disjunction is non-differentiable, so its tangent is a structural zero in the zero tangent space.
        let outputs = XorOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(true).unwrap()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(false).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(true).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero)),
        );
    }

    #[test]
    fn test_xor_transposition() {
        // Program transposition elides zero-space Boolean cotangents, so check the primitive's rejection directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            XorOperation::<ArrayType>::new().transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
                ],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `xor` is not transposable",
        ));
    }

    #[test]
    fn test_xor_for_primitives() {
        // The operator is logical for `bool` and bitwise for integer primitives.
        assert_eq!(Xor::xor(&true, &false), Ok(true));
        assert_eq!(Xor::xor(&0b1100u8, &0b1010), Ok(0b0110));
    }

    #[test]
    fn test_xor_for_array() {
        // Boolean elements follow the logical truth table and integer elements combine bitwise.
        let left = Array::vector(vec![true, true, false, false]).unwrap();
        let right = Array::vector(vec![true, false, true, false]).unwrap();
        assert_eq!(left.xor(&right), Array::vector(vec![false, true, true, false]));
        assert_eq!(Array::scalar(0b1100u8).unwrap().xor(&Array::scalar(0b1010u8).unwrap()), Array::scalar(0b0110u8));

        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(left.clone() ^ right.clone(), left.xor(&right).unwrap());
    }

    #[test]
    fn test_array_binary_logical() {
        let xor = |left: u8, right: u8| left ^ right;

        // General NumPy-style broadcasting maps each input coordinate into the common output shape.
        assert_eq!(
            Array::matrix(2, 1, vec![true, false]).unwrap().binary_logical(
                &Array::matrix(1, 3, vec![true, false, true]).unwrap(),
                "xor",
                xor
            ),
            Array::matrix(2, 3, vec![false, true, false, true, false, true]),
        );

        // Sub-byte encodings stay valid because combining two values bitwise never sets their unused high bits.
        assert_eq!(
            Array::scalar(u4::new(0b1100).unwrap())
                .unwrap()
                .binary_logical(&Array::scalar(u4::new(0b1010).unwrap()).unwrap(), "xor", xor)
                .unwrap()
                .elements::<u4>(),
            Ok(vec![u4::new(0b0110).unwrap()]),
        );

        // Physical layouts are traversed through addressing, so the output's holes stay zero.
        let strided_type =
            ArrayType::new_static(DataType::Boolean, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let strided = Array::new(strided_type.clone(), vec![1, 0, 0])
            .unwrap()
            .binary_logical(&Array::new(strided_type.clone(), vec![0, 0, 1]).unwrap(), "xor", xor)
            .unwrap();
        assert_eq!(strided.r#type().as_ref(), &strided_type);
        assert_eq!(strided.storage_bytes(), [1, 0, 1]);

        // Operands must share a Boolean or integer element type.
        for (left, right, data_types) in [
            (Array::scalar(1.0).unwrap(), Array::scalar(0.0).unwrap(), "`f64` and `f64`"),
            (Array::scalar(true).unwrap(), Array::scalar(1u8).unwrap(), "`bool` and `u8`"),
        ] {
            assert!(matches!(
                left.binary_logical(&right, "xor", xor),
                Err(ProgramError::Type(TypeError::Invalid { message }))
                    if message == format!("cannot apply `xor` to arrays of element data types {data_types}"),
            ));
        }

        // Eager inputs cannot implicitly acquire manual variation, including when no elements are traversed.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_type = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let varying = Array::from_elements(varying_type.clone(), &[true]).unwrap();
        let invariant = Array::scalar(false).unwrap();
        let expected = Err(TypeError::invalid(
            "`xor` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack an \
             axis, as `align_manual_variation` does",
        )
        .into());
        assert_eq!(invariant.binary_logical(&varying, "xor", xor), expected);
        assert_eq!(varying.binary_logical(&invariant, "xor", xor), expected);
        assert_eq!(varying.binary_logical(&varying, "xor", xor), Array::from_elements(varying_type, &[false]));
        let empty_type = ArrayType::new_static(DataType::Boolean, [0]);
        let empty = Array::from_elements::<bool>(empty_type.clone(), &[]).unwrap();
        let varying_empty = Array::from_elements::<bool>(
            empty_type
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[],
        )
        .unwrap();
        assert_eq!(empty.binary_logical(&varying_empty, "xor", xor), expected);
    }
}
