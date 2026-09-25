//! Operations that select elementwise minima and maxima of numeric values. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`MinOperation`]) together with a value capability trait (e.g., [`Min`])
//! whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code executes
//! immediately or records into a program depending on the value it runs on:
//!
//!   - [`Min`] and [`Max`] select the smaller and the larger of two values (i.e., `(a, b) ↦ min(a, b)` and
//!     `(a, b) ↦ max(a, b)`).
//!   - [`Clamp`] restricts a value to an inclusive interval as the composition `max(lower, min(input, upper))`, which
//!     is how StableHLO defines [`clamp`](https://openxla.org/stablehlo/spec#clamp). It is implemented for every value
//!     that supports [`Min`] and [`Max`] and stages no operation of its own.
//!
//! Inputs promote to a common numeric element type and broadcast, as for StableHLO's
//! [`minimum`](https://openxla.org/stablehlo/spec#minimum) and [`maximum`](https://openxla.org/stablehlo/spec#maximum),
//! and Boolean inputs are rejected. Real floating-point extrema propagate NaNs and order negative zero below positive
//! zero. Complex extrema compare real parts first and imaginary parts second, selecting one whole input, and ties and
//! unordered comparisons select the right input. Array inputs that carry partial sums over unreduced mesh axes are
//! rejected, and the reduced-axis markers of the inputs must agree.
//!
//! The tangent of an extremum is the tangent of the selected input. Real ties select the tangent of the left input,
//! whereas complex ties select the tangent of the right input, consistently with the primal selection. Neither
//! operation is linear, so reverse-mode differentiation transposes its linearization instead.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Clamp, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![-2.0f64, 0.5, 3.0])?;
//! let output = input.clamp(&Array::scalar(-1.0)?, &Array::scalar(1.0)?)?;
//! assert_eq!(output, Array::vector(vec![-1.0, 0.5, 1.0])?);
//! # Ok(())
//! # }
//! ```

use crate::arrays::ArrayElement;
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation,
};
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::complex::{Imaginary, Real};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::programs::{ProgramError, Type, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MinOperation`].
pub const MIN_OPERATION_NAME: &str = "min";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise minimum of two numeric values, promoting their
    /// element types and broadcasting their shapes. Real floating-point inputs propagate NaNs and order negative zero
    /// below positive zero. Complex inputs compare real components first, then imaginary components when the real
    /// components are equal, selecting one whole input. Ties and unordered deciding comparisons select the right
    /// complex input. Boolean inputs are not supported. Array inputs that still carry partial sums are rejected, and
    /// their reduced-axis markers must agree.
    MinOperation, MIN_OPERATION_NAME,
    Min, min,
    check_data_types = [@numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MinOperation,
    jvp<C>
    where
        C::Value: Compare<C::Value> + Imaginary + Real + Select + ZeroLike,
    {
        // Real ties retain the existing left-tangent convention. Complex selection uses strict lexicographic
        // comparisons, routing ties and unordered deciding comparisons to the right tangent like the primal.
        |(left, left_tangent), (right, _)| {
            let left_wins = if left.r#type().is_complex() || right.r#type().is_complex() {
                let left_real = if left.r#type().is_complex() { left.real()? } else { left.clone() };
                let right_real = if right.r#type().is_complex() { right.real()? } else { right.clone() };
                let left_imaginary = if left.r#type().is_complex() { left.imaginary()? } else { left.zero_like()? };
                let right_imaginary = if right.r#type().is_complex() { right.imaginary()? } else { right.zero_like()? };
                let same_real = left_real.compare(&right_real, ComparisonDirection::Equal)?;
                let real_wins = left_real.compare(&right_real, ComparisonDirection::LessThan)?;
                let imaginary_wins = left_imaginary.compare(&right_imaginary, ComparisonDirection::LessThan)?;
                C::Value::select(&same_real, &imaginary_wins, &real_wins)?
            } else {
                left.compare(&right, ComparisonDirection::LessThanOrEqual)?
            };
            C::Value::select(&left_wins, &left_tangent, &left_tangent.zero_like()?)?
        };
        |(left, _), (right, right_tangent)| {
            let left_wins = if left.r#type().is_complex() || right.r#type().is_complex() {
                let left_real = if left.r#type().is_complex() { left.real()? } else { left.clone() };
                let right_real = if right.r#type().is_complex() { right.real()? } else { right.clone() };
                let left_imaginary = if left.r#type().is_complex() { left.imaginary()? } else { left.zero_like()? };
                let right_imaginary = if right.r#type().is_complex() { right.imaginary()? } else { right.zero_like()? };
                let same_real = left_real.compare(&right_real, ComparisonDirection::Equal)?;
                let real_wins = left_real.compare(&right_real, ComparisonDirection::LessThan)?;
                let imaginary_wins = left_imaginary.compare(&right_imaginary, ComparisonDirection::LessThan)?;
                C::Value::select(&same_real, &imaginary_wins, &real_wins)?
            } else {
                left.compare(&right, ComparisonDirection::LessThanOrEqual)?
            };
            C::Value::select(&left_wins, &right_tangent.zero_like()?, &right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to select elementwise minima. Concrete arrays compute immediately while context-carrying
    /// values apply [`MinOperation`] through their context. Refer to that operation for supported types and
    /// exceptional-value behavior.
    Min,
    /// Returns the elementwise minimum of this value and `right`, promoting and broadcasting the inputs. Returns an
    /// error if the input types or metadata are unsupported.
    min(right),
    MinOperation,
);

impl_array_elementwise_operation!(
    @binary
    Min, min,
    operation = "min",
    inputs = @numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| Ok(ArrayElement::min(&lhs, &rhs)),
);

/// Implements [`Min`] for one host primitive type.
macro_rules! impl_min_for_primitive {
    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Min for $type {
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::min(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN inputs propagate, and signed zeros order
    // through the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Min for $type {
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Greater) {
                    *right
                } else {
                    *self
                })
            }
        }
    };
}

impl_min_for_primitive!(@integer i8);
impl_min_for_primitive!(@integer i16);
impl_min_for_primitive!(@integer i32);
impl_min_for_primitive!(@integer i64);
impl_min_for_primitive!(@integer i128);
impl_min_for_primitive!(@integer isize);
impl_min_for_primitive!(@integer u8);
impl_min_for_primitive!(@integer u16);
impl_min_for_primitive!(@integer u32);
impl_min_for_primitive!(@integer u64);
impl_min_for_primitive!(@integer u128);
impl_min_for_primitive!(@integer usize);
impl_min_for_primitive!(@float f32);
impl_min_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MaxOperation`].
pub const MAX_OPERATION_NAME: &str = "max";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise maximum of two numeric values, promoting their
    /// element types and broadcasting their shapes. Real floating-point inputs propagate NaNs and order negative zero
    /// below positive zero. Complex inputs compare real components first, then imaginary components when the real
    /// components are equal, selecting one whole input. Ties and unordered deciding comparisons select the right
    /// complex input. Boolean inputs are not supported. Array inputs that still carry partial sums are rejected, and
    /// their reduced-axis markers must agree.
    MaxOperation, MAX_OPERATION_NAME,
    Max, max,
    check_data_types = [@numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MaxOperation,
    jvp<C>
    where
        C::Value: Compare<C::Value> + Imaginary + Real + Select + ZeroLike,
    {
        // Real ties retain the existing left-tangent convention. Complex selection uses strict lexicographic
        // comparisons, routing ties and unordered deciding comparisons to the right tangent like the primal.
        |(left, left_tangent), (right, _)| {
            let left_wins = if left.r#type().is_complex() || right.r#type().is_complex() {
                let left_real = if left.r#type().is_complex() { left.real()? } else { left.clone() };
                let right_real = if right.r#type().is_complex() { right.real()? } else { right.clone() };
                let left_imaginary = if left.r#type().is_complex() { left.imaginary()? } else { left.zero_like()? };
                let right_imaginary = if right.r#type().is_complex() { right.imaginary()? } else { right.zero_like()? };
                let same_real = left_real.compare(&right_real, ComparisonDirection::Equal)?;
                let real_wins = left_real.compare(&right_real, ComparisonDirection::GreaterThan)?;
                let imaginary_wins = left_imaginary.compare(&right_imaginary, ComparisonDirection::GreaterThan)?;
                C::Value::select(&same_real, &imaginary_wins, &real_wins)?
            } else {
                left.compare(&right, ComparisonDirection::GreaterThanOrEqual)?
            };
            C::Value::select(&left_wins, &left_tangent, &left_tangent.zero_like()?)?
        };
        |(left, _), (right, right_tangent)| {
            let left_wins = if left.r#type().is_complex() || right.r#type().is_complex() {
                let left_real = if left.r#type().is_complex() { left.real()? } else { left.clone() };
                let right_real = if right.r#type().is_complex() { right.real()? } else { right.clone() };
                let left_imaginary = if left.r#type().is_complex() { left.imaginary()? } else { left.zero_like()? };
                let right_imaginary = if right.r#type().is_complex() { right.imaginary()? } else { right.zero_like()? };
                let same_real = left_real.compare(&right_real, ComparisonDirection::Equal)?;
                let real_wins = left_real.compare(&right_real, ComparisonDirection::GreaterThan)?;
                let imaginary_wins = left_imaginary.compare(&right_imaginary, ComparisonDirection::GreaterThan)?;
                C::Value::select(&same_real, &imaginary_wins, &real_wins)?
            } else {
                left.compare(&right, ComparisonDirection::GreaterThanOrEqual)?
            };
            C::Value::select(&left_wins, &right_tangent.zero_like()?, &right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to select elementwise maxima. Concrete arrays compute immediately while context-carrying
    /// values apply [`MaxOperation`] through their context. Refer to that operation for supported types and
    /// exceptional-value behavior.
    Max,
    /// Returns the elementwise maximum of this value and `right`, promoting and broadcasting the inputs. Returns an
    /// error if the input types or metadata are unsupported.
    max(right),
    MaxOperation,
);

impl_array_elementwise_operation!(
    @binary
    Max, max,
    operation = "max",
    inputs = @numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| Ok(ArrayElement::max(&lhs, &rhs)),
);

/// Implements [`Max`] for one host primitive type.
macro_rules! impl_max_for_primitive {
    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Max for $type {
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::max(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN inputs propagate, and signed zeros order
    // through the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Max for $type {
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Less) {
                    *right
                } else {
                    *self
                })
            }
        }
    };
}

impl_max_for_primitive!(@integer i8);
impl_max_for_primitive!(@integer i16);
impl_max_for_primitive!(@integer i32);
impl_max_for_primitive!(@integer i64);
impl_max_for_primitive!(@integer i128);
impl_max_for_primitive!(@integer isize);
impl_max_for_primitive!(@integer u8);
impl_max_for_primitive!(@integer u16);
impl_max_for_primitive!(@integer u32);
impl_max_for_primitive!(@integer u64);
impl_max_for_primitive!(@integer u128);
impl_max_for_primitive!(@integer usize);
impl_max_for_primitive!(@float f32);
impl_max_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Represents the ability to restrict values elementwise between `lower` and `upper`. [`Clamp`] is provided for
/// every value that supports [`Max`] and [`Min`] as the composition `max(lower, min(input, upper))`, which is how
/// [StableHLO defines `clamp`](https://openxla.org/stablehlo/spec#clamp). Inputs promote to a common numeric element
/// type and broadcast. Real inputs are clipped to the inclusive interval; complex inputs use the extrema operations'
/// lexicographic ordering. The composition inherits their NaN and tie behavior. For real inputs strictly inside the
/// interval, the tangent follows the input; outside the interval it follows the selected bound.
pub trait Clamp: Sized {
    /// Clamps this value elementwise to the inclusive `[lower, upper]` interval, promoting and broadcasting its inputs
    /// as needed. Returns an error if the input types or metadata are unsupported.
    ///
    /// # Parameters
    ///
    ///   - `lower`: Inclusive elementwise lower bound.
    ///   - `upper`: Inclusive elementwise upper bound.
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError>;
}

impl<V: Max + Min> Clamp for V {
    #[inline]
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError> {
        self.min(upper)?.max(lower)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::one_like::OneLike;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    /// Clamps `x` elementwise to the `[-1, 1]` interval, staging the bounds from `x` itself so that the helper works
    /// for both eager values and tracers.
    fn clamp_to_unit_interval<V: Clone + Clamp + OneLike + std::ops::Neg<Output = V>>(x: V) -> Result<V, ProgramError> {
        let upper = x.one_like()?;
        let lower = -upper.clone();
        x.clamp(&lower, &upper)
    }

    #[test]
    fn test_min_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MinOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "`min` does not support input data type `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_min_interpretation() {
        assert_eq!(
            MinOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.0f64).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(1.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_min_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MinOperation::new(),
            inputs = [Array::scalar(0.7).unwrap(), Array::scalar(0.3).unwrap()],
            expected = Array::scalar(0.3).unwrap(),
        );
    }

    #[test]
    fn test_min_batching() {
        check_operation_batching!(
            @exact,
            operation = MinOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.3, -1.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, -2.0]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MinOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(1.0).unwrap()],
                    tangent_outputs = [Array::scalar(5.0).unwrap()],
                },
                {
                    primals = [Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(1.0).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation_at_ties() {
        // Ties route the tangent to the left operand. The finite-difference oracle cannot check the
        // non-differentiable tie point, so the tie policy is asserted on the staged jvp program directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let outputs = jvp_program
            .interpret(vec![
                Array::scalar(2.0).unwrap(),
                Array::scalar(2.0).unwrap(),
                Array::scalar(3.0).unwrap(),
                Array::scalar(5.0).unwrap(),
            ])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]);
    }

    #[test]
    fn test_min_differentiation_complex() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::C128));
        let right = builder.add_input(ArrayType::scalar(DataType::C128));
        let output = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let left_tangent = Array::scalar(Complex::new(3.0, 4.0)).unwrap();
        let right_tangent = Array::scalar(Complex::new(5.0, 6.0)).unwrap();

        // The imaginary component decides when the real components agree.
        assert_eq!(
            jvp_program
                .interpret(vec![
                    Array::scalar(Complex::new(1.0, 1.0)).unwrap(),
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    left_tangent.clone(),
                    right_tangent.clone(),
                ])
                .unwrap(),
            vec![Array::scalar(Complex::new(1.0, 1.0)).unwrap(), left_tangent.clone()],
        );
        // Complex ties follow the right operand, unlike the existing real tie convention.
        assert_eq!(
            jvp_program
                .interpret(vec![
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    left_tangent,
                    right_tangent.clone(),
                ])
                .unwrap(),
            vec![Array::scalar(Complex::new(1.0, 2.0)).unwrap(), right_tangent],
        );
    }

    #[test]
    fn test_min_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_min() {
        assert_eq!(
            Array::scalar(2i32).unwrap().min(&Array::scalar(5i32).unwrap()).unwrap(),
            Array::scalar(2i32).unwrap()
        );
        assert_eq!(
            Array::scalar(-2i64).unwrap().min(&Array::scalar(-5i64).unwrap()).unwrap(),
            Array::scalar(-5i64).unwrap()
        );
        assert_eq!(
            Array::scalar(3u32).unwrap().min(&Array::scalar(7u32).unwrap()).unwrap(),
            Array::scalar(3u32).unwrap()
        );
        assert_eq!(
            Array::scalar(2.5f32).unwrap().min(&Array::scalar(1.5f32).unwrap()).unwrap(),
            Array::scalar(1.5f32).unwrap()
        );
        // Mixed-precision operands promote before comparing.
        assert_eq!(
            Array::scalar(2.5f32).unwrap().min(&Array::scalar(3.5f64).unwrap()).unwrap(),
            Array::scalar(2.5f64).unwrap()
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0))
                .unwrap()
                .min(&Array::scalar(bf16::from_f32(3.0)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(2.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).unwrap().min(&Array::scalar(f16::from_f32(3.0)).unwrap()).unwrap(),
            Array::scalar(f16::from_f32(2.0)).unwrap(),
        );
        // NaNs propagate and `-0.0` orders below `+0.0`.
        assert!(Array::scalar(f64::NAN).unwrap().min(&Array::scalar(1.0f64).unwrap()).unwrap().to_f64s()[0].is_nan());
        assert!(Array::scalar(1.0f64).unwrap().min(&Array::scalar(f64::NAN).unwrap()).unwrap().to_f64s()[0].is_nan());
        let zero = Array::scalar(-0.0f64).unwrap().min(&Array::scalar(0.0f64).unwrap()).unwrap().to_f64s()[0];
        assert!(zero == 0.0 && zero.is_sign_negative());
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).unwrap().min(&Array::vector(vec![0.3, 2.0]).unwrap()).unwrap(),
            Array::vector(vec![0.3, -1.0]).unwrap(),
        );
    }

    #[test]
    fn test_array_min_complex() {
        // The real component takes precedence, and mixed real/complex inputs promote before selection.
        let left = Array::scalar(Complex::new(1.0f32, 100.0)).unwrap();
        let right = Array::scalar(Complex::new(2.0f32, -100.0)).unwrap();
        assert_eq!(left.min(&right).unwrap(), Array::scalar(Complex::new(1.0f32, 100.0)).unwrap());
        assert_eq!(
            left.min(&Array::scalar(2.0f32).unwrap()).unwrap(),
            Array::scalar(Complex::new(1.0f32, 100.0)).unwrap(),
        );

        // Equal real components compare imaginary components, with scalar broadcasting across a vector.
        let values = Array::vector(vec![Complex::new(1.0f64, 1.0), Complex::new(1.0, 3.0)]).unwrap();
        assert_eq!(
            values.min(&Array::scalar(Complex::new(1.0f64, 2.0)).unwrap()).unwrap(),
            Array::vector(vec![Complex::new(1.0f64, 1.0), Complex::new(1.0, 2.0)]).unwrap(),
        );

        // Unordered real comparisons and signed-zero ties select the right whole operand.
        let unordered = Array::scalar(Complex::new(f32::NAN, 1.0)).unwrap();
        assert_eq!(unordered.min(&right).unwrap(), right);
        let zero = Array::scalar(Complex::new(1.0f32, -0.0)).unwrap();
        let result = zero.min(&Array::scalar(Complex::new(1.0f32, 0.0)).unwrap()).unwrap();
        assert_eq!(result.elements::<Complex<f32>>().unwrap()[0].im.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_array_min_encodings() {
        // Minimum preserves the selected input's IEEE signed-zero encoding.
        assert_eq!(
            Array::scalar(-0.0f32)
                .unwrap()
                .min(&Array::scalar(0.0f32).unwrap())
                .unwrap()
                .elements::<f32>()
                .unwrap()[0]
                .to_bits(),
            (-0.0f32).to_bits(),
        );
    }

    #[test]
    fn test_min_for_primitives() {
        assert_eq!(Min::min(&3usize, &4), Ok(3));
        assert!(Min::min(&1.0f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Min::min(&0.0f64, &-0.0).unwrap().to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn test_max_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MaxOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "`max` does not support input data type `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_max_interpretation() {
        assert_eq!(
            MaxOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.0f64).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(2.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_max_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MaxOperation::new(),
            inputs = [Array::scalar(0.7).unwrap(), Array::scalar(0.3).unwrap()],
            expected = Array::scalar(0.7).unwrap(),
        );
    }

    #[test]
    fn test_max_batching() {
        check_operation_batching!(
            @exact,
            operation = MaxOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 0.0]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MaxOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(2.0).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = max %0 %1
                            %5:bool[] = compare [direction=GreaterThanOrEqual] %0 %1
                            %6:f64[] = zero_like %2
                            %7:f64[] = select %5 %2 %6
                            %8:bool[] = compare [direction=GreaterThanOrEqual] %0 %1
                            %9:f64[] = zero_like %3
                            %10:f64[] = select %8 %9 %3
                            %11:f64[] = add %7 %10
                        in (%4, %11)
                    "},
                },
                {
                    primals = [Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(2.0).unwrap()],
                    tangent_outputs = [Array::scalar(5.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation_at_ties() {
        // Ties route the tangent to the left operand. The finite-difference oracle cannot check the
        // non-differentiable tie point, so the tie policy is asserted on the staged jvp program directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let outputs = jvp_program
            .interpret(vec![
                Array::scalar(2.0).unwrap(),
                Array::scalar(2.0).unwrap(),
                Array::scalar(3.0).unwrap(),
                Array::scalar(5.0).unwrap(),
            ])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]);
    }

    #[test]
    fn test_max_differentiation_complex() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::C128));
        let right = builder.add_input(ArrayType::scalar(DataType::C128));
        let output = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let left_tangent = Array::scalar(Complex::new(3.0, 4.0)).unwrap();
        let right_tangent = Array::scalar(Complex::new(5.0, 6.0)).unwrap();

        // The imaginary component decides when the real components agree.
        assert_eq!(
            jvp_program
                .interpret(vec![
                    Array::scalar(Complex::new(1.0, 1.0)).unwrap(),
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    left_tangent.clone(),
                    right_tangent.clone(),
                ])
                .unwrap(),
            vec![Array::scalar(Complex::new(1.0, 2.0)).unwrap(), right_tangent.clone()],
        );
        // Complex ties follow the right operand, unlike the existing real tie convention.
        assert_eq!(
            jvp_program
                .interpret(vec![
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    Array::scalar(Complex::new(1.0, 2.0)).unwrap(),
                    left_tangent,
                    right_tangent.clone(),
                ])
                .unwrap(),
            vec![Array::scalar(Complex::new(1.0, 2.0)).unwrap(), right_tangent],
        );
    }

    #[test]
    fn test_max_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_max() {
        assert_eq!(
            Array::scalar(2i32).unwrap().max(&Array::scalar(5i32).unwrap()).unwrap(),
            Array::scalar(5i32).unwrap()
        );
        assert_eq!(
            Array::scalar(-2i64).unwrap().max(&Array::scalar(-5i64).unwrap()).unwrap(),
            Array::scalar(-2i64).unwrap()
        );
        assert_eq!(
            Array::scalar(3u32).unwrap().max(&Array::scalar(7u32).unwrap()).unwrap(),
            Array::scalar(7u32).unwrap()
        );
        assert_eq!(
            Array::scalar(2.5f32).unwrap().max(&Array::scalar(1.5f32).unwrap()).unwrap(),
            Array::scalar(2.5f32).unwrap()
        );
        // Mixed-precision operands promote before comparing.
        assert_eq!(
            Array::scalar(2.5f32).unwrap().max(&Array::scalar(3.5f64).unwrap()).unwrap(),
            Array::scalar(3.5f64).unwrap()
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0))
                .unwrap()
                .max(&Array::scalar(bf16::from_f32(3.0)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(3.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).unwrap().max(&Array::scalar(f16::from_f32(3.0)).unwrap()).unwrap(),
            Array::scalar(f16::from_f32(3.0)).unwrap(),
        );
        // NaNs propagate and `-0.0` orders below `+0.0`.
        assert!(Array::scalar(f64::NAN).unwrap().max(&Array::scalar(1.0f64).unwrap()).unwrap().to_f64s()[0].is_nan());
        assert!(Array::scalar(1.0f64).unwrap().max(&Array::scalar(f64::NAN).unwrap()).unwrap().to_f64s()[0].is_nan());
        let zero = Array::scalar(-0.0f64).unwrap().max(&Array::scalar(0.0f64).unwrap()).unwrap().to_f64s()[0];
        assert!(zero == 0.0 && zero.is_sign_positive());
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).unwrap().max(&Array::vector(vec![0.3, 2.0]).unwrap()).unwrap(),
            Array::vector(vec![0.7, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_array_max_complex() {
        // The real component takes precedence, and mixed real/complex inputs promote before selection.
        let left = Array::scalar(Complex::new(1.0f32, 100.0)).unwrap();
        let right = Array::scalar(Complex::new(2.0f32, -100.0)).unwrap();
        assert_eq!(left.max(&right).unwrap(), Array::scalar(Complex::new(2.0f32, -100.0)).unwrap());
        assert_eq!(
            left.max(&Array::scalar(2.0f32).unwrap()).unwrap(),
            Array::scalar(Complex::new(2.0f32, 0.0)).unwrap(),
        );

        // Equal real components compare imaginary components, with scalar broadcasting across a vector.
        let values = Array::vector(vec![Complex::new(1.0f64, 1.0), Complex::new(1.0, 3.0)]).unwrap();
        assert_eq!(
            values.max(&Array::scalar(Complex::new(1.0f64, 2.0)).unwrap()).unwrap(),
            Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(1.0, 3.0)]).unwrap(),
        );

        // Unordered real comparisons and signed-zero ties select the right whole operand.
        let unordered = Array::scalar(Complex::new(f32::NAN, 1.0)).unwrap();
        assert_eq!(unordered.max(&right).unwrap(), right);
        let zero = Array::scalar(Complex::new(1.0f32, -0.0)).unwrap();
        let result = zero.max(&Array::scalar(Complex::new(1.0f32, 0.0)).unwrap()).unwrap();
        assert_eq!(result.elements::<Complex<f32>>().unwrap()[0].im.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_array_max_encodings() {
        // Elementwise extrema retain the selected input's NaN payload and IEEE signed-zero encoding.
        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(
            Array::scalar(nan).unwrap().max(&Array::scalar(1.0f32).unwrap()).unwrap().elements::<f32>().unwrap()[0]
                .to_bits(),
            nan.to_bits(),
        );
        assert_eq!(
            Array::scalar(-0.0f32)
                .unwrap()
                .max(&Array::scalar(0.0f32).unwrap())
                .unwrap()
                .elements::<f32>()
                .unwrap()[0]
                .to_bits(),
            0.0f32.to_bits(),
        );
    }

    #[test]
    fn test_max_for_primitives() {
        assert_eq!(Max::max(&3usize, &4), Ok(4));
        assert!(Max::max(&1.0f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Max::max(&0.0f64, &-0.0).unwrap().to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn test_clamp() {
        let lower = Array::scalar(-1.0f64).unwrap();
        let upper = Array::scalar(1.0f64).unwrap();
        assert_eq!(Array::scalar(0.5f64).unwrap().clamp(&lower, &upper).unwrap(), Array::scalar(0.5f64).unwrap());
        assert_eq!(Array::scalar(-2.5f64).unwrap().clamp(&lower, &upper).unwrap(), Array::scalar(-1.0f64).unwrap());
        assert_eq!(Array::scalar(2.5f64).unwrap().clamp(&lower, &upper).unwrap(), Array::scalar(1.0f64).unwrap());
        assert_eq!(
            Array::scalar(7i32)
                .unwrap()
                .clamp(&Array::scalar(0i32).unwrap(), &Array::scalar(5i32).unwrap())
                .unwrap(),
            Array::scalar(5i32).unwrap(),
        );

        assert_eq!(
            Array::vector(vec![-2.0, 0.5, 3.0])
                .unwrap()
                .clamp(&Array::scalar(-1.0).unwrap(), &Array::scalar(1.0).unwrap())
                .unwrap(),
            Array::vector(vec![-1.0, 0.5, 1.0]).unwrap(),
        );
    }

    #[test]
    fn test_clamp_differentiation() {
        // The gradient follows the clamped value: `1` strictly inside the interval and `0` outside it.
        let (value, gradient) =
            differentiate_at(Array::scalar(0.5).unwrap()).value_and_gradient(clamp_to_unit_interval).unwrap();
        assert_eq!(value.to_f64s(), vec![0.5]);
        assert_eq!(gradient.to_f64s(), vec![1.0]);
        let (value, gradient) =
            differentiate_at(Array::scalar(2.5).unwrap()).value_and_gradient(clamp_to_unit_interval).unwrap();
        assert_eq!(value.to_f64s(), vec![1.0]);
        assert_eq!(gradient.to_f64s(), vec![0.0]);
        let (value, gradient) =
            differentiate_at(Array::scalar(-2.5).unwrap()).value_and_gradient(clamp_to_unit_interval).unwrap();
        assert_eq!(value.to_f64s(), vec![-1.0]);
        assert_eq!(gradient.to_f64s(), vec![0.0]);
    }
}
