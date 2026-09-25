//! Operations that round real floating-point values elementwise to integral values. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`RoundOperation`]) together with a value capability trait (e.g.,
//! [`Round`]) whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code
//! executes immediately or records into a program depending on the value it runs on:
//!
//!   - [`Ceil`] rounds toward positive infinity (i.e., `x ↦ ⌈x⌉`).
//!   - [`Floor`] rounds toward negative infinity (i.e., `x ↦ ⌊x⌋`).
//!   - [`Round`] rounds to the nearest integer, resolving ties toward the even integer (e.g., `2.5 ↦ 2` and `3.5 ↦ 4`).
//!
//! Only real floating-point inputs are supported, similar to StableHLO's
//! [`ceil`](https://openxla.org/stablehlo/spec#ceil), [`floor`](https://openxla.org/stablehlo/spec#floor), and
//! [`round_nearest_even`](https://openxla.org/stablehlo/spec#round_nearest_even). The output keeps the element type
//! and array metadata of the input, NaNs and signed zeros pass through unchanged, and inputs that carry partial sums
//! over unreduced mesh axes are rejected. Every operation is piecewise constant, so its tangents and cotangents are
//! zero/
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, Round};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![1.5f64, 2.5, -1.5])?;
//! assert_eq!(input.round()?, Array::vector(vec![2.0, 2.0, -2.0])?);
//! # Ok(())
//! # }
//! ```

use crate::arrays::RealFloatingPointArrayElement;
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

/// Canonical operation name for [`CeilOperation`].
pub const CEIL_OPERATION_NAME: &str = "ceil";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise ceiling of one value (i.e., `x ↦ ⌈x⌉`, rounding
    /// toward positive infinity) while preserving its array metadata. Matching the input constraints of StableHLO's
    /// [`ceil`](https://openxla.org/stablehlo/spec#ceil), only real floating-point inputs are supported, and inputs
    /// that still carry partial sums are rejected.
    CeilOperation,
    CEIL_OPERATION_NAME,
    Ceil,
    ceil,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant CeilOperation);

define_elementwise_capability!(
    @unary
    /// Represents the ability to round elementwise toward positive infinity. Concrete arrays compute immediately while
    /// context-carrying values apply [`CeilOperation`] through their context.
    Ceil,
    /// Rounds each element toward positive infinity, preserving the input type. Returns an error if the input types or
    /// metadata are unsupported.
    ceil,
    CeilOperation,
);

impl_array_elementwise_operation!(
    @unary
    Ceil,
    ceil,
    operation = "ceil",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::ceil(input),
);

/// Implements [`Ceil`] for one host primitive type.
macro_rules! impl_ceil_for_primitive {
    ($type:ty) => {
        impl Ceil for $type {
            #[inline]
            fn ceil(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::ceil(*self))
            }
        }
    };
}

impl_ceil_for_primitive!(f32);
impl_ceil_for_primitive!(f64);

/// Canonical operation name for [`FloorOperation`].
pub const FLOOR_OPERATION_NAME: &str = "floor";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise floor of one value (i.e., `x ↦ ⌊x⌋`, rounding
    /// toward negative infinity) while preserving its array metadata. Matching the input constraints of StableHLO's
    /// [`floor`](https://openxla.org/stablehlo/spec#floor), only real floating-point inputs are supported, and inputs
    /// that still carry partial sums are rejected.
    FloorOperation,
    FLOOR_OPERATION_NAME,
    Floor,
    floor,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant FloorOperation);

define_elementwise_capability!(
    @unary
    /// Represents the ability to round elementwise toward negative infinity. Concrete arrays compute immediately while
    /// context-carrying values apply [`FloorOperation`] through their context.
    Floor,
    /// Rounds each element toward negative infinity, preserving the input type. Returns an error if the input types or
    /// metadata are unsupported.
    floor,
    FloorOperation,
);

impl_array_elementwise_operation!(
    @unary
    Floor,
    floor,
    operation = "floor",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::floor(input),
);

/// Implements [`Floor`] for one host primitive type.
macro_rules! impl_floor_for_primitive {
    ($type:ty) => {
        impl Floor for $type {
            #[inline]
            fn floor(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::floor(*self))
            }
        }
    };
}

impl_floor_for_primitive!(f32);
impl_floor_for_primitive!(f64);

/// Canonical operation name for [`RoundOperation`].
pub const ROUND_OPERATION_NAME: &str = "round";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that rounds one value elementwise to the nearest integer, with ties resolved
    /// toward the nearest even integer, while preserving its array metadata. Matching the input constraints of
    /// StableHLO's [`round_nearest_even`](https://openxla.org/stablehlo/spec#round_nearest_even), only real
    /// floating-point inputs are supported, and inputs that still carry partial sums are rejected.
    RoundOperation,
    ROUND_OPERATION_NAME,
    Round,
    round,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant RoundOperation);

define_elementwise_capability!(
    @unary
    /// Represents the ability to round elementwise to the nearest even integer. Concrete arrays compute immediately
    /// while context-carrying values apply [`RoundOperation`] through their context.
    Round,
    /// Rounds each element to the nearest integer, resolving ties toward the even integer. Returns an error if the
    /// input types or metadata are unsupported.
    round,
    RoundOperation,
);

impl_array_elementwise_operation!(
    @unary
    Round,
    round,
    operation = "round",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::round(input),
);

/// Implements [`Round`] for one host primitive type.
macro_rules! impl_round_for_primitive {
    ($type:ty) => {
        impl Round for $type {
            #[inline]
            fn round(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::round_ties_even(*self))
            }
        }
    };
}

impl_round_for_primitive!(f32);
impl_round_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_ceil_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = CeilOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`ceil` does not support input data type `i32`",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`ceil` does not support input data type `c64`",
                },
            ],
        );

        check_operation_type_inference!(
            @reject @unreduced,
            operation = CeilOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_ceil_interpretation() {
        assert_eq!(
            CeilOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(2.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_ceil_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CeilOperation::new(),
            inputs = [Array::scalar(2.3).unwrap()],
            expected = Array::scalar(3.0).unwrap(),
        );
    }

    #[test]
    fn test_ceil_batching() {
        check_operation_batching!(
            @exact,
            operation = CeilOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.5]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -1.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_ceil_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = CeilOperation::new(),
            cases = [{
                primals = [Array::scalar(2.5).unwrap()],
                tangents = [Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(3.0).unwrap()],
                tangent_outputs = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_ceil_transposition() {
        check_operation_transposition!(
            @exact,
            operation = CeilOperation::<ArrayType>::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_ceil() {
        assert_eq!(Array::scalar(2.3f32).unwrap().ceil().unwrap(), Array::scalar(3.0f32).unwrap());
        assert_eq!(Array::scalar(-2.7f64).unwrap().ceil().unwrap(), Array::scalar(-2.0f64).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(2.3)).unwrap().ceil().unwrap(),
            Array::scalar(bf16::from_f32(2.3f32.ceil())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.3)).unwrap().ceil().unwrap(),
            Array::scalar(f16::from_f32(2.3f32.ceil())).unwrap(),
        );

        // NaNs pass through unchanged.
        assert!(Array::scalar(f64::NAN).unwrap().ceil().unwrap().to_f64s()[0].is_nan());

        assert_eq!(
            Array::vector(vec![0.7, 1.0, -1.5]).unwrap().ceil().unwrap(),
            Array::vector(vec![1.0, 1.0, -1.0]).unwrap(),
        );

        assert_eq!(
            Array::vector(vec![-1.5f64, -0.0, 2.5, 3.5]).unwrap().ceil().unwrap(),
            Array::vector(vec![-1.0, -0.0, 3.0, 4.0]).unwrap(),
        );
    }

    #[test]
    fn test_ceil_for_primitives() {
        assert_eq!(Ceil::ceil(&1.25f64), Ok(2.0));
    }

    #[test]
    fn test_floor_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = FloorOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`floor` does not support input data type `i32`",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`floor` does not support input data type `c64`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = FloorOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_floor_interpretation() {
        assert_eq!(
            FloorOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(1.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_floor_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = FloorOperation::new(),
            inputs = [Array::scalar(2.7).unwrap()],
            expected = Array::scalar(2.0).unwrap(),
        );
    }

    #[test]
    fn test_floor_batching() {
        check_operation_batching!(
            @exact,
            operation = FloorOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.5]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, -2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_floor_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = FloorOperation::new(),
            cases = [{
                primals = [Array::scalar(-2.5).unwrap()],
                tangents = [Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(-3.0).unwrap()],
                tangent_outputs = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_floor_transposition() {
        check_operation_transposition!(
            @exact,
            operation = FloorOperation::<ArrayType>::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_floor() {
        assert_eq!(Array::scalar(2.7f32).unwrap().floor().unwrap(), Array::scalar(2.0f32).unwrap());
        assert_eq!(Array::scalar(-2.3f64).unwrap().floor().unwrap(), Array::scalar(-3.0f64).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(2.7)).unwrap().floor().unwrap(),
            Array::scalar(bf16::from_f32(2.7f32.floor())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.7)).unwrap().floor().unwrap(),
            Array::scalar(f16::from_f32(2.7f32.floor())).unwrap(),
        );

        // NaNs pass through unchanged.
        assert!(Array::scalar(f64::NAN).unwrap().floor().unwrap().to_f64s()[0].is_nan());

        assert_eq!(
            Array::vector(vec![-0.7, 0.0, 2.5]).unwrap().floor().unwrap(),
            Array::vector(vec![-1.0, 0.0, 2.0]).unwrap(),
        );

        assert_eq!(
            Array::vector(vec![-1.5f64, -0.0, 2.5, 3.5]).unwrap().floor().unwrap(),
            Array::vector(vec![-2.0, -0.0, 2.0, 3.0]).unwrap(),
        );
    }

    #[test]
    fn test_floor_for_primitives() {
        assert_eq!(Floor::floor(&1.75f64), Ok(1.0));
    }

    #[test]
    fn test_round_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = RoundOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`round` does not support input data type `i32`",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`round` does not support input data type `c64`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RoundOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_round_interpretation() {
        assert_eq!(
            RoundOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(2.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_round_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RoundOperation::new(),
            inputs = [Array::scalar(2.5).unwrap()],
            expected = Array::scalar(2.0).unwrap(),
        );
    }

    #[test]
    fn test_round_batching() {
        check_operation_batching!(
            @exact,
            operation = RoundOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 1.5]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_round_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RoundOperation::new(),
            cases = [{
                primals = [Array::scalar(2.4).unwrap()],
                tangents = [Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(2.0).unwrap()],
                tangent_outputs = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_round_transposition() {
        check_operation_transposition!(
            @exact,
            operation = RoundOperation::<ArrayType>::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_round() {
        // Ties resolve toward the nearest even integer.
        assert_eq!(Array::scalar(2.5f64).unwrap().round().unwrap(), Array::scalar(2.0f64).unwrap());
        assert_eq!(Array::scalar(3.5f64).unwrap().round().unwrap(), Array::scalar(4.0f64).unwrap());
        assert_eq!(Array::scalar(-2.5f32).unwrap().round().unwrap(), Array::scalar(-2.0f32).unwrap());
        assert_eq!(Array::scalar(2.3f64).unwrap().round().unwrap(), Array::scalar(2.0f64).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(2.5)).unwrap().round().unwrap(),
            Array::scalar(bf16::from_f32(2.0)).unwrap()
        );
        assert_eq!(
            Array::scalar(f16::from_f32(3.5)).unwrap().round().unwrap(),
            Array::scalar(f16::from_f32(4.0)).unwrap()
        );

        // NaNs pass through unchanged.
        assert!(Array::scalar(f64::NAN).unwrap().round().unwrap().to_f64s()[0].is_nan());

        assert_eq!(
            Array::vector(vec![0.5, 1.5, -2.5]).unwrap().round().unwrap(),
            Array::vector(vec![0.0, 2.0, -2.0]).unwrap(),
        );

        assert_eq!(
            Array::vector(vec![-1.5f64, -0.0, 2.5, 3.5]).unwrap().round().unwrap(),
            Array::vector(vec![-2.0, -0.0, 2.0, 4.0]).unwrap(),
        );
    }

    #[test]
    fn test_round_for_primitives() {
        assert_eq!(Round::round(&2.5f64), Ok(2.0));
        assert_eq!(Round::round(&3.5f64), Ok(4.0));
    }
}
