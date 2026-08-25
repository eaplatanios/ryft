use std::ops::{Add as StandardAdd, Div as StandardDiv};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::constants::one_like::OneLike;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`Log1pOperation`].
pub const LOG1P_OPERATION_NAME: &str = "log1p";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise natural logarithm of one plus its operand (i.e.,
    /// `x ↦ log(1 + x)`) while preserving its array metadata. The name matches the canonical mathematical spelling
    /// that Rust's own [`f64::ln_1p`] uses.
    ///
    /// The point of the primitive is accuracy near zero: evaluating `log(1 + x)` by first forming `1 + x` loses
    /// every bit of `x` below the precision of one, so a small `x` returns a result whose relative error grows
    /// without bound as `x` shrinks. Computing the composition as a single operation keeps full relative accuracy
    /// there, which is why `log1p` is the form used by log-likelihood and log-probability code.
    ///
    /// Only real floating-point operands are supported, and operands that still carry partial sums are rejected.
    /// Complex support is an explicit non-goal: the complex logarithm needs a different construction (a principal
    /// branch and a separate accurate magnitude near `-1`), so it is left out rather than approximated here.
    Log1pOperation, LOG1P_OPERATION_NAME,
    Log1p, log1p,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    Log1pOperation,
    jvp<C>
    where
        C::Value: OneLike + StandardAdd<Output = C::Value> + StandardDiv<Output = C::Value>,
    {
        // d(log1p(x)) = dx / (1 + x). The denominator is formed from the aligned input primal so that it carries the
        // tangent's element data type, and `one_like` supplies the one at exactly that type.
        |(input, input_tangent)| input_tangent / (input.one_like() + input)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise `log(1 + x)` capability. [`Log1p`] fills the same role for [`Log1pOperation`] that
    /// [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Log1p,
    /// Computes [`Log1pOperation`] elementwise for this value.
    log1p,
    Log1pOperation,
);

/// Implements [`Log1p`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Log1p for $type {
            fn log1p(&self) -> Result<Self, ProgramError> {
                Ok(self.ln_1p())
            }
        }
    };
}

impl_capability_for_primitive!(f32);
impl_capability_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };

    use super::*;

    #[test]
    fn test_log1p() {
        // Ordinary values in every supported floating-point width, each evaluated in its own precision.
        assert_eq!(Array::scalar(0.5f32).log1p().unwrap(), Array::scalar(0.5f32.ln_1p()));
        assert_eq!(Array::scalar(0.5f64).log1p().unwrap(), Array::scalar(0.5f64.ln_1p()));
        assert_eq!(Array::scalar(bf16::from_f32(0.5)).log1p().unwrap(), Array::scalar(bf16::from_f32(0.5f32.ln_1p())),);
        assert_eq!(Array::scalar(f16::from_f32(0.5)).log1p().unwrap(), Array::scalar(f16::from_f32(0.5f32.ln_1p())));

        // The fixed point and the boundary values of the real domain.
        assert_eq!(Array::scalar(0.0f64).log1p().unwrap(), Array::scalar(0.0f64));
        assert_eq!(Array::scalar(-1.0f64).log1p().unwrap(), Array::scalar(f64::NEG_INFINITY));
        assert!(Array::scalar(-2.0f64).log1p().unwrap().to_f64s()[0].is_nan());

        // The accuracy the primitive exists for: near zero, `log1p` keeps full relative precision while the naive
        // composition through `1 + x` has already lost most of it.
        assert_eq!(Array::scalar(1e-10f64).log1p().unwrap(), Array::scalar(1e-10f64.ln_1p()));
        assert_ne!(1e-10f64.ln_1p(), (1.0f64 + 1e-10).ln());
        assert!((1e-10f64.ln_1p() - 1e-10).abs() < 1e-20);

        assert_eq!(Array::scalar(0.5).log1p().unwrap(), Array::scalar(0.5f64.ln_1p()));
    }

    #[test]
    fn test_log1p_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = Log1pOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`log1p` does not support input data type c64",
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`log1p` does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = Log1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log1p_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = Log1pOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -0.5]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln_1p(), (-0.5f64).ln_1p()]))],
            }],
        );
    }

    #[test]
    fn test_log1p_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = Log1pOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.7f64.ln_1p())],
                tangent_outputs = [Array::scalar(3.0 / 1.7)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = log1p %0
                        %3:f64[] = one_like %0
                        %4:f64[] = add %3 %0
                        %5:f64[] = div %1 %4
                    in (%2, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_log1p_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = Log1pOperation::new(),
            inputs = [Array::scalar(0.7)],
            expected = Array::scalar(0.7f64.ln_1p()),
        );
    }

    #[test]
    fn test_log1p_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = Log1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log1p_for_primitives() {
        assert_eq!(Log1p::log1p(&0.0_f64), Ok(0.0));
        assert_eq!(Log1p::log1p(&0.0_f32), Ok(0.0));
    }
}
