//! Operations that compute special mathematical functions elementwise. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`ErfOperation`]) together with a value capability trait (e.g.,
//! [`Erf`]) whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code
//! executes immediately or records into a program depending on the value it runs on:
//!
//!   - [`Erf`] computes the Gauss error function (i.e., `x ↦ erf(x) = 2/√π · ∫₀ˣ e^{−t²} dt`).
//!
//! Only real floating-point inputs are supported. The output keeps the element type and array metadata of the input,
//! and inputs that carry partial sums over unreduced mesh axes are rejected. The derivative of the error function is
//! `2/√π · e^{−x²}`. It is nonlinear, so reverse-mode differentiation transposes its linearization instead.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Erf, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! assert_eq!(Array::scalar(0.0f64)?.erf()?, Array::scalar(0.0)?);
//! # Ok(())
//! # }
//! ```

use std::f64::consts::FRAC_2_SQRT_PI;

use crate::arrays::RealFloatingPointArrayElement;
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation,
};
use crate::operations::arithmetic::{Mul, Neg};
use crate::operations::constants::fill::Fill;
use crate::operations::exponential::Exp;
use crate::programs::{Typed, Value};

/// Canonical operation name for [`ErfOperation`].
pub const ERF_OPERATION_NAME: &str = "erf";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise Gauss error function of one value (i.e.,
    /// `x ↦ erf(x) = 2/√π · ∫₀ˣ e^{−t²} dt`) while preserving its array metadata. Only real floating-point inputs
    /// are supported, and inputs that still carry partial sums are rejected.
    ErfOperation,
    ERF_OPERATION_NAME,
    Erf,
    erf,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    ErfOperation,
    jvp<C>
    where
        C::Value: Neg + Mul + Exp,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        // `d(erf(x)) = (2/√π) · exp(-x²) · dx`, with the coefficient `2/√π` rounded to the aligned input's element
        // data type. Filling that type stages a scalar constant and broadcasts it when needed.
        |(input, input_tangent)| {
            let input_type = input.r#type().into_owned();
            let coefficient = input.dispatch_domain().fill(&input_type, FRAC_2_SQRT_PI)?;
            coefficient.mul(&input.mul(&input)?.neg()?.exp()?)?.mul(&input_tangent)?
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute the elementwise Gauss error function. Concrete arrays compute immediately
    /// while context-carrying values apply [`ErfOperation`] through their context.
    Erf,
    /// Computes the Gauss error function for each real floating-point element. Returns an error if the input types or
    /// metadata are unsupported.
    erf,
    ErfOperation,
);

impl_array_elementwise_operation!(
    @unary
    Erf,
    erf,
    operation = "erf",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::erf(input),
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType, f4e2m1fn, f8e8m0fnu};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_erf_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = ErfOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`erf` does not support input data type `c64`",
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`erf` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ErfOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_erf_interpretation() {
        assert_eq!(
            ErfOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_erf_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ErfOperation::new(),
            inputs = [Array::scalar(0.5).unwrap()],
            expected = Array::scalar(0.5204998778130465).unwrap(),
        );
    }

    #[test]
    fn test_erf_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = ErfOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5204998778130465, -0.8427007929497149]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_erf_differentiation() {
        let expected_tangent = 3.0 * FRAC_2_SQRT_PI * (-0.7f64 * 0.7).exp();
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ErfOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.6778011938374184).unwrap()],
                tangent_outputs = [Array::scalar(expected_tangent).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = erf %0
                        %3:f64[] = constant [value=1.1283791670955126]
                        %4:f64[] = mul %0 %0
                        %5:f64[] = neg %4
                        %6:f64[] = exp %5
                        %7:f64[] = mul %3 %6
                        %8:f64[] = mul %7 %1
                    in (%2, %8)
                "},
            }],
        );
        check_gradient!(
            |input| input.erf()?.reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![-2.5f64, -0.3, 0.0, 0.9, 3.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_erf_differentiation_low_precision() {
        // Form the derivative coefficient before multiplying by the tangent, retaining its accuracy in `f16`.
        let (_, tangent) = differentiate_at(Array::scalar(f16::from_f32(0.5)).unwrap())
            .jvp(Array::scalar(f16::from_f32(0.3)).unwrap(), |input| input.erf())
            .unwrap();
        assert_eq!(tangent, Array::scalar(f16::from_f64(0.263671875)).unwrap());

        // Exponent-only primals retain their storage type, while the derivative uses signed `f32` arithmetic.
        let (primal, tangent) = differentiate_at(Array::scalar(f8e8m0fnu::from_f64(0.5).unwrap()).unwrap())
            .jvp(Array::scalar(1.0f32).unwrap(), |input| input.erf())
            .unwrap();
        assert_eq!(primal.elements::<f8e8m0fnu>().unwrap()[0].to_bits(), 0x7e);
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.elements::<f32>().unwrap()[0], 0.8787826, epsilon = 1e-7);
    }

    #[test]
    fn test_erf_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = ErfOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_erf() {
        // Exact fixed points and symmetry.
        assert_eq!(Array::scalar(0.0f64).unwrap().erf().unwrap(), Array::scalar(0.0f64).unwrap());
        assert_eq!(Array::scalar(f64::INFINITY).unwrap().erf().unwrap(), Array::scalar(1.0f64).unwrap());
        assert_eq!(Array::scalar(f64::NEG_INFINITY).unwrap().erf().unwrap(), Array::scalar(-1.0f64).unwrap());
        assert_eq!(Array::scalar(1.5f64).unwrap().erf().unwrap(), -Array::scalar(-1.5f64).unwrap().erf().unwrap());
        assert!(Array::scalar(f64::NAN).unwrap().erf().unwrap().elements::<f64>().unwrap()[0].is_nan());

        // Logical element decoding retains signed zeros and the exact low-precision encodings.
        assert_eq!(
            Array::scalar(-0.0f64).unwrap().erf().unwrap().elements::<f64>().unwrap()[0].to_bits(),
            (-0.0f64).to_bits(),
        );
        assert_eq!(
            Array::scalar(f4e2m1fn::from_f64(-0.0).unwrap())
                .unwrap()
                .erf()
                .unwrap()
                .elements::<f4e2m1fn>()
                .unwrap()[0]
                .to_bits(),
            0x8,
        );
        assert_eq!(
            Array::scalar(f4e2m1fn::from_f64(0.5).unwrap())
                .unwrap()
                .erf()
                .unwrap()
                .elements::<f4e2m1fn>()
                .unwrap()[0]
                .to_bits(),
            0x1,
        );

        // The narrower variants round the double-precision evaluation to their own precision.
        assert_eq!(Array::scalar(0.5f32).unwrap().erf().unwrap(), Array::scalar(0.5204998778130465f64 as f32).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().erf().unwrap(),
            Array::scalar(bf16::from_f64(0.5204998778130465)).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().erf().unwrap(),
            Array::scalar(f16::from_f64(0.5204998778130465)).unwrap(),
        );

        assert_eq!(Array::scalar(0.5).unwrap().erf().unwrap(), Array::scalar(0.5204998778130465).unwrap());

        assert_abs_diff_eq!(
            Array::vector(vec![-1.0f64, 0.0, 1.0]).unwrap().erf().unwrap(),
            Array::vector(vec![-0.8427007929497149, 0.0, 0.8427007929497149]).unwrap(),
            epsilon = 1e-12,
        );
    }
}
