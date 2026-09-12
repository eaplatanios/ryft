use std::ops::{Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::arrays::{Array, NumericArrayElement};
use crate::differentiation::{DifferentiableType, ElementwiseDerivativeAlignment};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_array_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DivOperation`].
pub const DIV_OPERATION_NAME: &str = "div";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that divides two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that still carry partial sums are rejected, and their reduced-axis
    /// markers must agree.
    DivOperation, DIV_OPERATION_NAME,
    Div, div,
    check_data_types = [@numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

// Transposition accepts a linear numerator and a known denominator.
impl_differentiable_elementwise_operation! {
    @binary
    DivOperation,
    jvp<C>
    where
        C::Value: StandardNeg<Output = C::Value>
            + StandardMul<Output = C::Value>
            + StandardDiv<Output = C::Value>,
    {
        |(_, left_tangent), (right, _)| left_tangent / right;
        |(left, _), (right, right_tangent)| {
            let coefficient = -(left / (right.clone() * right));
            coefficient * right_tangent
        };
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<DivOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        [numerator = @linear, denominator = @known] =>
            |output_cotangent| output_cotangent.binary(&denominator, DivOperation::new());
    },
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise division capability. [`Div`] is the fallible Ryft counterpart to [`std::ops::Div`]
    /// that [`DivOperation`] interprets through, surfacing a [`ProgramError`] when something
    /// goes wrong, instead of panicking. Value types additionally provide [`std::ops::Div`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
    Div,
    /// Divides this value by `right`, returning a [`ProgramError`] if something goes wrong.
    div(right),
    DivOperation,
);

define_tracer_operator!(@binary std::ops::Div, div, capability = Div, method = div);

/// Implements [`Div`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Div for $type {
            fn div(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_div(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "`{}` divisor is zero or the result does not fit in {}",
                        DIV_OPERATION_NAME,
                        stringify!($type),
                    ),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Div for $type {
            fn div(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self / *right)
            }
        }
    };
}

impl_capability_for_primitive!(@integer i8);
impl_capability_for_primitive!(@integer i16);
impl_capability_for_primitive!(@integer i32);
impl_capability_for_primitive!(@integer i64);
impl_capability_for_primitive!(@integer i128);
impl_capability_for_primitive!(@integer isize);
impl_capability_for_primitive!(@integer u8);
impl_capability_for_primitive!(@integer u16);
impl_capability_for_primitive!(@integer u32);
impl_capability_for_primitive!(@integer u64);
impl_capability_for_primitive!(@integer u128);
impl_capability_for_primitive!(@integer usize);
impl_capability_for_primitive!(@float f32);
impl_capability_for_primitive!(@float f64);

impl_array_elementwise_operation!(
    @binary
    Div, div,
    operation = "div",
    inputs = @numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::div(lhs, rhs),
);

impl std::ops::Div for Array {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, differentiate_at,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::manipulation::conversions::ConvertElementType;
    use crate::programs::{EmptyRegionDriver, MaybeZero, TypeError};

    use super::*;

    #[test]
    fn test_div() {
        assert_eq!(DivOperation::<ArrayType>::new().to_string(), "div");
    }

    #[test]
    fn test_div_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = DivOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{DIV_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let unreduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let reduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let unreduced_error = || "`div` does not support unreduced operands";
        check_operation_type_inference!(
            operation = DivOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced.clone(), plain.clone()],
                    error = unreduced_error(),
                },
                {
                    input_types = [plain, unreduced.clone()],
                    error = unreduced_error(),
                },
                {
                    input_types = [unreduced, reduced],
                    error = unreduced_error(),
                },
            ],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = DivOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_div_interpretation() {
        let operation = DivOperation::<ArrayType>::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.0f32), Array::scalar(2.0f64)],
            ),
            Ok(vec![Array::scalar(3.5f64)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &DivOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.0), Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(3.5)]),
        );
        assert_eq!(Div::div(&7_usize, &2), Ok(3));
        assert_eq!(
            Div::div(&7_usize, &0),
            Err(ProgramError::InvalidArgument {
                message: "`div` divisor is zero or the result does not fit in usize".to_string(),
            }),
        );
        assert_abs_diff_eq!(
            match InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(Complex::new(1.0f64, 2.0)), Array::scalar(Complex::new(0.5f64, -1.0))],
            ) {
                Ok(outputs) => outputs[0].clone(),
                Err(error) => panic!("expected a complex quotient but got {error}"),
            },
            Array::scalar(Complex::new(1.0f64, 2.0) / Complex::new(0.5f64, -1.0)),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_div_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = DivOperation::new(),
            inputs = [Array::scalar(7.0), Array::scalar(2.0)],
            expected = Array::scalar(3.5),
        );
    }

    #[test]
    fn test_div_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = DivOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![3.0, -6.0])),
                    (@replicated, Array::scalar(3.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
            }],
        );
    }

    #[test]
    fn test_div_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = DivOperation::new(),
            cases = [{
                primals = [Array::scalar(6.0), Array::scalar(2.0)],
                tangents = [Array::scalar(3.0), Array::scalar(4.0)],
                primal_outputs = [Array::scalar(3.0)],
                tangent_outputs = [Array::scalar(-4.5)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = div %0 %1
                        %5:f64[] = div %2 %1
                        %6:f64[] = mul %1 %1
                        %7:f64[] = div %0 %6
                        %8:f64[] = neg %7
                        %9:f64[] = mul %8 %3
                        %10:f64[] = add %5 %9
                    in (%4, %10)
                "},
            }],
        );
    }

    #[test]
    fn test_div_differentiation_preserves_small_tangents() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let smallest_positive = f64::from_bits(1);
        let outputs = DivOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(Array::scalar(0.0), Array::scalar(smallest_positive)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(smallest_positive)).unwrap(),
                ],
            )
            .unwrap();
        match outputs[0].tangent() {
            MaybeZero::Value(tangent) => assert_eq!(tangent, &Array::scalar(1.0)),
            MaybeZero::Zero(_) => panic!("expected a live division tangent"),
        }
    }

    #[test]
    fn test_div_low_precision_differentiation_uses_widened_tangents() {
        let left = Array::scalar(4.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let right = Array::scalar(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent) = differentiate_at((left, right))
            .jvp((Array::scalar(1.0f32), Array::scalar(1.0f32)), |(left, right)| Ok(left / right))
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap());
        assert_eq!(tangent, Array::scalar(-0.5f32));
    }

    #[test]
    fn test_div_transposition() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @approx(epsilon = 1e-12),
            operation = DivOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = scalar_type.clone())),
                        (@known, Array::scalar(3.0)),
                    ],
                    output_cotangents = [Array::scalar(2.0)],
                    input_cotangents = [Array::scalar(2.0 / 3.0)],
                    pullback = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = div %0 %1
                        in (%2)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = scalar_type)),
                        (@known, Array::from_f64s(vector_type.clone(), vec![2.0, 4.0, 5.0])),
                    ],
                    output_cotangents = [Array::from_f64s(vector_type.clone(), vec![2.0, 4.0, 10.0])],
                    input_cotangents = [Array::scalar(4.0)],
                    pullback = indoc! {"
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = div %0 %1
                            %3:f64[] = reduce_sum [axes=[0]] %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_div_for_primitives() {
        assert_eq!(Div::div(&7_usize, &2), Ok(3));
        assert_eq!(
            Div::div(&7_usize, &0),
            Err(ProgramError::InvalidArgument {
                message: "`div` divisor is zero or the result does not fit in usize".to_string(),
            }),
        );
        assert_eq!(
            Div::div(&i8::MIN, &-1),
            Err(ProgramError::InvalidArgument {
                message: "`div` divisor is zero or the result does not fit in i8".to_string(),
            }),
        );
        assert_eq!(Div::div(&1.0_f64, &0.0), Ok(f64::INFINITY));
    }

    #[test]
    fn test_div_for_array() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(Div::div(&vector, &Array::scalar(2.0)).unwrap(), Array::vector(vec![0.5, 1.0, 1.5]));
    }

    #[test]
    fn test_div_for_array_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![1.0, 2.0]);
        let right = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![0.5, 0.25]);
        assert_eq!(Div::div(&left, &right).unwrap().to_f64s(), vec![2.0, 8.0]);
    }

    #[test]
    fn test_div_for_array_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)]);
        let right = Array::vector(vec![Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)]);
        let left_values = [Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)];
        let right_values = [Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)];
        assert_abs_diff_eq!(
            Div::div(&left, &right).unwrap(),
            Array::vector(vec![left_values[0] / right_values[0], left_values[1] / right_values[1]]),
            epsilon = 1e-12,
        );
        // Ratio-based division can still overflow when both denominator components are near the largest value.
        let large = Array::scalar(Complex::new(1e308f64, 1e308));
        let quotient = Div::div(&large, &large).unwrap().elements::<Complex<f64>>().unwrap()[0];
        assert!(quotient.re.is_nan());
        assert_eq!(quotient.im.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn test_div_for_array_integers() {
        // Exceptional integer division inputs return the same structured errors as native-width array
        // arithmetic rather than panicking.
        assert!(matches!(
            Div::div(&Array::vector(vec![1i32]), &Array::vector(vec![0i32])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide an integer scalar of data type `i32` by zero",
        ));
        assert!(matches!(
            Div::div(&Array::vector(vec![i8::MIN]), &Array::vector(vec![-1i8])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide the minimum integer scalar of data type `i8` by -1",
        ));
    }
}
