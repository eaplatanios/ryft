use crate::arrays::{Array, NumericArrayElement};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_array_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SubOperation`].
pub const SUB_OPERATION_NAME: &str = "sub";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that subtracts two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that carry partial sums must both be unreduced over exactly the same
    /// mesh axes (subtraction is linear, so the difference of two partial sums over the same axes is another valid
    /// partial sum); mixing an unreduced operand with an already reduced operand would duplicate the reduced
    /// contribution when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    SubOperation, SUB_OPERATION_NAME,
    Sub, sub,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    SubOperation,
    rule = [@positive, @negative]
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise subtraction capability. [`Sub`] is the fallible Ryft counterpart to [`std::ops::Sub`]
    /// that [`SubOperation`] interprets through, surfacing a [`ProgramError`] when something
    /// goes wrong, instead of panicking. Value types additionally provide [`std::ops::Sub`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
    Sub,
    /// Subtracts `right` from this value, returning a [`ProgramError`] if something goes wrong.
    sub(right),
    SubOperation,
);

define_tracer_operator!(@binary std::ops::Sub, sub, capability = Sub, method = sub);

/// Implements [`Sub`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Sub for $type {
            fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_sub(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` result does not fit in {}", SUB_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Sub for $type {
            fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self - *right)
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
    Sub, sub,
    operation = "sub",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::sub(lhs, rhs),
);

impl std::ops::Sub for Array {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayType, DataType, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
        i4,
    };
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_sub() {
        assert_eq!(SubOperation::<ArrayType>::new().to_string(), "sub");
    }

    #[test]
    fn test_sub_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = SubOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{SUB_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        check_operation_type_inference!(
            operation = SubOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "`sub` operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "`sub` operands must be unreduced over the same axes",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = SubOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sub_interpretation() {
        let operation = SubOperation::<ArrayType>::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32).unwrap(), Array::scalar(3.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(-1.5f64).unwrap()])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &SubOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            ),
            Ok(vec![Array::scalar(-1.5).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(Complex::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(Complex::new(0.5f64, -1.0)).unwrap()
                ],
            ),
            Ok(vec![Array::scalar(Complex::new(0.5f64, 3.0)).unwrap()]),
        );
    }

    #[test]
    fn test_sub_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SubOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            expected = Array::scalar(-1.5).unwrap(),
        );
    }

    #[test]
    fn test_sub_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SubOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-2.0, -5.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_sub_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SubOperation::new(),
            cases = [{
                primals = [Array::scalar(5.0).unwrap(), Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(3.0).unwrap()],
                tangent_outputs = [Array::scalar(2.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = sub %0 %1
                        %5:f64[] = sub %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_sub_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = SubOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0).unwrap()],
                    input_cotangents = [Array::scalar(3.0).unwrap(), Array::scalar(-3.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        let %1:f64[] = neg %0
                        in (%0, %1)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_f64s(vector_type.clone(), vec![2.0, 3.0, 4.0]).unwrap()],
                    input_cotangents = [
                        Array::scalar(9.0).unwrap(),
                        Array::from_f64s(vector_type, vec![-2.0, -3.0, -4.0]).unwrap(),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                            %2:f64[3] = neg %0
                        in (%1, %2)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_sub_for_primitives() {
        assert_eq!(Sub::sub(&5_usize, &3), Ok(2));
        assert_eq!(
            Sub::sub(&0_usize, &1),
            Err(ProgramError::InvalidArgument { message: "`sub` result does not fit in usize".to_string() }),
        );
        assert_eq!(Sub::sub(&2.5_f32, &0.5), Ok(2.0));
    }

    #[test]
    fn test_sub_for_array() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            vector.sub(&Array::vector(vec![0.5, 1.0, 1.5]).unwrap()).unwrap(),
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap()
        );
    }

    #[test]
    fn test_sub_for_array_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![1.0, 2.0]).unwrap();
        let right = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![0.5, 0.25]).unwrap();
        assert_eq!(left.sub(&right).unwrap().to_f64s(), vec![0.5, 1.75]);
    }

    #[test]
    fn test_sub_for_array_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)]).unwrap();
        let left_values = [Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)];
        let right_values = [Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)];
        assert_eq!(
            left.sub(&right).unwrap(),
            Array::vector(vec![left_values[0] - right_values[0], left_values[1] - right_values[1]]).unwrap(),
        );
    }

    #[test]
    fn test_sub_for_array_integers() {
        // Sub-byte arithmetic uses the declared bit width for every wrapping operation.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(
            narrow.sub(&Array::scalar(i4::new(1).unwrap()).unwrap()).unwrap().elements::<i4>(),
            Ok(vec![i4::new(6).unwrap(), i4::new(7).unwrap()]),
        );
    }
}
