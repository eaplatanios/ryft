use crate::arrays::{Array, NumericArrayElement};
use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_array_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AddOperation`].
pub const ADD_OPERATION_NAME: &str = "add";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that adds two numeric values elementwise, promoting their element
    /// [`DataType`](crate::arrays::DataType)s and broadcasting their [`Shape`](crate::arrays::Shape)s. Array operands
    /// that carry partial sums must both be unreduced over exactly the same mesh axes. Mixing an unreduced operand with
    /// an already reduced operand would duplicate the reduced contribution when the result is subsequently reduced.
    /// Their reduced-axis markers must likewise agree.
    AddOperation,
    ADD_OPERATION_NAME,
    Add,
    add,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    AddOperation,
    rule = [@positive, @positive],
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise addition capability. [`Add`] is the fallible Ryft counterpart to [`std::ops::Add`]
    /// that [`AddOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
    /// panicking. Value types additionally provide [`std::ops::Add`] as ergonomic (albeit panicking) sugar layered
    /// on top of this capability.
    Add,
    /// Adds `rhs` to this value.
    add(rhs),
    AddOperation,
);

define_tracer_operator!(@binary std::ops::Add, add, capability = Add, method = add);

/// Implements [`Add`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Add for $type {
            fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
                self.checked_add(*rhs).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` result does not fit in {}", ADD_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Add for $type {
            fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
                Ok(*self + *rhs)
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
    Add, add,
    operation = "add",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::add(lhs, rhs),
);

impl std::ops::Add for Array {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayType, DataType, Dimension, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension, StridedLayout, f8e4m3fn, i4,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::{EmptyRegionDriver, Typed};

    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(AddOperation::<ArrayType>::new().to_string(), "add");
    }

    #[test]
    fn test_add_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = AddOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{ADD_OPERATION_NAME}` input types are not broadcast-compatible"),
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
            operation = AddOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "`add` operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "`add` operands must be unreduced over the same axes",
                },
            ],
        );

        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = AddOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_add_interpretation() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &AddOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32).unwrap(), Array::scalar(3.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(5.5f64).unwrap()])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &AddOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap(), Array::vector(vec![3.5, -1.0]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![5.5, 1.0]).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &AddOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(Complex::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(Complex::new(0.5f64, -1.0)).unwrap()
                ],
            ),
            Ok(vec![Array::scalar(Complex::new(1.5f64, 1.0)).unwrap()]),
        );
    }

    #[test]
    fn test_add_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = AddOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            expected = Array::scalar(5.5).unwrap(),
        );
    }

    #[test]
    fn test_add_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AddOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![4.0, 1.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_add_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AddOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(-1.0).unwrap()],
                primal_outputs = [Array::scalar(7.0).unwrap()],
                tangent_outputs = [Array::scalar(2.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = add %0 %1
                        %5:f64[] = add %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_add_differentiation_low_precision() {
        // Rank-zero arrays support both half-precision variants through the ordinary array operations.
        assert_eq!(
            differentiate_at(Array::scalar(bf16::from_f32(3.0)).unwrap())
                .jvp(Array::scalar(bf16::ONE).unwrap(), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(bf16::from_f32(6.0)).unwrap(), Array::scalar(bf16::from_f32(2.0)).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(f16::from_f32(3.0)).unwrap())
                .jvp(Array::scalar(f16::ONE).unwrap(), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(f16::from_f32(6.0)).unwrap(), Array::scalar(f16::from_f32(2.0)).unwrap())),
        );
    }

    #[test]
    fn test_add_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = AddOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0).unwrap()],
                    input_cotangents = [Array::scalar(3.0).unwrap(), Array::scalar(3.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        in (%0, %0)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(vector_type.clone(), &[2.0, 3.0, 4.0]).unwrap()],
                    input_cotangents = [
                        Array::scalar(9.0).unwrap(),
                        Array::from_elements::<f64>(vector_type, &[2.0, 3.0, 4.0]).unwrap(),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                        in (%1, %0)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_add_for_primitives() {
        assert_eq!(Add::add(&2_usize, &3), Ok(5));
        assert_eq!(Add::add(&-2_i32, &3), Ok(1));
        assert_eq!(
            Add::add(&i8::MAX, &1),
            Err(ProgramError::InvalidArgument { message: "`add` result does not fit in i8".to_string() }),
        );
        assert_eq!(Add::add(&2.5_f64, &0.5), Ok(3.0));
    }

    #[test]
    fn test_add_for_array() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vector.add(&Array::scalar(1.0).unwrap()).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        // Mixed-precision operands promote to the common element data type.
        let promoted =
            Array::vector(vec![1.0f32, 2.0]).unwrap().add(&Array::vector(vec![0.5f64, 0.5]).unwrap()).unwrap();
        assert_eq!(promoted, Array::vector(vec![1.5f64, 2.5]).unwrap());
        // General broadcasting traverses arbitrary input layouts while mixed element types normalize through the
        // canonical conversion kernel.
        let left_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[1.0f32, 2.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[0.5f64, 1.0, 1.5]).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
        assert_eq!(sum.elements::<f64>(), Ok(vec![1.5, 2.0, 2.5, 2.5, 3.0, 3.5]));
        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(vector.clone() + Array::scalar(1.0).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        // Integer arithmetic wraps deterministically, matching the scalar reference backend.
        let wrapped = Array::vector(vec![255u8]).unwrap().add(&Array::vector(vec![1u8]).unwrap()).unwrap();
        assert_eq!(wrapped.elements::<u8>(), Ok(vec![0]));
    }

    #[test]
    fn test_add_for_array_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2]));
        assert_eq!(sum.to_f64s(), vec![1.5, 2.25]);
    }

    #[test]
    fn test_add_for_array_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)]).unwrap();
        let left_values = [Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)];
        let right_values = [Complex::new(0.5f64, -1.0), Complex::new(2.0f64, 0.5)];
        assert_eq!(
            left.add(&right).unwrap(),
            Array::vector(vec![left_values[0] + right_values[0], left_values[1] + right_values[1]]).unwrap(),
        );
    }

    #[test]
    fn test_add_for_array_integers() {
        // Sub-byte arithmetic wraps using the declared bit width.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(
            narrow.add(&Array::scalar(i4::new(1).unwrap()).unwrap()).unwrap().elements::<i4>(),
            Ok(vec![i4::MIN, i4::new(-7).unwrap()]),
        );
    }
}
