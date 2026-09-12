use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, ArrayElement, DataType};
use crate::macros::{
    check_types, define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    dispatch_on_array_element_type, impl_differentiable_elementwise_operation,
};
use crate::programs::{ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &str = "neg";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that negates one integer, floating-point, or complex value while preserving its array metadata
    /// and reduction state. Boolean, token, structural-zero, and the unsigned-only `f8e8m0fnu` data types are rejected.
    NegOperation, NEG_OPERATION_NAME,
    Neg, neg,
    infer_data_types = |input_types: &[DataType]| {
        check_types!(@numeric, NEG_OPERATION_NAME, input_types);
        let input_type = input_types[0];
        if input_type == DataType::F8E8M0FNU {
            return Err(TypeError::invalid(format!(
                "`{NEG_OPERATION_NAME}` does not support input data type `f8e8m0fnu`",
            )));
        }
        Ok(vec![input_type])
    },
);

impl_differentiable_elementwise_operation! {
    @linear
    NegOperation,
    rule = [@negative]
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise negation capability. [`Neg`] is the fallible Ryft counterpart to [`std::ops::Neg`]
    /// that [`NegOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
    /// panicking. Value types additionally provide [`std::ops::Neg`] as ergonomic (albeit panicking) sugar layered on
    /// top of this capability.
    Neg,
    /// Negates `self`, returning a [`ProgramError`] if something goes wrong.
    neg,
    NegOperation,
);

define_tracer_operator!(@unary std::ops::Neg, neg, NegOperation, "`neg` operation failed");

/// Implements [`Neg`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Signed integer primitives use checked negation so that the `MIN` overflow reports an error instead of
    // wrapping like the XLA-mirroring reference backends do on devices.
    (@signed $type:ty) => {
        impl Neg for $type {
            fn neg(&self) -> Result<Self, ProgramError> {
                self.checked_neg().ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` result does not fit in {}", NEG_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 negation, which cannot fail.
    (@float $type:ty) => {
        impl Neg for $type {
            fn neg(&self) -> Result<Self, ProgramError> {
                Ok(-*self)
            }
        }
    };
}

impl_capability_for_primitive!(@signed i8);
impl_capability_for_primitive!(@signed i16);
impl_capability_for_primitive!(@signed i32);
impl_capability_for_primitive!(@signed i64);
impl_capability_for_primitive!(@signed i128);
impl_capability_for_primitive!(@signed isize);
impl_capability_for_primitive!(@float f32);
impl_capability_for_primitive!(@float f64);

impl Neg for Array {
    fn neg(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_numeric() {
            return Err(TypeError::invalid(format!("cannot negate a scalar of data type `{data_type}`")).into());
        }
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.map_elements::<Element, Element>(self.r#type().into_owned(), <Element as ArrayElement>::neg)
        })
    }
}

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayType, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, i4,
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
    fn test_neg() {
        assert_eq!(NegOperation::<ArrayType>::new().to_string(), "neg");
    }

    #[test]
    fn test_neg_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = NegOperation,
            cases = [{
                input_data_types = [DataType::F64],
                output_data_types = [DataType::F64],
            }],
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::F8E8M0FNU] {
            let message = format!("`{NEG_OPERATION_NAME}` does not support input data type `{input_type}`");
            check_operation_type_inference!(
                @elementwise @unary,
                operation = NegOperation,
                cases = [{
                    input_data_types = [input_type],
                    error = message,
                }],
            );
        }

        // Negation is linear, so partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = NegOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced.clone()],
                output_types = [unreduced],
            }],
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = NegOperation::<ArrayType>::new(),
            cases = [{
                input_types = [reduced.clone()],
                output_types = [reduced],
            }],
        );
    }

    #[test]
    fn test_neg_interpretation() {
        let operation = NegOperation::<ArrayType>::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap()],
            ),
            Ok(vec![Array::scalar(-2.0).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1u8).unwrap()],
            ),
            Ok(vec![Array::scalar(u8::MAX).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &NegOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap()],
            ),
            Ok(vec![Array::scalar(-2.0).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(Complex::new(1.0f64, -2.0)).unwrap()],
            ),
            Ok(vec![Array::scalar(Complex::new(-1.0f64, 2.0)).unwrap()]),
        );
    }

    #[test]
    fn test_neg_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = NegOperation::new(),
            inputs = [Array::scalar(2.0).unwrap()],
            expected = Array::scalar(-2.0).unwrap(),
        );
    }

    #[test]
    fn test_neg_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = NegOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-1.0, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_neg_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = NegOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(-2.0).unwrap()],
                tangent_outputs = [Array::scalar(-3.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %0
                        %3:f64[] = neg %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_neg_transposition() {
        check_operation_transposition!(
            @exact,
            operation = NegOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(-3.0).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = neg %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_neg_for_primitives() {
        assert_eq!(Neg::neg(&5_i32), Ok(-5));
        assert_eq!(
            Neg::neg(&i8::MIN),
            Err(ProgramError::InvalidArgument { message: "`neg` result does not fit in i8".to_string() }),
        );
        assert_eq!(Neg::neg(&2.5_f64), Ok(-2.5));
    }

    #[test]
    fn test_neg_for_array() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vector.neg().unwrap(), Array::vector(vec![-1.0, -2.0, -3.0]).unwrap());
        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(-vector.clone(), Array::vector(vec![-1.0, -2.0, -3.0]).unwrap());
    }

    #[test]
    fn test_neg_for_array_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![1.0, 2.0]).unwrap();
        assert_eq!(left.neg().unwrap().to_f64s(), vec![-1.0, -2.0]);
    }

    #[test]
    fn test_neg_for_array_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)]).unwrap();
        let left_values = [Complex::new(1.0f64, 2.0), Complex::new(0.5f64, -1.0)];
        assert_eq!(left.neg().unwrap(), Array::vector(vec![-left_values[0], -left_values[1]]).unwrap());
    }

    #[test]
    fn test_neg_for_array_integers() {
        // Negation wraps deterministically for unsigned and two's-complement signed elements, matching the scalar
        // reference backend (and StableHLO's integer semantics), rather than panicking or saturating.
        let unsigned = Array::vector(vec![0u8, 1, 255]).unwrap();
        assert_eq!(unsigned.neg().unwrap().elements::<u8>(), Ok(vec![0, 255, 1]));
        let minimum = Array::vector(vec![i8::MIN, -5]).unwrap();
        assert_eq!(minimum.neg().unwrap().elements::<i8>(), Ok(vec![i8::MIN, 5]));
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(narrow.neg().unwrap().elements::<i4>(), Ok(vec![i4::new(-7).unwrap(), i4::MIN]));
    }
}
