use std::ops::{Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::complex::{Complex, Conjugate, Imaginary, Real};
use crate::operations::constants::{OneLike, ZeroLike};
use crate::operations::control_flow::{Select, SelectCondition};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::types::{Type, TypeError, Typed};
use crate::types::DataType;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AbsOperation`].
pub const ABS_OPERATION_NAME: &str = "abs";

/// Infers the result [`DataType`]s of an absolute-value operation from its input element data types.
fn infer_abs_output_data_types(input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
    let output_type = match input_types[0] {
        DataType::I2
        | DataType::I4
        | DataType::I8
        | DataType::I16
        | DataType::I32
        | DataType::I64
        | DataType::F4E2M1FN
        | DataType::F6E2M3FN
        | DataType::F6E3M2FN
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64 => input_types[0],
        DataType::C64 => DataType::F32,
        DataType::C128 => DataType::F64,
        input_type => Err(TypeError {
            message: format!("cannot compute the absolute value of a value of data type {input_type}"),
        })?,
    };
    Ok(vec![output_type])
}

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise absolute value of one value (i.e., `x ↦ |x|`, the magnitude `|z|`
    /// on complex operands with a real result) while preserving all other type metadata. Inputs that still represent
    /// partial sums over unreduced mesh axes are rejected because taking an absolute value does not preserve
    /// partial-sum semantics. Matching the operand constraints of
    /// [StableHLO's `abs`](https://openxla.org/stablehlo/spec#abs), signed-integer (including the sub-byte `si2` and
    /// `si4` types, with the minimum value wrapping to itself), floating-point, and complex inputs are supported,
    /// while unsigned-integer, Boolean, token, structural-zero, and single-bit `si1` inputs (whose only negative value
    /// `-1` has no representable absolute value) are rejected.
    AbsOperation, ABS_OPERATION_NAME,
    Abs, abs,
    infer_data_types = infer_abs_output_data_types,
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @custom
    AbsOperation,
    jvp<C>
    where
        C::Type: DifferentiableType,
        C::Value: Abs
            + Compare<Output = C::Value>
            + Complex
            + Conjugate
            + Imaginary
            + Real
            + Select<Condition = <C::Value as SelectCondition>::Condition>
            + SelectCondition
            + ZeroLike
            + OneLike
            + StandardNeg<Output = C::Value>
            + StandardMul<Output = C::Value>
            + StandardDiv<Output = C::Value>
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, _context, _driver, inputs| {
            // Away from zero, the real derivative is `d|x| = sign(x) · dx`, while the complex magnitude is a
            // ℂ → ℝ map with `d|z| = Re(z̄ · dz) / |z|`. At the real origin, choose the right derivative and return
            // `dx`; at the complex origin, replace the zero denominator with one so the zero numerator yields zero.
            // These conventions keep the rule finite and stable under higher-order transforms. A structural zero
            // tangent stays symbolic, retyped to the real output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().abs()?;
            let target = primal.r#type().tangent();
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(target),
                MaybeZero::Value(_) if target.is_zero_space() => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!("'abs' output type {} has no tangent space", primal.r#type()),
                    }
                    .into());
                }
                MaybeZero::Value(tangent) => {
                    if input.primal().r#type().is_complex() {
                        let denominator = primal.align_tangent(&target)?;
                        let zero = denominator.zero_like();
                        let one = denominator.one_like();
                        let denominator_is_zero =
                            denominator.compare(&zero, ComparisonDirection::Equal)?.select_condition()?;
                        let denominator = C::Value::select(&denominator_is_zero, &one, &denominator)?;
                        // Normalize `conj(z) / |z|` before multiplying by `dz`. Computing `conj(z) * dz` first is
                        // algebraically equivalent but can overflow even when the final directional derivative is
                        // finite.
                        let conjugate = input.primal().conjugate()?;
                        let coefficient =
                            (conjugate.real()? / denominator.clone()).complex(&(conjugate.imaginary()? / denominator))?;
                        let input_target = input.primal().r#type().tangent();
                        MaybeZero::Value(
                            (tangent.align_tangent(&input_target)? * coefficient).real()?.align_tangent(&target)?,
                        )
                    } else {
                        let input = input.primal().align_tangent(&target)?;
                        let tangent = tangent.align_tangent(&target)?;
                        let zero = input.zero_like();
                        let nonnegative =
                            input.compare(&zero, ComparisonDirection::GreaterThanOrEqual)?.select_condition()?;
                        MaybeZero::Value(C::Value::select(&nonnegative, &tangent, &-tangent.clone())?)
                    }
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise absolute-value capability. [`Abs`] fills the same role for [`AbsOperation`] that
    /// [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Abs,
    /// Computes the elementwise absolute value of this value (i.e., the magnitude for complex values, with a real
    /// result), returning a [`ProgramError`] if something goes wrong (e.g., when the value's data type carries no
    /// absolute value, such as a Boolean).
    abs,
    AbsOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, value_and_gradient};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition,
    };
    use crate::operations::math::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_abs() {
        let operation = AbsOperation;

        // Operation identity and concrete interpretation, including the complex magnitude with its real result.
        assert_eq!(Operation::<DataType>::name(&operation), ABS_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "AbsOperation");
        assert_eq!(format!("{operation}"), ABS_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(-2.0)],
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(ComplexNumber::new(3.0f64, -4.0f64))],
            ),
            Ok(vec![Scalar::from(5.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(-2.0)],
            ),
            Ok(vec![Array::scalar(2.0)]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <AbsOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(true)],
            ),
            Err(ProgramError::Type(TypeError {
                message: "cannot compute the absolute value of a scalar of data type bool".to_string(),
            })),
        );

        // Program rendering uses the canonical operation name, with the complex magnitude typed by its real part.
        let mut builder = ProgramBuilder::<Scalar, AbsOperation>::new();
        let input = builder.add_input(DataType::C128);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:c128 .
                let %1:f64 = abs %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_abs_type_inference() {
        // Signed-integer inputs, including the sub-byte `si2` and `si4` types, pass through unchanged.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AbsOperation, &[DataType::I32], &[]),
            Ok(vec![DataType::I32]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AbsOperation, &[DataType::I2], &[]),
            Ok(vec![DataType::I2]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AbsOperation, &[DataType::I4], &[]),
            Ok(vec![DataType::I4]),
        );

        // Complex element types map to their real part data type, preserving the shape for arrays.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AbsOperation, &[DataType::C64], &[]),
            Ok(vec![DataType::F32]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&AbsOperation, &[DataType::C128], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &AbsOperation,
                &[ArrayType::new(DataType::C128, Shape::new(vec![Size::Static(2)]))],
                &[],
            ),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
        );

        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::I1, DataType::U32] {
            let expected = TypeError {
                message: format!("cannot compute the absolute value of a value of data type {input_type}"),
            };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&AbsOperation, &[input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(&AbsOperation, &[ArrayType::scalar(input_type)], &[]),
                Err(expected),
            );
        }

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&AbsOperation, &[input], &[]),
            Err(TypeError { message: "'abs' does not support unreduced operands".to_string() }),
        );
    }

    #[test]
    fn test_abs_partial_evaluation() {
        check_operation_partial_evaluation!(operation = AbsOperation, inputs = [-2.0], expected = 2.0,);
    }

    #[test]
    fn test_abs_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AbsOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
            }],
        );
    }

    #[test]
    fn test_abs_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AbsOperation,
            cases = [
                {
                    primals = [Array::scalar(0.7)],
                    tangents = [Array::scalar(3.0)],
                    primal_outputs = [Array::scalar(0.7)],
                    tangent_outputs = [Array::scalar(3.0)],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = abs %0
                            %3:f64[] = zero_like %0
                            %4:bool[] = compare [direction=GreaterThanOrEqual] %0 %3
                            %5:f64[] = neg %1
                            %6:f64[] = select %4 %1 %5
                        in (%2, %6)
                    "},
                },
                {
                    primals = [Array::scalar(-2.5)],
                    tangents = [Array::scalar(2.0)],
                    primal_outputs = [Array::scalar(2.5)],
                    tangent_outputs = [Array::scalar(-2.0)],
                },
            ],
        );
        // The real rule uses +1 at and above zero and -1 below zero.
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(0.7f64)).unwrap(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(0.0f64)).unwrap(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(-0.7f64)).unwrap(), -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.abs().unwrap(), x).unwrap(), Scalar::from(0.0f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );
        check_gradient!(@scalar, |x| x.abs(), at = 0.7, step = 1e-6, tolerance = 1e-6);
        check_gradient!(@scalar, |x| x.abs(), at = -2.5, step = 1e-6, tolerance = 1e-6);

        // |z| is a ℂ → ℝ function and so it flows through the plain gradient entry point. With
        // d|z| = Re(z̄ · dz) / |z|, the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction):
        // the reverse-mode counterpart of ∇|z|² = 2z̄ after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) = value_and_gradient(|z| z.abs().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.norm()));
        let expected = z.conj() / z.norm();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - expected).norm() < 1e-12, "expected {expected} but got {actual}");
        check_gradient!(@scalar, |z| z.abs(), at = z, step = 1e-6, tolerance = 1e-6);

        // The array universe agrees: summing the elementwise magnitudes of a complex vector is again ℂⁿ → ℝ, and the
        // finite-difference oracle perturbs each element's real and imaginary parts independently.
        check_gradient!(
            @array,
            |z| z.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![ComplexNumber::new(0.7f64, -0.3), ComplexNumber::new(-1.2f64, 0.8)]),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // The complex rule replaces a zero magnitude denominator with one, so the zero numerator produces a finite
        // zero tangent and gradient at the origin.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(
            context.jvp(
                |z| z.abs(),
                Scalar::from(ComplexNumber::new(0.0f64, 0.0f64)),
                Scalar::from(ComplexNumber::new(1.0f64, 2.0f64)),
            ),
            Ok((Scalar::from(0.0f64), Scalar::from(0.0f64))),
        );
        assert_eq!(
            gradient(|z| z.abs().unwrap(), Scalar::from(ComplexNumber::new(0.0f64, 0.0f64))),
            Ok(Scalar::from(ComplexNumber::new(0.0f64, 0.0f64))),
        );

        // Normalizing the complex coefficient before applying the tangent avoids overflowing the otherwise finite
        // directional derivative `Re((conj(z) / |z|) * dz)`.
        assert_eq!(
            context.jvp(
                |z| z.abs(),
                Scalar::from(ComplexNumber::new(1e308f64, 0.0)),
                Scalar::from(ComplexNumber::new(2.0f64, 0.0)),
            ),
            Ok((Scalar::from(1e308f64), Scalar::from(2.0f64))),
        );

        // The coefficient and tangent are computed in the widened differential representation.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);

        let (_, tangent) = context.jvp(|input| input.abs(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_abs_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = AbsOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
