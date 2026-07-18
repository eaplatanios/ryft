use std::ops::{Add as StandardAdd, Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, binary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::{check_count, define_elementwise_capability, define_elementwise_operation, define_tracer_operator};
use crate::partial::PartialValue;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::values::Value;
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

impl<C: Context> DifferentiableOperation<C> for DivOperation
where
    C::Type: DifferentiableType,
    C::Value: StandardNeg<Output = C::Value>
        + StandardAdd<Output = C::Value>
        + StandardMul<Output = C::Value>
        + StandardDiv<Output = C::Value>
        + ElementwiseDerivativeAlignment<C::Type>,
    DivOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(x / y) = dx / y - (x / y²) · dy.
        binary_elementwise_jvp(
            self,
            inputs,
            |left, right| Ok(left.clone() / right.clone()),
            |operands, tangent| Ok(tangent / operands.right_primal()?),
            |operands, tangent| {
                let right = operands.right_primal()?;
                let coefficient = -(operands.left_primal()? / (right.clone() * right));
                Ok(coefficient * tangent)
            },
        )
    }
}

/// Transposes division when its numerator is linear and its denominator is a known runtime value.
impl<V: Value, O: Operation<V::Type> + From<DivOperation>> TransposableOperation<V, O> for DivOperation
where
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    DivOperation: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if !inputs[0].is_unknown() {
            return Err(ProgramError::UnsupportedOperation {
                message: "'div' with no linear numerator cannot be transposed".to_string(),
            }
            .into());
        }
        if inputs[1].is_unknown() {
            return Err(ProgramError::UnsupportedOperation {
                message: "'div' with a linear denominator is nonlinear and cannot be transposed".to_string(),
            }
            .into());
        }
        let numerator_type = inputs[0].r#type().cotangent();
        let numerator_contribution = match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(numerator_type),
            MaybeZero::Value(output_cotangent) => {
                if numerator_type.is_zero_space() {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "'div' numerator has no cotangent space".to_string(),
                    }
                    .into());
                }
                // The `is_unknown` check above guarantees the denominator is a known operand.
                let denominator = inputs[1].as_known().unwrap();
                let denominator = denominator.align_tangent(output_cotangent.r#type().as_ref())?;
                MaybeZero::Value(
                    output_cotangent.binary(&denominator, DivOperation).unalign_cotangent(&numerator_type)?,
                )
            }
        };
        Ok(vec![numerator_contribution, MaybeZero::Zero(inputs[1].r#type().cotangent())])
    }
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise division capability. [`Div`] is the fallible Ryft counterpart to [`std::ops::Div`]
    /// that [`DivOperation`] interprets through, surfacing a [`ProgramError`] when something
    /// goes wrong, instead of panicking. Value types additionally provide [`std::ops::Div`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
    Div, div, DivOperation,
);

define_tracer_operator!(@binary std::ops::Div, div, DivOperation, "`div` operation failed");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::OneLike;
    use crate::operations::manipulation::ConvertElementType;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_div() {
        let operation = DivOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), DIV_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "DivOperation");
        assert_eq!(format!("{operation}"), DIV_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(7.0f32), Scalar::from(2.0f64)],
            ),
            Ok(vec![Scalar::from(3.5f64)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.0), Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(3.5)]),
        );
        assert_abs_diff_eq!(
            match InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, 2.0)), Scalar::from(Complex::new(0.5f64, -1.0))],
            ) {
                Ok(outputs) => outputs[0].clone(),
                Err(error) => panic!("expected a complex quotient but got {error}"),
            },
            Scalar::from(Complex::new(1.0f64, 2.0) / Complex::new(0.5f64, -1.0)),
            epsilon = 1e-12,
        );

        // Array type inference broadcasts shapes and promotes data types.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))]);

        // Array type inference drops layout metadata when inputs disagree.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::scalar()).with_layout(Layout::Strided(StridedLayout::new(vec![]))),
                ArrayType::scalar(DataType::F32),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);

        // Array type inference tolerates compatible inputs that only disagree on varying manual axes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let right = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let output =
            <DivOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right], &[]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, DivOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = div %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_div_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&DivOperation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{DIV_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean] {
            let expected =
                TypeError { message: format!("'{DIV_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&DivOperation, &[input_type, input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(
                    &DivOperation,
                    &[ArrayType::scalar(input_type), ArrayType::scalar(input_type)],
                    &[],
                ),
                Err(expected),
            );
        }
        assert_eq!(
            <DivOperation as Operation<ArrayType>>::infer_output_types(
                &DivOperation,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError { message: format!("'{DIV_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]));
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
        let expected = Err(TypeError { message: "'div' does not support unreduced operands".to_string() });
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&DivOperation, &[unreduced.clone(), plain.clone()], &[]),
            expected,
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&DivOperation, &[plain, unreduced.clone()], &[]),
            expected,
        );
        assert_eq!(Operation::<ArrayType>::infer_output_types(&DivOperation, &[unreduced, reduced], &[]), expected);
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = DivOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_div_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = DivOperation,
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
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = DivOperation,
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
        let smallest_positive = f64::from_bits(1);
        let outputs = DivOperation
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(Scalar::from(0.0), Scalar::from(smallest_positive)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Scalar::from(smallest_positive)),
                ],
            )
            .unwrap();
        match outputs[0].tangent() {
            MaybeZero::Value(tangent) => assert_eq!(tangent, &Scalar::from(1.0)),
            MaybeZero::Zero(_) => panic!("expected a live division tangent"),
        }
        fn normalized_ratio<V>(input: V) -> V
        where
            V: Clone + OneLike + std::ops::Add<Output = V> + std::ops::Div<Output = V>,
        {
            input.clone() / (input.clone() + input.one_like())
        }
        check_gradient!(@scalar, normalized_ratio, at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = Complex::new(0.7f64, -0.3);
        let holomorphic_gradient = gradient_holomorphic(normalized_ratio, Scalar::from(input)).unwrap();
        assert_abs_diff_eq!(
            holomorphic_gradient,
            Scalar::from(Complex::new(1.0, 0.0) / (input + 1.0).powu(2)),
            epsilon = 1e-12,
        );

        // Second-order differentiation recovers d²(x / (x + 1))/dx² = -2 / (x + 1)³.
        assert_abs_diff_eq!(
            gradient(|x| gradient(normalized_ratio, x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            -2.0 / (1.7f64 * 1.7 * 1.7),
            epsilon = 1e-9,
        );

        let left = Scalar::from(4.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let right = Scalar::from(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent): (Scalar, Scalar) = context
            .jvp(|(left, right)| Ok(left / right), (left, right), (Scalar::from(1.0f32), Scalar::from(1.0f32)))
            .unwrap();
        assert_eq!(primal, Scalar::from(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap());
        assert_eq!(tangent, Scalar::from(-0.5f32));
    }

    #[test]
    fn test_div_partial_evaluation() {
        check_operation_partial_evaluation!(operation = DivOperation, inputs = [7.0, 2.0], expected = 3.5,);
    }

    #[test]
    fn test_div_transposition() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        check_operation_transposition!(
            @approx(epsilon = 1e-12),
            operation = DivOperation,
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
}
