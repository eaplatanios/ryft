use std::ops::Div as StandardDiv;

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, unary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::macros::{define_elementwise_capability, define_elementwise_operation, impl_non_transposable_operation};
use crate::programs::operations::Operation;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`LogOperation`].
pub const LOG_OPERATION_NAME: &str = "log";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise natural logarithm of one value (i.e.,
    /// `x ↦ ln(x)`, the principal branch `ln(z)` on complex operands) while preserving its array metadata. Only
    /// floating-point and complex operands are supported, and operands that still carry partial sums are rejected.
    LogOperation, LOG_OPERATION_NAME,
    Log, log,
    check_data_types = [@floating_or_complex],
    check_array_types = [@no_unreduced],
);

impl<C: Context> DifferentiableOperation<C> for LogOperation
where
    C::Type: DifferentiableType,
    C::Value: Log + StandardDiv<Output = C::Value> + ElementwiseDerivativeAlignment<C::Type>,
    LogOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(ln(x)) = dx / x.
        unary_elementwise_jvp(
            self,
            inputs,
            |input| input.log(),
            |operands| Ok(operands.input_tangent()? / operands.input_primal()?),
        )
    }
}

impl_non_transposable_operation!(LogOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise natural-logarithm capability. [`Log`] fills the same role for
    /// [`LogOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Log, log, LogOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::{TypeError, Typed};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_log() {
        assert_eq!(Scalar::from(0.5f32).log().unwrap(), 0.5f32.ln());
        assert_eq!(Scalar::from(0.5f64).log().unwrap(), 0.5f64.ln());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).log().unwrap(), bf16::from_f32(0.5f32.ln()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).log().unwrap(), f16::from_f32(0.5f32.ln()));
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Scalar::from(input).log().unwrap(), Scalar::from(input.ln()), epsilon = 1e-12);
        // The principal branch maps the negative real axis to `ln|x| + iπ`.
        assert_abs_diff_eq!(
            Scalar::from(ComplexNumber::new(-1.0f64, 0.0)).log().unwrap(),
            Scalar::from(ComplexNumber::new(0.0f64, std::f64::consts::PI)),
            epsilon = 1e-12,
        );

        let operation = LogOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), LOG_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "LogOperation");
        assert_eq!(format!("{operation}"), LOG_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.7)],
            ),
            Ok(vec![Scalar::from(0.7f64.ln())]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.7)],
            ),
            Ok(vec![Array::scalar(0.7f64.ln())]),
        );

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
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
            Ok(vec![input]),
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

        let mut builder = ProgramBuilder::<Scalar, LogOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = log %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_log_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&LogOperation, &[DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&LogOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'log' does not support input data type i32".to_string() }),
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = LogOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln(), 2.0f64.ln()]))],
            }],
        );
    }

    #[test]
    fn test_log_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogOperation,
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.7f64.ln())],
                tangent_outputs = [Array::scalar(3.0 / 0.7)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = log %0
                        %3:f64[] = div %1 %0
                    in (%2, %3)
                "},
            }],
        );
        check_gradient!(@scalar, |input| input.log(), at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.log().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(ComplexNumber::new(1.0, 0.0) / input)),
        );

        // Second-order differentiation recovers d²(ln(x))/dx² = -1/x².
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.log().unwrap(), x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            -1.0 / (0.7f64 * 0.7f64),
            epsilon = 1e-9,
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.log(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.values()[0], 1.5, epsilon = 1e-9);

        // The widened staged tangent program divides by the input converted to the widened differential
        // representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(LogOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = log %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = div %1 %3
                in (%2, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_log_partial_evaluation() {
        check_operation_partial_evaluation!(operation = LogOperation, inputs = [0.7], expected = 0.7f64.ln(),);
    }

    #[test]
    fn test_log_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
