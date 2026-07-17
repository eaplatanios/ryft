use std::ops::Mul as StandardMul;

use crate::contexts::Context;
use crate::define_elementwise_capability;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::{check_count, define_elementwise_operation};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::operations::broadcasting::ElementwiseDifferentiableValue;

/// Canonical operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "exp";

define_elementwise_operation!(
    @unary_base
    /// [`Operation`] that computes the elementwise natural exponential of one value (i.e.,
    /// `x ↦ eˣ`, the analytic continuation `e^z` on complex operands) while preserving its array metadata. Only
    /// floating-point and complex operands are supported, and operands that still carry partial sums are rejected.
    ExpOperation, EXP_OPERATION_NAME,
    Exp, exp,
    validate = super::validate_floating_or_complex_input_types,
    validate_array = super::validate_no_unreduced_inputs,
);

impl<C: Context> DifferentiableOperation<C> for ExpOperation
where
    C::Type: DifferentiableType,
    C::Value: Exp + StandardMul<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
    ExpOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().exp()?;
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'exp' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(target),
            MaybeZero::Value(tangent) => {
                let coefficient = input.primal().normalize_elementwise_tangent(&target)?.exp()?;
                MaybeZero::Value(coefficient * tangent.normalize_elementwise_tangent(&target)?)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Reports that a primal exponential is nonlinear and cannot occur on a linear operand in a valid tangent program.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for ExpOperation
where
    ExpOperation: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: format!("operation `{}` is not transposable", self.name()) }
            .into())
    }
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise natural-exponential capability. [`Exp`] fills the same role for
    /// [`ExpOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Exp, exp, ExpOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::gradient_holomorphic;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::{TypeError, Typed};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::{TestArray, check_gradient};
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_exp() {
        let operation = ExpOperation;
        assert_eq!(Operation::<DataType>::name(&operation), EXP_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ExpOperation");
        assert_eq!(format!("{operation}"), EXP_OPERATION_NAME);
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
            Ok(vec![Scalar::from(0.7f64.exp())]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(0.7)],
            ),
            Ok(vec![TestArray::scalar(0.7f64.exp())]),
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
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        let mut builder = ProgramBuilder::<Scalar, ExpOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = exp %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_exp_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&ExpOperation, &[DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&ExpOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'exp' does not support input data type i32".to_string() }),
        );
        crate::operations::math::tests::assert_rejects_unreduced(ExpOperation, EXP_OPERATION_NAME, 1);
    }

    #[test]
    fn test_exp_batching() {
        crate::operations::math::tests::assert_unary_batching(
            ExpOperation,
            &[0.5, -1.0],
            &[0.5f64.exp(), (-1.0f64).exp()],
        );
    }

    #[test]
    fn test_exp_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context.jvp(|input| input.exp(), Scalar::from(0.7), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(primal, 0.7f64.exp(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 * 0.7f64.exp(), epsilon = 1e-9);
        check_gradient!(|input| input.exp().unwrap(), 0.7, 1e-6, 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.exp().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(input.exp())),
        );

        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primal = TestArray::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.exp(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.values()[0], 3.0 * 2.0f64.exp(), epsilon = 1e-9);

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(ExpOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = exp %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = exp %3
                    %5:f32[] = mul %4 %1
                in (%2, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_exp_transposition() {
        crate::operations::math::tests::assert_rejects_nonlinear_transposition(ExpOperation, EXP_OPERATION_NAME, 1);
    }
}
