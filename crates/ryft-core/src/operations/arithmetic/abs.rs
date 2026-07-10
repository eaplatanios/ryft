use std::fmt::Display;
use std::ops::{Div, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDual, DifferentiationError, TransposableOperation,
};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::complex::{Conjugate, Real};
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`AbsOperation`].
pub const ABS_OPERATION_NAME: &'static str = "abs";

/// [`Operation`] that computes the elementwise absolute value of one value (i.e., `x ↦ |x|`, the magnitude `|z|` on
/// complex operands with a real result) while preserving all other type metadata. This is the analogue of
/// [JAX's `lax.abs`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html).
#[derive(Clone, Debug, Default)]
pub struct AbsOperation;

impl Display for AbsOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ABS_OPERATION_NAME)
    }
}

impl Operation<DataType> for AbsOperation {
    #[inline]
    fn name(&self) -> &'static str {
        ABS_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![match &input_types[0] {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other.clone(),
        }])
    }
}

impl Operation<ArrayType> for AbsOperation {
    #[inline]
    fn name(&self) -> &'static str {
        ABS_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![ArrayType {
            data_type: match input_types[0].data_type() {
                DataType::C64 => DataType::F32,
                DataType::C128 => DataType::F64,
                other => other,
            },
            ..input_types[0].clone()
        }])
    }
}

impl ElementwiseOperation for AbsOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types)
    }
}

impl<V: Clone + Value + Abs, C> InterpretableOperation<V, C> for AbsOperation
where
    Self: Operation<V::Type>,
{
    #[inline]
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].abs()?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for AbsOperation where C::Operation: From<AbsOperation> {}

// TODO(eaplatanios): Review this implementation.
impl<C: Context<Value: Abs + Conjugate + Real + Mul<Output = C::Value> + Div<Output = C::Value>, Operation: Clone>>
    DifferentiableOperation<C> for AbsOperation
where
    AbsOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The real derivative is `d|x| = (x / |x|) · dx` (i.e., `sign(x) · dx` and undefined at zero), and the complex
        // magnitude is a ℂ → ℝ map with `d|z| = Re(z̄ · dz) / |z|`, so the complex branch routes the tangent through
        // `conjugate` and `real` while the real branch reuses the primal directly. A structural zero tangent stays
        // symbolic, retyped to the (real) output type.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().abs()?;
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => {
                let numerator = if input.primal().r#type().is_complex() {
                    (input.primal().conjugate()? * tangent.clone()).real()?
                } else {
                    input.primal().clone() * tangent.clone()
                };
                MaybeZero::Value(numerator / primal.clone())
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AbsOperation
where
    AbsOperation: Operation<V::Type>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        // The absolute value is nonlinear in its operand and so it cannot be transposed. The reason we still implement
        // `TransposableOperation` and return a `ProgramError::UnsupportedOperation` is because we need the whole
        // operation family of a `Context` to implement `TransposableOperation`. Ryft used to have a separate linear
        // operation family type but that resulted in overly complicated backend and operation implementations for very
        // little benefit in practice, and so we ended up unifying them.
        Err(ProgramError::UnsupportedOperation { message: format!("operation `{}` is not transposable", self.name()) }
            .into())
    }
}

/// Value-level elementwise absolute-value capability. [`Abs`] fills the same role for [`AbsOperation`] that
/// [`Neg`](crate::Neg) fills for [`NegOperation`](crate::NegOperation).
pub trait Abs: Sized {
    /// Computes the elementwise absolute value of this value (i.e., the magnitude for complex values, with a real
    /// result), returning a [`ProgramError`] if something goes wrong (e.g., when the value's data type carries no
    /// absolute value, such as a Boolean).
    fn abs(&self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<AbsOperation>>>> Abs for V {
    #[inline]
    fn abs(&self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(AbsOperation, &[self.clone()])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, value_and_gradient};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::{TestArray, check_gradient};
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_abs() {
        let operation = AbsOperation;

        // Operation identity and concrete interpretation, including the complex magnitude with its real result.
        assert_eq!(Operation::<DataType>::name(&operation), ABS_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "AbsOperation");
        assert_eq!(format!("{operation}"), ABS_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]));
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::C64]), Ok(vec![DataType::F32]));
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::C128]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(-2.0)],
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(ComplexNumber::new(3.0f64, -4.0f64))],
            ),
            Ok(vec![Scalar::from(5.0)]),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(-2.0)],
            ),
            Ok(vec![TestArray::scalar(2.0)]),
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
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <AbsOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );

        // Complex array element types map to their real part data type while the shape is preserved.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[ArrayType::new(DataType::C128, Shape::new(vec![Size::Static(2)]))],
            ),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(&operation, &EagerContext::new(), &[],),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(true)],
            ),
            Err(ProgramError::Type(TypeError {
                message: "cannot compute the absolute value of a scalar of data type bool".to_string(),
            })),
        );

        // Program rendering uses the canonical operation name, with the complex magnitude typed by its real part.
        let mut builder = ProgramBuilder::<Scalar, AbsOperation>::new();
        let input = builder.add_input(DataType::C128);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
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

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_abs_derivative() {
        // d|x| = sign(x) away from the kink at zero is +1 above zero and -1 below zero.
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(0.7f64)).unwrap(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(-0.7f64)).unwrap(), -1.0, epsilon = 1e-9);
        check_gradient!(|x| x.abs().unwrap(), 0.7, 1e-6, 1e-6);
        check_gradient!(|x| x.abs().unwrap(), -2.5, 1e-6, 1e-6);

        // |z| is a ℂ → ℝ function and so it flows through the plain gradient entry point. With d|z| = Re(z̄ · dz) / |z|,
        // the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction), the reverse-mode
        // counterpart of `∇|z|² = 2z̄` after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) = value_and_gradient(|z| z.abs().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.norm()));
        let expected = z.conj() / z.norm();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - expected).norm() < 1e-12, "expected {expected} but got {actual}");
        check_gradient!(|z| z.abs().unwrap(), z, 1e-6, 1e-6);
    }
}
