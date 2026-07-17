use std::ops::{Add, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::math::MulOperation;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::programs::types::Typed;
use crate::tracing_v2::operations::broadcasting::ElementwiseDifferentiableValue;

impl<C: Context> DifferentiableOperation<C> for MulOperation
where
    MulOperation: Operation<C::Type>,
    C::Type: DifferentiableType,
    C::Value: Mul<Output = C::Value> + Add<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() * right.primal().clone();
        // Product rule: each surviving term is a primal-operand-times-tangent product, shape-congruent with the primal
        // output so the staged `Mul` needs no broadcasting. Zero terms are dropped so the program stays as small
        // as the capture-based pushforward.
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'mul' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let left_term = left
            .tangent()
            .as_value()
            .map(|tangent| {
                Ok::<_, DifferentiationError>(
                    right.primal().normalize_elementwise_tangent(&target)?
                        * tangent.normalize_elementwise_tangent(&target)?,
                )
            })
            .transpose()?;
        let right_term = right
            .tangent()
            .as_value()
            .map(|tangent| {
                Ok::<_, DifferentiationError>(
                    left.primal().normalize_elementwise_tangent(&target)?
                        * tangent.normalize_elementwise_tangent(&target)?,
                )
            })
            .transpose()?;
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for the bilinear `Mul`. A bilinear product is linear in each operand separately but
/// not in both jointly, so in a valid pushforward exactly one operand is linear and the other is a known runtime
/// value. The transpose of `x -> x * k` is `x_bar -> k * x_bar`, so the linear operand's cotangent is the known
/// operand's value times the output cotangent. The known operand receives no cotangent contribution. Scaling by a
/// known factor is self-adjoint, and here that factor is read from the known operand's pullback value atom rather
/// than from a closed-over constant.
///
/// The rule is generic over the type descriptor `T`, so it transposes both scalar [`Mul`]
/// ([`ScalarOperation`](crate::backends::scalars::ScalarOperation)) and array [`Mul`]
/// ([`ArrayOperation`](crate::tracing_v2::ArrayOperation)) tangent programs reached through their enums' partition-
/// aware transpose dispatch.
impl<V: Value, O: Operation<V::Type> + From<MulOperation>> TransposableOperation<V, O> for MulOperation
where
    MulOperation: Operation<V::Type>,
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDifferentiableValue<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        // `Mul` always has exactly two inputs (enforced by its type inference), so the reverse walk supplies a
        // length-two operand slice; index it directly.
        match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
            // Both operands linear corresponds to a bilinear product, which is not a linear map in both operands
            // jointly and therefore never appears in a valid pushforward.
            (true, true) => Err(ProgramError::UnsupportedOperation {
                message: "bilinear `Mul` with two linear operands cannot be transposed".to_string(),
            }
            .into()),
            // Exactly one operand is linear: its cotangent is the known operand's value times the output cotangent,
            // and the known operand contributes a structural zero. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let target = inputs[linear_index].r#type().cotangent();
                if target.is_zero_space() {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "'mul' linear input has no cotangent space".to_string(),
                    }
                    .into());
                }
                let contribution = match &outputs[0] {
                    MaybeZero::Zero(_) => MaybeZero::Zero(target),
                    MaybeZero::Value(output_cotangent) => {
                        // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                        let known_value = inputs[known_index]
                            .as_known()
                            .expect("dispatch guarantees a known operand carries its pullback value");
                        let output_type = output_cotangent.r#type();
                        let contribution = known_value
                            .normalize_elementwise_tangent(output_type.as_ref())?
                            .binary(output_cotangent, MulOperation);
                        MaybeZero::Value(contribution.unbroadcast_elementwise_cotangent(&target)?)
                    }
                };
                let mut contributions = inputs
                    .iter()
                    .map(|input| {
                        let input_type = input.r#type();
                        MaybeZero::Zero(input_type.cotangent())
                    })
                    .collect::<Vec<_>>();
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::operations::math::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
    use crate::types::{ArrayType, DataType};

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain
            .jvp(
                |(left, right)| Ok(left * right),
                (Scalar::from(2.0), Scalar::from(5.0)),
                (Scalar::from(3.0), Scalar::from(-1.0)),
            )
            .unwrap();

        assert_abs_diff_eq!(primal, 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 13.0, epsilon = 1e-9);
    }

    #[test]
    fn test_mul_partitioned_transpose_scales_cotangent_by_known_operand() {
        let scalar_type = ArrayType::scalar(DataType::F64);

        // Build the bilinear product `(residual, tangent) -> residual * tangent`, where the residual is a known
        // runtime value and the tangent is the linear input. This is the shape of the reverse program a
        // partition-aware transpose receives, where `partition` keeps the residual as a known program input.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let residual = builder.add_input(scalar_type.clone());
        let tangent = builder.add_input(scalar_type.clone());
        let product = builder.add_instruction(MulOperation, Vec::new(), vec![residual, tangent]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![product], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        // Transpose treating only the tangent as linear. The pullback inputs are the output cotangent followed by the
        // known residual value, and the single pullback output is the tangent's cotangent (no cotangent for the known
        // residual). Interpreting on `(output_cotangent = 1, residual = r)` yields `r * 1 = r`.
        let r = 4.0;
        let pullback = program.transpose_with_respect_to(&[1]).unwrap();
        assert_eq!(pullback.output_ids().len(), 1, "the known residual input must receive no cotangent output");
        let cotangents = pullback.interpret(vec![TestArray::scalar(1.0), TestArray::scalar(r)]).unwrap();
        assert_eq!(cotangents.len(), 1);
        approx_eq(cotangents[0].values()[0], r);
    }
}
