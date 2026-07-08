use std::ops::{Add, Mul};

use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::MulOperation;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::types::Typed;

impl<C: Context> DifferentiableOperation<C> for MulOperation
where
    C::Operation: Clone,
    C::Value: Mul<Output = C::Value> + Add<Output = C::Value>,
    MulOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() * right.primal().clone();
        // Product rule: each surviving term is a primal-operand-times-tangent product, shape-congruent with the primal
        // output so the staged `Mul` needs no broadcasting. Zero terms are dropped so the program stays as small
        // as the capture-based pushforward.
        let left_term = left.tangent().as_value().map(|tangent| right.primal().clone() * tangent.clone());
        let right_term = right.tangent().as_value().map(|tangent| left.primal().clone() * tangent.clone());
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(primal.r#type().into_owned()), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)])
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
/// ([`ScalarOperation`](crate::operations::scalars::ScalarOperation)) and array [`Mul`]
/// ([`ArrayOperation`](crate::tracing_v2::ArrayOperation)) tangent programs reached through their enums' partition-
/// aware transpose dispatch.
impl<V: Value, O: Operation<V::Type> + From<MulOperation>> TransposableOperation<V, O> for MulOperation
where
    MulOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("output", outputs, 1, ProgramError);
        // `Mul` always has exactly two inputs (enforced by its type inference), so the reverse walk supplies a
        // length-two operand slice; index it directly.
        match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
            // Both operands linear corresponds to a bilinear product, which is not a linear map in both operands
            // jointly and therefore never appears in a valid pushforward.
            (true, true) => Err(ProgramError::UnsupportedOperation {
                message: "bilinear `Mul` with two linear operands cannot be transposed".to_string(),
            }),
            // Exactly one operand is linear: its cotangent is the known operand's value times the output cotangent,
            // and the known operand contributes a structural zero. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let contribution = match &outputs[0] {
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                    MaybeZero::Value(output_cotangent) => {
                        // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                        let known_value = inputs[known_index]
                            .as_known()
                            .expect("dispatch guarantees a known operand carries its pullback value");
                        MaybeZero::Value(known_value.binary(output_cotangent, MulOperation))
                    }
                };
                let mut contributions =
                    inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect::<Vec<_>>();
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::scalars::Scalar;
    use crate::tests::TestArray;
    use crate::tracing_v2::test_util::assert_scalar_close;
    use crate::tracing_v2::{ArrayOperation, Differentiate};
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

        assert_scalar_close(primal, 10.0);
        assert_scalar_close(tangent, 13.0);
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
        let product = builder.add_instruction(MulOperation, vec![residual, tangent]).unwrap()[0];
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
