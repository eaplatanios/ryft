use std::fmt::Display;
use std::ops::BitXor;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Type};

/// Canonical operation name for [`XorOperation`].
pub const XOR_OPERATION_NAME: &'static str = "xor";

/// [`Operation`] that computes the elementwise exclusive disjunction (i.e., `left ^ right`) of two values. This
/// operation covers both logical (i.e., Boolean) and bitwise exclusive disjunction: the two semantics coincide on
/// Boolean element types, and StableHLO's [`xor`](https://openxla.org/stablehlo/spec#xor) operation likewise serves
/// both. Value types provide the elementwise behavior through the standard [`BitXor`] operator trait.
#[derive(Clone, Debug, Default)]
pub struct XorOperation;

impl Display for XorOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO(eaplatanios): Should this not be just `self.render`? Why the ambiguity?
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl ElementwiseOperation for XorOperation {
    #[inline]
    fn name(&self) -> &'static str {
        XOR_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl<V: Value<ArrayType> + BitXor<Output = V>> InterpretableOperation<ArrayType, V> for XorOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone() ^ inputs[1].clone()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`XorOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`XorOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsXor<T: Type> {
    /// Constructs an instance of [`XorOperation`] for this [`Operation`] type.
    fn xor_operation() -> Self;
}

impl<C: StagingContext<Operation: SupportsXor<C::Type>>> BitXor for Tracer<C> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.binary(&rhs, C::Operation::xor_operation())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_xor() {
        let operation = XorOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), XOR_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "XorOperation");
        assert_eq!(format!("{operation}"), XOR_OPERATION_NAME);
        let lhs = TestArray::vector(vec![1.0, 1.0, 0.0, 0.0]);
        let rhs = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        let outputs = operation.interpret(&[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].values(), &[0.0, 1.0, 1.0, 0.0]);

        // The `^` operator implementation matches the interpretation, including scalar broadcasting.
        let lhs = TestArray::vector(vec![1.0, 1.0, 0.0, 0.0]);
        let rhs = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!((lhs ^ rhs).values(), &[0.0, 1.0, 1.0, 0.0]);
        assert_eq!((TestArray::vector(vec![1.0, 0.0]) ^ TestArray::scalar(1.0)).values(), &[0.0, 1.0]);

        // Array type inference broadcasts the Boolean input types.
        let input_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[ArrayType::scalar(DataType::Boolean), input_type.clone()],
            ),
            Ok(vec![input_type.clone()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input_type)),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, XorOperation>::new();
        let left = builder.add_input(input_type.clone());
        let right = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![program_output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[4], %1:bool[4] .
                let %2:bool[4] = xor %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
