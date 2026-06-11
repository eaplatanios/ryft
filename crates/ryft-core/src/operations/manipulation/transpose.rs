use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Shape, Type, TypeError};

/// Canonical operation name for [`TransposeOperation`].
pub const TRANSPOSE_OPERATION_NAME: &'static str = "transpose";

/// [`Operation`] that reorders the axes of its input array according to a static permutation. The output shape is the
/// input shape with its axes permuted. Output dimension `i` is set to input dimension `permutation[i]`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransposeOperation {
    /// Axis permutation of this [`TransposeOperation`].
    permutation: Vec<usize>,
}

impl TransposeOperation {
    /// Creates a new [`TransposeOperation`] with the provided axis permutation.
    #[inline]
    pub fn new(permutation: Vec<usize>) -> Self {
        Self { permutation }
    }

    /// Returns the axis permutation of this [`TransposeOperation`].
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        self.permutation.as_slice()
    }
}

impl Display for TransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for TransposeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        TRANSPOSE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        let rank = input.rank();
        if self.permutation.len() != rank {
            return Err(TypeError {
                message: format!(
                    "transpose permutation has length {} but input has rank {rank}",
                    self.permutation.len(),
                ),
            });
        }
        let mut seen = vec![false; rank];
        for axis in &self.permutation {
            if *axis >= rank {
                return Err(TypeError { message: format!("transpose permutation axis {axis} is out of bounds") });
            }
            if seen[*axis] {
                return Err(TypeError { message: format!("transpose permutation contains duplicate axis {axis}") });
            }
            seen[*axis] = true;
        }
        let permuted = self.permutation.iter().map(|axis| input.dimension(*axis as isize)).collect::<Vec<_>>();
        Ok(vec![ArrayType::new(input.data_type(), Shape::new(permuted))])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("permutation", format_args!("{:?}", self.permutation)))
    }
}

impl<V: Value<ArrayType> + Transpose> InterpretableOperation<ArrayType, V> for TransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().transpose(self.permutation.clone())])
    }
}

/// Trait that represents [`Operation`] types that support/include [`TransposeOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`TransposeOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsTranspose<T: Type> {
    /// Constructs an instance of [`TransposeOperation`] for this [`Operation`] type with the provided axis
    /// permutation.
    fn transpose_operation(permutation: Vec<usize>) -> Self;
}

/// Represents the ability to transpose the axes of an array. [`Transpose`] fills the same role for
/// [`TransposeOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
/// [`Operation`]s.
pub trait Transpose: Sized {
    /// Reorders the axes of `self` according to the provided permutation.
    fn transpose(self, permutation: Vec<usize>) -> Self;
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsTranspose<ArrayType>>> Transpose for Tracer<C> {
    #[inline]
    fn transpose(self, permutation: Vec<usize>) -> Self {
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return self;
        }
        self.unary(C::Operation::transpose_operation(permutation))
    }
}

impl<V: Value<ArrayType> + Transpose> Transpose for Tangent<ArrayType, V> {
    #[inline]
    fn transpose(self, permutation: Vec<usize>) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(ArrayType::new(
                r#type.data_type(),
                Shape::new(permutation.iter().map(|axis| r#type.dimension(*axis as isize)).collect::<Vec<_>>()),
            )),
            Self::Value(value) => Self::Value(value.transpose(permutation)),
        }
    }
}

/// Returns the inverse permutation of `permutation` (i.e., the permutation that undoes it).
pub fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; permutation.len()];
    for (position, axis) in permutation.iter().enumerate() {
        inverse[*axis] = position;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{DataType, Size};

    use super::*;

    #[test]
    fn test_transpose() {
        let operation = TransposeOperation::new(vec![1, 0]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), TRANSPOSE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "TransposeOperation { permutation: [1, 0] }");
        assert_eq!(format!("{operation}"), "transpose [permutation=[1, 0]]");
        assert_eq!(operation.permutation(), &[1, 0]);

        // Type inference permutes the input shape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![output_type.clone()]));

        // Interpretation reorders the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation.interpret(std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].array_type(), output_type);
        assert_eq!(output[0].values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError { message: "transpose permutation has length 2 but input has rank 1".to_string() }),
        );
        assert_eq!(
            TransposeOperation::new(vec![0, 2]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "transpose permutation axis 2 is out of bounds".to_string() }),
        );
        assert_eq!(
            TransposeOperation::new(vec![0, 0]).infer_output_types(std::slice::from_ref(&input_type)),
            Err(TypeError { message: "transpose permutation contains duplicate axis 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured permutation.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TransposeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[3, 2] = transpose [permutation=[1, 0]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transpose_test_array() {
        // Rank-2 swap of a row-major 2x3 payload.
        let output = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose(vec![1, 0]);
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)])));
        assert_eq!(output.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Rank-3 permutation moving the last axis to the front.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type, values).transpose(vec![2, 0, 1]);
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)])),
        );
        assert_eq!(
            output.values,
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0,
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
            ],
        );

        // Rank-0 and empty payloads pass through unchanged.
        let output = TestArray::scalar(42.0).transpose(vec![]);
        assert_eq!(output.r#type, ArrayType::scalar(DataType::F64));
        assert_eq!(output.values, vec![42.0]);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0), Size::Static(2)]));
        let output = TestArray::new(input_type, Vec::new()).transpose(vec![1, 0]);
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(0)])));
        assert_eq!(output.values, Vec::<f64>::new());
    }

    #[test]
    fn test_inverse_permutation() {
        // Empty and identity permutations are their own inverses.
        assert_eq!(inverse_permutation(&[]), Vec::<usize>::new());
        assert_eq!(inverse_permutation(&[0, 1, 2]), vec![0, 1, 2]);

        // A swap is its own inverse, while a cycle inverts to the reverse cycle.
        assert_eq!(inverse_permutation(&[1, 0]), vec![1, 0]);
        assert_eq!(inverse_permutation(&[2, 0, 1]), vec![1, 2, 0]);

        // Inverting twice recovers the original permutation, and applying the inverse after the
        // permutation restores the identity ordering.
        let permutation = vec![3, 0, 2, 1];
        let inverse = inverse_permutation(permutation.as_slice());
        assert_eq!(inverse_permutation(inverse.as_slice()), permutation);
        let composed = inverse.iter().map(|axis| permutation[*axis]).collect::<Vec<_>>();
        assert_eq!(composed, vec![0, 1, 2, 3]);
    }
}
