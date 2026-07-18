pub mod abs;
pub mod add;
pub mod atan2;
pub mod cos;
pub mod div;
pub mod dot;
pub mod exp;
pub mod log;
pub mod mul;
pub mod neg;
pub mod reduce;
pub mod sin;
pub mod sqrt;
pub mod sub;

pub use abs::{ABS_OPERATION_NAME, Abs, AbsOperation};
pub use add::{ADD_OPERATION_NAME, Add, AddOperation};
pub use atan2::{ATAN2_OPERATION_NAME, Atan2, Atan2Operation};
pub use cos::{COS_OPERATION_NAME, Cos, CosOperation};
pub use div::{DIV_OPERATION_NAME, Div, DivOperation};
pub use dot::{DOT_OPERATION_NAME, Dot, DotDimensionNumbers, DotOperation, DotOps};
pub use exp::{EXP_OPERATION_NAME, Exp, ExpOperation};
pub use log::{LOG_OPERATION_NAME, Log, LogOperation};
pub use mul::{MUL_OPERATION_NAME, Mul, MulOperation};
pub use neg::{NEG_OPERATION_NAME, Neg, NegOperation};
pub use reduce::{Reduce, ReduceOperation, ReductionKind};
pub use sin::{SIN_OPERATION_NAME, Sin, SinOperation};
pub use sqrt::{SQRT_OPERATION_NAME, Sqrt, SqrtOperation};
pub use sub::{SUB_OPERATION_NAME, Sub, SubOperation};

#[cfg(test)]
pub(crate) mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationError;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Shape, Size};

    /// Checks the shared elementwise batching rule for a unary operation over a mapped vector of scalar batch items.
    pub(crate) fn assert_unary_batching<O>(operation: O, input: &[f64], expected: &[f64])
    where
        O: BatchableOperation<EagerContext<Array, ArrayOperation<Array>>>,
    {
        let physical_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(input.len())]));
        let input =
            ArrayBatch::new(physical_type.clone(), Array::from_f64s(physical_type, input.to_vec()), BatchAxis::new(0))
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), expected.len());
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        for (actual, expected) in outputs[0].value().to_f64s().iter().zip(expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }
    }

    /// Checks the shared elementwise batching rule for a binary operation with one mapped and one replicated operand.
    pub(crate) fn assert_binary_batching<O>(operation: O, left: &[f64], right: f64, expected: &[f64])
    where
        O: BatchableOperation<EagerContext<Array, ArrayOperation<Array>>>,
    {
        let physical_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(left.len())]));
        let left =
            ArrayBatch::new(physical_type.clone(), Array::from_f64s(physical_type, left.to_vec()), BatchAxis::new(0))
                .unwrap();
        let right =
            ArrayBatch::new(ArrayType::scalar(DataType::F64), Array::scalar(right), BatchAxis::replicated()).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), expected.len());
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[left, right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        for (actual, expected) in outputs[0].value().to_f64s().iter().zip(expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }
    }

    /// Checks the shared partial-evaluation behavior of an elementwise operation over `f64` scalars: all-known
    /// inputs constant-fold the operation away, while an unknown first input residualizes the operation into the
    /// residual program, which replays to the same result.
    pub(crate) fn assert_partial_evaluation<O: Clone + Into<ScalarOperation<Scalar>>>(
        operation: O,
        inputs: &[f64],
        expected: f64,
    ) {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input_ids = inputs.iter().map(|_| builder.add_input(DataType::F64)).collect::<Vec<_>>();
        let output = builder.add_instruction(operation, Vec::new(), input_ids).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder; inputs.len()], vec![Placeholder])
            .unwrap();

        // All-known inputs fold the operation away into a known output.
        let known = inputs.iter().map(|input| PartialValue::Known(Scalar::from(*input))).collect::<Vec<_>>();
        let evaluation = program.partially_evaluate(&known).unwrap();
        assert!(evaluation.program.instructions().is_empty());
        assert_eq!(evaluation.outputs.len(), 1);
        let PartialEvaluationOutput::Known(Scalar::F64(folded)) = &evaluation.outputs[0] else {
            panic!("expected a known f64 folded output");
        };
        assert_abs_diff_eq!(*folded, expected, epsilon = 1e-9);

        // An unknown first input residualizes the operation, and the residual program replays to the same result.
        let mut partial_inputs = known;
        partial_inputs[0] = PartialValue::Unknown(DataType::F64);
        let evaluation = program.partially_evaluate(&partial_inputs).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        let outputs = evaluation
            .interpret(&EagerContext::<Scalar, ScalarOperation<Scalar>>::new(), &[Scalar::from(inputs[0])])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let Scalar::F64(residual) = &outputs[0] else { panic!("expected an f64 residual output") };
        assert_abs_diff_eq!(*residual, expected, epsilon = 1e-9);
    }

    /// Checks that a nonlinear elementwise operation rejects an input that still carries a partial sum.
    pub(crate) fn assert_rejects_unreduced<O: Operation<ArrayType>>(
        operation: O,
        operation_name: &str,
        input_count: usize,
    ) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::scalar(DataType::F64)
            .with_sharding(
                Sharding::new(mesh, Vec::<ShardingDimension>::new()).unwrap().with_unreduced_axes(["x"]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&vec![input; input_count], &[]),
            Err(TypeError { message: format!("'{operation_name}' does not support unreduced operands") }),
        );
    }

    /// Checks that an ordinary binary elementwise operation rejects operands with mismatched reduced-axis markers.
    pub(crate) fn assert_rejects_mismatched_reduced<O: Operation<ArrayType>>(operation: O, operation_name: &str) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::new(mesh.clone(), Vec::<ShardingDimension>::new()).unwrap())
            .unwrap();
        let reduced = ArrayType::scalar(DataType::F64)
            .with_sharding(
                Sharding::new(mesh, Vec::<ShardingDimension>::new()).unwrap().with_reduced_axes(["x"]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[plain, reduced], &[]),
            Err(TypeError { message: format!("'{operation_name}' operands must be reduced over the same axes") }),
        );
    }

    /// Checks that a nonlinear elementwise operation cannot be transposed before it has been linearized by its JVP
    /// rule.
    pub(crate) fn assert_rejects_nonlinear_transposition<O: Into<ArrayOperation<Array>>>(
        operation: O,
        operation_name: &str,
        input_count: usize,
    ) {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = (0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))).collect::<Vec<_>>();
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![output], vec![Placeholder; input_count], Placeholder)
            .unwrap();
        let input_indices = (0..input_count).collect::<Vec<_>>();
        assert!(matches!(
            program.transpose_with_respect_to(input_indices.as_slice()),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("operation `{operation_name}` is not transposable"),
        ));
    }
}
