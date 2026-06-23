use std::fmt::{Debug, Display};

use crate::macros::{check_count, check_types};
use crate::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::types::{ArrayType, Type, TypeError};

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &'static str = "condition";

// TODO(eaplatanios): Review from here onwards.

/// [`Operation`] that evaluates one of two nested branch [`Program`]s depending on a Boolean predicate that is always
/// supplied as the first operation input (a scalar Boolean operand). The remaining operation inputs are forwarded to
/// the selected branch, and so both branches must consume the same input types and produce the same output types.
///
/// A predicate that is already known while *building* a program is naturally expressed with a plain Rust `if` that
/// chooses which operations to stage, so no `condition` operation is needed for it. A predicate that is staged as a
/// constant still lowers to a `stablehlo.if` operation whose constant predicate the backend folds away (via
/// [StableHLO canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes) and XLA's conditional
/// simplification), so `ryft` performs no predicate folding of its own.
///
/// The nested branches are stored as flat `Vec`-parameter [`Program`]s because they consume the operation operands
/// directly. Structured Rust parameters are flattened before a branch is captured (i.e., via [`Parameterized`]
/// helpers) and reconstructed later as needed. The operation itself only needs the ordered parameter signature for
/// type checking, interpretation, batching, differentiation, transposition, and other transforms.
#[derive(Clone, Debug)]
pub struct ConditionOperation<T: Type, V: Value<T>, O> {
    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is true.
    pub(crate) true_branch: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is false.
    pub(crate) false_branch: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> ConditionOperation<ArrayType, V, O> {
    /// Creates a new [`ConditionOperation`] whose predicate is supplied as the first operation input. The predicate
    /// input is not described by the operation itself: it must simply be a scalar Boolean type, which
    /// [`Operation::infer_output_types`] validates structurally against the actual first operand type.
    ///
    /// # Parameters
    ///
    ///   - `true_branch`: Branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Branch [`Program`] evaluated when the predicate is false. This program must have the same
    ///     input and output type signatures as `true_branch`.
    pub fn new(
        true_branch: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
        false_branch: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = true_branch.input_types();
        check_types!("condition branch input", &input_types, &false_branch.input_types());
        let output_types = true_branch.output_types();
        check_types!("condition branch output", &output_types, &false_branch.output_types());
        Ok(Self { true_branch, false_branch })
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> ConditionOperation<T, V, O> {
    /// Returns the branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is true.
    #[inline]
    pub fn true_branch(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.true_branch
    }

    /// Returns the branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is false.
    #[inline]
    pub fn false_branch(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.false_branch
    }

    /// Returns the output types produced by both branches of this [`ConditionOperation`].
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        self.true_branch.output_types()
    }
}

impl<T: Type, V: Value<T>, O> Display for ConditionOperation<T, V, O>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type + BooleanLike, V: Value<T>, O: Operation<T>> Operation<T> for ConditionOperation<T, V, O> {
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        let branch_input_types = self.true_branch.input_types();
        check_count!("input", input_types, branch_input_types.len() + 1, TypeError);
        if !input_types[0].is_scalar() || input_types[0] != input_types[0].as_boolean() {
            return Err(TypeError {
                message: format!("condition predicate type must be a scalar boolean, but got {}", input_types[0]),
            });
        }
        check_types!("condition operand", &branch_input_types, &input_types[1..]);
        Ok(self.output_types())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONDITION_OPERATION_NAME)?.bracketed(|operation| {
            operation.program("true_branch", &self.true_branch)?;
            operation.program("false_branch", &self.false_branch)
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for ConditionOperation<ArrayType, V, O>
where
    V: Value<ArrayType> + BooleanLike,
    O: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let (predicate, operands) = (inputs[0].boolean()?, &inputs[1..]);
        if predicate { &self.true_branch } else { &self.false_branch }.interpret_in_context(context, operands.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::TracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<TestArray>,
    ) -> Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, inputs).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::new(
            scalar_branch(ArrayOperation::Add(AddOperation)),
            scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation)),
        )
        .unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(operation.true_branch().input_types(), vec![operand_type.clone()]);
        assert_eq!(operation.true_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.false_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.output_types(), vec![operand_type.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ]
            "}
            .trim_end(),
        );

        // Type inference validates the predicate and operand types and returns the branch output types.
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), operand_type.clone()]),
            Ok(vec![operand_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[operand_type.clone(), operand_type.clone()]),
            Err(TypeError { message: "condition predicate type must be a scalar boolean, but got f64[]".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])),
                operand_type.clone(),
            ]),
            Err(TypeError {
                message: "condition predicate type must be a scalar boolean, but got bool[2]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                predicate_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "condition operand type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            }),
        );

        // Construction rejects mismatched branch signatures.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation, vec![input]).unwrap()[0];
        let boolean_output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![input, zero])
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            ConditionOperation::new(scalar_branch(ArrayOperation::Add(AddOperation)), boolean_branch).map(|_| ()),
            Err(TypeError {
                message: "condition branch output type signature mismatch: expected [f64[]] but got [bool[]]"
                    .to_string(),
            }),
        );

        // Interpretation extracts the predicate from the first input and selects between the two branches.
        let predicate = |value: f64| TestArray::new(predicate_type.clone(), vec![value]);
        let outputs =
            operation.interpret(&crate::EagerContext::new(), &[predicate(1.0), TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![8.0]);
        let outputs =
            operation.interpret(&crate::EagerContext::new(), &[predicate(0.0), TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        assert_eq!(
            operation.interpret(&crate::EagerContext::new(), &[] as &[TestArray]),
            Err(ProgramError::Type(TypeError { message: "expected 2 inputs but got 0".to_string() })),
        );

        // Staging records the condition payload into the active program instead of trying to concretize the staged
        // predicate.
        let domain = TestArrayDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new()));
        let context = TracingContext::new(&domain, builder.clone());
        let staged_predicate = context.input(predicate_type.clone());
        let staged_operand = context.input(operand_type.clone());
        let outputs = context
            .stage_operation(operation.clone(), &[staged_predicate.clone(), staged_operand.clone()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::Condition(_)));
        assert_eq!(
            builder.instructions()[0].inputs(),
            &[staged_predicate.atom_id().unwrap(), staged_operand.atom_id().unwrap()],
        );
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering uses the canonical operation name and includes the nested branch programs.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let program_predicate = builder.add_input(predicate_type);
        let program_operand = builder.add_input(operand_type);
        let program_output = builder
            .add_instruction(ArrayOperation::Condition(Box::new(operation)), vec![program_predicate, program_operand])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![program_output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
