use std::fmt::{Debug, Display};

use crate::macros::{check_count, check_types};
use crate::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::types::{ArrayType, Type, TypeError};

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &'static str = "condition";

/// Predicate source for a [`ConditionOperation`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConditionPredicate<T: PartialEq + Type> {
    /// The predicate is **dynamic** and is the first input to the operation.
    Dynamic(T),

    /// The predicate is **static** and captured in the operation (i.e., it is not an input to the operation).
    Static(bool),
}

// TODO(eaplatanios): Review from here onwards.

// TODO(eaplatanios): Re-order generic parameters to `T, V, O` here and elsewhere.
/// [`Operation`] that evaluates one of two nested branch [`Program`]s depending on a Boolean predicate. The predicate
/// is either supplied as the first operation input (i.e., [`ConditionPredicate::Dynamic`]) or captured in the operation
/// itself (i.e., [`ConditionPredicate::Static`]). The remaining operation inputs are forwarded to the selected branch,
/// and so both branches must consume the same input types and produce the same output types. The nested branches are
/// stored as flat `Vec`-parameter [`Program`]s because they consume the operation operands directly. Structured Rust
/// parameters are flattened before a branch is captured (i.e., via [`Parameterized`] helpers) and reconstructed later
/// as needed. The operation itself only needs the ordered parameter signature for type checking, interpretation,
/// batching, differentiation, transposition, and other transforms.
#[derive(Clone, Debug)]
pub struct ConditionOperation<V: Value<T>, O, T: PartialEq + Type> {
    /// Predicate source of this [`ConditionOperation`].
    pub(crate) predicate: ConditionPredicate<T>,

    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is true.
    pub(crate) true_branch: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is false.
    pub(crate) false_branch: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> ConditionOperation<V, O, ArrayType> {
    /// Creates a new [`ConditionOperation`] whose predicate is supplied as the first operation input.
    ///
    /// # Parameters
    ///
    ///   - `predicate_type`: [`ArrayType`] of the runtime predicate input. This must be a scalar Boolean type.
    ///   - `true_branch`: Branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Branch [`Program`] evaluated when the predicate is false. This program must have the same
    ///     input and output type signatures as `true_branch`.
    pub fn new(
        predicate_type: ArrayType,
        true_branch: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
        false_branch: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        if !predicate_type.is_scalar() || predicate_type != predicate_type.as_boolean() {
            return Err(TypeError {
                message: format!("condition predicate type must be a scalar boolean, but got {predicate_type}"),
            });
        }
        Self::from_parts(ConditionPredicate::Dynamic(predicate_type), true_branch, false_branch)
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> ConditionOperation<V, O, T> {
    /// Creates a new [`ConditionOperation`] whose predicate is captured in the operation instead of being supplied as
    /// an operation input.
    ///
    /// # Parameters
    ///
    ///   - `predicate`: Captured predicate value that selects the branch to evaluate.
    ///   - `true_branch`: Branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Branch [`Program`] evaluated when the predicate is false. This program must have the same
    ///     input and output type signatures as `true_branch`.
    pub fn with_captured_predicate(
        predicate: bool,
        true_branch: Program<T, V, O, Vec<V>, Vec<V>>,
        false_branch: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        Self::from_parts(ConditionPredicate::Static(predicate), true_branch, false_branch)
    }

    /// Creates a new [`ConditionOperation`] after validating that the two branches have identical input and output
    /// type signatures.
    fn from_parts(
        predicate: ConditionPredicate<T>,
        true_branch: Program<T, V, O, Vec<V>, Vec<V>>,
        false_branch: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = true_branch.input_types();
        check_types!("condition branch input", &input_types, &false_branch.input_types());
        let output_types = true_branch.output_types();
        check_types!("condition branch output", &output_types, &false_branch.output_types());
        Ok(Self { predicate, true_branch, false_branch })
    }

    /// Returns the predicate source of this [`ConditionOperation`].
    #[inline]
    pub fn predicate(&self) -> &ConditionPredicate<T> {
        &self.predicate
    }

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

    /// Returns the operand input types consumed by both branches of this [`ConditionOperation`]. These do not include
    /// the runtime predicate input, if one is present.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        self.true_branch.input_types()
    }

    /// Returns the output types produced by both branches of this [`ConditionOperation`].
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        self.true_branch.output_types()
    }

    /// Returns the branch of this [`ConditionOperation`] that is selected by `predicate`.
    pub(crate) fn selected_branch(&self, predicate: bool) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        if predicate { &self.true_branch } else { &self.false_branch }
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for ConditionOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: PartialEq + Type + BooleanLike, V: Value<T>, O: Operation<T>> Operation<T> for ConditionOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        let operand_input_types = self.input_types();
        let operand_start = match &self.predicate {
            ConditionPredicate::Dynamic(predicate_type) => {
                check_count!("input", input_types, operand_input_types.len() + 1, TypeError);
                if !input_types[0].is_scalar() || input_types[0] != input_types[0].as_boolean() {
                    return Err(TypeError {
                        message: format!(
                            "condition predicate type must be a scalar boolean, but got {}",
                            input_types[0],
                        ),
                    });
                }
                if &input_types[0] != predicate_type {
                    return Err(TypeError {
                        message: format!(
                            "condition predicate type mismatch: expected {predicate_type}, got {}",
                            input_types[0]
                        ),
                    });
                }
                1
            }
            ConditionPredicate::Static(_) => {
                check_count!("input", input_types, operand_input_types.len(), TypeError);
                0
            }
        };
        check_types!("condition operand", &operand_input_types, &input_types[operand_start..]);
        Ok(self.output_types())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONDITION_OPERATION_NAME)?.bracketed(|operation| {
            match &self.predicate {
                ConditionPredicate::Dynamic(predicate_type) => {
                    operation.field("predicate", format_args!("runtime_input(type={predicate_type})"))?;
                }
                ConditionPredicate::Static(predicate) => {
                    operation.field("predicate", format_args!("captured({predicate})"))?;
                }
            }
            operation.program("true_branch", &self.true_branch)?;
            operation.program("false_branch", &self.false_branch)
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for ConditionOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + BooleanLike,
    O: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let (predicate, operands) = match self.predicate {
            ConditionPredicate::Dynamic(_) => (inputs[0].boolean()?, &inputs[1..]),
            ConditionPredicate::Static(predicate) => (predicate, inputs),
        };
        self.selected_branch(predicate).interpret(operands.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::compare::ComparisonDirection;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Test [`Operation`] type used for the nested branch programs.
    type TestOperation = ArrayOperation<TestArray, ArrayType>;

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: TestOperation,
    ) -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, TestOperation::Add) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, inputs).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::new(
            predicate_type.clone(),
            scalar_branch(TestOperation::Add),
            scalar_branch(TestOperation::ZeroLike),
        )
        .unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(*operation.predicate(), ConditionPredicate::Dynamic(predicate_type.clone()));
        assert_eq!(operation.true_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.false_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.input_types(), vec![operand_type.clone()]);
        assert_eq!(operation.output_types(), vec![operand_type.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                condition [
                    predicate=runtime_input(type=bool[]),
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

        // Construction rejects non-Boolean predicate types and mismatched branch signatures.
        assert_eq!(
            ConditionOperation::<TestArray, TestOperation, ArrayType>::new(
                operand_type.clone(),
                scalar_branch(TestOperation::Add),
                scalar_branch(TestOperation::ZeroLike),
            )
            .map(|_| ()),
            Err(TypeError { message: "condition predicate type must be a scalar boolean, but got f64[]".to_string() }),
        );
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(TestOperation::ZeroLike, vec![input]).unwrap()[0];
        let boolean_output = builder
            .add_instruction(TestOperation::Compare { direction: ComparisonDirection::GreaterThan }, vec![input, zero])
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            ConditionOperation::with_captured_predicate(true, scalar_branch(TestOperation::Add), boolean_branch)
                .map(|_| ()),
            Err(TypeError {
                message: "condition branch output type signature mismatch: expected [f64[]] but got [bool[]]"
                    .to_string(),
            }),
        );

        // Interpretation with a runtime predicate selects between the two branches.
        let predicate = |value: f64| TestArray::new(predicate_type.clone(), vec![value]);
        let outputs = operation.interpret(&[predicate(1.0), TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![8.0]);
        let outputs = operation.interpret(&[predicate(0.0), TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        assert_eq!(
            operation.interpret(&[]),
            Err(ProgramError::Type(TypeError { message: "expected 2 inputs but got 0".to_string() })),
        );

        // Interpretation with a captured predicate consumes only the branch operands.
        let captured = ConditionOperation::with_captured_predicate(
            true,
            scalar_branch(TestOperation::Add),
            scalar_branch(TestOperation::ZeroLike),
        )
        .unwrap();
        assert_eq!(*captured.predicate(), ConditionPredicate::Static(true));
        assert_eq!(captured.infer_output_types(&[operand_type.clone()]), Ok(vec![operand_type.clone()]));
        let outputs = captured.interpret(&[TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![8.0]);
        let captured = ConditionOperation::with_captured_predicate(
            false,
            scalar_branch(TestOperation::Add),
            scalar_branch(TestOperation::ZeroLike),
        )
        .unwrap();
        let outputs = captured.interpret(&[TestArray::scalar(4.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);

        // Program rendering uses the canonical operation name and includes the nested branch programs.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let program_predicate = builder.add_input(predicate_type);
        let program_operand = builder.add_input(operand_type);
        let program_output = builder
            .add_instruction(TestOperation::Condition(Box::new(operation)), vec![program_predicate, program_operand])
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
                    predicate=runtime_input(type=bool[]),
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
