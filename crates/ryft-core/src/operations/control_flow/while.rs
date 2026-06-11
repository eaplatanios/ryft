use std::fmt::{Debug, Display};

use crate::macros::{check_count, check_types};
use crate::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::types::{ArrayType, Type, TypeError};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`WhileOperation`].
pub const WHILE_OPERATION_NAME: &'static str = "while";

/// [`Operation`] that repeatedly applies a nested body [`Program`] to a loop-carried state while a nested condition
/// [`Program`] over that same state produces a true scalar Boolean predicate. The condition and body consume identical
/// state type signatures, the body produces the next state with that same signature, and the operation outputs the
/// final state once the condition produces false.
///
/// The nested condition and body are stored as flat `Vec`-parameter [`Program`]s because they consume the
/// loop-carried state directly. Structured Rust parameters are flattened before a region is captured and
/// reconstructed by the surrounding API when needed; the operation itself only needs the ordered leaf signature for
/// type checking, interpretation, JVP, batching, and transposition.
#[derive(Clone, Debug)]
pub struct WhileOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Condition [`Program`] of this [`WhileOperation`] that maps the current loop state to one scalar Boolean
    /// predicate.
    pub(crate) condition: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Body [`Program`] of this [`WhileOperation`] that maps the current loop state to the next loop state.
    pub(crate) body: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> WhileOperation<V, O, ArrayType> {
    /// Creates a new [`WhileOperation`] with the provided condition and body programs.
    ///
    /// # Parameters
    ///
    ///   - `condition`: Condition [`Program`] that maps the loop-carried state to one scalar Boolean predicate.
    ///   - `body`: Body [`Program`] that maps the loop-carried state to the next loop state. This program must
    ///     consume and produce the same state type signature that `condition` consumes.
    pub fn new(
        condition: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
        body: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let state_types = condition.input_types();
        check_types!("while condition/body input", &state_types, &body.input_types());
        let condition_output_types = condition.output_types();
        if condition_output_types.len() != 1 {
            return Err(TypeError {
                message: format!(
                    "while condition must return exactly one predicate leaf but returned {}",
                    condition_output_types.len()
                ),
            });
        }
        if !condition_output_types[0].is_scalar() || condition_output_types[0] != condition_output_types[0].as_boolean()
        {
            return Err(TypeError {
                message: format!(
                    "while condition output type must be a scalar boolean, but got {}",
                    condition_output_types[0],
                ),
            });
        }
        check_types!("while body output", &state_types, &body.output_types());
        Ok(Self { condition, body })
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> WhileOperation<V, O, T> {
    /// Returns the condition [`Program`] of this [`WhileOperation`] that is evaluated before each loop iteration.
    #[inline]
    pub fn condition(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.condition
    }

    /// Returns the body [`Program`] of this [`WhileOperation`] that computes the next loop-carried state.
    #[inline]
    pub fn body(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.body
    }

    /// Returns the loop-carried state types of this [`WhileOperation`].
    #[inline]
    pub fn state_types(&self) -> Vec<T> {
        self.body.input_types()
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for WhileOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> Operation<T> for WhileOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        WHILE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        let state_types = self.state_types();
        check_count!("input", input_types, state_types.len(), TypeError);
        check_types!("while input", &state_types, input_types);
        Ok(state_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, WHILE_OPERATION_NAME)?.bracketed(|operation| {
            operation.program("condition", &self.condition)?;
            operation.program("body", &self.body)
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + BooleanLike,
    O: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let mut state = inputs.to_vec();
        loop {
            let condition_outputs = self.condition.interpret(state.clone())?;
            check_count!("output", condition_outputs, 1, ProgramError);
            if !condition_outputs[0].boolean()? {
                return Ok(state);
            }
            state = self.body.interpret(state)?;
            check_count!("output", state, self.state_types().len(), ProgramError);
        }
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

    /// Test [`Operation`] type used for the nested condition and body programs.
    type TestOperation = ArrayOperation<TestArray, ArrayType>;

    /// Builds a condition program that maps a scalar `f64` state to the scalar Boolean predicate `state > 0`.
    fn greater_than_zero_condition() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(TestOperation::ZeroLike, vec![state]).unwrap()[0];
        let predicate = builder
            .add_instruction(TestOperation::Compare { direction: ComparisonDirection::GreaterThan }, vec![state, zero])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a body program that maps a scalar `f64` state to `state - 1`.
    fn subtract_one_body() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_instruction(TestOperation::OneLike, vec![state]).unwrap()[0];
        let next_state = builder.add_instruction(TestOperation::Sub, vec![state, one]).unwrap()[0];
        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_while() {
        let state_type = ArrayType::scalar(DataType::F64);
        let operation = WhileOperation::new(greater_than_zero_condition(), subtract_one_body()).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), WHILE_OPERATION_NAME);
        assert_eq!(operation.condition().output_types(), vec![ArrayType::scalar(DataType::Boolean)]);
        assert_eq!(operation.body().output_types(), vec![state_type.clone()]);
        assert_eq!(operation.state_types(), vec![state_type.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                while [
                    condition={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                            %2:bool[] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = one_like %0
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
            "}
            .trim_end(),
        );

        // Type inference validates the state types and returns them as the output types.
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&state_type)), Ok(vec![state_type.clone()]));
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError {
                message: "while input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            }),
        );

        // Construction rejects mismatched condition/body state signatures, non-scalar-Boolean condition outputs,
        // multi-output conditions, and body outputs that do not match the state signature.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])));
        let zero = builder.add_instruction(TestOperation::ZeroLike, vec![state]).unwrap()[0];
        let vector_body = builder.build(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            WhileOperation::new(greater_than_zero_condition(), vector_body).map(|_| ()),
            Err(TypeError {
                message: "while condition/body input type signature mismatch: expected [f64[]] but got [f64[2]]"
                    .to_string(),
            }),
        );
        assert_eq!(
            WhileOperation::new(subtract_one_body(), subtract_one_body()).map(|_| ()),
            Err(TypeError {
                message: "while condition output type must be a scalar boolean, but got f64[]".to_string(),
            }),
        );
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let multi_output_condition =
            builder.build(vec![state, state], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();
        assert_eq!(
            WhileOperation::new(multi_output_condition, subtract_one_body()).map(|_| ()),
            Err(TypeError {
                message: "while condition must return exactly one predicate leaf but returned 2".to_string(),
            }),
        );
        assert_eq!(
            WhileOperation::new(greater_than_zero_condition(), greater_than_zero_condition()).map(|_| ()),
            Err(TypeError {
                message: "while body output type signature mismatch: expected [f64[]] but got [bool[]]".to_string(),
            }),
        );

        // Interpretation iterates the body until the condition produces false.
        let outputs = operation.interpret(&[TestArray::scalar(3.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        let outputs = operation.interpret(&[TestArray::scalar(-1.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![-1.0]);
        assert_eq!(
            operation.interpret(&[]),
            Err(ProgramError::Type(TypeError { message: "expected 1 input but got 0".to_string() })),
        );

        // Program rendering uses the canonical operation name and includes the nested condition and body programs.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let program_state = builder.add_input(state_type);
        let program_output =
            builder.add_instruction(TestOperation::While(Box::new(operation)), vec![program_state]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![program_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [
                    condition={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                            %2:bool[] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = one_like %0
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
