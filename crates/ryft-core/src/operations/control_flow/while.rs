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
///
/// **Iteration bounds are semantic.** A while loop built with [`Self::with_iteration_bound`] runs **at most** `bound`
/// iterations *by definition*: a loop whose condition would keep it running longer is truncated after `bound` body
/// applications. This is visible, defined behavior — not an unchecked promise — and every consumer enforces it:
/// interpretation exits the loop once the bound is reached even while the condition is still true, and the XLA
/// lowering threads an iteration counter through the `stablehlo.while` state and conjoins `counter < bound` into the
/// loop condition.
///
/// Differentiation through `while` follows one of three regimes:
///
///   - **Eager (unrolled).** When the differentiation context's primal values are concrete, the hybrid JVP rule
///     unrolls the loop (respecting any iteration bound), producing a straight-line — and therefore transposable —
///     pushforward, so eager reverse mode works.
///   - **Bounded staged (stored stacks + masked scan, reverse-capable).** When primal values are tracers and the
///     loop carries an iteration bound `B`, the rule stages an augmented primal while that *stores* every
///     per-iteration pushforward residual into a preallocated `[B, …]` stack (plus a Boolean validity mask), and the
///     tangent side becomes one masked linear [`scan`](super::scan::ScanOperation) of length `B` whose per-lane
///     `select` passes tangents through unchanged on the lanes beyond the actual trip count. The linear scan
///     transposes totally, so reverse mode composes through staged bounded loops.
///   - **Unbounded staged (fused recompute loop, forward-only).** Without a bound, no statically shaped residual
///     stack exists, so the rule stages a doubled-state linear loop that recomputes its residuals forward; that loop
///     rejects transposition, exactly like JAX's `while_loop`.
#[derive(Clone, Debug)]
pub struct WhileOperation<T, V, O>
where
    T: Type,
    V: Value<T>,
{
    /// Condition [`Program`] of this [`WhileOperation`] that maps the current loop state to one scalar Boolean
    /// predicate.
    pub(crate) condition: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Body [`Program`] of this [`WhileOperation`] that maps the current loop state to the next loop state.
    pub(crate) body: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Optional semantic iteration bound: when present, the loop runs at most this many iterations by definition,
    /// truncating even while the condition still produces true.
    pub(crate) iteration_bound: Option<usize>,
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> WhileOperation<ArrayType, V, O> {
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
        Ok(Self { condition, body, iteration_bound: None })
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> WhileOperation<T, V, O> {
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

    /// Returns the semantic iteration bound of this [`WhileOperation`], if any. Refer to the documentation of
    /// [`Self::with_iteration_bound`] for the truncation semantics a bound carries.
    #[inline]
    pub fn iteration_bound(&self) -> Option<usize> {
        self.iteration_bound
    }

    /// Returns this [`WhileOperation`] with its semantic iteration bound set to `bound` (or cleared when `bound` is
    /// `None`). The bound must be at least `1` when present.
    ///
    /// **A bounded while runs at most `bound` iterations by definition.** The bound is not a hint and not an
    /// unchecked promise: a loop whose condition would keep it running longer is *truncated* after `bound` body
    /// applications, which is visible, defined behavior. Interpretation exits the loop once the bound is reached
    /// even while the condition still produces true, and the XLA lowering threads an iteration counter through the
    /// `stablehlo.while` state and conjoins `counter < bound` into the loop condition. The bound is also what makes
    /// staged reverse-mode differentiation possible: it gives the loop's linearization a static residual-stack
    /// length (see the bounded staged regime described on [`WhileOperation`]).
    pub fn with_iteration_bound(mut self, bound: impl Into<Option<usize>>) -> Result<Self, ProgramError> {
        let bound = bound.into();
        if bound == Some(0) {
            return Err(TypeError { message: "while iteration bound must be at least 1".to_string() }.into());
        }
        self.iteration_bound = bound;
        Ok(self)
    }
}

impl<T: Type, V: Value<T>, O> Display for WhileOperation<T, V, O>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Operation<T> for WhileOperation<T, V, O> {
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
            if let Some(iteration_bound) = self.iteration_bound {
                operation.field("iteration_bound", iteration_bound)?;
            }
            operation.program("condition", &self.condition)?;
            operation.program("body", &self.body)
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for WhileOperation<ArrayType, V, O>
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
        let mut state = inputs.to_vec();
        let mut completed_iterations = 0;
        loop {
            // The iteration bound is semantic: a bounded loop runs at most `bound` iterations by definition, so the
            // loop exits here even while the condition still produces true.
            if self.iteration_bound.is_some_and(|bound| completed_iterations >= bound) {
                return Ok(state);
            }
            let condition_outputs = self.condition.interpret_in_context(context, state.clone())?;
            check_count!("output", condition_outputs, 1, ProgramError);
            if !condition_outputs[0].boolean()? {
                return Ok(state);
            }
            state = self.body.interpret_in_context(context, state)?;
            check_count!("output", state, self.state_types().len(), ProgramError);
            completed_iterations += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::arithmetic::SubOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{OneLikeOperation, ZeroLikeOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Test [`Operation`] type used for the nested condition and body programs.
    type TestOperation = ArrayOperation<ArrayType, TestArray>;

    /// Builds a condition program that maps a scalar `f64` state to the scalar Boolean predicate `state > 0`.
    fn greater_than_zero_condition() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a body program that maps a scalar `f64` state to `state - 1`.
    fn subtract_one_body() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
        let next_state = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
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
        let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
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

        // The semantic iteration bound defaults to absent, must be at least one, may be cleared with `None`, and is
        // reported by the accessor.
        assert_eq!(operation.iteration_bound(), None);
        let bounded = WhileOperation::new(greater_than_zero_condition(), subtract_one_body())
            .unwrap()
            .with_iteration_bound(2)
            .unwrap();
        assert_eq!(bounded.iteration_bound(), Some(2));
        assert_eq!(bounded.clone().with_iteration_bound(None).unwrap().iteration_bound(), None);
        assert_eq!(
            WhileOperation::new(greater_than_zero_condition(), subtract_one_body())
                .unwrap()
                .with_iteration_bound(0)
                .map(|_| ()),
            Err(ProgramError::Type(TypeError { message: "while iteration bound must be at least 1".to_string() })),
        );

        // The bound renders as an `iteration_bound=` field ahead of the nested programs.
        assert_eq!(
            format!("{bounded}"),
            indoc! {"
                while [
                    iteration_bound=2,
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

        // Interpretation iterates the body until the condition produces false.
        let outputs = operation.interpret(&crate::EagerContext::new(), &[TestArray::scalar(3.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        let outputs = operation.interpret(&crate::EagerContext::new(), &[TestArray::scalar(-1.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![-1.0]);
        assert_eq!(
            operation.interpret(&crate::EagerContext::new(), &[]),
            Err(ProgramError::Type(TypeError { message: "expected 1 input but got 0".to_string() })),
        );

        // A bounded while runs at most `bound` iterations by definition: the subtract-one loop at 5 would run five
        // iterations on its own, but the bound of 2 truncates it at 3 even though the condition is still true.
        let outputs = bounded.interpret(&crate::EagerContext::new(), &[TestArray::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![3.0]);
        // A loop that exits before reaching the bound is unaffected by it.
        let outputs = bounded.interpret(&crate::EagerContext::new(), &[TestArray::scalar(1.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);

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
