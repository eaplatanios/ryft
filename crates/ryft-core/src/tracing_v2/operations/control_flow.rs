use std::fmt::{Debug, Display};

use thiserror::Error;

use crate::batching::BatchingError;
use crate::compilation::CapturedConstant;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::{SupportsOne, SupportsZero};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{Instruction, Program, ProgramError, Value};
use crate::tracing::{AbstractTracer, AbstractTracingContext, Tracer, TracingContext};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext};
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, JvpTracer, LinearOperationOf, ResidualizedOperation,
    TangentContext,
};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Flat nested program shape used by control-flow operations.
///
/// Control-flow operations store nested regions as flat `Vec`-parameter programs because their
/// branch and loop bodies consume the operation operands directly. Structured Rust parameters are
/// flattened before a region is captured and reconstructed by the surrounding API when needed; the
/// operation itself only needs the ordered leaf signature for type checking, interpretation, JVP,
/// batching, and transposition.
pub type FlatProgram<V, O, T = ArrayType> = Program<T, V, O, Vec<V>, Vec<V>>;

/// Errors emitted by higher-order control-flow operations.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum ControlFlowError {
    /// A concrete value was used as a predicate but did not contain a scalar boolean.
    #[error("control-flow predicate value has type {type_}, but expected bool[]")]
    InvalidPredicateValue {
        /// Type metadata reported by the invalid predicate value.
        type_: ArrayType,
    },

    /// A transform reached a control-flow case that does not yet have a rule.
    #[error("control-flow operation does not yet provide a {transform} rule")]
    MissingTransformRule {
        /// Name of the missing transform.
        transform: &'static str,
    },

    /// Replaying a linear nested program needs an existing linear builder but no inputs were available.
    #[error("control-flow transform requires at least one tangent or cotangent leaf to supply a linear builder")]
    MissingLinearInvocationContext,
}

impl From<ControlFlowError> for ProgramError {
    /// Surfaces a control-flow error through [`ProgramError::Custom`], keeping control-flow extensibility out of the
    /// core [`ProgramError`] enum. Recover it with `error.as_any().downcast_ref::<ControlFlowError>()`.
    #[inline]
    fn from(error: ControlFlowError) -> Self {
        ProgramError::custom(error)
    }
}

/// Value-level predicate extraction used by interpreted control flow.
pub trait ControlFlowValue: Value<ArrayType> {
    /// Extracts a scalar boolean predicate from this value.
    fn control_flow_predicate(&self) -> Result<bool, ProgramError>;
}

impl<'domain, E> ControlFlowValue for JvpTracer<'domain, E>
where
    E: DifferentiationContext<Type = ArrayType>,
    E::Value: ControlFlowValue,
{
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        self.primal().control_flow_predicate()
    }
}

impl<C> ControlFlowValue for Tracer<C>
where
    C: Context<Type = ArrayType>,
{
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        Err(ControlFlowError::MissingTransformRule { transform: "traced predicate extraction" }.into())
    }
}

impl ControlFlowValue for ArrayType {
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        // `ArrayType` is only abstract staged-program metadata. It satisfies generic operation-enum bounds for
        // transform composition, but it never contains the concrete boolean needed to choose a branch.
        Err(ControlFlowError::MissingTransformRule { transform: "abstract predicate extraction" }.into())
    }
}

impl ControlFlowValue for CapturedConstant<ArrayType> {
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        // A captured constant is a reference into a side table, not the concrete predicate value itself. Control-flow
        // staging must keep predicates in the IR or add a transform-specific rule instead of trying to branch here.
        Err(ControlFlowError::MissingTransformRule { transform: "captured predicate extraction" }.into())
    }
}

impl<V: ControlFlowValue> ControlFlowValue for ArrayBatch<V> {
    fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
        if self.batch_axis().is_some() {
            return Err(ControlFlowError::MissingTransformRule { transform: "batched predicate control flow" }.into());
        }
        self.value().control_flow_predicate()
    }
}

/// Type metadata that can represent the scalar boolean predicate expected by control-flow operations.
pub(crate) trait ControlFlowPredicateType: PartialEq + Type {
    /// Validates that this metadata is the scalar boolean predicate type.
    fn ensure_scalar_bool_type(&self) -> Result<(), TypeError>;
}

impl ControlFlowPredicateType for ArrayType {
    #[inline]
    fn ensure_scalar_bool_type(&self) -> Result<(), TypeError> {
        ensure_array_scalar_bool_type(self)
    }
}

impl ControlFlowPredicateType for DataType {
    #[inline]
    fn ensure_scalar_bool_type(&self) -> Result<(), TypeError> {
        ensure_data_scalar_bool_type(self)
    }
}

/// Predicate source for a [`ConditionOperation`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConditionPredicate<T: PartialEq + Type = ArrayType> {
    /// The first operation input is the predicate.
    RuntimeInput(T),

    /// The predicate is captured in the operation and is not an operation input.
    Captured(bool),
}

/// Two-way conditional operation with nested true and false branch programs.
#[derive(Clone, Debug)]
pub struct ConditionOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Predicate source.
    predicate: ConditionPredicate<T>,

    /// Program evaluated when the predicate is true.
    true_branch: FlatProgram<V, O, T>,

    /// Program evaluated when the predicate is false.
    false_branch: FlatProgram<V, O, T>,
}

/// While-loop operation with nested condition and body programs over the same loop-carried state.
#[derive(Clone, Debug)]
pub struct WhileOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Program that maps the current loop state to one scalar boolean predicate.
    condition: FlatProgram<V, O, T>,

    /// Program that maps the current loop state to the next loop state.
    body: FlatProgram<V, O, T>,
}

/// Returns the flat input types of a nested control-flow program.
pub fn flat_program_input_types<T: Type, V: Value<T>, O: Operation<T>>(program: &FlatProgram<V, O, T>) -> Vec<T> {
    program.inputs().map(|input| input.r#type().into_owned()).collect()
}

/// Returns the flat output types of a nested control-flow program.
pub fn flat_program_output_types<T: Type, V: Value<T>, O: Operation<T>>(program: &FlatProgram<V, O, T>) -> Vec<T> {
    program.outputs().map(|output| output.r#type().into_owned()).collect()
}

/// Validates that `predicate_type` is exactly the canonical scalar boolean array type.
fn ensure_array_scalar_bool_type(predicate_type: &ArrayType) -> Result<(), TypeError> {
    let expected = ArrayType::scalar(DataType::Boolean);
    if predicate_type != &expected {
        return Err(TypeError {
            message: format!("control-flow predicate type must be {expected}, but got {predicate_type}"),
        });
    }
    Ok(())
}

/// Validates that `predicate_type` is exactly the canonical scalar boolean data type.
fn ensure_data_scalar_bool_type(predicate_type: &DataType) -> Result<(), TypeError> {
    let expected = DataType::Boolean;
    if predicate_type != &expected {
        return Err(TypeError {
            message: format!("control-flow predicate type must be {expected}, but got {predicate_type}"),
        });
    }
    Ok(())
}

/// Validates that two flat type signatures are identical.
pub(crate) fn ensure_types_match<T: PartialEq + Type>(
    context: &'static str,
    left: &[T],
    right: &[T],
) -> Result<(), TypeError> {
    if left != right {
        return Err(TypeError {
            message: format!(
                "{context} type mismatch: left has [{}], right has [{}]",
                left.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                right.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            ),
        });
    }
    Ok(())
}

/// Validates a flat operation input count.
fn ensure_input_count(expected: usize, actual: usize, operation: &'static str) -> Result<(), TypeError> {
    if expected != actual {
        return Err(TypeError { message: format!("{operation} expected {expected} input type(s) but got {actual}") });
    }
    Ok(())
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> ConditionOperation<V, O, ArrayType> {
    /// Creates a condition whose predicate is supplied as the first operation input.
    pub fn new(
        predicate_type: ArrayType,
        true_branch: FlatProgram<V, O, ArrayType>,
        false_branch: FlatProgram<V, O, ArrayType>,
    ) -> Result<Self, TypeError> {
        ensure_array_scalar_bool_type(&predicate_type)?;
        Self::from_parts(ConditionPredicate::RuntimeInput(predicate_type), true_branch, false_branch)
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> ConditionOperation<V, O, T> {
    /// Creates a condition whose predicate is captured in the operation.
    pub fn with_captured_predicate(
        predicate: bool,
        true_branch: FlatProgram<V, O, T>,
        false_branch: FlatProgram<V, O, T>,
    ) -> Result<Self, TypeError> {
        Self::from_parts(ConditionPredicate::Captured(predicate), true_branch, false_branch)
    }

    /// Creates a condition after validating branch signatures.
    fn from_parts(
        predicate: ConditionPredicate<T>,
        true_branch: FlatProgram<V, O, T>,
        false_branch: FlatProgram<V, O, T>,
    ) -> Result<Self, TypeError> {
        let input_types = flat_program_input_types(&true_branch);
        ensure_types_match("condition branch input", &input_types, &flat_program_input_types(&false_branch))?;
        let output_types = flat_program_output_types(&true_branch);
        ensure_types_match("condition branch output", &output_types, &flat_program_output_types(&false_branch))?;
        Ok(Self { predicate, true_branch, false_branch })
    }

    /// Returns the predicate source for this condition.
    #[inline]
    pub fn predicate(&self) -> &ConditionPredicate<T> {
        &self.predicate
    }

    /// Returns the branch program evaluated when the predicate is true.
    #[inline]
    pub fn true_branch(&self) -> &FlatProgram<V, O, T> {
        &self.true_branch
    }

    /// Returns the branch program evaluated when the predicate is false.
    #[inline]
    pub fn false_branch(&self) -> &FlatProgram<V, O, T> {
        &self.false_branch
    }

    /// Returns the operand input types consumed by both branches.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        flat_program_input_types(&self.true_branch)
    }

    /// Returns the output types produced by both branches.
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        flat_program_output_types(&self.true_branch)
    }

    /// Returns the branch selected by `predicate`.
    fn selected_branch(&self, predicate: bool) -> &FlatProgram<V, O, T> {
        if predicate { &self.true_branch } else { &self.false_branch }
    }

    fn infer_output_types_impl(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>
    where
        T: ControlFlowPredicateType,
    {
        let operand_input_types = self.input_types();
        let operand_start = match &self.predicate {
            ConditionPredicate::RuntimeInput(predicate_type) => {
                ensure_input_count(operand_input_types.len() + 1, input_types.len(), "condition")?;
                input_types[0].ensure_scalar_bool_type()?;
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
            ConditionPredicate::Captured(_) => {
                ensure_input_count(operand_input_types.len(), input_types.len(), "condition")?;
                0
            }
        };
        ensure_types_match("condition operand", &operand_input_types, &input_types[operand_start..])?;
        Ok(self.output_types())
    }

    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, "condition")?.bracketed(|operation| {
            match &self.predicate {
                ConditionPredicate::RuntimeInput(predicate_type) => {
                    operation.field("predicate", format_args!("runtime_input(type={predicate_type})"))?;
                }
                ConditionPredicate::Captured(predicate) => {
                    operation.field("predicate", format_args!("captured({predicate})"))?;
                }
            }
            operation.program("true_branch", &self.true_branch)?;
            operation.program("false_branch", &self.false_branch)
        })
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for ConditionOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: ControlFlowPredicateType, V: Value<T>, O: Operation<T>> Operation<T> for ConditionOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "condition"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        self.infer_output_types_impl(input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for ConditionOperation<V, O, ArrayType>
where
    V: ControlFlowValue,
    O: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let (predicate, operands) = match self.predicate {
            ConditionPredicate::RuntimeInput(_) => (inputs[0].control_flow_predicate()?, &inputs[1..]),
            ConditionPredicate::Captured(predicate) => (predicate, inputs),
        };
        self.selected_branch(predicate).interpret(operands.to_vec())
    }
}

impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for ConditionOperation<V, O, ArrayType>
where
    O: TransposableOperation<ArrayType, V, O>
        + crate::operations::constants::SupportsZero<ArrayType>
        + SupportsAdd<ArrayType>
        + From<ConditionOperation<V, O, ArrayType>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        let ConditionPredicate::Captured(predicate) = self.predicate else {
            return Err(
                ControlFlowError::MissingTransformRule { transform: "runtime-predicate condition transpose" }.into()
            );
        };
        if output_cotangents.is_empty() {
            return if self.input_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(ControlFlowError::MissingLinearInvocationContext.into())
            };
        }
        if output_cotangents.iter().all(Cotangent::is_zero) {
            return Ok(vec![Cotangent::Zero; self.input_types().len()]);
        }
        let transposed_condition = ConditionOperation::with_captured_predicate(
            predicate,
            context.transpose_nested(&self.true_branch)?,
            context.transpose_nested(&self.false_branch)?,
        )?;
        let materialized = output_cotangents
            .iter()
            .zip(self.output_types().iter())
            .map(|(cotangent, output_type)| stage_cotangent(context, cotangent, output_type))
            .collect::<Vec<_>>();
        let cotangents = context.stage_operation(O::from(transposed_condition), materialized.as_slice())?;
        check_count!("output", cotangents, self.input_types().len(), ProgramError);
        Ok(cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}

/// Returns a concrete cotangent atom for `cotangent`, staging a typed `Zero` op when the cotangent
/// is structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
pub(crate) fn stage_cotangent<'transpose, T: Type, V: Value<T>, O>(
    context: &AbstractTracingContext<'transpose, T, V, O>,
    cotangent: &Cotangent<'transpose, T, V, O>,
    output_type: &T,
) -> AbstractTracer<'transpose, T, V, O>
where
    O: Operation<T> + crate::operations::constants::SupportsZero<T>,
{
    match cotangent {
        Cotangent::Staged(cotangent) => return cotangent.clone(),
        Cotangent::Zero => {}
    }
    let builder = context.builder();
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow
        .instructions
        .push(Instruction::new(O::zero_operation(output_type.clone()), vec![], vec![output]));
    drop(builder_borrow);
    context.tracer(output, None)
}

impl<V, D, O> DifferentiableOperation<D> for ConditionOperation<V, O, ArrayType>
where
    V: ControlFlowValue,
    D: Domain<Type = ArrayType, Value = V> + Domain<Type = ArrayType, Value = V, Constant = V> + DifferentiationContext,
    O: Operation<ArrayType> + DifferentiableOperation<D> + InterpretableOperation<ArrayType, V>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Vec<V>: Parameterized<
            V,
            Family: ParameterizedFamily<D::Tangent>,
            To<V> = Vec<V>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let operand_count = self.input_types().len();
        let expected_count = operand_count + usize::from(matches!(self.predicate, ConditionPredicate::RuntimeInput(_)));
        check_count!("input", inputs, expected_count, ProgramError);
        let (predicate, operands) = match self.predicate {
            ConditionPredicate::RuntimeInput(_) => (inputs[0].primal().control_flow_predicate()?, &inputs[1..]),
            ConditionPredicate::Captured(predicate) => (predicate, inputs),
        };
        let primal_operands = operands.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        // Materialize symbolic-zero operand tangents into concrete linear-builder atoms before
        // inlining the branch's pushforward into the active JVP builder.
        let tangent_operands = operands
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let branch = self.selected_branch(predicate);
        let (primal_outputs, pushforward) = context.differentiable().linearize_program(branch, primal_operands)?;
        let pushforward_program = pushforward.program_with_residual_constants()?;
        let tangent_outputs = context.stage_program(&pushforward_program, tangent_operands)?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

/// JVP rule for `ConditionOperation` under an enclosing [`TracingContext`].
///
/// Predicate extraction does not work at trace time (the differentiable host value is a [`Tracer`],
/// whose [`ControlFlowValue::control_flow_predicate`] implementation errors), so this impl reports
/// [`ControlFlowError::MissingTransformRule`] for any traced JVP attempt.
impl<'domain, D, V, O> DifferentiableOperation<TracingContext<'domain, D, V>> for ConditionOperation<V, O, ArrayType>
where
    D: DifferentiationContext<Type = ArrayType, Value = V, Constant = V>
        + Domain<Type = ArrayType, Value = V, Constant = V, Operation = O>
        + 'domain,
    V: ControlFlowValue + Value<ArrayType>,
    O: Clone + Operation<ArrayType> + SupportsAdd<ArrayType> + SupportsZero<ArrayType> + SupportsOne<ArrayType>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
    TracingContext<'domain, D, V>: DifferentiationContext<Type = ArrayType, Constant = V>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, TracingContext<'domain, D, V>>,
        _inputs: &[JvpTracer<'jvp, TracingContext<'domain, D, V>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, D, V>>>, ProgramError>
    where
        TracingContext<'domain, D, V>: 'jvp,
    {
        Err(ControlFlowError::MissingTransformRule { transform: "linearization domain traced JVP" }.into())
    }
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> WhileOperation<V, O, ArrayType> {
    /// Creates a while loop from a condition program and a body program.
    pub fn new(condition: FlatProgram<V, O, ArrayType>, body: FlatProgram<V, O, ArrayType>) -> Result<Self, TypeError> {
        let state_types = flat_program_input_types(&condition);
        ensure_types_match("while condition/body input", &state_types, &flat_program_input_types(&body))?;
        let condition_output_types = flat_program_output_types(&condition);
        if condition_output_types.len() != 1 {
            return Err(TypeError {
                message: format!(
                    "while condition must return exactly one predicate leaf but returned {}",
                    condition_output_types.len()
                ),
            });
        }
        ensure_array_scalar_bool_type(&condition_output_types[0])?;
        ensure_types_match("while body output", &state_types, &flat_program_output_types(&body))?;
        Ok(Self { condition, body })
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> WhileOperation<V, O, T> {
    /// Returns the condition program evaluated before each loop iteration.
    #[inline]
    pub fn condition(&self) -> &FlatProgram<V, O, T> {
        &self.condition
    }

    /// Returns the body program that computes the next loop-carried state.
    #[inline]
    pub fn body(&self) -> &FlatProgram<V, O, T> {
        &self.body
    }

    /// Returns the loop-carried state types.
    #[inline]
    pub fn state_types(&self) -> Vec<T> {
        flat_program_input_types(&self.body)
    }

    fn infer_output_types_impl(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        let state_types = self.state_types();
        ensure_input_count(state_types.len(), input_types.len(), "while")?;
        ensure_types_match("while input", &state_types, input_types)?;
        Ok(state_types)
    }

    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, "while")?.bracketed(|operation| {
            operation.program("condition", &self.condition)?;
            operation.program("body", &self.body)
        })
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for WhileOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> Operation<T> for WhileOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "while"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        self.infer_output_types_impl(input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for WhileOperation<V, O, ArrayType>
where
    V: ControlFlowValue,
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
            if !condition_outputs[0].control_flow_predicate()? {
                return Ok(state);
            }
            state = self.body.interpret(state)?;
            check_count!("output", state, self.state_types().len(), ProgramError);
        }
    }
}

impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for WhileOperation<V, O, ArrayType>
where
    O: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ControlFlowError::MissingTransformRule { transform: "while transpose" }.into())
    }
}

/// JVP rule for `WhileOperation` under an enclosing [`TracingContext`]. See the matching
/// [`ConditionOperation`] impl for rationale; predicate extraction does not work at trace time.
impl<'domain, D, V, O> DifferentiableOperation<TracingContext<'domain, D, V>> for WhileOperation<V, O, ArrayType>
where
    D: DifferentiationContext<Type = ArrayType, Value = V, Constant = V>
        + Domain<Type = ArrayType, Value = V, Constant = V, Operation = O>
        + 'domain,
    V: ControlFlowValue + Value<ArrayType>,
    O: Clone + Operation<ArrayType> + SupportsAdd<ArrayType> + SupportsZero<ArrayType> + SupportsOne<ArrayType>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
    TracingContext<'domain, D, V>: DifferentiationContext<Type = ArrayType, Constant = V>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, TracingContext<'domain, D, V>>,
        _inputs: &[JvpTracer<'jvp, TracingContext<'domain, D, V>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, D, V>>>, ProgramError>
    where
        TracingContext<'domain, D, V>: 'jvp,
    {
        Err(ControlFlowError::MissingTransformRule { transform: "linearization domain traced JVP" }.into())
    }
}

impl<V, D, O> DifferentiableOperation<D> for WhileOperation<V, O, ArrayType>
where
    V: ControlFlowValue,
    D: Domain<Type = ArrayType, Value = V> + Domain<Type = ArrayType, Value = V, Constant = V> + DifferentiationContext,
    O: Operation<ArrayType> + DifferentiableOperation<D> + InterpretableOperation<ArrayType, V>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Vec<V>: Parameterized<
            V,
            Family: ParameterizedFamily<D::Tangent>,
            To<V> = Vec<V>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let state_count = self.state_types().len();
        check_count!("input", inputs, state_count, ProgramError);
        let mut state_primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        // Materialize symbolic-zero state tangents into concrete atoms at loop entry so each
        // body pushforward can be inlined into the active JVP builder.
        let mut state_tangents = inputs
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;

        loop {
            let condition_outputs = self.condition.interpret(state_primals.clone())?;
            check_count!("output", condition_outputs, 1, ProgramError);
            if !condition_outputs[0].control_flow_predicate()? {
                return Ok(state_primals
                    .into_iter()
                    .zip(state_tangents)
                    .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
                    .collect());
            }

            let (next_primals, pushforward) =
                context.differentiable().linearize_program(&self.body, state_primals.clone())?;
            let pushforward_program = pushforward.program_with_residual_constants()?;
            let next_tangents = context.stage_program(&pushforward_program, state_tangents)?;
            check_count!("output", next_primals, state_count, ProgramError);
            check_count!("output", next_tangents, state_count, ProgramError);
            state_primals = next_primals;
            state_tangents = next_tangents;
        }
    }
}

fn batch_condition_with_interpreter<VOperation, V, O, F>(
    condition: &ConditionOperation<VOperation, O, ArrayType>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: ControlFlowValue + crate::tracing_v2::operations::select::Select,
    O: Operation<ArrayType>,
    F: FnMut(&FlatProgram<VOperation, O>, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    match condition.predicate() {
        ConditionPredicate::Captured(predicate) => {
            let branch = if *predicate { condition.true_branch() } else { condition.false_branch() };
            interpret_program(branch, inputs.to_vec())
        }
        ConditionPredicate::RuntimeInput(_) => {
            let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
                return Err(BatchingError::UnsupportedOperation {
                    message: "cannot batch a condition operation with no predicate input".to_string(),
                }
                .into());
            };
            match predicate_batch.batch_axis() {
                None => {
                    let predicate = predicate_batch.value().control_flow_predicate()?;
                    let branch = if predicate { condition.true_branch() } else { condition.false_branch() };
                    interpret_program(branch, operand_inputs.to_vec())
                }
                Some(predicate_axis) => {
                    let true_outputs = interpret_program(condition.true_branch(), operand_inputs.to_vec())?;
                    let false_outputs = interpret_program(condition.false_branch(), operand_inputs.to_vec())?;
                    check_count!("output", true_outputs, false_outputs.len(), ProgramError);
                    true_outputs
                        .into_iter()
                        .zip(false_outputs)
                        .map(|(true_output, false_output)| -> Result<ArrayBatch<V>, ProgramError> {
                            let output_axis = match (true_output.batch_axis(), false_output.batch_axis()) {
                                (Some(left), Some(right)) if left != right => {
                                    return Err(BatchingError::MisalignedBatchAxes {
                                        message: format!(
                                            "condition branches produced lane-varying outputs at mismatched axes \
                                            ({left} vs {right})",
                                        ),
                                    }
                                    .into());
                                }
                                (Some(axis), _) | (_, Some(axis)) => axis,
                                (None, None) => predicate_axis,
                            };
                            let selected = V::select(
                                predicate_batch.value().clone(),
                                true_output.value().clone(),
                                false_output.value().clone(),
                            )?;
                            let output_type = selected.r#type().into_owned();
                            ArrayBatch::new(output_type, selected, Some(output_axis))
                        })
                        .collect()
                }
            }
        }
    }
}

impl<V, O> BatchableOperation<V, ()> for ConditionOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + ControlFlowValue + crate::tracing_v2::operations::select::Select,
    O: BatchableOperation<V, ()>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_condition_with_interpreter(self, inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
            )
        })
    }
}

impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for ConditionOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + ControlFlowValue,
    Tracer<C>: crate::tracing_v2::operations::select::Select,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        batch_condition_with_interpreter(self, inputs, |program, program_inputs| {
            context.interpret_program(program, program_inputs)
        })
    }
}

/// `Tangent`-specific batching for [`ConditionOperation`]. The generic impl above doesn't apply
/// because [`Tangent`] does not implement [`ControlFlowValue`] or
/// [`Select`](crate::tracing_v2::operations::select::Select) (those would require materializing
/// the inner symbolic-zero tangent at the tangent layer). For captured-predicate conditions we
/// pick the branch and recurse at the tangent layer. For runtime-predicate conditions we
/// materialize each input's [`Tangent::Zero`] to the matching `V::zero(t)` via the default
/// [`BatchableOperation`] rule, dispatch to the V-level [`ConditionOperation`] batching rule
/// (which itself handles lane-uniform vs lane-varying predicates by selecting per lane), and
/// re-wrap each output as `Tangent::Value`. This is the same materialize-then-dispatch pattern used by
/// [`LinearArrayOperation`](crate::tracing_v2::operations::primitive::LinearArrayOperation)'s tangent batching rule.
impl<V, O> BatchableOperation<Tangent<ArrayType, V>, ()> for ConditionOperation<V, O, ArrayType>
where
    Self: BatchableOperation<V>,
    V: Value<ArrayType>
        + crate::operations::constants::Zero<ArrayType>
        + ControlFlowValue
        + crate::tracing_v2::operations::select::Select,
    O: BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>, ()>,
{
    fn batch(
        &self,
        _context: &(),
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        match self.predicate() {
            ConditionPredicate::Captured(predicate) => {
                let branch = if *predicate { self.true_branch() } else { self.false_branch() };
                branch.interpret_with(
                    inputs.to_vec(),
                    |_, constant| Ok(ArrayBatch::unbatched(Tangent::Value(constant.clone()))),
                    |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
                )
            }
            ConditionPredicate::RuntimeInput(_) => {
                let materialized: Vec<ArrayBatch<V>> = inputs
                    .iter()
                    .map(|input| -> Result<ArrayBatch<V>, ProgramError> {
                        let value = match input.value() {
                            Tangent::Zero(t) => V::zero(t)?,
                            Tangent::Value(v) => v.clone(),
                        };
                        ArrayBatch::new(input.r#type().into_owned(), value, input.batch_axis())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let v_outputs = <Self as BatchableOperation<V>>::batch(self, &(), materialized.as_slice())?;
                v_outputs
                    .into_iter()
                    .map(|out| -> Result<ArrayBatch<Tangent<ArrayType, V>>, ProgramError> {
                        let output_type = out.r#type().into_owned();
                        let output_axis = out.batch_axis();
                        ArrayBatch::new(output_type, Tangent::Value(out.into_value()), output_axis)
                    })
                    .collect()
            }
        }
    }
}

fn batch_while_with_interpreter<VOperation, V, O, F>(
    while_operation: &WhileOperation<VOperation, O, ArrayType>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType>
        + ControlFlowValue
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::operations::manipulation::BroadcastInDim,
    O: Operation<ArrayType>,
    F: FnMut(&FlatProgram<VOperation, O>, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    // Run the condition once on the initial state to discover whether the predicate is
    // lane-uniform or lane-varying. The two cases diverge from here: lane-uniform takes the
    // original eager-loop path; lane-varying threads a per-lane mask through every iteration
    // and runs the body until no lane is still active.
    let mut state = inputs.to_vec();
    let initial_condition_outputs = interpret_program(while_operation.condition(), state.clone())?;
    check_count!("output", initial_condition_outputs, 1, ProgramError);
    let initial_predicate = initial_condition_outputs.into_iter().next().expect("checked above");
    if initial_predicate.batch_axis().is_none() {
        if !initial_predicate.value().control_flow_predicate()? {
            return Ok(state);
        }
        state = interpret_program(while_operation.body(), state)?;
        return run_lane_uniform_while_loop::<VOperation, V, O, F>(
            while_operation.condition(),
            while_operation.body(),
            state,
            &mut interpret_program,
        );
    }
    // Lane-varying path: the predicate carries a batch axis. Track a per-lane mask, mask
    // state updates per lane via `Select`, and exit once `any(mask)` is false.
    run_lane_varying_while_loop::<VOperation, V, O, F>(
        while_operation.condition(),
        while_operation.body(),
        state,
        initial_predicate,
        &mut interpret_program,
    )
}

impl<V, O> BatchableOperation<V, ()> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType>
        + ControlFlowValue
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::operations::manipulation::BroadcastInDim,
    O: BatchableOperation<V, ()>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_while_with_interpreter(self, inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
            )
        })
    }
}

impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for WhileOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + ControlFlowValue,
    Tracer<C>: crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::operations::manipulation::BroadcastInDim,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        batch_while_with_interpreter(self, inputs, |program, program_inputs| {
            context.interpret_program(program, program_inputs)
        })
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a lane-uniform
/// scalar Boolean predicate. Each iteration runs the body when the predicate is `true` and exits
/// when it becomes `false`. This is the original simple loop preserved for the lane-uniform case.
fn run_lane_uniform_while_loop<VOperation, V, O, F>(
    condition: &FlatProgram<VOperation, O>,
    body: &FlatProgram<VOperation, O>,
    mut state: Vec<ArrayBatch<V>>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: ControlFlowValue,
    F: FnMut(&FlatProgram<VOperation, O>, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    loop {
        let condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate_batch = &condition_outputs[0];
        if predicate_batch.batch_axis().is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop condition produced a lane-varying predicate mid-iteration after starting \
                    lane-uniform; this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        if !predicate_batch.value().control_flow_predicate()? {
            return Ok(state);
        }
        state = interpret_program(body, state)?;
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a lane-varying
/// predicate (one Boolean per mapped lane). Each iteration:
///
///   1. Updates the per-lane active mask by AND-ing with the current per-lane predicate.
///   2. Stops when no lane is still active (`any(mask) == false`).
///   3. Runs the body to produce candidate updated state.
///   4. Masks state updates per lane via [`Select`](crate::tracing_v2::operations::select::Select)
///      so inactive lanes retain their prior state forever.
///
/// This implementation requires a value type that supports [`Reduce`](
/// crate::tracing_v2::operations::reduce::Reduce) (for the `any` aggregation),
/// [`LogicalBinary`](crate::tracing_v2::operations::logical::LogicalBinary) (for `mask & current`),
/// [`Select`](crate::tracing_v2::operations::select::Select), and
/// [`BroadcastInDim`](crate::operations::manipulation::BroadcastInDim) — the same
/// primitives every staged value type already needs for the rest of the operation enum.
fn run_lane_varying_while_loop<VOperation, V, O, F>(
    condition: &FlatProgram<VOperation, O>,
    body: &FlatProgram<VOperation, O>,
    mut state: Vec<ArrayBatch<V>>,
    initial_predicate: ArrayBatch<V>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType>
        + ControlFlowValue
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::operations::manipulation::BroadcastInDim,
    F: FnMut(&FlatProgram<VOperation, O>, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    let predicate_axis = initial_predicate.batch_axis().expect("lane-varying entry guarantees a batched predicate");
    let mut active_mask = initial_predicate;
    loop {
        if !lane_varying_any_active(&active_mask, predicate_axis)? {
            return Ok(state);
        }
        let body_outputs = interpret_program(body, state.clone())?;
        check_count!("output", body_outputs, state.len(), ProgramError);
        state = state
            .into_iter()
            .zip(body_outputs)
            .map(|(prior, candidate)| mask_state_element(&active_mask, predicate_axis, candidate, prior))
            .collect::<Result<Vec<_>, _>>()?;
        let next_condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", next_condition_outputs, 1, ProgramError);
        let next_predicate = next_condition_outputs.into_iter().next().expect("checked above");
        if next_predicate.batch_axis().is_none() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop predicate became lane-uniform mid-iteration after starting lane-varying; \
                    this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        active_mask = combine_active_mask(active_mask, next_predicate)?;
    }
}

/// Returns `true` when at least one lane of `mask` is active by reducing along `predicate_axis`
/// and extracting the resulting scalar Boolean.
fn lane_varying_any_active<V: ControlFlowValue + crate::tracing_v2::operations::reduce::Reduce>(
    mask: &ArrayBatch<V>,
    predicate_axis: usize,
) -> Result<bool, ProgramError> {
    let reduced = mask
        .value()
        .clone()
        .reduce(&[predicate_axis], crate::tracing_v2::operations::reduce::ReductionKind::Any);
    reduced.control_flow_predicate()
}

/// Combines the prior `active_mask` with the current `next_predicate` via logical AND. Both must
/// be batched on the same physical axis; the result inherits that axis.
fn combine_active_mask<V: Value<ArrayType> + crate::tracing_v2::operations::logical::LogicalBinary>(
    active_mask: ArrayBatch<V>,
    next_predicate: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, ProgramError> {
    let axis = active_mask.batch_axis();
    let combined = active_mask
        .into_value()
        .logical_binary(next_predicate.into_value(), crate::tracing_v2::operations::logical::LogicalKind::And);
    let combined_type = combined.r#type().into_owned();
    ArrayBatch::new(combined_type, combined, axis)
}

/// Builds the masked update for one state element by broadcasting the per-lane mask to the
/// element's physical shape and selecting between the candidate body output and the prior state
/// per lane.
fn mask_state_element<V>(
    active_mask: &ArrayBatch<V>,
    predicate_axis: usize,
    candidate: ArrayBatch<V>,
    prior: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, ProgramError>
where
    V: Value<ArrayType>
        + crate::tracing_v2::operations::select::Select
        + crate::operations::manipulation::BroadcastInDim,
{
    let candidate_axis =
        candidate.batch_axis().or(prior.batch_axis()).ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "lane-varying while body produced a lane-uniform state element; this is not yet supported"
                .to_string(),
        })?;
    let candidate_type = candidate.r#type().into_owned();
    let mask_type = active_mask.r#type().into_owned();
    let mask_broadcast_dimensions: Vec<usize> = (0..mask_type.rank())
        .map(|i| {
            if i == predicate_axis {
                candidate_axis
            } else if i < predicate_axis {
                // mask axes left of the predicate axis carry over to the candidate left of `candidate_axis`.
                i
            } else {
                // mask axes right of the predicate axis carry over to the candidate right of `candidate_axis`.
                i + (candidate_type.rank() - mask_type.rank())
            }
        })
        .collect();
    let broadcasted_mask =
        active_mask.value().clone().broadcast_in_dim(candidate_type.clone(), mask_broadcast_dimensions);
    let selected = V::select(broadcasted_mask, candidate.into_value(), prior.into_value())?;
    let selected_type = selected.r#type().into_owned();
    ArrayBatch::new(selected_type, selected, Some(candidate_axis))
}

/// `Tangent`-specific batching for [`WhileOperation`]. Like the `ConditionOperation` Tangent impl
/// above, this exists because [`Tangent`] does not implement [`ControlFlowValue`]. Pushforward /
/// pullback programs do not emit `While` today (the JVP rule unrolls loops at trace time and
/// `WhileOperation::transpose` errors), so this path is unreachable from `jacfwd` / `jacrev`;
/// it returns [`BatchingError::UnsupportedOperation`] if a caller manually constructs a tangent `While`.
impl<V, O> BatchableOperation<Tangent<ArrayType, V>, ()> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + ControlFlowValue,
    O: BatchableOperation<Tangent<ArrayType, V>, ()>,
{
    fn batch(
        &self,
        _context: &(),
        _inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        Err(BatchingError::UnsupportedOperation {
            message: "missing batching rule for while over tangent runtime values".to_string(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::operations::InterpretableOperation as _;
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::domains::{AbstractDomain, Domain};
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, SUB_OPERATION_NAME, Scale, SupportsAdd, SupportsNeg, SupportsScale,
    };
    use crate::operations::constants::{One, OneLike, SupportsZero, Zero, ZeroLike};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{ProgramBuilder, Value};
    use crate::tracing_v2::{ArrayOperation, FactorParameterizedOperation};
    use crate::types::DataType;

    use super::*;

    #[derive(Clone, Debug, Parameter, PartialEq)]
    enum TestValue {
        Bool(bool),
        Number(f64),
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Bool(value) => Display::fmt(value, formatter),
                Self::Number(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Typed<ArrayType> for TestValue {
        fn r#type(&self) -> Cow<'_, ArrayType> {
            match self {
                Self::Bool(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean)),
                Self::Number(_) => Cow::Owned(ArrayType::scalar(DataType::F64)),
            }
        }
    }

    impl Value<ArrayType> for TestValue {}

    impl ZeroLike for TestValue {
        fn zero_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(false),
                Self::Number(_) => Self::Number(0.0),
            }
        }
    }

    impl OneLike for TestValue {
        fn one_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(true),
                Self::Number(_) => Self::Number(1.0),
            }
        }
    }

    impl Zero<ArrayType> for TestValue {
        fn zero(value_type: &ArrayType) -> Result<Self, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(Self::Bool(false)),
                DataType::F64 => Ok(Self::Number(0.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize zero for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl One<ArrayType> for TestValue {
        fn one(value_type: &ArrayType) -> Result<Self, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(Self::Bool(true)),
                DataType::F64 => Ok(Self::Number(1.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize one for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl ControlFlowValue for TestValue {
        fn control_flow_predicate(&self) -> Result<bool, ProgramError> {
            match self {
                Self::Bool(value) => Ok(*value),
                value => Err(ControlFlowError::InvalidPredicateValue { type_: value.r#type().into_owned() }.into()),
            }
        }
    }

    #[derive(Clone, Debug)]
    enum TestOperation {
        Add,
        Sub,
        IsPositive,
        Condition(Box<ConditionOperation<TestValue, TestOperation, ArrayType>>),
        While(Box<WhileOperation<TestValue, TestOperation, ArrayType>>),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => ADD_OPERATION_NAME,
                Self::Sub => SUB_OPERATION_NAME,
                Self::IsPositive => "is_positive",
                Self::Condition(condition) => condition.name(),
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add | Self::Sub => {
                    ensure_input_count(2, input_types.len(), self.name())?;
                    ensure_types_match(self.name(), &input_types[..1], &input_types[1..])?;
                    Ok(vec![input_types[0].clone()])
                }
                Self::IsPositive => {
                    ensure_input_count(1, input_types.len(), self.name())?;
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::Condition(condition) => condition.infer_output_types(input_types),
                Self::While(while_operation) => while_operation.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                Self::While(while_operation) => while_operation.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("add expected numeric inputs").into() }.into()),
                },
                Self::Sub => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left - right)]),
                    _ => Err(TypeError { message: ("sub expected numeric inputs").into() }.into()),
                },
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: ("is_positive expected a numeric input").into() }.into()),
                },
                Self::Condition(condition) => condition.interpret(inputs),
                Self::While(while_operation) => while_operation.interpret(inputs),
            }
        }
    }

    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        Add,
        Neg,
        Scale { factor: TestValue },
        Condition(Box<ConditionOperation<TestValue, TestLinearOperation, ArrayType>>),
    }

    impl Display for TestLinearOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => "linear_add",
                Self::Neg => "linear_neg",
                Self::Scale { .. } => "linear_scale",
                Self::Condition(condition) => condition.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add => {
                    ensure_input_count(2, input_types.len(), self.name())?;
                    ensure_types_match(self.name(), &input_types[..1], &input_types[1..])?;
                    Ok(vec![input_types[0].clone()])
                }
                Self::Neg | Self::Scale { .. } => {
                    ensure_input_count(1, input_types.len(), self.name())?;
                    Ok(vec![input_types[0].clone()])
                }
                Self::Condition(condition) => condition.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestLinearOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("linear add expected numeric inputs").into() }.into()),
                },
                Self::Neg => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(-value)]),
                    _ => Err(TypeError { message: ("linear neg expected a numeric input").into() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: ("linear scale expected numeric inputs").into() }.into()),
                },
                Self::Condition(condition) => condition.interpret(inputs),
            }
        }
    }

    impl TransposableOperation<ArrayType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            context: &mut AbstractTracingContext<'transpose, ArrayType, TestValue, TestLinearOperation>,
            input_types: &[&ArrayType],
            output_cotangents: &[Cotangent<'transpose, ArrayType, TestValue, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, ArrayType, TestValue, TestLinearOperation>>, ProgramError> {
            match self {
                Self::Add => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
                }
                Self::Neg => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    match &output_cotangents[0] {
                        Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                        Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    }
                }
                Self::Scale { factor } => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    match &output_cotangents[0] {
                        Cotangent::Staged(cotangent) => {
                            Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                        }
                        Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    }
                }
                Self::Condition(condition) => condition.transpose(context, input_types, output_cotangents),
            }
        }
    }

    impl<Factor: Value<ArrayType>> FactorParameterizedOperation<ArrayType, Factor> for TestLinearOperation {
        type WithFactor<MappedFactor: Value<ArrayType>> = Self;

        fn try_map_factors<MappedFactor: Value<ArrayType>, MapFactorFn>(
            &self,
            _map_factor: &mut MapFactorFn,
        ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
        where
            MapFactorFn: FnMut(&Factor) -> Result<MappedFactor, ProgramError>,
        {
            Ok(self.clone())
        }
    }

    impl SupportsAdd<ArrayType> for TestLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl crate::operations::constants::SupportsZero<ArrayType> for TestLinearOperation {
        fn zero_operation(_type: ArrayType) -> Self {
            // The test linear operation enum doesn't include a Zero variant; the tests below never disconnect
            // primal inputs, so this constructor is unreachable in practice.
            Self::Scale { factor: TestValue::Number(0.0) }
        }
    }

    impl SupportsNeg<ArrayType> for TestLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsScale<ArrayType, TestValue> for TestLinearOperation {
        fn scale_operation(factor: TestValue) -> Self {
            Self::Scale { factor }
        }
    }

    impl From<ConditionOperation<TestValue, TestLinearOperation, ArrayType>> for TestLinearOperation {
        fn from(op: ConditionOperation<TestValue, TestLinearOperation, ArrayType>) -> Self {
            Self::Condition(Box::new(op))
        }
    }

    #[derive(Clone, Debug)]
    enum TestDifferentiableOperation {
        Zero(ArrayType),
        IsPositive,
        SubtractOne,
        Scale { factor: TestValue },
    }

    impl Display for TestDifferentiableOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestDifferentiableOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Zero(_) => "zero",
                Self::IsPositive => "is_positive",
                Self::SubtractOne => "subtract_one",
                Self::Scale { .. } => "scale",
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Zero(value_type) => {
                    ensure_input_count(0, input_types.len(), self.name())?;
                    Ok(vec![value_type.clone()])
                }
                Self::IsPositive => {
                    ensure_input_count(1, input_types.len(), self.name())?;
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::SubtractOne | Self::Scale { .. } => {
                    ensure_input_count(1, input_types.len(), self.name())?;
                    Ok(vec![input_types[0].clone()])
                }
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, _indentation: usize) -> std::fmt::Result {
            Display::fmt(self, formatter)
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestDifferentiableOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    Ok(vec![TestValue::zero(value_type)?])
                }
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: ("is_positive expected a numeric input").into() }.into()),
                },
                Self::SubtractOne => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(value - 1.0)]),
                    _ => Err(TypeError { message: ("subtract_one expected a numeric input").into() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: ("scale expected numeric inputs").into() }.into()),
                },
            }
        }
    }

    impl SupportsZero<ArrayType> for TestDifferentiableOperation {
        fn zero_operation(r#type: ArrayType) -> Self {
            Self::Zero(r#type)
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestDomain;

    impl Domain for TestDomain {
        type Type = ArrayType;
        type Value = TestValue;
        type Constant = TestValue;
        type Operation = TestDifferentiableOperation;
    }

    impl Context for TestDomain {
        fn lift(&self, constant: TestValue) -> Result<TestValue, ProgramError> {
            Ok(constant)
        }

        fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
            operation.interpret(inputs)
        }
    }

    impl DifferentiationContext for TestDomain {
        type Tangent = TestValue;
        type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> = TestLinearOperation;

        fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError> {
            let mut outputs =
                self.bind(<Self::Operation as SupportsZero<Self::Type>>::zero_operation(type_.clone()), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.pop().expect("zero operation produces exactly one output"))
        }
    }

    fn test_transposition_context<'transpose>(
        domain: &'transpose AbstractDomain<ArrayType, TestValue, TestLinearOperation>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, TestValue, TestLinearOperation>>>,
    ) -> AbstractTracingContext<'transpose, ArrayType, TestValue, TestLinearOperation> {
        AbstractTracingContext::new(domain, builder)
    }

    impl DifferentiableOperation<TestDomain> for TestDifferentiableOperation {
        fn jvp<'jvp>(
            &self,
            context: &mut TangentContext<'jvp, TestDomain>,
            inputs: &[JvpTracer<'jvp, TestDomain>],
        ) -> Result<Vec<JvpTracer<'jvp, TestDomain>>, ProgramError>
        where
            TestDomain: 'jvp,
        {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    let mut primals = context.bind_primal(Self::Zero(value_type.clone()), &[])?;
                    check_count!("output", primals, 1, ProgramError);
                    Ok(vec![JvpTracer::from_zero_tangent(primals.pop().expect("checked above"), value_type.clone())])
                }
                Self::IsPositive => Err(ControlFlowError::MissingTransformRule { transform: "is_positive jvp" }.into()),
                Self::SubtractOne => {
                    ensure_input_count(1, inputs.len(), self.name())?;
                    let primal_outputs = self.interpret(std::slice::from_ref(inputs[0].primal()))?;
                    Ok(vec![JvpTracer::new(primal_outputs[0].clone(), inputs[0].tangent().clone())])
                }
                Self::Scale { factor } => {
                    ensure_input_count(1, inputs.len(), self.name())?;
                    let primal_outputs = self.interpret(std::slice::from_ref(inputs[0].primal()))?;
                    let materialized_tangent = context.materialize_tangent(inputs[0].tangent().clone())?;
                    let tangent_outputs = context.stage_operation(
                        TestLinearOperation::Scale { factor: factor.clone() },
                        &[materialized_tangent],
                    )?;
                    check_count!("output", tangent_outputs, 1, ProgramError);
                    Ok(vec![JvpTracer::from_value(primal_outputs[0].clone(), tangent_outputs[0].clone())])
                }
            }
        }
    }

    fn add_one_branch() -> FlatProgram<TestValue, TestOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Add, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn subtract_one_branch() -> FlatProgram<TestValue, TestOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Sub, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn identity_array_branch() -> FlatProgram<TestValue, ArrayOperation<TestValue, ArrayType>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, ArrayOperation<TestValue, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_linear_identity_branch() -> FlatProgram<TestValue, TestLinearOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_scale_branch(factor: f64) -> FlatProgram<TestValue, TestDifferentiableOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestDifferentiableOperation::Scale { factor: TestValue::Number(factor) }, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_while_condition_branch() -> FlatProgram<TestValue, TestDifferentiableOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let counter = builder.add_input(ArrayType::scalar(DataType::F64));
        let _value = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestDifferentiableOperation::IsPositive, vec![counter]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_while_body_branch() -> FlatProgram<TestValue, TestDifferentiableOperation> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let counter = builder.add_input(ArrayType::scalar(DataType::F64));
        let value = builder.add_input(ArrayType::scalar(DataType::F64));
        let next_counter = builder.add_instruction(TestDifferentiableOperation::SubtractOne, vec![counter]).unwrap()[0];
        let next_value = builder
            .add_instruction(TestDifferentiableOperation::Scale { factor: TestValue::Number(2.0) }, vec![value])
            .unwrap()[0];
        builder
            .build(vec![next_counter, next_value], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    #[test]
    fn test_condition_interprets_true_and_false_branches() {
        let condition =
            ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), subtract_one_branch())
                .unwrap();

        assert_eq!(
            condition.interpret(&[TestValue::Bool(true), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(4.0)]),
        );
        assert_eq!(
            condition.interpret(&[TestValue::Bool(false), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(2.0)]),
        );
    }

    #[test]
    fn test_condition_program_rendering_includes_nested_branches() {
        let condition =
            ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), subtract_one_branch())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition [
                    predicate=runtime_input(type=bool[]),
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = add %0 %1
                        in (%2)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_rejects_branch_output_mismatch() {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::IsPositive, vec![input]).unwrap()[0];
        let bool_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), bool_branch).is_err());
    }

    #[test]
    fn test_while_interprets_until_condition_is_false() {
        let mut condition_builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, subtract_one_branch()).unwrap();

        assert_eq!(while_operation.interpret(&[TestValue::Number(3.0)]), Ok(vec![TestValue::Number(0.0)]),);
    }

    #[test]
    fn test_while_program_rendering_includes_condition_and_body() {
        let mut condition_builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, subtract_one_branch()).unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::While(Box::new(while_operation)), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [
                    condition={
                        lambda %0:f64[] .
                        let %1:bool[] = is_positive %0
                        in (%1)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_operation_condition_infers_output_types() {
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            identity_array_branch(),
            identity_array_branch(),
        )
        .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)]),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    fn expect_tangent_value<'jvp, T: crate::types::Type, V: crate::programs::Value<T>>(tangent: &Tangent<T, V>) -> V {
        match tangent {
            Tangent::Value(value) => value.clone(),
            Tangent::Zero(_) => {
                panic!("expected a concrete tangent value, not a symbolic zero")
            }
        }
    }

    #[test]
    fn test_generic_condition_jvp_uses_custom_operations() {
        let condition =
            ConditionOperation::with_captured_predicate(true, custom_scale_branch(2.0), custom_scale_branch(3.0))
                .unwrap();
        let domain = TestDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let mut context = TangentContext::new(&domain, builder.clone());
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = condition
            .jvp(&mut context, &[JvpTracer::from_value(TestValue::Number(4.0), tangent_input)])
            .unwrap();

        assert_eq!(outputs[0].primal(), &TestValue::Number(8.0));
        let tangent_output = expect_tangent_value(outputs[0].tangent()).atom_id().unwrap();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![tangent_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(tangent_program.interpret(vec![TestValue::Number(10.0)]), Ok(vec![TestValue::Number(20.0)]));
    }

    #[test]
    fn test_generic_while_jvp_propagates_tangents_through_iterations() {
        let while_operation = WhileOperation::new(custom_while_condition_branch(), custom_while_body_branch()).unwrap();
        let domain = TestDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let mut context = TangentContext::new(&domain, builder.clone());
        let counter_tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let value_tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = while_operation
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_value(TestValue::Number(3.0), counter_tangent_input),
                    JvpTracer::from_value(TestValue::Number(5.0), value_tangent_input),
                ],
            )
            .unwrap();

        assert_eq!(
            outputs.iter().map(|output| output.primal().clone()).collect::<Vec<_>>(),
            vec![TestValue::Number(0.0), TestValue::Number(40.0)],
        );
        let tangent_outputs = outputs
            .iter()
            .map(|output| expect_tangent_value(output.tangent()).atom_id().unwrap())
            .collect::<Vec<_>>();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                tangent_outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            tangent_program.interpret(vec![TestValue::Number(0.0), TestValue::Number(1.0)]),
            Ok(vec![TestValue::Number(0.0), TestValue::Number(8.0)]),
        );
    }

    #[test]
    fn test_linear_condition_transpose_rejects_runtime_predicates() {
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            custom_linear_identity_branch(),
            custom_linear_identity_branch(),
        )
        .unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, builder);
        let cotangent = context.tracer(cotangent_input, None);

        // Control-flow errors ride up as a `ProgramError::Custom` payload; recover the concrete error with
        // `downcast_custom`.
        let Err(error) =
            condition.transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(cotangent)])
        else {
            panic!("runtime-predicate condition transpose should be rejected");
        };
        assert_eq!(
            error.downcast_custom::<ControlFlowError>(),
            Some(&ControlFlowError::MissingTransformRule { transform: "runtime-predicate condition transpose" }),
        );
    }

    #[test]
    fn test_generic_linear_condition_transpose_uses_custom_operation() {
        let condition = ConditionOperation::with_captured_predicate(
            true,
            custom_linear_identity_branch(),
            custom_linear_identity_branch(),
        )
        .unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, builder.clone());
        let cotangent = context.tracer(cotangent_input, None);
        let outputs = condition
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(cotangent)])
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(!outputs[0].is_zero());
        let builder = builder.borrow();
        assert!(matches!(builder.instructions()[0].operation(), TestLinearOperation::Condition(_)));
    }
}
