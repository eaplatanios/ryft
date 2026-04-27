use std::fmt::{Debug, Display};

use half::{bf16, f16};
use thiserror::Error;

use crate::parameters::Parameterized;
use crate::tracing::{
    Instruction, InterpretableOperation, Operation, OperationFormatter, Program, Traceable, TracingError, Value,
};
use crate::tracing_v2::engines::StagingEngine;
use crate::tracing_v2::forward::Differentiable;
use crate::tracing_v2::linear::linearize_program;
use crate::tracing_v2::operations::constants::{Zero, ZeroLike};
use crate::tracing_v2::{
    DifferentiableEngine, DifferentiableStagingEngine, JvpContext, JvpTracer, PrimitiveOperation, Tracer,
};
use crate::types::{ArrayType, DataType, TypeError, Typed};

use super::{DifferentiableOperation, LinearOperation, SupportsAdd, TracedLinearizationCarrier, TranspositionContext};

/// Flat nested program shape used by control-flow operations.
pub type FlatProgram<V, O> = Program<ArrayType, V, O, Vec<V>, Vec<V>>;

/// Errors emitted by higher-order control-flow operations.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
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

/// Value-level predicate extraction used by interpreted control flow.
pub trait ControlFlowValue: Traceable<ArrayType> {
    /// Extracts a scalar boolean predicate from this value.
    fn control_flow_predicate(&self) -> Result<bool, TracingError>;
}

impl ControlFlowValue for bool {
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        Ok(*self)
    }
}

macro_rules! impl_non_predicate_control_flow_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ControlFlowValue for $ty {
                #[inline]
                fn control_flow_predicate(&self) -> Result<bool, TracingError> {
                    Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
                }
            }
        )*
    };
}

impl_non_predicate_control_flow_value!(i8, i16, i32, i64, u8, u16, u32, u64, bf16, f16, f32, f64);

impl<V: ControlFlowValue, T: Clone + crate::parameters::Parameter> ControlFlowValue for JvpTracer<V, T> {
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        self.primal.control_flow_predicate()
    }
}

impl<'engine, V, E> ControlFlowValue for Tracer<'engine, E>
where
    V: Traceable<ArrayType>,
    E: StagingEngine<Type = ArrayType, Value = V> + ?Sized,
{
    #[inline]
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        Err(ControlFlowError::MissingTransformRule { transform: "traced predicate extraction" }.into())
    }
}

/// Predicate source for a [`ConditionOperation`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConditionPredicate {
    /// The first operation input is the predicate.
    RuntimeInput(ArrayType),

    /// The predicate is captured in the operation and is not an operation input.
    Captured(bool),
}

/// Two-way conditional operation with nested true and false branch programs.
#[derive(Clone)]
pub struct ConditionOperation<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType> = PrimitiveOperation<V>> {
    /// Predicate source.
    predicate: ConditionPredicate,

    // TODO(eaplatanios): Why are we limiting our control flow operations to flat programs only?
    /// Program evaluated when the predicate is true.
    true_branch: FlatProgram<V, O>,

    /// Program evaluated when the predicate is false.
    false_branch: FlatProgram<V, O>,
}

/// While-loop operation with nested condition and body programs over the same loop-carried state.
#[derive(Clone)]
pub struct WhileOperation<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType> = PrimitiveOperation<V>> {
    /// Program that maps the current loop state to one scalar boolean predicate.
    condition: FlatProgram<V, O>,

    /// Program that maps the current loop state to the next loop state.
    body: FlatProgram<V, O>,
}

/// Returns the flat input types of a nested control-flow program.
pub fn flat_program_input_types<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>>(
    program: &FlatProgram<V, O>,
) -> Vec<ArrayType> {
    program.inputs().map(|input| input.r#type().into_owned()).collect()
}

/// Returns the flat output types of a nested control-flow program.
pub fn flat_program_output_types<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>>(
    program: &FlatProgram<V, O>,
) -> Vec<ArrayType> {
    program.outputs().map(|output| output.r#type().into_owned()).collect()
}

/// Returns the canonical scalar boolean array type used by control predicates.
fn scalar_bool_type() -> ArrayType {
    ArrayType::scalar(DataType::Boolean)
}

/// Validates that `predicate_type` is exactly the canonical scalar boolean type.
fn ensure_scalar_bool_type(predicate_type: &ArrayType) -> Result<(), TypeError> {
    let expected = scalar_bool_type();
    if predicate_type != &expected {
        return Err(TypeError {
            message: format!("control-flow predicate type must be {expected}, but got {predicate_type}"),
        });
    }
    Ok(())
}

/// Validates that two flat type signatures are identical.
fn ensure_types_match(context: &'static str, left: &[ArrayType], right: &[ArrayType]) -> Result<(), TypeError> {
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
fn ensure_input_count(expected: usize, got: usize, operation: &'static str) -> Result<(), TypeError> {
    if expected != got {
        return Err(TypeError { message: format!("{operation} expected {expected} input type(s) but got {got}") });
    }
    Ok(())
}

/// Replays one staged linear program by inlining its instructions into an existing linear builder
/// owned by a [`JvpContext`].
fn replay_linear_program_on_atoms<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>>(
    context: &JvpContext<'_, V, O>,
    program: &FlatProgram<V, O>,
    inputs: &[crate::tracing::AtomId],
) -> Result<Vec<crate::tracing::AtomId>, TracingError> {
    if inputs.len() != program.input_ids.len() {
        return Err(TracingError::InvalidInputCount { expected: program.input_ids.len(), got: inputs.len() });
    }
    if inputs.is_empty() && program.output_ids.is_empty() {
        return Ok(Vec::new());
    }

    let builder = context.builder().clone();
    let mut values: Vec<Option<crate::tracing::AtomId>> = vec![None; program.atoms.len()];
    for (input_id, input) in program.input_ids.iter().copied().zip(inputs.iter().copied()) {
        values[input_id.index] = Some(input);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        if let crate::tracing::Atom::Constant(value) = atom {
            let atom = builder.borrow_mut().add_constant(value.clone());
            values[atom_index] = Some(atom);
        }
    }

    for instruction in program.instructions.iter() {
        let instruction_inputs = instruction
            .inputs
            .iter()
            .map(|input| values[input.index].ok_or(TracingError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = context.apply_operation(
            instruction_inputs.as_slice(),
            instruction.operation.clone(),
            instruction.outputs.len(),
        )?;
        for (output, value) in instruction.outputs.iter().copied().zip(outputs) {
            values[output.index] = Some(value);
        }
    }

    program
        .output_ids
        .iter()
        .map(|output| values[output.index].ok_or(TracingError::UnboundAtomId { id: *output }))
        .collect()
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> ConditionOperation<V, O> {
    /// Creates a condition whose predicate is supplied as the first operation input.
    pub fn new(
        predicate_type: ArrayType,
        true_branch: FlatProgram<V, O>,
        false_branch: FlatProgram<V, O>,
    ) -> Result<Self, TypeError> {
        ensure_scalar_bool_type(&predicate_type)?;
        Self::from_parts(ConditionPredicate::RuntimeInput(predicate_type), true_branch, false_branch)
    }

    /// Creates a condition whose predicate is captured in the operation.
    pub fn with_captured_predicate(
        predicate: bool,
        true_branch: FlatProgram<V, O>,
        false_branch: FlatProgram<V, O>,
    ) -> Result<Self, TypeError> {
        Self::from_parts(ConditionPredicate::Captured(predicate), true_branch, false_branch)
    }

    /// Creates a condition after validating branch signatures.
    fn from_parts(
        predicate: ConditionPredicate,
        true_branch: FlatProgram<V, O>,
        false_branch: FlatProgram<V, O>,
    ) -> Result<Self, TypeError> {
        if let ConditionPredicate::RuntimeInput(predicate_type) = &predicate {
            ensure_scalar_bool_type(predicate_type)?;
        }
        let input_types = flat_program_input_types(&true_branch);
        ensure_types_match("condition branch input", &input_types, &flat_program_input_types(&false_branch))?;
        let output_types = flat_program_output_types(&true_branch);
        ensure_types_match("condition branch output", &output_types, &flat_program_output_types(&false_branch))?;
        Ok(Self { predicate, true_branch, false_branch })
    }

    /// Returns the predicate source used by this condition.
    #[inline]
    pub fn predicate(&self) -> &ConditionPredicate {
        &self.predicate
    }

    /// Returns the true branch program.
    #[inline]
    pub fn true_branch(&self) -> &FlatProgram<V, O> {
        &self.true_branch
    }

    /// Returns the false branch program.
    #[inline]
    pub fn false_branch(&self) -> &FlatProgram<V, O> {
        &self.false_branch
    }

    /// Returns the operand input types consumed by both branches.
    #[inline]
    pub fn input_types(&self) -> Vec<ArrayType> {
        flat_program_input_types(&self.true_branch)
    }

    /// Returns the output types produced by both branches.
    #[inline]
    pub fn output_types(&self) -> Vec<ArrayType> {
        flat_program_output_types(&self.true_branch)
    }

    /// Returns the branch selected by `predicate`.
    fn selected_branch(&self, predicate: bool) -> &FlatProgram<V, O> {
        if predicate { &self.true_branch } else { &self.false_branch }
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Debug for ConditionOperation<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Condition").field("predicate", &self.predicate).finish()
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Display for ConditionOperation<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "condition")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation<ArrayType> for ConditionOperation<V, O> {
    fn name(&self) -> &'static str {
        "condition"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let operand_input_types = self.input_types();
        let operand_start = match &self.predicate {
            ConditionPredicate::RuntimeInput(predicate_type) => {
                ensure_input_count(operand_input_types.len() + 1, input_types.len(), self.name())?;
                ensure_scalar_bool_type(&input_types[0])?;
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
                ensure_input_count(operand_input_types.len(), input_types.len(), self.name())?;
                0
            }
        };
        ensure_types_match("condition operand", &operand_input_types, &input_types[operand_start..])?;
        Ok(self.output_types())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            match &self.predicate {
                ConditionPredicate::RuntimeInput(predicate_type) => {
                    operation.field("predicate", format_args!("runtime_input(type={predicate_type})"))?;
                }
                ConditionPredicate::Captured(predicate) => {
                    operation.field("predicate", format_args!("captured({predicate})"))?;
                }
            }
            operation.program("true_branch", self.true_branch())?;
            operation.program("false_branch", self.false_branch())
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for ConditionOperation<V, O>
where
    V: ControlFlowValue,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let (predicate, operands) = match self.predicate {
            ConditionPredicate::RuntimeInput(_) => (inputs[0].control_flow_predicate()?, &inputs[1..]),
            ConditionPredicate::Captured(predicate) => (predicate, inputs),
        };
        self.selected_branch(predicate).interpret(operands.to_vec())
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for ConditionOperation<V, O>
where
    V: Traceable<ArrayType>,
    O: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, O>
        + SupportsAdd<ArrayType, V>
        + crate::tracing_v2::operations::SupportsZero<ArrayType, V>
        + From<ConditionOperation<V, O>>,
{
    fn transpose(
        &self,
        context: &mut TranspositionContext<'_, ArrayType, V, O>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
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
        if output_cotangents.iter().all(Option::is_none) {
            return Ok(vec![None; self.input_types().len()]);
        }
        let transposed_condition = ConditionOperation::with_captured_predicate(
            predicate,
            context.transpose_nested_program(&self.true_branch)?,
            context.transpose_nested_program(&self.false_branch)?,
        )?;
        let materialized = output_cotangents
            .iter()
            .zip(self.output_types().iter())
            .map(|(cotangent, output_type)| stage_optional_cotangent(context, *cotangent, output_type))
            .collect::<Vec<_>>();
        let input_count = self.input_types().len();
        let cotangents =
            context.apply_operation(materialized.as_slice(), O::from(transposed_condition), input_count)?;
        Ok(cotangents.into_iter().map(Some).collect())
    }
}

/// Returns a concrete cotangent atom for `cotangent`, staging a typed `Zero` op when the cotangent
/// is structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
fn stage_optional_cotangent<V, O>(
    context: &TranspositionContext<'_, ArrayType, V, O>,
    cotangent: Option<crate::tracing::AtomId>,
    output_type: &ArrayType,
) -> crate::tracing::AtomId
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType> + crate::tracing_v2::operations::SupportsZero<ArrayType, V>,
{
    if let Some(atom) = cotangent {
        return atom;
    }
    let builder = context.builder();
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow.instructions.push(Instruction {
        operation: <O as crate::tracing_v2::operations::SupportsZero<ArrayType, V>>::zero_operation(
            output_type.clone(),
        ),
        inputs: vec![],
        outputs: vec![output],
    });
    output
}

impl<V, E, O> DifferentiableOperation<E> for ConditionOperation<V, O>
where
    V: ControlFlowValue + ZeroLike + Differentiable<ArrayType, Tangent = V> + Zero<ArrayType>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    O: Clone + DifferentiableOperation<E> + InterpretableOperation<ArrayType, V> + Operation<ArrayType>,
    <E as DifferentiableEngine>::LinearOperation: Operation<ArrayType>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut crate::tracing_v2::JvpContext<'_, V, E::LinearOperation>,
        inputs: &[JvpTracer<V, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<V, crate::tracing::AtomId>>, TracingError> {
        let operand_count = self.input_types().len();
        let expected_count = operand_count + usize::from(matches!(self.predicate, ConditionPredicate::RuntimeInput(_)));
        if inputs.len() != expected_count {
            return Err(TracingError::InvalidInputCount { expected: expected_count, got: inputs.len() });
        }
        let (predicate, operands) = match self.predicate {
            ConditionPredicate::RuntimeInput(_) => (inputs[0].primal.control_flow_predicate()?, &inputs[1..]),
            ConditionPredicate::Captured(predicate) => (predicate, inputs),
        };
        let primal_operands = operands.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_operands = operands.iter().map(|input| input.tangent).collect::<Vec<_>>();
        let branch = self.selected_branch(predicate);
        let primal_outputs = branch.interpret(primal_operands.clone())?;
        let pushforward = linearize_program(engine, branch, primal_operands)?;
        let tangent_outputs = replay_linear_program_on_atoms(context, &pushforward, tangent_operands.as_slice())?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect())
    }
}

/// JVP rule for `ConditionOperation` under
/// [`TracingEngine`](crate::tracing_v2::TracingEngine).
///
/// Predicate extraction does not work at trace time (the wrapper engine's `Value` is `Tracer`,
/// whose `control_flow_predicate` always errors), so this impl reports
/// [`ControlFlowError::MissingTransformRule`] for any traced JVP attempt.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing_v2::TracingEngine<'engine, EInner>>
    for ConditionOperation<V, EInner::Operation>
where
    V: ControlFlowValue + Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    EInner: DifferentiableStagingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    EInner::Operation: TracedLinearizationCarrier<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn jvp(
        &self,
        _engine: &crate::tracing_v2::TracingEngine<'engine, EInner>,
        _context: &mut crate::tracing_v2::JvpContext<
            '_,
            crate::tracing_v2::Tracer<'engine, EInner>,
            <EInner as DifferentiableStagingEngine>::LinearOperation<'engine>,
        >,
        _inputs: &[JvpTracer<crate::tracing_v2::Tracer<'engine, EInner>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<crate::tracing_v2::Tracer<'engine, EInner>, crate::tracing::AtomId>>, TracingError> {
        Err(ControlFlowError::MissingTransformRule { transform: "linearization engine traced JVP" }.into())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> WhileOperation<V, O> {
    /// Creates a while loop from a condition program and a body program.
    pub fn new(condition: FlatProgram<V, O>, body: FlatProgram<V, O>) -> Result<Self, TypeError> {
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
        ensure_scalar_bool_type(&condition_output_types[0])?;
        ensure_types_match("while body output", &state_types, &flat_program_output_types(&body))?;
        Ok(Self { condition, body })
    }

    /// Returns the condition program.
    #[inline]
    pub fn condition(&self) -> &FlatProgram<V, O> {
        &self.condition
    }

    /// Returns the body program.
    #[inline]
    pub fn body(&self) -> &FlatProgram<V, O> {
        &self.body
    }

    /// Returns the loop-carried state types.
    #[inline]
    pub fn state_types(&self) -> Vec<ArrayType> {
        flat_program_input_types(&self.body)
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Debug for WhileOperation<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("While").finish()
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Display for WhileOperation<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "while")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation<ArrayType> for WhileOperation<V, O> {
    fn name(&self) -> &'static str {
        "while"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let state_types = self.state_types();
        ensure_input_count(state_types.len(), input_types.len(), self.name())?;
        ensure_types_match("while input", &state_types, input_types)?;
        Ok(state_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.program("condition", self.condition())?;
            operation.program("body", self.body())
        })
    }
}

impl<V, O> InterpretableOperation<ArrayType, V> for WhileOperation<V, O>
where
    V: ControlFlowValue,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let mut state = inputs.to_vec();
        loop {
            let condition_outputs = self.condition.interpret(state.clone())?;
            if condition_outputs.len() != 1 {
                return Err(TracingError::InvalidOutputCount { expected: 1, got: condition_outputs.len() });
            }
            if !condition_outputs[0].control_flow_predicate()? {
                return Ok(state);
            }
            state = self.body.interpret(state)?;
            if state.len() != self.state_types().len() {
                return Err(TracingError::InvalidOutputCount { expected: self.state_types().len(), got: state.len() });
            }
        }
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for WhileOperation<V, O>
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, ArrayType, V, O>,
        _output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        Err(ControlFlowError::MissingTransformRule { transform: "while transpose" }.into())
    }
}

/// JVP rule for `WhileOperation` under
/// [`TracingEngine`](crate::tracing_v2::TracingEngine). See the matching
/// [`ConditionOperation`] impl for rationale; predicate extraction does not work at trace time.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing_v2::TracingEngine<'engine, EInner>>
    for WhileOperation<V, EInner::Operation>
where
    V: ControlFlowValue + Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    EInner: DifferentiableStagingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    EInner::Operation: TracedLinearizationCarrier<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn jvp(
        &self,
        _engine: &crate::tracing_v2::TracingEngine<'engine, EInner>,
        _context: &mut crate::tracing_v2::JvpContext<
            '_,
            crate::tracing_v2::Tracer<'engine, EInner>,
            <EInner as DifferentiableStagingEngine>::LinearOperation<'engine>,
        >,
        _inputs: &[JvpTracer<crate::tracing_v2::Tracer<'engine, EInner>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<crate::tracing_v2::Tracer<'engine, EInner>, crate::tracing::AtomId>>, TracingError> {
        Err(ControlFlowError::MissingTransformRule { transform: "linearization engine traced JVP" }.into())
    }
}

impl<V, E, O> DifferentiableOperation<E> for WhileOperation<V, O>
where
    V: ControlFlowValue + ZeroLike + Differentiable<ArrayType, Tangent = V> + Zero<ArrayType>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    O: Clone + DifferentiableOperation<E> + InterpretableOperation<ArrayType, V> + Operation<ArrayType>,
    <E as DifferentiableEngine>::LinearOperation: Operation<ArrayType>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut crate::tracing_v2::JvpContext<'_, V, E::LinearOperation>,
        inputs: &[JvpTracer<V, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<V, crate::tracing::AtomId>>, TracingError> {
        let state_count = self.state_types().len();
        if inputs.len() != state_count {
            return Err(TracingError::InvalidInputCount { expected: state_count, got: inputs.len() });
        }
        let mut state_primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let mut state_tangents = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();

        loop {
            let condition_outputs = self.condition.interpret(state_primals.clone())?;
            if condition_outputs.len() != 1 {
                return Err(TracingError::InvalidOutputCount { expected: 1, got: condition_outputs.len() });
            }
            if !condition_outputs[0].control_flow_predicate()? {
                return Ok(state_primals
                    .into_iter()
                    .zip(state_tangents)
                    .map(|(primal, tangent)| JvpTracer { primal, tangent })
                    .collect());
            }

            let pushforward = linearize_program(engine, self.body(), state_primals.clone())?;
            let next_primals = self.body.interpret(state_primals)?;
            let next_tangents = replay_linear_program_on_atoms(context, &pushforward, state_tangents.as_slice())?;
            if next_primals.len() != state_count {
                return Err(TracingError::InvalidOutputCount { expected: state_count, got: next_primals.len() });
            }
            if next_tangents.len() != state_count {
                return Err(TracingError::InvalidOutputCount { expected: state_count, got: next_tangents.len() });
            }
            state_primals = next_primals;
            state_tangents = next_tangents;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::parameters::{Parameter, Placeholder};
    use crate::tracing::{ProgramBuilder, Traceable, Value};
    use crate::tracing_v2::engines::{Engine, ScalarEngine};
    use crate::tracing_v2::operations::constants::{One, OneLike, Zero};
    use crate::tracing_v2::operations::{SupportsAdd, SupportsNeg, SupportsScale};
    use crate::tracing_v2::{Differentiable, LinearPrimitiveOperation};
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

    impl Traceable<ArrayType> for TestValue {}

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
        fn zero(value_type: &ArrayType) -> Result<Self, TracingError> {
            match value_type.data_type {
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
        fn one(value_type: &ArrayType) -> Result<Self, TracingError> {
            if value_type.rank() != 0 {
                return Err(crate::tracing_v2::DifferentiationError::NonScalarGradientOutput {
                    output_type: value_type.clone(),
                }
                .into());
            }
            match value_type.data_type {
                DataType::Boolean => Ok(Self::Bool(true)),
                DataType::F64 => Ok(Self::Number(1.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize one for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl Differentiable<ArrayType> for TestValue {
        type Tangent = Self;
    }

    impl ControlFlowValue for TestValue {
        fn control_flow_predicate(&self) -> Result<bool, TracingError> {
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
        Condition(Box<ConditionOperation<TestValue, TestOperation>>),
        While(Box<WhileOperation<TestValue, TestOperation>>),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestOperation {
        fn name(&self) -> &'static str {
            match self {
                Self::Add => "add",
                Self::Sub => "sub",
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
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: "add expected numeric inputs".to_string() }.into()),
                },
                Self::Sub => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left - right)]),
                    _ => Err(TypeError { message: "sub expected numeric inputs".to_string() }.into()),
                },
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: "is_positive expected a numeric input".to_string() }.into()),
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
        Condition(Box<ConditionOperation<TestValue, TestLinearOperation>>),
    }

    impl Display for TestLinearOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestLinearOperation {
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
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: "linear add expected numeric inputs".to_string() }.into()),
                },
                Self::Neg => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(-value)]),
                    _ => Err(TypeError { message: "linear neg expected a numeric input".to_string() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: "linear scale expected numeric inputs".to_string() }.into()),
                },
                Self::Condition(condition) => condition.interpret(inputs),
            }
        }
    }

    impl LinearOperation<ArrayType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose(
            &self,
            context: &mut TranspositionContext<'_, ArrayType, TestValue, TestLinearOperation>,
            output_cotangents: &[Option<crate::tracing::AtomId>],
        ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
            match self {
                Self::Add => {
                    ensure_input_count(1, output_cotangents.len(), self.name())?;
                    Ok(vec![output_cotangents[0], output_cotangents[0]])
                }
                Self::Neg => {
                    ensure_input_count(1, output_cotangents.len(), self.name())?;
                    match output_cotangents[0] {
                        Some(atom) => Ok(vec![Some(
                            context
                                .apply_operation(&[atom], Self::Neg, 1)?
                                .into_iter()
                                .next()
                                .expect("neg transpose should produce one cotangent contribution"),
                        )]),
                        None => Ok(vec![None]),
                    }
                }
                Self::Scale { factor } => {
                    ensure_input_count(1, output_cotangents.len(), self.name())?;
                    match output_cotangents[0] {
                        Some(atom) => Ok(vec![Some(
                            context
                                .apply_operation(&[atom], Self::Scale { factor: factor.clone() }, 1)?
                                .into_iter()
                                .next()
                                .expect("scale transpose should produce one cotangent contribution"),
                        )]),
                        None => Ok(vec![None]),
                    }
                }
                Self::Condition(condition) => condition.transpose(context, output_cotangents),
            }
        }
    }

    impl SupportsAdd<ArrayType, TestValue> for TestLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl crate::tracing_v2::operations::SupportsZero<ArrayType, TestValue> for TestLinearOperation {
        fn zero_operation(_type: ArrayType) -> Self {
            // Test linear carrier doesn't include a Zero variant; the tests below never disconnect
            // primal inputs, so this constructor is unreachable in practice.
            Self::Scale { factor: TestValue::Number(0.0) }
        }
    }

    impl SupportsNeg<ArrayType, TestValue> for TestLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsScale<ArrayType, TestValue> for TestLinearOperation {
        fn scale_operation(factor: TestValue) -> Self {
            Self::Scale { factor }
        }
    }

    impl From<ConditionOperation<TestValue, TestLinearOperation>> for TestLinearOperation {
        fn from(op: ConditionOperation<TestValue, TestLinearOperation>) -> Self {
            Self::Condition(Box::new(op))
        }
    }

    #[derive(Clone, Debug)]
    enum TestDifferentiableOperation {
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
        fn name(&self) -> &'static str {
            match self {
                Self::IsPositive => "is_positive",
                Self::SubtractOne => "subtract_one",
                Self::Scale { .. } => "scale",
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
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

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            let _ = indentation;
            Display::fmt(self, formatter)
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestDifferentiableOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
            match self {
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: "is_positive expected a numeric input".to_string() }.into()),
                },
                Self::SubtractOne => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(value - 1.0)]),
                    _ => Err(TypeError { message: "subtract_one expected a numeric input".to_string() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: "scale expected numeric inputs".to_string() }.into()),
                },
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TestEngine;

    impl Engine for TestEngine {
        type Type = ArrayType;
        type Value = TestValue;

        fn zero(&self, r#type: &ArrayType) -> Result<TestValue, TracingError> {
            if r#type.data_type == DataType::Boolean { Ok(TestValue::Bool(false)) } else { Ok(TestValue::Number(0.0)) }
        }

        fn one(&self, _type: &ArrayType) -> Result<TestValue, TracingError> {
            Ok(TestValue::Number(1.0))
        }
    }

    impl StagingEngine for TestEngine {
        type Operation = TestDifferentiableOperation;
    }

    impl DifferentiableEngine for TestEngine {
        type DifferentiableOperation = TestDifferentiableOperation;
        type LinearOperation = TestLinearOperation;
    }

    fn test_transposition_context(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, TestValue, TestLinearOperation>>>,
    ) -> TranspositionContext<'static, ArrayType, TestValue, TestLinearOperation> {
        TranspositionContext::new(builder)
    }

    impl DifferentiableOperation<TestEngine> for TestDifferentiableOperation {
        fn jvp(
            &self,
            _engine: &TestEngine,
            context: &mut JvpContext<'_, TestValue, TestLinearOperation>,
            inputs: &[JvpTracer<TestValue, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<TestValue, crate::tracing::AtomId>>, TracingError> {
            match self {
                Self::IsPositive => Err(ControlFlowError::MissingTransformRule { transform: "is_positive jvp" }.into()),
                Self::SubtractOne => {
                    ensure_input_count(1, inputs.len(), self.name())?;
                    let primal_outputs = self.interpret(std::slice::from_ref(&inputs[0].primal))?;
                    Ok(vec![JvpTracer { primal: primal_outputs[0].clone(), tangent: inputs[0].tangent }])
                }
                Self::Scale { factor } => {
                    ensure_input_count(1, inputs.len(), self.name())?;
                    let primal_outputs = self.interpret(std::slice::from_ref(&inputs[0].primal))?;
                    let tangent = context
                        .apply_operation(
                            &[inputs[0].tangent],
                            TestLinearOperation::Scale { factor: factor.clone() },
                            1,
                        )?
                        .into_iter()
                        .next()
                        .expect("test scale jvp should produce one tangent");
                    Ok(vec![JvpTracer { primal: primal_outputs[0].clone(), tangent }])
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

    fn identity_primitive_branch() -> FlatProgram<TestValue, PrimitiveOperation<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, PrimitiveOperation<TestValue>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn scalar_scale_branch(factor: f64) -> FlatProgram<f64, PrimitiveOperation<f64>> {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(PrimitiveOperation::Scale { factor }, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
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
    fn test_primitive_carrier_condition_interprets_captured_predicate() {
        let condition =
            ConditionOperation::with_captured_predicate(false, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let operation = PrimitiveOperation::Condition(Box::new(condition));

        assert_eq!(operation.interpret(&[4.0]), Ok(vec![12.0]));
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
    fn test_primitive_carrier_condition_infers_output_types() {
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            identity_primitive_branch(),
            identity_primitive_branch(),
        )
        .unwrap();
        let operation = PrimitiveOperation::Condition(Box::new(condition));

        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)]),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    #[test]
    fn test_condition_jvp_uses_selected_captured_branch() {
        let condition =
            ConditionOperation::with_captured_predicate(true, scalar_scale_branch(2.0), scalar_scale_branch(3.0))
                .unwrap();
        let engine = ScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new()));
        let tangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = JvpContext::new(builder.clone());
        let outputs =
            condition.jvp(&engine, &mut context, &[JvpTracer { primal: 4.0, tangent: tangent_input }]).unwrap();

        assert_eq!(outputs[0].primal, 8.0);
        let tangent_output = outputs[0].tangent;
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder.build::<f64, f64>(vec![tangent_output], Placeholder, Placeholder).unwrap();
        assert_eq!(tangent_program.interpret(10.0), Ok(20.0));
    }

    #[test]
    fn test_generic_condition_jvp_uses_custom_carriers() {
        let condition =
            ConditionOperation::with_captured_predicate(true, custom_scale_branch(2.0), custom_scale_branch(3.0))
                .unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let tangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = JvpContext::new(builder.clone());
        let outputs = condition
            .jvp(&TestEngine, &mut context, &[JvpTracer { primal: TestValue::Number(4.0), tangent: tangent_input }])
            .unwrap();

        assert_eq!(outputs[0].primal, TestValue::Number(8.0));
        let tangent_output = outputs[0].tangent;
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
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let counter_tangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let value_tangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = JvpContext::new(builder.clone());
        let outputs = while_operation
            .jvp(
                &TestEngine,
                &mut context,
                &[
                    JvpTracer { primal: TestValue::Number(3.0), tangent: counter_tangent_input },
                    JvpTracer { primal: TestValue::Number(5.0), tangent: value_tangent_input },
                ],
            )
            .unwrap();

        assert_eq!(
            outputs.iter().map(|output| output.primal.clone()).collect::<Vec<_>>(),
            vec![TestValue::Number(0.0), TestValue::Number(40.0)],
        );
        let tangent_outputs = outputs.iter().map(|output| output.tangent).collect::<Vec<_>>();
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
        let mut context = test_transposition_context(builder);

        assert!(matches!(
            condition.transpose(&mut context, &[Some(cotangent_input)]),
            Err(TracingError::ControlFlow(ControlFlowError::MissingTransformRule {
                transform: "runtime-predicate condition transpose",
            })),
        ));
    }

    #[test]
    fn test_generic_linear_condition_transpose_uses_custom_carrier() {
        let condition = ConditionOperation::with_captured_predicate(
            true,
            custom_linear_identity_branch(),
            custom_linear_identity_branch(),
        )
        .unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = test_transposition_context(builder.clone());
        let outputs = condition.transpose(&mut context, &[Some(cotangent_input)]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_some());
        let builder = builder.borrow();
        assert!(matches!(builder.instructions[0].operation, TestLinearOperation::Condition(_)));
    }
}
