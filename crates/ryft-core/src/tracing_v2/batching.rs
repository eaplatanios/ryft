use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, Scale, ScaleOperation, SubOperation,
};
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::domains::Tracer;
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::{
    ArrayOperation, ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, LinearArrayOperation,
    MatrixOps, NoOperationExtension, WhileOperation,
};
use crate::types::{ArrayType, Size, Typed};

/// Errors emitted by explicit batching and `vmap` helpers.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum BatchingError {
    /// No mapped array leaves were provided and no explicit axis size is available.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// Structured lanes did not share the same [`Parameterized`] shape in the reference fallback.
    #[error("mismatched parameter structures across batch lanes")]
    MismatchedParameterStructures,

    /// Different batched leaves disagreed on the mapped axis size.
    #[error("mismatched batch sizes across batched leaves")]
    MismatchedBatchSize,

    /// A primitive has no packed-array batching rule.
    #[error("missing batching rule for operation '{operation}'")]
    MissingBatchingRule {
        /// Name of the operation that could not be batched.
        operation: String,
    },

    /// A batching rule encountered batch axes it does not yet know how to align.
    #[error("{message}")]
    UnsupportedBatchAxisAlignment {
        /// Human-readable explanation of the unsupported axis placement.
        message: String,
    },

    /// A public `vmap` output did not carry the mapped axis.
    #[error("{message}")]
    UnbatchedOutput {
        /// Human-readable explanation of the output mismatch.
        message: String,
    },

    /// A mapped axis has dynamic size and no explicit axis size was provided.
    #[error("batch axis {axis} of array type {type_} has dynamic size")]
    DynamicBatchAxis {
        /// Physical array type containing the mapped axis.
        type_: ArrayType,

        /// Mapped axis.
        axis: usize,
    },

    /// A mapped axis is outside the rank of its array type.
    #[error("batch axis {axis} is out of bounds for array type {type_}")]
    InvalidBatchAxis {
        /// Physical array type.
        type_: ArrayType,

        /// Invalid axis.
        axis: usize,
    },

    /// Wrapper around parameter-lifting failures from the [`Parameterized`] infrastructure.
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}

/// Packed array value carrying lane metadata for one batching transform.
///
/// [`ArrayBatch`] is the production batching carrier for `tracing_v2`: its [`ArrayType`] is the
/// physical type of `value`, so it includes the mapped lane dimension when [`ArrayBatch::batch_axis`]
/// is `Some`. The logical per-lane type is derived by removing that dimension.
///
/// A `None` batch axis is an explicit lane-uniform state. It means the value does not contain a
/// physical dimension for the current batch lanes and should be interpreted as the same value for
/// every lane. For example, a traced constant in `vmap(|x| x + 1)` is represented with
/// `batch_axis == None`, while `x` carries the mapped input axis. Runtime control-flow predicates
/// also require `None` today because a single predicate can select one branch for all lanes, while
/// a lane-varying predicate would need a dedicated batching rule. `None` is not limited to
/// rank-0 values: any shaped constant or operand can be lane-uniform when none of its physical
/// dimensions indexes the current lanes.
#[derive(Clone, Debug, Parameter, PartialEq)]
pub struct ArrayBatch<V: Typed<ArrayType> + Parameter> {
    /// Physical array type of `value`.
    r#type: ArrayType,

    /// Packed array value.
    value: V,

    /// Axis in `type_` and `value` that represents the mapped batch dimension, or `None` when
    /// `value` is uniform across the current batch lanes.
    batch_axis: Option<usize>,
}

impl<V: Typed<ArrayType> + Parameter> ArrayBatch<V> {
    /// Creates a packed array batch from explicit physical metadata.
    ///
    /// # Parameters
    ///
    ///   - `type_`: Physical type of `value`. This type includes `batch_axis` when present.
    ///   - `value`: Physical array value.
    ///   - `batch_axis`: Mapped axis in `type_` and `value`, or `None` when `value` is shared
    ///     uniformly across lanes.
    pub fn new(type_: ArrayType, value: V, batch_axis: Option<usize>) -> Result<Self, TracingError> {
        if let Some(axis) = batch_axis
            && axis >= type_.rank()
        {
            return Err(BatchingError::InvalidBatchAxis { type_, axis }.into());
        }
        Ok(Self { r#type: type_, value, batch_axis })
    }

    /// Wraps a value that already contains a mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value.
    ///   - `batch_axis`: Mapped axis in `value`.
    pub fn mapped(value: V, batch_axis: usize) -> Result<Self, TracingError>
    where
        V: Traceable<ArrayType>,
    {
        Self::new(value.r#type().into_owned(), value, Some(batch_axis))
    }

    /// Wraps a value that is uniform across the current batch lanes.
    pub fn unbatched(value: V) -> Self
    where
        V: Traceable<ArrayType>,
    {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }

    /// Returns the mapped axis, if the physical value carries one.
    #[inline]
    pub fn batch_axis(&self) -> Option<usize> {
        self.batch_axis
    }

    /// Returns the physical value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes `self` and returns the physical value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the static mapped axis size, if this value is batched.
    pub fn axis_size(&self) -> Result<Option<usize>, TracingError> {
        let Some(axis) = self.batch_axis else {
            return Ok(None);
        };
        let Some(size) = self.r#type.dimension(axis as isize).value() else {
            return Err(BatchingError::DynamicBatchAxis { type_: self.r#type.clone(), axis }.into());
        };
        Ok(Some(size))
    }

    /// Returns the scalar-body type obtained by removing the mapped axis.
    pub fn logical_type(&self) -> Result<ArrayType, TracingError> {
        let Some(axis) = self.batch_axis else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }
}

impl<V: Typed<ArrayType> + Parameter> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display + Typed<ArrayType> + Parameter> Display for ArrayBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "batch(value={}, axis={:?})", self.value, self.batch_axis)
    }
}

impl<V: Traceable<ArrayType>> Traceable<ArrayType> for ArrayBatch<V> {}

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {}

impl<V: Traceable<ArrayType> + ControlFlowValue> ControlFlowValue for ArrayBatch<V> {
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        if self.batch_axis.is_some() {
            return Err(ControlFlowError::MissingTransformRule { transform: "batched predicate control flow" }.into());
        }
        self.value.control_flow_predicate()
    }
}

/// Packed-array batching rule for one staged operation.
///
/// Implementations receive physical array values plus mapped-axis metadata and must return physical
/// array values with correct output axis metadata. Missing rules should return
/// [`BatchingError::MissingBatchingRule`] rather than replaying per lane.
pub trait BatchableOperation<V: Traceable<ArrayType>>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Physical input values and their mapped-axis metadata.
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>;
}

impl<V: Traceable<ArrayType>> BatchableOperation<V> for NoOperationExtension {
    fn batch(&self, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match *self {}
    }
}

fn validate_axis_size<V: Typed<ArrayType> + Parameter>(
    expected_axis: &mut Option<usize>,
    expected_size: &mut Option<usize>,
    value: &ArrayBatch<V>,
) -> Result<(), TracingError> {
    let Some(axis) = value.batch_axis else {
        return Ok(());
    };
    match expected_axis {
        Some(existing_axis) if *existing_axis != axis => {
            return Err(BatchingError::UnsupportedBatchAxisAlignment {
                message: format!("cannot align batch axis {axis} with existing batch axis {existing_axis}"),
            }
            .into());
        }
        Some(_) => {}
        None => *expected_axis = Some(axis),
    }

    let size = value.axis_size()?.ok_or(BatchingError::EmptyBatch)?;
    match expected_size {
        Some(existing_size) if *existing_size != size => Err(BatchingError::MismatchedBatchSize.into()),
        Some(_) => Ok(()),
        None => {
            *expected_size = Some(size);
            Ok(())
        }
    }
}

fn common_batch_axis_and_size<V: Typed<ArrayType> + Parameter>(
    inputs: &[ArrayBatch<V>],
) -> Result<(Option<usize>, Option<usize>), TracingError> {
    let mut batch_axis = None;
    let mut axis_size = None;
    for input in inputs {
        validate_axis_size(&mut batch_axis, &mut axis_size, input)?;
    }
    Ok((batch_axis, axis_size))
}

fn validate_output_batch_axis<O: Operation<ArrayType>>(
    operation: &O,
    output_type: &ArrayType,
    batch_axis: Option<usize>,
    axis_size: Option<usize>,
) -> Result<(), TracingError> {
    if let (Some(axis), Some(size)) = (batch_axis, axis_size) {
        if axis >= output_type.rank() {
            return Err(BatchingError::UnsupportedBatchAxisAlignment {
                message: format!("operation '{}' removed batch axis {axis}", operation.name()),
            }
            .into());
        }
        if output_type.dimension(axis as isize) != Size::Static(size) {
            return Err(BatchingError::MismatchedBatchSize.into());
        }
    }
    Ok(())
}

fn batch_by_interpreting_physical_operation<V, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
{
    let (batch_axis, axis_size) = common_batch_axis_and_size(inputs)?;

    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(input_types.as_slice())?;
    let input_values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let output_values = operation.interpret(input_values.as_slice())?;
    check_count!("output", output_values, output_types.len(), TracingError);

    output_types
        .into_iter()
        .zip(output_values)
        .map(|(type_, value)| {
            validate_output_batch_axis(operation, &type_, batch_axis, axis_size)?;
            ArrayBatch::new(type_, value, batch_axis)
        })
        .collect()
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Sin
        + Cos
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V>,
> BatchableOperation<V> for ArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Zero(zero) => batch_by_interpreting_physical_operation(zero, inputs),
            Self::One(one) => batch_by_interpreting_physical_operation(one, inputs),
            Self::Add => batch_by_interpreting_physical_operation(&AddOperation, inputs),
            Self::Sub => batch_by_interpreting_physical_operation(&SubOperation, inputs),
            Self::Mul => batch_by_interpreting_physical_operation(&MulOperation, inputs),
            Self::Div => batch_by_interpreting_physical_operation(&DivOperation, inputs),
            Self::Neg => batch_by_interpreting_physical_operation(&NegOperation, inputs),
            Self::Sin => batch_by_interpreting_physical_operation(&SinOperation, inputs),
            Self::Cos => batch_by_interpreting_physical_operation(&CosOperation, inputs),
            Self::ZeroLike => {
                batch_by_interpreting_physical_operation(&crate::operations::constants::ZeroLikeOperation, inputs)
            }
            Self::OneLike => {
                batch_by_interpreting_physical_operation(&crate::operations::constants::OneLikeOperation, inputs)
            }
            Self::MatrixMultiply => {
                batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::MatMulOperation, inputs)
            }
            Self::Transpose => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::MatrixTransposeOperation,
                inputs,
            ),
            Self::Scale { factor } => {
                batch_by_interpreting_physical_operation(&ScaleOperation::new(factor.clone()), inputs)
            }
            Self::Reshape { input_shape, output_shape } => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                inputs,
            ),
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
        }
    }
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V> + InterpretableOperation<ArrayType, V>,
> BatchableOperation<V> for LinearArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            _ => batch_by_interpreting_physical_operation(self, inputs),
        }
    }
}

/// Symbolic-zero-aware batched interpretation of a [`LinearArrayOperation`].
///
/// When every input batch is structurally zero ([`Tangent::Zero`]), the operation produces
/// structurally zero outputs whose [`ArrayType`]s are derived from the operation's
/// [`Operation::infer_output_types`] without touching the leaf type's arithmetic. Otherwise,
/// each lane is materialized via [`Zero::zero`] for [`Tangent::Zero`] inputs and forwarded to
/// the existing [`BatchableOperation`] implementation over `V`.
///
/// `ZeroLike` and `OneLike` always need their exemplar input materialized to derive the output
/// value, so the short-circuit does not apply to those two variants.
fn batch_linear_with_symbolic_zero<V, Extension>(
    operation: &LinearArrayOperation<V, ArrayType, Extension>,
    inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    let (batch_axis, axis_size) = common_batch_axis_and_size(inputs)?;
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(input_types.as_slice())?;

    let always_materialize = matches!(operation, LinearArrayOperation::ZeroLike | LinearArrayOperation::OneLike);
    if !always_materialize && inputs.iter().all(|input| input.value().is_zero()) {
        return output_types
            .into_iter()
            .map(|output_type| {
                validate_output_batch_axis(operation, &output_type, batch_axis, axis_size)?;
                let value = Tangent::zero(output_type.clone());
                ArrayBatch::new(output_type, value, batch_axis)
            })
            .collect();
    }

    let materialized = inputs
        .iter()
        .map(|input| -> Result<ArrayBatch<V>, TracingError> {
            let materialized_value = match input.value() {
                Tangent::Zero(zero_type) => V::zero(zero_type)?,
                Tangent::Value(value) => value.clone(),
            };
            ArrayBatch::new(input.r#type().into_owned(), materialized_value, input.batch_axis())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let v_outputs = BatchableOperation::<V>::batch(operation, materialized.as_slice())?;

    v_outputs
        .into_iter()
        .map(|v_batch| -> Result<ArrayBatch<Tangent<ArrayType, V>>, TracingError> {
            let output_type = v_batch.r#type().into_owned();
            let output_batch_axis = v_batch.batch_axis();
            let output_value = v_batch.into_value();
            ArrayBatch::new(output_type, Tangent::Value(output_value), output_batch_axis)
        })
        .collect()
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>> + InterpretableOperation<ArrayType, V>,
> BatchableOperation<Tangent<ArrayType, V>> for LinearArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        match self {
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            _ => batch_linear_with_symbolic_zero(self, inputs),
        }
    }
}

/// Interprets a staged program once through packed-array batching rules.
pub fn interpret_batched_program<V, O, Input, Output>(
    program: &Program<ArrayType, V, O, Input, Output>,
    input: Input::To<ArrayBatch<V>>,
) -> Result<Output::To<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<V>,
    Input: Parameterized<V, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<ArrayBatch<V>>>,
    Output: Parameterized<V, Family: ParameterizedFamily<ArrayBatch<V>>>,
{
    let input_structure = input.parameter_structure();
    if input_structure != program.input_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{:?}", program.input_structure),
            right_structure: format!("{input_structure:?}"),
        }
        .into());
    }

    let outputs = program.interpret_with(
        input.into_parameters().collect(),
        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
        |instruction, inputs| instruction.operation.batch(inputs),
    )?;
    Ok(Output::To::<ArrayBatch<V>>::from_parameters(program.output_structure.clone(), outputs)?)
}

/// Maps a scalar traced function over one existing array axis.
///
/// The input values are already packed arrays. `vmap` traces `function` against the logical
/// per-example types obtained by removing `batch_axis`, then replays the staged program through
/// primitive batching rules over the original physical values.
#[allow(private_bounds)]
pub fn vmap<'domain, D, F, Input, Output, V>(
    domain: &'domain D,
    function: F,
    input: Input,
    batch_axis: usize,
) -> Result<Output, TracingError>
where
    D: crate::tracing::domains::TracingDomain<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + 'domain,
    Input: Parameterized<
            V,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Tracer<'domain, D>>,
        >,
    Output: Parameterized<
            V,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Tracer<'domain, D>>,
        >,
    Input::To<ArrayType>:
        Parameterized<ArrayType, To<V> = Input, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
    Output::To<ArrayType>:
        Parameterized<ArrayType, To<V> = Output, To<Tracer<'domain, D>> = Output::To<Tracer<'domain, D>>>,
    Output::To<Tracer<'domain, D>>:
        Parameterized<Tracer<'domain, D>, To<ArrayType> = Output::To<ArrayType>, To<V> = Output>,
    F: FnOnce(Input::To<Tracer<'domain, D>>) -> Result<Output::To<Tracer<'domain, D>>, TracingError>,
    D::OperationCarrier: Clone + BatchableOperation<V>,
{
    let structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    if input_values.is_empty() {
        return Err(BatchingError::EmptyBatch.into());
    }

    let mut axis_size = None;
    let mut logical_types = Vec::with_capacity(input_values.len());
    let mut batched_inputs = Vec::with_capacity(input_values.len());
    for value in input_values {
        let physical_type = value.r#type().into_owned();
        let (logical_type, dimension) = physical_type.without_dimension(batch_axis)?;
        let Some(size) = dimension.value() else {
            return Err(BatchingError::DynamicBatchAxis { type_: physical_type, axis: batch_axis }.into());
        };
        match axis_size {
            Some(existing_size) if existing_size != size => return Err(BatchingError::MismatchedBatchSize.into()),
            Some(_) => {}
            None => axis_size = Some(size),
        }
        logical_types.push(logical_type);
        batched_inputs.push(ArrayBatch::new(physical_type, value, Some(batch_axis))?);
    }

    let input_types = Input::To::<ArrayType>::from_parameters(structure.clone(), logical_types)?;
    let (_, program): (Output::To<ArrayType>, Program<ArrayType, V, D::OperationCarrier, Input, Output>) =
        domain.trace(function, input_types)?;
    let batched_input = Input::To::<ArrayBatch<V>>::from_parameters(structure, batched_inputs)?;
    let batched_output = interpret_batched_program(&program, batched_input)?;
    let output_structure = batched_output.parameter_structure();
    let output_values = batched_output
        .into_parameters()
        .map(|batch| {
            if batch.batch_axis() != Some(batch_axis) {
                return Err(BatchingError::UnbatchedOutput {
                    message: format!("vmap output has batch axis {:?} but expected {batch_axis}", batch.batch_axis()),
                }
                .into());
            }
            Ok(batch.into_value())
        })
        .collect::<Result<Vec<_>, TracingError>>()?;
    Ok(Output::from_parameters(output_structure, output_values)?)
}

/// Interprets a [`crate::tracing_v2::FlatProgram`] (a `Program` over `Vec<V>` input and output)
/// through batching rules, taking and returning packed [`ArrayBatch`]es. Used by the batching
/// implementations of [`ConditionOperation`] and [`WhileOperation`] to recurse into their nested
/// branch / condition / body programs over the same lane configuration.
fn interpret_batched_flat_program<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    inputs: Vec<ArrayBatch<V>>,
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<V>,
{
    program.interpret_with(
        inputs,
        |_, constant| Ok::<_, TracingError>(ArrayBatch::unbatched(constant.clone())),
        |instruction, instruction_inputs| instruction.operation.batch(instruction_inputs),
    )
}

impl<V, O> BatchableOperation<V> for ConditionOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<V>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let (predicate, operand_inputs) = match &self.predicate {
            ConditionPredicate::Captured(predicate) => (*predicate, inputs),
            ConditionPredicate::RuntimeInput(_) => {
                let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
                    return Err(BatchingError::MissingBatchingRule {
                        operation: "condition with no predicate input".to_string(),
                    }
                    .into());
                };
                if predicate_batch.batch_axis().is_some() {
                    return Err(BatchingError::MissingBatchingRule {
                        operation: "condition with lane-varying runtime predicate".to_string(),
                    }
                    .into());
                }
                (predicate_batch.value().control_flow_predicate()?, operand_inputs)
            }
        };
        let branch = if predicate { &self.true_branch } else { &self.false_branch };
        interpret_batched_flat_program(branch, operand_inputs.to_vec())
    }
}

impl<V, O> BatchableOperation<V> for WhileOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<V>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let mut state = inputs.to_vec();
        loop {
            let condition_outputs = interpret_batched_flat_program(&self.condition, state.clone())?;
            check_count!("output", condition_outputs, 1, TracingError);
            let predicate_batch = &condition_outputs[0];
            if predicate_batch.batch_axis().is_some() {
                return Err(BatchingError::MissingBatchingRule {
                    operation: "while with lane-varying loop predicate".to_string(),
                }
                .into());
            }
            if !predicate_batch.value().control_flow_predicate()? {
                return Ok(state);
            }
            state = interpret_batched_flat_program(&self.body, state)?;
        }
    }
}

/// Tangent-runtime counterpart of [`interpret_batched_flat_program`]: lifts each constant to
/// [`Tangent::Value`] and dispatches per-instruction batching through `BatchableOperation<Tangent<…>>`.
fn interpret_batched_flat_program_tangent<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    inputs: Vec<ArrayBatch<Tangent<ArrayType, V>>>,
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    program.interpret_with(
        inputs,
        |_, constant| Ok::<_, TracingError>(ArrayBatch::unbatched(Tangent::Value(constant.clone()))),
        |instruction, instruction_inputs| {
            BatchableOperation::<Tangent<ArrayType, V>>::batch(&instruction.operation, instruction_inputs)
        },
    )
}

impl<V, O> BatchableOperation<Tangent<ArrayType, V>> for ConditionOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    fn batch(
        &self,
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        let predicate = match &self.predicate {
            ConditionPredicate::Captured(predicate) => *predicate,
            ConditionPredicate::RuntimeInput(_) => {
                return Err(BatchingError::MissingBatchingRule {
                    operation: "condition with runtime predicate over tangent runtime values".to_string(),
                }
                .into());
            }
        };
        let branch = if predicate { &self.true_branch } else { &self.false_branch };
        interpret_batched_flat_program_tangent(branch, inputs.to_vec())
    }
}

impl<V, O> BatchableOperation<Tangent<ArrayType, V>> for WhileOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    fn batch(
        &self,
        _inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        // While loops over tangent runtime values cannot make a loop-control decision: the
        // condition program requires a primal value to derive its scalar boolean output, and
        // a `Tangent` lane carries only zero/value tangent metadata, not a primal predicate.
        // Pushforward/pullback programs do not emit `While` today (the JVP rule unrolls loops at
        // trace time and `WhileOperation::transpose` errors), so this path is unreachable from
        // `jacfwd` / `jacrev`; callers manually constructing a tangent `While` get a clear error.
        Err(BatchingError::MissingBatchingRule { operation: "while over tangent runtime values".to_string() }.into())
    }
}
