use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Neg};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::engines::Tracer;
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::{
    ArrayOperation, ControlFlowError, ControlFlowValue, Cos, LinearArrayOperation, MatrixOps, Sin,
};
use crate::types::{ArrayType, Size, Type, Typed};

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

    /// Lane leaves disagreed on their abstract type metadata in the reference fallback.
    #[error("mismatched batch leaf types across batch lanes")]
    MismatchedBatchLeafTypes,

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

/// Packed array value carrying one optional mapped axis.
///
/// [`ArrayBatch`] is the production batching carrier for `tracing_v2`: its [`ArrayType`] is the
/// physical type of `value` and therefore includes the mapped batch axis when [`ArrayBatch::batch_axis`]
/// is `Some`. The logical per-example type is derived by removing that axis.
#[derive(Clone, Debug, Parameter, PartialEq)]
pub struct ArrayBatch<V: Parameter> {
    /// Physical array type of `value`.
    r#type: ArrayType,

    /// Packed array value.
    value: V,

    // TODO(eaplatanios): When would this ever be `None`?
    /// Axis in `type_` and `value` that represents the mapped batch dimension.
    batch_axis: Option<usize>,
}

impl<V: Parameter> ArrayBatch<V> {
    /// Creates a packed array batch from explicit physical metadata.
    ///
    /// # Parameters
    ///
    ///   - `type_`: Physical type of `value`. This type includes `batch_axis` when present.
    ///   - `value`: Physical array value.
    ///   - `batch_axis`: Optional mapped axis in `type_` and `value`.
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

    /// Wraps an unbatched scalar-body value.
    pub fn unbatched(value: V) -> Self
    where
        V: Traceable<ArrayType>,
    {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }

    /// Returns the optional mapped axis.
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
        let Some(size) = self.r#type.dimension(axis as i32).value() else {
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

impl<V: Parameter> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display + Parameter> Display for ArrayBatch<V> {
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

fn ensure_compatible_array_type(expected: &ArrayType, got: &ArrayType) -> Result<(), BatchingError> {
    if expected.is_compatible_with(got) && got.is_compatible_with(expected) {
        Ok(())
    } else {
        Err(BatchingError::MismatchedBatchLeafTypes)
    }
}

fn validate_axis_size<V: Parameter>(
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

fn batch_by_interpreting_physical_operation<V, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
{
    let mut batch_axis = None;
    let mut axis_size = None;
    for input in inputs {
        validate_axis_size(&mut batch_axis, &mut axis_size, input)?;
    }

    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(input_types.as_slice())?;
    let input_values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let output_values = operation.interpret(input_values.as_slice())?;
    if output_values.len() != output_types.len() {
        return Err(TracingError::InvalidOutputCount { expected: output_types.len(), got: output_values.len() });
    }

    output_types
        .into_iter()
        .zip(output_values)
        .map(|(type_, value)| {
            if let (Some(axis), Some(size)) = (batch_axis, axis_size) {
                if axis >= type_.rank() {
                    return Err(BatchingError::UnsupportedBatchAxisAlignment {
                        message: format!("operation '{}' removed batch axis {axis}", operation.name()),
                    }
                    .into());
                }
                if type_.dimension(axis as i32) != Size::Static(size) {
                    return Err(BatchingError::MismatchedBatchSize.into());
                }
            }
            ArrayBatch::new(type_, value, batch_axis)
        })
        .collect()
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
> BatchableOperation<V> for ArrayOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Zero(zero) => batch_by_interpreting_physical_operation(zero, inputs),
            Self::One(one) => batch_by_interpreting_physical_operation(one, inputs),
            Self::Add => batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::AddOperation, inputs),
            Self::Mul => batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::MulOperation, inputs),
            Self::Neg => batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::NegOperation, inputs),
            Self::Sin => batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::SinOperation, inputs),
            Self::Cos => batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::CosOperation, inputs),
            Self::ZeroLike => {
                batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::ZeroLikeOperation, inputs)
            }
            Self::OneLike => {
                batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::OneLikeOperation, inputs)
            }
            Self::MatrixMultiply => {
                batch_by_interpreting_physical_operation(&crate::tracing_v2::operations::MatMulOperation, inputs)
            }
            Self::Transpose => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::MatrixTransposeOperation,
                inputs,
            ),
            Self::Scale { factor } => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::ScaleOperation::new(factor.clone()),
                inputs,
            ),
            Self::Reshape { input_shape, output_shape } => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                inputs,
            ),
            Self::Custom(_) | Self::Rematerialize(_) | Self::Condition(_) | Self::While(_) => {
                Err(BatchingError::MissingBatchingRule { operation: self.name().to_string() }.into())
            }
        }
    }
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
> BatchableOperation<V> for LinearArrayOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Custom(_) | Self::Rematerialize(_) | Self::Condition(_) | Self::While(_) => {
                Err(BatchingError::MissingBatchingRule { operation: self.name().to_string() }.into())
            }
            _ => batch_by_interpreting_physical_operation(self, inputs),
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
pub fn vmap<'engine, E, F, Input, Output, V>(
    engine: &'engine E,
    function: F,
    input: Input,
    batch_axis: usize,
) -> Result<Output, TracingError>
where
    E: crate::tracing::engines::TracingEngine<Type = ArrayType, Value = V> + ?Sized,
    V: Traceable<ArrayType>,
    Input: Parameterized<
            V,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Output: Parameterized<
            V,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Input::To<ArrayType>:
        Parameterized<ArrayType, To<V> = Input, To<Tracer<'engine, E>> = Input::To<Tracer<'engine, E>>>,
    Output::To<ArrayType>:
        Parameterized<ArrayType, To<V> = Output, To<Tracer<'engine, E>> = Output::To<Tracer<'engine, E>>>,
    Output::To<Tracer<'engine, E>>:
        Parameterized<Tracer<'engine, E>, To<ArrayType> = Output::To<ArrayType>, To<V> = Output>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    E::Operation: Clone + BatchableOperation<V>,
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
    let (_, program): (Output::To<ArrayType>, Program<ArrayType, V, E::Operation, Input, Output>) =
        engine.trace(function, input_types)?;
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

/// Reference lane carrier used only by dense Jacobian materialization until a backend provides
/// packed basis construction.
#[derive(Clone, Debug, Parameter, PartialEq)]
pub(crate) struct ReferenceBatch<V: Parameter> {
    /// Logical type shared by all lanes.
    r#type: ArrayType,

    /// Lane values in mapped-axis order.
    lanes: Vec<V>,
}

impl<V: Parameter> ReferenceBatch<V> {
    pub(crate) fn new(r#type: ArrayType, lanes: Vec<V>) -> Self {
        Self { r#type, lanes }
    }

    pub(crate) fn broadcast(value: V, lane_count: usize) -> Self
    where
        V: Traceable<ArrayType>,
    {
        Self { r#type: value.r#type().into_owned(), lanes: vec![value; lane_count] }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.lanes.len()
    }

    #[inline]
    pub(crate) fn lanes(&self) -> &[V] {
        self.lanes.as_slice()
    }

    #[inline]
    pub(crate) fn into_lanes(self) -> Vec<V> {
        self.lanes
    }
}

impl<V: Parameter> Typed<ArrayType> for ReferenceBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display + Parameter> Display for ReferenceBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reference_batch(")?;
        for (index, lane) in self.lanes.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            Display::fmt(lane, formatter)?;
        }
        write!(formatter, ")")
    }
}

impl<V: Traceable<ArrayType>> Traceable<ArrayType> for ReferenceBatch<V> {}

impl<V: Value<ArrayType>> Value<ArrayType> for ReferenceBatch<V> {}

impl<V: Traceable<ArrayType> + ControlFlowValue> ControlFlowValue for ReferenceBatch<V> {
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        Err(ControlFlowError::MissingTransformRule { transform: "reference-batched predicate control flow" }.into())
    }
}

fn validate_reference_lane_count<V: Parameter>(batches: &[ReferenceBatch<V>]) -> Result<usize, BatchingError> {
    let Some(first_batch) = batches.first() else {
        return Err(BatchingError::EmptyBatch);
    };
    let lane_count = first_batch.len();
    if batches.iter().any(|batch| batch.len() != lane_count) {
        return Err(BatchingError::MismatchedBatchSize);
    }
    Ok(lane_count)
}

pub(crate) fn reference_stack<V, Input>(inputs: Vec<Input>) -> Result<Input::To<ReferenceBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<V, ParameterStructure: PartialEq, Family: ParameterizedFamily<ReferenceBatch<V>>>,
{
    let mut inputs = inputs.into_iter();
    let first = inputs.next().ok_or(BatchingError::EmptyBatch)?;
    let structure = first.parameter_structure();
    let mut buckets = first
        .into_parameters()
        .map(|parameter| (parameter.r#type().into_owned(), vec![parameter]))
        .collect::<Vec<_>>();

    for input in inputs {
        if input.parameter_structure() != structure {
            return Err(BatchingError::MismatchedParameterStructures.into());
        }
        for ((expected_type, bucket), parameter) in buckets.iter_mut().zip(input.into_parameters()) {
            ensure_compatible_array_type(expected_type, &parameter.r#type())?;
            bucket.push(parameter);
        }
    }

    Ok(Input::To::<ReferenceBatch<V>>::from_parameters(
        structure,
        buckets.into_iter().map(|(r#type, lanes)| ReferenceBatch::new(r#type, lanes)),
    )?)
}

fn interpret_reference_instruction<V, O>(
    operation: &O,
    inputs: &[ReferenceBatch<V>],
    output_types: Vec<ArrayType>,
    lane_count: usize,
) -> Result<Vec<ReferenceBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: InterpretableOperation<ArrayType, V>,
{
    if inputs.iter().any(|input| input.len() != lane_count) {
        return Err(BatchingError::MismatchedBatchSize.into());
    }

    let mut output_lanes = output_types.iter().map(|_| Vec::with_capacity(lane_count)).collect::<Vec<Vec<V>>>();
    let mut lane_inputs = Vec::with_capacity(inputs.len());
    for lane_index in 0..lane_count {
        lane_inputs.clear();
        lane_inputs.extend(inputs.iter().map(|input| input.lanes()[lane_index].clone()));
        let lane_outputs = operation.interpret(lane_inputs.as_slice())?;
        if lane_outputs.len() != output_types.len() {
            return Err(TracingError::InvalidOutputCount { expected: output_types.len(), got: lane_outputs.len() });
        }
        for (bucket, output) in output_lanes.iter_mut().zip(lane_outputs) {
            bucket.push(output);
        }
    }

    Ok(output_types
        .into_iter()
        .zip(output_lanes)
        .map(|(r#type, lanes)| ReferenceBatch::new(r#type, lanes))
        .collect())
}

pub(crate) fn interpret_reference_batched_program<V, O, Input, Output>(
    program: &Program<ArrayType, V, O, Input, Output>,
    input: Input::To<ReferenceBatch<V>>,
) -> Result<Output::To<ReferenceBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
    Input: Parameterized<V, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<ReferenceBatch<V>>>,
    Output: Parameterized<V, Family: ParameterizedFamily<ReferenceBatch<V>>>,
{
    let input_structure = input.parameter_structure();
    if input_structure != program.input_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{:?}", program.input_structure),
            right_structure: format!("{input_structure:?}"),
        }
        .into());
    }

    let input_values = input.into_parameters().collect::<Vec<_>>();
    let lane_count = validate_reference_lane_count(input_values.as_slice())?;

    let outputs = program.interpret_with(
        input_values,
        |_, constant| Ok(ReferenceBatch::broadcast(constant.clone(), lane_count)),
        |instruction, inputs| {
            let output_types = instruction
                .outputs
                .iter()
                .map(|output| program.atoms[output.index].r#type().into_owned())
                .collect::<Vec<_>>();
            interpret_reference_instruction(&instruction.operation, inputs, output_types, lane_count)
        },
    )?;
    Ok(Output::To::<ReferenceBatch<V>>::from_parameters(program.output_structure.clone(), outputs)?)
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::ops::{Add, Mul, Neg};
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::broadcasting::Broadcastable;
    use crate::tracing::engines::{Engine, TracingEngine};
    use crate::tracing_v2::DifferentiableEngine;
    use crate::tracing_v2::operations::{ControlFlowError, ControlFlowValue, CustomPrimitive};
    use crate::types::{DataType, Shape};

    #[derive(Clone, Debug, PartialEq)]
    struct TestArray {
        r#type: ArrayType,
        values: Vec<f64>,
    }

    impl TestArray {
        fn scalar(value: f64) -> Self {
            Self { r#type: ArrayType::scalar(DataType::F64), values: vec![value] }
        }

        fn vector(values: Vec<f64>) -> Self {
            Self {
                r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())]), None, None)
                    .unwrap(),
                values,
            }
        }

        fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Self {
            assert_eq!(values.len(), rows * cols);
            Self {
                r#type: ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Size::Static(rows), Size::Static(cols)]),
                    None,
                    None,
                )
                .unwrap(),
                values,
            }
        }

        fn element_count(r#type: &ArrayType) -> usize {
            if r#type.rank() == 0 {
                1
            } else {
                r#type.shape.dimensions.iter().map(|dimension| dimension.value().unwrap()).product()
            }
        }

        fn broadcast_values(&self, output_len: usize) -> Vec<f64> {
            if self.values.len() == output_len {
                self.values.clone()
            } else if self.values.len() == 1 {
                vec![self.values[0]; output_len]
            } else {
                panic!("cannot broadcast {} values to {output_len}", self.values.len());
            }
        }

        fn binary(self, rhs: Self, function: impl Fn(f64, f64) -> f64) -> Self {
            let output_type = self.r#type.broadcast(&rhs.r#type).unwrap();
            let output_len = Self::element_count(&output_type);
            let left = self.broadcast_values(output_len);
            let right = rhs.broadcast_values(output_len);
            Self {
                r#type: output_type,
                values: left.into_iter().zip(right).map(|(left, right)| function(left, right)).collect(),
            }
        }
    }

    impl Parameter for TestArray {}

    impl Display for TestArray {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{:?}", self.values)
        }
    }

    impl Typed<ArrayType> for TestArray {
        fn r#type(&self) -> Cow<'_, ArrayType> {
            Cow::Borrowed(&self.r#type)
        }
    }

    impl Traceable<ArrayType> for TestArray {}

    impl Value<ArrayType> for TestArray {}

    impl ControlFlowValue for TestArray {
        fn control_flow_predicate(&self) -> Result<bool, TracingError> {
            Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
        }
    }

    impl ZeroLike for TestArray {
        fn zero_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![0.0; self.values.len()] }
        }
    }

    impl crate::tracing_v2::operations::constants::Zero<ArrayType> for TestArray {
        fn zero(value_type: &ArrayType) -> Result<Self, TracingError> {
            let mut element_count = 1usize;
            for dim in &value_type.shape.dimensions {
                element_count *= dim.value().ok_or_else(|| crate::types::TypeError {
                    message: format!("test array zero requires static shape but got {value_type}"),
                })?;
            }
            Ok(Self { r#type: value_type.clone(), values: vec![0.0; element_count] })
        }
    }

    impl crate::tracing_v2::operations::constants::One<ArrayType> for TestArray {
        fn one(value_type: &ArrayType) -> Result<Self, TracingError> {
            if value_type.rank() != 0 {
                return Err(crate::tracing_v2::DifferentiationError::NonScalarGradientOutput {
                    output_type: value_type.clone(),
                }
                .into());
            }
            Ok(Self { r#type: value_type.clone(), values: vec![1.0] })
        }
    }

    impl OneLike for TestArray {
        fn one_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![1.0; self.values.len()] }
        }
    }

    impl crate::tracing_v2::Differentiable<ArrayType> for TestArray {
        type Tangent = Self;
    }

    impl Add for TestArray {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left + right)
        }
    }

    impl Mul for TestArray {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left * right)
        }
    }

    impl Neg for TestArray {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self { r#type: self.r#type, values: self.values.into_iter().map(|value| -value).collect() }
        }
    }

    impl Sin for TestArray {
        fn sin(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::sin).collect() }
        }
    }

    impl Cos for TestArray {
        fn cos(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::cos).collect() }
        }
    }

    impl MatrixOps for TestArray {
        fn matmul(self, rhs: Self) -> Self {
            self * rhs
        }

        fn transpose_matrix(self) -> Self {
            self
        }
    }

    impl ReshapeOps for TestArray {
        fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
            let output_type = ArrayType::new(self.r#type.data_type, target_shape, None, None).unwrap();
            assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
            Ok(Self { r#type: output_type, values: self.values })
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestArrayEngine;

    impl Engine for TestArrayEngine {
        type Type = ArrayType;
        type Value = TestArray;

        fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            Ok(TestArray { r#type: r#type.clone(), values: vec![0.0; TestArray::element_count(r#type)] })
        }

        fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            Ok(TestArray { r#type: r#type.clone(), values: vec![1.0; TestArray::element_count(r#type)] })
        }
    }

    impl TracingEngine for TestArrayEngine {
        type Operation = ArrayOperation<TestArray>;
    }

    impl crate::tracing_v2::LinearEngine for TestArrayEngine {
        type LinearOperation = LinearArrayOperation<TestArray>;
    }

    impl DifferentiableEngine for TestArrayEngine {
        type DifferentiableOperation = ArrayOperation<TestArray>;
    }

    #[derive(Clone, Debug)]
    struct TestCustomOp;

    impl Display for TestCustomOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "custom_test")
        }
    }

    impl Operation<ArrayType> for TestCustomOp {
        fn name(&self) -> &'static str {
            "custom_test"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, crate::types::TypeError> {
            Ok(input_types.to_vec())
        }
    }

    impl InterpretableOperation<ArrayType, TestArray> for TestCustomOp {
        fn interpret(&self, inputs: &[TestArray]) -> Result<Vec<TestArray>, TracingError> {
            Ok(inputs.to_vec())
        }
    }

    #[test]
    fn test_array_batch_derives_logical_type_from_batch_axis() {
        let batch = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();

        assert_eq!(batch.axis_size(), Ok(Some(3)));
        assert_eq!(batch.logical_type(), Ok(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_vmap_uses_one_packed_array_value() {
        let engine = TestArrayEngine;

        let output = vmap::<TestArrayEngine, _, TestArray, TestArray, TestArray>(
            &engine,
            |x| Ok(x.clone() * x.clone() + x.sin()),
            TestArray::vector(vec![0.0, 1.0, 2.0]),
            0,
        )
        .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]), None, None).unwrap()
        );
        assert_eq!(output.values.len(), 3);
        for (actual, expected) in output.values.iter().zip([0.0, 1.0 + 1.0f64.sin(), 4.0 + 2.0f64.sin()]) {
            assert!((*actual - expected).abs() <= 1e-12, "expected {actual} ~= {expected}");
        }
    }

    #[test]
    fn test_vmap_broadcasts_scalar_constants_inside_packed_operations() {
        let engine = TestArrayEngine;

        let output = vmap::<TestArrayEngine, _, TestArray, TestArray, TestArray>(
            &engine,
            |x| Ok(x.clone() + x.one_like()),
            TestArray::vector(vec![2.0, 4.0, 6.0]),
            0,
        )
        .unwrap();

        assert_eq!(output.values, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_vmap_maps_structured_packed_inputs_and_outputs() {
        let engine = TestArrayEngine;

        let output = vmap::<TestArrayEngine, _, (TestArray, TestArray), (TestArray, TestArray), TestArray>(
            &engine,
            |(left, right)| Ok((left.clone() + right.clone(), left * right)),
            (TestArray::vector(vec![1.0, 3.0]), TestArray::vector(vec![2.0, 4.0])),
            0,
        )
        .unwrap();

        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batching_rule_rejects_unaligned_batch_axes() {
        let left = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0).unwrap();
        let right = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 1).unwrap();

        assert!(matches!(
            ArrayOperation::<TestArray>::Add.batch(&[left, right]),
            Err(TracingError::Batching(BatchingError::UnsupportedBatchAxisAlignment { .. }))
        ));
    }

    #[test]
    fn test_custom_primitive_requires_explicit_batching_rule() {
        let operation = ArrayOperation::<TestArray>::Custom(Arc::new(CustomPrimitive::new(TestCustomOp)));
        let input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0]), 0).unwrap();

        assert!(matches!(
            operation.batch(&[input]),
            Err(TracingError::Batching(BatchingError::MissingBatchingRule { operation })) if operation == "custom_test"
        ));
    }

    #[test]
    fn test_reference_stack_remains_private_lane_fallback() {
        let lanes =
            vec![(TestArray::scalar(1.0), TestArray::scalar(2.0)), (TestArray::scalar(3.0), TestArray::scalar(4.0))];

        let batched = reference_stack::<TestArray, _>(lanes).unwrap();

        assert_eq!(batched.0.lanes().iter().map(|lane| lane.values[0]).collect::<Vec<_>>(), vec![1.0, 3.0]);
        assert_eq!(batched.1.lanes().iter().map(|lane| lane.values[0]).collect::<Vec<_>>(), vec![2.0, 4.0]);
    }
}
