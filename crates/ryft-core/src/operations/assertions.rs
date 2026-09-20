//! Correctness assertions over Boolean predicates, with named observations on failure.
//!
//! Use [`Assert::assert_with_limit`] for bounded reporting over an array of predicates. To inspect per-lane
//! outcomes without raising, return the predicate array as an ordinary program output instead:
//!
//! ```rust
//! use ryft_core::{Array, Compare, ProgramError, ZeroLike};
//!
//! # fn main() -> Result<(), ProgramError> {
//! let divisors = Array::vector(vec![2_i32, 0, 4])?;
//! let valid = divisors.not_equal(&divisors.zero_like()?)?;
//! assert_eq!(valid.elements::<bool>()?, vec![true, false, true]);
//! # Ok(())
//! # }
//! ```
//!
//! A validation mask does not guard later operations; callers must ensure their inputs remain valid.

// TODO(eaplatanios): Review from here onwards.

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use half::{bf16, f16};

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, DimensionType, DimensionValue,
};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, ValueResolution};
use crate::differentiation::{DifferentiationContext, DifferentiationPolicy};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_non_transposable_operation,
    impl_reference_dischargeable_operation,
};
use crate::operations::compare::{CompareOperation, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_add::DimensionAddOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcastOperation};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::padding::PadOperation;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::DynamicSlice;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Concretizable, EffectClass, EffectClasses, Effects, Operation, OperationFormatter, ProgramError, RegionInterface,
    Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, TracingContext};

/// Failure of a correctness assertion, including the named values observed at that assertion.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssertionError {
    /// The Boolean condition was false.
    Failed { message: String, observations: Vec<(String, String)> },
    /// One or more logical lanes failed, with a bounded diagnostic sample and the number omitted.
    FailedLanes { message: String, lanes: Vec<AssertionLane>, omitted: usize },
}

/// Coordinates and named observations for one failed logical assertion lane.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AssertionLane {
    /// Logical row-major coordinates of the failed lane.
    pub index: Vec<usize>,
    /// Named scalar values observed at the failed lane.
    pub observations: Vec<(String, String)>,
}

impl Display for AssertionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FailedLanes { message, lanes, omitted } => {
                write!(formatter, "assertion failed: {message}")?;
                for lane in lanes {
                    write!(formatter, "; lane {:?}", lane.index)?;
                    for (label, value) in &lane.observations {
                        write!(formatter, ", {label}={value}")?;
                    }
                }
                if *omitted > 0 {
                    write!(formatter, "; {omitted} additional failing lanes omitted")?;
                }
                Ok(())
            }
            Self::Failed { message, observations } => {
                write!(formatter, "assertion failed: {message}")?;
                for (index, (label, value)) in observations.iter().enumerate() {
                    let separator = if index == 0 { "; " } else { ", " };
                    write!(formatter, "{separator}{label}={value}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AssertionError {}

impl From<AssertionError> for ProgramError {
    fn from(error: AssertionError) -> Self {
        Self::custom(error)
    }
}

/// Canonical operation name for [`AssertOperation`].
pub const ASSERT_OPERATION_NAME: &str = "assert";

/// Checks Boolean conditions and produces no outputs. By default, the condition and observations are scalar.
/// [`with_failure_limit`](Self::with_failure_limit) enables array conditions and bounded lane diagnostics.
/// Failed checks report the message and named observations. Assertions remain enabled independently of debug builds. Every residual instruction declares
/// [`EffectClass::OrderedAssertion`]; known conditions are handled by [`Assert`] and partial evaluation.
///
/// The first input is the condition; subsequent inputs correspond to [`labels`](Self::labels). Supported observations
/// are dimensions and scalar Boolean, integer, `bf16`, `f16`, `f32`, or `f64` arrays. Labels and messages are literal
/// descriptions and do not refer to type identities. With bounded reporting, array observations may instead have
/// the condition's shape. Separate assertions preserve separate ordered failures.
/// Default mapped batching reports the first failing lane at each level, with batch extents restricted to the
/// `i32` index range. Bounded reporting instead preserves all logical lanes and reports a row-major sample. Mixed array programs support dynamic extents, including empty batches, which pass vacuously. Observations
/// with a potentially empty mapped axis receive one unused padding lane so diagnostic selection remains valid;
/// the padded extent must also fit `i32`.
#[derive(Clone, Debug)]
pub struct AssertOperation<T: Type> {
    /// Refer to the documentation of [`message`](Self::message) for more information.
    message: String,

    /// Refer to the documentation of [`labels`](Self::labels) for more information.
    labels: Vec<String>,

    /// Refer to the documentation of [`failure_limit`](Self::failure_limit) for more information.
    failure_limit: Option<NonZeroUsize>,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type + Into<ArrayIrType>> AssertOperation<T> {
    /// Creates an assertion with the provided message and no diagnostic observations.
    pub fn new<M: Into<String>>(message: M) -> Self {
        Self { message: message.into(), labels: Vec::new(), failure_limit: None, marker: PhantomData }
    }

    /// Returns a copy of this [`AssertOperation`] with its diagnostic labels set to the provided `labels`.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Returns a copy of this [`AssertOperation`] with bounded lane reporting enabled with the provided `limit`.
    /// Conditions may then be Boolean arrays; observations must be scalars or have the condition's shape.
    /// At most `limit` failed lanes are reported in logical row-major order, along with the number omitted.
    pub fn with_failure_limit(mut self, limit: NonZeroUsize) -> Self {
        self.failure_limit = Some(limit);
        self
    }

    /// Returns the literal message reported when the condition is false.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the labels for the diagnostic inputs following the condition, in input order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the maximum reported failed lanes, or `None` for the default scalar assertion behavior.
    pub fn failure_limit(&self) -> Option<NonZeroUsize> {
        self.failure_limit
    }

    /// Batches logical scalar inputs, selecting one observed failing lane at the current transform level.
    fn batch_inputs<C, P, V, Project, Lift, Indices, MaskEmpty, PadEmpty, Pack, Driver>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &Driver,
        inputs: &[P::Batch],
        extent: Dimension,
        project: Project,
        lift: Lift,
        indices: Indices,
        mask_empty: MaskEmpty,
        mut pad_empty: PadEmpty,
        pack: Pack,
    ) -> Result<BatchedOutputs<C, P>, BatchingError>
    where
        C: AssertionContext<Type = T>,
        P: BatchingPolicy<C>,
        V: Value<Type = ArrayType> + Reduce + Select + DynamicSlice + Reshape + ConvertElementType + ZeroLike,
        Project: Fn(C::Value) -> Result<V, ProgramError>,
        Lift: Fn(V) -> C::Value,
        Indices: Fn(&ArrayType) -> Result<V, ProgramError>,
        MaskEmpty: Fn(V) -> Result<V, ProgramError>,
        PadEmpty: FnMut(C::Value) -> Result<C::Value, ProgramError>,
        Pack: Fn(&[P::Batch]) -> Result<Vec<C::Value>, BatchingError>,
        Driver: BatchingDriver<C, P>,
    {
        self.infer_output_types(
            &inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        // Empty batches pass even when the predicate is replicated and false. Check the extent before replaying.
        if extent.value() == Some(0) {
            return Ok(Vec::new().into());
        }
        if self.failure_limit.is_some() {
            let arguments = pack(inputs)?;
            context.parent().assert(self.clone(), &arguments)?;
            return Ok(Vec::new().into());
        }
        let mapped = inputs.iter().any(|input| P::batch_axis(input).axis().is_some());
        if !mapped {
            let mut arguments = inputs.iter().map(|input| P::value(input).clone()).collect::<Vec<_>>();
            arguments[0] = lift(mask_empty(project(arguments[0].clone())?)?);
            context.parent().assert(self.clone(), &arguments)?;
            return Ok(Vec::new().into());
        }
        let (_, maximum) = extent.bounds().representable_extent_range().map_err(ProgramError::from)?;
        if maximum > i32::MAX as usize {
            return Err(BatchingError::UnsupportedOperation {
                message: "mapped assertion batch extent exceeds the supported `i32` index range".to_owned(),
            });
        }
        // Backends also represent logical padded extents with signed indices; reserve the fallback lane before staging.
        if maximum == i32::MAX as usize
            && extent.bounds().lower() == 0
            && inputs[1..].iter().any(|input| P::batch_axis(input).axis().is_some())
        {
            return Err(BatchingError::UnsupportedOperation {
                message: "padded assertion batch extent exceeds the supported `i32` extent range".to_owned(),
            });
        }
        let condition = project(P::value(&inputs[0]).clone())?;
        let (condition, index) = if P::batch_axis(&inputs[0]).axis().is_some() {
            let coordinates = indices(&condition.r#type().into_owned().with_data_type(DataType::I32))?;
            // Passing lanes may tie the last coordinate: when any lane fails, the minimum still identifies the
            // first failure. Successful and empty batches are handled by the independent Boolean reduction.
            let sentinel = coordinates.reduce(&[0], ReductionKind::Max);
            let candidates = V::select(&condition, &sentinel, &coordinates)?;
            let index = candidates.reduce(&[0], ReductionKind::Min);
            let condition = condition.reduce(&[0], ReductionKind::All);
            // Empty and successful batches use index zero. A padded observation makes this valid even at extent zero.
            let index = V::select(&condition, &index.zero_like()?, &index)?;
            (condition, index)
        } else {
            let condition = mask_empty(condition)?;
            let index = condition.convert_element_type(DataType::I32)?.zero_like()?;
            (condition, index)
        };
        let mut arguments = vec![lift(condition)];
        for input in &inputs[1..] {
            if P::batch_axis(input).axis().is_some() {
                let input = project(pad_empty(P::value(input).clone())?)?;
                // The selected batch index is never negative, so the clamp-only policy keeps this slice on the single
                // gather path even when the mapped axis is dynamic.
                let observation =
                    input.dynamic_slice_with_negative_indices(std::slice::from_ref(&index), &[1], false)?;
                arguments.push(lift(observation.reshape([])?));
            } else {
                arguments.push(P::value(input).clone());
            }
        }
        arguments.push(lift(index));
        let mut labels = self.labels.clone();
        let mut label = "batch_index".to_owned();
        let mut suffix = 1;
        while labels.contains(&label) {
            label = format!("batch_index_{suffix}");
            suffix += 1;
        }
        labels.push(label);
        context.parent().assert(self.clone().with_labels(labels), &arguments)?;
        Ok(Vec::new().into())
    }

    /// Elides known successes at a staging boundary; transform contexts first apply their own rule.
    fn fold_or_bind<C: Context<Type = T, Constant: AssertionValue, Operation: From<Self>>>(
        &self,
        context: &C,
        inputs: &[C::Value],
    ) -> Result<(), ProgramError> {
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        if let ValueResolution::Constant(condition) = context.resolve(&inputs[0]) {
            match condition.concretize() {
                Ok(true) => return Ok(()),
                Ok(false) | Err(ProgramError::Concretization { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        context.bind(self.clone(), Vec::new(), inputs)?;
        Ok(())
    }

    /// Checks concrete logical lanes without changing ordinary scalar assertion behavior.
    fn check_lanes<V: AssertionValue<Type = T>>(
        &self,
        condition: &V,
        observations: &[(&str, V)],
    ) -> Result<(), ProgramError> {
        let input_types = std::iter::once(condition.r#type().into_owned())
            .chain(observations.iter().map(|(_, value)| value.r#type().into_owned()))
            .collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        let condition = condition.assertion_array()?.unwrap();
        let values = condition.elements::<bool>()?;
        let failure_count = values.iter().filter(|value| !**value).count();
        if failure_count == 0 {
            return Ok(());
        }
        let shape = condition
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| dimension.value().unwrap())
            .collect::<Vec<_>>();
        let observations = observations
            .iter()
            .map(|(label, value)| Ok((*label, value, value.assertion_array()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let limit = self.failure_limit.unwrap().get();
        let mut lanes = Vec::new();
        for position in values.iter().enumerate().filter_map(|(index, value)| (!value).then_some(index)).take(limit) {
            let mut remaining = position;
            let mut index = vec![0; shape.len()];
            for axis in (0..shape.len()).rev() {
                index[axis] = remaining % shape[axis];
                remaining /= shape[axis];
            }
            let starts =
                index.iter().map(|&coordinate| Array::scalar(coordinate as i64)).collect::<Result<Vec<_>, _>>()?;
            let observations = observations
                .iter()
                .map(|(label, value, array)| {
                    let observation = match array {
                        Some(array) if array.r#type().rank() > 0 => {
                            array.dynamic_slice(&starts, &vec![1; index.len()])?.reshape([])?.assertion_observation()?
                        }
                        Some(array) => array.assertion_observation()?,
                        None => value.assertion_observation()?,
                    };
                    Ok(((*label).to_owned(), observation))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            lanes.push(AssertionLane { index, observations });
        }
        Err(AssertionError::FailedLanes {
            message: self.message.clone(),
            omitted: failure_count.saturating_sub(limit),
            lanes,
        }
        .into())
    }

    /// Constructs a failure from validated observations, retaining dimension facts when values cannot be resolved.
    fn failure<V: AssertionValue<Type = T>, I: IntoIterator<Item = (Option<V>, T)>>(
        &self,
        observations: I,
    ) -> Result<AssertionError, ProgramError> {
        let observations = self
            .labels
            .iter()
            .zip(observations)
            .map(|(label, (input, r#type))| {
                let fallback = || match r#type.into() {
                    ArrayIrType::Dimension(dimension) => match dimension.extent() {
                        Some(extent) => extent.to_string(),
                        None => format!("<unknown: `{dimension}`>"),
                    },
                    _ => "<unknown>".to_owned(),
                };
                let observation = match input {
                    Some(input) => match input.assertion_observation() {
                        Ok(value) => value,
                        Err(ProgramError::Concretization { .. }) => fallback(),
                        Err(error) => return Err(error),
                    },
                    None => fallback(),
                };
                Ok((label.clone(), observation))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(AssertionError::Failed { message: self.message.clone(), observations })
    }
}

impl<T: Type + Into<ArrayIrType>> Display for AssertOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type + Into<ArrayIrType>> Operation for AssertOperation<T> {
    type Type = T;

    fn name(&self) -> &'static str {
        ASSERT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1 + self.labels.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let condition: ArrayIrType = input_types[0].clone().into();
        if !matches!(&condition, ArrayIrType::Array(array) if (self.failure_limit.is_some() || array.rank() == 0) && array.data_type() == DataType::Boolean)
        {
            return Err(TypeError::invalid(format!(
                "assertion condition must have type `{}` but has type `{condition}`",
                if self.failure_limit.is_some() { "bool[...]" } else { "bool[]" },
            )));
        }
        for (label, input) in self.labels.iter().zip(&input_types[1..]) {
            let input: ArrayIrType = input.clone().into();
            let supported = match &input {
                ArrayIrType::Dimension(_) => true,
                ArrayIrType::Array(array) => {
                    (array.rank() == 0
                        || (self.failure_limit.is_some()
                            && matches!(&condition, ArrayIrType::Array(condition) if array.shape() == condition.shape())))
                        && (array.data_type().is_integer()
                            || matches!(
                                array.data_type(),
                                DataType::Boolean | DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64
                            ))
                }
                _ => false,
            };
            if !supported {
                return Err(TypeError::invalid(format!(
                    "assertion observation `{label}` has unsupported type `{input}`"
                )));
            }
        }
        Ok(Vec::new())
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("message", format_args!("{:?}", self.message))?;
            operation.field("labels", format_args!("{:?}", self.labels))?;
            if let Some(limit) = self.failure_limit {
                operation.field("failure_limit", limit)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free <T> AssertOperation<T> where T: Type + Into<ArrayIrType>);

impl<D: Domain<Value: Assert>> InterpretableOperation<D> for AssertOperation<D::Type>
where
    D::Type: Into<ArrayIrType>,
{
    fn interpret<I: InterpretationDriver<D>>(
        &self,
        _context: &D,
        driver: &I,
        inputs: &[D::Value],
    ) -> Result<Vec<D::Value>, ProgramError> {
        check_count!("input", inputs, 1 + self.labels.len(), ProgramError);
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        let observations = self
            .labels
            .iter()
            .zip(&inputs[1..])
            .map(|(label, input)| (label.as_str(), input.clone()))
            .collect::<Vec<_>>();
        match self.failure_limit {
            Some(limit) => inputs[0].assert_with_limit(&self.message, &observations, limit)?,
            None => inputs[0].assert(&self.message, &observations)?,
        }
        Ok(Vec::new())
    }
}

impl<C: Context<Constant: AssertionValue, Operation: From<AssertOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for AssertOperation<C::Type>
where
    C::Type: Into<ArrayIrType>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1 + self.labels.len(), ProgramError);
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        if self.failure_limit.is_some() {
            return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
        }
        if let Some(condition) = inputs[0].as_known()
            && let ValueResolution::Constant(condition) = context.parent().resolve(condition)
        {
            match condition.concretize() {
                Ok(true) => return Ok(Vec::new()),
                Ok(false) => {
                    if !context.parent().is_eager() || !context.can_fold_effects(self.effects().classes()) {
                        return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
                    }
                }
                Err(ProgramError::Concretization { .. }) => {
                    return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
                }
                Err(error) => return Err(error),
            }
            return Err(self
                .failure(inputs[1..].iter().map(|input| {
                    let value = input.as_known().and_then(|input| match context.parent().resolve(input) {
                        ValueResolution::Constant(input) => Some(input),
                        _ => None,
                    });
                    (value, input.r#type().into_owned())
                }))?
                .into());
        }
        context.fold_or_residualize(self.clone(), Vec::new(), inputs)
    }
}

impl<C: AssertionContext<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for AssertOperation<ArrayType>
where
    C::Operation: From<Self> + From<IotaOperation<ArrayType>> + From<BroadcastOperation>,
    C::Value: Reduce + Select + DynamicSlice + Reshape + ConvertElementType + ZeroLike,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[<ArrayBatchingPolicy<P> as BatchingPolicy<C>>::Batch],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        let extent = P::axis_dimension(context)?;
        self.batch_inputs(
            context,
            driver,
            inputs,
            extent.clone(),
            Ok,
            |value| value,
            |r#type| {
                if extent.value().is_none() {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "dynamic mapped assertions require the mixed array batching policy".to_owned(),
                    });
                }
                let mut outputs = context.parent().bind(IotaOperation::new(r#type.clone(), 0)?, Vec::new(), &[])?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            },
            |condition| {
                if extent.bounds().lower() == 0 {
                    let condition = P::match_axis(context, &ArrayBatch::replicated(condition), Axis::from(0))?;
                    return Ok(condition.value().reduce(&[0], ReductionKind::All));
                }
                Ok(condition)
            },
            Ok,
            |inputs| {
                let condition = P::match_axis(context, &inputs[0], Axis::from(0))?.value().clone();
                let condition_type = condition.r#type().into_owned();
                let mut arguments = vec![condition];
                for input in &inputs[1..] {
                    let value = if input.batch_axis().is_replicated() {
                        input.value().clone()
                    } else {
                        P::match_axis(context, input, Axis::from(0))?.value().clone()
                    };
                    let input_type = value.r#type();
                    if input_type.rank() == 0 {
                        arguments.push(value);
                        continue;
                    }
                    let output_axes = if input.batch_axis().is_replicated() {
                        (1..=input_type.rank()).collect::<Vec<_>>()
                    } else {
                        (0..input_type.rank()).collect::<Vec<_>>()
                    };
                    let output_type = condition_type.clone().with_data_type(input_type.data_type());
                    let mut outputs = context.parent().bind(
                        BroadcastOperation::new(output_type, output_axes),
                        Vec::new(),
                        &[value],
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    arguments.push(outputs.remove(0));
                }
                Ok(arguments)
            },
        )
    }
}

impl<C: AssertionContext<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy>
    for AssertOperation<ArrayIrType>
where
    C::Operation: From<Self>
        + From<ConstantOperation<DimensionValue>>
        + From<IotaOperation<ArrayType>>
        + From<ZeroOperation<ArrayType>>
        + From<DimensionAddOperation>
        + From<CompareOperation<ArrayIrType>>
        + From<PadOperation<ArrayIrType>>
        + From<DynamicBroadcastOperation>
        + From<DimensionSizeOperation>,
    C::Value: ValueProjection<
            ArrayType,
            Projected: Value<Type = ArrayType>
                           + Reduce
                           + Select
                           + DynamicSlice
                           + Reshape
                           + ConvertElementType
                           + ZeroLike
                           + Transpose,
        >,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[<ArrayIrBatchingPolicy as BatchingPolicy<C>>::Batch],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let extent = match context.axis_extent().r#type().as_ref() {
            ArrayIrType::Dimension(dimension) => dimension.to_dimension(),
            _ => return Err(TypeError::invalid("assertion batch extent must be a dimension").into()),
        };
        let mut padded_extent: Option<C::Value> = None;
        self.batch_inputs(
            context,
            driver,
            inputs,
            extent.clone(),
            |value| Ok(value.into_projected()?),
            C::Value::from_projected,
            |r#type| {
                let dimensions =
                    if extent.value().is_some() { Vec::new() } else { vec![context.axis_extent().clone()] };
                let mut outputs =
                    context.parent().bind(IotaOperation::new(r#type.clone(), 0)?, Vec::new(), &dimensions)?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0).into_projected()?)
            },
            |condition| {
                if extent.bounds().lower() > 0 {
                    return Ok(condition);
                }
                let mut zero =
                    context.parent().bind(ConstantOperation::new(DimensionValue::constant(0)?), Vec::new(), &[])?;
                check_count!("output", zero, 1, ProgramError);
                let mut empty = context.parent().bind(
                    CompareOperation::new(ComparisonDirection::Equal),
                    Vec::new(),
                    &[context.axis_extent().clone(), zero.remove(0)],
                )?;
                check_count!("output", empty, 1, ProgramError);
                let empty = empty.remove(0).into_projected()?;
                Select::select(&empty, &empty, &condition)
            },
            |input| {
                if extent.bounds().lower() > 0 {
                    return Ok(input);
                }
                let input_type = <&ArrayType>::try_from(input.r#type().as_ref())?.clone();
                let mut zero = context.parent().bind(
                    ZeroOperation::new(ArrayType::scalar(input_type.data_type()).with_memory(input_type.memory())),
                    Vec::new(),
                    &[],
                )?;
                check_count!("output", zero, 1, ProgramError);
                let padded_extent = match &padded_extent {
                    Some(value) => value.clone(),
                    None => {
                        let mut one = context.parent().bind(
                            ConstantOperation::new(DimensionValue::constant(1)?),
                            Vec::new(),
                            &[],
                        )?;
                        check_count!("output", one, 1, ProgramError);
                        let one = one.remove(0);
                        let operation = DimensionAddOperation::new(
                            <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?,
                            <&DimensionType>::try_from(one.r#type().as_ref())?,
                        )?;
                        let mut outputs =
                            context.parent().bind(operation, Vec::new(), &[context.axis_extent().clone(), one])?;
                        check_count!("output", outputs, 1, ProgramError);
                        let value = outputs.remove(0);
                        padded_extent = Some(value.clone());
                        value
                    }
                };
                let arguments = vec![input, zero.remove(0), padded_extent];
                let operation = PadOperation::<ArrayIrType>::new(vec![0], vec![1], vec![0])?
                    .with_input_types(&arguments.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
                let mut outputs = context.parent().bind(operation, Vec::new(), &arguments)?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            },
            |inputs| {
                let value = inputs[0].value().clone();
                let condition_type = <&ArrayType>::try_from(value.r#type().as_ref())?.clone();
                let mut dimensions = vec![context.axis_extent().clone()];
                let mapped_axis = inputs[0].batch_axis_position();
                for axis in (0..condition_type.rank()).filter(|axis| Some(*axis) != mapped_axis) {
                    let mut outputs = context.parent().bind(
                        DimensionSizeOperation::new(&condition_type, axis)?,
                        Vec::new(),
                        std::slice::from_ref(&value),
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    dimensions.push(outputs.remove(0));
                }
                let mut arguments = Vec::new();
                for input in inputs {
                    let mut value = input.value().clone();
                    let input_type = value.r#type();
                    let ArrayIrType::Array(input_type) = input_type.as_ref() else {
                        arguments.push(value);
                        continue;
                    };
                    let rank = input_type.rank();
                    if !arguments.is_empty() && rank == 0 {
                        arguments.push(value);
                        continue;
                    }
                    let output_axes = if let Some(mapped_axis) = input.batch_axis_position() {
                        if mapped_axis != 0 {
                            let permutation = std::iter::once(mapped_axis)
                                .chain((0..rank).filter(|axis| *axis != mapped_axis))
                                .collect::<Vec<_>>();
                            value = C::Value::from_projected(value.into_projected()?.transpose(permutation)?);
                        }
                        (0..rank).collect::<Vec<_>>()
                    } else {
                        (1..=rank).collect::<Vec<_>>()
                    };
                    let inputs = std::iter::once(value).chain(dimensions.iter().cloned()).collect::<Vec<_>>();
                    let mut outputs =
                        context.parent().bind(DynamicBroadcastOperation::new(output_axes), Vec::new(), &inputs)?;
                    check_count!("output", outputs, 1, ProgramError);
                    arguments.push(outputs.remove(0));
                }
                Ok(arguments)
            },
        )
    }
}

impl_non_differentiable_operation!(<T> AssertOperation<T> where T: Type + Into<ArrayIrType>);
impl_non_transposable_operation!(<T> AssertOperation<T> where T: Type + Into<ArrayIrType>);

/// Checks Boolean conditions, reporting a message and named scalar observations when they fail.
/// Eager execution reports failures immediately. Tracing elides known successes and retains failing or symbolic
/// conditions as ordered assertions. All input types are validated before folding or staging.
pub trait Assert: Sized {
    /// Requires this scalar Boolean value to be true, reporting `message` and `observations` on failure.
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError>;

    /// Requires every logical lane to be true, reporting at most `limit` failed lanes in row-major order.
    /// Observations must be scalar values or arrays with the condition's shape. Empty conditions pass vacuously.
    /// This bounds diagnostic output, not the amount of data inspected or transferred to the host.
    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError>;
}

impl Assert for Array {
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect());
        let input_types = std::iter::once(self.r#type().into_owned())
            .chain(observations.iter().map(|(_, input)| input.r#type().into_owned()))
            .collect::<Vec<_>>();
        operation.infer_output_types(&input_types, &[])?;
        if Concretizable::<bool>::concretize(self)? {
            return Ok(());
        }
        Err(operation
            .failure(observations.iter().map(|(_, input)| (Some(input.clone()), input.r#type().into_owned())))?
            .into())
    }

    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect())
            .with_failure_limit(limit);
        operation.check_lanes(self, observations)
    }
}

impl<A: AssertionValue<Type = ArrayType>> Assert for ArrayIrValue<A> {
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect());
        let input_types = std::iter::once(self.r#type().into_owned())
            .chain(observations.iter().map(|(_, input)| input.r#type().into_owned()))
            .collect::<Vec<_>>();
        operation.infer_output_types(&input_types, &[])?;
        if Concretizable::<bool>::concretize(self)? {
            return Ok(());
        }
        Err(operation
            .failure(observations.iter().map(|(_, input)| (Some(input.clone()), input.r#type().into_owned())))?
            .into())
    }

    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect())
            .with_failure_limit(limit);
        operation.check_lanes(self, observations)
    }
}

impl<V: Value> Assert for V
where
    V::Type: Into<ArrayIrType>,
    V::DispatchDomain: AssertionContext,
{
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect());
        let inputs = std::iter::once(self.clone())
            .chain(observations.iter().map(|(_, input)| input.clone()))
            .collect::<Vec<_>>();
        self.dispatch_domain().assert(operation, &inputs)
    }

    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect())
            .with_failure_limit(limit);
        let inputs = std::iter::once(self.clone())
            .chain(observations.iter().map(|(_, input)| input.clone()))
            .collect::<Vec<_>>();
        self.dispatch_domain().assert(operation, &inputs)
    }
}

/// Context dispatch for [`Assert`]. Tracing elides known successes and stages potential failures. Transform
/// contexts bind first so batching applies its empty-batch semantics, partial evaluation honors effect-folding
/// policy, and differentiation retains assertions only in the primal computation.
pub trait AssertionContext:
    Context<Type: Into<ArrayIrType>, Operation: From<AssertOperation<<Self as Domain>::Type>>>
{
    /// Applies an assertion to parent-owned inputs after validating its complete signature.
    fn assert(&self, operation: AssertOperation<Self::Type>, inputs: &[Self::Value]) -> Result<(), ProgramError> {
        operation
            .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        self.bind(operation, Vec::new(), inputs)?;
        Ok(())
    }
}

// Concrete execution reaches the eager value capability through ordinary operation interpretation.
impl<V: Value, O: Operation<Type = V::Type>> AssertionContext for EagerContext<V, O>
where
    Self: Context<Type = V::Type, Operation: From<AssertOperation<V::Type>>>,
    V::Type: Into<ArrayIrType>,
{
}

impl<C: AssertionContext, T: Type + Into<ArrayIrType>> AssertionContext for ProjectedContext<C, T>
where
    Self: Context<Type = T, Operation: From<AssertOperation<T>>>,
    C::Value: ValueProjection<T, Projected = <Self as Domain>::Value>,
{
    fn assert(&self, operation: AssertOperation<T>, inputs: &[Self::Value]) -> Result<(), ProgramError> {
        operation
            .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        self.parent().assert(
            AssertOperation {
                message: operation.message,
                labels: operation.labels,
                failure_limit: operation.failure_limit,
                marker: PhantomData,
            },
            &inputs.iter().cloned().map(C::Value::from_projected).collect::<Vec<_>>(),
        )
    }
}

impl<V: AssertionValue, O: Operation<Type = V::Type> + From<AssertOperation<V::Type>>, C> AssertionContext
    for TracingContext<V, O, C>
where
    V::Type: Into<ArrayIrType>,
{
    fn assert(&self, operation: AssertOperation<V::Type>, inputs: &[Self::Value]) -> Result<(), ProgramError> {
        operation.fold_or_bind(self, inputs)
    }
}

impl<C: Context> AssertionContext for NestedTracingContext<C>
where
    C::Type: Into<ArrayIrType>,
    C::Constant: AssertionValue,
    C::Operation: From<AssertOperation<C::Type>>,
{
    fn assert(&self, operation: AssertOperation<Self::Type>, inputs: &[Self::Value]) -> Result<(), ProgramError> {
        operation.fold_or_bind(self, inputs)
    }
}

impl<C: Context> AssertionContext for PartialEvaluationContext<C>
where
    Self: Context<Type = C::Type, Operation: From<AssertOperation<C::Type>>>,
    C::Type: Into<ArrayIrType>,
{
}

impl<C: Context, P: crate::batching::BatchingPolicy<C>> AssertionContext for BatchingContext<C, P>
where
    Self: Context<Type = C::Type, Operation: From<AssertOperation<C::Type>>>,
    C::Type: Into<ArrayIrType>,
{
}

impl<C: Context, P: DifferentiationPolicy<C>> AssertionContext for DifferentiationContext<C, P>
where
    Self: Context<Type = C::Type, Operation: From<AssertOperation<C::Type>>>,
    C::Type: Into<ArrayIrType>,
{
}

/// Concrete scalar values that can report observations for a failed [`AssertOperation`].
/// Implementations preserve unsigned integer ranges and render only the observed scalar value.
pub trait AssertionValue: Value + Concretizable<bool> {
    /// Renders the concrete scalar observation, rejecting unsupported representations.
    fn assertion_observation(&self) -> Result<String, ProgramError>;

    /// Materializes an array for logical lane inspection, or returns `None` for a non-array scalar value.
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError>;
}

impl AssertionValue for Array {
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        Ok(Some(self.clone()))
    }

    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self.r#type().data_type() {
            DataType::Boolean => Ok(Concretizable::<bool>::concretize(self)?.to_string()),
            data_type if data_type.is_integer() => Ok(Concretizable::<i128>::concretize(self)?.to_string()),
            DataType::BF16 => Ok(f32::from(Concretizable::<bf16>::concretize(self)?).to_string()),
            DataType::F16 => Ok(f32::from(Concretizable::<f16>::concretize(self)?).to_string()),
            DataType::F32 => Ok(Concretizable::<f32>::concretize(self)?.to_string()),
            DataType::F64 => Ok(Concretizable::<f64>::concretize(self)?.to_string()),
            _ => Err(ProgramError::Concretization {
                message: format!("unsupported assertion observation type `{}`", self.r#type()),
            }),
        }
    }
}

impl<A: AssertionValue<Type = ArrayType>> AssertionValue for ArrayIrValue<A> {
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        match self {
            Self::Array(array) => array.assertion_array(),
            Self::Dimension(_) => Ok(None),
            Self::Reference(_) => {
                Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() })
            }
        }
    }

    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Array(array) => array.assertion_observation(),
            Self::Dimension(dimension) => Ok(Concretizable::<usize>::concretize(dimension)?.to_string()),
            Self::Reference(_) => {
                Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() })
            }
        }
    }
}

impl<T: Type> AssertionValue for CaptureReference<T> {
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        Err(ProgramError::Concretization {
            message: "cannot inspect a captured assertion array before execution".to_owned(),
        })
    }

    fn assertion_observation(&self) -> Result<String, ProgramError> {
        Err(ProgramError::Concretization {
            message: "cannot inspect a captured assertion observation before execution".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayBatch, ArrayIrBatch, ArrayIrOperation, ArrayOperation, DimensionBounds, DimensionType, DimensionValue,
        Shape,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::StagingContext;
    use crate::macros::check_operation_type_inference;
    use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::ConditionOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialTracer, PartialValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramRenderingMode, Provenance, ProvenanceScope};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_assert() {
        let operation =
            AssertOperation::<ArrayType>::new("input must be \"valid\"\nnext").with_labels(vec!["value".to_owned()]);
        assert_eq!(operation.name(), ASSERT_OPERATION_NAME);
        assert_eq!(operation.message(), "input must be \"valid\"\nnext");
        assert_eq!(operation.labels(), &["value"]);
        assert_eq!(operation.failure_limit(), None);
        let bounded = operation.clone().with_failure_limit(NonZeroUsize::new(3).unwrap());
        assert_eq!(bounded.failure_limit(), NonZeroUsize::new(3));
        assert_eq!(bounded.effects(), operation.effects());
        assert!(bounded.to_string().contains("failure_limit=3"));
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(operation.to_string(), r#"assert [message="input must be \"valid\"\nnext", labels=["value"]]"#);
    }

    #[test]
    fn test_assert_rendering() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::Boolean));
        builder
            .add_instruction(
                AssertOperation::new("valid"),
                Vec::new(),
                vec![input],
                Some(Provenance::scope(ProvenanceScope::new("check"), Provenance::unknown())),
            )
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder], vec![]).unwrap();
        assert_eq!(
            std::fmt::from_fn(|formatter| program.render(formatter, 0, ProgramRenderingMode::WithEffectsAndProvenance))
                .to_string(),
            indoc! {r#"
                lambda %0:bool[] .
                let () = assert [message="valid", labels=[]] %0 ; effects=[ordered_assertion] ; provenance=check
                in ()"#},
        );
    }

    #[test]
    fn test_assert_type_inference() {
        check_operation_type_inference!(
            operation = AssertOperation::<ArrayType>::new("condition must hold"),
            cases = [{
                input_types = [ArrayType::scalar(DataType::Boolean)],
                output_types = [],
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }, {
                input_types = [ArrayType::scalar(DataType::I32)],
                error = "assertion condition must have type `bool[]` but has type `i32[]`",
            }, {
                input_types = [ArrayType::new_static(DataType::Boolean, [1])],
                error = "assertion condition must have type `bool[]` but has type `bool[1]`",
            }],
        );
        check_operation_type_inference!(
            operation = AssertOperation::<ArrayType>::new("condition must hold").with_labels(vec!["value".to_owned()]),
            cases = [{
                input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::U64)],
                output_types = [],
            }, {
                input_types = [ArrayType::scalar(DataType::Boolean)],
                error = "expected 2 inputs but got 1",
            }, {
                input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::I32, [1])],
                error = "assertion observation `value` has unsupported type `i32[1]`",
            }, {
                input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::C64)],
                error = "assertion observation `value` has unsupported type `c64[]`",
            }],
        );
    }

    #[test]
    fn test_assert_interpretation() {
        let context = EagerContext::<Array>::new();
        let operation = AssertOperation::new("condition must hold");
        assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[Array::scalar(true).unwrap()]), Ok(vec![]));
        let error = operation.interpret(&context, &EmptyRegionDriver, &[Array::scalar(false).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "condition must hold".to_owned(), observations: vec![] })
        );
        assert_eq!(error.to_string(), "assertion failed: condition must hold");
        let error = Array::scalar(false)
            .unwrap()
            .assert(
                "observations",
                &[
                    ("unsigned", Array::scalar(u64::MAX).unwrap()),
                    ("signed", Array::scalar(i64::MIN).unwrap()),
                    ("boolean", Array::scalar(true).unwrap()),
                    ("float", Array::scalar(1.5_f32).unwrap()),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "observations".to_owned(),
                observations: vec![
                    ("unsigned".to_owned(), u64::MAX.to_string()),
                    ("signed".to_owned(), i64::MIN.to_string()),
                    ("boolean".to_owned(), "true".to_owned()),
                    ("float".to_owned(), "1.5".to_owned()),
                ],
            })
        );
        assert_eq!(
            error.to_string(),
            "assertion failed: observations; unsigned=18446744073709551615, signed=-9223372036854775808, boolean=true, float=1.5",
        );
        // Even a true condition must validate unsupported observations before returning.
        assert_eq!(
            Array::scalar(true)
                .unwrap()
                .assert("bad observation", &[("vector", Array::vector(vec![1_i32]).unwrap())]),
            Err(TypeError::invalid("assertion observation `vector` has unsupported type `i32[1]`").into()),
        );
        let condition = ArrayIrValue::Array(Array::scalar(false).unwrap());
        let error = condition
            .assert("extent", &[("size", ArrayIrValue::Dimension(DimensionValue::constant(7).unwrap()))])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "extent".to_owned(),
                observations: vec![("size".to_owned(), "7".to_owned())],
            })
        );
    }

    #[test]
    fn test_assert_rejects_regions_before_folding() {
        let (_, region) =
            TracingContext::<Array, ArrayOperation<Array>>::trace(Ok, ArrayType::scalar(DataType::Boolean)).unwrap();
        let region = region.to_flat_program();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            context.bind(AssertOperation::new("true"), vec![region.clone()], &[Array::scalar(true).unwrap()]),
            Err(ProgramError::MalformedProgram(
                "operation `assert` declares no region slots but 1 regions were attached".to_owned()
            )),
        );
        let partial = PartialEvaluationContext::new(context);
        let condition = partial.lift(Array::scalar(true).unwrap()).unwrap();
        assert_eq!(
            partial.bind(AssertOperation::new("true"), vec![region.clone()], &[condition]).map(|_| ()),
            Err(ProgramError::MalformedProgram(
                "operation `assert` declares no region slots but 1 regions were attached".to_owned()
            )),
        );
        let batching = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 0);
        let condition = batching.lift(Array::scalar(false).unwrap()).unwrap();
        assert_eq!(
            batching.bind(AssertOperation::new("empty"), vec![region], &[condition]).map(|_| ()),
            Err(ProgramError::MalformedProgram(
                "operation `assert` declares no region slots but 1 regions were attached".to_owned()
            )),
        );
    }

    #[test]
    fn test_assert_with_limit() {
        let condition = Array::matrix(2, 2, vec![false, true, false, false]).unwrap();
        let observations = Array::matrix(2, 2, vec![10_i32, 20, 30, 40]).unwrap();
        let error = condition
            .assert_with_limit("valid", &[("value", observations)], NonZeroUsize::new(2).unwrap())
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedLanes {
                message: "valid".to_owned(),
                lanes: vec![
                    AssertionLane { index: vec![0, 0], observations: vec![("value".to_owned(), "10".to_owned())] },
                    AssertionLane { index: vec![1, 0], observations: vec![("value".to_owned(), "30".to_owned())] },
                ],
                omitted: 1,
            })
        );
        assert_eq!(
            error.to_string(),
            "assertion failed: valid; lane [0, 0], value=10; lane [1, 0], value=30; 1 additional failing lanes omitted"
        );
        assert!(
            Array::vector(Vec::<bool>::new())
                .unwrap()
                .assert_with_limit("empty", &[], NonZeroUsize::MIN)
                .is_ok()
        );
        assert!(Array::vector(vec![true, true]).unwrap().assert_with_limit("true", &[], NonZeroUsize::MIN).is_ok());
        assert!(
            Array::vector(vec![true, true])
                .unwrap()
                .assert_with_limit("invalid", &[("value", Array::vector(vec![1_i32]).unwrap())], NonZeroUsize::MIN)
                .is_err()
        );
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |condition| condition.assert_with_limit("valid", &[], NonZeroUsize::new(2).unwrap()),
            ArrayType::new(DataType::Boolean, [3].into()),
        )
        .unwrap();
        assert!(program.to_string().contains("failure_limit=2"));
        let error = program.interpret(Array::vector(vec![true, false, false]).unwrap()).unwrap_err();
        assert_eq!(error.to_string(), "assertion failed: valid; lane [1]; lane [2]");
    }

    #[test]
    fn test_assert_with_limit_batching() {
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                condition.assert_with_limit("valid", &[("value", observation)], NonZeroUsize::new(2).unwrap())
            },
            (Array::vector(vec![true, false, false]).unwrap(), Array::vector(vec![10_i32, 20, 30]).unwrap()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            (),
            None,
        );
        assert_eq!(output.unwrap_err().to_string(), "assertion failed: valid; lane [1], value=20; lane [2], value=30");
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                condition.assert_with_limit("replicated", &[("value", observation)], NonZeroUsize::MIN)
            },
            (Array::scalar(false).unwrap(), Array::vector(vec![10_i32, 20]).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (),
            None,
        );
        assert_eq!(
            output.unwrap_err().to_string(),
            "assertion failed: replicated; lane [0], value=10; 1 additional failing lanes omitted"
        );
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                condition.assert_with_limit("empty", &[("value", observation)], NonZeroUsize::MIN)
            },
            (Array::scalar(false).unwrap(), Array::vector(Vec::<i32>::new()).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (),
            None,
        );
        assert_eq!(output, Ok(()));
    }

    #[test]
    fn test_assert_with_limit_nested_batching() {
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                batch(
                    |(condition, observation)| {
                        condition.assert_with_limit("nested", &[("value", observation)], NonZeroUsize::new(2).unwrap())
                    },
                    (condition, observation),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    (),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (
                Array::matrix(2, 2, vec![true, false, false, false]).unwrap(),
                Array::matrix(2, 2, vec![10_i32, 20, 30, 40]).unwrap(),
            ),
            (BatchAxis::new(1), BatchAxis::new(1)),
            (),
            None,
        );
        assert_eq!(
            output.unwrap_err().to_string(),
            "assertion failed: nested; lane [0, 1], value=30; lane [1, 0], value=20; 1 additional failing lanes omitted"
        );
    }

    #[test]
    fn test_per_lane_validation_outputs() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |divisors| divisors.not_equal(&divisors.zero_like()?),
            ArrayType::new(DataType::I32, [3].into()),
        )
        .unwrap();
        assert!(program.effects().classes().is_empty());
        let outcomes = program.interpret(Array::vector(vec![2_i32, 0, 4]).unwrap()).unwrap();
        assert_eq!(outcomes.elements::<bool>().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn test_assert_program() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.assert("condition must hold", &[]),
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
            lambda %0:bool[] .
            let () = assert [message="condition must hold", labels=[]] %0
            in ()"#}
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(program.simplified().unwrap().instructions().len(), 1);
        let (_, folded) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.dispatch_domain().lift(Array::scalar(true).unwrap())?;
                condition.assert("always true", &[])
            },
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        assert!(folded.instructions().is_empty());
        let (_, failing) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.dispatch_domain().lift(Array::scalar(false).unwrap())?;
                condition.assert("always false", &[])
            },
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        assert_eq!(failing.instructions().len(), 1);
        let error = failing.interpret(Array::scalar(true).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "always false".to_owned(), observations: vec![] })
        );
    }

    #[test]
    fn test_assert_conditional_staging() {
        let (_, failing) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.dispatch_domain().lift(Array::scalar(false).unwrap())?;
                condition.assert("selected failure", &[])
            },
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        let (_, passing) =
            TracingContext::<Array, ArrayOperation<Array>>::trace(|_| Ok(()), ArrayType::scalar(DataType::Boolean))
                .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let failing = builder.import_program(failing.to_flat_program());
        let passing = builder.import_program(passing.to_flat_program());
        builder
            .add_instruction(ConditionOperation::new(), vec![failing, passing], vec![predicate, input], None)
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        assert_eq!(
            program.interpret(vec![Array::scalar(false).unwrap(), Array::scalar(true).unwrap()]),
            Ok(Vec::new())
        );
        let error = program.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(true).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "selected failure".to_owned(), observations: Vec::new() })
        );
    }

    #[test]
    fn test_assert_partial_evaluation() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.assert("condition must hold", &[]),
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        let program = program.to_flat_program();
        let residual =
            program.partially_evaluate(&[PartialValue::Unknown(ArrayType::scalar(DataType::Boolean))]).unwrap();
        assert_eq!(residual.program().instructions().len(), 1);
        let folded = program.partially_evaluate(&[PartialValue::Known(Array::scalar(true).unwrap())]).unwrap();
        assert!(folded.program().instructions().is_empty());
        let error = program.partially_evaluate(&[PartialValue::Known(Array::scalar(false).unwrap())]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "condition must hold".to_owned(), observations: vec![] })
        );
    }

    #[test]
    fn test_assert_partial_evaluation_staged_observation() {
        let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
        let partial = PartialEvaluationContext::new(parent.clone());
        let condition = partial.lift(Array::scalar(false).unwrap()).unwrap();
        let observation =
            PartialTracer::new(partial.clone(), partial.unknown_input(ArrayType::scalar(DataType::I32), 0));
        condition.assert("residual failure", &[("value", observation.clone())]).unwrap();
        drop((condition, observation));
        let evaluation = partial.into_evaluation(Vec::new()).unwrap();
        assert!(parent.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program().instructions().len(), 1);
        assert_eq!(evaluation.program().input_count(), 1);
        let error = evaluation.program().interpret(vec![Array::scalar(7i32).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "residual failure".to_owned(),
                observations: vec![("value".to_owned(), "7".to_owned())]
            })
        );
    }

    #[test]
    fn test_assert_dimension_observations() {
        let exact = DimensionType::new("exact", DimensionBounds::new(4, Some(5)).unwrap());
        let bounded = DimensionType::new("bounded", DimensionBounds::new(2, Some(8)).unwrap());
        let input_types = vec![
            ArrayIrType::from(ArrayType::scalar(DataType::Boolean)),
            exact.into(),
            bounded.clone().into(),
            ArrayType::scalar(DataType::I32).into(),
        ];
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let observations = input_types[1..].iter().map(|r#type| trace.input(r#type.clone())).collect::<Vec<_>>();
        let condition = trace.constant(ArrayIrValue::Array(Array::scalar(false).unwrap()));
        condition
            .assert(
                "dimension requirement",
                &[
                    ("exact", observations[0].clone()),
                    ("bounded", observations[1].clone()),
                    ("scalar", observations[2].clone()),
                ],
            )
            .unwrap();
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
        let expected = format!(
            "assertion failed: dimension requirement; exact=4, bounded=<unknown: `{bounded}`>, scalar=<unknown>",
        );

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        builder
            .add_instruction(
                AssertOperation::new("dimension requirement").with_labels(vec![
                    "exact".to_owned(),
                    "bounded".to_owned(),
                    "scalar".to_owned(),
                ]),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 4], vec![])
            .unwrap();
        let mut inputs = input_types.into_iter().map(PartialValue::Unknown).collect::<Vec<_>>();
        inputs[0] = PartialValue::Known(ArrayIrValue::Array(Array::scalar(false).unwrap()));
        assert_eq!(program.partially_evaluate(&inputs).unwrap_err().to_string(), expected);

        // Concrete observations take precedence over their wider declared bounds.
        inputs[2] = PartialValue::Known(ArrayIrValue::Dimension(DimensionValue::new(bounded, 6).unwrap()));
        assert_eq!(
            program.partially_evaluate(&inputs).unwrap_err().to_string(),
            "assertion failed: dimension requirement; exact=4, bounded=6, scalar=<unknown>",
        );
    }

    #[test]
    fn test_assert_partial_evaluation_preserves_failure_order() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let failure = builder.add_constant(Array::scalar(false).unwrap());
        builder.add_instruction(AssertOperation::new("first"), Vec::new(), vec![condition], None).unwrap();
        builder.add_instruction(AssertOperation::new("second"), Vec::new(), vec![failure], None).unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder], vec![]).unwrap();
        let residual =
            program.partially_evaluate(&[PartialValue::Unknown(ArrayType::scalar(DataType::Boolean))]).unwrap();
        assert_eq!(residual.program().instructions().len(), 2);
        for (condition, message) in [(false, "first"), (true, "second")] {
            let error = residual.program().interpret(vec![Array::scalar(condition).unwrap()]).unwrap_err();
            assert_eq!(
                error.downcast_custom::<AssertionError>(),
                Some(&AssertionError::Failed { message: message.to_owned(), observations: vec![] })
            );
        }
    }

    #[test]
    fn test_assert_composed_predicates_and_splicing() {
        let left = DimensionType::new("left", DimensionBounds::new(0, Some(10)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(0, Some(10)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left_input = builder.add_input(left.clone().into());
        let right_input = builder.add_input(right.clone().into());
        for (direction, right_input, message) in [
            (ComparisonDirection::Equal, left_input, "self equality"),
            (ComparisonDirection::LessThanOrEqual, right_input, "left <= right"),
            (ComparisonDirection::Equal, right_input, "left == right"),
        ] {
            let condition = builder
                .add_instruction(
                    CompareOperation::<ArrayIrType>::new(direction),
                    Vec::new(),
                    vec![left_input, right_input],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    AssertOperation::new(message).with_labels(vec!["left".to_owned(), "right".to_owned()]),
                    Vec::new(),
                    vec![condition, left_input, right_input],
                    None,
                )
                .unwrap();
        }
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder, Placeholder], vec![])
            .unwrap();
        // Raw Boolean types carry no truth, so the raw assertion remains effectful until predicate folding.
        assert_eq!(
            program
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == ASSERT_OPERATION_NAME)
                .count(),
            3
        );
        // Dead-code elimination conservatively retains raw Boolean assertions. Partial evaluation first discovers
        // the self-equality proof; its now-unused comparison can then disappear with the proven assertion.
        let residual = program
            .partially_evaluate(&[
                PartialValue::Unknown(left.clone().into()),
                PartialValue::Unknown(right.clone().into()),
            ])
            .unwrap();
        let simplified = residual.program().simplified().unwrap();
        assert_eq!(
            simplified
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == ASSERT_OPERATION_NAME)
                .map(|instruction| instruction.operation().to_string())
                .collect::<Vec<_>>(),
            vec![
                r#"assert [message="left <= right", labels=["left", "right"]]"#,
                r#"assert [message="left == right", labels=["left", "right"]]"#,
            ]
        );
        for (left_extent, right_extent, message) in [(7, 3, "left <= right"), (3, 7, "left == right")] {
            let error = simplified
                .interpret(vec![
                    ArrayIrValue::Dimension(DimensionValue::new(left.clone(), left_extent).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::new(right.clone(), right_extent).unwrap()),
                ])
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<AssertionError>(),
                Some(&AssertionError::Failed {
                    message: message.to_owned(),
                    observations: vec![
                        ("left".to_owned(), left_extent.to_string()),
                        ("right".to_owned(), right_extent.to_string())
                    ],
                })
            );
        }
        let renamed_left = DimensionType::new("renamed_left", left.bounds());
        let renamed_right = DimensionType::new("renamed_right", right.bounds());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = [builder.add_input(renamed_left.clone().into()), builder.add_input(renamed_right.clone().into())];
        let outputs = builder.splice_program(&simplified, &inputs).unwrap();
        let imported = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![],
            )
            .unwrap();
        assert_eq!(imported.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let error = imported
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(renamed_left, 7).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(renamed_right, 3).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "left <= right".to_owned(),
                observations: vec![("left".to_owned(), "7".to_owned()), ("right".to_owned(), "3".to_owned())],
            })
        );
    }

    #[test]
    fn test_assert_batching() {
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| condition.assert("positive", &[("value", observation)]),
            (Array::vector(vec![true, false, false]).unwrap(), Array::vector(vec![10_i32, 20, 30]).unwrap()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            (),
            None,
        );
        let BatchingError::Program(error) = output.unwrap_err() else { panic!("expected an assertion failure") };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "positive".to_owned(),
                observations: vec![("value".to_owned(), "20".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
            })
        );
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| condition.assert("empty", &[("value", observation)]),
            (Array::scalar(false).unwrap(), Array::vector(Vec::<i32>::new()).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (),
            None,
        );
        assert_eq!(output, Ok(()));
        let output: Result<(), BatchingError> = batch(
            |condition| condition.assert("all true", &[]),
            Array::vector(vec![true, true]).unwrap(),
            BatchAxis::new(0),
            (),
            None,
        );
        assert_eq!(output, Ok(()));
    }

    #[test]
    fn test_assert_nested_batching() {
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                batch(
                    |(condition, observation)| condition.assert("nested", &[("batch_index", observation)]),
                    (condition, observation),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    (),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (
                Array::matrix(2, 3, vec![true, true, true, true, false, false]).unwrap(),
                Array::matrix(2, 3, vec![10_i32, 20, 30, 40, 50, 60]).unwrap(),
            ),
            (BatchAxis::new(1), BatchAxis::new(1)),
            (),
            None,
        );
        let BatchingError::Program(error) = output.unwrap_err() else { panic!("expected an assertion failure") };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "nested".to_owned(),
                observations: vec![
                    ("batch_index".to_owned(), "50".to_owned()),
                    ("batch_index_1".to_owned(), "1".to_owned()),
                    ("batch_index_2".to_owned(), "1".to_owned()),
                ],
            })
        );
    }

    #[test]
    fn test_assert_with_limit_dynamic_batching() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.to_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.dispatch_domain(), extent);
                AssertOperation::new("dynamic")
                    .with_labels(vec!["value".to_owned()])
                    .with_failure_limit(NonZeroUsize::new(2).unwrap())
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayIrBatch::new(condition, BatchAxis::new(0))?,
                            ArrayIrBatch::new(observation, BatchAxis::new(0))?,
                        ],
                    )?;
                Ok(())
            },
            (
                ArrayIrType::Dimension(extent_type.clone()),
                ArrayIrType::Array(ArrayType::new(DataType::Boolean, shape.clone())),
                ArrayIrType::Array(ArrayType::new(DataType::I32, shape)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<bool>::new()).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<i32>::new()).unwrap()),
            )),
            Ok(())
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 4).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true, false, false, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30, 40]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "assertion failed: dynamic; lane [1], value=20; lane [2], value=30; 1 additional failing lanes omitted"
        );
    }

    #[test]
    fn test_assert_dynamic_batching() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.to_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.dispatch_domain(), extent);
                AssertOperation::new("dynamic").with_labels(vec!["value".to_owned()]).batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[
                        ArrayIrBatch::new(condition, BatchAxis::new(0))?,
                        ArrayIrBatch::new(observation, BatchAxis::new(0))?,
                    ],
                )?;
                Ok(())
            },
            (
                ArrayIrType::Dimension(extent_type.clone()),
                ArrayIrType::Array(ArrayType::new(DataType::Boolean, shape.clone())),
                ArrayIrType::Array(ArrayType::new(DataType::I32, shape)),
            ),
        )
        .unwrap();
        assert!(!program.instructions().iter().any(|instruction| instruction.operation().name() == "sort"));
        assert!(program.instructions().iter().any(|instruction| instruction.operation().name() == "reduce_min"));
        for conditions in [vec![], vec![true], vec![true, true, true, true]] {
            let size = conditions.len();
            assert_eq!(
                program.interpret((
                    ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), size).unwrap()),
                    ArrayIrValue::Array(Array::vector(conditions).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![10_i32; size]).unwrap()),
                )),
                Ok(())
            );
        }
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true, false, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dynamic".to_owned(),
                observations: vec![("value".to_owned(), "20".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
            })
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 1).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![30_i32]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dynamic".to_owned(),
                observations: vec![("value".to_owned(), "30".to_owned()), ("batch_index".to_owned(), "0".to_owned())],
            })
        );
    }

    #[test]
    fn test_assert_dynamic_nested_batching() {
        let extent_type = DimensionType::new("inner", DimensionBounds::new(0, Some(4)).unwrap());
        let shape = Shape::new(vec![Dimension::Static(2), extent_type.to_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                batch(
                    |(extent, condition, observation)| {
                        let batching = BatchingContext::new(extent.dispatch_domain(), extent);
                        AssertOperation::new("nested dynamic").with_labels(vec!["value".to_owned()]).batch(
                            &batching,
                            &EmptyRegionDriver,
                            &[
                                ArrayIrBatch::new(condition, BatchAxis::new(0))?,
                                ArrayIrBatch::new(observation, BatchAxis::new(0))?,
                            ],
                        )?;
                        Ok(())
                    },
                    (extent, condition, observation),
                    (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::new(0)),
                    (),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (
                ArrayIrType::Dimension(extent_type.clone()),
                ArrayIrType::Array(ArrayType::new(DataType::Boolean, shape.clone())),
                ArrayIrType::Array(ArrayType::new(DataType::I32, shape)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 0, Vec::<bool>::new()).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 0, Vec::<i32>::new()).unwrap()),
            )),
            Ok(())
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![true, true, true, true, false, false]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_i32, 20, 30, 40, 50, 60]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "nested dynamic".to_owned(),
                observations: vec![
                    ("value".to_owned(), "50".to_owned()),
                    ("batch_index".to_owned(), "1".to_owned()),
                    ("batch_index_1".to_owned(), "1".to_owned()),
                ],
            })
        );
    }

    #[test]
    fn test_assert_dynamic_batching_replicated() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition)| {
                let batching = BatchingContext::new(extent.dispatch_domain(), extent.clone());
                AssertOperation::new("replicated").with_labels(vec!["size".to_owned()]).batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::replicated(condition), ArrayIrBatch::replicated(extent)],
                )?;
                Ok(())
            },
            (ArrayIrType::Dimension(extent_type.clone()), ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
            )),
            Ok(())
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
            )),
            Ok(())
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "replicated".to_owned(),
                observations: vec![("size".to_owned(), "3".to_owned())],
            })
        );
    }

    #[test]
    fn test_assert_dynamic_batching_replicated_predicate() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.dispatch_domain(), extent.clone());
                AssertOperation::new("mixed mapping")
                    .with_labels(vec!["value".to_owned(), "size".to_owned()])
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayIrBatch::replicated(condition),
                            ArrayIrBatch::new(observation, BatchAxis::new(0))?,
                            ArrayIrBatch::replicated(extent),
                        ],
                    )?;
                Ok(())
            },
            (
                ArrayIrType::Dimension(extent_type.clone()),
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::Array(ArrayType::new(DataType::U64, Shape::new(vec![extent_type.to_dimension()]))),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<u64>::new()).unwrap()),
            )),
            Ok(())
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 2).unwrap()),
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![u64::MAX, 1]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "mixed mapping".to_owned(),
                observations: vec![
                    ("value".to_owned(), u64::MAX.to_string()),
                    ("size".to_owned(), "2".to_owned()),
                    ("batch_index".to_owned(), "0".to_owned()),
                ],
            })
        );
    }

    #[test]
    fn test_assert_dynamic_batching_index_limit() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(i32::MAX as usize + 2)).unwrap());
        let extent = context.input(extent_type.clone().into());
        let predicate =
            context.input(ArrayType::new(DataType::Boolean, Shape::new(vec![extent_type.to_dimension()])).into());
        let batching = BatchingContext::new(context.clone(), extent);
        assert_eq!(
            AssertOperation::new("too large")
                .batch(&batching, &EmptyRegionDriver, &[ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap()],)
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation {
                message: "mapped assertion batch extent exceeds the supported `i32` index range".to_owned(),
            })
        );
        assert!(context.builder().borrow().instructions().is_empty());

        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(i32::MAX as usize + 1)).unwrap());
        let extent = context.input(extent_type.clone().into());
        let predicate =
            context.input(ArrayType::new(DataType::Boolean, Shape::new(vec![extent_type.to_dimension()])).into());
        let batching = BatchingContext::new(context.clone(), extent);
        let predicate = ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap();
        assert_eq!(
            AssertOperation::new("no room for padding")
                .with_labels(vec!["predicate".to_owned()])
                .batch(&batching, &EmptyRegionDriver, &[predicate.clone(), predicate.clone()])
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation {
                message: "padded assertion batch extent exceeds the supported `i32` extent range".to_owned(),
            })
        );
        assert!(context.builder().borrow().instructions().is_empty());
        // No diagnostic array is gathered, so this boundary does not require a padding lane.
        AssertOperation::new("no padding needed")
            .batch(&batching, &EmptyRegionDriver, &[predicate])
            .unwrap();
    }

    #[test]
    fn test_assert_projected_and_replicated_folding() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition = context.lift(ArrayIrValue::Array(Array::scalar(true).unwrap())).unwrap();
        ValueProjection::<ArrayType>::into_projected(condition)
            .unwrap()
            .assert("projected true", &[])
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let condition = context.lift(Array::scalar(true).unwrap()).unwrap();
        let batching = BatchingContext::new(context.clone(), 3);
        AssertOperation::new("replicated true")
            .batch(&batching, &EmptyRegionDriver, &[ArrayBatch::replicated(condition)])
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_assert_differentiation() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.assert("condition must hold", &[]),
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        let program = program.to_flat_program();
        let differentiated = program.jvp().unwrap();
        assert_eq!(differentiated.to_string(), program.to_string());
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization
                .primal()
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == ASSERT_OPERATION_NAME)
                .count(),
            1
        );
        assert!(
            linearization
                .tangent()
                .instructions()
                .iter()
                .all(|instruction| instruction.operation().name() != ASSERT_OPERATION_NAME)
        );
    }
}
