//! Operations that check Boolean conditions and declared type relationships at run time. Boolean assertions report a
//! message and named observations when a condition fails. Assertions are how programs state facts about values that the
//! type system cannot prove (e.g., that a divisor is non-zero, that indices lie within an axis, or that two runtime
//! dimensions agree). They come in the following shapes:
//!
//!   - The [`Assert`] value capability, whose [`assert`](Assert::assert) requires a scalar Boolean condition to hold
//!     and whose [`assert_with_limit`](Assert::assert_with_limit) requires every element of a Boolean array to hold,
//!     reporting at most a bounded number of failing elements in row-major order. Both take observations as
//!     `(label, value)` pairs that are rendered alongside the message on failure. Observations may be dimensions or
//!     Boolean, integer, and floating-point arrays, and with a failure limit they may also be arrays with the
//!     condition's shape, in which case each failing element reports its own observed values.
//!   - The [`AssertOperation`] that the capability stages, which consumes the condition and the observations and
//!     produces no outputs. It declares the [`EffectClass::OrderedAssertion`] effect, so dead-code elimination never
//!     removes it and separate assertions keep their relative order.
//!   - The [`GuardOperation`], which checks relationships required by declared input types (e.g., two inputs sharing a
//!     dimension variable must have equal extents). It preserves these input checks when a transform removes a program
//!     or region call, and returns each non-reference input with a type representation justified by the checks.
//!     It also declares [`EffectClass::OrderedAssertion`] to preserve the checks and their relative order.
//!
//! Eager execution reports failures immediately. The value capability omits validated conditions that resolve to
//! scalar `true`; all other conditions follow ordinary operation binding. Tracing stages potential failures for
//! execution, batching reports failing items, and differentiation retains assertions only in the primal computation.
//!
//! Assertions remain enabled independently of debug builds. A failing assertion does not guard the operations that
//! follow it within an eager computation, because the error is raised at the assertion itself. In a staged program,
//! the ordered effect guarantees that the failure is observed even when nothing consumes the values being checked.
//!
//! # Examples
//!
//! Eager assertions report their message and observations in the returned error:
//!
//! ```rust
//! # use ryft_core::{Array, Assert, AssertionError, Compare, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let left = Array::scalar(2i32)?;
//! let right = Array::scalar(3i32)?;
//! left.not_equal(&right)?.assert("left must differ from right", &[("left", left.clone())])?;
//! let error = left.equal(&right)?.assert("left must equal right", &[("left", left), ("right", right)]).unwrap_err();
//! assert_eq!(error.to_string(), "assertion failed: left must equal right; left=2, right=3");
//! assert!(matches!(error.downcast_custom::<AssertionError>(), Some(AssertionError::Failed { .. })));
//! # Ok(())
//! # }
//! ```
//!
//! Array conditions report a bounded sample of their failing elements, and traced assertions are staged as effectful
//! instructions that fail when the program is interpreted:
//!
//! ```rust
//! # use std::num::NonZeroUsize;
//! # use indoc::indoc;
//! # use ryft_core::{Array, ArrayOperation, ArrayType, Assert, DataType, ProgramError, Trace, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |condition| condition.assert_with_limit("all elements must hold", &[], NonZeroUsize::new(1).unwrap()),
//!     ArrayType::new_static(DataType::Boolean, [3]),
//! )?;
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {r#"
//!         lambda %0:bool[3] .
//!         let () = assert [message="all elements must hold", labels=[], failure_limit=1] %0
//!         in ()"#},
//! );
//! program.interpret(Array::vector(vec![true, true, true])?)?;
//! let error = program.interpret(Array::vector(vec![true, false, false])?).unwrap_err();
//! assert_eq!(
//!     error.to_string(),
//!     "assertion failed: all elements must hold; element [1]; 1 additional failing elements omitted",
//! );
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use half::{bf16, f16};

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy,
    ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionSource, DimensionType, DimensionValue,
    DynamicArrayExtentBatchingPolicy, StaticArrayExtentBatchingPolicy,
};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    BatchingPolicyProjection, reject_ragged_carriers,
};
use crate::captures::CaptureReference;
use crate::contexts::{Context, ContextProjection, Domain, ProjectedContext, ValueResolution};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_non_transposable_operation,
    impl_reference_dischargeable_operation,
};
use crate::operations::Capability;

use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;

use crate::operations::dimensions::dimension_add::DimensionAddOperation;
use crate::operations::manipulation::conversions::{CastOperation, ConvertElementType};

use crate::operations::manipulation::padding::PadOperation;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::DynamicSlice;
use crate::operations::reductions::{Reduce, ReductionKind};

use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Concretizable, EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProvider,
    ProgramError, ReferenceAccessMode, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceEffect, ReferenceMemberType, RegionInterface,
    Type, TypeError, TypeIdentity, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueDomainDispatch,
    ValueProjection,
};

/// Failure of a correctness assertion, including the named values observed at that assertion.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssertionError {
    /// The Boolean condition was false.
    Failed {
        /// Message that the assertion was created with, describing the fact that was required to hold (e.g.,
        /// `"left must equal right"`). It is reported verbatim after the `assertion failed: ` prefix.
        message: String,

        /// Named values observed when the condition failed, as `(label, value)` pairs in the order in which the
        /// assertion listed them, with each value rendered as text (e.g., `("left", "2")` and `("right", "3")`, which
        /// are reported as `left=2, right=3`). A dimension whose extent cannot be resolved when the failure is reported
        /// renders as `<unknown: ...>` with the dimension's type, and any other unresolvable value as `<unknown>`.
        observations: Vec<(String, String)>,
    },

    /// One or more logical elements of the condition were false, with a bounded diagnostic sample and the number
    /// of failures omitted from it.
    FailedElements {
        /// Message that the assertion was created with, describing the fact that was required to hold for every
        /// element (e.g., `"all elements must hold"`). It is reported verbatim after the `assertion failed: ` prefix.
        message: String,

        /// Failed elements in logical row-major order, holding at most as many elements as the assertion's failure
        /// limit (refer to [`Assert::assert_with_limit`]). For example, the condition `[true, false, false]` with a
        /// failure limit of `1` reports only the element at index `[1]`.
        failures: Vec<AssertionFailure>,

        /// Number of failed elements beyond the failure limit that [`failures`](Self::FailedElements::failures) does
        /// not report, which is `0` when every failed element is reported. For example, it is `1` for the condition
        /// `[true, false, false]` with a failure limit of `1`, since the element at index `[2]` is omitted.
        omitted: usize,
    },
}

impl AssertionError {
    /// Constructs a scalar [`AssertionError::Failed`] from a message and validated labeled observations, without
    /// checking the condition. Missing or non-concretizable dimension values use their extent or dimension type.
    /// Other unavailable values appear as `<unknown>` and other observation-rendering errors are propagated.
    fn from_observations<
        'a,
        V: AssertionValue<Type: Into<ArrayIrType>>,
        I: IntoIterator<Item = (&'a str, Option<V>, V::Type)>,
    >(
        message: &str,
        observations: I,
    ) -> Result<Self, ProgramError> {
        let observations = observations
            .into_iter()
            .map(|(label, input, r#type)| {
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
                Ok((label.to_owned(), observation))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(Self::Failed { message: message.to_owned(), observations })
    }
}

impl Display for AssertionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { message, observations } => {
                write!(formatter, "assertion failed: {message}")?;
                for (index, (label, value)) in observations.iter().enumerate() {
                    let separator = if index == 0 { "; " } else { ", " };
                    write!(formatter, "{separator}{label}={value}")?;
                }
                Ok(())
            }
            Self::FailedElements { message, failures, omitted } => {
                write!(formatter, "assertion failed: {message}")?;
                for failure in failures {
                    write!(formatter, "; element {:?}", failure.index)?;
                    for (label, value) in &failure.observations {
                        write!(formatter, ", {label}={value}")?;
                    }
                }
                if *omitted > 0 {
                    write!(formatter, "; {omitted} additional failing elements omitted")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AssertionError {}

impl From<AssertionError> for ProgramError {
    #[inline]
    fn from(error: AssertionError) -> Self {
        Self::custom(error)
    }
}

/// Coordinates and named observations for one failed logical element of an assertion condition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AssertionFailure {
    /// Logical row-major coordinates of the failed element, with one entry per axis of the condition (e.g., `[1, 0]`
    /// for the element in the second row and first column of a matrix condition).
    pub index: Vec<usize>,

    /// Named scalar values observed at the failed element, as `(label, value)` pairs in the order in which the
    /// assertion listed them. A scalar observation reports the same value for every failed element, while an
    /// observation with the condition's shape reports its own element at [`index`](Self::index) (e.g.,
    /// `("value", "-3")` for a negative entry that violated a non-negativity condition).
    pub observations: Vec<(String, String)>,
}

/// Canonical operation name for [`AssertOperation`].
pub const ASSERT_OPERATION_NAME: &str = "assert";

/// [`Operation`] that checks a Boolean condition and produces no outputs.
///
/// The first input is the condition and the remaining inputs are the observations named by [`labels`](Self::labels),
/// which are reported together with the message when the check fails. By default, the condition and the observations
/// are scalars. [`with_failure_limit`](Self::with_failure_limit) enables array conditions and bounded element
/// diagnostics, in which case array observations may instead have the condition's shape. Supported observations
/// are dimensions and Boolean, integer, `bf16`, `f16`, `f32`, and `f64` arrays. Labels and messages are literal
/// descriptions and do not refer to type identities.
///
/// Assertions remain enabled independently of debug builds. Every residual instruction declares
/// [`EffectClass::OrderedAssertion`], so separate assertions preserve separate ordered failures, while known
/// conditions are handled by [`Assert`] and partial evaluation. Default mapped batching reports the first failing
/// batch item at each batching level, with batch extents restricted to the `i32` index range, whereas bounded
/// reporting preserves all logical elements and reports a row-major sample. Mixed array programs support dynamic
/// extents, including empty batches, which pass vacuously. Materializing bounded element diagnostics rejects ragged
/// inputs because their padding is not logical data. Observations with a potentially empty mapped axis receive
/// one unused padding item so that diagnostic selection remains valid; the padded extent must also fit `i32`.
///
/// Refer to the documentation of [`Assert`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// Creates a new [`AssertOperation`] with the provided message and no diagnostic observations.
    #[inline]
    pub fn new<M: Into<String>>(message: M) -> Self {
        Self { message: message.into(), labels: Vec::new(), failure_limit: None, marker: PhantomData }
    }

    /// Returns this [`AssertOperation`] with its diagnostic labels set to the provided `labels`.
    #[inline]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Returns this [`AssertOperation`] with bounded element reporting enabled with the provided `limit`. Conditions
    /// may then be Boolean arrays and observations must be scalars or have the condition's shape. At most `limit`
    /// failed elements are reported in logical row-major order, along with the number omitted.
    #[inline]
    pub fn with_failure_limit(mut self, limit: NonZeroUsize) -> Self {
        self.failure_limit = Some(limit);
        self
    }

    /// Returns the literal message reported when the condition is false.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the labels for the diagnostic inputs following the condition, in input order.
    #[inline]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the maximum number of reported failed elements, or `None` for the default scalar assertion behavior.
    #[inline]
    pub fn failure_limit(&self) -> Option<NonZeroUsize> {
        self.failure_limit
    }
}

impl<T: Type + Into<ArrayIrType>> Display for AssertOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type + Into<ArrayIrType>> Operation for AssertOperation<T> {
    type Type = T;

    #[inline]
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

        let bounded = self.failure_limit.is_some();
        let condition = match input_types[0].clone().into() {
            ArrayIrType::Array(condition)
                if condition.data_type() == DataType::Boolean && (bounded || condition.rank() == 0) =>
            {
                condition
            }
            condition => {
                return Err(TypeError::invalid(format!(
                    "assertion condition must have type `{}` but has type `{}`",
                    if bounded { "bool[...]" } else { "bool[]" },
                    condition,
                )));
            }
        };

        for (label, input) in self.labels.iter().zip(&input_types[1..]) {
            let input: ArrayIrType = input.clone().into();
            let supported = match &input {
                ArrayIrType::Dimension(_) => true,
                ArrayIrType::Array(array) => {
                    (array.rank() == 0 || (bounded && array.shape() == condition.shape()))
                        && (array.data_type().is_integer()
                            || matches!(
                                array.data_type(),
                                DataType::Boolean | DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64,
                            ))
                }
                _ => false,
            };

            if !supported {
                return Err(TypeError::invalid(format!(
                    "assertion observation `{label}` has unsupported type `{input}`",
                )));
            }
        }

        Ok(Vec::new())
    }

    #[inline]
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

impl<C: Domain<Type: Into<ArrayIrType>, Value: Assert>> InterpretableOperation<C> for AssertOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
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

impl<C: Context<Type: Into<ArrayIrType>, Constant: AssertionValue, Operation: From<AssertOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for AssertOperation<C::Type>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Validate the complete signature before inspecting the condition so even a known success cannot hide
        // invalid observations or attached regions.
        check_count!("input", inputs, 1 + self.labels.len(), ProgramError);
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;

        // Elementwise assertions need the ordinary execution path to collect failing indices and observations.
        // The default policy decides whether to bind them in the parent or retain them in the residual program.
        if self.failure_limit.is_some() {
            return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
        }

        // A known input can still be symbolic in the parent context. Inspect it only if it resolves to a constant.
        if let Some(condition) = inputs[0].as_known()
            && let ValueResolution::Constant(condition) = context.parent().resolve(condition)
        {
            match condition.concretize() {
                Ok(true) => {
                    // A successful assertion has no observable effect and needs none of its diagnostic observations.
                    return Ok(Vec::new());
                }
                Ok(false) => {
                    // Raise immediately only under eager execution when effect folding is allowed and no earlier
                    // deferred ordered effect must run first. Otherwise, let the default policy place the assertion.
                    if !context.parent().is_eager() || !context.can_fold_effects(self.effects().classes()) {
                        return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
                    }
                }
                Err(ProgramError::Concretization { .. }) => {
                    // An unavailable concrete Boolean is not an assertion failure; defer to ordinary binding.
                    return context.fold_or_residualize(self.clone(), Vec::new(), inputs);
                }
                Err(error) => return Err(error),
            }

            // The condition is false and may fail now. Render resolved observations, using type information or
            // unknown placeholders for observations that are still symbolic rather than requiring their execution.
            return Err(AssertionError::from_observations(
                &self.message,
                self.labels.iter().zip(&inputs[1..]).map(|(label, input)| {
                    let value = input.as_known().and_then(|input| match context.parent().resolve(input) {
                        ValueResolution::Constant(input) => Some(input),
                        _ => None,
                    });
                    (label.as_str(), value, input.r#type().into_owned())
                }),
            )?
            .into());
        }

        // Without a concrete condition, preserve the assertion through the default partial evaluation policy.
        context.fold_or_residualize(self.clone(), Vec::new(), inputs)
    }
}

impl<C, P> BatchableOperation<C, ArrayBatchingPolicy<P>> for AssertOperation<ArrayType>
where
    C: Context<
            Type = ArrayType,
            Value: Assert + Reduce + Select + DynamicSlice + Reshape + ConvertElementType + ZeroLike,
        >,
    P: ArrayExtentBatchingPolicy<C> + ArrayAssertionBatchingPolicy<C, P>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Validate the per-item signature before eliminating an empty batch or changing the input shapes.
        self.infer_output_types(
            &inputs
                .iter()
                .map(|input| <ArrayBatchingPolicy<P> as BatchingPolicy<C>>::unbatched_type(input).into_owned())
                .collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;

        let Some(assertion) = BatchedAssertion::from_inputs(self, context, inputs)? else {
            return Ok(Vec::new().into());
        };

        // Delegate the assertion to the parent so any enclosing transforms apply their own rules.
        let labels = self.labels.iter().chain(&assertion.batch_index_label).map(String::as_str);
        let observations = labels.zip(assertion.observations).collect::<Vec<_>>();
        match self.failure_limit {
            Some(limit) => assertion.condition.assert_with_limit(&self.message, &observations, limit)?,
            None => assertion.condition.assert(&self.message, &observations)?,
        }

        Ok(Vec::new().into())
    }
}

impl<C: ContextProjection<ArrayType, Type = ArrayIrType, Value: Assert>> BatchableOperation<C, ArrayIrBatchingPolicy>
    for AssertOperation<ArrayIrType>
where
    <C::Value as ValueProjection<ArrayType>>::Projected:
        ZeroLike + ConvertElementType + Select + DynamicSlice + Reshape + Reduce,
    DynamicArrayExtentBatchingPolicy: ArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>, Extent = C::Value>
        + ArrayAssertionBatchingPolicy<ProjectedContext<C, ArrayType>, DynamicArrayExtentBatchingPolicy>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // Validate the per-item signature before eliminating an empty batch or changing the input shapes.
        self.infer_output_types(
            &inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;

        // Run the shared rule over the array view of this batching level, which shares its first-class extent. A
        // replicated dimension has no array view, but it is shared by every batch item, so it is reinserted unchanged
        // at its own position. Every other input crosses into that view through the policy projection, which owns the
        // array view of each carrier kind: a mapped dimension's packed value holds one extent per batch item and is
        // selected or materialized like an integer observation, and bounded ragged axes are kept so that
        // materialization still rejects them.
        let projected = inputs
            .iter()
            .map(|input| !matches!(input.value().r#type().as_ref(), ArrayIrType::Dimension(_)))
            .collect::<Vec<_>>();
        let array_inputs = inputs
            .iter()
            .zip(&projected)
            .filter(|(_, projected)| **projected)
            .map(|(input, _)| <ArrayIrBatchingPolicy as BatchingPolicyProjection<C, ArrayType>>::project_batch(input))
            .collect::<Result<Vec<_>, _>>()?;
        let projected_context =
            BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(context.parent().clone()),
                context.axis_extent().clone(),
            )
            .with_axis_name(context.axis_name().map(str::to_owned))
            .with_axis_sharding(context.axis_sharding().clone());
        let Some(assertion) = BatchedAssertion::from_inputs(self, &projected_context, &array_inputs)? else {
            return Ok(Vec::new().into());
        };
        let mut array_observations = assertion.observations.into_iter().map(C::Value::from_projected);
        let mut observations = Vec::with_capacity(inputs.len());
        for (input, projected) in inputs[1..].iter().zip(&projected[1..]) {
            observations.push(if *projected { array_observations.next().unwrap() } else { input.value().clone() });
        }
        observations.extend(array_observations);

        // Delegate the assertion to the parent so any enclosing transforms apply their own rules.
        let condition = C::Value::from_projected(assertion.condition);
        let labels = self.labels.iter().chain(&assertion.batch_index_label).map(String::as_str);
        let observations = labels.zip(observations).collect::<Vec<_>>();
        match self.failure_limit {
            Some(limit) => condition.assert_with_limit(&self.message, &observations, limit)?,
            None => condition.assert(&self.message, &observations)?,
        }
        Ok(Vec::new().into())
    }
}

impl_non_differentiable_operation!(<T> AssertOperation<T> where T: Type + Into<ArrayIrType>);
impl_non_transposable_operation!(<T> AssertOperation<T> where T: Type + Into<ArrayIrType>);

// TODO(eaplatanios): Review this.
/// Prepares array assertions for an [`ArrayExtentBatchingPolicy`]. An extent policy implements this trait for itself
/// to support assertion batching, where `P` is the extent policy of the supplied batching context.
///
/// One shared rule batches [`AssertOperation`]s over the array universe. It validates per-item types, skips statically
/// empty batches, and either selects the first failing batch item or, with a failure limit, materializes the inputs
/// along the batch axis, before delegating the assertion to the parent context. This trait supplies the batch item
/// indices and the empty-batch handling that the rule uses, while the rule materializes inputs through the extent
/// policy's own [`match_axis`](ArrayExtentBatchingPolicy::match_axis) and
/// [`broadcast_input`](ArrayExtentBatchingPolicy::broadcast_input). [`ArrayBatchingPolicy`] runs the rule with its own
/// extent policy. [`ArrayIrBatchingPolicy`] runs it over the array inputs of its projected
/// [`DynamicArrayExtentBatchingPolicy`] view, whose implementation of this trait binds the extent computations in the
/// composite parent context, and then reinserts replicated dimension observations, which have no array view.
///
/// [`StaticArrayExtentBatchingPolicy`] implements this trait for every array context and
/// [`DynamicArrayExtentBatchingPolicy`] for projected composite contexts. Both implementations are generic over `P`, so
/// a custom extent policy can reuse either one for its context without restricting its extent's Rust type. Individual
/// ordinary operation rules and guard/cast batching do not require this trait. An operation family's generated batching
/// implementation still requires support for every variant, including assertions when present.
///
/// Implementations must preserve logical extents, memory placement, and manual variation. They must reject unsupported
/// ragged inputs rather than report padding as logical assertion elements.
pub trait ArrayAssertionBatchingPolicy<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> {
    /// Returns the index of every batch item along the leading axis of `r#type`. Its extent must describe the logical
    /// batch, including when that extent is supplied as a first-class dimension value.
    fn batch_item_indices(
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        r#type: &ArrayType,
    ) -> Result<C::Value, ProgramError>;

    /// Makes the scalar `condition` true when the batch is empty. This gives a replicated false condition the same
    /// empty-batch semantics as a mapped condition reduced over zero items.
    fn mask_empty_batch_condition(
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        condition: C::Value,
    ) -> Result<C::Value, ProgramError>;

    /// Makes selecting item zero from a mapped observation safe when the batch may be empty. Static policies can
    /// return `observation` unchanged; dynamic policies pad it. `padded_extent` caches the padded extent across all
    /// observations of one assertion, avoiding repeated staging of the same extent calculation.
    fn pad_empty_batch_observation(
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        observation: C::Value,
        padded_extent: &mut Option<P::Extent>,
    ) -> Result<C::Value, ProgramError>;
}

// TODO(eaplatanios): Review this.
impl<
    C: Context<Type = ArrayType, Value: Reduce, Operation: From<IotaOperation<ArrayType>>>,
    P: ArrayExtentBatchingPolicy<C>,
> ArrayAssertionBatchingPolicy<C, P> for StaticArrayExtentBatchingPolicy
{
    fn batch_item_indices(
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        r#type: &ArrayType,
    ) -> Result<C::Value, ProgramError> {
        if P::axis_dimension(context)?.value().is_none() {
            return Err(ProgramError::UnsupportedOperation {
                message: "dynamic mapped assertions require the mixed array batching policy".to_owned(),
            });
        }
        let mut outputs = context.parent().bind(IotaOperation::new(r#type.clone(), 0)?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn mask_empty_batch_condition(
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        condition: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let extent = P::axis_dimension(context)?;
        if extent.bounds().lower() == 0 {
            let condition = P::match_axis(context, &ArrayBatch::replicated(condition), Axis::from(0))?;
            return condition.value().reduce(&[0], ReductionKind::All);
        }
        Ok(condition)
    }

    #[inline]
    fn pad_empty_batch_observation(
        _context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        observation: C::Value,
        _padded_extent: &mut Option<P::Extent>,
    ) -> Result<C::Value, ProgramError> {
        // The shared rule eliminates a statically empty batch before selecting observations, so every remaining
        // mapped observation has at least one item.
        Ok(observation)
    }
}

// TODO(eaplatanios): Review this.
// A projected dynamic batch uses the same first-class extent as its composite parent. That extent and the dimensions
// derived from it exist only as composite dimension values, so these functions bind their extent computations in the
// composite parent context and project the resulting arrays back into the array view.
impl<C, V: Value<Type = ArrayIrType>, P> ArrayAssertionBatchingPolicy<ProjectedContext<C, ArrayType>, P>
    for DynamicArrayExtentBatchingPolicy
where
    C: ContextProjection<
            ArrayType,
            Type = ArrayIrType,
            Value = V,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<ZeroOperation<ArrayType>>
                           + From<IotaOperation<ArrayType>>
                           + From<DimensionAddOperation>
                           + From<CompareOperation<ArrayIrType>>
                           + From<PadOperation<ArrayIrType>>,
        >,
    V: ValueProjection<ArrayType, Projected: Select>,
    P: ArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>, Extent = C::Value>,
{
    fn batch_item_indices(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<P>>,
        r#type: &ArrayType,
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let extent = P::axis_dimension(context)?;
        let dimensions = if extent.value().is_some() { Vec::new() } else { vec![context.axis_extent().clone()] };
        let mut outputs =
            context.parent().parent().bind(IotaOperation::new(r#type.clone(), 0)?, Vec::new(), &dimensions)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0).into_projected()?)
    }

    fn mask_empty_batch_condition(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<P>>,
        condition: <V as ValueProjection<ArrayType>>::Projected,
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let extent = P::axis_dimension(context)?;
        if extent.bounds().lower() > 0 {
            return Ok(condition);
        }
        let parent = context.parent().parent();
        let mut zero = parent.bind(ConstantOperation::new(DimensionValue::constant(0)?), Vec::new(), &[])?;
        check_count!("output", zero, 1, ProgramError);
        let mut empty = parent.bind(
            CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
            Vec::new(),
            &[context.axis_extent().clone(), zero.remove(0)],
        )?;
        check_count!("output", empty, 1, ProgramError);
        let empty = empty.remove(0).into_projected()?;
        empty.select(&empty, &condition)
    }

    fn pad_empty_batch_observation(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<P>>,
        observation: <V as ValueProjection<ArrayType>>::Projected,
        padded_extent: &mut Option<P::Extent>,
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let extent = P::axis_dimension(context)?;
        if extent.bounds().lower() > 0 {
            return Ok(observation);
        }

        // The padding zero is created with the observation's memory space and manual variation, which `pad` requires
        // of its padding value. This skips a `parallel_vary` transition, which is sound only because assertions are
        // not differentiable, so the padding never carries a tangent.
        let parent = context.parent().parent();
        let padding_type = observation.r#type().scalar_like()?;
        let mut zero = parent.bind(ZeroOperation::new(padding_type), Vec::new(), &[])?;
        check_count!("output", zero, 1, ProgramError);
        let padded_extent = match padded_extent.clone() {
            Some(value) => value,
            None => {
                let mut one = parent.bind(ConstantOperation::new(DimensionValue::constant(1)?), Vec::new(), &[])?;
                check_count!("output", one, 1, ProgramError);
                let one = one.remove(0);
                let operation = DimensionAddOperation::new(
                    <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?,
                    <&DimensionType>::try_from(one.r#type().as_ref())?,
                )?;
                let mut outputs = parent.bind(operation, Vec::new(), &[context.axis_extent().clone(), one])?;
                check_count!("output", outputs, 1, ProgramError);
                let value = outputs.remove(0);
                *padded_extent = Some(value.clone());
                value
            }
        };
        let arguments = vec![V::from_projected(observation), zero.remove(0), padded_extent];
        let operation = PadOperation::<ArrayIrType>::new(vec![0], vec![1], vec![0])?
            .with_input_types(&arguments.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let mut outputs = parent.bind(operation, Vec::new(), &arguments)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0).into_projected()?)
    }
}

/// Represents the ability to assert Boolean conditions in programs, reporting a message and named observations on
/// failure. Concrete values are checked immediately; context-carrying values apply an [`AssertOperation`] through
/// their context. Tracing omits known scalar successes and stages failing or symbolic conditions for execution.
/// Assertions produce no outputs and remain enabled independently of debug builds. Staged assertions declare
/// [`EffectClass::OrderedAssertion`] to preserve their failure order and survive dead-code elimination.
///
/// [`assert`](Self::assert) checks a scalar condition, while [`assert_with_limit`](Self::assert_with_limit) checks
/// every element of an array condition and bounds the number of reported failures. Observations may be dimensions
/// or Boolean, integer, `bf16`, `f16`, `f32`, and `f64` values. All input types are validated even when the condition
/// is known to pass.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
#[capability]
pub trait Assert<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Requires this scalar Boolean value to be true. A failure reports the message and named scalar observations.
    /// Under batching, reports the first failing item at each batching level and includes its batch index.
    ///
    /// # Parameters
    ///
    ///   - `message`: Literal description of the requirement, included in the failure diagnostic.
    ///   - `observations`: Label-value pairs of supported scalar observations to report on failure, in order.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if an input has an invalid type or the assertion cannot be applied in this value's
    /// context. An executed false condition returns an [`AssertionError`] wrapped in a [`ProgramError`] while a staged
    /// assertion reports that failure when the program executes.
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError>;

    /// Requires every logical element of this Boolean value to be true. Empty conditions pass. A failure reports
    /// up to `limit` failing indices in row-major order, their observations, and the number of omitted failures.
    /// Batching preserves individual elements for reporting instead of selecting only the first failing item.
    ///
    /// # Parameters
    ///
    ///   - `message`: Literal description of the requirement, included in the failure diagnostic.
    ///   - `observations`: Label-value pairs of supported observations to report on failure, in order. Scalars are
    ///     reused for every failed element; arrays must have the condition's shape and are sampled at each failing index.
    ///   - `limit`: Maximum number of failed elements to report. Bounds diagnostic output, not the amount of data
    ///     inspected or transferred to the host.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if an input has an invalid type or shape, or the assertion cannot be applied in this
    /// value's context. Executed failures return [`AssertionError::FailedElements`] wrapped in a [`ProgramError`] while
    /// a staged assertion reports those failures when the program executes.
    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError>;
}

impl Assert for Array {
    #[inline]
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        self.check_assertion(message, observations, None)
    }

    #[inline]
    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError> {
        self.check_assertion(message, observations, Some(limit))
    }
}

impl<A: AssertionValue<Type = ArrayType>> Assert for ArrayIrValue<A> {
    #[inline]
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        self.check_assertion(message, observations, None)
    }

    #[inline]
    fn assert_with_limit(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        limit: NonZeroUsize,
    ) -> Result<(), ProgramError> {
        self.check_assertion(message, observations, Some(limit))
    }
}

impl<
    T: Type + Into<ArrayIrType>,
    V: Value<
            Type = T,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Constant: Concretizable<bool>, Operation: From<AssertOperation<T>>>,
        >,
> Assert<T> for V
{
    fn assert(&self, message: &str, observations: &[(&str, Self)]) -> Result<(), ProgramError> {
        let operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect());
        let inputs = std::iter::once(self.clone())
            .chain(observations.iter().map(|(_, input)| input.clone()))
            .collect::<Vec<_>>();
        operation
            .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let context = self.domain();

        // A validated success has no observable effect. False or unresolved conditions must reach normal binding
        // so batching can handle empty batches and partial evaluation can preserve effect ordering.
        if let ValueResolution::Constant(condition) = context.resolve(self) {
            match condition.concretize() {
                Ok(true) => return Ok(()),
                Ok(false) | Err(ProgramError::Concretization { .. }) => {}
                Err(error) => return Err(error),
            }
        }

        context.bind(operation, Vec::new(), &inputs)?;
        Ok(())
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
        operation
            .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let context = self.domain();

        // A validated success has no observable effect. False or unresolved conditions must reach normal binding
        // so batching can handle empty batches and partial evaluation can preserve effect ordering.
        if let ValueResolution::Constant(condition) = context.resolve(self) {
            match condition.concretize() {
                Ok(true) => return Ok(()),
                Ok(false) | Err(ProgramError::Concretization { .. }) => {}
                Err(error) => return Err(error),
            }
        }

        context.bind(operation, Vec::new(), &inputs)?;
        Ok(())
    }
}

/// Represents concrete values that can be inspected to check assertions and render their observations. Implementations
/// preserve unsigned integer ranges when rendering scalar observations and provide an array view for elementwise
/// checks. [`check_assertion`](Self::check_assertion) supplies the shared concrete execution used by [`Assert`],
/// without binding an [`AssertOperation`].
pub trait AssertionValue: Value + Concretizable<bool> {
    /// Returns this scalar value rendered as text for an assertion failure diagnostic, without its observation label or
    /// type (e.g., `"42"` or `"false"`). Integer values retain their exact value, including unsigned integer ranges.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the value is not scalar, its representation is unsupported, or its concrete
    /// value is unavailable (e.g., a captured value that cannot be inspected before execution).
    fn assertion_observation(&self) -> Result<String, ProgramError>;

    /// Returns this value as a concrete [`Array`] so assertion checks can inspect its logical elements. Array values,
    /// including rank-zero arrays, return [`Some`]. Non-array scalar values such as dimensions return [`None`] and can
    /// be rendered directly through [`assertion_observation`](Self::assertion_observation).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the representation is unsupported or the array cannot be inspected.
    /// An unavailable array is an error, not [`None`], which is reserved for non-array scalar observations.
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError>;

    /// Validates and checks this concrete Boolean condition without binding an operation. Scalar assertions report
    /// their named observations on failure; a failure limit enables elementwise checking with a bounded sample of
    /// failing indices and observations in row-major order. Empty array conditions pass.
    ///
    /// # Parameters
    ///
    ///   - `message`: Literal description of the requirement, included in the failure diagnostic.
    ///   - `observations`: Label-value pairs to report on failure. Scalars are reused for each failing element.
    ///     Array observations must have the condition's shape and require a failure limit.
    ///   - `failure_limit`: Maximum number of failing elements to report, or [`None`] to require a scalar condition
    ///     and scalar observations. Limits diagnostic output, not the amount of data inspected.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if input validation or concrete value inspection fails, or an [`AssertionError`]
    /// wrapped in a [`ProgramError`] if the condition fails.
    fn check_assertion(
        &self,
        message: &str,
        observations: &[(&str, Self)],
        failure_limit: Option<NonZeroUsize>,
    ) -> Result<(), ProgramError>
    where
        Self::Type: Into<ArrayIrType>,
    {
        // Reuse operation type inference so eager and staged assertions validate the same signature. Validate
        // observations before checking the condition, including when it succeeds or has no elements.
        let mut operation = AssertOperation::new(message)
            .with_labels(observations.iter().map(|(label, _)| (*label).to_owned()).collect());
        if let Some(limit) = failure_limit {
            operation = operation.with_failure_limit(limit);
        }
        let input_types = std::iter::once(self.r#type().into_owned())
            .chain(observations.iter().map(|(_, value)| value.r#type().into_owned()))
            .collect::<Vec<_>>();
        operation.infer_output_types(&input_types, &[])?;

        // Without a failure limit, validation requires a scalar condition and scalar observations. A success
        // needs no diagnostic values while a failure renders the observations through the shared error constructor.
        let Some(limit) = failure_limit else {
            if Concretizable::<bool>::concretize(self)? {
                return Ok(());
            }
            return Err(AssertionError::from_observations(
                message,
                observations.iter().map(|(label, input)| (*label, Some(input.clone()), input.r#type().into_owned())),
            )?
            .into());
        };

        // Validation guarantees an array condition. Count all failures so the diagnostic can report how many
        // were omitted, even though only the first `limit` failures are rendered. Empty arrays pass vacuously.
        let condition = self.assertion_array()?.unwrap();
        let values = condition.elements::<bool>()?;
        let failure_count = values.iter().filter(|value| !**value).count();
        if failure_count == 0 {
            return Ok(());
        }

        // Concrete array dimensions give the shape needed to turn flat element positions into logical indices.
        let shape = condition
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| dimension.value().unwrap())
            .collect::<Vec<_>>();

        // Materialize each array observation once, and only after finding a failure. Non-array scalar
        // observations retain their original representation for rendering at each failing index.
        let observations = observations
            .iter()
            .map(|(label, value)| Ok((*label, value, value.assertion_array()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let limit = limit.get();
        let mut failures = Vec::new();
        for position in values.iter().enumerate().filter_map(|(index, value)| (!value).then_some(index)).take(limit) {
            // Decode the flat row-major position from the last axis backwards. A failing element guarantees
            // that no dimension is empty; a rank-zero condition naturally has an empty index.
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
                    // Sample shaped observations at the failing index and reshape the single element to a scalar
                    // for rendering. Scalar arrays and non-array scalars are reused unchanged at every index.
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
            failures.push(AssertionFailure { index, observations });
        }

        Err(AssertionError::FailedElements {
            message: message.to_owned(),
            failures,
            omitted: failure_count.saturating_sub(limit),
        }
        .into())
    }
}

// Assertion partial evaluation requires `Constant: AssertionValue`, including in domains that store captures as
// constants. A `CaptureReference` carries only an index and type, so inspection always fails with `Concretization`.
// This implementation keeps captured programs eligible for partial evaluation. Unavailable observations use type
// information or `<unknown>` in diagnostics. Removing it would reject those domains at compile time even when no
// captured value needs to be inspected. Constant enums containing captures can also delegate their inspection here.
impl<T: Type> AssertionValue for CaptureReference<T> {
    #[inline]
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        Err(ProgramError::Concretization {
            message: "cannot inspect a captured assertion observation before execution".to_owned(),
        })
    }

    #[inline]
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        Err(ProgramError::Concretization {
            message: "cannot inspect a captured assertion array before execution".to_owned(),
        })
    }
}

impl AssertionValue for Array {
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

    #[inline]
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        Ok(Some(self.clone()))
    }
}

impl<A: AssertionValue<Type = ArrayType>> AssertionValue for ArrayIrValue<A> {
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Array(array) => array.assertion_observation(),
            Self::Dimension(dimension) => Ok(Concretizable::<usize>::concretize(dimension)?.to_string()),
            Self::Reference(_) => {
                Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() })
            }
        }
    }

    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        match self {
            Self::Array(array) => array.assertion_array(),
            Self::Dimension(_) => Ok(None),
            Self::Reference(_) => {
                Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() })
            }
        }
    }
}

/// Condition and observations that one batched [`AssertOperation`] forwards to its parent context,
/// as prepared by [`from_inputs`](Self::from_inputs).
struct BatchedAssertion<V> {
    /// Condition reduced to one scalar Boolean or, with a failure limit, materialized along the batch axis.
    condition: V,

    /// Observations in the order of the prepared inputs, followed by the index of the first failing batch item when
    /// [`batch_index_label`](Self::batch_index_label) is present.
    observations: Vec<V>,

    /// Label of the trailing batch index observation, which is unique among the labels of the batched operation.
    batch_index_label: Option<String>,
}

impl<V: Value<Type = ArrayType>> BatchedAssertion<V> {
    /// Prepares one assertion in the parent context for an [`AssertOperation`] batched over the array universe,
    /// returning `None` for a statically empty batch, which passes vacuously. This is the shared batching rule behind
    /// both batching entry points: [`ArrayBatchingPolicy`] applies it directly, and [`ArrayIrBatchingPolicy`] applies
    /// it to the array inputs of its projected [`DynamicArrayExtentBatchingPolicy`] view. Without a failure limit, the
    /// rule reduces the condition to one scalar Boolean and selects the observations of the first failing batch item,
    /// adding that item's index as an observation when any input is mapped. With a failure limit, it instead
    /// materializes the inputs along the batch axis so that the parent reports individual failing elements. `P`
    /// supplies the batch item indices and the empty-batch handling through [`ArrayAssertionBatchingPolicy`],
    /// and the axis alignment and broadcasts of that materialization through [`ArrayExtentBatchingPolicy`].
    ///
    /// The constructor neither validates the per-item signature nor asserts. Callers validate in their own universe
    /// first, and they assert afterwards, so that observations without an array view can rejoin the prepared ones.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Assertion being batched, whose failure limit selects the preparation and whose labels keep the
    ///     added batch index label unique.
    ///   - `context`: Active array [`BatchingContext`] whose parent context receives the prepared assertion.
    ///   - `inputs`: Condition followed by the array observations, in operation order.
    ///
    /// # Errors
    ///
    /// Returns [`BatchingError::UnsupportedOperation`] when selecting a mapped input's first failure needs a batch
    /// index or a padded extent outside the `i32` range, and propagates the errors of the
    /// [`ArrayAssertionBatchingPolicy`] functions and of the array capabilities used for the selection.
    fn from_inputs<
        T: Type + Into<ArrayIrType>,
        C: Context<Type = ArrayType, Value = V>,
        P: ArrayExtentBatchingPolicy<C> + ArrayAssertionBatchingPolicy<C, P>,
    >(
        operation: &AssertOperation<T>,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Option<Self>, BatchingError>
    where
        V: ZeroLike + ConvertElementType + Select + DynamicSlice + Reshape + Reduce,
    {
        // Empty batches pass even when the predicate is replicated and false. Check the extent before replaying.
        let extent = P::axis_dimension(context)?;
        if extent.value() == Some(0) {
            return Ok(None);
        }

        // Preserve individual conditions and observations so the parent can report multiple failing elements.
        // Materialization broadcasts replicated inputs and arranges mapped inputs along the batch axis.
        if operation.failure_limit.is_some() {
            // Padding in bounded ragged inputs is not logical data and must not be reported as failed elements.
            ArrayBatch::reject_ragged_inputs(&AssertOperation::<ArrayType>::new(""), inputs)?;

            // Move a mapped condition's batch axis to the front, or insert a leading batch axis for a replicated one.
            let condition = P::match_axis(context, &inputs[0], Axis::from(0))?.into_value();
            let condition_type = condition.r#type().into_owned();
            let condition_rank = condition_type.rank();

            // Every shaped observation takes the aligned condition's dimensions: the transform's extent along the
            // batch axis, exact static extents, and reads of the aligned condition for dynamic dimensions.
            let dimension_sources = condition_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, dimension)| match dimension {
                    _ if axis == 0 => DimensionSource::BatchExtent,
                    Dimension::Static(extent) => DimensionSource::Static(*extent),
                    _ => DimensionSource::Value { source: condition.clone(), axis },
                })
                .collect::<Vec<_>>();

            let mut observations = Vec::with_capacity(inputs.len() - 1);
            for input in &inputs[1..] {
                let input_type = input.r#type();
                let rank = input_type.rank();

                // Scalar observations are shared by every failing element and do not need broadcasting.
                if rank == 0 {
                    observations.push(input.value().clone());
                    continue;
                }

                // A replicated observation gains the leading batch axis. A mapped observation moves its batch axis
                // to the front in the same broadcast, so the mapping is not monotonic for a non-leading batch axis.
                let output_axes = match input.batch_axis_position() {
                    Some(mapped_axis) => (0..rank)
                        .map(|axis| match axis.cmp(&mapped_axis) {
                            Ordering::Less => axis + 1,
                            Ordering::Equal => 0,
                            Ordering::Greater => axis,
                        })
                        .collect::<Vec<_>>(),
                    None => (1..=rank).collect::<Vec<_>>(),
                };

                // Each device records its own observations, so an observation keeps its memory space, placement, and
                // manual variation, with its sharding mapped onto the condition's axes and any new axis replicated.
                // Assertions do not require their inputs to share manual variation, and a broadcast cannot change it.
                let output_sharding = input_type
                    .sharding()
                    .map(|sharding| sharding.with_broadcasted_dimensions(condition_rank, &output_axes))
                    .transpose()
                    .map_err(|error| ProgramError::from(TypeError::invalid(error.to_string())))?;
                let output_type = ArrayType::new(input_type.data_type(), condition_type.shape().clone())
                    .with_memory(input_type.memory())
                    .with_sharding(output_sharding)
                    .map_err(|error| ProgramError::from(TypeError::invalid(error.to_string())))?;
                let output = P::broadcast_input(
                    context,
                    input,
                    output_type,
                    output_axes,
                    Axis::from(0),
                    dimension_sources.clone(),
                )?;

                observations.push(output.into_value());
            }
            return Ok(Some(Self { condition, observations, batch_index_label: None }));
        }

        let mapped = inputs.iter().any(|input| input.batch_axis().axis().is_some());
        if !mapped {
            // Every item has the same condition and observations, so one assertion suffices. A dynamic extent
            // may still be zero at execution time; mask the condition so that case passes vacuously.
            let condition = P::mask_empty_batch_condition(context, inputs[0].value().clone())?;
            let observations = inputs[1..].iter().map(|input| input.value().clone()).collect();
            return Ok(Some(Self { condition, observations, batch_index_label: None }));
        }

        // Failure selection uses signed 32-bit indices. Check the extent's upper bound before staging that work.
        let (_, maximum) = extent.bounds().representable_extent_range().map_err(ProgramError::from)?;
        if maximum > i32::MAX as usize {
            return Err(BatchingError::UnsupportedOperation {
                message: "mapped assertion batch extent exceeds the supported `i32` index range".to_owned(),
            });
        }

        // Backends also represent logical padded extents with signed indices; reserve the fallback item before staging.
        if maximum == i32::MAX as usize
            && extent.bounds().lower() == 0
            && inputs[1..].iter().any(|input| input.batch_axis().axis().is_some())
        {
            return Err(BatchingError::UnsupportedOperation {
                message: "padded assertion batch extent exceeds the supported `i32` extent range".to_owned(),
            });
        }

        let condition = inputs[0].value();
        let (condition, index) = if inputs[0].batch_axis().axis().is_some() {
            let coordinates_type = condition.r#type().into_owned().with_data_type(DataType::I32);
            let coordinates = P::batch_item_indices(context, &coordinates_type)?;

            // Passing items may tie the last coordinate: when any item fails, the minimum still identifies the
            // first failure. Successful and empty batches are handled by the independent Boolean reduction.
            let sentinel = coordinates.reduce(&[0], ReductionKind::Max)?;
            let candidates = condition.select(&sentinel, &coordinates)?;
            let index = candidates.reduce(&[0], ReductionKind::Min)?;
            let condition = condition.reduce(&[0], ReductionKind::All)?;

            // Empty and successful batches use index zero. A padded observation makes this valid even at extent zero.
            let index = condition.select(&index.zero_like()?, &index)?;
            (condition, index)
        } else {
            // Only observations vary across items. A replicated false condition fails first at index zero,
            // while a dynamically empty batch must still pass.
            let condition = P::mask_empty_batch_condition(context, condition.clone())?;
            let index = condition.convert_element_type(DataType::I32)?.zero_like()?;
            (condition, index)
        };

        // Select each mapped observation at the first failing index and keep replicated observations unchanged.
        // Padding supplies a valid fallback element when the dynamic batch is empty, even though it passes.
        let mut observations = Vec::with_capacity(inputs.len());
        let mut padded_extent = None;
        for input in &inputs[1..] {
            if input.batch_axis().axis().is_some() {
                let input = P::pad_empty_batch_observation(context, input.value().clone(), &mut padded_extent)?;

                // The selected batch index is never negative, so the clamp-only policy keeps this slice on the single
                // gather path even when the mapped axis is dynamic.
                let observation =
                    input.dynamic_slice_with_negative_indices(std::slice::from_ref(&index), &[1], false)?;
                observations.push(observation.reshape([])?);
            } else {
                observations.push(input.value().clone());
            }
        }
        observations.push(index);

        // Include the selected index in the diagnostic without colliding with user labels or indices already
        // introduced by another batching level.
        let mut label = "batch_index".to_owned();
        let mut suffix = 1;
        while operation.labels.contains(&label) {
            label = format!("batch_index_{suffix}");
            suffix += 1;
        }
        Ok(Some(Self { condition, observations, batch_index_label: Some(label) }))
    }
}

// TODO(eaplatanios): Review from here onward.

/// Canonical operation name for [`GuardOperation`].
pub const GUARD_OPERATION_NAME: &str = "guard";

/// [`Operation`] that checks declared type-identity relationships between its inputs at runtime and returns the
/// non-reference inputs with type representations justified by a successful check. The supplied types may admit these
/// relationships without proving them. A guard preserves the check when a transform removes the program or region
/// boundary that would otherwise enforce it.
///
/// A boundary may relate its inputs through shared type identities. For example, a region declared as `[dimension<n>,
/// f32[n]]` requires its dimension input to equal the extent of its array input. When the region is applied to a
/// symbolic `dimension<m>` and an exact `f32[2]`, type inference admits the application (the dimension may be 2 at
/// runtime), but the region must check that `m = 2` before executing its body. Staging replay, partial evaluation, and
/// backend lowering can remove the region invocation and its input checks, so they emit a [`GuardOperation`] in its
/// place, before executing the guarded body. [`for_entry`](Self::for_entry) derives the guard from the declared and
/// actual input types. Boundaries whose actual types prove every relationship need no guard.
///
/// Operation families construct guards through [`OperationProvider`]. Families with an ordinary
/// `From<GuardOperation<T>>` conversion receive a blanket provider implementation; other families reject guard requests
/// with [`unsupported_family_error`](Self::unsupported_family_error) through a provider that `#[derive(Operation)]`
/// generates for a derived family without a native guard variant and that
/// [`impl_guardless_operation!`](crate::impl_guardless_operation) generates for a hand-written family. A family can
/// also implement the provider explicitly to accept only the requests it can represent. No guard-specific derive
/// attribute is needed.
///
/// The guard is a checked cast: it produces one output per non-reference input, which is that input at the type that
/// [`TypeRefinements::retype_actual`](crate::TypeRefinements::retype_actual) derives from the guard's declared types
/// and its [retained](Self::retained) identities (e.g., `dimension<2>` and `f32[2]` in the example above). That is the
/// input narrowed by the facts that the checked relationships prove, except that retained identities stay symbolic and
/// the other inputs of their classes take compatible representatives, according to the declared component's bounds and
/// the [`TypeRefinements::retype_actual`](crate::TypeRefinements::retype_actual) fallback rules. Code that specializes
/// the body using those types consumes the outputs instead of the inputs, so that the specialization only takes effect
/// after the check while the program or region still accepts the supplied input types. Consumers that only need the
/// check ignore the outputs. The guard declares [`EffectClass::OrderedAssertion`], so dead-code elimination never
/// removes it and it orders like an assertion. Backends additionally order every later ordered I/O effect after it,
/// but, like any assertion, it does not order [unordered](EffectClass::UnorderedIo) effects, which a compiled program
/// may execute before a failing check. Interpretation fails with the same errors as the program's or region's input
/// checks (e.g., [`DimensionError::InputDimensionMismatch`](crate::DimensionError::InputDimensionMismatch)).
///
/// A reference member is checked through its referent's shape, which the reference fixes when it is allocated but which
/// another member may still contradict (e.g., supplying `ref<f32[3]>` and `f32[2]` as inputs declared as
/// `[ref<f32[n]>, f32[n]]`). The guard declares a [read](ReferenceAccessMode::Read) of each reference member, so that
/// the check is ordered with the other accesses of its allocation, but it never observes the referenced contents and
/// produces no output for a reference member. Reference discharge checks the referent of a discharged allocation as an
/// ordinary value.
///
/// Batching lifts the declared types by the mapped axis and guards the packed values at once, freshening the guard's
/// local identities so the inserted batch dimension cannot alias them. Bounded ragged carriers and mapped first-class
/// dimensions hold their per-item extents outside their packed types, which the lifted guard cannot relate, so
/// batching rejects them.
///
/// # Enforcement
///
/// A guard is enforced by interpretation in an eager context, by the corresponding rule under a transform, or by
/// staging it as an instruction that a later interpretation or backend enforces. Every enforcement upholds the
/// following contract:
///
///   1. The inputs are the complete guarded input sequence, aligned with [`declared_types`](Self::declared_types) and
///      including reference members. Malformed arities or types are rejected even when the input types prove every
///      declared relationship.
///   2. Enforcement yields one output per non-reference declaration, in declaration order, carrying the checked
///      representation that the guard infers (i.e., the input narrowed, renamed, or widened to a
///      [retained](Self::retained) symbolic representative). Returning the original symbolic inputs instead would lose
///      that representation.
///   3. Reference members are shape anchors: their referent relationships are checked without reading the referenced
///      contents, changing handle identities, or retyping the handles, and they yield no output. After reference
///      discharge, a former reference member is an ordinary value that yields an output, while the saved anchor
///      positions still constrain the representation of the other outputs; the discharge rule drops those extra
///      outputs.
///   4. The declared identities are local to the guard: type-identity renamings of the surrounding program change its
///      [retained](Self::retained) identities but never its declarations. Anchor positions, the
///      [boundary](Self::boundary) name, reference access effects, input order, and input multiplicity are preserved,
///      so a guard is never rebuilt from its declared types alone.
///   5. Success means that the check ran immediately or became a dependency that runs before the outputs are used.
///      Relationships that local types prove never permit dropping a guard, because its outputs may still transport
///      the ordering of an earlier ordered assertion.
///   6. Under a transform, enforcement applies that transform's rule for this operation and validates its carriers
///      (e.g., batching rejects bounded ragged carriers). A transform may forward the guard to its parent context only
///      after its rule has transformed the request and the inputs.
///   7. Errors keep their variants: interpreting a relationship that symbolic types leave unproven fails with
///      [`ProgramError::Concretization`], and a family or carrier that cannot represent the check fails with an
///      explicit unsupported-operation error (e.g., [`unsupported_family_error`](Self::unsupported_family_error)).
///      An unsupported check never counts as success.
///   8. Enforcement runs under the ambient [`Provenance`](crate::Provenance) of the replay that requested the guard.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds,
/// #     DimensionType, DimensionValue, GuardOperation, Operation, Placeholder, ProgramBuilder, ProgramError,
/// #     Shape,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// let n = DimensionType::new("n", DimensionBounds::positive(Some(8))?);
/// let declared: Vec<ArrayIrType> =
///     vec![n.clone().into(), ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into()];
/// let m = DimensionType::new("m", DimensionBounds::positive(Some(8))?);
/// let actual: Vec<ArrayIrType> = vec![m.into(), ArrayType::new_static(DataType::F32, [2]).into()];
/// let (guard, members) = GuardOperation::for_entry("region", &declared, &actual)?.unwrap();
/// assert_eq!(members, vec![0, 1]);
/// assert_eq!(
///     guard.infer_output_types(&actual, &[])?,
///     vec![DimensionType::from(Dimension::Static(2)).into(), actual[1].clone()],
/// );
///
/// let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
/// let inputs = actual.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
/// builder.add_instruction(guard, Vec::new(), inputs, None)?;
/// let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
///     Vec::new(),
///     vec![Placeholder; 2],
///     Vec::new(),
/// )?;
/// let two = ArrayIrValue::Array(Array::vector(vec![1f32; 2])?);
/// let dimension = |extent: usize| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
/// assert_eq!(program.interpret(vec![dimension(2), two.clone()]), Ok(Vec::new()));
/// assert!(program.interpret(vec![dimension(5), two]).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardOperation<T: Type> {
    /// Refer to the documentation of [`boundary`](Self::boundary) for more information.
    boundary: String,

    /// Refer to the documentation of [`declared_types`](Self::declared_types) for more information.
    declared_types: Vec<T>,

    /// Refer to the documentation of [`retained`](Self::retained) for more information.
    retained: Vec<T::Identity>,

    /// Positions of immutable reference shape anchors in the original declaration. Discharge replaces their references
    /// with values, but their declared identity correspondences for each input must still determine the guard's
    /// outputs.
    anchors: Vec<usize>,
}

impl<T: Type> GuardOperation<T> {
    /// Creates a new [`GuardOperation`] that checks its inputs against `declared_types`, using `boundary` as the
    /// diagnostic name of the guarded boundary.
    #[inline]
    pub fn new<B: Into<String>>(boundary: B, declared_types: Vec<T>) -> Self {
        let anchors = declared_types
            .iter()
            .enumerate()
            .filter_map(|(index, declared)| declared.is_reference().then_some(index))
            .collect();
        Self { boundary: boundary.into(), declared_types, retained: Vec::new(), anchors }
    }

    /// Returns a guard that checks inputs of the `actual` types against `declared`, together with the positions of the
    /// guarded members among those inputs (in increasing order), or [`None`] when the actual types prove every
    /// relationship expressed by `declared` (refer to [the standalone
    /// query](crate::programs::types::type_identities_needing_runtime_checks)). The guard takes every member that
    /// carries an unproven identity, including reference members, and every non-reference member whose actual type the
    /// checked relationships narrow, so that its outputs carry each narrowing justified by those checks. Without
    /// unproven relationships, a narrowing can only come from an actual identity whose own bounds admit a single
    /// extent, which needs no check.
    ///
    /// # Parameters
    ///
    ///   - `boundary`: Diagnostic name of the guarded boundary (e.g., the name of the operation that owns it).
    ///   - `declared`: Complete declared input signature of the boundary.
    ///   - `actual`: Actual input types, aligned with `declared`.
    ///
    /// # Errors
    ///
    /// Returns an error when `actual` cannot satisfy the declared input types (refer to
    /// [`TypeRefinements::new`](crate::TypeRefinements::new)).
    pub fn for_entry<B: Into<String>>(
        boundary: B,
        declared: &[T],
        actual: &[T],
    ) -> Result<Option<(Self, Vec<usize>)>, TypeError> {
        check_count!("input", actual, declared.len(), TypeError);
        let refinements = T::Refinements::new(declared.iter(), actual.iter())?;
        let unproven = refinements.identities_needing_runtime_checks();
        if unproven.is_empty() {
            return Ok(None);
        }
        let mut members = Vec::new();
        for index in 0..declared.len() {
            if declared[index].identities().any(|(_, identity)| unproven.contains(identity))
                || (!declared[index].is_reference() && refinements.refine_actual(&actual[index], &[])? != actual[index])
            {
                members.push(index);
            }
        }
        let declared_types = members.iter().map(|&index| declared[index].clone()).collect();
        Ok(Some((Self::new(boundary, declared_types), members)))
    }

    /// Returns this [`GuardOperation`] with its [retained](Self::retained) identities set to `retained`.
    #[inline]
    pub fn with_retained(mut self, retained: Vec<T::Identity>) -> Self {
        self.retained = retained;
        self
    }

    /// Returns this [`GuardOperation`] with its [retained](Self::retained) identities chosen from the relationships
    /// between its declared types and the actual `input_types`, so that each class of related identities keeps one
    /// representation. References take precedence over known values: every actual reference identity in a class is
    /// retained (several may be linked by unknown dimensions, and the guard recovers their per-formal anchors from its
    /// reference inputs), and a reference with an exact shape but no identity also wins over a symbolic known
    /// candidate, which may consequently become residual. Otherwise an unfixed class retains the first `pinned`
    /// identity it contains, else the boundary's own identity when an actual input carries it, else its first actual
    /// identity, which avoids unnecessary replay renaming; an exact class without a pinned identity retains nothing.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Actual types of the guarded members, aligned with [`declared_types`](Self::declared_types).
    ///   - `pinned`: Actual identities that the caller needs to keep symbolic (e.g., the identities of the references
    ///     entering a replayed region), preferred as a class's representation over the boundary's own identity.
    pub fn retaining(self, input_types: &[T], pinned: &[T::Identity]) -> Result<Self, TypeError> {
        let refinements = T::Refinements::from_local_declarations(self.declared_types().iter(), input_types.iter())?;
        let references = self
            .declared_types()
            .iter()
            .zip(input_types)
            .filter(|(declared, _)| declared.is_reference())
            .collect::<Vec<_>>();
        let mut retained = Vec::new();
        for class in refinements.identity_classes() {
            if class.actual().len() == 1 && !class.is_exact() {
                continue;
            }
            let reference_identities = references
                .iter()
                .flat_map(|(_, actual)| actual.identities().map(|(_, identity)| identity).collect::<Vec<_>>())
                .filter(|identity| class.actual().contains(identity))
                .collect::<Vec<_>>();
            if !reference_identities.is_empty() {
                // Keep all immutable names, including distinct reference identities linked by unknown dimensions.
                for identity in reference_identities {
                    if !retained.contains(identity) {
                        retained.push(identity.clone());
                    }
                }
                continue;
            }
            if references
                .iter()
                .any(|(declared, _)| declared.identities().any(|(_, identity)| class.declared().contains(identity)))
            {
                // A reference without a nominal actual identity has an exact shape. That representation must win over a
                // symbolic known candidate; the checked output may consequently become residual.
                continue;
            }
            let representative = pinned.iter().find(|identity| class.actual().contains(identity));
            let representative = representative.or_else(|| {
                if class.is_exact() {
                    None
                } else {
                    class
                        .declared()
                        .iter()
                        .find(|identity| class.actual().contains(identity))
                        .or_else(|| class.actual().first())
                }
            });
            if let Some(representative) = representative {
                retained.push(representative.clone());
            }
        }
        Ok(self.with_retained(retained))
    }

    /// Returns the diagnostic name of the guarded boundary.
    #[inline]
    pub fn boundary(&self) -> &str {
        &self.boundary
    }

    /// Returns the declared types of the guarded boundary members, in input order. Their identities are local to the
    /// guard: they name the relationships that the inputs must satisfy, not identities of the surrounding program, and
    /// so type-identity renamings of the surrounding program leave them unchanged.
    #[inline]
    pub fn declared_types(&self) -> &[T] {
        &self.declared_types
    }

    /// Returns the identities of the surrounding program that this guard's outputs keep symbolic even when the checked
    /// relationships fix them (empty by default). A consumer that seeds a body with the outputs retains the identities
    /// of the values that it cannot narrow alongside them (e.g., references, which are never cast), so that the body
    /// sees one type per identity: with a retained `n`, an `f32[n, m]` input whose checked relationships fix `n = 2`
    /// and `m = 3` becomes `f32[n, 3]`, next to a `ref<f32[n]>`. Unlike the declared types, these identities name
    /// values of the surrounding program, so they follow its type-identity renamings.
    #[inline]
    pub fn retained(&self) -> &[T::Identity] {
        &self.retained
    }

    /// Returns the [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error that an operation family which
    /// stores no guard returns from its [`OperationProvider`] implementation for this request, because it cannot
    /// represent the runtime input checks that this guard carries. `#[derive(Operation)]` generates that provider for
    /// a derived family without a native guard variant, and
    /// [`impl_guardless_operation!`](crate::impl_guardless_operation) generates it for a hand-written family. Both
    /// call this function, which owns the diagnostic text.
    pub fn unsupported_family_error(&self) -> ProgramError {
        ProgramError::UnsupportedOperation {
            message: format!(
                "the input types of `{}` leave identity relationships unproven, but the operation family \
                 cannot represent the guard required to preserve these input checks",
                self.boundary,
            ),
        }
    }
}

impl<T: Type> Display for GuardOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type + Hash> Hash for GuardOperation<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Type identities are not hashable, so only the number of retained identities is hashed. Equal guards still
        // hash equally, because equality compares the retained identities themselves.
        self.boundary.hash(state);
        self.declared_types.hash(state);
        self.retained.len().hash(state);
        self.anchors.hash(state);
    }
}

impl<T: Type> Operation for GuardOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        GUARD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, self.declared_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        // Mixed refinements are admitted exactly like the program or region call this guard replaces; only runtime
        // extents decide.
        // The declared identities are local to the guard, so its inputs enter them through scoped correspondences,
        // which a consistent renaming of the surrounding program does not affect. Each non-reference output is its
        // input narrowed by the actual facts that the declared relationships prove.
        let refinements = T::Refinements::from_local_declarations(self.declared_types.iter(), input_types.iter())?;
        // Original reference facts preserve the declared anchor of each immutable handle, even after discharge turns
        // that handle into a value. The complete invocation may connect two anchors through unknown dimensions, but
        // runtime equality does not permit retyping the handles. Cast values through their anchors first.
        let reference_refinements = T::Refinements::from_local_declarations(
            self.anchors.iter().map(|&index| &self.declared_types[index]),
            self.anchors.iter().map(|&index| &input_types[index]),
        )?;
        self.declared_types
            .iter()
            .zip(input_types)
            .filter(|(declared_type, _)| !declared_type.is_reference())
            .map(|(declared_type, input_type)| {
                let anchored = reference_refinements.retype_actual(declared_type, input_type, &self.retained)?;
                refinements.retype_actual(declared_type, &anchored, &self.retained)
            })
            .collect()
    }

    fn effects(&self) -> Cow<'_, Effects> {
        // Reference members are read for their shapes only, which orders the check with the other accesses of their
        // allocations. Each reference input is accessed once, so the declaration is always valid.
        let accesses = self
            .declared_types
            .iter()
            .enumerate()
            .filter(|(_, declared_type)| declared_type.is_reference())
            .map(|(input_index, _)| ReferenceEffect::Access { input_index, mode: ReferenceAccessMode::Read })
            .collect::<Vec<_>>();
        if accesses.is_empty() {
            return Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)));
        }
        Cow::Owned(Effects::new(EffectClasses::single(EffectClass::OrderedAssertion), accesses).unwrap())
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        // The declared identities are local to the guard (they only name the relationships that its inputs must
        // satisfy), so an ambient renaming of the surrounding program, which may also narrow bounds, must not change
        // them. The guard's outputs carry only identities of its inputs, which the program renames separately, and the
        // retained identities name those same input identities, so they follow the renaming.
        Ok(Self { retained: self.retained.iter().map(|identity| renaming.rename(identity)).collect(), ..self.clone() })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("boundary", format_args!("{:?}", self.boundary))?;
            operation.field(
                "types",
                format_args!(
                    "[{}]",
                    self.declared_types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            )?;
            if !self.retained.is_empty() {
                operation.field(
                    "retained",
                    format_args!("[{}]", self.retained.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),),
                )?;
            }
            if self.anchors.iter().copied().ne(self
                .declared_types
                .iter()
                .enumerate()
                .filter_map(|(index, declared)| declared.is_reference().then_some(index)))
            {
                operation.field("anchors", format_args!("{:?}", self.anchors))?;
            }
            Ok(())
        })
    }
}

impl<C: Context<Type: ReferenceMemberType, Operation: From<GuardOperation<C::Type>>>, P: ReferenceDischargePolicy<C>>
    ReferenceDischargeableOperation<C, P> for GuardOperation<C::Type>
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A discharged reference member is checked through its current value at its declared referent type, which has
        // the same shape, while a preserved reference member is checked as the reference that it still is. The rebuilt
        // guard therefore produces outputs for discharged reference members too, which are dropped so that the outputs
        // keep corresponding to the non-reference members of this guard.
        check_count!("input", inputs, self.declared_types.len(), ProgramError);
        let mut declared_types = Vec::with_capacity(inputs.len());
        let mut values = Vec::with_capacity(inputs.len());
        let mut kept_outputs = Vec::with_capacity(inputs.len());
        for (declared_type, input) in self.declared_types.iter().zip(inputs) {
            match input {
                ReferenceDischargeValue::Value(value) => {
                    declared_types.push(declared_type.clone());
                    values.push(value.clone());
                    kept_outputs.push(true);
                }
                ReferenceDischargeValue::Reference(reference) => match reference.preserved() {
                    Some(preserved) => {
                        declared_types.push(declared_type.clone());
                        values.push(preserved.clone());
                    }
                    None => {
                        let referent = declared_type.referent().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{GUARD_OPERATION_NAME}` received a reference for its member of type \
                                 `{declared_type}`",
                            ))
                        })?;
                        declared_types.push(C::Type::from(referent.clone()));
                        values.push(context.read(reference)?);
                        kept_outputs.push(false);
                    }
                },
            }
        }
        let outputs = context.parent().bind(Self { declared_types, ..self.clone() }, Vec::new(), &values)?;
        check_count!("output", outputs, kept_outputs.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(kept_outputs)
            .filter(|(_, kept)| *kept)
            .map(|(output, _)| ReferenceDischargeValue::Value(output))
            .collect())
    }
}

impl<C: Domain> InterpretableOperation<C> for GuardOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Interpreted values carry their concrete extents in their types, so deriving refinements from these types
        // performs the complete runtime check, with the same diagnostics as program or region input validation, and the
        // narrowed outputs are the non-reference inputs themselves. Interpreted types can still be symbolic (e.g.,
        // tracers interpreted through a program), in which case a relationship the types do not prove has no
        // interpretation here, exactly as for `cast`: such a guard must be bound through its context, whose rules stage
        // the check.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &driver.regions().map(|region| region.interface()).collect::<Vec<_>>())?;
        let refinements =
            <C::Type as Type>::Refinements::from_local_declarations(self.declared_types.iter(), input_types.iter())?;
        if !refinements.identities_needing_runtime_checks().is_empty() {
            return Err(ProgramError::Concretization {
                message: format!(
                    "`{GUARD_OPERATION_NAME}` cannot check symbolic type relationships during interpretation; bind the \
                     operation through its context"
                ),
            });
        }
        Ok(self
            .declared_types
            .iter()
            .zip(inputs)
            .filter(|(declared_type, _)| !declared_type.is_reference())
            .map(|(_, input)| input.clone())
            .collect())
    }
}

impl<C: Context<Operation: From<GuardOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for GuardOperation<C::Type>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Known inputs fold into the parent (which checks them immediately when it is eager), and any unknown input
        // keeps the guard in the residual program, where its ordered effect defers later ordered known effects and
        // where all of its outputs are unknown.
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        context.fold_or_residualize(self.clone(), Vec::new(), inputs)
    }
}

impl<C: Context<Operation: From<GuardOperation<C::Type>>>, P: BatchingPolicy<C>> BatchableOperation<C, P>
    for GuardOperation<C::Type>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Every batch item enters with the same per-item shape, because mapped members are dense along the batch axis.
        // Guarding the stacked values with the batch axis inserted into their declared types therefore checks every
        // item at once. An empty batch still carries the extents of its non-batch axes, which the guard checks. Each
        // output keeps the batch axis of its input, since narrowing never changes the position of an axis. Validate the
        // original declaration before lifting erases layout and adds batch-axis variation. The parent validates the
        // physical operands against the lifted contract, retaining every other input constraint.
        check_count!("input", inputs, self.declared_types.len(), ProgramError);
        self.infer_output_types(
            &self.declared_types,
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        // Bounded ragged carriers and mapped dimensions keep per-item extents outside their packed types, which the
        // lifted declarations cannot relate. Reject them before freshening makes their identities look unrelated.
        reject_ragged_carriers::<C, P>(GUARD_OPERATION_NAME, inputs)?;
        // The inserted batch dimension belongs to the surrounding program. Freshen the guard-local names together
        // before inserting it, so reusing a nominal identity across those scopes cannot create an extra equality.
        let mut renaming = TypeIdentityRenaming::new();
        for (_, identity) in self.declared_types.iter().flat_map(|r#type| r#type.identities()) {
            if renaming.target(identity).is_none() {
                renaming.insert(identity.clone(), identity.fresh())?;
            }
        }
        let declared_types = self
            .declared_types
            .iter()
            .zip(inputs)
            .map(|(declared_type, input)| {
                let declared_type = declared_type.rename_identities(&renaming)?;
                P::batched_type(context, input, &declared_type)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = inputs.iter().map(|input| P::value(input).clone()).collect::<Vec<_>>();
        let outputs = context.parent().bind(Self { declared_types, ..self.clone() }, Vec::new(), &values)?;
        let inputs = self
            .declared_types
            .iter()
            .zip(inputs)
            .filter(|(declared_type, _)| !declared_type.is_reference())
            .map(|(_, input)| input)
            .collect::<Vec<_>>();
        check_count!("output", outputs, inputs.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(inputs)
            .map(|(output, input)| P::batch(output, P::batch_axis(input)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<C: Context<Type = T>, T: DifferentiableType> DifferentiableOperation<C> for GuardOperation<T>
where
    C::Operation: From<GuardOperation<T>> + From<CastOperation<T>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Tangents use the checked output's representation through the shared tangent cast of `CastOperation`,
        // which saves input extents only for names lost from that representation. Every fused tangent retains the
        // guard's data dependency: even a proven guard orders its outputs after preceding standalone assertions.
        // Partitioned tangents run after the primal completes and therefore retain only the extents they need.
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let outputs = context.primal().bind(self.clone(), Vec::new(), &primals)?;
        let needs_ordering = !P::PRIMAL_COMPLETES_BEFORE_TANGENT;
        let members = self
            .declared_types
            .iter()
            .zip(inputs)
            .filter(|(declared_type, _)| !declared_type.is_reference())
            .collect::<Vec<_>>();
        check_count!("output", outputs, members.len(), ProgramError);
        outputs
            .into_iter()
            .zip(members)
            .map(|(output, (_, input))| match input.tangent() {
                MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(output),
                MaybeZero::Value(tangent) => {
                    let tangent =
                        CastOperation::bind_tangent(context, input.primal(), &output, tangent, needs_ordering)?;
                    DifferentiationDual::new(output, tangent)
                }
            })
            .collect()
    }
}

impl_non_transposable_operation!(<T> GuardOperation<T> where T: Type);

impl<T: Type, O: Operation<Type = T> + From<GuardOperation<T>>> OperationProvider<T, GuardOperation<T>> for O {
    type Operation = Self;

    #[inline]
    fn provide(request: GuardOperation<T>, input_types: &[&T]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, request.declared_types.len(), ProgramError);
        Ok(Self::from(request))
    }
}

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::tests::{ragged_array_batch, ragged_batch};
    use crate::arrays::{
        ArrayBatch, ArrayIrOperation, ArrayOperation, ArrayReference, Dimension, DimensionBounds, DimensionError,
        DimensionSource, DimensionVariable, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension, StaticArrayExtentBatchingPolicy, StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{
        BatchAxis, BatchedProgram, BatchingTracer, BoundaryPreservingBatchedProgram, ProgramBatchingOutputAxesPolicy,
        batch,
    };
    use crate::contexts::{DomainProjection, EagerContext, StagingContext};
    use crate::macros::check_operation_type_inference;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::manipulation::broadcasting::Broadcast;
    use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
    use crate::operations::references::ReferenceReadOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialTracer, PartialValue};
    use crate::programs::{EmptyRegionDriver, OperationProjection, ProgramBuilder, ReferenceType};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Returns the `[f32[n], f32[n]]` input signature shared by the guard tests.
    fn guard_signature() -> (DimensionType, Vec<ArrayIrType>) {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        (n, vec![vector_type.clone(), vector_type])
    }

    #[test]
    fn test_assertion_error() {
        let failure = AssertionError::Failed {
            message: "condition must hold".to_owned(),
            observations: vec![("value".to_owned(), "7".to_owned()), ("valid".to_owned(), "false".to_owned())],
        };
        assert_eq!(failure.to_string(), "assertion failed: condition must hold; value=7, valid=false");
        assert_eq!(
            format!("{failure:?}"),
            "Failed { message: \"condition must hold\", observations: [(\"value\", \"7\"), (\"valid\", \"false\")] }",
        );
        let error = ProgramError::from(failure.clone());
        assert_eq!(error.downcast_custom::<AssertionError>(), Some(&failure));

        // Bounded diagnostics retain logical indices and report omitted failures only when there are any.
        let failures =
            vec![AssertionFailure { index: vec![1, 0], observations: vec![("value".to_owned(), "3".to_owned())] }];
        assert_eq!(
            AssertionError::FailedElements { message: "valid".to_owned(), failures: failures.clone(), omitted: 0 }
                .to_string(),
            "assertion failed: valid; element [1, 0], value=3",
        );
        assert_eq!(
            AssertionError::FailedElements { message: "valid".to_owned(), failures, omitted: 2 }.to_string(),
            "assertion failed: valid; element [1, 0], value=3; 2 additional failing elements omitted",
        );
    }

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
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(operation.to_string(), r#"assert [message="input must be \"valid\"\nnext", labels=["value"]]"#);
        assert_eq!(
            bounded.to_string(),
            r#"assert [message="input must be \"valid\"\nnext", labels=["value"], failure_limit=3]"#,
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

        // Bounded assertions accept shaped Boolean conditions and either scalar or shape-matched observations.
        check_operation_type_inference!(
            operation = AssertOperation::<ArrayType>::new("all elements must hold")
                .with_labels(vec!["value".to_owned()]).with_failure_limit(NonZeroUsize::MIN),
            cases = [{
                input_types = [
                    ArrayType::new_static(DataType::Boolean, [2]),
                    ArrayType::new_static(DataType::I32, [2]),
                ],
                output_types = [],
            }, {
                input_types = [ArrayType::new_static(DataType::Boolean, [2]), ArrayType::scalar(DataType::I32)],
                output_types = [],
            }, {
                input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I32)],
                output_types = [],
            }, {
                input_types = [ArrayType::new_static(DataType::I32, [2]), ArrayType::scalar(DataType::I32)],
                error = "assertion condition must have type `bool[...]` but has type `i32[2]`",
            }, {
                input_types = [
                    ArrayType::new_static(DataType::Boolean, [2]),
                    ArrayType::new_static(DataType::I32, [1]),
                ],
                error = "assertion observation `value` has unsupported type `i32[1]`",
            }],
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
    fn test_assert_interpretation() {
        let context = EagerContext::<Array>::new();
        let operation = AssertOperation::new("condition must hold");
        assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[Array::scalar(true).unwrap()]), Ok(vec![]));
        let error = operation.interpret(&context, &EmptyRegionDriver, &[Array::scalar(false).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "condition must hold".to_owned(), observations: vec![] }),
        );
        assert_eq!(error.to_string(), "assertion failed: condition must hold");
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
            Some(&AssertionError::Failed { message: "condition must hold".to_owned(), observations: vec![] }),
        );
    }

    #[test]
    fn test_assert_partial_evaluation_staged_observation() {
        let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
        let partial = PartialEvaluationContext::new(parent.clone());
        let condition = partial.lift(Array::scalar(false).unwrap()).unwrap();
        let observation =
            PartialTracer::new(partial.clone(), partial.unknown_input(ArrayType::scalar(DataType::I32), 0));

        // Known successes need no diagnostic values, even when a failure limit is configured.
        let success = partial.lift(Array::scalar(true).unwrap()).unwrap();
        assert_eq!(success.assert("known success", &[("value", observation.clone())]), Ok(()));
        assert_eq!(
            success.assert_with_limit("known success", &[("value", observation.clone())], NonZeroUsize::MIN),
            Ok(()),
        );
        drop(success);

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
                observations: vec![("value".to_owned(), "7".to_owned())],
            }),
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

        // An unresolved earlier failure must be observed before the later known-false assertion.
        let error = residual.program().interpret(vec![Array::scalar(false).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "first".to_owned(), observations: vec![] }),
        );
        let error = residual.program().interpret(vec![Array::scalar(true).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "second".to_owned(), observations: vec![] }),
        );
    }

    #[test]
    fn test_assert_partial_evaluation_dimension_observations() {
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
        let error = program.partially_evaluate(&inputs).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dimension requirement".to_owned(),
                observations: vec![
                    ("exact".to_owned(), "4".to_owned()),
                    ("bounded".to_owned(), format!("<unknown: `{bounded}`>")),
                    ("scalar".to_owned(), "<unknown>".to_owned()),
                ],
            }),
        );
        assert_eq!(error.to_string(), expected);

        // Concrete observations take precedence over their wider declared bounds.
        inputs[2] = PartialValue::Known(ArrayIrValue::Dimension(DimensionValue::new(bounded, 6).unwrap()));
        let error = program.partially_evaluate(&inputs).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dimension requirement".to_owned(),
                observations: vec![
                    ("exact".to_owned(), "4".to_owned()),
                    ("bounded".to_owned(), "6".to_owned()),
                    ("scalar".to_owned(), "<unknown>".to_owned()),
                ],
            }),
        );
        assert_eq!(error.to_string(), "assertion failed: dimension requirement; exact=4, bounded=6, scalar=<unknown>");
    }

    #[test]
    fn test_assert_partial_evaluation_composed_predicates() {
        let left = DimensionType::new("left", DimensionBounds::new(0, Some(10)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(0, Some(10)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left_input = builder.add_input(left.clone().into());
        let right_input = builder.add_input(right.clone().into());
        let condition = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![left_input, left_input],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::new("self equality").with_labels(vec!["left".to_owned(), "right".to_owned()]),
                Vec::new(),
                vec![condition, left_input, right_input],
                None,
            )
            .unwrap();
        let condition = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![left_input, right_input],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::new("left <= right").with_labels(vec!["left".to_owned(), "right".to_owned()]),
                Vec::new(),
                vec![condition, left_input, right_input],
                None,
            )
            .unwrap();
        let condition = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![left_input, right_input],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::new("left == right").with_labels(vec!["left".to_owned(), "right".to_owned()]),
                Vec::new(),
                vec![condition, left_input, right_input],
                None,
            )
            .unwrap();
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
            3,
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
            ],
        );
        let error = simplified
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left.clone(), 7).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right.clone(), 3).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "left <= right".to_owned(),
                observations: vec![("left".to_owned(), "7".to_owned()), ("right".to_owned(), "3".to_owned())],
            }),
        );
        let error = simplified
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left.clone(), 3).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right.clone(), 7).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "left == right".to_owned(),
                observations: vec![("left".to_owned(), "3".to_owned()), ("right".to_owned(), "7".to_owned())],
            }),
        );
    }

    #[test]
    fn test_assert_batching() {
        // Mapped conditions select the first failure and its matching observation.
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| condition.assert("positive", &[("value", observation)]),
            (Array::vector(vec![true, false, false]).unwrap(), Array::vector(vec![10i32, 20, 30]).unwrap()),
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
            }),
        );

        // Replicated false conditions still pass over an empty batch.
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| condition.assert("empty", &[("value", observation)]),
            (Array::scalar(false).unwrap(), Array::vector(Vec::<i32>::new()).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (),
            None,
        );
        assert_eq!(output, Ok(()));

        // A nonempty batch of successful conditions also passes.
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
    fn test_assert_batching_custom_static_policy() {
        // Extents need only Clone + Debug; their representation need not be usize, Copy, or an array IR value.
        #[derive(Clone, Debug)]
        struct HostExtent {
            size: usize,
            metadata: String,
        }
        #[derive(Copy, Clone, Debug)]
        struct CustomExtent;
        impl<C: Context<Type = ArrayType>> BatchingPolicy<C> for CustomExtent {
            type Batch = ArrayBatch<C::Value>;
            type Extent = HostExtent;
            type Evidence = Vec<DimensionVariable>;
            type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;
            fn batch(value: C::Value, axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
                <StaticArrayExtentBatchingPolicy as BatchingPolicy<C>>::batch(value, axis)
            }
            fn replicated(value: C::Value) -> Self::Batch {
                <StaticArrayExtentBatchingPolicy as BatchingPolicy<C>>::replicated(value)
            }
            fn value(batch: &Self::Batch) -> &C::Value {
                batch.value()
            }
            fn batch_axis(batch: &Self::Batch) -> BatchAxis {
                batch.batch_axis()
            }
            fn batched_type(
                context: &BatchingContext<C, Self>,
                input: &Self::Batch,
                declared: &ArrayType,
            ) -> Result<ArrayType, BatchingError> {
                let context = BatchingContext::<C, StaticArrayExtentBatchingPolicy>::with_policy(
                    context.parent().clone(),
                    context.axis_extent().size,
                )
                .with_axis_sharding(context.axis_sharding().clone());
                StaticArrayExtentBatchingPolicy::batched_type(&context, input, declared)
            }
            fn unbatched_type(batch: &Self::Batch) -> Cow<'_, ArrayType> {
                Cow::Owned(batch.unbatched_type())
            }
            fn adapt_batched_program<
                CollapseFn: Fn(
                    &TracingContext<C::Constant, C::Operation>,
                    Tracer<TracingContext<C::Constant, C::Operation>>,
                    Axis,
                ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
            >(
                program: Self::BatchedProgram,
                required_output_axes: Option<&[BatchAxis]>,
                collapse_fn: CollapseFn,
            ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
                <StaticArrayExtentBatchingPolicy as BatchingPolicy<C>>::adapt_batched_program(
                    program,
                    required_output_axes,
                    collapse_fn,
                )
            }
        }
        impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> ArrayExtentBatchingPolicy<C> for CustomExtent {
            fn axis_dimension(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
            ) -> Result<Dimension, BatchingError> {
                Ok(Dimension::Static(context.axis_extent().size))
            }
            fn match_axis(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
                batch: &ArrayBatch<C::Value>,
                axis: Axis,
            ) -> Result<ArrayBatch<C::Value>, BatchingError> {
                batch.match_axis(axis, context.axis_extent().size, context.axis_sharding().clone())
            }
            fn broadcast_input(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
                input: &ArrayBatch<C::Value>,
                target: ArrayType,
                axes: Vec<usize>,
                batch_axis: Axis,
                sources: Vec<DimensionSource<C::Value>>,
            ) -> Result<ArrayBatch<C::Value>, BatchingError> {
                let context = BatchingContext::<C, ArrayBatchingPolicy>::new(
                    context.parent().clone(),
                    context.axis_extent().size,
                )
                .with_axis_sharding(context.axis_sharding().clone());
                StaticArrayExtentBatchingPolicy::broadcast_input(&context, input, target, axes, batch_axis, sources)
            }
        }

        impl<
            C: Context<Type = ArrayType, Value: Broadcast + Reduce + Transpose, Operation: From<IotaOperation<ArrayType>>>,
        > ArrayAssertionBatchingPolicy<C, Self> for CustomExtent
        {
            #[inline]
            fn batch_item_indices(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
                r#type: &ArrayType,
            ) -> Result<C::Value, ProgramError> {
                <StaticArrayExtentBatchingPolicy as ArrayAssertionBatchingPolicy<C, Self>>::batch_item_indices(
                    context, r#type,
                )
            }

            #[inline]
            fn mask_empty_batch_condition(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
                condition: C::Value,
            ) -> Result<C::Value, ProgramError> {
                <StaticArrayExtentBatchingPolicy as ArrayAssertionBatchingPolicy<C, Self>>::mask_empty_batch_condition(
                    context, condition,
                )
            }

            #[inline]
            fn pad_empty_batch_observation(
                context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
                observation: C::Value,
                padded_extent: &mut Option<HostExtent>,
            ) -> Result<C::Value, ProgramError> {
                <StaticArrayExtentBatchingPolicy as ArrayAssertionBatchingPolicy<C, Self>>::pad_empty_batch_observation(
                    context,
                    observation,
                    padded_extent,
                )
            }
        }

        // This family omits assertions and their diagnostic primitives. The custom preparation impl above does
        // not apply to its context, but generated dispatch must still support guards, casts, and ordinary batching.
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(dispatch(batching))]
        enum MinimalOperation<V: Value<Type = ArrayType>> {
            Constant(ConstantOperation<V>),
            Guard(GuardOperation<ArrayType>),
            Cast(CastOperation<ArrayType>),
            Transpose(TransposeOperation),
        }

        // A distinct public extent policy must retain all three rules. Built-in policies alone cannot catch
        // an implementation that accidentally specializes on their concrete types.
        for extent in [0, 1, 3] {
            let context = BatchingContext::<_, ArrayBatchingPolicy<CustomExtent>>::with_policy(
                EagerContext::<Array, ArrayOperation<Array>>::new(),
                HostExtent { size: extent, metadata: "custom host extent".to_owned() },
            );
            assert_eq!(context.axis_extent().metadata, "custom host extent");
            let failing_index = extent / 2;
            let observation = ArrayBatch::new(Array::vector(vec![2i32; extent]).unwrap(), BatchAxis::new(0)).unwrap();
            for limit in [None, NonZeroUsize::new(1)] {
                let mut operation = AssertOperation::new("custom policy").with_labels(vec!["value".to_owned()]);
                if let Some(limit) = limit {
                    operation = operation.with_failure_limit(limit);
                }
                let passing = ArrayBatch::new(Array::vector(vec![true; extent]).unwrap(), BatchAxis::new(0)).unwrap();
                assert_eq!(
                    operation
                        .batch(&context, &EmptyRegionDriver, &[passing, observation.clone()])
                        .map(|outputs| outputs.into_parts().0.len()),
                    Ok(0),
                );
                let condition = ArrayBatch::new(
                    Array::vector((0..extent).map(|index| index != failing_index).collect::<Vec<_>>()).unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap();
                let failure = operation.batch(&context, &EmptyRegionDriver, &[condition, observation.clone()]);
                if extent == 0 {
                    assert_eq!(failure.map(|outputs| outputs.into_parts().0.len()), Ok(0));
                } else {
                    let BatchingError::Program(error) = failure.unwrap_err() else {
                        panic!("expected a program error")
                    };
                    let expected = if limit.is_some() {
                        AssertionError::FailedElements {
                            message: "custom policy".to_owned(),
                            failures: vec![AssertionFailure {
                                index: vec![failing_index],
                                observations: vec![("value".to_owned(), "2".to_owned())],
                            }],
                            omitted: 0,
                        }
                    } else {
                        AssertionError::Failed {
                            message: "custom policy".to_owned(),
                            observations: vec![
                                ("value".to_owned(), "2".to_owned()),
                                ("batch_index".to_owned(), failing_index.to_string()),
                            ],
                        }
                    };
                    assert_eq!(error.downcast_custom::<AssertionError>(), Some(&expected));
                }
            }
            let member_type = ArrayType::new_static(DataType::F32, [2]);
            let value =
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [extent, 2]), &vec![1f32; extent * 2])
                    .unwrap();
            let input = ArrayBatch::new(value.clone(), BatchAxis::new(0)).unwrap();
            let guard = GuardOperation::new("custom policy", vec![member_type.clone()]);
            let guarded =
                guard.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
            assert_eq!(guarded, vec![input.clone()]);
            let cast = CastOperation::unchecked(member_type.clone(), vec![member_type.clone(), member_type]).unwrap();
            let casted =
                cast.batch(&context, &EmptyRegionDriver, &[input.clone(), input.clone()]).unwrap().into_parts().0;
            assert_eq!(casted, vec![input.clone()]);

            let context = BatchingContext::<_, ArrayBatchingPolicy<CustomExtent>>::with_policy(
                EagerContext::<Array, MinimalOperation<Array>>::new(),
                context.axis_extent().clone(),
            );
            let guarded = MinimalOperation::<Array>::from(guard)
                .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(guarded, vec![input.clone()]);
            let casted = MinimalOperation::<Array>::from(cast)
                .batch(&context, &EmptyRegionDriver, &[input.clone(), input.clone()])
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(casted, vec![input.clone()]);
            let transposed = MinimalOperation::<Array>::from(TransposeOperation::new(vec![0]))
                .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(transposed, vec![input]);
        }
    }

    #[test]
    fn test_assert_batching_policy_bounds() {
        // The array rule must compile for any extent policy with assertion preparation, given only the diagnostic
        // capabilities of the values. Concrete built-in policies alone would hide accidentally stronger bounds.
        fn array_policy<C, P>()
        where
            C: Context<
                    Type = ArrayType,
                    Value: Assert + Reduce + Select + DynamicSlice + Reshape + ConvertElementType + ZeroLike,
                >,
            P: ArrayExtentBatchingPolicy<C> + ArrayAssertionBatchingPolicy<C, P>,
        {
            let _ = <AssertOperation<ArrayType> as BatchableOperation<C, ArrayBatchingPolicy<P>>>::batch::<
                EmptyRegionDriver,
            >;
        }
        array_policy::<EagerContext<Array, ArrayOperation<Array>>, StaticArrayExtentBatchingPolicy>();
    }

    #[test]
    fn test_assert_batching_replicated_folding() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let condition = context.lift(Array::scalar(true).unwrap()).unwrap();
        let batching = BatchingContext::new(context.clone(), 3);
        AssertOperation::new("replicated true")
            .batch(&batching, &EmptyRegionDriver, &[ArrayBatch::replicated(condition)])
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_assert_batching_nested() {
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
                Array::matrix(2, 3, vec![10i32, 20, 30, 40, 50, 60]).unwrap(),
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
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.simplified_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.domain(), extent);
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
        assert!(program.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Array(ArrayOperation::Reduce(operation)) if operation.kind() == ReductionKind::Min,
        )));

        // Empty, singleton, and larger successful batches all satisfy the assertion.
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<bool>::new()).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10i32; 0]).unwrap()),
            )),
            Ok(()),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 1).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10i32; 1]).unwrap()),
            )),
            Ok(()),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true, true, true, true]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10i32; 4]).unwrap()),
            )),
            Ok(()),
        );

        // Report the first failing item, including a failure in a singleton batch.
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true, false, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dynamic".to_owned(),
                observations: vec![("value".to_owned(), "20".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
            }),
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 1).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![30i32]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dynamic".to_owned(),
                observations: vec![("value".to_owned(), "30".to_owned()), ("batch_index".to_owned(), "0".to_owned())],
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_manual_variation() {
        // Inside a manual region, the zero that pads a varying observation for a possibly empty batch takes the
        // observation's manual variation, as `pad` requires of its padding value.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let varying = |data_type: DataType| {
            ArrayIrType::Array(
                ArrayType::new(data_type, Shape::new(vec![extent_type.simplified_dimension()]))
                    .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
            )
        };
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |(extent, condition, observation)| {
                let batching = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                    ProjectedContext::new(extent.domain()),
                    extent,
                );
                AssertOperation::<ArrayType>::new("dynamic").with_labels(vec!["value".to_owned()]).batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(condition.into_projected()?, BatchAxis::new(0))?,
                        ArrayBatch::new(observation.into_projected()?, BatchAxis::new(0))?,
                    ],
                )?;
                Ok(())
            },
            (ArrayIrType::Dimension(extent_type.clone()), varying(DataType::Boolean), varying(DataType::I32)),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [0, 5)>, %1:bool[batch][sharding={mesh<['x'=2:manual]>, [{}], \
                    varying_manual={'x'}}], %2:i32[batch][sharding={mesh<['x'=2:manual]>, [{}], \
                    varying_manual={'x'}}] .
                let %3:i32[batch][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i32[batch][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ] %0
                    %4:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=max, \
                        axes=[0]] %3
                    %5:i32[batch][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %1 %4 %3
                    %6:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=min, \
                        axes=[0]] %5
                    %7:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=all, \
                        axes=[0]] %1
                    %8:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = zero_like %6
                    %9:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = select %7 %8 %6
                    %10:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = zero \
                        [type=i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                    %11:dimension<1> = constant [value=1]
                    %12:dimension<batch + 1 ∈ [1, 6)> = dimension_add %0 %11
                    %13:i32[batch + 1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = pad \
                        [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[0]] %2 %10 %12
                    %14:i32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = dynamic_slice \
                        [sizes=[1], allow_negative_indices=false] %13 %9
                    %15:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reshape [shape=[]] %14
                    () = assert [message=\"dynamic\", labels=[\"value\", \"batch_index\"]] %7 %15 %9
                in ()"
            },
        );
    }

    #[test]
    fn test_assert_batching_dynamic_nested() {
        let extent_type = DimensionType::new("inner", DimensionBounds::new(0, Some(4)).unwrap());
        let shape = Shape::new(vec![Dimension::Static(2), extent_type.simplified_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                batch(
                    |(extent, condition, observation)| {
                        let batching =
                            BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                                ProjectedContext::new(extent.domain()),
                                extent,
                            );
                        AssertOperation::<ArrayType>::new("nested dynamic")
                            .with_labels(vec!["value".to_owned()])
                            .batch(
                                &batching,
                                &EmptyRegionDriver,
                                &[
                                    ArrayBatch::new(condition.into_projected()?, BatchAxis::new(0))?,
                                    ArrayBatch::new(observation.into_projected()?, BatchAxis::new(0))?,
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
            Ok(()),
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![true, true, true, true, false, false]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10i32, 20, 30, 40, 50, 60]).unwrap()),
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
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_replicated() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition)| {
                let batching = BatchingContext::new(extent.domain(), extent.clone());
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
            Ok(()),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap()),
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
            )),
            Ok(()),
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
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_replicated_predicate() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.domain(), extent.clone());
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
                ArrayIrType::Array(ArrayType::new(DataType::U64, Shape::new(vec![extent_type.simplified_dimension()]))),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 0).unwrap()),
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<u64>::new()).unwrap()),
            )),
            Ok(()),
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
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_replicated_dimension_observation() {
        // A replicated dimension observation is shared by every batch item, so it reaches the parent assertion
        // unchanged at its own position, ahead of the selected or materialized array observation that follows it.
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.simplified_dimension()]);
        let trace = |limit: Option<NonZeroUsize>| {
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |(extent, condition, observation)| {
                    let batching = BatchingContext::new(extent.domain(), extent.clone());
                    let mut operation =
                        AssertOperation::new("dimension").with_labels(vec!["size".to_owned(), "value".to_owned()]);
                    if let Some(limit) = limit {
                        operation = operation.with_failure_limit(limit);
                    }
                    operation.batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayIrBatch::new(condition, BatchAxis::new(0))?,
                            ArrayIrBatch::replicated(extent),
                            ArrayIrBatch::new(observation, BatchAxis::new(0))?,
                        ],
                    )?;
                    Ok(())
                },
                (
                    ArrayIrType::Dimension(extent_type.clone()),
                    ArrayIrType::Array(ArrayType::new(DataType::Boolean, shape.clone())),
                    ArrayIrType::Array(ArrayType::new(DataType::I32, shape.clone())),
                ),
            )
            .unwrap()
            .1
        };
        let inputs = |extent: usize, condition: Vec<bool>| {
            (
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), extent).unwrap()),
                ArrayIrValue::Array(Array::vector(condition).unwrap()),
                ArrayIrValue::Array(Array::vector((1..=extent).map(|index| index as i32 * 10).collect()).unwrap()),
            )
        };

        // Without a failure limit, the first failing item is reported after the dimension observation.
        let program = trace(None);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [0, 5)>, %1:bool[batch], %2:i32[batch] .
                let %3:i32[batch] = iota [type=i32[batch], dimension=0] %0
                    %4:i32[] = reduce [kind=max, axes=[0]] %3
                    %5:i32[batch] = select %1 %4 %3
                    %6:i32[] = reduce [kind=min, axes=[0]] %5
                    %7:bool[] = reduce [kind=all, axes=[0]] %1
                    %8:i32[] = zero_like %6
                    %9:i32[] = select %7 %8 %6
                    %10:i32[] = zero [type=i32[]]
                    %11:dimension<1> = constant [value=1]
                    %12:dimension<batch + 1 ∈ [1, 6)> = dimension_add %0 %11
                    %13:i32[batch + 1] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[0]] \
                        %2 %10 %12
                    %14:i32[1] = dynamic_slice [sizes=[1], allow_negative_indices=false] %13 %9
                    %15:i32[] = reshape [shape=[]] %14
                    () = assert [message=\"dimension\", labels=[\"size\", \"value\", \"batch_index\"]] \
                        %7 %0 %15 %9
                in ()"
            },
        );
        assert_eq!(program.interpret(inputs(0, Vec::new())), Ok(()));
        let error = program.interpret(inputs(3, vec![true, false, false])).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "dimension".to_owned(),
                observations: vec![
                    ("size".to_owned(), "3".to_owned()),
                    ("value".to_owned(), "20".to_owned()),
                    ("batch_index".to_owned(), "1".to_owned()),
                ],
            }),
        );

        // With a failure limit, every failing element reports the same dimension observation.
        let program = trace(NonZeroUsize::new(2));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [0, 5)>, %1:bool[batch], %2:i32[batch] .
                let %3:bool[batch] = broadcast [output_axes=[0]] %1 %0
                    %4:i32[batch] = broadcast [output_axes=[0]] %2 %0
                    () = assert [message=\"dimension\", labels=[\"size\", \"value\"], failure_limit=2] %3 %0 %4
                in ()"
            },
        );
        assert_eq!(program.interpret(inputs(0, Vec::new())), Ok(()));
        let error = program.interpret(inputs(3, vec![true, false, false])).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "dimension".to_owned(),
                failures: vec![
                    AssertionFailure {
                        index: vec![1],
                        observations: vec![("size".to_owned(), "3".to_owned()), ("value".to_owned(), "20".to_owned())],
                    },
                    AssertionFailure {
                        index: vec![2],
                        observations: vec![("size".to_owned(), "3".to_owned()), ("value".to_owned(), "30".to_owned())],
                    },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_mapped_dimension_observation() {
        // A mapped dimension holds one extent per batch item; the assertion reports the failing items' extents.
        let dimension_type = DimensionType::new("n", DimensionBounds::new(0, Some(8)).unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            DimensionValue::constant(3).unwrap().into(),
        );
        let inputs = [
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![true, false, false]).unwrap()), BatchAxis::new(0))
                .unwrap(),
            ArrayIrBatch::mapped_dimension(
                ArrayIrValue::Array(Array::vector(vec![2i32, 5, 3]).unwrap()),
                BatchAxis::new(0),
                dimension_type,
            )
            .unwrap(),
        ];
        let operation = AssertOperation::<ArrayIrType>::new("mapped").with_labels(vec!["size".to_owned()]);
        let BatchingError::Program(error) =
            operation.batch(&context, &EmptyRegionDriver, &inputs).map(|_| ()).unwrap_err()
        else {
            panic!("expected an assertion failure")
        };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "mapped".to_owned(),
                observations: vec![("size".to_owned(), "5".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
            }),
        );
        let BatchingError::Program(error) = operation
            .with_failure_limit(NonZeroUsize::new(2).unwrap())
            .batch(&context, &EmptyRegionDriver, &inputs)
            .map(|_| ())
            .unwrap_err()
        else {
            panic!("expected an assertion failure")
        };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "mapped".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![1], observations: vec![("size".to_owned(), "5".to_owned())] },
                    AssertionFailure { index: vec![2], observations: vec![("size".to_owned(), "3".to_owned())] },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_assert_batching_dynamic_index_limit() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(i32::MAX as usize + 2)).unwrap());
        let extent = context.input(extent_type.clone().into());
        let predicate = context
            .input(ArrayType::new(DataType::Boolean, Shape::new(vec![extent_type.simplified_dimension()])).into());
        let batching = BatchingContext::new(context.clone(), extent);
        assert_eq!(
            AssertOperation::new("too large")
                .batch(&batching, &EmptyRegionDriver, &[ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap()],)
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation {
                message: "mapped assertion batch extent exceeds the supported `i32` index range".to_owned(),
            }),
        );
        assert!(context.builder().borrow().instructions().is_empty());

        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(i32::MAX as usize + 1)).unwrap());
        let extent = context.input(extent_type.clone().into());
        let predicate = context
            .input(ArrayType::new(DataType::Boolean, Shape::new(vec![extent_type.simplified_dimension()])).into());
        let batching = BatchingContext::new(context.clone(), extent);
        let predicate = ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap();
        assert_eq!(
            AssertOperation::new("no room for padding")
                .with_labels(vec!["predicate".to_owned()])
                .batch(&batching, &EmptyRegionDriver, &[predicate.clone(), predicate.clone()])
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation {
                message: "padded assertion batch extent exceeds the supported `i32` extent range".to_owned(),
            }),
        );
        assert!(context.builder().borrow().instructions().is_empty());

        // No diagnostic array is gathered, so this boundary does not require a padding item.
        AssertOperation::new("no padding needed")
            .batch(&batching, &EmptyRegionDriver, &[predicate])
            .unwrap();
    }

    #[test]
    fn test_assert_batching_projected_dynamic() {
        // Exercise the projected policy's independent mapped/replicated roles and both diagnostic modes. Empty
        // batches pass even with a replicated false predicate; nonempty failures retain exact indices and values.
        let extent_type = DimensionType::new("batch", DimensionBounds::non_negative(Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.simplified_dimension()]);
        for mapped_condition in [false, true] {
            for mapped_observation in [false, true] {
                for limit in [None, NonZeroUsize::new(2)] {
                    let condition_type = if mapped_condition {
                        ArrayType::new(DataType::Boolean, shape.clone())
                    } else {
                        ArrayType::scalar(DataType::Boolean)
                    };
                    let observation_type = if mapped_observation {
                        ArrayType::new(DataType::I32, shape.clone())
                    } else {
                        ArrayType::scalar(DataType::I32)
                    };
                    let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                        |(extent, condition, observation)| {
                            let batching = BatchingContext::<
                                    _,
                                    ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
                                >::with_policy(
                                    ProjectedContext::new(extent.domain()), extent
                                );
                            let mut operation =
                                AssertOperation::<ArrayType>::new("projected").with_labels(vec!["value".to_owned()]);
                            if let Some(limit) = limit {
                                operation = operation.with_failure_limit(limit);
                            }
                            let condition = ArrayBatch::new(
                                condition.into_projected()?,
                                if mapped_condition { BatchAxis::new(0) } else { BatchAxis::replicated() },
                            )?;
                            let observation = ArrayBatch::new(
                                observation.into_projected()?,
                                if mapped_observation { BatchAxis::new(0) } else { BatchAxis::replicated() },
                            )?;
                            operation.batch(&batching, &EmptyRegionDriver, &[condition, observation])?;
                            Ok(())
                        },
                        (
                            ArrayIrType::Dimension(extent_type.clone()),
                            ArrayIrType::Array(condition_type),
                            ArrayIrType::Array(observation_type),
                        ),
                    )
                    .unwrap();
                    for extent in [0, 1, 3] {
                        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
                        let observation = if mapped_observation {
                            Array::vector((1..=extent).map(|index| index as i32 * 10).collect()).unwrap()
                        } else {
                            Array::scalar(17i32).unwrap()
                        };
                        let passing = if mapped_condition {
                            Array::vector(vec![true; extent]).unwrap()
                        } else {
                            Array::scalar(true).unwrap()
                        };
                        assert_eq!(
                            program.interpret((dimension.clone(), passing.into(), observation.clone().into())),
                            Ok(()),
                        );
                        let failing = if mapped_condition {
                            let mut values = vec![false; extent];
                            if extent > 1 {
                                values[0] = true;
                            }
                            Array::vector(values).unwrap()
                        } else {
                            Array::scalar(false).unwrap()
                        };
                        let result = program.interpret((dimension, failing.into(), observation.into()));
                        if extent == 0 {
                            assert_eq!(result, Ok(()));
                            continue;
                        }
                        let first = usize::from(mapped_condition && extent > 1);
                        let observation = |index: usize| {
                            (
                                "value".to_owned(),
                                if mapped_observation { ((index + 1) * 10).to_string() } else { "17".to_owned() },
                            )
                        };
                        let expected = if let Some(limit) = limit {
                            let failures = (first..extent)
                                .take(limit.get())
                                .map(|index| AssertionFailure {
                                    index: vec![index],
                                    observations: vec![observation(index)],
                                })
                                .collect::<Vec<_>>();
                            let omitted = extent - first - failures.len();
                            AssertionError::FailedElements { message: "projected".to_owned(), failures, omitted }
                        } else {
                            let mut observations = vec![observation(first)];
                            if mapped_condition || mapped_observation {
                                observations.push(("batch_index".to_owned(), first.to_string()));
                            }
                            AssertionError::Failed { message: "projected".to_owned(), observations }
                        };
                        assert_eq!(result.unwrap_err().downcast_custom::<AssertionError>(), Some(&expected));
                    }
                }
            }
        }
    }

    #[test]
    fn test_assert_batching_with_limit_rejects_ragged() {
        // Both carriers must reject the logical ragged signature before physical padding can become a failed element,
        // irrespective of whether the padding happens to contain true or false.
        let dimension = DimensionVariable::new("item", DimensionBounds::non_negative(Some(4)).unwrap());
        let extents = Array::vector(vec![1i32, 2]).unwrap();
        let expected = Err(BatchingError::UnsupportedOperation {
            message: "`assert` does not support bounded ragged dimension `item` on input 0".to_owned(),
        });
        for padding in [false, true] {
            let condition = Array::matrix(2, 3, vec![true, padding, padding, true, true, padding]).unwrap();
            let input = ragged_array_batch(condition.clone(), extents.clone(), &dimension, 1).unwrap();
            let context =
                BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
            assert_eq!(
                AssertOperation::<ArrayType>::new("logical items")
                    .with_failure_limit(NonZeroUsize::new(3).unwrap())
                    .batch(&context, &EmptyRegionDriver, &[input.clone()])
                    .map(|_| ()),
                expected,
            );
            let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            );
            assert_eq!(
                AssertOperation::<ArrayType>::new("logical items")
                    .with_failure_limit(NonZeroUsize::new(3).unwrap())
                    .batch(&context, &EmptyRegionDriver, &[input])
                    .map(|_| ()),
                expected,
            );
            let input = ragged_batch(condition.into(), extents.clone().into(), &dimension, 1).unwrap();
            let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
                EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                DimensionValue::constant(2).unwrap().into(),
            );
            assert_eq!(
                AssertOperation::<ArrayIrType>::new("logical items")
                    .with_failure_limit(NonZeroUsize::new(3).unwrap())
                    .batch(&context, &EmptyRegionDriver, &[input])
                    .map(|_| ()),
                expected,
            );
        }
    }

    #[test]
    fn test_assert_batching_with_limit() {
        // Mapped inputs retain separate failures and their observations.
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                condition.assert_with_limit("valid", &[("value", observation)], NonZeroUsize::new(2).unwrap())
            },
            (Array::vector(vec![true, false, false]).unwrap(), Array::vector(vec![10i32, 20, 30]).unwrap()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            (),
            None,
        );
        let BatchingError::Program(error) = output.unwrap_err() else { panic!("expected an assertion failure") };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "valid".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![1], observations: vec![("value".to_owned(), "20".to_owned())] },
                    AssertionFailure { index: vec![2], observations: vec![("value".to_owned(), "30".to_owned())] },
                ],
                omitted: 0,
            }),
        );

        // A replicated false condition fails at every mapped item, subject to the reporting limit.
        let output: Result<(), BatchingError> = batch(
            |(condition, observation)| {
                condition.assert_with_limit("replicated", &[("value", observation)], NonZeroUsize::MIN)
            },
            (Array::scalar(false).unwrap(), Array::vector(vec![10i32, 20]).unwrap()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (),
            None,
        );
        let BatchingError::Program(error) = output.unwrap_err() else { panic!("expected an assertion failure") };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "replicated".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![0],
                    observations: vec![("value".to_owned(), "10".to_owned())],
                }],
                omitted: 1,
            }),
        );

        // Empty batches pass even when the replicated condition is false.
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
    fn test_assert_batching_with_limit_manual_variation() {
        type ArrayTrace = TracingContext<Array, ArrayOperation<Array>>;

        // Inside a manual region, each device records its own observations, so materializing an observation along
        // the batch axis keeps its own manual variation rather than taking that of the condition.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let condition_type = ArrayType::new_static(DataType::Boolean, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let observation_type = ArrayType::new_static(DataType::I32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let (_, program) = ArrayTrace::trace_with_named_axes(
            |(condition, observation): (Tracer<ArrayTrace>, Tracer<ArrayTrace>)| {
                let batching = BatchingContext::<_, ArrayBatchingPolicy>::new(condition.domain(), 2);
                AssertOperation::new("valid")
                    .with_labels(vec!["value".to_owned()])
                    .with_failure_limit(NonZeroUsize::MIN)
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayBatch::new(condition, BatchAxis::new(0))?,
                            ArrayBatch::new(observation, BatchAxis::new(0))?,
                        ],
                    )?;
                Ok(())
            },
            (condition_type, observation_type),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:i32[2][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %2:i32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast \
                    [output_type=i32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %1
                    () = assert [message=\"valid\", labels=[\"value\"], failure_limit=1] %0 %2
                in ()"
            },
        );
    }

    #[test]
    fn test_assert_batching_with_limit_nested() {
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
                Array::matrix(2, 2, vec![10i32, 20, 30, 40]).unwrap(),
            ),
            (BatchAxis::new(1), BatchAxis::new(1)),
            (),
            None,
        );
        let BatchingError::Program(error) = output.unwrap_err() else { panic!("expected an assertion failure") };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "nested".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![0, 1], observations: vec![("value".to_owned(), "30".to_owned())] },
                    AssertionFailure { index: vec![1, 0], observations: vec![("value".to_owned(), "20".to_owned())] },
                ],
                omitted: 1,
            }),
        );
    }

    #[test]
    fn test_assert_batching_with_limit_dynamic() {
        let extent_type = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let shape = Shape::new(vec![extent_type.simplified_dimension()]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, condition, observation)| {
                let batching = BatchingContext::new(extent.domain(), extent);
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
            Ok(()),
        );
        let error = program
            .interpret((
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 4).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![true, false, false, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10i32, 20, 30, 40]).unwrap()),
            ))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "dynamic".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![1], observations: vec![("value".to_owned(), "20".to_owned())] },
                    AssertionFailure { index: vec![2], observations: vec![("value".to_owned(), "30".to_owned())] },
                ],
                omitted: 1,
            }),
        );
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
            1,
        );
        assert!(
            linearization
                .tangent()
                .instructions()
                .iter()
                .all(|instruction| instruction.operation().name() != ASSERT_OPERATION_NAME),
        );
    }

    #[test]
    fn test_assert_staging() {
        // Symbolic conditions remain ordered effects, even with no outputs to keep them live.
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
            in ()"#},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(program.simplified().unwrap().instructions().len(), 1);

        // A known success disappears during tracing.
        let (_, folded) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.domain().lift(Array::scalar(true).unwrap())?;
                condition.assert("always true", &[])
            },
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        assert!(folded.instructions().is_empty());

        // A known failure is staged and reported only when the program executes.
        let (_, failing) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.domain().lift(Array::scalar(false).unwrap())?;
                condition.assert("always false", &[])
            },
            ArrayType::scalar(DataType::Boolean),
        )
        .unwrap();
        assert_eq!(failing.instructions().len(), 1);
        let error = failing.interpret(Array::scalar(true).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "always false".to_owned(), observations: vec![] }),
        );
    }

    #[test]
    fn test_assert_staging_with_limit() {
        // Tracing retains bounded reporting for execution.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |condition| condition.assert_with_limit("valid", &[], NonZeroUsize::new(2).unwrap()),
            ArrayType::new(DataType::Boolean, [3].into()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:bool[3] .
                let () = assert [message="valid", labels=[], failure_limit=2] %0
                in ()"#},
        );
        let error = program.interpret(Array::vector(vec![true, false, false]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "valid".to_owned(),
                failures: vec![
                    AssertionFailure { index: vec![1], observations: vec![] },
                    AssertionFailure { index: vec![2], observations: vec![] },
                ],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_assert_staging_projected() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition = context.lift(ArrayIrValue::Array(Array::scalar(true).unwrap())).unwrap();
        ValueProjection::<ArrayType>::into_projected(condition)
            .unwrap()
            .assert("projected true", &[])
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_assert_staging_validates_observations() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let condition = context.lift(Array::scalar(true).unwrap()).unwrap();

        // Validate observations before omitting a known success in either capability function.
        let invalid = context.lift(Array::vector(vec![1i32]).unwrap()).unwrap();
        assert_eq!(
            condition.assert("invalid observation", &[("value", invalid.clone())]),
            Err(TypeError::invalid("assertion observation `value` has unsupported type `i32[1]`").into()),
        );
        assert_eq!(
            condition.assert_with_limit("invalid observation", &[("value", invalid)], NonZeroUsize::MIN),
            Err(TypeError::invalid("assertion observation `value` has unsupported type `i32[1]`").into()),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_assert_staging_conditional() {
        let (_, failing) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                let condition = input.domain().lift(Array::scalar(false).unwrap())?;
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
            Ok(Vec::new()),
        );
        let error = program.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(true).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "selected failure".to_owned(), observations: Vec::new() }),
        );
    }

    #[test]
    fn test_assert_with_limit() {
        let error = Array::vector(vec![true, false])
            .unwrap()
            .assert_with_limit("valid", &[("value", Array::vector(vec![10i32, 20]).unwrap())], NonZeroUsize::MIN)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "valid".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![1],
                    observations: vec![("value".to_owned(), "20".to_owned())],
                }],
                omitted: 0,
            }),
        );
    }

    #[test]
    fn test_assertion_value_assertion_observation() {
        // Integer diagnostics preserve the entire unsigned and signed ranges without floating-point conversion.
        assert_eq!(Array::scalar(u64::MAX).unwrap().assertion_observation(), Ok("18446744073709551615".to_owned()));
        assert_eq!(Array::scalar(i64::MIN).unwrap().assertion_observation(), Ok("-9223372036854775808".to_owned()));
        assert_eq!(Array::scalar(true).unwrap().assertion_observation(), Ok("true".to_owned()));
        assert_eq!(Array::scalar(false).unwrap().assertion_observation(), Ok("false".to_owned()));

        // Floating-point observations use the logical scalar value, including low-precision representations.
        assert_eq!(Array::scalar(bf16::from_f32(1.5)).unwrap().assertion_observation(), Ok("1.5".to_owned()));
        assert_eq!(Array::scalar(f16::from_f32(-2.5)).unwrap().assertion_observation(), Ok("-2.5".to_owned()));
        assert_eq!(Array::scalar(3.25f32).unwrap().assertion_observation(), Ok("3.25".to_owned()));
        assert_eq!(Array::scalar(-4.75f64).unwrap().assertion_observation(), Ok("-4.75".to_owned()));

        // Composite values preserve array formatting and render dimensions as scalar extents.
        let array = ArrayIrValue::Array(Array::scalar(12i32).unwrap());
        assert_eq!(array.assertion_observation(), Ok("12".to_owned()));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(7).unwrap());
        assert_eq!(dimension.assertion_observation(), Ok("7".to_owned()));

        // Unsupported representations and unavailable captures have distinct concretization diagnostics.
        assert!(matches!(
            Array::scalar(Complex::new(1.0f32, 2.0)).unwrap().assertion_observation(),
            Err(ProgramError::Concretization { message })
                if message == "unsupported assertion observation type `c64[]`",
        ));
        assert!(matches!(
            Array::vector(vec![1i32]).unwrap().assertion_observation(),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete integer from `i32[1]`; expected a scalar integer",
        ));
        let capture = CaptureReference::new(0, ArrayType::scalar(DataType::I32));
        assert!(matches!(
            capture.assertion_observation(),
            Err(ProgramError::Concretization { message })
                if message == "cannot inspect a captured assertion observation before execution",
        ));
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1i32).unwrap()));
        assert_eq!(
            reference.assertion_observation(),
            Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() }),
        );
    }

    #[test]
    fn test_assertion_value_assertion_array() {
        // Both scalar and shaped arrays retain their concrete representation through the composite wrapper.
        let scalar = Array::scalar(true).unwrap();
        assert_eq!(scalar.assertion_array(), Ok(Some(scalar.clone())));
        assert_eq!(ArrayIrValue::Array(scalar.clone()).assertion_array(), Ok(Some(scalar)));
        let array = Array::matrix(1, 2, vec![1i32, 2]).unwrap();
        assert_eq!(array.assertion_array(), Ok(Some(array.clone())));
        assert_eq!(ArrayIrValue::Array(array.clone()).assertion_array(), Ok(Some(array)));

        // Non-array scalars return `None`; an unavailable array must instead report an inspection error.
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(7).unwrap());
        assert_eq!(dimension.assertion_array(), Ok(None));
        let capture = CaptureReference::new(0, ArrayType::scalar(DataType::Boolean));
        assert!(matches!(
            capture.assertion_array(),
            Err(ProgramError::Concretization { message })
                if message == "cannot inspect a captured assertion array before execution",
        ));
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1i32).unwrap()));
        assert_eq!(
            reference.assertion_array(),
            Err(ProgramError::Concretization { message: "assertion observations cannot be references".to_owned() }),
        );
    }

    #[test]
    fn test_assertion_value_check_assertion() {
        // Scalar failures retain observation labels and values, while scalar successes have no diagnostic output.
        let observations = [("value", Array::scalar(7i32).unwrap())];
        assert_eq!(Array::scalar(true).unwrap().check_assertion("valid", &observations, None), Ok(()));
        let error = Array::scalar(false).unwrap().check_assertion("valid", &observations, None).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "valid".to_owned(),
                observations: vec![("value".to_owned(), "7".to_owned())],
            }),
        );

        // Validation precedes success elision, including for vacuously successful empty conditions.
        assert_eq!(
            Array::scalar(true).unwrap().check_assertion(
                "invalid",
                &[("value", Array::vector(vec![1i32]).unwrap())],
                None,
            ),
            Err(TypeError::invalid("assertion observation `value` has unsupported type `i32[1]`").into()),
        );
        assert_eq!(
            Array::vector(Vec::<bool>::new()).unwrap().check_assertion(
                "invalid",
                &[("value", Array::vector(vec![1i32]).unwrap())],
                Some(NonZeroUsize::MIN),
            ),
            Err(TypeError::invalid("assertion observation `value` has unsupported type `i32[1]`").into()),
        );
        assert_eq!(
            Array::vector(Vec::<bool>::new()).unwrap().check_assertion("empty", &[], Some(NonZeroUsize::MIN)),
            Ok(()),
        );
        assert_eq!(
            Array::vector(vec![true, true]).unwrap().check_assertion("valid", &[], Some(NonZeroUsize::MIN)),
            Ok(()),
        );

        // A bounded scalar failure is the single rank-zero element, whose logical index is empty.
        let error = Array::scalar(false)
            .unwrap()
            .check_assertion("scalar", &observations, Some(NonZeroUsize::MIN))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "scalar".to_owned(),
                failures: vec![AssertionFailure {
                    index: vec![],
                    observations: vec![("value".to_owned(), "7".to_owned())],
                }],
                omitted: 0,
            }),
        );

        // Failures are sampled in row-major order; scalar arrays and dimensions are reused at every index.
        let condition = ArrayIrValue::Array(Array::matrix(2, 2, vec![false, true, false, false]).unwrap());
        let error = condition
            .check_assertion(
                "elements",
                &[
                    ("value", ArrayIrValue::Array(Array::matrix(2, 2, vec![10i32, 20, 30, 40]).unwrap())),
                    ("scalar", ArrayIrValue::Array(Array::scalar(7i32).unwrap())),
                    ("extent", ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())),
                ],
                NonZeroUsize::new(2),
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::FailedElements {
                message: "elements".to_owned(),
                failures: vec![
                    AssertionFailure {
                        index: vec![0, 0],
                        observations: vec![
                            ("value".to_owned(), "10".to_owned()),
                            ("scalar".to_owned(), "7".to_owned()),
                            ("extent".to_owned(), "4".to_owned()),
                        ],
                    },
                    AssertionFailure {
                        index: vec![1, 0],
                        observations: vec![
                            ("value".to_owned(), "30".to_owned()),
                            ("scalar".to_owned(), "7".to_owned()),
                            ("extent".to_owned(), "4".to_owned()),
                        ],
                    },
                ],
                omitted: 1,
            }),
        );
    }
    #[test]
    fn test_guard_operation() {
        let (n, declared) = guard_signature();
        let guard = GuardOperation::new("region", declared.clone());
        assert_eq!(guard.name(), GUARD_OPERATION_NAME);
        assert_eq!(guard.boundary(), "region");
        assert_eq!(guard.declared_types(), declared.as_slice());
        assert_eq!(guard.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(guard.to_string(), r#"guard [boundary="region", types=[f32[n], f32[n]]]"#);

        // Proven entries need no guard, while mixed entries guard every member that carries an unproven identity.
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![m.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        assert_eq!(GuardOperation::for_entry("region", &declared, &declared), Ok(None));
        assert_eq!(GuardOperation::for_entry("region", &declared, &[exact.clone(), exact.clone()]), Ok(None));
        assert_eq!(
            GuardOperation::for_entry("region", &declared, &[declared[0].clone(), exact.clone()]),
            Ok(Some((GuardOperation::new("region", declared.clone()), vec![0, 1]))),
        );
        assert_eq!(
            GuardOperation::for_entry(
                "region",
                &declared,
                &[exact.clone(), ArrayType::new_static(DataType::F32, [4]).into()]
            ),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 3,
                actual: 4,
            })),
        );
        assert_eq!(
            GuardOperation::for_entry("region", &declared, &[symbolic.clone(), symbolic.clone()]),
            Err(TypeError::invalid("dimension m does not refine declared dimension n")),
        );
        assert_eq!(
            GuardOperation::for_entry("region", &declared, &[]),
            Err(TypeError::invalid("expected 2 inputs but got 0")),
        );
        assert_eq!(
            GuardOperation::for_entry("region", &declared, &[exact.clone(), exact.clone(), exact.clone()]),
            Err(TypeError::invalid("expected 2 inputs but got 3")),
        );

        // An input whose supplied type the declared signature narrows is guarded even without an unproven identity,
        // so that its outputs carry that narrowing. Here, the actual `q` entering the declared `p` is fixed to 3, which
        // narrows the `f32[q]` member even though its declared identity `q` is proven.
        let p = DimensionType::new("p", DimensionBounds::positive(Some(8)).unwrap());
        let q = DimensionType::new("q", DimensionBounds::positive(Some(8)).unwrap());
        let vector = |dimension: &DimensionType| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()])).into()
        };
        let chained_actual = vec![q.clone().into(), exact.clone(), vector(&q)];
        let (chained_guard, members) =
            GuardOperation::for_entry("region", &[p.clone().into(), vector(&p), vector(&q)], &chained_actual)
                .unwrap()
                .unwrap();
        assert_eq!(members, vec![0, 1, 2]);
        assert_eq!(
            chained_guard.infer_output_types(&chained_actual, &[]),
            Ok(vec![DimensionType::from(Dimension::Static(3)).into(), exact.clone(), exact.clone()]),
        );

        // Reference members take part in the relationships through their referents' shapes, which another member may
        // contradict. The guard reads each reference member, and produces outputs only for the other members.
        let reference: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        let declared_reference: ArrayIrType =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]))).into();
        let declared_with_reference = vec![declared[0].clone(), declared_reference.clone(), declared[1].clone()];
        assert_eq!(
            GuardOperation::for_entry(
                "region",
                &declared_with_reference,
                &[declared[0].clone(), reference.clone(), exact.clone()],
            ),
            Ok(Some((GuardOperation::new("region", declared_with_reference), vec![0, 1, 2]))),
        );
        let (reference_guard, members) = GuardOperation::for_entry(
            "region",
            &[declared[0].clone(), declared_reference.clone()],
            &[declared[0].clone(), reference.clone()],
        )
        .unwrap()
        .unwrap();
        assert_eq!(members, vec![0, 1]);
        assert_eq!(reference_guard.declared_types(), &[declared[0].clone(), declared_reference]);
        assert_eq!(
            reference_guard.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Read }],
        );
        assert!(reference_guard.effects().classes().contains(EffectClass::OrderedAssertion));
        assert_eq!(reference_guard.infer_output_types(&[declared[0].clone(), reference], &[]), Ok(vec![exact.clone()]));

        // The guard's declared identities are local, so a renaming of the surrounding program, here to an identity with
        // narrower bounds, leaves them and the extents they admit unchanged.
        let narrow = DimensionType::new("narrow", DimensionBounds::positive(Some(4)).unwrap());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(n.variable().unwrap().clone(), narrow.variable().unwrap().clone()).unwrap();
        let renamed = guard.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed, guard);
        let five: ArrayIrType = ArrayType::new_static(DataType::F32, [5]).into();
        assert_eq!(renamed.infer_output_types(&[five.clone(), five.clone()], &[]), Ok(vec![five.clone(), five]));
    }

    #[test]
    fn test_guard_type_inference() {
        let (_, declared) = guard_signature();
        let guard = GuardOperation::new("region", declared.clone());
        let symbolic = declared[0].clone();
        let exact = |extent: usize| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));

        // Mixed refinements are admitted exactly like the program or region call the guard replaces, and the outputs
        // are the inputs narrowed by the facts that the declared relationships prove.
        assert_eq!(guard.infer_output_types(&[symbolic.clone(), exact(3)], &[]), Ok(vec![exact(3), exact(3)]));
        assert_eq!(guard.infer_output_types(&[symbolic.clone(), symbolic.clone()], &[]), Ok(declared.clone()));
        assert_eq!(
            guard.infer_output_types(&[exact(3), exact(4)], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 3,
                actual: 4,
            })),
        );
        assert_eq!(
            guard.infer_output_types(&[symbolic.clone()], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );

        // A reference member is checked through its referent's shape and produces no output.
        let ArrayIrType::Array(symbolic_array) = &symbolic else { unreachable!() };
        let declared_reference: ArrayIrType = ReferenceType::new(symbolic_array.clone()).into();
        let reference = |extent: usize| -> ArrayIrType {
            ReferenceType::new(ArrayType::new_static(DataType::F32, [extent])).into()
        };
        let reference_guard = GuardOperation::new("region", vec![declared_reference, symbolic.clone()]);
        assert_eq!(reference_guard.infer_output_types(&[reference(3), exact(3)], &[]), Ok(vec![exact(3)]));
        assert_eq!(
            reference_guard.infer_output_types(&[reference(3), exact(2)], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 3,
                actual: 2,
            })),
        );
    }

    #[test]
    fn test_guard_retained_identities() {
        // `[f32[p, q], f32[p], f32[q]]` entered with `[f32[a, b], f32[2], f32[3]]` fixes `a = 2` and `b = 3`. A guard
        // that retains `a` narrows only `b`, and gives the exact member of the class of `a` the identity `a` too, so
        // that its outputs agree. It renders its retained identities and renames them with the program, while its
        // declared identities stay local.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, a, b, c] = ["p", "q", "a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(dimensions)).into()
        };
        let declared = vec![
            array(vec![p.simplified_dimension(), q.simplified_dimension()]),
            array(vec![p.simplified_dimension()]),
            array(vec![q.simplified_dimension()]),
        ];
        let exact = |extent: usize| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));
        let inputs = vec![array(vec![a.simplified_dimension(), b.simplified_dimension()]), exact(2), exact(3)];
        let guard = GuardOperation::new("region", declared.clone());
        assert_eq!(guard.retained(), &[]);
        assert_eq!(
            guard.infer_output_types(&inputs, &[]),
            Ok(vec![array(vec![Dimension::Static(2), Dimension::Static(3)]), exact(2), exact(3)]),
        );
        let guard = guard.with_retained(vec![a.to_variable()]);
        assert_eq!(guard.retained(), &[a.to_variable()]);
        assert_eq!(
            guard.infer_output_types(&inputs, &[]),
            Ok(vec![
                array(vec![a.simplified_dimension(), Dimension::Static(3)]),
                array(vec![a.simplified_dimension()]),
                exact(3)
            ]),
        );
        assert_eq!(guard.to_string(), r#"guard [boundary="region", types=[f32[p, q], f32[p], f32[q]], retained=[a]]"#,);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(a.to_variable(), c.to_variable()).unwrap();
        let renamed = guard.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.retained(), &[c.to_variable()]);
        assert_eq!(renamed.declared_types(), declared.as_slice());
    }

    #[test]
    fn test_guard_retained_reference_anchors() {
        // Unknown dimension inputs join both reference classes, but each reference keeps its own nominal shape.
        // Guard-local declared names also survive an ambient renaming of the supplied reference identities.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, a, b, c] = ["p", "q", "a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let array = |first: &DimensionType, second: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![first.simplified_dimension(), second.simplified_dimension()]))
        };
        let reference = |dimension: &DimensionType| -> ArrayIrType {
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))).into()
        };
        let declared = vec![reference(&p), reference(&q), p.clone().into(), q.clone().into(), array(&p, &q).into()];
        let actual = vec![
            reference(&a),
            reference(&b),
            c.clone().into(),
            c.clone().into(),
            ArrayType::new_static(DataType::F32, [3, 3]).into(),
        ];
        let guard = GuardOperation::new("region", declared).with_retained(vec![a.to_variable(), b.to_variable()]);
        assert_eq!(
            guard.infer_output_types(&actual, &[]),
            Ok(vec![a.clone().into(), b.clone().into(), array(&a, &b).into()]),
        );

        // A crossed retained scalar is a relationship witness, not another immutable handle anchor. Selecting every
        // actual retained member would join the anchor-only classes and change these outputs to `b`, `a`, and `[a, a]`.
        let crossed = vec![reference(&a), reference(&b), b.clone().into(), c.clone().into(), array(&c, &c).into()];
        let expected = vec![a.clone().into(), b.clone().into(), array(&a, &b).into()];
        assert_eq!(guard.infer_output_types(&crossed, &[]), Ok(expected.clone()));
        let referents = |types: &[ArrayIrType]| {
            types
                .iter()
                .map(|r#type| match r#type {
                    ArrayIrType::Reference(reference) => ArrayIrType::Array(reference.referent().clone()),
                    r#type => r#type.clone(),
                })
                .collect::<Vec<_>>()
        };
        let discharged = GuardOperation { declared_types: referents(guard.declared_types()), ..guard.clone() };
        let outputs = discharged.infer_output_types(&referents(&crossed), &[]).unwrap();
        assert_eq!(&outputs[2..], expected.as_slice());
        assert_eq!(discharged.anchors, vec![0, 1]);

        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(a.to_variable(), p.to_variable()).unwrap();
        renaming.insert(b.to_variable(), q.to_variable()).unwrap();
        let renamed = discharged.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.anchors, discharged.anchors);
        assert_eq!(
            renamed
                .infer_output_types(
                    &referents(&crossed)
                        .iter()
                        .map(|r#type| r#type.rename_identities(&renaming).unwrap())
                        .collect::<Vec<_>>(),
                    &[],
                )
                .unwrap()[2..],
            expected.iter().map(|r#type| r#type.rename_identities(&renaming).unwrap()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_guard_retaining() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, c] = ["p", "q", "c"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let reference = |dimension: &DimensionType| ArrayIrType::from(ReferenceType::new(vector(dimension)));

        // A reference's shape wins over a known representation independently of member order.
        let declared = vec![vector(&p).into(), reference(&q), p.clone().into(), q.clone().into()];
        let actual = vec![vector(&p).into(), reference(&q), c.clone().into(), c.clone().into()];
        let pinned = vec![p.to_variable()];
        let guard = GuardOperation::new("region", declared.clone()).retaining(&actual, &pinned).unwrap();
        assert_eq!(guard.retained(), &[q.to_variable()]);
        assert_eq!(
            guard.infer_output_types(&actual, &[]),
            Ok(vec![vector(&q).into(), q.clone().into(), q.clone().into()])
        );
        let reversed_declared = declared.into_iter().rev().collect::<Vec<_>>();
        let reversed_actual = actual.into_iter().rev().collect::<Vec<_>>();
        let guard = GuardOperation::new("region", reversed_declared).retaining(&reversed_actual, &pinned).unwrap();
        assert_eq!(guard.retained(), &[q.to_variable()]);

        // Several immutable names in one complete class keep their own formal anchors.
        let declared = vec![reference(&p), reference(&q), p.clone().into(), q.clone().into()];
        let actual = vec![reference(&p), reference(&q), c.clone().into(), c.clone().into()];
        let guard = GuardOperation::new("region", declared).retaining(&actual, &[]).unwrap();
        assert_eq!(guard.retained(), &[p.to_variable(), q.to_variable()]);
        assert_eq!(guard.infer_output_types(&actual, &[]), Ok(vec![p.clone().into(), q.into()]));

        // An exact reference shape has priority even when it carries no nominal identity.
        let declared = vec![reference(&p), vector(&p).into()];
        let actual = vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into(), vector(&p).into()];
        let guard = GuardOperation::new("region", declared).retaining(&actual, &[p.to_variable()]).unwrap();
        assert_eq!(guard.retained(), &[]);
        assert_eq!(guard.infer_output_types(&actual, &[]), Ok(vec![ArrayType::new_static(DataType::F32, [3]).into()]));
    }

    #[test]
    fn test_guard_scoped_declarations() {
        // A guard's declared identities are local, so inputs whose identities a consistent renaming of the surrounding
        // program changed (here, from `n` to `m`) still enter them, while the declared bounds still bound the inputs.
        let (n, declared) = guard_signature();
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let wide = DimensionType::new("wide", DimensionBounds::positive(Some(16)).unwrap());
        let vector = |dimension: &DimensionType| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()])).into()
        };
        let exact = |extent: usize| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));
        let guard = GuardOperation::new("region", declared.clone());
        assert_eq!(guard.infer_output_types(&[vector(&m), exact(2)], &[]), Ok(vec![exact(2), exact(2)]));
        assert_eq!(guard.infer_output_types(&[vector(&m), vector(&m)], &[]), Ok(vec![vector(&m), vector(&m)]));
        assert_eq!(
            guard.infer_output_types(&[vector(&wide), exact(2)], &[]),
            Err(TypeError::invalid("dimension wide does not refine declared dimension n")),
        );

        // Renaming a program that contains a guard, and then replaying it at the renamed types, is valid.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = builder.add_input(vector(&n));
        let second = builder.add_input(exact(2));
        let outputs = builder.add_instruction(guard, Vec::new(), vec![first, second], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(n.variable().unwrap().clone(), m.variable().unwrap().clone()).unwrap();
        let renamed = program.rename_type_identities(&renaming).unwrap();
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = vec![context.input(vector(&m)), context.input(exact(2))];
        assert!(renamed.interpret_in_context(&context, inputs).is_ok());
        let vector_value = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        assert_eq!(
            renamed.interpret(vec![vector_value(2), vector_value(2)]),
            Ok(vec![vector_value(2), vector_value(2)])
        );
        assert!(renamed.interpret(vec![vector_value(3), vector_value(2)]).is_err());
    }

    #[test]
    fn test_guard_reference_discharge() {
        // Discharging a reference member checks the referent's current value at the declared referent type, and the
        // rebuilt guard's output for that member is dropped, so the outputs still match the non-reference members.
        let (_, declared) = guard_signature();
        let ArrayIrType::Array(vector_type) = &declared[0] else { unreachable!() };
        let reference_type: ArrayIrType = ReferenceType::new(vector_type.clone()).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(reference_type.clone());
        let array = builder.add_input(declared[0].clone());
        let outputs = builder
            .add_instruction(
                GuardOperation::new("region", vec![reference_type, declared[0].clone()]),
                Vec::new(),
                vec![reference, array],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[n], %3:f32[n] = guard [boundary="region", types=[f32[n], f32[n]], anchors=[0]] %0 %1
                in (%3)"#},
        );
    }

    #[test]
    fn test_guard_reference_discharge_preserves_partial_outputs() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, c] = ["p", "q", "c"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let reference = |dimension: &DimensionType| ArrayIrType::from(ReferenceType::new(vector(dimension)));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(reference(&p));
        let reference_input = builder.add_input(reference(&q));
        builder.add_input(p.clone().into());
        let dimension = builder.add_input(q.clone().into());
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference_input], None).unwrap()[0];
        let zero = builder.add_instruction(ZeroOperation::new(vector(&q)), vec![], vec![dimension], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), vec![], vec![read, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![sum],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(reference(&p)),
                PartialValue::Unknown(reference(&q)),
                PartialValue::Unknown(c.clone().into()),
                PartialValue::Unknown(c.into()),
            ])
            .unwrap();
        let discharged = evaluation.program().clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {r#"
                lambda %0:f32[p], %1:f32[q], %2:dimension<c ∈ [1, 8)>, %3:dimension<c ∈ [1, 8)> .
                let %4:f32[p], %5:f32[q], %6:dimension<p ∈ [1, 8)>, %7:dimension<q ∈ [1, 8)> = guard [
                    boundary="partial evaluation",
                    types=[f32[p], f32[q], dimension<p ∈ [1, 8)>, dimension<q ∈ [1, 8)>],
                    retained=[p, q],
                    anchors=[0, 1],
                ] %0 %1 %2 %3
                    %8:f32[q] = zero [type=f32[q]] %7
                    %9:f32[q] = add %1 %8
                in (%9)"#},
        );
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let array = |extent| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let reference =
            |extent| ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
        for extent in [3, 4] {
            let original_inputs = vec![reference(3), reference(extent), dimension.clone(), dimension.clone()];
            let direct = program.interpret(original_inputs.clone());
            let expected = if extent == 3 {
                Ok(vec![array(3)])
            } else {
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "q".to_owned(),
                    expected: 4,
                    actual: 3,
                })))
            };
            assert_eq!(direct, expected);
            assert_eq!(evaluation.interpret(&EagerContext::new(), &original_inputs), direct);
            assert_eq!(
                discharged.program().interpret(vec![array(3), array(extent), dimension.clone(), dimension.clone()]),
                direct,
            );
        }
    }

    #[test]
    fn test_guard_reference_discharge_anchors_reference_only_guards() {
        // Partially evaluating the sum of two reads, from a symbolic `ref<f32[n]>` and from a reference whose supplied
        // type is the exact `ref<f32[3]>`, checks the references with an entry guard that has no outputs. Discharge
        // turns it into a guard over the referents, anchored at both members, whose outputs are unused because the
        // discharged reads become the referent inputs themselves, in either read order.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference: ArrayIrType = ReferenceType::new(vector).into();
        let exact: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        for reversed_reads in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let first = builder.add_input(reference.clone());
            let second = builder.add_input(reference.clone());
            let references = if reversed_reads { [second, first] } else { [first, second] };
            let reads = references
                .into_iter()
                .map(|reference| {
                    builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()
                        [0]
                })
                .collect();
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), reads, None).unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![sum],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            let evaluation = program
                .partially_evaluate(&[PartialValue::Unknown(reference.clone()), PartialValue::Unknown(exact.clone())])
                .unwrap();
            let discharged = evaluation.program().clone().discharge_references(0).unwrap();
            let operands = if reversed_reads { "%1 %0" } else { "%0 %1" };
            assert_eq!(
                discharged.program().to_string(),
                formatdoc! {r#"
                lambda %0:f32[n], %1:f32[3] .
                let %2:f32[n], %3:f32[n] = guard [
                    boundary="partial evaluation",
                    types=[f32[n], f32[n]],
                    retained=[n],
                    anchors=[0, 1],
                ] %0 %1
                    %4:f32[n], %5:f32[n] = guard [boundary="add", types=[f32[n], f32[n]], retained=[n]] {operands}
                    %6:f32[n] = add %4 %5
                in (%6)"#},
            );
            let vector = |extent| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
            assert_eq!(
                discharged.program().interpret(vec![vector(3), vector(3)]),
                Ok(vec![ArrayIrValue::Array(Array::vector(vec![2f32; 3]).unwrap())]),
            );
            assert_eq!(
                discharged.program().interpret(vec![vector(4), vector(3)]),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_string(),
                    expected: 4,
                    actual: 3,
                }))),
            );
        }
    }

    #[test]
    fn test_guard_interpretation() {
        let (_, declared) = guard_signature();
        let guard = GuardOperation::<ArrayIrType>::new("region", declared);
        let vector = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[vector(3), vector(3)]),
            Ok(vec![vector(3), vector(3)]),
        );
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[vector(3), vector(4)]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 3,
                actual: 4,
            }))),
        );
    }

    #[test]
    fn test_guard_interpretation_symbolic_relationships() {
        // Symbolic values carry no concrete extents, so a declared relationship that still needs a runtime check is
        // rejected rather than trusted, exactly as for `cast`; a relationship the types prove needs no check.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b, n] = ["a", "b", "n"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let first = context.input(vector(&a));
        let second = context.input(vector(&b));
        let guard = GuardOperation::new("region", vec![vector(&n), vector(&n)]);
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[first.clone(), second]),
            Err(ProgramError::Concretization {
                message: "`guard` cannot check symbolic type relationships during interpretation; bind the \
                          operation through its context"
                    .to_owned(),
            }),
        );
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[first.clone(), first.clone()]),
            Ok(vec![first.clone(), first]),
        );
    }

    #[test]
    fn test_guard_partial_evaluation() {
        let (n, declared) = guard_signature();
        let vector_type = declared[0].clone();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = builder.add_input(vector_type.clone());
        let second = builder.add_input(vector_type);
        builder
            .add_instruction(GuardOperation::new("region", declared), Vec::new(), vec![first, second], None)
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let vector = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());

        // Known inputs fold into the eager parent, which checks them immediately.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(vector(3)), PartialValue::Known(vector(3))])
            .unwrap();
        assert_eq!(evaluation.program().to_string(), "lambda  .\nin ()");

        // An unknown input keeps the guard, and its known witness, in the residual program. That program accepts
        // the supplied `f32[n]`, and the root program guard returns it narrowed to `f32[3]` for the replayed body.
        let unknown: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(vector(3)), PartialValue::Unknown(unknown)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[3] .
                let %2:f32[3], %3:f32[3] = guard [boundary="partial evaluation", types=[f32[n], f32[n]]] %1 %0
                    %4:f32[3], %5:f32[3] = guard [boundary="region", types=[f32[n], f32[n]]] %1 %3
                in ()"#},
        );
    }

    #[test]
    fn test_guard_batching() {
        // A mapped `f32[4]` member and a replicated `f32[n]` member are guarded once over the stacked values, with the
        // batch axis inserted into the mapped member's declared type.
        let (n, declared) = guard_signature();
        let batch = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, mapped, replicated)| {
                let batching = BatchingContext::new(extent.domain(), extent);
                GuardOperation::new("region", declared.clone()).batch(
                    &batching,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::new(mapped, BatchAxis::new(0))?, ArrayIrBatch::replicated(replicated)],
                )?;
                Ok(())
            },
            (
                ArrayIrType::Dimension(batch.clone()),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![batch.simplified_dimension(), Dimension::Static(4)]),
                )),
                ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]))),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:dimension<batch ∈ [0, 5)>, %1:f32[batch, 4], %2:f32[n] .
                let %3:f32[batch, 4], %4:f32[4] = guard [boundary="region", types=[f32[batch, n], f32[n]]] %1 %2
                in ()"#},
        );
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let mapped = ArrayIrValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 4]), &[1.0; 8]).unwrap(),
        );
        let replicated = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        assert_eq!(program.interpret((extent.clone(), mapped.clone(), replicated(4))), Ok(()));
        assert_eq!(
            program.interpret((extent, mapped, replicated(3))),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 4,
                actual: 3,
            }))),
        );
    }

    #[test]
    fn test_guard_batching_cast() {
        // Entry guards feed casts in replay and derivative programs. Both must lift the original declaration rather
        // than reconstructing its layout or manual variation from the already packed operand.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        for (declared, output, axis_sharding) in [
            (
                plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4]))),
                plain.clone(),
                ShardingDimension::replicated(),
            ),
            (
                plain.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
                plain,
                ShardingDimension::sharded(["x"]),
            ),
        ] {
            let (_, source) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value| {
                    let context = value.domain();
                    let guarded =
                        context.bind(GuardOperation::new("entry", vec![declared.clone()]), Vec::new(), &[value])?;
                    let mut outputs = context.bind(
                        CastOperation::checked(output.clone(), vec![declared.clone()]).unwrap(),
                        Vec::new(),
                        &guarded,
                    )?;
                    Ok(outputs.remove(0))
                },
                declared.clone(),
            )
            .unwrap();
            let expected = if declared.sharding().is_some() {
                formatdoc! {r#"
                    lambda %0:{declared} .
                    let %1:{declared} = guard [boundary="entry", types=[{declared}]] %0
                        %2:{output} = cast [
                            type={output},
                            inputs=[{declared}],
                            checked=true,
                        ] %1
                    in (%2)"#}
            } else {
                formatdoc! {r#"
                    lambda %0:{declared} .
                    let %1:{declared} = guard [boundary="entry", types=[{declared}]] %0
                        %2:{output} = cast [type={output}, inputs=[{declared}], checked=true] %1
                    in (%2)"#}
            };
            assert_eq!(source.to_string(), expected);
            for extent in [0, 3] {
                let packed = declared.batched(0, Dimension::Static(extent), axis_sharding.clone()).unwrap();
                let (program, axes) = source
                    .to_flat_program()
                    .batched(
                        extent,
                        axis_sharding.clone(),
                        &[BatchAxis::new(0)],
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )
                    .unwrap()
                    .into_parts();
                assert_eq!(axes, vec![BatchAxis::new(0)]);
                let expected = if packed.sharding().is_some() {
                    formatdoc! {r#"
                        lambda %0:{packed} .
                        let %1:{packed} = guard [
                            boundary="entry",
                            types=[{packed}],
                        ] %0
                        in (%1)"#}
                } else {
                    formatdoc! {r#"
                        lambda %0:{packed} .
                        let %1:{packed} = guard [boundary="entry", types=[{packed}]] %0
                        in (%1)"#}
                };
                assert_eq!(program.to_string(), expected);
                let value = Array::from_elements(packed, &vec![1f32; extent * 2]).unwrap();
                assert_eq!(program.interpret(vec![value.clone()]), Ok(vec![value]));
            }
        }
    }

    #[test]
    fn test_guard_batching_policy_bounds() {
        // Guards and casts only lift their declarations through `BatchingPolicy::batched_type`, so these generic bodies
        // must compile without an array extent policy, assertion preparation, or numerical diagnostic capabilities.
        // Concrete built-in contexts alone would hide accidentally stronger bounds.
        fn static_policy<C: Context<Type = ArrayType>, P: BatchingPolicy<C, Batch = ArrayBatch<C::Value>>>()
        where
            C::Operation: From<GuardOperation<ArrayType>> + From<CastOperation<ArrayType>>,
        {
            let _ = <GuardOperation<ArrayType> as BatchableOperation<C, ArrayBatchingPolicy<P>>>::batch::<
                EmptyRegionDriver,
            >;
            let _ =
                <CastOperation<ArrayType> as BatchableOperation<C, ArrayBatchingPolicy<P>>>::batch::<EmptyRegionDriver>;
        }
        fn mixed_policy<C: Context<Type = ArrayIrType>>()
        where
            C::Operation: From<GuardOperation<ArrayIrType>> + From<CastOperation<ArrayIrType>>,
        {
            let _ = <GuardOperation<ArrayIrType> as BatchableOperation<C, ArrayIrBatchingPolicy>>::batch::<
                EmptyRegionDriver,
            >;
            let _ = <CastOperation<ArrayIrType> as BatchableOperation<C, ArrayIrBatchingPolicy>>::batch::<
                EmptyRegionDriver,
            >;
        }
        fn projected_policy<C: Context<Type = ArrayIrType> + DomainProjection<ArrayType>>()
        where
            C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            C::Operation: OperationProjection<
                    ArrayType,
                    Projected: From<GuardOperation<ArrayType>> + From<CastOperation<ArrayType>>,
                >,
        {
            let _ = <GuardOperation<ArrayType> as BatchableOperation<
                ProjectedContext<C, ArrayType>,
                ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
            >>::batch::<EmptyRegionDriver>;
            let _ = <CastOperation<ArrayType> as BatchableOperation<
                ProjectedContext<C, ArrayType>,
                ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
            >>::batch::<EmptyRegionDriver>;
        }
        static_policy::<EagerContext<Array, ArrayOperation<Array>>, StaticArrayExtentBatchingPolicy>();
        mixed_policy::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>();
        projected_policy::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>();
    }

    #[test]
    fn test_guard_batching_local_identities() {
        // Reusing the guard's local name as the ambient batch dimension must not equate batch and item extents.
        // Keep the distinct-name control and check that the original per-item equality still rejects a mismatch.
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let n = DimensionType::new("n", bounds);
        let item = DimensionType::new("item", bounds);
        let declared =
            vec![ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]),)); 2];
        for batch in [n.clone(), DimensionType::new("batch", bounds)] {
            let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |(extent, mapped, replicated)| {
                    let batching = BatchingContext::new(extent.domain(), extent);
                    let outputs = GuardOperation::new("region", declared.clone())
                        .with_retained(vec![batch.to_variable()])
                        .batch(
                            &batching,
                            &EmptyRegionDriver,
                            &[ArrayIrBatch::new(mapped, BatchAxis::new(0))?, ArrayIrBatch::replicated(replicated)],
                        )?
                        .into_parts()
                        .0;
                    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                    assert_eq!(outputs[1].batch_axis(), BatchAxis::replicated());
                    Ok(outputs.into_iter().map(|output| output.value().clone()).collect::<Vec<_>>())
                },
                (
                    ArrayIrType::Dimension(batch.clone()),
                    ArrayIrType::Array(ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![batch.simplified_dimension(), Dimension::Static(3)]),
                    )),
                    ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![item.simplified_dimension()]))),
                ),
            )
            .unwrap();
            assert_eq!(
                program.output_types(),
                vec![
                    ArrayType::new(DataType::F32, Shape::new(vec![batch.simplified_dimension(), Dimension::Static(3)]))
                        .into(),
                    ArrayType::new_static(DataType::F32, [3]).into(),
                ],
            );
            for extent in [0, 2, 4] {
                let dimension = ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
                let mapped = ArrayIrValue::Array(
                    Array::from_elements::<f32>(
                        ArrayType::new_static(DataType::F32, [extent, 3]),
                        &vec![1.0; extent * 3],
                    )
                    .unwrap(),
                );
                let replicated = |extent| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
                assert_eq!(
                    program.interpret((dimension.clone(), mapped.clone(), replicated(3))),
                    Ok(vec![mapped.clone(), replicated(3)]),
                );
                assert_eq!(
                    program.interpret((dimension, mapped, replicated(4))),
                    Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                        dimension: "n".to_owned(),
                        expected: 3,
                        actual: 4,
                    }))),
                );
            }
        }
    }

    #[test]
    fn test_guard_batching_manual_sharding() {
        // Removing the physical batch placement exposes manual variation in the per-item type. Lifting the checked
        // declaration must restore that placement for homogeneous arrays, composite arrays, and reference referents.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let packed = ArrayType::new_static(DataType::F32, [3, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        let varying = packed
            .clone()
            .with_sharding(packed.sharding().unwrap().clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        for packed in [packed, varying] {
            let per_item = packed.unbatched(BatchAxis::new(-1)).unwrap();
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |mapped| {
                    let batching = BatchingContext::<_, ArrayBatchingPolicy>::new(mapped.domain(), 2)
                        .with_axis_sharding(ShardingDimension::sharded(["x"]));
                    let outputs = GuardOperation::new("region", vec![per_item.clone()])
                        .batch(&batching, &EmptyRegionDriver, &[ArrayBatch::new(mapped, BatchAxis::new(-1))?])?
                        .into_parts()
                        .0;
                    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
                    Ok(outputs[0].value().clone())
                },
                packed.clone(),
            )
            .unwrap();
            assert_eq!(program.output_types(), vec![packed.clone()]);
            for reference in [false, true] {
                let declared = if reference {
                    ArrayIrType::Reference(ReferenceType::new(per_item.clone()))
                } else {
                    ArrayIrType::Array(per_item.clone())
                };
                let actual = if reference {
                    ArrayIrType::Reference(ReferenceType::new(packed.clone()))
                } else {
                    ArrayIrType::Array(packed.clone())
                };
                let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                    |(extent, mapped)| {
                        let batching = BatchingContext::new(extent.domain(), extent)
                            .with_axis_sharding(ShardingDimension::sharded(["x"]));
                        let outputs = GuardOperation::new("region", vec![declared.clone()])
                            .batch(&batching, &EmptyRegionDriver, &[ArrayIrBatch::new(mapped, BatchAxis::new(-1))?])?
                            .into_parts()
                            .0;
                        assert_eq!(outputs.len(), usize::from(!reference));
                        Ok(outputs.into_iter().map(|output| output.value().clone()).collect::<Vec<_>>())
                    },
                    (ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2))), actual.clone()),
                )
                .unwrap();
                let ArrayIrOperation::Guard(guard) = program.instructions()[0].operation() else {
                    panic!("expected a guard");
                };
                assert_eq!(guard.declared_types(), &[actual.clone()]);
                assert_eq!(program.output_types(), if reference { vec![] } else { vec![actual] });
            }
        }
    }

    #[test]
    fn test_guard_batching_mapped_references() {
        // A mapped reference carries its batch axis in its referent, so the guard checks the referent shape of every
        // batch item at once, here against a replicated known dimension.
        let (n, _) = guard_signature();
        let declared: Vec<ArrayIrType> = vec![
            n.clone().into(),
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]))).into(),
        ];
        let extent = ArrayIrValue::Dimension(DimensionValue::new(n.clone(), 3).unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            extent,
        );
        let packed = |columns: usize| {
            ArrayIrValue::Reference(ArrayReference::new(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [3, columns]),
                    &vec![1.0; 3 * columns],
                )
                .unwrap(),
            ))
        };
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let guard = GuardOperation::new("region", declared);
        let outputs = guard
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::replicated(dimension.clone()),
                    ArrayIrBatch::new(packed(2), BatchAxis::new(0)).unwrap(),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value(), &dimension);
        assert!(
            guard
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::replicated(dimension), ArrayIrBatch::new(packed(4), BatchAxis::new(0)).unwrap()],
                )
                .is_err()
        );
    }

    #[test]
    fn test_guard_batching_rejects_ragged_carriers() {
        // Bounded ragged carriers hold their per-item extents outside their packed types, which a lifted guard cannot
        // relate, so both carrier families reject them before lifting the declarations.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let declared = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let message = "`guard` does not support bounded ragged dimension `length` on input 0".to_owned();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let input = ragged_array_batch(
            Array::matrix(2, 3, vec![1f32; 6]).unwrap(),
            Array::vector(vec![1i32, 3]).unwrap(),
            &length,
            1,
        )
        .unwrap();
        assert_eq!(
            GuardOperation::new("region", vec![declared.clone()])
                .batch(&context, &EmptyRegionDriver, &[input])
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation { message: message.clone() }),
        );
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ragged_batch(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1f32; 6]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap()),
            &length,
            1,
        )
        .unwrap();
        assert_eq!(
            GuardOperation::new("region", vec![declared.into()])
                .batch(&context, &EmptyRegionDriver, &[input])
                .map(|_| ()),
            Err(BatchingError::UnsupportedOperation { message }),
        );
    }

    #[test]
    fn test_guard_batching_rejects_ragged_carriers_in_eager_replay() {
        // Replaying `[f32[length], f32[length]]` through a batching context over an eager parent, at a bounded ragged
        // `f32[length]` and a replicated `f32[3]`, leaves their relationship unproven by the carrier types. Although
        // the context is eager, replay must bind the entry guard through the batching context, whose guard rule
        // rejects the ragged carrier, rather than check the packed values directly in the eager parent.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(vector_type.clone());
        builder.add_input(vector_type);
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let ragged = ragged_array_batch(
            Array::matrix(2, 3, vec![1f32; 6]).unwrap(),
            Array::vector(vec![1i32, 3]).unwrap(),
            &length,
            1,
        )
        .unwrap();
        let replicated = ArrayBatch::replicated(Array::vector(vec![1f32; 3]).unwrap());
        let inputs =
            vec![BatchingTracer::new(context.clone(), ragged), BatchingTracer::new(context.clone(), replicated)];
        assert!(context.is_eager());
        assert_eq!(
            program.interpret_in_context(&context, inputs).map(|_| ()),
            Err(ProgramError::custom(BatchingError::UnsupportedOperation {
                message: "`guard` does not support bounded ragged dimension `length` on input 0".to_owned(),
            })),
        );
    }

    #[test]
    fn test_guard_batching_discharged_anchors() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, a, b, c] = ["p", "q", "a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let original = GuardOperation::new(
            "region",
            vec![
                ArrayIrType::from(ReferenceType::new(vector(&p))),
                ArrayIrType::from(ReferenceType::new(vector(&q))),
                p.clone().into(),
                q.clone().into(),
            ],
        )
        .with_retained(vec![a.to_variable(), b.to_variable()]);
        let discharged = GuardOperation {
            declared_types: vec![vector(&p).into(), vector(&q).into(), p.clone().into(), q.clone().into()],
            ..original
        };
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, first, second, dimension)| {
                let batching = BatchingContext::new(extent.domain(), extent);
                let outputs = discharged
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayIrBatch::replicated(first),
                            ArrayIrBatch::replicated(second),
                            ArrayIrBatch::replicated(dimension.clone()),
                            ArrayIrBatch::replicated(dimension),
                        ],
                    )?
                    .into_parts()
                    .0;
                Ok(outputs.into_iter().map(|output| output.value().clone()).collect::<Vec<_>>())
            },
            (
                ArrayIrType::from(DimensionType::from(Dimension::Static(2))),
                ArrayIrType::from(vector(&a)),
                ArrayIrType::from(vector(&b)),
                ArrayIrType::from(c),
            ),
        )
        .unwrap();
        assert_eq!(program.output_types(), vec![vector(&a).into(), vector(&b).into(), a.into(), b.into()]);
        let ArrayIrOperation::Guard(guard) = program.instructions()[0].operation() else {
            panic!("expected a guard");
        };
        assert_eq!(guard.anchors, vec![0, 1]);
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:dimension<2>, %1:f32[a], %2:f32[b], %3:dimension<c ∈ [1, 8)> .
                let %4:f32[a], %5:f32[b], %6:dimension<a ∈ [1, 8)>, %7:dimension<b ∈ [1, 8)> = guard [
                    boundary="region",
                    types=[f32[p], f32[q], dimension<p ∈ [1, 8)>, dimension<q ∈ [1, 8)>],
                    retained=[a, b],
                    anchors=[0, 1],
                ] %1 %2 %3 %3
                in (%4, %5, %6, %7)"#},
        );
    }

    #[test]
    fn test_guard_differentiation() {
        // The guard checks the primals, and the tangent of each narrowed output is its input tangent retyped behind the
        // guard, witnessed by the primal input and checked output. The tangent program contains only casts, whose
        // transposes retain those dependencies and restore the input cotangent representations. The program also
        // returns its symbolic input, so that the pullback accumulates the cotangents of both uses of that input.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(symbolic.clone());
        let second = builder.add_input(exact.clone());
        let mut outputs = builder
            .add_instruction(
                GuardOperation::new("region", vec![symbolic.clone(), symbolic.clone()]),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        outputs.push(first);
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[2], %2:f32[n], %3:f32[2] .
                let %4:f32[2], %5:f32[2] = guard [boundary="region", types=[f32[n], f32[n]]] %0 %1
                    %6:f32[n] = cast [type=f32[n], inputs=[f32[n], f32[2]]] %0 %4
                    %7:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %2 %6
                    %8:f32[2] = cast [type=f32[2], inputs=[f32[2], f32[2]]] %3 %5
                in (%4, %5, %0, %7, %8, %2)"#},
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[2], %2:f32[n] .
                let %3:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %2
                in (%3, %1, %0)"},
        );
        assert_eq!(linearization.residual_count(), 1);
        let pullback = linearization.tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[n], %3:f32[n] .
                let %4:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %3
                    %5:f32[n] = add %2 %4
                in (%5, %1)"},
        );
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                vector(vec![1.0, 2.0]),
                vector(vec![3.0, 4.0]),
                vector(vec![5.0, 6.0]),
                vector(vec![7.0, 8.0]),
            ]),
            Ok(vec![vector(vec![6.0, 8.0]), vector(vec![3.0, 4.0])]),
        );
    }

    #[test]
    fn test_guard_differentiation_with_retained_identities() {
        // A guard that retains `n` gives the exact input the type `f32[n]`, so its tangent is widened behind the guard,
        // witnessed by the primal output that carries `n`, and the pullback narrows the cotangent back to `f32[2]`. The
        // retained symbolic input keeps its type, but its tangent still depends on the checked output.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(symbolic.clone());
        let second = builder.add_input(exact.clone());
        let outputs = builder
            .add_instruction(
                GuardOperation::new("region", vec![symbolic.clone(), symbolic.clone()])
                    .with_retained(vec![n.to_variable()]),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), outputs, None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        assert_eq!(program.output_types(), vec![symbolic.clone()]);
        let jvp = program.jvp().unwrap().to_string();
        assert_eq!(jvp.matches("cast").count(), 2, "{jvp}");
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        let rendering = pullback.to_string();
        assert!(rendering.contains("cast [type=f32[2]"), "{rendering}");
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        let pullback_inputs = pullback
            .input_types()
            .iter()
            .enumerate()
            .map(|(index, _)| vector(vec![1.0 + 2.0 * index as f32, 2.0 + 2.0 * index as f32]))
            .collect::<Vec<_>>();
        let cotangent = pullback_inputs[0].clone();
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![cotangent.clone(), cotangent]));
    }

    #[test]
    fn test_guard_differentiation_with_renamed_identities() {
        // `[f32[a], f32[b]]` entering `[f32[p], f32[p]]` relates `a` and `b` without an exact extent, and a guard that
        // retains `a` gives both outputs the type `f32[a]`. The renamed tangent is converted by a `cast` whose
        // witnesses are the primal input and output, so that its transpose can
        // rename the cotangent back to `f32[b]`.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, a, b] = ["p", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(vector(&a));
        let second = builder.add_input(vector(&b));
        let outputs = builder
            .add_instruction(
                GuardOperation::new("region", vec![vector(&p), vector(&p)]).with_retained(vec![a.to_variable()]),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), outputs, None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {r#"
                lambda %0:f32[a], %1:f32[b], %2:f32[a], %3:f32[b] .
                let %4:f32[a], %5:f32[a] = guard [boundary="region", types=[f32[p], f32[p]], retained=[a]] %0 %1
                    %6:f32[a] = cast [type=f32[a], inputs=[f32[a], f32[a]]] %2 %4
                    %7:f32[a] = cast [type=f32[a], inputs=[f32[b], f32[b], f32[a]]] %3 %1 %5
                    %8:f32[a] = add %4 %5
                    %9:f32[a] = add %6 %7
                in (%8, %9)"#},
        );
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        let rendering = pullback.to_string();
        assert!(rendering.contains("cast [type=f32[b]"), "{rendering}");
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        let pullback_inputs = pullback
            .input_types()
            .iter()
            .enumerate()
            .map(|(index, _)| vector(vec![1.0 + 2.0 * index as f32, 2.0 + 2.0 * index as f32]))
            .collect::<Vec<_>>();
        let cotangent = pullback_inputs[0].clone();
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![cotangent.clone(), cotangent]));
    }

    #[test]
    fn test_guard_operation_provider() {
        let r#type = ArrayType::scalar(DataType::F32);
        let guard = GuardOperation::new("region", vec![r#type.clone()]);
        assert!(matches!(
            ArrayOperation::<Array>::provide(guard.clone(), &[&r#type]),
            Ok(ArrayOperation::Guard(actual)) if actual == guard,
        ));
        assert!(matches!(
            ArrayOperation::<Array>::provide(guard.clone(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        let composite_type = ArrayIrType::from(r#type);
        let composite_guard = GuardOperation::new("region", vec![composite_type.clone()]);
        assert!(matches!(
            ArrayIrOperation::<Array>::provide(composite_guard.clone(), &[&composite_type]),
            Ok(ArrayIrOperation::Guard(actual)) if actual == composite_guard,
        ));
        assert!(matches!(
            ConstantOperation::<Array>::provide(guard, &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the input types of `region` leave identity relationships unproven, but the operation \
                               family cannot represent the guard required to preserve these input checks",
        ));
    }

    #[test]
    fn test_guard_operation_provider_uses_ordinary_conversions() {
        // The derive recognizes the guard payload by its `GuardOperation` path segment and then leaves provision to the
        // blanket provider over the generated conversion. A payload that names the guard through a type alias is not
        // recognized and fails to compile instead (refer to the `ryft-macros-tests` compile-fail snapshots).
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(type(ArrayType), constant(Array))]
        enum Family {
            Constant(ConstantOperation<Array>),
            Checked(GuardOperation<ArrayType>),
        }

        let r#type = ArrayType::scalar(DataType::F32);
        let guard = GuardOperation::new("region", vec![r#type.clone()]);
        assert!(matches!(Family::from(guard.clone()), Family::Checked(actual) if actual == guard));
        assert!(matches!(
            Family::provide(guard.clone(), &[&r#type]),
            Ok(Family::Checked(actual)) if actual == guard,
        ));
    }
}
