//! Array-owned reference views and eager/staging implementations of generic reference capabilities.
//!
//! Indexing and slicing remain here because they require [`ArrayReferenceViewTransform`] geometry. The generic
//! allocation, read, replacement, additive-update, and freeze payloads live in [`crate::programs::references`]; this
//! module specializes them to [`ArrayIrType`] and implements their capabilities for array values and tracers.
//!
//! Staged and composite value calls validate the complete operand-type relationship before dispatching the operation.
//! Their type diagnostics can therefore precede any eager reference-state error. Once an eager reference or derived
//! view is reached, reference-state errors take precedence over replacement-type validation, as documented on that
//! runtime API.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::reference_views::{ArrayReference, ArrayReferenceView, ArrayReferenceViewTransform, ViewIndex};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::ir::ArrayIrType;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::{Add, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameter;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::references::forwarded_tangent;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, ProjectedValue, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceAliasKind, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceFreeze,
    ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceOperationSemantics, ReferenceOutput,
    ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, ReferenceType, ReferenceView,
    ReferenceViewOperation, ReferenceViewValidationError, ReferenceWrite, ReferenceWriteOperation, RegionInterface,
    TypeError, Typed, Value, ValueProjection, ViewSymbol, batch_reference_view_operation,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReferenceIndexOperation`].
pub const REFERENCE_INDEX_OPERATION_NAME: &str = "reference_index";

/// Canonical operation name for [`ReferenceSliceOperation`].
pub const REFERENCE_SLICE_OPERATION_NAME: &str = "reference_slice";

/// Derives an axis-removing indexed view of a reference without accessing its state.
pub trait ReferenceIndex<Output = Self>: Sized {
    /// Returns a reference view selecting `index` on `axis`.
    fn reference_index(&self, axis: usize, index: usize) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceIndex<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceIndexOperation>,
{
    fn reference_index(&self, axis: usize, index: usize) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceIndexOperation::new(axis, index), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V> ReferenceIndex<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ReferenceType<ArrayType>>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceIndexOperation>,
{
    fn reference_index(
        &self,
        axis: usize,
        index: usize,
    ) -> Result<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected, ProgramError> {
        self.value().reference_index(axis, index)?.into_projected().map_err(Into::into)
    }
}

/// Derives a rank-preserving static slice view of a reference without accessing its state.
pub trait ReferenceSlice<Output = Self>: Sized {
    /// Returns a reference view selecting `axes`, one static selection per input axis.
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceSlice<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSliceOperation>,
{
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceSliceOperation::new(axes.to_vec()), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V> ReferenceSlice<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ReferenceType<ArrayType>>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSliceOperation>,
{
    fn reference_slice(
        &self,
        axes: &[ArraySliceAxis],
    ) -> Result<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected, ProgramError> {
        self.value().reference_slice(axes)?.into_projected().map_err(Into::into)
    }
}

// Both view operations preserve the canonical allocation without accessing its state.
static REFERENCE_VIEW_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(
        Vec::new(),
        vec![ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
    )
});

/// Infers the derived reference type produced by one allocation-preserving view transform.
fn infer_view_output_types(
    transform: ArrayReferenceViewTransform,
    input_types: &[ArrayIrType],
    region_interfaces: &[RegionInterface<ArrayIrType>],
) -> Result<Vec<ArrayIrType>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    check_count!("region", region_interfaces, 0, TypeError);
    let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
    Ok(vec![ReferenceType::new(transform.output_type(reference.referent())?).into()])
}

/// Pure reference-to-reference operation selecting one coordinate and removing its axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceIndexOperation {
    /// Axis selected in the input reference view.
    axis: usize,

    /// Coordinate selected on `axis`.
    index: usize,
}

impl ReferenceIndexOperation {
    /// Creates a new reference index operation.
    pub const fn new(axis: usize, index: usize) -> Self {
        Self { axis, index }
    }

    /// Returns this operation's allocation-preserving view transform.
    pub const fn transform(&self) -> ArrayReferenceViewTransform {
        ArrayReferenceViewTransform::Index { axis: self.axis, index: ViewIndex::Static(self.index) }
    }
}

impl Display for ReferenceIndexOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceIndexOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_INDEX_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        infer_view_output_types(self.transform(), input_types, region_interfaces)
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_VIEW_OPERATION_SEMANTICS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis", self.axis)?;
            operation.field("index", self.index)
        })
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceIndex<C::Value>>> InterpretableOperation<C>
    for ReferenceIndexOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reference_index(self.axis, self.index)?])
    }
}

/// Pure reference-to-reference operation selecting one static range on every axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceSliceOperation {
    /// Per-axis selections in the input reference view.
    axes: Vec<ArraySliceAxis>,
}

impl ReferenceSliceOperation {
    /// Creates a new reference slice operation.
    #[inline]
    pub fn new(axes: Vec<ArraySliceAxis>) -> Self {
        Self { axes }
    }

    /// Returns this operation's allocation-preserving view transform.
    #[inline]
    pub fn transform(&self) -> ArrayReferenceViewTransform {
        ArrayReferenceViewTransform::Slice { axes: self.axes.clone() }
    }
}

impl Display for ReferenceSliceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceSliceOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_SLICE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        infer_view_output_types(self.transform(), input_types, region_interfaces)
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_VIEW_OPERATION_SEMANTICS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axes", format_args!("{:?}", self.axes)))
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceSlice<C::Value>>> InterpretableOperation<C>
    for ReferenceSliceOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reference_slice(self.axes.as_slice())?])
    }
}

/// Discharges one allocation-preserving array reference view by composing `transform` onto the incoming handle's alias.
///
/// A view creates a narrower alias of the same allocation, so on a discharged allocation the rewrite is metadata only:
/// it validates the composed referent type with exactly the eager handle's arithmetic, rejecting an invalid composition
/// before any handle exists, and then records the composed chain as the new handle's authoritative alias. Nothing is
/// bound into the destination, because the portion this handle selects is materialized at each access rather than at
/// the view. The step is closed over destination values: each [`ViewSymbol::Operand`] the transform reports binds the
/// destination value of that operand, so the operands of a view operation are its reference followed by one value per
/// symbol, and a static transform binds nothing.
///
/// On an allocation that partial discharge *preserved*, the view is additionally replayed into the destination, and the
/// reference it produces becomes the alias handle's own destination value, so that later accesses consume that
/// exact value instead of replaying the chain and duplicating the view operations. The composed alias is recorded
/// either way, which is what keeps one handle's view chain single-sourced whichever state its allocation is in.
///
/// # Parameters
///
///   - `operation`: View operation being discharged, replayed verbatim on a preserved allocation.
///   - `transform`: Coordinate transform this view operation applies to its operand's view.
///   - `context`: Active discharge context owning the allocation environment.
///   - `inputs`: Carriers supplied as the view operation's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::InvalidInputCount`] for an application that does not supply exactly one operand per
/// symbol beyond the reference, [`ProgramError::MalformedProgram`] when the first operand is a value rather than a
/// reference handle, when a symbol operand is a reference rather than a value, or when a symbol names the reference
/// operand or an operand outside the application, [`ProgramError::UnsupportedOperation`] for a
/// [`ViewSymbol::Iteration`] coordinate, and [`ProgramError::InvalidOutputCount`] when replaying the view on a
/// preserved allocation does not produce exactly one value. Propagates the view algebra's own [`TypeError`] when
/// `transform` does not compose onto the incoming handle's referent, and the discharge context's own
/// [`ProgramError::MalformedProgram`] when the replayed reference does not carry the composed type.
fn discharge_reference_view<C, P, O>(
    operation: &O,
    transform: ArrayReferenceViewTransform,
    context: &ReferenceDischargeContext<C, P>,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Type = ArrayIrType, Operation: From<O>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView<C::Value>>,
    O: Clone + Operation<Type = ArrayIrType>,
{
    let symbols = transform.symbols();
    check_count!("input", inputs, 1 + symbols.len(), ProgramError);
    let reference = inputs[0].try_as_reference("a reference to view")?;
    let referent = transform.output_type(reference.r#type().referent())?;
    let bindings = symbols
        .iter()
        .map(|symbol| match symbol {
            ViewSymbol::Operand(index) => match inputs.get(*index) {
                Some(input) if *index > 0 => input.try_as_value("a view coordinate").cloned(),
                _ => Err(ProgramError::MalformedProgram(format!(
                    "reference view symbol names operand {index} but the coordinate operands of a view with {} \
                     operands are 1..{}",
                    inputs.len(),
                    inputs.len(),
                ))),
            },
            ViewSymbol::Iteration => Err(ProgramError::UnsupportedOperation {
                message: "an iteration view is created by its region-carrying operation and never discharged as an \
                          instruction"
                    .to_string(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let alias = reference.alias().with_step(transform, bindings);
    Ok(vec![
        context
            .alias_reference(reference, alias, ReferenceType::new(referent), |value| {
                let mut outputs = context.parent().bind(operation.clone(), Vec::new(), std::slice::from_ref(value))?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            })?
            .into(),
    ])
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceIndexOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceIndexOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView<C::Value>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_reference_view(self, self.transform(), context, inputs)
    }
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceSliceOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceSliceOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView<C::Value>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_reference_view(self, self.transform(), context, inputs)
    }
}

// The default partial-evaluation behavior applies to both views: a view carries no effect of its own and is placed
// wherever its reference operand is.
impl<C: Context<Type = ArrayIrType, Operation: From<ReferenceIndexOperation>>> PartiallyEvaluatableOperation<C>
    for ReferenceIndexOperation
{
}

impl<C: Context<Type = ArrayIrType, Operation: From<ReferenceSliceOperation>>> PartiallyEvaluatableOperation<C>
    for ReferenceSliceOperation
{
}

macro_rules! impl_default_reference_view_transposition {
    // Installs the transposition rule for one array reference-view operation.
    ($operation:ty) => {
        // A view is aliasing metadata rather than a linear map of its own: the cotangent of a view operand is reached
        // by reapplying the view path to its root's cotangent reference inside the transposition context, so the
        // reverse sweep never needs this rule to run and every operand receives a structural zero.
        impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> TransposableOperation<V, O>
            for $operation
        {
            fn transpose<D: TranspositionDriver<V, O>>(
                &self,
                _context: &mut TranspositionContext<'_, V, O>,
                _driver: &D,
                inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
                _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
            ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
                inputs.iter().map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?))).collect()
            }
        }
    };
}

impl_default_reference_view_transposition!(ReferenceIndexOperation);
impl_default_reference_view_transposition!(ReferenceSliceOperation);

impl<C: Context<Type = ArrayIrType, Value: ReferenceIndex<C::Value>>> DifferentiableOperation<C>
    for ReferenceIndexOperation
{
    // A view is pure aliasing metadata, so the tangent reference receives the same view as the primal reference. A
    // plumbing reference (i.e., a reference dual whose tangent is a symbolic zero) carries no tangent reference, so its
    // view stays plumbing.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reference_index(self.axis, self.index)?;
        Ok(vec![forwarded_tangent(&inputs[0], primal, |tangent_reference| {
            tangent_reference.reference_index(self.axis, self.index)
        })?])
    }
}

impl<C: Context<Type = ArrayIrType, Value: ReferenceSlice<C::Value>>> DifferentiableOperation<C>
    for ReferenceSliceOperation
{
    // As in the `ReferenceIndexOperation` rule, the tangent reference receives the same view as the primal reference.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reference_slice(self.axes.as_slice())?;
        Ok(vec![forwarded_tangent(&inputs[0], primal, |tangent_reference| {
            tangent_reference.reference_slice(self.axes.as_slice())
        })?])
    }
}

impl<
    C: Context<Type = ArrayIrType, Operation: ReferenceViewOperation + From<ReferenceIndexOperation>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceIndexOperation
{
    // The axis arithmetic lives on the view description (`ArrayReferenceViewTransform::batch`); the shared rule moves
    // the source's batch axis through it and binds the batched view on the parent context.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        batch_reference_view_operation(self, context, inputs)
    }
}

impl<
    C: Context<Type = ArrayIrType, Operation: ReferenceViewOperation + From<ReferenceSliceOperation>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceSliceOperation
{
    // As for `ReferenceIndexOperation`, the shared rule batches the slice through its view description.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        batch_reference_view_operation(self, context, inputs)
    }
}

/// Validates one [`ArrayReferenceViewTransform`] as a step from the reference type `source` to the reference type
/// `output`. This is the array family's
/// [`ReferenceViewOperation::validate_view`](crate::programs::ReferenceViewOperation::validate_view) rule, shared by
/// every operation family that embeds the array view operations: the transform's
/// [`output_type`](ArrayReferenceViewTransform::output_type) must be defined for the source referent and must equal
/// the output referent exactly.
///
/// # Errors
///
/// Returns [`ReferenceViewValidationError::InvalidComposition`] when either type is not a reference type or the
/// transform cannot be applied to the source referent, and [`ReferenceViewValidationError::TypeMismatch`] when the
/// derived referent differs from the declared one.
pub fn validate_array_reference_view(
    view: &ArrayReferenceViewTransform,
    source: &ArrayIrType,
    output: &ArrayIrType,
) -> Result<(), ReferenceViewValidationError> {
    let invalid = |error: TypeError| ReferenceViewValidationError::InvalidComposition { message: error.to_string() };
    let source = <&ReferenceType<ArrayType>>::try_from(source).map_err(invalid)?;
    let output = <&ReferenceType<ArrayType>>::try_from(output).map_err(invalid)?;
    let expected = view.output_type(source.referent()).map_err(invalid)?;
    if expected != *output.referent() {
        return Err(ReferenceViewValidationError::TypeMismatch {
            expected: expected.to_string(),
            actual: output.referent().to_string(),
        });
    }
    Ok(())
}

/// Stages one [`ArrayReferenceViewTransform`] over the reference `source` through `context` and returns the derived
/// reference. This is the array family's
/// [`ReferenceViewOperation::reapply_view`](crate::programs::ReferenceViewOperation::reapply_view) rule, shared by
/// every operation family that embeds the array view operations: a static [`Index`](ArrayReferenceViewTransform::Index)
/// transform stages a [`ReferenceIndexOperation`] and a [`Slice`](ArrayReferenceViewTransform::Slice) transform
/// stages a [`ReferenceSliceOperation`], through the same conversions the eager array reference views use. `symbols`
/// supplies one value per symbol of `view`, which is none for both static transforms.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when `symbols` does not supply exactly one value per symbol of `view`,
/// [`ProgramError::UnsupportedOperation`] for a symbolic index (no array operation reapplies one until dynamic
/// indexing is supported), and [`ProgramError::InvalidOutputCount`] when the staged view does not produce exactly one
/// value. Propagates the staging error of `context`.
pub fn reapply_array_reference_view<C>(
    context: &C,
    view: &ArrayReferenceViewTransform,
    source: C::Value,
    symbols: &[C::Value],
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<ReferenceIndexOperation> + From<ReferenceSliceOperation>,
{
    let expected = view.symbols().len();
    if symbols.len() != expected {
        return Err(ProgramError::MalformedProgram(format!(
            "reapplying reference view `{view:?}` requires {expected} symbol values but received {}",
            symbols.len(),
        )));
    }
    let mut outputs = match view {
        ArrayReferenceViewTransform::Index { axis, index: ViewIndex::Static(index) } => {
            context.bind(ReferenceIndexOperation::new(*axis, *index), Vec::new(), std::slice::from_ref(&source))?
        }
        ArrayReferenceViewTransform::Index { index: ViewIndex::Symbolic(_), .. } => {
            return Err(ProgramError::UnsupportedOperation {
                message: "no array operation reapplies a symbolic reference index until dynamic indexing is supported"
                    .to_string(),
            });
        }
        ArrayReferenceViewTransform::Slice { axes } => {
            context.bind(ReferenceSliceOperation::new(axes.clone()), Vec::new(), std::slice::from_ref(&source))?
        }
    };
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

impl<V: Value<Type = ArrayIrType>> ReferenceNew<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceNewOperation<ArrayType, ArrayIrType>>,
{
    fn reference_new(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceNewOperation::new(), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V> ReferenceNew<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected> for ProjectedValue<ArrayType, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ReferenceType<ArrayType>>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceNewOperation<ArrayType, ArrayIrType>>,
{
    fn reference_new(&self) -> Result<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected, ProgramError> {
        self.value()
            .dispatch_domain()
            .bind(ReferenceNewOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceRead<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation<ArrayType, ArrayIrType>>,
{
    fn read(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V> ReferenceRead<<V as ValueProjection<ArrayType>>::Projected> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation<ArrayType, ArrayIrType>>,
{
    fn read(&self) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        self.value()
            .dispatch_domain()
            .bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceWrite<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType>>,
{
    fn write(&self, replacement: &V) -> Result<(), ProgramError> {
        self.dispatch_domain().bind(
            ReferenceWriteOperation::new(),
            Vec::new(),
            &[self.clone(), replacement.clone()],
        )?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceWrite<ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType>>,
{
    fn write(&self, replacement: &ProjectedValue<ArrayType, V>) -> Result<(), ProgramError> {
        self.value().dispatch_domain().bind(
            ReferenceWriteOperation::new(),
            Vec::new(),
            &[self.value().clone(), replacement.value().clone()],
        )?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceSwap<V, V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType>>,
{
    fn swap(&self, replacement: &V) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceSwapOperation::new(), Vec::new(), &[self.clone(), replacement.clone()])?
            .remove(0))
    }
}

impl<V> ReferenceSwap<ProjectedValue<ArrayType, V>, <V as ValueProjection<ArrayType>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType>>,
{
    fn swap(
        &self,
        replacement: &ProjectedValue<ArrayType, V>,
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        self.value()
            .dispatch_domain()
            .bind(ReferenceSwapOperation::new(), Vec::new(), &[self.value().clone(), replacement.value().clone()])?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceAddUpdate<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType>>,
{
    fn add_update(&self, update: &V) -> Result<(), ProgramError> {
        self.dispatch_domain()
            .bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[self.clone(), update.clone()])?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceAddUpdate<ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType>>,
{
    fn add_update(&self, update: &ProjectedValue<ArrayType, V>) -> Result<(), ProgramError> {
        self.value().dispatch_domain().bind(
            ReferenceAddUpdateOperation::new(),
            Vec::new(),
            &[self.value().clone(), update.value().clone()],
        )?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceFreeze<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(self) -> Result<V, ProgramError> {
        let domain = self.dispatch_domain();
        Ok(domain.bind(ReferenceFreezeOperation::new(), Vec::new(), std::slice::from_ref(&self))?.remove(0))
    }
}

impl<V> ReferenceFreeze<<V as ValueProjection<ArrayType>>::Projected> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(self) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let domain = self.value().dispatch_domain();
        domain
            .bind(ReferenceFreezeOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<A: Value<Type = ArrayType>> ReferenceNew for ArrayIrValue<A> {
    fn reference_new(&self) -> Result<Self, ProgramError> {
        ReferenceNewOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let value = <Self as ValueProjection<ArrayType>>::projected(self)?.clone();
        Ok(Self::Reference(ArrayReference::new(value)))
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice> ReferenceRead for ArrayIrValue<A> {
    fn read(&self) -> Result<Self, ProgramError> {
        ReferenceReadOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.read()?))
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ReferenceWrite for ArrayIrValue<A> {
    fn write(&self, replacement: &Self) -> Result<(), ProgramError> {
        ReferenceWriteOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        reference.write(replacement)
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ReferenceSwap for ArrayIrValue<A> {
    fn swap(&self, replacement: &Self) -> Result<Self, ProgramError> {
        ReferenceSwapOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        Ok(Self::Array(reference.swap(replacement)?))
    }
}

impl<A: Value<Type = ArrayType> + Add + Reshape + Slice + UpdateSlice> ReferenceAddUpdate for ArrayIrValue<A> {
    fn add_update(&self, update: &Self) -> Result<(), ProgramError> {
        ReferenceAddUpdateOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), update.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        reference.add_update(update)
    }
}

impl<A: Value<Type = ArrayType>> ReferenceFreeze for ArrayIrValue<A> {
    fn freeze(self) -> Result<Self, ProgramError> {
        ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(&self)?;
        Ok(Self::Array(reference.freeze()?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceIndex for ArrayIrValue<A> {
    fn reference_index(&self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        // Projection rejects value operands and `with_transform` validates the transform against the handle's
        // cached referent type, so a separate operation-level inference pass would only repeat both checks.
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let transform = ArrayReferenceViewTransform::Index { axis, index: ViewIndex::Static(index) };
        Ok(Self::Reference(reference.with_transform(transform)?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceSlice for ArrayIrValue<A> {
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        // Projection rejects value operands and `with_transform` validates the transform against the handle's
        // cached referent type, so a separate operation-level inference pass would only repeat both checks.
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceViewTransform::Slice { axes: axes.to_vec() })?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::batching::{ArrayIrBatch, ArrayIrBatching};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::reference_discharge::ArrayReferenceDischarge;
    use crate::arrays::reference_views::ArrayReferenceViewError;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::axes::Axis;
    use crate::batching::{BatchAxis, BatchingTracer};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDual, DifferentiationError, DifferentiationTracer,
        TransposableOperation, differentiate_at,
    };
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{
        Effect, Effects, EmptyRegionDriver, ProgramBuilder, ProgramError, REFERENCE_NEW_OPERATION_NAME,
        REFERENCE_READ_OPERATION_NAME, ReferenceError, TypeError, ViewSymbol,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestDestination = EagerContext<TestValue, TestOperation>;
    type TestNew = ReferenceNewOperation<ArrayType, ArrayIrType>;
    type TestRead = ReferenceReadOperation<ArrayType, ArrayIrType>;
    type TestWrite = ReferenceWriteOperation<ArrayType, ArrayIrType>;
    type TestSwap = ReferenceSwapOperation<ArrayType, ArrayIrType>;
    type TestAddUpdate = ReferenceAddUpdateOperation<ArrayType, ArrayIrType>;
    type TestFreeze = ReferenceFreezeOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_array_reference_view_operations() {
        let allocation_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let index = ReferenceIndexOperation::new(0, 1);
        assert_eq!(
            index.infer_output_types(std::slice::from_ref(&ReferenceType::new(allocation_type.clone()).into()), &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [4])).into()]),
        );
        assert_eq!(
            index.infer_output_types(std::slice::from_ref(&allocation_type.clone().into()), &[]),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
        assert!(index.effects().is_pure());
        assert_eq!(
            index.reference_semantics().outputs(),
            &[ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
        );
        assert!(index.reference_semantics().inputs().is_empty());
        assert_eq!(index.transform(), ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) });
        assert_eq!(index.to_string(), "reference_index [axis=0, index=1]");

        let slice = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)]);
        assert_eq!(
            slice.infer_output_types(std::slice::from_ref(&ReferenceType::new(allocation_type).into()), &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])).into()]),
        );
        assert_eq!(
            slice.transform(),
            ArrayReferenceViewTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
            },
        );
        assert_eq!(
            slice.to_string(),
            concat!(
                "reference_slice [\n",
                "    axes=[ArraySliceAxis { start: 1, size: 2, stride: 1 }, ",
                "ArraySliceAxis { start: 0, size: 3, stride: 1 }],\n",
                "]",
            ),
        );
    }

    #[test]
    fn test_array_reference_view_operation_reference_discharge() {
        // Both view rules create a narrower reference to the same allocation by composing their transform onto the
        // incoming alias, and bind nothing: a view's coordinates are materialized at each access instead.
        let context = ReferenceDischargeContext::<TestDestination, ArrayReferenceDischarge>::new(EagerContext::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let allocated = ReferenceDischargeValue::from(
            context
                .bind_discharged(
                    ReferenceType::new(allocation_type.clone()),
                    TestValue::Array(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect())),
                )
                .unwrap(),
        );
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().clone();
        let sliced = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)])
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&allocated))
            .unwrap();
        assert_eq!(sliced.len(), 1);
        let sliced = sliced[0].try_as_reference("the slice view").unwrap().clone();
        assert_eq!(sliced.allocation_id(), allocation.allocation_id());
        assert_eq!(sliced.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 2])));
        assert_eq!(sliced.preserved(), None);

        // Composition is onto the *incoming* handle's chain, so indexing the slice selects a row of the slice rather
        // than a row of the allocation, and the composed alias is what every later access applies.
        let indexed = ReferenceIndexOperation::new(0, 1)
            .discharge_references(&context, &EmptyRegionDriver, &[ReferenceDischargeValue::Reference(sliced.clone())])
            .unwrap();
        let indexed = indexed[0].try_as_reference("the index view").unwrap().clone();
        assert_eq!(indexed.allocation_id(), allocation.allocation_id());
        assert_eq!(indexed.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            indexed.alias(),
            &ArrayReferenceView::root()
                .with_transform_unchecked(ArrayReferenceViewTransform::Slice {
                    axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
                })
                .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) }),
        );
        assert_eq!(context.read(&indexed), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 8.0]))));

        // A composition that does not fit the incoming view is rejected before any handle exists, with the view
        // algebra's own diagnostic rather than a discharge-specific one.
        assert_eq!(
            ReferenceIndexOperation::new(0, 2).discharge_references(
                &context,
                &EmptyRegionDriver,
                &[ReferenceDischargeValue::Reference(sliced)],
            ),
            Err(TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into()),
        );

        // The operand must be a reference handle, and there must be exactly one of them.
        let pure = ReferenceDischargeValue::Value(TestValue::Array(Array::scalar(1.0_f32)));
        assert_eq!(
            ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)]).discharge_references(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&pure),
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to view but received a value".to_string(),
            )),
        );
        assert_eq!(
            ReferenceIndexOperation::new(0, 0).discharge_references(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // An allocation that partial discharge preserved survives in the destination, so the view is additionally replayed
        // there and the reference that replay produced becomes the view's own destination value. The
        // composed alias is recorded exactly as it is for a discharged allocation, which is what keeps one handle's view
        // chain single-sourced whichever state its allocation is in.
        let preserved = ReferenceDischargeValue::from(
            context
                .bind_preserved(
                    ReferenceType::new(allocation_type),
                    TestValue::Reference(ArrayReference::new(Array::matrix(
                        3,
                        3,
                        (1..=9).map(|value| value as f32).collect(),
                    ))),
                )
                .unwrap(),
        );
        let preserved_allocation = preserved.try_as_reference("the preserved allocation").unwrap().allocation_id();
        let view = ReferenceIndexOperation::new(0, 0)
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&preserved))
            .unwrap();
        assert_eq!(view.len(), 1);
        let view = view[0].try_as_reference("the preserved view").unwrap().clone();
        assert_eq!(view.allocation_id(), preserved_allocation);
        assert_eq!(view.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(
            view.alias(),
            &ArrayReferenceView::root()
                .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) }),
        );
        assert_eq!(
            view.preserved().map(|value| value.r#type().into_owned()),
            Some(ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])))),
        );

        // The replayed view denotes the coordinates the source named, which the eager destination proves by reading
        // through the view: the first row of the preserved allocation rather than the allocation itself.
        assert_eq!(
            view.preserved().map(ReferenceRead::read),
            Some(Ok(TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])))),
        );
    }

    #[test]
    fn test_array_reference_view_operations_jvp() {
        let context = DifferentiationContext::new(TestDestination::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(Array::from_f64s(allocation_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .reference_new()
            .unwrap();
        let tangent_reference =
            TestValue::Array(Array::from_f64s(allocation_type, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
                .reference_new()
                .unwrap();

        // An active reference's view is applied to its tangent reference with the same alias, so both views select the
        // same coordinates of their respective allocations.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );
        let indexed = context.bind(ReferenceIndexOperation::new(0, 1), Vec::new(), &[active.clone()]).unwrap();
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))));
        assert_eq!(
            indexed[0].tangent().as_value().unwrap().read(),
            Ok(TestValue::Array(Array::vector(vec![10.0_f32, 11.0, 12.0]))),
        );
        let axes = vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)];
        let sliced = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[active]).unwrap();
        let sliced_type = ArrayType::new_static(DataType::F32, [2, 2]);
        assert_eq!(
            sliced[0].primal().read(),
            Ok(TestValue::Array(Array::from_f64s(sliced_type.clone(), vec![2.0, 3.0, 5.0, 6.0]))),
        );
        assert_eq!(
            sliced[0].tangent().as_value().unwrap().read(),
            Ok(TestValue::Array(Array::from_f64s(sliced_type.clone(), vec![8.0, 9.0, 11.0, 12.0]))),
        );

        // A store through the tangent view lands in the tangent allocation and leaves the primal allocation untouched.
        let zeros = TestValue::Array(Array::from_f64s(sliced_type, vec![0.0; 4]));
        sliced[0].tangent().as_value().unwrap().write(&zeros).unwrap();
        assert_eq!(
            tangent_reference.read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [2, 3]),
                vec![7.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            ))),
        );
        assert_eq!(
            reference.read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [2, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))),
        );

        // The views of a plumbing reference stay plumbing, typed with the view's own reference type.
        let plumbing =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let indexed = context.bind(ReferenceIndexOperation::new(1, 2), Vec::new(), &[plumbing.clone()]).unwrap();
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![3.0_f32, 6.0]))));
        assert!(matches!(
            indexed[0].tangent(),
            MaybeZero::Zero(r#type)
                if *r#type == ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        ));
        let axes = vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let sliced = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[plumbing]).unwrap();
        assert!(matches!(
            sliced[0].tangent(),
            MaybeZero::Zero(r#type)
                if *r#type == ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [1, 3]))),
        ));
    }

    #[test]
    fn test_array_reference_view_operations_batching() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(TestDestination::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3, 4]);
        let reference = TestValue::Array(Array::from_f64s(packed_type, (0..24).map(f64::from).collect()))
            .reference_new()
            .unwrap();

        // A batch axis before the indexed axis shifts the packed indexed axis one position later and keeps the output
        // batch axis.
        let leading =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());
        let outputs = context.bind(ReferenceIndexOperation::new(1, 2), Vec::new(), &[leading]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [2, 3]),
                vec![2.0, 6.0, 10.0, 14.0, 18.0, 22.0],
            ))),
        );

        // A batch axis after the indexed axis leaves the packed indexed axis alone and moves the output batch axis one
        // position earlier.
        let inner =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceIndexOperation::new(0, 1), Vec::new(), &[inner.clone()]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [3, 4]),
                (12..24).map(f64::from).collect(),
            ))),
        );

        // A batch axis at the indexed axis position precedes the indexed per-item axis in the packed referent.
        let outputs = context.bind(ReferenceIndexOperation::new(1, 3), Vec::new(), &[inner.clone()]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [2, 3]),
                vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0],
            ))),
        );

        // Slicing inserts an identity selection at the batch axis position and keeps the output batch axis.
        let axes = vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)];
        let outputs = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[inner]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [1, 2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [1, 3, 2]),
                vec![13.0, 14.0, 17.0, 18.0, 21.0, 22.0],
            ))),
        );

        // Replicated references are viewed unchanged and stay replicated.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
        let outputs = context.bind(ReferenceIndexOperation::new(0, 1), Vec::new(), &[replicated.clone()]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 4]))),
        );
        let axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1), ArraySliceAxis::new(0, 4, 1)];
        let outputs = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[replicated]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [1, 3, 4]))),
        );

        // A static identity slice cannot be formed for a dynamically sized batch axis.
        let trace = TracingContext::<TestValue, TestOperation>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let extent = trace.input(DimensionType::new(batch.clone()).into());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        let reference = trace.input(ReferenceType::new(dynamic_type.clone()).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, extent);
        let batched = BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference, BatchAxis::new(0)).unwrap());
        let error = context
            .bind(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 3, 1)]), Vec::new(), &[batched])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic_type), axis: Axis::from(0) }),
        );
    }

    #[test]
    fn test_array_ir_reference_operation_conversions() {
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceNew(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceReadOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceRead(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceWriteOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceWrite(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceSwapOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceSwap(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceAddUpdateOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceAddUpdate(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceFreeze(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceIndexOperation::new(0, 0)),
            ArrayIrOperation::ReferenceIndex(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)])),
            ArrayIrOperation::ReferenceSlice(_),
        ));
    }

    #[test]
    fn test_eager_reference_allocation_and_read_roundtrip() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.reference_new().unwrap();
        assert!(matches!(reference, ArrayIrValue::Reference(_)));
        assert_eq!(reference.read().unwrap(), initial);
    }

    #[test]
    fn test_eager_reference_index_slice_and_composition() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let allocation = initial.reference_new().unwrap();
        let row = allocation.reference_index(0, 1).unwrap();
        assert_eq!(row.read(), Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))));

        let slice = allocation.reference_slice(&[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)]).unwrap();
        assert_eq!(
            slice.read(),
            Ok(ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),),
                vec![2.0, 3.0, 5.0, 6.0],
            ))),
        );
        let composed = slice.reference_index(0, 1).unwrap();
        assert_eq!(composed.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
    }

    #[test]
    fn test_eager_reference_indexed_mutation_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let allocation = initial.reference_new().unwrap();
        let row = allocation.reference_index(0, 1).unwrap();

        assert_eq!(
            row.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))),
        );
        row.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(
            allocation.read(),
            Ok(ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0],))),
        );
    }

    #[test]
    fn test_eager_reference_views_share_overlapping_allocation_state() {
        let allocation = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])).reference_new().unwrap();
        let left = allocation.reference_slice(&[ArraySliceAxis::new(0, 3, 1)]).unwrap();
        let right = allocation.reference_slice(&[ArraySliceAxis::new(1, 3, 1)]).unwrap();

        assert_eq!(
            left.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
        );
        assert_eq!(right.read(), Ok(ArrayIrValue::Array(Array::vector(vec![20.0_f32, 30.0, 4.0]))));
        right.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(allocation.read(), Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f32, 21.0, 32.0, 7.0]))));
    }

    #[test]
    fn test_eager_reference_view_validation_and_freeze_invalidation() {
        let allocation = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])).reference_new().unwrap();
        assert_eq!(
            allocation.reference_index(1, 0),
            Err(TypeError::invalid("reference index axis 1 is out of bounds for rank 1").into()),
        );
        assert_eq!(
            allocation.reference_index(0, 3),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3").into()),
        );
        assert_eq!(
            allocation.reference_slice(&[ArraySliceAxis::new(2, 2, 1)]),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 2 exceeds input size 3").into()),
        );
        assert_eq!(
            allocation.reference_slice(&[ArraySliceAxis::new(0, 2, 2)]),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )
            .into()),
        );

        let view = allocation.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        let same_view = allocation.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        assert_eq!(view, same_view);
        assert_ne!(view, allocation);
        let different_view = allocation.reference_slice(&[ArraySliceAxis::new(1, 2, 1)]).unwrap();
        let view_handle = <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&view)
            .unwrap()
            .clone();
        let same_view_handle =
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&same_view)
                .unwrap()
                .clone();
        let different_view_handle =
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&different_view)
                .unwrap()
                .clone();
        let allocation_handle =
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&allocation)
                .unwrap()
                .clone();
        assert!(allocation_handle.is_runtime_root_handle());
        assert!(!view_handle.is_runtime_root_handle());
        let Err(error) = view_handle.lock_root() else {
            panic!("reference view must not expose a complete-value transaction guard")
        };
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::InvalidRuntimeRoot),
        );
        let mut views = HashMap::new();
        views.insert(view_handle, 7);
        assert_eq!(views.get(&same_view_handle), Some(&7));
        assert_eq!(views.get(&different_view_handle), None);
        assert_eq!(views.get(&allocation_handle), None);
        // `freeze` consumes the handle it is given, so an alias that must outlive the consumption is cloned first;
        // the clone shares the holder and is therefore invalidated with the rest of the family.
        let error = view.clone().freeze().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotFreezeView)
        );
        assert_eq!(allocation.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));

        assert_eq!(allocation.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));
        let error = view.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_reject_mismatched_member_kinds() {
        let array = ArrayIrValue::<Array>::Array(Array::scalar(1.0_f32));
        assert_eq!(array.read(), Err(TypeError::invalid("expected reference type but got array type").into()));
        let reference = array.reference_new().unwrap();
        assert_eq!(
            reference.reference_new(),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(array.swap(&array), Err(TypeError::invalid("expected reference type but got array type").into()));
        assert_eq!(
            reference.swap(&reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(
            array.add_update(&array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            reference.add_update(&reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(array.freeze(), Err(TypeError::invalid("expected reference type but got array type").into()));
    }

    #[test]
    fn test_eager_reference_updates_enforce_exact_storage_and_preserve_rejected_state() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.reference_new().unwrap();

        let error = reference.swap(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.write(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_write` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.add_update(&ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_add_update` addition result type `f64[2]` must exactly match reference referent \
                 type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(reference.read(), Ok(initial));

        let replacement = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0]));
        assert_eq!(reference.swap(&replacement), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))),);
        assert_eq!(reference.read(), Ok(replacement));

        // Broadcasting is valid only because the computed result preserves the exact stored type.
        assert_eq!(reference.add_update(&ArrayIrValue::Array(Array::scalar(1.0_f32))), Ok(()));
        assert_eq!(reference.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
    }

    #[test]
    fn test_eager_reference_freeze_invalidates_composite_aliases() {
        let reference = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))));

        let error = alias.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.swap(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.add_update(&ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.freeze().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_preserve_dynamic_referents() {
        // Equality over a dynamically typed `Array` cannot address its elements, so every referent this test observes
        // is unwrapped here and then compared by its declared type and its exact physical storage.
        fn referent(value: ArrayIrValue<Array>) -> Array {
            <ArrayIrValue<Array> as ValueProjection<ArrayType>>::into_projected(value).unwrap()
        }

        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let replacement_bytes = 2.0_f32.to_le_bytes().to_vec();

        // `Array`'s checked constructors reject dynamically shaped types, so both referents come from the test-only
        // unchecked hatch. They declare exactly the same dynamic type, which is what makes each holder transition
        // observable only through the payload it returns.
        let initial = Array::with_unchecked_type(dynamic_type.clone(), initial_bytes.clone());
        let replacement = Array::with_unchecked_type(dynamic_type.clone(), replacement_bytes.clone());
        let reference = ArrayIrValue::Array(initial).reference_new().unwrap();

        let read = referent(reference.read().unwrap());
        assert_eq!(read.r#type().into_owned(), dynamic_type);
        assert_eq!(read.storage_bytes(), initial_bytes.as_slice());

        // Swapping installs the replacement and hands back exactly the previous payload, so the later freeze consumes
        // the installed replacement rather than the original value.
        let old = referent(reference.swap(&ArrayIrValue::Array(replacement)).unwrap());
        assert_eq!(old.r#type().into_owned(), dynamic_type);
        assert_eq!(old.storage_bytes(), initial_bytes.as_slice());
        let frozen = referent(reference.clone().freeze().unwrap());
        assert_eq!(frozen.r#type().into_owned(), dynamic_type);
        assert_eq!(frozen.storage_bytes(), replacement_bytes.as_slice());
        let error = reference.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_ir_mutating_reference_operations_stage_as_native_variants() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let update = builder.add_input(array_type.into());
        builder.add_instruction(TestWrite::new(), Vec::new(), vec![reference, update], None).unwrap();
        let old = builder.add_instruction(TestSwap::new(), Vec::new(), vec![reference, update], None).unwrap()[0];
        builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, update], None).unwrap();
        let frozen = builder.add_instruction(TestFreeze::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![old, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let reference_write %0 %1
                    %2:f32[2] = reference_swap %0 %1
                    reference_add_update %0 %1
                    %3:f32[2] = reference_freeze %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_array_ir_reference_write_and_swap_partial_evaluation_placement() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        // Under the `Stage` placement every reference operation stages regardless of operand knowledge: the live state
        // is untouched, the write produces nothing, and the swap's previous value is an unknown of the residual
        // program.
        let live = ArrayReference::new(Array::scalar(1.0_f32));
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let replacement = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32)));
        let staging =
            PartialEvaluationContext::new_with_reference_placement(TestContext::new(), ReferencePlacement::Stage);
        assert!(
            staging
                .fold_or_residualize(
                    TestOperation::ReferenceWrite(TestWrite::new()),
                    Vec::new(),
                    &[reference.clone(), replacement.clone()]
                )
                .unwrap()
                .is_empty()
        );
        let swapped = staging
            .fold_or_residualize(
                TestOperation::ReferenceSwap(TestSwap::new()),
                Vec::new(),
                &[reference.clone(), replacement.clone()],
            )
            .unwrap();
        assert_eq!(swapped.len(), 1);
        assert!(swapped[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32)));

        // Under the default `Execute` placement all-known reference operations fold: they run against the live state in
        // program order and the swap's previous value is known.
        let executing = PartialEvaluationContext::new(TestContext::new());
        assert!(
            executing
                .fold_or_residualize(
                    TestOperation::ReferenceWrite(TestWrite::new()),
                    Vec::new(),
                    &[reference.clone(), replacement.clone()]
                )
                .unwrap()
                .is_empty()
        );
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32)));
        let swapped = executing
            .fold_or_residualize(
                TestOperation::ReferenceSwap(TestSwap::new()),
                Vec::new(),
                &[reference, PartialEvaluationValue::known(TestValue::Array(Array::scalar(3.0_f32)))],
            )
            .unwrap();
        assert_eq!(swapped.len(), 1);
        assert_eq!(swapped[0].as_known(), Some(&TestValue::Array(Array::scalar(2.0_f32))));
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_array_ir_reference_write_and_swap_reject_unscoped_transposition() {
        // The stores transpose through the cotangent accumulator of their reference operand, which only a transposition
        // context scoped to the instruction being transposed can resolve, so a detached context rejects them.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let value_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(value_type.clone())];
        assert!(matches!(
            TestWrite::new().transpose(
                &mut TranspositionContext::new(TracingContext::<TestValue, TestOperation>::new()),
                &EmptyRegionDriver,
                &inputs,
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "operand 0 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let cotangent = context.input(value_type);
        assert!(matches!(
            TestSwap::new().transpose(
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "operand 0 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));
    }

    #[test]
    fn test_array_ir_reference_allocation_and_read_stage_as_native_variants() {
        type TestContext = TracingContext<TestValue, TestOperation>;

        let (output_type, program) = TestContext::trace(
            |input| input.reference_new()?.read(),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                    %2:f32[] = reference_read %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_reapply_array_reference_view_stages_the_view_over_another_reference() {
        type TestContext = TracingContext<TestValue, TestOperation>;

        // Reapplication rebuilds a view description over a different reference of compatible geometry (here a fresh
        // input standing in for a tangent or cotangent root) through the family's own view operations.
        let root_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 4])));
        let (output_type, program) = TestContext::trace(
            |input| {
                let sliced = reapply_array_reference_view(
                    input.context(),
                    &ArrayReferenceViewTransform::Slice {
                        axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 4, 1)],
                    },
                    input.clone(),
                    &[],
                )?;
                reapply_array_reference_view(
                    input.context(),
                    &ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) },
                    sliced,
                    &[],
                )
            },
            root_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4]))));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[3, 4]> .
                let %1:ref<f32[2, 4]> = reference_slice [
                    axes=[ArraySliceAxis { start: 1, size: 2, stride: 1 }, ArraySliceAxis { start: 0, size: 4, stride: 1 }],
                ] %0
                    %2:ref<f32[4]> = reference_index [axis=0, index=1] %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reapply_array_reference_view_rejects_symbolic_indices_and_symbol_count_mismatches() {
        type TestContext = TracingContext<TestValue, TestOperation>;

        // Reapplication receives exactly one value per symbol of the description, and no array operation can stage a
        // symbolic index yet, even when its coordinate value is supplied.
        let root_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let coordinate_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let error = TestContext::trace(
            |inputs: Vec<Tracer<TestContext>>| {
                let r#static = ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) };
                reapply_array_reference_view(inputs[0].context(), &r#static, inputs[0].clone(), &inputs[1..])
            },
            vec![root_type.clone(), coordinate_type.clone()],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::MalformedProgram(
                "reapplying reference view `Index { axis: 0, index: Static(1) }` requires 0 symbol values but \
                 received 1"
                    .to_string(),
            ),
        );
        let error = TestContext::trace(
            |inputs: Vec<Tracer<TestContext>>| {
                let symbolic =
                    ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) };
                reapply_array_reference_view(inputs[0].context(), &symbolic, inputs[0].clone(), &inputs[1..])
            },
            vec![root_type, coordinate_type],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "no array operation reapplies a symbolic reference index until dynamic indexing is supported"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_array_ir_projected_reference_capabilities_bind_through_parent() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestContext::trace(
            |inputs| {
                let [initial, replacement, update]: [TestTracer; 3] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let update = <TestTracer as ValueProjection<ArrayType>>::into_projected(update)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                reference.write(&replacement)?;
                reference.add_update(&update)?;
                let frozen = reference.freeze()?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &frozen;
                Ok(vec![frozen.into_value()])
            },
            vec![array_type.clone(), array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[2] .
                let %3:ref<f32[2]> = reference_new %0
                    reference_write %3 %1
                    reference_add_update %3 %2
                    %4:f32[2] = reference_freeze %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_projected_reference_view_chains_preserve_projected_results() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let input = <TestTracer as ValueProjection<ArrayType>>::into_projected(input)?;
                let reference = input.reference_new()?;
                let sliced = reference.reference_slice(&[ArraySliceAxis::new(0, 2, 1)])?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &sliced;
                let indexed = sliced.reference_index(0, 1)?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &indexed;
                let value = indexed.read()?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &value;
                Ok(value.into_value())
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        )
        .unwrap();

        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec![
                REFERENCE_NEW_OPERATION_NAME,
                REFERENCE_SLICE_OPERATION_NAME,
                REFERENCE_INDEX_OPERATION_NAME,
                REFERENCE_READ_OPERATION_NAME,
            ],
        );
    }

    #[test]
    fn test_traced_reference_misuse_is_rejected_where_it_is_staged() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 2]));
        let consumed = "`reference_read` reads a reference whose alias family `reference_freeze` already consumed";

        // Every clone of one tracer names the same staged atom, so a handle cloned before the freeze is invalidated
        // with the rest of the alias family and its next access is reported against the operation that performs it,
        // not against the freeze and not at discharge.
        let error = TestContext::trace(
            |input: TestTracer| {
                let reference = input.reference_new()?;
                let alias = reference.clone();
                reference.freeze()?;
                alias.read()
            },
            array_type.clone(),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::MalformedProgram(consumed.to_string()));

        // Consumption invalidates a view exactly as it invalidates the allocation, because the view is an alias
        // edge onto the same family rather than an independent resource.
        let error = TestContext::trace(
            |input: TestTracer| {
                let reference = input.reference_new()?;
                let row = reference.reference_index(0, 0)?;
                reference.freeze()?;
                row.read()
            },
            array_type.clone(),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::MalformedProgram(consumed.to_string()));

        // Freezing a view is rejected in the other direction too: consumption applies to the whole allocation, which is
        // what the eager handles enforce with `CannotFreezeView` and what discharge would otherwise discover only
        // when it compared the allocation's state type against the handle's.
        let error = TestContext::trace(
            |input: TestTracer| {
                let reference = input.reference_new()?;
                reference.reference_slice(&[ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 2, 1)])?.freeze()
            },
            array_type.clone(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::MalformedProgram(
                "`reference_freeze` consumes a derived reference view, but consumption invalidates the whole alias \
                 family; consume the root handle instead"
                    .to_string(),
            ),
        );

        // Independent allocations stay independent, and a whole-family consumption of one says nothing about the other.
        let (_, program) = TestContext::trace(
            |inputs: Vec<TestTracer>| {
                let first = inputs[0].reference_new()?;
                let second = inputs[1].reference_new()?;
                let frozen = first.freeze()?;
                Ok(vec![frozen, second.read()?])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2] .
                let %2:ref<f32[2, 2]> = reference_new %0
                    %3:ref<f32[2, 2]> = reference_new %1
                    %4:f32[2, 2] = reference_freeze %2
                    %5:f32[2, 2] = reference_read %3
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_traced_structured_reference_carry_joins_its_operand_alias_family() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        // A `while` declares nothing in its reference semantics; that its carry output denotes the same reference as
        // its carry input is stated through `reference_output_identity_input` instead. The trace-time liveness state
        // honors that hook, so the loop's own result belongs to its operand's alias family and an access through it
        // after the allocation has been frozen is still reported at the access that performs it.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar_type.clone()));
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder
            .add_constant(TestValue::Array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0])));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let carry = body_builder.add_input(reference_type);
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let error = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                let reference = input.reference_new()?;
                let carried = context
                    .bind(
                        WhileOperation::new(),
                        vec![condition.clone(), body.clone()],
                        std::slice::from_ref(&reference),
                    )?
                    .remove(0);
                reference.freeze()?;
                carried.read()
            },
            ArrayIrType::Array(scalar_type),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::MalformedProgram(
                "`reference_read` reads a reference whose alias family `reference_freeze` already consumed".to_string(),
            ),
        );
    }

    #[test]
    fn test_array_ir_reference_program_matches_eager_execution() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        let inputs = (
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            TestValue::Array(Array::vector(vec![5.0_f32, 6.0])),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
        );
        let (eager_outputs, program) = TestContext::new()
            .interpret_and_trace(
                |(initial, replacement, written, update)| {
                    let reference = initial.reference_new()?;
                    let snapshot = reference.read()?;
                    let old = reference.swap(&replacement)?;
                    reference.write(&written)?;
                    reference.add_update(&update)?;
                    Ok((snapshot, old, reference.freeze()?))
                },
                inputs.clone(),
            )
            .unwrap();
        let expected = (
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            TestValue::Array(Array::vector(vec![6.0_f32, 8.0])),
        );
        assert_eq!(eager_outputs, expected);
        assert_eq!(program.interpret(inputs), Ok(expected));
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_array_ir_reference_jvp_read_modify_write() {
        // The tangent of a read-modify-write is the tangent reference's contents plus the update's tangent, and both
        // the primal and the tangent references observe their respective stores.
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![0.5_f32, 0.25])).reference_new().unwrap();
        let (primal, tangent) =
            differentiate_at((reference.clone(), TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))))
                .jvp::<TestValue, _, _>(
                    (tangent_reference.clone(), TestValue::Array(Array::vector(vec![5.0_f32, 6.0]))),
                    |(reference, value)| {
                        reference.add_update(&value)?;
                        reference.read()
                    },
                )
                .unwrap();
        assert_eq!(primal, TestValue::Array(Array::vector(vec![4.0_f32, 6.0])));
        assert_eq!(tangent, TestValue::Array(Array::vector(vec![5.5_f32, 6.25])));
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 6.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![5.5_f32, 6.25]))));
    }

    #[test]
    fn test_array_ir_reference_program_jvp_matches_discharged_program() {
        // A program that allocates, writes, reads, accumulates into, and freezes a local reference has the same fused
        // JVP boundary and values as its discharged reference-free equivalent.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(array_type.clone().into());
        let replacement = builder.add_input(array_type.into());
        let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(TestWrite::new(), Vec::new(), vec![reference, replacement], None).unwrap();
        let read = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, initial], None).unwrap();
        let frozen = builder.add_instruction(TestFreeze::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        let discharged = program
            .clone()
            .discharge_references::<ArrayReferenceDischarge>(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();
        let discharged_jvp = discharged.jvp().unwrap();
        assert_eq!(jvp.input_types(), discharged_jvp.input_types());
        assert_eq!(jvp.output_types(), discharged_jvp.output_types());

        let inputs = vec![
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            TestValue::Array(Array::vector(vec![5.0_f32, 6.0])),
            TestValue::Array(Array::vector(vec![7.0_f32, 8.0])),
        ];
        let expected = vec![
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            TestValue::Array(Array::vector(vec![4.0_f32, 6.0])),
            TestValue::Array(Array::vector(vec![7.0_f32, 8.0])),
            TestValue::Array(Array::vector(vec![12.0_f32, 14.0])),
        ];
        assert_eq!(jvp.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged_jvp.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_array_ir_reference_program_replay_binds_external_references() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let replacement = builder.add_input(array_type.into());
        builder.add_instruction(TestSwap::new(), Vec::new(), vec![external, replacement], None).unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![external], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let initial = TestValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.reference_new().unwrap();
        let outputs = program
            .interpret(vec![reference.clone(), TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))])
            .unwrap();
        assert_eq!(outputs, vec![reference.clone()]);
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))));
    }

    #[test]
    fn test_array_ir_reference_program_discards_nested_local_allocations() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference =
            true_builder.add_instruction(TestNew::new(), Vec::new(), vec![true_input], None).unwrap()[0];
        let true_output =
            true_builder.add_instruction(TestRead::new(), Vec::new(), vec![true_reference], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![true_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let false_input = false_builder.add_input(array_type.clone().into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![false_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(array_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]));
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(true)), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(program.interpret(vec![TestValue::Array(Array::scalar(false)), value.clone()]), Ok(vec![value]));
    }

    #[test]
    fn test_array_ir_reference_program_forwards_checked_allocations_into_condition_branches() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = build_branch();
        let false_branch = build_branch();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(array_type.into());
        let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let context = EagerContext::<TestValue, TestOperation>::new();
        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]));
        for predicate in [true, false] {
            assert_eq!(
                program.interpret(vec![TestValue::Array(Array::scalar(predicate)), value.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                program.entry_region_ref().interpret_in_context(
                    &context,
                    vec![TestValue::Array(Array::scalar(predicate)), value.clone()],
                    None,
                ),
                Ok(vec![value.clone()]),
            );
        }
    }

    #[test]
    fn test_array_ir_reference_while_recreates_and_discards_local_allocations_per_invocation() {
        type Values = Vec<TestValue>;

        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            let predicate = builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap();
            builder.build::<Values, Values>(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state], None).unwrap()[0];
            let update = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32)));
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, update], None).unwrap();
            let state = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let done = builder.add_constant(TestValue::Array(Array::scalar(false)));
            builder
                .build::<Values, Values>(vec![state, done], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(array_type.into());
        let predicate = builder.add_input(boolean_type.into());
        let outputs = builder
            .add_instruction(
                ArrayIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state, predicate],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        assert_eq!(
            program.interpret(vec![
                TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                TestValue::Array(Array::scalar(true)),
            ]),
            Ok(vec![TestValue::Array(Array::vector(vec![2.0_f32, 3.0])), TestValue::Array(Array::scalar(false)),]),
        );
    }

    #[test]
    fn test_array_ir_reference_scan_recreates_and_discards_local_allocations_per_iteration() {
        type Values = Vec<TestValue>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![carry], None).unwrap()[0];
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, item], None).unwrap();
            let next = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Values, Values>(vec![next, next], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(stacked_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body_region], vec![carry, items], None)
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        assert_eq!(
            program.interpret(vec![
                TestValue::Array(Array::scalar(1.0_f32)),
                TestValue::Array(Array::vector(vec![1.0_f32, 3.0, 4.0])),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(9.0_f32)),
                TestValue::Array(Array::vector(vec![2.0_f32, 5.0, 9.0])),
            ]),
        );
    }
}
