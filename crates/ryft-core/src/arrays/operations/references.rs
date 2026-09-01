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
use crate::arrays::reference_views::{ArrayReference, ArrayReferenceView, ArrayReferenceViewTransform};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::ir::ArrayIrType;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::{Add, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, ProjectedValue, ReferenceAddUpdate, ReferenceAddUpdateOperation,
    ReferenceAliasKind, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew,
    ReferenceNewOperation, ReferenceOperationSemantics, ReferenceOutput, ReferenceRead, ReferenceReadOperation,
    ReferenceSwap, ReferenceSwapOperation, ReferenceType, ReferenceWrite, ReferenceWriteOperation, RegionInterface,
    TypeError, Typed, Value, ValueProjection,
};

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
        ArrayReferenceViewTransform::Index { axis: self.axis, index: self.index }
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
/// A view derives a narrower handle onto the same allocation, so on a discharged allocation the rewrite is metadata only: it
/// validates and derives the composed referent type with exactly the eager handle's arithmetic, rejecting an invalid
/// composition before any handle exists, and then records the composed chain as the derived handle's authoritative
/// alias. Nothing is bound into the destination, because the coordinates this handle selects are materialized at each
/// access rather than at the view.
///
/// On an allocation that partial discharge *preserved*, the view is additionally replayed into the destination, and the
/// reference it produces becomes the derived handle's own destination value, so that later accesses consume that
/// exact value instead of re-deriving the chain and duplicating the view operations. The composed alias is recorded
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
/// Returns [`ProgramError::InvalidInputCount`] for an application that does not supply exactly one operand,
/// [`ProgramError::MalformedProgram`] when that operand is an ordinary value rather than a reference handle, and
/// [`ProgramError::InvalidOutputCount`] when replaying the view on a preserved allocation does not produce exactly one
/// value. Propagates the view algebra's own [`TypeError`] when `transform` does not compose onto the incoming
/// handle's referent, and the discharge context's own [`ProgramError::MalformedProgram`] when the replayed
/// reference does not carry the composed type.
fn discharge_reference_view<C, P, O>(
    operation: &O,
    transform: ArrayReferenceViewTransform,
    context: &ReferenceDischargeContext<C, P>,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Type = ArrayIrType, Operation: From<O>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView>,
    O: Clone + Operation<Type = ArrayIrType>,
{
    check_count!("input", inputs, 1, ProgramError);
    let reference = inputs[0].expect_reference("a reference to view")?;
    let referent = transform.output_type(reference.r#type().referent())?;
    let alias = reference.alias().with_transform_unchecked(transform);
    Ok(vec![context.derive_reference(reference, alias, ReferenceType::new(referent), |value| {
        let mut outputs = context.parent().bind(operation.clone(), Vec::new(), std::slice::from_ref(value))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    })?])
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceIndexOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceIndexOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView>,
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
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceView>,
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

macro_rules! impl_unsupported_reference_view_transforms {
    // Installs the same conservative transform rejections for one unresolved array reference-view operation.
    ($operation:ty) => {
        impl_non_transposable_operation!($operation);

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> PartiallyEvaluatableOperation<C>
            for $operation
        {
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>, P: BatchingPolicy<C>> BatchableOperation<C, P>
            for $operation
        {
            fn batch<D: BatchingDriver<C, P>>(
                &self,
                _context: &BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<BatchedOutputs<C, P>, BatchingError> {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before batching", self.name()),
                })
            }
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> DifferentiableOperation<C> for $operation {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before differentiation", self.name()),
                }
                .into())
            }
        }
    };
}

impl_unsupported_reference_view_transforms!(ReferenceIndexOperation);
impl_unsupported_reference_view_transforms!(ReferenceSliceOperation);

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
        // Projection rejects ordinary operands and `with_transform` validates the transform against the handle's
        // cached referent type, so a separate operation-level inference pass would only repeat both checks.
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceViewTransform::Index { axis, index })?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceSlice for ArrayIrValue<A> {
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        // Projection rejects ordinary operands and `with_transform` validates the transform against the handle's
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
    use crate::arrays::batching::ArrayIrBatching;
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::reference_discharge::ArrayReferenceDischarge;
    use crate::arrays::reference_views::ArrayReferenceViewError;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DifferentiationError, TransposableOperation};
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialEvaluationContext;
    use crate::programs::{
        Effect, Effects, EmptyRegionDriver, ProgramBuilder, ProgramError, REFERENCE_NEW_OPERATION_NAME,
        REFERENCE_READ_OPERATION_NAME, ReferenceError, TypeError,
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
        assert_eq!(index.transform(), ArrayReferenceViewTransform::Index { axis: 0, index: 1 });
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
        // Both view rules derive a narrower handle onto the same allocation by composing their transform onto the incoming
        // alias, and bind nothing: a view's coordinates are materialized at each access instead.
        let context = ReferenceDischargeContext::<TestDestination, ArrayReferenceDischarge>::new(EagerContext::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let allocated = context
            .bind_discharged(
                ReferenceType::new(allocation_type.clone()),
                TestValue::Array(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect())),
            )
            .unwrap();
        let allocation = allocated.expect_reference("the allocated allocation").unwrap().clone();
        let sliced = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)])
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&allocated))
            .unwrap();
        assert_eq!(sliced.len(), 1);
        let sliced = sliced[0].expect_reference("the derived slice").unwrap().clone();
        assert_eq!(sliced.allocation_id(), allocation.allocation_id());
        assert_eq!(sliced.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 2])));
        assert_eq!(sliced.preserved(), None);

        // Composition is onto the *incoming* handle's chain, so indexing the slice selects a row of the slice rather
        // than a row of the allocation, and the composed alias is what every later access applies.
        let indexed = ReferenceIndexOperation::new(0, 1)
            .discharge_references(&context, &EmptyRegionDriver, &[ReferenceDischargeValue::Reference(sliced.clone())])
            .unwrap();
        let indexed = indexed[0].expect_reference("the derived index").unwrap().clone();
        assert_eq!(indexed.allocation_id(), allocation.allocation_id());
        assert_eq!(indexed.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            indexed.alias(),
            &ArrayReferenceView::root()
                .with_transform_unchecked(ArrayReferenceViewTransform::Slice {
                    axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
                })
                .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 1 }),
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
        let pure = ReferenceDischargeValue::Ordinary(TestValue::Array(Array::scalar(1.0_f32)));
        assert_eq!(
            ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)]).discharge_references(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&pure),
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to view but received an ordinary value".to_string(),
            )),
        );
        assert_eq!(
            ReferenceIndexOperation::new(0, 0).discharge_references(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // An allocation that partial discharge preserved survives in the destination, so the view is additionally replayed
        // there and the reference that replay produced becomes the derived handle's own destination value. The
        // composed alias is recorded exactly as it is for a discharged allocation, which is what keeps one handle's view
        // chain single-sourced whichever state its allocation is in.
        let preserved = context
            .bind_preserved(
                ReferenceType::new(allocation_type),
                TestValue::Reference(ArrayReference::new(Array::matrix(
                    3,
                    3,
                    (1..=9).map(|value| value as f32).collect(),
                ))),
            )
            .unwrap();
        let preserved_allocation = preserved.expect_reference("the preserved allocation").unwrap().allocation_id();
        let derived = ReferenceIndexOperation::new(0, 0)
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&preserved))
            .unwrap();
        assert_eq!(derived.len(), 1);
        let derived = derived[0].expect_reference("the derived preserved view").unwrap().clone();
        assert_eq!(derived.allocation_id(), preserved_allocation);
        assert_eq!(derived.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(
            derived.alias(),
            &ArrayReferenceView::root()
                .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 0 }),
        );
        assert_eq!(
            derived.preserved().map(|value| value.r#type().into_owned()),
            Some(ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])))),
        );

        // The replayed view denotes the coordinates the source named, which the eager destination proves by reading
        // through the derived handle: the first row of the preserved allocation rather than the allocation itself.
        assert_eq!(
            derived.preserved().map(ReferenceRead::read),
            Some(Ok(TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])))),
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
    fn test_array_ir_reference_write_and_swap_reject_transforms_before_discharge() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        let partial_context = PartialEvaluationContext::new(TestContext::new());
        assert!(matches!(
            TestWrite::new().partially_evaluate(&partial_context, &EmptyRegionDriver, &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_write` must be discharged before partial evaluation",
        ));
        assert!(matches!(
            TestSwap::new().partially_evaluate(&partial_context, &EmptyRegionDriver, &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before partial evaluation",
        ));
        let batching_context =
            BatchingContext::<_, ArrayIrBatching>::new(TestContext::new(), TestValue::Array(Array::scalar(2_i64)));
        assert!(matches!(
            TestWrite::new().batch(&batching_context, &EmptyRegionDriver, &[]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_write` must be discharged before batching",
        ));
        assert!(matches!(
            TestSwap::new().batch(&batching_context, &EmptyRegionDriver, &[]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before batching",
        ));
        assert!(matches!(
            TestWrite::new().jvp(&TestContext::new(), &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`reference_write` must be discharged before differentiation",
        ));
        assert!(matches!(
            TestSwap::new().jvp(&TestContext::new(), &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`reference_swap` must be discharged before differentiation",
        ));
        assert!(matches!(
            TestWrite::new().transpose(
                &mut TracingContext::<TestValue, TestOperation>::new(),
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `reference_write` is not transposable",
        ));
        assert!(matches!(
            TestSwap::new().transpose(
                &mut TracingContext::<TestValue, TestOperation>::new(),
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `reference_swap` is not transposable",
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

        // Consumption invalidates a derived view exactly as it invalidates the allocation, because the view is an alias
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
    fn test_array_ir_reference_program_boundary_validation_rejects_before_external_mutation() {
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
        assert_eq!(
            program.interpret(vec![reference.clone(), TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))]),
            Err(ProgramError::UnsupportedOperation {
                message: "program replay cannot bind external reference `input 0`; use a stateful \
                          compilation domain"
                    .to_string(),
            }),
        );
        assert_eq!(reference.read(), Ok(initial));
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
