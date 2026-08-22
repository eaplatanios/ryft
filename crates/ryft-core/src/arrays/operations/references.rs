//! Array-owned reference views and eager/staging implementations of generic reference capabilities.
//!
//! Indexing and slicing remain here because they require [`ArrayReferenceViewTransform`] geometry. The generic
//! allocation, read, replacement, additive-update, and freeze payloads live in [`crate::programs::references`]; this
//! module specializes them to [`ArrayIrType`] and implements their capabilities for array values and tracers.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::reference_views::{ArrayReference, ArrayReferenceViewTransform};
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
    FreezeReference, FreezeReferenceOperation, NewReference, NewReferenceOperation, Operation, OperationFormatter,
    ProgramError, ProjectedValue, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceAliasKind,
    ReferenceOperationSemantics, ReferenceOutputSemantics, ReferenceRead, ReferenceReadOperation, ReferenceSwap,
    ReferenceSwapOperation, ReferenceType, RegionInterface, TypeError, Typed, Value, ValueProjection,
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

impl<V: Value<Type = ArrayIrType>> ReferenceIndex<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceIndexOperation>,
{
    fn reference_index(&self, axis: usize, index: usize) -> Result<V, ProgramError> {
        self.value().reference_index(axis, index)
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

impl<V: Value<Type = ArrayIrType>> ReferenceSlice<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSliceOperation>,
{
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<V, ProgramError> {
        self.value().reference_slice(axes)
    }
}

// Both view operations preserve the canonical root without accessing its state.
static REFERENCE_VIEW_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(
        vec![ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
        Vec::new(),
    )
});

/// Infers the derived reference type produced by one root-preserving view transform.
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
    #[inline]
    pub const fn new(axis: usize, index: usize) -> Self {
        Self { axis, index }
    }

    /// Returns this operation's root-preserving view transform.
    #[inline]
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

    /// Returns this operation's root-preserving view transform.
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

impl<V: Value<Type = ArrayIrType>> NewReference<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<NewReferenceOperation<ArrayType, ArrayIrType>>,
{
    fn new_reference(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(NewReferenceOperation::new(), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> NewReference<V> for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<NewReferenceOperation<ArrayType, ArrayIrType>>,
{
    fn new_reference(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(NewReferenceOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
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

impl<V: Value<Type = ArrayIrType>> ReferenceRead<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation<ArrayType, ArrayIrType>>,
{
    fn read(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
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

impl<V: Value<Type = ArrayIrType>> ReferenceSwap<ProjectedValue<ArrayType, V>, V>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType>>,
{
    fn swap(&self, replacement: &ProjectedValue<ArrayType, V>) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(ReferenceSwapOperation::new(), Vec::new(), &[self.value().clone(), replacement.value().clone()])?
            .remove(0))
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

impl<V: Value<Type = ArrayIrType>> FreezeReference<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<FreezeReferenceOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(FreezeReferenceOperation::new(), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> FreezeReference<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<FreezeReferenceOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(FreezeReferenceOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

impl<A: Value<Type = ArrayType>> NewReference for ArrayIrValue<A> {
    fn new_reference(&self) -> Result<Self, ProgramError> {
        NewReferenceOperation::<ArrayType, ArrayIrType>::new()
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
        Ok(Self::Array(reference.read_view()?))
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

impl<A: Value<Type = ArrayType>> FreezeReference for ArrayIrValue<A> {
    fn freeze(&self) -> Result<Self, ProgramError> {
        FreezeReferenceOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.freeze()?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceIndex for ArrayIrValue<A> {
    fn reference_index(&self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        let operation = ReferenceIndexOperation::new(axis, index);
        operation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceViewTransform::Index { axis, index })?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceSlice for ArrayIrValue<A> {
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        let operation = ReferenceSliceOperation::new(axes.to_vec());
        operation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
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
    use crate::arrays::reference_views::ArrayReferenceViewError;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationError, TransposableOperation};
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialEvaluationContext;
    use crate::programs::{
        Effect, Effects, EmptyRegionDriver, InstructionId, ProgramBuilder, REFERENCE_READ_OPERATION_NAME,
        ReferenceAnalysisError, ReferenceError, ReferenceRoot, TypeError,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestNew = NewReferenceOperation<ArrayType, ArrayIrType>;
    type TestRead = ReferenceReadOperation<ArrayType, ArrayIrType>;
    type TestSwap = ReferenceSwapOperation<ArrayType, ArrayIrType>;
    type TestAddUpdate = ReferenceAddUpdateOperation<ArrayType, ArrayIrType>;
    type TestFreeze = FreezeReferenceOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_array_reference_view_operations() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let index = ReferenceIndexOperation::new(0, 1);
        assert_eq!(
            index.infer_output_types(std::slice::from_ref(&ReferenceType::new(root_type.clone()).into()), &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [4])).into()]),
        );
        assert_eq!(
            index.infer_output_types(std::slice::from_ref(&root_type.clone().into()), &[]),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
        assert!(index.effects().is_pure());
        assert_eq!(
            index.reference_semantics().outputs(),
            &[ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
        );
        assert!(index.reference_semantics().accesses().is_empty());
        assert_eq!(index.transform(), ArrayReferenceViewTransform::Index { axis: 0, index: 1 });
        assert_eq!(index.to_string(), "reference_index [axis=0, index=1]");

        let slice = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)]);
        assert_eq!(
            slice.infer_output_types(std::slice::from_ref(&ReferenceType::new(root_type).into()), &[]),
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
    fn test_array_ir_reference_operation_conversions() {
        assert!(matches!(
            ArrayIrOperation::<Array>::from(NewReferenceOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::NewReference(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceReadOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceRead(_),
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
            ArrayIrOperation::<Array>::from(FreezeReferenceOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::FreezeReference(_),
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
        let reference = initial.new_reference().unwrap();
        assert!(matches!(reference, ArrayIrValue::Reference(_)));
        assert_eq!(reference.read().unwrap(), initial);
    }

    #[test]
    fn test_eager_reference_index_slice_and_composition() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let root = initial.new_reference().unwrap();
        let row = root.reference_index(0, 1).unwrap();
        assert_eq!(row.read(), Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))));

        let slice = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)]).unwrap();
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
        let root = initial.new_reference().unwrap();
        let row = root.reference_index(0, 1).unwrap();

        assert_eq!(
            row.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))),
        );
        row.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(
            root.read(),
            Ok(ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0],))),
        );
    }

    #[test]
    fn test_eager_reference_views_share_overlapping_root_state() {
        let root = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])).new_reference().unwrap();
        let left = root.reference_slice(&[ArraySliceAxis::new(0, 3, 1)]).unwrap();
        let right = root.reference_slice(&[ArraySliceAxis::new(1, 3, 1)]).unwrap();

        assert_eq!(
            left.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
        );
        assert_eq!(right.read(), Ok(ArrayIrValue::Array(Array::vector(vec![20.0_f32, 30.0, 4.0]))));
        right.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(root.read(), Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f32, 21.0, 32.0, 7.0]))));
    }

    #[test]
    fn test_eager_reference_view_validation_and_freeze_invalidation() {
        let root = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])).new_reference().unwrap();
        assert_eq!(
            root.reference_index(1, 0),
            Err(TypeError::invalid("reference index axis 1 is out of bounds for rank 1").into()),
        );
        assert_eq!(
            root.reference_index(0, 3),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3").into()),
        );
        assert_eq!(
            root.reference_slice(&[ArraySliceAxis::new(2, 2, 1)]),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 2 exceeds input size 3",).into()),
        );
        assert_eq!(
            root.reference_slice(&[ArraySliceAxis::new(0, 2, 2)]),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )
            .into()),
        );

        let view = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        let same_view = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        assert_eq!(view, same_view);
        assert_ne!(view, root);
        let different_view = root.reference_slice(&[ArraySliceAxis::new(1, 2, 1)]).unwrap();
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
        let root_handle = <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&root)
            .unwrap()
            .clone();
        assert!(root_handle.is_runtime_root_handle());
        assert!(!view_handle.is_runtime_root_handle());
        let Err(error) = view_handle.lock_root() else {
            panic!("reference view must not expose a root transaction guard")
        };
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::InvalidRuntimeRoot),
        );
        let mut views = HashMap::new();
        views.insert(view_handle, 7);
        assert_eq!(views.get(&same_view_handle), Some(&7));
        assert_eq!(views.get(&different_view_handle), None);
        assert_eq!(views.get(&root_handle), None);
        let error = view.freeze().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotFreezeView)
        );
        assert_eq!(root.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));

        assert_eq!(root.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));
        let error = view.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_reject_mismatched_member_kinds() {
        let array = ArrayIrValue::<Array>::Array(Array::scalar(1.0_f32));
        assert_eq!(array.read(), Err(TypeError::invalid("expected reference type but got array type").into()));
        let reference = array.new_reference().unwrap();
        assert_eq!(
            reference.new_reference(),
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
        let reference = initial.new_reference().unwrap();

        let error = reference.swap(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
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
        let reference = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).new_reference().unwrap();
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
        let reference = ArrayIrValue::Array(initial).new_reference().unwrap();

        let read = referent(reference.read().unwrap());
        assert_eq!(read.r#type().into_owned(), dynamic_type);
        assert_eq!(read.storage_bytes(), initial_bytes.as_slice());

        // Swapping installs the replacement and hands back exactly the previous payload, so the later freeze consumes
        // the installed replacement rather than the original value.
        let old = referent(reference.swap(&ArrayIrValue::Array(replacement)).unwrap());
        assert_eq!(old.r#type().into_owned(), dynamic_type);
        assert_eq!(old.storage_bytes(), initial_bytes.as_slice());
        let frozen = referent(reference.freeze().unwrap());
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
        let old = builder.add_instruction(TestSwap::new(), Vec::new(), vec![reference, update]).unwrap()[0];
        builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, update]).unwrap();
        let frozen = builder.add_instruction(TestFreeze::new(), Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![old, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let %2:f32[2] = reference_swap %0 %1
                    reference_add_update %0 %1
                    %3:f32[2] = freeze_reference %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_array_ir_reference_swap_rejects_transforms_before_discharge() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        let partial_context = PartialEvaluationContext::new(TestContext::new());
        assert!(matches!(
            TestSwap::new().partially_evaluate(&partial_context, &EmptyRegionDriver, &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before partial evaluation",
        ));
        let batching_context =
            BatchingContext::<_, ArrayIrBatching>::new(TestContext::new(), TestValue::Array(Array::scalar(2_i64)));
        assert!(matches!(
            TestSwap::new().batch(&batching_context, &EmptyRegionDriver, &[]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before batching",
        ));
        assert!(matches!(
            TestSwap::new().jvp(&TestContext::new(), &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`reference_swap` must be discharged before differentiation",
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
            |input| input.new_reference()?.read(),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = new_reference %0
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
                let reference = initial.new_reference()?;
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(reference)?;
                reference.write(&replacement)?;
                reference.add_update(&update)?;
                Ok(vec![reference.freeze()?])
            },
            vec![array_type.clone(), array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[2] .
                let %3:ref<f32[2]> = new_reference %0
                    %4:f32[2] = reference_swap %3 %1
                    reference_add_update %3 %2
                    %5:f32[2] = freeze_reference %3
                in (%5)
            "}
            .trim_end(),
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
                    let reference = initial.new_reference()?;
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
        program.analyze_references(0).unwrap();
        assert_eq!(program.interpret(inputs), Ok(expected));
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_array_ir_reference_program_preflight_rejects_before_external_mutation() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let replacement = builder.add_input(array_type.into());
        builder.add_instruction(TestSwap::new(), Vec::new(), vec![external, replacement]).unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![external], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let initial = TestValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert!(matches!(
            program.interpret(vec![
                reference.clone(),
                TestValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            ]),
            Err(error)
                if error.downcast_custom::<ReferenceAnalysisError>()
                    == Some(&ReferenceAnalysisError::ReferenceOutput {
                        region: program.entry(),
                        output_index: 0,
                        root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                    }),
        ));
        assert_eq!(reference.read(), Ok(initial));
    }

    #[test]
    fn test_array_ir_reference_program_discards_nested_local_roots() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference = true_builder.add_instruction(TestNew::new(), Vec::new(), vec![true_input]).unwrap()[0];
        let true_output = true_builder.add_instruction(TestRead::new(), Vec::new(), vec![true_reference]).unwrap()[0];
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
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        program.analyze_references(0).unwrap();

        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]));
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(true)), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(program.interpret(vec![TestValue::Array(Array::scalar(false)), value.clone()]), Ok(vec![value]));
    }

    #[test]
    fn test_array_ir_reference_program_forwards_checked_roots_into_condition_branches() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![initial]).unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        program.analyze_references(0).unwrap();

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
    fn test_array_ir_reference_program_validates_attached_regions_before_selection() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let valid_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let invalid_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![input]).unwrap()[0];
            builder.add_instruction(TestFreeze::new(), Vec::new(), vec![reference]).unwrap();
            builder.add_instruction(TestRead::new(), Vec::new(), vec![reference]).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let invalid_region = invalid_branch.entry();

        let error = EagerContext::<TestValue, TestOperation>::new()
            .bind(
                ConditionOperation::new(),
                vec![valid_branch, invalid_branch],
                &[TestValue::Array(Array::scalar(true)), TestValue::Array(Array::vector(vec![1.0_f32, 2.0]))],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(invalid_region, 2),
                operation: REFERENCE_READ_OPERATION_NAME.to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(invalid_region, 0), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_array_ir_reference_while_recreates_and_discards_local_roots_per_invocation() {
        type Values = Vec<TestValue>;

        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            let predicate = builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state]).unwrap()[0];
            builder.add_instruction(TestRead::new(), Vec::new(), vec![reference]).unwrap();
            builder.build::<Values, Values>(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state]).unwrap()[0];
            let update = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32)));
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, update]).unwrap();
            let state = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            )
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        program.analyze_references(0).unwrap();
        assert_eq!(
            program.interpret(vec![
                TestValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                TestValue::Array(Array::scalar(true)),
            ]),
            Ok(vec![TestValue::Array(Array::vector(vec![2.0_f32, 3.0])), TestValue::Array(Array::scalar(false)),]),
        );
    }

    #[test]
    fn test_array_ir_reference_scan_recreates_and_discards_local_roots_per_iteration() {
        type Values = Vec<TestValue>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![carry]).unwrap()[0];
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, item]).unwrap();
            let next = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Values, Values>(vec![next, next], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(stacked_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body_region], vec![carry, items])
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        program.analyze_references(0).unwrap();
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
