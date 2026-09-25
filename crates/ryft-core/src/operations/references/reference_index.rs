use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::{
    ArrayIrType, ArrayIrValue, ArrayReferenceView, ArrayReferenceViewIndex, ArrayReferenceViewPath, ArrayType,
};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, TransposableOperation, TranspositionContext,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::parameters::Parameter;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    BatchableReferenceView, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, ProgramError,
    ProjectedValue, ReferenceAlias, ReferenceAliasKind, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceType,
    ReferenceViewOperation, RegionInterface, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ReferenceIndexOperation`].
pub const REFERENCE_INDEX_OPERATION_NAME: &str = "reference_index";

/// Pure reference-to-reference operation selecting one element by index and removing its axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceIndexOperation {
    /// Axis selected in the input reference view.
    axis: usize,

    /// Index selected on `axis`.
    index: usize,
}

impl ReferenceIndexOperation {
    /// Creates a new reference index operation.
    pub const fn new(axis: usize, index: usize) -> Self {
        Self { axis, index }
    }

    /// Returns this operation's allocation-preserving view transform.
    pub const fn transform(&self) -> ArrayReferenceView {
        ArrayReferenceView::Index { axis: self.axis, index: ArrayReferenceViewIndex::Static(self.index) }
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
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        Ok(vec![ReferenceType::new(self.transform().output_type(reference.referent())?).into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // A view preserves the canonical allocation without accessing its state, so it declares one view alias and no
        // effect class. Share one descriptor to avoid allocating and validating it on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(EffectClasses::NONE, Vec::new(), vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)])
                .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis", self.axis)?;
            operation.field("index", self.index)
        })
    }
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceIndexOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceIndexOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceViewPath<C::Value>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        self.transform().discharge(self, context, inputs)
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

// The default partial-evaluation behavior applies: a view carries no effect of its own and is placed wherever its
// reference operand is.
impl<C: Context<Type = ArrayIrType, Operation: From<ReferenceIndexOperation>>> PartiallyEvaluatableOperation<C>
    for ReferenceIndexOperation
{
}

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: ReferenceViewOperation<View: BatchableReferenceView> + From<ReferenceIndexOperation>,
        >,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceIndexOperation
{
    // The axis arithmetic lives on the view description (`ArrayReferenceView::batch`); the shared rule moves
    // the source's batch axis through it and binds the batched view on the parent context.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        ReferenceViewOperation::batch(&C::Operation::from(self.clone()), context, inputs)
    }
}

impl<C: Context<Type = ArrayIrType, Value: ReferenceIndex<C::Value>>> DifferentiableOperation<C>
    for ReferenceIndexOperation
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reference_index(self.axis, self.index)?;
        // Apply the same view to a live tangent reference. A reference without tangent storage keeps a symbolic
        // zero typed with the new view, without constructing a tangent reference.
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Value(reference) => {
                let tangent = reference.reference_index(self.axis, self.index)?;
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?
            }
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal)?,
        }])
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>>>
    TransposableOperation<V, O> for ReferenceIndexOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // A view is aliasing metadata rather than a linear map of its own: the cotangent of a view operand is reached by
        // reapplying the view path to its root's cotangent reference inside the transposition context, so the reverse
        // sweep never needs this rule to run and every operand receives a structural zero.
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);
        for (accumulator, input) in accumulators.iter().zip(inputs) {
            accumulator.accumulate(context, MaybeZero::Zero(input.r#type().cotangent()?))?;
        }
        Ok(())
    }
}

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

impl<A: Value<Type = ArrayType>> ReferenceIndex for ArrayIrValue<A> {
    fn reference_index(&self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        // Projection rejects value operands and `with_transform` validates the transform against the handle's
        // cached referent type, so a separate operation-level inference pass would only repeat both checks.
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let transform = ArrayReferenceView::Index { axis, index: ArrayReferenceViewIndex::Static(index) };
        Ok(Self::Reference(reference.with_transform(transform)?))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayReference, ArrayReferenceDischarge,
        ArraySliceAxis, DataType, DimensionBounds, DimensionType, DimensionValue,
    };
    use crate::batching::{BatchAxis, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::references::reference_new::ReferenceNew;
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::operations::references::reference_slice::ReferenceSliceOperation;
    use crate::programs::EmptyRegionDriver;

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestDestination = EagerContext<TestValue, TestOperation>;

    #[test]
    fn test_reference_index() {
        let operation = ReferenceIndexOperation::new(0, 1);
        assert_eq!(operation.to_string(), "reference_index [axis=0, index=1]");
        assert_eq!(
            operation.transform(),
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) },
        );
        assert!(operation.effects().is_pure());
        assert_eq!(operation.effects().reference_aliases(), &[ReferenceAlias::new(0, 0, ReferenceAliasKind::View)]);
        assert_eq!(operation.effects().reference_effects(), &[]);
    }

    #[test]
    fn test_reference_index_type_inference() {
        let allocation_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let operation = ReferenceIndexOperation::new(0, 1);
        assert_eq!(
            operation.infer_output_types(&[ReferenceType::new(allocation_type.clone()).into()], &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [4])).into()]),
        );
        assert_eq!(
            operation.infer_output_types(&[allocation_type.into()], &[]),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
    }

    #[test]
    fn test_reference_index_interpretation() {
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let allocation =
            TestValue::Array(Array::from_elements::<f32>(matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())
                .reference_new()
                .unwrap();
        let outputs = TestDestination::new()
            .bind(ReferenceIndexOperation::new(0, 1), Vec::new(), std::slice::from_ref(&allocation))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));
        assert_eq!(
            allocation.reference_index(2, 0),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2").into()),
        );
        assert_eq!(
            allocation.reference_index(0, 2),
            Err(TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into()),
        );
    }

    #[test]
    fn test_reference_index_batching() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestDestination::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3, 4]);
        let reference = TestValue::Array(
            Array::from_elements::<f32>(packed_type, &(0..24).map(|value| value as f32).collect::<Vec<_>>()).unwrap(),
        )
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
            Ok(TestValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    &[2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
                )
                .unwrap()
            )),
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
            Ok(TestValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [3, 4]),
                    &(12..24).map(|value| value as f32).collect::<Vec<_>>()
                )
                .unwrap()
            )),
        );

        // A batch axis at the indexed axis position precedes the indexed per-item axis in the packed referent.
        let outputs = context.bind(ReferenceIndexOperation::new(1, 3), Vec::new(), &[inner]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    &[3.0, 7.0, 11.0, 15.0, 19.0, 23.0]
                )
                .unwrap()
            )),
        );

        // Replicated references are viewed unchanged and stay replicated.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
        let outputs = context.bind(ReferenceIndexOperation::new(0, 1), Vec::new(), &[replicated]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 4]))),
        );
    }

    #[test]
    fn test_reference_index_differentiation() {
        let context = DifferentiationContext::fused(TestDestination::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(
            Array::from_elements::<f32>(allocation_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .reference_new()
        .unwrap();
        let tangent_reference =
            TestValue::Array(Array::from_elements::<f32>(allocation_type, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap())
                .reference_new()
                .unwrap();

        // An active reference's view is applied to its tangent reference with the same alias, so both views select the
        // same indices of their respective allocations.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference).unwrap(),
            context.clone(),
        );
        let indexed = context.bind(ReferenceIndexOperation::new(0, 1), Vec::new(), &[active]).unwrap();
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));
        assert_eq!(
            indexed[0].tangent().as_value().unwrap().read(),
            Ok(TestValue::Array(Array::vector(vec![10.0_f32, 11.0, 12.0]).unwrap())),
        );

        // The views of a plumbing reference stay plumbing, typed with the view's own reference type.
        let plumbing =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let indexed = context.bind(ReferenceIndexOperation::new(1, 2), Vec::new(), &[plumbing]).unwrap();
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![3.0_f32, 6.0]).unwrap())));
        assert!(matches!(
            indexed[0].tangent(),
            MaybeZero::Zero(r#type)
                if *r#type == ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        ));
    }

    #[test]
    fn test_reference_index_reference_discharge() {
        let context = ReferenceDischargeContext::<TestDestination, ArrayReferenceDischarge>::new(EagerContext::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let allocated = ReferenceDischargeValue::from(
            context
                .bind_discharged(
                    ReferenceType::new(allocation_type.clone()),
                    TestValue::Array(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()).unwrap()),
                )
                .unwrap(),
        );
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().clone();
        let sliced = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)])
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&allocated))
            .unwrap();
        let sliced = sliced[0].try_as_reference("the slice view").unwrap().clone();

        // Composition is onto the *incoming* handle's chain, so indexing the slice selects a row of the slice rather
        // than a row of the allocation, and the composed alias is what every later access applies. Nothing is bound,
        // because a view's indices are materialized at each access instead.
        let indexed = ReferenceIndexOperation::new(0, 1)
            .discharge_references(&context, &EmptyRegionDriver, &[ReferenceDischargeValue::Reference(sliced.clone())])
            .unwrap();
        assert_eq!(indexed.len(), 1);
        let indexed = indexed[0].try_as_reference("the index view").unwrap().clone();
        assert_eq!(indexed.allocation_id(), allocation.allocation_id());
        assert_eq!(indexed.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(indexed.preserved(), None);
        assert_eq!(
            indexed.alias(),
            &ArrayReferenceViewPath::root()
                .with_view(ArrayReferenceView::Slice {
                    axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
                })
                .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) }),
        );
        assert_eq!(context.read(&indexed), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

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

        // There must be exactly one reference operand.
        assert_eq!(
            ReferenceIndexOperation::new(0, 0).discharge_references(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // An allocation that partial discharge preserved survives in the destination, so the view is additionally
        // replayed there and the reference that replay produced becomes the view's own destination value. The composed
        // alias is recorded exactly as it is for a discharged allocation, which is what keeps one handle's view chain
        // single-sourced whichever state its allocation is in.
        let preserved = ReferenceDischargeValue::from(
            context
                .bind_preserved(
                    ReferenceType::new(allocation_type),
                    TestValue::Reference(ArrayReference::new(
                        Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()).unwrap(),
                    )),
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
            &ArrayReferenceViewPath::root()
                .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) }),
        );
        assert_eq!(
            view.preserved().map(|value| value.r#type().into_owned()),
            Some(ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])))),
        );

        // The replayed view denotes the indices the source named, which the eager destination proves by reading
        // through the view: the first row of the preserved allocation rather than the allocation itself.
        assert_eq!(
            view.preserved().map(ReferenceRead::read),
            Some(Ok(TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()))),
        );
    }

    #[test]
    fn test_reference_index_projected() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        // Indexing a projected reference member binds the operation through the parent tracer's context and hands back
        // the projected reference member.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(input)?;
                let indexed: ProjectedValue<ReferenceType<ArrayType>, TestTracer> = reference.reference_index(0, 1)?;
                Ok(indexed.into_value())
            },
            reference_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3]))));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2, 3]> .
                let %1:ref<f32[3]> = reference_index [axis=0, index=1] %0
                in (%1)"},
        );
    }
}
