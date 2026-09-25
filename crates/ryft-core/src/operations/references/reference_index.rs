use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

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

/// Pure reference-to-reference operation selecting the hyperplane at a static `index` along `axis` and removing that
/// axis. For example, index `1` on axis `0` of a `f32[2, 3]` reference selects its second row as a `f32[3]` reference.
/// The resulting reference aliases the same allocation, and constructing the view does not access its contents.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
        // A view is aliasing metadata rather than a linear map of its own: the cotangent of a view operand is reached
        // by reapplying the view path to its root's cotangent reference inside the transposition context, so the
        // reverse sweep never needs this rule to run and every operand receives a structural zero.
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
    /// Returns a reference view selecting the hyperplane at `index` along `axis`, removing that axis and sharing the
    /// original allocation. `index` must be in bounds for `axis`.
    fn reference_index(&self, axis: usize, index: usize) -> Result<Output, ProgramError>;
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayReference, ArrayReferenceDischarge,
        ArraySliceAxis, DataType, DimensionBounds, DimensionType, DimensionValue,
    };
    use crate::batching::{BatchAxis, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::{ReferenceRead, ReferenceReadOperation};
    use crate::operations::references::reference_slice::ReferenceSliceOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

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
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 4])));
        let row = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4])));
        check_operation_type_inference!(
            operation = ReferenceIndexOperation::new(0, 1),
            cases = [
                {
                    input_types = [reference.clone()],
                    output_types = [row],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 4]))],
                    error = "expected reference type but got array type",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ReferenceIndexOperation::new(2, 0),
            cases = [{
                input_types = [reference.clone()],
                error = "reference index axis 2 is out of bounds for rank 2",
            }],
        );
        check_operation_type_inference!(
            operation = ReferenceIndexOperation::new(0, 3),
            cases = [{
                input_types = [reference.clone()],
                error = "reference index 3 on axis 0 is out of bounds for size 3",
            }],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            ReferenceIndexOperation::new(0, 1)
                .infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
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
    fn test_reference_index_partial_evaluation() {
        let matrix = || {
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        };
        let live = TestValue::Reference(ArrayReference::new(matrix().unwrap()));
        let reference = PartialEvaluationValue::known(live.clone());

        // Under the default `Execute` placement, a known reference folds the view into a handle that aliases the same
        // allocation.
        let executing = PartialEvaluationContext::new(TestDestination::new());
        let outputs = executing
            .fold_or_residualize(ReferenceIndexOperation::new(0, 1), Vec::new(), std::slice::from_ref(&reference))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&live.reference_index(0, 1).unwrap()));

        // Under the `Stage` placement, the view stays residual regardless of input knowledge, so the residual program
        // keeps the complete alias chain of every later access.
        let staging =
            PartialEvaluationContext::new(TestDestination::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs =
            staging.fold_or_residualize(ReferenceIndexOperation::new(0, 1), Vec::new(), &[reference]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());

        // Program-level partial evaluation uses the `Stage` placement, so both a known and an unknown reference retain
        // the view in the residual program and replay it against the runtime reference.
        let known = TestValue::Reference(ArrayReference::new(matrix().unwrap()));
        let replay = TestValue::Reference(ArrayReference::new(matrix().unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ReferenceIndexOperation::new(0, 1),
            cases = [
                {
                    inputs = [(@known, known.clone())],
                    outputs = [(@residual, known.reference_index(0, 1).unwrap())],
                    residual_instructions = 1,
                },
                {
                    inputs = [(@unknown(type = reference_type, replay = replay.clone()))],
                    outputs = [(@residual, replay.reference_index(0, 1).unwrap())],
                    residual_instructions = 1,
                },
            ],
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
    fn test_reference_index_transposition() {
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);

        // `r = new(x); y = read(index(r))`: the view has no transpose of its own, because the read accumulates `ȳ` into
        // the indexed row of the cotangent reference of `r`, which the allocation then freezes into `x̄`.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(matrix_type.clone()));
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let row = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReferenceReadOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![row], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[2, 3] = zero [type=f32[2, 3]]
                    %2:ref<f32[2, 3]> = reference_new %1
                    %3:ref<f32[3]> = reference_index [axis=0, index=1] %2
                    () = reference_add_update %3 %0
                    %4:f32[2, 3] = reference_freeze %2
                in (%4)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())]),
            Ok(vec![TestValue::Array(
                Array::from_elements::<f32>(matrix_type.clone(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]).unwrap()
            )]),
        );

        // The rule itself gives the viewed reference a structural zero whatever its output cotangent is, so it never
        // stages anything into the transposition context.
        let inputs = [PartialValue::Unknown(ArrayIrType::Reference(ReferenceType::new(matrix_type)))];
        let tracing = TracingContext::<TestValue, TestOperation>::new();
        let cotangent =
            tracing.input(ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3]))));
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            ReferenceIndexOperation::new(0, 1).transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Ok(()),
        );
        assert!(context.builder().borrow().instructions().is_empty());
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
