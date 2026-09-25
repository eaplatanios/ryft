use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReferenceView, ArrayReferenceViewPath, ArraySliceAxis, ArrayType};
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

/// Canonical operation name for [`ReferenceSliceOperation`].
pub const REFERENCE_SLICE_OPERATION_NAME: &str = "reference_slice";

/// Pure reference-to-reference operation selecting one static range on every axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    pub fn transform(&self) -> ArrayReferenceView {
        ArrayReferenceView::Slice { axes: self.axes.clone() }
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
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axes", format_args!("{:?}", self.axes)))
    }
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceSliceOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceSliceOperation>>,
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

// The default partial-evaluation behavior applies: a view carries no effect of its own and is placed wherever its
// reference operand is.
impl<C: Context<Type = ArrayIrType, Operation: From<ReferenceSliceOperation>>> PartiallyEvaluatableOperation<C>
    for ReferenceSliceOperation
{
}

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: ReferenceViewOperation<View: BatchableReferenceView> + From<ReferenceSliceOperation>,
        >,
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
        ReferenceViewOperation::batch(&C::Operation::from(self.clone()), context, inputs)
    }
}

impl<C: Context<Type = ArrayIrType, Value: ReferenceSlice<C::Value>>> DifferentiableOperation<C>
    for ReferenceSliceOperation
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reference_slice(self.axes.as_slice())?;
        // Apply the same view to a live tangent reference. A reference without tangent storage keeps a symbolic
        // zero typed with the new view, without constructing a tangent reference.
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Value(reference) => {
                let tangent = reference.reference_slice(self.axes.as_slice())?;
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?
            }
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal)?,
        }])
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>>>
    TransposableOperation<V, O> for ReferenceSliceOperation
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

/// Derives a rank-preserving static slice view of a reference without accessing its state.
pub trait ReferenceSlice<Output = Self>: Sized {
    /// Returns a reference view selecting `axes`, one static selection per input axis.
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Output, ProgramError>;
}

impl<A: Value<Type = ArrayType>> ReferenceSlice for ArrayIrValue<A> {
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        // Projection rejects value operands and `with_transform` validates the transform against the handle's
        // cached referent type, so a separate operation-level inference pass would only repeat both checks.
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceView::Slice { axes: axes.to_vec() })?))
    }
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayReference, ArrayReferenceDischarge,
        DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape,
    };
    use crate::axes::Axis;
    use crate::batching::{BatchAxis, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::{ReferenceRead, ReferenceReadOperation};
    use crate::operations::references::reference_write::ReferenceWrite;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestDestination = EagerContext<TestValue, TestOperation>;

    #[test]
    fn test_reference_slice() {
        let operation = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)]);
        assert_eq!(
            operation.to_string(),
            concat!(
                "reference_slice [\n",
                "    axes=[ArraySliceAxis { start: 1, size: 2, stride: 1 }, ",
                "ArraySliceAxis { start: 0, size: 3, stride: 1 }],\n",
                "]",
            ),
        );
        assert_eq!(
            operation.transform(),
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] },
        );
        assert!(operation.effects().is_pure());
        assert_eq!(operation.effects().reference_aliases(), &[ReferenceAlias::new(0, 0, ReferenceAliasKind::View)]);
        assert_eq!(operation.effects().reference_effects(), &[]);
    }

    #[test]
    fn test_reference_slice_type_inference() {
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 4])));
        let axes = vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)];
        check_operation_type_inference!(
            operation = ReferenceSliceOperation::new(axes.clone()),
            cases = [
                {
                    input_types = [reference.clone()],
                    output_types = [
                        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3]))),
                    ],
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
        let vector = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        check_operation_type_inference!(
            operation = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(2, 2, 1)]),
            cases = [{
                input_types = [vector.clone()],
                error = "reference slice on axis 0 with start 2 and size 2 exceeds input size 3",
            }],
        );
        check_operation_type_inference!(
            operation = ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 2)]),
            cases = [{
                input_types = [vector],
                error = "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            }],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            ReferenceSliceOperation::new(axes)
                .infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_slice_interpretation() {
        let allocation = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        let outputs = TestDestination::new()
            .bind(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]),
                Vec::new(),
                std::slice::from_ref(&allocation),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].read(), Ok(TestValue::Array(Array::vector(vec![2.0_f32, 3.0]).unwrap())));
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
    }

    #[test]
    fn test_reference_slice_partial_evaluation() {
        let vector = || Array::vector(vec![1.0_f32, 2.0, 3.0]);
        let live = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let reference = PartialEvaluationValue::known(live.clone());
        let axes = vec![ArraySliceAxis::new(1, 2, 1)];

        // Under the default `Execute` placement, a known reference folds the view into a handle that aliases the same
        // allocation.
        let executing = PartialEvaluationContext::new(TestDestination::new());
        let outputs = executing
            .fold_or_residualize(
                ReferenceSliceOperation::new(axes.clone()),
                Vec::new(),
                std::slice::from_ref(&reference),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&live.reference_slice(&axes).unwrap()));

        // Under the `Stage` placement, the view stays residual regardless of input knowledge, so the residual program
        // keeps the complete alias chain of every later access.
        let staging =
            PartialEvaluationContext::new(TestDestination::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging
            .fold_or_residualize(ReferenceSliceOperation::new(axes.clone()), Vec::new(), &[reference])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());

        // Program-level partial evaluation uses the `Stage` placement, so both a known and an unknown reference retain
        // the view in the residual program and replay it against the runtime reference.
        let known = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let replay = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ReferenceSliceOperation::new(axes.clone()),
            cases = [
                {
                    inputs = [(@known, known.clone())],
                    outputs = [(@residual, known.reference_slice(&axes).unwrap())],
                    residual_instructions = 1,
                },
                {
                    inputs = [(@unknown(type = reference_type, replay = replay.clone()))],
                    outputs = [(@residual, replay.reference_slice(&axes).unwrap())],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_reference_slice_batching() {
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

        // Slicing inserts an identity selection at the batch axis position and keeps the output batch axis.
        let inner =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let axes = vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)];
        let outputs = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[inner]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [1, 2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [1, 3, 2]),
                    &[13.0, 14.0, 17.0, 18.0, 21.0, 22.0]
                )
                .unwrap()
            )),
        );

        // Replicated references are viewed unchanged and stay replicated.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
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
        let extent = trace.input(DimensionType::from(batch.clone()).into());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        let reference = trace.input(ReferenceType::new(dynamic_type.clone()).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace, extent);
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
    fn test_reference_slice_differentiation() {
        let context = DifferentiationContext::fused(TestDestination::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(
            Array::from_elements::<f32>(allocation_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .reference_new()
        .unwrap();
        let tangent_reference = TestValue::Array(
            Array::from_elements::<f32>(allocation_type.clone(), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
        )
        .reference_new()
        .unwrap();

        // An active reference's view is applied to its tangent reference with the same alias, so both views select the
        // same indices of their respective allocations.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );
        let axes = vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)];
        let sliced = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[active]).unwrap();
        assert_eq!(sliced.len(), 1);
        let sliced_type = ArrayType::new_static(DataType::F32, [2, 2]);
        assert_eq!(
            sliced[0].primal().read(),
            Ok(TestValue::Array(Array::from_elements::<f32>(sliced_type.clone(), &[2.0, 3.0, 5.0, 6.0]).unwrap())),
        );
        assert_eq!(
            sliced[0].tangent().as_value().unwrap().read(),
            Ok(TestValue::Array(Array::from_elements::<f32>(sliced_type.clone(), &[8.0, 9.0, 11.0, 12.0]).unwrap())),
        );

        // A store through the tangent view lands in the tangent allocation and leaves the primal allocation untouched.
        let zeros = TestValue::Array(Array::from_elements::<f32>(sliced_type, &[0.0; 4]).unwrap());
        sliced[0].tangent().as_value().unwrap().write(&zeros).unwrap();
        assert_eq!(
            tangent_reference.read(),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(allocation_type.clone(), &[7.0, 0.0, 0.0, 10.0, 0.0, 0.0]).unwrap()
            )),
        );
        assert_eq!(
            reference.read(),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(allocation_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
            )),
        );

        // The views of a plumbing reference stay plumbing, typed with the view's own reference type.
        let plumbing =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let axes = vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let sliced = context.bind(ReferenceSliceOperation::new(axes), Vec::new(), &[plumbing]).unwrap();
        assert!(matches!(
            sliced[0].tangent(),
            MaybeZero::Zero(r#type)
                if *r#type == ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [1, 3]))),
        ));
    }

    #[test]
    fn test_reference_slice_transposition() {
        let vector_type = ArrayType::new_static(DataType::F32, [4]);

        // `r = new(x); y = read(slice(r))`: the view has no transpose of its own, because the read accumulates `ȳ` into
        // the sliced range of the cotangent reference of `r`, which the allocation then freezes into `x̄`.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let sliced = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(ReferenceReadOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![sliced], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[4] = zero [type=f32[4]]
                    %2:ref<f32[4]> = reference_new %1
                    %3:ref<f32[2]> = reference_slice [axes=[ArraySliceAxis { start: 1, size: 2, stride: 1 }]] %2
                    () = reference_add_update %3 %0
                    %4:f32[4] = reference_freeze %2
                in (%4)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap())]),
            Ok(vec![TestValue::Array(Array::vector(vec![0.0_f32, 1.0, 2.0, 0.0]).unwrap())]),
        );

        // The rule itself gives the viewed reference a structural zero whatever its output cotangent is, so it never
        // stages anything into the transposition context.
        let inputs = [PartialValue::Unknown(ArrayIrType::Reference(ReferenceType::new(vector_type)))];
        let tracing = TracingContext::<TestValue, TestOperation>::new();
        let cotangent =
            tracing.input(ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))));
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]).transpose(
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
    fn test_reference_slice_reference_discharge() {
        // A slice creates a narrower reference to the same allocation by composing its transform onto the incoming
        // alias, and binds nothing: a view's indices are materialized at each access instead.
        let context = ReferenceDischargeContext::<TestDestination, ArrayReferenceDischarge>::new(EagerContext::new());
        let allocated = ReferenceDischargeValue::from(
            context
                .bind_discharged(
                    ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 3])),
                    TestValue::Array(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()).unwrap()),
                )
                .unwrap(),
        );
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().clone();
        let axes = vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)];
        let sliced = ReferenceSliceOperation::new(axes.clone())
            .discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&allocated))
            .unwrap();
        assert_eq!(sliced.len(), 1);
        let sliced = sliced[0].try_as_reference("the slice view").unwrap().clone();
        assert_eq!(sliced.allocation_id(), allocation.allocation_id());
        assert_eq!(sliced.r#type(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 2])));
        assert_eq!(sliced.preserved(), None);
        assert_eq!(sliced.alias(), &ArrayReferenceViewPath::root().with_view(ArrayReferenceView::Slice { axes }));
        assert_eq!(
            context.read(&sliced),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 2]), &[4.0, 5.0, 7.0, 8.0])
                    .unwrap()
            )),
        );

        // The operand must be a reference handle.
        let pure = ReferenceDischargeValue::Value(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
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
    }

    #[test]
    fn test_reference_slice_projected() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        // Slicing a projected reference member binds the operation through the parent tracer's context and hands back
        // the projected reference member.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(input)?;
                let sliced: ProjectedValue<ReferenceType<ArrayType>, TestTracer> =
                    reference.reference_slice(&[ArraySliceAxis::new(0, 2, 1)])?;
                Ok(sliced.into_value())
            },
            reference_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[3]> .
                let %1:ref<f32[2]> = reference_slice [axes=[ArraySliceAxis { start: 0, size: 2, stride: 1 }]] %0
                in (%1)"},
        );
    }
}
