//! Generic reference allocation operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReference, ArrayType, DataType};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider, TransposableOperation,
    TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::zero::Zero;
use crate::operations::math::add::AddOperation;
use crate::operations::references::reference_freeze::ReferenceFreezeOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClasses, Effects, MaybeZero, Operation, OperationProvider, ProgramError, ProjectedValue,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceType, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReferenceNewOperation`].
pub const REFERENCE_NEW_OPERATION_NAME: &str = "reference_new";

/// Allocates a reference allocation for a referent of type `T` in the enclosing type universe `U`.
#[derive(Clone, Debug)]
pub struct ReferenceNewOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceNewOperation<T, U> {
    /// Creates a new [`ReferenceNewOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceNewOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceNewOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_NEW_OPERATION_NAME)
    }
}

impl<T, U> Operation for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type + From<ReferenceType<T>>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_NEW_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let referent = <&T>::try_from(&input_types[0])?;
        if referent.is_reference() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` cannot allocate a reference whose referent type `{referent}` is \
                 itself a reference",
            )));
        }
        Ok(vec![ReferenceType::new(referent.clone()).into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // The effect descriptor is built once and shared by every `<T, U>` instantiation of this operation: the
        // `static` names no generic parameter, so this generic function owns exactly one instance, and `LazyLock` is
        // required because `Effects::new` validates and allocates at runtime. Nesting it here scopes it to its only
        // reader and lets `effects` hand out a borrowed `'static` descriptor without cloning on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new()).unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let initial = inputs[0].try_as_value("an initial reference state")?.clone();

        // The allocation's reference type is exactly the one this operation's own inference derives from the
        // initializer, so the rewrite never re-derives a referent that the type system already settled.
        let output_types = self.infer_output_types(&[initial.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let r#type = <&ReferenceType<P::Referent>>::try_from(&output_types[0]).map_err(|_| {
            ProgramError::MalformedProgram(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` inferred the non-reference output type `{}`",
                output_types[0],
            ))
        })?;
        let r#type = r#type.clone();
        if context.selects_internal(driver.source_instruction_id(), 0) {
            return Ok(vec![context.bind_discharged(r#type, initial)?.into()]);
        }

        // An unselected allocation target survives, so the operation is replayed and its result is the destination
        // reference bound to that allocation.
        let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![context.bind_preserved(r#type, outputs.remove(0))?.into()])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceNew<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reference_new()?])
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // A reference may later receive a batched value, and its batch axis is fixed by the packed referent at allocation
    // time, so the allocation is always batched: a mapped initial value keeps its axis and a replicated one is first
    // broadcast along a new leading batch axis through the driver.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let batch_axis = P::batch_axis(&inputs[0]).axis().unwrap_or(Axis::from(0));
        let initial = driver.align_batch_axis(context, inputs[0].clone(), batch_axis)?;
        let reference = context.parent().bind(*self, Vec::new(), std::slice::from_ref(P::value(&initial)))?.remove(0);
        Ok(vec![P::batch(reference, P::batch_axis(&initial))?].into())
    }
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<
            Type = U,
            Operation: From<ReferenceNewOperation<T, U>> + ResidualZeroProvider<U, Operation = C::Operation>,
        > + Zero<C::Value>,
{
    // Forward mode allocates a tangent reference beside the primal one, initialized from the initial value's tangent.
    // A symbolic zero tangent is instantiated first, because the tangent reference must exist as a concrete allocation
    // for later stores to land in: a reference type is never zero-space, so the allocation's dual always carries a live
    // tangent reference.
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.primal().bind(*self, Vec::new(), std::slice::from_ref(inputs[0].primal()))?.remove(0);
        let source = context.primal_to_tangent(inputs[0].primal().clone())?;
        let tangent = context
            .tangent()
            .bind(
                *self,
                Vec::new(),
                &[C::Operation::materialize_zero_from_residual_sources(
                    context.tangent(),
                    inputs[0].tangent().clone(),
                    std::iter::once(&source),
                )?],
            )?
            .remove(0);
        Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?])
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: Operation<Type = U> + From<AddOperation<U>> + From<ReferenceFreezeOperation<T, U>>,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
{
    // The allocation is the map from the initial value to the initial state, so its transpose is the final step of the
    // reverse sweep for its root: the cotangent accumulated into the root's cotangent reference is frozen into the
    // cotangent of the initial value. An accumulator that nothing ever reached was never allocated, so the initial
    // value's cotangent is a symbolic zero and neither `reference_new` nor `reference_freeze` is staged.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);
        let contribution = match context.take_reference_cotangent(driver, 0)? {
            Some(accumulator) => {
                MaybeZero::Value(context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[accumulator])?.remove(0))
            }
            None => MaybeZero::Zero(inputs[0].r#type().cotangent()?),
        };
        accumulators[0].accumulate(context, contribution)
    }
}

// Composite families select canonical operations; homogeneous families reject requests for reference state.
impl<O: Operation<Type = ArrayIrType> + From<ReferenceNewOperation<ArrayType, ArrayIrType>>>
    OperationProvider<ArrayIrType, ReferenceNewOperation<ArrayIrType, ArrayIrType>> for O
{
    type Operation = Self;

    fn provide(
        _request: ReferenceNewOperation<ArrayIrType, ArrayIrType>,
        input_types: &[&ArrayIrType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(ReferenceNewOperation::new().into())
    }
}

impl<O: Operation<Type = ArrayType>> OperationProvider<ArrayType, ReferenceNewOperation<ArrayType, ArrayType>> for O {
    type Operation = Self;

    fn provide(
        _request: ReferenceNewOperation<ArrayType, ArrayType>,
        _input_types: &[&ArrayType],
    ) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "this operation family does not support reference allocation".to_string(),
        })
    }
}

impl<O: Operation<Type = DataType>> OperationProvider<DataType, ReferenceNewOperation<DataType, DataType>> for O {
    type Operation = Self;

    fn provide(
        _request: ReferenceNewOperation<DataType, DataType>,
        _input_types: &[&DataType],
    ) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "this operation family does not support reference allocation".to_string(),
        })
    }
}

/// Creates a new reference initialized from this value.
pub trait ReferenceNew<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn reference_new(&self) -> Result<Output, ProgramError>;
}

impl<A: Value<Type = ArrayType>> ReferenceNew for ArrayIrValue<A> {
    fn reference_new(&self) -> Result<Self, ProgramError> {
        ReferenceNewOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let value = <Self as ValueProjection<ArrayType>>::projected(self)?.clone();
        Ok(Self::Reference(ArrayReference::new(value)))
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::check_operation_type_inference;
    use crate::operations::references::reference_read::{ReferenceRead, ReferenceReadOperation};
    use crate::operations::references::reference_write::ReferenceWrite;
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{
        EffectClass, EmptyRegionDriver, ProgramBuilder, ReferenceDischargeResult, TypeIdentityPosition,
    };

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrNew = ReferenceNewOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_new() {
        let operation = New::new();
        assert_eq!(operation.name(), REFERENCE_NEW_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_NEW_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!("ReferenceNewOperation({:?})", PhantomData::<fn() -> (TestReferent, TestType)>),
        );
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(operation.effects().reference_effects(), &[ReferenceEffect::Allocate { output_index: 0 }]);
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_new_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = New::new(),
            cases = [
                {
                    input_types = [value.clone()],
                    output_types = [reference.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [reference.clone()],
                    error = "expected value type but got reference type",
                },
            ],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&value), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // The inferred reference type carries the referent's identity and refinement, which is what later refinement
        // checks against the allocated reference compare.
        assert_eq!(
            referent.identities().collect::<Vec<_>>(),
            vec![(TypeIdentityPosition::Definition, &referent.identity)],
        );
        assert!(referent.is_refined_by(&TestReferent::new(7, 32)));

        // Allocation only requires a universe that embeds reference types and projects referents.
        check_operation_type_inference!(
            operation = ReferenceNewOperation::<TestReferent, NewUniverse>::new(),
            cases = [{
                input_types = [NewUniverse::Value(referent)],
                output_types = [NewUniverse::Reference(ReferenceType::new(referent))],
            }],
        );

        // A referent that is itself a reference is rejected even when the universe can represent the nesting.
        let nested_referent = ReferenceType::new(referent);
        check_operation_type_inference!(
            operation = ReferenceNewOperation::<ReferenceType<TestReferent>, NestedTestType>::new(),
            cases = [{
                input_types = [NestedTestType::Reference(nested_referent.clone())],
                error = format!(
                    "`reference_new` cannot allocate a reference whose referent type `{nested_referent}` is itself a \
                     reference",
                ),
            }],
        );

        // The array universe allocates array referents and rejects its other members.
        let array_type = ArrayType::scalar(DataType::F32);
        let dimension_type = DimensionType::new(DimensionVariable::new("n", DimensionBounds::unbounded()));
        check_operation_type_inference!(
            operation = TestIrNew::new(),
            cases = [
                {
                    input_types = [ArrayIrType::Array(array_type.clone())],
                    output_types = [ArrayIrType::Reference(ReferenceType::new(array_type.clone()))],
                },
                {
                    input_types = [ArrayIrType::Reference(ReferenceType::new(array_type))],
                    error = "expected array type but got reference type",
                },
                {
                    input_types = [ArrayIrType::Dimension(dimension_type)],
                    error = "expected array type but got dimension type",
                },
            ],
        );
    }

    #[test]
    fn test_reference_new_interpretation() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());

        // Allocation yields a live reference whose type and initial state come from the initializer.
        let outputs = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrNew::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&initial),
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0], TestIrValue::Reference(_)));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(outputs[0].read(), Ok(initial.clone()));

        // Every allocation is independent of the others allocated from the same initializer.
        let other = initial.reference_new().unwrap();
        assert!(matches!(other, TestIrValue::Reference(_)));
        other.write(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())).unwrap();
        assert_eq!(outputs[0].read(), Ok(initial.clone()));
        assert_eq!(other.read(), Ok(TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())));

        // Only array members can be allocated, so a reference initializer is rejected before any allocation happens.
        let mismatch: ProgramError = TypeError::invalid("expected array type but got reference type").into();
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrNew::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&other),
            ),
            Err(mismatch.clone()),
        );
        assert_eq!(other.reference_new(), Err(mismatch));

        // A dynamically shaped initializer allocates a reference of exactly its declared dynamic type. `Array`'s
        // checked constructors reject dynamically shaped types, so the referent comes from the test-only unchecked
        // hatch.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let dynamic =
            TestIrValue::Array(Array::with_unchecked_type(dynamic_type.clone(), 1.0_f32.to_le_bytes().to_vec()));
        let outputs = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrNew::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&dynamic),
        )
        .unwrap();
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayIrType::Reference(ReferenceType::new(dynamic_type)));
    }

    #[test]
    fn test_reference_new_partial_evaluation() {
        type TestContext = EagerContext<TestIrValue, TestIrOperation>;

        let initial = TestIrValue::Array(Array::scalar(1.0_f32).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // Under the default `Execute` placement a known initial value folds the allocation eagerly, so the known output
        // is a live reference holding the initial value.
        let executing = PartialEvaluationContext::new(TestContext::new());
        let outputs = executing
            .fold_or_residualize(TestIrNew::new(), Vec::new(), &[PartialEvaluationValue::known(initial.clone())])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let reference = outputs[0].as_known().unwrap();
        assert!(matches!(reference, TestIrValue::Reference(_)));
        assert_eq!(reference.read(), Ok(initial.clone()));

        // An unknown initial value residualizes the allocation, whose reference output is then unknown and is allocated
        // only when the residual program replays against the runtime initial value.
        let executing = PartialEvaluationContext::new(TestContext::new());
        let unknown = executing.unknown_input(scalar_type.clone(), 0);
        let outputs = executing.fold_or_residualize(TestIrNew::new(), Vec::new(), &[unknown]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        let evaluation = executing.into_evaluation(outputs).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                in (%1)"},
        );
        let replayed = evaluation.interpret(&TestContext::new(), std::slice::from_ref(&initial)).unwrap();
        assert_eq!(replayed.len(), 1);
        assert!(matches!(replayed[0], TestIrValue::Reference(_)));
        assert_eq!(replayed[0].read(), Ok(initial.clone()));

        // Under the `Stage` placement even a known initial value stages the allocation, so eager specialization never
        // creates live reference state. Program-level partial evaluation uses that placement, so a known initial value
        // becomes a residual input of the retained allocation.
        let staging =
            PartialEvaluationContext::new(TestContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging
            .fold_or_residualize(TestIrNew::new(), Vec::new(), &[PartialEvaluationValue::known(initial.clone())])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(scalar_type);
        let reference = builder.add_instruction(TestIrNew::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Known(initial.clone())]).unwrap();
        assert_eq!(evaluation.program().instructions().len(), 1);
        assert_eq!(evaluation.outputs().len(), 1);
        assert!(evaluation.outputs()[0].is_unknown());
        let replayed = evaluation.interpret(&TestContext::new(), &[]).unwrap();
        assert_eq!(replayed.len(), 1);
        assert!(matches!(replayed[0], TestIrValue::Reference(_)));
        assert_eq!(replayed[0].read(), Ok(initial));
    }

    #[test]
    fn test_reference_new_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            extent,
        );

        // A mapped initial value allocates a reference batched at the same axis.
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let initial =
            TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let input =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(initial.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3]))),
        );
        assert_eq!(outputs[0].batch().value().read(), Ok(initial));

        // A replicated initial value is broadcast along a new leading batch axis, so the allocation is always batched
        // and can later receive batched values.
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(initial));
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 2]), &[1.0, 2.0, 1.0, 2.0])
                    .unwrap()
            )),
        );
    }

    #[test]
    fn test_reference_new_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<TestIrValue, TestIrOperation>::new());
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());

        // A live initial tangent seeds an independent tangent reference beside the primal allocation.
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(initial.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(outputs[0].primal().read(), Ok(initial.clone()));
        let tangent_reference = outputs[0].tangent().as_value().unwrap();
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())));
        tangent_reference.write(&TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())).unwrap();
        assert_eq!(outputs[0].primal().read(), Ok(initial.clone()));

        // A symbolic zero initial tangent is instantiated as a zero-filled tangent reference, because a reference type
        // is never zero-space and later stores need a concrete allocation to land in.
        let input =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(initial).unwrap(), context.clone());
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(
            outputs[0].tangent().as_value().unwrap().read(),
            Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())),
        );
    }

    #[test]
    fn test_reference_new_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // `r = new(v); y = read(r)`: the read accumulates `ȳ` into the lazily allocated cotangent reference of `r`, and
        // the allocation's transpose then freezes that accumulator into `v̄`, so the transposed program allocates the
        // accumulator exactly once.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let reference = builder.add_instruction(TestIrNew::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output = builder
            .add_instruction(ReferenceReadOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                    %2:ref<f32[]> = reference_new %1
                    reference_add_update %2 %0
                    %3:f32[] = reference_freeze %2
                in (%3)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(3.0_f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(3.0_f32).unwrap())]),
        );

        // An allocation that nothing accumulates into was never allocated a cotangent reference, so its initial value's
        // cotangent is a symbolic zero and neither `reference_new` nor `reference_freeze` is staged.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let value = builder.add_input(scalar_type.clone());
        builder.add_instruction(TestIrNew::new(), Vec::new(), vec![initial], None).unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                in (%1, %0)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(4.0_f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(4.0_f32).unwrap()),
            ]),
        );

        // The rule takes the allocation's accumulator from the transposition scope, which only a context scoped to the
        // allocating instruction can resolve, so a detached context rejects it.
        let inputs = [PartialValue::Unknown(scalar_type)];
        let mut context = TranspositionContext::new(TracingContext::<TestIrValue, TestIrOperation>::new());
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrNew::new().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(reference_type)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "output 0 has no reference allocation in a transposition context that is not scoped to \
                    a reference-carrying instruction",
        ));
    }

    #[test]
    fn test_reference_new_reference_discharge() {
        // Allocation binds a fresh discharged reference whose entering state is the initializer and whose reference
        // type is the one this operation's own inference derives, exposed through the storage alias of a complete
        // value.
        let (context, reference) = allocated_reference(4);
        assert_eq!(context.live_allocation_ids(), vec![reference.allocation_id()]);
        assert_eq!(reference.r#type(), &ReferenceType::new(REFERENT));
        assert_eq!(reference.alias(), &TestAlias);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.discharged_state(reference.allocation_id()), Ok(TestValue::new(REFERENT, 4)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(false));

        // A reference operand is not an initial state, and the diagnostic says which operand the rule expected.
        let context = TestDischargeContext::new(TestDestination::new());
        let handle = ReferenceDischargeValue::Reference(reference);
        assert_eq!(
            New::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an initial reference state but received {handle}",
            ))),
        );

        // The rule reads its fresh allocation's reference type back out of its own inferred output type. A deliberately
        // inconsistent canonical conversion therefore cannot silently allocate an unclassifiable allocation.
        let disagreeing = TestDischargeContext::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Value(TestValue::new(NON_PROJECTING_REFERENT, 4));
        assert_eq!(
            New::new().discharge_references(&disagreeing, &EmptyRegionDriver, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "`reference_new` inferred the non-reference output type `ref<value<i7,p255>>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_new_discharge_preserves_unselected_allocation() {
        // The allocation rule consults its own replay position against the targets, so an unselected allocation
        // target is replayed rather than turned into threaded state, and its allocated reference survives in the
        // destination.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(REFERENT));
        let update = builder.add_input(TestType::Value(REFERENT));
        let allocation =
            builder.add_instruction(TestOperation::New(New::new()), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(TestOperation::AddUpdate(AddUpdate::new()), Vec::new(), vec![allocation, update], None)
            .unwrap();
        let frozen = builder
            .add_instruction(TestOperation::Freeze(Freeze::new()), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let preserved = source.clone().partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.output_count(), 1);
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
            lambda %0:value<i7,p16>, %1:value<i7,p16> .
            let %2:ref<value<i7,p16>> = reference_new %0
                reference_add_update %2 %1
                %3:value<i7,p16> = reference_freeze %2
            in (%3)"},
        );

        // Selecting that same target is the everything-selected case, so it must agree with full discharge exactly.
        let targets = source.reference_discharge_targets(0).unwrap();
        let selected = ReferenceDischargeResult::try_from(
            source.clone().partially_discharge_references(0, targets.as_slice()).unwrap(),
        )
        .unwrap();
        let full = source.discharge_references(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
        assert_eq!(
            full.program().to_string(),
            indoc! {"
            lambda %0:value<i7,p16>, %1:value<i7,p16> .
            let %2:value<i7,p16> = test.add %0 %1
            in (%2)"},
        );
    }

    #[test]
    fn test_reference_new_operation_provider() {
        // Composite array families construct the canonical reference allocation operation.
        assert!(matches!(
            TestIrOperation::provide(
                ReferenceNewOperation::<ArrayIrType, ArrayIrType>::new(),
                &[&ArrayIrType::Array(ArrayType::scalar(DataType::F32))],
            ),
            Ok(ArrayIrOperation::ReferenceNew(_)),
        ));

        // Homogeneous array and scalar families satisfy the provider contract without supporting references.
        assert!(matches!(
            ArrayOperation::<Array>::provide(ReferenceNewOperation::<ArrayType, ArrayType>::new(), &[]),
            Err(ProgramError::UnsupportedOperation { message, .. })
                if message == "this operation family does not support reference allocation",
        ));
        assert!(matches!(
            AddOperation::<DataType>::provide(ReferenceNewOperation::<DataType, DataType>::new(), &[]),
            Err(ProgramError::UnsupportedOperation { message, .. })
                if message == "this operation family does not support reference allocation",
        ));
    }

    #[test]
    fn test_reference_new_projected() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // Allocating through a projected array member binds the operation through the parent tracer's context and
        // hands back the projected reference member.
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(input)?;
                let reference: ProjectedValue<ReferenceType<ArrayType>, TestTracer> = initial.reference_new()?;
                reference.into_value().read()
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:ref<f32[2]> = reference_new %0
                    %2:f32[2] = reference_read %1
                in (%2)"},
        );
    }

    #[test]
    fn test_reference_new_staging() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;

        // A staged allocation is the native `reference_new` variant of the array operation family.
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
                in (%2)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
