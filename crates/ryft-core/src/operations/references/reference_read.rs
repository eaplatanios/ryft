//! Generic immutable reference read operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayType};
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
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::references::reference_add_update::ReferenceAddUpdate;
use crate::operations::references::reference_new::ReferenceNewOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClasses, Effects, MaybeZero, Operation, OperationProvider, ProgramError, ProjectedValue, ReferenceAccessMode,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceType, ReferenceViewOperation, RegionInterface, Type,
    TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::forwarded_tangent;

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Reads the current referent snapshot from a reference in the enclosing type universe `U`.
#[derive(Clone, Debug)]
pub struct ReferenceReadOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceReadOperation<T, U> {
    /// Creates a new [`ReferenceReadOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceReadOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceReadOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_READ_OPERATION_NAME)
    }
}

impl<T, U> Operation for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
                Vec::new(),
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to read")?;
        Ok(vec![ReferenceDischargeValue::Value(context.read(reference)?)])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceRead<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].read()?])
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // A read yields the packed referent, batched at the reference's own axis (or replicated with the reference).
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![P::batch(
            context.parent().bind(*self, Vec::new(), std::slice::from_ref(P::value(&inputs[0])))?.remove(0),
            P::batch_axis(&inputs[0]),
        )?]
        .into())
    }
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
{
    // Reading a reference reads its tangent reference alongside. A plumbing reference (i.e., a reference dual whose
    // tangent is a symbolic zero) carries no tangent reference, so the value read from it has a symbolic zero tangent.
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.primal().bind(*self, Vec::new(), std::slice::from_ref(inputs[0].primal()))?.remove(0);
        Ok(vec![forwarded_tangent(&inputs[0], primal, |reference| {
            Ok(context.tangent().bind(*self, Vec::new(), std::slice::from_ref(reference))?.remove(0))
        })?])
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U>
        + ResidualZeroProvider<U, Operation = O>
        + OperationProvider<U, ReferenceNewOperation<U, U>, Operation = O>,
    Tracer<TracingContext<V, O>>: ReferenceAddUpdate,
{
    // A read is the identity map from the referenced state to its result, so its transpose accumulates the result's
    // cotangent into the cotangent reference of the read root, viewed exactly as the operand views it. The reference
    // operand carries no value cotangent of its own; its state cotangent lives in that accumulator.
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
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let reference = context.cotangent_reference(driver, 0)?;
            reference.add_update(cotangent)?;
        }
        Ok(())
    }
}

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

impl<A: Value<Type = ArrayType> + Reshape + Slice> ReferenceRead for ArrayIrValue<A> {
    fn read(&self) -> Result<Self, ProgramError> {
        ReferenceReadOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.read()?))
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReference,
        ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_freeze::ReferenceFreeze;
    use crate::operations::references::reference_new::ReferenceNew;
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder, ReferenceError};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrRead = ReferenceReadOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_read() {
        let operation = Read::new();
        assert_eq!(operation.name(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!("ReferenceReadOperation({:?})", PhantomData::<fn() -> (TestReferent, TestType)>),
        );
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_read_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Read::new(),
            cases = [
                {
                    input_types = [reference.clone()],
                    output_types = [value.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [value],
                    error = "expected reference type but got value type",
                },
            ],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // Reading only requires a universe that embeds referents and projects reference types.
        check_operation_type_inference!(
            operation = ReferenceReadOperation::<TestReferent, ReadFreezeUniverse>::new(),
            cases = [{
                input_types = [ReadFreezeUniverse::Reference(ReferenceType::new(referent))],
                output_types = [ReadFreezeUniverse::Value(referent)],
            }],
        );

        // The array universe reads array referents and rejects its other members.
        let array_type = ArrayType::scalar(DataType::F32);
        let dimension_type = DimensionType::new(DimensionVariable::new("n", DimensionBounds::unbounded()));
        check_operation_type_inference!(
            operation = TestIrRead::new(),
            cases = [
                {
                    input_types = [ArrayIrType::Reference(ReferenceType::new(array_type.clone()))],
                    output_types = [ArrayIrType::Array(array_type.clone())],
                },
                {
                    input_types = [ArrayIrType::Array(array_type)],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [ArrayIrType::Dimension(dimension_type)],
                    error = "expected reference type but got dimension type",
                },
            ],
        );
    }

    #[test]
    fn test_reference_read_interpretation() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap()));

        // A read returns a snapshot of the current referent and leaves the reference live.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrRead::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&reference),
            ),
            Ok(vec![initial.clone()]),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        // Only reference members can be read.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrRead::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&initial),
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(initial.read(), Err(TypeError::invalid("expected reference type but got array type").into()));

        // A read of a dynamically typed referent returns exactly the stored dynamic type and physical bytes. Equality
        // over a dynamically typed `Array` cannot address its elements, so the referent is unwrapped and compared by
        // its declared type and storage. `Array`'s checked constructors reject dynamically shaped types, so the
        // referent comes from the test-only unchecked hatch.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let reference = TestIrValue::Reference(ArrayReference::new(Array::with_unchecked_type(
            dynamic_type.clone(),
            initial_bytes.clone(),
        )));
        let read = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrRead::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap()
        .remove(0);
        let read = <TestIrValue as ValueProjection<ArrayType>>::into_projected(read).unwrap();
        assert_eq!(read.r#type().into_owned(), dynamic_type);
        assert_eq!(read.storage_bytes(), initial_bytes.as_slice());

        // Reading a consumed reference fails against the shared allocation state.
        reference.clone().freeze().unwrap();
        let error = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrRead::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_read_partial_evaluation() {
        type TestContext = EagerContext<TestIrValue, TestIrOperation>;

        let value = TestIrValue::Array(Array::scalar(1.0_f32).unwrap());
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));

        // Under the default `Execute` placement a known reference folds the read against the live state, which the read
        // leaves untouched.
        let executing = PartialEvaluationContext::new(TestContext::new());
        let outputs = executing
            .fold_or_residualize(TestIrRead::new(), Vec::new(), std::slice::from_ref(&reference))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&value));
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));

        // Under the `Stage` placement the read stays residual regardless of operand knowledge, so eager specialization
        // never observes live reference state.
        let staging =
            PartialEvaluationContext::new(TestContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(TestIrRead::new(), Vec::new(), &[reference]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());

        // Program-level partial evaluation uses the `Stage` placement, so both a known and an unknown reference retain
        // the read in the residual program and replay it against the runtime reference.
        let known = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let replay = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = TestIrRead::new(),
            cases = [
                {
                    inputs = [(@known, known)],
                    outputs = [(@residual, value.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [(@unknown(type = reference_type, replay = replay))],
                    outputs = [(@residual, value)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_reference_read_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let packed =
            TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let reference = packed.reference_new().unwrap();

        // Reading a batched reference yields the packed referent at the reference's batch axis.
        let input =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].batch().value(), &packed);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));

        // Reading a replicated reference stays replicated.
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &packed);
    }

    #[test]
    fn test_reference_read_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<TestIrValue, TestIrOperation>::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();

        // Reading an active reference reads its tangent reference alongside.
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            outputs[0].tangent().as_value(),
            Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
        );

        // Reading a plumbing reference yields a symbolic zero tangent of the referent's tangent type.
        let input =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        ));
    }

    #[test]
    fn test_reference_read_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // `r = new(v); y = read(r)`: the read is the identity from the referenced state to its result, so its transpose
        // accumulates `ȳ` into the cotangent reference of the read root, which the allocation then freezes into `v̄`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let output = builder.add_instruction(TestIrRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
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
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );

        // The rule accumulates through the cotangent reference of its operand's root, which only a transposition
        // context scoped to the read instruction can resolve, so a detached context rejects a live result cotangent.
        let inputs = [PartialValue::Unknown(reference_type)];
        let tracing = TracingContext::<TestIrValue, TestIrOperation>::new();
        let cotangent = tracing.input(scalar_type);
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrRead::new().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));

        // A symbolic zero result cotangent contributes nothing, so the rule never touches the cotangent reference.
        assert_eq!(
            TestIrRead::new().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))],
                &accumulators,
            ),
            Ok(()),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_reference_read_reference_discharge() {
        // A read observes the allocation's current state without changing it, so the allocation stays unmutated.
        let (context, reference) = allocated_reference(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(false));

        // An value operand denotes no allocation, so the rule reports what it expected instead of reading a value.
        let pure: TestDischargeValue = ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4));
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&pure)),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received a value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_read_projected() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // Reading through a projected reference member binds the operation through the parent tracer's context and
        // hands back the projected array member.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(input)?;
                let read: ProjectedValue<ArrayType, TestTracer> = reference.read()?;
                Ok(read.into_value())
            },
            reference_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]> .
                let %1:f32[2] = reference_read %0
                in (%1)"},
        );
    }

    #[test]
    fn test_reference_read_staging() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;

        // A staged read is the native `reference_read` variant of the array operation family.
        let (output_type, program) = TestContext::trace(
            |input| input.read(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:f32[] = reference_read %0
                in (%1)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
