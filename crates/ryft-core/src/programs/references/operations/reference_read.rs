//! Generic immutable reference read operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceEffect};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::{ReferenceAddUpdateOperationProvider, ReferenceNewOperationProvider};

use crate::programs::references::types::ReferenceType;
use crate::programs::references::views::ReferenceViewOperation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

use super::forwarded_tangent;

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

static REFERENCE_READ_OPERATION_EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
    Effects::new(
        EffectClasses::NONE,
        vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
        Vec::new(),
    )
    .unwrap()
});

define_reference_primitive_payload!(
    /// Reads the current referent snapshot from a reference in the enclosing type universe `U`.
    ReferenceReadOperation
);

impl_reference_primitive_display!(ReferenceReadOperation, REFERENCE_READ_OPERATION_NAME);

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
        Cow::Borrowed(&REFERENCE_READ_OPERATION_EFFECTS)
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

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
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
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.bind(*self, Vec::new(), std::slice::from_ref(inputs[0].primal()))?.remove(0);
        Ok(vec![forwarded_tangent(&inputs[0], primal, |reference| {
            Ok(context.bind(*self, Vec::new(), std::slice::from_ref(reference))?.remove(0))
        })?])
    }
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

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U>
        + ResidualZeroProvider<U>
        + ReferenceNewOperationProvider<U>
        + ReferenceAddUpdateOperationProvider<U>,
{
    // A read is the identity map from the referenced state to its result, so its transpose accumulates the result's
    // cotangent into the cotangent reference of the read root, viewed exactly as the operand views it. The reference
    // operand carries no value cotangent of its own; its state cotangent lives in that accumulator.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let reference = context.cotangent_reference(0)?;
            context.bind(O::reference_add_update_operation()?, Vec::new(), &[reference, cotangent.clone()])?;
        }
        Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::programs::effects::EffectClass;
    use crate::programs::references::operations::ReferenceNew;
    use crate::programs::references::operations::tests::*;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;

    #[test]
    fn test_reference_read_operation() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Read::new());
        assert_eq!(Read::new().to_string(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(Read::new().effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            Read::new().effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }]
        );
        assert_eq!(Read::new().effects().reference_aliases(), &[]);

        let minimal_reference = ReadFreezeUniverse::Reference(ReferenceType::new(referent));
        assert_eq!(
            ReferenceReadOperation::<TestReferent, ReadFreezeUniverse>::new()
                .infer_output_types(std::slice::from_ref(&minimal_reference), &[]),
            Ok(vec![ReadFreezeUniverse::Value(referent)]),
        );
        assert_eq!(Read::new().infer_output_types(std::slice::from_ref(&reference), &[]), Ok(vec![value.clone()]),);
        assert_eq!(Read::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&value), &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_read_operation_reference_discharge() {
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
    fn test_reference_read_operation_jvp() {
        let context = DifferentiationContext::new(EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0])).reference_new().unwrap();

        // Reading an active reference reads its tangent reference alongside.
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0])));
        assert_eq!(outputs[0].tangent().as_value(), Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))));

        // Reading a plumbing reference yields a symbolic zero tangent of the referent's tangent type.
        let input =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0])));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        ));
    }

    #[test]
    fn test_reference_read_operation_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let packed = TestIrValue::Array(Array::from_f64s(packed_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
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
}
