//! Generic reference allocation operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::axes::Axis;
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
use crate::operations::Zero;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::{EffectClasses, Effects, ReferenceEffect};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::ReferenceFreezeOperation;
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReferenceNewOperation`].
pub const REFERENCE_NEW_OPERATION_NAME: &str = "reference_new";

/// Creates a new reference initialized from this value.
pub trait ReferenceNew<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn reference_new(&self) -> Result<Output, ProgramError>;
}

/// Supplies the allocation operation of a reference-capable operation family. Transforms use this family-owned
/// capability to allocate state without requiring downstream implementations for core-owned tracer types.
pub trait ReferenceNewOperationProvider<T: Type>: Operation<Type = T> {
    /// Returns the operation that allocates a reference from one initial referent value.
    fn reference_new_operation() -> Self;
}

static REFERENCE_NEW_OPERATION_EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
    Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new()).unwrap()
});

define_reference_primitive_payload!(
    /// Allocates a reference allocation for a referent of type `T` in the enclosing type universe `U`.
    ReferenceNewOperation
);

impl_reference_primitive_display!(ReferenceNewOperation, REFERENCE_NEW_OPERATION_NAME);

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
        Cow::Borrowed(&REFERENCE_NEW_OPERATION_EFFECTS)
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

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>> + ResidualZeroProvider<U>> + Zero<C::Value>,
{
    // Forward mode allocates a tangent reference beside the primal one, initialized from the initial value's tangent.
    // A symbolic zero tangent is instantiated first, because the tangent reference must exist as a concrete allocation
    // for later stores to land in: a reference type is never zero-space, so the allocation's dual always carries a live
    // tangent reference.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = context.bind(*self, Vec::new(), std::slice::from_ref(inputs[0].primal()))?.remove(0);
        let tangent = context
            .bind(
                *self,
                Vec::new(),
                &[C::Operation::materialize_zero_from_residual_sources(
                    context,
                    inputs[0].tangent().clone(),
                    std::iter::once(inputs[0].primal()),
                )?],
            )?
            .remove(0);
        Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?])
    }
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

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: Operation<Type = U> + From<ReferenceFreezeOperation<T, U>>,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
{
    // The allocation is the map from the initial value to the initial state, so its transpose is the final step of the
    // reverse sweep for its root: the cotangent accumulated into the root's cotangent reference is frozen into the
    // cotangent of the initial value. An accumulator that nothing ever reached was never allocated, so the initial
    // value's cotangent is a symbolic zero and neither `reference_new` nor `reference_freeze` is staged.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![match context.allocation_cotangent(0)? {
            Some(accumulator) => {
                MaybeZero::Value(context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[accumulator])?.remove(0))
            }
            None => MaybeZero::Zero(inputs[0].r#type().cotangent()?),
        }])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::EffectClass;
    use crate::programs::identities::TypeIdentityPosition;
    use crate::programs::references::discharge::ReferenceDischargeResult;
    use crate::programs::references::operations::tests::*;
    use crate::programs::references::operations::{ReferenceRead, ReferenceWrite};
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;

    #[test]
    fn test_reference_new_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(New::new());
        assert_eq!(New::new(), New::default());
        assert_eq!(format!("{:?}", New::new()), "ReferenceNewOperation");
        assert_eq!(New::new().to_string(), REFERENCE_NEW_OPERATION_NAME);
        assert_eq!(New::new().effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(New::new().effects().reference_effects(), &[ReferenceEffect::Allocate { output_index: 0 }]);
        assert_eq!(New::new().effects().reference_aliases(), &[]);

        assert_eq!(
            ReferenceNewOperation::<TestReferent, NewUniverse>::new()
                .infer_output_types(&[NewUniverse::Value(referent)], &[]),
            Ok(vec![NewUniverse::Reference(ReferenceType::new(referent))]),
        );
        assert_eq!(New::new().infer_output_types(std::slice::from_ref(&value), &[]), Ok(vec![reference.clone()]));
        assert_eq!(New::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&value), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        assert_eq!(
            referent.identities().collect::<Vec<_>>(),
            vec![(TypeIdentityPosition::Definition, &referent.identity)],
        );
        assert!(referent.is_refined_by(&promoted_refinement));

        let nested_referent = ReferenceType::new(referent);
        assert_eq!(
            ReferenceNewOperation::<ReferenceType<TestReferent>, NestedTestType>::new()
                .infer_output_types(&[NestedTestType::Reference(nested_referent.clone())], &[]),
            Err(TypeError::invalid(format!(
                "`reference_new` cannot allocate a reference whose referent type `{nested_referent}` is itself a \
                 reference",
            ))),
        );
    }

    #[test]
    fn test_reference_new_operation_reference_discharge() {
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
    fn test_reference_primitive_discharge_preserves_an_unselected_allocation() {
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
    fn test_reference_new_operation_jvp() {
        let context = DifferentiationContext::new(EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new());
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));

        // A live initial tangent seeds an independent tangent reference beside the primal allocation.
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(initial.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))).unwrap(),
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
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))));
        tangent_reference.write(&TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))).unwrap();
        assert_eq!(outputs[0].primal().read(), Ok(initial.clone()));

        // A symbolic zero initial tangent is instantiated as a zero-filled tangent reference, because a reference type
        // is never zero-space and later stores need a concrete allocation to land in.
        let input =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(initial).unwrap(), context.clone());
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(
            outputs[0].tangent().as_value().unwrap().read(),
            Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]))),
        );
    }

    #[test]
    fn test_reference_new_operation_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new(),
            extent,
        );

        // A mapped initial value allocates a reference batched at the same axis.
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let initial = TestIrValue::Array(Array::from_f64s(packed_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
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
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(initial));
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2]))),
        );
        assert_eq!(
            outputs[0].batch().value().read(),
            Ok(TestIrValue::Array(Array::from_f64s(
                ArrayType::new_static(DataType::F32, [2, 2]),
                vec![1.0, 2.0, 1.0, 2.0],
            ))),
        );
    }
}
