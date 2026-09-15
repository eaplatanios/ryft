//! Generic atomic additive reference update operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayType, DataType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
use crate::operations::math::add::{Add, AddOperation};
use crate::operations::references::reference_read::ReferenceReadOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClasses, Effects, MaybeZero, NoReferent, Operation, ProgramError, ProjectedValue, ReferenceAccessMode,
    ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceMemberType, ReferenceType, ReferenceViewOperation,
    RegionInterface, Type, TypeError, Typed, Value, ValueProjection,
};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceAtomicAddUpdateOperation`].
pub const REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME: &str = "reference_atomic_add_update";

/// Applies an atomic additive update whose result retains the exact referent type.
/// Refer to [`ReferenceAtomicAddUpdate`] for the ordering, scope, and caller contract.
#[derive(Clone, Debug)]
pub struct ReferenceAtomicAddUpdateOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceAtomicAddUpdateOperation<T, U> {
    /// Creates a new [`ReferenceAtomicAddUpdateOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceAtomicAddUpdateOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceAtomicAddUpdateOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME)
    }
}

impl<T, U> Operation for ReferenceAtomicAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    AddOperation<T>: Operation<Type = T>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let update = <&T>::try_from(&input_types[1])?;
        let addition_results =
            AddOperation::<T>::new().infer_output_types(&[reference.referent().clone(), update.clone()], &[])?;
        check_count!("output", addition_results, 1, TypeError);
        let addition_result = &addition_results[0];
        if addition_result != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME}` addition result type `{addition_result}` must exactly match \
                 reference referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // The effect descriptor is built once and shared by every `<T, U>` instantiation of this operation: the
        // `static` names no generic parameter, so this generic function owns exactly one instance, and `LazyLock` is
        // required because `Effects::new` validates and allocates at runtime. Nesting it here scopes it to its only
        // reader and lets `effects` hand out a borrowed `'static` descriptor without cloning on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::AtomicAccumulate }],
                Vec::new(),
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceAtomicAddUpdateOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
    P: ReferenceAccumulationPolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to accumulate into")?;
        let update = inputs[1].try_as_value("an update value")?.clone();

        // The sum of the handle's referent and the update must itself be the handle's referent, which is exactly what
        // this operation's own inference states and what a universe's addition alone does not guarantee.
        validate_operand_types(self, inputs)?;
        // Discharge is sequential replay, selecting one legal order while preserving the caller's state ordering.
        context.accumulate(reference, update)?;
        Ok(Vec::new())
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceAtomicAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceAtomicAddUpdate<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].atomic_add_update(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceAtomicAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceAtomicAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // The update is aligned with the reference's fixed batch axis before the packed accumulation.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let update = align_stored_batch(
            context,
            driver,
            REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME,
            &inputs[0],
            inputs[1].clone(),
        )?;
        context
            .parent()
            .bind(*self, Vec::new(), &[P::value(&inputs[0]).clone(), P::value(&update).clone()])?;
        Ok(Vec::new().into())
    }
}

impl_differentiable_operation! {
    <T, U> ReferenceAtomicAddUpdateOperation<T, U>,
    jvp<C>
    where
        T: Type,
        U: DifferentiableType,
        C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
    {
        |operation, context, _driver, inputs| {
            // Addition is linear, so the update's tangent is accumulated into the tangent reference exactly as the
            // primal update is accumulated into the primal reference. Accumulating a symbolic zero tangent is a no-op
            // and stages nothing. The tangent pairing is resolved before either accumulation so that a rejected
            // plumbing store leaves both references untouched.
            check_count!("input", inputs, 2, ProgramError);
            let stored = stored_tangents(REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME, &inputs[0], &inputs[1])?;
            context
                .primal()
                .bind(*operation, Vec::new(), &[inputs[0].primal().clone(), inputs[1].primal().clone()])?;
            if let Some((tangent_reference, MaybeZero::Value(tangent))) = stored {
                context.tangent().bind(*operation, Vec::new(), &[tangent_reference.clone(), tangent])?;
            }
            Ok(Vec::new())
        }
    },
    transpose<V, O>
    where
        T: Type,
        U: DifferentiableType,
        V: Value<Type = U>,
        O: ReferenceViewOperation<Type = U> + From<ReferenceReadOperation<T, U>>,
        ReferenceReadOperation<T, U>: Operation<Type = U>,
    {
        |_operation, context, driver, inputs, outputs, accumulators| {
            // An accumulation maps `(state, x) ↦ state + x`, so its transpose reads the cotangent reference as the
            // cotangent of the update and leaves the reference's contents unchanged for the earlier accesses. An
            // accumulator that nothing has reached yet holds zero, so nothing is staged and the update's cotangent
            // stays symbolic.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 0, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            let Some(accumulator) = context.cotangent_reference_if_allocated(driver, 0)? else {
                return Ok(());
            };
            if accumulators[1].is_needed() {
                let contribution = context.bind(ReferenceReadOperation::new(), Vec::new(), &[accumulator])?.remove(0);
                accumulators[1].accumulate(context, MaybeZero::Value(contribution))?;
            }
            Ok(())
        }
    },
}

/// Selects the canonical atomic accumulation operation for the reference member of a type universe.
/// Reference-free universes have no constructible referent and therefore cannot invoke this provider.
pub trait ReferenceAtomicAddUpdateOperationProvider<T: ReferenceMemberType>: Operation + Sized {
    /// Returns this family's operation that adds an update into a reference over `referent`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`] when this family provides no reference accumulation.
    fn reference_atomic_add_update(referent: &T::Referent) -> Result<Self, ProgramError>;
}

// Composite array families select the canonical payload; the reference-free array and scalar universes have no
// referent values, so their providers are unreachable by construction.
impl<O: Operation<Type = ArrayIrType> + From<ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>>>
    ReferenceAtomicAddUpdateOperationProvider<ArrayIrType> for O
{
    fn reference_atomic_add_update(_referent: &ArrayType) -> Result<Self, ProgramError> {
        Ok(ReferenceAtomicAddUpdateOperation::new().into())
    }
}

impl<O: Operation<Type = ArrayType>> ReferenceAtomicAddUpdateOperationProvider<ArrayType> for O {
    fn reference_atomic_add_update(referent: &NoReferent) -> Result<Self, ProgramError> {
        match *referent {}
    }
}

impl<O: Operation<Type = DataType>> ReferenceAtomicAddUpdateOperationProvider<DataType> for O {
    fn reference_atomic_add_update(referent: &NoReferent) -> Result<Self, ProgramError> {
        match *referent {}
    }
}

/// Atomically adds an update into a reference without returning the previous value.
///
/// Each selected scalar update occurs exactly once without tearing, with device-scoped sequential consistency:
/// atomic accesses share a total order consistent with each program instance's order. This does not make an entire
/// array update indivisible. The caller accepts any allowed ordering of competing additions, including differences
/// in floating-point results. Conflicting non-atomic accesses still require synchronization.
///
/// Sequential interpreters and reference discharge select one permitted execution order. Staged values retain the
/// atomic operation so parallel lowerings must implement its scope and ordering or reject it. The operation still
/// carries [`EffectClass::OrderedState`](crate::programs::EffectClass::OrderedState); generic transforms gain no
/// permission to reorder state effects.
pub trait ReferenceAtomicAddUpdate<Update = Self>: Sized {
    /// Atomically adds `update` to the selected stored elements under the capability's ordering contract.
    fn atomic_add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

impl<A: Value<Type = ArrayType> + Add + Reshape + Slice + UpdateSlice> ReferenceAtomicAddUpdate for ArrayIrValue<A> {
    fn atomic_add_update(&self, update: &Self) -> Result<(), ProgramError> {
        ReferenceAtomicAddUpdateOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), update.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        // The reference holder serializes the complete read/add/write transaction, including derived views.
        // This is a valid sequential execution of the per-element atomic contract.
        reference.add_update(update)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceAtomicAddUpdate<ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>>,
{
    fn atomic_add_update(&self, update: &ProjectedValue<ArrayType, V>) -> Result<(), ProgramError> {
        self.value().dispatch_domain().bind(
            ReferenceAtomicAddUpdateOperation::new(),
            Vec::new(),
            &[self.value().clone(), update.value().clone()],
        )?;
        Ok(())
    }
}

// Staged values delegate selection to their operation family over the referent of the reference being updated; a
// value that is not a reference member of its universe has no referent and is rejected before any selection.
impl<V: Value<Type: ReferenceMemberType>> ReferenceAtomicAddUpdate for V
where
    V::DispatchDomain: Context<Operation: ReferenceAtomicAddUpdateOperationProvider<V::Type>>,
{
    fn atomic_add_update(&self, update: &Self) -> Result<(), ProgramError> {
        let reference_type = self.r#type();
        let referent = reference_type
            .referent()
            .ok_or_else(|| TypeError::invalid(format!("expected reference type but got `{reference_type}`")))?;
        let operation = <V::DispatchDomain as Domain>::Operation::reference_atomic_add_update(referent)?;
        self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), update.clone()])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayReference, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::check_operation_type_inference;
    use crate::operations::references::reference_freeze::ReferenceFreezeOperation;
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestContext = EagerContext<TestValue, TestOperation>;
    type AtomicAddUpdate = ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_atomic_add_update() {
        let operation = AtomicAddUpdate::new();
        assert_eq!(operation.name(), REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::AtomicAccumulate }]
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_atomic_add_update_type_inference() {
        let referent = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = AtomicAddUpdate::new(),
            cases = [
                {
                    input_types = [ReferenceType::new(referent.clone()).into(), referent.clone().into()],
                    output_types = [],
                },
                {
                    input_types = [ReferenceType::new(referent.clone()).into(), ArrayType::scalar(DataType::F32).into()],
                    output_types = [],
                },
                {
                    input_types = [ReferenceType::new(referent).into(), ArrayType::scalar(DataType::F64).into()],
                    error = "`reference_atomic_add_update` addition result type `f64[2]` must exactly match reference referent type `f32[2]`",
                },
            ],
        );
    }

    #[test]
    fn test_reference_atomic_add_update_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1_i32, 2]).unwrap());
        let reference = TestValue::Reference(live.clone());
        assert_eq!(
            AtomicAddUpdate::new().interpret(
                &TestContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestValue::Array(Array::scalar(3_i32).unwrap())]
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4_i32, 5]).unwrap()));
        assert_eq!(reference.atomic_add_update(&TestValue::Array(Array::scalar(1.0_f64).unwrap())),
            Err(TypeError::invalid("`reference_atomic_add_update` addition result type `f64[2]` must exactly match reference referent type `i32[2]`").into()));
        assert_eq!(live.read(), Ok(Array::vector(vec![4_i32, 5]).unwrap()));
    }

    #[test]
    fn test_reference_atomic_add_update_interpretation_concurrent_updates() {
        let live = ArrayReference::new(Array::scalar(0_i32).unwrap());
        std::thread::scope(|scope| {
            let workers = (0..4)
                .map(|_| {
                    let reference = TestValue::Reference(live.clone());
                    scope.spawn(move || {
                        let update = TestValue::Array(Array::scalar(1_i32).unwrap());
                        for _ in 0..32 {
                            reference.atomic_add_update(&update).unwrap();
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(live.read(), Ok(Array::scalar(128_i32).unwrap()));
    }

    #[test]
    fn test_reference_atomic_add_update_partial_evaluation() {
        let live = ArrayReference::new(Array::scalar(2_i32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(3_i32).unwrap()));
        let staging =
            PartialEvaluationContext::new(TestContext::new()).with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(
            staging
                .fold_or_residualize(AtomicAddUpdate::new(), Vec::new(), &[reference.clone(), update.clone()])
                .map(|outputs| outputs.len()),
            Ok(0)
        );
        assert_eq!(live.read(), Ok(Array::scalar(2_i32).unwrap()));
        let executing = PartialEvaluationContext::new(TestContext::new());
        assert_eq!(
            executing
                .fold_or_residualize(AtomicAddUpdate::new(), Vec::new(), &[reference, update])
                .map(|outputs| outputs.len()),
            Ok(0)
        );
        assert_eq!(live.read(), Ok(Array::scalar(5_i32).unwrap()));
    }

    #[test]
    fn test_reference_atomic_add_update_batching() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let initial = TestValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // An update mapped at the reference's batch axis accumulates packed.
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestValue::Array(Array::from_elements::<f32>(packed_type.clone(), &[1.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let outputs = context.bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[batched, update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()
            )),
        );

        // A batched update cannot accumulate into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestValue::Array(Array::from_elements::<f32>(packed_type, &[1.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let error = context
            .bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[replicated, update])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message:
                    "`reference_atomic_add_update` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_atomic_add_update_differentiation() {
        let context = DifferentiationContext::fused(TestContext::new());
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Addition is linear, so the update's tangent accumulates into the tangent reference.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context
            .bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[active.clone(), update])
            .unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // A symbolic zero update tangent accumulates nothing and is not instantiated.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[active, update]).unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 9.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // Staged, the zero-tangent accumulation is therefore elided from the tangent side entirely: the fused program
        // accumulates the constant into the primal reference and leaves the tangent reference untouched.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference_atom = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let constant = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), vec![reference_atom, constant], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    reference_atomic_add_update %0 %2
                in (%0, %1)
            "}
            .trim_end(),
        );

        // A plumbing reference accepts an update without a live tangent, while a live update tangent has no tangent
        // reference to accumulate into and the rejection precedes the primal accumulation.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context
            .bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[plumbing.clone(), update])
            .unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
                TestValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error =
            context.bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[plumbing, update]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message:
                    "`reference_atomic_add_update` writes a live tangent into a reference that carries no tangent; pass the \
                     reference as a differentiated input instead of capturing it"
                        .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));
    }

    #[test]
    fn test_reference_atomic_add_update_reference_discharge() {
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar.clone());
        let update = builder.add_input(scalar);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(AtomicAddUpdate::new(), Vec::new(), vec![reference, update], None).unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let analysis = program.reference_analysis(0).unwrap();
        let root = analysis.roots().next().unwrap();
        assert!(analysis.is_mutated(root));
        assert_eq!(
            analysis.access_modes_for(root).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::AtomicAccumulate, ReferenceAccessMode::Consume]
        );
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![
                TestValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestValue::Array(Array::scalar(3.0_f32).unwrap())
            ]),
            Ok(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())])
        );
    }

    #[test]
    fn test_reference_atomic_add_update_transposition() {
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar.clone());
        let update = builder.add_input(scalar);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(AtomicAddUpdate::new(), Vec::new(), vec![reference, update], None).unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.interpret(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(5.0_f32).unwrap())
            ])
        );
    }
}
