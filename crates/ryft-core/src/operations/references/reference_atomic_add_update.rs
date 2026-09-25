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
use crate::operations::arithmetic::{Add, AddOperation};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
use crate::operations::references::reference_read::ReferenceReadOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClasses, Effects, MaybeZero, NoReferent, Operation, OperationProvider, ProgramError, ProjectedValue,
    ReferenceAccessMode, ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceEffect, ReferenceMemberType, ReferenceType,
    ReferenceViewOperation, RegionInterface, Type, TypeError, Typed, Value, ValueProjection,
};

/// Canonical operation name for [`ReferenceAtomicAddUpdateOperation`].
pub const REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME: &str = "reference_atomic_add_update";

/// Applies an atomic additive update that preserves the reference's exact referent type.
/// Refer to [`ReferenceAtomicAddUpdate`] for the ordering, scope, and caller contract.
#[derive(Clone, Debug)]
pub struct ReferenceAtomicAddUpdateOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceAtomicAddUpdateOperation<T, U> {
    /// Creates a new [`ReferenceAtomicAddUpdateOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Default for ReferenceAtomicAddUpdateOperation<T, U> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, U: Type> Copy for ReferenceAtomicAddUpdateOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceAtomicAddUpdateOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME)
    }
}

impl<T: Type, U: Type> Operation for ReferenceAtomicAddUpdateOperation<T, U>
where
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
        let addition_outputs =
            AddOperation::<T>::new().infer_output_types(&[reference.referent().clone(), update.clone()], &[])?;
        check_count!("output", addition_outputs, 1, TypeError);
        let addition_output = &addition_outputs[0];
        if addition_output != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{}` addition output type `{}` must exactly match reference referent type `{}`",
                REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME,
                addition_output,
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // Share one descriptor across all type instantiations to avoid allocating and validating it on every query.
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

impl<
    T: Type,
    U: Type + From<T> + From<ReferenceType<T>>,
    C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
    P: ReferenceAccumulationPolicy<C, Referent = T>,
> ReferenceDischargeableOperation<C, P> for ReferenceAtomicAddUpdateOperation<T, U>
where
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
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
        // this operation's own inference states and what a universe's addition alone does not guarantee. Rules can be
        // called outside a validated program, so check the input types before changing reference state.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;

        // Discharge is sequential replay, selecting one legal order while preserving the caller's state ordering.
        context.accumulate(reference, update)?;
        Ok(Vec::new())
    }
}

impl<T: Type, U: Type, C: Domain<Type = U, Value: ReferenceAtomicAddUpdate<C::Value>>> InterpretableOperation<C>
    for ReferenceAtomicAddUpdateOperation<T, U>
where
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
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

impl<T: Type, U: Type, C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>>
    PartiallyEvaluatableOperation<C> for ReferenceAtomicAddUpdateOperation<T, U>
{
}

impl<
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceAtomicAddUpdateOperation<T, U>>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceAtomicAddUpdateOperation<T, U>
where
    ReferenceAtomicAddUpdateOperation<T, U>: Operation<Type = U>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        // The reference's batch axis is fixed. Broadcast or move the stored value to match it.
        let update = match (P::batch_axis(&inputs[0]).axis(), P::batch_axis(&inputs[1]).axis()) {
            (Some(axis), _) => driver.align_batch_axis(context, inputs[1].clone(), axis)?,
            (None, None) => inputs[1].clone(),
            (None, Some(_)) => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME}` cannot store a batched value into an unbatched \
                         reference; pass the reference as a batched input instead",
                    ),
                });
            }
        };
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
            // Accumulate into the tangent reference alongside the primal as a zero update needs no tangent work, and
            // reject a live tangent without tangent storage before either reference can be changed.
            check_count!("input", inputs, 2, ProgramError);
            let stored = match (inputs[0].tangent(), inputs[1].tangent()) {
                (MaybeZero::Value(reference), tangent) => Some((reference, tangent.clone())),
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => None,
                (MaybeZero::Zero(_), MaybeZero::Value(_)) => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME}` writes a live tangent into a reference \
                             that carries no tangent; pass the reference as a differentiated input instead of \
                             capturing it",
                        ),
                    }
                    .into());
                }
            };
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

// The blanket ReferenceAtomicAddUpdate implementation requires this provider bound before its method can reject
// a non-reference input. Providing it keeps that capability available for generic scalar/array values, with a
// runtime error for unsupported calls. Differentiation itself does not require atomic accumulation.
impl<O: Operation<Type = DataType>> OperationProvider<DataType, ReferenceAtomicAddUpdateOperation<NoReferent, DataType>>
    for O
{
    type Operation = Self;

    fn provide(
        _request: ReferenceAtomicAddUpdateOperation<NoReferent, DataType>,
        input_types: &[&DataType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME}` is not supported in a reference-free type universe",
            ),
        })
    }
}

// The blanket ReferenceAtomicAddUpdate implementation requires this provider bound before its method can reject
// a non-reference input. Providing it keeps that capability available for generic scalar/array values, with a
// runtime error for unsupported calls. Differentiation itself does not require atomic accumulation.
impl<O: Operation<Type = ArrayType>>
    OperationProvider<ArrayType, ReferenceAtomicAddUpdateOperation<NoReferent, ArrayType>> for O
{
    type Operation = Self;

    fn provide(
        _request: ReferenceAtomicAddUpdateOperation<NoReferent, ArrayType>,
        input_types: &[&ArrayType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME}` is not supported in a reference-free type universe",
            ),
        })
    }
}

impl<O: Operation<Type = ArrayIrType> + From<ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>>>
    OperationProvider<ArrayIrType, ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>> for O
{
    type Operation = Self;

    fn provide(
        request: ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>,
        input_types: &[&ArrayIrType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        request.infer_output_types(&[input_types[0].clone(), input_types[1].clone()], &[])?;
        Ok(request.into())
    }
}

/// Capability to atomically add an update into a reference without returning the previous value. Each selected scalar
/// update occurs exactly once without tearing, with device-scoped sequential consistency: atomic accesses share a total
/// order consistent with each program instance's order. This does not make an entire array update indivisible. The
/// caller accepts any allowed ordering of competing additions, including differences in floating-point sums.
/// Conflicting non-atomic accesses still require synchronization.
///
/// Sequential interpreters and reference discharge select one permitted execution order. Staged values retain the
/// atomic operation so parallel lowering must implement its scope and ordering or reject it. The operation still
/// carries [`EffectClass::OrderedState`](crate::EffectClass::OrderedState) and so generic transforms gain no
/// permission to reorder state effects.
pub trait ReferenceAtomicAddUpdate<Update = Self>: Sized {
    /// Atomically adds `update` to the selected stored elements under the capability's ordering contract.
    fn atomic_add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

impl<A: Value<Type = ArrayType> + Add + Reshape + Slice + UpdateSlice> ReferenceAtomicAddUpdate for ArrayIrValue<A> {
    fn atomic_add_update(&self, update: &Self) -> Result<(), ProgramError> {
        let operation = ReferenceAtomicAddUpdateOperation::<ArrayType, ArrayIrType>::new();
        operation.infer_output_types(&[self.r#type().into_owned(), update.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;

        // The reference holder serializes the complete read/add/write transaction, including derived views.
        // This is a valid sequential execution of the per-element atomic contract.
        reference.add_update(update)
    }
}

// Staged values delegate selection to their operation family over the referent of the reference being updated;
// a value that is not a reference member of its universe has no referent and is rejected before any selection.
impl<
    V: Value<
            Type: ReferenceMemberType,
            DispatchDomain: Context<
                Operation: OperationProvider<
                    V::Type,
                    ReferenceAtomicAddUpdateOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
                    Operation = <V::DispatchDomain as Domain>::Operation,
                >,
            >,
        >,
> ReferenceAtomicAddUpdate for V
{
    fn atomic_add_update(&self, update: &Self) -> Result<(), ProgramError> {
        let reference_type = self.r#type();
        reference_type
            .referent()
            .ok_or_else(|| TypeError::invalid(format!("expected reference type but got `{reference_type}`")))?;
        let operation = <V::DispatchDomain as Domain>::Operation::provide(
            ReferenceAtomicAddUpdateOperation::new(),
            &[reference_type.as_ref(), update.r#type().as_ref()],
        )?;
        self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), update.clone()])?;
        Ok(())
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<
                Type = ArrayIrType,
                Operation: From<ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>>,
            >,
        >,
> ReferenceAtomicAddUpdate<ProjectedValue<ArrayType, V>> for ProjectedValue<ReferenceType<ArrayType>, V>
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayOperation, ArrayReference, DimensionBounds,
        DimensionType, DimensionValue,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::references::reference_freeze::ReferenceFreezeOperation;
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrContext = EagerContext<TestIrValue, TestIrOperation>;
    type TestIrReferenceAtomicAddUpdateOperation = ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_atomic_add_update() {
        let operation = TestIrReferenceAtomicAddUpdateOperation::new();
        assert_eq!(TestIrReferenceAtomicAddUpdateOperation::default().to_string(), operation.to_string());
        assert_eq!(operation.name(), REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!("ReferenceAtomicAddUpdateOperation({:?})", PhantomData::<fn() -> (ArrayType, ArrayIrType)>),
        );
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::AtomicAccumulate }],
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_atomic_add_update_type_inference() {
        let referent = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrReferenceAtomicAddUpdateOperation::new(),
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
                    error = "`reference_atomic_add_update` addition output type `f64[2]` must exactly match reference \
                             referent type `f32[2]`",
                },
            ],
        );
    }

    #[test]
    fn test_reference_atomic_add_update_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1_i32, 2]).unwrap());
        let reference = TestIrValue::Reference(live.clone());
        assert_eq!(
            TestIrReferenceAtomicAddUpdateOperation::new().interpret(
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::scalar(3_i32).unwrap())],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4_i32, 5]).unwrap()));
        assert_eq!(
            reference.atomic_add_update(&TestIrValue::Array(Array::scalar(1.0_f64).unwrap())),
            Err(TypeError::invalid(
                "`reference_atomic_add_update` addition output type `f64[2]` must exactly match reference \
                 referent type `i32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4_i32, 5]).unwrap()));
    }

    #[test]
    fn test_reference_atomic_add_update_interpretation_concurrent_updates() {
        let live = ArrayReference::new(Array::scalar(0_i32).unwrap());
        std::thread::scope(|scope| {
            let workers = (0..4)
                .map(|_| {
                    let reference = TestIrValue::Reference(live.clone());
                    scope.spawn(move || {
                        let update = TestIrValue::Array(Array::scalar(1_i32).unwrap());
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
        // Program replay keeps atomic updates residual for both known and unknown update values.
        let replayed = ArrayReference::new(Array::scalar(1_i32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceAtomicAddUpdateOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(replayed.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2_i32).unwrap())),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, TestIrValue::Reference(replayed.clone())),
                        (@unknown(
                            type = ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
                            replay = TestIrValue::Array(Array::scalar(3_i32).unwrap())
                        )),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(replayed.read(), Ok(Array::scalar(6_i32).unwrap()));

        // Staging leaves live state untouched; executing an all-known update changes it immediately.
        let live = ArrayReference::new(Array::scalar(2_i32).unwrap());
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));
        let update = PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(3_i32).unwrap()));
        let staging =
            PartialEvaluationContext::new(TestIrContext::new()).with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(
            staging
                .fold_or_residualize(
                    TestIrReferenceAtomicAddUpdateOperation::new(),
                    Vec::new(),
                    &[reference.clone(), update.clone()],
                )
                .map(|outputs| outputs.len()),
            Ok(0),
        );
        assert_eq!(live.read(), Ok(Array::scalar(2_i32).unwrap()));
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        assert_eq!(
            executing
                .fold_or_residualize(TestIrReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[reference, update])
                .map(|outputs| outputs.len()),
            Ok(0),
        );
        assert_eq!(live.read(), Ok(Array::scalar(5_i32).unwrap()));
    }

    #[test]
    fn test_reference_atomic_add_update_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let initial = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // An update mapped at the reference's batch axis accumulates packed.
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type.clone(), &[1.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let outputs = context.bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[batched, update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()
            )),
        );

        // A batched update cannot accumulate into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[1.0; 6]).unwrap()),
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
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Addition is linear, so the update's tangent accumulates into the tangent reference.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context
            .bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[active.clone(), update])
            .unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // A symbolic zero update tangent accumulates nothing and is not instantiated.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[active, update]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 9.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // Staged, the zero-tangent accumulation is therefore elided from the tangent side entirely: the fused program
        // accumulates the constant into the primal reference and leaves the tangent reference untouched.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference_atom = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let constant = builder.add_constant(TestIrValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), vec![reference_atom, constant], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    () = reference_atomic_add_update %0 %2
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
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context
            .bind(ReferenceAtomicAddUpdateOperation::new(), Vec::new(), &[plumbing.clone(), update])
            .unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
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
                    "`reference_atomic_add_update` writes a live tangent into a reference that carries no tangent; \
                     pass the reference as a differentiated input instead of capturing it"
                        .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));
    }

    #[test]
    fn test_reference_atomic_add_update_transposition() {
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar.clone());
        let update = builder.add_input(scalar);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(TestIrReferenceAtomicAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_reference_atomic_add_update_reference_discharge() {
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar.clone());
        let update = builder.add_input(scalar);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(TestIrReferenceAtomicAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let analysis = program.reference_analysis(0).unwrap();
        let root = analysis.roots().next().unwrap();
        assert!(analysis.is_mutated(root));
        assert_eq!(
            analysis.access_modes_for(root).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::AtomicAccumulate, ReferenceAccessMode::Consume],
        );
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(3.0_f32).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );
    }

    #[test]
    fn test_reference_atomic_add_update_provider() {
        let value_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(
            TestIrOperation::provide(TestIrReferenceAtomicAddUpdateOperation::new(), &[&reference_type, &value_type]),
            Ok(ArrayIrOperation::ReferenceAtomicAddUpdate(_)),
        ));
        assert!(matches!(
            TestIrOperation::provide(TestIrReferenceAtomicAddUpdateOperation::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ));
        assert!(matches!(
            TestIrOperation::provide(TestIrReferenceAtomicAddUpdateOperation::new(), &[&value_type, &value_type]),
            Err(ProgramError::Type(_)),
        ));
        assert!(matches!(
            AddOperation::<DataType>::provide(
                ReferenceAtomicAddUpdateOperation::new(),
                &[&DataType::F32, &DataType::F32],
            ),
            Err(ProgramError::UnsupportedOperation { .. }),
        ));
        let array_type = ArrayType::scalar(DataType::F32);
        assert!(matches!(
            ArrayOperation::<Array>::provide(ReferenceAtomicAddUpdateOperation::new(), &[&array_type, &array_type]),
            Err(ProgramError::UnsupportedOperation { .. }),
        ));
    }
}
