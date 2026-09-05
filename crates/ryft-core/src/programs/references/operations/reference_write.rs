//! Generic write-only reference replacement operation and its value-level capability.

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
use crate::operations::Zero;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceEffect};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::ReferenceSwapOperation;

use crate::programs::references::types::ReferenceType;
use crate::programs::references::views::ReferenceViewOperation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceWriteOperation`].
pub const REFERENCE_WRITE_OPERATION_NAME: &str = "reference_write";

/// Replaces the value stored by a reference without observing the previous value.
pub trait ReferenceWrite<Replacement = Self>: Sized {
    /// Replaces the stored value with `replacement` in program order.
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError>;
}

static REFERENCE_WRITE_OPERATION_EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
    Effects::new(
        EffectClasses::NONE,
        vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
        Vec::new(),
    )
    .unwrap()
});

define_reference_primitive_payload!(
    /// Replaces a reference's stored value with an exactly matching referent without observing the old value.
    ReferenceWriteOperation
);

impl_reference_primitive_display!(ReferenceWriteOperation, REFERENCE_WRITE_OPERATION_NAME);

impl<T, U> Operation for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_WRITE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let replacement = <&T>::try_from(&input_types[1])?;
        if replacement != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_WRITE_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Borrowed(&REFERENCE_WRITE_OPERATION_EFFECTS)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceWrite<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].write(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to write")?;
        let replacement = inputs[1].try_as_value("a replacement value")?.clone();
        validate_operand_types(self, inputs)?;
        context.write(reference, replacement)?;
        Ok(Vec::new())
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>> + ResidualZeroProvider<U>> + Zero<C::Value>,
{
    // The replacement's tangent is stored into the tangent reference exactly as the primal replacement is stored into
    // the primal reference. The tangent pairing is resolved before either store so that a rejected plumbing store
    // leaves both references untouched.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let stored = stored_tangents(REFERENCE_WRITE_OPERATION_NAME, &inputs[0], &inputs[1])?;
        context.bind(*self, Vec::new(), &[inputs[0].primal().clone(), inputs[1].primal().clone()])?;
        if let Some((tangent_reference, tangent)) = stored {
            // A zero replacement tangent is instantiated because the tangent reference must observe the store.
            context.bind(
                *self,
                Vec::new(),
                &[
                    tangent_reference.clone(),
                    C::Operation::materialize_zero_from_residual_sources(
                        context,
                        tangent,
                        std::iter::once(inputs[1].primal()),
                    )?,
                ],
            )?;
        }
        Ok(Vec::new())
    }
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // The replacement is aligned with the reference's fixed batch axis before the packed store.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let replacement =
            align_stored_batch(context, driver, REFERENCE_WRITE_OPERATION_NAME, &inputs[0], inputs[1].clone())?;
        context
            .parent()
            .bind(*self, Vec::new(), &[P::value(&inputs[0]).clone(), P::value(&replacement).clone()])?;
        Ok(Vec::new().into())
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U> + ResidualZeroProvider<U> + From<ReferenceSwapOperation<T, U>>,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
{
    // A write is a swap whose previous contents are discarded, so its transpose is the swap transpose with a zero
    // output cotangent: the cotangent reference is reset to zero (the pre-execution state no longer flows into
    // anything) and its previous contents become the cotangent of the stored value. An accumulator that nothing has
    // reached yet already holds zero, so nothing is staged and the stored value's cotangent stays symbolic.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 0, ProgramError);
        let reference_cotangent = MaybeZero::Zero(inputs[0].r#type().cotangent()?);
        let value_cotangent_type = inputs[1].r#type().cotangent()?;
        let Some(accumulator) = context.cotangent_reference_if_allocated(0)? else {
            return Ok(vec![reference_cotangent, MaybeZero::Zero(value_cotangent_type)]);
        };
        let zero = O::materialize_zero_from_residual_sources(
            &**context,
            MaybeZero::Zero(value_cotangent_type),
            context
                .geometry_sources()
                .chain(inputs.iter().filter_map(PartialValue::as_known))
                .chain(std::iter::once(&accumulator)),
        )?;
        let previous = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[accumulator, zero])?.remove(0);
        Ok(vec![reference_cotangent, MaybeZero::Value(previous)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::programs::effects::EffectClass;
    use crate::programs::references::operations::tests::*;
    use crate::programs::references::operations::{ReferenceNew, ReferenceRead};
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;

    #[test]
    fn test_reference_write_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Write::new());
        assert_eq!(Write::new().to_string(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(Write::new().effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            Write::new().effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }]
        );
        assert_eq!(Write::new().effects().reference_aliases(), &[]);

        assert_eq!(
            ReferenceWriteOperation::<TestReferent, WriteUniverse>::new().infer_output_types(
                &[WriteUniverse::Reference(ReferenceType::new(referent)), WriteUniverse::Value(referent)],
                &[],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(Write::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(Vec::new()));
        assert_eq!(
            Write::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent type \
                 `value<i7,p16>`",
            )),
        );
        assert_eq!(
            Write::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            Write::new().infer_output_types(&[value.clone(), value.clone()], &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            Write::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Write::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_write_operation_reference_discharge() {
        // A policy with no accumulation capability replaces state through `write`, produces no old-value result, and
        // marks the allocation mutated. Its `swap` path is an error, making accidental swap dispatch visible.
        let context =
            ReferenceDischargeContext::<TestDestination, WriteOnlyReferenceDischarge>::new(TestDestination::new());
        let initial = TestValue::new(REFERENT, 4);
        let allocated =
            ReferenceDischargeValue::from(context.bind_discharged(ReferenceType::new(REFERENT), initial).unwrap());
        let reference = allocated.try_as_reference("the allocated reference").unwrap().clone();
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(Write::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()), Ok(Vec::new()),);
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // Exact operand inference runs before mutation, so a rejected replacement leaves the allocation unchanged.
        let invalid = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            Write::new().discharge_references(&context, &EmptyRegionDriver, invalid.as_slice()),
            Err(TypeError::invalid(
                "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent type \
             `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
    }

    #[test]
    fn test_reference_write_operation_jvp() {
        let context = DifferentiationContext::new(EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0])).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // A live replacement tangent is written into the tangent reference alongside the primal replacement.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0])),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]))));

        // A symbolic zero replacement tangent is instantiated so that the tangent reference observes the store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]))).unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]))));

        // A plumbing reference accepts a replacement without a live tangent and records no tangent store.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0])))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]))));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![13.0_f32, 14.0])),
                TestIrValue::Array(Array::vector(vec![15.0_f32, 16.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DifferentiationError>(),
            Some(&DifferentiationError::PlumbingReferenceTangent { operation: REFERENCE_WRITE_OPERATION_NAME }),
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]))));
    }

    #[test]
    fn test_reference_write_operation_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<TestIrValue, ArrayIrOperation<Array>>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference =
            TestIrValue::Array(Array::from_f64s(packed_type.clone(), vec![0.0; 6])).reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // A replacement mapped at the reference's batch axis is stored packed.
        let aligned = TestIrValue::Array(Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(aligned.clone(), BatchAxis::new(0)).unwrap());
        let outputs =
            context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(aligned));

        // A replicated replacement is broadcast along the reference's batch axis.
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]))),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::from_f64s(packed_type.clone(), vec![7.0, 8.0, 9.0, 7.0, 8.0, 9.0]))),
        );

        // A replacement mapped at another axis is moved to the reference's batch axis.
        let transposed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let transposed = TestIrValue::Array(Array::from_f64s(transposed_type, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(transposed, BatchAxis::new(1)).unwrap());
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched, replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))),
        );

        // A replicated replacement is stored plainly into a replicated reference, while a batched replacement has no
        // batch axis to land in and the user is told to batch the reference instead.
        let unbatched = TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])).reference_new().unwrap();
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(unbatched.clone()));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
        );
        context
            .bind(ReferenceWriteOperation::new(), Vec::new(), &[replicated.clone(), replacement])
            .unwrap();
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(TestIrValue::Array(Array::from_f64s(packed_type, vec![0.0; 6])), BatchAxis::new(0))
                .unwrap(),
        );
        let error = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[replicated, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_write` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));
    }
}
