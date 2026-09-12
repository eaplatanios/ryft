//! Generic write-only reference replacement operation and its value-level capability.

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
use crate::operations::constants::zero::Zero;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
use crate::operations::math::add::AddOperation;
use crate::operations::references::reference_swap::ReferenceSwapOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClasses, Effects, MaybeZero, Operation, ProgramError, ProjectedValue, ReferenceAccessMode,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceType, ReferenceViewOperation, RegionInterface, Type,
    TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceWriteOperation`].
pub const REFERENCE_WRITE_OPERATION_NAME: &str = "reference_write";

static REFERENCE_WRITE_OPERATION_EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
    Effects::new(
        EffectClasses::NONE,
        vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
        Vec::new(),
    )
    .unwrap()
});

/// Replaces a reference's stored value with an exactly matching referent without observing the old value.
#[derive(Clone, Debug)]
pub struct ReferenceWriteOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceWriteOperation<T, U> {
    /// Creates a new [`ReferenceWriteOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceWriteOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceWriteOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_WRITE_OPERATION_NAME)
    }
}

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

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
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

impl<T, U, C> DifferentiableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Context<
            Type = U,
            Operation: From<ReferenceWriteOperation<T, U>> + ResidualZeroProvider<U, Operation = C::Operation>,
        > + Zero<C::Value>,
{
    // The replacement's tangent is stored into the tangent reference exactly as the primal replacement is stored into
    // the primal reference. The tangent pairing is resolved before either store so that a rejected plumbing store
    // leaves both references untouched.
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let stored = stored_tangents(REFERENCE_WRITE_OPERATION_NAME, &inputs[0], &inputs[1])?;
        context
            .primal()
            .bind(*self, Vec::new(), &[inputs[0].primal().clone(), inputs[1].primal().clone()])?;
        if let Some((tangent_reference, tangent)) = stored {
            // A zero replacement tangent is instantiated because the tangent reference must observe the store.
            let source = context.primal_to_tangent(inputs[1].primal().clone())?;
            context.tangent().bind(
                *self,
                Vec::new(),
                &[
                    tangent_reference.clone(),
                    C::Operation::materialize_zero_from_residual_sources(
                        context.tangent(),
                        tangent,
                        std::iter::once(&source),
                    )?,
                ],
            )?;
        }
        Ok(Vec::new())
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U>
        + From<AddOperation<U>>
        + ResidualZeroProvider<U, Operation = O>
        + From<ReferenceSwapOperation<T, U>>,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
{
    // A write is a swap whose previous contents are discarded, so its transpose is the swap transpose with a zero
    // output cotangent: the cotangent reference is reset to zero (the pre-execution state no longer flows into
    // anything) and its previous contents become the cotangent of the stored value. An accumulator that nothing has
    // reached yet already holds zero, so nothing is staged and the stored value's cotangent stays symbolic.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 0, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        let value_cotangent_type = inputs[1].r#type().cotangent()?;
        let Some(accumulator) = context.cotangent_reference_if_allocated(driver, 0)? else {
            return Ok(());
        };
        let zero = O::materialize_zero_from_residual_sources(
            &**context,
            MaybeZero::Zero(value_cotangent_type),
            context
                .dimension_sources()
                .chain(inputs.iter().filter_map(PartialValue::as_known))
                .chain(std::iter::once(&accumulator)),
        )?;
        let previous = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[accumulator, zero])?.remove(0);
        accumulators[1].accumulate(context, MaybeZero::Value(previous))
    }
}

/// Replaces the value stored by a reference without observing the previous value.
pub trait ReferenceWrite<Replacement = Self>: Sized {
    /// Replaces the stored value with `replacement` in program order.
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError>;
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ReferenceWrite for ArrayIrValue<A> {
    fn write(&self, replacement: &Self) -> Result<(), ProgramError> {
        ReferenceWriteOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        reference.write(replacement)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceWrite<ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType>>,
{
    fn write(&self, replacement: &ProjectedValue<ArrayType, V>) -> Result<(), ProgramError> {
        self.value().dispatch_domain().bind(
            ReferenceWriteOperation::new(),
            Vec::new(),
            &[self.value().clone(), replacement.value().clone()],
        )?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceWrite<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType>>,
{
    fn write(&self, replacement: &V) -> Result<(), ProgramError> {
        self.dispatch_domain().bind(
            ReferenceWriteOperation::new(),
            Vec::new(),
            &[self.clone(), replacement.clone()],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayReference, ArrayType,
        DataType, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_freeze::{ReferenceFreeze, ReferenceFreezeOperation};
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrContext = EagerContext<TestIrValue, TestIrOperation>;
    type TestIrWrite = ReferenceWriteOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_write() {
        let operation = Write::new();
        assert_eq!(operation.name(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            "ReferenceWriteOperation(PhantomData<fn() -> (ryft_core::operations::references::tests::TestReferent, \
             ryft_core::operations::references::tests::TestType)>)",
        );

        // A write orders against other state effects, accesses its reference operand for writing, and aliases nothing.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }]
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_write_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Write::new(),
            cases = [
                {
                    input_types = [reference.clone(), value.clone()],
                    output_types = [],
                },
                {
                    input_types = [reference.clone(), TestType::Value(TestReferent::new(7, 32))],
                    error = "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent \
                             type `value<i7,p16>`",
                },
                {
                    input_types = [reference.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [value.clone(), value.clone()],
                    error = "expected reference type but got value type",
                },
                {
                    input_types = [reference.clone(), reference.clone()],
                    error = "expected value type but got reference type",
                },
            ],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Write::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // A universe only needs to project references and values out of itself for a write to type-check.
        check_operation_type_inference!(
            operation = ReferenceWriteOperation::<TestReferent, WriteUniverse>::new(),
            cases = [{
                input_types = [WriteUniverse::Reference(ReferenceType::new(referent)), WriteUniverse::Value(referent)],
                output_types = [],
            }],
        );

        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrWrite::new(),
            cases = [
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(vector_type.clone()),
                    ],
                    output_types = [],
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                    ],
                    error = "`reference_write` replacement type `f32[3]` must exactly match reference referent type \
                             `f32[2]`",
                },
                {
                    input_types = [ArrayIrType::Array(vector_type.clone()), ArrayIrType::Array(vector_type.clone())],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Reference(ReferenceType::new(vector_type)),
                    ],
                    error = "expected array type but got reference type",
                },
            ],
        );
    }

    #[test]
    fn test_reference_write_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(live.clone());

        // A replacement carrying exactly the referent type replaces the stored value and produces no output.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrWrite::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Exact operand inference runs before the store, so a rejected replacement leaves the stored value unchanged.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrWrite::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0, 7.0]).unwrap())],
            ),
            Err(TypeError::invalid(
                "`reference_write` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Each operand must be the member kind the operation expects.
        let array = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap());
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrWrite::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[array.clone(), array],
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrWrite::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), reference],
            ),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));
    }

    #[test]
    fn test_reference_write_partial_evaluation() {
        // Program replay uses the `Stage` placement: a write stages regardless of operand knowledge, the live handle
        // is passed to the residual program as a known reference input, and the store runs only when that program
        // runs.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrWrite::new(),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2.0_f32).unwrap())),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@unknown(
                            type = ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                            replay = TestIrValue::Array(Array::scalar(3.0_f32).unwrap())
                        )),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the `Stage` placement the live state is untouched at partial evaluation time even when every operand
        // is known.
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));
        let replacement = PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(4.0_f32).unwrap()));
        let staging =
            PartialEvaluationContext::new(TestIrContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs =
            staging.fold_or_residualize(TestIrWrite::new(), Vec::new(), &[reference.clone(), replacement.clone()]);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the default `Execute` placement an all-known write folds: it runs against the live state in program
        // order at partial evaluation time.
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let outputs = executing.fold_or_residualize(TestIrWrite::new(), Vec::new(), &[reference, replacement]);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(4.0_f32).unwrap()));
    }

    #[test]
    fn test_reference_write_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestIrValue::Array(Array::from_elements::<f32>(packed_type.clone(), &[0.0; 6]).unwrap())
            .reference_new()
            .unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // A replacement mapped at the reference's batch axis is stored packed.
        let aligned = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(aligned.clone(), BatchAxis::new(0)).unwrap());
        let outputs =
            context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(aligned));

        // A replicated replacement is broadcast along the reference's batch axis.
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]).unwrap())),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[7.0, 8.0, 9.0, 7.0, 8.0, 9.0]).unwrap()
            )),
        );

        // A replacement mapped at another axis is moved to the reference's batch axis.
        let transposed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let transposed =
            TestIrValue::Array(Array::from_elements::<f32>(transposed_type, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(transposed, BatchAxis::new(1)).unwrap());
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched, replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
            )),
        );

        // A replicated replacement is stored plainly into a replicated reference, while a batched replacement has no
        // batch axis to land in and the user is told to batch the reference instead.
        let unbatched = TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]).unwrap()).reference_new().unwrap();
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(unbatched.clone()));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())),
        );
        context
            .bind(ReferenceWriteOperation::new(), Vec::new(), &[replicated.clone(), replacement])
            .unwrap();
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[0.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
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
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));
    }

    #[test]
    fn test_reference_write_differentiation() {
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // A live replacement tangent is written into the tangent reference alongside the primal replacement.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // A symbolic zero replacement tangent is instantiated so that the tangent reference observes the store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())));

        // A plumbing reference accepts a replacement without a live tangent and records no tangent store.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(
                Array::vector(vec![11.0_f32, 12.0]).unwrap(),
            ))
            .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![13.0_f32, 14.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![15.0_f32, 16.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message: "`reference_write` writes a live tangent into a reference that carries no tangent; pass the \
                          reference as a differentiated input instead of capturing it"
                    .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap())));
    }

    #[test]
    fn test_reference_write_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // The store transposes through the cotangent accumulator of its reference operand, which only a transposition
        // context scoped to the instruction being transposed can resolve, so a detached context rejects it.
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(scalar_type.clone())];
        let mut context = TranspositionContext::new(TracingContext::<TestIrValue, TestIrOperation>::new());
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrWrite::new().transpose(&mut context, &EmptyRegionDriver, &inputs, &[], &accumulators),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));

        // `r = new(v); write(r, x); y = freeze(r)`: the freeze lands `ȳ` in the allocation's accumulator, the write
        // swaps a zero into it (the pre-execution state no longer flows anywhere) and hands its previous contents to
        // `x̄`, and the allocation freezes the cleared accumulator into `v̄ = 0`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                    %2:ref<f32[]> = reference_new %1
                    reference_add_update %2 %0
                    %3:f32[] = zero [type=f32[]]
                    %4:f32[] = reference_swap %2 %3
                    %5:f32[] = reference_freeze %2
                in (%5, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );

        // `r = new(v); write(r, x)` with no later access leaves the allocation's accumulator unallocated, so the write
        // stages nothing and both cotangents stay symbolic zeros.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::<Placeholder>::new())
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[] = zero [type=f32[]]
                    %1:f32[] = zero [type=f32[]]
                in (%0, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(Vec::new()),
            Ok(vec![
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_reference_write_reference_discharge() {
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
    fn test_reference_write_projected() {
        type TestTracingContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestTracingContext>;

        // A projected reference binds the write through its parent tracer's context, so the projected surface stages
        // exactly the native operation.
        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestTracingContext::trace(
            |inputs| {
                let [initial, replacement]: [TestTracer; 2] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                reference.write(&replacement)?;
                let frozen = reference.freeze()?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &frozen;
                Ok(vec![frozen.into_value()])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:ref<f32[2]> = reference_new %0
                    reference_write %2 %1
                    %3:f32[2] = reference_freeze %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_write_staging() {
        // A traced reference stages the write as the native variant of its operation family, with no result and the
        // ordered-state effect of the program.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let (output_types, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| {
                let [reference, replacement]: [Tracer<_>; 2] = inputs.try_into().unwrap();
                reference.write(&replacement)?;
                Ok(vec![reference])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
                ArrayIrType::Array(array_type.clone()),
            ],
        )
        .unwrap();
        assert_eq!(output_types, vec![ArrayIrType::Reference(ReferenceType::new(array_type))]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let reference_write %0 %1
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
