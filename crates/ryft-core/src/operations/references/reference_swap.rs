//! Generic read-write reference replacement operation and its value-level capability.

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
use crate::operations::references::reference_new::ReferenceNewOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClasses, Effects, MaybeZero, Operation, OperationProvider, ProgramError, ProjectedValue, ReferenceAccessMode,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceType, ReferenceViewOperation, RegionInterface, Type,
    TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceSwapOperation`].
pub const REFERENCE_SWAP_OPERATION_NAME: &str = "reference_swap";

/// Replaces a reference's stored value with an exactly matching referent and returns the old value.
#[derive(Clone, Debug)]
pub struct ReferenceSwapOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceSwapOperation<T, U> {
    /// Creates a new [`ReferenceSwapOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceSwapOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceSwapOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_SWAP_OPERATION_NAME)
    }
}

impl<T, U> Operation for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_SWAP_OPERATION_NAME
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
                "`{REFERENCE_SWAP_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
                Vec::new(),
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to replace")?;
        let replacement = inputs[1].try_as_value("a replacement value")?.clone();

        // The replacement must carry exactly the handle's referent. A universe whose write mechanics only require the
        // replacement to fit inside the selected coordinates would otherwise perform a silent partial write, so the
        // rule re-derives the operand relationship its own inference already states.
        validate_operand_types(self, inputs)?;
        Ok(vec![ReferenceDischargeValue::Value(context.swap(reference, replacement)?)])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceSwap<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].swap(&inputs[1])?])
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // The replacement is aligned with the reference's fixed batch axis before the packed swap, and the previous packed
    // value is batched at that same axis.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let replacement =
            align_stored_batch(context, driver, REFERENCE_SWAP_OPERATION_NAME, &inputs[0], inputs[1].clone())?;
        let previous = context
            .parent()
            .bind(*self, Vec::new(), &[P::value(&inputs[0]).clone(), P::value(&replacement).clone()])?
            .remove(0);
        Ok(vec![P::batch(previous, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<
            Type = U,
            Operation: From<ReferenceSwapOperation<T, U>> + ResidualZeroProvider<U, Operation = C::Operation>,
        > + Zero<C::Value>,
{
    // The tangent reference is swapped exactly as the primal reference is, so the returned previous value pairs with
    // the previous tangent contents. A plumbing reference returns its previous value with a symbolic zero tangent. The
    // tangent pairing is resolved before either swap so that a rejected plumbing store leaves both references
    // untouched.
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let stored = stored_tangents(REFERENCE_SWAP_OPERATION_NAME, &inputs[0], &inputs[1])?;
        let previous = context
            .primal()
            .bind(*self, Vec::new(), &[inputs[0].primal().clone(), inputs[1].primal().clone()])?
            .remove(0);
        Ok(vec![match stored {
            Some((tangent_reference, tangent)) => {
                // A zero replacement tangent is instantiated because the tangent reference must observe the store.
                let source = context.primal_to_tangent(inputs[1].primal().clone())?;
                let tangent = C::Operation::materialize_zero_from_residual_sources(
                    context.tangent(),
                    tangent,
                    std::iter::once(&source),
                )?;
                DifferentiationDual::new(
                    previous,
                    MaybeZero::Value(
                        context.tangent().bind(*self, Vec::new(), &[tangent_reference.clone(), tangent])?.remove(0),
                    ),
                )?
            }
            None => DifferentiationDual::new_with_zero_tangent(previous)?,
        }])
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U>
        + From<AddOperation<U>>
        + ResidualZeroProvider<U, Operation = O>
        + OperationProvider<U, ReferenceNewOperation<U, U>, Operation = O>
        + From<ReferenceSwapOperation<T, U>>,
{
    // A swap maps `(state, x) ↦ (x, state)`, so its transpose swaps the output cotangent into the cotangent reference
    // and yields the previous contents as the cotangent of the stored value. A zero output cotangent swapped into an
    // accumulator that nothing has reached yet leaves both zero, so nothing is staged and the stored value's cotangent
    // stays symbolic.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        let accumulator = match &outputs[0] {
            MaybeZero::Value(_) => context.cotangent_reference(driver, 0)?,
            MaybeZero::Zero(_) => match context.cotangent_reference_if_allocated(driver, 0)? {
                Some(accumulator) => accumulator,
                None => return Ok(()),
            },
        };
        let cotangent = O::materialize_zero_from_residual_sources(
            &**context,
            outputs[0].clone(),
            context
                .dimension_sources()
                .chain(inputs.iter().filter_map(PartialValue::as_known))
                .chain(std::iter::once(&accumulator)),
        )?;
        let previous = context.bind(*self, Vec::new(), &[accumulator, cotangent])?.remove(0);
        accumulators[1].accumulate(context, MaybeZero::Value(previous))
    }
}

/// Replaces the value stored by a reference in program order and returns its previous immutable snapshot.
pub trait ReferenceSwap<Replacement = Self, Output = Replacement>: Sized {
    /// Replaces the stored value in program order and returns the previously stored value.
    fn swap(&self, replacement: &Replacement) -> Result<Output, ProgramError>;
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ReferenceSwap for ArrayIrValue<A> {
    fn swap(&self, replacement: &Self) -> Result<Self, ProgramError> {
        ReferenceSwapOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        Ok(Self::Array(reference.swap(replacement)?))
    }
}

impl<V> ReferenceSwap<ProjectedValue<ArrayType, V>, <V as ValueProjection<ArrayType>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType>>,
{
    fn swap(
        &self,
        replacement: &ProjectedValue<ArrayType, V>,
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        self.value()
            .dispatch_domain()
            .bind(ReferenceSwapOperation::new(), Vec::new(), &[self.value().clone(), replacement.value().clone()])?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceSwap<V, V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType>>,
{
    fn swap(&self, replacement: &V) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceSwapOperation::new(), Vec::new(), &[self.clone(), replacement.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayReference, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
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
    type TestIrSwap = ReferenceSwapOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_swap() {
        let operation = Swap::new();
        assert_eq!(operation.name(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            "ReferenceSwapOperation(PhantomData<fn() -> (ryft_core::operations::references::tests::TestReferent, \
             ryft_core::operations::references::tests::TestType)>)",
        );

        // A swap orders against other state effects, reads and writes its reference operand, and aliases nothing.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }]
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_swap_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Swap::new(),
            cases = [
                {
                    input_types = [reference.clone(), value.clone()],
                    output_types = [value.clone()],
                },
                {
                    input_types = [reference.clone(), TestType::Value(TestReferent::new(7, 32))],
                    error = "`reference_swap` replacement type `value<i7,p32>` must exactly match reference referent \
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
            Swap::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // A universe must project references and values out of itself and embed the previous value back into itself.
        check_operation_type_inference!(
            operation = ReferenceSwapOperation::<TestReferent, SwapUniverse>::new(),
            cases = [{
                input_types = [SwapUniverse::Reference(ReferenceType::new(referent)), SwapUniverse::Value(referent)],
                output_types = [SwapUniverse::Value(referent)],
            }],
        );

        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrSwap::new(),
            cases = [
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(vector_type.clone()),
                    ],
                    output_types = [ArrayIrType::Array(vector_type.clone())],
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                    ],
                    error = "`reference_swap` replacement type `f32[3]` must exactly match reference referent type \
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
    fn test_reference_swap_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(live.clone());

        // A replacement carrying exactly the referent type replaces the stored value and returns the previous value.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrSwap::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())],
            ),
            Ok(vec![TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Exact operand inference runs before the swap, so a rejected replacement leaves the stored value unchanged.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrSwap::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0, 7.0]).unwrap())],
            ),
            Err(TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Each operand must be the member kind the operation expects.
        let array = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap());
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrSwap::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[array.clone(), array],
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrSwap::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), reference],
            ),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Equality over a dynamically typed `Array` cannot address its elements, so a dynamically typed referent is
        // compared by its declared type and its exact physical storage. Both referents come from the test-only
        // unchecked hatch because the checked constructors reject dynamically shaped types, and they declare exactly
        // the same dynamic type, so the swap is observable only through the payloads it returns and installs.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let replacement_bytes = 2.0_f32.to_le_bytes().to_vec();
        let live = ArrayReference::new(Array::with_unchecked_type(dynamic_type.clone(), initial_bytes.clone()));
        let replacement = Array::with_unchecked_type(dynamic_type.clone(), replacement_bytes.clone());
        let mut outputs = InterpretableOperation::<TestIrContext>::interpret(
            &TestIrSwap::new(),
            &TestIrContext::new(),
            &EmptyRegionDriver,
            &[TestIrValue::Reference(live.clone()), TestIrValue::Array(replacement)],
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        let previous = <TestIrValue as ValueProjection<ArrayType>>::into_projected(outputs.remove(0)).unwrap();
        assert_eq!(previous.r#type().into_owned(), dynamic_type);
        assert_eq!(previous.storage_bytes(), initial_bytes.as_slice());
        let installed = live.read().unwrap();
        assert_eq!(installed.r#type().into_owned(), dynamic_type);
        assert_eq!(installed.storage_bytes(), replacement_bytes.as_slice());
    }

    #[test]
    fn test_reference_swap_partial_evaluation() {
        // Program replay uses the `Stage` placement: a swap stages regardless of operand knowledge, its previous value
        // is an unknown of the residual program, the live handle is passed to that program as a known reference input,
        // and the swap runs only when that program runs.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrSwap::new(),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2.0_f32).unwrap())),
                    ],
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(1.0_f32).unwrap()))],
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
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(2.0_f32).unwrap()))],
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
        let swapped = staging
            .fold_or_residualize(TestIrSwap::new(), Vec::new(), &[reference.clone(), replacement.clone()])
            .unwrap();
        assert_eq!(swapped.len(), 1);
        assert!(swapped[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the default `Execute` placement an all-known swap folds: it runs against the live state in program
        // order at partial evaluation time and its previous value is known.
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let swapped = executing.fold_or_residualize(TestIrSwap::new(), Vec::new(), &[reference, replacement]).unwrap();
        assert_eq!(swapped.len(), 1);
        assert_eq!(swapped[0].as_known(), Some(&TestIrValue::Array(Array::scalar(3.0_f32).unwrap())));
        assert_eq!(live.read(), Ok(Array::scalar(4.0_f32).unwrap()));
    }

    #[test]
    fn test_reference_swap_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let initial = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());

        // A replacement mapped at the reference's batch axis is swapped packed, and the previous packed value is
        // batched at the reference's axis.
        let aligned = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
        );
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(aligned.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[batched, replacement]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].batch().value(), &initial);
        assert_eq!(reference.read(), Ok(aligned));

        // A batched replacement cannot be swapped into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[0.0; 6]).unwrap()),
                BatchAxis::new(1),
            )
            .unwrap(),
        );
        let error = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[replicated, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_swap` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_swap_differentiation() {
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Swapping an active reference swaps its tangent reference alongside, so the previous value pairs with the
        // previous tangent contents.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            outputs[0].tangent().as_value(),
            Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // Swapping a plumbing reference with a replacement without a live tangent yields a symbolic zero tangent.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        let outputs =
            context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayType::new_static(DataType::F32, [2]).into(),
        ));
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal swap.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![13.0_f32, 14.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message: "`reference_swap` writes a live tangent into a reference that carries no tangent; pass the \
                          reference as a differentiated input instead of capturing it"
                    .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
    }

    #[test]
    fn test_reference_swap_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // The swap transposes through the cotangent accumulator of its reference operand, which only a transposition
        // context scoped to the instruction being transposed can resolve, so a detached context rejects it.
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(scalar_type.clone())];
        let tracing = TracingContext::<TestIrValue, TestIrOperation>::new();
        let cotangent = tracing.input(scalar_type.clone());
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrSwap::new().transpose(
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

        // `r = new(v); y = swap(r, x); z = freeze(r)`: the freeze lands `z̄` in the allocation's accumulator, the swap
        // swaps `ȳ` into it and hands its previous contents to `x̄ = z̄`, and the allocation freezes the accumulator
        // into `v̄ = ȳ`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = zero [type=f32[]]
                    %3:ref<f32[]> = reference_new %2
                    reference_add_update %3 %1
                    %4:f32[] = reference_swap %3 %0
                    %5:f32[] = reference_freeze %3
                in (%5, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![
                TestIrValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );

        // A dead swap result whose accumulator a later freeze allocated instantiates its zero cotangent, because the
        // accumulator must observe the swap: `x̄ = z̄` and `v̄ = 0`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
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

        // A dead swap result with an accumulator that nothing reached stages nothing and leaves both cotangents
        // symbolic zeros.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
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
    fn test_reference_swap_reference_discharge() {
        // A replacement returns the previous state and commits the successor, which marks the allocation mutated.
        let (context, reference) = allocated_reference(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // The replacement itself must be a value rather than a second reference handle.
        let handles =
            vec![ReferenceDischargeValue::Reference(reference.clone()), ReferenceDischargeValue::Reference(reference)];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, handles.as_slice()),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a replacement value but received {}",
                handles[1],
            ))),
        );
    }

    #[test]
    fn test_reference_swap_projected() {
        type TestTracingContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestTracingContext>;

        // A projected reference binds the swap through its parent tracer's context and projects the previous value,
        // so the projected surface stages exactly the native operation.
        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestTracingContext::trace(
            |inputs| {
                let [initial, replacement]: [TestTracer; 2] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                let previous = reference.swap(&replacement)?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &previous;
                let frozen = reference.freeze()?;
                Ok(vec![previous.into_value(), frozen.into_value()])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:ref<f32[2]> = reference_new %0
                    %3:f32[2] = reference_swap %2 %1
                    %4:f32[2] = reference_freeze %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_swap_staging() {
        // A traced reference stages the swap as the native variant of its operation family, with the previous value
        // as its result and the ordered-state effect of the program.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let (output_types, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| {
                let [reference, replacement]: [Tracer<_>; 2] = inputs.try_into().unwrap();
                Ok(vec![reference.swap(&replacement)?])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
                ArrayIrType::Array(array_type.clone()),
            ],
        )
        .unwrap();
        assert_eq!(output_types, vec![ArrayIrType::Array(array_type)]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let %2:f32[2] = reference_swap %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
