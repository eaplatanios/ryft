//! Generic read-write reference replacement operation and its value-level capability.

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
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::ReferenceNew;
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
use crate::programs::references::types::ReferenceType;
use crate::programs::references::views::ReferenceViewOperation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceSwapOperation`].
pub const REFERENCE_SWAP_OPERATION_NAME: &str = "reference_swap";

/// Replaces the value stored by a reference in program order and returns its previous immutable snapshot.
pub trait ReferenceSwap<Replacement = Self, Output = Replacement>: Sized {
    /// Replaces the stored value in program order and returns the previously stored value.
    fn swap(&self, replacement: &Replacement) -> Result<Output, ProgramError>;
}

static REFERENCE_SWAP_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)], Vec::new())
});

define_reference_primitive_payload!(
    /// Replaces a reference's stored value with an exactly matching referent and returns the old value.
    ReferenceSwapOperation
);

impl_reference_primitive_display!(ReferenceSwapOperation, REFERENCE_SWAP_OPERATION_NAME);

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
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_SWAP_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
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

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceSwap<C::Value>> + Zero<C::Value>,
{
    // The tangent reference is swapped exactly as the primal reference is, so the returned previous value pairs with
    // the previous tangent contents. A plumbing reference returns its previous value with a symbolic zero tangent. The
    // tangent pairing is resolved before either swap so that a rejected plumbing store leaves both references
    // untouched.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let stored = stored_tangents(REFERENCE_SWAP_OPERATION_NAME, &inputs[0], &inputs[1])?;
        let previous = inputs[0].primal().swap(inputs[1].primal())?;
        Ok(vec![match stored {
            Some((tangent_reference, tangent)) => {
                // A zero replacement tangent is instantiated because the tangent reference must observe the store.
                let tangent = tangent.materialize(context)?;
                DifferentiationDual::new(previous, MaybeZero::Value(tangent_reference.swap(&tangent)?))?
            }
            None => DifferentiationDual::new_with_zero_tangent(previous)?,
        }])
    }
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceSwap<C::Value>>,
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
        let previous = P::value(&inputs[0]).swap(P::value(&replacement))?;
        Ok(vec![P::batch(previous, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U> + ResidualZeroProvider<U>,
    Tracer<TracingContext<V, O>>: ReferenceNew<Tracer<TracingContext<V, O>>>
        + ReferenceSwap<Tracer<TracingContext<V, O>>, Tracer<TracingContext<V, O>>>,
{
    // A swap maps `(state, x) ↦ (x, state)`, so its transpose swaps the output cotangent into the cotangent reference
    // and yields the previous contents as the cotangent of the stored value. A zero output cotangent swapped into an
    // accumulator that nothing has reached yet leaves both zero, so nothing is staged and the stored value's cotangent
    // stays symbolic.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let reference_cotangent = MaybeZero::Zero(inputs[0].r#type().cotangent()?);
        let accumulator = match &outputs[0] {
            MaybeZero::Value(_) => context.cotangent_reference(0)?,
            MaybeZero::Zero(_) => match context.cotangent_reference_if_allocated(0)? {
                Some(accumulator) => accumulator,
                None => return Ok(vec![reference_cotangent, MaybeZero::Zero(inputs[1].r#type().cotangent()?)]),
            },
        };
        let cotangent = O::materialize_zero_from_residual_sources(
            &**context,
            outputs[0].clone(),
            inputs.iter().filter_map(PartialValue::as_known),
        )?;
        let previous = accumulator.swap(&cotangent)?;
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
    use crate::programs::references::operations::tests::*;
    use crate::programs::references::operations::{ReferenceNew, ReferenceRead};
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_reference_swap_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Swap::new());
        assert_eq!(Swap::new().to_string(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(Swap::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Swap::new().reference_semantics().outputs(), &[]);
        assert_eq!(
            Swap::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)],
        );

        assert_eq!(
            ReferenceSwapOperation::<TestReferent, SwapUniverse>::new().infer_output_types(
                &[SwapUniverse::Reference(ReferenceType::new(referent)), SwapUniverse::Value(referent)],
                &[],
            ),
            Ok(vec![SwapUniverse::Value(referent)]),
        );
        assert_eq!(Swap::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(vec![value.clone()]),);
        assert_eq!(
            Swap::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_swap` replacement type `value<i7,p32>` must exactly match reference referent type \
                 `value<i7,p16>`",
            )),
        );
        assert_eq!(
            Swap::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            Swap::new().infer_output_types(&[value.clone(), value.clone()], &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            Swap::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            Swap::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_swap_operation_reference_discharge() {
        // A replacement returns the previous state and commits the successor, which marks the allocation mutated.
        let (context, reference) = allocated_allocation(4);
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
    fn test_reference_swap_operation_jvp() {
        type TestValue = ArrayIrValue<Array>;

        let context = DifferentiationContext::new(EagerContext::<TestValue, ArrayIrOperation<Array>>::new());
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![3.0_f32, 4.0])).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Swapping an active reference swaps its tangent reference alongside, so the previous value pairs with the
        // previous tangent contents.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![5.0_f32, 6.0])),
                TestValue::Array(Array::vector(vec![7.0_f32, 8.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestValue::Array(Array::vector(vec![1.0_f32, 2.0])));
        assert_eq!(outputs[0].tangent().as_value(), Some(&TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))));
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 8.0]))));

        // Swapping a plumbing reference with a replacement without a live tangent yields a symbolic zero tangent.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::vector(vec![9.0_f32, 10.0]))).unwrap(),
            context.clone(),
        );
        let outputs =
            context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(outputs[0].primal(), &TestValue::Array(Array::vector(vec![5.0_f32, 6.0])));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayType::new_static(DataType::F32, [2]).into(),
        ));
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![9.0_f32, 10.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 8.0]))));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal swap.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![11.0_f32, 12.0])),
                TestValue::Array(Array::vector(vec![13.0_f32, 14.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DifferentiationError>(),
            Some(&DifferentiationError::PlumbingReferenceTangent { operation: REFERENCE_SWAP_OPERATION_NAME }),
        );
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![9.0_f32, 10.0]))));
    }

    #[test]
    fn test_reference_swap_operation_batching() {
        type TestValue = ArrayIrValue<Array>;

        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<TestValue, ArrayIrOperation<Array>>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let initial = TestValue::Array(Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());

        // A replacement mapped at the reference's batch axis is swapped packed, and the previous packed value is
        // batched at the reference's axis.
        let aligned = TestValue::Array(Array::from_f64s(packed_type.clone(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
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
            ArrayIrBatch::new(TestValue::Array(Array::from_f64s(packed_type, vec![0.0; 6])), BatchAxis::new(1))
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
}
