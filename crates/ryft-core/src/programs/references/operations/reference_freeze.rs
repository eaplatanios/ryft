//! Generic consuming reference finalization operation and its value-level capability.

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
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::{ReferenceAddUpdate, ReferenceNew};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
use crate::programs::references::types::ReferenceType;
use crate::programs::references::views::ReferenceViewOperation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

use super::forwarded_tangent;

/// Canonical operation name for [`ReferenceFreezeOperation`].
pub const REFERENCE_FREEZE_OPERATION_NAME: &str = "reference_freeze";

/// Consumes a reference, returning its final value and invalidating its complete alias family.
pub trait ReferenceFreeze<Output = Self>: Sized {
    /// Returns the final stored value and invalidates this reference and all aliases.
    ///
    /// The handle is taken by value, because consumption is linear: after this call the reference denotes nothing.
    /// Passing it by value makes the common single-handle misuse — freezing and then reading through the same
    /// binding — a compile error rather than a runtime one. Aliases obtained by cloning the handle are a different
    /// case and remain a dynamic failure, because the type system cannot see them: an eager alias fails at its next
    /// access against the shared reference state, and a staged alias fails while tracing, because every clone of one
    /// [`Tracer`](crate::Tracer) names the same staged atom. Freezing through a shared borrow is therefore an
    /// explicit clone-then-freeze, which reads as the deliberate act it is.
    ///
    /// ```compile_fail
    /// use ryft_core::{Array, ArrayIrValue, ReferenceFreeze, ReferenceNew, ReferenceRead};
    ///
    /// let allocation = ArrayIrValue::Array(Array::scalar(1.0_f32)).reference_new()?;
    /// let frozen = allocation.freeze()?;
    /// // The handle was consumed, so reading it again does not compile.
    /// let stale = allocation.read()?;
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    ///
    /// ```
    /// use ryft_core::{Array, ArrayIrValue, ReferenceFreeze, ReferenceNew, ReferenceError, ReferenceRead};
    ///
    /// // A clone is a separate handle onto the same reference allocation, so misuse is caught dynamically instead.
    /// let allocation = ArrayIrValue::Array(Array::scalar(1.0_f32)).reference_new()?;
    /// let alias = allocation.clone();
    /// assert_eq!(allocation.freeze()?, ArrayIrValue::Array(Array::scalar(1.0_f32)));
    /// assert_eq!(
    ///     alias.read().unwrap_err().downcast_custom::<ReferenceError>(),
    ///     Some(&ReferenceError::Frozen),
    /// );
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    fn freeze(self) -> Result<Output, ProgramError>;
}

static REFERENCE_FREEZE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)], Vec::new())
});

define_reference_primitive_payload!(
    /// Consumes an allocation reference, returning its final referent and invalidating its complete alias family.
    ReferenceFreezeOperation
);

impl_reference_primitive_display!(ReferenceFreezeOperation, REFERENCE_FREEZE_OPERATION_NAME);

impl<T, U> Operation for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_FREEZE_OPERATION_NAME
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
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_FREEZE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceFreeze<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);

        // Interpretation replays an already-built instruction, so the operand is borrowed from the environment rather
        // than owned, and cloning it is the faithful replay: a clone names the same allocation, so consuming it
        // invalidates the whole alias family exactly as the source program asked. The linearity the value-level
        // capability enforces is not weakened by the clone, because it was never this layer's to enforce: a staged
        // handle is held to it while the program is traced, and an eager clone shares the allocation that reports the
        // misuse.
        Ok(vec![inputs[0].clone().freeze()?])
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to freeze")?;
        Ok(vec![ReferenceDischargeValue::Value(context.consume(reference)?)])
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceFreeze<C::Value>>,
{
    // Freezing a reference freezes its tangent reference alongside, so the final value pairs with the final tangent
    // contents. A plumbing reference carries no tangent reference, so its final value has a symbolic zero tangent. The
    // operands are cloned before consumption for the same reason the interpretation rule clones them: the rule replays
    // an already-built application over borrowed duals, and a clone names the same allocation.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().freeze()?;
        Ok(vec![forwarded_tangent(&inputs[0], primal, |tangent_reference| tangent_reference.clone().freeze())?])
    }
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceFreeze<C::Value>>,
    P: BatchingPolicy<C>,
{
    // Freezing yields the final packed referent, batched at the reference's own axis.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        _context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![P::batch(P::value(&inputs[0]).clone().freeze()?, P::batch_axis(&inputs[0]))?].into())
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U> + ResidualZeroProvider<U>,
    Tracer<TracingContext<V, O>>:
        ReferenceNew<Tracer<TracingContext<V, O>>> + ReferenceAddUpdate<Tracer<TracingContext<V, O>>>,
{
    // A freeze reads the final state and consumes the allocation, so its transpose accumulates the frozen value's
    // cotangent into the root's cotangent reference exactly like a read. The cotangent reference stays live for the
    // earlier (in program order) accesses that the reverse sweep visits next.
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
            context.cotangent_reference(0)?.add_update(cotangent)?;
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
    use crate::programs::references::operations::tests::*;
    use crate::programs::references::operations::{ReferenceNew, ReferenceRead};
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;

    use super::*;

    #[test]
    fn test_reference_freeze_operation() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Freeze::new());
        assert_eq!(Freeze::new().to_string(), REFERENCE_FREEZE_OPERATION_NAME);
        assert_eq!(Freeze::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Freeze::new().reference_semantics().outputs(), &[]);
        assert_eq!(
            Freeze::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::Consume)],
        );

        let minimal_reference = ReadFreezeUniverse::Reference(ReferenceType::new(referent));
        assert_eq!(
            ReferenceFreezeOperation::<TestReferent, ReadFreezeUniverse>::new()
                .infer_output_types(std::slice::from_ref(&minimal_reference), &[]),
            Ok(vec![ReadFreezeUniverse::Value(referent)]),
        );
        assert_eq!(Freeze::new().infer_output_types(std::slice::from_ref(&reference), &[]), Ok(vec![value.clone()]),);
        assert_eq!(Freeze::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            Freeze::new().infer_output_types(std::slice::from_ref(&value), &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            Freeze::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_freeze_operation_reference_discharge() {
        // A freeze yields the allocation's final state and unbinds the allocation, so every later access is a
        // use-after-consume.
        let (context, reference) = allocated_allocation(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Freeze::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.live_allocation_ids(), Vec::new());
        assert_eq!(
            Freeze::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed consumed {}",
                reference.allocation_id(),
            ))),
        );
    }

    #[test]
    fn test_reference_freeze_operation_jvp() {
        type TestValue = ArrayIrValue<Array>;

        let context = DifferentiationContext::new(EagerContext::<TestValue, ArrayIrOperation<Array>>::new());
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![3.0_f32, 4.0])).reference_new().unwrap();

        // Freezing an active reference freezes its tangent reference alongside and invalidates both alias families.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[active]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestValue::Array(Array::vector(vec![1.0_f32, 2.0])));
        assert_eq!(outputs[0].tangent().as_value(), Some(&TestValue::Array(Array::vector(vec![3.0_f32, 4.0]))));
        assert!(reference.read().is_err());
        assert!(tangent_reference.read().is_err());

        // Freezing a plumbing reference yields a symbolic zero tangent of the referent's tangent type.
        let reference = TestValue::Array(Array::vector(vec![5.0_f32, 6.0])).reference_new().unwrap();
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[plumbing]).unwrap();
        assert_eq!(outputs[0].primal(), &TestValue::Array(Array::vector(vec![5.0_f32, 6.0])));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        ));
        assert!(reference.read().is_err());
    }

    #[test]
    fn test_reference_freeze_operation_batching() {
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
        let packed = TestValue::Array(Array::from_f64s(packed_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // Freezing a batched reference yields the final packed referent at the reference's batch axis.
        let reference = packed.reference_new().unwrap();
        let input =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].batch().value(), &packed);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        assert!(reference.read().is_err());

        // Freezing a replicated reference stays replicated.
        let reference = packed.reference_new().unwrap();
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &packed);
    }
}
