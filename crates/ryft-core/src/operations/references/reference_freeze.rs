//! Generic consuming reference finalization operation and its value-level capability.

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
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider, TransposableOperation,
    TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::references::reference_add_update::ReferenceAddUpdate;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClasses, Effects, MaybeZero, NoReferent, Operation, ProgramError, ProjectedValue, ReferenceAccessMode,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceEffect, ReferenceMemberType, ReferenceType, ReferenceViewOperation,
    RegionInterface, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::{ReferenceNewOperationProvider, forwarded_tangent};

/// Canonical operation name for [`ReferenceFreezeOperation`].
pub const REFERENCE_FREEZE_OPERATION_NAME: &str = "reference_freeze";

/// Consumes an allocation reference, returning its final referent and invalidating its complete alias family.
#[derive(Clone, Debug)]
pub struct ReferenceFreezeOperation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

impl<T: Type, U: Type> ReferenceFreezeOperation<T, U> {
    /// Creates a new [`ReferenceFreezeOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type, U: Type> Copy for ReferenceFreezeOperation<T, U> {}

impl<T: Type, U: Type> Display for ReferenceFreezeOperation<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REFERENCE_FREEZE_OPERATION_NAME)
    }
}

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
    fn effects(&self) -> Cow<'_, Effects> {
        // The effect descriptor is built once and shared by every `<T, U>` instantiation of this operation: the
        // `static` names no generic parameter, so this generic function owns exactly one instance, and `LazyLock` is
        // required because `Effects::new` validates and allocates at runtime. Nesting it here scopes it to its only
        // reader and lets `effects` hand out a borrowed `'static` descriptor without cloning on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }],
                Vec::new(),
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
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

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
    P: BatchingPolicy<C>,
{
    // Freezing yields the final packed referent, batched at the reference's own axis.
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

impl<T, U, C> DifferentiableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
{
    // Freezing a reference freezes its tangent reference alongside, so the final value pairs with the final tangent
    // contents. A plumbing reference carries no tangent reference, so its final value has a symbolic zero tangent. The
    // operands are cloned before consumption for the same reason the interpretation rule clones them: the rule replays
    // an already-built application over borrowed duals, and a clone names the same allocation.
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

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: DifferentiableType + ReferenceMemberType,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U> + ResidualZeroProvider<U, Operation = O> + ReferenceNewOperationProvider<U>,
    Tracer<TracingContext<V, O>>: ReferenceAddUpdate,
{
    // A freeze reads the final state and consumes the allocation, so its transpose accumulates the frozen value's
    // cotangent into the root's cotangent reference exactly like a read. The cotangent reference stays live for the
    // earlier (in program order) accesses that the reverse sweep visits next.
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

// TODO(eaplatanios): Restore the strict `Operation<Type = T>` super-trait bound on the three reference operation
//  providers once the next-generation trait solver stabilizes. The current solver cannot discharge this projection
//  equality at bound sites whose tracing context is built from the bounded operation family (E0284); every
//  implementation constrains its target to `Operation<Type = T>` instead.
/// Selects the operation of an [`Operation`] family over the universe `T` that consumes a reference over a referent
/// and returns its final value, or reports that the family provides none. Reverse-mode differentiation freezes
/// completed cotangent references through it. Refer to the [module documentation](super) for the shared provider
/// contract, including how reference-free universes implement it.
pub trait ReferenceFreezeOperationProvider<T: ReferenceMemberType>: Operation + Sized {
    /// Returns this family's operation that freezes a reference over `referent`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`] when this family provides no reference freezing.
    fn reference_freeze(referent: &T::Referent) -> Result<Self, ProgramError>;
}

// Composite array families select the canonical payload; the reference-free array and scalar universes have no
// referent values, so their providers are unreachable by construction.
impl<O: Operation<Type = ArrayIrType> + From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>>
    ReferenceFreezeOperationProvider<ArrayIrType> for O
{
    fn reference_freeze(_referent: &ArrayType) -> Result<Self, ProgramError> {
        Ok(ReferenceFreezeOperation::new().into())
    }
}

impl<O: Operation<Type = ArrayType>> ReferenceFreezeOperationProvider<ArrayType> for O {
    fn reference_freeze(referent: &NoReferent) -> Result<Self, ProgramError> {
        match *referent {}
    }
}

impl<O: Operation<Type = DataType>> ReferenceFreezeOperationProvider<DataType> for O {
    fn reference_freeze(referent: &NoReferent) -> Result<Self, ProgramError> {
        match *referent {}
    }
}

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
    /// let allocation = ArrayIrValue::Array(Array::scalar(1.0_f32).unwrap()).reference_new()?;
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
    /// let allocation = ArrayIrValue::Array(Array::scalar(1.0_f32).unwrap()).reference_new()?;
    /// let alias = allocation.clone();
    /// assert_eq!(allocation.freeze()?, ArrayIrValue::Array(Array::scalar(1.0_f32).unwrap()));
    /// assert_eq!(
    ///     alias.read().unwrap_err().downcast_custom::<ReferenceError>(),
    ///     Some(&ReferenceError::Frozen),
    /// );
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    fn freeze(self) -> Result<Output, ProgramError>;
}

impl<A: Value<Type = ArrayType>> ReferenceFreeze for ArrayIrValue<A> {
    fn freeze(self) -> Result<Self, ProgramError> {
        ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new()
            .infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(&self)?;
        Ok(Self::Array(reference.freeze()?))
    }
}

impl<V> ReferenceFreeze<<V as ValueProjection<ArrayType>>::Projected> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(self) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let domain = self.value().dispatch_domain();
        domain
            .bind(ReferenceFreezeOperation::new(), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceFreeze<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>,
{
    fn freeze(self) -> Result<V, ProgramError> {
        let domain = self.dispatch_domain();
        Ok(domain.bind(ReferenceFreezeOperation::new(), Vec::new(), std::slice::from_ref(&self))?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::math::add::AddOperation;
    use crate::operations::references::reference_add_update::ReferenceAddUpdate;
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::operations::references::reference_swap::ReferenceSwap;
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder, ReferenceError};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrFreeze = ReferenceFreezeOperation<ArrayType, ArrayIrType>;

    #[test]
    fn test_reference_freeze() {
        let operation = Freeze::new();
        assert_eq!(operation.name(), REFERENCE_FREEZE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_FREEZE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!("ReferenceFreezeOperation({:?})", PhantomData::<fn() -> (TestReferent, TestType)>),
        );
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }],
        );
        assert_eq!(operation.effects().reference_aliases(), &[]);
    }

    #[test]
    fn test_reference_freeze_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Freeze::new(),
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
            Freeze::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // Freezing only requires a universe that embeds referents and projects reference types.
        check_operation_type_inference!(
            operation = ReferenceFreezeOperation::<TestReferent, ReadFreezeUniverse>::new(),
            cases = [{
                input_types = [ReadFreezeUniverse::Reference(ReferenceType::new(referent))],
                output_types = [ReadFreezeUniverse::Value(referent)],
            }],
        );

        // The array universe freezes array referents and rejects its other members.
        let array_type = ArrayType::scalar(DataType::F32);
        let dimension_type = DimensionType::new(DimensionVariable::new("n", DimensionBounds::unbounded()));
        check_operation_type_inference!(
            operation = TestIrFreeze::new(),
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
    fn test_reference_freeze_interpretation() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let value = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        let alias = reference.clone();

        // A freeze returns the final referent and invalidates the complete alias family, so every clone of the handle
        // fails its next access against the shared allocation state.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrFreeze::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&reference),
            ),
            Ok(vec![value.clone()]),
        );
        let error = alias.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.swap(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.add_update(&TestIrValue::Array(Array::scalar(1.0_f32).unwrap())).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.freeze().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrFreeze::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));

        // The value-level capability consumes its handle and agrees with the operation.
        let reference = value.reference_new().unwrap();
        assert_eq!(reference.freeze(), Ok(value.clone()));

        // Only reference members can be frozen.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrFreeze::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&value),
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(value.freeze(), Err(TypeError::invalid("expected reference type but got array type").into()));

        // Freezing a dynamically typed referent returns exactly the installed dynamic payload rather than the original
        // one, and the consumed handle fails afterwards. Equality over a dynamically typed `Array` cannot address its
        // elements, so the frozen referent is unwrapped and compared by its declared type and storage. `Array`'s
        // checked constructors reject dynamically shaped types, so both referents come from the test-only unchecked
        // hatch.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let replacement_bytes = 2.0_f32.to_le_bytes().to_vec();
        let initial = Array::with_unchecked_type(dynamic_type.clone(), 1.0_f32.to_le_bytes().to_vec());
        let replacement = Array::with_unchecked_type(dynamic_type.clone(), replacement_bytes.clone());
        let reference = TestIrValue::Reference(ArrayReference::new(initial));
        reference.swap(&TestIrValue::Array(replacement)).unwrap();
        let frozen = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrFreeze::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap()
        .remove(0);
        let frozen = <TestIrValue as ValueProjection<ArrayType>>::into_projected(frozen).unwrap();
        assert_eq!(frozen.r#type().into_owned(), dynamic_type);
        assert_eq!(frozen.storage_bytes(), replacement_bytes.as_slice());
        let error = reference.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_freeze_partial_evaluation() {
        type TestContext = EagerContext<TestIrValue, TestIrOperation>;

        let value = TestIrValue::Array(Array::scalar(1.0_f32).unwrap());
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));

        // Under the `Stage` placement the freeze stays residual regardless of operand knowledge, so eager
        // specialization never consumes live reference state.
        let staging =
            PartialEvaluationContext::new(TestContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging
            .fold_or_residualize(TestIrFreeze::new(), Vec::new(), std::slice::from_ref(&reference))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));

        // Under the default `Execute` placement a known reference folds the freeze against the live state, which the
        // freeze consumes.
        let executing = PartialEvaluationContext::new(TestContext::new());
        let outputs = executing.fold_or_residualize(TestIrFreeze::new(), Vec::new(), &[reference]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&value));
        assert_eq!(live.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));

        // Program-level partial evaluation uses the `Stage` placement, so both a known and an unknown reference retain
        // the freeze in the residual program and replay it against the runtime reference.
        let known = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let replay = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = TestIrFreeze::new(),
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
    fn test_reference_freeze_batching() {
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

    #[test]
    fn test_reference_freeze_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<TestIrValue, TestIrOperation>::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();

        // Freezing an active reference freezes its tangent reference alongside and invalidates both alias families.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[active]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            outputs[0].tangent().as_value(),
            Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
        );
        assert!(reference.read().is_err());
        assert!(tangent_reference.read().is_err());

        // Freezing a plumbing reference yields a symbolic zero tangent of the referent's tangent type.
        let reference = TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()).reference_new().unwrap();
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceFreezeOperation::new(), Vec::new(), &[plumbing]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        ));
        assert!(reference.read().is_err());
    }

    #[test]
    fn test_reference_freeze_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // `r = new(v); y = freeze(r)` reads the final state and consumes the allocation, so the freeze's transpose
        // accumulates `ȳ` into the root's cotangent reference exactly like a read, and the allocation's transpose then
        // freezes that accumulator into `v̄`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let output = builder.add_instruction(TestIrFreeze::new(), Vec::new(), vec![reference], None).unwrap()[0];
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
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(2.0_f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(2.0_f32).unwrap())]),
        );

        // The rule accumulates through the cotangent reference of its operand's root, which only a transposition
        // context scoped to the freeze instruction can resolve, so a detached context rejects a live result cotangent.
        let inputs = [PartialValue::Unknown(reference_type)];
        let tracing = TracingContext::<TestIrValue, TestIrOperation>::new();
        let cotangent = tracing.input(scalar_type);
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrFreeze::new().transpose(
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
            TestIrFreeze::new().transpose(
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
    fn test_reference_freeze_reference_discharge() {
        // A freeze yields the allocation's final state and unbinds the allocation, so every later access is a
        // use-after-consume.
        let (context, reference) = allocated_reference(4);
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
    fn test_reference_freeze_provider() {
        // Composite array families select the canonical consuming operation for an array referent, keeping alias
        // invalidation in the reference machinery.
        assert!(matches!(
            TestIrOperation::reference_freeze(&ArrayType::scalar(DataType::F32)),
            Ok(ArrayIrOperation::ReferenceFreeze(_)),
        ));

        // Reference-free array and scalar families satisfy the provider contract without any runtime rejection, so
        // the contract is checked at compile time only.
        fn assert_provider<T: ReferenceMemberType, O: ReferenceFreezeOperationProvider<T>>() {}
        assert_provider::<ArrayType, ArrayOperation<Array>>();
        assert_provider::<DataType, AddOperation<DataType>>();
    }

    #[test]
    fn test_reference_freeze_projected() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // Freezing through a projected reference member binds the operation through the parent tracer's context and
        // hands back the projected array member.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(input)?;
                let frozen: ProjectedValue<ArrayType, TestTracer> = reference.freeze()?;
                Ok(frozen.into_value())
            },
            reference_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]> .
                let %1:f32[2] = reference_freeze %0
                in (%1)"},
        );
    }

    #[test]
    fn test_reference_freeze_staging() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;

        // A staged freeze is the native `reference_freeze` variant of the array operation family.
        let (output_type, program) = TestContext::trace(
            |input| input.freeze(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:f32[] = reference_freeze %0
                in (%1)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
