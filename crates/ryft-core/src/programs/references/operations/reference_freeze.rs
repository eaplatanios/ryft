//! Generic consuming reference finalization operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};

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
        let reference = inputs[0].expect_reference("a reference to freeze")?;
        Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
    }
}

impl_unsupported_reference_transforms!(ReferenceFreezeOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceFreezeOperation<T, U>
    where
        T: Type,
        U: Type,
);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::references::operations::tests::*;
    use crate::programs::regions::EmptyRegionDriver;

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
            Ok(vec![ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.live_allocations(), Vec::new());
        assert_eq!(
            Freeze::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed consumed {}",
                reference.allocation_id(),
            ))),
        );
    }
}
