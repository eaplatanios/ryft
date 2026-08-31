//! Generic immutable reference read operation and its value-level capability.

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

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

static REFERENCE_READ_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Read)], Vec::new())
});

define_reference_primitive_payload!(
    /// Reads the current referent snapshot from a reference in the enclosing type universe `U`.
    ReferenceReadOperation
);

impl_reference_primitive_display!(ReferenceReadOperation, REFERENCE_READ_OPERATION_NAME);

impl<T, U> Operation for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
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
        Cow::Borrowed(&REFERENCE_READ_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceRead<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].read()?])
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
    P: ReferenceDischargePolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let reference = inputs[0].expect_reference("a reference to read")?;
        Ok(vec![ReferenceDischargeValue::Ordinary(context.read(reference)?)])
    }
}

impl_unsupported_reference_transforms!(ReferenceReadOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceReadOperation<T, U>
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
    fn test_reference_read_operation() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Read::new());
        assert_eq!(Read::new().to_string(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(Read::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Read::new().reference_semantics().outputs(), &[]);
        assert_eq!(Read::new().reference_semantics().inputs(), &[ReferenceInput::new(0, ReferenceAccessMode::Read)],);

        let minimal_reference = ReadFreezeUniverse::Reference(ReferenceType::new(referent));
        assert_eq!(
            ReferenceReadOperation::<TestReferent, ReadFreezeUniverse>::new()
                .infer_output_types(std::slice::from_ref(&minimal_reference), &[]),
            Ok(vec![ReadFreezeUniverse::Value(referent)]),
        );
        assert_eq!(Read::new().infer_output_types(std::slice::from_ref(&reference), &[]), Ok(vec![value.clone()]),);
        assert_eq!(Read::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&value), &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_read_operation_reference_discharge() {
        // A read observes the allocation's current state without changing it, so the allocation stays unmutated.
        let (context, reference) = allocated_allocation(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(false));

        // An ordinary operand denotes no allocation, so the rule reports what it expected instead of reading a value.
        let pure: TestDischargeValue = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4));
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&pure)),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received an ordinary value".to_string(),
            )),
        );
    }
}
