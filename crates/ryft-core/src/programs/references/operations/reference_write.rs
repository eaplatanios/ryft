//! Generic write-only reference replacement operation and its value-level capability.

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

use super::validate_operand_types;

/// Canonical operation name for [`ReferenceWriteOperation`].
pub const REFERENCE_WRITE_OPERATION_NAME: &str = "reference_write";

/// Replaces the value stored by a reference without observing the previous value.
pub trait ReferenceWrite<Replacement = Self>: Sized {
    /// Replaces the stored value with `replacement` in program order.
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError>;
}

static REFERENCE_WRITE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Write)], Vec::new())
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
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_WRITE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
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

impl_unsupported_reference_transforms!(ReferenceWriteOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceWriteOperation<T, U>
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
    fn test_reference_write_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(Write::new());
        assert_eq!(Write::new().to_string(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(Write::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Write::new().reference_semantics().outputs(), &[]);
        assert_eq!(Write::new().reference_semantics().inputs(), &[ReferenceInput::new(0, ReferenceAccessMode::Write)],);

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
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
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
        let allocated = context.bind_discharged(ReferenceType::new(REFERENT), initial).unwrap();
        let reference = allocated.try_as_reference("the allocated allocation").unwrap().clone();
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
}
