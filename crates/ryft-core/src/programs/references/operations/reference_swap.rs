//! Generic read-write reference replacement operation and its value-level capability.

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

impl_unsupported_reference_transforms!(ReferenceSwapOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceSwapOperation<T, U>
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
}
