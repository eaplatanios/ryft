//! Generic ordered additive reference update operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::AddOperation;
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};

use super::validate_operand_types;

/// Canonical operation name for [`ReferenceAddUpdateOperation`].
pub const REFERENCE_ADD_UPDATE_OPERATION_NAME: &str = "reference_add_update";

/// Adds an update into the value stored by a reference in program order.
pub trait ReferenceAddUpdate<Update = Self>: Sized {
    /// Adds `update` to the stored value in program order.
    fn add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

static REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Accumulate)], Vec::new())
});

define_reference_primitive_payload!(
    /// Applies an ordered additive update whose result must retain the reference's exact referent type.
    ReferenceAddUpdateOperation
);

impl_reference_primitive_display!(ReferenceAddUpdateOperation, REFERENCE_ADD_UPDATE_OPERATION_NAME);

impl<T, U> Operation for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    AddOperation<T>: Operation<Type = T>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ADD_UPDATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let update = <&T>::try_from(&input_types[1])?;
        let addition_results =
            AddOperation::<T>::new().infer_output_types(&[reference.referent().clone(), update.clone()], &[])?;
        check_count!("output", addition_results, 1, TypeError);
        let addition_result = &addition_results[0];
        if addition_result != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` addition result type `{addition_result}` must exactly match \
                 reference referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceAddUpdate<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].add_update(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U>>>,
    P: ReferenceAccumulationPolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].expect_reference("a reference to accumulate into")?;
        let update = inputs[1].expect_ordinary("an update value")?.clone();

        // The sum of the handle's referent and the update must itself be the handle's referent, which is exactly what
        // this operation's own inference states and what a universe's addition alone does not guarantee.
        validate_operand_types(self, inputs)?;
        context.accumulate(reference, update)?;
        Ok(Vec::new())
    }
}

impl_unsupported_reference_transforms!(ReferenceAddUpdateOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceAddUpdateOperation<T, U>
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
    fn test_reference_add_update_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(AddUpdate::new());
        assert_eq!(AddUpdate::new().to_string(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(AddUpdate::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(AddUpdate::new().reference_semantics().outputs(), &[]);
        assert_eq!(
            AddUpdate::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::Accumulate)],
        );

        assert_eq!(
            ReferenceAddUpdateOperation::<TestReferent, AddUpdateUniverse>::new().infer_output_types(
                &[AddUpdateUniverse::Reference(ReferenceType::new(referent)), AddUpdateUniverse::Value(referent)],
                &[],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(AddUpdate::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(Vec::new()));
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
                 type `value<i7,p16>`",
            )),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(&[value.clone(), value.clone()], &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_add_update_operation_reference_discharge() {
        // An accumulation produces no result and replaces the current state with its sum with the update.
        let (context, reference) = allocated_allocation(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(Vec::new()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // An update whose sum with the referent would not itself be the referent is rejected by this operation's own
        // inference before the universe accumulates anything, so the allocation keeps its previous state.
        let promoted = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, promoted.as_slice()),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
             type `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
    }
}
