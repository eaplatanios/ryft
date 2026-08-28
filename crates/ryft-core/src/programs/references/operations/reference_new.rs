//! Generic reference allocation operation and its value-level capability.

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
use crate::programs::references::semantics::{ReferenceOperationSemantics, ReferenceOutput};
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};

/// Canonical operation name for [`ReferenceNewOperation`].
pub const REFERENCE_NEW_OPERATION_NAME: &str = "reference_new";

/// Creates a new reference initialized from this value.
pub trait ReferenceNew<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn reference_new(&self) -> Result<Output, ProgramError>;
}

static REFERENCE_NEW_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }])
});

define_reference_primitive_payload!(
    /// Allocates a reference allocation for a referent of type `T` in the enclosing type universe `U`.
    ReferenceNewOperation
);

impl_reference_primitive_display!(ReferenceNewOperation, REFERENCE_NEW_OPERATION_NAME);

impl<T, U> Operation for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type + From<ReferenceType<T>>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_NEW_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let referent = <&T>::try_from(&input_types[0])?;
        if referent.is_reference() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` cannot allocate a reference whose referent type `{referent}` is \
                 itself a reference",
            )));
        }
        Ok(vec![ReferenceType::new(referent.clone()).into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_NEW_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceNew<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reference_new()?])
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let initial = inputs[0].expect_ordinary("an initial reference state")?.clone();

        // The allocation's reference type is exactly the one this operation's own inference derives from the
        // initializer, so the rewrite never re-derives a referent that the type system already settled.
        let output_types = self.infer_output_types(&[initial.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let r#type = P::project_reference_type(&output_types[0]).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` inferred the non-reference output type `{}`",
                output_types[0],
            ))
        })?;
        if context.selects_allocation(driver.instruction(), 0) {
            return Ok(vec![context.allocate_discharged(r#type, initial)?]);
        }

        // An unselected allocation site survives, so the operation is replayed and its result is the destination
        // reference bound to that allocation.
        let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![context.bind_preserved(r#type, outputs.remove(0))?])
    }
}

impl_unsupported_reference_transforms!(ReferenceNewOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceNewOperation<T, U>
    where
        T: Type,
        U: Type,
);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::identities::TypeIdentityPosition;
    use crate::programs::references::operations::tests::*;
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_reference_new_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(New::new());
        assert_eq!(New::new(), New::default());
        assert_eq!(format!("{:?}", New::new()), "ReferenceNewOperation");
        assert_eq!(New::new().to_string(), REFERENCE_NEW_OPERATION_NAME);
        assert_eq!(New::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(New::new().reference_semantics().inputs(), &[]);
        assert_eq!(New::new().reference_semantics().outputs(), &[ReferenceOutput::Allocation { output_index: 0 }],);

        assert_eq!(
            ReferenceNewOperation::<TestReferent, NewUniverse>::new()
                .infer_output_types(&[NewUniverse::Value(referent)], &[]),
            Ok(vec![NewUniverse::Reference(ReferenceType::new(referent))]),
        );
        assert_eq!(New::new().infer_output_types(std::slice::from_ref(&value), &[]), Ok(vec![reference.clone()]));
        assert_eq!(New::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&value), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        assert_eq!(
            referent.identities().collect::<Vec<_>>(),
            vec![(TypeIdentityPosition::Definition, &referent.identity)],
        );
        assert!(referent.is_refined_by(&promoted_refinement));

        let nested_referent = ReferenceType::new(referent);
        assert_eq!(
            ReferenceNewOperation::<ReferenceType<TestReferent>, NestedTestType>::new()
                .infer_output_types(&[NestedTestType::Reference(nested_referent.clone())], &[]),
            Err(TypeError::invalid(format!(
                "`reference_new` cannot allocate a reference whose referent type `{nested_referent}` is itself a \
                 reference",
            ))),
        );
    }

    #[test]
    fn test_reference_new_operation_reference_discharge() {
        // Allocation binds a fresh discharged reference whose entering state is the initializer and whose reference
        // type is the one this operation's own inference derives, exposed through the storage alias of a complete
        // value.
        let (context, reference) = allocated_allocation(4);
        assert_eq!(context.live_allocations(), vec![reference.allocation()]);
        assert_eq!(reference.r#type(), &ReferenceType::new(REFERENT));
        assert_eq!(reference.alias(), &TestAlias);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.discharged_state(reference.allocation()), Ok(TestValue::new(REFERENT, 4)));
        assert_eq!(context.is_mutated(reference.allocation()), Ok(false));

        // A reference operand is not an initial state, and the diagnostic says which operand the rule expected.
        let context = TestDischargeContext::new(TestDestination::new());
        let handle = ReferenceDischargeValue::Reference(reference);
        assert_eq!(
            New::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an initial reference state but received {handle}",
            ))),
        );

        // The rule reads its fresh allocation's reference type back out of its own inferred output type, so a policy
        // whose projection disagrees with that inference cannot silently allocate an unclassifiable allocation.
        let disagreeing =
            ReferenceDischargeContext::<TestDestination, NonProjectingReferenceDischarge>::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4));
        assert_eq!(
            New::new().discharge_references(&disagreeing, &EmptyRegionDriver, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "`reference_new` inferred the non-reference output type `ref<value<i7,p16>>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_primitive_discharge_preserves_an_unselected_allocation() {
        // The allocation rule consults its own replay position against the selection, so an unselected allocation
        // site is replayed rather than turned into threaded state, and its allocated reference survives in the
        // destination.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(REFERENT));
        let update = builder.add_input(TestType::Value(REFERENT));
        let allocation =
            builder.add_instruction(TestOperation::New(New::new()), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(TestOperation::AddUpdate(AddUpdate::new()), Vec::new(), vec![allocation, update], None)
            .unwrap();
        let frozen = builder
            .add_instruction(TestOperation::Freeze(Freeze::new()), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let preserved =
            source.clone().partially_discharge_references_with_policy::<TestReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.public_output_count(), 1);
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
            lambda %0:value<i7,p16>, %1:value<i7,p16> .
            let %2:ref<value<i7,p16>> = reference_new %0
                reference_add_update %2 %1
                %3:value<i7,p16> = reference_freeze %2
            in (%3)"},
        );

        // Selecting that same site is the everything-selected case, so it must agree with full discharge exactly.
        let sites = source.reference_discharge_sites(0).unwrap();
        let selected = source
            .clone()
            .partially_discharge_references_with_policy::<TestReferenceDischarge>(0, sites.as_slice())
            .unwrap()
            .try_into_full()
            .unwrap();
        let full = source.discharge_references_with_policy::<TestReferenceDischarge>(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
        assert_eq!(
            full.program().to_string(),
            indoc! {"
            lambda %0:value<i7,p16>, %1:value<i7,p16> .
            let %2:value<i7,p16> = test.add %0 %1
            in (%2)"},
        );
    }
}
