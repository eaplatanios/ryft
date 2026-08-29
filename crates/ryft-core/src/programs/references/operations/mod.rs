//! Generic reference primitive operations and their value-level capabilities.
//!
//! Each child module owns one complete primitive: its value-level capability, type-indexed operation payload,
//! inference and effects, eager interpretation, reference-discharge rewrite, transform behavior, and unit tests.
//! This facade retains only machinery genuinely shared by multiple primitives and re-exports the established public
//! operation surface.

// TODO(eaplatanios): Review this module.

use crate::contexts::Domain;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, Typed};

use super::discharge::{ReferenceDischargePolicy, ReferenceDischargeValue};
use super::types::ReferenceType;

macro_rules! define_reference_primitive_payload {
    // Defines one type-indexed zero-sized payload without deriving unnecessary bounds on its phantom type parameters.
    ($(#[$documentation:meta])* $operation:ident) => {
        $(#[$documentation])*
        pub struct $operation<T: $crate::programs::Type, U: $crate::programs::Type>(
            std::marker::PhantomData<fn() -> (T, U)>,
        );

        impl<T: $crate::programs::Type, U: $crate::programs::Type> $operation<T, U> {
            #[doc = concat!("Creates a new [`", stringify!($operation), "`].")]
            #[inline]
            pub const fn new() -> Self {
                Self(std::marker::PhantomData)
            }
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> Copy for $operation<T, U> {}

        impl<T: $crate::programs::Type, U: $crate::programs::Type> Clone for $operation<T, U> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> std::fmt::Debug for $operation<T, U> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(stringify!($operation))
            }
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> Default for $operation<T, U> {
            #[inline]
            fn default() -> Self {
                Self::new()
            }
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> PartialEq for $operation<T, U> {
            #[inline]
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> Eq for $operation<T, U> {}

        impl<T: $crate::programs::Type, U: $crate::programs::Type> std::hash::Hash for $operation<T, U> {
            #[inline]
            fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
        }

        impl<T: $crate::programs::Type, U: $crate::programs::Type> $crate::parameters::Parameter
            for $operation<T, U>
        {
        }
    };
}

macro_rules! impl_reference_primitive_display {
    // Renders one payload using its canonical operation name without requiring its conversion seam.
    ($operation:ident, $name:ident) => {
        impl<T: $crate::programs::Type, U: $crate::programs::Type> std::fmt::Display for $operation<T, U> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str($name)
            }
        }
    };
}

/// Re-derives one reference primitive's own type inference over the carriers it received, so that a rewrite acts only
/// on operands the operation itself accepts.
///
/// A [`Program`](crate::Program) built through a [`ProgramBuilder`](crate::ProgramBuilder) already ran this inference
/// when the instruction was added, but a rule invoked outside a checked program replay has no such guarantee, and the
/// rules that relate two operands to each other cannot recover that relationship from the carriers alone. Only those
/// rules call this: an allocation derives its own output type instead, and a read or a freeze relates no operands, so
/// re-deriving would restate the projection the rule already performs.
///
/// # Parameters
///
///   - `operation`: Reference primitive whose inference is re-derived.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
fn validate_operand_types<U: Type, C, P, O>(
    operation: &O,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<(), ProgramError>
where
    C: Domain<Type = U>,
    U: From<ReferenceType<P::Referent>>,
    P: ReferenceDischargePolicy<C>,
    O: Operation<Type = U>,
{
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    operation.infer_output_types(input_types.as_slice(), &[])?;
    Ok(())
}

macro_rules! impl_unsupported_reference_transforms {
    // Defines the same conservative transform rejections for one unresolved generic reference primitive.
    ($operation:ident) => {
        impl<T, U, C> $crate::partial::PartiallyEvaluatableOperation<C> for $operation<T, U>
        where
            T: $crate::programs::Type,
            U: $crate::programs::Type,
            C: $crate::contexts::Context<Type = U, Operation: From<$operation<T, U>>>,
        {
        }

        impl<T, U, C, P> $crate::batching::BatchableOperation<C, P> for $operation<T, U>
        where
            T: $crate::programs::Type,
            U: $crate::programs::Type,
            $operation<T, U>: $crate::programs::Operation<Type = U>,
            C: $crate::contexts::Context<Type = U, Operation: From<$operation<T, U>>>,
            P: $crate::batching::BatchingPolicy<C>,
        {
            fn batch<D: $crate::batching::BatchingDriver<C, P>>(
                &self,
                _context: &$crate::batching::BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<$crate::batching::BatchedOutputs<C, P>, $crate::batching::BatchingError> {
                Err($crate::batching::BatchingError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before batching", self.name()),
                })
            }
        }

        impl<T, U, C> $crate::differentiation::DifferentiableOperation<C> for $operation<T, U>
        where
            T: $crate::programs::Type,
            U: $crate::programs::Type,
            $operation<T, U>: $crate::programs::Operation<Type = U>,
            C: $crate::contexts::Context<Type = U, Operation: From<$operation<T, U>>>,
        {
            fn jvp<D: $crate::differentiation::DifferentiationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[$crate::differentiation::DifferentiationDual<C::Value>],
            ) -> Result<
                Vec<$crate::differentiation::DifferentiationDual<C::Value>>,
                $crate::differentiation::DifferentiationError,
            > {
                Err($crate::programs::ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before differentiation", self.name()),
                }
                .into())
            }
        }
    };
}

mod reference_add_update;
mod reference_freeze;
mod reference_new;
mod reference_read;
mod reference_swap;
mod reference_write;

pub use reference_add_update::{REFERENCE_ADD_UPDATE_OPERATION_NAME, ReferenceAddUpdate, ReferenceAddUpdateOperation};
pub use reference_freeze::{REFERENCE_FREEZE_OPERATION_NAME, ReferenceFreeze, ReferenceFreezeOperation};
pub use reference_new::{REFERENCE_NEW_OPERATION_NAME, ReferenceNew, ReferenceNewOperation};
pub use reference_read::{REFERENCE_READ_OPERATION_NAME, ReferenceRead, ReferenceReadOperation};
pub use reference_swap::{REFERENCE_SWAP_OPERATION_NAME, ReferenceSwap, ReferenceSwapOperation};
pub use reference_write::{REFERENCE_WRITE_OPERATION_NAME, ReferenceWrite, ReferenceWriteOperation};

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::fmt::{Debug, Display};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, EagerContext};
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::{Add, AddOperation};
    use crate::parameters::{Parameter, Parameterized, Placeholder};
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::identities::{TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming};
    use crate::programs::references::discharge::{
        ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargeReference,
        ReferenceDischargeableOperation, discharge_reference_free_operation,
    };
    use crate::programs::references::semantics::ReferenceOperationSemantics;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{EmptyRegionDriver, RegionInterface};
    use crate::programs::types::{Type, TypeError};
    use crate::programs::values::Value;

    use super::*;

    /// Type identity used by the generic reference-operation test universes.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(crate) struct TestIdentity(pub(crate) u8);

    impl Display for TestIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "i{}", self.0)
        }
    }

    impl TypeIdentity for TestIdentity {
        fn fresh(&self) -> Self {
            Self(self.0.wrapping_add(128))
        }
    }

    /// Ordinary referent type used to exercise generic reference operations without array-specific behavior.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(crate) struct TestReferent {
        pub(crate) identity: TestIdentity,
        pub(crate) precision: u8,
    }

    impl TestReferent {
        /// Creates a test referent with the provided identity and precision.
        pub(crate) const fn new(identity: u8, precision: u8) -> Self {
            Self { identity: TestIdentity(identity), precision }
        }
    }

    impl Display for TestReferent {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "value<{},p{}>", self.identity, self.precision)
        }
    }

    impl Parameter for TestReferent {}

    impl Type for TestReferent {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            std::iter::once((TypeIdentityPosition::Definition, &self.identity))
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(Self { identity: renaming.rename(&self.identity), precision: self.precision })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            self.identity == other.identity && self.precision <= other.precision
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self.identity == other.identity && self.precision <= other.precision
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl Operation for AddOperation<TestReferent> {
        type Type = TestReferent;

        fn name(&self) -> &'static str {
            "test_add"
        }

        fn infer_output_types(
            &self,
            input_types: &[TestReferent],
            region_interfaces: &[RegionInterface<TestReferent>],
        ) -> Result<Vec<TestReferent>, TypeError> {
            check_count!("input", input_types, 2, TypeError);
            check_count!("region", region_interfaces, 0, TypeError);
            Ok(vec![TestReferent {
                identity: input_types[0].identity,
                precision: input_types.iter().map(|r#type| r#type.precision).max().unwrap(),
            }])
        }
    }

    /// Complete value/reference type universe shared by the reference-operation tests.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum TestType {
        Value(TestReferent),
        Reference(ReferenceType<TestReferent>),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Value(r#type) => Display::fmt(r#type, formatter),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for TestType {}

    impl From<TestReferent> for TestType {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl From<ReferenceType<TestReferent>> for TestType {
        fn from(r#type: ReferenceType<TestReferent>) -> Self {
            Self::Reference(r#type)
        }
    }

    impl<'t> TryFrom<&'t TestType> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t TestType) -> Result<Self, Self::Error> {
            match r#type {
                TestType::Value(r#type) => Ok(r#type),
                TestType::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t TestType> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t TestType) -> Result<Self, Self::Error> {
            match r#type {
                // This sentinel deliberately violates the otherwise canonical embedding/projection round trip. It
                // lets the allocation-discharge test pin its malformed-inference diagnostic without inventing a
                // second policy-level type conversion seam solely for tests.
                TestType::Reference(r#type) if r#type.referent() != &NON_PROJECTING_REFERENT => Ok(r#type),
                TestType::Reference(_) => {
                    Err(TypeError::invalid("the non-projecting test reference is deliberately not recognized"))
                }
                TestType::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl Type for TestType {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            match self {
                Self::Value(r#type) => r#type.identities().collect::<Vec<_>>(),
                Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
            }
            .into_iter()
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Value(r#type) => Self::Value(r#type.rename_identities(renaming)?),
                Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
            })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Value(left), Self::Value(right)) => left.is_compatible_with(right),
                (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                _ => false,
            }
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Value(left), Self::Value(right)) => left.is_refined_by(right),
                (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                _ => false,
            }
        }

        fn is_scalar(&self) -> bool {
            matches!(self, Self::Value(r#type) if r#type.is_scalar())
        }

        fn is_complex(&self) -> bool {
            matches!(self, Self::Value(r#type) if r#type.is_complex())
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Test universe that can represent a reference whose immediate referent is itself a reference.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum NestedTestType {
        Reference(ReferenceType<TestReferent>),
        Nested(ReferenceType<ReferenceType<TestReferent>>),
    }

    impl Display for NestedTestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
                Self::Nested(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for NestedTestType {}

    impl From<ReferenceType<ReferenceType<TestReferent>>> for NestedTestType {
        fn from(r#type: ReferenceType<ReferenceType<TestReferent>>) -> Self {
            Self::Nested(r#type)
        }
    }

    impl<'t> TryFrom<&'t NestedTestType> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t NestedTestType) -> Result<Self, Self::Error> {
            match r#type {
                NestedTestType::Reference(r#type) => Ok(r#type),
                NestedTestType::Nested(_) => {
                    Err(TypeError::invalid("expected reference type but got nested reference type"))
                }
            }
        }
    }

    impl Type for NestedTestType {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            match self {
                Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
                Self::Nested(r#type) => r#type.identities().collect::<Vec<_>>(),
            }
            .into_iter()
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
                Self::Nested(r#type) => Self::Nested(r#type.rename_identities(renaming)?),
            })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                (Self::Nested(left), Self::Nested(right)) => left.is_compatible_with(right),
                _ => false,
            }
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                (Self::Nested(left), Self::Nested(right)) => left.is_refined_by(right),
                _ => false,
            }
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            true
        }
    }

    macro_rules! define_partial_test_universe {
        // Defines one non-array universe whose conversion implementations are selected independently below.
        ($name:ident) => {
            /// Minimal type universe used to verify one reference primitive's required conversion boundary.
            #[derive(Clone, Debug, PartialEq, ryft_macros::Parameter)]
            pub(crate) enum $name {
                Value(TestReferent),
                Reference(ReferenceType<TestReferent>),
            }

            impl Display for $name {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        Self::Value(r#type) => Display::fmt(r#type, formatter),
                        Self::Reference(r#type) => Display::fmt(r#type, formatter),
                    }
                }
            }

            impl Type for $name {
                type Identity = TestIdentity;
                type Refinements = ();

                fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
                    match self {
                        Self::Value(r#type) => r#type.identities().collect::<Vec<_>>(),
                        Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
                    }
                    .into_iter()
                }

                fn rename_identities(
                    &self,
                    renaming: &TypeIdentityRenaming<Self::Identity>,
                ) -> Result<Self, TypeError> {
                    Ok(match self {
                        Self::Value(r#type) => Self::Value(r#type.rename_identities(renaming)?),
                        Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
                    })
                }

                fn is_compatible_with(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::Value(left), Self::Value(right)) => left.is_compatible_with(right),
                        (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                        _ => false,
                    }
                }

                fn is_refined_by(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::Value(left), Self::Value(right)) => left.is_refined_by(right),
                        (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                        _ => false,
                    }
                }

                fn is_scalar(&self) -> bool {
                    matches!(self, Self::Value(r#type) if r#type.is_scalar())
                }

                fn is_complex(&self) -> bool {
                    matches!(self, Self::Value(r#type) if r#type.is_complex())
                }

                fn is_reference(&self) -> bool {
                    matches!(self, Self::Reference(_))
                }
            }
        };
    }

    define_partial_test_universe!(NewUniverse);
    define_partial_test_universe!(ReadFreezeUniverse);
    define_partial_test_universe!(WriteUniverse);
    define_partial_test_universe!(SwapUniverse);
    define_partial_test_universe!(AddUpdateUniverse);

    impl From<ReferenceType<TestReferent>> for NewUniverse {
        fn from(r#type: ReferenceType<TestReferent>) -> Self {
            Self::Reference(r#type)
        }
    }

    impl<'t> TryFrom<&'t NewUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t NewUniverse) -> Result<Self, Self::Error> {
            match r#type {
                NewUniverse::Value(r#type) => Ok(r#type),
                NewUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl From<TestReferent> for ReadFreezeUniverse {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl<'t> TryFrom<&'t ReadFreezeUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t ReadFreezeUniverse) -> Result<Self, Self::Error> {
            match r#type {
                ReadFreezeUniverse::Reference(r#type) => Ok(r#type),
                ReadFreezeUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl<'t> TryFrom<&'t WriteUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t WriteUniverse) -> Result<Self, Self::Error> {
            match r#type {
                WriteUniverse::Value(r#type) => Ok(r#type),
                WriteUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t WriteUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t WriteUniverse) -> Result<Self, Self::Error> {
            match r#type {
                WriteUniverse::Reference(r#type) => Ok(r#type),
                WriteUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl From<TestReferent> for SwapUniverse {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl<'t> TryFrom<&'t SwapUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t SwapUniverse) -> Result<Self, Self::Error> {
            match r#type {
                SwapUniverse::Value(r#type) => Ok(r#type),
                SwapUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t SwapUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t SwapUniverse) -> Result<Self, Self::Error> {
            match r#type {
                SwapUniverse::Reference(r#type) => Ok(r#type),
                SwapUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl<'t> TryFrom<&'t AddUpdateUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t AddUpdateUniverse) -> Result<Self, Self::Error> {
            match r#type {
                AddUpdateUniverse::Value(r#type) => Ok(r#type),
                AddUpdateUniverse::Reference(_) => {
                    Err(TypeError::invalid("expected value type but got reference type"))
                }
            }
        }
    }

    impl<'t> TryFrom<&'t AddUpdateUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t AddUpdateUniverse) -> Result<Self, Self::Error> {
            match r#type {
                AddUpdateUniverse::Reference(r#type) => Ok(r#type),
                AddUpdateUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    pub(crate) type New = ReferenceNewOperation<TestReferent, TestType>;
    pub(crate) type Read = ReferenceReadOperation<TestReferent, TestType>;
    pub(crate) type Write = ReferenceWriteOperation<TestReferent, TestType>;
    pub(crate) type Swap = ReferenceSwapOperation<TestReferent, TestType>;
    pub(crate) type AddUpdate = ReferenceAddUpdateOperation<TestReferent, TestType>;
    pub(crate) type Freeze = ReferenceFreezeOperation<TestReferent, TestType>;

    // The fixtures below give the reference-primitive discharge rules a destination to write into. The universe is
    // deliberately view-less, so that these tests isolate the rules themselves: composed views are the policy's
    // concern and are covered where a universe with real view mechanics lives.
    //
    // Two destinations are named because the rules serve two kinds of allocation. A discharged reference's rewrite
    // reaches only ordinary values, so the eager destination executes it and the tests read the results directly. A
    // preserved reference is a *reference* of the destination universe, which an eager value of this fixture cannot
    // be, so the preserved replay is exercised against the staging destination and read back as the program it
    // recorded.

    /// Destination universe of the discharge-rule tests.
    pub(crate) type TestDestination = EagerContext<TestValue, TestOperation>;

    /// Discharge context over the discharge-rule test destination.
    pub(crate) type TestDischargeContext = ReferenceDischargeContext<TestDestination, TestReferenceDischarge>;

    /// Carrier flowing through the discharge-rule tests.
    pub(crate) type TestDischargeValue = ReferenceDischargeValue<TestDestination, TestReferenceDischarge>;

    /// Operation family of the discharge-rule test destinations.
    ///
    /// It carries the six reference primitives, because replaying an access to a preserved reference binds the access
    /// itself into the destination, and one addition, because that is how this universe's accumulation policy reaches
    /// its sum. A real family reaches the same shape through a dispatch derive.
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum TestOperation {
        Add,
        New(New),
        Read(Read),
        Write(Write),
        Swap(Swap),
        AddUpdate(AddUpdate),
        Freeze(Freeze),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for TestOperation {
        type Type = TestType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "test.add",
                Self::New(operation) => operation.name(),
                Self::Read(operation) => operation.name(),
                Self::Write(operation) => operation.name(),
                Self::Swap(operation) => operation.name(),
                Self::AddUpdate(operation) => operation.name(),
                Self::Freeze(operation) => operation.name(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[TestType],
            region_interfaces: &[RegionInterface<TestType>],
        ) -> Result<Vec<TestType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    if input_types[0] != input_types[1] {
                        return Err(TypeError::invalid(format!(
                            "`test.add` cannot add `{}` to `{}`",
                            input_types[1], input_types[0],
                        )));
                    }
                    Ok(vec![input_types[0].clone()])
                }
                Self::New(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Read(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Write(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Swap(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::AddUpdate(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Freeze(operation) => operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            match self {
                Self::Add => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                Self::New(operation) => operation.reference_semantics(),
                Self::Read(operation) => operation.reference_semantics(),
                Self::Write(operation) => operation.reference_semantics(),
                Self::Swap(operation) => operation.reference_semantics(),
                Self::AddUpdate(operation) => operation.reference_semantics(),
                Self::Freeze(operation) => operation.reference_semantics(),
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add => Effects::PURE,
                Self::New(operation) => operation.effects(),
                Self::Read(operation) => operation.effects(),
                Self::Write(operation) => operation.effects(),
                Self::Swap(operation) => operation.effects(),
                Self::AddUpdate(operation) => operation.effects(),
                Self::Freeze(operation) => operation.effects(),
            }
        }
    }

    // Only the addition is executable: a reference primitive reaches an eager destination exclusively as the replay of
    // an access to a preserved reference, and a preserved reference lives in the staging destination, which records
    // rather than executes.
    impl<C: Domain<Type = TestType, Value = TestValue>> InterpretableOperation<C> for TestOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => {
                    check_count!("input", inputs, 2, ProgramError);
                    Ok(vec![inputs[0].add(&inputs[1])?])
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before interpretation", self.name()),
                }),
            }
        }
    }

    macro_rules! impl_test_operation_from_reference_primitive {
        // Lifts one reference primitive into the test destination family, which is the conversion seam a rule spends
        // when it replays an access to a preserved reference.
        ($variant:ident, $payload:ident) => {
            impl From<$payload> for TestOperation {
                fn from(operation: $payload) -> Self {
                    Self::$variant(operation)
                }
            }
        };
    }

    impl_test_operation_from_reference_primitive!(New, New);
    impl_test_operation_from_reference_primitive!(Read, Read);
    impl_test_operation_from_reference_primitive!(Write, Write);
    impl_test_operation_from_reference_primitive!(Swap, Swap);
    impl_test_operation_from_reference_primitive!(AddUpdate, AddUpdate);
    impl_test_operation_from_reference_primitive!(Freeze, Freeze);

    // The family delegates each variant to the primitive rule that owns it, which is exactly what a dispatch derive
    // generates, and is what lets the program-level entry point drive these rules over this universe.
    impl<C, P> ReferenceDischargeableOperation<C, P> for TestOperation
    where
        C: Context<Type = TestType, Operation = TestOperation>,
        P: ReferenceAccumulationPolicy<C, Referent = TestReferent>,
    {
        fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
            &self,
            context: &ReferenceDischargeContext<C, P>,
            driver: &D,
            inputs: &[ReferenceDischargeValue<C, P>],
        ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
            match self {
                Self::Add => discharge_reference_free_operation(self, context, driver, inputs),
                Self::New(operation) => operation.discharge_references(context, driver, inputs),
                Self::Read(operation) => operation.discharge_references(context, driver, inputs),
                Self::Write(operation) => operation.discharge_references(context, driver, inputs),
                Self::Swap(operation) => operation.discharge_references(context, driver, inputs),
                Self::AddUpdate(operation) => operation.discharge_references(context, driver, inputs),
                Self::Freeze(operation) => operation.discharge_references(context, driver, inputs),
            }
        }
    }

    /// Destination value of the discharge-rule tests: one integer payload carrying its own referent type.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub(crate) struct TestValue {
        referent: TestReferent,
        payload: i64,
    }

    impl TestValue {
        /// Creates a test value with the provided referent type and integer payload.
        pub(crate) const fn new(referent: TestReferent, payload: i64) -> Self {
            Self { referent, payload }
        }
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}:{}", self.payload, self.referent)
        }
    }

    impl Parameter for TestValue {}

    impl Typed for TestValue {
        type Type = TestType;

        fn r#type(&self) -> Cow<'_, TestType> {
            Cow::Owned(TestType::Value(self.referent))
        }
    }

    impl Value for TestValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    impl Add for TestValue {
        fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
            if self.referent != rhs.referent {
                return Err(ProgramError::MalformedProgram(format!(
                    "cannot add `{}` to `{}`",
                    rhs.referent, self.referent,
                )));
            }
            Ok(Self::new(self.referent, self.payload + rhs.payload))
        }
    }

    /// View chain of the discharge-rule test universe, which has no interior structure to select.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub(crate) struct TestAlias;

    /// Reference discharge policy of the discharge-rule test universe.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct TestReferenceDischarge;

    // The policy leaves the destination value generic, which is what lets one implementation serve both the eager and
    // the staging destination: a view-less universe needs no destination capability at all to read or replace, and it
    // reaches its sum by binding this family's addition rather than by requiring value-level arithmetic.
    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
        for TestReferenceDischarge
    {
        type Referent = TestReferent;
        type Alias = TestAlias;

        fn storage_alias(_referent: &TestReferent) -> TestAlias {
            TestAlias
        }

        fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
            Ok(current.clone())
        }

        fn write(
            _context: &C,
            _current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            Ok(replacement)
        }
    }

    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceAccumulationPolicy<C>
        for TestReferenceDischarge
    {
        fn accumulate(
            context: &C,
            current: &C::Value,
            update: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            let mut outputs = context.bind(TestOperation::Add, Vec::new(), &[current.clone(), update])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }
    }

    /// Reference policy that deliberately supports write discharge without supporting accumulation.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct WriteOnlyReferenceDischarge;

    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
        for WriteOnlyReferenceDischarge
    {
        type Referent = TestReferent;
        type Alias = TestAlias;

        fn storage_alias(_referent: &TestReferent) -> TestAlias {
            TestAlias
        }

        fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
            Ok(current.clone())
        }

        fn write(
            _context: &C,
            _current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            Ok(replacement)
        }

        fn swap(
            _context: &C,
            _current: &C::Value,
            _replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<(C::Value, C::Value), ProgramError> {
            Err(ProgramError::MalformedProgram("write-only discharge policy must not swap".to_string()))
        }
    }

    /// Handle to one live allocation in the discharge-rule test universe.
    pub(crate) type TestDischargeReference = ReferenceDischargeReference<TestDestination, TestReferenceDischarge>;

    /// Referent every discharge-rule test allocates its allocation over.
    pub(crate) const REFERENT: TestReferent = TestReferent::new(7, 16);

    /// Referent whose canonical reference projection deliberately fails to exercise the allocation diagnostic.
    pub(crate) const NON_PROJECTING_REFERENT: TestReferent = TestReferent::new(7, u8::MAX);

    /// Allocates a reference containing `payload` through the allocation discharge rule.
    ///
    /// Returns the discharge context together with the handle denoting the new allocation.
    pub(crate) fn allocated_allocation(payload: i64) -> (TestDischargeContext, TestDischargeReference) {
        let context = TestDischargeContext::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, payload));
        let allocated = New::new().discharge_references(&context, &EmptyRegionDriver, &[initial]).unwrap();
        let reference = allocated[0].expect_reference("the allocated allocation").unwrap().clone();
        (context, reference)
    }

    /// Asserts that `parameter` round-trips through its parameter structure.
    pub(crate) fn assert_parameter_roundtrip<P>(parameter: P)
    where
        P: Copy + Debug + PartialEq + Parameter,
    {
        let structure = <P as Parameterized<P>>::parameter_structure(&parameter);
        assert_eq!(<P as Parameterized<P>>::from_parameters(structure, [parameter]), Ok(parameter));
    }

    #[test]
    fn test_reference_primitive_discharge_replays_accesses_to_a_preserved_allocation() {
        // An allocation that partial discharge preserved survives in the destination as an ordinary reference, so the
        // dispatch path replays every access verbatim over the handle's destination value instead of acting on
        // threaded state, and the access rules themselves never run. The rewritten program therefore performs the
        // same reference operations the source did, in the same order, and the consumed allocation contributes no
        // binding.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(TestType::Reference(ReferenceType::new(REFERENT)));
        let update = builder.add_input(TestType::Value(REFERENT));
        let observed = builder
            .add_instruction(TestOperation::Read(Read::new()), Vec::new(), vec![reference], None)
            .unwrap()[0];
        builder
            .add_instruction(TestOperation::Write(Write::new()), Vec::new(), vec![reference, update], None)
            .unwrap();
        let previous = builder
            .add_instruction(TestOperation::Swap(Swap::new()), Vec::new(), vec![reference, update], None)
            .unwrap()[0];
        builder
            .add_instruction(TestOperation::AddUpdate(AddUpdate::new()), Vec::new(), vec![reference, update], None)
            .unwrap();
        let frozen = builder
            .add_instruction(TestOperation::Freeze(Freeze::new()), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let preserved = source.partially_discharge_references_with_policy::<TestReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.output_count(), 3);
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<value<i7,p16>>, %1:value<i7,p16> .
                let %2:value<i7,p16> = reference_read %0
                    reference_write %0 %1
                    %3:value<i7,p16> = reference_swap %0 %1
                    reference_add_update %0 %1
                    %4:value<i7,p16> = reference_freeze %0
                in (%2, %3, %4)"},
        );
    }
}
