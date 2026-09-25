//! Operations over _references_ (i.e., handles to mutable state that programs allocate, read, update in place, and
//! finally consume). A reference has a [`ReferenceType`] naming the referent it stores. Reads and updates declare
//! reference effects, so a program that touches state keeps its accesses in order and its reference discharge can
//! later turn that state into explicit dataflow for backends without mutable buffers.
//!
//! The core operations are:
//!
//!   - [`ReferenceNew`], which allocates a reference holding an initial value, and [`ReferenceFreeze`],
//!     which consumes the reference and returns its final value.
//!   - [`ReferenceRead`], which copies the current value out, and [`ReferenceWrite`] and [`ReferenceSwap`],
//!     which replace it, with the swap also returning the previous value.
//!   - [`ReferenceAddUpdate`], which adds an update into the referent, and [`ReferenceAtomicAddUpdate`],
//!     its variant for updates that may race (e.g., within a kernel implementation).
//!
//! Array references additionally support _views_, which derive a narrower reference to the same allocation without
//! accessing its state: [`ReferenceIndex`] and [`ReferenceDynamicIndex`] select one element on an axis (at a static
//! index and at an index supplied as a value, respectively) and remove that axis, and [`ReferenceSlice`] selects one
//! static range on every axis. Each view is described by an [`ArrayReferenceView`], and reads and updates through a
//! view touch only the elements it selects.
//!
//! Every function is a value capability on the reference handle, so the same code updates eager state immediately
//! and records reference instructions when the handle is a tracer.
//!
//! # Examples
//!
//! Eager references hold a value that updates mutate in place:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayIrValue, ProgramError, ReferenceAddUpdate, ReferenceFreeze, ReferenceNew, ReferenceRead,
//! #     ReferenceWrite,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! let buffer = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])?).reference_new()?;
//! buffer.add_update(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0])?))?;
//! assert_eq!(buffer.read()?, ArrayIrValue::Array(Array::vector(vec![11.0_f32, 22.0])?));
//! buffer.write(&ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0])?))?;
//! assert_eq!(buffer.freeze()?, ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0])?));
//! # Ok(())
//! # }
//! ```
//!
//! Traced references record the same steps as instructions, and interpreting the program replays them:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, EagerContext, ProgramError,
//! #     ReferenceAddUpdate, ReferenceFreeze, ReferenceNew, Trace,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
//!     |input| {
//!         let buffer = input.reference_new()?;
//!         buffer.add_update(&input)?;
//!         buffer.freeze()
//!     },
//!     ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
//! )?;
//! assert_eq!(
//!     program.interpret(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])?))?,
//!     ArrayIrValue::Array(Array::vector(vec![2.0_f32, 4.0])?),
//! );
//! # Ok(())
//! # }
//! ```

mod reference_add_update;
mod reference_atomic_add_update;
mod reference_dynamic_index;
mod reference_freeze;
mod reference_index;
mod reference_new;
mod reference_read;
mod reference_slice;
mod reference_swap;
mod reference_write;

pub use reference_add_update::{REFERENCE_ADD_UPDATE_OPERATION_NAME, ReferenceAddUpdate, ReferenceAddUpdateOperation};
pub use reference_atomic_add_update::{
    REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME, ReferenceAtomicAddUpdate, ReferenceAtomicAddUpdateOperation,
};
pub use reference_dynamic_index::{
    REFERENCE_DYNAMIC_INDEX_OPERATION_NAME, ReferenceDynamicIndex, ReferenceDynamicIndexOperation,
};
pub use reference_freeze::{REFERENCE_FREEZE_OPERATION_NAME, ReferenceFreeze, ReferenceFreezeOperation};
pub use reference_index::{REFERENCE_INDEX_OPERATION_NAME, ReferenceIndex, ReferenceIndexOperation};
pub use reference_new::{REFERENCE_NEW_OPERATION_NAME, ReferenceNew, ReferenceNewOperation};
pub use reference_read::{REFERENCE_READ_OPERATION_NAME, ReferenceRead, ReferenceReadOperation};
pub use reference_slice::{REFERENCE_SLICE_OPERATION_NAME, ReferenceSlice, ReferenceSliceOperation};
pub use reference_swap::{REFERENCE_SWAP_OPERATION_NAME, ReferenceSwap, ReferenceSwapOperation};
pub use reference_write::{REFERENCE_WRITE_OPERATION_NAME, ReferenceWrite, ReferenceWriteOperation};

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::fmt::{Debug, Display};

    use ryft_macros::Parameter;

    use crate::contexts::{Context, Domain, EagerContext};
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::{Add, AddOperation};
    use crate::parameters::Parameter;
    use crate::programs::{
        Effects, EmptyRegionDriver, Operation, ProgramError, ReferenceAccumulationPolicy, ReferenceDischargeContext,
        ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceDischargeValue,
        ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceType, RegionInterface, Type, TypeError,
        TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming, Typed, Value, discharge_reference_free_operation,
    };

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

    /// Referent type used to exercise generic reference operations without array-specific behavior.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Parameter)]
    pub(crate) struct TestReferent {
        /// Identity preserved by reference operations.
        pub(crate) identity: TestIdentity,

        /// Precision used to exercise referent compatibility and exact update type checks.
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

    macro_rules! define_partial_test_universe {
        // Defines a shared test universe; conversions are supplied separately for each operation.
        ($name:ident) => {
            /// Minimal type universe used to verify reference operations' required conversions.
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

    define_partial_test_universe!(SwapUniverse);

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

    define_partial_test_universe!(ReadFreezeUniverse);
    define_partial_test_universe!(StoreUniverse);

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

    impl<'t> TryFrom<&'t StoreUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t StoreUniverse) -> Result<Self, Self::Error> {
            match r#type {
                StoreUniverse::Value(r#type) => Ok(r#type),
                StoreUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t StoreUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t StoreUniverse) -> Result<Self, Self::Error> {
            match r#type {
                StoreUniverse::Reference(r#type) => Ok(r#type),
                StoreUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    pub(crate) type New = ReferenceNewOperation<TestReferent, TestType>;
    pub(crate) type Read = ReferenceReadOperation<TestReferent, TestType>;
    pub(crate) type Write = ReferenceWriteOperation<TestReferent, TestType>;
    pub(crate) type Swap = ReferenceSwapOperation<TestReferent, TestType>;
    pub(crate) type AddUpdate = ReferenceAddUpdateOperation<TestReferent, TestType>;
    pub(crate) type Freeze = ReferenceFreezeOperation<TestReferent, TestType>;

    // These fixtures exercise discharge without array or view behavior. Addition executes eagerly; program-level
    // tests stage preserved reference operations.

    /// Destination universe of the discharge-rule tests.
    pub(crate) type TestDestination = EagerContext<TestValue, TestOperation>;

    /// Discharge context over the discharge-rule test destination.
    pub(crate) type TestDischargeContext = ReferenceDischargeContext<TestDestination, TestReferenceDischarge>;

    /// Carrier flowing through the discharge-rule tests.
    pub(crate) type TestDischargeValue = ReferenceDischargeValue<TestDestination, TestReferenceDischarge>;

    /// Operation family required by the discharge rules; only addition executes in the eager destination.
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

        fn effects(&self) -> Cow<'_, Effects> {
            match self {
                Self::Add => Cow::Borrowed(Effects::empty()),
                Self::New(operation) => operation.effects(),
                Self::Read(operation) => operation.effects(),
                Self::Write(operation) => operation.effects(),
                Self::Swap(operation) => operation.effects(),
                Self::AddUpdate(operation) => operation.effects(),
                Self::Freeze(operation) => operation.effects(),
            }
        }
    }

    // Allocation-preservation tests replay whole programs through this operation family.
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
        // Supplies the operation conversion required by each discharge rule's context bound.
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

    /// Destination value of the discharge-rule tests: one integer payload carrying its own referent type.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub(crate) struct TestValue {
        /// Type carried by this value.
        referent: TestReferent,

        /// Integer stored in the discharged reference state.
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

    impl ReferenceDischargeableType for TestType {
        type Policy = TestReferenceDischarge;
    }

    // Reads and replacements need no value-level capabilities; accumulation binds the test addition operation.
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

    /// Default referent type used by the discharge-rule tests.
    pub(crate) const REFERENT: TestReferent = TestReferent::new(7, 16);

    /// Referent whose canonical reference projection deliberately fails to exercise the allocation diagnostic.
    pub(crate) const NON_PROJECTING_REFERENT: TestReferent = TestReferent::new(7, u8::MAX);

    /// Allocates a reference containing `payload` through the allocation discharge rule.
    /// Returns the discharge context together with the handle denoting the new allocation.
    pub(crate) fn allocated_reference(
        payload: i64,
    ) -> (TestDischargeContext, ReferenceDischargeReference<TestDestination, TestReferenceDischarge>) {
        let context = TestDischargeContext::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Value(TestValue::new(REFERENT, payload));
        let allocated = New::new().discharge_references(&context, &EmptyRegionDriver, &[initial]).unwrap();
        let reference = allocated[0].try_as_reference("the allocated reference").unwrap().clone();
        (context, reference)
    }
}
