//! Contains machinery related to _reference discharge_, which is the process of rewriting mutable reference state into
//! explicit immutable dataflow. Specifically, a program containing reference operations is not ordinary functional
//! Single Static Assignment (SSA) dataflow. A read operation depends on the latest write operation to the same
//! allocation even though that dependency is represented by a reference handle rather than by an SSA operand carrying
//! the current value. Many transforms and backends require the dependency to be explicit. Reference discharge makes it
//! explicit by replaying the program while replacing each selected allocation with an immutable state value that is
//! threaded from one access to the next.
//!
//! # Example
//!
//! Consider this schematic program with one local reference:
//!
//! ```text
//! %reference = reference_new(%initial)
//! %before = reference_swap(%reference, %replacement)
//! %after = reference_read(%reference)
//! %final = reference_freeze(%reference)
//! return %before, %after, %final
//! ```
//!
//! Discharge removes the mutable allocation and exposes the same dependencies as immutable values:
//!
//! ```text
//! %state0 = %initial
//! %before = %state0
//! %state1 = %replacement
//! %after = %state1
//! %final = %state1
//! return %before, %after, %final
//! ```
//!
//! The exact rewritten program may simplify aliases such as `%state0`, but the semantic relationship is the same:
//! replacement returns the previous state and produces a successor state, while every subsequent access consumes that
//! successor. The reference is an implementation detail of the source program, and no reference survives in the full
//! result.
//!
//! An external reference follows the same rewrite but its state crosses the program boundary. Its reference-typed input
//! becomes an ordinary input carrying the entering referent. If the program mutates the reference, its final state is
//! appended after the public outputs as a hidden output. [`ExternalReferenceBinding`] records which capture or public
//! argument owns that state and which hidden output must replace the caller's reference value. Its discharged input
//! position follows from that logical source and the result's capture count. A read-only external reference has no
//! hidden output.
//!
//! Discharge rewrites a program. It does not itself lock eager reference state or execute a backend. The stateful
//! compilation surface uses the result's binding metadata together with the runtime reference protocol after
//! compilation.
//!
//! # Using Reference Discharge
//!
//! Callers normally invoke [`Program::discharge_references`](crate::Program::discharge_references) to eliminate every
//! reference, or [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) when
//! selected allocations should become explicit state while
//! other allocations remain references. The full entry point returns a [`ReferenceDischargeResult`], whose program is
//! proven reference-free. The partial entry point returns a [`PartialReferenceDischargeResult`], which describes only
//! the discharged references and can be converted into a full result after proving that no references remain. A
//! caller that needs a bare program with no caller-owned reference bindings converts a full result through
//! [`ReferenceDischargeResult::into_program_without_external_references`].
//!
//! A reference universe participates by implementing [`ReferenceDischargePolicy`], selecting that policy through
//! [`ReferenceDischargeableType`], and, when supported, implementing [`ReferenceAccumulationPolicy`]. Each operation
//! implements [`ReferenceDischargeableOperation`] to rewrite its own reference effects. Region-free operations can
//! delegate to [`discharge_reference_free_operation`]; structured operations use [`ReferenceDischargeDriver`],
//! [`ReferenceRegionSummary`], and
//! [`ReferenceRegionDischargeBoundary`] to rebuild attached regions with the necessary state positions.
//!
//! # Full and Partial Discharge
//!
//! [`ReferenceDischargeResult`] is the full-discharge contract. Its program is proven reference-free across the
//! complete attached-region closure, its public outputs form a prefix of its complete outputs, and the remaining
//! suffix contains exactly the final states of mutated external allocations in canonical boundary order.
//!
//! [`PartialReferenceDischargeResult`] permits selected allocations to become immutable state while unselected references
//! and their operations remain in the program. Callers select external references or internal allocations through
//! [`ReferenceDischargeTarget`]. This is useful when normalizing a pipeline's internal state while deliberately
//! preserving references that a kernel will lower to target memory operations. Conversion from a partial result to a
//! full result performs a closure-wide proof that no reference type or reference operation remains.
//!
//! # Interpreter Architecture
//!
//! Discharge follows Ryft's context-and-per-operation-rule transform architecture:
//!
//! - [`ReferenceDischargePolicy`] names everything that varies by reference universe: the referent type family, the
//!   handle's composed alias metadata, type lifting and projection, and the mechanics of reading and replacing a
//!   selected value. [`ReferenceAccumulationPolicy`] adds ordered accumulation only for universes that support it.
//! - [`ReferenceDischargeValue`] is the context-free carrier flowing between rules. It contains either an ordinary
//!   destination value or an opaque [`ReferenceDischargeReference`] handle.
//! - [`ReferenceDischargeContext`] owns the live allocation environment. Each allocation is either `Discharged`, with a
//!   current immutable state and mutation bit, or `Preserved`, with the exact destination reference value that survived.
//! - [`ReferenceDischargeableOperation`] is the rule implemented by each operation. Reference primitives rewrite
//!   their own *discharged* accesses, structured operations own their boundary widening, and reference-free
//!   operations replay unchanged through the parent context. Accesses to preserved references never reach a rule: the
//!   replay path itself replays every region-free, access-only application over exclusively preserved references
//!   verbatim.
//! - [`ReferenceDischargeDriver`] exposes the current source instruction and attached regions. It can replay a region
//!   against the live environment or rebuild one against an isolated environment and return a sealed
//!   [`ReferenceRegionDischargeFork`]. [`ReferenceRegionSummary`] supplies the transitive access facts a structured
//!   rule needs before choosing its state boundary.
//!
//! The driver provides shared mechanics but never chooses how an operation is rewritten. This keeps the system open:
//! a third-party primitive or structured operation participates by implementing its own rule, while a non-array
//! value family supplies its own policy without changing this interpreter.
//!
//! # Allocation Identities and Boundaries
//!
//! [`ReferenceDischargeTarget`] names a source program location used before replay to select an external reference or
//! a locally allocated reference. [`ReferenceDischargeAllocationId`] is different: it is a temporary identity minted
//! inside one live discharge environment. IDs from isolated region forks cannot address parent allocations, and fork
//! results carry sealed programs and context-free summaries rather than child-context values.
//!
//! Structured rules use [`ReferenceRegionDischargeBoundary`] to describe their declared inputs plus the discharged
//! allocations that must enter and leave a rebuilt region. Read-only allocations are pruned where the operation's boundary permits
//! it; loop-shaped operations retain the symmetry their fixed-point contracts require. Every rebuilt region is
//! validated against the allocations and mutations its summary predicted before the parent environment accepts its outputs.
//!
//! # End-to-End Flow
//!
//! 1. The program entry point validates the selected targets and binds each external reference as either discharged
//!    state or a preserved reference in a new [`ReferenceDischargeContext`].
//! 2. The driver replays each instruction. An access involving only preserved references is replayed unchanged; every
//!    other application dispatches to its [`ReferenceDischargeableOperation`] rule.
//! 3. A rule reads or updates the allocation environment. A structured rule may first summarize an attached region, widen
//!    its boundary, and rebuild it in an isolated context before merging the resulting state into its caller.
//! 4. The transform reconstructs the program's public outputs, appends one hidden final-state output for each mutated
//!    external allocation, validates the complete boundary, and returns the appropriate result envelope.
//!
//! # Glossary
//!
//! - An **external source** is the capture or public input through which caller-owned reference state enters, named by
//!   [`ReferenceSource`].
//! - A **discharge target** is a stable source program location used to select an external reference or internal
//!   allocation for partial discharge, represented by [`ReferenceDischargeTarget`].
//! - An **allocation identity** is the temporary [`ReferenceDischargeAllocationId`] by which one running interpreter
//!   identifies a reference allocation. It does not identify a source program location and cannot cross between
//!   isolated environments.
//! - A **discharged reference** is represented by its current immutable state. A **preserved reference** remains
//!   represented by a reference value and its reference operations are replayed rather than rewritten.
//! - A **carrier** is a [`ReferenceDischargeValue`]: either an ordinary destination value or a temporary reference
//!   handle passed between operation rules.
//! - A summary allocation is **reached** when a region's closure accesses, returns, or otherwise rematerializes it. It is
//!   **accessed** only when the closure performs a semantic reference access on it. Reached allocations may need to cross a
//!   rebuilt boundary even when they are read nowhere inside the region.
//! - A widening's **threaded** allocations cross a structured boundary, its **entering** allocations require added inputs because
//!   no declared input already carries them, and its **published** allocations require added outputs because the region may
//!   mutate them and no declared output already publishes their state.

// TODO(eaplatanios): Review this module.

mod interpreter;
mod policies;
mod results;
mod targets;
mod transform;

pub use interpreter::{
    RecursiveReferenceDischargeDriver, ReferenceDischargeDriver, ReferenceDischargeRegionDestination,
    ReferenceDischargeableOperation, ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork,
    ReferenceRegionStateInsertion, ReferenceRegionSummary, ReferenceStateWidening,
    discharge_positional_region_operation, discharge_preserved_access, discharge_reference_free_operation,
};
pub use policies::{ReferenceAccumulationPolicy, ReferenceDischargePolicy, ReferenceDischargeableType};
pub use results::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, ReferenceDischargeResult, ReferenceSource,
};
pub use targets::ReferenceDischargeTarget;
pub use transform::{
    ReferenceDischargeAllocationId, ReferenceDischargeContext, ReferenceDischargeReference, ReferenceDischargeValue,
};

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::fmt::Display;

    use crate::contexts::{Context, Domain, EagerContext};
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::Add;
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::identities::NoIdentity;
    use crate::programs::instructions::InstructionId;
    use crate::programs::operations::Operation;
    use crate::programs::programs::Program;

    use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
    use crate::programs::types::{Type, TypeError, Typed};
    use crate::programs::values::Value;

    use crate::captures::CaptureReference;

    use super::super::semantics::{
        ReferenceAccessMode, ReferenceAliasKind, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput,
    };
    use super::super::types::ReferenceType;
    use super::*;

    /// Minimal generic type universe for the boundary tests below: opaque indexed values plus references over them.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum TestType {
        Value(u8),
        Reference(Box<ReferenceType<TestType>>),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Value(index) => write!(formatter, "value<{index}>"),
                Self::Reference(reference) => Display::fmt(reference, formatter),
            }
        }
    }

    impl Parameter for TestType {}

    impl Type for TestType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Constant payload of the minimal universe. The boundary tests never materialize concrete values, so the capture
    /// reference stand-in is all they need.
    pub(crate) type TestValue = CaptureReference<TestType>;

    /// Returns a reference type over the opaque value type with the given index.
    pub(crate) fn reference_type(index: u8) -> TestType {
        TestType::Reference(Box::new(ReferenceType::new(TestType::Value(index))))
    }

    /// Minimal generic operation universe for the boundary tests below: allocation, read, and consumption of a
    /// reference, plus one positional call-like region operation.
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum TestOperation {
        NewAllocation,
        Read,
        Consume,
        Call,
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
                Self::NewAllocation => "test.new_allocation",
                Self::Read => "test.read",
                Self::Consume => "test.consume",
                Self::Call => "test.call",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Call => const { &[RegionSlot::computation("callee")] },
                _ => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[TestType],
            region_interfaces: &[RegionInterface<TestType>],
        ) -> Result<Vec<TestType>, TypeError> {
            match self {
                Self::NewAllocation => {
                    Ok(vec![TestType::Reference(Box::new(ReferenceType::new(input_types[0].clone())))])
                }
                Self::Read | Self::Consume => match input_types.first() {
                    Some(TestType::Reference(reference)) => Ok(vec![reference.referent().clone()]),
                    _ => Err(TypeError::invalid("test operation expected a reference input")),
                },
                Self::Call => Ok(region_interfaces[0].output_types().to_vec()),
            }
        }

        fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<usize> {
            matches!(self, Self::Call).then_some(input_index)
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                _ => Vec::new(),
            }
        }

        fn allows_reference_access_through_region_input(
            &self,
            _region_index: usize,
            mode: ReferenceAccessMode,
        ) -> bool {
            matches!(self, Self::Call) && mode != ReferenceAccessMode::Consume
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Call => Effects::PURE,
                _ => Effects::single(Effect::OrderedState),
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            let semantics = match self {
                Self::NewAllocation => {
                    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }])
                }
                Self::Read => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                    Vec::new(),
                ),
                Self::Consume => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)],
                    Vec::new(),
                ),
                Self::Call => ReferenceOperationSemantics::default(),
            };
            Cow::Owned(semantics)
        }
    }

    // The fixtures below are the non-array prototype universe shared by the interpreter tests: a deliberately small
    // reference universe whose referents are fixed-length integer lists and whose views are contiguous sub-ranges. It
    // is the standing proof that the discharge architecture has not silently become array-shaped, because nothing in
    // it mentions arrays and its alias mechanics are real rather than trivial.

    thread_local! {
        pub(crate) static OBSERVED_ALLOCATION_POSITIONS: RefCell<Vec<Option<InstructionId>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Destination universe of the prototype programs.
    pub(crate) type ListDestination = EagerContext<ListIrValue, ListOperation>;

    /// Discharge context over the prototype destination universe.
    pub(crate) type ListDischargeContext = ReferenceDischargeContext<ListDestination, ListReferenceDischarge>;

    /// Carrier flowing through prototype discharge.
    pub(crate) type ListDischargeValue = ReferenceDischargeValue<ListDestination, ListReferenceDischarge>;

    /// Referent type of the prototype universe: a list of integers with a fixed length.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct ListType {
        pub(crate) length: usize,
    }

    impl Display for ListType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "list<{}>", self.length)
        }
    }

    impl Parameter for ListType {}

    impl Type for ListType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            self.length == 1
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    /// Type universe of the prototype programs, pairing ordinary lists with references to them.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum ListIrType {
        List(ListType),
        Reference(ReferenceType<ListType>),
    }

    impl Display for ListIrType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::List(r#type) => Display::fmt(r#type, formatter),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ListIrType {}

    impl From<ListType> for ListIrType {
        fn from(r#type: ListType) -> Self {
            Self::List(r#type)
        }
    }

    impl From<ReferenceType<ListType>> for ListIrType {
        fn from(r#type: ReferenceType<ListType>) -> Self {
            Self::Reference(r#type)
        }
    }

    impl<'t> TryFrom<&'t ListIrType> for &'t ReferenceType<ListType> {
        type Error = TypeError;

        fn try_from(r#type: &'t ListIrType) -> Result<Self, Self::Error> {
            match r#type {
                ListIrType::Reference(r#type) => Ok(r#type),
                ListIrType::List(_) => Err(TypeError::invalid("expected reference type but got list type")),
            }
        }
    }

    impl Type for ListIrType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            matches!(self, Self::List(r#type) if r#type.is_scalar())
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Value universe of the prototype programs.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum ListIrValue {
        /// Concrete list payload.
        List(Vec<i64>),

        /// Reference value surviving in a destination program. The prototype universe has no runtime allocation behind
        /// it, which is what makes it a good test of the machinery that must never look inside one.
        Reference(ReferenceType<ListType>),
    }

    impl ListIrValue {
        /// Returns the list payload of this value, or an error when it is a reference.
        fn list(&self) -> Result<&[i64], ProgramError> {
            match self {
                Self::List(elements) => Ok(elements.as_slice()),
                Self::Reference(r#type) => {
                    Err(ProgramError::MalformedProgram(format!("expected a list but got `{}`", r#type)))
                }
            }
        }
    }

    impl Display for ListIrValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::List(elements) => write!(formatter, "{elements:?}"),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ListIrValue {}

    impl Typed for ListIrValue {
        type Type = ListIrType;

        fn r#type(&self) -> Cow<'_, ListIrType> {
            match self {
                Self::List(elements) => Cow::Owned(ListIrType::List(ListType { length: elements.len() })),
                Self::Reference(r#type) => Cow::Owned(ListIrType::Reference(r#type.clone())),
            }
        }
    }

    impl Value for ListIrValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    impl Add for ListIrValue {
        fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
            let (lhs, rhs) = (self.list()?, rhs.list()?);
            if lhs.len() != rhs.len() {
                return Err(ProgramError::MalformedProgram(format!(
                    "cannot add lists of lengths {} and {}",
                    lhs.len(),
                    rhs.len(),
                )));
            }
            Ok(Self::List(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect()))
        }
    }

    /// View chain of the prototype universe: one contiguous sub-range of the allocation list.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub(crate) struct ListAlias {
        pub(crate) offset: usize,
        pub(crate) length: usize,
    }

    // Binds one single-result prototype operation into a destination. Routing the alias mechanics through the
    // operation family is what keeps this universe's policy independent of whether its destination executes the work
    // or stages it, which one policy implementation must be for a rebuilt region to be traced at all.
    fn bind_list<C: Context<Type = ListIrType, Operation: From<ListOperation>>>(
        context: &C,
        operation: ListOperation,
        inputs: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        let mut outputs = context.bind(operation, Vec::new(), inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Reference discharge policy of the prototype universe.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct ListReferenceDischarge;

    impl ReferenceDischargeableType for ListIrType {
        type Policy = ListReferenceDischarge;
    }

    // The policy leaves the destination value generic and reaches its alias mechanics through the operation family,
    // which is what one implementation must do to serve both the eager destination this universe's own tests use and
    // the fresh staging destination a rebuilt region is traced into.
    impl<C: Context<Type = ListIrType, Operation: From<ListOperation>>> ReferenceDischargePolicy<C>
        for ListReferenceDischarge
    {
        type Referent = ListType;
        type Alias = ListAlias;

        fn storage_alias(referent: &ListType) -> ListAlias {
            ListAlias { offset: 0, length: referent.length }
        }

        fn read(context: &C, current: &C::Value, alias: &ListAlias) -> Result<C::Value, ProgramError> {
            bind_list(context, ListOperation::Select { offset: alias.offset, length: alias.length }, &[current.clone()])
        }

        fn write(
            context: &C,
            current: &C::Value,
            replacement: C::Value,
            alias: &ListAlias,
        ) -> Result<C::Value, ProgramError> {
            bind_list(context, ListOperation::Splice { offset: alias.offset }, &[current.clone(), replacement])
        }
    }

    // The prototype universe accumulates by lifting its own addition into the destination, which is the shape a
    // universe whose values carry no arithmetic capability of their own uses.
    impl<C: Context<Type = ListIrType, Operation: From<ListOperation>>> ReferenceAccumulationPolicy<C>
        for ListReferenceDischarge
    {
        fn accumulate(
            context: &C,
            current: &C::Value,
            update: C::Value,
            alias: &ListAlias,
        ) -> Result<C::Value, ProgramError> {
            let selected = Self::read(context, current, alias)?;
            let accumulated = bind_list(context, ListOperation::Add, &[selected, update])?;
            bind_list(context, ListOperation::Splice { offset: alias.offset }, &[current.clone(), accumulated])
        }
    }

    /// Operation family of the prototype universe.
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum ListOperation {
        Add,
        Select { offset: usize, length: usize },
        Splice { offset: usize },
        ReferenceNew,
        Slice { offset: usize, length: usize },
        Read,
        Write,
        Swap,
        AddUpdate,
        Freeze,
        UnreportedFreeze,
        Call,
    }

    impl Display for ListOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for ListOperation {
        type Type = ListIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "list.add",
                Self::Select { .. } => "list.select",
                Self::Splice { .. } => "list.splice",
                Self::ReferenceNew => "list.reference_new",
                Self::Slice { .. } => "list.slice",
                Self::Read => "list.read",
                Self::Write => "list.write",
                Self::Swap => "list.swap",
                Self::AddUpdate => "list.add_update",
                Self::Freeze => "list.freeze",
                Self::UnreportedFreeze => "test.unreported_freeze",
                Self::Call => "list.call",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Call => const { &[RegionSlot::computation("callee")] },
                _ => &[],
            }
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
            (matches!(self, Self::Call) && region_index == 0).then_some(input_index)
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                _ => Vec::new(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ListIrType],
            region_interfaces: &[RegionInterface<ListIrType>],
        ) -> Result<Vec<ListIrType>, TypeError> {
            let referent = |index: usize| match input_types.get(index) {
                Some(ListIrType::Reference(reference)) => Ok(reference.referent().clone()),
                _ => Err(TypeError::invalid(format!("`{}` expects a reference operand", self.name()))),
            };
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Select { offset, length } => {
                    check_count!("input", input_types, 1, TypeError);
                    let ListIrType::List(source) = &input_types[0] else {
                        return Err(TypeError::invalid("`list.select` expects a list operand"));
                    };
                    if offset + length > source.length {
                        return Err(TypeError::invalid(format!(
                            "selection [{offset}, {}) does not fit `{source}`",
                            offset + length,
                        )));
                    }
                    Ok(vec![ListIrType::List(ListType { length: *length })])
                }
                Self::Splice { offset } => {
                    check_count!("input", input_types, 2, TypeError);
                    let (ListIrType::List(target), ListIrType::List(update)) = (&input_types[0], &input_types[1])
                    else {
                        return Err(TypeError::invalid("`list.splice` expects two list operands"));
                    };
                    if offset + update.length > target.length {
                        return Err(TypeError::invalid(format!(
                            "splice [{offset}, {}) does not fit `{target}`",
                            offset + update.length,
                        )));
                    }
                    Ok(vec![input_types[0].clone()])
                }
                Self::ReferenceNew => {
                    check_count!("input", input_types, 1, TypeError);
                    let ListIrType::List(referent) = &input_types[0] else {
                        return Err(TypeError::invalid("`list.reference_new` expects a list operand"));
                    };
                    Ok(vec![ListIrType::Reference(ReferenceType::new(referent.clone()))])
                }
                Self::Slice { offset, length } => {
                    check_count!("input", input_types, 1, TypeError);
                    let referent = referent(0)?;
                    if offset + length > referent.length {
                        return Err(TypeError::invalid(format!(
                            "view [{offset}, {}) does not fit `{referent}`",
                            offset + length,
                        )));
                    }
                    Ok(vec![ListIrType::Reference(ReferenceType::new(ListType { length: *length }))])
                }
                Self::Read | Self::Freeze | Self::UnreportedFreeze => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ListIrType::List(referent(0)?)])
                }
                Self::Write => {
                    check_count!("input", input_types, 2, TypeError);
                    referent(0)?;
                    Ok(Vec::new())
                }
                Self::Swap => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![ListIrType::List(referent(0)?)])
                }
                Self::AddUpdate => {
                    check_count!("input", input_types, 2, TypeError);
                    referent(0)?;
                    Ok(Vec::new())
                }
                Self::Call => {
                    check_count!("region", region_interfaces, 1, TypeError);
                    check_count!("input", input_types, region_interfaces[0].input_types().len(), TypeError);
                    Ok(region_interfaces[0].output_types().to_vec())
                }
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            let semantics = match self {
                Self::ReferenceNew => {
                    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }])
                }
                Self::Slice { .. } => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
                ),
                Self::Read => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                    Vec::new(),
                ),
                Self::Write => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Write)],
                    Vec::new(),
                ),
                Self::Swap => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)],
                    Vec::new(),
                ),
                Self::AddUpdate => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Accumulate)],
                    Vec::new(),
                ),
                Self::Freeze => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)],
                    Vec::new(),
                ),
                Self::Add | Self::Select { .. } | Self::Splice { .. } | Self::UnreportedFreeze | Self::Call => {
                    return Cow::Borrowed(ReferenceOperationSemantics::empty());
                }
            };
            Cow::Owned(semantics)
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add | Self::Select { .. } | Self::Splice { .. } | Self::Slice { .. } | Self::Call => {
                    Effects::PURE
                }
                _ => Effects::single(Effect::OrderedState),
            }
        }
    }

    /// Region-policy stand-in that permits exactly one non-consuming reference access mode.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct SingleModeRegionOperation(pub(crate) ReferenceAccessMode);

    impl Display for SingleModeRegionOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for SingleModeRegionOperation {
        type Type = ListIrType;

        fn name(&self) -> &'static str {
            "test.single_mode_region"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ListIrType],
            _region_interfaces: &[RegionInterface<ListIrType>],
        ) -> Result<Vec<ListIrType>, TypeError> {
            Ok(Vec::new())
        }

        fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
            region_index == 0 && mode == self.0
        }
    }

    impl<C: Domain<Type = ListIrType, Value = ListIrValue>> InterpretableOperation<C> for ListOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[ListIrValue],
        ) -> Result<Vec<ListIrValue>, ProgramError> {
            match self {
                Self::Call => driver.interpret_region(context, 0, inputs.to_vec()),
                Self::Add => {
                    check_count!("input", inputs, 2, ProgramError);
                    Ok(vec![inputs[0].add(&inputs[1])?])
                }
                Self::Select { offset, length } => {
                    check_count!("input", inputs, 1, ProgramError);
                    let elements = inputs[0].list()?;
                    let selected = elements.get(*offset..offset + length).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "selection [{offset}, {}) does not fit a list of length {}",
                            offset + length,
                            elements.len(),
                        ))
                    })?;
                    Ok(vec![ListIrValue::List(selected.to_vec())])
                }
                Self::Splice { offset } => {
                    check_count!("input", inputs, 2, ProgramError);
                    let mut spliced = inputs[0].list()?.to_vec();
                    let update = inputs[1].list()?;
                    let length = spliced.len();
                    let range = spliced.get_mut(*offset..offset + update.len()).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "splice [{offset}, {}) does not fit a list of length {length}",
                            offset + update.len(),
                        ))
                    })?;
                    range.clone_from_slice(update);
                    Ok(vec![ListIrValue::List(spliced)])
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before interpretation", self.name()),
                }),
            }
        }
    }

    // One implementation covers every prototype operation, which is why the accumulating rule's
    // `ReferenceAccumulationPolicy` requirement appears as an implementation-level bound here: closed operation-enum
    // dispatch reintroduces the union that the policy split otherwise keeps separate.
    impl<C> ReferenceDischargeableOperation<C, ListReferenceDischarge> for ListOperation
    where
        C: Context<Type = ListIrType, Operation: From<ListOperation>>,
    {
        fn discharge_references<D: ReferenceDischargeDriver<C, ListReferenceDischarge>>(
            &self,
            context: &ReferenceDischargeContext<C, ListReferenceDischarge>,
            driver: &D,
            inputs: &[ReferenceDischargeValue<C, ListReferenceDischarge>],
        ) -> Result<Vec<ReferenceDischargeValue<C, ListReferenceDischarge>>, ProgramError> {
            match self {
                Self::Add | Self::Select { .. } | Self::Splice { .. } => {
                    discharge_reference_free_operation(self, context, driver, inputs)
                }
                Self::ReferenceNew => {
                    check_count!("input", inputs, 1, ProgramError);
                    OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(|positions| positions.push(driver.instruction()));
                    let initial = inputs[0].expect_ordinary("an initial state")?.clone();
                    let initial_type = initial.r#type().into_owned();
                    let output_type = self.infer_output_types(std::slice::from_ref(&initial_type), &[])?.remove(0);
                    let r#type = <&ReferenceType<ListType>>::try_from(&output_type)
                        .map_err(|_| {
                            ProgramError::MalformedProgram(
                                "`list.reference_new` produced a non-reference type".to_string(),
                            )
                        })?
                        .clone();
                    if context.selects_internal(driver.instruction(), 0) {
                        return Ok(vec![context.bind_discharged(r#type, initial)?]);
                    }
                    let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(vec![context.bind_preserved(r#type, outputs.remove(0))?])
                }
                Self::Slice { offset, length } => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to view")?;
                    let alias = reference.alias();
                    if offset + length > alias.length {
                        return Err(ProgramError::MalformedProgram(format!(
                            "view [{offset}, {}) does not fit `{}`",
                            offset + length,
                            reference.r#type(),
                        )));
                    }
                    let composed = ListAlias { offset: alias.offset + offset, length: *length };
                    let r#type = ReferenceType::new(ListType { length: *length });
                    Ok(vec![context.alias_reference(reference, composed, r#type, |value| {
                        let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(value))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    })?])
                }
                Self::Read => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to read")?;
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.read(reference)?)])
                }
                Self::Write => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to write")?;
                    let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                    context.write(reference, replacement)?;
                    Ok(Vec::new())
                }
                Self::Swap => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to replace")?;
                    let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.swap(reference, replacement)?)])
                }
                Self::AddUpdate => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to accumulate into")?;
                    let update = inputs[1].expect_ordinary("an update value")?.clone();
                    context.accumulate(reference, update)?;
                    Ok(Vec::new())
                }
                Self::Freeze => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to freeze")?;
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
                }
                Self::UnreportedFreeze => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to freeze")?;
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
                }
                Self::Call => discharge_positional_region_operation(self, context, driver, inputs, 0),
            }
        }
    }

    /// Builds a reference-free test program with the requested flat boundary arities.
    pub(crate) fn boundary_program(
        input_count: usize,
        output_count: usize,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        for index in 0..input_count {
            builder.add_input(TestType::Value(index as u8));
        }
        let outputs = (0..output_count)
            .map(|index| builder.add_constant(CaptureReference::new(index, TestType::Value(index as u8))))
            .collect::<Vec<_>>();
        builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }

    // Capture seam for the prototype universe, which has no capture constants of its own: a reference-typed constant
    // names the capture position given by its referent length. The seam is the only universe-specific part of capture
    // resolution, so supplying one here exercises every branch of it without inventing a second constant family.
    pub(crate) fn list_capture_position(constant: &ListIrValue) -> Option<usize> {
        match constant {
            ListIrValue::Reference(r#type) => Some(r#type.referent().length),
            ListIrValue::List(_) => None,
        }
    }
}
