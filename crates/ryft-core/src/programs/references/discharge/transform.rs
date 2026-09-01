use std::borrow::Cow;
use std::cell::{Ref, RefCell};
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::captures::{CaptureConstant, ClosedProgram};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::{Parameterized, Placeholder};
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::interpreter::{
    RecursiveReferenceDischargeDriver, ReferenceDischargeDriver, ReferenceDischargeableOperation,
};
use crate::programs::references::discharge::policies::{
    ReferenceAccumulationPolicy, ReferenceDischargePolicy, ReferenceDischargeableType,
};
use crate::programs::references::discharge::results::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, ReferenceDischargeResult, ReferenceSource,
};
use crate::programs::references::discharge::targets::{ReferenceDischargeTarget, ReferenceDischargeTargets};
use crate::programs::references::types::ReferenceType;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;

// TODO(eaplatanios): Order declarations as `ReferenceDischargeableType` -> `ReferenceDischargePolicy` ->
//  `ReferenceAccumulationPolicy` -> `ReferenceDischargeDriver` -> `ReferenceDischargeContext`.

/// Active state of a reference discharge transform. Reference discharge interprets a source [`Program`] into a
/// destination [`Program`], one [`Region`](crate::Region) at a time through a [`ReferenceDischargeDriver`]. Each
/// replayed [`Instruction`](crate::Instruction) dispatches to its [`ReferenceDischargeableOperation`] implementation
/// with this context, and that implementation emits destination work through [`parent`](Self::parent).
///
/// Each source reference allocation is bound into this context exactly once. [`bind_discharged`](Self::bind_discharged)
/// records an allocation as explicit immutable state, while [`bind_preserved`](Self::bind_preserved) records the
/// exact destination reference value when partial discharge leaves the allocation intact. That fate never changes.
/// A [`ReferenceDischargeableOperation`] implementation uses [`derive_reference`](Self::derive_reference) to construct
/// another view of the same allocation, then either rewrites accesses through [`read`](Self::read),
/// [`write`](Self::write), [`swap`](Self::swap), and [`accumulate`](Self::accumulate), or replays
/// them against the preserved destination reference.
///
/// The allocation environment lives on the context rather than on flowing values because references carry identity:
/// several reference values can denote different views of the same allocation, and every one of them must observe the
/// same current state and liveness. Clones therefore share one environment. A structured rule that must rebuild an
/// attached region instead uses [`ReferenceDischargeDriver::discharge_region_program`], which creates an isolated
/// environment and commits nothing here until the rule explicitly merges its outputs.
pub struct ReferenceDischargeContext<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Destination context that owns the discharged values and executes or stages the rewritten work.
    parent: C,

    /// [`ReferenceDischargeEnvironment`] shared by every clone of this context.
    environment: Rc<RefCell<ReferenceDischargeEnvironment<P::Referent, C::Value>>>,

    /// [`ReferenceDischargeCaptureScope`] that contains allocations the capture prefix of the scope this context
    /// discharges binds. A region that inherits its parent's capture prefix discharges under the same scope. A region
    /// fork rebuilds the scope in its own allocation terms.
    captures: ReferenceDischargeCaptureScope<C::Constant>,

    /// [`ReferenceDischargeTargets`] that the current reference discharge transform normalizes into immutable state.
    /// Every allocation they omit is preserved, and the targets are shared unchanged by every clone and by every region
    /// fork, because a target names the same source program location wherever the replay reaches it.
    targets: ReferenceDischargeTargets,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
    /// Creates a new [`ReferenceDischargeContext`] over `parent` that discharges every reference it reaches. The
    /// context starts with no live allocations or capture bindings. Capture bindings are populated while the program
    /// boundary is threaded because they refer to allocations minted by this context. To request partial discharge,
    /// use [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) instead of
    /// constructing a context directly; that function validates the targets against the program in which they were
    /// identified.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::new_with_targets(parent, ReferenceDischargeTargets::everything())
    }

    /// Creates a new [`ReferenceDischargeContext`] with an empty [`ReferenceDischargeEnvironment`] and an empty
    /// [`ReferenceDischargeCaptureScope`] over the provided destination context, discharging exactly the references
    /// named by `targets`.
    #[inline]
    pub fn new_with_targets(parent: C, targets: ReferenceDischargeTargets) -> Self {
        Self {
            parent,
            environment: Rc::new(RefCell::new(ReferenceDischargeEnvironment {
                id: ReferenceDischargeEnvironmentId::next(),
                allocations: Vec::new(),
            })),
            captures: ReferenceDischargeCaptureScope::default(),
            targets,
        }
    }

    /// Returns this [`ReferenceDischargeContext`] discharging under a different [`ReferenceDischargeCaptureScope`],
    /// sharing its [`ReferenceDischargeEnvironment`]. A region fork reaches its own scope this way, because the
    /// allocations that scope binds are minted by the fork itself and therefore exist only once its boundary has
    /// been threaded.
    #[inline]
    pub fn with_captures(&self, captures: ReferenceDischargeCaptureScope<C::Constant>) -> Self
    where
        C: Clone,
    {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures,
            targets: self.targets.clone(),
        }
    }

    /// Returns the destination context that owns the discharged values of this [`ReferenceDischargeContext`].
    pub const fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ReferenceDischargeCaptureScope`] that this [`ReferenceDischargeContext`] discharges under.
    pub const fn captures(&self) -> &ReferenceDischargeCaptureScope<C::Constant> {
        &self.captures
    }

    /// Returns the [`ReferenceDischargeTargets`] that this [`ReferenceDischargeContext`] discharges.
    pub const fn targets(&self) -> &ReferenceDischargeTargets {
        &self.targets
    }

    /// Returns whether the allocation an [`Instruction`](crate::Instruction) performs was selected for discharge,
    /// which is what an allocation rule asks before deciding between a discharged reference and one that survives in
    /// the destination. An operation application that did not come from a replayed instruction (i.e., a region-free
    /// rule invocation through [`EmptyRegionDriver`](crate::programs::EmptyRegionDriver)) has no source program
    /// location and is always discharged as no [`ReferenceDischargeTarget`] can name it, and so declining it would
    /// express nothing about the caller's choice.
    ///
    /// This is the only target query a rule ever makes, which is why it is the only one exposed. Whether an entry
    /// boundary allocation was selected is decided once, by the program-level entry point that threads the boundary,
    /// and no rule is in a position to ask it.
    ///
    /// # Parameters
    ///
    ///   - `instruction`: Replay location of the application, from [`ReferenceDischargeDriver::instruction`].
    ///   - `output_index`: Output position at which the application defines the fresh allocation.
    #[inline]
    pub fn selects_internal(&self, instruction: Option<InstructionId>, output_index: usize) -> bool {
        instruction.is_none_or(|instruction| {
            self.targets.selects(ReferenceDischargeTarget::Internal { instruction, output_index })
        })
    }

    /// Returns whether one entry boundary allocation was selected for discharge.
    #[inline]
    pub fn selects_external(&self, source: ReferenceSource) -> bool {
        self.targets.selects(ReferenceDischargeTarget::External(source))
    }

    /// Returns the complete current immutable state of one live discharged allocation. Operation rules normally use
    /// [`read`](Self::read) to observe the portion selected by a reference value. Structured operation implementations
    /// use this function when they must thread the allocation's complete state across a destination boundary.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment, has
    /// already been consumed, or was preserved rather than discharged.
    #[inline]
    pub fn discharged_state(&self, allocation: ReferenceDischargeAllocationId) -> Result<C::Value, ProgramError> {
        match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { current, .. } => Ok(current.clone()),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        }
    }

    /// Returns whether one live discharged allocation has been mutated during this transform. A direct write, swap, or
    /// accumulation marks the allocation as mutated. Structured operation implementations use this fact to publish only
    /// final states that the source program could have changed; read-only state need not become a hidden output.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment, has
    /// already been consumed, or was preserved rather than discharged.
    #[inline]
    pub fn is_mutated(&self, allocation: ReferenceDischargeAllocationId) -> Result<bool, ProgramError> {
        match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { mutated, .. } => Ok(*mutated),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {allocation}",
            ))),
        }
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of every allocation still live in this context, in binding order.
    /// Consumption retains the allocation's environment slot but removes its entry, so consumed allocations are omitted
    /// while the relative order of all remaining allocations stays stable.
    #[inline]
    pub fn live_allocation_ids(&self) -> Vec<ReferenceDischargeAllocationId> {
        let environment = self.environment.borrow();
        environment
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_some())
            .map(|(index, _)| ReferenceDischargeAllocationId { environment: environment.id, index })
            .collect()
    }

    /// Returns whether one live allocation is discharged rather than preserved. This function is useful when structured
    /// operation code has only an allocation ID and must choose whether to thread immutable state or a destination
    /// reference value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment,
    /// or has already been consumed.
    #[inline]
    pub fn is_allocation_discharged(&self, allocation: ReferenceDischargeAllocationId) -> Result<bool, ProgramError> {
        Ok(matches!(&self.allocation_entry(allocation)?.state, ReferenceDischargeAllocationState::Discharged { .. }))
    }

    /// Returns an unviewed [`ReferenceDischargeValue`] denoting a live allocation already bound in this context. This
    /// function does not bind another allocation. Region threading and capture resolution use it to reconstruct the
    /// complete value reference associated with an allocation ID. A preserved allocation carries the exact destination
    /// reference originally supplied to [`bind_preserved`](Self::bind_preserved).
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment,
    /// or has already been consumed.
    pub fn allocation_reference(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let (r#type, binding) = {
            let entry = self.allocation_entry(allocation)?;
            let binding = match &entry.state {
                ReferenceDischargeAllocationState::Discharged { .. } => ReferenceDischargeBinding::Discharged,
                ReferenceDischargeAllocationState::Preserved { reference } => {
                    ReferenceDischargeBinding::Preserved { reference: reference.clone() }
                }
            };
            (entry.r#type.clone(), binding)
        };
        let alias = P::storage_alias(r#type.referent());
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            denotes_complete_value: true,
            alias,
            r#type,
            binding,
        }))
    }

    /// Returns an immutable borrow of one live [`ReferenceDischargeAllocationEntry`]. The returned guard keeps the
    /// allocation environment immutably borrowed. Callers should copy or clone the fields they need and let the guard
    /// drop before invoking any operation that may borrow the environment mutably.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` belongs to another environment, was never bound,
    /// or has already been consumed.
    pub(crate) fn allocation_entry(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<Ref<'_, ReferenceDischargeAllocationEntry<P::Referent, C::Value>>, ProgramError> {
        let environment = self.environment.borrow();
        if environment.slot(allocation)?.is_none() {
            return Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}")));
        }
        Ok(Ref::map(environment, |environment| {
            // The check above proved that this exact slot exists and contains a live entry.
            environment.allocations[allocation.index].as_ref().unwrap()
        }))
    }

    /// Binds an allocation selected for discharge and returns its unviewed reference value. The allocation is fresh to
    /// this context even when it represents a reference that already existed at the source program's entry boundary.
    /// Its `initial` value becomes the immutable state exposed by [`discharged_state`](Self::discharged_state),
    /// observed through [`read`](Self::read), and transformed through [`write`](Self::write), [`swap`](Self::swap),
    /// and [`accumulate`](Self::accumulate).
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`ReferenceType`] of the allocation.
    ///   - `initial`: Destination value that becomes the allocation's initial immutable state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `initial` does not carry the lifted referent type of `r#type`.
    pub fn bind_discharged(
        &self,
        r#type: ReferenceType<P::Referent>,
        initial: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let expected = C::Type::from(r#type.referent().clone());
        let actual = initial.r#type();
        if actual.as_ref() != &expected {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state has type `{actual}` but allocation `{type}` requires `{expected}`",
            )));
        }
        Ok(self.bind_allocation(
            r#type,
            ReferenceDischargeAllocationState::Discharged { current: initial, mutated: false },
            ReferenceDischargeBinding::Discharged,
        ))
    }

    /// Binds an allocation preserved by partial discharge and returns its unviewed reference value. The environment
    /// retains `reference` so structured boundaries can thread the allocation, while each derived reference retains
    /// the exact destination value produced when its view operation is replayed. A preserved allocation never becomes
    /// discharged later in this transform.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`ReferenceType`] of the allocation.
    ///   - `reference`: Destination reference-typed value denoting the allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `reference` does not carry the reference type `r#type`.
    pub fn bind_preserved(
        &self,
        r#type: ReferenceType<P::Referent>,
        reference: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let reference_type = reference.r#type();
        let actual = <&ReferenceType<P::Referent>>::try_from(reference_type.as_ref()).map_err(|_| {
            ProgramError::MalformedProgram(format!(
                "reference discharge preserved an allocation as `{reference_type}` which is not a reference type",
            ))
        })?;
        if actual != &r#type {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge preserved an allocation as `{actual}` but its handle exposes `{type}`",
            )));
        }
        Ok(self.bind_allocation(
            r#type,
            ReferenceDischargeAllocationState::Preserved { reference: reference.clone() },
            ReferenceDischargeBinding::Preserved { reference },
        ))
    }

    /// Inserts one validated allocation entry and returns its unviewed reference value. [`Self::bind_discharged`] and
    /// [`Self::bind_preserved`] validate their fate-specific values before calling this shared environment primitive.
    fn bind_allocation(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceDischargeAllocationState<C::Value>,
        binding: ReferenceDischargeBinding<C::Value>,
    ) -> ReferenceDischargeValue<C, P> {
        let alias = P::storage_alias(r#type.referent());
        let allocation = {
            // The environment owns the complete type and state even after every handle to this allocation disappears.
            // Keep the mutable borrow scoped to inserting that record; the returned handle carries only its identity.
            let mut environment = self.environment.borrow_mut();
            environment
                .allocations
                .push(Some(ReferenceDischargeAllocationEntry { r#type: r#type.clone(), state }));
            ReferenceDischargeAllocationId { environment: environment.id, index: environment.allocations.len() - 1 }
        };
        ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            denotes_complete_value: true,
            alias,
            r#type,
            binding,
        })
    }

    // TODO(eaplatanios): Review from here onwards.

    /// Derives another reference value for the same allocation with the provided composed view and exposed type.
    ///
    /// `alias` is the authoritative complete view chain, not merely the newest view step. For a discharged allocation,
    /// later accesses apply that chain to the allocation's immutable state and `derive_preserved` is not called. For a
    /// preserved allocation, `derive_preserved` receives the parent reference's exact destination value and normally
    /// replays the source view operation; the returned value is retained on the derived reference so later accesses
    /// use it directly instead of replaying the view again.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Reference value the view is composed onto.
    ///   - `alias`: Complete composed view chain of the derived reference.
    ///   - `r#type`: Reference type the derived reference exposes.
    ///   - `derive_preserved`: Produces the derived reference's destination value from the parent reference's value,
    ///     invoked only when the allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is no longer live or when the derived destination
    /// value does not carry the reference type `r#type`, and propagates every `derive_preserved` failure.
    pub fn derive_reference(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        alias: P::Alias,
        r#type: ReferenceType<P::Referent>,
        derive_preserved: impl FnOnce(&C::Value) -> Result<C::Value, ProgramError>,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let allocation = reference.allocation_id();

        // Handles can outlive the allocation they denote, so resolve the ID against the active environment before
        // deriving another handle. This reports foreign, never-bound, and consumed allocations at the attempted use.
        self.allocation_entry(allocation)?;

        let binding = match &reference.binding {
            ReferenceDischargeBinding::Discharged => ReferenceDischargeBinding::Discharged,
            ReferenceDischargeBinding::Preserved { reference: parent } => {
                let derived = derive_preserved(parent)?;
                let derived_type = derived.r#type();
                let actual = <&ReferenceType<P::Referent>>::try_from(derived_type.as_ref()).map_err(|_| {
                    ProgramError::MalformedProgram(format!(
                        "reference discharge preserved an allocation as `{derived_type}` which is not a reference type",
                    ))
                })?;
                if actual != &r#type {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge preserved an allocation as `{actual}` but its handle exposes `{type}`",
                    )));
                }
                ReferenceDischargeBinding::Preserved { reference: derived }
            }
        };

        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            denotes_complete_value: false,
            alias,
            r#type,
            binding,
        }))
    }

    /// Reads the portion that `reference` selects from its discharged allocation's current state.
    ///
    /// Reference-operation rules call this function only for discharged references. An access to a preserved reference
    /// must instead replay the source operation against [`ReferenceDischargeReference::preserved`].
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live, and propagates the policy's error when
    /// the alias cannot be applied. Reading a preserved reference through this function is rejected, because a preserved
    /// access must replay verbatim in the destination instead.
    pub fn read(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let current = self.discharged_state(reference.allocation_id())?;
        P::read(&self.parent, &current, reference.alias())
    }

    /// Replaces the portion that `reference` selects in a discharged allocation.
    ///
    /// The policy returns a complete successor state, which this function installs through
    /// [`update_discharged_state`](Self::update_discharged_state) and records as a mutation.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to replace.
    ///   - `replacement`: Value written into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the write cannot be applied.
    pub fn write(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let successor = P::write(&self.parent, &current, replacement, reference.alias())?;
        self.update_discharged_state(allocation, successor, true)
    }

    /// Replaces the portion that `reference` selects and returns its previous contents.
    ///
    /// Like [`write`](Self::write), this function installs the policy's complete successor state and records a
    /// mutation; unlike `write`, it also returns the value selected before replacement.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to replace.
    ///   - `replacement`: Value written into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied.
    pub fn swap(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let (previous, successor) = P::swap(&self.parent, &current, replacement, reference.alias())?;
        self.update_discharged_state(allocation, successor, true)?;
        Ok(previous)
    }

    /// Accumulates `update` into the portion that `reference` selects in a discharged allocation.
    ///
    /// The policy returns a complete successor state, which this function installs and records as a mutation.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to update.
    ///   - `update`: Value accumulated into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied or the universe forbids accumulation.
    pub fn accumulate(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        update: C::Value,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
        P: ReferenceAccumulationPolicy<C>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let successor = P::accumulate(&self.parent, &current, update, reference.alias())?;
        self.update_discharged_state(allocation, successor, true)
    }

    /// Consumes a discharged allocation and returns its complete current immutable state.
    ///
    /// Consumption removes the allocation's live environment entry, so every later access reports a use-after-consume.
    /// It always yields the complete stored value and deliberately ignores aliases; only the unviewed reference value
    /// returned when the allocation was bound can therefore name the transition. For a preserved allocation, the
    /// destination operation performs the semantic consumption and
    /// [`mark_preserved_consumed`](Self::mark_preserved_consumed) records the matching liveness transition.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live, was preserved rather than discharged, or
    /// is named through a derived handle rather than an unviewed handle.
    pub fn consume(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let allocation = reference.allocation_id();
        let current_type = match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { current, .. } => Ok(current.r#type().into_owned()),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        }?;
        if !reference.denotes_complete_value() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the derived view `{}`; consumption yields the \
                 complete stored value, whose referent is `{}`",
                reference.r#type(),
                current_type,
            )));
        }
        let mut environment = self.environment.borrow_mut();
        // The inspection above proved that this handle belongs to this environment and names a live discharged reference.
        let entry = environment.allocations[allocation.index].take().unwrap();
        let ReferenceDischargeAllocationState::Discharged { current, .. } = entry.state else { unreachable!() };
        Ok(current)
    }

    /// Records that a replayed destination operation consumed one preserved allocation.
    ///
    /// This is [`consume`](Self::consume)'s bookkeeping counterpart for an allocation that survives in the destination.
    /// It returns no value because the replayed operation already produced the destination result; it only removes the
    /// environment entry so later accesses report a use-after-consume. Consumption still applies to the complete stored
    /// value, so only the unviewed reference returned by [`bind_preserved`](Self::bind_preserved) can name it.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live, was discharged rather than preserved, or
    /// is named through a derived handle rather than an unviewed handle.
    pub fn mark_preserved_consumed(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<(), ProgramError> {
        let allocation = reference.allocation_id();
        let whole = self.allocation_entry(allocation)?.r#type.clone();
        if self.is_allocation_discharged(allocation)? {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge marked discharged {allocation} as consumed through the preserved path",
            )));
        }
        if !reference.denotes_complete_value() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the derived view `{}`; consumption yields the \
                 complete stored value, whose reference type is `{}`",
                reference.r#type(),
                whole,
            )));
        }

        // The type lookup above proved that this handle belongs to this environment and names a live allocation.
        self.environment.borrow_mut().allocations[allocation.index] = None;
        Ok(())
    }

    /// Installs the complete current state of one live discharged allocation and merges its mutation status.
    ///
    /// Reference-operation functions pass `true`. Structured-boundary code passes its access summary because symmetric
    /// boundaries also return unchanged state for read-only allocations, which must not cause those allocations to
    /// publish hidden final-state outputs. Once an allocation is marked mutated, a later `false` never clears that fact.
    ///
    /// # Parameters
    ///
    ///   - `allocation`: Live discharged allocation whose complete state is being installed.
    ///   - `current`: New complete immutable state.
    ///   - `mutated`: Whether this transition should mark the allocation as mutated.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is not a live discharged allocation or `current`
    /// does not carry its referent type.
    pub fn update_discharged_state(
        &self,
        allocation: ReferenceDischargeAllocationId,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        // Validate before taking the mutable environment borrow so a type error leaves both the current state and its
        // mutation bit unchanged.
        let r#type = self.allocation_entry(allocation)?.r#type.clone();
        let expected = C::Type::from(r#type.referent().clone());
        let actual = current.r#type();
        if actual.as_ref() != &expected {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state has type `{actual}` but allocation `{type}` requires `{expected}`",
            )));
        }
        let mut environment = self.environment.borrow_mut();
        environment.slot(allocation)?;
        match environment.allocations[allocation.index].as_mut().map(|entry| &mut entry.state) {
            Some(ReferenceDischargeAllocationState::Discharged { current: state, mutated: previous_mutated }) => {
                *state = current;
                *previous_mutated |= mutated;
                Ok(())
            }
            Some(ReferenceDischargeAllocationState::Preserved { .. }) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge updated the state of preserved {allocation}",
            ))),
            None => Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}"))),
        }
    }
}

impl<C: Clone + Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeContext<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures: self.captures.clone(),
            targets: self.targets.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeContext<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live_allocation_count =
            self.environment.borrow().allocations.iter().filter(|allocation| allocation.is_some()).count();
        formatter
            .debug_struct("ReferenceDischargeContext")
            .field("live_allocation_count", &live_allocation_count)
            .finish_non_exhaustive()
    }
}

/// Identity of one reference allocation inside a running reference discharge.
///
/// IDs are minted by [`ReferenceDischargeContext`] as allocations enter its environment, so they are temporary
/// discharge identities rather than source program locations. They exist only for the duration of one discharge and
/// are meaningful only against the environment that produced them. Pre-transform identity for caller-facing targets
/// is [`ReferenceDischargeTarget`] instead.
///
/// Each ID records which environment minted it, so an ID from an unrelated discharge is reported rather than silently
/// addressing whichever allocation happens to occupy the same position. That is also what isolates a structured
/// rule's region fork: the fork mints its own environment, so a caller ID cannot address a fork allocation and a fork
/// ID cannot address a caller allocation. The one table relating the two lives inside
/// [`ReferenceDischargeDriver::discharge_region_program`], which reports its results in caller terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceDischargeAllocationId {
    /// Environment that minted this ID.
    environment: ReferenceDischargeEnvironmentId,

    /// Position of the allocation in that environment.
    index: usize,
}

impl Display for ReferenceDischargeAllocationId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reference allocation {}:{}", self.environment.0, self.index)
    }
}

/// Identity of one reference discharge allocation environment, shared by every clone of the context that owns it and
/// distinct for every environment a structured rule's region fork mints.
///
/// No caller names this identity directly. It makes [`ReferenceDischargeAllocationId`] addressable only in the
/// environment that minted it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReferenceDischargeEnvironmentId(
    /// Process-local numeric environment identity.
    usize,
);

impl ReferenceDischargeEnvironmentId {
    /// Returns a fresh environment identity, distinct from every identity handed out so far in this process.
    fn next() -> Self {
        static NEXT_ENVIRONMENT_ID: AtomicUsize = AtomicUsize::new(0);
        Self(NEXT_ENVIRONMENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Live allocation environment of one reference discharge, shared by every clone of its context.
struct ReferenceDischargeEnvironment<T: Type, V> {
    /// Identity that every allocation ID minted from this environment records.
    id: ReferenceDischargeEnvironmentId,

    /// State of every allocation minted so far, indexed by [`ReferenceDischargeAllocationId`]. A consumed allocation
    /// keeps its slot and becomes [`None`], so a use-after-consume is reported against the exact allocation rather than
    /// as an unknown ID.
    allocations: Vec<Option<ReferenceDischargeAllocationEntry<T, V>>>,
}

impl<T: Type, V> ReferenceDischargeEnvironment<T, V> {
    /// Returns the state slot that `allocation` names, or an error when the ID belongs to another environment or names
    /// a position this environment never minted.
    fn slot(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<&Option<ReferenceDischargeAllocationEntry<T, V>>, ProgramError> {
        if allocation.environment != self.id {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed {allocation}, which belongs to an environment other than the active `{}`",
                self.id.0,
            )));
        }
        self.allocations.get(allocation.index).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("reference discharge accessed never-bound {allocation}"))
        })
    }
}

/// Complete environment record of one live allocation: the reference type of its complete stored value and what
/// discharge turned that allocation into.
///
/// The reference type is recorded because an allocation's identity outlives every handle that denotes it. A structured rule
/// threading an inherited allocation through a rebuilt region boundary holds only that allocation's handle, never a handle it
/// could read a type off, so the environment is where the complete-value type has to live.
pub(crate) struct ReferenceDischargeAllocationEntry<T: Type, V> {
    /// Reference type of the complete stored value, whose referent types the immutable state a discharged reference threads.
    pub(crate) r#type: ReferenceType<T>,

    /// What discharge turned this allocation into.
    state: ReferenceDischargeAllocationState<V>,
}

/// Environment entry describing what one live reference allocation became during reference discharge.
#[derive(Debug)]
enum ReferenceDischargeAllocationState<V> {
    /// Allocation selected for discharge, which threads through the destination program as immutable state.
    Discharged {
        /// Current immutable state of the complete stored value.
        current: V,

        /// Whether any ordered write or accumulation has been applied to this allocation. Read-only allocations are pruned from
        /// hidden outputs and from structured-operation widening, so this is the fact that pruning consults.
        mutated: bool,
    },

    /// Allocation not selected for discharge, which survives in the destination program as a reference value. This is the
    /// allocation's own destination reference value and is what boundary threading uses; a handle derived from it through
    /// a view carries its own exact destination value instead.
    Preserved {
        /// Destination reference-typed value denoting the allocation.
        reference: V,
    },
}

/// Reference allocations the capture prefix of one discharge scope binds.
///
/// A capture-lifted program names its caller's references through constants rather than through its own boundary: the
/// entry boundary carries the lifted capture prefix, and an attached region inside that program names the very same
/// references through capture constants. Resolving one is therefore a property of the scope a region discharges
/// under, not of any rule, so the scope rides on [`ReferenceDischargeContext`] beside the allocation environment and is
/// recomputed at every region boundary — inherited by default, and replaced by a fresh prefix wherever an operation
/// declares one through [`Operation::region_capture_input_count`].
///
/// Recognizing a capture is a *constant-family* question, and the interpreter deliberately serves families that are
/// not capture-bearing at all, so the resolver is a function pointer supplied by the entry point that knows the family
/// rather than a [`CaptureConstant`] bound on the whole architecture. The [`Default`] scope recognizes nothing and
/// binds nothing, which is exactly the behavior of a program that has no captures.
pub struct ReferenceDischargeCaptureScope<Constant> {
    /// Capture position a constant names, or [`None`] when it is an ordinary constant of its family.
    capture_index_of: fn(&Constant) -> Option<usize>,

    /// Allocation each capture position binds, or [`None`] when that position carries an ordinary value rather than a
    /// reference. A capture position past the end of this list binds nothing.
    allocations: Rc<[Option<ReferenceDischargeAllocationId>]>,
}

impl<Constant> ReferenceDischargeCaptureScope<Constant> {
    /// Creates a capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_index_of`: Function reporting the capture position a constant of this family names.
    ///   - `allocations`: Allocation each capture position binds, in capture order.
    #[inline]
    pub fn new(
        capture_index_of: fn(&Constant) -> Option<usize>,
        allocations: Vec<Option<ReferenceDischargeAllocationId>>,
    ) -> Self {
        Self { capture_index_of, allocations: allocations.into() }
    }

    /// Returns the allocation each capture position binds, in capture order.
    #[inline]
    pub(super) fn allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.allocations.as_ref()
    }

    /// Returns the allocation one constant denotes, or [`None`] when the constant names no capture position or that
    /// position binds no allocation. A constant this scope cannot resolve is an ordinary constant of its family, and a
    /// reference-typed one that no scope resolves is rejected where it is lifted.
    #[inline]
    pub(super) fn resolve(&self, constant: &Constant) -> Option<ReferenceDischargeAllocationId> {
        (self.capture_index_of)(constant).and_then(|index| self.allocations.get(index).copied().flatten())
    }

    /// Returns this scope's resolver over a different set of bound allocations, which is how a nested region's scope and a
    /// region fork's remapped scope are built without restating the constant family's recognition rule.
    #[inline]
    pub(super) fn with_allocations(&self, allocations: Vec<Option<ReferenceDischargeAllocationId>>) -> Self {
        Self { capture_index_of: self.capture_index_of, allocations: allocations.into() }
    }
}

impl<Constant> Default for ReferenceDischargeCaptureScope<Constant> {
    #[inline]
    fn default() -> Self {
        Self { capture_index_of: |_| None, allocations: Rc::from([]) }
    }
}

impl<Constant> Clone for ReferenceDischargeCaptureScope<Constant> {
    #[inline]
    fn clone(&self) -> Self {
        Self { capture_index_of: self.capture_index_of, allocations: Rc::clone(&self.allocations) }
    }
}

impl<Constant> Debug for ReferenceDischargeCaptureScope<Constant> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeCaptureScope")
            .field("allocations", &self.allocations)
            .finish_non_exhaustive()
    }
}

/// Handle to one live reference allocation flowing through reference discharge.
///
/// The fields are private and only [`ReferenceDischargeContext`] constructs them, so a rule can read a handle but
/// cannot fabricate an allocation, an alias, a derived type, or a preserved destination value. That keeps allocation
/// identity and view composition checked even though the rule trait is open to third-party operations.
pub struct ReferenceDischargeReference<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Identity of the allocation this handle denotes.
    allocation_id: ReferenceDischargeAllocationId,

    /// Whether this handle denotes the complete stored value rather than a derived view of it.
    denotes_complete_value: bool,

    /// Composed policy-owned view chain from the allocation to this handle.
    alias: P::Alias,

    /// Reference type this exact handle exposes, which differs from the allocation's type under a composed view.
    r#type: ReferenceType<P::Referent>,

    /// Destination fate of this handle's allocation, fixed at construction.
    binding: ReferenceDischargeBinding<C::Value>,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeReference<C, P> {
    /// Returns the identity of the allocation this handle denotes.
    pub const fn allocation_id(&self) -> ReferenceDischargeAllocationId {
        self.allocation_id
    }

    /// Returns whether this handle denotes the complete stored value rather than a derived view.
    pub(super) const fn denotes_complete_value(&self) -> bool {
        self.denotes_complete_value
    }

    /// Returns the composed view chain from the allocation to this handle.
    pub const fn alias(&self) -> &P::Alias {
        &self.alias
    }

    /// Returns the reference type this exact handle exposes.
    pub const fn r#type(&self) -> &ReferenceType<P::Referent> {
        &self.r#type
    }

    /// Returns the exact destination reference value of a preserved handle, or [`None`] when the allocation was
    /// discharged.
    pub const fn preserved(&self) -> Option<&C::Value> {
        match &self.binding {
            ReferenceDischargeBinding::Discharged => None,
            ReferenceDischargeBinding::Preserved { reference } => Some(reference),
        }
    }

    /// Returns how this handle's allocation is represented in the destination program.
    pub(super) const fn binding(&self) -> &ReferenceDischargeBinding<C::Value> {
        &self.binding
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeReference<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            allocation_id: self.allocation_id,
            denotes_complete_value: self.denotes_complete_value,
            alias: self.alias.clone(),
            r#type: self.r#type.clone(),
            binding: self.binding.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeReference<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeReference")
            .field("allocation_id", &self.allocation_id)
            .field("denotes_complete_value", &self.denotes_complete_value)
            .field("alias", &self.alias)
            .field("type", &self.r#type)
            .field("binding", &self.binding)
            .finish()
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeReference<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.allocation_id, self.r#type)
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeReference<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.allocation_id == other.allocation_id
            && self.denotes_complete_value == other.denotes_complete_value
            && self.alias == other.alias
            && self.r#type == other.r#type
            && self.binding == other.binding
    }
}

/// Destination fate a [`ReferenceDischargeReference`] handle carries for its allocation.
///
/// The binding is fixed when the handle is constructed and always agrees with the allocation's environment state, because
/// allocations never move between the discharged and preserved fates after they are bound: carrying the fate on the handle
/// makes a handle/environment disagreement unrepresentable rather than defensively re-checked at every access.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ReferenceDischargeBinding<V> {
    /// The allocation became explicit immutable state, so accesses through this handle rewrite into state reads and
    /// writes against the environment.
    Discharged,

    /// The allocation survives in the destination program, and this exact handle denotes `reference` there.
    ///
    /// A preserved handle must consume this value rather than re-deriving its view chain per access, because
    /// re-deriving would duplicate and reorder the replayed view operations in the destination program.
    Preserved {
        /// Exact destination reference value this handle denotes.
        reference: V,
    },
}

/// Context-free carrier flowing through reference discharge.
///
/// Rules receive and return carriers; the context that owns the allocation environment travels separately as an explicit
/// rule argument rather than being stamped onto every value. It is public because the rule trait names it, and
/// because enum variant fields are always as public as their enum, the reference payload is the opaque
/// [`ReferenceDischargeReference`] rather than inline fields.
pub enum ReferenceDischargeValue<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Ordinary destination value, carrying no reference and replayed as-is.
    Ordinary(C::Value),

    /// Handle to one live reference allocation.
    Reference(ReferenceDischargeReference<C, P>),
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeValue<C, P> {
    /// Returns the ordinary destination value this carrier holds, or an error naming `expectation` when it holds a
    /// reference handle instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_ordinary(&self, expectation: &str) -> Result<&C::Value, ProgramError> {
        match self {
            Self::Ordinary(value) => Ok(value),
            Self::Reference(reference) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received {reference}",
            ))),
        }
    }

    /// Returns the reference handle this carrier holds, or an error naming `expectation` when it holds an ordinary
    /// value instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_reference(&self, expectation: &str) -> Result<&ReferenceDischargeReference<C, P>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Ordinary(_) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received an ordinary value",
            ))),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeValue<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(value) => Self::Ordinary(value.clone()),
            Self::Reference(reference) => Self::Reference(reference.clone()),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeValue<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => formatter.debug_tuple("Ordinary").field(value).finish(),
            Self::Reference(reference) => formatter.debug_tuple("Reference").field(reference).finish(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeValue<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => Display::fmt(value, formatter),
            Self::Reference(reference) => Display::fmt(reference, formatter),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeValue<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ordinary(value), Self::Ordinary(other)) => value == other,
            (Self::Reference(reference), Self::Reference(other)) => reference == other,
            _ => false,
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Typed for ReferenceDischargeValue<C, P>
where
    C::Type: From<ReferenceType<P::Referent>>,
{
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match self {
            Self::Ordinary(value) => value.r#type(),
            Self::Reference(reference) => Cow::Owned(C::Type::from(reference.r#type().clone())),
        }
    }
}

// TODO(eaplatanios): Review up to here.

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Rewrites every [`Reference`](crate::Reference) in this [`Program`] as explicit immutable state and returns the
    /// resulting reference-free program together with bindings for its external references.
    ///
    /// A reference-typed input keeps its position but becomes an ordinary input carrying the reference's initial state.
    /// Local reference allocations disappear. The source program's public outputs remain first and in the same order;
    /// the final state of each mutated external reference is appended as a hidden output in entry-boundary order.
    /// A read-only external reference adds no hidden output.
    ///
    /// Each operation defines how its reference effects are rewritten through [`ReferenceDischargeableOperation`].
    /// A structured operation must also thread through the reference state used by its attached
    /// [`Region`](crate::Region)s; a structured operation whose complete region closure is reference-free is replayed
    /// unchanged. The returned [`ReferenceDischargeResult`] proves that no reference type or reference operation
    /// remains anywhere in the rewritten region closure.
    ///
    /// Use [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// instead when the program was produced by lifting a [`ClosedProgram`]'s captures and attached regions may refer
    /// to those captures through capture constants.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the input count, when an external
    /// reference is consumed, when a reference reaches a region without a boundary or capture binding, when a
    /// structured operation has no rule for its reference-using regions, or when the rewritten program is not fully
    /// reference-free. Errors reported by individual operation rules, including use after consumption and access to
    /// an unbound reference, propagate unchanged.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites every [`Reference`](crate::Reference) in a capture-lifted [`Program`] as explicit immutable state and
    /// returns a proven reference-free result.
    ///
    /// A capture-lifted program has the captures of a [`ClosedProgram`] represented by a leading input prefix. Attached
    /// [`Region`](crate::Region)s may still name those captures through capture constants. This function resolves each
    /// reference-typed capture constant to the external reference bound at the corresponding prefix position.
    ///
    /// Apart from that capture resolution, this function has the same boundary rewrite, hidden final-state outputs,
    /// and reference-freedom guarantee as [`discharge_references`](Self::discharge_references). The two functions
    /// produce the same result when no attached region uses a reference-typed capture constant.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix. It determines both the capture scope and the
    ///     split between [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`discharge_references`](Self::discharge_references). It also returns
    /// [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            CaptureConstant::capture_index,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s as explicit immutable state while preserving
    /// every unselected reference.
    ///
    /// The selected references follow the same rewrite as [`discharge_references`](Self::discharge_references). An
    /// unselected reference keeps its reference-typed boundary position or allocation operation, and its accesses and
    /// derived views are replayed unchanged. It contributes no [`ExternalReferenceBinding`] or hidden final-state
    /// output because it never becomes explicit state.
    ///
    /// Preserved references can cross structured-[`Region`](crate::Region) boundaries beside discharged state. A
    /// declared reference position remains a reference position; when an attached region reaches a preserved reference
    /// only through an inherited capture, the rewrite adds a reference-typed position to keep the rebuilt region
    /// self-contained.
    ///
    /// The returned [`PartialReferenceDischargeResult`] may therefore still contain reference types and operations.
    /// If the selected targets were expected to cover every reference, convert it through
    /// [`ReferenceDischargeResult::try_from`] to validate and obtain the full-discharge guarantee.
    ///
    /// A capture-lifted program instead uses [`Self::partially_discharge_references_in_capture_lifted_program`], which
    /// performs the same target selection under a populated capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table,
    ///     used to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///   - `targets`: Reference targets to discharge, enumerated from this same program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `targets` does not belong to this program or otherwise violates
    /// the target-selection contract. It also returns the applicable errors documented by
    /// [`discharge_references`](Self::discharge_references), except that consuming a preserved external reference is
    /// allowed because its consuming operation remains in the rewritten program. Consuming a discharged external
    /// reference remains invalid because its state is still owned by the caller.
    #[inline]
    pub fn partially_discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, |_| None, targets)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s of a capture-lifted [`Program`] as explicit
    /// immutable state and preserves every unselected reference.
    ///
    /// This function combines the capture-constant resolution of
    /// [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// with the selection and partial-result behavior of
    /// [`partially_discharge_references`](Self::partially_discharge_references). The lifted capture prefix
    /// establishes the capture scope; only references named by `targets` become immutable state.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix.
    ///   - `targets`: Reference targets to discharge, enumerated from this same capture-lifted program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`partially_discharge_references`](Self::partially_discharge_references). It also
    /// returns [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn partially_discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, CaptureConstant::capture_index, targets)
    }

    /// Performs the shared partial-discharge rewrite for one validated target selection. `capture_index_of` resolves
    /// reference-typed capture constants when the input is capture-lifted; ordinary programs provide a resolver that
    /// matches no constant. The function always returns a [`PartialReferenceDischargeResult`]. Full-discharge entry
    /// points select every reference and then validate the result through [`ReferenceDischargeResult::try_from`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs that originated in the source program's capture table.
    ///   - `capture_index_of`: Function returning the capture position named by a stored constant.
    ///   - `targets`: Reference targets to discharge; every allocation they omit is preserved.
    ///
    /// # Errors
    ///
    /// Returns any validation or operation-rule error produced while rewriting the program.
    fn discharge_references_helper<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        capture_index_of: fn(&V) -> Option<usize>,
        targets: ReferenceDischargeTargets,
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let input_types = self.input_types();
        let input_count = input_types.len();
        if capture_count > input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge requests {capture_count} captures but the program has {input_count} inputs",
            )));
        }
        let output_count = self.output_count();

        // A program that touches no reference anywhere is already its own discharge, so it is returned untouched rather
        // than replayed into a fresh trace. This is not only cheaper on the two transform adapters that discharge
        // unconditionally: re-tracing would also renumber its atoms, drop its dead constants, and abandon the region
        // transform cache its regions carry, all for a rewrite that has nothing to rewrite.
        let entry = self.entry_region_ref();
        if !entry.contains_references_in_closure() {
            return PartialReferenceDischargeResult::new(self, capture_count, output_count, Vec::new());
        }

        // The block scopes the destination context, the discharge context, and every carrier, because recovering the
        // traced program below requires unique ownership of the shared builder and therefore that every other handle
        // to it has been released.
        let (builder, output_ids, external_reference_bindings) = {
            let destination = TracingContext::<V, O>::new();
            let builder = destination.builder().clone();
            let context = ReferenceDischargeContext::new_with_targets(destination.clone(), targets);
            let mut inputs = Vec::with_capacity(input_count);
            let mut discharged_allocations = Vec::new();
            let mut capture_allocations = vec![None; capture_count];
            for (input_index, input_type) in input_types.into_iter().enumerate() {
                let Ok(reference_type) = <&ReferenceType<P::Referent>>::try_from(&input_type) else {
                    inputs.push(ReferenceDischargeValue::Ordinary(destination.input(input_type)));
                    continue;
                };
                let reference_type = reference_type.clone();
                let source = ReferenceSource::from_flat_input_index(input_index, capture_count);
                let selected = context.selects_external(source);
                let carrier = if selected {
                    let state = destination.input(V::Type::from(reference_type.referent().clone()));
                    context.bind_discharged(reference_type, state)?
                } else {
                    // An unselected external allocation keeps its reference-typed boundary position exactly as the
                    // source declared it, so the caller still supplies the reference, and every access to it replays
                    // verbatim.
                    context.bind_preserved(reference_type, destination.input(input_type))?
                };
                let allocation = carrier.expect_reference("an entry-boundary reference allocation")?.allocation_id();
                if selected {
                    discharged_allocations.push((source, allocation));
                }
                if input_index < capture_count {
                    capture_allocations[input_index] = Some(allocation);
                }
                inputs.push(carrier);
            }

            // The capture scope can only be established once the prefix has minted its allocations, and it is what lets
            // a nested region resolve the caller references it names through capture constants rather than through its
            // own boundary.
            let context =
                context.with_captures(ReferenceDischargeCaptureScope::new(capture_index_of, capture_allocations));

            let regions = [self];
            let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
            let outputs = driver.discharge_region(&context, 0, inputs)?;
            let mut output_ids = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| match output {
                    ReferenceDischargeValue::Ordinary(value) => value.atom_id(),
                    ReferenceDischargeValue::Reference(reference) => {
                        // A preserved reference survives in the rewritten program, so returning one returns its
                        // destination reference value. A discharged reference has no such value, because it became
                        // state. Returning an allocation is a use of it like any other, so its liveness is resolved
                        // against the environment rather than taken from the handle, which is what reports an
                        // allocation the program already consumed.
                        context.allocation_entry(reference.allocation_id())?;
                        reference
                            .preserved()
                            .ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference discharge expected an ordinary value for output {output_index} but \
                                     received {reference}",
                                ))
                            })?
                            .atom_id()
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            // A mutated external allocation publishes its final state as a hidden output. A read-only one publishes
            // nothing, which is what keeps a read-only program's boundary identical to its source boundary. A preserved
            // external allocation binds nothing at all (i.e., because it never became state, so there is no state for a
            // caller to supply or to write back).
            let mut external_reference_bindings = Vec::with_capacity(discharged_allocations.len());
            for (source, allocation) in discharged_allocations {
                // External state remains caller-owned, so consuming its allocation during replay invalidates the
                // transform even when no later source operation tries to use it.
                if context.allocation_entry(allocation).is_err() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge consumed external {source}, whose state must remain owned by the caller",
                    )));
                }
                let output_index = if context.is_mutated(allocation)? {
                    output_ids.push(context.discharged_state(allocation)?.atom_id()?);
                    Some(output_ids.len() - 1)
                } else {
                    None
                };
                external_reference_bindings.push(ExternalReferenceBinding::new(source, output_index));
            }
            (builder, output_ids, external_reference_bindings)
        };

        let complete_output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program =
            builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; complete_output_count])?;
        PartialReferenceDischargeResult::new(program, capture_count, output_count, external_reference_bindings)
    }
}

impl<
    Capture: Value,
    V: CaptureConstant<Type = Capture::Type>,
    O: Operation<Type = Capture::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> ClosedProgram<Capture, V, O, Input, Output>
{
    /// Lifts this [`ClosedProgram`]'s captures into leading inputs and rewrites every [`Reference`](crate::Reference)
    /// as explicit immutable state.
    ///
    /// The returned [`ReferenceDischargeResult`] remains reference-free and records which leading inputs originated as
    /// captures rather than ordinary inputs. The concrete capture values remain owned by this closed program; their
    /// mutable contents are not embedded in the rewritten program.
    ///
    /// # Errors
    ///
    /// Returns errors produced while lifting the captures or performing capture-aware reference discharge.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        &self,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        Capture::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t Capture::Type>,
    {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_in_capture_lifted_program::<P>(capture_count)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::captures::CaptureReference;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::instructions::InstructionId;
    use crate::programs::operations::Operation;
    use crate::programs::references::discharge::tests::*;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::types::Typed;

    use super::*;

    /// Capture-constant family used by the capture-aware transform tests.
    type ListCapture = CaptureReference<ListIrType>;

    /// Closed list program used by the capture-aware transform tests.
    type ClosedListProgram = ClosedProgram<ListIrValue, ListCapture, ListOperation, Vec<ListCapture>, Vec<ListCapture>>;

    /// Builds a closed program whose attached region reads a reference solely through a capture constant.
    fn closed_list_program_with_nested_reference_capture() -> ClosedListProgram {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut callee_builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let captured_reference =
            callee_builder.add_constant(ListCapture::new(0, ListIrType::Reference(reference_type.clone())));
        let observed = callee_builder
            .add_instruction(ListOperation::Read, Vec::new(), vec![captured_reference], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let callee = builder.import_program(callee);
        let observed = builder.add_instruction(ListOperation::Call, vec![callee], Vec::new(), None).unwrap()[0];
        let source = builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();
        ClosedProgram::new(source, vec![ListIrValue::Reference(reference_type)]).unwrap()
    }

    #[test]
    fn test_reference_discharge_context_bind_discharged() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 4 });
        let allocated = context.bind_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2, 3, 4])).unwrap();
        let reference = allocated.expect_reference("the allocated allocation").unwrap().clone();
        let allocation = reference.allocation_id();

        // A fresh allocation starts unmutated, exposes its identity alias and reference type, and carries no destination
        // reference value because it was discharged rather than preserved.
        assert_eq!(context.live_allocation_ids(), vec![allocation]);
        assert_eq!(context.is_mutated(allocation), Ok(false));
        assert_eq!(context.is_allocation_discharged(allocation), Ok(true));
        assert_eq!(context.allocation_reference(allocation), Ok(allocated.clone()));
        assert_eq!(reference.alias(), &ListAlias { offset: 0, length: 4 });
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 2, 3, 4])));

        // A derived handle narrows the view without touching the allocation's identity, and its accesses act only
        // on the portion it selects.
        let view = context
            .derive_reference(
                &reference,
                ListAlias { offset: 1, length: 2 },
                ReferenceType::new(ListType { length: 2 }),
                |_| unreachable!("the allocation is discharged"),
            )
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap().clone();
        assert_eq!(view.allocation_id(), allocation);
        assert_eq!(context.read(&view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.write(&view, ListIrValue::List(vec![10, 11])), Ok(()));
        assert_eq!(context.swap(&view, ListIrValue::List(vec![20, 30])), Ok(ListIrValue::List(vec![10, 11])));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 20, 30, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(true));
        assert_eq!(context.accumulate(&view, ListIrValue::List(vec![1, 1])), Ok(()));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));

        // Consumption is a complete-value event. Provenance, not type equality, distinguishes the complete-value handle
        // from a derived view: a policy may derive a view whose referent happens to have the allocation's exact type.
        let same_type_view = context
            .derive_reference(&reference, ListAlias { offset: 0, length: 4 }, reference_type.clone(), |_| {
                unreachable!("the allocation is discharged")
            })
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type derived view").unwrap();
        assert_eq!(
            context.operand_allocation(
                &ReferenceDischargeValue::Reference(same_type_view.clone()),
                ListOperation::Call.name(),
            ),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` passes the derived view `ref<list<4>>` of {allocation} across a region boundary, \
                 which carries the complete stored value `ref<list<4>>`; derive the view inside the region instead",
            ))),
        );
        assert_eq!(
            context.consume(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the derived view `ref<list<4>>`; consumption \
                 yields the complete stored value, whose referent is `list<4>`",
            ))),
        );

        // A narrower derived view is rejected by the same provenance check rather than silently yielding the whole
        // allocation's value under the view's type.
        assert_eq!(
            context.consume(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the derived view `ref<list<2>>`; consumption \
                 yields the complete stored value, whose referent is `list<4>`",
            ))),
        );

        // Through the complete-value handle it yields the complete state and unbinds the allocation, so every later
        // access through any handle of that allocation is reported against the exact allocation.
        assert_eq!(context.consume(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));
        assert_eq!(context.live_allocation_ids(), Vec::new());
        let consumed = ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}"));
        assert_eq!(context.read(&reference), Err(consumed.clone()));
        assert_eq!(context.update_discharged_state(allocation, ListIrValue::List(vec![0; 4]), true), Err(consumed),);

        // An allocation ID minted by an unrelated discharge is reported instead of silently addressing whichever
        // allocation occupies the same position here.
        let other = ListDischargeContext::new(ListDestination::new());
        let foreign = other.bind_discharged(reference_type, ListIrValue::List(vec![0; 4])).unwrap();
        let foreign = foreign.expect_reference("the unrelated allocation").unwrap().allocation_id();
        let prefix =
            format!("reference discharge accessed {foreign}, which belongs to an environment other than the active");
        assert!(matches!(
            context.discharged_state(foreign),
            Err(ProgramError::MalformedProgram(message)) if message.starts_with(&prefix),
        ));
    }

    #[test]
    fn test_reference_discharge_context_bind_preserved() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });

        // Binding validates the destination value before inserting an allocation into the environment.
        assert_eq!(
            context.bind_preserved(reference_type.clone(), ListIrValue::List(vec![1, 2])),
            Err(ProgramError::MalformedProgram(
                "reference discharge preserved an allocation as `list<2>` which is not a reference type".to_string(),
            )),
        );
        assert_eq!(
            context.bind_preserved(
                reference_type.clone(),
                ListIrValue::Reference(ReferenceType::new(ListType { length: 1 })),
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge preserved an allocation as `ref<list<1>>` but its handle exposes `ref<list<2>>`"
                    .to_string(),
            )),
        );
        assert_eq!(context.live_allocation_ids(), Vec::new());

        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let bound = context.bind_preserved(reference_type.clone(), destination_reference.clone()).unwrap();
        let reference = bound.expect_reference("the preserved reference").unwrap().clone();
        let allocation = reference.allocation_id();

        // A preserved reference keeps its destination reference value on the handle, so a later access can replay
        // verbatim instead of re-deriving the handle.
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), Some(&destination_reference));
        assert_eq!(context.is_allocation_discharged(allocation), Ok(false));
        assert_eq!(context.allocation_reference(allocation), Ok(bound.clone()));
        assert_eq!(
            context.operand_value(&ReferenceDischargeValue::Reference(reference.clone())),
            Ok(destination_reference.clone()),
        );

        // Every discharged-state service rejects a preserved reference by name rather than silently treating it as state.
        assert_eq!(
            context.read(&reference),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        );
        assert_eq!(
            context.is_mutated(allocation),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {allocation}"
            ))),
        );
        assert_eq!(
            context.update_discharged_state(allocation, ListIrValue::List(vec![0, 0]), true),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge updated the state of preserved {allocation}",
            ))),
        );

        // Deriving on a preserved reference hands the closure the parent handle's exact destination value, so the derived
        // handle cannot disagree with its allocation's fate by construction.
        let view_type = ReferenceType::new(ListType { length: 1 });
        let view_alias = ListAlias { offset: 0, length: 1 };
        let view = context
            .derive_reference(&reference, view_alias, view_type.clone(), |parent| {
                assert_eq!(parent, &ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
                Ok(ListIrValue::Reference(view_type.clone()))
            })
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap();
        assert_eq!(view.allocation_id(), allocation);
        assert_eq!(view.preserved(), Some(&ListIrValue::Reference(view_type)));
    }

    #[test]
    fn test_reference_discharge_context_mark_preserved_consumed() {
        // Consuming a preserved reference yields no state — the replayed operation already produced the destination's own
        // result — but it must still stop the environment from handing the allocation out again, and only a handle denoting
        // the complete stored value can name a consumption.
        let context = ListDischargeContext::new(ListDestination::new());
        let referent = ListType { length: 2 };
        let preserved = context
            .bind_preserved(
                ReferenceType::new(referent.clone()),
                ListIrValue::Reference(ReferenceType::new(referent.clone())),
            )
            .unwrap();
        let reference = preserved.expect_reference("the preserved reference").unwrap().clone();
        let discharged = context.bind_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_allocation = discharged.expect_reference("the discharged reference").unwrap().allocation_id();

        let same_type_view = context
            .derive_reference(&reference, ListAlias { offset: 0, length: 2 }, reference.r#type().clone(), |_| {
                Ok(ListIrValue::Reference(reference.r#type().clone()))
            })
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type preserved view").unwrap();
        assert_eq!(
            context.mark_preserved_consumed(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<2>>`; consumption yields \
                 the complete stored value, whose reference type is `ref<list<2>>`",
                reference.allocation_id(),
            ))),
        );

        let view = context
            .derive_reference(
                &reference,
                ListAlias { offset: 0, length: 1 },
                ReferenceType::new(ListType { length: 1 }),
                |_| Ok(ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))),
            )
            .unwrap();
        let view = view.expect_reference("the derived preserved view").unwrap().clone();
        assert_eq!(
            context.mark_preserved_consumed(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<1>>`; consumption yields \
                 the complete stored value, whose reference type is `ref<list<2>>`",
                reference.allocation_id(),
            ))),
        );
        assert_eq!(context.mark_preserved_consumed(&reference), Ok(()));
        assert_eq!(context.live_allocation_ids(), vec![discharged_allocation]);
        assert_eq!(
            context.mark_preserved_consumed(&reference),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed consumed {}",
                reference.allocation_id()
            ))),
        );

        // A discharged reference is not unbound through the preserved path, which is what keeps the two states from being
        // confused by a rule that dispatched on the wrong one.
        let discharged = discharged.expect_reference("the discharged reference").unwrap();
        assert_eq!(
            context.mark_preserved_consumed(discharged),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge marked discharged {discharged_allocation} as consumed through the preserved path",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_context_update_discharged_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let wrong_state = ListIrValue::List(vec![1]);
        let error = ProgramError::MalformedProgram(
            "reference discharge state has type `list<1>` but allocation `ref<list<2>>` requires `list<2>`".to_string(),
        );

        // A malformed allocation is rejected before an allocation is inserted into the environment.
        assert_eq!(context.bind_discharged(reference_type.clone(), wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.live_allocation_ids(), Vec::new());

        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated allocation").unwrap();
        let allocation = reference.allocation_id();

        // Boundary reconciliation can install a successor without marking a read-only allocation as mutated.
        assert_eq!(context.update_discharged_state(allocation, ListIrValue::List(vec![3, 4]), false), Ok(()));
        assert_eq!(context.discharged_state(allocation), Ok(ListIrValue::List(vec![3, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(false));

        // State updates validate before taking the mutable environment borrow, so failure preserves the prior state
        // and mutation bit.
        assert_eq!(context.update_discharged_state(allocation, wrong_state, true), Err(error));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![3, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_reference_discharge_context_clone() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated allocation").unwrap().clone();

        // A clone shares the environment rather than copying it, which is the contract every stateful Ryft context
        // follows: several handles can denote one allocation, and every one of them must observe the same current state.
        // Isolation is therefore never implicit — a structured rule that must not commit rebuilds its region against
        // an environment of its own through `discharge_region_program`.
        let clone = context.clone();
        clone.accumulate(&reference, ListIrValue::List(vec![10, 10])).unwrap();
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![11, 12])));
        context.accumulate(&reference, ListIrValue::List(vec![1, 1])).unwrap();
        assert_eq!(clone.read(&reference), Ok(ListIrValue::List(vec![12, 13])));
    }

    #[test]
    fn test_reference_discharge_capture_scope() {
        // A scope binds one allocation per capture position. Positions carrying an ordinary value, positions past the end of
        // the scope, and constants that name no capture position at all all resolve to nothing, which is what leaves
        // an unresolvable reference-typed constant to the rejection at the lift site.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the captured allocation").unwrap().allocation_id();

        let empty = ReferenceDischargeCaptureScope::<ListIrValue>::default();
        assert_eq!(empty.allocations(), &[]);
        assert_eq!(empty.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), None);

        let scope = ReferenceDischargeCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]);
        assert_eq!(scope.allocations(), &[None, None, Some(allocation)]);
        assert_eq!(
            scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))),
            Some(allocation)
        );
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 9 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::List(vec![1, 2])), None);

        // Rebinding keeps the seam, which is how a nested region's scope and a fork's remapped scope are built.
        let rebound = scope.with_allocations(vec![Some(allocation)]);
        assert_eq!(rebound.allocations(), &[Some(allocation)]);
        assert_eq!(
            rebound.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 0 }))),
            Some(allocation)
        );
    }

    #[test]
    fn test_reference_discharge_value_reports_operand_kind_mismatches() {
        // A rule that receives the wrong carrier kind gets a diagnostic naming what it expected, which is what keeps
        // an open set of third-party rules diagnosable without each of them inventing its own message.
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 1 });
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1])).unwrap();
        let allocation = allocated.expect_reference("the allocated allocation").unwrap().allocation_id();
        let ordinary: ListDischargeValue = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1]));

        assert_eq!(ordinary.expect_ordinary("an update value"), Ok(&ListIrValue::List(vec![1])));
        assert_eq!(
            allocated.expect_ordinary("an update value"),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an update value but received {allocation} ref<list<1>>",
            ))),
        );
        assert_eq!(
            ordinary.expect_reference("a reference to read"),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received an ordinary value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_value_reports_its_type_and_display() {
        let context = ListDischargeContext::new(ListDestination::new());
        let ordinary =
            ReferenceDischargeValue::<ListDestination, ListReferenceDischarge>::Ordinary(ListIrValue::List(vec![1]));
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let reference = context.bind_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2])).unwrap();

        // An ordinary carrier reports the wrapped destination value's type, while a reference handle reports its own
        // reference type lifted into the destination universe.
        assert_eq!(ordinary.r#type().into_owned(), ListIrType::List(ListType { length: 1 }));
        assert_eq!(reference.r#type().into_owned(), ListIrType::Reference(reference_type));
        assert_eq!(ordinary.to_string(), "[1]");
        let allocation = reference.expect_reference("the allocated allocation").unwrap().allocation_id();
        assert_eq!(reference.to_string(), format!("{allocation} ref<list<2>>"));
        assert_eq!(
            format!("{reference:?}"),
            format!(
                "Reference(ReferenceDischargeReference {{ allocation_id: {allocation:?}, denotes_complete_value: true, \
                 alias: ListAlias \
                 {{ offset: 0, length: 2 }}, type: ReferenceType {{ referent: ListType {{ length: 2 }} }}, \
                 binding: Discharged }})",
            ),
        );
    }

    #[test]
    fn test_program_discharge_references() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.capture_count(), 0);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2> = list.select %0
                    %3:list<2> = list.splice %0 %1
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_program_discharge_references_rejects_consumed_external_allocations() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose state must remain owned by the caller"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_discharge_references_in_capture_lifted_program() {
        // The attached region names the caller-owned reference only through a capture constant. Capture lifting moves
        // that reference into the entry input prefix while leaving the nested constant for the discharge transform to
        // resolve through its capture scope.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();

        let targets = lifted.reference_discharge_targets(1).unwrap();
        assert_eq!(targets, vec![ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })]);

        let discharged = lifted.discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1).unwrap();
        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_program_partially_discharge_references() {
        // The kernel-pipeline shape, in a universe that mentions no arrays: one caller-owned allocation is selected
        // and becomes threaded state, while the other survives as a reference the rewritten program still accesses
        // through the very operations the source used.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![kernel], None).unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![pipeline, update], None).unwrap();
        builder.add_instruction(ListOperation::Swap, Vec::new(), vec![kernel, observed], None).unwrap();
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        let discharged = source.partially_discharge_references(0, &targets[..1]);
        let discharged = discharged.unwrap();

        // The selected allocation became an ordinary state input at its own boundary position and publishes its final
        // state as a hidden output; the preserved reference kept its reference type, so it binds nothing at all.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>>, %2:list<2> .
                let %3:list<2> = list.read %1
                    %4:list<2> = list.select %0
                    %5:list<2> = list.add %4 %2
                    %6:list<2> = list.splice %0 %5
                    %7:list<2> = list.swap %1 %3
                in (%3, %6)"},
        );

        // The result deliberately proves nothing about reference freedom, and asking for the proof reports the
        // surviving reference rather than converting.
        assert_eq!(
            ReferenceDischargeResult::try_from(discharged).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_preserves_an_unselected_internal_target() {
        // An interior allocation is selectable in its own right, so a program can normalize its pipeline state while
        // the allocation a kernel body addresses is allocated, viewed, accessed, and consumed as a reference
        // throughout.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![allocation], None)
            .unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Selecting nothing preserves the allocation, so the whole reference language survives: the view operation is
        // replayed too, and the derived handle consumes the reference that replay produced rather than re-deriving
        // the chain at the access.
        let discharged = source.clone().partially_discharge_references(0, &[]);
        let discharged = discharged.unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<4>, %1:list<2> .
                let %2:ref<list<4>> = list.reference_new %0
                    %3:ref<list<2>> = list.slice %2
                    list.add_update %3 %1
                    %4:list<4> = list.freeze %2
                in (%4)"},
        );

        // Selecting the allocation instead discharges it, which is the everything-selected case and therefore has to
        // agree with full discharge exactly.
        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![ReferenceDischargeTarget::Internal {
                instruction: InstructionId::new(source.entry_region_ref().id(), 0),
                output_index: 0,
            }],
        );
        let selected = source.clone().partially_discharge_references(0, targets.as_slice());
        let selected = ReferenceDischargeResult::try_from(selected.unwrap()).unwrap();
        let full = source.discharge_references(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
    }

    #[test]
    fn test_program_partially_discharge_references_allows_consuming_a_preserved_external_allocation() {
        // A preserved reference has no external binding to describe: the program keeps the consuming operation, and
        // the caller passes its reference to that operation directly, so partial discharge preserves the source
        // program's consumption.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.freeze %0
                in (%1)"},
        );
        // Returning the allocation afterwards is a use of it like any other, so the consumed allocation is reported at
        // the output that names it rather than published as a stale reference.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![frozen, external],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // An allocation rendering embeds the identity of the environment that minted it, which is process-global,
        // so the assertion pins everything around it.
        let error = source.partially_discharge_references(0, &[]).unwrap_err();
        let ProgramError::MalformedProgram(message) = &error else {
            panic!("expected a malformed-program rejection but got {error:?}");
        };
        assert!(message.starts_with("reference discharge accessed consumed reference allocation "), "{message}");
        assert!(message.ends_with(":0"), "{message}");
    }

    #[test]
    fn test_program_partially_discharge_references_threads_a_preserved_allocation_beside_discharged_state() {
        // A structured boundary carries both kinds of allocation at once: a discharged carry crosses as immutable
        // state and is widened with a published successor, while a preserved carry crosses as the reference it already
        // is, at its own declared operand position, and widens nothing at all.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_state = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee_kernel = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed =
            callee_builder.add_instruction(ListOperation::Read, Vec::new(), vec![callee_kernel], None).unwrap()[0];
        callee_builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![callee_state, observed], None)
            .unwrap();
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee = builder.import_program(callee);
        let observed =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![pipeline, kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.partially_discharge_references(0, &targets[..1]).unwrap();

        // The selected allocation's entering state occupies its own operand position and its successor is appended as
        // a published output; the preserved reference's operand position still carries a reference, and the rebuilt
        // callee performs the read on it exactly as the source did.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>> .
                let %2:list<2>, %3:list<2> = list.call %0 %1 [
                    callee={
                        lambda %0:list<2>, %1:ref<list<2>> .
                        let %2:list<2> = list.read %1
                            %3:list<2> = list.select %0
                            %4:list<2> = list.add %3 %2
                            %5:list<2> = list.splice %0 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_program_partially_discharge_references_validates_targets_against_the_program() {
        // The targets are checked before anything is replayed, so a target this program does not expose is reported
        // against the program rather than surfacing later as an allocation that never appeared.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source
                .partially_discharge_references(
                    0,
                    &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 3 })],
                )
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 3 which is not selectable in this program"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_in_capture_lifted_program() {
        // Selecting nothing preserves the capture as a reference and explicitly threads it into the nested region.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();
        let discharged = lifted
            .partially_discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1, &[])
            .unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:ref<list<2>> .
                        let %1:list<2> = list.read %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_closed_program_discharge_references() {
        let closed = closed_list_program_with_nested_reference_capture();
        let discharged = closed.discharge_references::<ListReferenceDischarge>().unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }
}
