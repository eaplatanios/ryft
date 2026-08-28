use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::references::ReferenceError;
use crate::programs::references::types::ReferenceType;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;

/// Process-local identity shared by all aliased [`Reference`] handles backed by the same allocation. It supports alias
/// identity checks and diagnostics within one process, carries no structural type information, and is never serialized
/// into a program or compilation key. The identity may be reused after all handles, observations, and replacement
/// transactions tied to that allocation are dropped.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId(usize);

impl ReferenceId {
    /// Creates a [`ReferenceId`] from a live [`ReferenceHolder`] memory address.
    #[inline]
    fn from_address(address: usize) -> Self {
        Self(address)
    }
}

/// Monotonically increasing version of one [`Reference`]'s mutable state. A newly allocated reference starts at
/// generation zero. Each committed synchronous mutation or submitted asynchronous mutation advances the generation,
/// and every asynchronous replacement transaction and completion path retains the generation it belongs to. The
/// reference applies a delayed completion only while that generation remains current, preventing an older execution
/// from completing or poisoning state committed by a newer mutation. Generations are local to one reference allocation
/// and do not identify references across allocations or processes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceGeneration(u64);

impl ReferenceGeneration {
    /// Returns the initial [`ReferenceGeneration`] for a newly allocated [`ReferenceHolder`].
    #[inline]
    fn initial() -> Self {
        Self(0)
    }

    /// Returns the successor [`ReferenceGeneration`], or [`None`] when the supported number of generations
    /// is exhausted.
    #[inline]
    fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Cloneable handle for a referenced [`Value`]. Cloning a [`Reference`] creates an alias that shares the same
/// mutable state. Reading clones the current value. Writes and swaps validate the exact declared referent type before
/// atomically replacing the shared snapshot, and a consuming freeze invalidates the complete alias family. Reference
/// referents are expected to have immutable value clone semantics so that later replacements and updates cannot change
/// any initializers, read results, or swap results retained by the caller.
///
/// A directly created eager handle has no [`Program`](crate::Program)-owned scope. It remains valid until an explicit
/// freeze or until the last handle in its alias family is dropped. Statically validated local program roots that are
/// never frozen are implicitly discarded when their non-escaping interpretation environment is released.
///
/// # Reference State Machine
///
/// Each reference alias family has one synchronized state. The following diagram illustrates its lifecycle:
///
/// ```mermaid
/// stateDiagram-v2
///   [*] --> Ready: new at generation 0
///   Ready --> Ready: write, swap, or update (generation + 1)
///   Ready --> Taken: synchronous take or asynchronous submission (generation + 1)
///   Pending --> Taken: chained asynchronous submission (generation + 1)
///   Taken --> Ready: replace synchronously
///   Taken --> Pending: commit asynchronous replacement
///   Taken --> Poisoned: guard drop or backend failure
///   Pending --> Ready: backend completion succeeds
///   Pending --> Poisoned: backend completion fails
///   Ready --> Frozen: freeze
/// ```
///
/// `Ready` exposes a completed immutable snapshot. `Taken` contains no accessible value: the value was either returned
/// by [`ReadyReferenceGuard::take`] or made unavailable after [`PreparedReferenceReplacement::begin`] recorded a
/// submitted asynchronous mutation. The resulting [`TakenReferenceGuard`] or [`ReferenceReplacementTransaction`]
/// retains the state lock and owns the obligation to replace the missing value or poison the reference. Dropping either
/// owning type without fulfilling that obligation poisons the reference automatically.
///
/// Committing an asynchronous replacement changes the state to `Pending`, which stores the replacement together with
/// its cumulative [`ReferenceCompletion`]. Ordinary value access waits for that completion without holding the state
/// lock, then changes the state to `Ready` on success or `Poisoned` on failure. Generations ensure that completion of
/// an older execution cannot affect a newer mutation. `Frozen` and `Poisoned` are terminal states.
///
/// Read-only asynchronous executions do not enter a separate state. They publish completion leases on `Ready` or
/// `Pending`. A mutation waits for those leases before it may remove the value, preventing donated storage from racing
/// a reader that still observes the current snapshot.
///
/// Equality and hashing identify the mutable storage location, not this handle's structural type. Clones and
/// identity-renamed handles into the same alias family therefore compare equal and hash identically even when their
/// handle-local referent types use different type-identity namespaces.
///
/// Handles are pointer-sized. Cloning a handle is a single reference-count increment because exact clones share one
/// immutable handle containing the same local referent type and identity mappings. Identity renaming allocates a new
/// handle with a different type-identity namespace, shares the same reference allocation, and maps values
/// bidirectionally at the shared-state boundary. Because exact clones share that metadata, `Reference<V>` is [`Send`]
/// and [`Sync`] only when the referent value and its type and identity metadata are all safe to share across threads
/// (i.e., they are all `Send + Sync`).
#[cfg_attr(doc, aquamarine::aquamarine)]
#[derive(Parameter)]
pub struct Reference<V: Value> {
    /// Immutable [`ReferenceHandle`] shared by exact clones. All runtime mutability lives behind its shared state lock.
    handle: Arc<ReferenceHandle<V>>,
}

impl<V: Value> Reference<V> {
    /// Creates a new independent [`Reference`] whose underlying storage is initialized with `value`.
    pub fn new(value: V) -> Result<Self, ReferenceError> {
        let referent_type = value.r#type().into_owned();
        if referent_type.is_reference() {
            return Err(ReferenceError::NestedReferent { referent_type: referent_type.to_string() });
        }
        Ok(Self {
            handle: Arc::new(ReferenceHandle {
                holder: Arc::new(ReferenceHolder {
                    referent_type: referent_type.clone(),
                    state: Mutex::new(ReferenceState::Ready {
                        value,
                        generation: ReferenceGeneration::initial(),
                        read_leases: Vec::new(),
                    }),
                }),
                r#type: ReferenceType::new(referent_type),
                storage_to_handle: TypeIdentityRenaming::new(),
                handle_to_storage: TypeIdentityRenaming::new(),
            }),
        })
    }

    /// Returns the process-local [`ReferenceId`] of this [`Reference`], which remains stable while any alias is alive.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        ReferenceId::from_address(Arc::as_ptr(&self.handle.holder) as usize)
    }

    /// Returns `true` if this handle uses the reference allocation's original type identities, and `false` otherwise.
    /// When this function returns `true`, values cross between the handle-local representation and shared reference
    /// state without type-identity renaming. A handle produced by [`Self::rename_type_identities`] still refers to the
    /// same reference allocation but returns `false` when that operation changes any identity.
    #[inline]
    pub fn uses_storage_type_identities(&self) -> bool {
        self.handle.storage_to_handle.is_identity() && self.handle.handle_to_storage.is_identity()
    }

    /// Returns an immutable snapshot of this [`Reference`]'s current value. If a submitted mutation is still pending,
    /// this function waits for its completion without holding the state mutex and then reads the committed value. It
    /// does not wait for active read-only executions because they may safely share the same immutable snapshot. For an
    /// identity-renamed alias, the returned value uses that alias's referent type identities.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] after the alias family has been consumed,
    /// [`ReferenceError::ExecutionPoisoned`] after a submitted mutation fails, or [`ReferenceError::Poisoned`] after
    /// unexpected mutex poisoning, and [`ReferenceError::ValueReconstruction`] if the stored value cannot be converted
    /// to this alias's type identities.
    pub fn read(&self) -> Result<V, ReferenceError> {
        let state = self.lock_ready_state(false)?;
        let ReferenceState::Ready { value, .. } = &*state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        value
            .rename_type_identities(&self.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
    }

    /// Atomically replaces this [`Reference`]'s current value. This function first waits for pending mutations and
    /// active read-only executions to finish. It then verifies that `replacement` exactly matches this alias's referent
    /// type, converts identity-renamed values to the shared storage representation, commits the replacement, and
    /// advances the reference generation. The previous value is not reconstructed or returned.
    ///
    /// Reference-state errors take precedence over replacement validation. Once the current value is ready, type
    /// validation, identity conversion, and generation-exhaustion errors leave the stored value and generation
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns the applicable lifecycle-state error if the reference cannot be mutated,
    /// [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type,
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails, or
    /// [`ReferenceError::GenerationExhausted`] if no next generation can be assigned.
    pub fn write(&self, replacement: V) -> Result<(), ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        let replacement = self.prepare_replacement(replacement)?;
        Self::commit_replacement(current, generation, replacement)
    }

    /// Atomically replaces this [`Reference`]'s current value and returns its previous value. Like [`Self::write`],
    /// this function waits for pending mutations and active read-only executions, requires an exact replacement type,
    /// converts identity-renamed values to and from shared storage, and advances the reference generation. The returned
    /// snapshot uses this alias's referent type identities.
    ///
    /// Reference-state errors take precedence over replacement validation. Once the current value is ready, the
    /// replacement, previous-value reconstruction, and next generation are all prepared before either the stored value
    /// or generation is changed, so errors during those steps leave the reference unchanged.
    ///
    /// # Errors
    ///
    /// Returns the applicable lifecycle-state error if the reference cannot be mutated,
    /// [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type,
    /// [`ReferenceError::ValueReconstruction`] if either identity conversion fails, or
    /// [`ReferenceError::GenerationExhausted`] if no next generation can be assigned.
    pub fn swap(&self, replacement: V) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        let replacement = self.prepare_replacement(replacement)?;
        let previous = current
            .rename_type_identities(&self.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        Self::commit_replacement(current, generation, replacement)?;
        Ok(previous)
    }

    /// Atomically transforms this [`Reference`]'s stored value and returns a result derived from its previous snapshot.
    /// This function waits until the reference can be mutated, converts the current value into this reference's type
    /// identity space, and passes that immutable snapshot to `update`. The closure returns both the replacement value
    /// and the result returned to the caller. The replacement is validated and converted back to the shared storage
    /// representation before either the value or its generation is changed, so every error leaves the reference
    /// unchanged.
    ///
    /// `update` runs while this reference's non-reentrant state mutex is locked. It must not access this reference
    /// or another alias of the same reference, because doing so would attempt to acquire that mutex recursively.
    ///
    /// # Errors
    ///
    /// Returns errors from acquiring or reconstructing the current reference state, propagates errors returned by
    /// `update`, rejects a replacement whose type does not exactly match the reference's referent type, and returns
    /// a generation-exhaustion error before modifying the reference.
    pub fn update<R, F: FnOnce(&V) -> Result<(V, R), ProgramError>>(&self, update: F) -> Result<R, ProgramError> {
        let mut state = self.lock_ready_state(true).map_err(ProgramError::custom)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        let local = current
            .rename_type_identities(&self.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
            .map_err(ProgramError::custom)?;
        let (updated, result) = update(&local)?;
        let stored = self.prepare_replacement(updated).map_err(ProgramError::custom)?;
        Self::commit_replacement(current, generation, stored).map_err(ProgramError::custom)?;
        Ok(result)
    }

    /// Returns this [`Reference`]'s current value and permanently invalidates its complete alias family. This function
    /// waits for pending mutations and active read-only executions, converts the stored value to this alias's referent
    /// type identities, and only then transitions the shared state to `Frozen`. After it succeeds, every alias reports
    /// [`ReferenceError::Frozen`] on subsequent access. A failed value conversion leaves the reference unfrozen.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] if the alias family was already consumed,
    /// [`ReferenceError::ExecutionPoisoned`] after a submitted mutation fails, or [`ReferenceError::Poisoned`] after
    /// unexpected mutex poisoning, and [`ReferenceError::ValueReconstruction`] if the stored value cannot be converted
    /// to this alias's type identities.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value, .. } = &*state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        let value = value
            .rename_type_identities(&self.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        *state = ReferenceState::Frozen;
        Ok(value)
    }

    /// Acquires this [`Reference`]'s observable lifecycle state for backend-managed access. This function returns as
    /// soon as the state mutex is acquired and deliberately does not await a `Pending` value or active read leases.
    /// The returned [`ReadyOrPendingReferenceGuard`] always protects `Ready` or `Pending` state and can observe the
    /// current snapshot, publish a submitted read, or begin one of the typed replacement protocols.
    ///
    /// Ordinary value access should use [`read`](Self::read), [`write`](Self::write), [`swap`](Self::swap),
    /// [`update`](Self::update), or [`freeze`](Self::freeze), which reconcile pending work before accessing the value.
    /// Backends that lock multiple references must acquire them in ascending [`ReferenceId`] order and retain every
    /// resulting guard or replacement typestate until all submitted hidden replacements have been validated and
    /// committed. The owning typestates keep the mutex locked while the value is `Taken`, so another call to `lock`
    /// cannot observe that internal state.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] after the alias family is consumed, [`ReferenceError::ExecutionPoisoned`]
    /// after a replacement obligation or backend execution fails, or [`ReferenceError::Poisoned`] after unexpected
    /// mutex poisoning.
    #[inline]
    pub fn lock(&self) -> Result<ReadyOrPendingReferenceGuard<'_, V>, ReferenceError> {
        let state = self.lock_holder_state()?;
        match &*state {
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => {
                Ok(ReadyOrPendingReferenceGuard { guard: ReferenceGuard { reference: self, state } })
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => {
                unreachable!("the typestate owning a taken reference retains its state mutex")
            }
        }
    }

    /// Acquires the [`Reference`] in `Ready` state after reconciling work that prevents the requested value access.
    /// Each iteration acquires the raw mutex through [`Self::lock_holder_state`]. A `Pending` value causes this
    /// function to release the mutex, await its cumulative completion, reacquire the mutex, and apply the result only
    /// if that generation is still current. When read leases must also finish, the function prunes completed leases,
    /// releases the mutex while awaiting the remaining leases, and retries. It never awaits backend work while holding
    /// the mutex.
    ///
    /// Unlike [`lock`](Self::lock), this function is the private ordinary-access path: it hides pending-state
    /// reconciliation and returns a raw guard proven to contain `Ready`. It does not return
    /// [`ReadyOrPendingReferenceGuard`] because ordinary reads and mutations do not expose `Pending` state or use the
    /// `Taken` state and its guard-drop poisoning rule.
    ///
    /// # Parameters
    ///
    ///   - `wait_for_read_leases`: Whether the returned state must also have no active readers. Reads pass `false`
    ///     because they may share the current immutable snapshot while writes, swaps, updates, and freezing pass `true`
    ///     because they may replace or consume storage still observed by a submitted reader.
    ///
    /// # Errors
    ///
    /// Returns the lifecycle error represented by `Frozen` or `Poisoned`, propagates unexpected mutex poisoning from
    /// [`Self::lock_holder_state`], and reports a failed pending completion as [`ReferenceError::ExecutionPoisoned`].
    /// The internal `Taken` state cannot be observed because its owning typestate retains the mutex.
    fn lock_ready_state(
        &self,
        wait_for_read_leases: bool,
    ) -> Result<MutexGuard<'_, ReferenceState<V>>, ReferenceError> {
        loop {
            let mut state = self.lock_holder_state()?;
            let wait = match &mut *state {
                ReferenceState::Ready { read_leases, .. } if wait_for_read_leases => {
                    read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
                    (!read_leases.is_empty()).then(|| (None, read_leases.clone()))
                }
                ReferenceState::Ready { .. } => return Ok(state),
                ReferenceState::Pending { generation, completion, .. } => {
                    Some((Some((*generation, completion.clone())), Vec::new()))
                }
                ReferenceState::Frozen => return Err(ReferenceError::Frozen),
                ReferenceState::Poisoned(reason) => {
                    return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
                }
                ReferenceState::Taken { .. } => {
                    unreachable!("the typestate owning a taken reference retains its state mutex")
                }
            };

            let Some((pending, read_leases)) = wait else {
                return Ok(state);
            };

            // `wait` owns cloned completion tokens, so it no longer borrows the protected state. Release the
            // non-reentrant mutex before awaiting backend work: holding it would block unrelated reference access,
            // and the pending-completion path below must reacquire this same mutex to apply the result conditionally.
            drop(state);

            if let Some((generation, completion)) = pending {
                let result = completion.r#await();
                let mut state = self.lock_holder_state()?;
                Self::apply_pending_completion(&mut state, generation, result);
            } else {
                for lease in read_leases {
                    match lease.r#await() {
                        Ok(()) | Err(_) => {
                            // A completed read lease is safe to release regardless of its owning execution result.
                        }
                    }
                }
            }
        }
    }

    /// Acquires the raw lifecycle-state mutex without interpreting the current [`ReferenceState`]. This is the common
    /// lowest-level locking primitive for [`lock`](Self::lock) and [`Self::lock_ready_state`]. It neither resolves a
    /// `Pending` completion nor waits for read leases. It recovers standard-library mutex poisoning only when
    /// [`TakenReferenceGuard::drop`] already replaced the lifecycle state with explicit terminal `Poisoned` state
    /// before an unwind released the mutex. Any other mutex poisoning represents an unexpected panic while the
    /// reference was in a usable state and is reported rather than cleared.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Poisoned`] when the mutex was poisoned while the lifecycle state was not explicitly
    /// `Poisoned`.
    fn lock_holder_state(&self) -> Result<MutexGuard<'_, ReferenceState<V>>, ReferenceError> {
        match self.handle.holder.state.lock() {
            Ok(state) => Ok(state),
            Err(poisoned) => {
                let state = poisoned.into_inner();
                if matches!(*state, ReferenceState::Poisoned(_)) {
                    self.handle.holder.state.clear_poison();
                    Ok(state)
                } else {
                    Err(ReferenceError::Poisoned)
                }
            }
        }
    }

    /// Creates an alias of this [`Reference`] whose referent type uses the provided [`TypeIdentityRenaming`]. The
    /// returned reference shares the same stored value and mutation state as `self`. Reading through the alias renames
    /// the stored value's type identities, and writing through it translates those identities back before validating
    /// the replacement. Because values must be translated in both directions, `renaming` must map every identity used
    /// by the referent type uniquely and reversibly.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        // First, apply the requested current-to-renamed mapping to this alias's referent type. The remaining work
        // derives the two complete value conversions needed to keep the new alias backed by the existing allocation.
        let current_type = self.handle.r#type.referent();
        let renamed_type = current_type.rename_identities(renaming)?;

        // A renaming that merges two of the referent's identities cannot be inverted for value reconstruction. Detect
        // the collision here and report it in the caller's direction. Deriving the inverse below would otherwise
        // surface it backwards, as a target renamed to two sources.
        let mut renamed_identities: Vec<(<V::Type as Type>::Identity, <V::Type as Type>::Identity)> = Vec::new();
        for (_, identity) in current_type.identities() {
            let renamed = renaming.rename(identity);
            match renamed_identities.iter().find(|(existing, _)| *existing == renamed) {
                Some((_, previous)) if previous != identity => {
                    return Err(TypeError::invalid(format!(
                        "type identities `{previous}` and `{identity}` are both renamed to `{renamed}`",
                    )));
                }
                Some(_) => {}
                None => renamed_identities.push((renamed, identity.clone())),
            }
        }

        // Derive the inverse step from the renamed referent back to the current referent. The explicit collision check
        // above keeps errors for non-injective user mappings oriented from their source identities to their target.
        let inverse_step =
            V::Type::derive_identity_renaming(std::slice::from_ref(&renamed_type), std::slice::from_ref(current_type))?;

        // Reads begin with values represented in the shared storage identity space. Compose the existing
        // storage-to-current conversion with the requested current-to-renamed step to produce the new read conversion.
        let referent_type = &self.handle.holder.referent_type;
        let mut storage_to_handle = TypeIdentityRenaming::new();
        for (_, identity) in referent_type.identities() {
            storage_to_handle
                .insert(identity.clone(), renaming.rename(&self.handle.storage_to_handle.rename(identity)))?;
        }

        // Writes travel in the opposite direction. Compose the renamed-to-current inverse with this alias's existing
        // current-to-storage conversion to produce the new write conversion.
        let mut handle_to_storage = TypeIdentityRenaming::new();
        for (_, identity) in renamed_type.identities() {
            handle_to_storage
                .insert(identity.clone(), self.handle.handle_to_storage.rename(&inverse_step.rename(identity)))?;
        }

        // Check each complete conversion against its concrete endpoint types. This catches inconsistent identity
        // implementations even when every individual mapping above was constructible.
        if referent_type.rename_identities(&storage_to_handle)? != renamed_type
            || renamed_type.rename_identities(&handle_to_storage)? != *referent_type
        {
            return Err(TypeError::invalid(
                "reference identity renaming must admit an exact bidirectional value reconstruction",
            ));
        }

        // Only alias-local type and conversion metadata is new. Sharing the existing allocation keeps the returned
        // alias attached to the same mutable value and lifecycle state.
        Ok(Self {
            handle: Arc::new(ReferenceHandle {
                holder: Arc::clone(&self.handle.holder),
                r#type: ReferenceType::new(renamed_type),
                storage_to_handle,
                handle_to_storage,
            }),
        })
    }

    /// Resolves the pending value for `generation` using its backend completion result. If `state` is still `Pending`
    /// at `generation`, a successful result makes its value and read leases `Ready`, while a failed result replaces it
    /// with terminal `Poisoned` state carrying the backend failure reason. If the state is no longer pending at that
    /// generation, this function leaves it unchanged and returns `false`. This prevents a stale completion from
    /// overwriting newer reference state.
    fn apply_pending_completion(
        state: &mut ReferenceState<V>,
        generation: ReferenceGeneration,
        result: Result<(), Arc<str>>,
    ) -> bool {
        if !matches!(state, ReferenceState::Pending { generation: current, .. } if *current == generation) {
            return false;
        }
        match result {
            Ok(()) => {
                // The placeholder is unobservable. The state mutex is held and `*state` is rewritten immediately
                // in both directions below, so no other thread can see the transient `Frozen`.
                let previous = std::mem::replace(state, ReferenceState::Frozen);
                let ReferenceState::Pending { value, generation, read_leases, .. } = previous else {
                    unreachable!("completion generation was validated as pending")
                };
                *state = ReferenceState::Ready { value, generation, read_leases };
            }
            Err(reason) => *state = ReferenceState::Poisoned(reason),
        }
        true
    }

    /// Validates a handle-local replacement value and converts it to the shared storage representation. The input must
    /// exactly match this [`Reference`]'s referent type. After applying the handle-to-storage identity mapping, the
    /// converted value must exactly match the reference allocation's canonical referent type. Both checks finish before
    /// any reference state is changed, so callers may safely commit the returned value.
    fn prepare_replacement(&self, value: V) -> Result<V, ReferenceError> {
        // `Typed::r#type` may return a `Cow` that borrows its value. Keeping that borrow inside this closure lets each
        // validation finish before the corresponding value is moved into identity renaming or returned to the caller.
        let validate_type = |value: &V, expected: &V::Type| -> Result<(), ReferenceError> {
            let actual = value.r#type();
            if actual.as_ref() == expected {
                return Ok(());
            }
            Err(ReferenceError::ReferentTypeMismatch { expected: expected.to_string(), actual: actual.to_string() })
        };

        // Reject the replacement in the handle's public type-identity namespace before translating it for storage.
        validate_type(&value, self.handle.r#type.referent())?;

        // Every alias stores values in the reference allocation's shared identity namespace. This mapping converts
        // the handle-local replacement into that storage representation.
        let stored = value
            .rename_type_identities(&self.handle.handle_to_storage)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;

        // Identity renaming is implemented by the value family, so verify that its result exactly matches the root
        // referent type before allowing the value to reach shared state.
        validate_type(&stored, &self.handle.holder.referent_type)?;

        Ok(stored)
    }

    /// Commits a prepared replacement to an already-locked `Ready` state and advances its [`ReferenceGeneration`].
    /// The next generation is computed before either field is changed. If the generation space is exhausted, this
    /// function returns [`ReferenceError::GenerationExhausted`] and leaves both `current` and `generation` unchanged.
    /// `replacement` must already have been validated and converted to the shared storage representation by
    /// [`Self::prepare_replacement`].
    fn commit_replacement(
        current: &mut V,
        generation: &mut ReferenceGeneration,
        replacement: V,
    ) -> Result<(), ReferenceError> {
        let next_generation = generation.next().ok_or(ReferenceError::GenerationExhausted)?;
        *current = replacement;
        *generation = next_generation;
        Ok(())
    }
}

impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
        // Exact clones share one immutable handle and its type-identity mappings, so cloning is a single
        // reference-count increment.
        Self { handle: Arc::clone(&self.handle) }
    }
}

impl<V: Value> Debug for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Reference")
            .field("id", &self.id())
            .field("type", &self.handle.r#type)
            .finish()
    }
}

impl<V: Value> Display for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ReferenceType::fmt` supplies the `ref<...>` wrapper. `Display` deliberately renders only that handle-local
        // type because the `Value` rendering contract requires deterministic outputs (that is because this rendering
        // backs diagnostics, rendered-program tests, and the debug-assertions transform-cache determinism checks), so
        // the process-local holder address must not leak here. Runtime identity remains visible through `Debug`.
        Display::fmt(&self.handle.r#type, formatter)
    }
}

impl<V: Value> PartialEq for Reference<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle.holder, &other.handle.holder)
    }
}

impl<V: Value> Eq for Reference<V> {}

impl<V: Value> Hash for Reference<V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.handle.holder).hash(state);
    }
}

impl<V: Value> Typed for Reference<V> {
    type Type = ReferenceType<V::Type>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.handle.r#type)
    }
}

/// Immutable metadata for one [`Reference`] handle. Exact clones share this metadata. An identity-renamed alias
/// receives a new [`ReferenceHandle`] with its own [`ReferenceType`] and conversion mappings while continuing to
/// share the same [`ReferenceHolder`] and runtime state.
struct ReferenceHandle<V: Value> {
    /// Shared allocation whose pointer identity defines reference equality, hashing, and [`ReferenceId`].
    holder: Arc<ReferenceHolder<V>>,

    /// [`ReferenceType`] exposed through this handle.
    r#type: ReferenceType<V::Type>,

    /// [`TypeIdentityRenaming`] that converts stored values from the allocation's identities to this
    /// [`ReferenceHandle`]'s identities.
    storage_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// [`TypeIdentityRenaming`] that converts values from this [`ReferenceHandle`]'s identities to the allocation's
    /// identities.
    handle_to_storage: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

/// Allocation shared by every handle in one [`Reference`] alias family.
struct ReferenceHolder<V: Value> {
    /// Canonical referent [`Type`] of stored values. Identity-renamed aliases may expose a different handle-local
    /// type and convert at this allocation boundary. The canonical type is immutable and available without locking
    /// so replacement validation does not need to acquire the lifecycle mutex.
    referent_type: V::Type,

    /// Synchronized value and lifecycle state.
    state: Mutex<ReferenceState<V>>,
}

/// Value and lifecycle state shared by one [`Reference`] alias family.
enum ReferenceState<V: Value> {
    /// Current value is available and the mutation that produced it has completed.
    Ready {
        /// Current immutable value in the allocation's canonical identity space.
        value: V,

        /// [`ReferenceGeneration`] of `value`.
        generation: ReferenceGeneration,

        /// Submitted read-only executions that may still be using `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// Current replacement is committed, but the execution producing it has not completed.
    Pending {
        /// Pending replacement in the allocation's canonical identity space.
        value: V,

        /// [`ReferenceGeneration`] assigned to `value`.
        generation: ReferenceGeneration,

        /// Completion of this [`ReferenceGeneration`] and every predecessor on which it depends.
        completion: ReferenceCompletion,

        /// Submitted read-only executions that may still be using `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// No value is accessible while a backend finishes replacing it at the next [`ReferenceGeneration`].
    Taken {
        /// [`ReferenceGeneration`] assigned to the missing replacement.
        generation: ReferenceGeneration,
    },

    /// No trustworthy value remains after a submitted mutation failed or a `Taken` guard was released
    /// without replacing the missing value.
    Poisoned(Arc<str>),

    /// Value was consumed by [`Reference::freeze`].
    Frozen,
}

/// Backend-facing exclusive access to a [`Reference`] in observable `Ready` or `Pending` state. This guard acquires
/// the reference mutex without waiting for pending backend work. It can capture a coherent [`ReferenceObservation`],
/// preserve the current value for a submitted read-only execution, prepare an asynchronous replacement, or wait for
/// the stricter [`ReadyReferenceGuard`] needed by synchronous extraction. Each function that creates a mandatory
/// replacement obligation consumes this guard and returns a more specific owning type. The guard holds the mutex for
/// its entire lifetime and cannot move to another thread.
///
/// An asynchronous read keeps the state `Ready` or `Pending` and records how long its snapshot must remain valid:
///
/// ```mermaid
/// flowchart LR
///   state["Ready or Pending"] -->|observe| observation["ReferenceObservation"]
///   observation --> submission["Submit backend read"]
///   submission -->|use snapshot after dependency| execution["Backend execution"]
///   submission -->|preserve value until completion| state
/// ```
///
/// [`Self::preserve_value_until`] is infallible because this guard can protect only observable state. Before a later
/// mutation removes the value, [`Self::prepare_replacement`] prunes completed read records and returns every completion
/// for an execution that may still be reading the snapshot.
///
/// An asynchronous mutation first observes the value and prepares backend work. It then reacquires the guard and
/// verifies that the observed generation is still current before submitting that work:
///
/// ```mermaid
/// flowchart LR
///   state["Ready or Pending"] -->|observe| prepare["Prepare execution"]
///   prepare --> validate["Reacquire guard and check observation"]
///   validate -->|stale| state
///   validate -->|prepare_replacement| preparation{"Read leases active?"}
///   preparation -->|yes| retry["Wait and retry"]
///   retry --> state
///   preparation -->|no| prepared["PreparedReferenceReplacement"]
///   prepared -->|submission succeeds; begin| transaction["ReferenceReplacementTransaction (Taken)"]
///   transaction -->|validate| validated["ValidatedPendingReplacementTransaction"]
///   validated -->|commit| pending["ReadyOrPendingReferenceGuard (Pending)"]
///   pending -->|completion succeeds| ready["Ready"]
///   pending -->|completion fails| poisoned["Poisoned"]
///   transaction -->|poison or drop| poisoned
/// ```
///
/// Preparation checks active readers and computes the next generation before submission, while leaving the state
/// unchanged. [`PreparedReferenceReplacement::begin`] is called only after submission succeeds. It changes the state
/// to `Taken` and returns a [`ReferenceReplacementTransaction`] that owns both the exact guard and the submitted
/// completion. Successful validation returns a [`ValidatedPendingReplacementTransaction`] whose commit cannot fail. A
/// multi-reference backend validates every replacement before committing any of them, then retains every guard returned
/// by commit until all commits finish. This prevents another thread from observing a partially published state update.
/// Guards for multiple references must be acquired and retained in ascending [`ReferenceId`] order.
///
/// A synchronous or potentially donating backend follows the following shorter extraction protocol:
///
/// ```mermaid
/// flowchart LR
///   observable["ReadyOrPendingReferenceGuard"] -->|wait_until_ready| ready["ReadyReferenceGuard (Ready)"]
///   ready -->|take| taken["TakenReferenceGuard (Taken)"]
///   taken -->|replace| ready
///   taken -->|poison or drop| poisoned["Poisoned"]
/// ```
///
/// [`Self::wait_until_ready`] releases this guard while waiting for a pending mutation or read leases, then returns a
/// guard proven to protect the `Ready` state. [`ReadyReferenceGuard::take`] returns the value and the only typestate
/// that can resolve the resulting `Taken` state. Dropping that [`TakenReferenceGuard`] poisons the reference because
/// the previous value is no longer available and may already have been consumed by backend work.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct ReadyOrPendingReferenceGuard<'g, V: Value> {
    /// Underlying [`ReferenceGuard`].
    guard: ReferenceGuard<'g, V>,
}

impl<'g, V: Value> ReadyOrPendingReferenceGuard<'g, V> {
    /// Observes the underlying [`Reference`]'s value, generation, and pending dependency without waiting for backend
    /// work. In `Ready` or `Pending` state, this function converts the stored value to this handle's type identities
    /// and returns all three fields as one coherent [`ReferenceObservation`]. A `Pending` value is returned immediately
    /// together with the cumulative dependency that backend work must wait for before using it. After preparing work
    /// and reacquiring the guard, use [`ReferenceObservation::is_current`] to determine whether the observed value is
    /// still the one stored by this reference.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn observe(&self) -> Result<ReferenceObservation<V>, ReferenceError> {
        let (value, generation, dependency) = match &*self.guard.state {
            ReferenceState::Ready { value, generation, .. } => (value, *generation, None),
            ReferenceState::Pending { value, generation, completion, .. } => {
                (value, *generation, Some(completion.clone()))
            }
            ReferenceState::Frozen | ReferenceState::Poisoned(_) | ReferenceState::Taken { .. } => {
                unreachable!("`ReadyOrPendingReferenceGuard` protects only `Ready` or `Pending` state")
            }
        };
        let snapshot = value
            .rename_type_identities(&self.guard.reference.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        Ok(ReferenceObservation {
            holder: Arc::downgrade(&self.guard.reference.handle.holder),
            generation,
            snapshot,
            dependency,
        })
    }

    /// Preserves the current value until a submitted read-only execution completes. The caller must retain this guard
    /// after confirming that its [`ReferenceObservation`] is current and through backend submission, then call this
    /// function before releasing the guard. `completion` must cover the submitted read and, when the snapshot came from
    /// `Pending` state, that snapshot's prior dependency. This function returns immediately and does not keep the mutex
    /// locked. Instead, later attempts to take, replace, or freeze the value wait until `completion` resolves.
    /// Completed read records are removed first so a reference used only for reads does not retain backend
    /// resources indefinitely.
    ///
    /// # Parameters
    ///
    ///   - `completion`: [`ReferenceCompletion`] for the submitted read-only execution and its prior dependency.
    pub fn preserve_value_until(&mut self, completion: ReferenceCompletion) {
        match &mut *self.guard.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => {
                // Preserving the value may be the only lifecycle update made for a read-only reference.
                // Prune completed executions here so they cannot accumulate when no mutation path runs.
                read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
                read_leases.push(completion);
            }
            ReferenceState::Frozen | ReferenceState::Poisoned(_) | ReferenceState::Taken { .. } => {
                unreachable!("`ReadyOrPendingReferenceGuard` protects only `Ready` or `Pending` state")
            }
        }
    }

    /// Consumes this [`ReadyOrPendingReferenceGuard`] and prepares its value for asynchronous replacement. Completed
    /// read leases are removed first. If submitted readers may still be using the value, the guard is released and
    /// their completion tokens are returned in [`ReferenceReplacementPreparation::Waiting`]. Otherwise,
    /// the next [`ReferenceGeneration`] is computed and returned with the still-locked guard in
    /// [`ReferenceReplacementPreparation::Prepared`]. Apart from pruning completed read leases, the stored value,
    /// lifecycle variant, and generation do not change until [`PreparedReferenceReplacement::begin`] records a
    /// successful backend submission.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::GenerationExhausted`] when the current generation has no successor.
    pub fn prepare_replacement(mut self) -> Result<ReferenceReplacementPreparation<'g, V>, ReferenceError> {
        // Both states protected by this typestate store every submitted execution that may still be reading the current
        // value. Other lifecycle states cannot be reached while this guard exists.
        let read_leases = match &mut *self.guard.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => read_leases,
            ReferenceState::Frozen | ReferenceState::Poisoned(_) | ReferenceState::Taken { .. } => {
                unreachable!("`ReadyOrPendingReferenceGuard` protects only `Ready` or `Pending` state")
            }
        };

        // A terminal lease can no longer access the value, regardless of whether it succeeded or failed. Remove those
        // leases so they do not retain backend resources or unnecessarily delay replacement.
        read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));

        // Replacing the value while a submitted reader may still access it would invalidate that reader's snapshot.
        // Return cloned completion tokens for the caller to await; consuming `self` releases the mutex while it waits,
        // while the originals remain recorded until a later attempt observes that they are terminal and prunes them.
        if !read_leases.is_empty() {
            return Ok(ReferenceReplacementPreparation::Waiting(read_leases.clone()));
        }

        // Compute the generation that the replacement will receive before backend submission, but leave the lifecycle
        // variant and current generation unchanged until `PreparedReferenceReplacement::begin` records successful
        // submission.
        let generation = match &*self.guard.state {
            ReferenceState::Ready { generation, .. } | ReferenceState::Pending { generation, .. } => {
                generation.next().ok_or(ReferenceError::GenerationExhausted)?
            }
            ReferenceState::Frozen | ReferenceState::Poisoned(_) | ReferenceState::Taken { .. } => {
                unreachable!("`ReadyOrPendingReferenceGuard` protects only `Ready` or `Pending` state")
            }
        };

        // Retain the lock across backend submission so no other access can invalidate the checked prerequisites
        // or the generation selected above before the caller begins the replacement transaction.
        Ok(ReferenceReplacementPreparation::Prepared(PreparedReferenceReplacement { guard: self, generation }))
    }

    /// Waits until this reference is `Ready` with no active read leases and returns the corresponding typestate guard.
    /// This function releases the current mutex guard before awaiting pending mutation completion or read leases. The
    /// returned guard therefore protects the newest value at the time the wait completes, which may differ from an
    /// earlier [`ReferenceObservation`].
    ///
    /// # Errors
    ///
    /// Returns the applicable terminal lifecycle error, unexpected mutex poisoning, or a failed pending completion.
    pub fn wait_until_ready(self) -> Result<ReadyReferenceGuard<'g, V>, ReferenceError> {
        let reference = self.guard.reference;

        // Release the current state mutex before `lock_ready_state` reacquires it to reconcile pending work and read
        // leases. The borrowed reference remains valid for the guard lifetime after this guard is dropped.
        drop(self);

        let state = reference.lock_ready_state(true)?;
        Ok(ReadyReferenceGuard { guard: ReferenceGuard { reference, state } })
    }
}

/// Provides exclusive access to a [`Reference`] that is in the `Ready` state and thus has no active read leases.
/// [`ReadyOrPendingReferenceGuard::wait_until_ready`] creates this guard after awaiting all work that could prevent
/// safe synchronous extraction. Its only state-changing function is [`Self::take`], which removes the value and returns
/// the owning [`TakenReferenceGuard`] that must replace or poison it.
pub struct ReadyReferenceGuard<'g, V: Value> {
    /// Underlying [`ReferenceGuard`].
    guard: ReferenceGuard<'g, V>,
}

impl<'g, V: Value> ReadyReferenceGuard<'g, V> {
    /// Removes the ready value of the underlying [`Reference`] and returns it together with the only guard that can
    /// resolve the resulting `Taken` state (i.e., a [`TakenReferenceGuard`]). The value is converted to this handle's
    /// type identities before the state changes. The next generation is also computed first, so either failure leaves
    /// the reference in the `Ready` state. After this function succeeds, the returned [`TakenReferenceGuard`] must be
    /// replaced or poisoned. Dropping the returned [`TakenReferenceGuard`] poisons the reference automatically.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ValueReconstruction`] if identity conversion fails or
    /// [`ReferenceError::GenerationExhausted`] when the current generation has no successor.
    pub fn take(mut self) -> Result<(V, TakenReferenceGuard<'g, V>), ReferenceError> {
        let (value, generation) = match &*self.guard.state {
            ReferenceState::Ready { value, generation, .. } => {
                let value = value
                    .rename_type_identities(&self.guard.reference.handle.storage_to_handle)
                    .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
                let generation = generation.next().ok_or(ReferenceError::GenerationExhausted)?;
                (value, generation)
            }
            ReferenceState::Pending { .. }
            | ReferenceState::Taken { .. }
            | ReferenceState::Poisoned(_)
            | ReferenceState::Frozen => unreachable!("`ReadyReferenceGuard` protects only ready reference state"),
        };
        *self.guard.state = ReferenceState::Taken { generation };
        Ok((value, TakenReferenceGuard { guard: Some(self.guard) }))
    }
}

/// Exclusive ownership of a [`Reference`] whose value has been removed and must be replaced or poisoned.
/// [`ReadyReferenceGuard::take`] creates this guard for synchronous extraction. Asynchronous replacement transactions
/// own the same typestate internally after submission. [`Self::replace`] restores a ready value, while [`Self::poison`]
/// records why no replacement can be provided. Dropping the guard performs the same terminal transition with a generic
/// abandonment reason so a missing replacement can never become observable as usable state.
#[must_use = "a taken reference must be replaced or poisoned"]
pub struct TakenReferenceGuard<'g, V: Value> {
    /// Underlying [`ReferenceGuard`] or [`None`] when an owning transition consumes this guard.
    guard: Option<ReferenceGuard<'g, V>>,
}

impl<'g, V: Value> TakenReferenceGuard<'g, V> {
    /// Stores a synchronous replacement and changes the reference state from `Taken` to `Ready`. This function consumes
    /// this [`TakenReferenceGuard`], making a second replacement or poisoning transition impossible. If the replacement
    /// has the wrong type or cannot be converted to shared storage, the reference is poisoned with that validation
    /// error before the error is returned.
    ///
    /// # Parameters
    ///
    ///   - `replacement`: New value expressed using the originating reference handle's type identities.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn replace(mut self, replacement: V) -> Result<ReadyReferenceGuard<'g, V>, ReferenceError> {
        let guard = self.guard.as_mut().unwrap();
        let replacement = match guard.reference.prepare_replacement(replacement) {
            Ok(replacement) => replacement,
            Err(error) => {
                *guard.state = ReferenceState::Poisoned(error.to_string().into());
                return Err(error);
            }
        };
        let generation = match *guard.state {
            ReferenceState::Taken { generation } => generation,
            ReferenceState::Ready { .. }
            | ReferenceState::Pending { .. }
            | ReferenceState::Poisoned(_)
            | ReferenceState::Frozen => unreachable!("`TakenReferenceGuard` protects only taken reference state"),
        };
        *guard.state = ReferenceState::Ready { value: replacement, generation, read_leases: Vec::new() };
        Ok(ReadyReferenceGuard { guard: self.guard.take().unwrap() })
    }

    /// Permanently poisons the taken [`Reference`] because no valid replacement can be provided.
    ///
    /// # Parameters
    ///
    ///   - `reason`: Failure description reported by later attempts to access the reference.
    #[inline]
    pub fn poison<R: Into<Arc<str>>>(mut self, reason: R) {
        let guard = self.guard.as_mut().unwrap();
        debug_assert!(matches!(*guard.state, ReferenceState::Taken { .. }));
        *guard.state = ReferenceState::Poisoned(reason.into());
    }
}

impl<V: Value> Drop for TakenReferenceGuard<'_, V> {
    #[inline]
    fn drop(&mut self) {
        let Some(guard) = &mut self.guard else {
            return;
        };
        if matches!(*guard.state, ReferenceState::Taken { .. }) {
            // No value remains in the `Taken` state, and the previous value may already have been consumed by
            // irreversible backend work. Make an abandoned replacement obligation a permanent, explicit failure.
            *guard.state = ReferenceState::Poisoned("stateful transaction ended without restoring state".into());
        }
    }
}

/// Storage owned by each reference guard typestate (i.e., [`ReadyOrPendingReferenceGuard`],
/// [`ReadyReferenceGuard`], or [`TakenReferenceGuard`]).
struct ReferenceGuard<'g, V: Value> {
    /// Handle that supplies this guard's alias-local type conversions.
    reference: &'g Reference<V>,

    /// Locked [`ReferenceState`] of the shared [`Reference`] allocation.
    state: MutexGuard<'g, ReferenceState<V>>,
}

/// Outcome of preparing an observable [`Reference`] for asynchronous replacement. Preparation consumes the original
/// [`ReadyOrPendingReferenceGuard`]. It either returns a locked [`PreparedReferenceReplacement`] or releases the lock
/// and returns the read completions that must be awaited before restarting the complete observation-and-preparation
/// attempt.
pub enum ReferenceReplacementPreparation<'g, V: Value> {
    /// Every prerequisite is satisfied and the guard may remain locked through backend submission.
    Prepared(PreparedReferenceReplacement<'g, V>),

    /// Submitted reads may still be using the observed value. The guard has been released, and the caller should wait
    /// for these completions before reacquiring the reference and retrying the complete observation attempt.
    Waiting(Vec<ReferenceCompletion>),
}

/// Locked [`Reference`] that is ready to cross an asynchronous replacement's submission boundary. This type proves that
/// no active read lease can still access the current value and that a next [`ReferenceGeneration`] exists. Dropping it
/// simply releases the unchanged observable state. After backend submission succeeds, call [`Self::begin`] to make the
/// value unavailable and create the owning [`ReferenceReplacementTransaction`].
#[must_use = "a prepared reference replacement must be submitted or released"]
pub struct PreparedReferenceReplacement<'g, V: Value> {
    /// [`ReadyOrPendingReferenceGuard`] retained across the backend submission boundary.
    guard: ReadyOrPendingReferenceGuard<'g, V>,

    /// [`ReferenceGeneration`] assigned if submission succeeds.
    generation: ReferenceGeneration,
}

impl<'g, V: Value> PreparedReferenceReplacement<'g, V> {
    /// Records successful backend submission, changes the underlying [`Reference`] state to `Taken`, and returns
    /// the transaction that must provide its replacement. `completion` must include the submitted mutation and the
    /// dependency captured by the observation used to prepare that mutation. After this function returns, dropping the
    /// transaction without committing or explicitly poisoning it poisons the underlying reference automatically.
    ///
    /// # Parameters
    ///
    ///   - `completion`: Cumulative [`ReferenceCompletion`] of the submitted mutation and all predecessor dependencies.
    #[inline]
    pub fn begin(mut self, completion: ReferenceCompletion) -> ReferenceReplacementTransaction<'g, V> {
        *self.guard.guard.state = ReferenceState::Taken { generation: self.generation };
        ReferenceReplacementTransaction { taken: TakenReferenceGuard { guard: Some(self.guard.guard) }, completion }
    }
}

/// Represents a coherent observation of a [`Reference`] value and the work that must precede its use.
/// [`ReadyOrPendingReferenceGuard::observe`] captures the handle-local value, its [`ReferenceGeneration`], and its
/// optional [`ReferenceCompletion`] under one lock. A backend can prepare work from [`Self::snapshot`], reacquire any
/// handle for the same reference allocation, and use [`Self::is_current`] to verify that the observation is still
/// valid. When [`Self::dependency`] is present, submitted work must wait for it before accessing the snapshot.
pub struct ReferenceObservation<V: Value> {
    /// [`Reference`] allocation from which this observation was created. Keeping its weak control block alive prevents
    /// a later allocation from reusing the same address while this observation exists.
    holder: Weak<ReferenceHolder<V>>,

    /// [`ReferenceGeneration`] of the observed value.
    generation: ReferenceGeneration,

    /// Observed [`Value`] using the originating handle's type identities.
    snapshot: V,

    /// Cumulative [`ReferenceCompletion`] that must complete before the snapshot may be used.
    dependency: Option<ReferenceCompletion>,
}

impl<V: Value> ReferenceObservation<V> {
    /// Returns the [`ReferenceGeneration`] of the observed value.
    #[inline]
    pub fn generation(&self) -> ReferenceGeneration {
        self.generation
    }

    /// Returns the observed [`Value`] using the originating [`Reference`] handle's type identities.
    #[inline]
    pub fn snapshot(&self) -> &V {
        &self.snapshot
    }

    /// Returns the cumulative work that must complete before the observed value may be used. [`None`] means that the
    /// value was already `Ready` when observed. The returned completion may already have resolved because pending
    /// reference state is reconciled lazily.
    #[inline]
    pub fn dependency(&self) -> Option<&ReferenceCompletion> {
        self.dependency.as_ref()
    }

    /// Returns `true` if this [`ReferenceObservation`] still describes the value protected by `guard` and `false`
    /// otherwise. An observation is current for any handle backed by the same reference allocation while that
    /// allocation remains at the same generation. This includes identity-renamed aliases because the observation owns
    /// its already-converted snapshot. A `Pending` value may become `Ready` without changing its generation; the
    /// observation remains valid in that case and its captured dependency is simply already complete. This function
    /// returns `false` for another allocation or generation instead of treating an optimistic retry as an error.
    ///
    /// # Parameters
    ///
    ///   - `guard`: Guard protecting the reference state against which this observation is checked.
    #[inline]
    pub fn is_current(&self, guard: &ReadyOrPendingReferenceGuard<'_, V>) -> bool {
        match &*guard.guard.state {
            ReferenceState::Ready { generation, .. } | ReferenceState::Pending { generation, .. } => {
                std::ptr::eq(self.holder.as_ptr(), Arc::as_ptr(&guard.guard.reference.handle.holder))
                    && *generation == self.generation
            }
            ReferenceState::Frozen | ReferenceState::Poisoned(_) | ReferenceState::Taken { .. } => {
                unreachable!("`ReadyOrPendingReferenceGuard` protects only `Ready` or `Pending` state")
            }
        }
    }
}

/// Cloneable backend-neutral token for work that reads or replaces a [`Reference`] value. A token has one immutable
/// terminal result: success or a backend-owned failure reason. It can be polled with [`Self::is_ready`], waited on with
/// [`Self::r#await`], and combined in dependency order with [`Self::joined`].
#[derive(Clone)]
pub struct ReferenceCompletion {
    /// Private representation kept behind this public wrapper so callers cannot construct or match its variants and
    /// bypass [`Self::joined`]'s normalization.
    storage: ReferenceCompletionStorage,
}

impl ReferenceCompletion {
    /// Wraps a [`ReferenceCompletionBackend`] into a new [`ReferenceCompletion`].
    #[inline]
    pub fn new<B: ReferenceCompletionBackend>(backend: B) -> Self {
        Self { storage: ReferenceCompletionStorage::Backend(Arc::new(backend)) }
    }

    /// Creates a new [`ReferenceCompletion`] token with an already-known terminal `result`.
    #[inline]
    pub fn ready(result: Result<(), Arc<str>>) -> Self {
        Self::new(ReadyReferenceCompletion(result))
    }

    /// Returns a new [`ReferenceCompletion`] that completes after every input token and reports the first failure in
    /// input order. Nested joins are flattened. Inputs that have already succeeded are discarded because their terminal
    /// result cannot change. Pending and failed inputs are retained in order. The join remains pending until every
    /// retained input is terminal, and waiting on it waits for every input even after observing a failure.
    pub fn joined<C: IntoIterator<Item = Self>>(completions: C) -> Self {
        let mut flattened = Vec::new();
        for completion in completions {
            // Keep primitive order while removing dependencies whose immutable terminal result is already success.
            // Failed completions must remain because the earliest one determines the joined result.
            match &completion.storage {
                ReferenceCompletionStorage::Backend(_) => {
                    if !matches!(completion.is_ready(), Ok(true)) {
                        flattened.push(completion);
                    }
                }
                ReferenceCompletionStorage::Joined(joined) => flattened.extend(
                    joined.completions.iter().filter(|completion| !matches!(completion.is_ready(), Ok(true))).cloned(),
                ),
            }
        }
        let completions = flattened;
        match completions.len() {
            0 => Self::ready(Ok(())),
            1 => completions.into_iter().next().unwrap(),
            _ => Self {
                storage: ReferenceCompletionStorage::Joined(Arc::new(JoinedReferenceCompletion { completions })),
            },
        }
    }

    /// Checks whether all underlying work has completed without blocking. Returns `Ok(false)` while any work is
    /// pending, `Ok(true)` after success, or the backend's terminal failure reason. Failures use `Arc<str>` so cloned
    /// completions and joined dependencies can cheaply retain and propagate the same immutable diagnostic across
    /// threads without tying it to a backend object's lifetime.
    #[inline]
    pub fn is_ready(&self) -> Result<bool, Arc<str>> {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.is_ready(),
            ReferenceCompletionStorage::Joined(joined) => joined.is_ready(),
        }
    }

    /// Blocks until all underlying work completes and returns its terminal result. A joined completion waits for every
    /// dependency even after observing a failure and returns the first failure in input order. The shared `Arc<str>`
    /// failure reason can be retained by poisoned reference state and propagated through backend execution results
    /// without copying its contents.
    #[inline]
    pub fn r#await(&self) -> Result<(), Arc<str>> {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.r#await(),
            ReferenceCompletionStorage::Joined(joined) => joined.r#await(),
        }
    }
}

impl Debug for ReferenceCompletion {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ReferenceCompletion").field("is_ready", &self.is_ready()).finish()
    }
}

/// Private representation of one primitive [`ReferenceCompletion`] or a flattened join. This is separate from the
/// public [`ReferenceCompletion`] because keeping the enum private prevents callers from constructing nested joins
/// directly, ensuring every joined completion passes through [`ReferenceCompletion::joined`], which flattens nested
/// joins, removes completed successes, and preserves the original input order.
#[derive(Clone)]
enum ReferenceCompletionStorage {
    /// Wraps a [`ReferenceCompletionBackend`].
    Backend(Arc<dyn ReferenceCompletionBackend>),

    /// Ordered list of [`ReferenceCompletion`]s flattened from one or more joins.
    Joined(Arc<JoinedReferenceCompletion>),
}

/// Backend implementation stored behind a type-erased [`ReferenceCompletion`]. [`Self::is_ready`] may return
/// `Ok(false)` before completion. Once either function observes success or failure, that terminal result must never
/// change, and both functions must agree on it. [`ReferenceCompletion::joined`] relies on this contract when it discards
/// completed successes.
pub trait ReferenceCompletionBackend: 'static + Send + Sync {
    /// Blocks until the represented work completes and returns its terminal result.
    fn r#await(&self) -> Result<(), Arc<str>>;

    /// Checks without blocking, returning `Ok(false)` while pending, `Ok(true)` after success, or the terminal failure.
    fn is_ready(&self) -> Result<bool, Arc<str>>;
}

/// [`ReferenceCompletion`] backend with an immediately available terminal result.
struct ReadyReferenceCompletion(Result<(), Arc<str>>);

impl ReferenceCompletionBackend for ReadyReferenceCompletion {
    #[inline]
    fn r#await(&self) -> Result<(), Arc<str>> {
        self.0.clone()
    }

    #[inline]
    fn is_ready(&self) -> Result<bool, Arc<str>> {
        self.0.clone().map(|_| true)
    }
}

/// Flattened and ordered [`ReferenceCompletion`] join created by [`ReferenceCompletion::joined`].
struct JoinedReferenceCompletion {
    /// Flat, input-ordered primitive [`ReferenceCompletion`]s.
    completions: Vec<ReferenceCompletion>,
}

impl ReferenceCompletionBackend for JoinedReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        // A joined token represents completion of every member, so a failure does not permit an early return.
        // Retain the first failure while continuing to wait for the remaining dependencies.
        let mut result = Ok(());
        for completion in &self.completions {
            if let Err(error) = completion.r#await()
                && result.is_ok()
            {
                result = Err(error);
            }
        }
        result
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        // A known failure is not yet the join's terminal result while a later member remains pending.
        // Once every member is terminal, report the first failure in the original order.
        let mut result = Ok(true);
        for completion in &self.completions {
            match completion.is_ready() {
                Ok(false) => return Ok(false),
                Ok(true) => {}
                Err(error) if result.is_ok() => result = Err(error),
                Err(_) => {}
            }
        }
        result
    }
}

/// Owning typestate for a submitted asynchronous [`Reference`] replacement whose value has not yet been reconstructed.
/// [`PreparedReferenceReplacement::begin`] creates this transaction while changing the reference to `Taken`. The
/// transaction owns that reference's guard and cumulative completion, so it cannot be paired with another allocation or
/// generation. It must be validated and committed or explicitly poisoned; dropping it poisons the reference through
/// its owned [`TakenReferenceGuard`].
#[must_use = "a submitted reference replacement must be committed or its reference must be poisoned"]
pub struct ReferenceReplacementTransaction<'g, V: Value> {
    /// Exclusive ownership of the reference's `Taken` state.
    taken: TakenReferenceGuard<'g, V>,

    /// Cumulative [`ReferenceCompletion`] of the submitted mutation and its predecessor dependencies.
    completion: ReferenceCompletion,
}

impl<'g, V: Value> ReferenceReplacementTransaction<'g, V> {
    /// Validates and converts the reconstructed replacement without changing the reference's `Taken` state.
    ///
    /// This function consumes the submitted transaction. Success returns an owning
    /// [`ValidatedPendingReplacementTransaction`] that can be committed without failure. A validation failure poisons
    /// this reference with the validation error before returning it. A multi-reference backend should retain every
    /// validated transaction until all replacements validate, then commit them while retaining the returned guards.
    ///
    /// # Parameters
    ///
    ///   - `replacement`: New value expressed using the submitted reference handle's type identities.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn validate(self, replacement: V) -> Result<ValidatedPendingReplacementTransaction<'g, V>, ReferenceError> {
        let replacement = match self.taken.guard.as_ref().unwrap().reference.prepare_replacement(replacement) {
            Ok(replacement) => replacement,
            Err(error) => {
                self.taken.poison(error.to_string());
                return Err(error);
            }
        };
        Ok(ValidatedPendingReplacementTransaction {
            taken: self.taken,
            value: replacement,
            completion: self.completion,
        })
    }

    /// Permanently poisons the submitted replacement because no valid value can be committed.
    ///
    /// # Parameters
    ///
    ///   - `reason`: Failure description reported by later attempts to access the reference.
    #[inline]
    pub fn poison<R: Into<Arc<str>>>(self, reason: R) {
        self.taken.poison(reason);
    }
}

/// A replacement prepared to change its [`Reference`] from `Taken` to `Pending` without failure.
/// [`ReferenceReplacementTransaction::validate`] creates this transaction after validating and converting the
/// replacement while retaining exclusive ownership of its `Taken` reference guard. A multi-reference backend can keep
/// one validated transaction per reference, then commit them only after every replacement validates successfully.
#[must_use = "a validated reference replacement must be committed or its reference must be poisoned"]
pub struct ValidatedPendingReplacementTransaction<'g, V: Value> {
    /// Exclusive ownership of the reference's validated `Taken` state.
    taken: TakenReferenceGuard<'g, V>,

    /// Replacement [`Value`] using the allocation's canonical type identities.
    value: V,

    /// Cumulative [`ReferenceCompletion`] of the submitted mutation and its predecessor dependencies.
    completion: ReferenceCompletion,
}

impl<'g, V: Value> ValidatedPendingReplacementTransaction<'g, V> {
    /// Stores the validated replacement and changes its [`Reference`] state from `Taken` to `Pending`. This function is
    /// infallible because validation established every precondition while retaining the exact taken guard. After the
    /// stored completion resolves, ordinary reference access reconciles `Pending` to `Ready` on success or `Poisoned`
    /// on backend failure.
    #[inline]
    pub fn commit(self) -> ReadyOrPendingReferenceGuard<'g, V> {
        let Self { mut taken, value, completion } = self;

        // Move the shared guard out before `taken` is dropped so its drop implementation does not poison
        // the reference after this successful replacement.
        let mut guard = taken.guard.take().unwrap();

        let generation = match *guard.state {
            ReferenceState::Taken { generation } => generation,
            ReferenceState::Ready { .. }
            | ReferenceState::Pending { .. }
            | ReferenceState::Poisoned(_)
            | ReferenceState::Frozen => unreachable!("`TakenReferenceGuard` protects only taken reference state"),
        };

        *guard.state = ReferenceState::Pending { value, generation, completion, read_leases: Vec::new() };
        ReadyOrPendingReferenceGuard { guard }
    }

    /// Permanently poisons the validated replacement instead of committing it.
    ///
    /// # Parameters
    ///
    ///   - `reason`: Failure description reported by later attempts to access the reference.
    #[inline]
    pub fn poison<R: Into<Arc<str>>>(self, reason: R) {
        self.taken.poison(reason);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc::{TryRecvError, channel};
    use std::sync::{Arc, Condvar, Mutex, Weak};
    use std::thread;
    use std::time::Duration;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrValue, ArrayReference, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };
    use crate::captures::CaptureReference;
    use crate::operations::Add;

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(10);

    /// Simulates the successful-submission and replacement-commit phases used by asynchronous backend tests.
    fn commit_pending_replacement<V: Value>(
        guard: ReadyOrPendingReferenceGuard<V>,
        completion: ReferenceCompletion,
        replacement: V,
    ) -> Result<(ReadyOrPendingReferenceGuard<V>, ReferenceGeneration), ReferenceError> {
        let prepared = match guard.prepare_replacement()? {
            ReferenceReplacementPreparation::Prepared(prepared) => prepared,
            ReferenceReplacementPreparation::Waiting(_) => {
                panic!("test reference unexpectedly has a preserved reader")
            }
        };
        let guard = prepared.begin(completion).validate(replacement)?.commit();
        let generation = guard.observe()?.generation();
        Ok((guard, generation))
    }

    /// Completion backend with a bounded waiter handshake so a broken asynchronous protocol fails instead of hanging.
    #[derive(Clone)]
    struct ControlledCompletion {
        state: Arc<(Mutex<ControlledCompletionState>, Condvar)>,
    }

    struct ControlledCompletionState {
        awaiting: bool,
        result: Option<Result<(), Arc<str>>>,
    }

    impl ControlledCompletion {
        fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledCompletionState { awaiting: false, result: None }),
                    Condvar::new(),
                )),
            }
        }

        fn wait_until_awaited(&self) {
            let (state, changed) = &*self.state;
            let state = state.lock().unwrap();
            let (state, _) = changed.wait_timeout_while(state, TEST_TIMEOUT, |state| !state.awaiting).unwrap();
            assert!(state.awaiting, "completion was not awaited within {TEST_TIMEOUT:?}");
        }

        fn complete(&self, result: Result<(), Arc<str>>) {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            assert!(state.result.is_none());
            state.result = Some(result);
            ready.notify_all();
        }
    }

    impl ReferenceCompletionBackend for ControlledCompletion {
        fn r#await(&self) -> Result<(), Arc<str>> {
            let (state, changed) = &*self.state;
            let mut state = state.lock().unwrap();
            state.awaiting = true;
            changed.notify_all();
            let (state, _) = changed.wait_timeout_while(state, TEST_TIMEOUT, |state| state.result.is_none()).unwrap();
            assert!(state.result.is_some(), "completion was not resolved within {TEST_TIMEOUT:?}");
            state.result.clone().unwrap()
        }

        fn is_ready(&self) -> Result<bool, Arc<str>> {
            let state = self.state.0.lock().unwrap();
            state.result.clone().map_or(Ok(false), |result| result.map(|_| true))
        }
    }

    #[test]
    fn test_reference_new_rejects_an_immediate_reference_referent() {
        let nested = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        assert!(matches!(
            Reference::new(nested),
            Err(ReferenceError::NestedReferent { referent_type }) if referent_type == "ref<f32[]>"
        ));
    }

    #[test]
    fn test_reference_clone_preserves_allocation_identity_equality_and_hashing() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = Reference::new(initial.clone()).unwrap();
        let alias = reference.clone();
        let distinct = Reference::new(initial).unwrap();

        assert_eq!(&reference, &reference);
        assert_eq!(reference, alias);
        assert_ne!(reference, distinct);
        assert_eq!(reference.id(), alias.id());
        assert_ne!(reference.id(), distinct.id());
        assert_eq!(reference.read(), Ok(Array::vector(vec![1.0_f32, 2.0])));
        assert_eq!(reference.r#type(), alias.r#type());
        assert_eq!(reference.r#type(), distinct.r#type());
        assert!(reference.uses_storage_type_identities());
        assert!(alias.uses_storage_type_identities());
        assert!(Arc::ptr_eq(&alias.handle, &reference.handle));

        let mut reference_hasher = DefaultHasher::new();
        reference.hash(&mut reference_hasher);
        let mut alias_hasher = DefaultHasher::new();
        alias.hash(&mut alias_hasher);
        assert_eq!(reference_hasher.finish(), alias_hasher.finish());

        let references = HashMap::from([(reference.clone(), "allocation")]);
        assert_eq!(references.get(&alias), Some(&"allocation"));
        assert_eq!(references.get(&distinct), None);
    }

    #[test]
    fn test_reference_display_and_debug_render_type_and_allocation_identity() {
        let reference = Reference::new(Array::vector(vec![1.0_f32, 2.0])).unwrap();
        let distinct = Reference::new(Array::vector(vec![3.0_f32, 4.0])).unwrap();

        // Display is deterministic and type-based, while Debug also exposes process-local allocation identity.
        assert_eq!(reference.to_string(), "ref<f32[2]>");
        assert_eq!(reference.to_string(), distinct.to_string());
        assert_eq!(
            format!("{reference:?}"),
            format!("Reference {{ id: {:?}, type: {:?} }}", reference.id(), reference.r#type()),
        );
    }

    #[test]
    fn test_reference_has_pointer_sized_representation() {
        assert_eq!(size_of::<Reference<Array>>(), size_of::<usize>());
    }

    #[test]
    fn test_reference_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_reference_read_write_swap_and_update_preserve_value_ownership() {
        let initializer = Array::vector(vec![1.0_f32, 2.0]);
        let first = Reference::new(initializer.clone()).unwrap();
        let read_snapshot = first.read().unwrap();
        let replacement = Array::vector(vec![3.0_f32, 4.0]);
        let retained_replacement = replacement.clone();
        assert_eq!(first.write(replacement), Ok(()));
        assert_eq!(first.swap(Array::vector(vec![7.0_f32, 8.0])), Ok(Array::vector(vec![3.0_f32, 4.0])));
        assert_eq!(
            first.update(|current| {
                current.add(&Array::vector(vec![10.0_f32, 20.0])).map(|updated| (updated, "updated"))
            }),
            Ok("updated"),
        );
        assert_eq!(initializer, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(read_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(retained_replacement, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(first.read(), Ok(Array::vector(vec![17.0_f32, 28.0])));
    }

    #[test]
    fn test_reference_mutations_do_not_affect_distinct_allocations() {
        let initializer = Array::vector(vec![1.0_f32, 2.0]);
        let first = Reference::new(initializer.clone()).unwrap();
        let second = Reference::new(initializer.clone()).unwrap();
        assert_eq!(first.write(Array::vector(vec![3.0_f32, 4.0])), Ok(()));
        assert_eq!(second.read(), Ok(initializer.clone()));
        assert_eq!(second.swap(Array::vector(vec![5.0_f32, 6.0])), Ok(initializer));
        assert_eq!(first.read(), Ok(Array::vector(vec![3.0_f32, 4.0])));
        assert_eq!(second.read(), Ok(Array::vector(vec![5.0_f32, 6.0])));
    }

    #[test]
    fn test_reference_rejected_replacements_and_updates_preserve_state() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = Reference::new(initial.clone()).unwrap();
        let mismatch =
            ReferenceError::ReferentTypeMismatch { expected: "f32[2]".to_string(), actual: "f32[3]".to_string() };

        assert_eq!(reference.swap(Array::vector(vec![3.0_f32, 4.0, 5.0])), Err(mismatch.clone()));
        assert_eq!(reference.read(), Ok(initial.clone()));
        assert_eq!(reference.write(Array::vector(vec![3.0_f32, 4.0, 5.0])), Err(mismatch.clone()));
        assert_eq!(reference.read(), Ok(initial.clone()));

        let update_error = ProgramError::InvalidArgument { message: "test update failed".to_string() };
        assert_eq!(reference.update::<(), _>(|_| Err(update_error.clone())), Err(update_error));
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.update(|_| Ok((Array::vector(vec![3.0_f32, 4.0, 5.0]), ()))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&mismatch));
        assert_eq!(reference.read(), Ok(initial));
    }

    #[test]
    fn test_reference_write_commits_calls_from_multiple_threads() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let initial_generation = reference.lock().unwrap().observe().unwrap().generation();
        let first_reference = reference.clone();
        let second_reference = reference.clone();
        let (sender, receiver) = channel();
        let first_sender = sender.clone();
        let first = thread::spawn(move || first_sender.send(first_reference.write(Array::scalar(2.0_f32))).unwrap());
        let second = thread::spawn(move || sender.send(second_reference.write(Array::scalar(3.0_f32))).unwrap());

        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| { panic!("first reference write result within {TEST_TIMEOUT:?}: {error}") }),
            Ok(()),
        );
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| { panic!("second reference write result within {TEST_TIMEOUT:?}: {error}") }),
            Ok(()),
        );
        first.join().unwrap();
        second.join().unwrap();
        assert!(
            matches!(reference.read(), Ok(value) if value == Array::scalar(2.0_f32) || value == Array::scalar(3.0_f32))
        );
        assert_eq!(reference.lock().unwrap().observe().unwrap().generation().0, initial_generation.0 + 2);
    }

    #[test]
    fn test_reference_read_waits_for_pending_completion_without_holding_the_lock() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let backend = ControlledCompletion::new();
        let guard = reference.lock().unwrap();
        let (guard, _) =
            commit_pending_replacement(guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32))
                .unwrap();
        assert!(guard.observe().unwrap().dependency().is_some());
        drop(guard);

        let (sender, receiver) = channel();
        let reader_reference = Arc::clone(&reference);
        let reader = thread::spawn(move || sender.send(reader_reference.read()).unwrap());
        backend.wait_until_awaited();
        assert!(reference.lock().unwrap().observe().unwrap().dependency().is_some());
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        backend.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference read result within {TEST_TIMEOUT:?}: {error}")),
            Ok(Array::scalar(2.0_f32)),
        );
        reader.join().unwrap();
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_write_waits_for_pending_completion_without_holding_the_lock() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let backend = ControlledCompletion::new();
        let guard = reference.lock().unwrap();
        let (guard, _) =
            commit_pending_replacement(guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32))
                .unwrap();
        drop(guard);

        let (sender, receiver) = channel();
        let writing_reference = Arc::clone(&reference);
        let writer = thread::spawn(move || sender.send(writing_reference.write(Array::scalar(3.0_f32))).unwrap());
        backend.wait_until_awaited();
        assert!(reference.lock().unwrap().observe().unwrap().dependency().is_some());
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        backend.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference write result within {TEST_TIMEOUT:?}: {error}")),
            Ok(()),
        );
        writer.join().unwrap();
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_read_reports_a_failed_pending_completion_as_execution_poisoned() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let backend = ControlledCompletion::new();
        let guard = reference.lock().unwrap();
        let (guard, _) =
            commit_pending_replacement(guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32))
                .unwrap();
        drop(guard);

        // The completion resolves before the read reaches it, so the read observes the failure through the same lazy
        // reconciliation path and reports the backend-owned reason. Poisoning is terminal for every later access.
        backend.complete(Err("device execution failed".into()));
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "device execution failed".to_string() };
        assert_eq!(reference.read(), Err(poisoned.clone()));
        assert_eq!(reference.write(Array::scalar(3.0_f32)), Err(poisoned.clone()));
        assert_eq!(reference.swap(Array::scalar(3.0_f32)), Err(poisoned.clone()));
        assert_eq!(reference.freeze(), Err(poisoned));
    }

    #[test]
    fn test_reference_read_ignores_a_stale_pending_completion() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let first_backend = ControlledCompletion::new();
        let second_backend = ControlledCompletion::new();

        let first_generation = {
            let guard = reference.lock().unwrap();
            let (guard, generation) = commit_pending_replacement(
                guard,
                ReferenceCompletion::new(first_backend.clone()),
                Array::scalar(2.0_f32),
            )
            .unwrap();
            drop(guard);
            generation
        };

        // The reader captures generation one and releases the mutex while awaiting its completion.
        let (sender, receiver) = channel();
        let reader_reference = Arc::clone(&reference);
        let reader = thread::spawn(move || sender.send(reader_reference.read()).unwrap());
        first_backend.wait_until_awaited();

        // Commit generation two while the reader is waiting. Its completion includes generation one's dependency,
        // matching the cumulative dependency contract required of chained submissions.
        let second_generation = {
            let guard = reference.lock().unwrap();
            let observation = guard.observe().unwrap();
            assert_eq!(observation.generation(), first_generation);
            let completion = ReferenceCompletion::joined([
                observation.dependency().unwrap().clone(),
                ReferenceCompletion::new(second_backend.clone()),
            ]);
            let (guard, generation) = commit_pending_replacement(guard, completion, Array::scalar(3.0_f32)).unwrap();
            drop(guard);
            generation
        };
        assert_eq!(second_generation, ReferenceGeneration(first_generation.0 + 1));

        // Resolving generation one wakes the reader, but its stale result cannot promote generation two. Waiting until
        // the reader reaches the second completion proves that it retried through the ordinary reconciliation path.
        first_backend.complete(Ok(()));
        second_backend.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        second_backend.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference read result within {TEST_TIMEOUT:?}: {error}")),
            Ok(Array::scalar(3.0_f32)),
        );
        reader.join().unwrap();

        let observation = reference.lock().unwrap().observe().unwrap();
        assert_eq!(observation.generation(), second_generation);
        assert_eq!(observation.snapshot(), &Array::scalar(3.0_f32));
        assert!(observation.dependency().is_none());
    }

    #[test]
    fn test_reference_write_waits_until_the_preserved_value_is_released() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let preservation = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(preservation.clone()));
        drop(guard);

        let (sender, receiver) = channel();
        let writing_reference = Arc::clone(&reference);
        let writer = thread::spawn(move || sender.send(writing_reference.write(Array::scalar(2.0_f32))).unwrap());
        preservation.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        preservation.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference write result within {TEST_TIMEOUT:?}: {error}")),
            Ok(()),
        );
        writer.join().unwrap();
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_swap_waits_until_the_preserved_value_is_released() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let preservation = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(preservation.clone()));
        drop(guard);

        // Preserving the current value prevents the replacement from committing until the completion resolves.
        let (sender, receiver) = channel();
        let swapping_reference = Arc::clone(&reference);
        let swapper = thread::spawn(move || sender.send(swapping_reference.swap(Array::scalar(2.0_f32))).unwrap());
        preservation.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        preservation.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference swap result within {TEST_TIMEOUT:?}: {error}")),
            Ok(Array::scalar(1.0_f32)),
        );
        swapper.join().unwrap();
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_freeze_waits_until_the_preserved_value_is_released() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let preservation = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(preservation.clone()));
        drop(guard);

        // Consuming the value must respect the same preservation barrier as replacing it.
        let (sender, receiver) = channel();
        let freezing_reference = Arc::clone(&reference);
        let freezer = thread::spawn(move || sender.send(freezing_reference.freeze()).unwrap());
        preservation.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        preservation.complete(Ok(()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("reference freeze result within {TEST_TIMEOUT:?}: {error}")),
            Ok(Array::scalar(1.0_f32)),
        );
        freezer.join().unwrap();
        assert_eq!(reference.read(), Err(ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_freeze_invalidates_every_alias_before_running_later_updates() {
        let reference = Reference::new(Array::vector(vec![1.0_f32, 2.0])).unwrap();
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));

        assert_eq!(alias.read(), Err(ReferenceError::Frozen));
        assert_eq!(alias.write(Array::vector(vec![3.0_f32, 4.0])), Err(ReferenceError::Frozen));
        assert_eq!(alias.swap(Array::vector(vec![3.0_f32, 4.0])), Err(ReferenceError::Frozen));
        assert_eq!(reference.freeze(), Err(ReferenceError::Frozen));

        let update_executed = Cell::new(false);
        let error = alias
            .update(|_| {
                update_executed.set(true);
                Ok((Array::vector(vec![3.0_f32, 4.0]), ()))
            })
            .unwrap_err();
        assert!(!update_executed.get());
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_generation_advances_only_after_committed_mutations() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let initial_generation = reference.lock().unwrap().observe().unwrap().generation();

        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.lock().unwrap().observe().unwrap().generation(), initial_generation);

        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
        let swapped_generation = reference.lock().unwrap().observe().unwrap().generation();
        assert_eq!(swapped_generation, ReferenceGeneration(initial_generation.0 + 1));

        assert_eq!(reference.write(Array::scalar(3.0_f32)), Ok(()));
        let written_generation = reference.lock().unwrap().observe().unwrap().generation();
        assert_eq!(written_generation, ReferenceGeneration(swapped_generation.0 + 1));

        let rejected = ProgramError::InvalidArgument { message: "rejected update".to_string() };
        assert_eq!(reference.update::<(), _>(|_| Err(rejected.clone())), Err(rejected));
        assert_eq!(reference.lock().unwrap().observe().unwrap().generation(), written_generation);

        assert_eq!(reference.update(|_| Ok((Array::scalar(4.0_f32), "updated"))), Ok("updated"));
        let updated_generation = reference.lock().unwrap().observe().unwrap().generation();
        assert_eq!(updated_generation, ReferenceGeneration(written_generation.0 + 1));
    }

    #[test]
    fn test_reference_mutations_preserve_state_when_the_generation_is_exhausted() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        {
            let mut state = reference.handle.holder.state.lock().unwrap();
            let ReferenceState::Ready { generation, .. } = &mut *state else {
                unreachable!("a newly allocated reference is ready")
            };
            *generation = ReferenceGeneration(u64::MAX);
        }

        assert_eq!(reference.write(Array::scalar(2.0_f32)), Err(ReferenceError::GenerationExhausted));
        assert!(matches!(reference.lock().unwrap().prepare_replacement(), Err(ReferenceError::GenerationExhausted),));
        let ready = reference.lock().unwrap().wait_until_ready().unwrap();
        assert!(matches!(ready.take(), Err(ReferenceError::GenerationExhausted)));
        let observation = reference.lock().unwrap().observe().unwrap();
        assert_eq!(observation.generation(), ReferenceGeneration(u64::MAX));
        assert_eq!(observation.snapshot(), &Array::scalar(1.0_f32));
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_reference_access_reports_unexpected_mutex_poisoning() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let allocation = Arc::clone(&reference.handle.holder);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = allocation.state.lock().unwrap();
                panic!("poison reference allocation");
            }))
            .is_err(),
        );
        assert_eq!(reference.read(), Err(ReferenceError::Poisoned));
        assert_eq!(reference.write(Array::scalar(2.0_f32)), Err(ReferenceError::Poisoned));
    }

    #[test]
    fn test_reference_rename_type_identities_composes_aliases_and_preserves_allocation_identity() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let middle = DimensionVariable::new("middle", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let middle_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(middle.clone())]));
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let reference = Reference::new(CaptureReference::new(0, source_type.clone())).unwrap();

        let no_op = reference.rename_type_identities(&TypeIdentityRenaming::new()).unwrap();
        let mut source_to_middle = TypeIdentityRenaming::new();
        source_to_middle.insert(source, middle.clone()).unwrap();
        let renamed = reference.rename_type_identities(&source_to_middle).unwrap();
        let mut middle_to_target = TypeIdentityRenaming::new();
        middle_to_target.insert(middle, target).unwrap();
        let chained = renamed.rename_type_identities(&middle_to_target).unwrap();

        // Renaming creates distinct handle-local type and identity metadata over the same allocation. Equality and
        // hashing follow allocation identity rather than alias-local metadata.
        assert!(reference.uses_storage_type_identities());
        assert!(no_op.uses_storage_type_identities());
        assert!(!renamed.uses_storage_type_identities());
        assert!(!chained.uses_storage_type_identities());
        assert!(!Arc::ptr_eq(&renamed.handle, &reference.handle));
        assert!(Arc::ptr_eq(&renamed.handle.holder, &reference.handle.holder));
        assert_eq!(renamed, reference);
        assert_eq!(renamed.r#type().as_ref(), &ReferenceType::new(middle_type.clone()));
        assert_eq!(chained.r#type().as_ref(), &ReferenceType::new(target_type.clone()));
        assert_eq!(renamed.read(), Ok(CaptureReference::new(0, middle_type.clone())));
        assert_eq!(chained.read(), Ok(CaptureReference::new(0, target_type.clone())));
        assert_eq!(HashMap::from([(reference.clone(), "allocation")]).get(&renamed), Some(&"allocation"));

        // Observation validity follows allocation identity, while its snapshot retains the originating handle's type
        // identities and therefore needs no reinterpretation through the validating alias.
        let observation = reference.lock().unwrap().observe().unwrap();
        let unrelated = Reference::new(CaptureReference::new(0, source_type.clone())).unwrap();
        assert!(observation.is_current(&reference.lock().unwrap()));
        assert!(observation.is_current(&renamed.lock().unwrap()));
        assert!(!observation.is_current(&unrelated.lock().unwrap()));

        assert_eq!(chained.write(CaptureReference::new(1, target_type.clone())), Ok(()));
        assert_eq!(reference.read(), Ok(CaptureReference::new(1, source_type.clone())));
        assert_eq!(renamed.read(), Ok(CaptureReference::new(1, middle_type.clone())));

        // Both guard replacement protocols accept values in the locking handle's identity space and convert them back
        // to the allocation's storage identities before making them observable through every alias.
        let ready = chained.lock().unwrap().wait_until_ready().unwrap();
        let (value, taken) = ready.take().unwrap();
        assert_eq!(value, CaptureReference::new(1, target_type.clone()));
        let ready = taken.replace(CaptureReference::new(2, target_type)).unwrap();
        drop(ready);
        assert_eq!(reference.read(), Ok(CaptureReference::new(2, source_type.clone())));

        let guard = renamed.lock().unwrap();
        let (guard, _) = commit_pending_replacement(
            guard,
            ReferenceCompletion::ready(Ok(())),
            CaptureReference::new(3, middle_type),
        )
        .unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(CaptureReference::new(3, source_type)));
    }

    #[test]
    fn test_reference_rename_type_identities_rejects_a_non_injective_mapping() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let second = DimensionVariable::new("second", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Dynamic(second.clone())]),
        );
        let reference = Reference::new(CaptureReference::new(0, source_type)).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        renaming.insert(second, target).unwrap();

        assert_eq!(
            reference.rename_type_identities(&renaming),
            Err(TypeError::invalid("type identities `source` and `second` are both renamed to `target`")),
        );
    }

    // Ready-or-pending guards and coherent observations.

    #[test]
    fn test_reference_observation_tracks_allocation_and_generation() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let alias = reference.clone();
        let distinct = Reference::new(Array::scalar(1.0_f32)).unwrap();

        let observation = reference.lock().unwrap().observe().unwrap();
        assert_eq!(observation.generation(), ReferenceGeneration(0));
        assert_eq!(observation.snapshot(), &Array::scalar(1.0_f32));
        assert!(observation.dependency().is_none());
        assert!(observation.is_current(&reference.lock().unwrap()));
        assert!(observation.is_current(&alias.lock().unwrap()));
        assert!(!observation.is_current(&distinct.lock().unwrap()));

        assert_eq!(reference.write(Array::scalar(2.0_f32)), Ok(()));
        assert!(!observation.is_current(&reference.lock().unwrap()));
        let current = reference.lock().unwrap().observe().unwrap();
        assert_eq!(current.generation(), ReferenceGeneration(1));
        assert_eq!(current.snapshot(), &Array::scalar(2.0_f32));
        assert!(current.dependency().is_none());
    }

    #[test]
    fn test_reference_observation_remains_current_when_pending_value_becomes_ready() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let completion = ControlledCompletion::new();
        let guard = reference.lock().unwrap();
        let (guard, generation) =
            commit_pending_replacement(guard, ReferenceCompletion::new(completion.clone()), Array::scalar(2.0_f32))
                .unwrap();
        let pending = guard.observe().unwrap();
        assert_eq!(pending.generation(), generation);
        assert_eq!(pending.snapshot(), &Array::scalar(2.0_f32));
        assert!(pending.dependency().is_some());
        drop(guard);

        completion.complete(Ok(()));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));

        // Reconciliation changes only the lifecycle variant, so the pending observation remains current while a new
        // observation no longer carries an already-completed dependency.
        let guard = reference.lock().unwrap();
        assert!(pending.is_current(&guard));
        assert!(guard.observe().unwrap().dependency().is_none());
    }

    #[test]
    fn test_ready_or_pending_reference_guard_preserve_value_until_releases_terminal_completions() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let first = ControlledCompletion::new();
        let second = ControlledCompletion::new();
        let third = ControlledCompletion::new();
        first.complete(Ok(()));
        second.complete(Err("read failed".into()));

        // Preserving the value may be the only lifecycle update made for a read-only reference, so it must also release
        // terminal completions instead of retaining their backend resources indefinitely. The strong counts make those
        // releases observable here.
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(first.clone()));
        guard.preserve_value_until(ReferenceCompletion::new(second.clone()));
        assert_eq!(Arc::strong_count(&first.state), 1);
        assert_eq!(Arc::strong_count(&second.state), 2);
        guard.preserve_value_until(ReferenceCompletion::new(third.clone()));
        assert_eq!(Arc::strong_count(&second.state), 1);
        assert_eq!(Arc::strong_count(&third.state), 2);

        // Only the pending completion remains. It blocks the next submitted mutation until completion and pruning.
        let ReferenceReplacementPreparation::Waiting(preservations) = guard.prepare_replacement().unwrap() else {
            panic!("active preservation unexpectedly allowed replacement preparation")
        };
        assert_eq!(preservations.len(), 1);
        third.complete(Ok(()));
        drop(preservations);
        let ReferenceReplacementPreparation::Prepared(prepared) =
            reference.lock().unwrap().prepare_replacement().unwrap()
        else {
            panic!("completed preservation unexpectedly blocked replacement preparation")
        };
        assert_eq!(Arc::strong_count(&third.state), 1);
        drop(prepared);
    }

    #[test]
    fn test_ready_or_pending_reference_guard_prepare_replacement_prunes_terminal_preservation_records() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::ready(Ok(())));
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("completed preservation unexpectedly blocked replacement preparation")
        };
        drop(prepared);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_ready_or_pending_reference_guard_prepare_replacement_returns_active_preservation_completions() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let preservation = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(preservation.clone()));

        // Preparing a mutation releases the mutex and returns every completion for an execution that can still read
        // the current value. The caller waits for them before restarting observation and replacement preparation.
        let ReferenceReplacementPreparation::Waiting(preservations) = guard.prepare_replacement().unwrap() else {
            panic!("active preservation unexpectedly allowed replacement preparation")
        };
        assert_eq!(preservations.len(), 1);
        preservation.complete(Ok(()));
        assert_eq!(preservations[0].r#await(), Ok(()));

        let ready = reference.lock().unwrap().wait_until_ready().unwrap();
        let (value, taken) = ready.take().unwrap();
        assert_eq!(value, Array::scalar(1.0_f32));
        let ready = taken.replace(Array::scalar(4.0_f32)).unwrap();
        drop(ready);
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
    }

    #[test]
    fn test_ready_or_pending_reference_guard_wait_until_ready_releases_the_mutex_while_waiting() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let pending = ControlledCompletion::new();
        let preservation = ControlledCompletion::new();
        let guard = reference.lock().unwrap();
        let (mut guard, generation) =
            commit_pending_replacement(guard, ReferenceCompletion::new(pending.clone()), Array::scalar(2.0_f32))
                .unwrap();
        guard.preserve_value_until(ReferenceCompletion::new(preservation.clone()));

        // While `wait_until_ready` awaits each completion, another thread must be able to acquire the same reference
        // mutex. Its first observation sees `Pending`; its second sees the reconciled value while preservation remains.
        let waiting_reference = Arc::clone(&reference);
        let (sender, receiver) = channel();
        let waiter = thread::spawn(move || {
            pending.wait_until_awaited();
            let observation = waiting_reference.lock().unwrap().observe().unwrap();
            sender.send((observation.generation(), observation.dependency().is_some())).unwrap();
            pending.complete(Ok(()));

            preservation.wait_until_awaited();
            let observation = waiting_reference.lock().unwrap().observe().unwrap();
            sender.send((observation.generation(), observation.dependency().is_some())).unwrap();
            preservation.complete(Ok(()));
        });

        let ready = guard.wait_until_ready().unwrap();
        assert_eq!(receiver.recv_timeout(TEST_TIMEOUT), Ok((generation, true)));
        assert_eq!(receiver.recv_timeout(TEST_TIMEOUT), Ok((generation, false)));
        waiter.join().unwrap();
        drop(ready);
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    // Ready and taken guards.

    #[test]
    fn test_ready_reference_guard_take_and_replace_advance_the_generation() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let ready = reference.lock().unwrap().wait_until_ready().unwrap();
        let (value, taken) = ready.take().unwrap();
        assert_eq!(value, Array::scalar(1.0_f32));

        let ready = taken.replace(Array::scalar(2.0_f32)).unwrap();
        drop(ready);
        let observation = reference.lock().unwrap().observe().unwrap();
        assert_eq!(observation.generation(), ReferenceGeneration(1));
        assert_eq!(observation.snapshot(), &Array::scalar(2.0_f32));
    }

    #[test]
    fn test_taken_reference_guard_replace_validation_failure_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let ready = reference.lock().unwrap().wait_until_ready().unwrap();
        let (_, taken) = ready.take().unwrap();

        let error = match taken.replace(Array::vector(vec![2.0_f32])) {
            Ok(_) => panic!("replacement with the wrong type unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ReferenceError::ReferentTypeMismatch { expected: "f32[]".to_string(), actual: "f32[1]".to_string() },
        );
        assert_eq!(reference.read(), Err(ReferenceError::ExecutionPoisoned { reason: error.to_string() }),);
    }

    #[test]
    fn test_taken_reference_guard_poison_affects_only_the_taken_reference() {
        let first = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let second = Reference::new(Array::scalar(2.0_f32)).unwrap();
        let first_guard = first.lock().unwrap().wait_until_ready().unwrap();
        let second_guard = second.lock().unwrap();
        let (_, taken) = first_guard.take().unwrap();
        taken.poison("test execution failed");
        drop(second_guard);
        assert_eq!(
            first.read(),
            Err(ReferenceError::ExecutionPoisoned { reason: "test execution failed".to_string() }),
        );
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_taken_reference_guard_drop_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let ready = reference.lock().unwrap().wait_until_ready().unwrap();
        let (value, taken) = ready.take().unwrap();
        assert_eq!(value, Array::scalar(1.0_f32));
        drop(taken);
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

    // Asynchronous replacement transactions.

    #[test]
    fn test_reference_replacement_transaction_commits_its_prepared_generation() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let guard = reference.lock().unwrap();
        let (guard, first) =
            commit_pending_replacement(guard, ReferenceCompletion::ready(Ok(())), Array::scalar(2.0_f32)).unwrap();
        let (guard, second) =
            commit_pending_replacement(guard, ReferenceCompletion::ready(Ok(())), Array::scalar(3.0_f32)).unwrap();
        assert_eq!(first.next(), Some(second));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_replacement_transaction_validate_failure_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let guard = reference.lock().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has a preserved reader")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));

        let error = match transaction.validate(Array::vector(vec![2.0_f32])) {
            Ok(_) => panic!("replacement with the wrong type unexpectedly validated"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ReferenceError::ReferentTypeMismatch { expected: "f32[]".to_string(), actual: "f32[1]".to_string() },
        );
        assert_eq!(reference.read(), Err(ReferenceError::ExecutionPoisoned { reason: error.to_string() }),);
    }

    #[test]
    fn test_validated_pending_replacement_transaction_drop_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let guard = reference.lock().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has a preserved reader")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));
        let validated = transaction.validate(Array::scalar(2.0_f32)).unwrap();

        drop(validated);
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_replacement_transactions_poison_every_replacement_after_validation_failure() {
        let first = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let second = Reference::new(Array::scalar(2.0_f32)).unwrap();
        let first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        let ReferenceReplacementPreparation::Prepared(first_prepared) = first_guard.prepare_replacement().unwrap()
        else {
            panic!("new reference unexpectedly has a preserved reader")
        };
        let ReferenceReplacementPreparation::Prepared(second_prepared) = second_guard.prepare_replacement().unwrap()
        else {
            panic!("new reference unexpectedly has a preserved reader")
        };
        let first_transaction = first_prepared.begin(ReferenceCompletion::ready(Ok(())));
        let second_transaction = second_prepared.begin(ReferenceCompletion::ready(Ok(())));
        let first_validated = first_transaction.validate(Array::scalar(3.0_f32)).unwrap();

        let error = match second_transaction.validate(Array::vector(vec![4.0_f32])) {
            Ok(_) => panic!("replacement with the wrong type unexpectedly validated"),
            Err(error) => error,
        };
        first_validated.poison(error.to_string());

        let expected = ReferenceError::ExecutionPoisoned { reason: error.to_string() };
        assert_eq!(first.read(), Err(expected.clone()));
        assert_eq!(second.read(), Err(expected));
    }

    #[test]
    fn test_reference_replacement_transaction_drop_during_unwind_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let guard = reference.lock().unwrap();
                let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
                    panic!("new reference unexpectedly has a preserved reader")
                };
                let _transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));
                panic!("injected backend unwind");
            }))
            .is_err(),
        );
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

    // Reference completions and joins.

    #[test]
    fn test_reference_completion_ready_reports_terminal_results_and_debug_state() {
        let succeeded = ReferenceCompletion::ready(Ok(()));
        let failed = ReferenceCompletion::ready(Err("execution failed".into()));

        assert_eq!(succeeded.is_ready(), Ok(true));
        assert_eq!(succeeded.r#await(), Ok(()));
        assert_eq!(format!("{succeeded:?}"), "ReferenceCompletion { is_ready: Ok(true) }");
        assert_eq!(failed.is_ready(), Err(Arc::<str>::from("execution failed")));
        assert_eq!(failed.r#await(), Err(Arc::<str>::from("execution failed")));
        assert_eq!(format!("{failed:?}"), "ReferenceCompletion { is_ready: Err(\"execution failed\") }",);
    }

    #[test]
    fn test_reference_completion_join_normalizes_empty_singleton_and_succeeded_inputs() {
        let pending = ControlledCompletion::new();
        let singleton = ReferenceCompletion::joined([ReferenceCompletion::new(pending.clone())]);

        assert_eq!(ReferenceCompletion::joined([]).is_ready(), Ok(true));
        assert_eq!(ReferenceCompletion::joined([ReferenceCompletion::ready(Ok(()))]).is_ready(), Ok(true));
        assert_eq!(singleton.is_ready(), Ok(false));
        pending.complete(Ok(()));
        assert_eq!(singleton.is_ready(), Ok(true));
        assert_eq!(singleton.r#await(), Ok(()));
    }

    #[test]
    fn test_reference_completion_join_reports_first_failure_in_input_order() {
        let completion = ReferenceCompletion::joined([
            ReferenceCompletion::ready(Ok(())),
            ReferenceCompletion::ready(Err("first failure".into())),
            ReferenceCompletion::ready(Err("second failure".into())),
        ]);
        assert_eq!(completion.is_ready(), Err(Arc::<str>::from("first failure")));
        assert_eq!(completion.r#await(), Err(Arc::<str>::from("first failure")));
    }

    #[test]
    fn test_reference_completion_join_flattens_nested_joins_and_waits_for_every_input() {
        let first = ControlledCompletion::new();
        let second = ControlledCompletion::new();
        let third = ControlledCompletion::new();
        let joined = ReferenceCompletion::joined([
            ReferenceCompletion::joined([
                ReferenceCompletion::new(first.clone()),
                ReferenceCompletion::new(second.clone()),
            ]),
            ReferenceCompletion::new(third.clone()),
        ]);

        // An early failure does not make the join terminal while later work remains pending.
        first.complete(Err("first failure".into()));
        assert_eq!(joined.is_ready(), Ok(false));
        let waiting = joined.clone();
        let (sender, receiver) = channel();
        let waiter = thread::spawn(move || sender.send(waiting.r#await()).unwrap());
        second.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        second.complete(Err("second failure".into()));
        third.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        third.complete(Err("third failure".into()));
        assert_eq!(
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .unwrap_or_else(|error| panic!("joined completion result within {TEST_TIMEOUT:?}: {error}")),
            Err(Arc::<str>::from("first failure")),
        );
        waiter.join().unwrap();
        assert_eq!(joined.is_ready(), Err(Arc::<str>::from("first failure")));
    }

    #[test]
    fn test_reference_completion_join_releases_successes_and_retains_pending_work_and_failures() {
        let succeeded = ControlledCompletion::new();
        let failed = ControlledCompletion::new();
        let pending = ControlledCompletion::new();
        let succeeded_state = Arc::downgrade(&succeeded.state);
        let failed_state = Arc::downgrade(&failed.state);
        let pending_state = Arc::downgrade(&pending.state);
        let nested = ReferenceCompletion::joined([
            ReferenceCompletion::new(succeeded.clone()),
            ReferenceCompletion::new(failed.clone()),
        ]);
        succeeded.complete(Ok(()));
        failed.complete(Err("retained failure".into()));
        drop(succeeded);
        drop(failed);

        let joined = ReferenceCompletion::joined([
            nested,
            ReferenceCompletion::new(pending.clone()),
            ReferenceCompletion::ready(Ok(())),
        ]);

        // These weak pointers show that joining releases completed successes while retaining pending work and failures.
        assert!(succeeded_state.upgrade().is_none());
        assert!(failed_state.upgrade().is_some());
        assert!(pending_state.upgrade().is_some());
        assert_eq!(joined.is_ready(), Ok(false));

        pending.complete(Ok(()));
        drop(pending);
        assert_eq!(joined.is_ready(), Err(Arc::<str>::from("retained failure")));
        let rejoined = ReferenceCompletion::joined([joined]);
        assert!(pending_state.upgrade().is_none());
        assert!(failed_state.upgrade().is_some());
        assert_eq!(rejoined.r#await(), Err(Arc::<str>::from("retained failure")));
        drop(rejoined);
        assert!(failed_state.upgrade().is_none());
    }

    #[test]
    fn test_reference_completion_repeated_joins_bound_succeeded_backend_retention() {
        let mut cumulative = ReferenceCompletion::ready(Ok(()));
        let mut completed_states = Vec::new();
        for _ in 0..32 {
            let current = ControlledCompletion::new();
            let current_state = Arc::downgrade(&current.state);

            // Each new join must release the previously succeeded backend instead of retaining an ever-growing chain.
            cumulative = ReferenceCompletion::joined([cumulative, ReferenceCompletion::new(current.clone())]);
            assert!(completed_states.iter().all(|state: &Weak<_>| state.upgrade().is_none()));

            current.complete(Ok(()));
            drop(current);
            assert!(current_state.upgrade().is_some());
            completed_states.push(current_state);
        }

        cumulative = ReferenceCompletion::joined([cumulative]);
        assert!(completed_states.iter().all(|state| state.upgrade().is_none()));
        assert_eq!(cumulative.r#await(), Ok(()));
    }
}
