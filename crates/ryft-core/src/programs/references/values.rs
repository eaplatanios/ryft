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
/// by [`ReferenceGuard::take`] or removed after an asynchronous mutation was submitted. The [`ReferenceGuard`] that
/// made this transition keeps the shared state locked and must replace the missing value or mark the reference
/// `Poisoned` before it is released. Committing an asynchronous replacement changes the state to `Pending`, which
/// stores the replacement together with its cumulative [`ReferenceCompletion`]. Ordinary value access waits for that
/// completion without holding the state lock, then changes the state to `Ready` on success or `Poisoned` on failure.
/// Generations ensure that completion of an older execution cannot affect a newer mutation. `Frozen` and `Poisoned` are
/// terminal states.
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

    /// Acquires this [`Reference`]'s shared lifecycle state for backend-managed access. This is the backend-facing
    /// wrapper around the private raw state lock. It returns as soon as the state mutex is acquired and deliberately
    /// does not await a `Pending` value or active read leases. The returned [`ReferenceGuard`] can observe unresolved
    /// state, record read-only executions, or replace the value while keeping every state transition under the same
    /// exclusive lock.
    ///
    /// Ordinary value access should use [`read`](Self::read), [`write`](Self::write), [`swap`](Self::swap),
    /// [`update`](Self::update), or [`freeze`](Self::freeze), which reconcile pending work before accessing the value.
    /// Backends that lock multiple references must acquire them in ascending [`ReferenceId`] order and retain that
    /// order until every submitted hidden replacement has been validated and committed. After a guard changes the state
    /// to `Taken`, no value is accessible; the guard must replace that value or mark the reference `Poisoned` before it
    /// is released.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Poisoned`] when the state mutex was poisoned without the explicit terminal `Poisoned`
    /// lifecycle state that [`ReferenceGuard`] sets defensively during an unwind.
    #[inline]
    pub fn lock(&self) -> Result<ReferenceGuard<'_, V>, ReferenceError> {
        let state = self.lock_holder_state()?;
        Ok(ReferenceGuard { reference: self, state })
    }

    /// Acquires the [`Reference`] in `Ready` state after reconciling work that prevents the requested value access.
    /// Each iteration acquires the raw mutex through [`Self::lock_holder_state`]. A `Pending` value causes this
    /// function to release the mutex, await its cumulative completion, reacquire the mutex, and apply the result only
    /// if that generation is still current. When read leases must also finish, the function prunes completed leases,
    /// releases the mutex while awaiting the remaining leases, and retries. It never awaits backend work while holding
    /// the mutex.
    ///
    /// Unlike [`lock`](Self::lock), this function is the private ordinary-access path: it hides pending-state
    /// reconciliation and returns a raw guard proven to contain `Ready`. It does not return [`ReferenceGuard`] because
    /// ordinary reads and mutations do not expose `Pending` state or use the `Taken` state and its guard-drop poisoning
    /// rule.
    ///
    /// # Parameters
    ///
    ///   - `wait_for_read_leases`: Whether the returned state must also have no active readers. Reads pass `false`
    ///     because they may share the current immutable snapshot while writes, swaps, updates, and freezing pass `true`
    ///     because they may replace or consume storage still observed by a submitted reader.
    ///
    /// # Errors
    ///
    /// Returns the lifecycle error represented by `Frozen`, `Poisoned`, or `Taken`, propagates unexpected mutex
    /// poisoning from [`Self::lock_holder_state`], and reports a failed pending completion as
    /// [`ReferenceError::ExecutionPoisoned`].
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
                    // The guard that changed the state to `Taken` retains this mutex until it either stores a
                    // replacement or poisons the reference. Competing accesses therefore block before reading this
                    // state. Keep the arm as a defensive contract check for misuse.
                    return Err(ReferenceError::TransactionInProgress);
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
    /// [`ReferenceGuard::drop`] already replaced the lifecycle state with the explicit terminal `Poisoned` state before
    /// an unwind released the mutex. Any other mutex poisoning represents an unexpected panic while the reference was
    /// in a usable state and is reported rather than cleared.
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

// TODO(eaplatanios): Review this block.
/// Backend-facing exclusive access to one [`Reference`] allocation.
///
/// Unlike ordinary [`Reference`] access, this guard exposes unresolved `Pending` state without waiting. A backend uses
/// it either to observe and retain a value for read-only work or to replace a value synchronously or asynchronously.
/// The guard holds the reference mutex for its entire lifetime and cannot move to another thread.
///
/// An asynchronous read keeps the state `Ready` or `Pending` and publishes its completion as a read lease:
///
/// ```mermaid
/// flowchart LR
///   state["Ready or Pending"] -->|observe| observation["ReferenceObservation"]
///   observation --> submission["Submit backend read"]
///   submission -->|use snapshot after dependency| execution["Backend execution"]
///   submission -->|publish completion as read lease| state
/// ```
///
/// Before removing the value, a later mutation prunes completed leases and waits for every lease still using the
/// snapshot. Publication is split into [`Self::validate_read_lease_publication`] and
/// [`Self::publish_read_lease_unchecked`] so a multi-reference backend can validate every reference before crossing the
/// irreversible submission boundary, then publish every lease infallibly after submission succeeds.
///
/// An asynchronous mutation first observes the value and prepares backend work. It then reacquires the guard and
/// verifies that the observed generation is still current before submitting that work:
///
/// ```mermaid
/// flowchart LR
///   state["Ready or Pending"] -->|observe| prepare["Prepare execution"]
///   prepare --> validate["Reacquire guard and check observation"]
///   validate -->|stale| state
///   validate -->|submission succeeds; begin_replacement| taken["Taken"]
///   taken -->|transaction.validate| validated["Validated replacement"]
///   validated -->|commit| pending["Pending"]
///   pending -->|completion succeeds| ready["Ready"]
///   pending -->|completion fails| poisoned["Poisoned"]
///   taken -->|poison or guard drop| poisoned
/// ```
///
/// [`Self::next_replacement_generation`] checks before submission that the value may be replaced.
/// [`Self::begin_replacement`] is called only after submission succeeds; it changes the state to `Taken`, in which no
/// value is accessible, and returns a [`ReferenceReplacementTransaction`]. The backend validates each reconstructed
/// replacement with its transaction. Successful validation returns a [`ValidatedPendingReplacementTransaction`] that
/// keeps its guard exclusively borrowed. A multi-reference backend validates every replacement before committing any
/// of them, preventing malformed results from being committed partially. Guards for multiple references must be
/// acquired and retained in ascending [`ReferenceId`] order throughout this process.
///
/// A synchronous or potentially donating backend follows the shorter extraction protocol:
///
/// ```mermaid
/// flowchart LR
///   ready["Ready"] -->|take| taken["Taken"]
///   taken -->|replace| ready
///   taken -->|poison or guard drop| poisoned["Poisoned"]
/// ```
///
/// [`Self::take`] returns the stored value and changes the state to `Taken`. Before releasing the guard, the backend
/// must call [`Self::replace`] to store its replacement or [`Self::poison`] to record that no valid replacement can be
/// provided. Dropping a guard while the state is `Taken` automatically marks the reference `Poisoned`, because the
/// previous value is no longer available and may already have been consumed by submitted work. A panic while the guard
/// instead protects `Ready` or `Pending` poisons the mutex, and later access reports [`ReferenceError::Poisoned`].
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct ReferenceGuard<'a, V: Value> {
    /// Handle that supplies this guard's alias-local type conversions.
    reference: &'a Reference<V>,

    /// Locked state of the shared reference allocation.
    state: MutexGuard<'a, ReferenceState<V>>,
}

// TODO(eaplatanios): Review this block.
impl<V: Value> ReferenceGuard<'_, V> {
    // Observation and retry coordination.

    /// Observes the reference's value, generation, and pending dependency without waiting for backend work.
    ///
    /// In `Ready` or `Pending` state, this function converts the stored value to this handle's type identities and
    /// returns all three fields as one coherent [`ReferenceObservation`]. A `Pending` value is returned immediately
    /// together with the cumulative dependency that backend work must wait for before using it. After preparing work
    /// and reacquiring the guard, use [`ReferenceObservation::is_current`] to determine whether the observed value is
    /// still the one stored by this reference.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] for `Frozen` state, [`ReferenceError::ExecutionPoisoned`] for `Poisoned`
    /// state, [`ReferenceError::TransactionInProgress`] for `Taken` state, or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn observe(&self) -> Result<ReferenceObservation<V>, ReferenceError> {
        let (value, generation, dependency) = match &*self.state {
            ReferenceState::Ready { value, generation, .. } => Ok((value, *generation, None)),
            ReferenceState::Pending { value, generation, completion, .. } => {
                Ok((value, *generation, Some(completion.clone())))
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }?;
        let snapshot = value
            .rename_type_identities(&self.reference.handle.storage_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        Ok(ReferenceObservation {
            holder: Arc::downgrade(&self.reference.handle.holder),
            generation,
            snapshot,
            dependency,
        })
    }

    /// Removes completed read leases and returns the leases for executions that may still be reading the value.
    ///
    /// Both successful and failed completed leases are removed because neither can still access the value. In `Ready`
    /// or `Pending` state, the returned tokens can be awaited before retrying a mutation. This function returns an empty
    /// vector in every other state.
    pub fn active_read_leases(&mut self) -> Vec<ReferenceCompletion> {
        let read_leases = match &mut *self.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => read_leases,
            _ => return Vec::new(),
        };
        // A terminal failure matters to the invocation that submitted the lease, but it no longer pins this reference's
        // value. Retain only executions that can still be reading the snapshot.
        read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
        read_leases.clone()
    }

    // Asynchronous read publication.

    /// Verifies that a read lease may be recorded while this guard remains locked.
    ///
    /// This succeeds only in `Ready` or `Pending` state and does not change the state. A backend that submits reads of
    /// multiple references can call this function for every guard before submission, then call
    /// [`Self::publish_read_lease_unchecked`] for each guard after submission succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] for `Frozen` state, [`ReferenceError::ExecutionPoisoned`] for `Poisoned`
    /// state, or [`ReferenceError::TransactionInProgress`] for `Taken` state.
    pub fn validate_read_lease_publication(&self) -> Result<(), ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Records a submitted read-only execution after [`Self::validate_read_lease_publication`] succeeds.
    ///
    /// The same guard must remain locked between validation and this function. `lease` must cover the submitted read
    /// and, when its snapshot came from `Pending` state, that snapshot's prior dependency. It must remain pending for
    /// as long as the execution may access the snapshot. Completed leases are removed before the new lease is recorded
    /// so a reference used only for reads does not retain backend resources indefinitely.
    ///
    /// # Parameters
    ///
    ///   - `lease`: Completion token for the submitted read-only execution.
    pub fn publish_read_lease_unchecked(&mut self, lease: ReferenceCompletion) {
        match &mut *self.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => {
                // Publication is the only lifecycle update guaranteed for a read-only reference. Pruning here keeps
                // completed executions from accumulating when no mutation path runs.
                read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
                read_leases.push(lease);
            }
            _ => unreachable!("read lease publication was validated under the same reference guard"),
        }
    }

    // Asynchronous replacement transactions.

    /// Checks whether the value can be replaced asynchronously and returns the replacement's generation.
    ///
    /// This function succeeds only when the state is `Ready` or `Pending`, the current generation has a successor, and
    /// no read lease is recorded. It does not change the reference state. The caller must retain this guard through
    /// submission and pass the returned generation to [`Self::begin_replacement`] after submission succeeds.
    ///
    /// Any recorded read lease rejects the mutation, even if that lease has already completed, because this function
    /// borrows the state immutably and cannot prune it. Call [`Self::active_read_leases`] first; if pending leases are
    /// returned, release the guard, await them, and retry the replacement.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::TransactionInProgress`] while a read lease is recorded or the state is `Taken`, the
    /// applicable terminal-state error for `Frozen` or `Poisoned`, or [`ReferenceError::GenerationExhausted`] when the
    /// current generation has no successor.
    pub fn next_replacement_generation(&self) -> Result<ReferenceGeneration, ReferenceError> {
        let generation = match &*self.state {
            ReferenceState::Ready { generation, read_leases, .. } => {
                if !read_leases.is_empty() {
                    return Err(ReferenceError::TransactionInProgress);
                }
                generation.next().ok_or(ReferenceError::GenerationExhausted)?
            }
            ReferenceState::Pending { generation, read_leases, .. } => {
                if !read_leases.is_empty() {
                    return Err(ReferenceError::TransactionInProgress);
                }
                generation.next().ok_or(ReferenceError::GenerationExhausted)?
            }
            ReferenceState::Frozen => return Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => {
                return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
            }
            ReferenceState::Taken { .. } => return Err(ReferenceError::TransactionInProgress),
        };
        Ok(generation)
    }

    /// Records a successfully submitted asynchronous replacement and makes the previous value inaccessible.
    ///
    /// The caller must first obtain `generation` from [`Self::next_replacement_generation`] under this same guard. This
    /// function changes `Ready` or `Pending` state to `Taken`, which retains the replacement generation but exposes no
    /// value. The returned [`ReferenceReplacementTransaction`] binds this reference allocation and generation to
    /// `completion`, preventing the reconstructed replacement from being paired with a different execution. The caller
    /// must keep the guard on the submitting thread until the replacement has been validated and committed, or call
    /// [`Self::poison`] if no valid replacement can be provided. Dropping the guard while the state remains `Taken`
    /// poisons the reference automatically.
    ///
    /// # Parameters
    ///
    ///   - `generation`: Generation returned by the immediately preceding [`Self::next_replacement_generation`] call.
    ///   - `completion`: Cumulative completion of the submitted mutation and all of its predecessor dependencies.
    pub fn begin_replacement(
        &mut self,
        generation: ReferenceGeneration,
        completion: ReferenceCompletion,
    ) -> ReferenceReplacementTransaction<V> {
        debug_assert_eq!(self.next_replacement_generation(), Ok(generation));
        *self.state = ReferenceState::Taken { generation };
        ReferenceReplacementTransaction {
            holder: Arc::downgrade(&self.reference.handle.holder),
            generation,
            completion,
        }
    }

    // Synchronous extraction.

    /// Extracts the current `Ready` value for a synchronous or potentially donating backend invocation.
    ///
    /// In `Ready` state, this function converts the stored value to the handle's type identities, advances its
    /// generation, returns the value, and changes the state to `Taken`. The `Taken` state retains only the new
    /// generation; it does not retain a copy of the returned value. Completed read leases are removed first. Any lease
    /// that remains pending rejects extraction because a donating execution could otherwise mutate storage that a
    /// submitted reader still uses. A `Pending` value must instead use the asynchronous replacement protocol. Before
    /// releasing the guard, the caller must store a replacement with [`Self::replace`] or record the failure with
    /// [`Self::poison`].
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::TransactionInProgress`] for a pending value or active read lease, the applicable
    /// terminal-state error for `Frozen` or `Poisoned`, [`ReferenceError::ValueReconstruction`] if identity conversion
    /// fails, or [`ReferenceError::GenerationExhausted`] when the current generation has no successor.
    pub fn take(&mut self) -> Result<V, ReferenceError> {
        if !self.active_read_leases().is_empty() {
            return Err(ReferenceError::TransactionInProgress);
        }
        let local = self.observe()?.snapshot;
        let generation = match &*self.state {
            ReferenceState::Ready { generation, .. } => generation.next().ok_or(ReferenceError::GenerationExhausted)?,
            _ => return Err(ReferenceError::TransactionInProgress),
        };
        *self.state = ReferenceState::Taken { generation };
        Ok(local)
    }

    /// Stores a synchronous replacement after [`Self::take`] and changes the state from `Taken` to `Ready`.
    ///
    /// The reference must still be `Taken` by this guard. This function validates the replacement's referent type,
    /// converts its type identities for shared storage, and stores it at the generation assigned by [`Self::take`]. A
    /// submitted asynchronous mutation must instead use [`Self::begin_replacement`], validate the returned
    /// [`ReferenceReplacementTransaction`] with its replacement, and commit the resulting token so the backend
    /// completion remains attached to the replacement.
    ///
    /// If replacement validation fails, the reference remains `Taken`; the caller may retry with a valid value or call
    /// [`Self::poison`] before releasing the guard.
    ///
    /// # Parameters
    ///
    ///   - `replacement`: New value for the reference, expressed using this handle's type identities.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::TransactionInProgress`] unless the reference state is `Taken`,
    /// [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type, or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn replace(&mut self, replacement: V) -> Result<(), ReferenceError> {
        let ReferenceState::Taken { generation } = *self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        let replacement = self.reference.prepare_replacement(replacement)?;
        *self.state = ReferenceState::Ready { value: replacement, generation, read_leases: Vec::new() };
        Ok(())
    }

    /// Changes from the `Taken` state to the terminal `Poisoned` state when no valid replacement can be provided.
    /// Call this after [`Self::take`] or [`Self::begin_replacement`] if the backend cannot produce or validate a
    /// replacement. Later access reports `reason` through [`ReferenceError::ExecutionPoisoned`]. This function is
    /// infallible so cleanup cannot replace the original failure with another error. It has no effect unless the
    /// reference state is `Taken`.
    ///
    /// # Parameters
    ///
    ///   - `reason`: Failure description reported by later attempts to access the underlying [`Reference`].
    #[inline]
    pub fn poison<R: Into<Arc<str>>>(&mut self, reason: R) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state = ReferenceState::Poisoned(reason.into());
        }
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    #[inline]
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            // `Taken` contains no value, and the previous value may already have been donated to irreversible backend
            // work. Make the missing replacement a permanent, explicit failure instead of fabricating a value.
            *self.state = ReferenceState::Poisoned("stateful transaction ended without restoring state".into());
        }
    }
}

/// Represents a coherent observation of a [`Reference`] value and the work that must precede its use.
/// [`ReferenceGuard::observe`] captures the handle-local value, its [`ReferenceGeneration`], and its optional
/// [`ReferenceCompletion`] under one lock. A backend can prepare work from [`Self::snapshot`], reacquire any handle for
/// the same reference allocation, and use [`Self::is_current`] to verify that the observation is still valid. When
/// [`Self::dependency`] is present, submitted work must wait for it before accessing the snapshot.
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
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] for `Frozen` state, [`ReferenceError::ExecutionPoisoned`] for `Poisoned`
    /// state, and [`ReferenceError::TransactionInProgress`] for `Taken` state.
    #[inline]
    pub fn is_current(&self, guard: &ReferenceGuard<'_, V>) -> Result<bool, ReferenceError> {
        match &*guard.state {
            ReferenceState::Ready { generation, .. } | ReferenceState::Pending { generation, .. } => {
                Ok(std::ptr::eq(self.holder.as_ptr(), Arc::as_ptr(&guard.reference.handle.holder))
                    && *generation == self.generation)
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
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

/// Proof that an asynchronous [`Reference`] replacement was submitted successfully and is awaiting its reconstructed
/// value. [`ReferenceGuard::begin_replacement`] creates this transaction while changing the reference state to `Taken`.
/// It binds the reference allocation, the generation assigned to its missing replacement, and the cumulative backend
/// completion. Calling [`Self::validate`] checks those bindings and prepares the replacement value before returning a
/// [`ValidatedPendingReplacementTransaction`]. A backend that replaces multiple references should validate every
/// replacement before committing any of them. Dropping this transaction does not change the `Taken` reference state.
/// After any failure, the backend must call [`ReferenceGuard::poison`] before releasing the guard.
#[must_use = "a submitted reference replacement must be committed or its reference must be poisoned"]
pub struct ReferenceReplacementTransaction<V: Value> {
    /// Weak pointer identifying the [`Reference`] allocation whose value is being replaced.
    holder: Weak<ReferenceHolder<V>>,

    /// [`ReferenceGeneration`] assigned to the submitted replacement.
    generation: ReferenceGeneration,

    /// Cumulative [`ReferenceCompletion`] of the submitted mutation and its predecessor dependencies.
    completion: ReferenceCompletion,
}

impl<V: Value> ReferenceReplacementTransaction<V> {
    /// Validates a replacement value and keeps its [`ReferenceGuard`] exclusively borrowed until commit. This function
    /// verifies that the transaction and guard refer to the same allocation and replacement generation, checks the
    /// replacement's referent type, and converts its type identities for shared storage without changing the `Taken`
    /// state. The returned [`ValidatedPendingReplacementTransaction`] contains everything needed to commit without
    /// failure. Keeping `guard` borrowed prevents the reference state from changing between validation and commit.
    /// A multi-reference backend can therefore validate every replacement before committing any of them.
    ///
    /// # Parameters
    ///
    ///   - `guard`: [`ReferenceGuard`] for the reference allocation being replaced.
    ///   - `replacement`: New value for the reference, expressed using `guard`'s handle-local type identities.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReplacementTransactionMismatch`] if this transaction belongs to another allocation or
    /// does not match the generation stored in `Taken` state, the applicable state error unless the guard protects
    /// `Taken` state, [`ReferenceError::ReferentTypeMismatch`] if `replacement` has the wrong type, or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn validate<'g, 'r>(
        self,
        guard: &'g mut ReferenceGuard<'r, V>,
        replacement: V,
    ) -> Result<ValidatedPendingReplacementTransaction<'g, 'r, V>, ReferenceError> {
        if !std::ptr::eq(self.holder.as_ptr(), Arc::as_ptr(&guard.reference.handle.holder)) {
            return Err(ReferenceError::ReplacementTransactionMismatch);
        }
        match &*guard.state {
            ReferenceState::Taken { generation } if *generation == self.generation => {}
            ReferenceState::Taken { .. } => return Err(ReferenceError::ReplacementTransactionMismatch),
            ReferenceState::Frozen => return Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => {
                return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
            }
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => {
                return Err(ReferenceError::TransactionInProgress);
            }
        }
        let replacement = guard.reference.prepare_replacement(replacement)?;
        Ok(ValidatedPendingReplacementTransaction {
            guard,
            generation: self.generation,
            value: replacement,
            completion: self.completion,
        })
    }
}

/// A replacement prepared to change its [`Reference`] from `Taken` to `Pending` without failure.
/// [`ReferenceReplacementTransaction::validate`] creates this transaction after confirming that its source transaction,
/// replacement value, and exclusively borrowed guard all refer to the same allocation and generation. Holding the guard
/// borrow prevents the `Taken` state from changing before [`Self::commit`]. A multi-reference backend can keep one
/// validated transaction per disjoint guard, then commit them only after every replacement validates successfully.
#[must_use = "a validated reference replacement must be committed or its reference must be poisoned"]
pub struct ValidatedPendingReplacementTransaction<'g, 'r, V: Value> {
    /// Exclusively borrowed [`ReferenceGuard`] protecting the validated `Taken` state.
    guard: &'g mut ReferenceGuard<'r, V>,

    /// [`ReferenceGeneration`] assigned to the submitted replacement.
    generation: ReferenceGeneration,

    /// Replacement [`Value`] using the allocation's canonical type identities.
    value: V,

    /// Cumulative [`ReferenceCompletion`] of the submitted mutation and its predecessor dependencies.
    completion: ReferenceCompletion,
}

impl<V: Value> ValidatedPendingReplacementTransaction<'_, '_, V> {
    /// Stores the validated replacement and changes its [`Reference`] state from `Taken` to `Pending`. This function is
    /// infallible because validation established every precondition while exclusively borrowing the [`ReferenceGuard`].
    /// After the stored completion resolves, ordinary reference access reconciles `Pending` to `Ready` on success or
    /// `Poisoned` on backend failure.
    #[inline]
    pub fn commit(self) {
        let Self { guard, generation, value, completion } = self;
        *guard.state = ReferenceState::Pending { value, generation, completion, read_leases: Vec::new() };
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

    // Simulates the successful-submission and replacement-commit phases used by asynchronous backend tests.
    fn transition_to_pending<V: Value>(
        guard: &mut ReferenceGuard<'_, V>,
        completion: ReferenceCompletion,
        replacement: V,
    ) -> Result<ReferenceGeneration, ReferenceError> {
        let generation = guard.next_replacement_generation()?;
        let transaction = guard.begin_replacement(generation, completion);
        transaction.validate(guard, replacement)?.commit();
        Ok(generation)
    }

    // Completion backend with a bounded waiter handshake so a broken asynchronous protocol fails instead of hanging.
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
    fn test_reference_clone_shares_allocation_identity_hashing_and_rendering() {
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

        // Display is deterministic and type-based, while Debug also exposes process-local allocation identity.
        assert_eq!(reference.to_string(), "ref<f32[2]>");
        assert_eq!(reference.to_string(), distinct.to_string());
        assert_eq!(
            format!("{reference:?}"),
            format!("Reference {{ id: {:?}, type: {:?} }}", reference.id(), reference.r#type()),
        );

        let mut reference_hasher = DefaultHasher::new();
        reference.hash(&mut reference_hasher);
        let mut alias_hasher = DefaultHasher::new();
        alias.hash(&mut alias_hasher);
        assert_eq!(reference_hasher.finish(), alias_hasher.finish());

        let references = HashMap::from([(reference.clone(), "root")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&distinct), None);
    }

    #[test]
    fn test_reference_clone_shares_one_word_handle_storage() {
        assert_eq!(size_of::<Reference<Array>>(), size_of::<usize>());

        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let alias = reference.clone();
        assert!(Arc::ptr_eq(&alias.handle, &reference.handle));
    }

    #[test]
    fn test_reference_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_reference_mutations_preserve_snapshots_and_independent_allocations() {
        let initializer = Array::vector(vec![1.0_f32, 2.0]);
        let first = Reference::new(initializer.clone()).unwrap();
        let second = Reference::new(initializer.clone()).unwrap();
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
        assert_eq!(second.read(), Ok(initializer.clone()));
        assert_eq!(second.swap(Array::vector(vec![5.0_f32, 6.0])), Ok(initializer.clone()));

        assert_eq!(initializer, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(read_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(retained_replacement, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(first.read(), Ok(Array::vector(vec![17.0_f32, 28.0])));
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
    fn test_reference_generation_advances_only_after_committed_mutations() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let initial_observation = reference.lock().unwrap().observe().unwrap();
        let initial_generation = initial_observation.generation();

        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.lock().unwrap().observe().unwrap().generation(), initial_generation);
        assert!(initial_observation.is_current(&reference.lock().unwrap()).unwrap());

        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
        let swapped_generation = reference.lock().unwrap().observe().unwrap().generation();
        assert_eq!(swapped_generation, ReferenceGeneration(initial_generation.0 + 1));
        assert!(!initial_observation.is_current(&reference.lock().unwrap()).unwrap());

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
        let mut guard = reference.lock().unwrap();
        assert_eq!(guard.next_replacement_generation(), Err(ReferenceError::GenerationExhausted));
        assert_eq!(guard.take(), Err(ReferenceError::GenerationExhausted));
        let observation = guard.observe().unwrap();
        assert_eq!(observation.generation(), ReferenceGeneration(u64::MAX));
        assert_eq!(observation.snapshot(), &Array::scalar(1.0_f32));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
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
        assert_eq!(HashMap::from([(reference.clone(), "root")]).get(&renamed), Some(&"root"));

        // Observation validity follows allocation identity, while its snapshot retains the originating handle's type
        // identities and therefore needs no reinterpretation through the validating alias.
        let observation = reference.lock().unwrap().observe().unwrap();
        let unrelated = Reference::new(CaptureReference::new(0, source_type.clone())).unwrap();
        assert!(observation.is_current(&reference.lock().unwrap()).unwrap());
        assert!(observation.is_current(&renamed.lock().unwrap()).unwrap());
        assert!(!observation.is_current(&unrelated.lock().unwrap()).unwrap());

        assert_eq!(chained.write(CaptureReference::new(1, target_type.clone())), Ok(()));
        assert_eq!(reference.read(), Ok(CaptureReference::new(1, source_type.clone())));
        assert_eq!(renamed.read(), Ok(CaptureReference::new(1, middle_type.clone())));

        // Both guard replacement protocols accept values in the locking handle's identity space and convert them back
        // to the allocation's storage identities before making them observable through every alias.
        let mut guard = chained.lock().unwrap();
        assert_eq!(guard.take(), Ok(CaptureReference::new(1, target_type.clone())));
        guard.replace(CaptureReference::new(2, target_type)).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(CaptureReference::new(2, source_type.clone())));

        let mut guard = renamed.lock().unwrap();
        let generation = guard.next_replacement_generation().unwrap();
        let transaction = guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));
        transaction.validate(&mut guard, CaptureReference::new(3, middle_type)).unwrap().commit();
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

    #[test]
    fn test_reference_observation_reports_terminal_state_before_identity_mismatch() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let unrelated = Reference::new(Array::scalar(2.0_f32)).unwrap();
        let unrelated_observation = unrelated.lock().unwrap().observe().unwrap();
        let mut guard = reference.lock().unwrap();

        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        assert!(matches!(guard.observe(), Err(ReferenceError::TransactionInProgress)));
        assert_eq!(unrelated_observation.is_current(&guard), Err(ReferenceError::TransactionInProgress));
        guard.poison("injected failure");
        assert!(matches!(
            guard.observe(),
            Err(ReferenceError::ExecutionPoisoned { reason }) if reason == "injected failure",
        ));
        assert_eq!(
            unrelated_observation.is_current(&guard),
            Err(ReferenceError::ExecutionPoisoned { reason: "injected failure".to_string() }),
        );
    }

    #[test]
    fn test_reference_guard_replace_checks_state_before_replacement_type() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();

        assert_eq!(guard.replace(Array::vector(vec![2.0_f32])), Err(ReferenceError::TransactionInProgress),);
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_reference_guard_replace_preserves_taken_state_after_type_mismatch() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));

        assert!(matches!(
            guard.replace(Array::vector(vec![2.0_f32])),
            Err(ReferenceError::ReferentTypeMismatch { expected, actual })
                if expected == "f32[]" && actual == "f32[1]"
        ));
        assert!(matches!(guard.observe(), Err(ReferenceError::TransactionInProgress)));
        assert_eq!(guard.replace(Array::scalar(3.0_f32)), Ok(()));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_replacement_transaction_rejects_another_reference_allocation() {
        let first = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let second = Reference::new(Array::scalar(2.0_f32)).unwrap();
        let mut first_guard = first.lock().unwrap();
        let mut second_guard = second.lock().unwrap();
        let generation = first_guard.next_replacement_generation().unwrap();
        let transaction = first_guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));

        assert!(matches!(
            transaction.validate(&mut second_guard, Array::scalar(3.0_f32)),
            Err(ReferenceError::ReplacementTransactionMismatch),
        ));
        first_guard.poison("test cleanup");
        drop(first_guard);
        drop(second_guard);
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_replacement_transaction_reports_terminal_state_during_validation() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        let generation = guard.next_replacement_generation().unwrap();
        let transaction = guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));
        guard.poison("injected failure");

        assert!(matches!(
            transaction.validate(&mut guard, Array::scalar(2.0_f32)),
            Err(ReferenceError::ExecutionPoisoned { reason }) if reason == "injected failure",
        ));
    }

    #[test]
    fn test_reference_replacement_transaction_preserves_taken_state_after_type_mismatch() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        let generation = guard.next_replacement_generation().unwrap();
        let transaction = guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));

        assert!(matches!(
            transaction.validate(&mut guard, Array::vector(vec![2.0_f32])),
            Err(ReferenceError::ReferentTypeMismatch { expected, actual })
                if expected == "f32[]" && actual == "f32[1]"
        ));
        assert!(matches!(guard.observe(), Err(ReferenceError::TransactionInProgress)));
        guard.poison("test cleanup");
    }

    #[test]
    fn test_reference_read_ignores_a_stale_pending_completion() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let first_backend = ControlledCompletion::new();
        let second_backend = ControlledCompletion::new();

        let first_generation = {
            let mut guard = reference.lock().unwrap();
            transition_to_pending(&mut guard, ReferenceCompletion::new(first_backend.clone()), Array::scalar(2.0_f32))
                .unwrap()
        };

        // The reader captures generation one and releases the mutex while awaiting its completion.
        let (sender, receiver) = channel();
        let reader_reference = Arc::clone(&reference);
        let reader = thread::spawn(move || sender.send(reader_reference.read()).unwrap());
        first_backend.wait_until_awaited();

        // Commit generation two while the reader is waiting. Its completion includes generation one's dependency,
        // matching the cumulative dependency contract required of chained submissions.
        let second_generation = {
            let mut guard = reference.lock().unwrap();
            let observation = guard.observe().unwrap();
            assert_eq!(observation.generation(), first_generation);
            let completion = ReferenceCompletion::joined([
                observation.dependency().unwrap().clone(),
                ReferenceCompletion::new(second_backend.clone()),
            ]);
            transition_to_pending(&mut guard, completion, Array::scalar(3.0_f32)).unwrap()
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
    fn test_reference_guard_drop_during_submitted_mutation_unwind_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut guard = reference.lock().unwrap();
                let generation = guard.next_replacement_generation().unwrap();
                let _transaction = guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));
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

    #[test]
    fn test_reference_guard_next_replacement_generation_requires_terminal_read_leases_to_be_pruned() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::ready(Ok(())));
        assert_eq!(guard.next_replacement_generation(), Err(ReferenceError::TransactionInProgress));
        assert!(guard.active_read_leases().is_empty());
        let generation = guard.next_replacement_generation().unwrap();
        let _transaction = guard.begin_replacement(generation, ReferenceCompletion::ready(Ok(())));
        guard.poison("test cleanup");
    }

    #[test]
    fn test_reference_read_waits_for_pending_completion_without_holding_the_lock() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        transition_to_pending(&mut guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32)).unwrap();
        let pending_observation = guard.observe().unwrap();
        assert!(pending_observation.dependency().is_some());
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

        // Applying the successful completion made the shared value ready without changing its generation. The pending
        // observation remains current, while a new observation no longer carries the completed dependency.
        let guard = reference.lock().unwrap();
        assert!(pending_observation.is_current(&guard).unwrap());
        assert!(guard.observe().unwrap().dependency().is_none());
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_write_waits_for_pending_completion_without_holding_the_lock() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        transition_to_pending(&mut guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32)).unwrap();
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
        let mut guard = reference.lock().unwrap();
        transition_to_pending(&mut guard, ReferenceCompletion::new(backend.clone()), Array::scalar(2.0_f32)).unwrap();
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
    fn test_reference_write_waits_for_an_active_read_lease() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        let (sender, receiver) = channel();
        let writing_reference = Arc::clone(&reference);
        let writer = thread::spawn(move || sender.send(writing_reference.write(Array::scalar(2.0_f32))).unwrap());
        lease.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        lease.complete(Ok(()));
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
    fn test_reference_swap_waits_for_an_active_read_lease() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        // A leased snapshot pins the current value, so the replacement cannot be committed until the lease completes.
        let (sender, receiver) = channel();
        let swapping_reference = Arc::clone(&reference);
        let swapper = thread::spawn(move || sender.send(swapping_reference.swap(Array::scalar(2.0_f32))).unwrap());
        lease.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        lease.complete(Ok(()));
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
    fn test_reference_freeze_waits_for_an_active_read_lease() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)).unwrap());
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        // Consumption is a mutation of the alias family, so it waits for the same leases a replacement would.
        let (sender, receiver) = channel();
        let freezing_reference = Arc::clone(&reference);
        let freezer = thread::spawn(move || sender.send(freezing_reference.freeze()).unwrap());
        lease.wait_until_awaited();
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        lease.complete(Ok(()));
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
    fn test_reference_replacement_transaction_commits_its_claimed_generation() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        let first =
            transition_to_pending(&mut guard, ReferenceCompletion::ready(Ok(())), Array::scalar(2.0_f32)).unwrap();
        let second = guard.next_replacement_generation().unwrap();
        let transaction = guard.begin_replacement(second, ReferenceCompletion::ready(Ok(())));
        transaction.validate(&mut guard, Array::scalar(3.0_f32)).unwrap().commit();
        assert_eq!(first.next(), Some(second));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_guard_take_rejects_an_active_read_lease() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));

        // Handing leased buffers to a donating execution would let the device mutate storage a submitted read-only
        // execution still observes, so extraction is rejected until the lease completes and is pruned.
        assert_eq!(guard.take(), Err(ReferenceError::TransactionInProgress));
        lease.complete(Ok(()));
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        guard.replace(Array::scalar(4.0_f32)).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_affects_only_the_taken_reference() {
        let first = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let second = Reference::new(Array::scalar(2.0_f32)).unwrap();
        let mut first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        first_guard.take().unwrap();
        first_guard.poison("test execution failed");
        drop(first_guard);
        drop(second_guard);
        assert_eq!(
            first.read(),
            Err(ReferenceError::ExecutionPoisoned { reason: "test execution failed".to_string() }),
        );
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_leaves_a_ready_reference_unchanged() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        let generation = guard.observe().unwrap().generation();

        // Poisoning is infallible so a cleanup path cannot replace the original backend error with a guard-state error.
        // A guard that does not own a `Taken` transaction has nothing to invalidate, so the reference remains ready at
        // its current generation.
        guard.poison("unrelated backend failure");
        let observation = guard.observe().unwrap();
        assert_eq!(observation.generation(), generation);
        assert_eq!(observation.snapshot(), &Array::scalar(1.0_f32));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_leaves_a_pending_reference_unchanged() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let completion = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let generation =
            transition_to_pending(&mut guard, ReferenceCompletion::new(completion.clone()), Array::scalar(2.0_f32))
                .unwrap();

        // Explicit poisoning applies only to an uncommitted `Taken` transaction. Once a replacement is committed,
        // its cumulative completion remains the sole authority for promoting or poisoning that pending generation.
        guard.poison("unrelated backend failure");
        let observation = guard.observe().unwrap();
        assert_eq!(observation.generation(), generation);
        assert!(observation.dependency().is_some());
        drop(guard);
        completion.complete(Ok(()));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_guard_read_lease_publication_releases_terminal_leases() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let first = ControlledCompletion::new();
        let second = ControlledCompletion::new();
        let third = ControlledCompletion::new();
        first.complete(Ok(()));
        second.complete(Err("read failed".into()));

        // Lease publication is the only lifecycle update guaranteed for a read-only reference, so it must also release
        // terminal leases instead of retaining their backend resources indefinitely. The strong counts make those
        // releases observable here.
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(first.clone()));
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(second.clone()));
        assert_eq!(Arc::strong_count(&first.state), 1);
        assert_eq!(Arc::strong_count(&second.state), 2);
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(third.clone()));
        assert_eq!(Arc::strong_count(&second.state), 1);
        assert_eq!(Arc::strong_count(&third.state), 2);

        // Only the running lease remains. It blocks the next submitted mutation until completion and pruning.
        assert_eq!(guard.active_read_leases().len(), 1);
        assert_eq!(guard.next_replacement_generation(), Err(ReferenceError::TransactionInProgress));
        third.complete(Ok(()));
        assert!(guard.active_read_leases().is_empty());
        assert_eq!(Arc::strong_count(&third.state), 1);
        assert_eq!(guard.next_replacement_generation(), Ok(ReferenceGeneration(1)));
    }

    #[test]
    fn test_reference_guard_drop_after_take_poisons_reference() {
        let reference = Reference::new(Array::scalar(1.0_f32)).unwrap();
        let mut guard = reference.lock().unwrap();
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        drop(guard);
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

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
