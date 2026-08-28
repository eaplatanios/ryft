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
/// into a program or compilation key. The identity may be reused after all handles and prepared replacements tied to
/// that allocation are dropped.
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
/// and every asynchronous installation and completion path retains the generation it belongs to. The reference applies
/// a delayed completion only while that generation remains current, preventing an older execution from completing or
/// poisoning state installed by a newer mutation. Generations are local to one reference allocation and do not
/// identify references across allocations or processes.
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
///   Taken --> Ready: install synchronous replacement
///   Taken --> Pending: install asynchronous replacement
///   Taken --> Poisoned: guard drop or backend failure
///   Pending --> Ready: backend completion succeeds
///   Pending --> Poisoned: backend completion fails
///   Ready --> Frozen: freeze
/// ```
///
/// `Ready` exposes a completed immutable snapshot. `Taken` is an exclusive transaction whose [`ReferenceGuard`] retains
/// exclusive access to the shared state while a synchronous backend prepares a replacement or an asynchronous backend
/// reconstructs submitted hidden state. Dropping that guard before installation poisons the reference defensively. An
/// asynchronous installation changes the state to `Pending`, where the new value remains inaccessible until its
/// cumulative [`ReferenceCompletion`] succeeds. Generations ensure that completion of an older execution cannot
/// affect a newer mutation. `Frozen` and `Poisoned` are terminal states.
///
/// Read-only asynchronous executions do not enter a separate state. They publish completion leases on `Ready` or
/// `Pending`. Mutating transactions wait for those leases before they may take the reference, preventing donated
/// storage from racing a reader that still observes the current snapshot.
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
        let root_type = value.r#type().into_owned();
        if root_type.is_reference() {
            return Err(ReferenceError::NestedReferent { referent_type: root_type.to_string() });
        }
        Ok(Self {
            handle: Arc::new(ReferenceHandle {
                holder: Arc::new(ReferenceHolder {
                    root_type: root_type.clone(),
                    state: Mutex::new(ReferenceState::Ready {
                        value,
                        generation: ReferenceGeneration::initial(),
                        read_leases: Vec::new(),
                    }),
                }),
                r#type: ReferenceType::new(root_type),
                root_to_handle: TypeIdentityRenaming::new(),
                handle_to_root: TypeIdentityRenaming::new(),
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
    pub fn uses_root_type_identities(&self) -> bool {
        self.handle.root_to_handle.is_identity() && self.handle.handle_to_root.is_identity()
    }

    /// Returns an immutable snapshot of this [`Reference`]'s current value. If a submitted mutation is still pending,
    /// this function waits for its completion without holding the state mutex and then reads the installed value. It
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
            .rename_type_identities(&self.handle.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
    }

    /// Atomically replaces this [`Reference`]'s current value. This function first waits for pending mutations and
    /// active read-only executions to finish. It then verifies that `replacement` exactly matches this alias's referent
    /// type, converts identity-renamed values to the shared storage representation, installs the replacement, and
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
            .rename_type_identities(&self.handle.root_to_handle)
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
            .rename_type_identities(&self.handle.root_to_handle)
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
            .rename_type_identities(&self.handle.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        *state = ReferenceState::Frozen;
        Ok(value)
    }

    /// Acquires this [`Reference`]'s shared lifecycle state for one backend-managed transaction. This is the
    /// backend-facing wrapper around the private raw state lock. It returns as soon as the state mutex is acquired and
    /// deliberately does not await a `Pending` value or active read leases. The returned [`ReferenceGuard`] exposes the
    /// pending dependency and the validated transitions a backend needs to compose, submit, and install stateful work
    /// while retaining exclusive access.
    ///
    /// Ordinary value access should use [`read`](Self::read), [`write`](Self::write), [`swap`](Self::swap),
    /// [`update`](Self::update), or [`freeze`](Self::freeze), which reconcile pending work before accessing the value.
    /// Backends that lock multiple references must acquire them in ascending [`ReferenceId`] order and retain that
    /// order until every submitted hidden replacement has been validated and installed. A submitted mutation remains
    /// represented by `Taken` while this guard is held, so dropping the guard before installation poisons the
    /// reference.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Poisoned`] when the state mutex was poisoned without the explicit terminal `Poisoned`
    /// lifecycle state that [`ReferenceGuard`] installs defensively during an unwind.
    #[inline]
    pub fn lock(&self) -> Result<ReferenceGuard<'_, V>, ReferenceError> {
        let state = self.lock_holder_state()?;
        Ok(ReferenceGuard { reference: self, state })
    }

    /// Acquires the [`Reference`] in `Ready` state after reconciling work that prevents the requested value access.
    /// Each iteration acquires the raw mutex through [`Self::lock_holder_state`]. A `Pending` value causes this method
    /// to release the mutex, await its cumulative completion, reacquire the mutex, and apply the result only if that
    /// generation is still current. When read leases must also finish, the method prunes completed leases, releases the
    /// mutex while awaiting the remaining leases, and retries. It never awaits backend work while holding the mutex.
    ///
    /// Unlike [`lock`](Self::lock), this method is the private ordinary-access path: it hides pending-state
    /// reconciliation and returns a raw guard proven to contain `Ready`. It does not return [`ReferenceGuard`] because
    /// ordinary reads and mutations do not participate in the backend transaction protocol or its `Taken`-on-drop
    /// poisoning rule.
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
                    // `Taken` is unobservable during a valid transaction because its guard retains this mutex.
                    // Competing accesses block before reading the state. Keep the arm as a defensive contract
                    // check for misuse.
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

        // Reads begin with values represented in the allocation's original identity space. Compose the existing
        // root-to-current conversion with the requested current-to-renamed step to produce the new read conversion.
        let root_type = &self.handle.holder.root_type;
        let mut root_to_handle = TypeIdentityRenaming::new();
        for (_, identity) in root_type.identities() {
            root_to_handle.insert(identity.clone(), renaming.rename(&self.handle.root_to_handle.rename(identity)))?;
        }

        // Writes travel in the opposite direction. Compose the renamed-to-current inverse with this alias's existing
        // current-to-root conversion to produce the new write conversion.
        let mut handle_to_root = TypeIdentityRenaming::new();
        for (_, identity) in renamed_type.identities() {
            handle_to_root
                .insert(identity.clone(), self.handle.handle_to_root.rename(&inverse_step.rename(identity)))?;
        }

        // Check each complete conversion against its concrete endpoint types. This catches inconsistent identity
        // implementations even when every individual mapping above was constructible.
        if root_type.rename_identities(&root_to_handle)? != renamed_type
            || renamed_type.rename_identities(&handle_to_root)? != *root_type
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
                root_to_handle,
                handle_to_root,
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

    /// Validates a handle-local replacement value and converts it to the shared root representation. The input must
    /// exactly match this [`Reference`]'s referent type. After applying the handle-to-root identity mapping, the
    /// converted value must exactly match the reference allocation's root referent type. Both checks finish before
    /// any reference state is changed, so callers may safely commit or install the returned value.
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

        // Every alias stores values in the reference allocation's root identity namespace. This mapping converts the
        // handle-local replacement into that shared representation.
        let stored = value
            .rename_type_identities(&self.handle.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;

        // Identity renaming is implemented by the value family, so verify that its result exactly matches the root
        // referent type before allowing the value to reach shared state.
        validate_type(&stored, &self.handle.holder.root_type)?;

        Ok(stored)
    }

    /// Installs a prepared replacement into an already-locked `Ready` state and advances its [`ReferenceGeneration`].
    /// The next generation is computed before either field is changed. If the generation space is exhausted, this
    /// function returns [`ReferenceError::GenerationExhausted`] and leaves both `current` and `generation` unchanged.
    /// `replacement` must already have been validated and converted to the shared root representation by
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

// TODO(eaplatanios): Review from here onwards.

/// Immutable metadata for one [`Reference`] handle.
///
/// Exact clones share this metadata. An identity-renamed alias receives a new `ReferenceHandle` with its own referent
/// type and conversion mappings while continuing to share the same [`ReferenceHolder`] and runtime state. The fields
/// are never changed after construction.
struct ReferenceHandle<V: Value> {
    /// Shared allocation whose pointer identity defines reference equality, hashing, and [`ReferenceId`].
    holder: Arc<ReferenceHolder<V>>,

    /// Referent type exposed through this handle.
    r#type: ReferenceType<V::Type>,

    /// Converts stored values from the allocation's identities to this handle's identities.
    root_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Converts values from this handle's identities to the allocation's identities.
    handle_to_root: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

/// Allocation shared by every handle in one reference alias family.
struct ReferenceHolder<V: Value> {
    /// Canonical referent type of stored values.
    ///
    /// Identity-renamed aliases may expose a different handle-local type and convert at this allocation boundary. The
    /// canonical type is immutable and available without locking so replacement validation does not need to acquire
    /// the lifecycle mutex.
    root_type: V::Type,

    /// Synchronized value and lifecycle state.
    state: Mutex<ReferenceState<V>>,
}

/// Value and lifecycle state shared by one reference alias family.
enum ReferenceState<V: Value> {
    /// Current value is available and the mutation that produced it has completed.
    Ready {
        /// Current immutable value in the allocation's canonical identity space.
        value: V,

        /// Generation of `value`.
        generation: ReferenceGeneration,

        /// Submitted read-only executions that may still be using `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// Current replacement is installed, but the execution producing it has not completed.
    Pending {
        /// Pending replacement in the allocation's canonical identity space.
        value: V,

        /// Generation that installed `value`.
        generation: ReferenceGeneration,

        /// Completion of this generation and every predecessor on which it depends.
        completion: ReferenceCompletion,

        /// Submitted read-only executions that may still be using `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// A backend transaction has claimed the next generation but has not installed its replacement.
    Taken {
        /// Generation claimed by the transaction.
        generation: ReferenceGeneration,
    },

    /// No trustworthy value remains after a submitted mutation failed or a transaction ended without a replacement.
    Poisoned(Arc<str>),

    /// Value was consumed by [`Reference::freeze`].
    Frozen,
}

/// Backend-facing exclusive access to one reference allocation.
///
/// Unlike ordinary [`Reference`] access, this guard exposes `Pending` state without waiting so an asynchronous backend
/// can add [`Self::dependency`] to its next submission. A mutation uses a validate-then-commit protocol: obtain a
/// generation with [`Self::next_generation`], retain the same guard while submitting work, call
/// [`Self::begin_submitted_mutation`] after submission succeeds, prepare and validate every replacement in the batch,
/// and then call [`Self::install_pending_unchecked`]. Multi-reference backends must hold guards in ascending
/// [`ReferenceId`] order throughout this protocol.
///
/// A synchronous backend instead calls [`Self::take`] followed by [`Self::install`] on success or [`Self::poison`] if
/// the extracted value cannot be restored safely.
///
/// Dropping a guard in `Taken` state marks the reference `Poisoned`, because an extracted value may already have been
/// consumed. A panic while the guard instead protects `Ready` or `Pending` state poisons the mutex and later access
/// reports [`ReferenceError::Poisoned`]. The guard is not transferable between threads; submission and installation
/// must happen on the thread that acquired it.
pub struct ReferenceGuard<'a, V: Value> {
    /// Handle that supplies this guard's alias-local type conversions.
    reference: &'a Reference<V>,

    /// Locked state of the shared reference allocation.
    state: MutexGuard<'a, ReferenceState<V>>,
}

impl<V: Value> ReferenceGuard<'_, V> {
    /// Returns the current `Ready` or `Pending` generation without resolving pending work.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] for `Frozen` state, [`ReferenceError::ExecutionPoisoned`] for `Poisoned`
    /// state, or [`ReferenceError::TransactionInProgress`] for `Taken` state.
    pub fn current_generation(&self) -> Result<ReferenceGeneration, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { generation, .. } => Ok(*generation),
            ReferenceState::Pending { generation, .. } => Ok(*generation),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Clones the current `Ready` or `Pending` value and converts it to this handle's type identities.
    ///
    /// This method does not extract the stored value or wait for a `Pending` generation. Before submitting work that
    /// uses a pending snapshot, a backend must include [`Self::dependency`] in that submission.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::Frozen`] for `Frozen` state, [`ReferenceError::ExecutionPoisoned`] for `Poisoned`
    /// state, [`ReferenceError::TransactionInProgress`] for `Taken` state, or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn snapshot(&self) -> Result<V, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { value, .. } | ReferenceState::Pending { value, .. } => value
                .rename_type_identities(&self.reference.handle.root_to_handle)
                .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() }),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Returns the `Pending` generation's cumulative completion dependency.
    ///
    /// This completion must precede any backend use of the snapshot returned by [`Self::snapshot`]. It may already be
    /// terminal because pending state is reconciled lazily by ordinary reference access. Returns `None` for every other
    /// state.
    pub fn dependency(&self) -> Option<ReferenceCompletion> {
        match &*self.state {
            ReferenceState::Pending { completion, .. } => Some(completion.clone()),
            _ => None,
        }
    }

    /// Prunes completed read leases and returns clones of those that remain pending.
    ///
    /// Both successful and failed terminal leases are removed because neither can still access the value. Returns an
    /// empty vector unless the reference is `Ready` or `Pending`.
    pub fn active_read_leases(&mut self) -> Vec<ReferenceCompletion> {
        let read_leases = match &mut *self.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => read_leases,
            _ => return Vec::new(),
        };
        // A terminal failure matters to the invocation that owns the lease, but it no longer pins this reference's
        // value. Retain only executions that can still be reading the snapshot.
        read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
        read_leases.clone()
    }

    /// Verifies that the reference can accept a read lease without changing its state.
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

    /// Publishes a read lease after [`Self::validate_read_lease_publication`] succeeded under this same guard.
    ///
    /// `lease` must cover the submitted read and, when its snapshot came from `Pending` state, that snapshot's prior
    /// dependency. It must remain pending for as long as the execution may access the snapshot. Completed leases are
    /// pruned during publication so a reference used only for reads does not retain backend resources indefinitely.
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
            _ => unreachable!("read lease publication was validated under the same holder guard"),
        }
    }

    /// Runs the complete mutation-publication protocol for tests.
    ///
    /// `completion` must include this reference's prior pending dependency and the newly submitted execution. This is a
    /// submission-time safety obligation: joining the predecessor after submission cannot prevent the backend from
    /// reading or replacing pending storage before the predecessor finishes.
    #[cfg(test)]
    fn submit_pending(
        &mut self,
        completion: ReferenceCompletion,
        replacement: ReferenceReplacement<V>,
    ) -> Result<ReferenceGeneration, ReferenceError> {
        let generation = self.next_generation()?;
        self.begin_submitted_mutation(generation);
        self.validate_pending_install(generation, &replacement)?;
        self.install_pending_unchecked(generation, completion, replacement);
        Ok(generation)
    }

    /// Returns the generation that a new mutation would claim without changing the reference state.
    ///
    /// The caller must retain this guard through submission and pass the returned generation to
    /// [`Self::begin_submitted_mutation`] from the backend's successful-submission callback.
    ///
    /// Any recorded read lease rejects the mutation, even if that lease has already completed, because this method
    /// borrows the state immutably and cannot prune it. Call [`Self::active_read_leases`] first; if pending leases are
    /// returned, release the guard, await them, and retry the transaction.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::TransactionInProgress`] while a read lease is recorded or the state is `Taken`, the
    /// applicable terminal-state error for `Frozen` or `Poisoned`, or [`ReferenceError::GenerationExhausted`] when the
    /// current generation has no successor.
    pub fn next_generation(&self) -> Result<ReferenceGeneration, ReferenceError> {
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

    /// Marks a successfully submitted mutation as `Taken`.
    ///
    /// The caller must first obtain `generation` from [`Self::next_generation`] under this same guard. This transition
    /// makes the previous value unavailable while the backend reconstructs its replacement. The caller must retain the
    /// guard on the submitting thread until [`Self::install_pending_unchecked`] installs that replacement or
    /// [`Self::poison`] records a failure; dropping it first poisons the reference automatically.
    ///
    /// # Parameters
    ///
    ///   - `generation`: Generation returned by the immediately preceding [`Self::next_generation`] call.
    pub fn begin_submitted_mutation(&mut self, generation: ReferenceGeneration) {
        debug_assert_eq!(self.next_generation(), Ok(generation));
        *self.state = ReferenceState::Taken { generation };
    }

    /// Validates a pending replacement without changing the reference state.
    ///
    /// The replacement must have been prepared for this allocation, and `generation` must match its current `Taken`
    /// state. Multi-reference backends use this fallible phase for every replacement before committing any replacement
    /// with [`Self::install_pending_unchecked`].
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReplacementHolderMismatch`] for another allocation,
    /// [`ReferenceError::TransactionInProgress`] unless the reference is `Taken`, or
    /// [`ReferenceError::StaleGeneration`] when `generation` does not match the claimed generation.
    pub fn validate_pending_install(
        &self,
        generation: ReferenceGeneration,
        replacement: &ReferenceReplacement<V>,
    ) -> Result<(), ReferenceError> {
        self.accepts(replacement)?;
        let ReferenceState::Taken { generation: current } = &*self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        if *current != generation {
            return Err(ReferenceError::StaleGeneration);
        }
        Ok(())
    }

    /// Installs a pending replacement after [`Self::validate_pending_install`] succeeded under this same guard.
    ///
    /// This moves the reference from `Taken` to `Pending`. `completion` must cover both the submitted execution and its
    /// predecessor dependency; it will eventually make the replacement `Ready` or poison the reference. Every
    /// replacement in a multi-reference mutation must be validated before the first replacement is installed. This
    /// method then performs only moves and state assignment. It must run through the same guard, on the same thread, as
    /// [`Self::begin_submitted_mutation`].
    pub fn install_pending_unchecked(
        &mut self,
        generation: ReferenceGeneration,
        completion: ReferenceCompletion,
        replacement: ReferenceReplacement<V>,
    ) {
        debug_assert!(self.validate_pending_install(generation, &replacement).is_ok());
        *self.state =
            ReferenceState::Pending { value: replacement.value, generation, completion, read_leases: Vec::new() };
    }

    /// Reconciles one completion in tests, but only if `generation` is still current.
    ///
    /// This is the test entry point for the lazy completion path used by ordinary reference access.
    #[cfg(test)]
    fn complete(&mut self, generation: ReferenceGeneration, result: Result<(), Arc<str>>) -> bool {
        Reference::<V>::apply_pending_completion(&mut self.state, generation, result)
    }

    /// Extracts the current `Ready` value for a synchronous or potentially donating backend invocation.
    ///
    /// This converts the value to the handle's type identities, advances its generation, and transitions the reference
    /// to `Taken`. It prunes completed read leases first. Any remaining lease rejects extraction because a donating
    /// execution could otherwise mutate storage that a submitted reader still uses. A `Pending` value must be chained
    /// through the asynchronous submission protocol instead of being extracted here. Before releasing the guard, the
    /// caller must restore the extracted value with [`Self::install`] or invalidate it with [`Self::poison`].
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
        let local = self.snapshot()?;
        let generation = match &*self.state {
            ReferenceState::Ready { generation, .. } => generation.next().ok_or(ReferenceError::GenerationExhausted)?,
            _ => return Err(ReferenceError::TransactionInProgress),
        };
        *self.state = ReferenceState::Taken { generation };
        Ok(local)
    }

    /// Validates `value`, converts it to the allocation's type identities, and binds it to this allocation.
    ///
    /// This method does not change the reference state. The returned [`ReferenceReplacement`] can be installed only
    /// through a guard for this same allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReferentTypeMismatch`] if `value` has the wrong type or
    /// [`ReferenceError::ValueReconstruction`] if identity conversion fails.
    pub fn prepare_replacement(&self, value: V) -> Result<ReferenceReplacement<V>, ReferenceError> {
        let stored = self.reference.prepare_replacement(value)?;
        Ok(ReferenceReplacement { holder: Arc::downgrade(&self.reference.handle.holder), value: stored })
    }

    /// Verifies that `replacement` was prepared for this reference allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReplacementHolderMismatch`] for a replacement prepared through another allocation.
    pub(crate) fn accepts(&self, replacement: &ReferenceReplacement<V>) -> Result<(), ReferenceError> {
        if std::ptr::eq(replacement.holder.as_ptr(), Arc::as_ptr(&self.reference.handle.holder)) {
            Ok(())
        } else {
            Err(ReferenceError::ReplacementHolderMismatch)
        }
    }

    /// Installs a prepared replacement into the current `Taken` transaction and returns the reference to `Ready`.
    ///
    /// This is the synchronous counterpart to [`Self::take`], and the replacement inherits the generation claimed by
    /// the transaction. A submitted asynchronous mutation must instead use [`Self::validate_pending_install`] followed
    /// by [`Self::install_pending_unchecked`] so its completion remains attached to the replacement.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReplacementHolderMismatch`] if `replacement` belongs to another allocation or
    /// [`ReferenceError::TransactionInProgress`] unless this guard owns a `Taken` value.
    pub fn install(&mut self, replacement: ReferenceReplacement<V>) -> Result<(), ReferenceError> {
        self.accepts(&replacement)?;
        let ReferenceState::Taken { generation } = *self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        *self.state = ReferenceState::Ready { value: replacement.value, generation, read_leases: Vec::new() };
        Ok(())
    }

    /// Marks the current `Taken` transaction as terminally failed.
    ///
    /// Later access reports `reason` through [`ReferenceError::ExecutionPoisoned`]. This method is infallible so a
    /// cleanup path cannot replace the original error with another failure. It does nothing unless the guard currently
    /// owns a `Taken` transaction.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state = ReferenceState::Poisoned(reason.into());
        }
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    #[inline]
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            // The extracted value may already have been donated to irreversible backend work, so restoring it would
            // be unsound. Make the missing replacement a permanent, explicit failure instead.
            *self.state = ReferenceState::Poisoned("stateful transaction ended without restoring state".into());
        }
    }
}

/// Type-checked replacement prepared for one reference allocation.
///
/// [`ReferenceGuard::prepare_replacement`] converts a handle-local value to the allocation's canonical type identities
/// and records which allocation it belongs to. Installation verifies that identity before moving the value into shared
/// state, preventing a multi-reference transaction from exchanging two otherwise type-compatible replacements.
pub struct ReferenceReplacement<V: Value> {
    /// Weak pointer identifying the allocation for which this replacement was prepared.
    ///
    /// Keeping the weak control block alive prevents its address from being reused while this replacement exists, so
    /// pointer equality remains a stable allocation-identity check even after the last strong handle is dropped.
    holder: Weak<ReferenceHolder<V>>,

    /// Value using the reference allocation's canonical type identities.
    value: V,
}

/// Cloneable backend-neutral token for work that reads or replaces a reference value.
///
/// A token has one immutable terminal result: success or a backend-owned failure reason. It can be polled with
/// [`Self::is_ready`], waited on with [`Self::r#await`], and combined in dependency order with [`Self::join`].
#[derive(Clone)]
pub struct ReferenceCompletion {
    /// One backend completion or a core-owned flattened join.
    storage: ReferenceCompletionStorage,
}

impl ReferenceCompletion {
    /// Wraps a backend completion in a cloneable type-erased token.
    pub fn new(backend: impl ReferenceCompletionBackend) -> Self {
        Self { storage: ReferenceCompletionStorage::Backend(Arc::new(backend)) }
    }

    /// Creates a token with an already-known terminal `result`.
    pub fn ready(result: Result<(), Arc<str>>) -> Self {
        Self::new(ReadyReferenceCompletion(result))
    }

    /// Returns a token that completes after every input token and reports the first failure in input order.
    ///
    /// Nested joins are flattened. Inputs that have already succeeded are discarded because their terminal result
    /// cannot change. Pending and failed inputs are retained in order. The join remains pending until every retained
    /// input is terminal, and waiting on it waits for every input even after observing a failure.
    pub fn join(completions: impl IntoIterator<Item = Self>) -> Self {
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

    /// Blocks until all represented work completes and returns its terminal result.
    #[inline]
    pub fn r#await(&self) -> Result<(), Arc<str>> {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.r#await(),
            ReferenceCompletionStorage::Joined(joined) => joined.r#await(),
        }
    }

    /// Returns `Ok(false)` while any represented work is pending, `Ok(true)` after success, or the terminal failure.
    #[inline]
    pub fn is_ready(&self) -> Result<bool, Arc<str>> {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.is_ready(),
            ReferenceCompletionStorage::Joined(joined) => joined.is_ready(),
        }
    }
}

impl Debug for ReferenceCompletion {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ReferenceCompletion").field("is_ready", &self.is_ready()).finish()
    }
}

/// Internal representation of one primitive completion or a flattened join.
///
/// Keeping this enum private ensures joined tokens remain flat and preserve their original input order.
#[derive(Clone)]
enum ReferenceCompletionStorage {
    /// One backend completion.
    Backend(Arc<dyn ReferenceCompletionBackend>),

    /// Ordered primitive completions flattened from one or more joins.
    Joined(Arc<JoinedReferenceCompletion>),
}

/// Backend implementation stored behind a type-erased [`ReferenceCompletion`].
///
/// [`Self::is_ready`] may return `Ok(false)` before completion. Once either method observes success or failure, that
/// terminal result must never change, and both methods must agree on it. [`ReferenceCompletion::join`] relies on this
/// contract when it discards completed successes.
pub trait ReferenceCompletionBackend: Send + Sync + 'static {
    /// Blocks until the represented work completes and returns its terminal result.
    fn r#await(&self) -> Result<(), Arc<str>>;

    /// Checks without blocking, returning `Ok(false)` while pending, `Ok(true)` after success, or the terminal failure.
    fn is_ready(&self) -> Result<bool, Arc<str>>;
}

/// Completion backend with an immediately available terminal result.
struct ReadyReferenceCompletion(Result<(), Arc<str>>);

impl ReferenceCompletionBackend for ReadyReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        self.0.clone()
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        self.0.clone().map(|_| true)
    }
}

/// Flattened ordered join created by [`ReferenceCompletion::join`].
struct JoinedReferenceCompletion {
    /// Flat, input-ordered primitive completions.
    completions: Vec<ReferenceCompletion>,
}

impl ReferenceCompletionBackend for JoinedReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        // A joined token represents completion of every member, so a failure does not permit an early return. Retain
        // the first failure while continuing to wait for the remaining dependencies.
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
        // A known failure is not yet the join's terminal result while a later member remains pending. Once every member
        // is terminal, report the first failure in the original order.
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Condvar;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrValue, ArrayReference, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };
    use crate::captures::CaptureReference;
    use crate::operations::Add;

    use super::*;

    fn reference_new<V: Value>(value: V) -> Reference<V> {
        Reference::new(value).unwrap()
    }

    // Completion backend that lets tests observe when a waiter blocks and resolve it explicitly.
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
            let (state, awaiting) = &*self.state;
            let mut state = state.lock().unwrap();
            while !state.awaiting {
                state = awaiting.wait(state).unwrap();
            }
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
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            state.awaiting = true;
            ready.notify_all();
            while state.result.is_none() {
                state = ready.wait(state).unwrap();
            }
            state.result.clone().unwrap()
        }

        fn is_ready(&self) -> Result<bool, Arc<str>> {
            let state = self.state.0.lock().unwrap();
            state.result.clone().map_or(Ok(false), |result| result.map(|_| true))
        }
    }

    #[test]
    fn test_reference_completion_join_preserves_ordered_failure() {
        let completion = ReferenceCompletion::join([
            ReferenceCompletion::ready(Ok(())),
            ReferenceCompletion::ready(Err("first failure".into())),
            ReferenceCompletion::ready(Err("second failure".into())),
        ]);
        assert_eq!(completion.is_ready(), Err(Arc::<str>::from("first failure")));
        assert_eq!(completion.r#await(), Err(Arc::<str>::from("first failure")));
    }

    #[test]
    fn test_reference_completion_flat_join_waits_for_all_and_preserves_input_failure_order() {
        let first = ControlledCompletion::new();
        let second = ControlledCompletion::new();
        let third = ControlledCompletion::new();
        let joined = ReferenceCompletion::join([
            ReferenceCompletion::join([
                ReferenceCompletion::new(first.clone()),
                ReferenceCompletion::new(second.clone()),
            ]),
            ReferenceCompletion::new(third.clone()),
        ]);

        // Resolve later failures first. The join must remain pending until every member is terminal so input order,
        // rather than completion order, determines the reported failure.
        third.complete(Err("third failure".into()));
        second.complete(Err("second failure".into()));
        assert_eq!(joined.is_ready(), Ok(false));
        let waiting = joined.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || sender.send(waiting.r#await()).unwrap());
        first.wait_until_awaited();
        assert!(receiver.try_recv().is_err());
        first.complete(Err("first failure".into()));
        assert_eq!(receiver.recv().unwrap(), Err(Arc::<str>::from("first failure")));
        waiter.join().unwrap();
        assert_eq!(joined.is_ready(), Err(Arc::<str>::from("first failure")));
    }

    #[test]
    fn test_reference_completion_rejoin_releases_successes_and_preserves_dependencies_and_failures() {
        let succeeded = ControlledCompletion::new();
        let failed = ControlledCompletion::new();
        let pending = ControlledCompletion::new();
        let succeeded_state = Arc::downgrade(&succeeded.state);
        let failed_state = Arc::downgrade(&failed.state);
        let pending_state = Arc::downgrade(&pending.state);
        let nested = ReferenceCompletion::join([
            ReferenceCompletion::new(succeeded.clone()),
            ReferenceCompletion::new(failed.clone()),
        ]);
        succeeded.complete(Ok(()));
        failed.complete(Err("retained failure".into()));
        drop(succeeded);
        drop(failed);

        let joined = ReferenceCompletion::join([
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
        let rejoined = ReferenceCompletion::join([joined]);
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
            cumulative = ReferenceCompletion::join([cumulative, ReferenceCompletion::new(current.clone())]);
            assert!(completed_states.iter().all(|state: &Weak<_>| state.upgrade().is_none()));

            current.complete(Ok(()));
            drop(current);
            assert!(current_state.upgrade().is_some());
            completed_states.push(current_state);
        }

        cumulative = ReferenceCompletion::join([cumulative]);
        assert!(completed_states.iter().all(|state| state.upgrade().is_none()));
        assert_eq!(cumulative.r#await(), Ok(()));
    }

    #[test]
    fn test_reference_generation_advances_only_after_committed_mutations() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let initial_generation = reference.lock().unwrap().current_generation().unwrap();

        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(initial_generation));

        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
        let swapped_generation = reference.lock().unwrap().current_generation().unwrap();
        assert!(swapped_generation > initial_generation);

        assert_eq!(reference.write(Array::scalar(3.0_f32)), Ok(()));
        let written_generation = reference.lock().unwrap().current_generation().unwrap();
        assert!(written_generation > swapped_generation);

        let rejected = ProgramError::InvalidArgument { message: "rejected update".to_string() };
        assert_eq!(reference.update::<(), _>(|_| Err(rejected.clone())), Err(rejected));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(written_generation));

        assert_eq!(reference.update(|_| Ok((Array::scalar(4.0_f32), "updated"))), Ok("updated"));
        let updated_generation = reference.lock().unwrap().current_generation().unwrap();
        assert!(updated_generation > written_generation);
    }

    #[test]
    fn test_reference_write_preserves_state_when_the_generation_is_exhausted() {
        let reference = reference_new(Array::scalar(1.0_f32));
        {
            let mut state = reference.handle.holder.state.lock().unwrap();
            let ReferenceState::Ready { generation, .. } = &mut *state else {
                unreachable!("a newly allocated reference is ready")
            };
            *generation = ReferenceGeneration(u64::MAX);
        }

        assert_eq!(reference.write(Array::scalar(2.0_f32)), Err(ReferenceError::GenerationExhausted));
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(ReferenceGeneration(u64::MAX)));
    }

    #[test]
    fn test_concurrent_reference_writes_serialize_on_one_holder() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let initial_generation = reference.lock().unwrap().current_generation().unwrap();
        let first = reference.clone();
        let second = reference.clone();
        let first = std::thread::spawn(move || first.write(Array::scalar(2.0_f32)));
        let second = std::thread::spawn(move || second.write(Array::scalar(3.0_f32)));

        assert_eq!(first.join().unwrap(), Ok(()));
        assert_eq!(second.join().unwrap(), Ok(()));
        assert!(
            matches!(reference.read(), Ok(value) if value == Array::scalar(2.0_f32) || value == Array::scalar(3.0_f32))
        );
        assert_eq!(reference.lock().unwrap().current_generation().unwrap().0, initial_generation.0 + 2);
    }

    #[test]
    fn test_reference_read_reports_a_poisoned_holder() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let holder = Arc::clone(&reference.handle.holder);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = holder.state.lock().unwrap();
                panic!("poison reference holder");
            }))
            .is_err(),
        );
        assert_eq!(reference.read(), Err(ReferenceError::Poisoned));
        assert_eq!(reference.write(Array::scalar(2.0_f32)), Err(ReferenceError::Poisoned));
    }

    #[test]
    fn test_reference_guard_prepares_replacement_for_the_exact_holder() {
        let first = reference_new(Array::scalar(1.0_f32));
        let second = reference_new(Array::scalar(2.0_f32));
        let mut first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        assert_eq!(first_guard.take(), Ok(Array::scalar(1.0_f32)));
        let replacement = first_guard.prepare_replacement(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(second_guard.accepts(&replacement), Err(ReferenceError::ReplacementHolderMismatch));
        first_guard.install(replacement).unwrap();
        drop(first_guard);
        drop(second_guard);
        assert_eq!(first.read(), Ok(Array::scalar(3.0_f32)));
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_replacement_pins_holder_allocation_identity_after_last_handle_is_dropped() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let original_id = reference.id();
        let replacement = reference.lock().unwrap().prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        let original_holder = replacement.holder.clone();
        drop(reference);
        assert!(original_holder.upgrade().is_none());

        // The surviving weak control block prevents reuse of the retired allocation's address. Pointer equality
        // therefore remains a stable ownership check after every strong handle to the original allocation is gone.
        let new_reference = reference_new(Array::scalar(3.0_f32));
        assert_ne!(new_reference.id(), original_id);
        assert_eq!(new_reference.lock().unwrap().accepts(&replacement), Err(ReferenceError::ReplacementHolderMismatch),);
    }

    #[test]
    fn test_reference_pending_generations_ignore_stale_completion() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let first_value = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        let first = guard.submit_pending(ReferenceCompletion::ready(Ok(())), first_value).unwrap();

        let second_value = guard.prepare_replacement(Array::scalar(3.0_f32)).unwrap();
        let second = guard.submit_pending(ReferenceCompletion::ready(Ok(())), second_value).unwrap();
        assert!(!guard.complete(first, Err("stale failure".into())));
        assert!(guard.complete(second, Ok(())));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_guard_drop_poisons_an_unfinished_submitted_mutation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        {
            let mut guard = reference.lock().unwrap();
            let generation = guard.next_generation().unwrap();
            guard.begin_submitted_mutation(generation);
        }
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

    #[test]
    fn test_unwinding_an_unfinished_submitted_mutation_reports_execution_poisoning() {
        let reference = reference_new(Array::scalar(1.0_f32));
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut guard = reference.lock().unwrap();
                let generation = guard.next_generation().unwrap();
                guard.begin_submitted_mutation(generation);
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
    fn test_reference_read_lease_must_be_pruned_before_submitted_mutation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::ready(Ok(())));
        assert_eq!(guard.next_generation(), Err(ReferenceError::TransactionInProgress));
        assert!(guard.active_read_leases().is_empty());
        let generation = guard.next_generation().unwrap();
        guard.begin_submitted_mutation(generation);
        guard.poison("test cleanup");
    }

    #[test]
    fn test_reference_read_awaits_a_pending_completion_resolved_by_another_thread() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let value = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        guard.submit_pending(ReferenceCompletion::new(backend.clone()), value).unwrap();
        drop(guard);

        // A read cannot return before the installed value's cumulative completion resolves, so the reader's slot is
        // guaranteed to still be empty here regardless of how far the spawned thread has progressed.
        let observed = Arc::new(Mutex::new(None));
        let reader_reference = Arc::clone(&reference);
        let reader_observed = Arc::clone(&observed);
        let reader = std::thread::spawn(move || {
            let value = reader_reference.read();
            *reader_observed.lock().unwrap() = Some(value);
        });
        backend.wait_until_awaited();
        assert_eq!(*observed.lock().unwrap(), None);
        backend.complete(Ok(()));
        reader.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(2.0_f32))));

        // Applying the successful completion made the shared value ready and removed its pending dependency.
        assert!(reference.lock().unwrap().dependency().is_none());
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_write_awaits_a_pending_completion_resolved_by_another_thread() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let value = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        guard.submit_pending(ReferenceCompletion::new(backend.clone()), value).unwrap();
        drop(guard);

        let observed = Arc::new(Mutex::new(None));
        let writing_reference = Arc::clone(&reference);
        let writing_observed = Arc::clone(&observed);
        let writer = std::thread::spawn(move || {
            let result = writing_reference.write(Array::scalar(3.0_f32));
            *writing_observed.lock().unwrap() = Some(result);
        });
        backend.wait_until_awaited();
        assert_eq!(*observed.lock().unwrap(), None);
        backend.complete(Ok(()));
        writer.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(())));
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_read_reports_a_failed_pending_completion_as_execution_poisoned() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let value = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        guard.submit_pending(ReferenceCompletion::new(backend.clone()), value).unwrap();
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
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        let observed = Arc::new(Mutex::new(None));
        let writing_reference = Arc::clone(&reference);
        let writing_observed = Arc::clone(&observed);
        let writer = std::thread::spawn(move || {
            let result = writing_reference.write(Array::scalar(2.0_f32));
            *writing_observed.lock().unwrap() = Some(result);
        });
        lease.wait_until_awaited();
        assert_eq!(*observed.lock().unwrap(), None);
        lease.complete(Ok(()));
        writer.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(())));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_swap_waits_for_an_active_read_lease() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        // A leased snapshot pins the current value, so the replacement cannot be installed until the lease completes.
        let observed = Arc::new(Mutex::new(None));
        let swapping_reference = Arc::clone(&reference);
        let swapping_observed = Arc::clone(&observed);
        let swapper = std::thread::spawn(move || {
            let old = swapping_reference.swap(Array::scalar(2.0_f32));
            *swapping_observed.lock().unwrap() = Some(old);
        });
        lease.wait_until_awaited();
        assert_eq!(*observed.lock().unwrap(), None);
        lease.complete(Ok(()));
        swapper.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(1.0_f32))));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_freeze_waits_for_an_active_read_lease() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));
        drop(guard);

        // Consumption is a mutation of the alias family, so it waits for the same leases a replacement would.
        let observed = Arc::new(Mutex::new(None));
        let freezing_reference = Arc::clone(&reference);
        let freezing_observed = Arc::clone(&observed);
        let freezer = std::thread::spawn(move || {
            let value = freezing_reference.freeze();
            *freezing_observed.lock().unwrap() = Some(value);
        });
        lease.wait_until_awaited();
        assert_eq!(*observed.lock().unwrap(), None);
        lease.complete(Ok(()));
        freezer.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(1.0_f32))));
        assert_eq!(reference.read(), Err(ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_pending_install_rejects_a_stale_generation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let first_value = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        let first = guard.submit_pending(ReferenceCompletion::ready(Ok(())), first_value).unwrap();
        let second = guard.next_generation().unwrap();
        guard.begin_submitted_mutation(second);

        // Rejecting a late installation leaves the newer `Taken` transition intact, allowing that transaction to
        // install its own replacement.
        let value = guard.prepare_replacement(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(guard.validate_pending_install(first, &value), Err(ReferenceError::StaleGeneration));
        guard.install_pending_unchecked(second, ReferenceCompletion::ready(Ok(())), value);
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_take_rejects_an_active_read_lease() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));

        // Handing leased buffers to a donating execution would let the device mutate storage a submitted read-only
        // execution still observes, so extraction is rejected until the lease completes and is pruned.
        assert_eq!(guard.take(), Err(ReferenceError::TransactionInProgress));
        lease.complete(Ok(()));
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        let restored = guard.prepare_replacement(Array::scalar(4.0_f32)).unwrap();
        guard.install(restored).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_isolated_to_the_extracted_holder() {
        let first = reference_new(Array::scalar(1.0_f32));
        let second = reference_new(Array::scalar(2.0_f32));
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
    fn test_reference_guard_poison_leaves_an_idle_holder_untouched() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let generation = guard.current_generation().unwrap();

        // Poisoning is infallible so a cleanup path cannot replace the original backend error with a guard-state error.
        // A guard that does not own a `Taken` transaction has nothing to invalidate, so the reference remains ready at
        // its current generation.
        guard.poison("unrelated backend failure");
        assert_eq!(guard.current_generation(), Ok(generation));
        assert_eq!(guard.snapshot(), Ok(Array::scalar(1.0_f32)));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_leaves_a_pending_holder_untouched() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let completion = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let replacement = guard.prepare_replacement(Array::scalar(2.0_f32)).unwrap();
        let generation = guard.submit_pending(ReferenceCompletion::new(completion.clone()), replacement).unwrap();

        // Explicit poisoning applies only to an uninstalled `Taken` transaction. Once a replacement is installed,
        // its cumulative completion remains the sole authority for promoting or poisoning that pending generation.
        guard.poison("unrelated backend failure");
        assert_eq!(guard.current_generation(), Ok(generation));
        assert!(guard.dependency().is_some());
        drop(guard);
        completion.complete(Ok(()));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_read_lease_publication_releases_completed_leases() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let first = ControlledCompletion::new();
        let second = ControlledCompletion::new();
        let third = ControlledCompletion::new();
        first.complete(Ok(()));
        second.complete(Ok(()));

        // Lease publication is the only lifecycle update guaranteed for a read-only reference, so it must also release
        // completed leases instead of retaining their backend resources indefinitely. The strong counts make those
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
        assert_eq!(guard.next_generation(), Err(ReferenceError::TransactionInProgress));
        third.complete(Ok(()));
        assert!(guard.active_read_leases().is_empty());
        assert_eq!(Arc::strong_count(&third.state), 1);
        assert!(guard.next_generation().is_ok());
    }

    #[test]
    fn test_dropping_reference_guard_poisons_extracted_holder() {
        let reference = reference_new(Array::scalar(1.0_f32));
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
    fn test_reference_clones_alias_one_holder_and_reads_snapshots() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = reference_new(initial.clone());
        let alias = reference.clone();
        let distinct = reference_new(initial);
        assert_eq!(reference, alias);
        assert_ne!(reference, distinct);
        assert_eq!(reference.id(), alias.id());
        assert_eq!(reference.read().unwrap(), Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(reference.r#type(), alias.r#type());
        assert_eq!(reference.r#type(), distinct.r#type());
        // `Display` is deterministic and type-based; including the allocation address would make diagnostics and
        // program renderings nondeterministic. Runtime identity remains available through `Debug`.
        assert_eq!(reference.to_string(), "ref<f32[2]>");
        assert_eq!(reference.to_string(), distinct.to_string());
        assert_eq!(
            format!("{reference:?}"),
            format!("Reference {{ id: {:?}, type: {:?} }}", reference.id(), reference.r#type()),
        );

        let mut reference_type_hasher = DefaultHasher::new();
        reference.r#type().hash(&mut reference_type_hasher);
        let mut distinct_type_hasher = DefaultHasher::new();
        distinct.r#type().hash(&mut distinct_type_hasher);
        assert_eq!(reference_type_hasher.finish(), distinct_type_hasher.finish());

        let references = HashMap::from([(reference.clone(), "root")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&distinct), None);
    }

    #[test]
    fn test_reference_is_send_and_sync() {
        // Exact clones share immutable handle-local type and identity metadata across threads, so the production array
        // reference type must remain `Send + Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_reference_handle_layout_and_clone_sharing() {
        // `Reference` stores one `Arc`; exact clones share its immutable type and identity metadata, so cloning requires
        // only a reference-count increment.
        assert_eq!(size_of::<Reference<Array>>(), size_of::<usize>());

        let reference = reference_new(Array::scalar(1.0_f32));
        let clone = reference.clone();
        assert!(Arc::ptr_eq(&clone.handle, &reference.handle));
    }

    #[test]
    fn test_reference_allocation_rejects_an_immediate_reference_referent() {
        let nested = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        assert!(matches!(
            Reference::new(nested),
            Err(ReferenceError::NestedReferent { referent_type }) if referent_type == "ref<f32[]>"
        ));
    }

    #[test]
    fn test_reference_freeze_invalidates_the_complete_alias_family() {
        let reference = reference_new(Array::vector(vec![1.0_f32, 2.0]));
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));

        assert_eq!(alias.read(), Err(ReferenceError::Frozen));
        assert_eq!(alias.write(Array::vector(vec![3.0_f32, 4.0])), Err(ReferenceError::Frozen));
        assert_eq!(alias.swap(Array::vector(vec![3.0_f32, 4.0])), Err(ReferenceError::Frozen));
        assert_eq!(reference.freeze(), Err(ReferenceError::Frozen));

        // Once the alias family is frozen, state validation must reject the update before invoking value-family code.
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
    fn test_reference_rejected_replacements_and_updates_leave_the_holder_unchanged() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = reference_new(initial.clone());

        assert_eq!(
            reference.swap(Array::vector(vec![3.0_f32, 4.0, 5.0])),
            Err(ReferenceError::ReferentTypeMismatch { expected: "f32[2]".to_string(), actual: "f32[3]".to_string() }),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        assert_eq!(
            reference.write(Array::vector(vec![3.0_f32, 4.0, 5.0])),
            Err(ReferenceError::ReferentTypeMismatch { expected: "f32[2]".to_string(), actual: "f32[3]".to_string() }),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        let update_error = ProgramError::InvalidArgument { message: "test update failed".to_string() };
        assert_eq!(reference.update::<(), _>(|_| Err(update_error.clone())), Err(update_error));
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.update(|_| Ok((Array::vector(vec![3.0_f32, 4.0, 5.0]), ()))).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[2]".to_string(),
                actual: "f32[3]".to_string(),
            }),
        );
        assert_eq!(reference.read(), Ok(initial));
    }

    #[test]
    fn test_reference_mutation_preserves_snapshots_and_independent_roots() {
        let initializer = Array::vector(vec![1.0_f32, 2.0]);
        let first = reference_new(initializer.clone());
        let second = reference_new(initializer.clone());
        let read_snapshot = first.read().unwrap();
        let replacement = Array::vector(vec![3.0_f32, 4.0]);
        let retained_replacement = replacement.clone();

        first.write(replacement).unwrap();
        let swapped_snapshot = first.swap(Array::vector(vec![7.0_f32, 8.0])).unwrap();
        first
            .update(|current| current.add(&Array::vector(vec![10.0_f32, 20.0])).map(|updated| (updated, ())))
            .unwrap();
        assert_eq!(second.read(), Ok(initializer.clone()));
        assert_eq!(second.swap(Array::vector(vec![5.0_f32, 6.0])), Ok(initializer.clone()));

        assert_eq!(initializer, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(read_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(swapped_snapshot, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(retained_replacement, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(first.read(), Ok(Array::vector(vec![17.0_f32, 28.0])));
        assert_eq!(second.read(), Ok(Array::vector(vec![5.0_f32, 6.0])));
    }

    #[test]
    fn test_identity_renamed_reference_preserves_location_equality_hashing_and_replacement_ownership() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let reference = reference_new(CaptureReference::new(0, source_type.clone()));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target).unwrap();
        let renamed = reference.rename_type_identities(&renaming).unwrap();

        // Renaming creates distinct handle-local type and identity metadata over the same allocation. Equality and
        // hashing follow allocation identity rather than alias-local metadata.
        assert!(!Arc::ptr_eq(&renamed.handle, &reference.handle));
        assert!(Arc::ptr_eq(&renamed.handle.holder, &reference.handle.holder));
        assert_eq!(renamed, reference);
        assert_ne!(renamed.r#type(), reference.r#type());
        let mut reference_hasher = DefaultHasher::new();
        reference.hash(&mut reference_hasher);
        let mut renamed_hasher = DefaultHasher::new();
        renamed.hash(&mut renamed_hasher);
        assert_eq!(reference_hasher.finish(), renamed_hasher.finish());
        assert_eq!(HashMap::from([(reference.clone(), "root")]).get(&renamed), Some(&"root"));

        let mut guard = renamed.lock().unwrap();
        assert_eq!(guard.take(), Ok(CaptureReference::new(0, target_type.clone())));
        let replacement = guard.prepare_replacement(CaptureReference::new(1, target_type)).unwrap();
        guard.install(replacement).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(CaptureReference::new(1, source_type)));
    }
}
