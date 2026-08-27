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

    /// Acquires this [`Reference`]'s shared lifecycle state for one backend-managed transaction. This is the
    /// backend-facing wrapper around the private raw state lock. It returns as soon as the state mutex is acquired and
    /// deliberately does not await a `Pending` value or active read leases. The returned [`ReferenceGuard`] exposes the
    /// pending dependency and the validated transitions a backend needs to compose, submit, and install stateful work
    /// while retaining exclusive access.
    ///
    /// Ordinary value access should use [`read`](Self::read), [`write`](Self::write), [`swap`](Self::swap), or
    /// [`freeze`](Self::freeze), which reconcile pending work before accessing the value. Backends that lock multiple
    /// references must acquire them in ascending [`ReferenceId`] order and retain that order until every submitted
    /// hidden replacement has been validated and installed. A submitted mutation remains represented by `Taken` while
    /// this guard is held, so dropping the guard before installation poisons the reference.
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
            drop(state);
            if let Some((generation, completion)) = pending {
                let result = completion.r#await();
                let mut state = self.lock_holder_state()?;
                Self::apply_completion(&mut state, generation, result);
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

    // TODO(eaplatanios): Review from here onward.

    /// Returns a clone of the currently stored value, which is an immutable snapshot for a valid reference referent.
    pub fn read(&self) -> Result<V, ReferenceError> {
        let state = self.lock_ready_state(false)?;
        let ReferenceState::Ready { value, .. } = &*state else {
            unreachable!("`lock_ready_state` yields only ready states")
        };
        self.reconstruct_local(value)
    }

    /// Atomically replaces the stored value without reconstructing or returning the previous handle-local value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the reference
    /// unchanged. Reference-state errors such as freezing, poisoning, or an active transaction take precedence over a
    /// replacement-type error, because the reference must first admit the mutation before its replacement is validated.
    pub fn write(&self, replacement: V) -> Result<(), ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready_state yields only ready states")
        };
        let stored_replacement = self.prepare_stored(replacement)?;
        Self::commit_ready(current, generation, stored_replacement)
    }

    /// Atomically replaces the stored value and returns the previous referent value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the reference
    /// unchanged. Reference-state errors such as freezing, poisoning, or an active transaction take precedence over a
    /// replacement-type error, because the reference must first admit the mutation before its replacement is validated.
    pub fn swap(&self, replacement: V) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready_state yields only ready states")
        };
        let stored_replacement = self.prepare_stored(replacement)?;
        let old = self.reconstruct_local(current)?;
        Self::commit_ready(current, generation, stored_replacement)?;
        Ok(old)
    }

    /// Consumes this reference's current value and invalidates every handle in its alias family.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready_state(true)?;
        let ReferenceState::Ready { value, .. } = &*state else {
            unreachable!("lock_ready_state yields only ready states")
        };
        let value = self.reconstruct_local(value)?;
        *state = ReferenceState::Frozen;
        Ok(value)
    }

    /// Atomically computes and installs an updated value while retaining the old value on every failure.
    ///
    /// This crate-visible primitive keeps value-family-specific update logic (such as array addition) outside the
    /// generic reference state while ensuring no other access can interleave between reading the old state and
    /// installing the new one.
    pub(crate) fn update_with(&self, update: impl FnOnce(&V) -> Result<V, ProgramError>) -> Result<(), ProgramError> {
        self.update_locked_with_result(|current| Ok((update(current)?, ())))
    }

    /// Atomically maps this handle's current value to a replacement and an operation result.
    ///
    /// Both handle-local reconstruction directions complete before the shared state is committed, so every failure
    /// leaves the live reference unchanged. `update` runs while this reference's non-reentrant state mutex is locked
    /// and therefore must not access this reference or any other handle in the same alias family.
    pub(crate) fn update_locked_with_result<R>(
        &self,
        update: impl FnOnce(&V) -> Result<(V, R), ProgramError>,
    ) -> Result<R, ProgramError> {
        let mut state = self.lock_ready_state(true).map_err(ProgramError::custom)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready_state yields only ready states")
        };
        let local = self.reconstruct_local(current).map_err(ProgramError::custom)?;
        let (updated, result) = update(&local)?;
        let stored = self.prepare_stored(updated).map_err(ProgramError::custom)?;
        Self::commit_ready(current, generation, stored).map_err(ProgramError::custom)?;
        Ok(result)
    }

    /// Returns a handle-local identity-renamed view of this same reference allocation.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let current_type = self.handle.r#type.referent();
        let renamed_type = current_type.rename_identities(renaming)?;
        // A renaming that merges two of the referent's identities cannot be inverted for value reconstruction.
        // Detect the collision here and report it in the caller's direction; deriving the inverse below would
        // otherwise surface it backwards, as a target renamed to two sources.
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
        let inverse_step =
            V::Type::derive_identity_renaming(std::slice::from_ref(&renamed_type), std::slice::from_ref(current_type))?;
        let root_type = &self.handle.holder.root_type;
        let root_to_handle = Self::compose_renamings(
            &self.handle.root_to_handle,
            renaming,
            root_type.identities().map(|(_, identity)| identity),
        )?;
        let handle_to_root = Self::compose_renamings(
            &inverse_step,
            &self.handle.handle_to_root,
            renamed_type.identities().map(|(_, identity)| identity),
        )?;
        if root_type.rename_identities(&root_to_handle)? != renamed_type
            || renamed_type.rename_identities(&handle_to_root)? != *root_type
        {
            return Err(TypeError::invalid(
                "reference identity renaming must admit an exact bidirectional value reconstruction",
            ));
        }
        Ok(Self {
            handle: Arc::new(ReferenceHandle {
                holder: Arc::clone(&self.handle.holder),
                r#type: ReferenceType::new(renamed_type),
                root_to_handle,
                handle_to_root,
            }),
        })
    }

    /// Applies `result` only when `generation` remains the shared state's current generation.
    fn apply_completion(
        state: &mut ReferenceState<V>,
        generation: ReferenceGeneration,
        result: Result<(), Arc<str>>,
    ) -> bool {
        if !matches!(state, ReferenceState::Pending { generation: current, .. } if *current == generation) {
            return false;
        }
        match result {
            Ok(()) => {
                // The placeholder is unobservable: the state mutex is held and `*state` is rewritten immediately in
                // both directions below, so no other thread can see the transient `Frozen`.
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

    /// Composes two simultaneous identity mappings over the provided source identities.
    fn compose_renamings<'a>(
        first: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
        second: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
        sources: impl Iterator<Item = &'a <V::Type as Type>::Identity>,
    ) -> Result<TypeIdentityRenaming<<V::Type as Type>::Identity>, TypeError>
    where
        <V::Type as Type>::Identity: 'a,
    {
        let mut result = TypeIdentityRenaming::new();
        for source in sources {
            result.insert(source.clone(), second.rename(&first.rename(source)))?;
        }
        Ok(result)
    }

    /// Reconstructs one root-stored value in this handle's type-identity namespace.
    fn reconstruct_local(&self, value: &V) -> Result<V, ReferenceError> {
        value
            .rename_type_identities(&self.handle.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
    }

    /// Validates and reconstructs one handle-local value for storage in the shared root representation.
    fn prepare_stored(&self, value: V) -> Result<V, ReferenceError> {
        self.validate_referent_type(&value)?;
        let stored = value
            .rename_type_identities(&self.handle.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        self.validate_root_type(&stored)?;
        Ok(stored)
    }

    /// Commits one prepared replacement and advances the ready reference generation.
    fn commit_ready(
        current: &mut V,
        generation: &mut ReferenceGeneration,
        replacement: V,
    ) -> Result<(), ReferenceError> {
        let next_generation = generation.next().ok_or(ReferenceError::GenerationExhausted)?;
        *current = replacement;
        *generation = next_generation;
        Ok(())
    }

    /// Validates that `value` preserves this reference's exact declared referent type.
    fn validate_referent_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == self.handle.r#type.referent() {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch {
            expected: self.handle.r#type.referent().to_string(),
            actual: actual.to_string(),
        })
    }

    /// Validates the exact value type stored behind every handle-local mapping.
    fn validate_root_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        let root_type = &self.handle.holder.root_type;
        if actual.as_ref() == root_type {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch { expected: root_type.to_string(), actual: actual.to_string() })
    }
}

// Exact clones share one immutable handle and its type-identity mappings, so cloning is a single reference-count
// increment.
impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
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

// `Display` deliberately renders only the handle-local type: the Value rendering contract requires deterministic
// output (renderings back diagnostics, rendered-program tests, and the debug-assertions transform-cache determinism
// recheck), so the process-local holder address must not leak here. Runtime identity remains visible through `Debug`.
impl<V: Value> Display for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// Immutable handle-local type and identity mappings shared by exact clones of one [`Reference`].
///
/// Every binding is fixed at construction and never reassigned: the `Reference` API exposes no way to mutate handle
/// metadata, derivation ([`Reference::rename_type_identities`]) constructs a new handle rather than modifying an
/// existing one, and all runtime mutability lives behind the [`ReferenceHolder`]'s state mutex. Private code must
/// preserve that invariant — `Arc` alone does not prevent mutation through `Arc::get_mut`, and sharing this metadata
/// between exact clones relies on Ryft's semantic contract that structural [`Type`] metadata remains stable for a
/// value's lifetime.
struct ReferenceHandle<V: Value> {
    /// Shared [`ReferenceHolder`] whose allocation defines this reference's runtime identity.
    holder: Arc<ReferenceHolder<V>>,

    /// Handle-local structural referent type.
    r#type: ReferenceType<V::Type>,

    /// Identity mapping applied when a stored value crosses into this handle.
    root_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Inverse identity mapping applied before a handle-local value enters the shared [`ReferenceHolder`].
    handle_to_root: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

/// Storage shared by one reference alias family.
struct ReferenceHolder<V: Value> {
    /// Structural referent type of values stored in this [`ReferenceHolder`]. Every alias agrees on it, it is
    /// immutable for the holder's lifetime, and it is deliberately readable without the state lock so validation
    /// paths never have to acquire or order against the lifecycle mutex.
    root_type: V::Type,

    /// Lifecycle state owned by this [`ReferenceHolder`].
    state: Mutex<ReferenceState<V>>,
}

/// Lifecycle state shared by every handle in one reference alias family.
enum ReferenceState<V: Value> {
    /// Live reference containing its current immutable value snapshot.
    Ready {
        /// Current immutable value.
        value: V,

        /// Most recently submitted mutation generation.
        generation: ReferenceGeneration,

        /// Submitted read-only executions that may still observe `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// Live value produced by an execution whose cumulative completion is still pending.
    Pending {
        /// Current possibly-pending immutable value.
        value: V,

        /// Generation that installed `value`.
        generation: ReferenceGeneration,

        /// Cumulative completion chain for this generation.
        completion: ReferenceCompletion,

        /// Submitted read-only executions that may still observe `value`.
        read_leases: Vec<ReferenceCompletion>,
    },

    /// Value temporarily unavailable while a backend transaction holds the shared state mutex.
    Taken {
        /// Generation claimed by the transaction.
        generation: ReferenceGeneration,
    },

    /// Value that may have been consumed by an irreversible failed backend invocation.
    Poisoned(Arc<str>),

    /// Consumed reference whose value was returned by `freeze`.
    Frozen,
}

/// Exclusive state guard for one reference alias family, used by stateful compilation backends.
///
/// Synchronous backends may extract the current value, but must then either install a type-compatible replacement or
/// poison the reference before dropping the guard. Asynchronous backends acquire multiple guards in stable
/// [`ReferenceId`] order, validate every lease publication or generation transition first, and then use the matching
/// unchecked commit methods while those same guards remain held. After successful submission they transition every
/// mutated reference to `Taken`, reconstruct and validate every hidden replacement, install every replacement as
/// `Pending`, and release all guards before any potentially blocking public-output reconstruction. The install loop
/// contains only moves and state assignments after batch validation; ordinary errors occur before the first install.
/// This guarantees batch atomicity for ordinary `Result`-based failures, but not for a panic inside the commit loop.
/// Dropping a guard while it still owns a `Taken` transaction poisons that reference defensively. A panic while retained
/// guards protect `Ready` or `Pending` state—including read-only references and any already-installed mutation—instead
/// poisons their synchronization primitives, so later access reports [`ReferenceError::Poisoned`].
///
/// The same guard must be retained from submission through installation on one thread. [`ReferenceGuard`] is not
/// transferable between threads; a backend requiring cross-thread installation needs a different synchronization
/// protocol.
#[doc(hidden)]
pub struct ReferenceGuard<'a, V: Value> {
    /// Reference handle whose shared state and handle-local type mapping this guard protects.
    reference: &'a Reference<V>,

    /// Locked reference lifecycle state.
    state: MutexGuard<'a, ReferenceState<V>>,
}

impl<V: Value> ReferenceGuard<'_, V> {
    /// Returns the generation of the current ready or installed pending value.
    pub fn current_generation(&self) -> Result<ReferenceGeneration, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { generation, .. } => Ok(*generation),
            ReferenceState::Pending { generation, .. } => Ok(*generation),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Returns a handle-local immutable snapshot without extracting the shared reference state.
    pub fn snapshot(&self) -> Result<V, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { value, .. } | ReferenceState::Pending { value, .. } => {
                self.reference.reconstruct_local(value)
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Returns the cumulative dependency of the current snapshot, if it was produced by a pending mutation.
    pub fn dependency(&self) -> Option<ReferenceCompletion> {
        match &*self.state {
            ReferenceState::Pending { completion, .. } => Some(completion.clone()),
            _ => None,
        }
    }

    /// Returns outstanding read leases after pruning every lease whose execution has completed.
    pub fn active_read_leases(&mut self) -> Vec<ReferenceCompletion> {
        let read_leases = match &mut *self.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => read_leases,
            _ => return Vec::new(),
        };
        read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
        read_leases.clone()
    }

    /// Validates read-lease publication without changing the shared reference state.
    pub fn validate_read_lease_publication(&self) -> Result<(), ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::Poisoned(reason) => Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }),
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Publishes a lease after [`Self::validate_read_lease_publication`] succeeded under this same guard.
    pub fn publish_read_lease_unchecked(&mut self, lease: ReferenceCompletion) {
        match &mut *self.state {
            ReferenceState::Ready { read_leases, .. } | ReferenceState::Pending { read_leases, .. } => {
                // Publication is the one point every read-only invocation passes through, so completed leases are
                // pruned here: a holder that is only ever read never takes the mutation paths that otherwise prune,
                // and an unpruned vector would grow (and pin each lease's backend completion) once per invocation
                // forever.
                read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
                read_leases.push(lease);
            }
            _ => unreachable!("read lease publication was validated under the same holder guard"),
        }
    }

    /// Begins and installs one submitted mutation. Test-only convenience combinator for the production
    /// validate-then-commit protocol.
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

    /// Validates a submitted mutation and returns its next generation without changing the shared reference state.
    ///
    /// Validation computes the generation that a successfully submitted execution will claim. The caller must retain
    /// this same guard through submission and pass the returned generation to [`Self::begin_submitted_mutation`] from
    /// the backend's successful-submission publication callback.
    ///
    /// Any published lease still recorded on the reference rejects the mutation, including one whose execution has
    /// already completed: this borrows the state immutably and therefore cannot prune. Backends must drain
    /// completed leases through [`Self::active_read_leases`] (releasing the guard and awaiting the returned
    /// completions when any remain) before validating a mutation, exactly as the multi-reference retry protocol
    /// does.
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

    /// Commits a successful execution handoff after [`Self::next_generation`] succeeded under this same guard.
    ///
    /// This moves a `Ready` or `Pending` reference to `Taken`: the logical mutation has committed and the old value is no
    /// longer observable, while the submitted execution's hidden replacement is reconstructed. The caller must retain
    /// this guard on the submitting thread until [`Self::install_pending_unchecked`] installs that replacement;
    /// dropping it first poisons the reference through the guard's defensive cleanup.
    pub fn begin_submitted_mutation(&mut self, generation: ReferenceGeneration) {
        debug_assert_eq!(self.next_generation(), Ok(generation));
        *self.state = ReferenceState::Taken { generation };
    }

    /// Validates one pending replacement installation without changing the shared reference state.
    ///
    /// The replacement must belong to this exact reference allocation, and `generation` must identify its current
    /// `Taken` transition. This method is the fallible first phase used to validate every member of a multi-reference
    /// batch before any member is committed through [`Self::install_pending_unchecked`].
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

    /// Commits a replacement after [`Self::validate_pending_install`] succeeded under this same guard.
    ///
    /// This moves the reference from `Taken` to `Pending` and attaches the cumulative completion that will eventually
    /// make the replacement `Ready` or poison the reference. After every replacement in a multi-reference mutation has
    /// been validated, this commit consists only of moves and state assignment and is infallible under those
    /// preconditions. It must run through the same guard, on the same thread, as [`Self::begin_submitted_mutation`].
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

    /// Applies a completion result only if `generation` is still current. Test-only entry into the lazy completion
    /// reconciliation that value accesses perform through `lock_ready_state`.
    #[cfg(test)]
    fn complete(&mut self, generation: ReferenceGeneration, result: Result<(), Arc<str>>) -> bool {
        Reference::<V>::apply_completion(&mut self.state, generation, result)
    }

    /// Extracts the handle-local current value for a potentially donating backend invocation.
    ///
    /// Extraction is rejected while any published read lease is still active: a leased snapshot pins the current
    /// value, and handing its buffers to a donating execution would let the device mutate storage a submitted
    /// read-only execution still observes. Completed leases are pruned rather than counted.
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

    /// Converts and validates a prospective replacement without changing the shared reference state.
    pub fn prepare_replacement(&self, value: V) -> Result<ReferenceReplacement<V>, ReferenceError> {
        let stored = self.reference.prepare_stored(value)?;
        Ok(ReferenceReplacement { holder: Arc::downgrade(&self.reference.handle.holder), value: stored })
    }

    /// Validates that `replacement` was prepared for this exact reference allocation.
    pub(crate) fn accepts(&self, replacement: &ReferenceReplacement<V>) -> Result<(), ReferenceError> {
        if std::ptr::eq(replacement.holder.as_ptr(), Arc::as_ptr(&self.reference.handle.holder)) {
            Ok(())
        } else {
            Err(ReferenceError::ReplacementHolderMismatch)
        }
    }

    /// Installs a synchronous replacement after [`Self::take`] extracted the previous value.
    ///
    /// Installation fails when `replacement` belongs to another reference allocation or this guard does not own an
    /// extracted value.
    /// A submitted asynchronous mutation must instead use [`Self::validate_pending_install`] followed by
    /// [`Self::install_pending_unchecked`] so its completion remains attached to the installed value.
    pub fn install(&mut self, replacement: ReferenceReplacement<V>) -> Result<(), ReferenceError> {
        self.accepts(&replacement)?;
        let ReferenceState::Taken { generation } = *self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        *self.state = ReferenceState::Ready { value: replacement.value, generation, read_leases: Vec::new() };
        Ok(())
    }

    /// Invalidates a taken value after an irreversible backend failure, recording `reason` as the cause every later
    /// reference access reports. Poisoning is infallible so that failure paths can never trade the original backend
    /// error for a guard-state error; a guard that does not own a `Taken` transaction is deliberately left untouched.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state = ReferenceState::Poisoned(reason.into());
        }
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state = ReferenceState::Poisoned("stateful transaction ended without restoring state".into());
        }
    }
}

/// Validated replacement bound to one reference allocation.
///
/// [`ReferenceGuard::prepare_replacement`] converts a handle-local value into the root's type-identity namespace,
/// validates its exact referent type, and records the reference allocation for which it was prepared. Installation
/// consumes the replacement only after verifying that allocation, preventing replacements prepared for different
/// references from being exchanged during a multi-reference transaction.
#[doc(hidden)]
pub struct ReferenceReplacement<V: Value> {
    /// Weak identity of the reference allocation for which this replacement was prepared.
    ///
    /// Retaining the allocation's weak control block prevents its address from being recycled while this replacement
    /// can still present it as allocation-identity proof.
    holder: Weak<ReferenceHolder<V>>,

    /// Value represented in the reference allocation's root type-identity space.
    value: V,
}

/// Cloneable backend-neutral dependency and completion token used by external references.
#[doc(hidden)]
#[derive(Clone)]
pub struct ReferenceCompletion {
    /// Primitive backend or core-owned flattened join.
    storage: ReferenceCompletionStorage,
}

impl ReferenceCompletion {
    /// Erases `backend` behind a cloneable completion token.
    pub fn new(backend: impl ReferenceCompletionBackend) -> Self {
        Self { storage: ReferenceCompletionStorage::Backend(Arc::new(backend)) }
    }

    /// Creates an already-completed token.
    pub fn ready(result: Result<(), Arc<str>>) -> Self {
        Self::new(ReadyReferenceCompletion(result))
    }

    /// Joins `completions`, preserving the first failure in input order.
    ///
    /// Flattening discards members that have already succeeded, because completion backends expose an immutable
    /// terminal result. Pending members remain as dependencies, and failed members remain so their original ordering
    /// continues to determine which failure is reported.
    pub fn join(completions: impl IntoIterator<Item = Self>) -> Self {
        let mut flattened = Vec::new();
        for completion in completions {
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

    /// Blocks until completion and returns its terminal result.
    #[inline]
    pub fn r#await(&self) -> Result<(), Arc<str>> {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.r#await(),
            ReferenceCompletionStorage::Joined(joined) => joined.r#await(),
        }
    }

    /// Returns `false` while pending, `true` after successful completion, or the terminal failure.
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

/// Private storage prevents third-party backends from misrepresenting a primitive dependency as a join.
#[derive(Clone)]
enum ReferenceCompletionStorage {
    /// One backend completion.
    Backend(Arc<dyn ReferenceCompletionBackend>),

    /// Flat ordered primitive completions.
    Joined(Arc<JoinedReferenceCompletion>),
}

/// Backend implementation stored behind a type-erased [`ReferenceCompletion`].
///
/// Implementations must make every method observe the same immutable terminal result.
#[doc(hidden)]
pub trait ReferenceCompletionBackend: Send + Sync + 'static {
    /// Blocks until completion and returns its terminal result.
    fn r#await(&self) -> Result<(), Arc<str>>;

    /// Returns `false` while pending, `true` after successful completion, or the terminal failure.
    fn is_ready(&self) -> Result<bool, Arc<str>>;
}

/// Already-completed backend used by [`ReferenceCompletion::ready`].
struct ReadyReferenceCompletion(Result<(), Arc<str>>);

impl ReferenceCompletionBackend for ReadyReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        self.0.clone()
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        self.0.clone().map(|_| true)
    }
}

/// Composite backend used by [`ReferenceCompletion::join`].
struct JoinedReferenceCompletion {
    /// Ordered completions whose first failure is retained.
    completions: Vec<ReferenceCompletion>,
}

impl ReferenceCompletionBackend for JoinedReferenceCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
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
        assert_eq!(reference.update_with(|_| Err(rejected.clone())), Err(rejected));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(written_generation));

        reference.update_with(|_| Ok(Array::scalar(4.0_f32))).unwrap();
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

        // The surviving weak control block keeps the retired allocation address unavailable to a new holder. This
        // makes pointer equality a stable ownership proof even after every strong handle to the original is gone.
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

        // The successful completion was applied to the holder, so the value is now ready without any dependency.
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

        // A late installation for the superseded generation is rejected without changing holder state, so the current
        // submitted mutation still installs its own value.
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

        // Poisoning is infallible so that a failure path can never trade the original backend error for a guard-state
        // error. A guard that does not own a taken transaction has nothing to invalidate, so the
        // holder must stay ready at its current generation and remain readable afterwards.
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

        // Publication is the one point every read-only invocation passes through, so a holder that is only ever read
        // must release each completed lease there instead of pinning its backend completion forever. Each backend's
        // shared state is retained exactly while the holder still holds that lease, so the strong count observes the
        // release directly.
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

        // Only the one still-running lease remains recorded, and it alone blocks the next submitted mutation until
        // it completes and the holder prunes it.
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
        // `Display` is deterministic and type-based (the holder address would leak nondeterminism into diagnostics
        // and program renderings); runtime identity remains visible through `Debug`.
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
        // Exact clones share immutable handle-local type and identity metadata, so the production array reference
        // family requires that metadata to remain `Send + Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_reference_handle_layout_and_clone_sharing() {
        // Handles are pointer-sized and exact clones share one immutable handle with its type-identity mappings, so
        // cloning is a single reference-count increment.
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

        // A rejected update must not invoke value-family code after the shared holder has been consumed.
        let update_executed = Cell::new(false);
        let error = alias
            .update_with(|_| {
                update_executed.set(true);
                Ok(Array::vector(vec![3.0_f32, 4.0]))
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
        assert_eq!(reference.update_with(|_| Err(update_error.clone())), Err(update_error));
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.update_with(|_| Ok(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
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
        first.update_with(|current| current.add(&Array::vector(vec![10.0_f32, 20.0]))).unwrap();
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

        // Renaming allocates distinct handle-local type and identity metadata over the same shared holder. Equality and
        // hashing follow the holder rather than that metadata.
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
