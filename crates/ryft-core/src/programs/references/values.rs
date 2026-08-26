// TODO(eaplatanios): Review from here onward.

//! Eager reference values and their holder-facing operations.
//!
//! [`Reference<V>`](Reference) is the generic cloneable root holder. Clones alias one synchronized holder, reads
//! return immutable snapshots, writes and swaps atomically preserve the exact declared referent type, and a consuming
//! freeze invalidates the complete alias family. Runtime identity belongs to the holder and never participates in
//! structural type equality, hashing, or retained-program specialization.
//!
//! A direct eager reference remains live until it is explicitly frozen or its last handle is dropped. Handle-local
//! type-identity mappings reconstruct values bidirectionally at the shared-holder boundary. Array indexing and slicing
//! live in the array-owned [`ArrayReference`](crate::arrays::ArrayReference) wrapper.
//!
//! References are second-class program values: they may appear as instruction intermediates, inputs, or captures, but
//! never as public program outputs or in ordinary numeric use. Staged lifetime and second-class boundary checks live
//! in the reference operations and discharge transform; this module owns only the eager value surface.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;

use super::runtime::{
    ReferenceCompletionResult, ReferenceError, ReferenceGeneration, ReferenceGuard, ReferenceHolder, ReferenceId,
    ReferenceState,
};
use super::types::ReferenceType;

/// Cloneable holder for a referenced [`Value`].
///
/// Cloning a reference aliases the same holder. Reading clones the current value. Reference referents are expected to
/// have immutable-value clone semantics so that later replacement or update cannot change an initializer, read
/// result, or swap result retained by the caller. Array IR admits only array referents, whose SSA/copy-on-write
/// semantics satisfy this requirement; resource handles such as references are not valid referents there.
///
/// A directly created eager handle has no program-owned scope: it remains valid until an explicit freeze or until the
/// last handle in its alias family is dropped. Statically validated local program roots that are never frozen are
/// implicitly discarded when their nonescaping interpretation environment is released.
///
/// Equality and hashing identify the mutable storage location, not this handle's structural type. Clones and
/// identity-renamed handles into the same alias family therefore compare equal and hash identically even when their
/// handle-local referent types use different identity vocabularies.
///
/// Handles are pointer-sized: cloning one is a single reference-count increment, because exact clones share one
/// immutable handle vocabulary. Identity renaming allocates a new vocabulary sharing the same holder. Because that
/// vocabulary is shared across clones, `Reference<V>` is `Send` and `Sync` only when the referent value and its type
/// and identity metadata are all safe to share across threads (`Send + Sync`).
pub struct Reference<V: Value> {
    /// Immutable handle vocabulary shared by exact clones, including the shared mutable holder. All runtime
    /// mutability lives behind the holder's state lock.
    pub(super) inner: Arc<ReferenceHandle<V>>,
}

/// Handle-local vocabulary shared by exact clones of one [`Reference`].
///
/// Every binding is fixed at construction and never reassigned: the `Reference` API exposes no way to mutate handle
/// vocabulary, derivation ([`Reference::rename_type_identities`]) constructs a new handle rather than modifying an
/// existing one, and all runtime mutability lives behind the holder's state mutex. Private code must preserve that
/// invariant — `Arc` alone does not prevent mutation through `Arc::get_mut`, and sharing this metadata between exact
/// clones relies on Ryft's semantic contract that structural [`Type`] metadata remains stable for a value's lifetime.
pub(super) struct ReferenceHandle<V: Value> {
    /// Shared mutable holder whose allocation defines this reference's runtime identity.
    pub(super) holder: Arc<ReferenceHolder<V>>,

    /// Handle-local structural referent type.
    r#type: ReferenceType<V::Type>,

    /// Identity mapping applied when a stored value crosses into this handle.
    root_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Inverse identity mapping applied before a handle-local value enters the shared holder.
    handle_to_root: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

impl<V: Value> Reference<V> {
    /// Creates a new independent reference initialized with `value`.
    pub fn new(value: V) -> Result<Self, ReferenceError> {
        let root_type = value.r#type().into_owned();
        if root_type.is_reference() {
            return Err(ReferenceError::NestedReferent { referent_type: root_type.to_string() });
        }
        let r#type = ReferenceType::new(root_type.clone());
        Ok(Self {
            inner: Arc::new(ReferenceHandle {
                holder: Arc::new(ReferenceHolder {
                    root_type,
                    state: Mutex::new(ReferenceState::Ready {
                        value,
                        generation: ReferenceGeneration::initial(),
                        read_leases: Vec::new(),
                    }),
                }),
                r#type,
                root_to_handle: TypeIdentityRenaming::new(),
                handle_to_root: TypeIdentityRenaming::new(),
            }),
        })
    }

    /// Returns this holder's process-local identity, which remains stable while any alias is alive.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        ReferenceId::from_address(Arc::as_ptr(&self.inner.holder) as usize)
    }

    /// Returns whether this handle exposes the holder's root type without a handle-local identity mapping.
    #[doc(hidden)]
    pub fn is_root_handle(&self) -> bool {
        self.inner.root_to_handle.is_identity() && self.inner.handle_to_root.is_identity()
    }

    /// Locks this holder for one synchronous stateful backend transaction.
    #[doc(hidden)]
    pub fn lock(&self) -> Result<ReferenceGuard<'_, V>, ReferenceError> {
        let mut state = self.inner.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
        Self::apply_reservation_abandonment(&mut state);
        Ok(ReferenceGuard { reference: self, state })
    }

    /// Waits until the short post-submission reservation window has installed a pending value.
    ///
    /// Multi-holder runtimes call this before taking any ordered guards so they never wait while retaining another
    /// holder lock needed by the installer.
    #[doc(hidden)]
    pub fn wait_until_accessible(&self) -> Result<(), ReferenceError> {
        loop {
            let mut state = self.inner.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
            Self::apply_reservation_abandonment(&mut state);
            let reservation = match &*state {
                ReferenceState::Reserved { reservation, .. } => Arc::clone(reservation),
                ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => return Ok(()),
                ReferenceState::Frozen => return Err(ReferenceError::Frozen),
                ReferenceState::ExecutionPoisoned(reason) => {
                    return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
                }
                // `Taken` exists only while a guard holds this mutex, so this arm is defensive.
                ReferenceState::Taken { .. } => return Err(ReferenceError::TransactionInProgress),
            };
            drop(state);
            reservation.wait_until_resolved();
        }
    }

    /// Returns a clone of the currently stored value, which is an immutable snapshot for a valid reference referent.
    pub fn read(&self) -> Result<V, ReferenceError> {
        let state = self.lock_ready(false)?;
        let ReferenceState::Ready { value, .. } = &*state else { unreachable!("lock_ready yields only ready states") };
        self.reconstruct_local(value)
    }

    /// Atomically replaces the stored value without reconstructing or returning the previous handle-local value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the live holder
    /// unchanged. Holder-state errors such as freezing, poisoning, or an active transaction take precedence over a
    /// replacement-type error, because the holder must first admit the mutation before its replacement is validated.
    pub fn write(&self, replacement: V) -> Result<(), ReferenceError> {
        let mut state = self.lock_ready(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready yields only ready states")
        };
        let stored_replacement = self.prepare_stored(replacement)?;
        Self::commit_ready(current, generation, stored_replacement)
    }

    /// Atomically replaces the stored value and returns the previous referent value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the live holder
    /// unchanged. Holder-state errors such as freezing, poisoning, or an active transaction take precedence over a
    /// replacement-type error, because the holder must first admit the mutation before its replacement is validated.
    pub fn swap(&self, replacement: V) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready yields only ready states")
        };
        let stored_replacement = self.prepare_stored(replacement)?;
        let old = self.reconstruct_local(current)?;
        Self::commit_ready(current, generation, stored_replacement)?;
        Ok(old)
    }

    /// Atomically computes and installs an updated value while retaining the old value on every failure.
    ///
    /// This crate-visible primitive keeps value-family-specific update logic (such as array addition) outside the
    /// generic holder while ensuring no other access can interleave between reading the old state and installing the
    /// new one.
    pub(crate) fn update_with(&self, update: impl FnOnce(&V) -> Result<V, ProgramError>) -> Result<(), ProgramError> {
        self.update_locked_with_result(|current| Ok((update(current)?, ())))
    }

    /// Atomically maps this handle's current value to a replacement and an operation result.
    ///
    /// Both handle-local reconstruction directions complete before the shared state is committed, so every failure
    /// leaves the live holder unchanged. `update` runs while this holder's non-reentrant mutex is locked and therefore
    /// must not access this reference or any other handle in the same alias family.
    pub(crate) fn update_locked_with_result<R>(
        &self,
        update: impl FnOnce(&V) -> Result<(V, R), ProgramError>,
    ) -> Result<R, ProgramError> {
        let mut state = self.lock_ready(true).map_err(ProgramError::custom)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready yields only ready states")
        };
        let local = self.reconstruct_local(current).map_err(ProgramError::custom)?;
        let (updated, result) = update(&local)?;
        let stored = self.prepare_stored(updated).map_err(ProgramError::custom)?;
        Self::commit_ready(current, generation, stored).map_err(ProgramError::custom)?;
        Ok(result)
    }

    /// Consumes this reference's current value and invalidates every handle in its alias family.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready(true)?;
        let ReferenceState::Ready { value, .. } = &*state else { unreachable!("lock_ready yields only ready states") };
        let value = self.reconstruct_local(value)?;
        *state = ReferenceState::Frozen;
        Ok(value)
    }

    /// Returns a handle-local identity-renamed view of this same shared holder.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let current_type = self.inner.r#type.referent();
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
        let root_type = &self.inner.holder.root_type;
        let root_to_handle = Self::compose_renamings(
            &self.inner.root_to_handle,
            renaming,
            root_type.identities().map(|(_, identity)| identity),
        )?;
        let handle_to_root = Self::compose_renamings(
            &inverse_step,
            &self.inner.handle_to_root,
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
            inner: Arc::new(ReferenceHandle {
                holder: Arc::clone(&self.inner.holder),
                r#type: ReferenceType::new(renamed_type),
                root_to_handle,
                handle_to_root,
            }),
        })
    }

    /// Locks this holder after resolving a pending mutation and, when requested, every active read lease.
    fn lock_ready(&self, wait_for_read_leases: bool) -> Result<MutexGuard<'_, ReferenceState<V>>, ReferenceError> {
        loop {
            let mut state = self.inner.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
            Self::apply_reservation_abandonment(&mut state);
            let wait = match &mut *state {
                ReferenceState::Ready { read_leases, .. } if wait_for_read_leases => {
                    read_leases.retain(|lease| matches!(lease.is_ready(), Ok(false)));
                    (!read_leases.is_empty()).then(|| (None, read_leases.clone()))
                }
                ReferenceState::Ready { .. } => return Ok(state),
                ReferenceState::Pending { generation, reservation, .. } => {
                    Some((Some((*generation, Arc::clone(reservation))), Vec::new()))
                }
                ReferenceState::Frozen => return Err(ReferenceError::Frozen),
                ReferenceState::ExecutionPoisoned(reason) => {
                    return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
                }
                ReferenceState::Taken { .. } => {
                    return Err(ReferenceError::TransactionInProgress);
                }
                ReferenceState::Reserved { reservation, .. } => {
                    let reservation = Arc::clone(reservation);
                    drop(state);
                    reservation.wait_until_resolved();
                    continue;
                }
            };
            let Some((pending, read_leases)) = wait else {
                return Ok(state);
            };
            drop(state);
            if let Some((generation, reservation)) = pending {
                let result = reservation.wait_until_terminal();
                let mut state = self.inner.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
                Self::apply_reservation_abandonment(&mut state);
                if let Some(result) = result {
                    Self::apply_completion(&mut state, generation, result);
                }
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

    /// Applies `result` only when `generation` remains the holder's current generation.
    pub(super) fn apply_completion(
        state: &mut ReferenceState<V>,
        generation: ReferenceGeneration,
        result: ReferenceCompletionResult,
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
            Err(reason) => *state = ReferenceState::ExecutionPoisoned(reason),
        }
        true
    }

    /// Converts an abandoned current reservation into a terminal holder failure while its mutex is held.
    fn apply_reservation_abandonment(state: &mut ReferenceState<V>) -> bool {
        let reservation = match state {
            ReferenceState::Reserved { reservation, .. } | ReferenceState::Pending { reservation, .. } => reservation,
            _ => return false,
        };
        let Some(reason) = reservation.abandonment_reason() else { return false };
        *state = ReferenceState::ExecutionPoisoned(reason);
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

    /// Reconstructs one root-stored value in this handle's type-identity vocabulary.
    pub(super) fn reconstruct_local(&self, value: &V) -> Result<V, ReferenceError> {
        value
            .rename_type_identities(&self.inner.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
    }

    /// Validates and reconstructs one handle-local value for storage in the shared root holder.
    pub(super) fn prepare_stored(&self, value: V) -> Result<V, ReferenceError> {
        self.validate_referent_type(&value)?;
        let stored = value
            .rename_type_identities(&self.inner.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        self.validate_root_type(&stored)?;
        Ok(stored)
    }

    /// Commits one prepared replacement and advances the ready holder's generation.
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

    /// Validates that `value` preserves this holder's exact declared referent type.
    fn validate_referent_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == self.inner.r#type.referent() {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch {
            expected: self.inner.r#type.referent().to_string(),
            actual: actual.to_string(),
        })
    }

    /// Validates the exact value type stored behind every handle-local mapping.
    fn validate_root_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        let root_type = &self.inner.holder.root_type;
        if actual.as_ref() == root_type {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch { expected: root_type.to_string(), actual: actual.to_string() })
    }
}

// Exact clones share one immutable handle vocabulary, so cloning is a single reference-count increment.
impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<V: Value> Debug for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Reference")
            .field("id", &self.id())
            .field("type", &self.inner.r#type)
            .finish()
    }
}

// `Display` deliberately renders only the handle-local type: the Value rendering contract requires deterministic
// output (renderings back diagnostics, rendered-program tests, and the debug-assertions transform-cache determinism
// recheck), so the process-local holder address must not leak here. Runtime identity remains visible through `Debug`.
impl<V: Value> Display for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.inner.r#type, formatter)
    }
}

impl<V: Value> PartialEq for Reference<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner.holder, &other.inner.holder)
    }
}

impl<V: Value> Eq for Reference<V> {}

impl<V: Value> Hash for Reference<V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner.holder).hash(state);
    }
}

impl<V: Value> Parameter for Reference<V> {}

impl<V: Value> Typed for Reference<V> {
    type Type = ReferenceType<V::Type>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.inner.r#type)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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
        // Sharing the handle vocabulary between exact clones requires it to be safe for concurrent access, so the
        // production array reference family must remain `Send + Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Reference<Array>>();
    }

    #[test]
    fn test_reference_handle_layout_and_clone_sharing() {
        // Handles are pointer-sized and exact clones share one immutable handle vocabulary, so cloning is a single
        // reference-count increment.
        assert_eq!(size_of::<Reference<Array>>(), size_of::<usize>());

        let reference = reference_new(Array::scalar(1.0_f32));
        let clone = reference.clone();
        assert!(Arc::ptr_eq(&clone.inner, &reference.inner));
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
    fn test_identity_renamed_reference_preserves_location_equality_hashing_and_prepared_ownership() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let reference = reference_new(CaptureReference::new(0, source_type.clone()));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target).unwrap();
        let renamed = reference.rename_type_identities(&renaming).unwrap();

        // Renaming allocates a distinct handle vocabulary over the same shared holder, which is exactly the
        // handle-vocabulary/storage split: equality and hashing follow the holder, not the vocabulary.
        assert!(!Arc::ptr_eq(&renamed.inner, &reference.inner));
        assert!(Arc::ptr_eq(&renamed.inner.holder, &reference.inner.holder));
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
        let prepared = guard.prepare(CaptureReference::new(1, target_type)).unwrap();
        guard.install(prepared).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(CaptureReference::new(1, source_type)));
    }
}
