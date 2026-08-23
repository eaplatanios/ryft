//! Runtime reference holders and backend completion coordination.
//!
//! References give programs mutable state with explicit runtime identity. This module owns the program-facing
//! surface of that feature:
//!
//! - [`ReferenceType`] is the structural [`Type`] of a reference. It carries only referent metadata: runtime
//!   identity belongs to [`Reference`] and never participates in structural equality, hashing, or retained-program
//!   specialization.
//! - [`Reference`] is the generic cloneable root holder. Clones alias one holder, reads return
//!   immutable value snapshots, replacement is atomic and preserves the exact declared referent type, handle-local
//!   type-identity mappings reconstruct values bidirectionally at the holder boundary, and a consuming freeze
//!   invalidates the complete alias family. Array indexing and slicing live in the array-owned
//!   [`ArrayReference`](crate::arrays::ArrayReference) wrapper rather than this generic layer.
//! - [`ReferenceError`] reports failed accesses to an eager holder.
//!
//! References are second-class program values: they may appear as instruction intermediates, inputs, or captures,
//! but never as public program outputs or in ordinary numeric use. Trace-time staging checks and discharge enforce
//! that root, lifetime, and second-class boundary contract, and eager program replay rejects an entry that expects
//! an external reference outright. The reference operations themselves —
//! allocation, snapshot read, replacement, ordered additive update, and consuming freeze — are independent operation
//! payloads defined in [`crate::programs::references::operations`] rather than one homogeneous reference operation
//! family, and binding-level sugar such as `write` is defined over `swap` instead of adding IR operations.
//!
//! Array reference views are statically validated and eliminated by canonical slice, reshape, and update-slice
//! discharge. Local references compose with the supported program transforms only after that discharge. External
//! holders use guarded state transactions around compiled execution. Views remain root-local across structured-region
//! boundaries: a region must receive the root handle and recreate any index or slice view inside its own body.
//!
//! # Local References and External State
//!
//! A reference allocated inside a program is an implementation detail when it is frozen or discarded without
//! escaping: discharge rewrites the complete lifetime into immutable array SSA, so the enclosing callable is
//! reference-free to its caller. A reference supplied as a public input or capture is different. Reads observe
//! caller-owned state and mutations install a new value into that holder, so the enclosing callable is externally
//! stateful and must use a backend's explicit stateful invocation surface.
//!
//! # Lifetime and Snapshot Rules
//!
//! References are second-class. They cannot be public outputs or constants, nested inside references or list values,
//! duplicated at region boundaries, or selected from multiple possible roots. [`Reference::freeze`] consumes a
//! locally owned root and invalidates every clone and view in its alias family; external roots cannot be frozen by a
//! program. A read or the old value returned by a swap is an immutable snapshot: later mutation of the holder never
//! changes a snapshot that has already been returned. Direct eager handles have no implicit program scope and remain
//! live until an explicit freeze or the last handle is dropped.
//!
//! ```
//! use ryft_core::{
//!     Array, ArrayIrValue, ArraySliceAxis, FreezeReference, NewReference, ReferenceRead, ReferenceSlice,
//!     ReferenceSwap,
//! };
//!
//! let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
//! let root = initial.new_reference()?;
//! let tail = root.reference_slice(&[ArraySliceAxis::new(1, 2, 1)])?;
//! let snapshot = tail.read()?;
//! tail.write(&ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0])))?;
//!
//! assert_eq!(snapshot, ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])));
//! assert_eq!(root.freeze()?, ArrayIrValue::Array(Array::vector(vec![1.0_f32, 4.0, 5.0])));
//! # Ok::<(), ryft_core::ProgramError>(())
//! ```
//!
//! # Backend State Protocol
//!
//! Stateful compilation backends coordinate external holders through generation-checked reservations, cumulative
//! completion dependencies, and read leases. Once a backend crosses its submission boundary, failure cannot restore
//! an unambiguously current value; the affected mutated holders become poisoned and every later access reports the
//! failure. Poisoning is terminal; recovery means constructing a new holder from independently trusted state. These
//! transaction types are a hidden, unstable backend service-provider interface. User code should use the stateful
//! call surface and await its [`ReferenceExecution`](crate::compilation::ReferenceExecution) instead of acquiring
//! holder guards directly.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use thiserror::Error;

use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;

use super::semantics::ReferenceType;

/// Error produced while accessing a [`Reference`]'s eager holder.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum ReferenceError {
    /// The reference and its complete alias family were invalidated by a consuming freeze.
    #[error("reference is frozen")]
    Frozen,

    /// The holder's synchronization primitive was poisoned by a panic during an earlier access.
    #[error("reference holder is poisoned")]
    Poisoned,

    /// The holder was invalidated after a stateful backend invocation crossed its irreversible execution boundary.
    #[error("reference state is poisoned: {reason}")]
    ExecutionPoisoned {
        /// Backend-owned reason the state can no longer be used safely.
        reason: String,
    },

    /// A guarded transaction attempted an operation incompatible with an extraction, reservation, or active lease.
    #[error("reference holder has a conflicting transaction or execution lease")]
    TransactionInProgress,

    /// The holder exhausted its monotonically increasing mutation generation space.
    #[error("reference holder mutation generation is exhausted")]
    GenerationExhausted,

    /// A pending value or completion targeted an older holder generation.
    #[error("reference completion targets a stale holder generation")]
    StaleGeneration,

    /// A prepared replacement belongs to a different holder transaction.
    #[error("prepared reference value belongs to a different holder")]
    TransactionHolderMismatch,

    /// A replacement or update result did not preserve the holder's exact declared referent type.
    #[error("reference value type `{actual}` must exactly match declared referent type `{expected}`")]
    ReferentTypeMismatch {
        /// Exact declared referent type.
        expected: String,

        /// Actual replacement or update-result type.
        actual: String,
    },

    /// A handle-local metadata mapping could not reconstruct a value crossing the shared-holder boundary.
    #[error("reference value reconstruction failed: {message}")]
    ValueReconstruction {
        /// Underlying value-family reconstruction diagnostic.
        message: String,
    },
}

/// Result reported by a backend-neutral reference completion token.
#[doc(hidden)]
pub type ReferenceCompletionResult = Result<(), Arc<str>>;

/// Callback invoked exactly once when a [`ReferenceCompletion`] finishes.
#[doc(hidden)]
pub type ReferenceCompletionCallback = Box<dyn FnOnce(ReferenceCompletionResult) + Send + 'static>;

/// Backend implementation stored behind a type-erased [`ReferenceCompletion`].
///
/// Implementations must make every method observe the same immutable terminal result. Callback registration may
/// invoke `callback` before returning when completion has already occurred.
#[doc(hidden)]
pub trait ReferenceCompletionBackend: Send + Sync + 'static {
    /// Blocks until completion and returns its terminal result.
    fn r#await(&self) -> ReferenceCompletionResult;

    /// Returns `false` while pending, `true` after successful completion, or the terminal failure.
    fn is_ready(&self) -> Result<bool, Arc<str>>;

    /// Registers a callback that is invoked exactly once with the terminal result.
    fn on_ready(&self, callback: ReferenceCompletionCallback);
}

/// Cloneable backend-neutral dependency and completion token used by external reference holders.
#[doc(hidden)]
#[derive(Clone)]
pub struct ReferenceCompletion {
    /// Primitive backend or core-owned flattened join.
    storage: ReferenceCompletionStorage,
}

/// Private storage prevents third-party backends from misrepresenting a primitive dependency as a join.
#[derive(Clone)]
enum ReferenceCompletionStorage {
    /// One backend completion.
    Backend(Arc<dyn ReferenceCompletionBackend>),

    /// Flat ordered primitive completions.
    Joined(Arc<JoinedReferenceCompletion>),
}

impl ReferenceCompletion {
    /// Erases `backend` behind a cloneable completion token.
    pub fn new(backend: impl ReferenceCompletionBackend) -> Self {
        Self { storage: ReferenceCompletionStorage::Backend(Arc::new(backend)) }
    }

    /// Creates an already-completed token.
    pub fn ready(result: ReferenceCompletionResult) -> Self {
        Self::new(ReadyReferenceCompletion(result))
    }

    /// Joins `completions`, preserving the first failure in input order.
    pub fn join(completions: impl IntoIterator<Item = Self>) -> Self {
        let mut flattened = Vec::new();
        for completion in completions {
            match &completion.storage {
                ReferenceCompletionStorage::Backend(_) => flattened.push(completion),
                ReferenceCompletionStorage::Joined(joined) => flattened.extend(joined.completions.iter().cloned()),
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
    pub fn r#await(&self) -> ReferenceCompletionResult {
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

    /// Registers a callback that is invoked exactly once with the terminal result.
    #[inline]
    pub fn on_ready(&self, callback: ReferenceCompletionCallback) {
        match &self.storage {
            ReferenceCompletionStorage::Backend(backend) => backend.on_ready(callback),
            ReferenceCompletionStorage::Joined(joined) => joined.on_ready(callback),
        }
    }
}

impl Debug for ReferenceCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ReferenceCompletion").field("is_ready", &self.is_ready()).finish()
    }
}

/// Already-completed backend used by [`ReferenceCompletion::ready`].
struct ReadyReferenceCompletion(ReferenceCompletionResult);

impl ReferenceCompletionBackend for ReadyReferenceCompletion {
    fn r#await(&self) -> ReferenceCompletionResult {
        self.0.clone()
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        self.0.clone().map(|_| true)
    }

    fn on_ready(&self, callback: ReferenceCompletionCallback) {
        callback(self.0.clone())
    }
}

/// Composite backend used by [`ReferenceCompletion::join`].
struct JoinedReferenceCompletion {
    /// Ordered completions whose first failure is retained.
    completions: Vec<ReferenceCompletion>,
}

impl ReferenceCompletionBackend for JoinedReferenceCompletion {
    fn r#await(&self) -> ReferenceCompletionResult {
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

    fn on_ready(&self, callback: ReferenceCompletionCallback) {
        struct State {
            remaining: usize,
            results: Vec<Option<ReferenceCompletionResult>>,
            callback: Option<ReferenceCompletionCallback>,
        }

        let state = Arc::new(Mutex::new(State {
            remaining: self.completions.len(),
            results: vec![None; self.completions.len()],
            callback: Some(callback),
        }));
        for (index, completion) in self.completions.iter().enumerate() {
            let state = Arc::clone(&state);
            completion.on_ready(Box::new(move |result| {
                let callback_and_result = {
                    let mut state = state.lock().expect("joined reference completion callback mutex poisoned");
                    if state.results[index].is_some() {
                        return;
                    }
                    state.results[index] = Some(result);
                    state.remaining -= 1;
                    (state.remaining == 0).then(|| {
                        let result = state
                            .results
                            .iter()
                            .filter_map(Option::as_ref)
                            .find_map(|result| result.as_ref().err().cloned())
                            .map_or(Ok(()), Err);
                        (state.callback.take().unwrap(), result)
                    })
                };
                if let Some((callback, result)) = callback_and_result {
                    callback(result);
                }
            }));
        }
    }
}

/// Opaque process-local identity that remains stable for the lifetime of one eager [`Reference`] holder.
///
/// The identity supports alias-identity checks and diagnostics inside one process. It carries no structural type
/// information, is never serialized into a program or compilation key, and may be reused after the last handle for
/// the original holder is dropped.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId(usize);

/// Monotonic generation of one holder mutation reservation.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceGeneration(u64);

impl ReferenceGeneration {
    /// Returns the successor generation, or [`None`] when the monotonic counter is exhausted.
    #[inline]
    fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

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
pub struct Reference<V: Value> {
    /// Shared mutable holder whose allocation defines this reference's runtime identity.
    holder: Arc<ReferenceHolder<V>>,

    /// Handle-local structural referent type.
    r#type: ReferenceType<V::Type>,

    /// Structural referent type of values stored in the shared holder.
    root_type: V::Type,

    /// Identity mapping applied when a stored value crosses into this handle.
    root_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Inverse identity mapping applied before a handle-local value enters the shared holder.
    handle_to_root: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

/// Synchronization state shared by one reference alias family.
struct ReferenceHolder<V: Value> {
    /// Holder lifecycle state.
    state: Mutex<ReferenceState<V>>,

    /// Notification used only for the short submitted-before-install reservation window.
    installed: Condvar,
}

/// Lifecycle state shared by every handle in one reference alias family.
enum ReferenceState<V: Value> {
    /// Live reference containing its current immutable value snapshot.
    Ready {
        /// Current immutable value.
        value: V,

        /// Most recently reserved mutation generation.
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

    /// Value temporarily extracted by a synchronous backend transaction holding this holder's mutex.
    Taken {
        /// Generation preserved across the synchronous transaction.
        generation: ReferenceGeneration,
    },

    /// Post-submission mutation reservation awaiting validated hidden final state construction.
    Reserved {
        /// Reserved generation.
        generation: ReferenceGeneration,

        /// Cumulative completion chain for the submitted mutation.
        completion: ReferenceCompletion,
    },

    /// Value that may have been consumed by an irreversible failed backend invocation.
    ExecutionPoisoned(Arc<str>),

    /// Consumed reference whose value was returned by `freeze`.
    Frozen,
}

impl<V: Value> Reference<V> {
    /// Creates a new independent reference initialized with `value`.
    pub fn new(value: V) -> Self {
        let root_type = value.r#type().into_owned();
        let r#type = ReferenceType::new(root_type.clone());
        Self {
            holder: Arc::new(ReferenceHolder {
                state: Mutex::new(ReferenceState::Ready {
                    value,
                    generation: ReferenceGeneration(0),
                    read_leases: Vec::new(),
                }),
                installed: Condvar::new(),
            }),
            r#type,
            root_type,
            root_to_handle: TypeIdentityRenaming::new(),
            handle_to_root: TypeIdentityRenaming::new(),
        }
    }

    /// Returns this holder's process-local identity, which remains stable while any alias is alive.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        ReferenceId(Arc::as_ptr(&self.holder) as usize)
    }

    /// Returns whether this handle exposes the holder's root type without a handle-local identity mapping.
    #[doc(hidden)]
    pub fn is_root_handle(&self) -> bool {
        self.root_to_handle.is_identity() && self.handle_to_root.is_identity()
    }

    /// Locks this holder for one synchronous stateful backend transaction.
    #[doc(hidden)]
    pub fn lock(&self) -> Result<ReferenceGuard<'_, V>, ReferenceError> {
        Ok(ReferenceGuard {
            id: self.id(),
            r#type: &self.r#type,
            root_type: &self.root_type,
            root_to_handle: &self.root_to_handle,
            handle_to_root: &self.handle_to_root,
            installed: &self.holder.installed,
            state: self.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?,
        })
    }

    /// Waits until the short post-submission reservation window has installed a pending value.
    ///
    /// Multi-holder runtimes call this before taking any ordered guards so they never wait while retaining another
    /// holder lock needed by the installer.
    #[doc(hidden)]
    pub fn wait_until_accessible(&self) -> Result<(), ReferenceError> {
        let mut state = self.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
        while matches!(*state, ReferenceState::Reserved { .. }) {
            state = self.holder.installed.wait(state).map_err(|_| ReferenceError::Poisoned)?;
        }
        match &*state {
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            // `Taken` exists only while a guard holds this mutex, and the wait loop above exits only once the state
            // is no longer `Reserved`; both arms are defensive.
            ReferenceState::Taken { .. } | ReferenceState::Reserved { .. } => {
                Err(ReferenceError::TransactionInProgress)
            }
        }
    }

    /// Returns a clone of the currently stored value, which is an immutable snapshot for a valid reference referent.
    pub fn read(&self) -> Result<V, ReferenceError> {
        let state = self.lock_ready(false)?;
        let ReferenceState::Ready { value, .. } = &*state else { unreachable!("lock_ready yields only ready states") };
        value
            .rename_type_identities(&self.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })
    }

    /// Atomically replaces the stored value and returns the previous referent value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the live holder
    /// unchanged.
    pub fn swap(&self, replacement: V) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready(true)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready yields only ready states")
        };
        self.validate_referent_type(&replacement)?;
        let stored_replacement = replacement
            .rename_type_identities(&self.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        self.validate_root_type(&stored_replacement)?;
        let old = current
            .rename_type_identities(&self.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        let next_generation = generation.next().ok_or(ReferenceError::GenerationExhausted)?;
        *current = stored_replacement;
        *generation = next_generation;
        Ok(old)
    }

    /// Atomically computes and installs an updated value while retaining the old value on every failure.
    ///
    /// This crate-visible primitive keeps value-family-specific update logic (such as array addition) outside the
    /// generic holder while ensuring no other access can interleave between reading the old state and installing the
    /// new one.
    pub(crate) fn update_with(&self, update: impl FnOnce(&V) -> Result<V, ProgramError>) -> Result<(), ProgramError> {
        self.update_with_result(|current| Ok((update(current)?, ())))
    }

    /// Atomically maps this handle's current value to a replacement and an operation result.
    ///
    /// Both handle-local reconstruction directions complete before the shared state is committed, so every failure
    /// leaves the live holder unchanged.
    pub(crate) fn update_with_result<R>(
        &self,
        update: impl FnOnce(&V) -> Result<(V, R), ProgramError>,
    ) -> Result<R, ProgramError> {
        let mut state = self.lock_ready(true).map_err(ProgramError::custom)?;
        let ReferenceState::Ready { value: current, generation, .. } = &mut *state else {
            unreachable!("lock_ready yields only ready states")
        };
        let local = current.rename_type_identities(&self.root_to_handle).map_err(|error| {
            ProgramError::custom(ReferenceError::ValueReconstruction { message: error.to_string() })
        })?;
        let (updated, result) = update(&local)?;
        self.validate_referent_type(&updated).map_err(ProgramError::custom)?;
        let stored = updated.rename_type_identities(&self.handle_to_root).map_err(|error| {
            ProgramError::custom(ReferenceError::ValueReconstruction { message: error.to_string() })
        })?;
        self.validate_root_type(&stored).map_err(ProgramError::custom)?;
        let next_generation =
            generation.next().ok_or_else(|| ProgramError::custom(ReferenceError::GenerationExhausted))?;
        *current = stored;
        *generation = next_generation;
        Ok(result)
    }

    /// Consumes this reference's current value and invalidates every handle in its alias family.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.lock_ready(true)?;
        let ReferenceState::Ready { value, .. } = &*state else { unreachable!("lock_ready yields only ready states") };
        let value = value
            .rename_type_identities(&self.root_to_handle)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        *state = ReferenceState::Frozen;
        Ok(value)
    }

    /// Returns a handle-local identity-renamed view of this same shared holder.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let current_type = self.r#type.referent();
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
        let root_to_handle = Self::compose_renamings(
            &self.root_to_handle,
            renaming,
            self.root_type.identities().map(|(_, identity)| identity),
        )?;
        let handle_to_root = Self::compose_renamings(
            &inverse_step,
            &self.handle_to_root,
            renamed_type.identities().map(|(_, identity)| identity),
        )?;
        if self.root_type.rename_identities(&root_to_handle)? != renamed_type
            || renamed_type.rename_identities(&handle_to_root)? != self.root_type
        {
            return Err(TypeError::invalid(
                "reference identity renaming must admit an exact bidirectional value reconstruction",
            ));
        }
        Ok(Self {
            holder: self.holder.clone(),
            r#type: ReferenceType::new(renamed_type),
            root_type: self.root_type.clone(),
            root_to_handle,
            handle_to_root,
        })
    }

    /// Fails if this holder has reached a terminal state, without reading or changing its value and without waiting.
    ///
    /// Pure state-free consumers such as view derivation call this: a holder with a pending completion or a
    /// submitted reservation is still live (its next value access resolves those), so this check must never block
    /// on device work the caller does not observe.
    pub(crate) fn validate_live(&self) -> Result<(), ReferenceError> {
        match &*self.holder.state.lock().map_err(|_| ReferenceError::Poisoned)? {
            ReferenceState::Ready { .. } | ReferenceState::Reserved { .. } | ReferenceState::Pending { .. } => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            // `Taken` exists only while a guard holds this mutex, so this arm is defensive and mirrors every other
            // non-guard access.
            ReferenceState::Taken { .. } => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Locks this holder after resolving a pending mutation and, when requested, every active read lease.
    fn lock_ready(&self, wait_for_read_leases: bool) -> Result<MutexGuard<'_, ReferenceState<V>>, ReferenceError> {
        loop {
            let mut state = self.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
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
                ReferenceState::ExecutionPoisoned(reason) => {
                    return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
                }
                ReferenceState::Taken { .. } => {
                    return Err(ReferenceError::TransactionInProgress);
                }
                ReferenceState::Reserved { .. } => {
                    state = self.holder.installed.wait(state).map_err(|_| ReferenceError::Poisoned)?;
                    drop(state);
                    continue;
                }
            };
            let Some((pending, read_leases)) = wait else {
                return Ok(state);
            };
            drop(state);
            if let Some((generation, completion)) = pending {
                let result = completion.r#await();
                let mut state = self.holder.state.lock().map_err(|_| ReferenceError::Poisoned)?;
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

    /// Applies `result` only when `generation` remains the holder's current generation.
    fn apply_completion(
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

    /// Validates that `value` preserves this holder's exact declared referent type.
    fn validate_referent_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == self.r#type.referent() {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch {
            expected: self.r#type.referent().to_string(),
            actual: actual.to_string(),
        })
    }

    /// Validates the exact value type stored behind every handle-local mapping.
    fn validate_root_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == &self.root_type {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch { expected: self.root_type.to_string(), actual: actual.to_string() })
    }
}

/// Exclusive holder guard used by stateful compilation backends.
///
/// Synchronous backends may extract the current value, but must then either install a type-compatible replacement or
/// poison the holder before dropping the guard. Asynchronous backends acquire multiple guards in stable
/// [`ReferenceId`] order, validate every lease publication or generation transition first, and then use the matching
/// unchecked commit methods while those same guards remain held. They release every guard immediately after
/// submission-time publication and never retain a holder mutex through device execution. Dropping a guard with an
/// extracted synchronous value poisons the holder defensively.
#[doc(hidden)]
pub struct ReferenceGuard<'a, V: Value> {
    /// Identity of the locked shared holder.
    id: ReferenceId,

    /// Handle-local referent type.
    r#type: &'a ReferenceType<V::Type>,

    /// Structural type stored by the shared root holder.
    root_type: &'a V::Type,

    /// Mapping applied when a root value crosses into this handle.
    root_to_handle: &'a TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Mapping applied when a handle-local value enters the root holder.
    handle_to_root: &'a TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Locked holder lifecycle state.
    state: MutexGuard<'a, ReferenceState<V>>,

    /// Notification for a completed reservation installation.
    installed: &'a Condvar,
}

/// Root-normalized holder value whose fallible reconstruction and type validation have completed.
#[doc(hidden)]
pub struct PreparedReferenceValue<V: Value> {
    /// Identity of the holder against which this value was prepared.
    reference_id: ReferenceId,

    /// Value represented in the shared root holder's type identity space.
    value: V,
}

impl<V: Value> ReferenceGuard<'_, V> {
    /// Returns whether this guard observes the short submitted-before-install reservation state.
    #[inline]
    pub fn reservation_pending(&self) -> bool {
        matches!(*self.state, ReferenceState::Reserved { .. })
    }

    /// Returns the generation of the current ready or installed pending value.
    pub fn current_generation(&self) -> Result<ReferenceGeneration, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { generation, .. } => Ok(*generation),
            ReferenceState::Pending { generation, .. } => Ok(*generation),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken { .. } | ReferenceState::Reserved { .. } => {
                Err(ReferenceError::TransactionInProgress)
            }
        }
    }

    /// Returns a handle-local immutable snapshot without extracting holder state.
    pub fn snapshot(&self) -> Result<V, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { value, .. } | ReferenceState::Pending { value, .. } => value
                .rename_type_identities(self.root_to_handle)
                .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() }),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken { .. } | ReferenceState::Reserved { .. } => {
                Err(ReferenceError::TransactionInProgress)
            }
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

    /// Validates read-lease publication without changing holder state.
    pub fn validate_read_lease_publication(&self) -> Result<(), ReferenceError> {
        match &*self.state {
            ReferenceState::Ready { .. } | ReferenceState::Pending { .. } => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken { .. } | ReferenceState::Reserved { .. } => {
                Err(ReferenceError::TransactionInProgress)
            }
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

    /// Reserves the next mutation generation after successful backend submission. Test-only convenience combinator
    /// for the production validate-then-commit protocol.
    ///
    /// `completion` must include this holder's prior pending dependency and the newly submitted execution.
    #[cfg(test)]
    fn reserve_pending(&mut self, completion: ReferenceCompletion) -> Result<ReferenceGeneration, ReferenceError> {
        let generation = self.next_generation()?;
        self.reserve_pending_unchecked(generation, completion);
        Ok(generation)
    }

    /// Validates a mutation reservation and returns its next generation without changing holder state.
    ///
    /// Any published lease still recorded on the holder rejects the reservation, including one whose execution has
    /// already completed: this borrows the holder immutably and therefore cannot prune. Backends must drain
    /// completed leases through [`Self::active_read_leases`] (releasing the guard and awaiting the returned
    /// completions when any remain) before validating a reservation, exactly as the multi-holder retry protocol
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
            ReferenceState::ExecutionPoisoned(reason) => {
                return Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() });
            }
            ReferenceState::Taken { .. } | ReferenceState::Reserved { .. } => {
                return Err(ReferenceError::TransactionInProgress);
            }
        };
        Ok(generation)
    }

    /// Commits a reservation after [`Self::next_generation`] succeeded under this same guard.
    pub fn reserve_pending_unchecked(&mut self, generation: ReferenceGeneration, completion: ReferenceCompletion) {
        debug_assert_eq!(self.next_generation(), Ok(generation));
        *self.state = ReferenceState::Reserved { generation, completion };
    }

    /// Installs a prepared final value for `generation` while leaving it pending on its cumulative completion.
    /// Test-only convenience combinator for the production validate-then-commit protocol.
    #[cfg(test)]
    fn install_pending(
        &mut self,
        generation: ReferenceGeneration,
        value: PreparedReferenceValue<V>,
    ) -> Result<(), ReferenceError> {
        self.validate_pending_install(generation, &value)?;
        self.install_pending_unchecked(generation, value);
        Ok(())
    }

    /// Validates one prepared pending-value installation without changing holder state.
    pub fn validate_pending_install(
        &self,
        generation: ReferenceGeneration,
        value: &PreparedReferenceValue<V>,
    ) -> Result<(), ReferenceError> {
        self.accepts(value)?;
        let ReferenceState::Reserved { generation: current, .. } = &*self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        if *current != generation {
            return Err(ReferenceError::StaleGeneration);
        }
        Ok(())
    }

    /// Commits a prepared value after [`Self::validate_pending_install`] succeeded under this same guard.
    pub fn install_pending_unchecked(&mut self, generation: ReferenceGeneration, value: PreparedReferenceValue<V>) {
        debug_assert!(self.validate_pending_install(generation, &value).is_ok());
        let ReferenceState::Reserved { completion, .. } = &*self.state else {
            unreachable!("pending installation was validated under the same holder guard")
        };
        *self.state = ReferenceState::Pending {
            value: value.value,
            generation,
            completion: completion.clone(),
            read_leases: Vec::new(),
        };
        self.installed.notify_all();
    }

    /// Applies a completion result only if `generation` is still current. Test-only entry into the lazy completion
    /// reconciliation that value accesses perform through `lock_ready`.
    #[cfg(test)]
    fn complete(&mut self, generation: ReferenceGeneration, result: ReferenceCompletionResult) -> bool {
        Reference::<V>::apply_completion(&mut self.state, generation, result)
    }

    /// Poisons a matching reserved or installed pending generation without affecting any newer generation.
    ///
    /// This is the generation-safe failure path for errors after submission publication but before or after hidden
    /// final-state installation. A matching reservation wakes every waiter blocked on its installation window.
    pub fn poison_pending(&mut self, generation: ReferenceGeneration, reason: impl Into<Arc<str>>) -> bool {
        if !matches!(
            *self.state,
            ReferenceState::Reserved { generation: current, .. }
                | ReferenceState::Pending { generation: current, .. }
                if current == generation
        ) {
            return false;
        }
        *self.state = ReferenceState::ExecutionPoisoned(reason.into());
        self.installed.notify_all();
        true
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

    /// Validates a prospective restored or installed value without changing holder state.
    pub fn prepare(&self, value: V) -> Result<PreparedReferenceValue<V>, ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() != self.r#type.referent() {
            return Err(ReferenceError::ReferentTypeMismatch {
                expected: self.r#type.referent().to_string(),
                actual: actual.to_string(),
            });
        }
        let stored = value
            .rename_type_identities(self.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        let actual = stored.r#type();
        if actual.as_ref() != self.root_type {
            return Err(ReferenceError::ReferentTypeMismatch {
                expected: self.root_type.to_string(),
                actual: actual.to_string(),
            });
        }
        Ok(PreparedReferenceValue { reference_id: self.id, value: stored })
    }

    /// Validates that `value` was prepared against this exact holder.
    pub(crate) fn accepts(&self, value: &PreparedReferenceValue<V>) -> Result<(), ReferenceError> {
        if value.reference_id == self.id { Ok(()) } else { Err(ReferenceError::TransactionHolderMismatch) }
    }

    /// Installs a value whose reconstruction and type checks completed through [`Self::prepare`].
    ///
    /// Installation fails when `value` was prepared for another holder or this guard does not own an extracted value.
    pub fn install(&mut self, value: PreparedReferenceValue<V>) -> Result<(), ReferenceError> {
        self.accepts(&value)?;
        let ReferenceState::Taken { generation } = *self.state else {
            return Err(ReferenceError::TransactionInProgress);
        };
        *self.state = ReferenceState::Ready { value: value.value, generation, read_leases: Vec::new() };
        self.installed.notify_all();
        Ok(())
    }

    /// Invalidates an extracted value or a submitted mutation reservation after an irreversible backend failure,
    /// recording `reason` as the cause every later holder access reports. Poisoning is infallible so that failure
    /// paths can never trade the original backend error for a guard-state error: a guard that neither extracted a
    /// value nor holds a reservation has nothing to invalidate and is deliberately left untouched, and a pending
    /// completion is poisoned through the generation-checked [`Self::poison_pending`] instead.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) {
        if matches!(*self.state, ReferenceState::Taken { .. } | ReferenceState::Reserved { .. }) {
            *self.state = ReferenceState::ExecutionPoisoned(reason.into());
            self.installed.notify_all();
        }
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state =
                ReferenceState::ExecutionPoisoned("stateful transaction ended without restoring state".into());
            self.installed.notify_all();
        }
    }
}

impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            holder: self.holder.clone(),
            r#type: self.r#type.clone(),
            root_type: self.root_type.clone(),
            root_to_handle: self.root_to_handle.clone(),
            handle_to_root: self.handle_to_root.clone(),
        }
    }
}

impl<V: Value> Debug for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Reference").field("id", &self.id()).field("type", &self.r#type).finish()
    }
}

// `Display` deliberately renders only the handle-local type: the Value rendering contract requires deterministic
// output (renderings back diagnostics, rendered-program tests, and the debug-assertions transform-cache determinism
// recheck), so the process-local holder address must not leak here. Runtime identity remains visible through `Debug`.
impl<V: Value> Display for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.r#type, formatter)
    }
}

impl<V: Value> PartialEq for Reference<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.holder, &other.holder)
    }
}

impl<V: Value> Eq for Reference<V> {}

impl<V: Value> Hash for Reference<V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.holder).hash(state);
    }
}

impl<V: Value> Parameter for Reference<V> {}

impl<V: Value> Typed for Reference<V> {
    type Type = ReferenceType<V::Type>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use pretty_assertions::assert_eq;

    use crate::arrays::Array;
    use crate::operations::Add;

    use super::*;

    #[derive(Clone)]
    struct ControlledCompletion {
        state: Arc<(Mutex<ControlledCompletionState>, Condvar)>,
    }

    struct ControlledCompletionState {
        result: Option<ReferenceCompletionResult>,
        callbacks: Vec<ReferenceCompletionCallback>,
    }

    impl ControlledCompletion {
        fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledCompletionState { result: None, callbacks: Vec::new() }),
                    Condvar::new(),
                )),
            }
        }

        fn complete(&self, result: ReferenceCompletionResult) {
            let callbacks = {
                let (state, ready) = &*self.state;
                let mut state = state.lock().unwrap();
                assert!(state.result.is_none());
                state.result = Some(result.clone());
                ready.notify_all();
                std::mem::take(&mut state.callbacks)
            };
            for callback in callbacks {
                callback(result.clone());
            }
        }
    }

    impl ReferenceCompletionBackend for ControlledCompletion {
        fn r#await(&self) -> ReferenceCompletionResult {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            while state.result.is_none() {
                state = ready.wait(state).unwrap();
            }
            state.result.clone().unwrap()
        }

        fn is_ready(&self) -> Result<bool, Arc<str>> {
            let state = self.state.0.lock().unwrap();
            state.result.clone().map_or(Ok(false), |result| result.map(|_| true))
        }

        fn on_ready(&self, callback: ReferenceCompletionCallback) {
            let callback = {
                let mut state = self.state.0.lock().unwrap();
                if let Some(result) = &state.result {
                    Some((callback, result.clone()))
                } else {
                    state.callbacks.push(callback);
                    None
                }
            };
            if let Some((callback, result)) = callback {
                callback(result);
            }
        }
    }

    #[test]
    fn test_reference_clones_alias_one_holder_and_reads_snapshots() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = Reference::new(initial.clone());
        let alias = reference.clone();
        let distinct = Reference::new(initial);
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
    fn test_reference_freeze_invalidates_the_complete_alias_family() {
        let reference = Reference::new(Array::vector(vec![1.0_f32, 2.0]));
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));

        assert_eq!(alias.read(), Err(ReferenceError::Frozen));
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
        let reference = Reference::new(initial.clone());

        assert_eq!(
            reference.swap(Array::vector(vec![3.0_f32, 4.0, 5.0])),
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
        let first = Reference::new(initializer.clone());
        let second = Reference::new(initializer.clone());
        let read_snapshot = first.read().unwrap();
        let replacement = Array::vector(vec![3.0_f32, 4.0]);
        let retained_replacement = replacement.clone();

        let swapped_snapshot = first.swap(replacement).unwrap();
        first.update_with(|current| current.add(&Array::vector(vec![10.0_f32, 20.0]))).unwrap();
        assert_eq!(second.read(), Ok(initializer.clone()));
        assert_eq!(second.swap(Array::vector(vec![5.0_f32, 6.0])), Ok(initializer.clone()));

        assert_eq!(initializer, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(read_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(swapped_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(retained_replacement, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(first.read(), Ok(Array::vector(vec![13.0_f32, 24.0])));
        assert_eq!(second.read(), Ok(Array::vector(vec![5.0_f32, 6.0])));
    }

    #[test]
    fn test_reference_generation_advances_only_after_committed_mutations() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let initial_generation = reference.lock().unwrap().current_generation().unwrap();

        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(initial_generation));

        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
        let swapped_generation = reference.lock().unwrap().current_generation().unwrap();
        assert!(swapped_generation > initial_generation);

        let rejected = ProgramError::InvalidArgument { message: "rejected update".to_string() };
        assert_eq!(reference.update_with(|_| Err(rejected.clone())), Err(rejected));
        assert_eq!(reference.lock().unwrap().current_generation(), Ok(swapped_generation));

        reference.update_with(|_| Ok(Array::scalar(3.0_f32))).unwrap();
        let updated_generation = reference.lock().unwrap().current_generation().unwrap();
        assert!(updated_generation > swapped_generation);
    }

    #[test]
    fn test_reference_read_reports_a_poisoned_holder() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let holder = Arc::clone(&reference.holder);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = holder.state.lock().unwrap();
                panic!("poison reference holder");
            }))
            .is_err(),
        );
        assert_eq!(reference.read(), Err(ReferenceError::Poisoned));
    }

    #[test]
    fn test_reference_guard_prepares_transaction_values_for_the_exact_holder() {
        let first = Reference::new(Array::scalar(1.0_f32));
        let second = Reference::new(Array::scalar(2.0_f32));
        let mut first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        assert_eq!(first_guard.take(), Ok(Array::scalar(1.0_f32)));
        let prepared = first_guard.prepare(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(second_guard.accepts(&prepared), Err(ReferenceError::TransactionHolderMismatch));
        first_guard.install(prepared).unwrap();
        drop(first_guard);
        drop(second_guard);
        assert_eq!(first.read(), Ok(Array::scalar(3.0_f32)));
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
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

        let observed = Arc::new(Mutex::new(None));
        let callback_observed = Arc::clone(&observed);
        completion.on_ready(Box::new(move |result| {
            *callback_observed.lock().unwrap() = Some(result);
        }));
        assert_eq!(*observed.lock().unwrap(), Some(Err(Arc::<str>::from("first failure"))));
    }

    #[test]
    fn test_reference_completion_reports_deferred_callback_once() {
        let backend = ControlledCompletion::new();
        let completion = ReferenceCompletion::new(backend.clone());
        assert_eq!(completion.is_ready(), Ok(false));
        let observed = Arc::new(Mutex::new(Vec::new()));
        let callback_observed = Arc::clone(&observed);
        completion.on_ready(Box::new(move |result| callback_observed.lock().unwrap().push(result)));
        assert!(observed.lock().unwrap().is_empty());
        backend.complete(Ok(()));
        assert_eq!(*observed.lock().unwrap(), vec![Ok(())]);
        assert_eq!(completion.is_ready(), Ok(true));
        assert_eq!(completion.r#await(), Ok(()));
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
        let observed = Arc::new(Mutex::new(None));
        let callback_observed = Arc::clone(&observed);
        joined.on_ready(Box::new(move |result| *callback_observed.lock().unwrap() = Some(result)));
        third.complete(Err("third failure".into()));
        second.complete(Err("second failure".into()));
        assert_eq!(joined.is_ready(), Ok(false));
        assert_eq!(*observed.lock().unwrap(), None);
        first.complete(Err("first failure".into()));
        assert_eq!(joined.r#await(), Err(Arc::<str>::from("first failure")));
        assert_eq!(*observed.lock().unwrap(), Some(Err(Arc::<str>::from("first failure"))));
    }

    #[test]
    fn test_reference_pending_generations_ignore_stale_completion() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let first = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();

        let second = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let second_value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        guard.install_pending(second, second_value).unwrap();
        assert!(!guard.complete(first, Err("stale failure".into())));
        assert!(guard.complete(second, Ok(())));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_pending_poison_is_generation_safe() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let first = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();
        let second = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let second_value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        guard.install_pending(second, second_value).unwrap();
        assert!(!guard.poison_pending(first, "stale failure"));
        assert!(guard.poison_pending(second, "current failure"));
        drop(guard);
        assert_eq!(reference.read(), Err(ReferenceError::ExecutionPoisoned { reason: "current failure".to_string() }));
    }

    #[test]
    fn test_reference_reservation_waiter_wakes_after_installation() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)));
        let generation = reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let waiting_reference = Arc::clone(&reference);
        let waiting_barrier = Arc::clone(&barrier);
        let waiter = std::thread::spawn(move || {
            waiting_barrier.wait();
            waiting_reference.wait_until_accessible()
        });
        barrier.wait();
        let mut guard = reference.lock().unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        drop(guard);
        assert_eq!(waiter.join().unwrap(), Ok(()));
    }

    #[test]
    fn test_reference_read_lease_must_be_pruned_before_mutation_reservation() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::ready(Ok(())));
        assert_eq!(
            guard.reserve_pending(ReferenceCompletion::ready(Ok(()))),
            Err(ReferenceError::TransactionInProgress),
        );
        assert!(guard.active_read_leases().is_empty());
        assert!(guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).is_ok());
        guard.poison("test cleanup");
    }

    #[test]
    fn test_reference_read_awaits_a_pending_completion_resolved_by_another_thread() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let generation = guard.reserve_pending(ReferenceCompletion::new(backend.clone())).unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
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
        assert_eq!(*observed.lock().unwrap(), None);
        backend.complete(Ok(()));
        reader.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(2.0_f32))));

        // The successful completion was applied to the holder, so the value is now ready without any dependency.
        assert!(reference.lock().unwrap().dependency().is_none());
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_read_reports_a_failed_pending_completion_as_execution_poisoned() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let generation = guard.reserve_pending(ReferenceCompletion::new(backend.clone())).unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        drop(guard);

        // The completion resolves before the read reaches it, so the read observes the failure through the same lazy
        // reconciliation path and reports the backend-owned reason. Poisoning is terminal for every later access.
        backend.complete(Err("device execution failed".into()));
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "device execution failed".to_string() };
        assert_eq!(reference.read(), Err(poisoned.clone()));
        assert_eq!(reference.swap(Array::scalar(3.0_f32)), Err(poisoned.clone()));
        assert_eq!(reference.freeze(), Err(poisoned));
    }

    #[test]
    fn test_reference_swap_waits_for_an_active_read_lease() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)));
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
        assert_eq!(*observed.lock().unwrap(), None);
        lease.complete(Ok(()));
        swapper.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(1.0_f32))));
        assert_eq!(reference.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_freeze_waits_for_an_active_read_lease() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)));
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
        assert_eq!(*observed.lock().unwrap(), None);
        lease.complete(Ok(()));
        freezer.join().unwrap();
        assert_eq!(*observed.lock().unwrap(), Some(Ok(Array::scalar(1.0_f32))));
        assert_eq!(reference.read(), Err(ReferenceError::Frozen));
    }

    #[test]
    fn test_poisoning_a_reservation_wakes_an_accessibility_waiter() {
        let reference = Arc::new(Reference::new(Array::scalar(1.0_f32)));
        let generation = reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let waiting_reference = Arc::clone(&reference);
        let waiting_barrier = Arc::clone(&barrier);
        let waiter = std::thread::spawn(move || {
            waiting_barrier.wait();
            waiting_reference.wait_until_accessible()
        });
        barrier.wait();

        // A reservation that fails after submission never installs a value, so its waiter must be woken with the
        // terminal failure instead of blocking forever on an installation that will never happen.
        let mut guard = reference.lock().unwrap();
        assert!(guard.poison_pending(generation, "submission failed"));
        drop(guard);
        assert_eq!(
            waiter.join().unwrap(),
            Err(ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() }),
        );
    }

    #[test]
    fn test_reference_pending_install_rejects_a_stale_generation() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let first = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();
        let second = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();

        // A late installation for the superseded generation is rejected without changing holder state, so the current
        // reservation still installs its own value.
        let value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(guard.validate_pending_install(first, &value), Err(ReferenceError::StaleGeneration));
        guard.install_pending(second, value).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_take_rejects_an_active_read_lease() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let lease = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::new(lease.clone()));

        // Handing leased buffers to a donating execution would let the device mutate storage a submitted read-only
        // execution still observes, so extraction is rejected until the lease completes and is pruned.
        assert_eq!(guard.take(), Err(ReferenceError::TransactionInProgress));
        lease.complete(Ok(()));
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        let restored = guard.prepare(Array::scalar(4.0_f32)).unwrap();
        guard.install(restored).unwrap();
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_isolated_to_the_extracted_holder() {
        let first = Reference::new(Array::scalar(1.0_f32));
        let second = Reference::new(Array::scalar(2.0_f32));
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
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let generation = guard.current_generation().unwrap();

        // Poisoning is infallible so that a failure path can never trade the original backend error for a guard-state
        // error. A guard that neither extracted a value nor holds a reservation has nothing to invalidate, so the
        // holder must stay ready at its current generation and remain readable afterwards.
        guard.poison("unrelated backend failure");
        assert_eq!(guard.current_generation(), Ok(generation));
        assert_eq!(guard.snapshot(), Ok(Array::scalar(1.0_f32)));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
        assert_eq!(reference.swap(Array::scalar(2.0_f32)), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_read_lease_publication_releases_completed_leases() {
        let reference = Reference::new(Array::scalar(1.0_f32));
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

        // Only the one still-running lease remains recorded, and it alone blocks the next mutation reservation until
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
        let reference = Reference::new(Array::scalar(1.0_f32));
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
}
