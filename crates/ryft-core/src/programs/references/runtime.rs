//! Runtime reference holders and backend completion coordination.
//!
//! The synchronized holder state machine backs eager [`Reference`] access and stateful compiled execution. Backends
//! coordinate external holders through generation-checked reservations, cumulative completion dependencies, and read
//! leases.
//!
//! Once a backend crosses its submission boundary, failure cannot restore an unambiguously current value; the
//! affected mutated holders become poisoned and every later access reports the failure. Poisoning is terminal;
//! recovery means constructing a new holder from independently trusted state. These transaction types are a hidden,
//! unstable backend service-provider interface. User code should use the stateful call surface and await its
//! [`ReferenceExecution`](crate::compilation::ReferenceExecution) instead of acquiring holder guards directly.

// TODO(eaplatanios): Review this module.

use std::fmt::Debug;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

use crate::programs::values::Value;

use super::values::Reference;

/// Error produced while accessing a [`Reference`]'s eager holder.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum ReferenceError {
    /// A reference allocation attempted to store another reference as its immediate referent.
    #[error("reference referent type `{referent_type}` must not itself be a reference")]
    NestedReferent {
        /// Rejected immediate referent type.
        referent_type: String,
    },

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
/// information, is never serialized into a program or compilation key, and may be reused after the last handle and
/// every prepared transaction value retaining the original holder allocation are dropped.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId(usize);

impl ReferenceId {
    /// Creates an identity from one live holder address.
    #[inline]
    pub(super) fn from_address(address: usize) -> Self {
        Self(address)
    }
}

/// Monotonic generation of one holder mutation reservation.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceGeneration(u64);

impl ReferenceGeneration {
    /// Returns the initial generation of a newly allocated holder.
    #[inline]
    pub(super) fn initial() -> Self {
        Self(0)
    }

    /// Returns the successor generation, or [`None`] when the monotonic counter is exhausted.
    #[inline]
    pub(super) fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Storage shared by one reference alias family.
pub(super) struct ReferenceHolder<V: Value> {
    /// Structural referent type of values stored in this holder. Every alias agrees on it, it is immutable for the
    /// holder's lifetime, and it is deliberately readable without the state lock so validation paths never have to
    /// acquire or order against the lifecycle mutex.
    pub(super) root_type: V::Type,

    /// Holder lifecycle state.
    pub(super) state: Mutex<ReferenceState<V>>,
}

/// Lifecycle state shared by every handle in one reference alias family.
pub(super) enum ReferenceState<V: Value> {
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

        /// Separate signal owned by the submitted execution until the complete mutation batch is installed.
        reservation: Arc<ReferenceReservationSignal>,

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

        /// Separate signal owned by the submitted execution until the complete mutation batch is installed.
        reservation: Arc<ReferenceReservationSignal>,
    },

    /// Value that may have been consumed by an irreversible failed backend invocation.
    ExecutionPoisoned(Arc<str>),

    /// Consumed reference whose value was returned by `freeze`.
    Frozen,
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
    /// Reference handle whose holder and handle-local type mapping this guard protects.
    pub(super) reference: &'a Reference<V>,

    /// Locked holder lifecycle state.
    pub(super) state: MutexGuard<'a, ReferenceState<V>>,
}

/// Root-normalized holder value whose fallible reconstruction and type validation have completed.
#[doc(hidden)]
pub struct PreparedReferenceValue<V: Value> {
    /// Weak identity of the holder allocation against which this value was prepared.
    ///
    /// Retaining the allocation's weak control block prevents its address from being recycled while this prepared
    /// value can still present it as transaction-ownership proof.
    holder: Weak<ReferenceHolder<V>>,

    /// Value represented in the shared root holder's type identity space.
    value: V,
}

/// Lifecycle of one separately synchronized mutation reservation publication.
#[derive(Clone)]
enum ReferenceReservationStatus {
    /// The holder remains reserved and no hidden final value has been installed.
    AwaitingInstallation,

    /// The hidden final value was installed and its cumulative backend dependency remains pending.
    Installed,

    /// The cumulative backend dependency reached its immutable terminal result.
    Completed(ReferenceCompletionResult),

    /// The owning execution ended before complete-batch installation, with an optional explicit backend failure.
    Abandoned(Option<Arc<str>>),
}

/// Signal synchronized independently from the holder mutex for one submitted mutation reservation.
pub(super) struct ReferenceReservationSignal {
    /// Current reservation publication lifecycle.
    status: Mutex<ReferenceReservationStatus>,

    /// Notification for installation or abandonment.
    resolved: Condvar,

    /// Whether a test reader has entered the terminal-result wait.
    #[cfg(test)]
    terminal_waiter_entered: AtomicBool,
}

impl ReferenceReservationSignal {
    /// Creates a signal awaiting hidden final-state installation.
    fn new() -> Self {
        Self {
            status: Mutex::new(ReferenceReservationStatus::AwaitingInstallation),
            resolved: Condvar::new(),
            #[cfg(test)]
            terminal_waiter_entered: AtomicBool::new(false),
        }
    }

    /// Waits without holding the reference holder mutex until installation or abandonment is published.
    pub(super) fn wait_until_resolved(&self) {
        let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while matches!(*status, ReferenceReservationStatus::AwaitingInstallation) {
            status = self.resolved.wait(status).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Waits until either the cumulative backend dependency completes or the reservation is abandoned.
    pub(super) fn wait_until_terminal(&self) -> Option<ReferenceCompletionResult> {
        let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            match &*status {
                ReferenceReservationStatus::Completed(result) => return Some(result.clone()),
                ReferenceReservationStatus::Abandoned(_) => return None,
                ReferenceReservationStatus::AwaitingInstallation | ReferenceReservationStatus::Installed => {
                    #[cfg(test)]
                    {
                        self.terminal_waiter_entered.store(true, Ordering::Release);
                        self.resolved.notify_all();
                    }
                    status = self.resolved.wait(status).unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            }
        }
    }

    /// Waits until a test reader is blocked on terminal completion or abandonment.
    #[cfg(test)]
    fn wait_until_terminal_waiter(&self) {
        let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while !self.terminal_waiter_entered.load(Ordering::Acquire) {
            status = self.resolved.wait(status).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Marks hidden final-state installation without overwriting prior abandonment.
    fn install(self: &Arc<Self>, completion: ReferenceCompletion) {
        {
            let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if !matches!(*status, ReferenceReservationStatus::AwaitingInstallation) {
                return;
            }
            *status = ReferenceReservationStatus::Installed;
            self.resolved.notify_all();
        }
        let reservation = Arc::clone(self);
        completion.on_ready(Box::new(move |result| reservation.complete(result)));
    }

    /// Publishes the cumulative backend dependency's immutable terminal result unless abandonment won the race.
    fn complete(&self, result: ReferenceCompletionResult) {
        let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*status, ReferenceReservationStatus::Installed) {
            *status = ReferenceReservationStatus::Completed(result);
            self.resolved.notify_all();
        }
    }

    /// Marks this reservation abandoned without acquiring the reference holder mutex.
    fn abandon(&self, reason: Option<Arc<str>>) {
        let mut status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(*status, ReferenceReservationStatus::Abandoned(_)) {
            *status = ReferenceReservationStatus::Abandoned(reason);
            self.resolved.notify_all();
        }
    }

    /// Returns the published abandonment reason, using the defensive default for owner drop.
    pub(super) fn abandonment_reason(&self) -> Option<Arc<str>> {
        let status = self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let ReferenceReservationStatus::Abandoned(reason) = &*status else { return None };
        Some(
            reason
                .clone()
                .unwrap_or_else(|| Arc::from("submitted reference execution ended before final state installation")),
        )
    }

    /// Returns whether hidden final-state installation has completed.
    fn is_installed(&self) -> bool {
        matches!(
            *self.status.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            ReferenceReservationStatus::Installed | ReferenceReservationStatus::Completed(_),
        )
    }
}

/// RAII ownership token returned by publishing one reference mutation reservation.
///
/// Dropping an armed token never acquires the holder mutex. It marks the reservation abandoned and wakes waiters; the
/// next holder access then converts the matching current reservation into a terminal execution failure under the
/// holder's ordinary lock. A stale token retains only its retired abandonment state and cannot affect a later
/// generation.
#[doc(hidden)]
#[must_use = "a published reference reservation must remain owned until its complete mutation batch is installed"]
pub struct PendingReferenceReservation {
    /// Signal stored in the matching reserved or pending generation.
    reservation: Arc<ReferenceReservationSignal>,

    /// Whether drop still owns cleanup responsibility.
    armed: bool,
}

impl PendingReferenceReservation {
    /// Marks this reservation abandoned, optionally retaining an explicit backend failure.
    fn abandon(&mut self, reason: Option<Arc<str>>) {
        if !self.armed {
            return;
        }
        self.reservation.abandon(reason);
        self.armed = false;
    }

    /// Disarms cleanup after the complete mutation batch has installed every pending final value.
    fn disarm(&mut self) {
        if self.reservation.is_installed() {
            self.armed = false;
        }
    }
}

impl Drop for PendingReferenceReservation {
    fn drop(&mut self) {
        if self.armed {
            self.abandon(None);
        }
    }
}

/// RAII cleanup owner for one submitted batch of reference mutation reservations.
#[doc(hidden)]
#[must_use = "submitted reference reservations must remain owned until the complete mutation batch is installed"]
pub struct PendingReferenceReservations {
    /// Per-holder ownership tokens returned by reservation publication.
    reservations: Vec<PendingReferenceReservation>,
}

impl PendingReferenceReservations {
    /// Collects the ownership tokens returned while publishing one complete mutation batch.
    pub fn new(reservations: Vec<PendingReferenceReservation>) -> Self {
        Self { reservations }
    }

    /// Marks every still-owned reservation abandoned with the same backend failure.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) {
        let reason = reason.into();
        for reservation in &mut self.reservations {
            reservation.abandon(Some(Arc::clone(&reason)));
        }
    }

    /// Disarms cleanup only after every reservation has installed its pending final value.
    pub fn disarm(&mut self) {
        debug_assert!(self.reservations.iter().all(|reservation| reservation.reservation.is_installed()));
        for reservation in &mut self.reservations {
            reservation.disarm();
        }
    }
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
            ReferenceState::Ready { value, .. } | ReferenceState::Pending { value, .. } => {
                self.reference.reconstruct_local(value)
            }
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
    /// `completion` must include this holder's prior pending dependency and the newly submitted execution. This is a
    /// submission-time safety obligation: joining the predecessor after submission cannot prevent the backend from
    /// reading or replacing pending storage before the predecessor finishes.
    #[cfg(test)]
    fn reserve_pending(
        &mut self,
        completion: ReferenceCompletion,
    ) -> Result<(ReferenceGeneration, PendingReferenceReservation), ReferenceError> {
        let generation = self.next_generation()?;
        let reservation = self.reserve_pending_unchecked(generation, completion);
        Ok((generation, reservation))
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
    ///
    /// `completion` must cumulatively include both this holder's prior pending dependency and the newly submitted
    /// execution. The predecessor must be part of the submission-time dependency: joining it afterwards cannot
    /// prevent the backend from reading or replacing pending storage before that predecessor finishes.
    #[must_use = "the returned reservation token must be retained until complete batch installation"]
    pub fn reserve_pending_unchecked(
        &mut self,
        generation: ReferenceGeneration,
        completion: ReferenceCompletion,
    ) -> PendingReferenceReservation {
        debug_assert_eq!(self.next_generation(), Ok(generation));
        let reservation = Arc::new(ReferenceReservationSignal::new());
        *self.state = ReferenceState::Reserved { generation, completion, reservation: Arc::clone(&reservation) };
        PendingReferenceReservation { reservation, armed: true }
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
        let ReferenceState::Reserved { completion, reservation, .. } = &*self.state else {
            unreachable!("pending installation was validated under the same holder guard")
        };
        let completion = completion.clone();
        let reservation = Arc::clone(reservation);
        *self.state = ReferenceState::Pending {
            value: value.value,
            generation,
            completion: completion.clone(),
            reservation: Arc::clone(&reservation),
            read_leases: Vec::new(),
        };
        reservation.install(completion);
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
        let reason = reason.into();
        let reservation = match &*self.state {
            ReferenceState::Reserved { reservation, .. } | ReferenceState::Pending { reservation, .. } => {
                Arc::clone(reservation)
            }
            _ => unreachable!("matching generation was validated as reserved or pending"),
        };
        reservation.abandon(Some(Arc::clone(&reason)));
        *self.state = ReferenceState::ExecutionPoisoned(reason);
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
        let stored = self.reference.prepare_stored(value)?;
        Ok(PreparedReferenceValue { holder: Arc::downgrade(&self.reference.inner.holder), value: stored })
    }

    /// Validates that `value` was prepared against this exact holder.
    pub(crate) fn accepts(&self, value: &PreparedReferenceValue<V>) -> Result<(), ReferenceError> {
        if std::ptr::eq(value.holder.as_ptr(), Arc::as_ptr(&self.reference.inner.holder)) {
            Ok(())
        } else {
            Err(ReferenceError::TransactionHolderMismatch)
        }
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
        Ok(())
    }

    /// Invalidates an extracted value or a submitted mutation reservation after an irreversible backend failure,
    /// recording `reason` as the cause every later holder access reports. Poisoning is infallible so that failure
    /// paths can never trade the original backend error for a guard-state error: a guard that neither extracted a
    /// value nor holds a reservation has nothing to invalidate and is deliberately left untouched, and a pending
    /// completion is poisoned through the generation-checked [`Self::poison_pending`] instead.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) {
        if matches!(*self.state, ReferenceState::Taken { .. } | ReferenceState::Reserved { .. }) {
            let reason = reason.into();
            if let ReferenceState::Reserved { reservation, .. } = &*self.state {
                reservation.abandon(Some(Arc::clone(&reason)));
            }
            *self.state = ReferenceState::ExecutionPoisoned(reason);
        }
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken { .. }) {
            *self.state =
                ReferenceState::ExecutionPoisoned("stateful transaction ended without restoring state".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use pretty_assertions::assert_eq;

    use crate::arrays::Array;
    use crate::programs::ProgramError;

    use super::*;

    #[derive(Clone)]
    struct ControlledCompletion {
        state: Arc<(Mutex<ControlledCompletionState>, Condvar)>,
    }

    struct ControlledCompletionState {
        awaiting: bool,
        result: Option<ReferenceCompletionResult>,
        callbacks: Vec<ReferenceCompletionCallback>,
    }

    impl ControlledCompletion {
        fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledCompletionState { awaiting: false, result: None, callbacks: Vec::new() }),
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

    fn reference_new<V: Value>(value: V) -> Reference<V> {
        Reference::new(value).unwrap()
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
            let mut state = reference.inner.holder.state.lock().unwrap();
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
        let holder = Arc::clone(&reference.inner.holder);
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
    fn test_reference_guard_prepares_transaction_values_for_the_exact_holder() {
        let first = reference_new(Array::scalar(1.0_f32));
        let second = reference_new(Array::scalar(2.0_f32));
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
    fn test_prepared_reference_value_pins_holder_allocation_identity_after_last_handle_is_dropped() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let original_id = reference.id();
        let prepared = reference.lock().unwrap().prepare(Array::scalar(2.0_f32)).unwrap();
        let original_holder = prepared.holder.clone();
        drop(reference);
        assert!(original_holder.upgrade().is_none());

        // The surviving weak control block keeps the retired allocation address unavailable to a new holder. This
        // makes pointer equality a stable ownership proof even after every strong handle to the original is gone.
        let replacement = reference_new(Array::scalar(3.0_f32));
        assert_ne!(replacement.id(), original_id);
        assert_eq!(replacement.lock().unwrap().accepts(&prepared), Err(ReferenceError::TransactionHolderMismatch),);
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
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let (first, mut first_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();
        first_reservation.disarm();

        let (second, mut second_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let second_value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        guard.install_pending(second, second_value).unwrap();
        second_reservation.disarm();
        assert!(!guard.complete(first, Err("stale failure".into())));
        assert!(guard.complete(second, Ok(())));
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_reference_pending_poison_is_generation_safe() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let (first, mut first_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();
        first_reservation.disarm();
        let (second, second_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let second_value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        guard.install_pending(second, second_value).unwrap();
        assert!(!guard.poison_pending(first, "stale failure"));
        assert!(guard.poison_pending(second, "current failure"));
        drop(guard);
        drop(second_reservation);
        assert_eq!(reference.read(), Err(ReferenceError::ExecutionPoisoned { reason: "current failure".to_string() }));
    }

    #[test]
    fn test_reference_reservation_waiter_wakes_after_installation() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let (generation, mut reservation) =
            reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
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
        reservation.disarm();
        drop(guard);
        assert_eq!(waiter.join().unwrap(), Ok(()));
    }

    #[test]
    fn test_reference_read_lease_must_be_pruned_before_mutation_reservation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        guard.validate_read_lease_publication().unwrap();
        guard.publish_read_lease_unchecked(ReferenceCompletion::ready(Ok(())));
        assert!(matches!(
            guard.reserve_pending(ReferenceCompletion::ready(Ok(()))),
            Err(ReferenceError::TransactionInProgress),
        ));
        assert!(guard.active_read_leases().is_empty());
        let _reservation = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        guard.poison("test cleanup");
    }

    #[test]
    fn test_reference_read_awaits_a_pending_completion_resolved_by_another_thread() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let (generation, mut reservation) = guard.reserve_pending(ReferenceCompletion::new(backend.clone())).unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        let reservation_signal = Arc::clone(&reservation.reservation);
        reservation.disarm();
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
        reservation_signal.wait_until_terminal_waiter();
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
        let (generation, mut reservation) = guard.reserve_pending(ReferenceCompletion::new(backend.clone())).unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        let reservation_signal = Arc::clone(&reservation.reservation);
        reservation.disarm();
        drop(guard);

        let observed = Arc::new(Mutex::new(None));
        let writing_reference = Arc::clone(&reference);
        let writing_observed = Arc::clone(&observed);
        let writer = std::thread::spawn(move || {
            let result = writing_reference.write(Array::scalar(3.0_f32));
            *writing_observed.lock().unwrap() = Some(result);
        });
        reservation_signal.wait_until_terminal_waiter();
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
        let (generation, mut reservation) = guard.reserve_pending(ReferenceCompletion::new(backend.clone())).unwrap();
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        reservation.disarm();
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
    fn test_poisoning_a_reservation_wakes_an_accessibility_waiter() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let (generation, reservation) =
            reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
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
        drop(reservation);
        assert_eq!(
            waiter.join().unwrap(),
            Err(ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() }),
        );
    }

    #[test]
    fn test_dropping_reservation_ownership_while_holder_guard_is_locked_does_not_relock() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let (_, reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();

        // Drop cannot acquire the mutex held by `guard`; it only marks the shared reservation state abandoned.
        drop(reservation);
        assert!(guard.reservation_pending());
        drop(guard);
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "submitted reference execution ended before final state installation".to_string(),
            }),
        );
    }

    #[test]
    fn test_dropping_reservation_batch_wakes_waiter_with_abandonment_failure() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let (_, reservation) = reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let reservations = PendingReferenceReservations::new(vec![reservation]);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let waiting_reference = Arc::clone(&reference);
        let waiting_barrier = Arc::clone(&barrier);
        let waiter = std::thread::spawn(move || {
            waiting_barrier.wait();
            waiting_reference.wait_until_accessible()
        });
        barrier.wait();
        drop(reservations);
        assert_eq!(
            waiter.join().unwrap(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "submitted reference execution ended before final state installation".to_string(),
            }),
        );
    }

    #[test]
    fn test_dropping_installed_reservation_wakes_pending_waiter_before_backend_completion() {
        let reference = Arc::new(reference_new(Array::scalar(1.0_f32)));
        let pending_backend = ControlledCompletion::new();
        let mut guard = reference.lock().unwrap();
        let (generation, reservation) =
            guard.reserve_pending(ReferenceCompletion::new(pending_backend.clone())).unwrap();
        let reservation_signal = Arc::clone(&reservation.reservation);
        let value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(generation, value).unwrap();
        let reservations = PendingReferenceReservations::new(vec![reservation]);
        drop(guard);

        let (sender, receiver) = std::sync::mpsc::channel();
        let waiting_reference = Arc::clone(&reference);
        let waiter = std::thread::spawn(move || sender.send(waiting_reference.read()).unwrap());
        reservation_signal.wait_until_terminal_waiter();

        // Installation alone does not disarm batch cleanup. Abandoning that ownership must preempt a backend
        // completion that may never arrive and wake a reader already blocked on the pending generation.
        drop(reservations);
        let result = receiver.recv_timeout(std::time::Duration::from_secs(1));
        if result.is_err() {
            pending_backend.complete(Ok(()));
        }
        assert_eq!(
            result.unwrap(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "submitted reference execution ended before final state installation".to_string(),
            }),
        );
        waiter.join().unwrap();
    }

    #[test]
    fn test_explicit_batch_poison_covers_installed_and_uninstalled_reservations() {
        let first = reference_new(Array::scalar(1.0_f32));
        let second = reference_new(Array::scalar(2.0_f32));
        let mut first_guard = first.lock().unwrap();
        let mut second_guard = second.lock().unwrap();
        let (first_generation, first_reservation) =
            first_guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let (_, second_reservation) = second_guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = first_guard.prepare(Array::scalar(3.0_f32)).unwrap();
        first_guard.install_pending(first_generation, first_value).unwrap();
        let mut reservations = PendingReferenceReservations::new(vec![first_reservation, second_reservation]);

        // Explicit cleanup is also non-locking, so a partial batch can be invalidated while every publication guard
        // remains held. The installed Pending member and the uninstalled Reserved member fail atomically on access.
        reservations.poison("batch installation failed");
        drop(first_guard);
        drop(second_guard);
        let expected = Err(ReferenceError::ExecutionPoisoned { reason: "batch installation failed".to_string() });
        assert_eq!(first.read(), expected.clone());
        assert_eq!(second.read(), expected);
    }

    #[test]
    fn test_stale_reservation_abandonment_cannot_poison_a_newer_generation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let (first_generation, first_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first_generation, first_value).unwrap();
        let (second_generation, mut second_reservation) =
            guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();

        drop(first_reservation);
        let second_value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        guard.install_pending(second_generation, second_value).unwrap();
        second_reservation.disarm();
        drop(guard);
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_unwinding_drops_reservation_owner_and_poisons_the_holder() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let (_, reservation) = reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _reservations = PendingReferenceReservations::new(vec![reservation]);
                panic!("injected backend unwind");
            }))
            .is_err(),
        );
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "submitted reference execution ended before final state installation".to_string(),
            }),
        );
    }

    #[test]
    fn test_reservation_owner_drop_does_not_double_panic_after_holder_mutex_poisoning() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let holder = Arc::clone(&reference.inner.holder);
        let (_, reservation) = reference.lock().unwrap().reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let reservations = PendingReferenceReservations::new(vec![reservation]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = holder.state.lock().unwrap();
                panic!("poison holder mutex");
            }))
            .is_err(),
        );
        assert!(catch_unwind(AssertUnwindSafe(|| drop(reservations))).is_ok());
        assert_eq!(reference.read(), Err(ReferenceError::Poisoned));
    }

    #[test]
    fn test_abandonment_preempts_completion_state_before_installation() {
        let pending_backend = ControlledCompletion::new();
        let references = [
            reference_new(Array::scalar(1.0_f32)),
            reference_new(Array::scalar(2.0_f32)),
            reference_new(Array::scalar(3.0_f32)),
        ];
        let completions = [
            ReferenceCompletion::ready(Ok(())),
            ReferenceCompletion::ready(Err("completed execution failure".into())),
            ReferenceCompletion::new(pending_backend),
        ];
        let reservations = references
            .iter()
            .zip(completions)
            .map(|(reference, completion)| reference.lock().unwrap().reserve_pending(completion).unwrap().1)
            .collect();
        drop(PendingReferenceReservations::new(reservations));

        let expected = Err(ReferenceError::ExecutionPoisoned {
            reason: "submitted reference execution ended before final state installation".to_string(),
        });
        for reference in references {
            assert_eq!(reference.read(), expected.clone());
        }
    }

    #[test]
    fn test_reference_pending_install_rejects_a_stale_generation() {
        let reference = reference_new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        let (first, mut first_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();
        let first_value = guard.prepare(Array::scalar(2.0_f32)).unwrap();
        guard.install_pending(first, first_value).unwrap();
        first_reservation.disarm();
        let (second, mut second_reservation) = guard.reserve_pending(ReferenceCompletion::ready(Ok(()))).unwrap();

        // A late installation for the superseded generation is rejected without changing holder state, so the current
        // reservation still installs its own value.
        let value = guard.prepare(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(guard.validate_pending_install(first, &value), Err(ReferenceError::StaleGeneration));
        guard.install_pending(second, value).unwrap();
        second_reservation.disarm();
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
        let restored = guard.prepare(Array::scalar(4.0_f32)).unwrap();
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
}
