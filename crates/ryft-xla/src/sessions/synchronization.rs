use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use ryft_pjrt::{Buffer, Error as PjrtError, Event, ExecutionFence};

use crate::{Error, ExecutionError};

/// Tracks submissions from the moment they are registered until they complete. An owner registers each submission
/// before its first native enqueue, so a wait over a snapshot of the registered identities also covers submissions that
/// are still being prepared. Successful submissions are retired as soon as they complete, while failures are kept in
/// registration order until the owner removes them. A [`SessionWorkTracker`] uses several trackers for submissions
/// from different threads. Each [`EffectScope`](crate::sessions::effects::EffectScope) owns one tracker for its
/// executions with effects and keeps their failures until they are acknowledged. `T` is the tag stored with each
/// submission, and `E` is the failure that a submission may record.
pub(super) struct WorkTracker<T, E> {
    /// Registration order, pending submissions, and failures that the owner has yet to remove.
    state: Mutex<WorkTrackerState<T, E>>,

    /// Wakes waiting callers when a submission completes.
    completed: Condvar,
}

impl<T, E> WorkTracker<T, E> {
    /// Registers a submission with the provided `tag` and returns its identity. The submission stays pending until
    /// [`Self::complete`] has been called as many times as it has outstanding completion notifications, which is
    /// once for a newly registered submission.
    pub(super) fn register(&self, tag: T) -> u64 {
        let mut state = self.lock();
        let identity = state.next_identity;
        state.next_identity += 1;
        state.insert_pending(identity, tag);
        identity
    }

    /// Records the result of one completion notification of the submission `identity`, keeping the first failure,
    /// and retires the submission once all of its notifications have arrived. A submission that failed moves to the
    /// failures of this tracker. Notifications for submissions that are no longer pending are ignored.
    pub(super) fn complete(&self, identity: u64, error: Option<E>) {
        let mut state = self.lock();
        let Entry::Occupied(mut entry) = state.pending.entry(identity) else {
            return;
        };

        let pending = entry.get_mut();
        if pending.error.is_none() {
            pending.error = error;
        }

        pending.remaining -= 1;
        if pending.remaining != 0 {
            return;
        }

        let PendingWork { tag, error, .. } = entry.remove();
        if let Some(error) = error {
            state.failures.insert(identity, (tag, error));
        }

        // Publish the result before waking any waiting callers.
        drop(state);

        self.completed.notify_all();
    }

    /// Returns the exclusive upper bound of the identities in a snapshot of the submissions registered so far.
    pub(super) fn snapshot(&self) -> u64 {
        self.lock().next_identity
    }

    /// Blocks until no submission registered before `end` is pending. Returns the locked state, so that the caller
    /// can read or remove the failures of that snapshot before any other thread does.
    pub(super) fn wait(&self, end: u64) -> MutexGuard<'_, WorkTrackerState<T, E>> {
        let mut state = self.lock();
        while state.pending.first_key_value().is_some_and(|(&identity, _)| identity < end) {
            state = self.completed.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state
    }

    /// Locks the state of this tracker. Completion callbacks run on PJRT threads and reservations complete while
    /// they are dropped, so no caller can afford to panic on a poisoned mutex. Every update leaves the state
    /// consistent, so a poisoned mutex is repaired and its state is used as it is.
    pub(super) fn lock(&self) -> MutexGuard<'_, WorkTrackerState<T, E>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.state.clear_poison();
        state
    }
}

impl<T, E> Default for WorkTracker<T, E> {
    fn default() -> Self {
        Self { state: Mutex::new(WorkTrackerState::default()), completed: Condvar::new() }
    }
}

/// Identities and results of the submissions of a [`WorkTracker`].
pub(super) struct WorkTrackerState<T, E> {
    /// Identity to use for the next registration in a standalone tracker.
    pub(super) next_identity: u64,

    /// Submissions that still have work to publish or finish.
    pub(super) pending: BTreeMap<u64, PendingWork<T, E>>,

    /// Failures of completed submissions, with the tags of those submissions, that the owner has yet to remove.
    pub(super) failures: BTreeMap<u64, (T, E)>,
}

impl<T, E> WorkTrackerState<T, E> {
    /// Adds the pending submission for an identity allocated while this state is locked.
    fn insert_pending(&mut self, identity: u64, tag: T) {
        self.pending.insert(identity, PendingWork { tag, remaining: 1, error: None });
    }
}

impl<T, E> Default for WorkTrackerState<T, E> {
    fn default() -> Self {
        Self { next_identity: 0, pending: BTreeMap::new(), failures: BTreeMap::new() }
    }
}

/// Tracks submissions across all handles of one [`Session`](crate::Session). Partitions submissions by thread across
/// several mutexes while sharing a registration sequence and reporting each failure once.
#[derive(Default)]
pub(super) struct SessionWorkTracker {
    /// Exclusive upper bound of the identities registered across all shards.
    next_identity: AtomicU64,

    /// Pending work and failures partitioned by the submitting thread.
    shards: [Arc<WorkTracker<(), Error>>; Self::SHARD_COUNT],

    /// Serializes removal of failures across shards after the caller's snapshot has completed.
    reporting: Mutex<()>,
}

impl SessionWorkTracker {
    /// Number of independent mutexes for session submissions.
    const SHARD_COUNT: usize = 8;

    /// Registers a submission before its first native enqueue.
    pub(super) fn reserve(&self) -> WorkReservation {
        SESSION_WORK_SHARD.with(|&shard| self.reserve_in_shard(shard))
    }

    /// Returns the exclusive upper bound of a snapshot of all registered submissions.
    pub(super) fn snapshot(&self) -> u64 {
        self.next_identity.load(Ordering::SeqCst)
    }

    /// Waits for the submissions in one snapshot and reports their failures in registration order.
    pub(super) fn synchronize(&self) -> Result<(), Error> {
        self.wait_for_snapshot(self.snapshot())
    }

    /// Returns whether any shard still has outstanding submissions.
    #[cfg(test)]
    pub(super) fn has_pending_work(&self) -> bool {
        self.shards.iter().any(|shard| !shard.lock().pending.is_empty())
    }

    /// Registers a submission in the given shard.
    fn reserve_in_shard(&self, shard: usize) -> WorkReservation {
        let work = &self.shards[shard];
        let mut state = work.lock();

        // Take the identity with the shard locked so a snapshot that includes it also observes the insertion.
        let identity = self.next_identity.fetch_add(1, Ordering::SeqCst);
        state.insert_pending(identity, ());

        // Keep reservation construction outside the shared lock.
        drop(state);

        WorkReservation {
            work: Arc::clone(work),
            identity: Some(identity),
            fence: None,
            events: Vec::new(),
            error: None,
        }
    }

    /// Waits for work registered before `end` and removes its failures from the session report.
    fn wait_for_snapshot(&self, end: u64) -> Result<(), Error> {
        let mut has_failures = false;

        for shard in &self.shards {
            let state = shard.wait(end);
            has_failures |= state.failures.first_key_value().is_some_and(|(&identity, _)| identity < end);
        }

        if !has_failures {
            return Ok(());
        }

        // Every shard has retired the snapshot. Claim its errors across the session as one operation.
        let reporting = self.reporting.lock().unwrap_or_else(PoisonError::into_inner);
        let mut failures = Vec::new();
        for shard in &self.shards {
            let mut state = shard.lock();
            let later_failures = state.failures.split_off(&end);
            let earlier_failures = std::mem::replace(&mut state.failures, later_failures);

            // Allow this shard to accept more work while the report collects its failures.
            drop(state);

            failures.extend(earlier_failures.into_iter().map(|(identity, ((), error))| (identity, error)));
        }

        // The claimed errors are independent of subsequent reports.
        drop(reporting);

        if failures.is_empty() {
            return Ok(());
        }

        failures.sort_unstable_by_key(|(identity, _)| *identity);
        Err(ExecutionError::SynchronizationFailures { failures }.into())
    }
}

thread_local! {
    /// Stable shard choice for each submitting thread. Completions use the original reservation's shard.
    static SESSION_WORK_SHARD: usize = {
        static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);
        NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SessionWorkTracker::SHARD_COUNT
    };
}

/// Completion for a registered execution or batch of transfers. Dropping it publishes whatever work was
/// enqueued, including on early returns. A reservation without any attached work completes immediately.
pub(crate) struct WorkReservation {
    /// Shard that owns this reservation and receives its completion notifications.
    work: Arc<WorkTracker<(), Error>>,

    /// Registration ID. Taking it prevents publication from running twice.
    identity: Option<u64>,

    /// Whole-execution fence of the submission, if one has been attached.
    fence: Option<ExecutionFence>,

    /// Native events of transfers that return no output payload.
    events: Vec<Event<()>>,

    /// First error while observing or validating work already handed off to PJRT.
    error: Option<Error>,
}

impl WorkReservation {
    /// Tracks the given completion independently of any value that carries it.
    pub(crate) fn track_fence(&mut self, fence: ExecutionFence) {
        self.fence = Some(match self.fence.take() {
            Some(previous) => ExecutionFence::join([previous, fence]),
            None => fence,
        });
    }

    /// Tracks native send or transfer events.
    pub(crate) fn track_events<E: IntoIterator<Item = Event<()>>>(&mut self, events: E) {
        self.events.extend(events);
    }

    /// Captures a destination buffer's ready event before the caller can drop or donate the buffer.
    pub(crate) fn track_buffer(&mut self, buffer: &Buffer<'_>) {
        match buffer.ready() {
            Ok(event) => self.events.push(event),
            Err(error) => self.track_error(error.into()),
        }
    }

    /// Observes a host copy through its native event while leaving the payload available through [`HostCopy::wait`].
    pub(crate) fn track_host_copy<O: Send + 'static>(&mut self, event: Event<O>) -> HostCopy<O> {
        if event.ready().unwrap_or(false) {
            let result = event.r#await();
            if let Err(error) = &result {
                self.track_error(error.clone().into());
            }
            return HostCopy(HostCopyCompletion::Ready(result));
        }

        // A reservation normally completes once, when its fence is ready. This copy cannot join that fence, because the
        // caller still needs the event to read the copied bytes out of it. The copy therefore reports its completion
        // through a callback on its own event, and the submission is made to wait for that one additional notification.
        // It is retired only after the fence and every such copy have reported, and it records the first failure among
        // them.
        let identity = self.identity.unwrap();
        self.work.lock().pending.get_mut(&identity).unwrap().remaining += 1;
        let work = Arc::downgrade(&self.work);
        let copy = Arc::new(PendingHostCopy::new());

        // The callback holds its own reference so an abandoned readback retains the event through completion.
        let callback_copy = Arc::clone(&copy);
        let registration = event.on_ready(move |error| {
            if let Some(work) = work.upgrade() {
                work.complete(identity, error.map(Error::from));
            }
            callback_copy.complete();
        });

        if let Err(registration_error) = registration {
            let result = event.r#await();
            let error = result.as_ref().err().cloned().unwrap_or(registration_error);
            self.work.complete(identity, Some(error.into()));
            return HostCopy(HostCopyCompletion::Ready(result));
        }

        // Registration can invoke the callback inline. Keep the event on this stack until registration returns.
        copy.install(event);
        HostCopy(HostCopyCompletion::Pending(copy))
    }

    /// Records a failure discovered after native enqueue. Any attached work still has to complete.
    pub(crate) fn track_error(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    /// Hands off completion tracking to the session and returns the combined [`ExecutionFence`], if any,
    /// without waiting for it. Fence construction can wait for native completion if PJRT callback registration fails.
    /// Cross-host sends have no local destination buffers, so the placement code attaches this fence to the returned
    /// [`Array`](crate::Array). This lets the array's readiness functions wait for the sends and report a send failure.
    /// Dropping the reservation is enough for [`Session::synchronize`](crate::Session::synchronize) to wait for the
    /// submission.
    pub(crate) fn finish(mut self) -> Option<ExecutionFence> {
        self.publish()
    }

    /// Publishes the terminal result once the attached work finishes.
    fn publish(&mut self) -> Option<ExecutionFence> {
        let identity = self.identity.take()?;
        if !self.events.is_empty() {
            let events = std::mem::take(&mut self.events);
            self.track_fence(ExecutionFence::new(events));
        }

        let fence = self.fence.take();
        let error = self.error.take();
        if let Some(fence) = &fence {
            let work = Arc::downgrade(&self.work);
            fence.on_ready(move |result| {
                if let Some(work) = work.upgrade() {
                    work.complete(identity, error.or_else(|| result.err().map(Error::from)));
                }
            });
        } else {
            self.work.complete(identity, error);
        }
        fence
    }
}

impl Drop for WorkReservation {
    fn drop(&mut self) {
        self.publish();
    }
}

/// A pending submission and its outstanding completion notifications.
pub(super) struct PendingWork<T, E> {
    /// Tag that the owner stored with the submission.
    pub(super) tag: T,

    /// Number of completion notifications that the submission still waits for. It starts at one, for the
    /// submission itself, and a [`WorkReservation`] adds one for each pending host copy that it tracks.
    pub(super) remaining: usize,

    /// First failure reported by a notification.
    pub(super) error: Option<E>,
}

/// A copy whose payload can be consumed independently of session synchronization.
pub(crate) struct HostCopy<O>(HostCopyCompletion<O>);

impl<O> HostCopy<O> {
    /// Returns the payload once the native copy completes.
    pub(crate) fn wait(self) -> Result<O, PjrtError> {
        match self.0 {
            HostCopyCompletion::Ready(result) => result,
            HostCopyCompletion::Pending(copy) => {
                // Await consumes the event and can return before the callback runs. Wait for the callback first
                // so the event stays alive until it reports the copy's result.
                let mut state = copy.state.lock().unwrap_or_else(PoisonError::into_inner);
                while matches!(*state, HostCopyState::Pending(_)) {
                    state = copy.completed.wait(state).unwrap_or_else(PoisonError::into_inner);
                }

                let HostCopyState::Ready(event) = std::mem::replace(&mut *state, HostCopyState::Taken) else {
                    unreachable!("a readback must have its event installed before waiting");
                };

                // Release the state lock before calling into PJRT.
                drop(state);

                event.r#await()
            }
        }
    }
}

/// Payload access for a session-tracked host copy.
enum HostCopyCompletion<O> {
    /// The event and its callback are pending when tracking starts.
    Pending(Arc<PendingHostCopy<O>>),

    /// Output of an already completed copy or of a synchronous fallback after callback registration failed.
    Ready(Result<O, PjrtError>),
}

/// Holds the native event until its callback reports completion and the caller can consume the payload.
struct PendingHostCopy<O> {
    /// Event ownership and callback completion, accessed under the same lock.
    state: Mutex<HostCopyState<O>>,

    /// Wakes the caller after the callback records completion.
    completed: Condvar,
}

impl<O> PendingHostCopy<O> {
    /// Creates shared state before registering the event's callback.
    fn new() -> Self {
        Self { state: Mutex::new(HostCopyState::Registering), completed: Condvar::new() }
    }

    /// Stores the event after registration, preserving any completion that arrived inline.
    fn install(&self, event: Event<O>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *state = match *state {
            HostCopyState::Registering => HostCopyState::Pending(event),
            HostCopyState::CompletedBeforeInstallation => HostCopyState::Ready(event),
            HostCopyState::Pending(_) | HostCopyState::Ready(_) | HostCopyState::Taken => {
                unreachable!("host copy event already installed")
            }
        };
    }

    /// Records callback completion and wakes the caller if it is waiting for the event.
    fn complete(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *state = match std::mem::replace(&mut *state, HostCopyState::CompletedBeforeInstallation) {
            HostCopyState::Registering => HostCopyState::CompletedBeforeInstallation,
            HostCopyState::Pending(event) => HostCopyState::Ready(event),
            completed @ (HostCopyState::CompletedBeforeInstallation
            | HostCopyState::Ready(_)
            | HostCopyState::Taken) => completed,
        };

        // Release the state lock before waking the reader.
        drop(state);

        self.completed.notify_one();
    }
}

/// Event ownership and callback completion for a [`PendingHostCopy`].
enum HostCopyState<O> {
    /// The submitting thread holds the event while it registers the callback.
    Registering,

    /// The event is installed and its callback has yet to report completion.
    Pending(Event<O>),

    /// The callback reported completion while the submitting thread still holds the event.
    CompletedBeforeInstallation,

    /// The event is installed and its callback has reported completion. The reader can take the event.
    Ready(Event<O>),

    /// The reader has taken the completed event to retrieve its payload.
    Taken,
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;
    use std::sync::Arc;
    use std::sync::mpsc::channel;
    use std::thread;
    use std::time::Duration;

    use pretty_assertions::assert_eq;

    use ryft_pjrt::{Error as PjrtError, ExecutionFence};

    use crate::tests::execution_client;
    use crate::{Error, ExecutionError};

    use super::{HostCopy, HostCopyCompletion, HostCopyState, PendingHostCopy, SessionWorkTracker, WorkTracker};

    #[test]
    fn test_work_tracker_register() {
        let tracker = WorkTracker::<bool, &str>::default();
        assert_eq!(tracker.register(true), 0);
        assert_eq!(tracker.register(false), 1);

        let state = tracker.lock();
        assert_eq!(
            state
                .pending
                .iter()
                .map(|(&identity, pending)| (identity, pending.tag, pending.remaining, pending.error))
                .collect::<Vec<_>>(),
            vec![(0, true, 1, None), (1, false, 1, None)],
        );
        assert!(state.failures.is_empty());
    }

    #[test]
    fn test_work_tracker_complete() {
        let tracker = WorkTracker::<bool, &str>::default();
        let successful = tracker.register(false);
        let failed = tracker.register(true);
        let dependent = tracker.register(false);

        // Submissions complete in any order. A success is retired at once, while a failure is kept with its tag.
        tracker.complete(failed, Some("failed"));
        tracker.complete(successful, None);
        assert_eq!(tracker.lock().pending.keys().copied().collect::<Vec<_>>(), vec![dependent]);
        assert_eq!(tracker.lock().failures.iter().collect::<Vec<_>>(), vec![(&failed, &(true, "failed"))]);

        // A submission that waits for two notifications is retired by the second and keeps the first failure.
        tracker.lock().pending.get_mut(&dependent).unwrap().remaining += 1;
        tracker.complete(dependent, Some("first"));
        assert_eq!(tracker.lock().pending.keys().copied().collect::<Vec<_>>(), vec![dependent]);
        tracker.complete(dependent, Some("second"));
        assert!(tracker.lock().pending.is_empty());
        assert_eq!(tracker.lock().failures.get(&dependent), Some(&(false, "first")));

        // Notifications for submissions that are no longer pending are ignored.
        tracker.complete(successful, Some("late"));
        assert_eq!(tracker.lock().failures.keys().copied().collect::<Vec<_>>(), vec![failed, dependent]);
    }

    #[test]
    fn test_work_tracker_snapshot() {
        let tracker = WorkTracker::<(), ()>::default();
        assert_eq!(tracker.snapshot(), 0);
        let first = tracker.register(());
        let snapshot = tracker.snapshot();
        let later = tracker.register(());
        assert_eq!(snapshot, later);

        // Retiring submissions leaves the exclusive registration boundary in place.
        tracker.complete(first, None);
        tracker.complete(later, None);
        assert_eq!(tracker.snapshot(), 2);
    }

    #[test]
    fn test_work_tracker_wait() {
        let tracker = Arc::new(WorkTracker::<(), &str>::default());
        let first = tracker.register(());
        let snapshot = tracker.snapshot();
        let later = tracker.register(());
        let (sender, receiver) = channel();
        let waiting_tracker = Arc::clone(&tracker);
        let waiter = thread::spawn(move || {
            let state = waiting_tracker.wait(snapshot);
            let failures = state.failures.keys().copied().collect::<Vec<_>>();
            let pending = state.pending.keys().copied().collect::<Vec<_>>();
            sender.send((failures, pending)).unwrap();
        });

        // The waiter must observe the first failure while the later submission is still pending.
        tracker.complete(first, Some("failed"));
        let result = receiver.recv_timeout(Duration::from_secs(5));
        tracker.complete(later, None);
        assert_eq!(result, Ok((vec![first], vec![later])));
        waiter.join().unwrap();
    }

    #[test]
    fn test_work_tracker_lock() {
        let tracker = WorkTracker::<bool, &str>::default();
        let identity = tracker.register(true);
        let poisoned = catch_unwind(|| {
            let _state = tracker.state.lock().unwrap();
            panic!("fixture poisoned tracker state");
        });
        assert_eq!(poisoned.unwrap_err().downcast_ref::<&str>(), Some(&"fixture poisoned tracker state"));
        assert!(tracker.state.is_poisoned());

        let state = tracker.lock();
        assert!(!tracker.state.is_poisoned());
        let pending = state.pending.get(&identity).unwrap();
        assert_eq!((pending.tag, pending.remaining, pending.error), (true, 1, None));
        assert!(state.failures.is_empty());
    }

    #[test]
    fn test_session_work_tracker_reserve() {
        let work = SessionWorkTracker::default();
        let reservation = work.reserve();
        let shard = Arc::clone(&reservation.work);
        assert_eq!(shard.lock().pending.get(&0).map(|pending| pending.remaining), Some(1));

        // Canceling before enqueue retires the registration without reporting a failure.
        drop(reservation);
        let state = shard.lock();
        assert!(state.pending.is_empty());
        assert!(state.failures.is_empty());
    }

    #[test]
    fn test_session_work_tracker_synchronize() {
        let work = SessionWorkTracker::default();
        let mut first = work.reserve_in_shard(1);
        let mut second = work.reserve_in_shard(0);
        let snapshot = work.snapshot();
        let mut later = work.reserve_in_shard(0);

        // Failures arrive in reverse registration order, and the latest failure belongs to a later snapshot.
        later.track_error(PjrtError::aborted("later failed").into());
        drop(later);
        second.track_error(PjrtError::aborted("second failed").into());
        drop(second);
        first.track_error(PjrtError::aborted("first failed").into());
        drop(first);

        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) =
            work.wait_for_snapshot(snapshot)
        else {
            panic!("synchronization should report both failures");
        };
        assert!(matches!(
            &failures[..],
            [(0, Error::Pjrt(PjrtError::Aborted { message: first, .. })),
             (1, Error::Pjrt(PjrtError::Aborted { message: second, .. }))]
                if first == "first failed" && second == "second failed",
        ));
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(2, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "later failed"),
        ));
        work.synchronize().unwrap();
    }

    #[test]
    fn test_session_work_tracker_synchronize_concurrent_snapshots() {
        let work = SessionWorkTracker::default();
        let mut first = work.reserve_in_shard(1);
        let mut second = work.reserve_in_shard(0);
        let snapshot = work.snapshot();
        let later = work.reserve_in_shard(0);

        thread::scope(|threads| {
            let (sender, receiver) = channel();
            for _ in 0..2 {
                let sender = sender.clone();
                let work = &work;
                threads.spawn(move || sender.send(work.wait_for_snapshot(snapshot)).unwrap());
            }
            first.track_error(PjrtError::aborted("first failed").into());
            second.track_error(PjrtError::aborted("second failed").into());
            drop(second);
            drop(first);

            // Both callers must return while the later reservation is pending. One caller claims both failures.
            let first_result = receiver.recv_timeout(Duration::from_secs(5));
            let second_result = receiver.recv_timeout(Duration::from_secs(5));
            drop(later);
            let (Ok(()), Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))) =
                (match (first_result.unwrap(), second_result.unwrap()) {
                    (Err(error), Ok(())) => (Ok(()), Err(error)),
                    results => results,
                })
            else {
                panic!("one caller should claim both failures");
            };
            assert_eq!(failures.into_iter().map(|(identity, _)| identity).collect::<Vec<_>>(), vec![0, 1]);
        });
        work.synchronize().unwrap();
    }

    #[test]
    fn test_work_reservation_track_host_copy() {
        let client = execution_client();
        let work = SessionWorkTracker::default();
        let (event, promise) = client.event(vec![1, 2, 3]).unwrap();
        let abandoned_payload = Arc::new(());
        let retained_payload = Arc::downgrade(&abandoned_payload);
        let (abandoned_event, abandoned_promise) = client.event(abandoned_payload).unwrap();
        let mut reservation = work.reserve_in_shard(0);
        let copy = reservation.track_host_copy(event);
        let abandoned = reservation.track_host_copy(abandoned_event);
        drop(reservation);
        drop(abandoned);

        // The callback retains the abandoned copy's event and payload while the copy is pending.
        assert_eq!(retained_payload.strong_count(), 1);
        assert_eq!(work.shards[0].lock().pending.get(&0).map(|pending| pending.remaining), Some(2));

        // The held copy can complete without its consumer, and its payload stays available after synchronization.
        promise.set(None).unwrap();
        abandoned_promise.set(Some(PjrtError::aborted("abandoned copy failed"))).unwrap();
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "abandoned copy failed"),
        ));
        assert_eq!(copy.wait(), Ok(vec![1, 2, 3]));
    }

    #[test]
    fn test_work_reservation_track_host_copy_already_ready() {
        let work = SessionWorkTracker::default();
        let (event, promise) = execution_client().event(()).unwrap();
        promise.set(Some(PjrtError::aborted("copy failed before tracking"))).unwrap();
        let mut reservation = work.reserve_in_shard(0);
        let copy = reservation.track_host_copy(event);

        // A ready readback reports its native error without adding a pending dependency.
        assert_eq!(work.shards[0].lock().pending.get(&0).map(|pending| pending.remaining), Some(1));
        assert!(matches!(
            copy.wait(),
            Err(PjrtError::Aborted { message, .. }) if message == "copy failed before tracking",
        ));
        drop(reservation);
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "copy failed before tracking"),
        ));
    }

    #[test]
    fn test_work_reservation_finish() {
        let client = execution_client();
        let work = SessionWorkTracker::default();
        let (first_event, first_promise) = client.event(()).unwrap();
        let (second_event, second_promise) = client.event(()).unwrap();
        let second_fence = ExecutionFence::new(vec![second_event]);
        let mut reservation = work.reserve_in_shard(0);
        reservation.track_fence(ExecutionFence::new(vec![first_event]));
        reservation.track_fence(second_fence.clone());
        let fence = reservation.finish().unwrap();

        // The second fence can fail while the joined fence still waits for the first.
        second_promise.set(Some(PjrtError::aborted("second fence failed"))).unwrap();
        let error = second_fence.block_until_ready().unwrap_err();
        assert_eq!(fence.is_ready(), Ok(false));
        assert!(work.shards[0].lock().pending.contains_key(&0));

        // The returned fence and the session tracker both observe the joined completion.
        first_promise.set(None).unwrap();
        assert_eq!(fence.block_until_ready(), Err(error));
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "second fence failed"),
        ));
    }

    #[test]
    fn test_work_reservation_drop() {
        let work = SessionWorkTracker::default();
        let (event, promise) = execution_client().event(()).unwrap();
        let mut reservation = work.reserve_in_shard(0);
        reservation.track_events([event]);

        // Leaving the submission early still tracks the transfer already enqueued.
        drop(reservation);
        assert!(work.shards[0].lock().pending.contains_key(&0));
        promise.set(Some(PjrtError::aborted("copy failed"))).unwrap();
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "copy failed"),
        ));
    }

    #[test]
    fn test_host_copy_wait() {
        let (event, promise) = execution_client().event(42).unwrap();
        let copy = Arc::new(PendingHostCopy::new());
        promise.set(None).unwrap();

        // Simulate an inline callback by reporting completion before installing the event.
        copy.complete();
        assert!(matches!(*copy.state.lock().unwrap(), HostCopyState::CompletedBeforeInstallation));
        copy.install(event);
        assert!(matches!(*copy.state.lock().unwrap(), HostCopyState::Ready(_)));
        assert_eq!(HostCopy(HostCopyCompletion::Pending(Arc::clone(&copy))).wait(), Ok(42));
        assert!(matches!(*copy.state.lock().unwrap(), HostCopyState::Taken));
    }
}
