use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use ryft_pjrt::{Buffer, Error as PjrtError, Event, ExecutionFence};

use crate::{Error, ExecutionError};

// TODO(eaplatanios): Review from here onwards.

/// Tracks submissions from the moment they are registered until they complete. An owner registers each submission
/// before its first native enqueue, so a wait over a snapshot of the registered identities also covers submissions
/// that are still being prepared. Successful submissions are retired as soon as they complete, while failures are
/// kept in registration order until the owner removes them.
///
/// A [`Session`](crate::Session) owns one tracker for the work of all of its handles and reports each failure once.
/// Each [`EffectScope`](crate::sessions::effects::EffectScope) owns another one for its executions with effects and
/// keeps their failures until they are acknowledged. `T` is a tag that the owner stores with each submission, and `E`
/// is the failure that a submission may record.
pub(super) struct WorkTracker<T, E> {
    /// Registration order, pending submissions, and failures that the owner has yet to remove.
    state: Mutex<WorkTrackerState<T, E>>,

    /// Wakes waiting callers when a submission completes.
    completed: Condvar,
}

impl<T, E> WorkTracker<T, E> {
    /// Registers a submission with the provided `tag` and returns its identity. The submission stays pending until
    /// [`Self::complete`] has been called once for it and once for each dependency that was added to it.
    pub(super) fn register(&self, tag: T) -> u64 {
        let mut state = self.lock();
        let identity = state.next_identity;
        state.next_identity += 1;
        state.pending.insert(identity, PendingWork { tag, remaining: 1, error: None });
        identity
    }

    /// Records a copy that the pending submission `identity` will observe through its own event callback.
    fn add_dependency(&self, identity: u64) {
        self.lock().pending.get_mut(&identity).unwrap().remaining += 1;
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

impl WorkTracker<(), Error> {
    /// Registers a submission before its first native enqueue.
    pub(super) fn reserve(self: &Arc<Self>) -> WorkReservation {
        WorkReservation {
            work: Arc::clone(self),
            identity: Some(self.register(())),
            fence: None,
            events: Vec::new(),
            error: None,
        }
    }

    /// Waits for every submission already registered and reports its failures in registration order.
    pub(super) fn synchronize(&self) -> Result<(), Error> {
        self.wait_for_snapshot(self.snapshot())
    }

    /// Waits for work registered before `end` and removes its failures from the session report.
    fn wait_for_snapshot(&self, end: u64) -> Result<(), Error> {
        let mut state = self.wait(end);
        let later_failures = state.failures.split_off(&end);
        let failures = std::mem::replace(&mut state.failures, later_failures);
        // Convert errors after releasing the tracker mutex.
        drop(state);

        if failures.is_empty() {
            Ok(())
        } else {
            let failures = failures.into_iter().map(|(identity, ((), error))| (identity, error)).collect();
            Err(ExecutionError::SynchronizationFailures { failures }.into())
        }
    }
}

impl<T, E> Default for WorkTracker<T, E> {
    fn default() -> Self {
        Self { state: Mutex::new(WorkTrackerState::default()), completed: Condvar::new() }
    }
}

/// Completion for a registered execution or batch of transfers. Dropping it publishes whatever work was
/// enqueued, including on early returns. A reservation without any attached work completes immediately.
pub(crate) struct WorkReservation {
    /// Tracker shared by the submitting handle and every other handle of its session.
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
    pub(crate) fn track_events(&mut self, events: impl IntoIterator<Item = Event<()>>) {
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

        let identity = self.identity.unwrap();
        self.work.add_dependency(identity);
        let work = Arc::downgrade(&self.work);
        let event_slot = Arc::new(Mutex::new(None));
        let retained_event = Arc::clone(&event_slot);
        let (ready, completion) = channel();
        let registration = event.on_ready(move |error| {
            if let Some(work) = work.upgrade() {
                work.complete(identity, error.map(Error::from));
            }
            // A caller may discard its copy handle before completion. Keep the event alive through this callback.
            drop(retained_event);
            // A disconnected receiver means the caller discarded its copy.
            ready.send(()).unwrap_or(());
        });
        if let Err(registration_error) = registration {
            let result = event.r#await();
            let error = result.as_ref().err().cloned().unwrap_or(registration_error);
            self.work.complete(identity, Some(error.into()));
            return HostCopy(HostCopyCompletion::Ready(result));
        }

        // Registration can invoke the callback inline. Keep the event on this stack until registration returns.
        *event_slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(event);
        HostCopy(HostCopyCompletion::Pending { event: event_slot, completion })
    }

    /// Records a failure discovered after native enqueue. Any attached work still has to complete.
    pub(crate) fn track_error(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    /// Publishes completion and returns the fence for callers that also need to attach it to an output.
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

/// Identities and results of the submissions of a [`WorkTracker`].
pub(super) struct WorkTrackerState<T, E> {
    /// Identity to use for the next registration.
    next_identity: u64,

    /// Submissions that still have work to publish or finish.
    pub(super) pending: BTreeMap<u64, PendingWork<T, E>>,

    /// Failures of completed submissions, with the tags of those submissions, that the owner has yet to remove.
    pub(super) failures: BTreeMap<u64, (T, E)>,
}

impl<T, E> Default for WorkTrackerState<T, E> {
    fn default() -> Self {
        Self { next_identity: 0, pending: BTreeMap::new(), failures: BTreeMap::new() }
    }
}

/// A pending submission and its outstanding completion notifications.
pub(super) struct PendingWork<T, E> {
    /// Tag that the owner stored with the submission.
    pub(super) tag: T,

    /// One notification for the submission itself and one for each dependency that was added to it (e.g., each
    /// subscribed host copy of a [`WorkReservation`]).
    remaining: usize,

    /// First failure reported by a notification.
    error: Option<E>,
}

/// A copy whose payload can be consumed independently of session synchronization.
pub(crate) struct HostCopy<O>(HostCopyCompletion<O>);

impl<O> HostCopy<O> {
    /// Returns the payload once the native copy completes.
    pub(crate) fn wait(self) -> Result<O, PjrtError> {
        match self.0 {
            HostCopyCompletion::Ready(result) => result,
            HostCopyCompletion::Pending { event, completion } => {
                completion.recv().map_err(|_| PjrtError::internal("host copy callback dropped before completion"))?;
                let event = event.lock().unwrap_or_else(PoisonError::into_inner).take().unwrap();
                event.r#await()
            }
        }
    }
}

/// Payload access for a session-tracked host copy.
enum HostCopyCompletion<O> {
    /// The callback holds a share of the event slot until it has reported completion.
    Pending {
        /// Event carrying the payload, consumed only by the caller.
        event: Arc<Mutex<Option<Event<O>>>>,

        /// Notifies the caller that it can take the event and read its payload.
        completion: Receiver<()>,
    },

    /// Output of an already completed copy or of a synchronous fallback after callback registration failed.
    Ready(Result<O, PjrtError>),
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;
    use std::sync::Arc;
    use std::sync::mpsc::channel;
    use std::thread;
    use std::time::Duration;

    use ryft_core::{ArrayType, DataType, Neg};
    use ryft_pjrt::{Error as PjrtError, ExecutionFence};

    use crate::sessions::tests::{domain_mesh, session_with_client};
    use crate::tests::{cpu_client, execution_client, values_to_bytes};
    use crate::{Array, DevicePutTarget, Error, ExecutionError};

    use super::WorkTracker;

    #[test]
    fn test_work_tracker_register() {
        let tracker = WorkTracker::<bool, &str>::default();
        assert_eq!(tracker.snapshot(), 0);
        assert_eq!(tracker.register(true), 0);
        assert_eq!(tracker.register(false), 1);
        assert_eq!(tracker.snapshot(), 2);
        let state = tracker.lock();
        assert_eq!(
            state.pending.iter().map(|(&identity, pending)| (identity, pending.tag)).collect::<Vec<_>>(),
            vec![(0, true), (1, false)]
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

        // A submission with a dependency waits for both notifications and keeps the first failure among them.
        tracker.add_dependency(dependent);
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
    fn test_work_tracker_wait() {
        let tracker = WorkTracker::<bool, &str>::default();
        let first = tracker.register(true);
        let snapshot = tracker.snapshot();
        let later = tracker.register(false);
        let (sender, receiver) = channel();
        let result = thread::scope(|threads| {
            let tracker = &tracker;
            threads.spawn(move || {
                let failures = tracker.wait(snapshot).failures.keys().copied().collect::<Vec<_>>();
                sender.send(failures).unwrap();
            });

            // Only the first submission belongs to the snapshot, so its completion releases the waiter.
            tracker.complete(first, Some("failed"));
            let result = receiver.recv_timeout(Duration::from_secs(5));

            // Complete the later submission so that the waiter can return even if it missed the deadline.
            tracker.complete(later, None);
            result
        });
        assert_eq!(result, Ok(vec![first]));
    }

    #[test]
    fn test_work_tracker_lock() {
        let tracker = WorkTracker::<bool, &str>::default();
        let identity = tracker.register(true);
        let poisoned = catch_unwind(|| {
            let _state = tracker.state.lock().unwrap();
            panic!("poison tracker state");
        });
        assert!(poisoned.is_err());
        assert!(tracker.state.is_poisoned());

        // Locking repairs the mutex, so completion callbacks and waiters keep working after a panic.
        tracker.complete(identity, Some("failed"));
        assert!(!tracker.state.is_poisoned());
        assert_eq!(tracker.wait(tracker.snapshot()).failures.get(&identity), Some(&(true, "failed")));
    }

    #[test]
    fn test_work_tracker_synchronize_waits_for_its_snapshot() {
        let work = Arc::new(WorkTracker::default());
        let mut first = work.reserve();
        let canceled = work.reserve();
        let mut second = work.reserve();
        let snapshot = work.snapshot();
        let mut later = work.reserve();
        let (event, promise) = execution_client().event(()).unwrap();
        let (sender, receiver) = channel();

        let result = thread::scope(|threads| {
            threads.spawn(|| sender.send(work.wait_for_snapshot(snapshot)).unwrap());

            // Complete in a different order from registration. The canceled entry has no completion to publish.
            second.track_error(PjrtError::aborted("second failed").into());
            drop(second);
            drop(canceled);
            first.track_fence(ExecutionFence::new(vec![event]));
            drop(first);
            promise.set(Some(PjrtError::aborted("first failed"))).unwrap();

            // Release the later reservation even if the waiter fails to return within the deadline.
            let result = receiver.recv_timeout(Duration::from_secs(5));
            later.track_error(PjrtError::aborted("later failed").into());
            drop(later);
            result
        });
        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) = result.unwrap() else {
            panic!("synchronization should report both failures");
        };
        assert!(matches!(
            &failures[..],
            [(0, Error::Pjrt(PjrtError::Aborted { message: first, .. })),
             (2, Error::Pjrt(PjrtError::Aborted { message: second, .. }))]
                if first == "first failed" && second == "second failed",
        ));
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(3, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "later failed"),
        ));
        work.synchronize().unwrap();
    }

    #[test]
    fn test_work_reservation_drop_publishes_partial_batch() {
        let work = Arc::new(WorkTracker::default());
        let (event, promise) = execution_client().event(()).unwrap();
        let mut reservation = work.reserve();
        reservation.track_events([event]);
        // Leaving the submission early still publishes completion for the transfer already enqueued.
        drop(reservation);
        assert_eq!(work.lock().pending.first_key_value().map(|(&identity, _)| identity), Some(0));
        promise.set(Some(PjrtError::aborted("copy failed"))).unwrap();
        assert!(matches!(
            work.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(0, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                    if message == "copy failed"),
        ));
    }

    #[test]
    fn test_work_reservation_track_host_copy() {
        let client = execution_client();
        let work = Arc::new(WorkTracker::default());
        let (event, promise) = client.event(vec![1, 2, 3]).unwrap();
        let (abandoned_event, abandoned_promise) = client.event(vec![4]).unwrap();
        let mut reservation = work.reserve();
        let copy = reservation.track_host_copy(event);
        let abandoned = reservation.track_host_copy(abandoned_event);
        drop(reservation);
        drop(abandoned);
        assert_eq!(work.lock().pending.get(&0).map(|pending| pending.remaining), Some(2));

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
    fn test_work_tracker_reserve_tracks_execution_and_transfers() {
        let client = cpu_client(2);
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 2);
        let (source_mesh, _) = DevicePutTarget::device(mesh.devices()[0]).resolve(1).unwrap();
        let scoped = session.with_new_effect_scope().with_device_mesh(source_mesh);

        let input = Array::from_host_buffer(
            &scoped,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1.0f32, 2.0]),
        )
        .unwrap();
        assert_eq!(session.state.work.snapshot(), 1);

        let output = Neg::neg(&input).unwrap();
        assert_eq!(session.state.work.snapshot(), 2);

        let placed = output.into_placement(&scoped, DevicePutTarget::device(mesh.devices()[1])).unwrap();
        assert_eq!(session.state.work.snapshot(), 3);

        let copy = placed.begin_copy_to_host().unwrap();
        assert_eq!(session.state.work.snapshot(), 4);
        drop((input, placed, copy));
        session.synchronize().unwrap();
    }
}
