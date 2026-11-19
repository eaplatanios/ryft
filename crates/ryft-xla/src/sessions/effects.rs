use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use ryft_core::DeviceId;
use ryft_pjrt::{Buffer, ExecutionFence};

use crate::sessions::synchronization::WorkTracker;
use crate::{Error, ExecutionError};

/// Tracks executions with effects and keeps ordered I/O in reservation order. Cloning a [`Session`](crate::Session)
/// shares its effect scope. Use [`Session::with_new_effect_scope`](crate::Session::with_new_effect_scope) to give a
/// handle independent ordering and completion tracking. The same compiled program can run in either scope. Each
/// execution with effects reserves a place in the scope before dispatch. Ordered executions pass device tokens from
/// one execution to the next, preserving effect order even when device work runs asynchronously. The scope also tracks
/// completion of unordered executions. [`Self::barrier`] waits for work already reserved when the barrier takes its
/// snapshot and reports failures in reservation order. [`Self::acknowledge`] clears reported failures once the
/// affected work has completed.
///
/// Completion is tracked by a [`WorkTracker`], which is the same mechanism that tracks the work of a whole session
/// for [`Session::synchronize`](crate::Session::synchronize). The two differ in how long they keep a failure. A session
/// reports each failure once, while a scope keeps its failures until they are acknowledged, and an ordered failure
/// blocks further ordered reservations until then. The scope itself adds the ordered chain of token publications.
#[derive(Default)]
pub(super) struct EffectScope {
    /// Ordered chain and recovery state, which is never held across PJRT calls or device waits. A thread that holds
    /// this mutex may lock the tracker, but never the other way around.
    state: Mutex<EffectScopeState>,

    /// Reservations of this scope and their completion. The tag of a reservation records whether it is ordered.
    /// The fence callbacks of published reservations share the tracker, which holds no client resources.
    work: Arc<WorkTracker<bool, Arc<str>>>,
}

impl EffectScope {
    /// Reserves an execution and captures its ordered predecessor, if any, while holding the scope mutex.
    ///
    /// # Parameters
    ///
    ///   - `devices`: [`None`] tracks completion without joining the ordered I/O chain. For ordered I/O, pass the
    ///     IDs of the addressable devices participating in this execution, in the order of PJRT's per-device input
    ///     and output groups. There is one token buffer per device in that same order. The scope copies the list when
    ///     starting a chain and compares later lists against it so each successor can pass predecessor tokens back
    ///     to the devices that produced them.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::ScopeDeviceMismatch`] if the device IDs or their order differ from the chain's
    /// current list. Returns [`ExecutionError::ScopeNeedsRecovery`] for an ordered reservation after an ordered
    /// failure has been recorded and before it is [acknowledged](Self::acknowledge). Returns
    /// [`ExecutionError::ReentrantDispatch`] from within a managed effect callback.
    pub(super) fn reserve(&self, devices: Option<&[DeviceId]>) -> Result<EffectReservation, Error> {
        ensure_effect_dispatch_is_allowed()?;

        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        if devices.is_some() && self.work.lock().failures.values().any(|(ordered, _)| *ordered) {
            return Err(ExecutionError::ScopeNeedsRecovery.into());
        }

        if let Some(devices) = devices {
            match state.devices.as_deref() {
                Some(previous) if previous != devices => {
                    return Err(ExecutionError::ScopeDeviceMismatch {
                        expected: previous.to_vec(),
                        actual: devices.to_vec(),
                    }
                    .into());
                }
                Some(_) => {}
                None => state.devices = Some(devices.to_vec()),
            }
        }

        let identity = self.work.register(devices.is_some());
        let submission = Arc::new(EffectSubmission {
            publication: Mutex::new(EffectTokenPublication::Pending),
            publication_ready: Condvar::new(),
        });
        let predecessor = if devices.is_some() { state.tail.replace(Arc::clone(&submission)) } else { None };

        Ok(EffectReservation { work: Arc::clone(&self.work), identity, submission, predecessor, committed: false })
    }

    /// Takes a snapshot of the scope's current reservations, then waits for each one to finish or be canceled.
    /// This includes reservations still preparing device work. Later reservations are handled by a subsequent
    /// barrier. Reported failures remain in the scope until [`Self::acknowledge`] clears them.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::EffectSubmissionFailures`] with all failures from the snapshot in reservation
    /// order. Returns [`ExecutionError::ReentrantDispatch`] from within a managed effect callback.
    pub(super) fn barrier(&self) -> Result<(), Error> {
        ensure_effect_dispatch_is_allowed()?;
        self.barrier_at(self.work.snapshot())
    }

    /// Implements [`Self::barrier`] for the snapshot of the reservations that were made before `end`.
    fn barrier_at(&self, end: u64) -> Result<(), Error> {
        let failures = self
            .work
            .wait(end)
            .failures
            .range(..end)
            .map(|(&identity, (_, reason))| (identity, Arc::clone(reason)))
            .collect::<Vec<_>>();

        // The tracker has been released by now. Reservations and acknowledgments lock the scope first.
        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        state.observed_end = state.observed_end.max(end);
        drop(state);

        if failures.is_empty() {
            return Ok(());
        }

        Err(ExecutionError::EffectSubmissionFailures { failures }.into())
    }

    /// Clears failures reported by [`Self::barrier`] without waiting for pending work. If an ordered execution
    /// failed, this also resets the ordered chain so the next ordered reservation can choose a new device list.
    /// Pending unordered work remains tracked.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::PrematureAcknowledgment`] if any recorded failure has yet to be reported by a
    /// barrier, or if an ordered failure has been recorded and any ordered reservation is still pending. Returns
    /// [`ExecutionError::ReentrantDispatch`] from within a managed effect callback.
    pub(super) fn acknowledge(&self) -> Result<(), Error> {
        ensure_effect_dispatch_is_allowed()?;

        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        let mut work = self.work.lock();

        // A barrier reports every failure of its snapshot, so only failures of later reservations are unreported.
        if work.failures.last_key_value().is_some_and(|(&identity, _)| identity >= state.observed_end) {
            return Err(ExecutionError::PrematureAcknowledgment.into());
        }

        let ordered_failure = work.failures.values().any(|(ordered, _)| *ordered);
        if ordered_failure && work.pending.values().any(|pending| pending.tag) {
            return Err(ExecutionError::PrematureAcknowledgment.into());
        }

        let failures = std::mem::take(&mut work.failures);
        drop(work);
        let tail = if ordered_failure {
            state.devices = None;
            state.tail.take()
        } else {
            None
        };

        // Unlock the scope before freeing the acknowledged failures and destroying the token buffers
        // of the former tail.
        drop(state);
        drop((failures, tail));

        Ok(())
    }
}

/// Ordered chain and recovery state protected by the [`EffectScope`]'s mutex. The scope's [`WorkTracker`] tracks
/// the completion of its reservations independently of this state.
#[derive(Default)]
struct EffectScopeState {
    /// Latest ordered invocation, retained even after it has completed successfully.
    tail: Option<Arc<EffectSubmission>>,

    /// End of the snapshot of the latest barrier. A barrier reports every failure of its snapshot, so the failures
    /// of the reservations that were made before this identity have all been reported.
    observed_end: u64,

    /// Device list in token order for the current ordered chain. Acknowledging an ordered failure clears the list.
    devices: Option<Vec<DeviceId>>,
}

/// Reservation for an execution with effects. Dropping an uncommitted reservation cancels it and lets its successors
/// follow its predecessor's token publication. Cancellation can proceed without waiting for the predecessor. A canceled
/// reservation completes without a failure, so a barrier stops waiting for it.
pub(crate) struct EffectReservation {
    /// Tracker of the reserving scope, which records the completion of this reservation.
    work: Arc<WorkTracker<bool, Arc<str>>>,

    /// Identity of this reservation in [`Self::work`], which is local to its scope.
    identity: u64,

    /// Invocation being prepared or published.
    submission: Arc<EffectSubmission>,

    /// Predecessor whose token publication successors will follow if this reservation is canceled.
    predecessor: Option<Arc<EffectSubmission>>,

    /// Whether publication and callback registration have disarmed cancellation on drop.
    committed: bool,
}

impl EffectReservation {
    /// Waits for the predecessor's token publication on the host, following any canceled reservations. Use the
    /// returned token buffers as inputs even if they are still pending on their devices. Returns [`None`] when there
    /// are no predecessor tokens and propagates any failure published by the predecessor.
    pub(crate) fn wait_for_predecessor_tokens(&self) -> Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>> {
        self.predecessor.as_ref().map_or(Ok(None), |previous| previous.wait_for_tokens())
    }

    /// Consumes the reservation and publishes its ordered I/O token buffers or execution boundary failure to its
    /// successors. Pass `Ok(None)` for a successful submission without ordered tokens. The fence callback records
    /// the boundary failure if one was published, or the fence's result otherwise, in the tracker of the scope. It
    /// shares that tracker independently of the reservation and of the scope, and it can run inline if the fence is
    /// already ready. Cancellation on drop is disarmed only after callback registration returns.
    pub(crate) fn publish(
        mut self,
        tokens: Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>,
        fence: &ExecutionFence,
    ) {
        let boundary_failure = tokens.as_ref().err().cloned();
        self.submission.publish(tokens);
        let (work, identity) = (Arc::clone(&self.work), self.identity);
        fence.on_ready(move |result| {
            let failure = boundary_failure.or_else(|| result.err().map(|error| Arc::from(error.to_string())));
            work.complete(identity, failure);
        });
        self.committed = true;
    }
}

impl Drop for EffectReservation {
    fn drop(&mut self) {
        if !self.committed {
            // Cancellation can forward immediately, even when the predecessor is still being prepared.
            let mut publication = self.submission.publication.lock().unwrap_or_else(PoisonError::into_inner);

            // Keep any previous token buffers alive until both sets of waiters have been notified.
            let _ = std::mem::replace(&mut *publication, EffectTokenPublication::Forward(self.predecessor.take()));
            self.submission.publication.clear_poison();

            // Release the repaired publication before waking successors.
            drop(publication);

            self.submission.publication_ready.notify_all();
            self.work.complete(self.identity, None);
        }
    }
}

thread_local! {
    /// Depth of managed callbacks on the current thread. It only detects reentrancy, while effect ordering lives in
    /// the explicit session scope.
    static EFFECT_CALLBACK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Tracks a managed callback on the thread that entered it. Dropping the guard restores the thread's callback depth,
/// including when the callback returns an error or unwinds.
pub(crate) struct EffectCallbackGuard {
    /// Keeps the guard on the thread whose callback depth it will restore.
    marker: PhantomData<*mut ()>,
}

impl EffectCallbackGuard {
    /// Marks the current thread as inside a managed callback. Keep the guard alive for the duration of the callback
    /// to prevent reentrant execution, barriers, and acknowledgment on that thread.
    pub(crate) fn enter() -> Self {
        EFFECT_CALLBACK_DEPTH.update(|depth| depth + 1);
        Self { marker: PhantomData }
    }
}

impl Drop for EffectCallbackGuard {
    fn drop(&mut self) {
        EFFECT_CALLBACK_DEPTH.update(|depth| depth - 1);
    }
}

/// Prevents the current thread from starting an execution, waiting at an effect barrier, or acknowledging effect
/// errors while it is running a managed callback such as a print or assertion handler. Call this before taking scope
/// or reference locks. A nested dispatch or barrier can wait for the current execution, which needs the callback to
/// return before it can finish.
///
/// # Errors
///
/// Returns [`ExecutionError::ReentrantDispatch`] if an [`EffectCallbackGuard`] is active on the current thread.
pub(crate) fn ensure_effect_dispatch_is_allowed() -> Result<(), Error> {
    if EFFECT_CALLBACK_DEPTH.get() != 0 { Err(ExecutionError::ReentrantDispatch.into()) } else { Ok(()) }
}

/// Host publication state for a submission's per-device ordered I/O token buffers.
#[derive(Clone, Default)]
enum EffectTokenPublication {
    /// The submission is still preparing its publication.
    #[default]
    Pending,

    /// The submission published its token buffers, a result without tokens, or a failure at the execution boundary.
    Published(Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>),

    /// The reservation was canceled, so successors follow its predecessor's publication.
    Forward(Option<Arc<EffectSubmission>>),
}

/// Token publication of one reserved execution with effects, which is a node of its scope's ordered chain when the
/// execution is ordered. Ordered I/O tokens are stored as PJRT buffers.
struct EffectSubmission {
    /// Token publication or forwarding link for a canceled reservation.
    publication: Mutex<EffectTokenPublication>,

    /// Wakes successors when this submission publishes a result or forwards its predecessor.
    publication_ready: Condvar,
}

impl EffectSubmission {
    /// Publishes this submission's token result or failure after execution handoff and wakes its successors.
    fn publish(&self, tokens: Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>) {
        *self.publication.lock().expect("effect publication mutex poisoned") =
            EffectTokenPublication::Published(tokens);
        self.publication_ready.notify_all();
    }

    /// Waits for this submission to publish or provide a forwarding link. The caller can follow the link after the
    /// function releases the mutex.
    fn wait_for_publication(&self) -> EffectTokenPublication {
        let mut publication = self.publication.lock().expect("effect publication mutex poisoned");
        while matches!(*publication, EffectTokenPublication::Pending) {
            publication = self.publication_ready.wait(publication).expect("effect publication mutex poisoned");
        }
        publication.clone()
    }

    /// Waits for token publication on the host and follows canceled reservations iteratively. The returned buffers
    /// may still be pending on their devices.
    fn wait_for_tokens(&self) -> Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>> {
        let mut publication = self.wait_for_publication();
        loop {
            match publication {
                EffectTokenPublication::Pending => unreachable!("publication waits for a terminal host state"),
                EffectTokenPublication::Published(result) => return result,
                EffectTokenPublication::Forward(None) => return Ok(None),
                EffectTokenPublication::Forward(Some(predecessor)) => publication = predecessor.wait_for_publication(),
            }
        }
    }
}

impl Drop for EffectSubmission {
    fn drop(&mut self) {
        // Canceled reservations can form a long forwarding chain. Detach uniquely owned links before dropping
        // each node so scope teardown does not consume one stack frame per canceled submission. Exclusive access
        // lets us dismantle each link even if its publication mutex was poisoned.
        let mut publication = std::mem::take(self.publication.get_mut().unwrap_or_else(PoisonError::into_inner));
        while let EffectTokenPublication::Forward(Some(predecessor)) = publication {
            match Arc::try_unwrap(predecessor) {
                Ok(mut predecessor) => {
                    publication =
                        std::mem::take(predecessor.publication.get_mut().unwrap_or_else(PoisonError::into_inner));
                }
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;
    use std::sync::mpsc::{Sender, channel};
    use std::sync::{Arc, Condvar, Mutex, PoisonError};
    use std::thread;
    use std::time::{Duration, Instant};

    use pretty_assertions::assert_eq;

    use ryft_core::{
        ArrayType, CustomCallOperation, DataType, Device, DeviceMesh, EffectClass, LogicalMesh, MeshAxis, MeshAxisType,
        Placeholder, Print, PrintOperation, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, Typed,
    };
    use ryft_pjrt::{Buffer, BufferSpecification, BufferType, Client, EventPromise, ExecutionFence};

    use crate::sessions::tests::{f64_vector, two_device_mesh};
    use crate::tests::{cpu_client, values_to_bytes};
    use crate::{
        Array, Constant, Error, ExecutionError, FromPjrt, Operation, Options, PRINT_CUSTOM_CALL_TARGET, ProgramBuilder,
        Session, ensure_print_handler_registered, with_captured_prints,
    };

    #[cfg(feature = "cuda-13")]
    use crate::tests::cuda_client;

    use super::{
        EFFECT_CALLBACK_DEPTH, EffectCallbackGuard, EffectScope, EffectTokenPublication,
        ensure_effect_dispatch_is_allowed,
    };

    const CALLBACK_TIMEOUT: Duration = Duration::from_secs(5);

    /// Creates a fence whose completion the test controls through its event promise.
    fn pending_fence() -> (ExecutionFence, EventPromise) {
        let (event, promise) = cpu_client(1).event(()).unwrap();
        (ExecutionFence::new(vec![event]), promise)
    }

    /// Holds a poisoned guard until the parent knows the waiter is about to park.
    fn wait_for_cancellation<T>(
        state: &Mutex<T>,
        ready: &Condvar,
        is_pending: impl Fn(&T) -> bool,
        started: Sender<()>,
    ) {
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(is_pending(&state));
        started.send(()).unwrap();

        let deadline = Instant::now() + CALLBACK_TIMEOUT;
        while is_pending(&state) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "cancellation did not wake the waiter");
            let (next, timeout) = ready.wait_timeout(state, remaining).unwrap_or_else(PoisonError::into_inner);
            assert!(!timeout.timed_out(), "cancellation did not wake the waiter");
            state = next;
        }
    }

    /// Chooses one addressable device so the fixture uses the same mesh on CPU and CUDA.
    fn single_device_mesh(client: &Client<'_>) -> DeviceMesh {
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        DeviceMesh::new(mesh, vec![device]).unwrap()
    }

    /// Reads the ordered tail's tokens after releasing the scope lock.
    fn ordered_tail_tokens(scope: &EffectScope) -> Vec<Arc<Buffer<'static>>> {
        let tail = Arc::clone(scope.state.lock().unwrap().tail.as_ref().unwrap());
        tail.wait_for_tokens().unwrap().unwrap()
    }

    /// Submits two prints through one session on one device of `client` and checks that they print in submission order.
    /// With `hold_predecessor`, the two prints are additionally submitted behind an outstanding predecessor token
    /// (i.e., a token buffer whose transfer from the host has not happened yet) and must stay pending until it is
    /// released. That fixture only works on plugins that submit executables without waiting for unrecorded input
    /// definition events on the host, such as the CPU plugin. The CUDA plugin blocks the submitting thread until
    /// every input event is recorded, so that case checks print order after session synchronization.
    fn assert_session_array_ordered_effects(client: &'static Client<'static>, hold_predecessor: bool) {
        let mesh = single_device_mesh(client);
        let session = Session::builder().client(client).build().unwrap();
        let first = f64_vector(&session, &mesh, &[1.0]);
        let second = f64_vector(&session, &mesh, &[2.0]);

        // Model an outstanding predecessor without relying on compiler cost heuristics or expensive dummy work.
        let manager = hold_predecessor.then(|| {
            let manager = client
                .host_to_device_transfer_manager(
                    vec![BufferSpecification::new(BufferType::Token, [])],
                    client.addressable_devices().unwrap()[0].default_memory().unwrap(),
                )
                .unwrap();
            let predecessor = session.reserve_effects(Some(&[mesh.devices()[0].id()])).unwrap();
            predecessor.publish(
                Ok(Some(vec![Arc::new(manager.retrieve_buffer(0).unwrap())])),
                &ExecutionFence::new(Vec::new()),
            );
            manager
        });

        let (readiness, lines) = with_captured_prints(|| {
            let readiness = thread::scope(|threads| {
                let (sender, receiver) = channel();
                let scope = &session.effect_scope;
                threads.spawn(move || {
                    let _ = first.print("first").unwrap();
                    let first_tokens = ordered_tail_tokens(scope);
                    let _ = second.print("second").unwrap();
                    let second_tokens = ordered_tail_tokens(scope);
                    let readiness = hold_predecessor.then(|| {
                        (
                            first_tokens[0].ready().unwrap().ready().unwrap(),
                            second_tokens[0].ready().unwrap().ready().unwrap(),
                        )
                    });
                    sender.send(readiness).unwrap();
                });

                // Release the token even if dispatch blocked on it, so the worker can finish.
                let readiness = receiver.recv_timeout(CALLBACK_TIMEOUT);
                if let Some(manager) = manager {
                    manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
                }
                readiness
            });
            session.synchronize().unwrap();
            readiness
        });

        assert_eq!(readiness, Ok(hold_predecessor.then_some((false, false))));
        assert_eq!(lines, vec!["first: [1.0]", "second: [2.0]"]);
    }

    /// Checks that an outputless unordered execution registers with the effect scope and preserves the ordered tail.
    fn assert_outputless_unordered_execution(client: &'static Client<'static>) {
        let mesh = single_device_mesh(client);
        let session = Session::builder().client(client).build().unwrap();
        let input = f64_vector(&session, &mesh, &[3.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        builder
            .add_instruction(
                PrintOperation::<ArrayType>::new("unordered").with_effect_class(EffectClass::UnorderedIo),
                Vec::new(),
                vec![argument],
                None,
            )
            .unwrap();
        let source = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let lowered = session.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let program = session.compile_program(&lowered).unwrap();
        assert!(program.signature().has_unordered_io());
        assert!(!program.signature().has_ordered_io());

        // Leave a completed entry at the ordered tail so any unordered reservation that joins the chain is visible.
        let ordered = session.reserve_effects(Some(&[mesh.devices()[0].id()])).unwrap();
        let ordered_submission = Arc::clone(&ordered.submission);
        ordered.publish(Ok(None), &ExecutionFence::new(Vec::new()));
        let ((), lines) = with_captured_prints(|| {
            let execution = session.execute_compiled_async(&program, vec![input]).unwrap();
            assert!(execution.output().is_empty());

            // The execution made the second reservation of the scope and left the ordered tail in place.
            let scope = &session.effect_scope;
            assert_eq!(scope.work.snapshot(), 2);
            assert!(Arc::ptr_eq(scope.state.lock().unwrap().tail.as_ref().unwrap(), &ordered_submission));

            session.synchronize().unwrap();
        });

        assert_eq!(lines, vec!["unordered: [3.0]"]);
    }

    #[test]
    fn test_effect_scope_reserve() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let unordered = scope.reserve(None).unwrap();
        let second = scope.reserve(Some(&[0])).unwrap();

        assert_eq!(first.identity, 0);
        assert_eq!(unordered.identity, 1);
        assert_eq!(second.identity, 2);
        assert!(first.predecessor.is_none());
        assert!(unordered.predecessor.is_none());
        assert!(Arc::ptr_eq(second.predecessor.as_ref().unwrap(), &first.submission));
    }

    #[test]
    fn test_effect_scope_reserve_device_assignment() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0, 1])).unwrap();
        let first_submission = Arc::clone(&first.submission);
        let device_storage = scope.state.lock().unwrap().devices.as_ref().unwrap().as_ptr();
        assert!(matches!(
            scope.reserve(Some(&[1, 0])),
            Err(Error::Execution(ExecutionError::ScopeDeviceMismatch { expected, actual }))
                if expected == vec![0, 1] && actual == vec![1, 0],
        ));

        // Cancellation preserves the predecessor and device assignment for the next reservation.
        drop(first);
        let successor = scope.reserve(Some(&[0, 1])).unwrap();
        assert!(Arc::ptr_eq(successor.predecessor.as_ref().unwrap(), &first_submission));
        assert_eq!(scope.state.lock().unwrap().devices.as_ref().unwrap().as_ptr(), device_storage);
    }

    #[test]
    fn test_effect_scope_barrier() {
        let scope = EffectScope::default();
        let first = scope.reserve(None).unwrap();
        let successful = scope.reserve(None).unwrap();
        let last = scope.reserve(None).unwrap();

        // Publish failures in reverse order to distinguish reservation order from completion order.
        let ready = ExecutionFence::new(Vec::new());
        last.publish(Err(Arc::from("last effect failed")), &ready);
        successful.publish(Ok(None), &ready);
        first.publish(Err(Arc::from("first effect failed")), &ready);

        let error = scope.barrier().unwrap_err();
        assert!(matches!(
            &error,
            Error::Execution(ExecutionError::EffectSubmissionFailures { failures })
                if *failures == vec![(0, Arc::from("first effect failed")), (2, Arc::from("last effect failed"))],
        ));
        assert_eq!(
            error.to_string(),
            "submission 0: first effect failed; submission 2: last effect failed; \
             call `effects_barrier()` again if more work was submitted, then `acknowledge_effect_errors()` \
             to recover once affected work is complete",
        );
        let tracked = scope.work.lock().failures.keys().copied().collect::<Vec<_>>();
        assert_eq!(tracked, vec![0, 2]);
    }

    #[test]
    fn test_effect_scope_barrier_waits_for_fixed_snapshot() {
        let scope = EffectScope::default();
        let first = scope.reserve(None).unwrap();
        let (fence, promise) = pending_fence();
        first.publish(Ok(None), &fence);
        let snapshot = scope.work.snapshot();
        let later = scope.reserve(None).unwrap();
        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let scope = &scope;
            threads.spawn(move || sender.send(scope.barrier_at(snapshot)).unwrap());

            // Only the first reservation belongs to the snapshot, so its completion releases the barrier.
            promise.set(None).unwrap();
            let result = receiver.recv_timeout(CALLBACK_TIMEOUT);
            let later_pending = scope.work.lock().pending.contains_key(&later.identity);

            // Cancel the later reservation so that the barrier can return even if it missed the deadline.
            drop(later);
            result.unwrap().unwrap();
            assert!(later_pending);
        });
    }

    #[test]
    fn test_effect_scope_acknowledge() {
        let scope = EffectScope::default();
        let failed = scope.reserve(Some(&[0])).unwrap();
        let tail = scope.reserve(Some(&[0])).unwrap();
        let tail_submission = Arc::downgrade(&tail.submission);
        let ready = ExecutionFence::new(Vec::new());
        tail.publish(Ok(None), &ready);
        failed.publish(Err(Arc::from("ordered effect failed")), &ready);
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        assert!(matches!(scope.reserve(Some(&[0])), Err(Error::Execution(ExecutionError::ScopeNeedsRecovery))));

        assert!(matches!(
            scope.barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("ordered effect failed"))],
        ));
        // The scope is the only owner of the successful tail of the failed chain.
        assert!(tail_submission.upgrade().is_some());

        let unordered = scope.reserve(None).unwrap();
        scope.acknowledge().unwrap();
        assert!(tail_submission.upgrade().is_none());
        assert_eq!(scope.work.lock().pending.keys().copied().collect::<Vec<_>>(), vec![unordered.identity]);

        let recovered = scope.reserve(Some(&[1])).unwrap();
        assert!(recovered.predecessor.is_none());
        assert!(scope.work.lock().failures.is_empty());
    }

    #[test]
    fn test_effect_scope_acknowledge_rejects_pending_ordered_work_after_failure() {
        let scope = EffectScope::default();
        let failed = scope.reserve(Some(&[0])).unwrap();
        let (fence, promise) = pending_fence();
        failed.publish(Ok(None), &fence);
        let snapshot = scope.work.snapshot();
        let later = scope.reserve(Some(&[0])).unwrap();

        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let scope = &scope;
            threads.spawn(move || sender.send(scope.barrier_at(snapshot)).unwrap());

            // Have the barrier report the failure while the later ordered reservation is still pending.
            promise.set(Some(ryft_pjrt::Error::invalid_argument("ordered effect failed"))).unwrap();
            let report = receiver.recv_timeout(CALLBACK_TIMEOUT);
            let acknowledgment = scope.acknowledge();

            // Cancel the later reservation so that the barrier can return even if it missed the deadline.
            drop(later);
            assert!(matches!(
                report.unwrap(),
                Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                    if failures == vec![(0, Arc::from("ordered effect failed"))],
            ));
            assert!(matches!(acknowledgment, Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        });

        // The later reservation has been canceled, so no ordered work is pending anymore.
        scope.acknowledge().unwrap();
    }

    #[test]
    fn test_effect_scope_acknowledge_preserves_pending_unordered_work() {
        let scope = EffectScope::default();
        let pending = scope.reserve(None).unwrap();
        let (fence, promise) = pending_fence();
        pending.publish(Ok(None), &fence);
        scope.acknowledge().unwrap();

        promise.set(Some(ryft_pjrt::Error::invalid_argument("late completion failure"))).unwrap();
        assert!(matches!(
            scope.barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("late completion failure"))],
        ));
        scope.acknowledge().unwrap();
        let work = scope.work.lock();
        assert!(work.pending.is_empty() && work.failures.is_empty());
    }

    #[test]
    fn test_effect_reservation_wait_for_predecessor_tokens() {
        let client = cpu_client(1);
        let device = &client.addressable_devices().unwrap()[0];
        let devices = [Device::from_pjrt(device).unwrap().id()];
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&devices)).unwrap();
        let successor = scope.reserve(Some(&devices)).unwrap();
        assert!(first.wait_for_predecessor_tokens().unwrap().is_none());

        let manager = client
            .host_to_device_transfer_manager(
                vec![BufferSpecification::new(BufferType::Token, [])],
                device.default_memory().unwrap(),
            )
            .unwrap();
        let token = Arc::new(manager.retrieve_buffer(0).unwrap());
        manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
        first.publish(Ok(Some(vec![Arc::clone(&token)])), &ExecutionFence::new(Vec::new()));
        let tokens = successor.wait_for_predecessor_tokens().unwrap().unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(Arc::ptr_eq(&tokens[0], &token));
    }

    #[test]
    fn test_effect_reservation_wait_for_predecessor_tokens_orders_cpu_executions() {
        assert_session_array_ordered_effects(cpu_client(1), true);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_effect_reservation_wait_for_predecessor_tokens_orders_cuda_executions() {
        // CUDA waits on unrecorded input events during dispatch, so exercise the chain with ordinary launches.
        assert_session_array_ordered_effects(cuda_client(), false);
    }

    #[test]
    fn test_effect_reservation_wait_for_predecessor_tokens_preserves_per_device_dependencies() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let session = Session::builder().client(client).build().unwrap();
        ensure_print_handler_registered(client).unwrap();
        let r#type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(
                Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Sharded(vec!["x".to_string()])])
                    .unwrap(),
            )
            .unwrap();
        let first =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[0.0f64, 1.0]))
                .unwrap();
        let second =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[2.0f64, 3.0]))
                .unwrap();
        let boundary = ShardMapBoundary::new(
            mesh.logical_mesh().clone(),
            vec![r#type.sharding().unwrap().clone()],
            Vec::new(),
            vec!["x".to_string()],
        )
        .unwrap();
        let local_type = boundary.local_input_type(0, &r#type).unwrap();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(local_type.clone().into());
        builder
            .add_instruction(
                CustomCallOperation::new(PRINT_CUSTOM_CALL_TARGET, Vec::new())
                    .with_attribute("label", "effect")
                    .with_effect_class(EffectClass::DeviceOrderedIo),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let body = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&body, vec![r#type.clone().into()], boundary).unwrap();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.into());
        let body = builder.import_program(body);
        builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![input], None)
            .unwrap();
        let source = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let lowered = session.lower_program(&source, 0, &Options::new(mesh)).unwrap();
        let program = session.compile_program(&lowered).unwrap();
        assert!(program.signature().has_device_ordered_io());
        let devices = program.executable().addressable_devices().unwrap();
        let managers = devices
            .iter()
            .map(|device| {
                client
                    .host_to_device_transfer_manager(
                        vec![BufferSpecification::new(BufferType::Token, [])],
                        device.default_memory().unwrap(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let scope = &session.effect_scope;
        let device_ids = devices.iter().map(|device| Device::from_pjrt(device).unwrap().id()).collect::<Vec<_>>();
        let predecessor = scope.reserve(Some(&device_ids)).unwrap();
        predecessor.publish(
            Ok(Some(managers.iter().map(|manager| Arc::new(manager.retrieve_buffer(0).unwrap())).collect())),
            &ExecutionFence::new(Vec::new()),
        );
        managers[1].transfer_data(0, Arc::new([]), 0, true).unwrap().r#await().unwrap();
        let (progress, lines) = with_captured_prints(|| {
            let progress = thread::scope(|threads| {
                let (sender, receiver) = channel();
                let session = &session;
                threads.spawn(move || {
                    session.execute_program(&program, vec![first]).unwrap();
                    let tokens = ordered_tail_tokens(scope);

                    // CPU serializes multi-device launches. Check device 1's readiness on the first execution
                    // and that the second dispatch returns while device 0's token is pending.
                    tokens[1].ready().unwrap().r#await().unwrap();
                    session.execute_program(&program, vec![second]).unwrap();
                    sender.send(()).unwrap();
                });

                let progress = receiver.recv_timeout(CALLBACK_TIMEOUT);

                // Unblock both launches before joining the worker if either operation missed the deadline.
                managers[0].transfer_data(0, Arc::new([]), 0, true).unwrap().r#await().unwrap();
                progress
            });
            session.synchronize().unwrap();
            progress
        });

        assert_eq!(progress, Ok(()), "the ready device and second dispatch should progress independently");
        assert_eq!(lines.first().map(String::as_str), Some("effect: [1.0]"));
        let first_on_first_device = lines.iter().position(|line| line == "effect: [0.0]").unwrap();
        let second_on_first_device = lines.iter().position(|line| line == "effect: [2.0]").unwrap();
        assert!(first_on_first_device < second_on_first_device);

        let mut sorted_lines = lines;
        sorted_lines.sort_unstable();
        assert_eq!(sorted_lines, vec!["effect: [0.0]", "effect: [1.0]", "effect: [2.0]", "effect: [3.0]"]);
    }

    #[test]
    fn test_effect_reservation_publish() {
        let scope = EffectScope::default();
        let reservation = scope.reserve(None).unwrap();
        let (fence, promise) = pending_fence();
        let submission = Arc::downgrade(&reservation.submission);
        let work = Arc::clone(&scope.work);
        reservation.publish(Ok(None), &fence);

        // The fence callback keeps only the tracker alive, so the submission can be released before it fires.
        drop(scope);
        let submission_released = submission.upgrade().is_none();
        let pending = work.lock().pending.contains_key(&0);
        promise.set(None).unwrap();
        assert!(submission_released);
        assert!(pending);
        assert!(work.wait(1).failures.is_empty());
    }

    #[test]
    fn test_effect_reservation_publish_boundary_failure() {
        let scope = EffectScope::default();
        let reservation = scope.reserve(Some(&[0])).unwrap();
        let successor = scope.reserve(Some(&[0])).unwrap();
        let (fence, promise) = pending_fence();
        let submission = Arc::clone(&reservation.submission);
        reservation.publish(Err(Arc::from("boundary failed")), &fence);

        // Capture publication before completing the fence without risking a blocking token lookup.
        let published = matches!(
            &*submission.publication.lock().unwrap(),
            EffectTokenPublication::Published(Err(reason)) if reason.as_ref() == "boundary failed",
        );
        let pending = scope.work.lock().pending.contains_key(&0);
        promise.set(Some(ryft_pjrt::Error::invalid_argument("execution fence failed"))).unwrap();

        assert!(published);
        assert!(pending);
        assert!(matches!(successor.wait_for_predecessor_tokens(), Err(reason) if reason.as_ref() == "boundary failed"));
        assert_eq!(scope.work.wait(1).failures.get(&0), Some(&(true, Arc::from("boundary failed"))));
    }

    #[test]
    fn test_effect_reservation_publish_tracks_outputless_unordered_cpu_execution() {
        assert_outputless_unordered_execution(cpu_client(1));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_effect_reservation_publish_tracks_outputless_unordered_cuda_execution() {
        assert_outputless_unordered_execution(cuda_client());
    }

    #[test]
    fn test_effect_reservation_drop() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let second = scope.reserve(Some(&[0])).unwrap();
        let canceled = Arc::clone(&second.submission);
        let canceled_identity = second.identity;
        thread::scope(|threads| {
            let (sender, receiver) = channel();
            threads.spawn(move || {
                // Signal the test thread once cancellation has finished.
                drop(second);
                sender.send(()).unwrap();
            });

            let cancellation = receiver.recv_timeout(CALLBACK_TIMEOUT);
            let predecessor_pending = scope.work.lock().pending.contains_key(&first.identity);

            // Release any waiter on the predecessor even if cancellation timed out.
            drop(first);

            assert_eq!(cancellation, Ok(()));
            assert!(predecessor_pending);
        });
        // A canceled reservation completes without a failure, so a barrier stops waiting for it.
        let work = scope.work.wait(canceled_identity + 1);
        assert!(!work.failures.contains_key(&canceled_identity));
        drop(work);
        assert!(canceled.wait_for_tokens().unwrap().is_none());
    }

    #[test]
    fn test_effect_reservation_drop_repairs_poisoned_state_and_wakes_waiters() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let reservation = scope.reserve(Some(&[0])).unwrap();
        let submission = Arc::clone(&reservation.submission);
        let identity = reservation.identity;
        first.publish(Err(Arc::from("predecessor publication failed")), &ExecutionFence::new(Vec::new()));

        let poisoned = catch_unwind(|| {
            let _publication = submission.publication.lock().unwrap();
            let _work = scope.work.lock();
            panic!("poison cancellation state");
        });
        assert!(poisoned.is_err());
        assert!(submission.publication.is_poisoned());

        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let publication = &submission.publication;
            let publication_ready = &submission.publication_ready;
            threads.spawn(move || {
                wait_for_cancellation(
                    publication,
                    publication_ready,
                    |publication| matches!(publication, EffectTokenPublication::Pending),
                    sender,
                );
            });
            let started = receiver.recv_timeout(CALLBACK_TIMEOUT);

            // Cancel after the waiter has reached its pending state.
            drop(reservation);
            assert_eq!(started, Ok(()));
        });

        // Cancellation completed the reservation in the tracker, whose mutex the panic had poisoned as well.
        let work = scope.work.wait(identity + 1);
        assert!(!work.failures.contains_key(&identity));
        drop(work);
        assert!(matches!(
            submission.wait_for_tokens(),
            Err(reason) if reason.as_ref() == "predecessor publication failed",
        ));
    }

    #[test]
    fn test_effect_reservation_drop_forwards_predecessor_publication() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let second = scope.reserve(Some(&[0])).unwrap();
        let third = scope.reserve(Some(&[0])).unwrap();
        let successor = scope.reserve(Some(&[0])).unwrap();

        first.publish(Err(Arc::from("predecessor publication failed")), &ExecutionFence::new(Vec::new()));

        // Create the first forwarding link to the published failure.
        drop(second);

        // Extend the chain so the successor has to follow both links.
        drop(third);

        assert!(matches!(
            successor.wait_for_predecessor_tokens(),
            Err(reason) if reason.as_ref() == "predecessor publication failed",
        ));
    }

    #[test]
    fn test_effect_callback_guard_enter() {
        assert_eq!(EFFECT_CALLBACK_DEPTH.get(), 0);
        let outer = EffectCallbackGuard::enter();
        assert_eq!(EFFECT_CALLBACK_DEPTH.get(), 1);
        let inner = EffectCallbackGuard::enter();
        assert_eq!(EFFECT_CALLBACK_DEPTH.get(), 2);
        assert_eq!(thread::spawn(|| EFFECT_CALLBACK_DEPTH.get()).join().unwrap(), 0);

        // Dropping the inner guard leaves the outer callback active on the original thread.
        drop(inner);
        assert_eq!(EFFECT_CALLBACK_DEPTH.get(), 1);

        // Dropping the outer guard ends the last active callback on this thread.
        drop(outer);
        assert_eq!(EFFECT_CALLBACK_DEPTH.get(), 0);
    }

    #[test]
    fn test_ensure_effect_dispatch_is_allowed() {
        ensure_effect_dispatch_is_allowed().unwrap();
        let _guard = EffectCallbackGuard::enter();
        let error = ensure_effect_dispatch_is_allowed().unwrap_err();
        assert!(matches!(error, Error::Execution(ExecutionError::ReentrantDispatch)));
        assert_eq!(error.to_string(), "execution and effect barriers cannot be entered from a managed callback");

        let scope = EffectScope::default();
        assert!(matches!(scope.reserve(None), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.barrier(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
    }

    #[test]
    fn test_effect_submission_drop_releases_long_canceled_chain() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let head = Arc::downgrade(&first.submission);

        // Cancel the head so the rest of the reservations form forwarding links to it.
        drop(first);

        for _ in 1..10_000 {
            let _reservation = scope.reserve(Some(&[0])).unwrap();
        }
        let tail = Arc::clone(scope.state.lock().unwrap().tail.as_ref().unwrap());
        assert!(tail.wait_for_tokens().unwrap().is_none());

        // Release the scope's reference so the retained tail owns the entire chain.
        drop(scope);
        assert!(head.upgrade().is_some());
        // Releasing the tail must also release the head of the chain without overflowing the stack.
        drop(tail);
        assert!(head.upgrade().is_none());
    }

    #[test]
    fn test_effect_submission_drop_recovers_poisoned_publication() {
        for poison_tail in [true, false] {
            let scope = EffectScope::default();
            let predecessor = scope.reserve(Some(&[0])).unwrap();
            let tail = scope.reserve(Some(&[0])).unwrap();
            let predecessor_submission = Arc::clone(&predecessor.submission);
            let tail_submission = Arc::clone(&tail.submission);
            let predecessor_released = Arc::downgrade(&predecessor_submission);

            // Cancel both reservations so the tail holds a forwarding link to its predecessor.
            drop((tail, predecessor));

            let publication =
                if poison_tail { &tail_submission.publication } else { &predecessor_submission.publication };
            let poisoned = catch_unwind(|| {
                let _guard = publication.lock().unwrap();
                panic!("poison effect publication");
            });
            assert!(poisoned.is_err());
            assert!(publication.is_poisoned());

            // Release the extra owners so teardown can take exclusive ownership of the forwarding link.
            drop((scope, predecessor_submission));

            // Exercise teardown after poisoning either the tail itself or the link it owns.
            drop(tail_submission);

            assert!(predecessor_released.upgrade().is_none());
        }
    }
}
