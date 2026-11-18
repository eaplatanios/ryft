use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use ryft_core::DeviceId;
use ryft_pjrt::{Buffer, ExecutionFence};

use crate::{Error, ExecutionError};

/// Tracks executions with effects and keeps ordered I/O in reservation order. Cloning a [`Session`](crate::Session)
/// shares its effect scope. Use [`Session::with_new_effect_scope`](crate::Session::with_new_effect_scope) to give a
/// handle independent ordering and completion tracking. The same compiled program can run in either scope. Each
/// execution with effects reserves a place in the scope before dispatch. Ordered executions pass device tokens from
/// one execution to the next, preserving effect order even when device work runs asynchronously. The scope also tracks
/// completion of unordered executions. [`Self::barrier`] waits for work already reserved when the barrier takes its
/// snapshot and reports failures in reservation order. [`Self::acknowledge`] clears reported failures once the
/// affected work has completed.
#[derive(Default)]
pub(super) struct EffectScope {
    /// Reservation, observation, and recovery state, which is never held across PJRT calls or device waits.
    state: Mutex<EffectScopeState>,
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

        // Hold extracted submissions until after the scope guard is released, even if reading a completion panics.
        let mut retired = Vec::new();
        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        if devices.is_some()
            && state
                .submissions
                .iter()
                .any(|submission| submission.ordered && matches!(submission.completion.result(), Some(Err(_))))
        {
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

        let submission = Arc::new(EffectSubmission {
            identity: state.next_identity,
            ordered: devices.is_some(),
            completion: Arc::new(EffectCompletion::default()),
            publication: Mutex::new(EffectTokenPublication::Pending),
            publication_ready: Condvar::new(),
        });
        state.next_identity += 1;
        let predecessor = if devices.is_some() { state.tail.replace(Arc::clone(&submission)) } else { None };
        retired.extend(
            state
                .submissions
                .extract_if(.., |submission| matches!(submission.completion.result(), Some(Ok(())))),
        );
        state.submissions.push(Arc::clone(&submission));

        Ok(EffectReservation { submission, predecessor, committed: false })
    }

    /// Takes a snapshot of the scope's current reservations, then waits for each one to finish or be cancelled.
    /// This includes reservations still preparing device work. Later reservations are handled by a subsequent
    /// barrier. Reported failures remain in the scope until [`Self::acknowledge`] clears them.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::EffectSubmissionFailures`] with all failures from the snapshot in reservation
    /// order. Returns [`ExecutionError::ReentrantDispatch`] from within a managed effect callback.
    pub(super) fn barrier(&self) -> Result<(), Error> {
        ensure_effect_dispatch_is_allowed()?;

        let submissions = self.state.lock().expect("effect scope mutex poisoned").submissions.clone();
        let mut failures = Vec::new();
        let mut successful = HashSet::new();
        for submission in &submissions {
            match submission.completion.wait() {
                Ok(()) => {
                    successful.insert(submission.identity);
                }
                Err(error) => failures.push((submission.identity, error)),
            }
        }

        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        state.observed.extend(failures.iter().map(|(identity, _)| *identity));
        state.submissions.retain(|submission| !successful.contains(&submission.identity));

        // Unlock before dropping the snapshot, which still holds the removed submissions and their token buffers.
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

        // Keep the former tail and extracted submissions alive past the scope guard.
        let mut retired = Vec::new();
        let mut state = self.state.lock().expect("effect scope mutex poisoned");

        // Callbacks publish independently of the scope mutex. Read each completion at most once so a failure
        // recorded after its submission was checked remains tracked.
        let mut completed = HashSet::new();
        let mut ordered_failure = false;
        let mut ordered_pending = false;
        for submission in &state.submissions {
            let Some(result) = submission.completion.result() else {
                ordered_pending |= submission.ordered;
                continue;
            };
            if result.is_err() {
                if !state.observed.contains(&submission.identity) {
                    return Err(ExecutionError::PrematureAcknowledgment.into());
                }
                ordered_failure |= submission.ordered;
            }
            completed.insert(submission.identity);
        }

        if ordered_failure && ordered_pending {
            return Err(ExecutionError::PrematureAcknowledgment.into());
        }

        retired.extend(state.submissions.extract_if(.., |submission| completed.contains(&submission.identity)));
        if ordered_failure {
            retired.extend(state.tail.take());
            state.devices = None;
        }

        state.observed.clear();

        // Unlock the scope before freeing the completed ID set and destroying retired token buffers.
        drop(state);

        Ok(())
    }
}

/// Scope metadata protected by the [`EffectScope`]'s mutex. Completion callbacks retain their submission's
/// [`EffectCompletion`] independently of this state.
#[derive(Default)]
struct EffectScopeState {
    /// Identity of the next reserved invocation.
    next_identity: u64,

    /// Latest ordered invocation, retained even after its successful completion is retired.
    tail: Option<Arc<EffectSubmission>>,

    /// Outstanding invocations and unacknowledged failures, in reservation order.
    submissions: Vec<Arc<EffectSubmission>>,

    /// Failures reported by completed barriers.
    observed: HashSet<u64>,

    /// Device list in token order for the current ordered chain. Acknowledging an ordered failure clears the list.
    devices: Option<Vec<DeviceId>>,
}

/// Reservation for an execution with effects. Dropping an uncommitted reservation cancels it and lets its successors
/// follow its predecessor's token publication. Cancellation can proceed without waiting for the predecessor.
pub(crate) struct EffectReservation {
    /// Invocation being prepared or published.
    submission: Arc<EffectSubmission>,

    /// Predecessor whose token publication successors will follow if this reservation is canceled.
    predecessor: Option<Arc<EffectSubmission>>,

    /// Whether publication and callback registration have disarmed cancellation on drop.
    committed: bool,
}

impl EffectReservation {
    /// Waits for the predecessor's token publication on the host, following any cancelled reservations. Use the
    /// returned token buffers as inputs even if they are still pending on their devices. Returns [`None`] when there
    /// are no predecessor tokens and propagates any failure published by the predecessor.
    pub(crate) fn wait_for_predecessor_tokens(&self) -> Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>> {
        self.predecessor.as_ref().map_or(Ok(None), |previous| previous.wait_for_tokens())
    }

    /// Consumes the reservation and publishes its ordered I/O token buffers or execution boundary failure to its
    /// successors. Pass `Ok(None)` for a successful submission without ordered tokens. The fence callback records
    /// the boundary failure if one was published, or the fence's result otherwise. It owns the completion state
    /// independently of the reservation and can run inline if the fence is already ready. Cancellation on drop is
    /// disarmed only after callback registration returns.
    pub(crate) fn publish(
        mut self,
        tokens: Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>,
        fence: &ExecutionFence,
    ) {
        let boundary_failure = tokens.as_ref().err().cloned();
        self.submission.publish(tokens);
        let completion = Arc::clone(&self.submission.completion);
        fence.on_ready(move |result| {
            let result = match boundary_failure {
                Some(reason) => Err(reason),
                None => result.map_err(|error| Arc::from(error.to_string())),
            };
            completion.complete(result);
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
            self.submission.completion.cancel();
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

/// Records the outcome of one execution with effects. The result is pending until device work finishes or the
/// reservation is canceled. A canceled reservation records success so a barrier can stop waiting for it. A PJRT
/// completion callback can own this state independently of the session because it holds no client resources.
#[derive(Default)]
struct EffectCompletion {
    /// Result recorded once when the execution completes or the reservation is canceled.
    result: Mutex<Option<Result<(), Arc<str>>>>,

    /// Wakes barrier callers when the terminal outcome becomes available.
    ready: Condvar,
}

impl EffectCompletion {
    /// Stores the result and wakes waiting threads. The callback can record the result while another thread holds
    /// the effect scope's mutex.
    fn complete(&self, result: Result<(), Arc<str>>) {
        *self.result.lock().expect("effect completion mutex poisoned") = Some(result);
        self.ready.notify_all();
    }

    /// Records a canceled reservation as complete and repairs any poisoned completion state before waking waiters.
    fn cancel(&self) {
        let mut result = self.result.lock().unwrap_or_else(PoisonError::into_inner);
        *result = Some(Ok(()));
        self.result.clear_poison();

        // Release the repaired result before waking barrier callers.
        drop(result);

        self.ready.notify_all();
    }

    /// Returns the recorded result, or [`None`] if completion is pending.
    fn result(&self) -> Option<Result<(), Arc<str>>> {
        self.result.lock().expect("effect completion mutex poisoned").clone()
    }

    /// Blocks the current thread until the execution finishes or the reservation is canceled, then returns the
    /// recorded result. The wait can begin before device work is submitted.
    fn wait(&self) -> Result<(), Arc<str>> {
        let mut result = self.result.lock().expect("effect completion mutex poisoned");
        while result.is_none() {
            result = self.ready.wait(result).expect("effect completion mutex poisoned");
        }
        result.as_ref().unwrap().clone()
    }
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

/// One reserved execution with effects. Its ordered I/O tokens are stored as PJRT buffers.
struct EffectSubmission {
    /// Monotonically increasing submission identity, local to its scope.
    identity: u64,

    /// Whether the invocation participates in the ordered I/O chain.
    ordered: bool,

    /// State without buffers that the completion observer retains.
    completion: Arc<EffectCompletion>,

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

    use crate::sessions::tests::{f64_vector, session_with_client, two_device_mesh};
    use crate::tests::{cpu_client, values_to_bytes};
    use crate::{
        Array, Constant, Error, ExecutionError, FromPjrt, Operation, Options, PRINT_CUSTOM_CALL_TARGET, ProgramBuilder,
        ensure_print_handler_registered, with_captured_prints,
    };

    #[cfg(feature = "cuda-13")]
    use crate::tests::cuda_client;

    use super::{
        EFFECT_CALLBACK_DEPTH, EffectCallbackGuard, EffectScope, EffectSubmission, EffectTokenPublication,
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

    /// Waits for the barrier to retain the submission in its snapshot and release the scope lock.
    fn wait_for_barrier_snapshot(
        scope: &EffectScope,
        submission: &Arc<EffectSubmission>,
        initial_references: usize,
    ) -> bool {
        let deadline = Instant::now() + CALLBACK_TIMEOUT;
        while Instant::now() < deadline {
            if Arc::strong_count(submission) > initial_references && scope.state.try_lock().is_ok() {
                return true;
            }
            thread::yield_now();
        }
        false
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
    /// every input event is recorded, so there the chain is proven through submission order and barrier completion
    /// alone.
    fn assert_session_array_ordered_effects(client: &'static Client<'static>, hold_predecessor: bool) {
        let mesh = single_device_mesh(client);
        let session = session_with_client(client);
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
            session.effects_barrier().unwrap();
            readiness
        });

        assert_eq!(readiness, Ok(hold_predecessor.then_some((false, false))));
        assert_eq!(lines, vec!["first: [1.0]", "second: [2.0]"]);
    }

    /// Checks that an outputless unordered execution is tracked by the barrier and leaves the ordered tail in place.
    fn assert_outputless_unordered_execution(client: &'static Client<'static>) {
        let mesh = single_device_mesh(client);
        let session = session_with_client(client);
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
        let ordered_identity = ordered.submission.identity;
        ordered.publish(Ok(None), &ExecutionFence::new(Vec::new()));
        let ((), lines) = with_captured_prints(|| {
            let execution = session.execute_compiled_async(&program, vec![input]).unwrap();
            assert!(execution.output().is_empty());
            let state = session.effect_scope.state.lock().unwrap();
            assert_eq!(state.submissions.len(), 1);
            assert!(!state.submissions[0].ordered);
            assert_eq!(state.tail.as_ref().unwrap().identity, ordered_identity);

            // Release the scope mutex before the barrier takes its snapshot.
            drop(state);

            session.effects_barrier().unwrap();
        });

        assert_eq!(lines, vec!["unordered: [3.0]"]);
    }

    #[test]
    fn test_effect_scope_reserve() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let unordered = scope.reserve(None).unwrap();
        let second = scope.reserve(Some(&[0])).unwrap();

        assert_eq!(first.submission.identity, 0);
        assert_eq!(unordered.submission.identity, 1);
        assert_eq!(second.submission.identity, 2);
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
    fn test_effect_scope_reserve_retires_successful_submissions() {
        let scope = EffectScope::default();
        let completed = scope.reserve(None).unwrap();
        completed.publish(Ok(None), &ExecutionFence::new(Vec::new()));
        let pending = scope.reserve(None).unwrap();
        let state = scope.state.lock().unwrap();
        assert_eq!(state.submissions.len(), 1);
        assert!(Arc::ptr_eq(&state.submissions[0], &pending.submission));
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
        let tracked = scope
            .state
            .lock()
            .unwrap()
            .submissions
            .iter()
            .map(|submission| submission.identity)
            .collect::<Vec<_>>();
        assert_eq!(tracked, vec![0, 2]);
    }

    #[test]
    fn test_effect_scope_barrier_waits_for_fixed_snapshot() {
        let scope = EffectScope::default();
        let first = scope.reserve(None).unwrap();
        let (fence, promise) = pending_fence();
        let first_submission = Arc::clone(&first.submission);
        first.publish(Ok(None), &fence);
        let initial_references = Arc::strong_count(&first_submission);
        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let scope = &scope;
            threads.spawn(move || sender.send(scope.barrier()).unwrap());

            let snapshot_captured = wait_for_barrier_snapshot(scope, &first_submission, initial_references);
            let later = snapshot_captured.then(|| scope.reserve(None).unwrap());
            promise.set(None).unwrap();
            let result = receiver.recv_timeout(CALLBACK_TIMEOUT);
            assert!(snapshot_captured, "the barrier did not capture its snapshot and release the scope mutex");
            result.unwrap().unwrap();
            assert_eq!(later.unwrap().submission.completion.result(), None);
        });
    }

    #[test]
    fn test_effect_scope_acknowledge() {
        let scope = EffectScope::default();
        let failed = scope.reserve(Some(&[0])).unwrap();
        let tail = scope.reserve(Some(&[0])).unwrap();
        let failed_completion = Arc::clone(&failed.submission.completion);
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
        // The scope is now the only owner of the successful tail, which the barrier removed from its tracked list.
        assert!(tail_submission.upgrade().is_some());

        let unordered = scope.reserve(None).unwrap();
        scope.acknowledge().unwrap();
        assert!(tail_submission.upgrade().is_none());
        assert!(Arc::ptr_eq(&scope.state.lock().unwrap().submissions[0], &unordered.submission));
        // Cancel the unordered reservation before checking the barrier after recovery.
        drop(unordered);
        scope.barrier().unwrap();
        let recovered = scope.reserve(Some(&[1])).unwrap();
        assert!(recovered.predecessor.is_none());
        assert_eq!(failed_completion.result(), Some(Err(Arc::from("ordered effect failed"))));
    }

    #[test]
    fn test_effect_scope_acknowledge_rejects_pending_ordered_work_after_failure() {
        let scope = EffectScope::default();
        let failed = scope.reserve(Some(&[0])).unwrap();
        let (fence, promise) = pending_fence();
        let failed_submission = Arc::clone(&failed.submission);
        failed.publish(Ok(None), &fence);
        let initial_references = Arc::strong_count(&failed_submission);

        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let scope = &scope;
            threads.spawn(move || sender.send(scope.barrier()).unwrap());

            // Have the barrier report the failure while a later ordered reservation is still pending.
            let snapshot_captured = wait_for_barrier_snapshot(scope, &failed_submission, initial_references);
            let later = snapshot_captured.then(|| scope.reserve(Some(&[0])).unwrap());
            promise.set(Some(ryft_pjrt::Error::invalid_argument("ordered effect failed"))).unwrap();
            let report = receiver.recv_timeout(CALLBACK_TIMEOUT);
            assert!(snapshot_captured, "the barrier did not capture its snapshot and release the scope mutex");
            assert!(matches!(
                report.unwrap(),
                Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                    if failures == vec![(0, Arc::from("ordered effect failed"))],
            ));
            assert_eq!(later.as_ref().unwrap().submission.completion.result(), None);
            assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        });

        // The later reservation was cancelled when the scoped thread closure ended.
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
        assert_eq!(scope.state.lock().unwrap().submissions.len(), 0);
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
        let session = session_with_client(client);
        ensure_print_handler_registered(client).unwrap();
        let r#type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(
                Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Sharded(vec!["x".to_string()])])
                    .unwrap(),
            )
            .unwrap();
        let first =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), values_to_bytes(&[0.0f64, 1.0])).unwrap();
        let second =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), values_to_bytes(&[2.0f64, 3.0])).unwrap();
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
            session.effects_barrier().unwrap();
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
        let completion = Arc::clone(&reservation.submission.completion);
        reservation.publish(Ok(None), &fence);

        // The fence callback keeps only the completion alive, so the submission can be released before it fires.
        drop(scope);
        let submission_released = submission.upgrade().is_none();
        let pending = completion.result();
        promise.set(None).unwrap();
        assert!(submission_released);
        assert_eq!(pending, None);
        assert_eq!(completion.wait(), Ok(()));
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
        let pending = submission.completion.result();
        promise.set(Some(ryft_pjrt::Error::invalid_argument("execution fence failed"))).unwrap();

        assert!(published);
        assert_eq!(pending, None);
        assert!(matches!(successor.wait_for_predecessor_tokens(), Err(reason) if reason.as_ref() == "boundary failed"));
        assert_eq!(submission.completion.wait(), Err(Arc::from("boundary failed")));
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
        let cancelled = Arc::clone(&second.submission);
        thread::scope(|threads| {
            let (sender, receiver) = channel();
            threads.spawn(move || {
                // Signal the test thread once cancellation has finished.
                drop(second);
                sender.send(()).unwrap();
            });

            let cancellation = receiver.recv_timeout(CALLBACK_TIMEOUT);
            let predecessor_pending = first.submission.completion.result().is_none();

            // Release any waiter on the predecessor even if cancellation timed out.
            drop(first);

            assert_eq!(cancellation, Ok(()));
            assert!(predecessor_pending);
        });
        assert_eq!(cancelled.completion.result(), Some(Ok(())));
        assert!(cancelled.wait_for_tokens().unwrap().is_none());
    }

    #[test]
    fn test_effect_reservation_drop_repairs_poisoned_state_and_wakes_waiters() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(&[0])).unwrap();
        let reservation = scope.reserve(Some(&[0])).unwrap();
        let submission = Arc::clone(&reservation.submission);
        first.publish(Err(Arc::from("predecessor publication failed")), &ExecutionFence::new(Vec::new()));

        let poisoned = catch_unwind(|| {
            let _publication = submission.publication.lock().unwrap();
            let _completion = submission.completion.result.lock().unwrap();
            panic!("poison cancellation state");
        });
        assert!(poisoned.is_err());
        assert!(submission.publication.is_poisoned());
        assert!(submission.completion.result.is_poisoned());

        thread::scope(|threads| {
            let (sender, receiver) = channel();
            let publication_started = sender.clone();
            let publication = &submission.publication;
            let publication_ready = &submission.publication_ready;
            threads.spawn(move || {
                wait_for_cancellation(
                    publication,
                    publication_ready,
                    |publication| matches!(publication, EffectTokenPublication::Pending),
                    publication_started,
                );
            });
            let completion = &submission.completion;
            threads
                .spawn(move || wait_for_cancellation(&completion.result, &completion.ready, Option::is_none, sender));
            let first_started = receiver.recv_timeout(CALLBACK_TIMEOUT);
            let second_started = receiver.recv_timeout(CALLBACK_TIMEOUT);

            // Cancel after both waiters have reached their pending state.
            drop(reservation);
            assert_eq!(first_started, Ok(()));
            assert_eq!(second_started, Ok(()));
        });

        assert_eq!(submission.completion.wait(), Ok(()));
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
    fn test_effect_submission_drop_releases_long_cancelled_chain() {
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
        scope.barrier().unwrap();

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
