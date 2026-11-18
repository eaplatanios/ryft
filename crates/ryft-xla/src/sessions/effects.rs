use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex};

use ryft_core::DeviceId;
use ryft_pjrt::Buffer;

use crate::{Error, ExecutionError};

// TODO(eaplatanios): Review from here onwards.

/// Tracks executions with effects and keeps ordered I/O in submission order.
///
/// Cloning a [`Session`](crate::Session) shares its effect scope. Use
/// [`Session::with_new_effect_scope`](crate::Session::with_new_effect_scope) to give a handle independent ordering and
/// completion tracking. The same compiled program can run in either scope.
///
/// Each execution with effects reserves a place in the scope before dispatch. Ordered executions pass device tokens
/// from one execution to the next, preserving effect order even when device work runs asynchronously. The scope also
/// tracks completion of unordered executions.
///
/// [`Self::barrier`] waits for work already reserved when the barrier starts and reports failures in submission
/// order. [`Self::acknowledge`] clears reported failures once the affected work has completed.
#[derive(Default)]
pub(crate) struct EffectScope {
    /// Reservation, observation, and recovery state, which is never held across PJRT calls or device waits.
    state: Mutex<EffectScopeState>,
}

impl EffectScope {
    /// Reserves a position and snapshots the predecessor atomically before releasing the scope mutex.
    pub(crate) fn reserve(&self, device: Option<Vec<DeviceId>>) -> Result<EffectReservation, Error> {
        ensure_effect_dispatch_allowed()?;
        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        if device.is_some()
            && state
                .submissions
                .iter()
                .any(|submission| submission.ordered && matches!(submission.completion.result(), Some(Err(_))))
        {
            return Err(ExecutionError::ScopeNeedsRecovery.into());
        }
        if let Some(device) = &device {
            if let Some(previous) = &state.device
                && previous != device
            {
                return Err(
                    ExecutionError::ScopeDeviceMismatch { expected: previous.clone(), actual: device.clone() }.into()
                );
            }
            state.device = Some(device.clone());
        }
        let submission = Arc::new(EffectSubmission {
            identity: state.next_identity,
            ordered: device.is_some(),
            completion: Arc::new(EffectCompletion::default()),
            carrier: Mutex::new(EffectCarrier::Pending),
            published: Condvar::new(),
        });
        state.next_identity += 1;
        let predecessor = if device.is_some() { state.tail.replace(Arc::clone(&submission)) } else { None };
        let mut retired = Vec::new();
        state.submissions.retain(|submission| {
            if matches!(submission.completion.result(), Some(Ok(()))) {
                retired.push(Arc::clone(submission));
                false
            } else {
                true
            }
        });
        state.submissions.push(Arc::clone(&submission));
        drop(state);
        drop(retired);
        Ok(EffectReservation { submission, predecessor, published: false })
    }

    /// Waits for a fixed snapshot and reports its failures in submission order.
    pub(super) fn barrier(&self) -> Result<(), Error> {
        ensure_effect_dispatch_allowed()?;
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
        let mut retired = Vec::new();
        state.submissions.retain(|submission| {
            if successful.contains(&submission.identity) {
                retired.push(Arc::clone(submission));
                false
            } else {
                true
            }
        });
        drop(state);
        drop(retired);
        if failures.is_empty() {
            return Ok(());
        }
        Err(ExecutionError::EffectSubmissionFailures { failures }.into())
    }

    /// Clears only observed failures after affected invocations have completed, without waiting.
    pub(super) fn acknowledge(&self) -> Result<(), Error> {
        ensure_effect_dispatch_allowed()?;
        let mut state = self.state.lock().expect("effect scope mutex poisoned");
        // Callbacks publish independently of the scope mutex. Read each completion exactly once so a failure
        // arriving after validation cannot be discarded by the retirement step below.
        let completions = state
            .submissions
            .iter()
            .map(|submission| (submission.identity, submission.ordered, submission.completion.result()))
            .collect::<Vec<_>>();
        let ordered_failure = completions.iter().any(|(_, ordered, result)| *ordered && matches!(result, Some(Err(_))));
        if completions.iter().any(|(identity, ordered, result)| {
            (ordered_failure && *ordered && result.is_none())
                || (matches!(result, Some(Err(_))) && !state.observed.contains(identity))
        }) {
            return Err(ExecutionError::PrematureAcknowledgment.into());
        }
        let completed = completions
            .iter()
            .filter_map(|(identity, _, result)| result.as_ref().map(|_| *identity))
            .collect::<HashSet<_>>();
        let retired_tail = if ordered_failure { state.tail.take() } else { None };
        if ordered_failure {
            state.device = None;
        }
        let mut retired = Vec::new();
        state.submissions.retain(|submission| {
            if completed.contains(&submission.identity) {
                retired.push(Arc::clone(submission));
                false
            } else {
                true
            }
        });
        state.observed.clear();
        drop(state);
        drop(retired);
        drop(retired_tail);
        Ok(())
    }
}

/// Reservation for an execution with effects. Dropping an uncommitted reservation cancels it and lets its successors
/// inherit its predecessor's carrier. Cancellation can proceed without waiting for the predecessor to publish.
pub(crate) struct EffectReservation {
    /// Invocation being prepared or published.
    submission: Arc<EffectSubmission>,

    /// Predecessor whose carrier will be forwarded if submission fails before handoff.
    predecessor: Option<Arc<EffectSubmission>>,

    /// Whether device work and its carrier have been published.
    published: bool,
}

impl EffectReservation {
    /// Waits for the predecessor of this reservation to publish and returns its ordered carrier, which this submission
    /// must consume. Returns [`None`] when the reservation has no predecessor or the predecessor carries no tokens.
    pub(crate) fn predecessor_carrier(&self) -> Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>> {
        self.predecessor.as_ref().map_or(Ok(None), |previous| previous.carrier())
    }

    /// Publishes the ordered carrier or the boundary failure of this submission to its successors. The returned
    /// callback owns the completion state. Call it once the submitted device work finishes.
    pub(crate) fn publish(
        &self,
        carrier: Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>,
    ) -> impl FnOnce(Result<(), Arc<str>>) + Send + 'static + use<> {
        self.submission.publish(carrier);
        let completion = Arc::clone(&self.submission.completion);
        move |result| completion.complete(result)
    }

    /// Marks this reservation as published, so that dropping it no longer cancels its submission.
    pub(crate) fn commit(&mut self) {
        self.published = true;
    }
}

impl Drop for EffectReservation {
    fn drop(&mut self) {
        if !self.published {
            // Cancellation must not wait, because the predecessor may still be prepared on this same calling thread.
            *self.submission.carrier.lock().expect("effect publication mutex poisoned") =
                EffectCarrier::Forward(self.predecessor.take());
            self.submission.published.notify_all();
            self.submission.completion.complete(Ok(()));
        }
    }
}

thread_local! {
    /// Depth of managed callbacks on the current thread. It only detects reentrancy, while effect ordering lives in
    /// the explicit session scope.
    static EFFECT_CALLBACK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Restores managed callback depth even when a callback returns an error or unwinds.
pub(crate) struct EffectCallbackGuard;

impl EffectCallbackGuard {
    /// Enters a managed callback, preventing reentrant submission and barriers on its PJRT thread.
    pub(crate) fn enter() -> Self {
        EFFECT_CALLBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for EffectCallbackGuard {
    fn drop(&mut self) {
        EFFECT_CALLBACK_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Rejects reentrancy from managed callbacks before any scope or reference locks are taken.
pub(crate) fn ensure_effect_dispatch_allowed() -> Result<(), Error> {
    if EFFECT_CALLBACK_DEPTH.with(|depth| depth.get() != 0) {
        Err(ExecutionError::ReentrantDispatch.into())
    } else {
        Ok(())
    }
}

/// Renders the failures observed by an effect barrier in their original submission order.
pub(crate) fn format_effect_failures(failures: &[(u64, Arc<str>)]) -> String {
    failures
        .iter()
        .map(|(identity, error)| format!("submission {identity}: {error}"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Records the outcome of one execution with effects. The result is pending until device work finishes or the
/// reservation is cancelled. A cancelled reservation records success so a barrier can stop waiting for it.
///
/// A PJRT completion callback can own this state independently of the session because it holds no client resources.
#[derive(Default)]
struct EffectCompletion {
    /// Result recorded once when the execution completes or the reservation is cancelled.
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

    /// Returns the recorded result, or [`None`] if completion is pending.
    fn result(&self) -> Option<Result<(), Arc<str>>> {
        self.result.lock().expect("effect completion mutex poisoned").clone()
    }

    /// Blocks the current thread until the execution finishes or the reservation is cancelled, then returns the
    /// recorded result. The wait can begin before device work is submitted.
    fn wait(&self) -> Result<(), Arc<str>> {
        let mut result = self.result.lock().expect("effect completion mutex poisoned");
        while result.is_none() {
            result = self.ready.wait(result).expect("effect completion mutex poisoned");
        }
        result.as_ref().unwrap().clone()
    }
}

/// Host publication state for an effect carrier, including cancellation forwarding without a host wait.
#[derive(Clone, Default)]
enum EffectCarrier {
    /// The invocation has not published its physical result yet.
    #[default]
    Pending,

    /// The invocation published its physical result or submission failure.
    Published(Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>),

    /// Preparation was cancelled, so successors inherit the eventual carrier of the preceding invocation.
    Forward(Option<Arc<EffectSubmission>>),
}

/// One reserved invocation. Carrier ownership remains tied to the client lifetime through ordinary buffers.
struct EffectSubmission {
    /// Monotonically increasing submission identity, local to its scope.
    identity: u64,

    /// Whether the invocation participates in the ordered I/O chain.
    ordered: bool,

    /// State without buffers that the completion observer retains.
    completion: Arc<EffectCompletion>,

    /// Physical carrier publication or nonblocking forwarding of a cancelled reservation.
    carrier: Mutex<EffectCarrier>,

    /// Wakes submitting threads once the predecessor's physical result is available.
    published: Condvar,
}

impl EffectSubmission {
    /// Waits only for host publication, following cancelled reservations without recursion or device waits.
    pub(super) fn carrier(&self) -> Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>> {
        let mut carrier = self.publication();
        loop {
            match carrier {
                EffectCarrier::Published(result) => return result,
                EffectCarrier::Forward(None) => return Ok(None),
                EffectCarrier::Forward(Some(predecessor)) => carrier = predecessor.publication(),
                EffectCarrier::Pending => unreachable!("publication waits for a terminal host state"),
            }
        }
    }

    /// Publishes a carrier after execution handoff, or forwards the predecessor after a cancelled submission.
    pub(super) fn publish(&self, carrier: Result<Option<Vec<Arc<Buffer<'static>>>>, Arc<str>>) {
        *self.carrier.lock().expect("effect publication mutex poisoned") = EffectCarrier::Published(carrier);
        self.published.notify_all();
    }

    /// Waits for this reservation's publication state, without following forwarding links under its mutex.
    fn publication(&self) -> EffectCarrier {
        let mut carrier = self.carrier.lock().expect("effect publication mutex poisoned");
        while matches!(*carrier, EffectCarrier::Pending) {
            carrier = self.published.wait(carrier).expect("effect publication mutex poisoned");
        }
        carrier.clone()
    }
}

impl Drop for EffectSubmission {
    fn drop(&mut self) {
        // Cancelled reservations can form a long forwarding chain. Detach uniquely owned links before dropping
        // each node so scope teardown does not consume one stack frame per cancelled submission.
        let mut carrier = std::mem::take(self.carrier.get_mut().expect("effect publication mutex poisoned"));
        while let EffectCarrier::Forward(Some(predecessor)) = carrier {
            match Arc::try_unwrap(predecessor) {
                Ok(mut predecessor) => {
                    carrier = std::mem::take(predecessor.carrier.get_mut().expect("effect publication mutex poisoned"));
                }
                Err(_) => break,
            }
        }
    }
}

/// Synchronized scope metadata. Callbacks never acquire this mutex or own this state.
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

    /// Ordered device assignment selected by the chain. Changing it requires a new scope.
    device: Option<Vec<DeviceId>>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, Context, CustomCallOperation, DataType, Device, DeviceMesh, EffectClass, LogicalMesh, MeshAxis,
        MeshAxisType, OneOperation, Placeholder, Print, PrintOperation, ProgramError, Shape, ShardMapBoundary,
        ShardMapOperation, Sharding, ShardingDimension, StaticShape, Typed,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, Client};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    #[cfg(feature = "cuda-13")]
    use crate::sessions::platforms::Platform;
    use crate::sessions::tests::{domain_mesh, f64_vector, session_with_client};
    use crate::tests::{cpu_client, execution_client, values_to_bytes};
    use crate::{
        Array, CompilationError, Constant, FromPjrt, Operation, Options, PRINT_CUSTOM_CALL_TARGET, ProgramBuilder,
        ensure_print_handler_registered, with_captured_prints,
    };

    use super::*;

    /// Submits two prints through one session on one device of `client` and checks that they print in submission order.
    /// With `hold_predecessor`, the two prints are additionally submitted behind an outstanding predecessor token
    /// (i.e., a token buffer whose transfer from the host has not happened yet) and must stay pending until it is
    /// released.
    /// That fixture only works on plugins that submit executables without waiting for unrecorded input definition
    /// events on the host, such as the CPU plugin. The CUDA plugin blocks the submitting thread until every input event
    /// is recorded, so there the chain is proven through submission order and barrier completion alone.
    fn assert_session_array_ordered_effects(client: &'static Client<'static>, hold_predecessor: bool) {
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let r#type = ArrayType::new(DataType::F64, Shape::from(StaticShape::new(vec![1])));
        let first = session.array(r#type.clone(), 1.0f64.to_ne_bytes()).unwrap();
        let second = session.array(r#type, 2.0f64.to_ne_bytes()).unwrap();
        // Model an outstanding predecessor without relying on compiler cost heuristics or expensive dummy work.
        let manager = hold_predecessor.then(|| {
            let manager = client
                .host_to_device_transfer_manager(
                    vec![BufferSpecification::new(BufferType::Token, [])],
                    client.addressable_devices().unwrap()[0].default_memory().unwrap(),
                )
                .unwrap();
            let mut predecessor = session.effect_scope().reserve(Some(vec![mesh.devices()[0].id()])).unwrap();
            predecessor.submission.publish(Ok(Some(vec![Arc::new(manager.retrieve_buffer(0).unwrap())])));
            predecessor.submission.completion.complete(Ok(()));
            predecessor.published = true;
            manager
        });
        let tail_pending = || {
            !session.effect_scope().state.lock().unwrap().tail.as_ref().unwrap().carrier().unwrap().unwrap()[0]
                .ready()
                .unwrap()
                .ready()
                .unwrap()
        };
        let ((first_pending, second_pending), lines) = with_captured_prints(|| {
            drop(first.print("first").unwrap());
            let first_pending = tail_pending();
            drop(second.print("second").unwrap());
            let second_pending = tail_pending();
            if let Some(manager) = &manager {
                manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
            }
            session.effects_barrier().unwrap();
            (first_pending, second_pending)
        });
        if hold_predecessor {
            assert!(first_pending);
            assert!(second_pending);
        }
        assert_eq!(lines, vec!["first: [1.0]", "second: [2.0]"]);
        assert_eq!(session.compilation_cache().program_count(), 2);
    }

    /// Compiles ordered and unordered prints on `client`, checks the persisted effect metadata roundtrip and its
    /// corruption detection, then executes the compiled, restored, and unordered programs and verifies the ordered
    /// prints fire twice and the unordered print once. With `hold_predecessor`, the executions are submitted behind
    /// an outstanding predecessor token that is released only afterwards. Refer to
    /// [`assert_session_array_ordered_effects`] for why that fixture only works on CPU.
    fn assert_domain_effect_metadata_reload_and_unordered_completion(
        client: &'static Client<'static>,
        hold_predecessor: bool,
    ) {
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        ensure_print_handler_registered(client).unwrap();
        let input = f64_vector(&session, &mesh, &[3.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("ordered"), Vec::new(), vec![argument], None)
            .unwrap();
        let source = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let lowered = session.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        assert!(compiled.signature().has_ordered_io());
        let serialized = compiled.serialize().unwrap().unwrap();
        let restored = session.deserialize_compiled_program(&serialized).unwrap().unwrap();
        assert_eq!(restored.signature(), compiled.signature());
        let corrupted = corrupt_ordered_io_metadata(&serialized);
        assert!(matches!(
            session.deserialize_compiled_program(&corrupted),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "effect signature does not match executable physical outputs",
        ));
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
        let unordered = session.compile_program(&lowered).unwrap();
        assert!(unordered.signature().has_unordered_io());
        assert!(!unordered.signature().has_ordered_io());
        let manager = hold_predecessor.then(|| {
            let manager = client
                .host_to_device_transfer_manager(
                    vec![BufferSpecification::new(BufferType::Token, [])],
                    client.addressable_devices().unwrap()[0].default_memory().unwrap(),
                )
                .unwrap();
            let mut predecessor = session.effect_scope().reserve(Some(vec![mesh.devices()[0].id()])).unwrap();
            predecessor.submission.publish(Ok(Some(vec![Arc::new(manager.retrieve_buffer(0).unwrap())])));
            predecessor.submission.completion.complete(Ok(()));
            predecessor.published = true;
            manager
        });
        let ((), lines) = with_captured_prints(|| {
            assert!(session.execute_program(&compiled, vec![input.clone()]).unwrap().is_empty());
            assert!(session.execute_program(&restored, vec![input.clone()]).unwrap().is_empty());
            let independent = session.execute_compiled_async(&unordered, vec![input]).unwrap();
            if let Some(manager) = &manager {
                manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
            }
            independent.fence().block_until_ready().unwrap();
            session.effects_barrier().unwrap();
        });
        assert_eq!(lines.iter().filter(|line| line.as_str() == "ordered: [3.0]").count(), 2);
        assert_eq!(lines.iter().filter(|line| line.as_str() == "unordered: [3.0]").count(), 1);
    }

    /// Clears the persisted ordered I/O declaration while preserving the native executable bytes.
    fn corrupt_ordered_io_metadata(serialized: &[u8]) -> Vec<u8> {
        let header_size = 8 + size_of::<u64>();
        let metadata_size = u64::from_le_bytes(serialized[8..header_size].try_into().unwrap()) as usize;
        let metadata_end = header_size + metadata_size;
        let mut metadata: serde_json::Value = serde_json::from_slice(&serialized[header_size..metadata_end]).unwrap();
        metadata["ordered_io"] = false.into();
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut corrupted = serialized[..8].to_vec();
        corrupted.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        corrupted.extend_from_slice(&metadata);
        corrupted.extend_from_slice(&serialized[metadata_end..]);
        corrupted
    }

    #[test]
    fn test_effect_scope_reserve() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(vec![0])).unwrap();
        let unordered = scope.reserve(None).unwrap();
        let second = scope.reserve(Some(vec![0])).unwrap();
        assert_eq!(first.submission.identity, 0);
        assert_eq!(unordered.submission.identity, 1);
        assert_eq!(second.submission.identity, 2);
        assert!(first.predecessor.is_none());
        assert!(unordered.predecessor.is_none());
        assert!(Arc::ptr_eq(second.predecessor.as_ref().unwrap(), &first.submission));
        assert!(matches!(
            scope.reserve(Some(vec![1])),
            Err(Error::Execution(ExecutionError::ScopeDeviceMismatch { expected, actual }))
                if expected == vec![0] && actual == vec![1],
        ));
        // Cancelling the successor while its predecessor is unpublished must return without waiting.
        let cancelled = Arc::clone(&second.submission);
        std::thread::scope(|threads| {
            let (sender, receiver) = std::sync::mpsc::channel();
            threads.spawn(move || {
                drop(second);
                sender.send(()).unwrap();
            });
            let cancellation = receiver.recv_timeout(Duration::from_secs(5));
            let predecessor_pending = first.submission.completion.result().is_none();
            drop(first);
            assert_eq!(cancellation, Ok(()));
            assert!(predecessor_pending);
        });
        assert_eq!(cancelled.completion.result(), Some(Ok(())));
        assert!(cancelled.carrier().unwrap().is_none());
        drop(unordered);
        scope.barrier().unwrap();
    }

    #[test]
    fn test_effect_scope_reserve_device_assignment() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(vec![0, 1])).unwrap();
        let first_submission = Arc::clone(&first.submission);
        assert!(matches!(
            scope.reserve(Some(vec![1, 0])),
            Err(Error::Execution(ExecutionError::ScopeDeviceMismatch { expected, actual }))
                if expected == vec![0, 1] && actual == vec![1, 0],
        ));
        drop(first);
        let successor = scope.reserve(Some(vec![0, 1])).unwrap();
        assert!(Arc::ptr_eq(successor.predecessor.as_ref().unwrap(), &first_submission));
        drop(successor);
        scope.barrier().unwrap();
    }

    #[test]
    fn test_effect_scope_barrier_waits_for_fixed_snapshot() {
        let scope = EffectScope::default();
        let mut first = scope.reserve(None).unwrap();
        first.published = true;
        let references = Arc::strong_count(&first.submission);
        std::thread::scope(|threads| {
            let (sender, receiver) = std::sync::mpsc::channel();
            let scope = &scope;
            threads.spawn(move || sender.send(scope.barrier()).unwrap());
            // The barrier owns a submission reference only after capturing its fixed snapshot.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while Arc::strong_count(&first.submission) == references && std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
            let captured = Arc::strong_count(&first.submission) > references;
            let later = scope.reserve(None).unwrap();
            first.submission.completion.complete(Ok(()));
            let result = receiver.recv_timeout(Duration::from_secs(5));
            let later_pending = later.submission.completion.result().is_none();
            drop(later);
            assert!(captured);
            result.unwrap().unwrap();
            assert!(later_pending);
        });
        scope.barrier().unwrap();
    }

    #[test]
    fn test_effect_scope_acknowledge() {
        let scope = EffectScope::default();
        let mut first = scope.reserve(Some(vec![0])).unwrap();
        let mut second = scope.reserve(Some(vec![0])).unwrap();
        first.published = true;
        second.published = true;
        first.submission.publish(Ok(None));
        second.submission.publish(Ok(None));
        first.submission.completion.complete(Err(Arc::from("first effect failed")));
        assert!(matches!(scope.reserve(Some(vec![0])), Err(Error::Execution(ExecutionError::ScopeNeedsRecovery))));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        second.submission.completion.complete(Err(Arc::from("second effect failed")));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        let expected = concat!(
            "submission 0: first effect failed; submission 1: second effect failed; ",
            "call `effects_barrier()` again if more work was submitted, then `acknowledge_effect_errors()` ",
            "to recover once affected work is complete",
        );
        for _ in 0..2 {
            let error = scope.barrier().unwrap_err();
            assert!(matches!(&error,
                Error::Execution(ExecutionError::EffectSubmissionFailures { failures })
                    if failures == &vec![(0, Arc::from("first effect failed")), (1, Arc::from("second effect failed"))],
            ));
            assert_eq!(error.to_string(), expected);
        }
        scope.acknowledge().unwrap();
        scope.barrier().unwrap();
        let recovered = scope.reserve(Some(vec![1])).unwrap();
        assert!(recovered.predecessor.is_none());
        drop(recovered);
        // Acknowledgment changes scope bookkeeping, not outcomes retained by callers.
        assert_eq!(first.submission.completion.result(), Some(Err(Arc::from("first effect failed"))));
    }

    #[test]
    fn test_effect_scope_acknowledge_preserves_pending() {
        let scope = EffectScope::default();
        let mut pending = scope.reserve(None).unwrap();
        pending.published = true;
        scope.acknowledge().unwrap();
        assert_eq!(scope.state.lock().unwrap().submissions.len(), 1);
        pending.submission.completion.complete(Err(Arc::from("late completion failure")));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        assert!(matches!(
            scope.barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("late completion failure"))],
        ));
        scope.acknowledge().unwrap();
        assert_eq!(scope.state.lock().unwrap().submissions.len(), 0);
    }

    #[test]
    fn test_effect_reservation_drop_forwards_predecessor_carrier() {
        let scope = EffectScope::default();
        let mut first = scope.reserve(Some(vec![0])).unwrap();
        first.published = true;
        let second = scope.reserve(Some(vec![0])).unwrap();
        let third = scope.reserve(Some(vec![0])).unwrap();
        let final_submission = Arc::clone(&third.submission);
        std::thread::scope(|threads| {
            let (sender, receiver) = std::sync::mpsc::channel();
            threads.spawn(move || {
                drop(third);
                drop(second);
                sender.send(()).unwrap();
            });
            let cancellation = receiver.recv_timeout(Duration::from_secs(5));
            first.submission.publish(Err(Arc::from("predecessor publication failed")));
            first.submission.completion.complete(Err(Arc::from("predecessor publication failed")));
            assert_eq!(cancellation, Ok(()));
        });
        assert!(matches!(
            final_submission.carrier(),
            Err(reason) if reason.as_ref() == "predecessor publication failed",
        ));
        assert!(matches!(
            scope.barrier(),
            Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                if failures == vec![(0, Arc::from("predecessor publication failed"))],
        ));
    }

    #[test]
    fn test_ensure_effect_dispatch_allowed_rejects_managed_callback_reentrancy() {
        let client = execution_client();
        let session = session_with_client(client);
        let scope = EffectScope::default();
        let guard = EffectCallbackGuard::enter();
        let reason = "execution and effect barriers cannot be entered from a managed callback";
        assert!(matches!(scope.reserve(None), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.barrier(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(
            session.bind(OneOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), &[]),
            Err(ProgramError::InvalidArgument { message })
                if message == reason,
        ));
        drop(guard);
        scope.barrier().unwrap();
    }

    #[test]
    fn test_format_effect_failures() {
        assert_eq!(format_effect_failures(&[]), "");
        assert_eq!(
            format_effect_failures(&[(0, Arc::from("first effect failed")), (3, Arc::from("second effect failed"))]),
            "submission 0: first effect failed; submission 3: second effect failed",
        );
    }

    #[test]
    fn test_effect_submission_drop_releases_long_cancelled_chain() {
        let scope = EffectScope::default();
        for _ in 0..10_000 {
            drop(scope.reserve(Some(vec![0])).unwrap());
        }
        let tail = Arc::clone(scope.state.lock().unwrap().tail.as_ref().unwrap());
        assert!(tail.carrier().unwrap().is_none());
        assert_eq!(tail.completion.result(), Some(Ok(())));
        scope.barrier().unwrap();
        drop(scope);
        // The retained tail is the last owner of the entire cancelled chain.
        drop(tail);
    }

    #[test]
    fn test_session_array_ordered_effects() {
        let client = cpu_client(1);
        assert_session_array_ordered_effects(client, true);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_array_ordered_effects_on_cuda() {
        // This proves token transport on a second plugin. Native tokens created through the asynchronous transfer path
        // from the host to the device gate CUDA executions, and the CUDA print handler observes device values in
        // submission order.
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        assert_session_array_ordered_effects(&client, false);
    }

    #[test]
    fn test_session_effect_metadata_reload_and_unordered_completion() {
        let client = cpu_client(1);
        assert_domain_effect_metadata_reload_and_unordered_completion(client, true);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_effect_metadata_reload_and_unordered_completion_on_cuda() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        assert_domain_effect_metadata_reload_and_unordered_completion(&client, false);
    }

    #[test]
    fn test_session_device_ordered_effects_preserve_per_device_dependencies() {
        let client = cpu_client(2);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let session = session_with_client(client);
        ensure_print_handler_registered(client).unwrap();
        let r#type = ArrayType::new(DataType::F64, Shape::from(StaticShape::new(vec![2])))
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
        let scope = session.effect_scope();
        let mut predecessor = scope
            .reserve(Some(devices.iter().map(|device| Device::from_pjrt(device).unwrap().id()).collect()))
            .unwrap();
        predecessor
            .submission
            .publish(Ok(Some(managers.iter().map(|manager| Arc::new(manager.retrieve_buffer(0).unwrap())).collect())));
        predecessor.submission.completion.complete(Ok(()));
        predecessor.published = true;
        managers[1].transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
        let (independent, lines) = with_captured_prints(|| {
            session.execute_program(&program, vec![first]).unwrap();
            let tokens = scope.state.lock().unwrap().tail.as_ref().unwrap().carrier().unwrap().unwrap();
            let independent = std::thread::scope(|threads| {
                let (sender, receiver) = std::sync::mpsc::channel();
                threads.spawn(move || sender.send(tokens[1].ready().unwrap().r#await()).unwrap());
                let independent = receiver.recv_timeout(Duration::from_secs(2));
                // CPU itself serializes multi-device launches. Submit the next invocation without adding a Ryft
                // join, but observe the readiness of each device on the first invocation rather than promising overlap.
                session.execute_program(&program, vec![second]).unwrap();
                managers[0].transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
                independent
            });
            session.effects_barrier().unwrap();
            independent
        });
        assert!(matches!(independent, Ok(Ok(()))));
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "effect: [1.0]");
        let position = |value: &str| lines.iter().position(|line| line == value).unwrap();
        assert!(position("effect: [0.0]") < position("effect: [2.0]"));
        assert!(position("effect: [1.0]") < position("effect: [3.0]"));
    }
}
