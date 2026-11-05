//! Ordered effect reservation, completion, and recovery.

use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex};

use ryft_core::DeviceId;
use ryft_pjrt::Buffer;

use crate::Error;
use crate::domains::execution::ExecutionError;

// TODO(eaplatanios): Review this module.

/// Renders the failures observed by an effect barrier in their original submission order.
pub(super) fn format_effect_failures(failures: &[(u64, Arc<str>)]) -> String {
    failures
        .iter()
        .map(|(identity, error)| format!("submission {identity}: {error}"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Completion-only state that is safe to retain in PJRT callbacks without retaining any client-borrowing resource.
#[derive(Default)]
pub(super) struct EffectCompletion {
    /// Immutable terminal outcome, published once after submission or cancellation.
    result: Mutex<Option<Result<(), Arc<str>>>>,
    /// Wakes barrier callers when the terminal outcome becomes available.
    ready: Condvar,
}

impl EffectCompletion {
    /// Records completion without entering scope state or running user code.
    pub(super) fn complete(&self, result: Result<(), Arc<str>>) {
        *self.result.lock().expect("effect completion mutex poisoned") = Some(result);
        self.ready.notify_all();
    }

    /// Returns a nonblocking snapshot of the completion state.
    fn result(&self) -> Option<Result<(), Arc<str>>> {
        self.result.lock().expect("effect completion mutex poisoned").clone()
    }

    /// Waits for this invocation, including its reservation-to-submission interval.
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
enum EffectCarrier<'c> {
    /// The invocation has not published its physical result yet.
    #[default]
    Pending,

    /// The invocation published its physical result or submission failure.
    Published(Result<Option<Vec<Arc<Buffer<'c>>>>, Arc<str>>),

    /// Preparation was cancelled; successors inherit the preceding invocation's eventual carrier.
    Forward(Option<Arc<EffectSubmission<'c>>>),
}

/// One reserved invocation. Carrier ownership remains tied to the client lifetime through ordinary buffers.
pub(super) struct EffectSubmission<'c> {
    /// Monotonically increasing scope-local submission identity.
    identity: u64,
    /// Whether the invocation participates in the ordered I/O chain.
    ordered: bool,
    /// Buffer-free state retained by the completion observer.
    pub(super) completion: Arc<EffectCompletion>,
    /// Physical carrier publication or nonblocking forwarding of a cancelled reservation.
    carrier: Mutex<EffectCarrier<'c>>,
    /// Wakes submitting threads once the predecessor's physical result is available.
    published: Condvar,
}

impl<'c> EffectSubmission<'c> {
    /// Publishes a carrier after execution handoff, or forwards the predecessor after a cancelled submission.
    pub(super) fn publish(&self, carrier: Result<Option<Vec<Arc<Buffer<'c>>>>, Arc<str>>) {
        *self.carrier.lock().expect("effect publication mutex poisoned") = EffectCarrier::Published(carrier);
        self.published.notify_all();
    }

    /// Waits only for host publication, following cancelled reservations without recursion or device waits.
    pub(super) fn carrier(&self) -> Result<Option<Vec<Arc<Buffer<'c>>>>, Arc<str>> {
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

    /// Waits for this reservation's publication state, without following forwarding links under its mutex.
    fn publication(&self) -> EffectCarrier<'c> {
        let mut carrier = self.carrier.lock().expect("effect publication mutex poisoned");
        while matches!(*carrier, EffectCarrier::Pending) {
            carrier = self.published.wait(carrier).expect("effect publication mutex poisoned");
        }
        carrier.clone()
    }
}

impl Drop for EffectSubmission<'_> {
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

/// Synchronized scope metadata; callbacks never acquire this mutex or own this state.
#[derive(Default)]
struct EffectScopeState<'c> {
    /// Identity of the next reserved invocation.
    next_identity: u64,
    /// Latest ordered invocation, retained even after its successful completion is retired.
    tail: Option<Arc<EffectSubmission<'c>>>,
    /// Outstanding invocations and unacknowledged failures, in reservation order.
    submissions: Vec<Arc<EffectSubmission<'c>>>,
    /// Failures reported by completed barriers.
    observed: HashSet<u64>,
    /// Ordered device assignment selected by the chain; changes require an explicit scope fork.
    device: Option<Vec<DeviceId>>,
}

/// Session-owned ordering and completion scope, independent of numerical values and compilation artifacts.
#[derive(Default)]
pub(super) struct EffectScope<'c> {
    /// Reservation, observation, and recovery state; never held across PJRT calls or device waits.
    state: Mutex<EffectScopeState<'c>>,
}

impl<'c> EffectScope<'c> {
    /// Reserves a position and snapshots the predecessor atomically before releasing the scope mutex.
    pub(super) fn reserve(&self, device: Option<Vec<DeviceId>>) -> Result<EffectReservation<'c>, Error> {
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

/// Client-borrowing reservation, released on every preparation or submission failure.
pub(super) struct EffectReservation<'c> {
    /// Invocation being prepared or published.
    pub(super) submission: Arc<EffectSubmission<'c>>,
    /// Predecessor whose carrier will be forwarded if submission fails before handoff.
    pub(super) predecessor: Option<Arc<EffectSubmission<'c>>>,
    /// Whether device work and its carrier have been published.
    pub(super) published: bool,
}

impl Drop for EffectReservation<'_> {
    fn drop(&mut self) {
        if !self.published {
            // Cancellation must not wait: the predecessor may still be prepared on this same calling thread.
            *self.submission.carrier.lock().expect("effect publication mutex poisoned") =
                EffectCarrier::Forward(self.predecessor.take());
            self.submission.published.notify_all();
            self.submission.completion.complete(Ok(()));
        }
    }
}

std::thread_local! {
    /// Reentrancy detection only; effect ordering lives in the explicit session scope.
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

/// Rejects managed-callback reentrancy before any scope or reference locks are taken.
pub(super) fn ensure_effect_dispatch_allowed() -> Result<(), Error> {
    if EFFECT_CALLBACK_DEPTH.with(|depth| depth.get() != 0) {
        Err(ExecutionError::ReentrantDispatch.into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, ArrayType, AssertOperation, BatchAxis, CompareOperation, ComparisonDirection, Context,
        CustomCallOperation, DataType, Device, DeviceMesh, Dimension as CoreDimension, DimensionBounds,
        DimensionFromScalarOperation, DimensionType, EffectClass, LogicalMesh, MeshAxis, MeshAxisType, OneOperation,
        Placeholder, PrintOperation, ProgramError, Shape, ShardMapBoundary, ShardMapOperation, Sharding,
        ShardingDimension, StaticShape, Typed, batch,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::domains::compilation::CompilationError;
    use crate::domains::options::Options;
    use crate::domains::persistence::tests::corrupt_ordered_io_metadata;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::experimental::ops::{Constant, Operation, ProgramBuilder};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{Array, FromPjrt};

    use super::*;

    #[test]
    fn test_session_array_ordered_effects() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        assert_session_array_ordered_effects(&client, true);
    }
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_array_ordered_effects_on_cuda() {
        // The second-plugin transport proof: native tokens created through the async host-to-device transfer path
        // gate CUDA executions, and the CUDA print handler observes device values in submission order.
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        assert_session_array_ordered_effects(&client, false);
    }
    #[test]
    fn test_domain_effect_metadata_reload_and_unordered_completion() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        assert_domain_effect_metadata_reload_and_unordered_completion(&client, true);
    }
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_domain_effect_metadata_reload_and_unordered_completion_on_cuda() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        assert_domain_effect_metadata_reload_and_unordered_completion(&client, false);
    }
    #[test]
    fn test_domain_device_ordered_effects_preserve_per_device_dependencies() {
        use crate::operations::debugging::{
            PRINT_CUSTOM_CALL_TARGET, ensure_print_handler_registered, with_captured_prints,
        };

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
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
        let domain = Session::new(&client).domain();
        ensure_print_handler_registered(&client).unwrap();
        let r#type = ArrayType::new(DataType::F64, Shape::from(StaticShape::new(vec![2])))
            .with_sharding(
                Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Sharded(vec!["x".to_string()])])
                    .unwrap(),
            )
            .unwrap();
        let first =
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes(&[0.0f64, 1.0])).unwrap();
        let second =
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes(&[2.0f64, 3.0])).unwrap();
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
        let lowered = domain.lower_program(&source, 0, &Options::new(mesh)).unwrap();
        let program = domain.compile_program(&lowered).unwrap();
        assert!(program.signature.has_device_ordered_io());
        let devices = program.executable.addressable_devices().unwrap();
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
        let scope = domain.effect_scope.as_ref();
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
            domain.execute_program(&program, vec![first]).unwrap();
            let tokens = scope.state.lock().unwrap().tail.as_ref().unwrap().carrier().unwrap().unwrap();
            let independent = std::thread::scope(|threads| {
                let (sender, receiver) = std::sync::mpsc::channel();
                threads.spawn(move || sender.send(tokens[1].ready().unwrap().r#await()).unwrap());
                let independent = receiver.recv_timeout(Duration::from_secs(2));
                // CPU itself serializes multi-device launches. Submit the next invocation without adding a Ryft
                // join, but observe per-device readiness on the first invocation rather than promising overlap.
                domain.execute_program(&program, vec![second]).unwrap();
                managers[0].transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
                independent
            });
            domain.effects_barrier().unwrap();
            independent
        });
        assert!(matches!(independent, Ok(Ok(()))));
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "effect: [1.0]");
        let position = |value: &str| lines.iter().position(|line| line == value).unwrap();
        assert!(position("effect: [0.0]") < position("effect: [2.0]"));
        assert!(position("effect: [1.0]") < position("effect: [3.0]"));
    }
    #[test]
    fn test_eager_bind_executes_print_effect() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        assert_eager_bind_executes_print_effect(&client);
    }
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_eager_bind_executes_print_effect_on_cuda() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        assert_eager_bind_executes_print_effect(&client);
    }
    #[test]
    fn test_batched_print_executes_on_cpu() {
        use ryft_core::Print;

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input = f64_vector(&domain, &mesh, &[1.5, 2.5]);

        // Eager batching binds one print of the whole batch through the `@ryft.print` host callback.
        let (output, lines) = with_captured_prints(|| {
            batch(|item| item.print("x"), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap()
        });
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(read_f64s(&client, &output), vec![1.5, 2.5]);

        // Batching under a staging parent keeps the print in the staged program, which prints the whole batch once
        // per execution.
        let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let output =
                    batch(|item| item.print("x"), inputs[0].clone(), BatchAxis::new(0), BatchAxis::new(0), None);
                vec![output.unwrap()]
            },
            vec![input.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh)).unwrap();
        assert!(lowered.stable_hlo().contains("@ryft.print"), "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let (outputs, lines) = with_captured_prints(|| domain.execute_program(&compiled, vec![input]).unwrap());
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f64s(&client, &outputs[0]), vec![1.5, 2.5]);
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
            assert!(cancellation.is_ok());
            assert!(predecessor_pending);
        });
        assert_eq!(cancelled.completion.result(), Some(Ok(())));
        assert!(cancelled.carrier().unwrap().is_none());
        drop(unordered);
        assert!(scope.barrier().is_ok());
    }
    #[test]
    fn test_effect_reservation_cancelled_forwarding() {
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
            assert!(cancellation.is_ok());
        });
        assert!(
            matches!(final_submission.carrier(), Err(reason) if reason.as_ref() == "predecessor publication failed")
        );
        assert!(matches!(scope.barrier(), Err(Error::Execution(ExecutionError::EffectSubmissionFailures { .. }))));
    }
    #[test]
    fn test_effect_submission_drop_cancelled_chain() {
        let scope = EffectScope::default();
        for _ in 0..10_000 {
            drop(scope.reserve(Some(vec![0])).unwrap());
        }
        let tail = Arc::clone(scope.state.lock().unwrap().tail.as_ref().unwrap());
        assert!(tail.carrier().unwrap().is_none());
        assert_eq!(tail.completion.result(), Some(Ok(())));
        assert!(scope.barrier().is_ok());
        drop(scope);
        // The retained tail is the last owner of the entire cancelled chain.
        drop(tail);
    }
    #[test]
    fn test_effect_scope_reserve_device_assignment() {
        let scope = EffectScope::default();
        let first = scope.reserve(Some(vec![0, 1])).unwrap();
        assert!(matches!(scope.reserve(Some(vec![1, 0])),
            Err(Error::Execution(ExecutionError::ScopeDeviceMismatch { expected, actual }))
                if expected == vec![0, 1] && actual == vec![1, 0]));
        drop(first);
        let successor = scope.reserve(Some(vec![0, 1])).unwrap();
        assert!(successor.predecessor.is_some());
        drop(successor);
        assert!(scope.barrier().is_ok());
    }
    #[test]
    fn test_effect_scope_barrier_snapshot() {
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
            assert!(result.unwrap().is_ok());
            assert!(later_pending);
        });
        assert!(scope.barrier().is_ok());
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
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment)),));
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
        assert!(scope.acknowledge().is_ok());
        assert!(scope.barrier().is_ok());
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
        assert!(scope.acknowledge().is_ok());
        assert_eq!(scope.state.lock().unwrap().submissions.len(), 1);
        pending.submission.completion.complete(Err(Arc::from("late completion failure")));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::PrematureAcknowledgment))));
        assert!(matches!(scope.barrier(), Err(Error::Execution(ExecutionError::EffectSubmissionFailures { .. }))));
        assert!(scope.acknowledge().is_ok());
        assert_eq!(scope.state.lock().unwrap().submissions.len(), 0);
    }
    #[test]
    fn test_effect_scope_managed_callback_reentrancy() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let scope = EffectScope::default();
        let guard = EffectCallbackGuard::enter();
        let reason = "execution and effect barriers cannot be entered from a managed callback";
        assert!(matches!(scope.reserve(None), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.barrier(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(scope.acknowledge(), Err(Error::Execution(ExecutionError::ReentrantDispatch))));
        assert!(matches!(
            domain.bind(OneOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), &[]),
            Err(ProgramError::InvalidArgument { message })
                if message == reason,
        ));
        drop(guard);
        assert!(scope.barrier().is_ok());
    }

    /// Submits two prints through one session on one device of `client` and checks that they print in submission order.
    /// With `hold_predecessor`, the two prints are additionally submitted behind an outstanding predecessor token (a
    /// not-yet-transferred host-to-device token buffer) and must stay pending until it is released. That fixture only
    /// works on plugins that submit executables without waiting for unrecorded input definition events on the host (the
    /// CPU plugin); the CUDA plugin blocks the submitting thread until every input event is recorded, so there the
    /// chain is proven through submission order and barrier completion alone.
    fn assert_session_array_ordered_effects(client: &Client<'_>, hold_predecessor: bool) {
        use crate::operations::debugging::with_captured_prints;
        use ryft_core::Print;

        let mesh = domain_mesh(client, "x", 1);
        let session = Session::new(client);
        let r#type = ArrayType::new(DataType::F64, Shape::from(StaticShape::new(vec![1])));
        let first = session.array(r#type.clone(), mesh.clone(), 1.0f64.to_ne_bytes()).unwrap();
        let second = session.array(r#type, mesh.clone(), 2.0f64.to_ne_bytes()).unwrap();
        // Model an outstanding predecessor without relying on compiler cost heuristics or expensive dummy work.
        let manager = hold_predecessor.then(|| {
            let manager = client
                .host_to_device_transfer_manager(
                    vec![BufferSpecification::new(BufferType::Token, [])],
                    client.addressable_devices().unwrap()[0].default_memory().unwrap(),
                )
                .unwrap();
            let mut predecessor = session.default_effect_scope.reserve(Some(vec![mesh.devices()[0].id()])).unwrap();
            predecessor.submission.publish(Ok(Some(vec![Arc::new(manager.retrieve_buffer(0).unwrap())])));
            predecessor.submission.completion.complete(Ok(()));
            predecessor.published = true;
            manager
        });
        let tail_pending = || {
            !session
                .default_effect_scope
                .state
                .lock()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .carrier()
                .unwrap()
                .unwrap()[0]
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
        assert_eq!(session.compilation_context().program_count(), 2);
    }

    /// Compiles ordered and unordered prints on `client`, checks the persisted effect metadata roundtrip and its
    /// corruption detection, then executes the compiled, restored, and unordered programs and verifies the ordered
    /// prints fire twice and the unordered print once. With `hold_predecessor`, the executions are submitted behind
    /// an outstanding predecessor token that is released only afterwards; refer to
    /// [`assert_session_array_ordered_effects`] for why that fixture is CPU-only.
    fn assert_domain_effect_metadata_reload_and_unordered_completion(client: &Client<'_>, hold_predecessor: bool) {
        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

        let mesh = domain_mesh(client, "x", 1);
        let session = Session::new(client);
        let domain = session.domain();
        ensure_print_handler_registered(client).unwrap();
        let input = f64_vector(&domain, &mesh, &[3.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("ordered"), Vec::new(), vec![argument], None)
            .unwrap();
        let source = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let lowered = domain.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        assert!(compiled.signature.has_ordered_io());
        let serialized = domain.serialize_compiled_program(&compiled).unwrap().unwrap();
        let restored = domain.deserialize_compiled_program(&serialized).unwrap().unwrap();
        assert_eq!(restored.signature, compiled.signature);
        let corrupted = corrupt_ordered_io_metadata(&serialized);
        assert!(matches!(
            domain.deserialize_compiled_program(&corrupted),
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
        let lowered = domain.lower_program(&source, 0, &Options::new(mesh.clone())).unwrap();
        let unordered = domain.compile_program(&lowered).unwrap();
        assert!(unordered.signature.has_unordered_io());
        assert!(!unordered.signature.has_ordered_io());
        let manager = hold_predecessor.then(|| {
            let manager = client
                .host_to_device_transfer_manager(
                    vec![BufferSpecification::new(BufferType::Token, [])],
                    client.addressable_devices().unwrap()[0].default_memory().unwrap(),
                )
                .unwrap();
            let mut predecessor = session.default_effect_scope.reserve(Some(vec![mesh.devices()[0].id()])).unwrap();
            predecessor.submission.publish(Ok(Some(vec![Arc::new(manager.retrieve_buffer(0).unwrap())])));
            predecessor.submission.completion.complete(Ok(()));
            predecessor.published = true;
            manager
        });
        let ((), lines) = with_captured_prints(|| {
            assert!(domain.execute_program(&compiled, vec![input.clone()]).unwrap().is_empty());
            assert!(domain.execute_program(&restored, vec![input.clone()]).unwrap().is_empty());
            let independent = domain.execute_compiled_async(&unordered, vec![input]).unwrap();
            if let Some(manager) = &manager {
                manager.transfer_data(0, Arc::new([] as [u8; 0]), 0, true).unwrap().r#await().unwrap();
            }
            independent.fence().block_until_ready().unwrap();
            domain.effects_barrier().unwrap();
        });
        assert_eq!(lines.iter().filter(|line| line.as_str() == "ordered: [3.0]").count(), 2);
        assert_eq!(lines.iter().filter(|line| line.as_str() == "unordered: [3.0]").count(), 1);
    }

    /// Eagerly binds one `print` on `client` and verifies that the handler fires once with the device value and that
    /// the payload passes through unchanged.
    fn assert_eager_bind_executes_print_effect(client: &Client<'_>) {
        use ryft_core::PrintOperation;

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

        let mesh = domain_mesh(client, "x", 1);
        let domain = array_domain(client);
        ensure_print_handler_registered(client).unwrap();

        // The effectful `print` rides the compiled per-operation program as a token-threaded `@ryft.print` custom
        // call: eagerly binding it fires the host callback once and passes the payload through unchanged.
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_host_buffer(
            domain.parent(),
            r#type,
            mesh.clone(),
            values_to_bytes::<f64>(&[1.5, 2.5]).as_slice(),
        )
        .unwrap();
        let (outputs, lines) =
            with_captured_prints(|| domain.bind(PrintOperation::<ArrayType>::new("x"), Vec::new(), &[input]).unwrap());

        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(outputs.len(), 1);
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = outputs[0]
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f64>(bytes.as_slice()), vec![1.5, 2.5]);
    }

    #[test]
    fn test_compiled_dimension_assertion_does_not_merge_ordered_io_chains_on_cpu() {
        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);

        let mut builder = ProgramBuilder::new();
        let left_input = builder.add_input(input_type.clone().into());
        let right_input = builder.add_input(input_type.clone().into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("left"), Vec::new(), vec![left_input], None)
            .unwrap();
        let left = builder
            .add_instruction(DimensionFromScalarOperation::new(left_type.clone()), Vec::new(), vec![left_input], None)
            .unwrap()[0];
        let right = builder
            .add_instruction(DimensionFromScalarOperation::new(right_type.clone()), Vec::new(), vec![right_input], None)
            .unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
                Vec::new(),
                vec![left, right],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("`left` <= `right`")
                    .with_labels(vec!["left".to_owned(), "right".to_owned()]),
                Vec::new(),
                vec![predicate, left, right],
                None,
            )
            .unwrap();
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("right"), Vec::new(), vec![right_input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        // Only assertions create a local chain; ordered I/O is seeded from the hidden executable argument.
        assert!(lowered.signature.has_ordered_io());
        assert!(lowered.signature.requires_assertion_handler());
        assert_eq!(lowered.stable_hlo().matches("stablehlo.after_all").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };

        let (output, lines) = with_captured_prints(|| {
            let output = domain.execute_program(&compiled, vec![input(3), input(4)]).unwrap();
            assert_eq!(read_i64s(&client, &output[0]), vec![4]);
            output
        });
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("left: "), "{:?}", lines);
        assert!(lines[1].starts_with("right: "), "{:?}", lines);
        assert_eq!(read_i64s(&client, &output[0]), vec![4]);
    }
}
