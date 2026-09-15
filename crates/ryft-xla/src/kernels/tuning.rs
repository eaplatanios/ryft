//! Bounded schedule measurements using whole-invocation completion and the existing auxiliary disk cache.
//!
//! Samples measure host submission through [`ReferenceExecution::await`], including dispatch and synchronization.
//! They are not device timestamps. Preparation and warmup are excluded from samples but included in the deadline.
//! Cancellation and deadlines are cooperative: submitted work is always awaited, and its buffers must remain owned
//! by the runner until completion. Share one [`KernelTuner`] for a device to serialize measurements; other processes
//! and unrelated device workloads remain the caller's responsibility. No golden performance threshold is assumed.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use ryft_core::kernels::{KernelCompiler, KernelExtension, KernelSchedule, VerifiedKernel};
use ryft_core::{DiskCache, ReferenceExecution};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::kernels::staging::{XlaKernelExecutionFacts, XlaKernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};

/// Invalid tuning input, incomplete execution, or incompatible persisted measurements.
#[derive(Debug, Error)]
pub enum KernelTuningError {
    /// Input or persisted metadata violates the bounded measurement contract.
    #[error("invalid kernel tuning request: {message}")]
    Invalid {
        /// Exact invalid request or measurement condition.
        message: String,
    },

    /// Another measurement owns this tuner's device gate.
    #[error("kernel tuner is busy")]
    Busy,

    /// The caller cancelled before a complete result could be published.
    #[error("kernel tuning was cancelled")]
    Cancelled,

    /// The total preparation, warmup, and execution deadline expired.
    #[error("kernel tuning time budget expired")]
    Timeout,

    /// Adapter admission or preparation failed.
    #[error("kernel tuning compilation failed: {message}")]
    Compiler {
        /// Concrete compiler or preparation diagnostic.
        message: String,
    },

    /// Existing XLA embedding validation failed.
    #[error(transparent)]
    Embedding(#[from] KernelEmbeddingError),

    /// Persistent storage failed checksum, envelope, or filesystem validation.
    #[error(transparent)]
    Storage(#[from] std::io::Error),

    /// Versioned measurement metadata could not be decoded.
    #[error(transparent)]
    Payload(#[from] serde_json::Error),
}

/// Explicit limits shared by admission, execution, and persisted-result validation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KernelTuningBudget {
    /// Maximum number of candidate schedules, capped at 256.
    maximum_candidates: usize,

    /// Completed unmeasured invocations per candidate, capped at 1000.
    warmups: usize,

    /// Completed measured invocations per candidate, in `1..=1000`.
    repeats: usize,

    /// Cooperative total wall-time budget, including candidate preparation.
    maximum_time: Duration,
}

impl KernelTuningBudget {
    /// Validates explicit count and time limits. Zero warmup is allowed; zero time is rejected.
    pub fn new(
        maximum_candidates: usize,
        warmups: usize,
        repeats: usize,
        maximum_time: Duration,
    ) -> Result<Self, KernelTuningError> {
        if !(1..=256).contains(&maximum_candidates)
            || warmups > 1000
            || !(1..=1000).contains(&repeats)
            || maximum_time.is_zero()
        {
            return Err(KernelTuningError::Invalid {
                message: "invalid candidate, warmup, repeat, or time limit".into(),
            });
        }
        Ok(Self { maximum_candidates, warmups, repeats, maximum_time })
    }
}

/// Immutable, fingerprinted search request with exact semantic, compiler, embedding, device, and workload identity.
#[derive(Clone, Debug)]
pub struct KernelTuningRequest {
    /// Finite schedules in stable caller order, with duplicates rejected.
    candidates: Vec<KernelSchedule>,

    /// Validated limits, also included in persistent identity.
    budget: KernelTuningBudget,

    /// Canonical versioned request encoding, including every candidate's compiler key.
    identity: Vec<u8>,
}

impl KernelTuningRequest {
    /// Fingerprints a verified definition and finite schedules using side-effect-free configuration queries.
    /// Compiler admission is deliberately deferred to the runner's budgeted preparation, because it may invoke tools.
    /// Target validation must inspect only the supplied immutable execution facts; configuration queries must not
    /// launch compiler processes. Existing Mosaic and cuTile configuration queries satisfy this contract.
    ///
    /// `environment` must identify workload inputs, driver/runtime settings not covered by `facts`, contention policy,
    /// and any preparation behavior that can affect measurements. Reuse is valid only for identical workloads and
    /// environments. The compiler and embedding supply their existing complete configuration identities.
    pub fn new<Extension, Compiler, Embedding>(
        kernel: &VerifiedKernel<'_, Extension>,
        compiler: &Compiler,
        target: &Compiler::Target,
        options: &Compiler::Options,
        embedding: &Embedding,
        facts: &XlaKernelExecutionFacts,
        environment: &[u8],
        candidates: Vec<KernelSchedule>,
        budget: KernelTuningBudget,
    ) -> Result<Self, KernelTuningError>
    where
        Extension: KernelExtension,
        Compiler: KernelCompiler<Extension>,
        Compiler::Target: XlaKernelTarget,
        Embedding: KernelOutputEmbedding<Compiler::Output, Extension>,
    {
        if candidates.is_empty() || candidates.len() > budget.maximum_candidates || environment.is_empty() {
            return Err(KernelTuningError::Invalid {
                message: "empty search, excessive candidates, or empty environment".into(),
            });
        }
        target.admit_execution(facts)?;
        let mut keys = Vec::with_capacity(candidates.len());
        for (index, schedule) in candidates.iter().enumerate() {
            if candidates[..index].contains(schedule) {
                return Err(KernelTuningError::Invalid { message: "duplicate candidate schedule".into() });
            }
            let key = compiler
                .configuration_key(target, options, schedule)
                .map_err(|error| KernelTuningError::Compiler { message: error.to_string() })?;
            keys.push((
                schedule.pipeline_stages().map(NonZeroUsize::get),
                schedule.buffering_depth().map(NonZeroUsize::get),
                schedule.maximum_scratch_bytes(),
                key,
            ));
        }
        let semantic = kernel
            .definition()
            .semantic_key()
            .map_err(|error| KernelTuningError::Invalid { message: error.to_string() })?;
        let identity = serde_json::to_vec(&(
            1,
            "host-submit-completion-upper-median-v1",
            semantic,
            embedding.configuration_key()?,
            facts.configuration_key()?,
            environment,
            &budget,
            keys,
        ))?;
        // A JSON byte needs at most four bytes; a Duration sample needs fewer than 64. Reject requests
        // whose complete bounded result could exceed storage limits before running any workload.
        let maximum_record = identity
            .len()
            .checked_mul(4)
            .and_then(|size| size.checked_add(candidates.len() * budget.repeats * 64 + 4096));
        if maximum_record.is_none_or(|size| size > MAXIMUM_RECORD_BYTES) {
            return Err(KernelTuningError::Invalid {
                message: "request identity or sample budget exceeds size limit".into(),
            });
        }
        Ok(Self { candidates, budget, identity })
    }

    /// Returns candidates in the exact order used for warmup, measurement, and equal-median tie breaking.
    pub fn candidates(&self) -> &[KernelSchedule] {
        &self.candidates
    }
}

/// Execution integration for one fixed workload. Preparation must select the supplied fingerprinted schedule.
///
/// Each invocation must use equivalent inputs. Mutable workloads must include deterministic reset in `execute` and
/// its measured completion; functional workloads can reuse immutable inputs. `prepare` must run normal compiler
/// admission through [`VerifiedKernel::compile`] or the XLA compiler binding before artifact reuse or execution. It
/// may compile and allocate, but must finish asynchronous setup before returning. `execute` must return the existing
/// whole-invocation completion, retaining all needed buffers until it completes. The tuner always awaits it; returning
/// an already-ready wrapper for merely submitted device work violates this contract. Compilation that can be cancelled
/// should use the same cancellation signal supplied to [`KernelTuner::run`].
pub trait KernelTuningRunner {
    /// Prepares one candidate outside sample timing, including deterministic input initialization.
    fn prepare(&mut self, schedule: &KernelSchedule) -> Result<(), KernelTuningError>;

    /// Submits one equivalent invocation with its actual completion and error chain.
    fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError>;
}

/// Private persistence record, decoded only through request-aware validation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasurementRecord {
    /// Measurement encoding version.
    version: u32,

    /// Exact fingerprinted request, including methodology and limits.
    identity: Vec<u8>,

    /// Submission-through-completion samples, ordered by candidate then repetition.
    samples: Vec<Vec<Duration>>,

    /// Lowest upper median, with ties resolved by the first candidate.
    best_candidate: usize,
}

/// Complete measurements in candidate order. Loading validates the complete private persistence record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelTuningResult {
    /// Request-validated measured samples and ranking.
    record: MeasurementRecord,
}

impl KernelTuningResult {
    /// Returns completed wall-time samples; no warmup or compilation times are included.
    pub fn samples(&self) -> &[Vec<Duration>] {
        &self.record.samples
    }

    /// Returns the winning index into the original request's candidate sequence.
    pub fn best_candidate(&self) -> usize {
        self.record.best_candidate
    }

    /// Checks exact compatibility and all bounded record invariants before exposing persisted measurements.
    fn validate(&self, request: &KernelTuningRequest) -> Result<(), KernelTuningError> {
        if self.record.version != 1
            || self.record.identity != request.identity
            || self.record.samples.len() != request.candidates.len()
            || self.record.samples.iter().any(|samples| {
                samples.len() != request.budget.repeats
                    || samples.iter().any(|sample| *sample > request.budget.maximum_time)
            })
            || self
                .record
                .samples
                .iter()
                .flatten()
                .try_fold(Duration::ZERO, |total, sample| total.checked_add(*sample))
                .is_none_or(|total| total > request.budget.maximum_time)
            || self.record.best_candidate != Self::winner(&self.record.samples)
        {
            return Err(KernelTuningError::Invalid { message: "stale or inconsistent measurement record".into() });
        }
        Ok(())
    }

    /// Selects the lowest upper median, retaining candidate order for ties.
    fn winner(samples: &[Vec<Duration>]) -> usize {
        samples
            .iter()
            .enumerate()
            .min_by_key(|(_, samples)| {
                let mut sorted = (*samples).clone();
                sorted.sort_unstable();
                sorted[sorted.len() / 2]
            })
            .map(|(index, _)| index)
            .unwrap()
    }
}

/// Maximum decoded measurement payload, before JSON allocation. DiskCache separately bounds decompression.
const MAXIMUM_RECORD_BYTES: usize = 8 * 1024 * 1024;

/// Serialized per-device measurements with optional atomic, checksummed auxiliary persistence.
///
/// A busy tuner rejects immediately rather than waiting outside the caller's deadline. Separate processes may
/// atomically replace complete records; they must coordinate device exclusivity themselves for meaningful timings.
pub struct KernelTuner {
    /// Exclusive measurement gate; never protects partially published results.
    gate: Mutex<()>,

    /// Existing persistent storage, using a distinct namespace and metadata entries instead of executables.
    cache: Option<Arc<DiskCache>>,
}

impl KernelTuner {
    /// Creates a measurement owner. `None` disables persistence without changing measurement semantics.
    pub fn new(cache: Option<Arc<DiskCache>>) -> Self {
        Self { gate: Mutex::new(()), cache }
    }

    /// Loads only a complete compatible measurement record; corruption remains an explicit error.
    pub fn load(&self, request: &KernelTuningRequest) -> Result<Option<KernelTuningResult>, KernelTuningError> {
        let Some(cache) = &self.cache else {
            return Ok(None);
        };
        let Some(bytes) = cache.get_auxiliary("kernel-tuning-v1", &request.identity)? else {
            return Ok(None);
        };
        if bytes.len() > MAXIMUM_RECORD_BYTES {
            return Err(KernelTuningError::Invalid { message: "measurement record exceeds size limit".into() });
        }
        let result = KernelTuningResult { record: serde_json::from_slice(&bytes)? };
        result.validate(request)?;
        Ok(Some(result))
    }

    /// Measures every candidate and publishes only after all invocations complete within the budget.
    ///
    /// This function deliberately does not consult the cache: callers explicitly choose `load` or a new experiment.
    /// Cancellation and timeout never interrupt submitted work; they reject its measurement after awaiting completion.
    /// Invocation failures abort the entire experiment. No partial record replaces an earlier successful result.
    pub fn run(
        &self,
        request: &KernelTuningRequest,
        runner: &mut impl KernelTuningRunner,
        cancelled: &AtomicBool,
    ) -> Result<KernelTuningResult, KernelTuningError> {
        let started = Instant::now();
        let check = || {
            if cancelled.load(Ordering::Acquire) {
                Err(KernelTuningError::Cancelled)
            } else if started.elapsed() >= request.budget.maximum_time {
                Err(KernelTuningError::Timeout)
            } else {
                Ok(())
            }
        };
        check()?;
        let _guard = match self.gate.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Err(KernelTuningError::Busy),
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
        };
        let mut samples = Vec::with_capacity(request.candidates.len());
        for schedule in &request.candidates {
            check()?;
            runner.prepare(schedule)?;
            check()?;
            for _ in 0..request.budget.warmups {
                runner.execute().r#await()?;
                check()?;
            }
            let mut candidate = Vec::with_capacity(request.budget.repeats);
            for _ in 0..request.budget.repeats {
                check()?;
                let submitted = Instant::now();
                runner.execute().r#await()?;
                let elapsed = submitted.elapsed();
                check()?;
                candidate.push(elapsed);
            }
            samples.push(candidate);
        }
        let result = KernelTuningResult {
            record: MeasurementRecord {
                version: 1,
                identity: request.identity.clone(),
                best_candidate: KernelTuningResult::winner(&samples),
                samples,
            },
        };
        result.validate(request)?;
        let bytes = serde_json::to_vec(&result.record)?;
        if bytes.len() > MAXIMUM_RECORD_BYTES {
            return Err(KernelTuningError::Invalid { message: "measurement record exceeds size limit".into() });
        }
        check()?;
        if let Some(cache) = &self.cache {
            cache.put_auxiliary("kernel-tuning-v1", &request.identity, &bytes)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{Grid, KernelCallOperation, KernelCompilationError, KernelDefinition};
    use ryft_core::operations::custom_call::CustomCallOperation;
    use ryft_core::{ReferenceCompletion, ReferenceCompletionBackend};

    use super::*;

    /// Minimal compiler whose options and schedules have independent exact identities.
    struct Compiler;

    /// Explicit local fixture execution target.
    struct Target;

    impl XlaKernelTarget for Target {
        fn admit_execution(&self, _facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
            Ok(())
        }
    }

    impl KernelCompiler for Compiler {
        type Target = Target;
        type Options = u8;
        type Output = ();
        type Error = std::io::Error;

        fn admit(
            &self,
            _kernel: &VerifiedKernel<'_>,
            _target: &Target,
            _options: &u8,
            _schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<Self::Error>> {
            panic!("constructing or loading measurements must not invoke compiler admission")
        }

        fn configuration_key(
            &self,
            _target: &Target,
            options: &u8,
            _schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<Self::Error>> {
            Ok(vec![*options])
        }

        fn compile(
            &self,
            _kernel: &VerifiedKernel<'_>,
            _target: &Target,
            _options: &u8,
            _schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<Self::Error>> {
            Ok(())
        }
    }

    /// Fixture embedding with a distinct ABI identity.
    struct Embedding(u8);

    impl KernelOutputEmbedding<()> for Embedding {
        fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
            Ok(vec![self.0])
        }

        fn custom_call(
            &self,
            _kernel: &VerifiedKernel<'_>,
            _output: &(),
        ) -> Result<CustomCallOperation, KernelEmbeddingError> {
            Ok(CustomCallOperation::new("fixture", vec![]))
        }
    }

    /// Constructs a fingerprinted empty kernel workload without external state or device dependencies.
    fn request() -> KernelTuningRequest {
        request_with(
            1,
            2,
            "1",
            b"empty-workload",
            vec![KernelSchedule::default(), KernelSchedule::default().with_maximum_scratch_bytes(0)],
        )
        .unwrap()
    }

    /// Varies independently owned identity components while retaining one semantic workload.
    fn request_with(
        options: u8,
        embedding: u8,
        platform_version: &str,
        environment: &[u8],
        candidates: Vec<KernelSchedule>,
    ) -> Result<KernelTuningRequest, KernelTuningError> {
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(), |_| Ok(()))
                .unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        KernelTuningRequest::new(
            &kernel,
            &Compiler,
            &Target,
            &options,
            &Embedding(embedding),
            &XlaKernelExecutionFacts {
                platform_name: "fixture".into(),
                platform_version: platform_version.into(),
                pjrt_version: ryft_pjrt::Version { major: 0, minor: 115 },
                has_ffi_extension: false,
                attributes: Default::default(),
                devices: vec![],
            },
            environment,
            candidates,
            KernelTuningBudget::new(2, 1, 2, Duration::from_secs(10)).unwrap(),
        )
    }

    /// Counts actual completion observation separately from submission.
    struct Completion(Arc<AtomicUsize>);

    impl ReferenceCompletionBackend for Completion {
        fn r#await(&self) -> Result<(), Arc<str>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn is_ready(&self) -> Result<bool, Arc<str>> {
            Ok(false)
        }
    }

    /// Executes completed CPU work through the canonical completion wrapper without supplying fabricated timings.
    #[derive(Default)]
    struct Runner {
        /// Schedule preparation order.
        schedules: Vec<KernelSchedule>,

        /// Independently observed completions.
        completed: Arc<AtomicUsize>,
    }

    impl KernelTuningRunner for Runner {
        fn prepare(&mut self, schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
            self.schedules.push(schedule.clone());
            Ok(())
        }
        fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
            ReferenceExecution::pending(
                Ok(()),
                ReferenceCompletion::new(Completion(self.completed.clone())),
                |message| KernelTuningError::Invalid { message: message.to_string() },
            )
        }
    }

    #[test]
    fn test_kernel_tuning_budget_new() {
        let budget = KernelTuningBudget::new(2, 0, 3, Duration::from_secs(1)).unwrap();
        assert_eq!((budget.maximum_candidates, budget.warmups, budget.repeats), (2, 0, 3));
        for (candidates, warmups, repeats, time) in
            [(0, 0, 1, 1), (257, 0, 1, 1), (1, 1001, 1, 1), (1, 0, 0, 1), (1, 0, 1001, 1), (1, 0, 1, 0)]
        {
            assert!(matches!(
                KernelTuningBudget::new(candidates, warmups, repeats, Duration::from_secs(time)),
                Err(KernelTuningError::Invalid { message })
                    if message == "invalid candidate, warmup, repeat, or time limit",
            ));
        }
    }

    #[test]
    fn test_kernel_tuning_request_new() {
        let first = request();
        let second = request();
        assert_eq!(first.identity, second.identity);
        assert_eq!(first.candidates, second.candidates);
        assert!(first.identity.len() < MAXIMUM_RECORD_BYTES / 2);
    }

    #[test]
    fn test_kernel_tuning_request_new_identity_and_search_validation() {
        let baseline = request();
        for (options, embedding, platform, environment) in [
            (3, 2, "1", b"empty-workload".as_slice()),
            (1, 3, "1", b"empty-workload".as_slice()),
            (1, 2, "2", b"empty-workload".as_slice()),
            (1, 2, "1", b"different-workload".as_slice()),
        ] {
            let changed = request_with(options, embedding, platform, environment, baseline.candidates.clone()).unwrap();
            assert_ne!(baseline.identity, changed.identity);
        }
        assert!(matches!(
            request_with(1, 2, "1", b"workload", vec![]),
            Err(KernelTuningError::Invalid { message })
                if message == "empty search, excessive candidates, or empty environment",
        ));
        assert!(matches!(
            request_with(1, 2, "1", b"workload", vec![KernelSchedule::default(); 2]),
            Err(KernelTuningError::Invalid { message })
                if message == "duplicate candidate schedule",
        ));
    }

    #[test]
    fn test_kernel_tuning_request_candidates() {
        assert_eq!(
            request().candidates(),
            &[KernelSchedule::default(), KernelSchedule::default().with_maximum_scratch_bytes(0)]
        );
    }

    #[test]
    fn test_kernel_tuning_runner_prepare() {
        let mut runner = Runner::default();
        runner.prepare(&KernelSchedule::default()).unwrap();
        assert_eq!(runner.schedules, vec![KernelSchedule::default()]);
        assert_eq!(runner.completed.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_kernel_tuning_runner_execute() {
        let mut runner = Runner::default();
        let execution = runner.execute();
        assert_eq!(runner.completed.load(Ordering::SeqCst), 0);
        execution.r#await().unwrap();
        assert_eq!(runner.completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_kernel_tuning_result_samples() {
        let request = request();
        let result = KernelTuner::new(None).run(&request, &mut Runner::default(), &AtomicBool::new(false)).unwrap();
        assert_eq!(result.samples().iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
    }

    #[test]
    fn test_kernel_tuning_result_best_candidate() {
        let request = request();
        // Hand-authored persisted data tests ranking only; these values are never presented as measured execution.
        let result = KernelTuningResult {
            record: MeasurementRecord {
                version: 1,
                identity: request.identity.clone(),
                samples: vec![
                    vec![Duration::from_nanos(1), Duration::from_nanos(3)],
                    vec![Duration::from_nanos(2), Duration::from_nanos(2)],
                ],
                best_candidate: 1,
            },
        };
        result.validate(&request).unwrap();
        assert_eq!(result.best_candidate(), 1);
        assert_eq!(KernelTuningResult::winner(&[vec![Duration::ZERO], vec![Duration::ZERO]]), 0);
    }

    #[test]
    fn test_kernel_tuner_new() {
        let tuner = KernelTuner::new(None);
        assert!(tuner.cache.is_none());
        assert!(tuner.gate.try_lock().is_ok());
    }

    #[test]
    fn test_kernel_tuner_load() {
        let directory = tempfile::tempdir().unwrap();
        let cache = Arc::new(DiskCache::open(directory.path()).unwrap());
        let tuner = KernelTuner::new(Some(cache.clone()));
        let request = request();
        assert_eq!(tuner.load(&request).unwrap(), None);
        let result = tuner.run(&request, &mut Runner::default(), &AtomicBool::new(false)).unwrap();
        assert_eq!(tuner.load(&request).unwrap(), Some(result.clone()));
        let mut stale = result;
        stale.record.version = 2;
        cache
            .put_auxiliary("kernel-tuning-v1", &request.identity, &serde_json::to_vec(&stale.record).unwrap())
            .unwrap();
        assert!(matches!(
            tuner.load(&request),
            Err(KernelTuningError::Invalid { message })
                if message == "stale or inconsistent measurement record",
        ));
        stale.record.version = 1;
        stale.record.samples =
            vec![vec![request.budget.maximum_time; request.budget.repeats]; request.candidates.len()];
        stale.record.best_candidate = 0;
        cache
            .put_auxiliary("kernel-tuning-v1", &request.identity, &serde_json::to_vec(&stale.record).unwrap())
            .unwrap();
        assert!(matches!(
            tuner.load(&request),
            Err(KernelTuningError::Invalid { message })
                if message == "stale or inconsistent measurement record",
        ));
        cache.put_auxiliary("kernel-tuning-v1", &request.identity, b"corrupt").unwrap();
        assert!(matches!(tuner.load(&request), Err(KernelTuningError::Payload(_))));
    }

    #[test]
    fn test_kernel_tuner_run() {
        let request = request();
        let mut runner = Runner::default();
        let result = KernelTuner::new(None).run(&request, &mut runner, &AtomicBool::new(false)).unwrap();
        assert_eq!(runner.schedules, request.candidates);
        assert_eq!(runner.completed.load(Ordering::SeqCst), 6);
        result.validate(&request).unwrap();
    }

    #[test]
    fn test_kernel_tuner_run_cancelled_timeout_and_busy() {
        let tuner = KernelTuner::new(None);
        let mut request = request();
        let mut runner = Runner::default();
        assert!(matches!(tuner.run(&request, &mut runner, &AtomicBool::new(true)), Err(KernelTuningError::Cancelled)));
        let guard = tuner.gate.lock().unwrap();
        assert!(matches!(tuner.run(&request, &mut runner, &AtomicBool::new(false)), Err(KernelTuningError::Busy)));
        drop(guard);
        request.budget.maximum_time = Duration::ZERO;
        assert!(matches!(tuner.run(&request, &mut runner, &AtomicBool::new(false)), Err(KernelTuningError::Timeout)));
        assert_eq!(runner.completed.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_kernel_tuner_run_preparation_deadline() {
        /// Preparation exceeds its explicit deadline without submitting work.
        struct SlowPreparation;
        impl KernelTuningRunner for SlowPreparation {
            fn prepare(&mut self, _schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
                std::thread::sleep(Duration::from_millis(20));
                Ok(())
            }
            fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
                panic!("expired preparation must not submit work")
            }
        }
        let mut request = request();
        request.budget = KernelTuningBudget::new(2, 1, 2, Duration::from_millis(10)).unwrap();
        assert!(matches!(
            KernelTuner::new(None).run(&request, &mut SlowPreparation, &AtomicBool::new(false)),
            Err(KernelTuningError::Timeout)
        ));
    }

    #[test]
    fn test_kernel_tuner_run_failure_preserves_completed_record() {
        /// Fails during completion, after submission has succeeded.
        struct FailedRunner;
        impl KernelTuningRunner for FailedRunner {
            fn prepare(&mut self, _schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
                Ok(())
            }
            fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
                ReferenceExecution::pending(
                    Ok(()),
                    ReferenceCompletion::ready(Err("execution failed".into())),
                    |message| KernelTuningError::Invalid { message: message.to_string() },
                )
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let tuner = KernelTuner::new(Some(Arc::new(DiskCache::open(directory.path()).unwrap())));
        let request = request();
        let previous = tuner.run(&request, &mut Runner::default(), &AtomicBool::new(false)).unwrap();
        assert!(matches!(
            tuner.run(&request, &mut FailedRunner, &AtomicBool::new(false)),
            Err(KernelTuningError::Invalid { message })
                if message == "execution failed",
        ));
        assert_eq!(tuner.load(&request).unwrap(), Some(previous));
    }
    #[test]
    fn test_kernel_tuner_run_cancellation_awaits_submitted_work() {
        /// Cancels during submission while returning a completion that still must be observed.
        struct CancellingRunner<'a> {
            /// Shared cooperative cancellation signal.
            cancelled: &'a AtomicBool,

            /// Completion counter independent of cancellation.
            completed: Arc<AtomicUsize>,
        }
        impl KernelTuningRunner for CancellingRunner<'_> {
            fn prepare(&mut self, _schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
                Ok(())
            }
            fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
                self.cancelled.store(true, Ordering::Release);
                ReferenceExecution::pending(
                    Ok(()),
                    ReferenceCompletion::new(Completion(self.completed.clone())),
                    |message| KernelTuningError::Invalid { message: message.to_string() },
                )
            }
        }
        let cancelled = AtomicBool::new(false);
        let mut runner = CancellingRunner { cancelled: &cancelled, completed: Arc::new(AtomicUsize::new(0)) };
        let tuner = KernelTuner::new(None);
        assert!(matches!(tuner.run(&request(), &mut runner, &cancelled), Err(KernelTuningError::Cancelled)));
        assert_eq!(runner.completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_kernel_tuner_run_concurrent_measurements() {
        /// Holds preparation until the competing caller has checked the device gate.
        struct BlockingRunner {
            /// Announces that preparation owns the gate.
            entered: std::sync::mpsc::Sender<()>,

            /// Explicit release, avoiding timing-dependent sleeps.
            release: std::sync::mpsc::Receiver<()>,
        }
        impl KernelTuningRunner for BlockingRunner {
            fn prepare(&mut self, _schedule: &KernelSchedule) -> Result<(), KernelTuningError> {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
                Ok(())
            }
            fn execute(&mut self) -> ReferenceExecution<(), KernelTuningError> {
                ReferenceExecution::ready(Ok(()))
            }
        }
        let tuner = KernelTuner::new(None);
        let request = request_with(1, 2, "1", b"workload", vec![KernelSchedule::default()]).unwrap();
        let (entered_sender, entered_receiver) = std::sync::mpsc::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let tuner = &tuner;
            let request = &request;
            let worker = scope.spawn(move || {
                tuner.run(
                    &request,
                    &mut BlockingRunner { entered: entered_sender, release: release_receiver },
                    &AtomicBool::new(false),
                )
            });
            entered_receiver.recv().unwrap();
            assert!(matches!(
                tuner.run(&request, &mut Runner::default(), &AtomicBool::new(false)),
                Err(KernelTuningError::Busy)
            ));
            release_sender.send(()).unwrap();
            worker.join().unwrap().unwrap();
        });
    }
}
