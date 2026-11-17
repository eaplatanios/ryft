//! Adaptive XLA profile-guided recompilation.
//!
//! [`AdaptiveProfileGuidedFunction`] mirrors JAX's automatic Profile-Guided Latency Estimator flow while keeping
//! the policy and lifecycle explicit. One caller at a time profiles the baseline executable. Other callers continue
//! dispatching without waiting, and compilation occurs outside all adaptive state and dispatch locks. A compatible
//! optimized executable is published only after compilation succeeds, and every failure path is bounded and observable.

// TODO(eaplatanios): Review this module.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ryft_core::{ArrayIrType, ArrayType, CompilationCacheDomain, Parameterized, ParameterizedFamily};
use ryft_pjrt::extensions::profiler::FeedbackDirectedProfile as PjrtFeedbackDirectedProfile;
use ryft_pjrt::protos::{ProfileDeviceType, ProfileOptions};

use crate::compilation::CompilationError;
use crate::compilation::functions::ExecutableFunction;
use crate::compilation::options::{FeedbackDirectedProfile, Options};
use crate::compilation::programs::LoweredProgram;
use crate::{Array, Constant, Error, Session, Value};

/// Namespace of the auxiliary disk cache entries that persist aggregated adaptive profiles.
const ADAPTIVE_PROFILE_CACHE_NAMESPACE: &str = "xla-adaptive-profile-v1";

/// Policy for one adaptive profile-guided recompilation lifecycle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AdaptiveProfileGuidedOptions {
    /// Number of successful profiles to collect before recompiling.
    sample_count: usize,

    /// Maximum number of executions selected for profiling, including failed attempts.
    maximum_profile_attempts: usize,

    /// Percentile in `0..=100` at which OpenXLA aggregates the instruction costs of the collected profiles.
    aggregation_percentile: u8,

    /// Profile version forwarded in the PJRT compilation options of the recompilation.
    profile_version: i64,
}

impl AdaptiveProfileGuidedOptions {
    /// Creates a policy that collects `sample_count` successful profiles in at most `maximum_profile_attempts`
    /// selected executions, with an aggregation percentile of 90 and a profile version of 0.
    ///
    /// # Errors
    ///
    /// Returns an error when `sample_count` is zero or when `maximum_profile_attempts` is smaller than `sample_count`.
    pub fn new(sample_count: usize, maximum_profile_attempts: usize) -> Result<Self, Error> {
        if sample_count == 0 {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "adaptive profile `sample_count` must be greater than zero".to_string(),
            }));
        }
        if maximum_profile_attempts < sample_count {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "adaptive `maximum_profile_attempts` must be at least `sample_count`".to_string(),
            }));
        }
        Ok(Self { sample_count, maximum_profile_attempts, aggregation_percentile: 90, profile_version: 0 })
    }

    /// Sets the percentile at which OpenXLA aggregates instruction costs.
    ///
    /// # Errors
    ///
    /// Returns an error when `percentile` is greater than 100.
    pub fn with_aggregation_percentile(mut self, percentile: u8) -> Result<Self, Error> {
        if percentile > 100 {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "adaptive aggregation percentile must be in `0..=100`".to_string(),
            }));
        }
        self.aggregation_percentile = percentile;
        Ok(self)
    }

    /// Sets the profile version forwarded in PJRT compilation options.
    #[inline]
    pub fn with_profile_version(mut self, version: i64) -> Self {
        self.profile_version = version;
        self
    }

    /// Returns the required number of successful profiles.
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the maximum number of selected profiling executions.
    #[inline]
    pub fn maximum_profile_attempts(&self) -> usize {
        self.maximum_profile_attempts
    }

    /// Returns the OpenXLA aggregation percentile.
    #[inline]
    pub fn aggregation_percentile(&self) -> u8 {
        self.aggregation_percentile
    }

    /// Returns the XLA profile version forwarded at recompilation.
    #[inline]
    pub fn profile_version(&self) -> i64 {
        self.profile_version
    }
}

impl Default for AdaptiveProfileGuidedOptions {
    fn default() -> Self {
        Self::new(3, 5).unwrap()
    }
}

/// Observable phase of an adaptive profile-guided executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdaptiveProfileGuidedState {
    /// The baseline remains active while successful profiles are collected.
    Profiling {
        /// Number of successful profiles collected so far.
        profiles_collected: usize,

        /// Number of executions selected for profiling so far, including failed attempts.
        profile_attempts: usize,

        /// Whether a selected execution is currently being profiled.
        profile_in_flight: bool,

        /// Reason of the most recent failed profiling attempt or persistent profile restoration, if any.
        last_failure: Option<String>,
    },

    /// The required samples were collected and one caller is compiling an optimized executable.
    Compiling {
        /// Number of successful profiles collected.
        profiles_collected: usize,

        /// Number of executions selected for profiling.
        profile_attempts: usize,
    },

    /// The compatible optimized executable is active.
    Optimized {
        /// Number of successful profiles collected, which is zero when a persisted profile was restored.
        profiles_collected: usize,

        /// Number of executions selected for profiling, which is zero when a persisted profile was restored.
        profile_attempts: usize,
    },

    /// Adaptation stopped permanently, and the baseline executable remains active.
    Failed {
        /// Number of successful profiles collected.
        profiles_collected: usize,

        /// Number of executions selected for profiling.
        profile_attempts: usize,

        /// Reason why adaptation stopped.
        reason: String,
    },
}

/// Lock-free activity snapshot for an [`AdaptiveProfileGuidedFunction`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveProfileGuidedStatistics {
    /// Total calls dispatched through this wrapper.
    executions: usize,

    /// Calls dispatched to the baseline, including selected profiling calls.
    baseline_executions: usize,

    /// Baseline calls selected to own a profiling attempt.
    profiled_executions: usize,

    /// Calls dispatched after an optimized executable was installed.
    optimized_executions: usize,

    /// Profiling attempts that failed before yielding an OpenXLA profile.
    profile_failures: usize,

    /// Profile aggregation and recompilation attempts. This is at most one.
    recompilations: usize,

    /// Terminal aggregation or recompilation failures.
    recompilation_failures: usize,

    /// Compatible optimized executable installations. This is at most one.
    executable_replacements: usize,

    /// Persisted aggregated profiles restored and installed without new sampling.
    profile_restore_hits: usize,

    /// Configured persistent profile lookups that found no sidecar.
    profile_restore_misses: usize,

    /// Persistent profile read, validation, or recompilation failures that fell back to bounded sampling.
    profile_restore_failures: usize,

    /// Aggregated profiles persisted after a successful replacement.
    profile_persistences: usize,

    /// Writes of profile sidecars that failed after a successful replacement.
    profile_persistence_failures: usize,
}

impl AdaptiveProfileGuidedStatistics {
    /// Returns the total calls dispatched through this wrapper.
    #[inline]
    pub fn executions(&self) -> usize {
        self.executions
    }

    /// Returns the calls dispatched to the baseline, including selected profiling calls.
    #[inline]
    pub fn baseline_executions(&self) -> usize {
        self.baseline_executions
    }

    /// Returns the baseline calls selected to own a profiling attempt.
    #[inline]
    pub fn profiled_executions(&self) -> usize {
        self.profiled_executions
    }

    /// Returns the calls dispatched after an optimized executable was installed.
    #[inline]
    pub fn optimized_executions(&self) -> usize {
        self.optimized_executions
    }

    /// Returns the profiling attempts that failed before yielding an OpenXLA profile.
    #[inline]
    pub fn profile_failures(&self) -> usize {
        self.profile_failures
    }

    /// Returns the profile aggregation and recompilation attempts. This is at most one.
    #[inline]
    pub fn recompilations(&self) -> usize {
        self.recompilations
    }

    /// Returns the terminal aggregation or recompilation failures.
    #[inline]
    pub fn recompilation_failures(&self) -> usize {
        self.recompilation_failures
    }

    /// Returns the compatible optimized executable installations. This is at most one.
    #[inline]
    pub fn executable_replacements(&self) -> usize {
        self.executable_replacements
    }

    /// Returns the persisted aggregated profiles restored and installed without new sampling.
    #[inline]
    pub fn profile_restore_hits(&self) -> usize {
        self.profile_restore_hits
    }

    /// Returns the configured persistent profile lookups that found no sidecar.
    #[inline]
    pub fn profile_restore_misses(&self) -> usize {
        self.profile_restore_misses
    }

    /// Returns the persistent profile read, validation, or recompilation failures that fell back to bounded sampling.
    #[inline]
    pub fn profile_restore_failures(&self) -> usize {
        self.profile_restore_failures
    }

    /// Returns the aggregated profiles persisted after a successful replacement.
    #[inline]
    pub fn profile_persistences(&self) -> usize {
        self.profile_persistences
    }

    /// Returns the writes of profile sidecars that failed after a successful replacement.
    #[inline]
    pub fn profile_persistence_failures(&self) -> usize {
        self.profile_persistence_failures
    }
}

/// Executable that performs one bounded adaptive profile-guided recompilation lifecycle at runtime.
///
/// Clones share dispatch state. Only a selected profiling call synchronizes its own baseline execution so the XProf
/// trace is complete. Concurrent calls copy the currently active executable handle and execute without holding the
/// state or dispatch locks. Recompilation happens at most once and is terminal. It either installs one compatible
/// executable or records a permanent failure while continuing to serve the baseline.
pub struct AdaptiveProfileGuidedFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// State shared by all clones of this function.
    inner: Arc<AdaptiveProfileGuidedFunctionInner<'c, In, Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone
    for AdaptiveProfileGuidedFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> AdaptiveProfileGuidedFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Creates an adaptive function over `baseline`, the executable compiled from `lowered_program`, and installs an
    /// executable compiled with a persisted profile right away when the disk cache holds one.
    ///
    /// # Errors
    ///
    /// Returns an error when `compilation_options` already carry a feedback-directed profile, when the platform of
    /// `session` is neither CUDA nor ROCm, when the client has no profiler extension, or when the persistent key of the
    /// profile cannot be computed.
    pub(crate) fn new(
        session: Session<'c>,
        baseline: ExecutableFunction<'c, In, Out>,
        lowered_program: LoweredProgram,
        compilation_options: Options,
        options: AdaptiveProfileGuidedOptions,
    ) -> Result<Self, Error> {
        if compilation_options.feedback_directed_profile().is_some() {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: "adaptive profile-guided recompilation requires an unprofiled baseline".to_string(),
            }));
        }
        let client = session.client();
        validate_adaptive_profile_guided_platform(client.platform_name()?.as_ref())?;
        client.profiler_extension()?;
        let profile_sidecar_key = adaptive_profile_sidecar_key(&session, &lowered_program, options)?;
        let function = Self {
            inner: Arc::new(AdaptiveProfileGuidedFunctionInner {
                session,
                active: RwLock::new(baseline.clone()),
                baseline,
                lowered_program,
                options,
                profile_sidecar_key,
                state: Mutex::new(AdaptiveProfileGuidedStateMachine::new()),
                statistics: AtomicAdaptiveProfileGuidedStatistics::default(),
            }),
        };
        function.restore_persisted_profile();
        Ok(function)
    }

    /// Returns this dispatcher's immutable sampling and aggregation policy.
    #[inline]
    pub fn options(&self) -> AdaptiveProfileGuidedOptions {
        self.inner.options
    }

    /// Returns a point-in-time lifecycle state snapshot.
    pub fn state(&self) -> AdaptiveProfileGuidedState {
        self.inner
            .state
            .lock()
            .expect("adaptive profile-guided state mutex should not be poisoned")
            .state
            .clone()
    }

    /// Returns a lock-free activity snapshot.
    #[inline]
    pub fn statistics(&self) -> AdaptiveProfileGuidedStatistics {
        self.inner.statistics.snapshot()
    }

    /// Returns the flat output types of the currently active compatible executable.
    pub fn output_types(&self) -> Vec<ArrayType> {
        self.inner
            .active
            .read()
            .expect("adaptive executable lock should not be poisoned")
            .output_types()
            .to_vec()
    }

    /// Executes with the active executable and advances adaptive profiling when this call wins ownership.
    pub fn interpret(&self, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, Error>
    where
        In::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<Value<'c>>,
        Out::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<Value<'c>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        self.inner.statistics.executions.fetch_add(1, Ordering::Relaxed);
        let (dispatch, executable) = {
            let mut state =
                self.inner.state.lock().expect("adaptive profile-guided state mutex should not be poisoned");
            let dispatch = state.claim_dispatch(self.inner.options);
            let executable = self.inner.active.read().expect("adaptive executable lock should not be poisoned").clone();
            (dispatch, executable)
        };
        match dispatch {
            AdaptiveDispatch::Profile => {
                self.inner.statistics.baseline_executions.fetch_add(1, Ordering::Relaxed);
                self.inner.statistics.profiled_executions.fetch_add(1, Ordering::Relaxed);
                self.profile_baseline(executable, inputs)
            }
            AdaptiveDispatch::Baseline => {
                self.inner.statistics.baseline_executions.fetch_add(1, Ordering::Relaxed);
                executable.execute(&self.inner.session, inputs)
            }
            AdaptiveDispatch::Optimized => {
                self.inner.statistics.optimized_executions.fetch_add(1, Ordering::Relaxed);
                executable.execute(&self.inner.session, inputs)
            }
        }
    }

    /// Executes `executable`, which is the baseline, under the profiler and records the resulting profile, recompiling
    /// once enough profiles were collected. Profiler failures are recorded as failed attempts while the outputs of the
    /// execution are still returned.
    fn profile_baseline(
        &self,
        executable: ExecutableFunction<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Out::To<Array<'c>>, Error>
    where
        In::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<Value<'c>>,
        Out::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<Value<'c>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let profiler_options = ProfileOptions {
            version: 1,
            device_type: ProfileDeviceType::Gpu as i32,
            host_tracing_level: 0,
            device_tracing_level: 1,
            python_tracing_level: 0,
            enable_hlo_proto: true,
            raise_error_on_start_failure: true,
            ..ProfileOptions::default()
        };
        let profiler = match self.inner.session.client().profiler(&profiler_options) {
            Ok(profiler) => profiler,
            Err(error) => {
                let outputs = executable.execute(&self.inner.session, inputs)?;
                self.record_profile_failure(error.to_string());
                return Ok(outputs);
            }
        };
        if let Err(error) = profiler.start() {
            let outputs = executable.execute(&self.inner.session, inputs)?;
            self.record_profile_failure(error.to_string());
            return Ok(outputs);
        }
        let execution = match executable.execute_async(&self.inner.session, inputs) {
            Ok(execution) => execution,
            Err(error) => {
                let reason = match profiler.stop() {
                    Ok(()) => error.to_string(),
                    Err(stop_error) => format!("{error}; profiler stop also failed: {stop_error}"),
                };
                self.record_profile_failure(reason);
                return Err(error);
            }
        };
        let outputs = match execution.block_until_ready() {
            Ok(outputs) => outputs,
            Err(error) => {
                let reason = match profiler.stop() {
                    Ok(()) => error.to_string(),
                    Err(stop_error) => format!("{error}; profiler stop also failed: {stop_error}"),
                };
                self.record_profile_failure(reason);
                return Err(error.into());
            }
        };
        let profile = profiler
            .stop()
            .and_then(|()| profiler.results())
            .and_then(|results| PjrtFeedbackDirectedProfile::from_x_space(&results));
        match profile {
            Ok(profile) => {
                let profiles = self
                    .inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .profile_succeeded(profile, self.inner.options);
                if let Some(profiles) = profiles {
                    self.recompile(profiles);
                }
            }
            Err(error) => self.record_profile_failure(error.to_string()),
        }
        Ok(outputs)
    }

    /// Records one failed profiling attempt.
    fn record_profile_failure(&self, reason: String) {
        self.inner.statistics.profile_failures.fetch_add(1, Ordering::Relaxed);
        self.inner
            .state
            .lock()
            .expect("adaptive profile-guided state mutex should not be poisoned")
            .profile_failed(reason, self.inner.options);
    }

    /// Compiles the baseline lowering with the aggregated `profile` attached and validates the result as a replacement
    /// of the baseline executable.
    fn compile_profile(&self, profile: Vec<u8>) -> Result<ExecutableFunction<'c, In, Out>, Error> {
        let profile = FeedbackDirectedProfile::new(profile).with_version(self.inner.options.profile_version);
        let lowered_program = self.inner.lowered_program.clone().with_feedback_directed_profile(profile);
        let session = &self.inner.session;
        let program = session.compilation_cache().compile(session, &lowered_program)?;
        session.replace_executable_program(&self.inner.baseline, program)
    }

    /// Installs an executable compiled with the persisted aggregated profile, if the disk cache holds one. Failures
    /// are recorded and leave the function profiling.
    fn restore_persisted_profile(&self) {
        let Some(key) = self.inner.profile_sidecar_key.as_deref() else {
            return;
        };
        let Some(disk_cache) = self.inner.session.compilation_cache().disk_cache() else {
            return;
        };
        let profile = match disk_cache.get_auxiliary(ADAPTIVE_PROFILE_CACHE_NAMESPACE, key) {
            Ok(Some(profile)) => profile,
            Ok(None) => {
                self.inner.statistics.profile_restore_misses.fetch_add(1, Ordering::Relaxed);
                return;
            }
            Err(error) => {
                self.inner.statistics.profile_restore_failures.fetch_add(1, Ordering::Relaxed);
                self.inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .profile_restore_failed(error.to_string());
                return;
            }
        };
        match self.compile_profile(profile) {
            Ok(candidate) => {
                let mut state =
                    self.inner.state.lock().expect("adaptive profile-guided state mutex should not be poisoned");
                *self.inner.active.write().expect("adaptive executable lock should not be poisoned") = candidate;
                state.profile_restored();
                drop(state);
                self.inner.statistics.profile_restore_hits.fetch_add(1, Ordering::Relaxed);
                self.inner.statistics.executable_replacements.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                self.inner.statistics.profile_restore_failures.fetch_add(1, Ordering::Relaxed);
                self.inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .profile_restore_failed(error.to_string());
            }
        }
    }

    /// Persists the aggregated `profile` in the disk cache, counting rather than propagating failures.
    fn persist_profile(&self, profile: &[u8]) {
        let Some(key) = self.inner.profile_sidecar_key.as_deref() else {
            return;
        };
        let Some(disk_cache) = self.inner.session.compilation_cache().disk_cache() else {
            return;
        };
        match disk_cache.put_auxiliary(ADAPTIVE_PROFILE_CACHE_NAMESPACE, key, profile) {
            Ok(()) => {
                self.inner.statistics.profile_persistences.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.inner.statistics.profile_persistence_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Aggregates `profiles`, compiles the optimized executable, and installs it, or stops adaptation permanently when
    /// aggregation or compilation fails.
    fn recompile(&self, profiles: Vec<PjrtFeedbackDirectedProfile>) {
        self.inner.statistics.recompilations.fetch_add(1, Ordering::Relaxed);
        let (profile, candidate) = match PjrtFeedbackDirectedProfile::aggregated(
            profiles.as_slice(),
            self.inner.options.aggregation_percentile,
        ) {
            Ok(profile) => {
                let candidate = self.compile_profile(profile.bytes().to_vec());
                (Some(profile), candidate)
            }
            Err(error) => (None, Err(Error::from(error))),
        };
        match candidate {
            Ok(candidate) => {
                let mut state =
                    self.inner.state.lock().expect("adaptive profile-guided state mutex should not be poisoned");
                *self.inner.active.write().expect("adaptive executable lock should not be poisoned") = candidate;
                state.compilation_succeeded();
                drop(state);
                self.inner.statistics.executable_replacements.fetch_add(1, Ordering::Relaxed);
                self.persist_profile(profile.unwrap().bytes());
            }
            Err(error) => {
                self.inner.statistics.recompilation_failures.fetch_add(1, Ordering::Relaxed);
                self.inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .compilation_failed(error.to_string());
            }
        }
    }
}

/// Counters behind [`AdaptiveProfileGuidedStatistics`], updated with relaxed atomic operations so that dispatch never
/// takes a lock to record activity.
#[derive(Default)]
struct AtomicAdaptiveProfileGuidedStatistics {
    /// Counter of [`AdaptiveProfileGuidedStatistics::executions`].
    executions: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::baseline_executions`].
    baseline_executions: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profiled_executions`].
    profiled_executions: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::optimized_executions`].
    optimized_executions: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_failures`].
    profile_failures: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::recompilations`].
    recompilations: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::recompilation_failures`].
    recompilation_failures: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::executable_replacements`].
    executable_replacements: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_restore_hits`].
    profile_restore_hits: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_restore_misses`].
    profile_restore_misses: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_restore_failures`].
    profile_restore_failures: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_persistences`].
    profile_persistences: AtomicUsize,

    /// Counter of [`AdaptiveProfileGuidedStatistics::profile_persistence_failures`].
    profile_persistence_failures: AtomicUsize,
}

impl AtomicAdaptiveProfileGuidedStatistics {
    /// Returns a snapshot of all counters. Counters are read independently, so the snapshot need not be consistent
    /// across counters that concurrent calls update.
    fn snapshot(&self) -> AdaptiveProfileGuidedStatistics {
        AdaptiveProfileGuidedStatistics {
            executions: self.executions.load(Ordering::Relaxed),
            baseline_executions: self.baseline_executions.load(Ordering::Relaxed),
            profiled_executions: self.profiled_executions.load(Ordering::Relaxed),
            optimized_executions: self.optimized_executions.load(Ordering::Relaxed),
            profile_failures: self.profile_failures.load(Ordering::Relaxed),
            recompilations: self.recompilations.load(Ordering::Relaxed),
            recompilation_failures: self.recompilation_failures.load(Ordering::Relaxed),
            executable_replacements: self.executable_replacements.load(Ordering::Relaxed),
            profile_restore_hits: self.profile_restore_hits.load(Ordering::Relaxed),
            profile_restore_misses: self.profile_restore_misses.load(Ordering::Relaxed),
            profile_restore_failures: self.profile_restore_failures.load(Ordering::Relaxed),
            profile_persistences: self.profile_persistences.load(Ordering::Relaxed),
            profile_persistence_failures: self.profile_persistence_failures.load(Ordering::Relaxed),
        }
    }
}

/// Executable that one call of an [`AdaptiveProfileGuidedFunction`] is dispatched to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AdaptiveDispatch {
    /// The baseline executable, executed under the profiler because this call owns a profiling attempt.
    Profile,

    /// The baseline executable, executed without profiling.
    Baseline,

    /// The optimized executable.
    Optimized,
}

/// Lifecycle state of an [`AdaptiveProfileGuidedFunction`] together with the profiles collected so far.
struct AdaptiveProfileGuidedStateMachine {
    /// Observable lifecycle phase.
    state: AdaptiveProfileGuidedState,

    /// Successful profiles collected while profiling, cleared once compilation succeeds or fails.
    profiles: Vec<PjrtFeedbackDirectedProfile>,
}

impl AdaptiveProfileGuidedStateMachine {
    /// Creates a state machine that is profiling and has collected no profiles.
    fn new() -> Self {
        Self {
            state: AdaptiveProfileGuidedState::Profiling {
                profiles_collected: 0,
                profile_attempts: 0,
                profile_in_flight: false,
                last_failure: None,
            },
            profiles: Vec::new(),
        }
    }

    /// Selects the executable for one call. A call claims a profiling attempt, which it then owns, only while no other
    /// attempt is in flight and the attempt budget of `options` is not exhausted.
    fn claim_dispatch(&mut self, options: AdaptiveProfileGuidedOptions) -> AdaptiveDispatch {
        match &mut self.state {
            AdaptiveProfileGuidedState::Profiling { profile_attempts, profile_in_flight, .. }
                if !*profile_in_flight && *profile_attempts < options.maximum_profile_attempts =>
            {
                *profile_attempts += 1;
                *profile_in_flight = true;
                AdaptiveDispatch::Profile
            }
            AdaptiveProfileGuidedState::Optimized { .. } => AdaptiveDispatch::Optimized,
            AdaptiveProfileGuidedState::Profiling { .. }
            | AdaptiveProfileGuidedState::Compiling { .. }
            | AdaptiveProfileGuidedState::Failed { .. } => AdaptiveDispatch::Baseline,
        }
    }

    /// Records a successful profile of the in-flight attempt. Returns all collected profiles, and enters the compiling
    /// phase, once `options` require no more samples. Fails permanently when the attempt budget is exhausted before
    /// enough profiles were collected.
    fn profile_succeeded(
        &mut self,
        profile: PjrtFeedbackDirectedProfile,
        options: AdaptiveProfileGuidedOptions,
    ) -> Option<Vec<PjrtFeedbackDirectedProfile>> {
        let AdaptiveProfileGuidedState::Profiling { profile_attempts, profile_in_flight, .. } = &mut self.state else {
            return None;
        };
        *profile_in_flight = false;
        self.profiles.push(profile);
        let profiles_collected = self.profiles.len();
        if profiles_collected < options.sample_count {
            let profile_attempts = *profile_attempts;
            if profile_attempts >= options.maximum_profile_attempts {
                self.state = AdaptiveProfileGuidedState::Failed {
                    profiles_collected,
                    profile_attempts,
                    reason: format!(
                        "collected {profiles_collected} successful profile(s), fewer than the required {} after \
                         exhausting {profile_attempts} attempt(s)",
                        options.sample_count,
                    ),
                };
                return None;
            }
            self.state = AdaptiveProfileGuidedState::Profiling {
                profiles_collected,
                profile_attempts,
                profile_in_flight: false,
                last_failure: None,
            };
            return None;
        }
        let profile_attempts = *profile_attempts;
        self.state = AdaptiveProfileGuidedState::Compiling { profiles_collected, profile_attempts };
        Some(self.profiles.clone())
    }

    /// Records a failed profiling attempt, which fails adaptation permanently once the attempt budget of `options` is
    /// exhausted.
    fn profile_failed(&mut self, reason: String, options: AdaptiveProfileGuidedOptions) {
        let AdaptiveProfileGuidedState::Profiling { profile_attempts, profile_in_flight, .. } = &mut self.state else {
            return;
        };
        *profile_in_flight = false;
        let profiles_collected = self.profiles.len();
        let profile_attempts = *profile_attempts;
        if profile_attempts >= options.maximum_profile_attempts {
            self.state = AdaptiveProfileGuidedState::Failed { profiles_collected, profile_attempts, reason };
        } else {
            self.state = AdaptiveProfileGuidedState::Profiling {
                profiles_collected,
                profile_attempts,
                profile_in_flight: false,
                last_failure: Some(reason),
            };
        }
    }

    /// Records that the optimized executable was installed.
    fn compilation_succeeded(&mut self) {
        let AdaptiveProfileGuidedState::Compiling { profiles_collected, profile_attempts } = self.state.clone() else {
            return;
        };
        self.state = AdaptiveProfileGuidedState::Optimized { profiles_collected, profile_attempts };
        self.profiles.clear();
    }

    /// Records that aggregation or recompilation failed, which stops adaptation permanently.
    fn compilation_failed(&mut self, reason: String) {
        let AdaptiveProfileGuidedState::Compiling { profiles_collected, profile_attempts } = self.state.clone() else {
            return;
        };
        self.state = AdaptiveProfileGuidedState::Failed { profiles_collected, profile_attempts, reason };
        self.profiles.clear();
    }

    /// Records that an executable compiled with a persisted profile was installed without sampling.
    fn profile_restored(&mut self) {
        self.state = AdaptiveProfileGuidedState::Optimized { profiles_collected: 0, profile_attempts: 0 };
        self.profiles.clear();
    }

    /// Records that restoring a persisted profile failed while profiling continues as a fallback.
    fn profile_restore_failed(&mut self, reason: String) {
        if let AdaptiveProfileGuidedState::Profiling { last_failure, .. } = &mut self.state {
            *last_failure = Some(format!("persistent adaptive profile restore failed: {reason}"));
        }
    }
}

/// State shared by the clones of an [`AdaptiveProfileGuidedFunction`].
///
/// The `active` lock is only ever taken while the `state` mutex is held, and neither is held while an executable
/// executes or compiles. A call claims its dispatch and clones the active executable handle under one hold of the
/// `state` mutex, and the single caller that owns recompilation or restoration replaces the executable and advances
/// the state under one hold as well. A call therefore always executes the executable that its dispatch names.
struct AdaptiveProfileGuidedFunctionInner<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Domain that executes and recompiles the function.
    session: Session<'c>,

    /// Unprofiled baseline executable, which also defines the invocation contract of every replacement.
    baseline: ExecutableFunction<'c, In, Out>,

    /// Executable that calls currently dispatch to, which is the baseline until an optimized executable is installed.
    active: RwLock<ExecutableFunction<'c, In, Out>>,

    /// Lowering of the baseline, which recompilation reuses with a feedback-directed profile attached.
    lowered_program: LoweredProgram,

    /// Sampling and aggregation policy.
    options: AdaptiveProfileGuidedOptions,

    /// Key of the persisted aggregated profile, or `None` when profiles are not persisted.
    profile_sidecar_key: Option<Vec<u8>>,

    /// Lifecycle state and collected profiles.
    state: Mutex<AdaptiveProfileGuidedStateMachine>,

    /// Activity counters.
    statistics: AtomicAdaptiveProfileGuidedStatistics,
}

/// Validates that `platform_name` names a CUDA or ROCm platform, which are the platforms whose profiles OpenXLA can
/// use for profile-guided recompilation.
fn validate_adaptive_profile_guided_platform(platform_name: &str) -> Result<(), Error> {
    if matches!(platform_name.to_ascii_lowercase().as_str(), "cuda" | "rocm") {
        Ok(())
    } else {
        Err(Error::Compilation(CompilationError::InvalidOptions {
            reason: format!(
                "adaptive profile-guided recompilation requires a CUDA or ROCm PJRT platform, but found \
                 `{platform_name}`",
            ),
        }))
    }
}

/// Returns the key under which the aggregated profile of `lowered_program` is persisted, or `None` when the compilation
/// context of `session` has no disk cache or no persistent key for the program. The key combines the persistent key of
/// the baseline program with the sampling and aggregation policy of `options`.
fn adaptive_profile_sidecar_key<'c>(
    session: &Session<'c>,
    lowered_program: &LoweredProgram,
    options: AdaptiveProfileGuidedOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let cache = session.compilation_cache();
    if cache.disk_cache().is_none() {
        return Ok(None);
    }
    let compilation_key = session.compilation_key(lowered_program)?;
    let Some(baseline_key) = session.persistent_cache_key(&compilation_key) else {
        return Ok(None);
    };
    let baseline_key_length = u64::try_from(baseline_key.len()).map_err(|_| Error::SizeLimitExceeded {
        message: "baseline persistent key length does not fit in `u64`".to_string(),
    })?;
    let sample_count = u64::try_from(options.sample_count).map_err(|_| {
        Error::Compilation(CompilationError::InvalidOptions {
            reason: "adaptive sample count does not fit in `u64`".to_string(),
        })
    })?;
    let maximum_profile_attempts = u64::try_from(options.maximum_profile_attempts).map_err(|_| {
        Error::Compilation(CompilationError::InvalidOptions {
            reason: "adaptive maximum profile attempts does not fit in `u64`".to_string(),
        })
    })?;
    let mut key = Vec::with_capacity(baseline_key.len() + 48);
    key.extend_from_slice(b"RYFT-ADAPTIVE-PROFILE\0");
    key.extend_from_slice(&baseline_key_length.to_le_bytes());
    key.extend_from_slice(baseline_key.as_slice());
    key.extend_from_slice(&sample_count.to_le_bytes());
    key.extend_from_slice(&maximum_profile_attempts.to_le_bytes());
    key.push(options.aggregation_percentile);
    key.extend_from_slice(&options.profile_version.to_le_bytes());
    Ok(Some(key))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn options(sample_count: usize, maximum_profile_attempts: usize) -> AdaptiveProfileGuidedOptions {
        AdaptiveProfileGuidedOptions::new(sample_count, maximum_profile_attempts).unwrap()
    }

    #[test]
    fn test_adaptive_profile_guided_options_new() {
        let options = AdaptiveProfileGuidedOptions::new(2, 3).unwrap();
        assert_eq!(options.sample_count(), 2);
        assert_eq!(options.maximum_profile_attempts(), 3);
        assert_eq!(options.aggregation_percentile(), 90);
        assert_eq!(options.profile_version(), 0);
        assert_eq!(AdaptiveProfileGuidedOptions::default(), AdaptiveProfileGuidedOptions::new(3, 5).unwrap());
    }

    #[test]
    fn test_adaptive_profile_guided_options_new_rejects_unreachable_sampling_policies() {
        assert!(matches!(
            AdaptiveProfileGuidedOptions::new(0, 1),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "adaptive profile `sample_count` must be greater than zero",
        ));
        assert!(matches!(
            AdaptiveProfileGuidedOptions::new(2, 1),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "adaptive `maximum_profile_attempts` must be at least `sample_count`",
        ));
    }

    #[test]
    fn test_adaptive_profile_guided_options_with_aggregation_percentile() {
        assert_eq!(options(1, 1).with_aggregation_percentile(100).unwrap().aggregation_percentile(), 100);
        assert!(matches!(
            options(1, 1).with_aggregation_percentile(101),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "adaptive aggregation percentile must be in `0..=100`",
        ));
    }

    #[test]
    fn test_adaptive_profile_guided_options_with_profile_version() {
        assert_eq!(options(1, 1).with_profile_version(7).profile_version(), 7);
    }

    #[test]
    fn test_atomic_adaptive_profile_guided_statistics_snapshot() {
        let statistics = AtomicAdaptiveProfileGuidedStatistics::default();
        statistics.executions.fetch_add(3, Ordering::Relaxed);
        statistics.baseline_executions.fetch_add(2, Ordering::Relaxed);
        statistics.profile_failures.fetch_add(1, Ordering::Relaxed);
        let snapshot = statistics.snapshot();
        assert_eq!(
            snapshot,
            AdaptiveProfileGuidedStatistics {
                executions: 3,
                baseline_executions: 2,
                profile_failures: 1,
                ..AdaptiveProfileGuidedStatistics::default()
            },
        );
        assert_eq!(snapshot.executions(), 3);
        assert_eq!(snapshot.baseline_executions(), 2);
        assert_eq!(snapshot.profile_failures(), 1);
        assert_eq!(snapshot.optimized_executions(), 0);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_claim_dispatch_allows_one_profile_in_flight() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 3);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline);
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Profiling {
                profiles_collected: 0,
                profile_attempts: 1,
                profile_in_flight: true,
                last_failure: None,
            },
        );
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_profile_succeeded_triggers_one_compilation() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 3);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        assert_eq!(machine.profile_succeeded(PjrtFeedbackDirectedProfile::from_bytes(vec![1]), options), None);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        assert_eq!(
            machine.profile_succeeded(PjrtFeedbackDirectedProfile::from_bytes(vec![2]), options),
            Some(vec![
                PjrtFeedbackDirectedProfile::from_bytes(vec![1]),
                PjrtFeedbackDirectedProfile::from_bytes(vec![2])
            ]),
        );
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline);
        machine.compilation_succeeded();
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Optimized);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_profile_succeeded_fails_when_attempts_are_exhausted() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 2);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        machine.profile_failed("first attempt failed".to_string(), options);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        assert_eq!(machine.profile_succeeded(PjrtFeedbackDirectedProfile::from_bytes(vec![1]), options), None);
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Failed {
                profiles_collected: 1,
                profile_attempts: 2,
                reason: "collected 1 successful profile(s), fewer than the required 2 after exhausting 2 attempt(s)"
                    .to_string(),
            },
        );
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_profile_failed_bounds_retries() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 2);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        machine.profile_failed("first failure".to_string(), options);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        machine.profile_failed("second failure".to_string(), options);
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Failed {
                profiles_collected: 0,
                profile_attempts: 2,
                reason: "second failure".to_string(),
            },
        );
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_compilation_failed_is_terminal() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(1, 2);
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Profile);
        assert_eq!(
            machine.profile_succeeded(PjrtFeedbackDirectedProfile::from_bytes(vec![1]), options),
            Some(vec![PjrtFeedbackDirectedProfile::from_bytes(vec![1])]),
        );
        machine.compilation_failed("compile failed".to_string());
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Failed {
                profiles_collected: 1,
                profile_attempts: 1,
                reason: "compile failed".to_string(),
            },
        );
        assert_eq!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_profile_restored() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        machine.profile_restored();
        assert_eq!(machine.state, AdaptiveProfileGuidedState::Optimized { profiles_collected: 0, profile_attempts: 0 });
        assert_eq!(machine.claim_dispatch(options(1, 1)), AdaptiveDispatch::Optimized);
    }

    #[test]
    fn test_adaptive_profile_guided_state_machine_profile_restore_failed_falls_back_to_profiling() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        machine.profile_restore_failed("checksum mismatch".to_string());

        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Profiling {
                profiles_collected: 0,
                profile_attempts: 0,
                profile_in_flight: false,
                last_failure: Some("persistent adaptive profile restore failed: checksum mismatch".to_string()),
            },
        );
        assert_eq!(machine.claim_dispatch(options(1, 1)), AdaptiveDispatch::Profile);
    }

    #[test]
    fn test_validate_adaptive_profile_guided_platform() {
        assert!(matches!(validate_adaptive_profile_guided_platform("cuda"), Ok(())));
        assert!(matches!(validate_adaptive_profile_guided_platform("ROCm"), Ok(())));
        assert!(matches!(
            validate_adaptive_profile_guided_platform("cpu"),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "adaptive profile-guided recompilation requires a CUDA or ROCm PJRT platform, but \
                              found `cpu`",
        ));
    }
}
