//! Adaptive XLA profile-guided recompilation.
//!
//! [`AdaptiveProfileGuidedXlaFunction`] mirrors JAX's automatic Profile-Guided Latency Estimator flow while keeping
//! the policy and lifecycle explicit. One caller at a time profiles the baseline executable. Other callers continue
//! dispatching without waiting, and compilation occurs outside all adaptive-state and dispatch locks. A compatible
//! optimized executable is published only after compilation succeeds; every failure path is bounded and observable.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ryft_core::{ArrayIrType, ArrayIrValue, ArrayType, CompilationCacheDomain, Parameterized, ParameterizedFamily};
use ryft_pjrt::extensions::profiler::FeedbackDirectedProfile;
use ryft_pjrt::protos::{ProfileDeviceType, ProfileOptions};

use crate::experimental::XlaDomainError;
use crate::experimental::domains::XlaLoweredProgram;
use crate::experimental::ops::XlaConstant;
use crate::{Array, ExecutableXlaFunction, XlaDomain, XlaFeedbackDirectedProfile, XlaOptions};

const ADAPTIVE_PROFILE_CACHE_NAMESPACE: &str = "xla-adaptive-profile-v1";

/// Policy for one adaptive profile-guided recompilation lifecycle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AdaptiveProfileGuidedOptions {
    sample_count: usize,
    maximum_profile_attempts: usize,
    aggregation_percentile: u8,
    profile_version: i64,
}

impl AdaptiveProfileGuidedOptions {
    /// Creates a policy that collects `sample_count` successful profiles in at most
    /// `maximum_profile_attempts` selected executions.
    pub fn new(sample_count: usize, maximum_profile_attempts: usize) -> Result<Self, XlaDomainError> {
        if sample_count == 0 {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "adaptive profile sample_count must be greater than zero".to_string(),
            });
        }
        if maximum_profile_attempts < sample_count {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "adaptive maximum_profile_attempts must be at least sample_count".to_string(),
            });
        }
        Ok(Self { sample_count, maximum_profile_attempts, aggregation_percentile: 90, profile_version: 0 })
    }

    /// Sets the OpenXLA instruction-cost aggregation percentile.
    pub fn with_aggregation_percentile(mut self, percentile: u8) -> Result<Self, XlaDomainError> {
        if percentile > 100 {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "adaptive aggregation percentile must be in 0..=100".to_string(),
            });
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
        Self::new(3, 5).expect("the default adaptive profile-guided policy should be valid")
    }
}

/// Observable phase of an adaptive profile-guided executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdaptiveProfileGuidedState {
    /// The baseline remains active while successful profiles are collected.
    Profiling {
        profiles_collected: usize,
        profile_attempts: usize,
        profile_in_flight: bool,
        last_failure: Option<String>,
    },
    /// The required samples were collected and one caller is compiling an optimized executable.
    Compiling { profiles_collected: usize, profile_attempts: usize },
    /// The compatible optimized executable is active.
    Optimized { profiles_collected: usize, profile_attempts: usize },
    /// Adaptation stopped permanently; the baseline executable remains active.
    Failed { profiles_collected: usize, profile_attempts: usize, reason: String },
}

/// Lock-free activity snapshot for an [`AdaptiveProfileGuidedXlaFunction`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveProfileGuidedStatistics {
    /// Total calls dispatched through this wrapper.
    pub executions: usize,
    /// Calls dispatched to the baseline, including selected profiling calls.
    pub baseline_executions: usize,
    /// Baseline calls selected to own a profiling attempt.
    pub profiled_executions: usize,
    /// Calls dispatched after an optimized executable was installed.
    pub optimized_executions: usize,
    /// Profiling attempts that failed before yielding an OpenXLA profile.
    pub profile_failures: usize,
    /// Profile aggregation and recompilation attempts. This is at most one.
    pub recompilations: usize,
    /// Terminal aggregation or recompilation failures.
    pub recompilation_failures: usize,
    /// Compatible optimized executable installations. This is at most one.
    pub executable_replacements: usize,
    /// Persisted aggregated profiles restored and installed without new sampling.
    pub profile_restore_hits: usize,
    /// Configured persistent profile lookups that found no sidecar.
    pub profile_restore_misses: usize,
    /// Persistent profile read, validation, or recompilation failures that fell back to bounded sampling.
    pub profile_restore_failures: usize,
    /// Aggregated profiles persisted after a successful replacement.
    pub profile_persistences: usize,
    /// Profile-sidecar writes that failed after a successful replacement.
    pub profile_persistence_failures: usize,
}

#[derive(Default)]
struct AtomicAdaptiveProfileGuidedStatistics {
    executions: AtomicUsize,
    baseline_executions: AtomicUsize,
    profiled_executions: AtomicUsize,
    optimized_executions: AtomicUsize,
    profile_failures: AtomicUsize,
    recompilations: AtomicUsize,
    recompilation_failures: AtomicUsize,
    executable_replacements: AtomicUsize,
    profile_restore_hits: AtomicUsize,
    profile_restore_misses: AtomicUsize,
    profile_restore_failures: AtomicUsize,
    profile_persistences: AtomicUsize,
    profile_persistence_failures: AtomicUsize,
}

impl AtomicAdaptiveProfileGuidedStatistics {
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

enum AdaptiveDispatch {
    Profile,
    Baseline,
    Optimized,
}

struct AdaptiveProfileGuidedStateMachine {
    state: AdaptiveProfileGuidedState,
    profiles: Vec<FeedbackDirectedProfile>,
}

impl AdaptiveProfileGuidedStateMachine {
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

    fn profile_succeeded(
        &mut self,
        profile: FeedbackDirectedProfile,
        options: AdaptiveProfileGuidedOptions,
    ) -> Option<Vec<FeedbackDirectedProfile>> {
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

    fn compilation_succeeded(&mut self) {
        let AdaptiveProfileGuidedState::Compiling { profiles_collected, profile_attempts } = self.state.clone() else {
            return;
        };
        self.state = AdaptiveProfileGuidedState::Optimized { profiles_collected, profile_attempts };
        self.profiles.clear();
    }

    fn compilation_failed(&mut self, reason: String) {
        let AdaptiveProfileGuidedState::Compiling { profiles_collected, profile_attempts } = self.state.clone() else {
            return;
        };
        self.state = AdaptiveProfileGuidedState::Failed { profiles_collected, profile_attempts, reason };
        self.profiles.clear();
    }

    fn profile_restored(&mut self) {
        self.state = AdaptiveProfileGuidedState::Optimized { profiles_collected: 0, profile_attempts: 0 };
        self.profiles.clear();
    }

    fn profile_restore_failed(&mut self, reason: String) {
        if let AdaptiveProfileGuidedState::Profiling { last_failure, .. } = &mut self.state {
            *last_failure = Some(format!("persistent adaptive profile restore failed: {reason}"));
        }
    }
}

fn validate_adaptive_profile_guided_platform(platform_name: &str) -> Result<(), XlaDomainError> {
    if matches!(platform_name.to_ascii_lowercase().as_str(), "cuda" | "rocm") {
        Ok(())
    } else {
        Err(XlaDomainError::InvalidCompilationOptions {
            reason: format!(
                "adaptive profile-guided recompilation requires a CUDA or ROCm PJRT platform, but found \
                 {platform_name}",
            ),
        })
    }
}

fn adaptive_profile_sidecar_key<'c>(
    domain: &crate::experimental::domains::XlaDomain<'c>,
    lowered_program: &XlaLoweredProgram,
    options: AdaptiveProfileGuidedOptions,
) -> Result<Option<Vec<u8>>, XlaDomainError> {
    let cache = domain.compilation_context();
    if cache.disk_cache().is_none() {
        return Ok(None);
    }
    let compilation_key = domain.compilation_key(lowered_program)?;
    let Some(baseline_key) = domain.persistent_cache_key(&compilation_key) else {
        return Ok(None);
    };
    let baseline_key_length =
        u64::try_from(baseline_key.len()).map_err(|_| XlaDomainError::InvalidCompilationOptions {
            reason: "baseline persistent key length does not fit in u64".to_string(),
        })?;
    let sample_count = u64::try_from(options.sample_count).map_err(|_| XlaDomainError::InvalidCompilationOptions {
        reason: "adaptive sample count does not fit in u64".to_string(),
    })?;
    let maximum_profile_attempts =
        u64::try_from(options.maximum_profile_attempts).map_err(|_| XlaDomainError::InvalidCompilationOptions {
            reason: "adaptive maximum profile attempts does not fit in u64".to_string(),
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

/// Runtime-only executable that performs one bounded adaptive profile-guided recompilation lifecycle.
///
/// Clones share dispatch state. Only a selected profiling call synchronizes its own baseline execution so the XProf
/// trace is complete. Concurrent calls copy the currently active executable handle and execute without holding the
/// state or dispatch locks. Recompilation is single-flight and terminal: it either installs one compatible executable
/// or records a permanent failure while continuing to serve the baseline.
pub struct AdaptiveProfileGuidedXlaFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    inner: Arc<AdaptiveProfileGuidedXlaFunctionInner<'c, In, Out>>,
}

struct AdaptiveProfileGuidedXlaFunctionInner<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    domain: XlaDomain<'c>,
    baseline: ExecutableXlaFunction<'c, In, Out>,
    active: RwLock<ExecutableXlaFunction<'c, In, Out>>,
    lowered_program: XlaLoweredProgram,
    options: AdaptiveProfileGuidedOptions,
    profile_sidecar_key: Option<Vec<u8>>,
    state: Mutex<AdaptiveProfileGuidedStateMachine>,
    statistics: AtomicAdaptiveProfileGuidedStatistics,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone
    for AdaptiveProfileGuidedXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> AdaptiveProfileGuidedXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    pub(crate) fn new(
        domain: XlaDomain<'c>,
        baseline: ExecutableXlaFunction<'c, In, Out>,
        lowered_program: XlaLoweredProgram,
        compilation_options: XlaOptions,
        options: AdaptiveProfileGuidedOptions,
    ) -> Result<Self, XlaDomainError> {
        if compilation_options.feedback_directed_profile.is_some() {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "adaptive profile-guided recompilation requires an unprofiled baseline".to_string(),
            });
        }
        let client = domain.client()?;
        validate_adaptive_profile_guided_platform(client.platform_name()?.as_ref())?;
        client.profiler_extension()?;
        let profile_sidecar_key = adaptive_profile_sidecar_key(&domain, &lowered_program, options)?;
        let function = Self {
            inner: Arc::new(AdaptiveProfileGuidedXlaFunctionInner {
                domain,
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
    pub fn interpret(&self, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<ArrayIrValue<Array<'c>>>,
        Out::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<ArrayIrValue<Array<'c>>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        self.inner.statistics.executions.fetch_add(1, Ordering::Relaxed);
        let dispatch = self
            .inner
            .state
            .lock()
            .expect("adaptive profile-guided state mutex should not be poisoned")
            .claim_dispatch(self.inner.options);
        let executable = self.inner.active.read().expect("adaptive executable lock should not be poisoned").clone();
        match dispatch {
            AdaptiveDispatch::Profile => {
                self.inner.statistics.baseline_executions.fetch_add(1, Ordering::Relaxed);
                self.inner.statistics.profiled_executions.fetch_add(1, Ordering::Relaxed);
                self.profile_baseline(executable, inputs)
            }
            AdaptiveDispatch::Baseline => {
                self.inner.statistics.baseline_executions.fetch_add(1, Ordering::Relaxed);
                self.inner.domain.interpret(&executable, inputs)
            }
            AdaptiveDispatch::Optimized => {
                self.inner.statistics.optimized_executions.fetch_add(1, Ordering::Relaxed);
                self.inner.domain.interpret(&executable, inputs)
            }
        }
    }

    fn profile_baseline(
        &self,
        executable: ExecutableXlaFunction<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<ArrayIrValue<Array<'c>>>,
        Out::Family: ParameterizedFamily<Array<'c>> + ParameterizedFamily<ArrayIrValue<Array<'c>>>,
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
        let profiler = match self.inner.domain.client()?.profiler(&profiler_options) {
            Ok(profiler) => profiler,
            Err(error) => {
                let outputs = self.inner.domain.interpret(&executable, inputs)?;
                self.record_profile_failure(error.to_string());
                return Ok(outputs);
            }
        };
        if let Err(error) = profiler.start() {
            let outputs = self.inner.domain.interpret(&executable, inputs)?;
            self.record_profile_failure(error.to_string());
            return Ok(outputs);
        }
        let execution = match self.inner.domain.interpret_async(&executable, inputs) {
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
            .and_then(|results| FeedbackDirectedProfile::from_x_space(&results));
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

    fn record_profile_failure(&self, reason: String) {
        self.inner.statistics.profile_failures.fetch_add(1, Ordering::Relaxed);
        self.inner
            .state
            .lock()
            .expect("adaptive profile-guided state mutex should not be poisoned")
            .profile_failed(reason, self.inner.options);
    }

    fn compile_profile(&self, profile: Vec<u8>) -> Result<ExecutableXlaFunction<'c, In, Out>, XlaDomainError> {
        let profile = XlaFeedbackDirectedProfile::new(profile).with_version(self.inner.options.profile_version);
        let lowered_program = self.inner.lowered_program.clone().with_feedback_directed_profile(profile);
        let domain = &self.inner.domain;
        let key = domain.compilation_key(&lowered_program)?;
        let program = domain
            .compilation_context()
            .get_or_compile(domain, key, || domain.compile_xla_program(&lowered_program))?;
        domain.replace_executable_xla_program(&self.inner.baseline, program)
    }

    fn restore_persisted_profile(&self) {
        let Some(key) = self.inner.profile_sidecar_key.as_deref() else {
            return;
        };
        let Some(disk_cache) = self.inner.domain.compilation_context().disk_cache() else {
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
                *self.inner.active.write().expect("adaptive executable lock should not be poisoned") = candidate;
                self.inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .profile_restored();
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

    fn persist_profile(&self, profile: &[u8]) {
        let Some(key) = self.inner.profile_sidecar_key.as_deref() else {
            return;
        };
        let Some(disk_cache) = self.inner.domain.compilation_context().disk_cache() else {
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

    fn recompile(&self, profiles: Vec<FeedbackDirectedProfile>) {
        self.inner.statistics.recompilations.fetch_add(1, Ordering::Relaxed);
        let (profile, candidate) =
            match FeedbackDirectedProfile::aggregated(profiles.as_slice(), self.inner.options.aggregation_percentile) {
                Ok(profile) => {
                    let candidate = self.compile_profile(profile.bytes().to_vec());
                    (Some(profile), candidate)
                }
                Err(error) => (None, Err(XlaDomainError::from(error))),
            };
        match candidate {
            Ok(candidate) => {
                *self.inner.active.write().expect("adaptive executable lock should not be poisoned") = candidate;
                self.inner
                    .state
                    .lock()
                    .expect("adaptive profile-guided state mutex should not be poisoned")
                    .compilation_succeeded();
                self.inner.statistics.executable_replacements.fetch_add(1, Ordering::Relaxed);
                self.persist_profile(profile.expect("a successful candidate requires an aggregated profile").bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn options(sample_count: usize, maximum_profile_attempts: usize) -> AdaptiveProfileGuidedOptions {
        AdaptiveProfileGuidedOptions::new(sample_count, maximum_profile_attempts).unwrap()
    }

    #[test]
    fn test_options_reject_unreachable_sampling_policies() {
        assert!(AdaptiveProfileGuidedOptions::new(0, 1).is_err());
        assert!(AdaptiveProfileGuidedOptions::new(2, 1).is_err());
        assert!(AdaptiveProfileGuidedOptions::new(1, 1).unwrap().with_aggregation_percentile(101).is_err());
    }

    #[test]
    fn test_only_one_profiling_call_can_be_in_flight() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 3);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline));
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
    fn test_profile_retries_are_bounded_and_failure_is_terminal() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 2);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        machine.profile_failed("first failure".to_string(), options);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        machine.profile_failed("second failure".to_string(), options);
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Failed {
                profiles_collected: 0,
                profile_attempts: 2,
                reason: "second failure".to_string(),
            },
        );
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline));
    }

    #[test]
    fn test_successful_samples_trigger_exactly_one_compilation() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 3);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        assert!(machine.profile_succeeded(FeedbackDirectedProfile::from_bytes(vec![1]), options).is_none());
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        assert_eq!(
            machine.profile_succeeded(FeedbackDirectedProfile::from_bytes(vec![2]), options),
            Some(vec![FeedbackDirectedProfile::from_bytes(vec![1]), FeedbackDirectedProfile::from_bytes(vec![2])]),
        );
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline));
        machine.compilation_succeeded();
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Optimized));
    }

    #[test]
    fn test_compilation_failure_is_terminal_without_reentering_profiling() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(1, 2);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        assert_eq!(
            machine.profile_succeeded(FeedbackDirectedProfile::from_bytes(vec![1]), options),
            Some(vec![FeedbackDirectedProfile::from_bytes(vec![1])]),
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
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline));
    }

    #[test]
    fn test_mixed_profile_results_fail_when_attempt_budget_cannot_reach_sample_count() {
        let mut machine = AdaptiveProfileGuidedStateMachine::new();
        let options = options(2, 2);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        machine.profile_failed("first attempt failed".to_string(), options);
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Profile));
        assert!(machine.profile_succeeded(FeedbackDirectedProfile::from_bytes(vec![1]), options).is_none());
        assert_eq!(
            machine.state,
            AdaptiveProfileGuidedState::Failed {
                profiles_collected: 1,
                profile_attempts: 2,
                reason: "collected 1 successful profile(s), fewer than the required 2 after exhausting 2 attempt(s)"
                    .to_string(),
            },
        );
        assert!(matches!(machine.claim_dispatch(options), AdaptiveDispatch::Baseline));
    }

    #[test]
    fn test_persistent_restore_failure_remains_observable_while_profiling_falls_back() {
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
        assert!(matches!(machine.claim_dispatch(options(1, 1)), AdaptiveDispatch::Profile));
    }
}
