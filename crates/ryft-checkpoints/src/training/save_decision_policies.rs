//! Policies that decide when a [`training::Checkpointer`](crate::training::Checkpointer) saves a checkpoint,
//! mirroring `save_decision_policy.py` of the training API of Orbax.
//!
//! A [`SaveDecisionPolicy`] decides whether to save a step, given the [`CheckpointMetadata`] of the checkpoints that
//! already exist, ordered from the oldest to the newest, and a [`DecisionContext`] that says whether a save is in
//! progress and whether the process is about to be preempted. Policies combine with [`AnySavePolicy`], which saves
//! when any of its policies would.
//!
//! A [`PreemptionSignal`] is the flag through which a process tells its checkpointer that it is about to be
//! preempted. This crate installs no signal handler. The process sets the flag itself (e.g., from its own `SIGTERM`
//! handler or when its scheduler announces a preemption), and a [`PreemptionCheckpointingPolicy`] then saves the next
//! step.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::errors::CheckpointError;
use crate::training::CheckpointMetadata;

// TODO(eaplatanios): Review this module.

/// State of the process that a [`SaveDecisionPolicy`] decides in, which mirrors the `DecisionContext` of Orbax.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DecisionContext {
    /// Whether an asynchronous save is in progress.
    is_saving_in_progress: bool,

    /// Step of the asynchronous save in progress, if any.
    in_progress_step: Option<u64>,

    /// Whether the process has been notified that it is about to be preempted.
    reached_preemption: bool,
}

impl DecisionContext {
    /// Creates a new [`DecisionContext`] without a save in progress and without a preemption.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns this context with the asynchronous save of `step` in progress, or with none when `step` is [`None`].
    #[inline]
    pub fn with_in_progress_step(mut self, step: Option<u64>) -> Self {
        self.is_saving_in_progress = step.is_some();
        self.in_progress_step = step;
        self
    }

    /// Returns this context with the provided preemption flag.
    #[inline]
    pub fn with_reached_preemption(mut self, reached_preemption: bool) -> Self {
        self.reached_preemption = reached_preemption;
        self
    }

    /// Returns whether an asynchronous save is in progress.
    #[inline]
    pub fn is_saving_in_progress(&self) -> bool {
        self.is_saving_in_progress
    }

    /// Returns the step of the asynchronous save in progress, if any.
    #[inline]
    pub fn in_progress_step(&self) -> Option<u64> {
        self.in_progress_step
    }

    /// Returns whether the process has been notified that it is about to be preempted.
    #[inline]
    pub fn reached_preemption(&self) -> bool {
        self.reached_preemption
    }
}

/// Flag through which a process announces that it is about to be preempted, which a
/// [`training::Checkpointer`](crate::training::Checkpointer) reports to its save decision policy as
/// [`DecisionContext::reached_preemption`]. Clones share one flag, so a clone can be moved into whatever notices the
/// preemption. This crate installs no signal handler, and the process sets the flag itself (e.g., from its own
/// `SIGTERM` handler).
#[derive(Clone, Debug, Default)]
pub struct PreemptionSignal {
    /// Shared flag.
    notified: Arc<AtomicBool>,
}

impl PreemptionSignal {
    /// Creates a new [`PreemptionSignal`] that has not been notified.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that the process is about to be preempted. This only stores to an atomic flag, so it is safe to call
    /// from a signal handler.
    #[inline]
    pub fn notify(&self) {
        self.notified.store(true, Ordering::SeqCst);
    }

    /// Clears the flag (e.g., after the preemption was cancelled).
    #[inline]
    pub fn reset(&self) {
        self.notified.store(false, Ordering::SeqCst);
    }

    /// Returns `true` if the process has been notified that it is about to be preempted.
    #[inline]
    pub fn is_notified(&self) -> bool {
        self.notified.load(Ordering::SeqCst)
    }
}

/// Policy that decides whether to save a checkpoint, which mirrors the `SaveDecisionPolicy` of Orbax.
pub trait SaveDecisionPolicy: Send + Sync {
    /// Returns `true` if the checkpoint described by `step` should be saved.
    ///
    /// # Parameters
    ///
    ///   - `step`: Step to decide on and the current time. Its metrics are not known yet and are [`None`].
    ///   - `previous`: Existing checkpoints, ordered from the oldest to the newest.
    ///   - `context`: State of the process (e.g., whether a save is in progress).
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool;
}

/// [`SaveDecisionPolicy`] that saves every step that is a multiple of a fixed interval.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedIntervalPolicy {
    /// Interval between saved steps.
    interval: u64,
}

impl FixedIntervalPolicy {
    /// Creates a new [`FixedIntervalPolicy`] that saves every step that is a multiple of `interval`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `interval` is zero.
    pub fn new(interval: u64) -> Result<Self, CheckpointError> {
        if interval == 0 {
            return Err(CheckpointError::InvalidArgument { message: "save intervals must be positive".to_string() });
        }
        Ok(Self { interval })
    }
}

impl SaveDecisionPolicy for FixedIntervalPolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        let _ = (previous, context);
        step.step().is_multiple_of(self.interval)
    }
}

/// [`SaveDecisionPolicy`] that saves a fixed set of steps.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpecificStepsPolicy {
    /// Steps to save.
    steps: BTreeSet<u64>,
}

impl SpecificStepsPolicy {
    /// Creates a new [`SpecificStepsPolicy`] that saves exactly the provided steps.
    #[inline]
    pub fn new<S: IntoIterator<Item = u64>>(steps: S) -> Self {
        Self { steps: steps.into_iter().collect() }
    }
}

impl SaveDecisionPolicy for SpecificStepsPolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        let _ = (previous, context);
        self.steps.contains(&step.step())
    }
}

/// [`SaveDecisionPolicy`] that saves when there is no checkpoint yet, so that training that resumes from scratch has
/// a checkpoint to restart from.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct InitialSavePolicy;

impl SaveDecisionPolicy for InitialSavePolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        let _ = (step, context);
        previous.is_empty()
    }
}

/// [`SaveDecisionPolicy`] that saves as often as allowed, which mirrors the `ContinuousCheckpointingPolicy` of Orbax.
/// It applies these rules in order:
///
///   1. A step that is a multiple of the target interval is saved.
///   2. Otherwise, nothing is saved while a save is in progress.
///   3. Otherwise, a step is saved when there is no previous checkpoint.
///   4. Otherwise, a step is skipped when the next multiple of the target interval is at most the deferral window
///      ahead, so that the save happens at that target step instead.
///   5. Otherwise, a step is saved when there is no minimum interval.
///   6. Otherwise, a step is saved once at least the minimum interval has passed since the latest checkpoint.
///
/// Orbax additionally cancels a save in progress that is not on the target interval in the first rule. Saves in
/// progress are never cancelled here, so a target step waits for the save in progress instead.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContinuousCheckpointingPolicy {
    /// Minimum time between the latest checkpoint and a new one, if any.
    minimum_interval: Option<Duration>,

    /// Interval of the steps that are always saved, if any.
    target_interval_steps: Option<u64>,

    /// Number of steps before a target step within which no other step is saved, if any.
    deferral_window_steps: Option<u64>,
}

impl ContinuousCheckpointingPolicy {
    /// Creates a new [`ContinuousCheckpointingPolicy`] with the provided minimum time between checkpoints, without
    /// target interval, and without deferral window.
    #[inline]
    pub fn new(minimum_interval: Option<Duration>) -> Self {
        Self { minimum_interval, target_interval_steps: None, deferral_window_steps: None }
    }

    /// Returns this policy with every multiple of `target_interval_steps` saved. A zero interval is treated as no
    /// interval.
    #[inline]
    pub fn with_target_interval_steps(mut self, target_interval_steps: Option<u64>) -> Self {
        self.target_interval_steps = target_interval_steps.filter(|interval| *interval > 0);
        self
    }

    /// Returns this policy with no other step saved within `deferral_window_steps` steps before a target step. The
    /// window only has an effect together with a target interval.
    #[inline]
    pub fn with_deferral_window_steps(mut self, deferral_window_steps: Option<u64>) -> Self {
        self.deferral_window_steps = deferral_window_steps;
        self
    }
}

impl SaveDecisionPolicy for ContinuousCheckpointingPolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        if self.target_interval_steps.is_some_and(|interval| step.step().is_multiple_of(interval)) {
            return true;
        }
        if context.is_saving_in_progress() {
            return false;
        }
        let Some(latest) = previous.last() else {
            return true;
        };
        if let (Some(interval), Some(window)) = (self.target_interval_steps, self.deferral_window_steps) {
            let steps_to_target = interval - step.step() % interval;
            if steps_to_target <= window {
                return false;
            }
        }
        match self.minimum_interval {
            None => true,
            Some(minimum_interval) => {
                step.time().duration_since(latest.time()).is_ok_and(|elapsed| elapsed >= minimum_interval)
            }
        }
    }
}

/// [`SaveDecisionPolicy`] that saves when the process is about to be preempted (i.e., when
/// [`DecisionContext::reached_preemption`] is set), which mirrors the `PreemptionCheckpointingPolicy` of Orbax.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PreemptionCheckpointingPolicy;

impl SaveDecisionPolicy for PreemptionCheckpointingPolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        let _ = (step, previous);
        context.reached_preemption()
    }
}

/// [`SaveDecisionPolicy`] that saves when any of its policies would, and never when it has no policies.
pub struct AnySavePolicy {
    /// Combined policies.
    policies: Vec<Box<dyn SaveDecisionPolicy>>,
}

impl AnySavePolicy {
    /// Creates a new [`AnySavePolicy`] that combines `policies`.
    #[inline]
    pub fn new(policies: Vec<Box<dyn SaveDecisionPolicy>>) -> Self {
        Self { policies }
    }
}

impl SaveDecisionPolicy for AnySavePolicy {
    fn should_save(
        &self,
        step: &CheckpointMetadata,
        previous: &[CheckpointMetadata],
        context: &DecisionContext,
    ) -> bool {
        self.policies.iter().any(|policy| policy.should_save(step, previous, context))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::training::tests::checkpoint;

    use super::*;

    #[test]
    fn test_decision_context() {
        let context = DecisionContext::new();
        assert!(!context.is_saving_in_progress());
        assert_eq!(context.in_progress_step(), None);
        assert!(!context.reached_preemption());
        let context = context.with_in_progress_step(Some(4)).with_reached_preemption(true);
        assert!(context.is_saving_in_progress());
        assert_eq!(context.in_progress_step(), Some(4));
        assert!(context.reached_preemption());
        assert!(!context.with_in_progress_step(None).is_saving_in_progress());
    }

    #[test]
    fn test_preemption_signal() {
        let signal = PreemptionSignal::new();
        let clone = signal.clone();
        assert!(!signal.is_notified());
        clone.notify();
        assert!(signal.is_notified());
        signal.reset();
        assert!(!clone.is_notified());
    }

    #[test]
    fn test_fixed_interval_policy() {
        let policy = FixedIntervalPolicy::new(5).unwrap();
        assert_eq!(
            [0, 3, 5, 12, 15].map(|step| policy.should_save(&checkpoint(step, 0, None), &[], &DecisionContext::new())),
            [true, false, true, false, true],
        );
        assert_eq!(
            FixedIntervalPolicy::new(0),
            Err(CheckpointError::InvalidArgument { message: "save intervals must be positive".to_string() }),
        );
    }

    #[test]
    fn test_specific_steps_policy() {
        let policy = SpecificStepsPolicy::new([2, 7]);
        assert_eq!(
            [1, 2, 7, 8].map(|step| policy.should_save(&checkpoint(step, 0, None), &[], &DecisionContext::new())),
            [false, true, true, false],
        );
    }

    #[test]
    fn test_initial_save_policy() {
        assert!(InitialSavePolicy.should_save(&checkpoint(5, 0, None), &[], &DecisionContext::new()));
        assert!(!InitialSavePolicy.should_save(
            &checkpoint(5, 0, None),
            &[checkpoint(1, 0, None)],
            &DecisionContext::new()
        ));
    }

    #[test]
    fn test_continuous_checkpointing_policy() {
        let previous = [checkpoint(1, 100, None), checkpoint(2, 200, None)];
        let unlimited = ContinuousCheckpointingPolicy::new(None);
        assert!(unlimited.should_save(&checkpoint(3, 200, None), &previous, &DecisionContext::new()));

        // With a minimum interval, the time since the latest checkpoint decides, and a first checkpoint is saved.
        let limited = ContinuousCheckpointingPolicy::new(Some(Duration::from_secs(60)));
        assert!(limited.should_save(&checkpoint(3, 0, None), &[], &DecisionContext::new()));
        assert!(!limited.should_save(&checkpoint(3, 259, None), &previous, &DecisionContext::new()));
        assert!(limited.should_save(&checkpoint(3, 260, None), &previous, &DecisionContext::new()));
        assert!(!limited.should_save(&checkpoint(3, 150, None), &previous, &DecisionContext::new()));
    }

    #[test]
    fn test_continuous_checkpointing_policy_target_interval() {
        let previous = [checkpoint(1, 100, None)];
        let idle = DecisionContext::new();
        let busy = DecisionContext::new().with_in_progress_step(Some(1));
        let policy = ContinuousCheckpointingPolicy::new(Some(Duration::from_secs(60)))
            .with_target_interval_steps(Some(10))
            .with_deferral_window_steps(Some(2));

        // Target steps always save, even while a save is in progress and before the minimum interval has passed.
        assert!(policy.should_save(&checkpoint(10, 101, None), &previous, &busy));
        assert!(policy.should_save(&checkpoint(20, 101, None), &previous, &idle));

        // Other steps are skipped while a save is in progress and within the deferral window before a target step,
        // and otherwise follow the minimum interval.
        assert!(!policy.should_save(&checkpoint(5, 500, None), &previous, &busy));
        assert!(policy.should_save(&checkpoint(5, 500, None), &[], &idle));
        assert!(!policy.should_save(&checkpoint(8, 500, None), &previous, &idle));
        assert!(!policy.should_save(&checkpoint(9, 500, None), &previous, &idle));
        assert!(policy.should_save(&checkpoint(7, 500, None), &previous, &idle));
        assert!(!policy.should_save(&checkpoint(7, 120, None), &previous, &idle));

        // Without a minimum interval, every step outside the deferral window is saved.
        let unlimited = ContinuousCheckpointingPolicy::new(None).with_target_interval_steps(Some(0));
        assert_eq!(unlimited, ContinuousCheckpointingPolicy::new(None));
        assert!(unlimited.should_save(&checkpoint(7, 100, None), &previous, &idle));
        assert!(!unlimited.should_save(&checkpoint(7, 100, None), &previous, &busy));
    }

    #[test]
    fn test_preemption_checkpointing_policy() {
        let context = DecisionContext::new();
        assert!(!PreemptionCheckpointingPolicy.should_save(&checkpoint(3, 0, None), &[], &context));
        assert!(PreemptionCheckpointingPolicy.should_save(
            &checkpoint(3, 0, None),
            &[],
            &context.with_reached_preemption(true)
        ));
    }

    #[test]
    fn test_any_save_policy() {
        let policy = AnySavePolicy::new(vec![
            Box::new(FixedIntervalPolicy::new(10).unwrap()),
            Box::new(SpecificStepsPolicy::new([3])),
        ]);
        assert_eq!(
            [3, 4, 10].map(|step| policy.should_save(&checkpoint(step, 0, None), &[], &DecisionContext::new())),
            [true, false, true],
        );
        assert!(!AnySavePolicy::new(Vec::new()).should_save(&checkpoint(0, 0, None), &[], &DecisionContext::new()));
    }
}
