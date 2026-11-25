//! Policies that decide which checkpoints a [`training::Checkpointer`](crate::training::Checkpointer) keeps,
//! mirroring `preservation_policy.py` of the training API of Orbax.
//!
//! A [`PreservationPolicy`] sees the [`CheckpointMetadata`] of all checkpoints, ordered from the oldest to the newest,
//! and decides which of them to keep. A checkpointer deletes every checkpoint that its policy does not preserve,
//! except the one that it just saved. Policies combine with [`AnyPreservationPolicy`], which preserves a checkpoint
//! when any of its policies would.

use std::cmp::Ordering;
use std::time::{Duration, SystemTime};

use crate::errors::CheckpointError;
use crate::training::CheckpointMetadata;

// TODO(eaplatanios): Review this module.

/// Policy that decides which checkpoints to keep, which mirrors the `PreservationPolicy` of Orbax.
pub trait PreservationPolicy: Send + Sync {
    /// Returns whether to keep every checkpoint of `checkpoints`, which are ordered from the oldest to the newest, in
    /// the same order.
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool>;
}

/// [`PreservationPolicy`] that keeps every checkpoint.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PreserveAll;

impl PreservationPolicy for PreserveAll {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        vec![true; checkpoints.len()]
    }
}

/// [`PreservationPolicy`] that keeps the newest checkpoints.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LatestN {
    /// Number of checkpoints to keep.
    count: usize,
}

impl LatestN {
    /// Creates a new [`LatestN`] policy that keeps the newest `count` checkpoints.
    #[inline]
    pub fn new(count: usize) -> Self {
        Self { count }
    }
}

impl PreservationPolicy for LatestN {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        let first_kept = checkpoints.len().saturating_sub(self.count);
        (0..checkpoints.len()).map(|index| index >= first_kept).collect()
    }
}

/// [`PreservationPolicy`] that keeps every checkpoint whose step is a multiple of a fixed interval.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EveryNSteps {
    /// Interval between kept steps.
    interval: u64,
}

impl EveryNSteps {
    /// Creates a new [`EveryNSteps`] policy that keeps every checkpoint whose step is a multiple of `interval`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `interval` is zero.
    pub fn new(interval: u64) -> Result<Self, CheckpointError> {
        if interval == 0 {
            return Err(CheckpointError::InvalidArgument {
                message: "preservation intervals must be positive".to_string(),
            });
        }
        Ok(Self { interval })
    }
}

impl PreservationPolicy for EveryNSteps {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        checkpoints.iter().map(|checkpoint| checkpoint.step().is_multiple_of(self.interval)).collect()
    }
}

/// [`PreservationPolicy`] that keeps checkpoints at least a fixed time apart. It keeps the oldest checkpoint and then
/// every checkpoint that was saved at least the interval after the last kept one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EveryNSeconds {
    /// Minimum time between kept checkpoints.
    interval: Duration,
}

impl EveryNSeconds {
    /// Creates a new [`EveryNSeconds`] policy that keeps checkpoints at least `interval` apart.
    #[inline]
    pub fn new(interval: Duration) -> Self {
        Self { interval }
    }
}

impl PreservationPolicy for EveryNSeconds {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        let mut last_kept: Option<SystemTime> = None;
        checkpoints
            .iter()
            .map(|checkpoint| {
                let keep = last_kept.is_none_or(|last_kept| {
                    checkpoint.time().duration_since(last_kept).is_ok_and(|elapsed| elapsed >= self.interval)
                });
                if keep {
                    last_kept = Some(checkpoint.time());
                }
                keep
            })
            .collect()
    }
}

/// Direction in which a metric improves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MetricMode {
    /// Larger values are better (e.g., accuracy).
    Maximize,

    /// Smaller values are better (e.g., loss).
    Minimize,
}

impl MetricMode {
    /// Compares two metric values such that the better one is [`Ordering::Less`], using the total order of `f64` for
    /// values that are not `NaN`. A `NaN` is worse than every other value in both modes, so that the checkpoint of a
    /// diverged step is never the best one.
    fn compare(self, left: f64, right: f64) -> Ordering {
        match (left.is_nan(), right.is_nan(), self) {
            (true, true, _) => Ordering::Equal,
            (true, false, _) => Ordering::Greater,
            (false, true, _) => Ordering::Less,
            (false, false, Self::Maximize) => right.total_cmp(&left),
            (false, false, Self::Minimize) => left.total_cmp(&right),
        }
    }
}

/// [`PreservationPolicy`] that keeps the checkpoints with the best values of one metric. Among checkpoints with equal
/// values the newer ones are preferred. Checkpoints without the metric are kept or not as a whole, as configured with
/// [`Self::with_keep_checkpoints_without_metrics`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BestN {
    /// Name of the metric.
    metric: String,

    /// Direction in which the metric improves.
    mode: MetricMode,

    /// Number of checkpoints with the metric to keep.
    count: usize,

    /// Whether to keep checkpoints that do not have the metric.
    keep_checkpoints_without_metrics: bool,
}

impl BestN {
    /// Creates a new [`BestN`] policy that keeps the `count` checkpoints with the best values of the metric named
    /// `metric`, and every checkpoint that does not have that metric.
    #[inline]
    pub fn new<M: Into<String>>(metric: M, mode: MetricMode, count: usize) -> Self {
        Self { metric: metric.into(), mode, count, keep_checkpoints_without_metrics: true }
    }

    /// Returns this policy configured to keep checkpoints that do not have the metric when
    /// `keep_checkpoints_without_metrics` is `true` and to delete them otherwise.
    #[inline]
    pub fn with_keep_checkpoints_without_metrics(mut self, keep_checkpoints_without_metrics: bool) -> Self {
        self.keep_checkpoints_without_metrics = keep_checkpoints_without_metrics;
        self
    }
}

impl PreservationPolicy for BestN {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        let mut ranked = checkpoints
            .iter()
            .enumerate()
            .filter_map(|(index, checkpoint)| checkpoint.metric(&self.metric).map(|value| (index, value)))
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_index, left), (right_index, right)| {
            self.mode.compare(*left, *right).then(right_index.cmp(left_index))
        });
        let mut preserve = checkpoints
            .iter()
            .map(|checkpoint| checkpoint.metric(&self.metric).is_none() && self.keep_checkpoints_without_metrics)
            .collect::<Vec<_>>();
        for (index, _) in ranked.into_iter().take(self.count) {
            preserve[index] = true;
        }
        preserve
    }
}

/// [`PreservationPolicy`] that keeps a checkpoint when any of its policies would, and none when it has no policies.
pub struct AnyPreservationPolicy {
    /// Combined policies.
    policies: Vec<Box<dyn PreservationPolicy>>,
}

impl AnyPreservationPolicy {
    /// Creates a new [`AnyPreservationPolicy`] that combines `policies`.
    #[inline]
    pub fn new(policies: Vec<Box<dyn PreservationPolicy>>) -> Self {
        Self { policies }
    }
}

impl PreservationPolicy for AnyPreservationPolicy {
    fn should_preserve(&self, checkpoints: &[CheckpointMetadata]) -> Vec<bool> {
        let mut preserve = vec![false; checkpoints.len()];
        for policy in &self.policies {
            for (preserved, keep) in preserve.iter_mut().zip(policy.should_preserve(checkpoints)) {
                *preserved |= keep;
            }
        }
        preserve
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::training::tests::checkpoint;

    use super::*;

    #[test]
    fn test_preserve_all() {
        assert_eq!(PreserveAll.should_preserve(&[checkpoint(1, 0, None), checkpoint(2, 0, None)]), vec![true, true]);
    }

    #[test]
    fn test_latest_n() {
        let checkpoints = [checkpoint(1, 0, None), checkpoint(2, 0, None), checkpoint(3, 0, None)];
        assert_eq!(LatestN::new(2).should_preserve(&checkpoints), vec![false, true, true]);
        assert_eq!(LatestN::new(5).should_preserve(&checkpoints), vec![true, true, true]);
        assert_eq!(LatestN::new(0).should_preserve(&checkpoints), vec![false, false, false]);
    }

    #[test]
    fn test_every_n_steps() {
        let checkpoints = [checkpoint(5, 0, None), checkpoint(10, 0, None), checkpoint(15, 0, None)];
        assert_eq!(EveryNSteps::new(10).unwrap().should_preserve(&checkpoints), vec![false, true, false]);
        assert_eq!(
            EveryNSteps::new(0),
            Err(CheckpointError::InvalidArgument { message: "preservation intervals must be positive".to_string() }),
        );
    }

    #[test]
    fn test_every_n_seconds() {
        let checkpoints = [
            checkpoint(1, 100, None),
            checkpoint(2, 130, None),
            checkpoint(3, 160, None),
            checkpoint(4, 170, None),
            checkpoint(5, 300, None),
        ];
        assert_eq!(
            EveryNSeconds::new(Duration::from_secs(60)).should_preserve(&checkpoints),
            vec![true, false, true, false, true],
        );
    }

    #[test]
    fn test_best_n() {
        let checkpoints = [
            checkpoint(1, 0, Some(0.5)),
            checkpoint(2, 0, None),
            checkpoint(3, 0, Some(0.25)),
            checkpoint(4, 0, Some(0.75)),
            checkpoint(5, 0, Some(0.25)),
        ];
        assert_eq!(
            BestN::new("loss", MetricMode::Minimize, 2).should_preserve(&checkpoints),
            vec![false, true, true, false, true],
        );
        assert_eq!(
            BestN::new("loss", MetricMode::Maximize, 2).should_preserve(&checkpoints),
            vec![true, true, false, true, false],
        );

        // A metric that is not a number is worse than every other value in both modes.
        let diverged = [checkpoint(1, 0, Some(0.5)), checkpoint(2, 0, Some(f64::NAN)), checkpoint(3, 0, Some(0.75))];
        assert_eq!(BestN::new("loss", MetricMode::Minimize, 1).should_preserve(&diverged), vec![true, false, false]);
        assert_eq!(BestN::new("loss", MetricMode::Maximize, 1).should_preserve(&diverged), vec![false, false, true]);

        // Ties prefer the newer checkpoint, and checkpoints without the metric can be deleted.
        assert_eq!(
            BestN::new("loss", MetricMode::Minimize, 1)
                .with_keep_checkpoints_without_metrics(false)
                .should_preserve(&checkpoints),
            vec![false, false, false, false, true],
        );
        assert_eq!(
            BestN::new("accuracy", MetricMode::Maximize, 1)
                .with_keep_checkpoints_without_metrics(false)
                .should_preserve(&checkpoints),
            vec![false; 5],
        );
    }

    #[test]
    fn test_any_preservation_policy() {
        let checkpoints = [checkpoint(10, 0, None), checkpoint(15, 0, None), checkpoint(17, 0, None)];
        let policy =
            AnyPreservationPolicy::new(vec![Box::new(LatestN::new(1)), Box::new(EveryNSteps::new(10).unwrap())]);
        assert_eq!(policy.should_preserve(&checkpoints), vec![true, false, true]);
        assert_eq!(AnyPreservationPolicy::new(Vec::new()).should_preserve(&checkpoints), vec![false; 3]);
    }
}
