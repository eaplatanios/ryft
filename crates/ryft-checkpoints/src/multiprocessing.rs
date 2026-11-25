//! Coordination of the processes that save one checkpoint together, mirroring the `MultiprocessingOptions`, primary
//! host, and barriers of Orbax.
//!
//! In multi-process training, every process calls the same checkpointing functions for the same steps. The processes
//! agree on one directory per checkpoint and synchronize at [`ProcessBarrier`]s, so that the checkpoint is finalized
//! only once every process has written its part. [`save_checkpointables`](crate::saving::save_checkpointables), with
//! a [`Context`](crate::contexts::Context) that carries [`MultiprocessingOptions`], saves in four phases that are
//! separated by barriers whose keys include the destination path, so that distinct saves never collide:
//!
//! ```text
//!   barrier `save_start`
//!   primary: handle an existing destination, remove a stale temporary directory, and create the directories
//!   barrier `directory_created`
//!   every process: `CheckpointableSaver::save` of every checkpointable into the same directories
//!   barrier `write_complete`
//!   primary: write the step metadata and finalize (rename or commit file)
//!   barrier `commit_complete`
//!   primary: call `CheckpointableHandler::finalize` of every checkpointable
//! ```
//!
//! Every saver learns the role of its process from the options that
//! [`CheckpointableSaver::save`](crate::handlers::CheckpointableSaver::save) receives. In this version, leaves are
//! assumed to be held completely by every process (i.e., replicated), so the handlers of this crate write arrays,
//! parameterized metadata, and JSON values only in the primary process, and the other processes write nothing, which
//! is also what Orbax does for replicated arrays. **Leaves that are split across
//! processes (i.e., where each process holds only some shards) are not supported yet.** Loading reads every array in
//! every process, so it needs a file system that every process can read (e.g., a shared disk or an object store).
//!
//! The processes must agree on which steps they save, so the save decision policy of a
//! [`training::Checkpointer`](crate::training::Checkpointer) must then depend on the step only. Policies that decide
//! from the clock or from a [`PreemptionSignal`](crate::training::PreemptionSignal) are evaluated in every process
//! separately and can disagree, which leaves the processes that save waiting at a barrier. **The decision of the
//! primary process is not broadcast to the other processes yet.**
//!
//! A process that fails before a barrier leaves the other processes waiting at that barrier until its timeout, which
//! is why production barriers should always have one. After a failed save the processes may no longer agree on how
//! often they have passed each barrier, so a multi-process job should end when a save fails, as a multi-host JAX job
//! does when one of its processes fails, rather than retry the save.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::errors::CheckpointError;

// TODO(eaplatanios): Review this module.

/// Barrier across the processes that save one checkpoint together. Implementations typically wrap the barrier of a
/// distributed runtime (e.g., the key value store of a coordination service).
pub trait ProcessBarrier: Debug + Send + Sync {
    /// Returns once every participating process has called this function with the same `key`, or fails when
    /// `timeout` passes first.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Barrier`] if the barrier times out or fails.
    fn wait(&self, key: &str, timeout: Option<Duration>) -> Result<(), CheckpointError>;
}

/// [`ProcessBarrier`] for a fixed number of participants within one process (e.g., threads that play the role of
/// processes in tests). Each key is reusable, because every completed round of a key starts a new one.
pub struct LocalProcessBarrier {
    /// Number of participants that every round waits for.
    participant_count: usize,

    /// Round state of every key that has been used.
    rounds: Mutex<HashMap<String, LocalBarrierRound>>,

    /// Condition that participants wait on until their round completes.
    condition: Condvar,
}

impl LocalProcessBarrier {
    /// Creates a new [`LocalProcessBarrier`] for `participant_count` participants.
    #[inline]
    pub fn new(participant_count: usize) -> Self {
        Self { participant_count, rounds: Mutex::new(HashMap::new()), condition: Condvar::new() }
    }

    /// Returns the number of participants that every round waits for.
    #[inline]
    pub fn participant_count(&self) -> usize {
        self.participant_count
    }
}

impl Debug for LocalProcessBarrier {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalProcessBarrier")
            .field("participant_count", &self.participant_count)
            .finish()
    }
}

impl ProcessBarrier for LocalProcessBarrier {
    fn wait(&self, key: &str, timeout: Option<Duration>) -> Result<(), CheckpointError> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let mut rounds = self.rounds.lock().expect("the process barrier lock was poisoned");
        let round = rounds.entry(key.to_string()).or_default();
        let generation = round.generation;
        round.arrived += 1;
        if round.arrived >= self.participant_count {
            round.arrived = 0;
            round.generation += 1;
            self.condition.notify_all();
            return Ok(());
        }
        loop {
            let remaining = match deadline {
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) if !remaining.is_zero() => Some(remaining),
                    _ => None,
                },
                None => Some(Duration::MAX),
            };
            let round = rounds.get_mut(key).unwrap();
            if round.generation != generation {
                return Ok(());
            }
            let Some(remaining) = remaining else {
                // The participant leaves the round, so that a later attempt with the same key starts from a
                // consistent count.
                round.arrived -= 1;
                return Err(CheckpointError::Barrier {
                    key: key.to_string(),
                    message: format!(
                        "timed out after {} of {} participants arrived",
                        round.arrived + 1,
                        self.participant_count,
                    ),
                });
            };
            rounds = match deadline {
                Some(_) => {
                    self.condition.wait_timeout(rounds, remaining).expect("the process barrier lock was poisoned").0
                }
                None => self.condition.wait(rounds).expect("the process barrier lock was poisoned"),
            };
        }
    }
}

/// Role of one process among the processes that save checkpoints together, which mirrors the `MultiprocessingOptions`
/// of Orbax. The default describes a single process that is its own primary process and needs no barrier.
#[derive(Clone)]
pub struct MultiprocessingOptions {
    /// Index of this process.
    process_index: usize,

    /// Number of processes.
    process_count: usize,

    /// Index of the primary process, or [`None`] if every process acts as the primary process.
    primary_process: Option<usize>,

    /// Barrier across the processes, which is required when there is more than one process.
    barrier: Option<Arc<dyn ProcessBarrier>>,

    /// Timeout of every barrier, if any.
    barrier_timeout: Option<Duration>,
}

impl MultiprocessingOptions {
    /// Creates new [`MultiprocessingOptions`] for the process `process_index` of `process_count` processes, with
    /// process 0 as the primary process, without a barrier, and without a barrier timeout.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `process_index` is not smaller than `process_count`.
    pub fn new(process_index: usize, process_count: usize) -> Result<Self, CheckpointError> {
        if process_index >= process_count {
            return Err(CheckpointError::InvalidArgument {
                message: format!("process index {process_index} is not smaller than the process count {process_count}"),
            });
        }
        Ok(Self { process_index, process_count, primary_process: Some(0), barrier: None, barrier_timeout: None })
    }

    /// Returns these options with `primary_process` as the primary process, or with every process acting as the
    /// primary process when it is [`None`]. Without a primary process every process creates, finalizes, and deletes
    /// checkpoint directories, so every process must then save to a directory of its own (e.g., on its local disk).
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `primary_process` is not smaller than the process count.
    pub fn with_primary_process(mut self, primary_process: Option<usize>) -> Result<Self, CheckpointError> {
        if let Some(primary_process) = primary_process
            && primary_process >= self.process_count
        {
            return Err(CheckpointError::InvalidArgument {
                message: format!(
                    "primary process {primary_process} is not smaller than the process count {}",
                    self.process_count,
                ),
            });
        }
        self.primary_process = primary_process;
        Ok(self)
    }

    /// Returns these options with `barrier` synchronizing the processes.
    #[inline]
    pub fn with_barrier(mut self, barrier: Arc<dyn ProcessBarrier>) -> Self {
        self.barrier = Some(barrier);
        self
    }

    /// Returns these options with `barrier_timeout` as the timeout of every barrier.
    #[inline]
    pub fn with_barrier_timeout(mut self, barrier_timeout: Option<Duration>) -> Self {
        self.barrier_timeout = barrier_timeout;
        self
    }

    /// Returns the index of this process.
    #[inline]
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Returns the number of processes.
    #[inline]
    pub fn process_count(&self) -> usize {
        self.process_count
    }

    /// Returns the index of the primary process, or [`None`] if every process acts as the primary process.
    #[inline]
    pub fn primary_process(&self) -> Option<usize> {
        self.primary_process
    }

    /// Returns the barrier across the processes, if any.
    #[inline]
    pub fn barrier(&self) -> Option<&Arc<dyn ProcessBarrier>> {
        self.barrier.as_ref()
    }

    /// Returns the timeout of every barrier, if any.
    #[inline]
    pub fn barrier_timeout(&self) -> Option<Duration> {
        self.barrier_timeout
    }

    /// Returns `true` if this process is the primary process or if every process acts as the primary process.
    #[inline]
    pub fn is_primary(&self) -> bool {
        self.primary_process.is_none_or(|primary_process| primary_process == self.process_index)
    }

    /// Waits at the barrier with `key` if there is a barrier, and returns immediately otherwise.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ProcessBarrier::wait`], and [`CheckpointError::InvalidArgument`] if several processes
    /// share a primary process without a barrier, because the other processes could then not tell when the primary
    /// process has finalized the checkpoint.
    pub(crate) fn synchronize(&self, key: &str) -> Result<(), CheckpointError> {
        match &self.barrier {
            Some(barrier) => barrier.wait(key, self.barrier_timeout),
            None if self.process_count > 1 && self.primary_process.is_some() => Err(CheckpointError::InvalidArgument {
                message: format!(
                    "{} processes that share a primary process need a barrier to save a checkpoint together",
                    self.process_count,
                ),
            }),
            None => Ok(()),
        }
    }
}

impl Default for MultiprocessingOptions {
    fn default() -> Self {
        Self { process_index: 0, process_count: 1, primary_process: Some(0), barrier: None, barrier_timeout: None }
    }
}

impl Debug for MultiprocessingOptions {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MultiprocessingOptions")
            .field("process_index", &self.process_index)
            .field("process_count", &self.process_count)
            .field("primary_process", &self.primary_process)
            .field("barrier", &self.barrier)
            .field("barrier_timeout", &self.barrier_timeout)
            .finish()
    }
}

/// State of the current round of one key of a [`LocalProcessBarrier`].
#[derive(Default)]
struct LocalBarrierRound {
    /// Number of participants that have arrived in the current round.
    arrived: usize,

    /// Number of completed rounds.
    generation: u64,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_local_process_barrier() {
        let barrier = Arc::new(LocalProcessBarrier::new(3));
        assert_eq!(barrier.participant_count(), 3);

        // Every participant returns once all of them have arrived, and the key can be reused for another round.
        for _ in 0..2 {
            let threads = (0..3)
                .map(|_| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || barrier.wait("round", Some(Duration::from_secs(60))))
                })
                .collect::<Vec<_>>();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), Ok(()));
            }
        }
    }

    #[test]
    fn test_local_process_barrier_timeout() {
        // Two of three participants time out instead of waiting forever.
        let barrier = Arc::new(LocalProcessBarrier::new(3));
        let threads = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || barrier.wait("partial", Some(Duration::from_millis(50))))
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert!(matches!(
                thread.join().unwrap(),
                Err(CheckpointError::Barrier { key, message }) if key == "partial" && message.starts_with("timed out"),
            ));
        }
    }

    #[test]
    fn test_multiprocessing_options() {
        let options = MultiprocessingOptions::default();
        assert_eq!((options.process_index(), options.process_count(), options.primary_process()), (0, 1, Some(0)));
        assert!(options.is_primary());
        assert!(options.barrier().is_none());
        assert_eq!(options.synchronize("key"), Ok(()));

        let barrier: Arc<dyn ProcessBarrier> = Arc::new(LocalProcessBarrier::new(1));
        let options = MultiprocessingOptions::new(1, 3)
            .unwrap()
            .with_barrier(barrier)
            .with_barrier_timeout(Some(Duration::from_secs(1)));
        assert!(!options.is_primary());
        assert_eq!(options.barrier_timeout(), Some(Duration::from_secs(1)));
        assert_eq!(options.synchronize("key"), Ok(()));
        assert!(options.clone().with_primary_process(Some(1)).unwrap().is_primary());
        assert!(options.clone().with_primary_process(None).unwrap().is_primary());

        // Several processes that share a primary process cannot save together without a barrier.
        let options_without_barrier = MultiprocessingOptions::new(1, 3).unwrap();
        assert_eq!(
            options_without_barrier.synchronize("key"),
            Err(CheckpointError::InvalidArgument {
                message: "3 processes that share a primary process need a barrier to save a checkpoint together"
                    .to_string(),
            }),
        );
        assert_eq!(options_without_barrier.with_primary_process(None).unwrap().synchronize("key"), Ok(()));

        assert_eq!(
            MultiprocessingOptions::new(3, 3).map(|_| ()),
            Err(CheckpointError::InvalidArgument {
                message: "process index 3 is not smaller than the process count 3".to_string(),
            }),
        );
        assert_eq!(
            options.with_primary_process(Some(5)).map(|_| ()),
            Err(CheckpointError::InvalidArgument {
                message: "primary process 5 is not smaller than the process count 3".to_string(),
            }),
        );
    }
}
