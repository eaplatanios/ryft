//! Communication checks for the CTA operations emitted by GPU lowering.
//!
//! The lowerer records each thread's actual memory accesses, `cp.async` copies, committed groups, waits, and barrier
//! sites while emitting the corresponding typed NVGPU/GPU operations. Storage owners are canonical [`ValueId`]s;
//! reference aliases must be resolved to the same owner before recording. Ordinary array arithmetic retains the core
//! interpreter's semantics. This module models only the target communication and does not interpret a second value IR.
//! Initialization is established by portable kernel qualification and the lowerer's complete transport writes; this
//! communication model detects conflicting or unfinished accesses, but does not independently prove initialized bytes.
//!
//! The baseline uses full-CTA barriers and drains all committed copy groups at each wait. Copy completion makes data
//! available to the issuing thread; another thread needs a CTA barrier before consuming it. Relative copy offsets
//! and widths are checked here; native admission must additionally prove global/shared address spaces and base
//! alignment. See the
//! [NVGPU copy contract](https://mlir.llvm.org/docs/Dialects/NVGPU/#nvgpudevice_async_copy-nvgpudeviceasynccopyop)
//! and [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::ops::Range;

use ryft_core::{
    ArrayAddressing, ArrayReferenceView, ArrayReferenceViewPath, ArrayType, InstructionId, ProgramError,
    ReferenceAccessMode, ValueId,
};
use thiserror::Error;

/// Invalid communication geometry, unsynchronized access, or incomplete CTA progress.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum SynchronizationError {
    /// The target or recorded trace exceeds the bounded baseline simulator.
    #[error("cta synchronization exceeds the supported {resource} limit {maximum}")]
    Limit { resource: &'static str, maximum: usize },

    /// An event names a thread outside the declared CTA.
    #[error("cta thread {thread} is outside the declared {participants} participants")]
    Thread { thread: u32, participants: u32 },

    /// A memory event does not name an allocation in the lowering plan.
    #[error("cta memory event names unknown storage {value:?}")]
    Storage { value: ValueId },

    /// The baseline requires statically resolved canonical views.
    #[error("cta storage {value:?} has an unsupported static selection")]
    Selection { value: ValueId },

    /// Canonical array geometry rejected the access.
    #[error(transparent)]
    Geometry(#[from] ProgramError),

    /// A copy does not satisfy the exact contiguous aligned PTX transfer-width contract.
    #[error("cta async copy requires matching contiguous aligned 4-, 8-, or 16-byte selections")]
    CopyAlignment,

    /// Only ordinary reads and writes are part of this baseline communication plan.
    #[error("cta synchronization does not support `{mode}` memory accesses")]
    AccessMode { mode: ReferenceAccessMode },

    /// An access touches a still-reserved copy source or destination.
    #[error("cta thread {thread} at {instruction} accesses storage {value:?} before its async copy is waited")]
    PendingAccess { thread: u32, instruction: InstructionId, value: ValueId },

    /// Two threads access overlapping storage without a separating CTA barrier.
    #[error("cta threads {first_thread} at {first} and {second_thread} at {second} race on storage {value:?}")]
    Race { value: ValueId, first_thread: u32, first: InstructionId, second_thread: u32, second: InstructionId },

    /// Some participants cannot reach the same next barrier site.
    #[error("cta barrier deadlock among participants {waiting:?}")]
    Deadlock { waiting: Vec<(u32, usize, InstructionId)> },

    /// A full-CTA barrier cannot substitute for completion of outstanding asynchronous copies.
    #[error("cta thread {thread} reaches barrier {site} before its async copies are waited")]
    PendingBarrier { thread: u32, site: usize },

    /// A thread exits with transfers that have not passed through a matching wait.
    #[error("cta thread {thread} exits with {copies} uncompleted async copies")]
    PendingExit { thread: u32, copies: usize },
}

/// One communication operation emitted for a particular CTA thread.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SynchronizationEvent {
    /// Read or write through a canonical static view of a lowering-owned buffer.
    Access { value: ValueId, view: ArrayReferenceView, mode: ReferenceAccessMode },

    /// One native `cp.async` transfer, pending until its committed group is waited.
    AsyncCopy { source: (ValueId, ArrayReferenceView), destination: (ValueId, ArrayReferenceView) },

    /// Commits the issuing thread's preceding uncommitted transfers, including an empty group.
    CommitGroup,

    /// Waits for every committed group of the issuing thread; uncommitted transfers remain pending.
    WaitGroup,

    /// Full-CTA rendezvous at a native barrier site, shared by every participating thread.
    Barrier { site: usize },
}

/// Recorded communication for one logical CTA. Lowering must record every executed memory access and normalize
/// aliases to the same storage owner. Uniform control-flow paths must be checked separately; loop reuse requires
/// checking the backedge as well as the first iteration. The plan is excluded from portable semantic identity.
#[derive(Clone, Debug)]
pub struct CtaSynchronization {
    /// Declared target CTA membership.
    participants: NonZeroU32,

    /// Canonical types of all global and shared transport buffers.
    storage: HashMap<ValueId, ArrayType>,

    /// Actual event order for each thread.
    events: Vec<Vec<(InstructionId, SynchronizationEvent)>>,

    /// Total recorded event count, bounded before growth.
    event_count: usize,
}

/// One checked access, retaining physical ranges derived by canonical array addressing.
#[derive(Clone, Debug)]
struct Access {
    /// Canonical owner shared by aliases.
    value: ValueId,

    /// Physical byte ranges selected within that owner.
    ranges: Vec<Range<usize>>,

    /// Whether the access changes selected memory.
    writes: bool,
}

/// One pending native copy whose source and destination remain reserved.
#[derive(Clone, Debug)]
struct PendingCopy {
    /// Source bytes that cannot be overwritten before completion.
    source: Access,

    /// Destination bytes that cannot be accessed before completion.
    destination: Access,
}

impl CtaSynchronization {
    /// Declares a CTA of at most 1024 threads and statically addressable storage. Native allocation base alignment
    /// remains the lowerer's responsibility; this plan validates canonical relative byte ranges.
    pub fn new(participants: NonZeroU32, storage: HashMap<ValueId, ArrayType>) -> Result<Self, SynchronizationError> {
        if participants.get() > 1024 {
            return Err(SynchronizationError::Limit { resource: "participant", maximum: 1024 });
        }
        for r#type in storage.values() {
            ArrayAddressing::new(r#type.clone())?;
        }
        Ok(Self { participants, storage, events: vec![vec![]; participants.get() as usize], event_count: 0 })
    }

    /// Returns the declared full-CTA participant count.
    pub fn participants(&self) -> NonZeroU32 {
        self.participants
    }

    /// Records one emitted operation, checking its geometry before retaining it. At most 65,536 events and 4,096
    /// canonical contiguous ranges per selection are admitted, keeping this deterministic specification bounded.
    pub fn record(
        &mut self,
        thread: u32,
        instruction: InstructionId,
        event: SynchronizationEvent,
    ) -> Result<(), SynchronizationError> {
        if thread >= self.participants.get() {
            return Err(SynchronizationError::Thread { thread, participants: self.participants.get() });
        }
        if self.event_count == 65_536 {
            return Err(SynchronizationError::Limit { resource: "event", maximum: 65_536 });
        }
        match &event {
            SynchronizationEvent::Access { value, view, mode } => {
                if !matches!(mode, ReferenceAccessMode::Read | ReferenceAccessMode::Write) {
                    return Err(SynchronizationError::AccessMode { mode: *mode });
                }
                self.access(*value, view, *mode == ReferenceAccessMode::Write)?;
            }
            SynchronizationEvent::AsyncCopy { source, destination } => {
                let source = self.access(source.0, &source.1, false)?;
                let destination = self.access(destination.0, &destination.1, true)?;
                if source.ranges.len() != 1 || destination.ranges.len() != 1 {
                    return Err(SynchronizationError::CopyAlignment);
                }
                let source = &source.ranges[0];
                let destination = &destination.ranges[0];
                let width = source.len();
                if !matches!(width, 4 | 8 | 16)
                    || width != destination.len()
                    || source.start % width != 0
                    || destination.start % width != 0
                {
                    return Err(SynchronizationError::CopyAlignment);
                }
            }
            _ => {}
        }
        self.events[thread as usize].push((instruction, event));
        self.event_count += 1;
        Ok(())
    }

    /// Checks complete communication in deterministic round-robin order and returns the successful event order.
    /// Race checks compare every conflicting access between barriers regardless of this chosen traversal order.
    /// A wait completes issuing-thread copies only; full-CTA release establishes cross-thread visibility.
    pub fn simulate(&self) -> Result<Vec<(u32, InstructionId)>, SynchronizationError> {
        let count = self.participants.get() as usize;
        let mut positions = vec![0; count];
        let mut uncommitted: Vec<Vec<PendingCopy>> = vec![vec![]; count];
        let mut groups: Vec<Vec<Vec<PendingCopy>>> = vec![vec![]; count];
        let mut history: Vec<(u32, InstructionId, Access)> = Vec::new();
        let mut trace = Vec::with_capacity(self.event_count);
        while trace.len() < self.event_count {
            let mut progressed = false;
            for thread in 0..count {
                let Some((instruction, event)) = self.events[thread].get(positions[thread]) else { continue };
                let accesses = match event {
                    SynchronizationEvent::Barrier { .. } => continue,
                    SynchronizationEvent::CommitGroup => {
                        groups[thread].push(std::mem::take(&mut uncommitted[thread]));
                        vec![]
                    }
                    SynchronizationEvent::WaitGroup => {
                        groups[thread].clear();
                        vec![]
                    }
                    SynchronizationEvent::Access { value, view, mode } => {
                        vec![self.access(*value, view, *mode == ReferenceAccessMode::Write)?]
                    }
                    SynchronizationEvent::AsyncCopy { source, destination } => {
                        vec![
                            self.access(source.0, &source.1, false)?,
                            self.access(destination.0, &destination.1, true)?,
                        ]
                    }
                };
                for access in &accesses {
                    for copy in uncommitted.iter().flatten().chain(groups.iter().flatten().flatten()) {
                        if access.overlaps(&copy.destination) || (access.writes && access.overlaps(&copy.source)) {
                            return Err(SynchronizationError::PendingAccess {
                                thread: thread as u32,
                                instruction: *instruction,
                                value: access.value,
                            });
                        }
                    }
                    for (previous_thread, previous, previous_access) in &history {
                        if *previous_thread != thread as u32
                            && (access.writes || previous_access.writes)
                            && access.overlaps(previous_access)
                        {
                            return Err(SynchronizationError::Race {
                                value: access.value,
                                first_thread: *previous_thread,
                                first: *previous,
                                second_thread: thread as u32,
                                second: *instruction,
                            });
                        }
                    }
                }
                if matches!(event, SynchronizationEvent::AsyncCopy { .. }) {
                    if accesses[0].overlaps(&accesses[1]) {
                        return Err(SynchronizationError::PendingAccess {
                            thread: thread as u32,
                            instruction: *instruction,
                            value: accesses[1].value,
                        });
                    }
                    uncommitted[thread]
                        .push(PendingCopy { source: accesses[0].clone(), destination: accesses[1].clone() });
                }
                history.extend(accesses.into_iter().map(|access| (thread as u32, *instruction, access)));
                positions[thread] += 1;
                trace.push((thread as u32, *instruction));
                progressed = true;
            }
            let waiting = (0..count)
                .filter_map(|thread| {
                    let (instruction, SynchronizationEvent::Barrier { site }) =
                        self.events[thread].get(positions[thread])?
                    else {
                        return None;
                    };
                    Some((thread as u32, *site, *instruction))
                })
                .collect::<Vec<_>>();
            if waiting.len() == count && waiting.iter().all(|(_, site, _)| *site == waiting[0].1) {
                for (thread, site, instruction) in &waiting {
                    let position = *thread as usize;
                    if !uncommitted[position].is_empty() || groups[position].iter().any(|group| !group.is_empty()) {
                        return Err(SynchronizationError::PendingBarrier { thread: *thread, site: *site });
                    }
                    positions[position] += 1;
                    trace.push((*thread, *instruction));
                }
                history.clear();
                progressed = true;
            }
            if !progressed {
                return Err(SynchronizationError::Deadlock { waiting });
            }
        }
        for thread in 0..count {
            let copies = uncommitted[thread].len() + groups[thread].iter().map(Vec::len).sum::<usize>();
            if copies != 0 {
                return Err(SynchronizationError::PendingExit { thread: thread as u32, copies });
            }
        }
        Ok(trace)
    }

    /// Resolves a static view through the canonical root-selection and addressing implementations.
    fn access(&self, value: ValueId, view: &ArrayReferenceView, writes: bool) -> Result<Access, SynchronizationError> {
        let r#type = self.storage.get(&value).ok_or(SynchronizationError::Storage { value })?;
        let path = ArrayReferenceViewPath::root().with_view(view.clone());
        let Some(ArrayReferenceView::Slice { axes }) = path.root_slice(r#type) else {
            return Err(SynchronizationError::Selection { value });
        };
        let addressing = ArrayAddressing::new(r#type.clone())?;
        let mut ranges = Vec::new();
        for range in addressing.ranges(&axes)? {
            if ranges.len() == 4096 {
                return Err(SynchronizationError::Limit { resource: "selection range", maximum: 4096 });
            }
            ranges.push(range.bytes());
        }
        Ok(Access { value, ranges, writes })
    }
}

impl Access {
    /// Returns whether two canonical selections address any common physical bytes of the same storage owner.
    fn overlaps(&self, other: &Self) -> bool {
        self.value == other.value
            && self
                .ranges
                .iter()
                .any(|left| other.ranges.iter().any(|right| left.start < right.end && right.start < left.end))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArraySliceAxis, AtomId, DataType, RegionId};

    use super::*;

    /// Identifies a source instruction in the fixture's single canonical region.
    fn instruction(index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(0), index)
    }

    /// Identifies a lowering-owned buffer in the same canonical region.
    fn value(index: usize) -> ValueId {
        ValueId::new(RegionId::new(0), AtomId::new(index))
    }

    /// Selects an exact byte interval in the fixture's byte-valued buffers.
    fn view(start: usize, size: usize) -> ArrayReferenceView {
        ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(start, size, 1)] }
    }

    /// Declares one global source and one shared destination, each with 16 physical bytes.
    fn plan(participants: u32) -> CtaSynchronization {
        CtaSynchronization::new(
            NonZeroU32::new(participants).unwrap(),
            HashMap::from([
                (value(0), ArrayType::new_static(DataType::U8, [16])),
                (value(1), ArrayType::new_static(DataType::U8, [16])),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn test_cta_synchronization_new() {
        assert_eq!(plan(2).participants().get(), 2);
        assert_eq!(plan(1).simulate(), Ok(vec![]));
        assert_eq!(
            CtaSynchronization::new(NonZeroU32::new(1025).unwrap(), HashMap::new()).unwrap_err(),
            SynchronizationError::Limit { resource: "participant", maximum: 1024 },
        );
    }

    #[test]
    fn test_cta_synchronization_participants() {
        assert_eq!(plan(128).participants(), NonZeroU32::new(128).unwrap());
    }

    #[test]
    fn test_cta_synchronization_record() {
        let mut plan = plan(1);
        assert_eq!(plan.record(0, instruction(0), SynchronizationEvent::CommitGroup), Ok(()));
        let error = plan.record(1, instruction(1), SynchronizationEvent::WaitGroup).unwrap_err();
        assert_eq!(error, SynchronizationError::Thread { thread: 1, participants: 1 });
        assert_eq!(error.to_string(), "cta thread 1 is outside the declared 1 participants");
        let error = plan
            .record(
                0,
                instruction(1),
                SynchronizationEvent::Access { value: value(2), view: view(0, 4), mode: ReferenceAccessMode::Read },
            )
            .unwrap_err();
        assert_eq!(error, SynchronizationError::Storage { value: value(2) });
        let error = plan
            .record(
                0,
                instruction(1),
                SynchronizationEvent::Access {
                    value: value(0),
                    view: view(0, 4),
                    mode: ReferenceAccessMode::AtomicAccumulate,
                },
            )
            .unwrap_err();
        assert_eq!(error, SynchronizationError::AccessMode { mode: ReferenceAccessMode::AtomicAccumulate });
    }

    #[test]
    fn test_cta_synchronization_record_copy_alignment() {
        for width in [4, 8, 16] {
            assert_eq!(
                plan(1).record(
                    0,
                    instruction(0),
                    SynchronizationEvent::AsyncCopy {
                        source: (value(0), view(0, width)),
                        destination: (value(1), view(0, width)),
                    }
                ),
                Ok(())
            );
        }
        for (start, source_width, destination_width) in [(1, 4, 4), (0, 3, 3), (0, 4, 8)] {
            let error = plan(1)
                .record(
                    0,
                    instruction(0),
                    SynchronizationEvent::AsyncCopy {
                        source: (value(0), view(start, source_width)),
                        destination: (value(1), view(0, destination_width)),
                    },
                )
                .unwrap_err();
            assert_eq!(error, SynchronizationError::CopyAlignment);
            assert_eq!(
                error.to_string(),
                "cta async copy requires matching contiguous aligned 4-, 8-, or 16-byte selections"
            );
        }
    }

    #[test]
    fn test_cta_synchronization_simulate() {
        let mut plan = plan(2);
        for thread in 0..2 {
            plan.record(
                thread,
                instruction(0),
                SynchronizationEvent::AsyncCopy {
                    source: (value(0), view(thread as usize * 4, 4)),
                    destination: (value(1), view(thread as usize * 4, 4)),
                },
            )
            .unwrap();
            plan.record(thread, instruction(1), SynchronizationEvent::CommitGroup).unwrap();
            plan.record(thread, instruction(2), SynchronizationEvent::WaitGroup).unwrap();
            plan.record(thread, instruction(3), SynchronizationEvent::Barrier { site: 0 }).unwrap();
            plan.record(
                thread,
                instruction(4),
                SynchronizationEvent::Access { value: value(1), view: view(0, 8), mode: ReferenceAccessMode::Read },
            )
            .unwrap();
        }
        let expected = (0..5).flat_map(|index| [(0, instruction(index)), (1, instruction(index))]).collect::<Vec<_>>();
        assert_eq!(plan.simulate(), Ok(expected.clone()));
        assert_eq!(plan.simulate(), Ok(expected));
    }

    #[test]
    fn test_cta_synchronization_simulate_pending_access() {
        for (owner, mode) in [(value(0), ReferenceAccessMode::Write), (value(1), ReferenceAccessMode::Read)] {
            let mut plan = plan(1);
            plan.record(
                0,
                instruction(0),
                SynchronizationEvent::AsyncCopy { source: (value(0), view(0, 4)), destination: (value(1), view(0, 4)) },
            )
            .unwrap();
            plan.record(0, instruction(1), SynchronizationEvent::Access { value: owner, view: view(0, 4), mode })
                .unwrap();
            assert_eq!(
                plan.simulate(),
                Err(SynchronizationError::PendingAccess { thread: 0, instruction: instruction(1), value: owner })
            );
        }
    }

    #[test]
    fn test_cta_synchronization_simulate_wait_requires_commit_and_barrier() {
        let mut uncommitted = plan(1);
        uncommitted
            .record(
                0,
                instruction(0),
                SynchronizationEvent::AsyncCopy { source: (value(0), view(0, 4)), destination: (value(1), view(0, 4)) },
            )
            .unwrap();
        uncommitted.record(0, instruction(1), SynchronizationEvent::WaitGroup).unwrap();
        assert_eq!(uncommitted.simulate(), Err(SynchronizationError::PendingExit { thread: 0, copies: 1 }));

        let mut unsynchronized = plan(2);
        unsynchronized
            .record(
                0,
                instruction(0),
                SynchronizationEvent::AsyncCopy { source: (value(0), view(0, 4)), destination: (value(1), view(0, 4)) },
            )
            .unwrap();
        unsynchronized.record(0, instruction(1), SynchronizationEvent::CommitGroup).unwrap();
        unsynchronized.record(0, instruction(2), SynchronizationEvent::WaitGroup).unwrap();
        unsynchronized.record(1, instruction(3), SynchronizationEvent::CommitGroup).unwrap();
        unsynchronized.record(1, instruction(4), SynchronizationEvent::CommitGroup).unwrap();
        unsynchronized
            .record(
                1,
                instruction(5),
                SynchronizationEvent::Access { value: value(1), view: view(0, 4), mode: ReferenceAccessMode::Read },
            )
            .unwrap();
        assert_eq!(
            unsynchronized.simulate(),
            Err(SynchronizationError::Race {
                value: value(1),
                first_thread: 0,
                first: instruction(0),
                second_thread: 1,
                second: instruction(5),
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_barrier_does_not_complete_copies() {
        let mut plan = plan(2);
        plan.record(
            0,
            instruction(0),
            SynchronizationEvent::AsyncCopy { source: (value(0), view(0, 4)), destination: (value(1), view(0, 4)) },
        )
        .unwrap();
        plan.record(0, instruction(1), SynchronizationEvent::Barrier { site: 0 }).unwrap();
        plan.record(1, instruction(1), SynchronizationEvent::Barrier { site: 0 }).unwrap();
        assert_eq!(plan.simulate(), Err(SynchronizationError::PendingBarrier { thread: 0, site: 0 }));
    }

    #[test]
    fn test_cta_synchronization_simulate_deadlock() {
        let mut divergent = plan(2);
        divergent.record(0, instruction(0), SynchronizationEvent::Barrier { site: 0 }).unwrap();
        divergent.record(1, instruction(1), SynchronizationEvent::Barrier { site: 1 }).unwrap();
        assert_eq!(
            divergent.simulate(),
            Err(SynchronizationError::Deadlock { waiting: vec![(0, 0, instruction(0)), (1, 1, instruction(1))] })
        );
        let mut missing = plan(2);
        missing.record(0, instruction(0), SynchronizationEvent::Barrier { site: 0 }).unwrap();
        assert_eq!(missing.simulate(), Err(SynchronizationError::Deadlock { waiting: vec![(0, 0, instruction(0))] }));
    }
}
