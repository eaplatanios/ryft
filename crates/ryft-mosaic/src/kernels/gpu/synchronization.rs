//! Communication checks for the CTA operations emitted by GPU lowering.
//!
//! The lowerer records each thread's actual memory accesses, `cp.async` copies, committed groups, waits, and barrier
//! sites while emitting the corresponding typed MLIR operations. Storage owners are canonical [`ValueId`]s;
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
//!
//! TMA transaction barriers use the canonical copy-token owner and count exact issued transfer bytes. A matching
//! generation wait acquires only its transfers for the waiting thread; a subsequent full-CTA barrier publishes that
//! completion to other threads. It cannot complete an unwaited generation. Reuse and invalidation require completion
//! acquired by the declared CTA. This first single-CTA protocol excludes overlapping unrelated cooperative work.
//! Native parity is derived from the generation, but stale generations remain distinguishable in this model.
//! Native address-space, descriptor, swizzle, and base-alignment eligibility remains the lowerer's responsibility.
//!
//! WGMMA events require exactly one complete 128-thread warpgroup. Every participant must reach the same fence,
//! issue, commit, and wait instruction. Register fences and asynchronous-proxy fences establish distinct prerequisites;
//! waits release only completed FIFO matrix groups. Outstanding operands remain reserved across partial waits and
//! cannot be made available by an unrelated copy wait or CTA barrier. Canonical operand owners identify the lowerer's
//! private packed transports; descriptor geometry and numerical accumulator contents are checked by lowering.
//!
//! Tensor-memory allocation and release rendezvous the first warp. Matrix issue/commit/wait belong to the elected
//! thread; a CTA barrier publishes allocation and completed matrix contents. Allocation never initializes contents,
//! accumulation requires a prior completed overwrite, and every completion token is committed and waited exactly
//! once. Loads rendezvous all CTA threads at the emitted instruction with their checked per-thread selections.
//! Native packing/proxy fences and tensor-memory load fences remain inseparable steps in the owning lowerer.
//!
//! Two-CTA replay keeps each CTA's storage and pending resources separate while retaining canonical value identities.
//! Native cluster rendezvous follow local completion barriers. Distributed row copies are full-cluster collectives
//! between those rendezvous; exact canonical byte selections detect cross-CTA conflicts. A tensor instruction spanning
//! two CTAs is recorded only in its elected CTA and applies its actual collective effects to both allocation states.
//! Each CTA acquires its own completion barrier, and matrix loads may access only that CTA's logical row half. Scale
//! copies initialize the complete replicated scale allocation in both CTAs. No peer issue or completion event is
//! synthesized.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU32;
use std::ops::Range;

use ryft_core::{
    ArrayAddressing, ArrayReferenceTransform, ArrayReferenceTransformPath, ArraySliceAxis, ArrayType, InstructionId,
    ProgramError, ReferenceAccessMode, ValueId,
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

    /// The baseline requires statically resolved canonical transforms.
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

    /// Two CTAs cannot reach the same cluster rendezvous with every declared thread.
    #[error("cluster barrier deadlock among participants {waiting:?}")]
    ClusterDeadlock { waiting: Vec<(u32, u32, usize, InstructionId)> },

    /// A distributed copy is not the checked full-cluster transport surrounded by publication barriers.
    #[error("cluster copy at {instruction} {reason}")]
    ClusterCopy { instruction: InstructionId, reason: &'static str },

    /// Simultaneous distributed accesses overlap one physical CTA's shared allocation.
    #[error(
        "cluster cta {first_block} thread {first_thread} and cta {second_block} thread {second_thread} \
         race on {value:?}"
    )]
    ClusterRace { value: ValueId, first_block: u32, first_thread: u32, second_block: u32, second_thread: u32 },

    /// A full-CTA barrier cannot substitute for completion of outstanding asynchronous copies.
    #[error("cta thread {thread} reaches barrier {site} before its async copies are waited")]
    PendingBarrier { thread: u32, site: usize },

    /// A collectively emitted WGMMA operation violates membership, ordering, or completion requirements.
    #[error("cta warpgroup at {instruction} {reason}")]
    Wgmma { instruction: InstructionId, reason: &'static str },

    /// A tensor-memory operation violates collective membership, initialization, completion, or lifetime.
    #[error("cta tensor-memory resource {value:?} at {instruction} {reason}")]
    TensorMemory { value: ValueId, instruction: InstructionId, reason: &'static str },

    /// A transaction-barrier operation violates its initialization, generation, or completion contract.
    #[error("cta transaction barrier {barrier:?} at {instruction} {reason}")]
    TransactionBarrier { barrier: ValueId, instruction: InstructionId, reason: &'static str },

    /// No participant can complete a transaction-barrier wait with the recorded arrivals and transfers.
    #[error("cta transaction barrier deadlock among participants {waiting:?}")]
    TransactionDeadlock { waiting: Vec<(u32, ValueId, u64, InstructionId)> },

    /// A transaction barrier remains initialized when its recorded lifetime ends.
    #[error("cta transaction barrier {barrier:?} exits without invalidation")]
    TransactionExit { barrier: ValueId },

    /// A thread exits with transfers that have not passed through a matching wait.
    #[error("cta thread {thread} exits with {copies} uncompleted async copies")]
    PendingExit { thread: u32, copies: usize },
}

/// One communication operation emitted for a particular CTA thread.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SynchronizationEvent {
    /// Read or write through a canonical static transform of a lowering-owned buffer.
    Access { value: ValueId, transform: ArrayReferenceTransform, mode: ReferenceAccessMode },

    /// One native `cp.async` transfer, pending until its committed group is waited.
    AsyncCopy { source: (ValueId, ArrayReferenceTransform), destination: (ValueId, ArrayReferenceTransform) },

    /// Commits the issuing thread's preceding uncommitted transfers, including an empty group.
    CommitGroup,

    /// Waits for every committed group of the issuing thread; uncommitted transfers remain pending.
    WaitGroup,

    /// Publishes preceding generic shared writes to the asynchronous proxy in every warpgroup thread.
    AsyncProxyFence,

    /// Orders the actual initialized accumulator registers before asynchronous WGMMA accesses.
    WgmmaFence { accumulator: ValueId },

    /// Issues one collective matrix operation using the canonical operands and accumulator owner.
    WgmmaIssue {
        accumulator: ValueId,
        left: (ValueId, ArrayReferenceTransform),
        right: (ValueId, ArrayReferenceTransform),
    },

    /// Commits preceding collectively issued matrix operations as one FIFO group.
    WgmmaCommit,

    /// Completes oldest committed groups until at most `remaining` groups are outstanding.
    WgmmaWait { remaining: usize },

    /// Allocates one canonical tensor-memory root collectively in lanes zero through 31.
    AllocateTensorMemory { value: ValueId },

    /// Issues one logical matrix operation and its canonical completion token from the elected thread.
    IssueTensorMemory {
        token: ValueId,
        destination: ValueId,
        left: (ValueId, ArrayReferenceTransform),
        right: (ValueId, ArrayReferenceTransform),
        accumulate: bool,
        scales: Vec<ValueId>,
    },

    /// Copies shared scale storage into tensor memory under a canonical completion token.
    CopyTensorMemory { token: ValueId, source: (ValueId, ArrayReferenceTransform), destination: ValueId },

    /// Issues one two-CTA matrix instruction from CTA zero, retaining each CTA's actual operand selections.
    IssueTensorMemoryCluster {
        token: ValueId,
        destination: ValueId,
        left: [(ValueId, ArrayReferenceTransform); 2],
        right: [(ValueId, ArrayReferenceTransform); 2],
        accumulate: bool,
        scales: Vec<ValueId>,
    },

    /// Copies the elected CTA's shared scales into both CTAs' tensor-memory allocations.
    CopyTensorMemoryCluster { token: ValueId, source: (ValueId, ArrayReferenceTransform), destination: ValueId },

    /// Multicasts completion of one two-CTA instruction to both local hardware barriers.
    CommitTensorMemoryCluster { token: ValueId },

    /// Attaches the elected thread's tensor-memory completion token to its hardware barrier exactly once.
    CommitTensorMemory { token: ValueId },

    /// Acquires a committed tensor-memory completion in the elected thread before CTA publication.
    WaitTensorMemory { token: ValueId },

    /// Reads the actual thread-owned tensor-memory selection as part of one uniform full-CTA load.
    LoadTensorMemory { value: ValueId, transform: ArrayReferenceTransform },

    /// Releases a completed allocation collectively in lanes zero through 31.
    ReleaseTensorMemory { value: ValueId },

    /// Initializes a transaction barrier owned by a canonical copy-token value. Record the issuing thread only.
    InitializeBarrier { barrier: ValueId, arrival_count: u32 },

    /// Arrives once and adds the exact expected transfer bytes to the current barrier generation.
    ArriveExpectTransaction { barrier: ValueId, bytes: usize },

    /// Issues a TMA transfer tracked by the current transaction-barrier generation; completion is simulated internally.
    TmaCopy {
        barrier: ValueId,
        source: (ValueId, ArrayReferenceTransform),
        destination: (ValueId, ArrayReferenceTransform),
    },

    /// Acquires one completed generation for this thread. Native parity is the low bit of `generation`.
    WaitBarrier { barrier: ValueId, generation: u64 },

    /// Invalidates a completed barrier after completion has been acquired by the declared CTA.
    InvalidateBarrier { barrier: ValueId },

    /// Full-CTA rendezvous at a native barrier site, shared by every participating thread.
    Barrier { site: usize },

    /// Copies one thread-owned shared selection from the named peer CTA into local shared storage.
    /// Every CTA thread participates at the same instruction, immediately between cluster publication barriers.
    DistributedCopy {
        source_block: u32,
        source: (ValueId, ArrayReferenceTransform),
        destination: (ValueId, ArrayReferenceTransform),
    },

    /// Rendezvous with the peer CTA after the preceding local completion barrier.
    ClusterBarrier { site: usize },
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

/// One canonical tensor-memory allocation and its publication/initialization state.
#[derive(Clone, Debug)]
struct TensorMemory {
    /// Whether allocation has crossed its first CTA barrier.
    published: bool,

    /// Whether a completed overwriting matrix operation initialized its contents.
    initialized: bool,

    /// Current operation, retained until its completion is published to the CTA.
    pending: Option<ValueId>,

    /// Whether the allocating warp has released this allocation.
    released: bool,
}

/// A tensor-memory operation tied to one canonical completion token.
#[derive(Clone, Debug)]
struct TensorOperation {
    /// Canonical destination allocation.
    destination: ValueId,

    /// Immutable operand selections reserved until completion acquisition.
    sources: Vec<Access>,

    /// Whether native commit attached the operation to its barrier.
    committed: bool,

    /// Whether the elected thread waited for completion.
    waited: bool,

    /// Whether that completion was published by a CTA barrier.
    published: bool,
}

/// Resumable communication state for one CTA, shared by standalone and clustered replay.
#[derive(Clone, Debug)]
struct CtaState {
    /// Next recorded event in each thread.
    positions: Vec<usize>,

    /// Uncommitted copies for each thread.
    uncommitted: Vec<Vec<PendingCopy>>,

    /// Committed copy groups for each thread.
    groups: Vec<Vec<Vec<PendingCopy>>>,

    /// Accesses since the relevant synchronization boundary.
    history: Vec<(u32, InstructionId, Access, Option<Completion>)>,

    /// Matrix issues not yet committed.
    mma_uncommitted: Vec<PendingMma>,

    /// FIFO matrix groups and their completion generations.
    mma_groups: Vec<(u64, Vec<PendingMma>)>,

    /// Generation assigned to the next matrix group.
    mma_group: u64,

    /// Accumulators whose register dependencies are fenced.
    mma_fenced: BTreeSet<ValueId>,

    /// Whether preceding shared transport writes are proxy-fenced.
    proxy_fenced: bool,

    /// Canonical tensor-memory allocation states.
    tensor_memory: BTreeMap<ValueId, TensorMemory>,

    /// Canonical tensor-memory completion tokens.
    tensor_operations: BTreeMap<ValueId, TensorOperation>,

    /// Transaction-barrier generations and reservations.
    transactions: BTreeMap<ValueId, TransactionBarrier>,

    /// Threads that acquired each transaction generation.
    acquired: BTreeMap<(ValueId, u64), Vec<bool>>,

    /// Executed communication events in deterministic order.
    trace: Vec<(u32, InstructionId)>,
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

/// One live transaction barrier, retaining its exact arrival/byte contract and reserved transfer selections.
#[derive(Clone, Debug)]
struct TransactionBarrier {
    /// Thread that initialized this object before publishing it to the CTA.
    initialized_by: u32,

    /// Whether a CTA barrier has made initialization visible to the remaining participants.
    initialization_published: bool,

    /// Number of arrivals required by each generation.
    arrival_count: u32,

    /// Current monotonically increasing generation; native parity alone cannot identify stale waits.
    generation: u64,

    /// Arrivals recorded for the current generation.
    arrivals: u32,

    /// Total bytes promised by those arrivals.
    expected_bytes: usize,

    /// Total bytes covered by issued transfers.
    transferred_bytes: usize,

    /// Transfers whose completion and visibility are acquired by a matching wait.
    copies: Vec<PendingCopy>,

    /// Whether the owner has invalidated this barrier.
    invalidated: bool,
}

/// Completion identity attached only to accesses made by an asynchronous native engine.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Completion {
    /// Canonical barrier owner and its monotonic generation.
    Transaction(ValueId, u64),

    /// FIFO WGMMA group number within the one declared warpgroup.
    Wgmma(u64),

    /// One canonical tensor-memory completion token.
    Tensor(ValueId),
}

/// Operand reservations for one collectively issued matrix operation.
#[derive(Clone, Debug)]
struct PendingMma {
    /// Shared operand selections whose contents must remain stable until completion.
    sources: [Access; 2],

    /// Logical accumulator whose ordinary accesses must wait for asynchronous completion.
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
            SynchronizationEvent::AsyncProxyFence
            | SynchronizationEvent::WgmmaFence { .. }
            | SynchronizationEvent::WgmmaIssue { .. }
            | SynchronizationEvent::WgmmaCommit
            | SynchronizationEvent::WgmmaWait { .. }
                if self.participants.get() != 128 =>
            {
                return Err(SynchronizationError::Wgmma {
                    instruction,
                    reason: "requires exactly one complete 128-thread warpgroup",
                });
            }
            SynchronizationEvent::WgmmaWait { remaining } if *remaining > 7 => {
                return Err(SynchronizationError::Wgmma { instruction, reason: "wait group count must be in [0, 7]" });
            }
            SynchronizationEvent::WgmmaIssue { accumulator, left, right } => {
                self.access(left.0, &left.1, false)?;
                self.access(right.0, &right.1, false)?;
                if !self.storage.contains_key(accumulator) {
                    return Err(SynchronizationError::Storage { value: *accumulator });
                }
            }
            SynchronizationEvent::AllocateTensorMemory { value }
            | SynchronizationEvent::ReleaseTensorMemory { value }
            | SynchronizationEvent::LoadTensorMemory { value, .. } => {
                if self.participants.get() != 128 {
                    return Err(SynchronizationError::TensorMemory {
                        value: *value,
                        instruction,
                        reason: "requires exactly one complete 128-thread cta",
                    });
                }
                if !self.storage.contains_key(value) {
                    return Err(SynchronizationError::Storage { value: *value });
                }
                if matches!(
                    &event,
                    SynchronizationEvent::AllocateTensorMemory { .. }
                        | SynchronizationEvent::ReleaseTensorMemory { .. }
                ) && thread >= 32
                {
                    return Err(SynchronizationError::TensorMemory {
                        value: *value,
                        instruction,
                        reason: "allocation and release require the first warp",
                    });
                }
                if let SynchronizationEvent::LoadTensorMemory { transform, .. } = &event {
                    self.access(*value, transform, false)?;
                }
            }
            SynchronizationEvent::IssueTensorMemory { token, destination, left, right, scales, .. } => {
                if self.participants.get() != 128 || thread != 0 {
                    return Err(SynchronizationError::TensorMemory {
                        value: *token,
                        instruction,
                        reason: "issue, commit, and wait require the elected cta thread",
                    });
                }
                if !self.storage.contains_key(destination) {
                    return Err(SynchronizationError::Storage { value: *destination });
                }
                self.access(left.0, &left.1, false)?;
                self.access(right.0, &right.1, false)?;
                if !matches!(scales.len(), 0 | 2) {
                    return Err(SynchronizationError::TensorMemory {
                        value: *token,
                        instruction,
                        reason: "requires zero or two scale allocations",
                    });
                }
                for value in scales {
                    if !self.storage.contains_key(value) {
                        return Err(SynchronizationError::Storage { value: *value });
                    }
                }
            }
            SynchronizationEvent::IssueTensorMemoryCluster { token, .. }
                if self.participants.get() != 128 || thread != 0 =>
            {
                return Err(SynchronizationError::TensorMemory {
                    value: *token,
                    instruction,
                    reason: "issue, commit, and wait require the elected cta thread",
                });
            }
            SynchronizationEvent::CopyTensorMemory { token, source, destination }
            | SynchronizationEvent::CopyTensorMemoryCluster { token, source, destination } => {
                if self.participants.get() != 128 || thread != 0 {
                    return Err(SynchronizationError::TensorMemory {
                        value: *token,
                        instruction,
                        reason: "issue, commit, and wait require the elected cta thread",
                    });
                }
                let source = self.access(source.0, &source.1, false)?;
                let destination =
                    self.storage.get(destination).ok_or(SynchronizationError::Storage { value: *destination })?;
                let size = ArrayAddressing::new(destination.clone())?.logical_byte_len();
                if size == 0 || source.ranges.iter().map(Range::len).sum::<usize>() != size {
                    return Err(SynchronizationError::TensorMemory {
                        value: *token,
                        instruction,
                        reason: "copy requires matching nonempty logical byte counts",
                    });
                }
            }
            SynchronizationEvent::CommitTensorMemory { token }
            | SynchronizationEvent::CommitTensorMemoryCluster { token }
            | SynchronizationEvent::WaitTensorMemory { token }
                if self.participants.get() != 128 || thread != 0 =>
            {
                return Err(SynchronizationError::TensorMemory {
                    value: *token,
                    instruction,
                    reason: "issue, commit, and wait require the elected cta thread",
                });
            }
            SynchronizationEvent::Access { value, transform, mode } => {
                if !matches!(mode, ReferenceAccessMode::Read | ReferenceAccessMode::Write) {
                    return Err(SynchronizationError::AccessMode { mode: *mode });
                }
                self.access(*value, transform, *mode == ReferenceAccessMode::Write)?;
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
            SynchronizationEvent::InitializeBarrier { barrier, arrival_count } if *arrival_count == 0 => {
                return Err(SynchronizationError::TransactionBarrier {
                    barrier: *barrier,
                    instruction,
                    reason: "requires a positive arrival count",
                });
            }
            SynchronizationEvent::TmaCopy { barrier, source, destination } => {
                let source = self.access(source.0, &source.1, false)?;
                let destination = self.access(destination.0, &destination.1, true)?;
                let source_bytes = source.ranges.iter().map(Range::len).sum::<usize>();
                let destination_bytes = destination.ranges.iter().map(Range::len).sum::<usize>();
                if source_bytes == 0 || source_bytes != destination_bytes {
                    return Err(SynchronizationError::TransactionBarrier {
                        barrier: *barrier,
                        instruction,
                        reason: "requires matching nonempty transfer selections",
                    });
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
    /// A copy-group wait completes issuing-thread copies only; a transaction wait acquires its exact generation.
    /// Full-CTA release establishes cross-thread visibility. Transaction waits block until recorded arrivals and
    /// transfer bytes match; missing progress reports deadlock. Every initialized transaction barrier must be
    /// invalidated.
    pub fn simulate(&self) -> Result<Vec<(u32, InstructionId)>, SynchronizationError> {
        let mut state = CtaState::new(self);
        while state.trace.len() < self.event_count {
            if !state.step(self, false)? {
                return Err(SynchronizationError::ClusterDeadlock { waiting: state.cluster_waiting(self, 0) });
            }
        }
        state.finish()
    }

    /// Replays two CTAs with independent local storage and resources, rendezvousing at actual cluster barriers.
    /// Each cluster event must immediately follow the lowering's local barrier in every thread. A cluster barrier
    /// does not replace either CTA's explicit copy or tensor-memory completion protocol.
    pub fn simulate_cluster(&self, peer: &Self) -> Result<Vec<(u32, u32, InstructionId)>, SynchronizationError> {
        for (value, r#type) in self.storage.iter().chain(&peer.storage) {
            if self.storage.get(value) != Some(r#type) || peer.storage.get(value) != Some(r#type) {
                return Err(SynchronizationError::Selection { value: *value });
            }
        }
        let plans = [self, peer];
        let mut states = [CtaState::new(self), CtaState::new(peer)];
        let mut trace = Vec::new();
        loop {
            let mut progressed = false;
            for (block, (plan, state)) in plans.iter().zip(&mut states).enumerate() {
                if state.trace.len() < plan.event_count {
                    let previous = state.trace.len();
                    progressed |= state.step(plan, true)?;
                    trace.extend(
                        state.trace[previous..]
                            .iter()
                            .map(|(thread, instruction)| (block as u32, *thread, *instruction)),
                    );
                }
            }
            if states.iter().zip(plans).all(|(state, plan)| state.trace.len() == plan.event_count) {
                for state in states {
                    state.finish()?;
                }
                return Ok(trace);
            }
            let tensor = plans[0].events.iter().enumerate().find_map(|(thread, events)| {
                events.get(states[0].positions[thread]).filter(|(_, event)| {
                    matches!(
                        event,
                        SynchronizationEvent::AllocateTensorMemory { .. }
                            | SynchronizationEvent::ReleaseTensorMemory { .. }
                            | SynchronizationEvent::LoadTensorMemory { .. }
                    )
                })
            });
            if let Some((instruction, event)) = tensor {
                if plans.iter().any(|plan| plan.participants.get() != 128) {
                    return Err(SynchronizationError::ClusterDeadlock { waiting: vec![] });
                }
                let (value, participants) = match event {
                    SynchronizationEvent::AllocateTensorMemory { value }
                    | SynchronizationEvent::ReleaseTensorMemory { value } => (*value, 32),
                    SynchronizationEvent::LoadTensorMemory { value, .. } => (*value, 128),
                    _ => unreachable!(),
                };
                let uniform = states.iter().zip(plans).all(|(state, plan)| {
                    (0..participants).all(|thread| {
                        plan.events[thread].get(state.positions[thread]).is_some_and(|(other_instruction, other)| {
                            *instruction == *other_instruction
                                && match (event, other) {
                                    (
                                        SynchronizationEvent::LoadTensorMemory { value, .. },
                                        SynchronizationEvent::LoadTensorMemory { value: other, .. },
                                    ) => value == other,
                                    _ => event == other,
                                }
                        })
                    })
                });
                if uniform {
                    for (block, (state, plan)) in states.iter_mut().zip(plans).enumerate() {
                        if matches!(event, SynchronizationEvent::LoadTensorMemory { .. }) {
                            let dimensions = plan.storage[&value].static_shape().unwrap();
                            let rows = dimensions.dimensions()[0] / 2;
                            for thread in 0..participants {
                                let (_, SynchronizationEvent::LoadTensorMemory { transform, .. }) =
                                    &plan.events[thread][state.positions[thread]]
                                else {
                                    unreachable!()
                                };
                                let Some(ArrayReferenceTransform::Slice { axes }) = ArrayReferenceTransformPath::root()
                                    .with_transform(transform.clone())
                                    .root_slice(&plan.storage[&value])
                                else {
                                    unreachable!()
                                };
                                if axes[0].start() < block * rows
                                    || axes[0].start() + axes[0].size() > (block + 1) * rows
                                {
                                    return Err(SynchronizationError::TensorMemory {
                                        value,
                                        instruction: *instruction,
                                        reason: "load exceeds this cta's allocated tensor-memory rows",
                                    });
                                }
                            }
                        }
                        TensorMemory::collective(
                            &mut state.tensor_memory,
                            &state.tensor_operations,
                            value,
                            *instruction,
                            event,
                        )?;
                        for thread in 0..participants {
                            state.positions[thread] += 1;
                            state.trace.push((thread as u32, *instruction));
                            trace.push((block as u32, thread as u32, *instruction));
                        }
                    }
                    progressed = true;
                }
            }
            if let Some((instruction, event)) = plans[0].events[0].get(states[0].positions[0]).filter(|(_, event)| {
                matches!(
                    event,
                    SynchronizationEvent::IssueTensorMemoryCluster { .. }
                        | SynchronizationEvent::CopyTensorMemoryCluster { .. }
                        | SynchronizationEvent::CommitTensorMemoryCluster { .. }
                )
            }) {
                match event {
                    SynchronizationEvent::CommitTensorMemoryCluster { token } => {
                        for state in &mut states {
                            TensorOperation::commit(&mut state.tensor_operations, *token, *instruction)?;
                        }
                    }
                    SynchronizationEvent::IssueTensorMemoryCluster {
                        token,
                        destination,
                        left,
                        right,
                        accumulate,
                        scales,
                    } => {
                        for (block, (state, plan)) in states.iter_mut().zip(plans).enumerate() {
                            let position = state.positions[0];
                            if position == 0
                                || !matches!(
                                    plan.events[0][position - 1].1,
                                    SynchronizationEvent::ClusterBarrier { .. }
                                )
                            {
                                return Err(SynchronizationError::TensorMemory {
                                    value: *token,
                                    instruction: *instruction,
                                    reason: "collective issue requires both ctas' completed transport publication",
                                });
                            }
                            TensorOperation::issue(
                                plan,
                                *instruction,
                                &SynchronizationEvent::IssueTensorMemory {
                                    token: *token,
                                    destination: *destination,
                                    left: left[block].clone(),
                                    right: right[block].clone(),
                                    accumulate: *accumulate,
                                    scales: scales.clone(),
                                },
                                &mut state.tensor_memory,
                                &mut state.tensor_operations,
                            )?;
                        }
                    }
                    SynchronizationEvent::CopyTensorMemoryCluster { token, source, destination } => {
                        for (state, plan) in states.iter_mut().zip(plans) {
                            let position = state.positions[0];
                            if position == 0
                                || !matches!(
                                    plan.events[0][position - 1].1,
                                    SynchronizationEvent::ClusterBarrier { .. }
                                )
                            {
                                return Err(SynchronizationError::TensorMemory {
                                    value: *token,
                                    instruction: *instruction,
                                    reason: "collective issue requires both ctas' completed transport publication",
                                });
                            }
                            TensorOperation::issue(
                                plans[0],
                                *instruction,
                                &SynchronizationEvent::CopyTensorMemory {
                                    token: *token,
                                    source: source.clone(),
                                    destination: *destination,
                                },
                                &mut state.tensor_memory,
                                &mut state.tensor_operations,
                            )?;
                        }
                        // One native instruction reads the elected CTA's source and writes both destinations.
                        states[1].tensor_operations.get_mut(token).unwrap().sources.clear();
                    }
                    _ => unreachable!(),
                }
                states[0].positions[0] += 1;
                states[0].trace.push((0, *instruction));
                trace.push((0, 0, *instruction));
                progressed = true;
            }
            let waiting = states
                .iter()
                .zip(plans)
                .enumerate()
                .flat_map(|(block, (state, plan))| state.cluster_waiting(plan, block as u32))
                .collect::<Vec<_>>();
            let participant_count = self.participants.get() as usize + peer.participants.get() as usize;
            if waiting.len() == participant_count
                && waiting
                    .iter()
                    .all(|(_, _, site, instruction)| (*site, *instruction) == (waiting[0].2, waiting[0].3))
            {
                if waiting.iter().any(|(block, thread, _, _)| {
                    let position = states[*block as usize].positions[*thread as usize];
                    position == 0
                        || !matches!(
                            plans[*block as usize].events[*thread as usize][position - 1].1,
                            SynchronizationEvent::Barrier { .. }
                        )
                }) {
                    return Err(SynchronizationError::ClusterDeadlock { waiting });
                }
                for (block, thread, _, instruction) in &waiting {
                    let state = &mut states[*block as usize];
                    state.positions[*thread as usize] += 1;
                    state.trace.push((*thread, *instruction));
                    trace.push((*block, *thread, *instruction));
                }
                progressed = true;
            }
            let copies = states
                .iter()
                .zip(plans)
                .enumerate()
                .flat_map(|(block, (state, plan))| {
                    state.positions.iter().enumerate().filter_map(move |(thread, position)| {
                        let (instruction, event @ SynchronizationEvent::DistributedCopy { .. }) =
                            plan.events[thread].get(*position)?
                        else {
                            return None;
                        };
                        Some((block, thread, *instruction, event))
                    })
                })
                .collect::<Vec<_>>();
            if copies.len() == participant_count {
                let instruction = copies[0].2;
                let error = |reason| SynchronizationError::ClusterCopy { instruction, reason };
                let mut accesses = Vec::new();
                for (block, thread, other_instruction, event) in &copies {
                    let position = states[*block].positions[*thread];
                    let events = &plans[*block].events[*thread];
                    if *other_instruction != instruction
                        || position == 0
                        || !matches!(events[position - 1].1, SynchronizationEvent::ClusterBarrier { .. })
                        || !matches!(events.get(position + 1), Some((_, SynchronizationEvent::Barrier { .. })))
                        || !matches!(events.get(position + 2), Some((_, SynchronizationEvent::ClusterBarrier { .. })))
                    {
                        return Err(error("requires uniform participation between cluster publication barriers"));
                    }
                    let SynchronizationEvent::DistributedCopy { source_block, source, destination } = event else {
                        unreachable!()
                    };
                    if *source_block as usize != 1 - *block {
                        return Err(error("requires the other cta as its source"));
                    }
                    let source_access = plans[*source_block as usize].access(source.0, &source.1, false)?;
                    let destination_access = plans[*block].access(destination.0, &destination.1, true)?;
                    if source_access.ranges.len() != 1 || destination_access.ranges.len() != 1 {
                        return Err(error("requires nonempty contiguous source and destination selections"));
                    }
                    if source_access.ranges.iter().map(Range::len).sum::<usize>()
                        != destination_access.ranges.iter().map(Range::len).sum::<usize>()
                    {
                        return Err(error("requires matching source and destination byte counts"));
                    }
                    for (storage_block, access) in [(*source_block, source_access), (*block as u32, destination_access)]
                    {
                        for (other_storage, other_block, other_thread, other_access) in &accesses {
                            if *other_storage == storage_block
                                && access.overlaps(other_access)
                                && (access.writes || other_access.writes)
                            {
                                return Err(SynchronizationError::ClusterRace {
                                    value: access.value,
                                    first_block: *other_block,
                                    first_thread: *other_thread,
                                    second_block: *block as u32,
                                    second_thread: *thread as u32,
                                });
                            }
                        }
                        accesses.push((storage_block, *block as u32, *thread as u32, access));
                    }
                }
                for (block, thread, instruction, _) in copies {
                    states[block].positions[thread] += 1;
                    states[block].trace.push((thread as u32, instruction));
                    trace.push((block as u32, thread as u32, instruction));
                }
                progressed = true;
            }
            if !progressed {
                for (state, plan) in states.iter().zip(plans) {
                    for (thread, position) in state.positions.iter().enumerate() {
                        if let Some((instruction, SynchronizationEvent::WaitTensorMemory { token })) =
                            plan.events[thread].get(*position)
                        {
                            return Err(SynchronizationError::TensorMemory {
                                value: *token,
                                instruction: *instruction,
                                reason: "collective wait has no committed issue",
                            });
                        }
                    }
                }
                return Err(SynchronizationError::ClusterDeadlock { waiting });
            }
        }
    }

    /// Resolves a static transform through the canonical root-selection and addressing implementations.
    fn access(
        &self,
        value: ValueId,
        transform: &ArrayReferenceTransform,
        writes: bool,
    ) -> Result<Access, SynchronizationError> {
        let r#type = self.storage.get(&value).ok_or(SynchronizationError::Storage { value })?;
        let path = ArrayReferenceTransformPath::root().with_transform(transform.clone());
        let Some(ArrayReferenceTransform::Slice { axes }) = path.root_slice(r#type) else {
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

impl TensorMemory {
    /// Applies the storage transition after the native allocation, load, or release participants rendezvous.
    fn collective(
        tensor_memory: &mut BTreeMap<ValueId, TensorMemory>,
        tensor_operations: &BTreeMap<ValueId, TensorOperation>,
        value: ValueId,
        instruction: InstructionId,
        event: &SynchronizationEvent,
    ) -> Result<(), SynchronizationError> {
        let error = |reason| SynchronizationError::TensorMemory { value, instruction, reason };
        if matches!(event, SynchronizationEvent::AllocateTensorMemory { .. }) {
            if tensor_memory.contains_key(&value) {
                return Err(error("is allocated more than once"));
            }
            tensor_memory
                .insert(value, TensorMemory { published: false, initialized: false, pending: None, released: false });
        } else {
            let memory = tensor_memory.get_mut(&value).ok_or_else(|| error("is not allocated"))?;
            if memory.released {
                return Err(error("is released"));
            }
            if !memory.published {
                return Err(error("allocation is not published to the cta"));
            }
            if memory.pending.is_some() {
                return Err(error("matrix completion is not published to the cta"));
            }
            if matches!(event, SynchronizationEvent::LoadTensorMemory { .. }) {
                if !memory.initialized {
                    return Err(error("is loaded before initialization"));
                }
            } else {
                if tensor_operations.values().any(|operation| {
                    !operation.published && operation.sources.iter().any(|source| source.value == value)
                }) {
                    return Err(error("operand is reserved by unfinished matrix work"));
                }
                memory.released = true;
            }
        }
        Ok(())
    }
}

impl TensorOperation {
    /// Applies one CTA's actual storage effects of a local or collective native tensor instruction.
    fn issue(
        plan: &CtaSynchronization,
        instruction: InstructionId,
        event: &SynchronizationEvent,
        tensor_memory: &mut BTreeMap<ValueId, TensorMemory>,
        tensor_operations: &mut BTreeMap<ValueId, TensorOperation>,
    ) -> Result<Vec<Access>, SynchronizationError> {
        let (SynchronizationEvent::IssueTensorMemory { token, destination, .. }
        | SynchronizationEvent::CopyTensorMemory { token, destination, .. }) = event
        else {
            unreachable!()
        };
        let error = |reason| SynchronizationError::TensorMemory { value: *destination, instruction, reason };
        let memory = tensor_memory.get(destination).ok_or_else(|| error("is not allocated"))?;
        if memory.released {
            return Err(error("is released"));
        }
        if !memory.published {
            return Err(error("allocation is not published to the cta"));
        }
        if memory.pending.is_some() {
            return Err(error("has an unfinished matrix operation"));
        }
        if matches!(event, SynchronizationEvent::IssueTensorMemory { accumulate: true, .. }) && !memory.initialized {
            return Err(error("accumulates before initialization"));
        }
        if tensor_operations.contains_key(token) {
            return Err(SynchronizationError::TensorMemory {
                value: *token,
                instruction,
                reason: "completion token is reused",
            });
        }
        if tensor_operations.values().any(|operation| {
            !operation.published && operation.sources.iter().any(|source| source.value == *destination)
        }) {
            return Err(error("operand is reserved by unfinished matrix work"));
        }
        let mut sources = match event {
            SynchronizationEvent::IssueTensorMemory { left, right, .. } => {
                vec![plan.access(left.0, &left.1, false)?, plan.access(right.0, &right.1, false)?]
            }
            SynchronizationEvent::CopyTensorMemory { source, .. } => {
                vec![plan.access(source.0, &source.1, false)?]
            }
            _ => unreachable!(),
        };
        if sources.iter().any(|source| tensor_memory.contains_key(&source.value)) {
            return Err(error("matrix operands require shared transport storage"));
        }
        let ordinary_sources = sources.clone();
        if let SynchronizationEvent::IssueTensorMemory { scales, .. } = event {
            if !matches!(scales.len(), 0 | 2) {
                return Err(error("requires zero or two scale allocations"));
            }
            for value in scales {
                let memory = tensor_memory.get(value).ok_or_else(|| error("scale is not allocated"))?;
                if memory.released || !memory.published || !memory.initialized || memory.pending.is_some() {
                    return Err(error("scale initialization is not published to the cta"));
                }
                let r#type = &plan.storage[value];
                let transform = ArrayReferenceTransform::Slice {
                    axes: r#type
                        .shape()
                        .dimensions()
                        .iter()
                        .map(|extent| {
                            let ryft_core::Dimension::Static(extent) = extent else { unreachable!() };
                            ryft_core::ArraySliceAxis::new(0, *extent, 1)
                        })
                        .collect(),
                };
                sources.push(plan.access(*value, &transform, false)?);
            }
        }
        tensor_memory.get_mut(destination).unwrap().pending = Some(*token);
        tensor_operations.insert(
            *token,
            TensorOperation {
                destination: *destination,
                sources: sources.clone(),
                committed: false,
                waited: false,
                published: false,
            },
        );
        Ok(ordinary_sources)
    }

    /// Attaches an issued completion token to its native barrier exactly once.
    fn commit(
        operations: &mut BTreeMap<ValueId, TensorOperation>,
        token: ValueId,
        instruction: InstructionId,
    ) -> Result<(), SynchronizationError> {
        let error = |reason| SynchronizationError::TensorMemory { value: token, instruction, reason };
        let operation = operations.get_mut(&token).ok_or_else(|| error("is not issued"))?;
        if operation.committed {
            return Err(error("is committed more than once"));
        }
        operation.committed = true;
        Ok(())
    }
}

impl CtaState {
    /// Creates empty communication state for a checked CTA trace.
    fn new(plan: &CtaSynchronization) -> Self {
        let count = plan.participants.get() as usize;
        let positions = vec![0; count];
        let uncommitted: Vec<Vec<PendingCopy>> = vec![vec![]; count];
        let groups: Vec<Vec<Vec<PendingCopy>>> = vec![vec![]; count];
        let history: Vec<(u32, InstructionId, Access, Option<Completion>)> = Vec::new();
        let mma_uncommitted: Vec<PendingMma> = Vec::new();
        let mma_groups: Vec<(u64, Vec<PendingMma>)> = Vec::new();
        let mma_group = 0u64;
        let mma_fenced = BTreeSet::new();
        let proxy_fenced = false;
        let tensor_memory: BTreeMap<ValueId, TensorMemory> = BTreeMap::new();
        let tensor_operations: BTreeMap<ValueId, TensorOperation> = BTreeMap::new();
        let transactions: BTreeMap<ValueId, TransactionBarrier> = BTreeMap::new();
        let acquired: BTreeMap<(ValueId, u64), Vec<bool>> = BTreeMap::new();
        let trace = Vec::with_capacity(plan.event_count);
        Self {
            positions,
            uncommitted,
            groups,
            history,
            mma_uncommitted,
            mma_groups,
            mma_group,
            mma_fenced,
            proxy_fenced,
            tensor_memory,
            tensor_operations,
            transactions,
            acquired,
            trace,
        }
    }

    /// Advances one deterministic round while preserving every pending resource and access history.
    fn step(&mut self, plan: &CtaSynchronization, clustered: bool) -> Result<bool, SynchronizationError> {
        let count = plan.participants.get() as usize;
        let Self {
            positions,
            uncommitted,
            groups,
            history,
            mma_uncommitted,
            mma_groups,
            mma_group,
            mma_fenced,
            proxy_fenced,
            tensor_memory,
            tensor_operations,
            transactions,
            acquired,
            trace,
        } = self;

        let mut progressed = false;
        for thread in 0..count {
            let Some((instruction, event)) = plan.events[thread].get(positions[thread]) else {
                continue;
            };
            let transaction_error = |barrier, reason| SynchronizationError::TransactionBarrier {
                barrier,
                instruction: *instruction,
                reason,
            };
            let mut transaction = None;
            let accesses = match event {
                SynchronizationEvent::Barrier { .. }
                | SynchronizationEvent::ClusterBarrier { .. }
                | SynchronizationEvent::DistributedCopy { .. }
                | SynchronizationEvent::IssueTensorMemoryCluster { .. }
                | SynchronizationEvent::CopyTensorMemoryCluster { .. }
                | SynchronizationEvent::CommitTensorMemoryCluster { .. }
                | SynchronizationEvent::AsyncProxyFence
                | SynchronizationEvent::WgmmaFence { .. }
                | SynchronizationEvent::WgmmaIssue { .. }
                | SynchronizationEvent::WgmmaCommit
                | SynchronizationEvent::WgmmaWait { .. }
                | SynchronizationEvent::AllocateTensorMemory { .. }
                | SynchronizationEvent::ReleaseTensorMemory { .. }
                | SynchronizationEvent::LoadTensorMemory { .. } => continue,
                SynchronizationEvent::IssueTensorMemory { token, .. }
                | SynchronizationEvent::CopyTensorMemory { token, .. } => {
                    let accesses = TensorOperation::issue(plan, *instruction, event, tensor_memory, tensor_operations)?;
                    transaction = Some(Completion::Tensor(*token));
                    accesses
                }
                SynchronizationEvent::CommitTensorMemory { token } => {
                    TensorOperation::commit(tensor_operations, *token, *instruction)?;
                    vec![]
                }
                SynchronizationEvent::WaitTensorMemory { token } => {
                    if clustered && tensor_operations.get(token).is_none_or(|operation| !operation.committed) {
                        continue;
                    }
                    let error = |reason| SynchronizationError::TensorMemory {
                        value: *token,
                        instruction: *instruction,
                        reason,
                    };
                    let operation = tensor_operations.get_mut(token).ok_or_else(|| error("is not issued"))?;
                    if !operation.committed {
                        return Err(error("is waited before commit"));
                    }
                    if operation.waited {
                        return Err(error("is waited more than once"));
                    }
                    operation.waited = true;
                    tensor_memory.get_mut(&operation.destination).unwrap().initialized = true;
                    vec![]
                }
                SynchronizationEvent::InitializeBarrier { barrier, arrival_count } => {
                    if transactions.contains_key(barrier) {
                        return Err(transaction_error(*barrier, "is initialized more than once"));
                    }
                    transactions.insert(
                        *barrier,
                        TransactionBarrier {
                            initialized_by: thread as u32,
                            initialization_published: false,
                            arrival_count: *arrival_count,
                            generation: 0,
                            arrivals: 0,
                            expected_bytes: 0,
                            transferred_bytes: 0,
                            copies: vec![],
                            invalidated: false,
                        },
                    );
                    vec![]
                }
                SynchronizationEvent::ArriveExpectTransaction { barrier, bytes } => {
                    let state = transactions
                        .get_mut(barrier)
                        .ok_or_else(|| transaction_error(*barrier, "is not initialized"))?;
                    if state.invalidated {
                        return Err(transaction_error(*barrier, "is invalidated"));
                    }
                    if state.initialized_by != thread as u32 && !state.initialization_published {
                        return Err(transaction_error(*barrier, "initialization is not published to this thread"));
                    }
                    if let Some(observers) = acquired.get(&(*barrier, state.generation)) {
                        if !observers.iter().all(|observed| *observed) {
                            return Err(transaction_error(*barrier, "is reused before completion is published"));
                        }
                        state.generation = state
                            .generation
                            .checked_add(1)
                            .ok_or_else(|| transaction_error(*barrier, "generation overflows"))?;
                        state.arrivals = 0;
                        state.expected_bytes = 0;
                        state.transferred_bytes = 0;
                        state.copies.clear();
                    }
                    if state.arrivals == state.arrival_count {
                        return Err(transaction_error(*barrier, "has too many arrivals"));
                    }
                    state.arrivals += 1;
                    state.expected_bytes = state
                        .expected_bytes
                        .checked_add(*bytes)
                        .ok_or_else(|| transaction_error(*barrier, "expected byte count overflows"))?;
                    vec![]
                }
                SynchronizationEvent::TmaCopy { barrier, source, destination } => {
                    let state =
                        transactions.get(barrier).ok_or_else(|| transaction_error(*barrier, "is not initialized"))?;
                    if state.invalidated {
                        return Err(transaction_error(*barrier, "is invalidated"));
                    }
                    if state.initialized_by != thread as u32 && !state.initialization_published {
                        return Err(transaction_error(*barrier, "initialization is not published to this thread"));
                    }
                    if state.arrivals == 0 || acquired.contains_key(&(*barrier, state.generation)) {
                        return Err(transaction_error(*barrier, "transfer is outside an active arrival phase"));
                    }
                    transaction = Some(Completion::Transaction(*barrier, state.generation));
                    vec![plan.access(source.0, &source.1, false)?, plan.access(destination.0, &destination.1, true)?]
                }
                SynchronizationEvent::WaitBarrier { barrier, generation } => {
                    let state =
                        transactions.get(barrier).ok_or_else(|| transaction_error(*barrier, "is not initialized"))?;
                    if state.invalidated {
                        return Err(transaction_error(*barrier, "is invalidated"));
                    }
                    if state.initialized_by != thread as u32 && !state.initialization_published {
                        return Err(transaction_error(*barrier, "initialization is not published to this thread"));
                    }
                    if *generation != state.generation {
                        return Err(transaction_error(*barrier, "wait names a stale or unissued generation"));
                    }
                    if state.arrivals != state.arrival_count || state.transferred_bytes != state.expected_bytes {
                        continue;
                    }
                    acquired.entry((*barrier, *generation)).or_insert_with(|| vec![false; count])[thread] = true;
                    vec![]
                }
                SynchronizationEvent::InvalidateBarrier { barrier } => {
                    let state = transactions
                        .get_mut(barrier)
                        .ok_or_else(|| transaction_error(*barrier, "is not initialized"))?;
                    if state.invalidated {
                        return Err(transaction_error(*barrier, "is invalidated"));
                    }
                    if state.initialized_by != thread as u32 && !state.initialization_published {
                        return Err(transaction_error(*barrier, "initialization is not published to this thread"));
                    }
                    if state.arrivals != 0
                        && acquired
                            .get(&(*barrier, state.generation))
                            .is_none_or(|observers| !observers.iter().all(|observed| *observed))
                    {
                        return Err(transaction_error(*barrier, "is invalidated before completion is published"));
                    }
                    state.invalidated = true;
                    vec![]
                }
                SynchronizationEvent::CommitGroup => {
                    groups[thread].push(std::mem::take(&mut uncommitted[thread]));
                    vec![]
                }
                SynchronizationEvent::WaitGroup => {
                    groups[thread].clear();
                    vec![]
                }
                SynchronizationEvent::Access { value, transform, mode } => {
                    vec![plan.access(*value, transform, *mode == ReferenceAccessMode::Write)?]
                }
                SynchronizationEvent::AsyncCopy { source, destination } => {
                    vec![plan.access(source.0, &source.1, false)?, plan.access(destination.0, &destination.1, true)?]
                }
            };
            for access in &accesses {
                if tensor_memory.contains_key(&access.value) {
                    return Err(SynchronizationError::TensorMemory {
                        value: access.value,
                        instruction: *instruction,
                        reason: "requires a tensor-memory load or matrix operation",
                    });
                }
                if tensor_operations.values().any(|operation| {
                    !(operation.published || operation.waited && thread == 0)
                        && access.writes
                        && operation.sources.iter().any(|source| access.overlaps(source))
                }) {
                    return Err(SynchronizationError::TensorMemory {
                        value: access.value,
                        instruction: *instruction,
                        reason: "operand is reserved by unfinished matrix work",
                    });
                }
                if access.writes {
                    *proxy_fenced = false;
                    mma_fenced.remove(&access.value);
                }
                for pending in mma_uncommitted.iter().chain(mma_groups.iter().flat_map(|(_, operations)| operations)) {
                    if access.overlaps(&pending.destination)
                        || access.writes && pending.sources.iter().any(|source| access.overlaps(source))
                    {
                        return Err(SynchronizationError::Wgmma {
                            instruction: *instruction,
                            reason: "accesses reserved operands before matrix completion",
                        });
                    }
                }
                for copy in uncommitted.iter().flatten().chain(groups.iter().flatten().flatten()) {
                    if access.overlaps(&copy.destination) || (access.writes && access.overlaps(&copy.source)) {
                        return Err(SynchronizationError::PendingAccess {
                            thread: thread as u32,
                            instruction: *instruction,
                            value: access.value,
                        });
                    }
                }
                for (barrier, state) in transactions.iter() {
                    if state.invalidated
                        || acquired.get(&(*barrier, state.generation)).is_some_and(|observers| observers[thread])
                    {
                        continue;
                    }
                    for copy in &state.copies {
                        if access.overlaps(&copy.destination) || (access.writes && access.overlaps(&copy.source)) {
                            return Err(SynchronizationError::PendingAccess {
                                thread: thread as u32,
                                instruction: *instruction,
                                value: access.value,
                            });
                        }
                    }
                }
                for (previous_thread, previous, previous_access, previous_transaction) in history.iter() {
                    let visible = match previous_transaction {
                        Some(Completion::Transaction(barrier, generation)) => {
                            acquired.get(&(*barrier, *generation)).is_some_and(|observers| observers[thread])
                        }
                        Some(Completion::Tensor(token)) => tensor_operations
                            .get(token)
                            .is_some_and(|operation| operation.published || operation.waited && thread == 0),
                        _ => false,
                    };
                    if !visible
                        && *previous_thread != thread as u32
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
            if matches!(event, SynchronizationEvent::AsyncCopy { .. } | SynchronizationEvent::TmaCopy { .. }) {
                if accesses[0].overlaps(&accesses[1]) {
                    return Err(SynchronizationError::PendingAccess {
                        thread: thread as u32,
                        instruction: *instruction,
                        value: accesses[1].value,
                    });
                }
                let copy = PendingCopy { source: accesses[0].clone(), destination: accesses[1].clone() };
                if let Some(Completion::Transaction(barrier, _)) = transaction {
                    let state = transactions.get_mut(&barrier).unwrap();
                    let bytes = copy.source.ranges.iter().map(Range::len).sum::<usize>();
                    state.transferred_bytes = state
                        .transferred_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| transaction_error(barrier, "transfer byte count overflows"))?;
                    if state.transferred_bytes > state.expected_bytes && state.arrivals == state.arrival_count {
                        return Err(transaction_error(barrier, "transfers exceed the expected byte count"));
                    }
                    state.copies.push(copy);
                } else {
                    uncommitted[thread].push(copy);
                }
            }
            history.extend(accesses.into_iter().map(|access| (thread as u32, *instruction, access, transaction)));
            positions[thread] += 1;
            trace.push((thread as u32, *instruction));
            progressed = true;
        }
        let collective = plan.events[0].get(positions[0]).filter(|(_, event)| {
            matches!(
                event,
                SynchronizationEvent::AsyncProxyFence
                    | SynchronizationEvent::WgmmaFence { .. }
                    | SynchronizationEvent::WgmmaIssue { .. }
                    | SynchronizationEvent::WgmmaCommit
                    | SynchronizationEvent::WgmmaWait { .. }
            )
        });
        if let Some((instruction, event)) = collective
            .filter(|event| (0..count).all(|thread| plan.events[thread].get(positions[thread]) == Some(*event)))
        {
            let error = |reason| SynchronizationError::Wgmma { instruction: *instruction, reason };
            match event {
                SynchronizationEvent::AsyncProxyFence => *proxy_fenced = true,
                SynchronizationEvent::WgmmaFence { accumulator } => {
                    if !plan.storage.contains_key(accumulator) {
                        return Err(SynchronizationError::Storage { value: *accumulator });
                    }
                    if mma_uncommitted
                        .iter()
                        .chain(mma_groups.iter().flat_map(|(_, operations)| operations))
                        .any(|pending| pending.destination.value == *accumulator)
                    {
                        return Err(error("fences accumulator registers before matrix completion"));
                    }
                    mma_fenced.insert(*accumulator);
                }
                SynchronizationEvent::WgmmaIssue { accumulator, left, right } => {
                    if !*proxy_fenced {
                        return Err(error("requires a shared asynchronous-proxy fence before issue"));
                    }
                    if !mma_fenced.contains(accumulator) {
                        return Err(error("requires an accumulator register fence before issue"));
                    }
                    let sources = [plan.access(left.0, &left.1, false)?, plan.access(right.0, &right.1, false)?];
                    let r#type = &plan.storage[accumulator];
                    let shape = r#type.static_shape().ok_or(SynchronizationError::Selection { value: *accumulator })?;
                    let transform = ArrayReferenceTransform::Slice {
                        axes: shape.dimensions().iter().map(|extent| ArraySliceAxis::new(0, *extent, 1)).collect(),
                    };
                    let destination = plan.access(*accumulator, &transform, true)?;
                    if sources.iter().any(|source| source.overlaps(&destination)) {
                        return Err(error("operand aliases the matrix accumulator"));
                    }
                    for access in sources.iter().chain(std::iter::once(&destination)) {
                        for copy in uncommitted.iter().flatten().chain(groups.iter().flatten().flatten()) {
                            if access.overlaps(&copy.destination) || access.writes && access.overlaps(&copy.source) {
                                return Err(error("matrix issue accesses an unfinished asynchronous copy"));
                            }
                        }
                        for (barrier, state) in transactions.iter() {
                            if !state.invalidated
                                && acquired
                                    .get(&(*barrier, state.generation))
                                    .is_none_or(|observers| !observers.iter().all(|observed| *observed))
                                && state.copies.iter().any(|copy| {
                                    access.overlaps(&copy.destination) || access.writes && access.overlaps(&copy.source)
                                })
                            {
                                return Err(error("matrix issue accesses an unpublished transaction"));
                            }
                        }
                        for pending in
                            mma_uncommitted.iter().chain(mma_groups.iter().flat_map(|(_, operations)| operations))
                        {
                            if !access.writes && access.overlaps(&pending.destination)
                                || access.writes && pending.sources.iter().any(|source| access.overlaps(source))
                            {
                                return Err(error("matrix issue accesses a reserved operand"));
                            }
                        }
                        for (_, _, previous, completion) in history.iter() {
                            if !matches!(completion, Some(Completion::Wgmma(_)))
                                && (access.writes || previous.writes)
                                && access.overlaps(previous)
                            {
                                return Err(error("matrix operands lack a separating cta barrier"));
                            }
                        }
                    }
                    history.extend(
                        sources
                            .iter()
                            .cloned()
                            .chain(std::iter::once(destination.clone()))
                            .map(|access| (0, *instruction, access, Some(Completion::Wgmma(*mma_group)))),
                    );
                    mma_uncommitted.push(PendingMma { sources, destination });
                }
                SynchronizationEvent::WgmmaCommit => {
                    if mma_groups.len() == 8 {
                        return Err(error("exceeds eight outstanding committed matrix groups"));
                    }
                    mma_groups.push((*mma_group, std::mem::take(mma_uncommitted)));
                    *mma_group += 1;
                }
                SynchronizationEvent::WgmmaWait { remaining } => {
                    let completed = mma_groups.len().saturating_sub(*remaining);
                    for (group, _) in mma_groups.drain(..completed) {
                        history.retain(|(_, _, _, completion)| *completion != Some(Completion::Wgmma(group)));
                    }
                }
                _ => unreachable!(),
            }
            for (thread, position) in positions.iter_mut().enumerate() {
                *position += 1;
                trace.push((thread as u32, *instruction));
            }
            progressed = true;
        }
        let tensor_collective = plan.events[0].get(positions[0]).filter(|(_, event)| {
            matches!(
                event,
                SynchronizationEvent::AllocateTensorMemory { .. }
                    | SynchronizationEvent::ReleaseTensorMemory { .. }
                    | SynchronizationEvent::LoadTensorMemory { .. }
            )
        });
        if !clustered && let Some((instruction, event)) = tensor_collective {
            let (value, participants) = match event {
                SynchronizationEvent::AllocateTensorMemory { value }
                | SynchronizationEvent::ReleaseTensorMemory { value } => (*value, 32),
                SynchronizationEvent::LoadTensorMemory { value, .. } => (*value, 128),
                _ => unreachable!(),
            };
            let uniform = (0..participants).all(|thread| {
                plan.events[thread].get(positions[thread]).is_some_and(|(other_instruction, other)| {
                    instruction == other_instruction
                        && match (event, other) {
                            (
                                SynchronizationEvent::LoadTensorMemory { value, .. },
                                SynchronizationEvent::LoadTensorMemory { value: other, .. },
                            ) => value == other,
                            _ => event == other,
                        }
                })
            });
            if uniform {
                TensorMemory::collective(tensor_memory, tensor_operations, value, *instruction, event)?;
                for (thread, position) in positions.iter_mut().enumerate().take(participants) {
                    *position += 1;
                    trace.push((thread as u32, *instruction));
                }
                progressed = true;
            }
        }
        let waiting = (0..count)
            .filter_map(|thread| {
                let (instruction, SynchronizationEvent::Barrier { site }) =
                    plan.events[thread].get(positions[thread])?
                else {
                    return None;
                };
                Some((thread as u32, *site, *instruction))
            })
            .collect::<Vec<_>>();
        if waiting.len() == count && waiting.iter().all(|(_, site, _)| *site == waiting[0].1) {
            if !mma_uncommitted.is_empty() || mma_groups.iter().any(|(_, operations)| !operations.is_empty()) {
                return Err(SynchronizationError::Wgmma {
                    instruction: waiting[0].2,
                    reason: "reaches a cta barrier before matrix groups are waited",
                });
            }
            for (thread, site, instruction) in &waiting {
                let position = *thread as usize;
                if !uncommitted[position].is_empty() || groups[position].iter().any(|group| !group.is_empty()) {
                    return Err(SynchronizationError::PendingBarrier { thread: *thread, site: *site });
                }
                positions[position] += 1;
                trace.push((*thread, *instruction));
            }
            for (token, operation) in tensor_operations.iter_mut() {
                if operation.published {
                    continue;
                }
                if !operation.waited {
                    return Err(SynchronizationError::TensorMemory {
                        value: *token,
                        instruction: waiting[0].2,
                        reason: "reaches a cta barrier before its matrix wait",
                    });
                }
                operation.published = true;
                tensor_memory.get_mut(&operation.destination).unwrap().pending = None;
            }
            for memory in tensor_memory.values_mut() {
                memory.published = true;
            }
            for (barrier, state) in transactions.iter_mut() {
                state.initialization_published = true;
                if state.invalidated || state.arrivals == 0 {
                    continue;
                }
                let observers = acquired.get_mut(&(*barrier, state.generation)).ok_or_else(|| {
                    SynchronizationError::TransactionBarrier {
                        barrier: *barrier,
                        instruction: waiting[0].2,
                        reason: "reaches a cta barrier before its generation is waited",
                    }
                })?;
                observers.fill(true);
            }
            history.clear();
            progressed = true;
        }
        if !progressed {
            if clustered
                && positions.iter().enumerate().any(|(thread, position)| {
                    plan.events[thread].get(*position).is_some_and(|(_, event)| {
                        matches!(
                            event,
                            SynchronizationEvent::IssueTensorMemoryCluster { .. }
                                | SynchronizationEvent::CopyTensorMemoryCluster { .. }
                                | SynchronizationEvent::CommitTensorMemoryCluster { .. }
                                | SynchronizationEvent::AllocateTensorMemory { .. }
                                | SynchronizationEvent::ReleaseTensorMemory { .. }
                                | SynchronizationEvent::LoadTensorMemory { .. }
                                | SynchronizationEvent::WaitTensorMemory { .. }
                        )
                    })
                })
            {
                return Ok(false);
            }
            if positions.iter().enumerate().any(|(thread, position)| {
                matches!(
                    plan.events[thread].get(*position),
                    Some((
                        _,
                        SynchronizationEvent::ClusterBarrier { .. } | SynchronizationEvent::DistributedCopy { .. }
                    ))
                )
            }) {
                return Ok(false);
            }
            if let Some((instruction, value)) = (0..count).find_map(|thread| {
                let (instruction, event) = plan.events[thread].get(positions[thread])?;
                match event {
                    SynchronizationEvent::AllocateTensorMemory { value }
                    | SynchronizationEvent::ReleaseTensorMemory { value }
                    | SynchronizationEvent::LoadTensorMemory { value, .. } => Some((*instruction, *value)),
                    _ => None,
                }
            }) {
                return Err(SynchronizationError::TensorMemory {
                    value,
                    instruction,
                    reason: "deadlocks on nonuniform collective participation",
                });
            }
            if let Some((instruction, _)) = (0..count).find_map(|thread| {
                plan.events[thread].get(positions[thread]).filter(|(_, event)| {
                    matches!(
                        event,
                        SynchronizationEvent::AsyncProxyFence
                            | SynchronizationEvent::WgmmaFence { .. }
                            | SynchronizationEvent::WgmmaIssue { .. }
                            | SynchronizationEvent::WgmmaCommit
                            | SynchronizationEvent::WgmmaWait { .. }
                    )
                })
            }) {
                return Err(SynchronizationError::Wgmma {
                    instruction: *instruction,
                    reason: "deadlocks on nonuniform collective participation",
                });
            }
            let transactions = (0..count)
                .filter_map(|thread| {
                    let (instruction, SynchronizationEvent::WaitBarrier { barrier, generation }) =
                        plan.events[thread].get(positions[thread])?
                    else {
                        return None;
                    };
                    Some((thread as u32, *barrier, *generation, *instruction))
                })
                .collect::<Vec<_>>();
            if !transactions.is_empty() {
                return Err(SynchronizationError::TransactionDeadlock { waiting: transactions });
            }
            return Err(SynchronizationError::Deadlock { waiting });
        }
        Ok(progressed)
    }

    /// Returns the exact blocked participants for the cluster driver's rendezvous and diagnostics.
    fn cluster_waiting(&self, plan: &CtaSynchronization, block: u32) -> Vec<(u32, u32, usize, InstructionId)> {
        self.positions
            .iter()
            .enumerate()
            .filter_map(|(thread, position)| {
                let (instruction, SynchronizationEvent::ClusterBarrier { site }) =
                    plan.events[thread].get(*position)?
                else {
                    return None;
                };
                Some((block, thread as u32, *site, *instruction))
            })
            .collect()
    }

    /// Verifies resource lifetimes after the last event and returns the completed trace.
    fn finish(self) -> Result<Vec<(u32, InstructionId)>, SynchronizationError> {
        let Self {
            uncommitted,
            groups,
            mma_uncommitted,
            mma_groups,
            tensor_operations,
            tensor_memory,
            transactions,
            trace,
            ..
        } = self;
        let count = uncommitted.len();

        if let Some((token, _)) = tensor_operations.iter().find(|(_, operation)| !operation.published) {
            return Err(SynchronizationError::TensorMemory {
                value: *token,
                instruction: trace.last().unwrap().1,
                reason: "exits without publishing matrix completion",
            });
        }
        if let Some((value, _)) = tensor_memory.iter().find(|(_, memory)| !memory.released) {
            return Err(SynchronizationError::TensorMemory {
                value: *value,
                instruction: trace.last().unwrap().1,
                reason: "exits without release",
            });
        }
        if !mma_uncommitted.is_empty() || mma_groups.iter().any(|(_, operations)| !operations.is_empty()) {
            return Err(SynchronizationError::Wgmma {
                instruction: trace.last().unwrap().1,
                reason: "exits with unfinished matrix groups",
            });
        }
        for thread in 0..count {
            let copies = uncommitted[thread].len() + groups[thread].iter().map(Vec::len).sum::<usize>();
            if copies != 0 {
                return Err(SynchronizationError::PendingExit { thread: thread as u32, copies });
            }
        }
        if let Some((barrier, _)) = transactions.iter().find(|(_, state)| !state.invalidated) {
            return Err(SynchronizationError::TransactionExit { barrier: *barrier });
        }
        Ok(trace)
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
    fn transform(start: usize, size: usize) -> ArrayReferenceTransform {
        ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(start, size, 1)] }
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

    /// Records the lowering's leader-owned initialization, CTA publication, and one pending 16-byte TMA transfer.
    fn transaction_plan(participants: u32) -> CtaSynchronization {
        let mut plan = plan(participants);
        plan.record(0, instruction(0), SynchronizationEvent::InitializeBarrier { barrier: value(2), arrival_count: 1 })
            .unwrap();
        for thread in 0..participants {
            plan.record(thread, instruction(1), SynchronizationEvent::Barrier { site: 0 }).unwrap();
        }
        plan.record(0, instruction(2), SynchronizationEvent::ArriveExpectTransaction { barrier: value(2), bytes: 16 })
            .unwrap();
        plan.record(
            0,
            instruction(3),
            SynchronizationEvent::TmaCopy {
                barrier: value(2),
                source: (value(0), transform(0, 16)),
                destination: (value(1), transform(0, 16)),
            },
        )
        .unwrap();
        plan
    }

    /// Records a uniform collective at one canonical instruction for every declared participant.
    fn collective(plan: &mut CtaSynchronization, index: usize, event: SynchronizationEvent) {
        for thread in 0..plan.participants().get() {
            plan.record(thread, instruction(index), event.clone()).unwrap();
        }
    }

    /// Creates the consumed matrix protocol through one committed group with a distinct accumulator owner.
    fn matrix_plan() -> CtaSynchronization {
        let mut plan = plan(128);
        plan.storage.insert(value(2), ArrayType::new_static(DataType::U8, [16]));
        collective(&mut plan, 0, SynchronizationEvent::AsyncProxyFence);
        collective(&mut plan, 1, SynchronizationEvent::WgmmaFence { accumulator: value(2) });
        collective(
            &mut plan,
            2,
            SynchronizationEvent::WgmmaIssue {
                accumulator: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
            },
        );
        collective(&mut plan, 3, SynchronizationEvent::WgmmaCommit);
        plan
    }

    /// Declares tensor-memory storage and completes its allocating warp's CTA publication.
    fn tensor_plan() -> CtaSynchronization {
        let mut plan = plan(128);
        plan.storage.insert(value(2), ArrayType::new_static(DataType::F32, [128, 8]));
        for thread in 0..32 {
            plan.record(thread, instruction(0), SynchronizationEvent::AllocateTensorMemory { value: value(2) })
                .unwrap();
        }
        collective(&mut plan, 1, SynchronizationEvent::Barrier { site: 0 });
        plan
    }

    /// Records one leader-owned matrix issue, commit, wait and CTA completion publication.
    fn tensor_completion(plan: &mut CtaSynchronization, start: usize, token: ValueId, accumulate: bool) {
        plan.record(
            0,
            instruction(start),
            SynchronizationEvent::IssueTensorMemory {
                token,
                destination: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
                accumulate,
                scales: vec![],
            },
        )
        .unwrap();
        plan.record(0, instruction(start + 1), SynchronizationEvent::CommitTensorMemory { token }).unwrap();
        plan.record(0, instruction(start + 2), SynchronizationEvent::WaitTensorMemory { token }).unwrap();
        collective(plan, start + 3, SynchronizationEvent::Barrier { site: start });
    }

    /// Records the allocating warp's collective release.
    fn tensor_release(plan: &mut CtaSynchronization, index: usize) {
        for thread in 0..32 {
            plan.record(thread, instruction(index), SynchronizationEvent::ReleaseTensorMemory { value: value(2) })
                .unwrap();
        }
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
                SynchronizationEvent::Access {
                    value: value(2),
                    transform: transform(0, 4),
                    mode: ReferenceAccessMode::Read,
                },
            )
            .unwrap_err();
        assert_eq!(error, SynchronizationError::Storage { value: value(2) });
        let error = plan
            .record(
                0,
                instruction(1),
                SynchronizationEvent::Access {
                    value: value(0),
                    transform: transform(0, 4),
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
                        source: (value(0), transform(0, width)),
                        destination: (value(1), transform(0, width)),
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
                        source: (value(0), transform(start, source_width)),
                        destination: (value(1), transform(0, destination_width)),
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
    fn test_cta_synchronization_record_transaction_geometry() {
        assert_eq!(
            plan(1).record(
                0,
                instruction(0),
                SynchronizationEvent::InitializeBarrier { barrier: value(2), arrival_count: 0 }
            ),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(0),
                reason: "requires a positive arrival count",
            }),
        );
        for size in [0, 8] {
            assert_eq!(
                plan(1).record(
                    0,
                    instruction(0),
                    SynchronizationEvent::TmaCopy {
                        barrier: value(2),
                        source: (value(0), transform(0, 16)),
                        destination: (value(1), transform(0, size)),
                    }
                ),
                Err(SynchronizationError::TransactionBarrier {
                    barrier: value(2),
                    instruction: instruction(0),
                    reason: "requires matching nonempty transfer selections",
                }),
            );
        }
    }

    #[test]
    fn test_cta_synchronization_record_wgmma_membership_and_wait_count() {
        let mut incomplete = plan(127);
        assert_eq!(
            incomplete.record(0, instruction(0), SynchronizationEvent::AsyncProxyFence),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(0),
                reason: "requires exactly one complete 128-thread warpgroup"
            })
        );
        let mut complete = plan(128);
        assert_eq!(
            complete.record(0, instruction(0), SynchronizationEvent::WgmmaWait { remaining: 8 }),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(0),
                reason: "wait group count must be in [0, 7]"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_record_tensor_memory_membership() {
        let mut plan = tensor_plan();
        assert_eq!(
            plan.record(32, instruction(2), SynchronizationEvent::ReleaseTensorMemory { value: value(2) }),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(2),
                reason: "allocation and release require the first warp"
            })
        );
        assert_eq!(
            plan.record(1, instruction(2), SynchronizationEvent::WaitTensorMemory { token: value(3) }),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(2),
                reason: "issue, commit, and wait require the elected cta thread"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate() {
        let mut plan = plan(2);
        for thread in 0..2 {
            plan.record(
                thread,
                instruction(0),
                SynchronizationEvent::AsyncCopy {
                    source: (value(0), transform(thread as usize * 4, 4)),
                    destination: (value(1), transform(thread as usize * 4, 4)),
                },
            )
            .unwrap();
            plan.record(thread, instruction(1), SynchronizationEvent::CommitGroup).unwrap();
            plan.record(thread, instruction(2), SynchronizationEvent::WaitGroup).unwrap();
            plan.record(thread, instruction(3), SynchronizationEvent::Barrier { site: 0 }).unwrap();
            plan.record(
                thread,
                instruction(4),
                SynchronizationEvent::Access {
                    value: value(1),
                    transform: transform(0, 8),
                    mode: ReferenceAccessMode::Read,
                },
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
                SynchronizationEvent::AsyncCopy {
                    source: (value(0), transform(0, 4)),
                    destination: (value(1), transform(0, 4)),
                },
            )
            .unwrap();
            plan.record(
                0,
                instruction(1),
                SynchronizationEvent::Access { value: owner, transform: transform(0, 4), mode },
            )
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
                SynchronizationEvent::AsyncCopy {
                    source: (value(0), transform(0, 4)),
                    destination: (value(1), transform(0, 4)),
                },
            )
            .unwrap();
        uncommitted.record(0, instruction(1), SynchronizationEvent::WaitGroup).unwrap();
        assert_eq!(uncommitted.simulate(), Err(SynchronizationError::PendingExit { thread: 0, copies: 1 }));

        let mut unsynchronized = plan(2);
        unsynchronized
            .record(
                0,
                instruction(0),
                SynchronizationEvent::AsyncCopy {
                    source: (value(0), transform(0, 4)),
                    destination: (value(1), transform(0, 4)),
                },
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
                SynchronizationEvent::Access {
                    value: value(1),
                    transform: transform(0, 4),
                    mode: ReferenceAccessMode::Read,
                },
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
            SynchronizationEvent::AsyncCopy {
                source: (value(0), transform(0, 4)),
                destination: (value(1), transform(0, 4)),
            },
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

    #[test]
    fn test_cta_synchronization_simulate_transaction() {
        let mut plan = transaction_plan(2);
        plan.record(0, instruction(4), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        for thread in 0..2 {
            plan.record(thread, instruction(5), SynchronizationEvent::Barrier { site: 1 }).unwrap();
        }
        plan.record(0, instruction(6), SynchronizationEvent::InvalidateBarrier { barrier: value(2) })
            .unwrap();
        for thread in 0..2 {
            plan.record(thread, instruction(7), SynchronizationEvent::Barrier { site: 2 }).unwrap();
        }
        plan.record(
            1,
            instruction(8),
            SynchronizationEvent::Access {
                value: value(1),
                transform: transform(0, 16),
                mode: ReferenceAccessMode::Read,
            },
        )
        .unwrap();
        let expected = vec![
            (0, instruction(0)),
            (0, instruction(1)),
            (1, instruction(1)),
            (0, instruction(2)),
            (0, instruction(3)),
            (0, instruction(4)),
            (0, instruction(5)),
            (1, instruction(5)),
            (0, instruction(6)),
            (0, instruction(7)),
            (1, instruction(7)),
            (1, instruction(8)),
        ];
        assert_eq!(plan.simulate(), Ok(expected.clone()));
        assert_eq!(plan.simulate(), Ok(expected));
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_generations() {
        let mut plan = transaction_plan(1);
        plan.record(0, instruction(4), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        plan.record(0, instruction(5), SynchronizationEvent::ArriveExpectTransaction { barrier: value(2), bytes: 16 })
            .unwrap();
        plan.record(
            0,
            instruction(6),
            SynchronizationEvent::TmaCopy {
                barrier: value(2),
                source: (value(0), transform(0, 16)),
                destination: (value(1), transform(0, 16)),
            },
        )
        .unwrap();
        plan.record(0, instruction(7), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 1 })
            .unwrap();
        plan.record(0, instruction(8), SynchronizationEvent::InvalidateBarrier { barrier: value(2) })
            .unwrap();
        assert_eq!(plan.simulate(), Ok((0..9).map(|index| (0, instruction(index))).collect()));
        plan.events[0][7].1 = SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 };
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(7),
                reason: "wait names a stale or unissued generation",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_incomplete() {
        for (arrival_count, expected_bytes) in [(2, 16), (1, 32)] {
            let mut plan = transaction_plan(1);
            plan.events[0][0].1 = SynchronizationEvent::InitializeBarrier { barrier: value(2), arrival_count };
            plan.events[0][2].1 =
                SynchronizationEvent::ArriveExpectTransaction { barrier: value(2), bytes: expected_bytes };
            plan.record(0, instruction(4), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
                .unwrap();
            assert_eq!(
                plan.simulate(),
                Err(SynchronizationError::TransactionDeadlock { waiting: vec![(0, value(2), 0, instruction(4))] })
            );
        }
        let mut plan = transaction_plan(1);
        plan.events[0][2].1 = SynchronizationEvent::ArriveExpectTransaction { barrier: value(2), bytes: 8 };
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(3),
                reason: "transfers exceed the expected byte count",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_lifetime() {
        assert_eq!(transaction_plan(1).simulate(), Err(SynchronizationError::TransactionExit { barrier: value(2) }));
        let mut pending = transaction_plan(1);
        pending
            .record(0, instruction(4), SynchronizationEvent::InvalidateBarrier { barrier: value(2) })
            .unwrap();
        assert_eq!(
            pending.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(4),
                reason: "is invalidated before completion is published",
            })
        );
        let mut completed = transaction_plan(1);
        completed
            .record(0, instruction(4), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        completed
            .record(0, instruction(5), SynchronizationEvent::InvalidateBarrier { barrier: value(2) })
            .unwrap();
        completed
            .record(0, instruction(6), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        assert_eq!(
            completed.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(6),
                reason: "is invalidated",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_visibility() {
        let mut pending = transaction_plan(1);
        pending.record(0, instruction(4), SynchronizationEvent::Barrier { site: 1 }).unwrap();
        assert_eq!(
            pending.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(4),
                reason: "reaches a cta barrier before its generation is waited",
            })
        );
        let mut unpublished = transaction_plan(2);
        unpublished
            .record(0, instruction(4), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        unpublished
            .record(0, instruction(5), SynchronizationEvent::InvalidateBarrier { barrier: value(2) })
            .unwrap();
        assert_eq!(
            unpublished.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(5),
                reason: "is invalidated before completion is published",
            })
        );
        unpublished.events[0].last_mut().unwrap().1 =
            SynchronizationEvent::ArriveExpectTransaction { barrier: value(2), bytes: 16 };
        assert_eq!(
            unpublished.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(5),
                reason: "is reused before completion is published",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_reservations() {
        for (owner, mode) in [(value(0), ReferenceAccessMode::Write), (value(1), ReferenceAccessMode::Read)] {
            let mut plan = transaction_plan(1);
            plan.record(
                0,
                instruction(4),
                SynchronizationEvent::Access { value: owner, transform: transform(0, 16), mode },
            )
            .unwrap();
            assert_eq!(
                plan.simulate(),
                Err(SynchronizationError::PendingAccess { thread: 0, instruction: instruction(4), value: owner })
            );
        }
        let mut plan = transaction_plan(1);
        plan.record(
            0,
            instruction(4),
            SynchronizationEvent::AsyncCopy {
                source: (value(0), transform(0, 4)),
                destination: (value(0), transform(8, 4)),
            },
        )
        .unwrap();
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::PendingAccess { thread: 0, instruction: instruction(4), value: value(0) })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_initialization() {
        let mut unknown = plan(1);
        unknown
            .record(0, instruction(0), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        let error = SynchronizationError::TransactionBarrier {
            barrier: value(2),
            instruction: instruction(0),
            reason: "is not initialized",
        };
        assert_eq!(unknown.simulate(), Err(error.clone()));
        assert_eq!(
            error.to_string(),
            format!("cta transaction barrier {:?} at {} is not initialized", value(2), instruction(0))
        );
        let mut duplicate = transaction_plan(1);
        duplicate
            .record(0, instruction(4), SynchronizationEvent::InitializeBarrier { barrier: value(2), arrival_count: 1 })
            .unwrap();
        assert_eq!(
            duplicate.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(4),
                reason: "is initialized more than once",
            })
        );
        let mut unpublished = plan(2);
        unpublished
            .record(0, instruction(0), SynchronizationEvent::InitializeBarrier { barrier: value(2), arrival_count: 1 })
            .unwrap();
        unpublished
            .record(1, instruction(1), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        assert_eq!(
            unpublished.simulate(),
            Err(SynchronizationError::TransactionBarrier {
                barrier: value(2),
                instruction: instruction(1),
                reason: "initialization is not published to this thread",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_transaction_wait_does_not_complete_copy_groups() {
        let mut plan = transaction_plan(1);
        plan.storage.insert(value(3), ArrayType::new_static(DataType::U8, [16]));
        plan.record(
            0,
            instruction(4),
            SynchronizationEvent::AsyncCopy {
                source: (value(0), transform(0, 4)),
                destination: (value(3), transform(0, 4)),
            },
        )
        .unwrap();
        plan.record(0, instruction(5), SynchronizationEvent::CommitGroup).unwrap();
        plan.record(0, instruction(6), SynchronizationEvent::WaitBarrier { barrier: value(2), generation: 0 })
            .unwrap();
        plan.record(0, instruction(7), SynchronizationEvent::Barrier { site: 1 }).unwrap();
        assert_eq!(plan.simulate(), Err(SynchronizationError::PendingBarrier { thread: 0, site: 1 }));
    }
    #[test]
    fn test_cta_synchronization_simulate_wgmma_completion() {
        let mut plan = matrix_plan();
        collective(&mut plan, 4, SynchronizationEvent::WgmmaWait { remaining: 0 });
        plan.record(
            1,
            instruction(5),
            SynchronizationEvent::Access {
                value: value(2),
                transform: transform(0, 16),
                mode: ReferenceAccessMode::Read,
            },
        )
        .unwrap();
        let mut expected = (0..5)
            .flat_map(|index| (0..128).map(move |thread| (thread, instruction(index))))
            .collect::<Vec<_>>();
        expected.push((1, instruction(5)));
        assert_eq!(plan.simulate(), Ok(expected));
    }

    #[test]
    fn test_cta_synchronization_simulate_wgmma_partial_wait_retains_reservations() {
        let mut plan = matrix_plan();
        collective(
            &mut plan,
            4,
            SynchronizationEvent::WgmmaIssue {
                accumulator: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
            },
        );
        collective(&mut plan, 5, SynchronizationEvent::WgmmaCommit);
        collective(&mut plan, 6, SynchronizationEvent::WgmmaWait { remaining: 1 });
        let mut finished = plan.clone();
        collective(&mut finished, 7, SynchronizationEvent::WgmmaWait { remaining: 0 });
        assert_eq!(finished.simulate().unwrap().len(), 8 * 128);
        plan.record(
            0,
            instruction(7),
            SynchronizationEvent::Access {
                value: value(0),
                transform: transform(0, 16),
                mode: ReferenceAccessMode::Write,
            },
        )
        .unwrap();
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(7),
                reason: "accesses reserved operands before matrix completion"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_wgmma_missing_fences() {
        let mut plan = matrix_plan();
        for events in &mut plan.events {
            events.remove(0);
        }
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(2),
                reason: "requires a shared asynchronous-proxy fence before issue"
            })
        );
        let mut plan = matrix_plan();
        for events in &mut plan.events {
            events.remove(1);
        }
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(2),
                reason: "requires an accumulator register fence before issue"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_wgmma_nonuniform_and_unfinished_groups() {
        let mut plan = matrix_plan();
        collective(&mut plan, 4, SynchronizationEvent::WgmmaWait { remaining: 0 });
        plan.events[127].pop().unwrap();
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(4),
                reason: "deadlocks on nonuniform collective participation"
            })
        );
        let mut plan = matrix_plan();
        collective(&mut plan, 4, SynchronizationEvent::Barrier { site: 0 });
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(4),
                reason: "reaches a cta barrier before matrix groups are waited"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_wgmma_requires_commit_and_bounded_groups() {
        let mut uncommitted = matrix_plan();
        for events in &mut uncommitted.events {
            events.pop().unwrap();
        }
        uncommitted.event_count -= 128;
        collective(&mut uncommitted, 4, SynchronizationEvent::WgmmaWait { remaining: 0 });
        assert_eq!(
            uncommitted.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(4),
                reason: "exits with unfinished matrix groups",
            })
        );
        let mut overflow = matrix_plan();
        for index in 4..12 {
            collective(&mut overflow, index, SynchronizationEvent::WgmmaCommit);
        }
        assert_eq!(
            overflow.simulate(),
            Err(SynchronizationError::Wgmma {
                instruction: instruction(11),
                reason: "exceeds eight outstanding committed matrix groups",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_wgmma_wait_preserves_copy_groups() {
        let mut plan = matrix_plan();
        plan.storage.insert(value(3), ArrayType::new_static(DataType::U8, [16]));
        plan.record(
            0,
            instruction(4),
            SynchronizationEvent::AsyncCopy {
                source: (value(0), transform(0, 4)),
                destination: (value(3), transform(0, 4)),
            },
        )
        .unwrap();
        plan.record(0, instruction(5), SynchronizationEvent::CommitGroup).unwrap();
        collective(&mut plan, 6, SynchronizationEvent::WgmmaWait { remaining: 0 });
        collective(&mut plan, 7, SynchronizationEvent::Barrier { site: 0 });
        assert_eq!(plan.simulate(), Err(SynchronizationError::PendingBarrier { thread: 0, site: 0 }));
    }
    #[test]
    fn test_cta_synchronization_simulate_tensor_memory() {
        let mut plan = tensor_plan();
        tensor_completion(&mut plan, 2, value(3), false);
        tensor_completion(&mut plan, 6, value(4), true);
        for thread in 0..128 {
            plan.record(
                thread,
                instruction(10),
                SynchronizationEvent::LoadTensorMemory {
                    value: value(2),
                    transform: ArrayReferenceTransform::Slice {
                        axes: vec![ArraySliceAxis::new(thread as usize, 1, 1), ArraySliceAxis::new(0, 8, 1)],
                    },
                },
            )
            .unwrap();
        }
        tensor_release(&mut plan, 11);
        let trace = plan.simulate().unwrap();
        assert_eq!(trace.len(), 582);
        assert_eq!(trace.last(), Some(&(31, instruction(11))));
    }

    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_initialization_and_lifetime() {
        let mut accumulate = tensor_plan();
        tensor_completion(&mut accumulate, 2, value(3), true);
        assert_eq!(
            accumulate.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(2),
                reason: "accumulates before initialization"
            })
        );
        let mut load = tensor_plan();
        collective(
            &mut load,
            2,
            SynchronizationEvent::LoadTensorMemory {
                value: value(2),
                transform: ArrayReferenceTransform::Slice {
                    axes: vec![ArraySliceAxis::new(0, 128, 1), ArraySliceAxis::new(0, 8, 1)],
                },
            },
        );
        assert_eq!(
            load.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(2),
                reason: "is loaded before initialization"
            })
        );
        let leaked = tensor_plan();
        assert_eq!(
            leaked.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(1),
                reason: "exits without release"
            })
        );
        let mut released = tensor_plan();
        tensor_release(&mut released, 2);
        tensor_completion(&mut released, 3, value(3), false);
        assert_eq!(
            released.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(3),
                reason: "is released"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_commit_and_wait() {
        let mut missing = tensor_plan();
        missing
            .record(
                0,
                instruction(2),
                SynchronizationEvent::IssueTensorMemory {
                    token: value(3),
                    destination: value(2),
                    left: (value(0), transform(0, 16)),
                    right: (value(1), transform(0, 16)),
                    accumulate: false,
                    scales: vec![],
                },
            )
            .unwrap();
        missing
            .record(0, instruction(3), SynchronizationEvent::WaitTensorMemory { token: value(3) })
            .unwrap();
        assert_eq!(
            missing.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(3),
                reason: "is waited before commit"
            })
        );
        let mut duplicate = tensor_plan();
        tensor_completion(&mut duplicate, 2, value(3), false);
        duplicate
            .record(0, instruction(6), SynchronizationEvent::WaitTensorMemory { token: value(3) })
            .unwrap();
        assert_eq!(
            duplicate.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(6),
                reason: "is waited more than once"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_pending_reservations() {
        let mut plan = tensor_plan();
        plan.record(
            0,
            instruction(2),
            SynchronizationEvent::IssueTensorMemory {
                token: value(3),
                destination: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
                accumulate: false,
                scales: vec![],
            },
        )
        .unwrap();
        let mut raced = plan.clone();
        raced
            .record(
                0,
                instruction(3),
                SynchronizationEvent::Access {
                    value: value(0),
                    transform: transform(0, 16),
                    mode: ReferenceAccessMode::Write,
                },
            )
            .unwrap();
        assert_eq!(
            raced.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(0),
                instruction: instruction(3),
                reason: "operand is reserved by unfinished matrix work"
            })
        );
        collective(&mut plan, 3, SynchronizationEvent::Barrier { site: 1 });
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(3),
                reason: "reaches a cta barrier before its matrix wait"
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_collective_deadlock() {
        let mut plan = plan(128);
        plan.storage.insert(value(2), ArrayType::new_static(DataType::F32, [128, 8]));
        for thread in 0..31 {
            plan.record(thread, instruction(0), SynchronizationEvent::AllocateTensorMemory { value: value(2) })
                .unwrap();
        }
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(0),
                reason: "deadlocks on nonuniform collective participation"
            })
        );
    }
    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_scale_copy() {
        let mut plan = tensor_plan();
        plan.storage.insert(value(4), ArrayType::new_static(DataType::F8E8M0FNU, [16]));
        for thread in 0..32 {
            plan.record(thread, instruction(2), SynchronizationEvent::AllocateTensorMemory { value: value(4) })
                .unwrap();
        }
        collective(&mut plan, 3, SynchronizationEvent::Barrier { site: 3 });
        plan.record(
            0,
            instruction(4),
            SynchronizationEvent::CopyTensorMemory {
                token: value(5),
                source: (value(0), transform(0, 16)),
                destination: value(4),
            },
        )
        .unwrap();
        plan.record(0, instruction(5), SynchronizationEvent::CommitTensorMemory { token: value(5) })
            .unwrap();
        plan.record(0, instruction(6), SynchronizationEvent::WaitTensorMemory { token: value(5) }).unwrap();
        collective(&mut plan, 7, SynchronizationEvent::Barrier { site: 7 });
        plan.record(
            0,
            instruction(8),
            SynchronizationEvent::IssueTensorMemory {
                token: value(6),
                destination: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
                accumulate: false,
                scales: vec![value(4), value(4)],
            },
        )
        .unwrap();
        let mut released = plan.clone();
        for thread in 0..32 {
            released
                .record(thread, instruction(9), SynchronizationEvent::ReleaseTensorMemory { value: value(4) })
                .unwrap();
        }
        assert_eq!(
            released.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(4),
                instruction: instruction(9),
                reason: "operand is reserved by unfinished matrix work",
            })
        );
        let mut overwritten = plan.clone();
        overwritten
            .record(
                0,
                instruction(9),
                SynchronizationEvent::CopyTensorMemory {
                    token: value(7),
                    source: (value(0), transform(0, 16)),
                    destination: value(4),
                },
            )
            .unwrap();
        assert_eq!(
            overwritten.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(4),
                instruction: instruction(9),
                reason: "operand is reserved by unfinished matrix work",
            })
        );
        plan.record(0, instruction(9), SynchronizationEvent::CommitTensorMemory { token: value(6) })
            .unwrap();
        plan.record(0, instruction(10), SynchronizationEvent::WaitTensorMemory { token: value(6) }).unwrap();
        collective(&mut plan, 11, SynchronizationEvent::Barrier { site: 11 });
        tensor_release(&mut plan, 12);
        for thread in 0..32 {
            plan.record(thread, instruction(13), SynchronizationEvent::ReleaseTensorMemory { value: value(4) })
                .unwrap();
        }
        let trace = plan.simulate().unwrap();
        assert_eq!(trace.len(), 646);
        assert_eq!(trace.last(), Some(&(31, instruction(13))));
    }

    #[test]
    fn test_cta_synchronization_simulate_tensor_memory_uninitialized_scales() {
        let mut plan = tensor_plan();
        plan.storage.insert(value(4), ArrayType::new_static(DataType::F8E8M0FNU, [16]));
        for thread in 0..32 {
            plan.record(thread, instruction(2), SynchronizationEvent::AllocateTensorMemory { value: value(4) })
                .unwrap();
        }
        collective(&mut plan, 3, SynchronizationEvent::Barrier { site: 3 });
        plan.record(
            0,
            instruction(4),
            SynchronizationEvent::IssueTensorMemory {
                token: value(6),
                destination: value(2),
                left: (value(0), transform(0, 16)),
                right: (value(1), transform(0, 16)),
                accumulate: false,
                scales: vec![value(4), value(4)],
            },
        )
        .unwrap();
        assert_eq!(
            plan.simulate(),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(4),
                reason: "scale initialization is not published to the cta",
            })
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_cluster() {
        let mut left = plan(2);
        collective(&mut left, 0, SynchronizationEvent::Barrier { site: 0 });
        collective(&mut left, 1, SynchronizationEvent::ClusterBarrier { site: 1 });
        let mut right = left.clone();
        // The same canonical owner denotes independent CTA-local storage.
        left.record(
            0,
            instruction(2),
            SynchronizationEvent::Access {
                value: value(0),
                transform: transform(0, 16),
                mode: ReferenceAccessMode::Write,
            },
        )
        .unwrap();
        right
            .record(
                1,
                instruction(2),
                SynchronizationEvent::Access {
                    value: value(0),
                    transform: transform(0, 16),
                    mode: ReferenceAccessMode::Write,
                },
            )
            .unwrap();
        assert_eq!(
            left.simulate_cluster(&right),
            Ok(vec![
                (0, 0, instruction(0)),
                (0, 1, instruction(0)),
                (1, 0, instruction(0)),
                (1, 1, instruction(0)),
                (0, 0, instruction(1)),
                (0, 1, instruction(1)),
                (1, 0, instruction(1)),
                (1, 1, instruction(1)),
                (0, 0, instruction(2)),
                (1, 1, instruction(2)),
            ])
        );
    }

    #[test]
    fn test_cta_synchronization_simulate_cluster_deadlock() {
        let mut left = plan(2);
        collective(&mut left, 0, SynchronizationEvent::Barrier { site: 0 });
        collective(&mut left, 1, SynchronizationEvent::ClusterBarrier { site: 1 });
        assert_eq!(
            left.simulate_cluster(&plan(2)),
            Err(SynchronizationError::ClusterDeadlock {
                waiting: vec![(0, 0, 1, instruction(1)), (0, 1, 1, instruction(1))],
            })
        );
        assert_eq!(
            left.simulate(),
            Err(SynchronizationError::ClusterDeadlock {
                waiting: vec![(0, 0, 1, instruction(1)), (0, 1, 1, instruction(1))],
            })
        );
        let mut right = plan(2);
        collective(&mut right, 0, SynchronizationEvent::Barrier { site: 0 });
        collective(&mut right, 1, SynchronizationEvent::ClusterBarrier { site: 2 });
        assert_eq!(
            left.simulate_cluster(&right),
            Err(SynchronizationError::ClusterDeadlock {
                waiting: vec![
                    (0, 0, 1, instruction(1)),
                    (0, 1, 1, instruction(1)),
                    (1, 0, 2, instruction(1)),
                    (1, 1, 2, instruction(1)),
                ],
            })
        );
    }
    #[test]
    fn test_cta_synchronization_simulate_cluster_distributed_copy() {
        let mut left = plan(2);
        let mut right = plan(2);
        for (block, plan) in [&mut left, &mut right].into_iter().enumerate() {
            collective(plan, 0, SynchronizationEvent::Barrier { site: 0 });
            collective(plan, 1, SynchronizationEvent::ClusterBarrier { site: 1 });
            for thread in 0..2 {
                let selected = transform((1 - block) * 8 + thread as usize * 4, 4);
                plan.record(
                    thread,
                    instruction(2),
                    SynchronizationEvent::DistributedCopy {
                        source_block: (1 - block) as u32,
                        source: (value(0), selected.clone()),
                        destination: (value(0), selected),
                    },
                )
                .unwrap();
            }
            collective(plan, 3, SynchronizationEvent::Barrier { site: 3 });
            collective(plan, 4, SynchronizationEvent::ClusterBarrier { site: 4 });
        }
        let trace = left.simulate_cluster(&right).unwrap();
        assert_eq!(trace.len(), 20);
        assert_eq!(
            &trace[8..12],
            &[(0, 0, instruction(2)), (0, 1, instruction(2)), (1, 0, instruction(2)), (1, 1, instruction(2)),]
        );
        let mut raced = right.clone();
        let SynchronizationEvent::DistributedCopy { destination, .. } = &mut raced.events[0][2].1 else {
            unreachable!()
        };
        destination.1 = transform(8, 4);
        assert_eq!(
            left.simulate_cluster(&raced),
            Err(SynchronizationError::ClusterRace {
                value: value(0),
                first_block: 0,
                first_thread: 0,
                second_block: 1,
                second_thread: 0,
            })
        );
        let mut noncontiguous = [left.clone(), right.clone()];
        for plan in &mut noncontiguous {
            plan.storage.insert(value(0), ArrayType::new_static(DataType::U8, [4, 4]));
            for thread in 0..2 {
                let SynchronizationEvent::DistributedCopy { source, destination, .. } = &mut plan.events[thread][2].1
                else {
                    unreachable!()
                };
                source.1 = ArrayReferenceTransform::Slice {
                    axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 2, 1)],
                };
                destination.1 = source.1.clone();
            }
        }
        assert_eq!(
            noncontiguous[0].simulate_cluster(&noncontiguous[1]),
            Err(SynchronizationError::ClusterCopy {
                instruction: instruction(2),
                reason: "requires nonempty contiguous source and destination selections",
            })
        );
        let mut unprotected = right.clone();
        for events in &mut unprotected.events {
            events.truncate(3);
        }
        unprotected.event_count = 6;
        assert_eq!(
            left.simulate_cluster(&unprotected),
            Err(SynchronizationError::ClusterCopy {
                instruction: instruction(2),
                reason: "requires uniform participation between cluster publication barriers",
            })
        );
    }
    #[test]
    fn test_cta_synchronization_simulate_cluster_tensor_memory() {
        let mut plans = [plan(128), plan(128)];
        for plan in &mut plans {
            plan.storage.insert(value(0), ArrayType::new_static(DataType::F16, [256, 16]));
            plan.storage.insert(value(1), ArrayType::new_static(DataType::F16, [16, 16]));
            plan.storage.insert(value(2), ArrayType::new_static(DataType::F32, [256, 16]));
            for thread in 0..32 {
                plan.record(thread, instruction(0), SynchronizationEvent::AllocateTensorMemory { value: value(2) })
                    .unwrap();
            }
            collective(plan, 1, SynchronizationEvent::Barrier { site: 1 });
            collective(plan, 2, SynchronizationEvent::ClusterBarrier { site: 2 });
        }
        plans[0]
            .record(
                0,
                instruction(3),
                SynchronizationEvent::IssueTensorMemoryCluster {
                    token: value(3),
                    destination: value(2),
                    left: [0, 128].map(|start| {
                        (
                            value(0),
                            ArrayReferenceTransform::Slice {
                                axes: vec![ArraySliceAxis::new(start, 128, 1), ArraySliceAxis::new(0, 16, 1)],
                            },
                        )
                    }),
                    right: [0, 8].map(|start| {
                        (
                            value(1),
                            ArrayReferenceTransform::Slice {
                                axes: vec![ArraySliceAxis::new(0, 16, 1), ArraySliceAxis::new(start, 8, 1)],
                            },
                        )
                    }),
                    accumulate: false,
                    scales: vec![],
                },
            )
            .unwrap();
        plans[0]
            .record(0, instruction(4), SynchronizationEvent::CommitTensorMemoryCluster { token: value(3) })
            .unwrap();
        for (block, plan) in plans.iter_mut().enumerate() {
            plan.record(0, instruction(5), SynchronizationEvent::WaitTensorMemory { token: value(3) }).unwrap();
            collective(plan, 6, SynchronizationEvent::Barrier { site: 6 });
            collective(plan, 7, SynchronizationEvent::ClusterBarrier { site: 7 });
            for thread in 0..128 {
                plan.record(
                    thread,
                    instruction(8),
                    SynchronizationEvent::LoadTensorMemory {
                        value: value(2),
                        transform: ArrayReferenceTransform::Slice {
                            axes: vec![
                                ArraySliceAxis::new(block * 128 + thread as usize, 1, 1),
                                ArraySliceAxis::new(0, 16, 1),
                            ],
                        },
                    },
                )
                .unwrap();
            }
            tensor_release(plan, 9);
        }
        let trace = plans[0].simulate_cluster(&plans[1]).unwrap();
        assert_eq!(trace.len(), 1412);
        assert_eq!(
            trace.iter().filter(|(_, _, source)| *source == instruction(3)).copied().collect::<Vec<_>>(),
            vec![(0, 0, instruction(3))]
        );
        assert_eq!(
            trace.iter().filter(|(_, _, source)| *source == instruction(4)).copied().collect::<Vec<_>>(),
            vec![(0, 0, instruction(4))]
        );
        let mut wrong_rows = plans[1].clone();
        let (_, SynchronizationEvent::LoadTensorMemory { transform, .. }) = wrong_rows.events[0]
            .iter_mut()
            .find(|(_, event)| matches!(event, SynchronizationEvent::LoadTensorMemory { .. }))
            .unwrap()
        else {
            unreachable!()
        };
        *transform =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 16, 1)] };
        assert_eq!(
            plans[0].simulate_cluster(&wrong_rows),
            Err(SynchronizationError::TensorMemory {
                value: value(2),
                instruction: instruction(8),
                reason: "load exceeds this cta's allocated tensor-memory rows",
            })
        );
        let mut missing_wait = plans[1].clone();
        missing_wait.events[0].retain(|(source, _)| *source != instruction(5));
        missing_wait.event_count -= 1;
        assert_eq!(
            plans[0].simulate_cluster(&missing_wait),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(3),
                reason: "collective issue requires both ctas' completed transport publication",
            })
        );
    }
    #[test]
    fn test_cta_synchronization_simulate_cluster_tensor_memory_copy() {
        let mut plans = [plan(128), plan(128)];
        let scale_type = ArrayType::new_static(DataType::F8E8M0FNU, [128, 4]);
        let whole =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 128, 1), ArraySliceAxis::new(0, 4, 1)] };
        for plan in &mut plans {
            plan.storage.insert(value(0), scale_type.clone());
            plan.storage.insert(value(2), scale_type.clone());
            for thread in 0..32 {
                plan.record(thread, instruction(0), SynchronizationEvent::AllocateTensorMemory { value: value(2) })
                    .unwrap();
            }
            collective(plan, 1, SynchronizationEvent::Barrier { site: 1 });
            collective(plan, 2, SynchronizationEvent::ClusterBarrier { site: 2 });
        }
        plans[0]
            .record(
                0,
                instruction(3),
                SynchronizationEvent::CopyTensorMemoryCluster {
                    token: value(3),
                    source: (value(0), whole),
                    destination: value(2),
                },
            )
            .unwrap();
        plans[0]
            .record(0, instruction(4), SynchronizationEvent::CommitTensorMemoryCluster { token: value(3) })
            .unwrap();
        for plan in &mut plans {
            plan.record(0, instruction(5), SynchronizationEvent::WaitTensorMemory { token: value(3) }).unwrap();
            collective(plan, 6, SynchronizationEvent::Barrier { site: 6 });
            collective(plan, 7, SynchronizationEvent::ClusterBarrier { site: 7 });
            tensor_release(plan, 8);
        }
        let trace = plans[0].simulate_cluster(&plans[1]).unwrap();
        assert_eq!(trace.len(), 1156);
        assert_eq!(
            trace.iter().filter(|(_, _, source)| *source == instruction(3)).copied().collect::<Vec<_>>(),
            vec![(0, 0, instruction(3))]
        );
        let mut duplicate_commit = plans[0].clone();
        duplicate_commit.events[0]
            .insert(5, (instruction(4), SynchronizationEvent::CommitTensorMemoryCluster { token: value(3) }));
        duplicate_commit.event_count += 1;
        assert_eq!(
            duplicate_commit.simulate_cluster(&plans[1]),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(4),
                reason: "is committed more than once",
            })
        );
        let mut uncommitted = plans[0].clone();
        uncommitted.events[0].retain(|(source, _)| *source != instruction(4));
        uncommitted.event_count -= 1;
        assert_eq!(
            uncommitted.simulate_cluster(&plans[1]),
            Err(SynchronizationError::TensorMemory {
                value: value(3),
                instruction: instruction(5),
                reason: "collective wait has no committed issue",
            })
        );
    }
}
