//! Contains the definite initialization analysis and the static launch qualification of portable kernel bodies.
//!
//! A kernel body may read only elements that are definitely initialized on every path that reaches the read, and it
//! must definitely initialize the window of every write-only parameter. Its launch must give every write-only output
//! windows that cover the full array and give every mutable parameter windows that are disjoint, ordered by the grid,
//! or only accumulated atomically. This module proves these properties statically, before any memory is allocated or
//! any grid point executes, and reports a body or launch that it cannot prove with a [`KernelInitializationError`].
//!
//! # Entry Points
//!
//! [`validate_kernel_initialization`] qualifies an attached body against the exact [`KernelCallOperation`] that
//! launches it. [`VerifiedKernel::new`](crate::compilation::VerifiedKernel::new) calls it before compiler admission,
//! and [`KernelDefinition::interpret_interleavings`](crate::calls::KernelDefinition::interpret_interleavings) calls it
//! before exploring schedules. The host interpreter behind
//! [`KernelDefinition::interpret`](crate::calls::KernelDefinition::interpret) runs the same validation through the
//! `validated_kernel_references` function of this crate, which also returns the [`KernelReferenceSummary`] of the
//! validated body, so that the interpreter replays the body without validating it a second time.
//!
//! # Coverage State
//!
//! The analysis keeps, for every canonical [`ReferenceRoot`] that it has encountered (i.e., a body parameter or an
//! allocation), the static array type of that root and a sorted list of the disjoint ranges of its row-major element
//! offsets that are definitely initialized. Reference inputs of an attached region resolve to the canonical roots of
//! the attaching region through the bindings of that particular attachment, so a region that is attached in several
//! places is analyzed separately for each attachment. Read-only and read-write parameters start complete and
//! write-only parameters start empty, except at the later steps of a carry axis described below. Every access relates
//! to its root through the root slice axes of its canonical transform path, which form its static selection.
//!
//! # Transfer Rules
//!
//! The analysis visits the instructions of a region in execution order and applies one rule to each kind of
//! operation. Kernel operations without a rule fail with [`KernelInitializationError::UnsupportedOperation`].
//!
//!   - **Reads and writes.** A pure write (i.e., a write or a swap whose old value is unused, which lowers to a store)
//!     adds the elements that it selects to the coverage of its root. Every other access reads its selection, so each
//!     element that it selects must be initialized, and otherwise it fails with
//!     [`KernelInitializationError::UninitializedRead`]. Every read of an instruction is checked against the coverage
//!     that enters that instruction, and its writes are added only once all of its accesses are checked, so an
//!     instruction never reads an element that only it writes, whatever the order of its accesses. A
//!     [`ReferenceNewOperation`](ryft_core::ReferenceNewOperation) allocation starts complete, because it is created
//!     from a value.
//!   - **Masked accesses.** A masked load, store, or swap whose mask is a constant reads and writes only the elements
//!     that its mask selects. A runtime mask may select any element, so a masked read requires its whole selection to
//!     be initialized and a masked write establishes no coverage. The only exceptions are two forms of masked store
//!     that provably write every element of their selection, i.e., a store without transforms under the
//!     [`WindowMaskOperation`](crate::memory::WindowMaskOperation) validity mask of the very reference that it writes
//!     through, and two stores of the same region with the same selection of the same root under a mask and its
//!     logical negation. Taking a window mask accesses no storage, so the analysis skips it.
//!   - **Dynamic selections.** An access whose transform path is dynamic or otherwise unsupported may select any
//!     element of its root, so it counts as an access of the whole root and its mask is ignored. A read through such
//!     a selection requires the whole root to be initialized, and otherwise it fails with
//!     [`KernelInitializationError::UnknownSelection`]. A write through it establishes no coverage.
//!   - **Asynchronous copies.** An [`AsyncCopyOperation`](crate::memory::AsyncCopyOperation) requires its source
//!     selection to be initialized when it starts, rejects source and destination selections that overlap within one
//!     root, and reserves both selections until a [`WaitOperation`](crate::memory::WaitOperation) of the same region
//!     consumes its completion token. While a copy is pending, an access that overlaps its destination or a mutation
//!     that overlaps its source fails with [`KernelInitializationError::PendingCopyAccess`], and every other access of
//!     its token fails with [`KernelInitializationError::InvalidCopyTokenAccess`], except for the extension commits
//!     described below. The coverage of a static destination selection is published only at the wait. An instruction
//!     with attached regions may run while copies are pending only when none of its transitive accesses touches a
//!     reserved root or a pending token, and every copy that a region starts must complete in that region.
//!   - **Scratch allocations.** A [`ScratchOperation`](crate::memory::ScratchOperation) allocation starts empty.
//!   - **Extensions.** An extension operation declares its timing through [`KernelExtension::memory_semantics`],
//!     which must agree with its canonical reference effects, and otherwise it fails with
//!     [`KernelInitializationError::InvalidExtension`]. The completion input of a commit or a wait must reference a
//!     scalar token, and otherwise it fails with [`KernelInitializationError::InvalidCopyToken`]. These checks are
//!     structural, so they run once for every extension of the body (and once for a region that is attached in several
//!     places), including for an empty launch and in branches that no analysis reaches. Synchronous operations access
//!     memory like portable operations, and allocations start empty. Asynchronous operations reserve their accesses
//!     like asynchronous copies until a wait consumes their completion allocation, and commits must read a pending
//!     completion of the same region. Releases retire an allocation of the current region that is neither a pending
//!     completion nor reserved by one. Extensions with attached regions are rejected.
//!   - **Completion protocols.** Every pending token carries the completion protocol of the operation that produced
//!     it, which is the portable protocol for an async copy and [`KernelExtension::completion_protocol`] for an
//!     asynchronous extension. A commit or a wait must follow that protocol, and a protocol that requires a commit
//!     admits exactly one commit before the wait. Otherwise the instruction fails with
//!     [`KernelInitializationError::IncompatibleCompletion`] before any reservation is retired or any deferred write is
//!     published, so neither a portable wait nor an extension commit or wait can complete the token of another family.
//!
//! # Control Flow
//!
//! Each branch of a condition is analyzed from the state that enters the condition, and the coverage after the
//! condition is the intersection of the coverage of its branches. A predicate that compares a grid coordinate that is
//! known to be zero in the current analysis case with a dimension constant has a known value, so only the selected
//! branch is analyzed. A `while` loop must have an iteration bound. Its condition region is analyzed from the entering
//! state and its body region once from the state after the condition, and the loop leaves the state after its first
//! condition, so writes that only the body performs are never credited after the loop, which may run no iteration.
//! Entering roots never lose coverage and allocations of the body start anew at each region entry, so the analysis of
//! one iteration covers every later iteration. Other portable operations with attached regions are unsupported.
//!
//! # Carry Axes
//!
//! A write-only parameter whose separable windows repeat only along one sequential grid axis carries its window along
//! that axis, its carry axis (i.e., the only grid axis with an extent greater than 1 that its mapping leaves unused,
//! when that axis is sequential). The grid points that share such a window form a chain ordered by that axis, so the
//! body may initialize the window only at the first step of the chain (e.g., the Pallas form `pl.when(k == 0)`) and
//! accumulate into it at every step. The body is analyzed once for every combination of a first case and a later case
//! of each carry axis. In the first case, the carry coordinate is zero and the parameters carried along that axis start
//! empty. In the later case, they start complete, because the previous step of the chain left the same window
//! complete. The coordinate of a grid axis with extent 1 is zero in every case. Every case must leave every
//! write-only parameter complete, and otherwise it fails with [`KernelInitializationError::IncompleteBody`]. Every
//! carry axis has an extent of at least 2, so a launch with grid points has at most as many cases as grid points.
//!
//! # Launch Qualification
//!
//! The launch must have no scalar prefetch, static grid extents, and static full array shapes, and otherwise it fails
//! with [`KernelInitializationError::UnsupportedLaunch`]. A parameter whose mapping has a separable tiling proof from
//! [`BlockMapping::tiling_axes`](crate::mappings::BlockMapping::tiling_axes) that uses every grid axis at most once is
//! qualified without enumerating the grid. Interpreting its mapping at the maximal grid coordinate checks the bounds of
//! every window and the coverage of a write-only output, and its mutable windows may repeat only along one sequential
//! grid axis, which orders the grid points that share a window. Every other mapping is interpreted at every grid point,
//! which requires the grid to have at most `maximum_programs` points, and otherwise the launch fails with
//! [`KernelInitializationError::QualificationLimit`] before the body is analyzed. The overlapping mutable windows of
//! such a mapping must belong to grid points that the grid orders, and the union of its windows must cover a
//! write-only output. Mutable windows that are only accumulated atomically may overlap freely. Overlapping windows
//! fail with [`KernelInitializationError::OverlappingWindows`] and uncovered outputs with
//! [`KernelInitializationError::IncompleteOutput`]. Finally, every nonempty reading access without a mask that the
//! body analysis recorded on a masked parameter must stay within the elements that are valid in the window of every
//! grid point, and otherwise it fails with [`KernelMemoryError::UnmaskedWindowAccess`].
//!
//! An empty launch (i.e., a grid with an axis of extent 0) never runs its body. Its body still passes the validation
//! of the call, and every write-only output with an element still fails with
//! [`KernelInitializationError::IncompleteOutput`], because no window covers it. The accesses of the body are not
//! analyzed, so no carry case is enumerated and no read needs to be initialized.

// Initialized element intervals are lists of ranges that often hold one interval, which this lint mistakes for a
// range to collect.
#![allow(clippy::single_range_in_vec_init)]

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use ryft_core::{
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformPath, ArraySliceAxis, ArrayType, Atom, AtomId, ComparisonDirection, DataType, Dimension,
    DimensionValue, InstructionId, Operation, ProgramError, ReferenceAccessMode, ReferenceEffect, ReferenceRoot,
    ReferenceType, ReferenceViewOverlap, RegionId, RegionRef, Typed, TypedProgram, ValueId,
};
use thiserror::Error;

use crate::calls::{KernelCallOperation, KernelError};
use crate::grids::{GridError, GridExecution};
use crate::mappings::{BlockMappingError, BoundaryPolicy};
use crate::memory::{KernelMemoryError, WAIT_OPERATION_NAME};
use crate::operations::{
    KernelCompletionProtocol, KernelExtension, KernelExtensionMemory, KernelOperation, admits_completion,
};
use crate::validation::{KernelParameterAccess, KernelReferenceSummary, KernelSwapLowering};

/// A body or launch lacks a definite initialization or disjointness proof.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelInitializationError {
    /// An ordinary access selects invalid elements of a padded parameter window.
    #[error(transparent)]
    Memory(#[from] KernelMemoryError),

    /// The call's type, reference boundary, or grid access contract rejected the body.
    #[error(transparent)]
    Call(#[from] KernelError),

    /// Canonical addressing or dimension binding failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// A mapping could not produce a valid full array window.
    #[error(transparent)]
    Mapping(#[from] BlockMappingError),

    /// A completion input does not name a pending copy token from this region.
    #[error("reference {value:?} is not a pending async copy token from this region")]
    InvalidCopyToken {
        /// Completion input that names no pending copy token.
        value: ValueId,
    },

    /// Only a wait may access a pending copy's token resource.
    #[error("kernel instruction {instruction} accesses copy token {token} without `{WAIT_OPERATION_NAME}`")]
    InvalidCopyTokenAccess {
        /// Instruction that accesses the token.
        instruction: InstructionId,

        /// Canonical completion allocation of the pending copy.
        token: ReferenceRoot,
    },

    /// A commit or wait does not follow the completion protocol of the operation that produced its pending token (i.e.,
    /// it belongs to another protocol, it waits before a commit that the protocol requires, or it repeats a commit
    /// that the protocol requires exactly once). The reservations of the token stay in place.
    #[error("kernel instruction {instruction} does not follow the completion protocol of pending copy {token}")]
    IncompatibleCompletion {
        /// Commit or wait instruction.
        instruction: InstructionId,

        /// Canonical completion allocation of the pending copy.
        token: ReferenceRoot,
    },

    /// A memory access conflicts with an outstanding source or destination reservation.
    #[error("kernel instruction {instruction} conflicts with pending copy {token}")]
    PendingCopyAccess {
        /// Instruction whose access overlaps the reservation.
        instruction: InstructionId,

        /// Canonical completion allocation of the pending copy that holds the reservation.
        token: ReferenceRoot,
    },

    /// A copy source and destination overlap within one allocation.
    #[error("async copy at {instruction} has overlapping source and destination views")]
    OverlappingCopy {
        /// Copy instruction.
        instruction: InstructionId,
    },

    /// An instruction with attached regions accesses, anywhere in those regions, a root that a pending copy reserves,
    /// including its completion token.
    #[error("pending async copies cannot cross the region boundary at {instruction}")]
    CrossRegionCopy {
        /// Instruction that attaches the regions.
        instruction: InstructionId,
    },

    /// A region exits while a copy that it started still owns its reservations.
    #[error("kernel region exits without waiting for async copy {token}")]
    UnwaitedCopy {
        /// Canonical completion allocation of the copy that is still pending.
        token: ReferenceRoot,
    },

    /// An executable loop lacks a finite iteration bound.
    #[error("kernel loop {instruction} requires an iteration bound")]
    UnboundedLoop {
        /// Loop instruction.
        instruction: InstructionId,
    },

    /// A body instruction requires unsupported initialization semantics.
    #[error("kernel initialization does not support operation `{operation}` at {instruction}")]
    UnsupportedOperation {
        /// Canonical operation name.
        operation: &'static str,

        /// Instruction needing a dedicated transfer rule.
        instruction: InstructionId,
    },

    /// An extension classification disagrees with its canonical effects or types.
    #[error("invalid kernel extension memory contract at {instruction}: {message}")]
    InvalidExtension {
        /// Extension instruction.
        instruction: InstructionId,

        /// Description of the disagreement.
        message: String,
    },

    /// An allocation has not been established or was already released.
    #[error("kernel reference {value:?} is not a live allocation")]
    UnavailableReference {
        /// Reference input whose root is not live.
        value: ValueId,
    },

    /// A reading access selects a view that cannot be resolved statically from a root that is not completely
    /// initialized, so it may read an uninitialized element. Selections are per access site, which means that the same
    /// root may be statically selected at one access and dynamically at another.
    #[error("kernel initialization cannot prove the selection of reference input {input_index} at {instruction}")]
    UnknownSelection {
        /// Accessing instruction.
        instruction: InstructionId,

        /// Position among the instruction's inputs of the reference input with a dynamic or unsupported transform path.
        input_index: usize,
    },

    /// An access reads coordinates not definitely initialized before that instruction.
    #[error("kernel instruction {instruction} reads uninitialized elements of {root}")]
    UninitializedRead {
        /// Reading instruction.
        instruction: InstructionId,

        /// Canonical allocation or parameter root.
        root: ReferenceRoot,
    },

    /// A write-only body parameter has uncovered coordinates on some path.
    #[error("kernel body does not definitely initialize write-only parameter {parameter}")]
    IncompleteBody {
        /// Parameter position in the call.
        parameter: usize,
    },

    /// Static qualification cannot resolve launch metadata (e.g., unspecialized scalar prefetch, a dynamic grid
    /// extent, or a dynamic full array shape).
    #[error("kernel initialization cannot statically resolve the {boundary} of the launch")]
    UnsupportedLaunch {
        /// Metadata whose runtime semantics need a later qualification rule.
        boundary: &'static str,
    },

    /// Exact static grid enumeration exceeds the caller's qualification budget.
    #[error("kernel qualification requires {programs} programs, exceeding the limit {maximum}")]
    QualificationLimit {
        /// Static grid size.
        programs: usize,

        /// Enumeration limit provided by the caller.
        maximum: usize,
    },

    /// Two mutable windows may execute on the same full array coordinates.
    #[error("kernel parameter {parameter} has overlapping mutable windows")]
    OverlappingWindows {
        /// Mutable parameter position.
        parameter: usize,
    },

    /// A write-only output contains coordinates absent from every window.
    #[error("kernel grid does not cover write-only parameter {parameter}")]
    IncompleteOutput {
        /// Write-only parameter position.
        parameter: usize,
    },
}

/// Validates definite body writes and complete, disjoint static output windows before memory allocation or execution.
///
/// The supplied region is validated against this exact call. Read-only and read-write inputs begin initialized, and
/// write-only outputs begin empty. A write-only output whose separable windows repeat only along one sequential grid
/// axis, its carry axis, begins empty only at the first step along that axis and begins complete at every later step,
/// because the previous step along that axis left the same window complete. The body is validated for every
/// combination of first and later steps across the carry axes. A condition whose predicate compares a dimension
/// constant with a grid coordinate that is zero (i.e., a carry coordinate at the first step or the coordinate of an
/// axis with extent 1) runs only its selected branch, and every other condition intersects the coverage of its
/// branches. Masked windows preserve fixed logical tile types, with raw reads checked against actual valid elements and
/// output publication clipped to the full array. Unresolved selections never establish writes, and reads through them
/// require their whole root to be initialized. Static separable tiling uses the canonical mapping program to check the
/// maximal coordinate and proves coverage from its extents along each axis. Other mappings use bounded enumeration
/// through the same interpreter. Overlapping mutable windows require grid ordering or exclusively atomic accumulation.
/// Conflicts between atomic and other accesses remain races, because deterministic host traversal does not provide
/// ordering that the grid contract lacks. A launch without grid points never runs its body, so only its call
/// validation and the coverage of its write-only outputs are checked, and any write-only output with an element is
/// uncovered.
///
/// # Parameters
///
///   - `region`: actual attached body being qualified.
///   - `call`: its canonical grid, full array types, accesses, and mappings.
///   - `maximum_programs`: maximum grid size to enumerate when a mapping lacks a separable tiling proof or when its
///     recognized axes repeat a grid axis. Such larger launches are rejected before body traversal. This resource
///     limit does not participate in kernel semantic identity. Only the debug interpreter entry points (e.g.,
///     [`KernelDefinition::interpret_with_limits`](crate::calls::KernelDefinition::interpret_with_limits)) also cap
///     grid points when no enumeration is needed here.
///
/// # Errors
///
/// Returns a [`KernelInitializationError`] when the body or its launch lacks a proof, i.e., when the body fails the
/// type, boundary, or reference validation of `call`, reads an element that is not definitely initialized, leaves a
/// write-only parameter incomplete, misuses a pending asynchronous copy, or has no transfer rule for one of its
/// operations, or when the launch has scalar prefetch or metadata that is not static, needs more than
/// `maximum_programs` grid points to be enumerated, leaves a write-only output uncovered, has overlapping mutable
/// windows that the grid does not order, or reads the padding of a masked window without a mask.
pub fn validate_kernel_initialization<Extension: KernelExtension>(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    call: &KernelCallOperation,
    maximum_programs: usize,
) -> Result<(), KernelInitializationError> {
    validated_kernel_references(region, call, maximum_programs).map(|_| ())
}

/// Validates `region` against `call` exactly like [`validate_kernel_initialization`] and returns the reference summary
/// of the validated body. The host interpreter replays the body with that summary, so it does not validate the body a
/// second time.
pub(crate) fn validated_kernel_references<Extension: KernelExtension>(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    call: &KernelCallOperation,
    maximum_programs: usize,
) -> Result<KernelReferenceSummary, KernelInitializationError> {
    if !call.prefetch_types().is_empty() {
        return Err(KernelInitializationError::UnsupportedLaunch { boundary: "unspecialized scalar prefetch" });
    }
    let extents = call
        .grid()
        .dimensions()
        .iter()
        .map(|dimension| match dimension.extent() {
            Dimension::Static(extent) => Ok(*extent),
            _ => Err(KernelInitializationError::UnsupportedLaunch { boundary: "grid extents" }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let points = call.grid().points(&extents).map_err(KernelError::from)?;
    let tiling_axes = call
        .parameters()
        .iter()
        .map(|parameter| {
            let axes = parameter.mapping().tiling_axes()?;
            let mut used = BTreeSet::new();
            axes.iter().flatten().all(|axis| used.insert(*axis)).then_some(axes)
        })
        .collect::<Vec<_>>();
    if points.len() > maximum_programs && tiling_axes.iter().any(Option::is_none) {
        return Err(KernelInitializationError::QualificationLimit {
            programs: points.len(),
            maximum: maximum_programs,
        });
    }
    let references = call.validate_body(region)?;
    validate_extensions(region)?;
    let carry_axes = call
        .parameters()
        .iter()
        .zip(&tiling_axes)
        .map(|(declaration, axes)| {
            let axes = axes.as_ref().filter(|_| declaration.access() == KernelParameterAccess::WriteOnly)?;
            let [axis] = repeated_axes(&extents, axes)[..] else {
                return None;
            };
            (call.grid().dimensions()[axis].execution() == GridExecution::Sequential).then_some(axis)
        })
        .collect::<Vec<_>>();
    let carried_axes = carry_axes.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
    // A write-only parameter carries its window along its carry axis, the only sequential grid axis with an extent
    // greater than 1 that its separable mapping leaves unused, when it has no other such axis of either execution.
    // Separable windows of distinct coordinates along the used axes are disjoint, so the grid points that share such a
    // window differ only in the carry coordinate, and no other grid point accesses that window. The sequential axis
    // orders those points into a chain `k = 0, 1, 2, ...`. The body is analyzed once for every assignment of a first
    // or a later case to each carry axis. In the first case, the carry coordinate is zero and every parameter carried
    // along that axis starts empty. In the later case, every parameter carried along that axis starts complete. Every
    // analysis requires every write-only parameter to end complete, and the coverage of an entering root never shrinks
    // during a body, because writes only add coverage and branch joins intersect states that each contain the
    // entering coverage. By induction along each chain, the step at `k = 0` leaves its window complete, and every step
    // at `k > 0` enters with the window that its predecessor left complete, which is the later case. Masked windows
    // reload the valid elements that their predecessor published and keep zero padding, so their tiles are
    // deterministic too. Every combination of cases occurs at some grid point, and because every carry axis has an
    // extent of at least 2, a launch with grid points has at most as many cases as grid points. A grid axis with
    // extent 1 always has coordinate zero, which every case knows. An empty launch runs no body, so no case is
    // analyzed, even though its carry axes may be too many to count their cases. The structural extension declarations
    // were already validated above, independently of the cases.
    let cases = if points.len() == 0 {
        0
    } else {
        u32::try_from(carried_axes.len())
            .ok()
            .and_then(|count| 1usize.checked_shl(count))
            .ok_or(KernelError::from(GridError::ProgramCountOverflow))?
    };
    let mut bindings = BTreeMap::new();
    for parameter in 0..call.parameters().len() {
        let root = ReferenceRoot::RegionInput { region: region.id(), input_index: parameter };
        bindings.insert(root, root);
    }
    let mut unmasked_accesses = Vec::new();
    for case in 0..cases {
        let later = |axis: usize| {
            carried_axes
                .iter()
                .position(|&carried| carried == axis)
                .is_some_and(|position| (case >> position) & 1 == 1)
        };
        let zero_coordinates = extents
            .iter()
            .enumerate()
            .filter(|&(axis, &extent)| extent == 1 || (carried_axes.contains(&axis) && !later(axis)))
            .map(|(axis, _)| ValueId::new(region.id(), region.input_ids()[call.parameters().len() + axis]))
            .collect();
        let mut initialization = Initialization {
            references: &references,
            states: BTreeMap::new(),
            types: BTreeMap::new(),
            unmasked_accesses: Vec::new(),
            pending_copies: BTreeMap::new(),
            zero_coordinates,
        };
        for (parameter, declaration) in call.parameters().iter().enumerate() {
            let root = ReferenceRoot::RegionInput { region: region.id(), input_index: parameter };
            let ArrayIrType::Reference(r#type) = declaration.body_type() else { unreachable!() };
            let r#type = r#type.referent().clone();
            let count = ArrayAddressing::new(r#type.clone())?.element_count();
            let empty =
                declaration.access() == KernelParameterAccess::WriteOnly && !carry_axes[parameter].is_some_and(later);
            initialization.states.insert(root, if empty || count == 0 { vec![] } else { vec![0..count] });
            initialization.types.insert(root, r#type);
        }
        initialization.region(region, &bindings)?;
        for (parameter, declaration) in call.parameters().iter().enumerate() {
            let root = ReferenceRoot::RegionInput { region: region.id(), input_index: parameter };
            let count = ArrayAddressing::new(initialization.types[&root].clone())?.element_count();
            if declaration.access() == KernelParameterAccess::WriteOnly
                && !contains(&initialization.states[&root], &(0..count))
            {
                return Err(KernelInitializationError::IncompleteBody { parameter });
            }
        }
        unmasked_accesses.extend(initialization.unmasked_accesses);
    }

    let mut valid_shapes = BTreeMap::new();
    for (parameter, declaration) in call.parameters().iter().enumerate() {
        let mapping = declaration.mapping();
        let summary = references.parameter(parameter).unwrap();
        let mutable = summary.is_mutated();
        let atomic_only = mutable && summary.modes().iter().all(|mode| *mode == ReferenceAccessMode::AtomicAccumulate);
        let mut valid_shape = mapping.block_shape().to_vec();
        let r#type = declaration.r#type();
        let shape = r#type
            .static_shape()
            .ok_or(KernelInitializationError::UnsupportedLaunch { boundary: "full array shapes" })?;
        let addressing = ArrayAddressing::new(r#type.into_owned())?;
        if let Some(axes) = &tiling_axes[parameter] {
            if points.len() == 0 {
                if declaration.access() == KernelParameterAccess::WriteOnly && addressing.element_count() != 0 {
                    return Err(KernelInitializationError::IncompleteOutput { parameter });
                }
                continue;
            }
            // Every recognized start is monotone. Interpreting the maximal coordinate preserves canonical input
            // binding, checked arithmetic, and bounds diagnostics without enumerating the launch.
            let inputs = mapping
                .program()
                .input_types()
                .into_owned()
                .into_iter()
                .zip(&extents)
                .map(|(r#type, extent)| {
                    let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                    DimensionValue::new(r#type, extent - 1).map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let last = mapping.evaluate(&inputs, shape.dimensions())?;
            if mapping.boundary_policy() == BoundaryPolicy::Masked {
                let ArrayReferenceTransform::Slice { axes } = last.valid_transform() else { unreachable!() };
                valid_shapes.insert(
                    ReferenceRoot::RegionInput { region: region.id(), input_index: parameter },
                    axes.iter().map(|axis| axis.size()).collect::<Vec<_>>(),
                );
            }
            let empty = addressing.element_count() == 0 || mapping.block_shape().contains(&0);
            if mutable && !atomic_only && !empty {
                let repeated_axes = repeated_axes(&extents, axes);
                // Equal windows differ only on unused axes. A single sequential axis totally orders those points, and
                // two varying sequential axes already contain incomparable opposite moves.
                if repeated_axes.len() > 1
                    || repeated_axes
                        .first()
                        .is_some_and(|&axis| call.grid().dimensions()[axis].execution() == GridExecution::Parallel)
                {
                    return Err(KernelInitializationError::OverlappingWindows { parameter });
                }
            }
            if declaration.access() == KernelParameterAccess::WriteOnly
                && addressing.element_count() != 0
                && last
                    .starts()
                    .iter()
                    .zip(mapping.block_shape())
                    .zip(shape.dimensions())
                    .any(|((&start, &block), &extent)| (start + block).min(extent) != extent)
            {
                return Err(KernelInitializationError::IncompleteOutput { parameter });
            }
            continue;
        }
        let mut covered = Vec::new();
        let mut previous_windows: Vec<(Vec<usize>, Vec<Range<usize>>)> = Vec::new();
        for point in points.clone() {
            let inputs = mapping
                .program()
                .input_types()
                .into_owned()
                .into_iter()
                .zip(&point)
                .map(|(r#type, &extent)| {
                    let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                    DimensionValue::new(r#type, extent).map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let window = mapping.evaluate(&inputs, shape.dimensions())?;
            let ArrayReferenceTransform::Slice { axes } = window.valid_transform() else { unreachable!() };
            for (valid, axis) in valid_shape.iter_mut().zip(axes) {
                *valid = (*valid).min(axis.size());
            }
            if mutable {
                let ranges = addressing.ranges(axes)?.map(|range| range.elements()).collect::<Vec<_>>();
                if !atomic_only {
                    for (previous_point, previous_ranges) in &previous_windows {
                        if !call.grid().points_are_ordered(previous_point, &point)
                            && previous_ranges.iter().any(|previous| {
                                ranges.iter().any(|range| previous.start < range.end && range.start < previous.end)
                            })
                        {
                            return Err(KernelInitializationError::OverlappingWindows { parameter });
                        }
                    }
                }
                for range in &ranges {
                    insert(&mut covered, range.clone());
                }
                if !atomic_only {
                    previous_windows.push((point, ranges));
                }
            }
        }
        if mapping.boundary_policy() == BoundaryPolicy::Masked && points.len() != 0 {
            valid_shapes
                .insert(ReferenceRoot::RegionInput { region: region.id(), input_index: parameter }, valid_shape);
        }
        if declaration.access() == KernelParameterAccess::WriteOnly
            && !contains(&covered, &(0..addressing.element_count()))
        {
            return Err(KernelInitializationError::IncompleteOutput { parameter });
        }
    }
    for (root, operation, axes) in &unmasked_accesses {
        if let Some(valid_shape) = valid_shapes.get(root)
            && !axes.iter().any(|axis| axis.size() == 0)
            && axes.iter().zip(valid_shape).any(|(axis, &valid)| axis.start() + axis.size() > valid)
        {
            return Err(KernelMemoryError::UnmaskedWindowAccess { operation }.into());
        }
    }

    Ok(references)
}

/// Ordered coverage state, keyed by canonical roots after substituting the inputs of each particular attachment.
struct Initialization<'o> {
    /// Canonical access and view analysis of the whole closure of the body.
    references: &'o KernelReferenceSummary,

    /// Definitely initialized logical intervals for every encountered root.
    states: BTreeMap<ReferenceRoot, Vec<Range<usize>>>,

    /// Static root array types used to interpret canonical selections.
    types: BTreeMap<ReferenceRoot, ArrayType>,

    /// Ordinary reads and read-write accesses checked against actual launch validity after body initialization. Swaps
    /// lowered to stores never read, so they are not recorded.
    unmasked_accesses: Vec<(ReferenceRoot, &'static str, Vec<ArraySliceAxis>)>,

    /// Outstanding asynchronous access reservations keyed by their canonical completion allocation.
    pending_copies: BTreeMap<ReferenceRoot, PendingCopy>,

    /// Grid coordinate inputs of the body's entry region that are zero in this analysis case.
    zero_coordinates: BTreeSet<ValueId>,
}

impl Initialization<'_> {
    /// Visits instructions in execution order, in which releases retire allocations and joins retain definite
    /// coverage. Copies that are pending when the region is entered stay pending throughout it, and every copy that the
    /// region starts must complete in it, so the pending copies on exit are exactly those on entry.
    fn region<Extension: KernelExtension>(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(), KernelInitializationError> {
        let entering_copies = self.pending_copies.keys().copied().collect::<BTreeSet<_>>();
        for (index, instruction) in region.instructions().iter().enumerate() {
            let id = InstructionId::new(region.id(), index);
            // Attached regions may run while copies are pending, as long as no access anywhere in them touches a root
            // that a pending copy reserves, including its completion token. This is sound because the attached regions
            // then neither observe nor modify reserved storage, and they cannot complete a copy started outside them,
            // since waits and commits only accept tokens of their own region and the token root is untouched. The
            // check is at root granularity because the transitive accesses of an instruction carry no selections.
            if !instruction.regions().is_empty()
                && !self.pending_copies.is_empty()
                && let Some(access) = self.references.analysis().analysis().transitive_access(id)
                && access.roots().any(|root| {
                    let root = bindings.get(&root).copied().unwrap_or(root);
                    self.pending_copies.iter().any(|(&token, copy)| {
                        token == root || copy.accesses.iter().any(|(reserved, _, _)| *reserved == root)
                    })
                })
            {
                return Err(KernelInitializationError::CrossRegionCopy { instruction: id });
            }
            if matches!(instruction.operation(), KernelOperation::AsyncCopy(_)) {
                let (source, source_axes) =
                    self.selection(id, 0, ValueId::new(region.id(), instruction.inputs()[0]), bindings)?;
                let (destination, destination_axes) =
                    self.selection(id, 1, ValueId::new(region.id(), instruction.inputs()[1]), bindings)?;
                if source == destination {
                    // A selection that is not statically known may select any element, so it overlaps every
                    // nonempty selection of the same root.
                    let source_path =
                        ArrayReferenceTransformPath::<ValueId>::root().with_transform(ArrayReferenceTransform::Slice {
                            axes: source_axes.clone().unwrap_or_else(|| self.whole(source)),
                        });
                    let destination_path =
                        ArrayReferenceTransformPath::<ValueId>::root().with_transform(ArrayReferenceTransform::Slice {
                            axes: destination_axes.clone().unwrap_or_else(|| self.whole(destination)),
                        });
                    let r#type = ArrayIrType::Reference(ReferenceType::new(self.types[&source].clone()));
                    if source_path.overlap(&destination_path, &r#type) != ReferenceViewOverlap::Disjoint {
                        return Err(KernelInitializationError::OverlappingCopy { instruction: id });
                    }
                }
                self.accesses(region, id, bindings, None, true)?;
                let token = ReferenceRoot::Allocation { instruction: id, output_index: 0 };
                self.pending_copies.insert(
                    token,
                    PendingCopy {
                        accesses: vec![
                            (source, source_axes, ReferenceAccessMode::Read),
                            (destination, destination_axes, ReferenceAccessMode::Write),
                        ],
                        protocol: None,
                        committed: false,
                    },
                );
                self.types.insert(token, ArrayType::scalar(DataType::Token));
                self.states.insert(token, vec![0..1]);
                continue;
            }
            if matches!(instruction.operation(), KernelOperation::Wait(_)) {
                let value = ValueId::new(region.id(), instruction.inputs()[0]);
                let original = self.references.analysis().analysis().root_of(value).unwrap();
                let token = bindings.get(&original).copied().unwrap_or(original);
                if !matches!(
                    token,
                    ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region.id(),
                ) {
                    return Err(KernelInitializationError::InvalidCopyToken { value });
                }
                self.complete(id, value, token, None, false)?;
                continue;
            }
            if let KernelOperation::Scratch(operation) = instruction.operation() {
                let root = ReferenceRoot::Allocation { instruction: id, output_index: 0 };
                self.states.insert(root, Vec::new());
                self.types.insert(root, operation.referent().clone());
                continue;
            }
            if matches!(instruction.operation(), KernelOperation::WindowMask(_)) {
                // Window validity accesses no storage, so it reads no uninitialized element, never conflicts with a
                // pending copy, and is not an unmasked access of a padded window.
                continue;
            }
            let mask_index = match instruction.operation() {
                KernelOperation::MaskedLoad(_) => Some(1),
                KernelOperation::MaskedStore(_) | KernelOperation::MaskedSwap(_) => Some(2),
                _ => None,
            };
            if let Some(mask_index) = mask_index {
                self.accesses(region, id, bindings, Some(instruction.inputs()[mask_index]), false)?;
                continue;
            }
            if matches!(instruction.operation(), KernelOperation::TileLoad(_)) {
                self.accesses(region, id, bindings, None, false)?;
                continue;
            }
            if let KernelOperation::Extension(operation) = instruction.operation() {
                self.extension(region, id, operation, bindings)?;
                continue;
            }
            let KernelOperation::Portable(operation) = instruction.operation() else {
                return Err(KernelInitializationError::UnsupportedOperation {
                    operation: instruction.operation().name(),
                    instruction: id,
                });
            };
            if !instruction.regions().is_empty() {
                match operation {
                    ArrayIrOperation::Condition(_) => {
                        // Every branch leaves the pending copies exactly as it found them, so only coverage is joined.
                        // A predicate that is known in this analysis case runs only its branch, and the true branch
                        // is attached first.
                        let taken =
                            self.known_predicate(region, instruction.inputs()[0]).map(|known| usize::from(!known));
                        let entering = self.states.clone();
                        let mut joined = None;
                        for (position, &nested) in instruction.regions().iter().enumerate() {
                            if taken.is_some_and(|taken| taken != position) {
                                continue;
                            }
                            self.states = entering.clone();
                            let nested_bindings = self.bindings(id, position, bindings);
                            self.region(region.with_id(nested).unwrap(), &nested_bindings)?;
                            joined = Some(match joined {
                                None => self.states.clone(),
                                Some(previous) => intersect_states(previous, &self.states),
                            });
                        }
                        self.states = joined.unwrap_or(entering);
                    }
                    ArrayIrOperation::While(operation) => {
                        if operation.iteration_bound().is_none() {
                            return Err(KernelInitializationError::UnboundedLoop { instruction: id });
                        }
                        // The first condition always executes, including on the path that runs no body. Existing
                        // lifetime validation forbids consuming entering roots, and stores only add coverage, so
                        // validating the first body from this state suffices for the initialization safety of
                        // subsequent iterations. This is the monotone fixed point subset, because local allocations
                        // reset at each region entry, entering roots never lose initialized elements, and reads are
                        // checked before each write. The path that runs no iteration prevents crediting writes that
                        // only the body performs after the loop.
                        let condition_bindings = self.bindings(id, 0, bindings);
                        self.region(region.with_id(instruction.regions()[0]).unwrap(), &condition_bindings)?;
                        let condition_state = self.states.clone();
                        let body_bindings = self.bindings(id, 1, bindings);
                        self.region(region.with_id(instruction.regions()[1]).unwrap(), &body_bindings)?;
                        self.states = condition_state;
                    }
                    _ => {
                        return Err(KernelInitializationError::UnsupportedOperation {
                            operation: operation.name(),
                            instruction: id,
                        });
                    }
                }
                continue;
            }
            if matches!(operation, ArrayIrOperation::ReferenceNew(_)) {
                let root = ReferenceRoot::Allocation { instruction: id, output_index: 0 };
                let r#type = region.atoms()[instruction.outputs()[0].index()].r#type();
                let ArrayIrType::Reference(r#type) = r#type.as_ref() else { unreachable!() };
                let r#type = r#type.referent().clone();
                let count = ArrayAddressing::new(r#type.clone())?.element_count();
                self.states.insert(root, if count == 0 { vec![] } else { vec![0..count] });
                self.types.insert(root, r#type);
            }
            self.accesses(region, id, bindings, None, false)?;
        }
        // Copies pending on entry cannot complete in this region, so only copies that it started can be reported.
        if let Some(&token) = self.pending_copies.keys().find(|token| !entering_copies.contains(token)) {
            return Err(KernelInitializationError::UnwaitedCopy { token });
        }
        Ok(())
    }

    /// Applies a commit (when `commit` is set) or a wait that follows `protocol` (i.e., [`None`] for the portable
    /// wait) to the pending copy of `token`, which the completion input `value` of instruction `id` names. Both must
    /// follow the protocol of the producer, which is checked before anything changes. A commit only records that it
    /// observed the token. A wait retires the token and its reservations and publishes the deferred writes of the
    /// copy, where a write whose selection is not statically known may select any element, so it establishes no
    /// coverage.
    fn complete(
        &mut self,
        id: InstructionId,
        value: ValueId,
        token: ReferenceRoot,
        protocol: Option<KernelCompletionProtocol>,
        commit: bool,
    ) -> Result<(), KernelInitializationError> {
        let pending =
            self.pending_copies.get_mut(&token).ok_or(KernelInitializationError::InvalidCopyToken { value })?;
        if !admits_completion(pending.protocol, pending.committed, protocol, commit) {
            return Err(KernelInitializationError::IncompatibleCompletion { instruction: id, token });
        }
        if commit {
            pending.committed = true;
            return Ok(());
        }
        let pending = self.pending_copies.remove(&token).unwrap();
        self.states.remove(&token);
        for (root, axes, mode) in pending.accesses {
            if mode == ReferenceAccessMode::Read {
                continue;
            }
            let Some(axes) = axes else {
                continue;
            };
            let addressing = ArrayAddressing::new(self.types[&root].clone())?;
            let state = self.states.get_mut(&root).unwrap();
            for range in addressing.ranges(&axes)? {
                insert(state, range.elements());
            }
        }
        Ok(())
    }

    /// Applies extension timing to canonical reference effects without redefining access windows. The structural
    /// declarations of the extension were validated by [`validate_extension_declarations`] before any analysis case.
    fn extension<Extension: KernelExtension>(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        id: InstructionId,
        operation: &Extension,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(), KernelInitializationError> {
        let instruction = &region.instructions()[id.index()];
        let (semantics, accesses) = validate_extension_declarations(region, id, operation)?;
        match semantics {
            KernelExtensionMemory::Synchronous => {
                self.accesses(region, id, bindings, None, false)?;
            }
            KernelExtensionMemory::Allocation { output_index }
            | KernelExtensionMemory::Asynchronous { completion_output_index: output_index } => {
                let r#type = region.atoms()[instruction.outputs()[output_index].index()].r#type();
                let ArrayIrType::Reference(reference) = r#type.as_ref() else { unreachable!() };
                let root = ReferenceRoot::Allocation { instruction: id, output_index };
                if matches!(semantics, KernelExtensionMemory::Asynchronous { .. }) {
                    self.accesses(region, id, bindings, None, true)?;
                    let reservations = accesses
                        .iter()
                        .map(|(input, mode)| {
                            self.selection(
                                id,
                                *input,
                                ValueId::new(region.id(), instruction.inputs()[*input]),
                                bindings,
                            )
                            .map(|(root, axes)| (root, axes, *mode))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.pending_copies.insert(
                        root,
                        PendingCopy {
                            accesses: reservations,
                            protocol: Some(operation.completion_protocol()),
                            committed: false,
                        },
                    );
                }
                self.types.insert(root, reference.referent().clone());
                self.states.insert(root, Vec::new());
            }
            KernelExtensionMemory::Commit { completion_input_index: input_index } => {
                let value = ValueId::new(region.id(), instruction.inputs()[input_index]);
                let (root, _) = self.selection(id, input_index, value, bindings)?;
                if !matches!(root, ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region.id())
                {
                    return Err(KernelInitializationError::InvalidCopyToken { value });
                }
                self.complete(id, value, root, Some(operation.completion_protocol()), true)?;
            }
            KernelExtensionMemory::Wait { completion_input_index: input_index }
            | KernelExtensionMemory::Release { input_index } => {
                let value = ValueId::new(region.id(), instruction.inputs()[input_index]);
                let (root, _) = self.selection(id, input_index, value, bindings)?;
                if !matches!(root, ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region.id())
                {
                    return Err(KernelInitializationError::InvalidExtension {
                        instruction: id,
                        message: "wait or release requires an allocation from the current region".to_owned(),
                    });
                }
                if matches!(semantics, KernelExtensionMemory::Wait { .. }) {
                    self.complete(id, value, root, Some(operation.completion_protocol()), false)?;
                } else {
                    if self.pending_copies.contains_key(&root) {
                        return Err(KernelInitializationError::InvalidCopyTokenAccess { instruction: id, token: root });
                    }
                    if let Some((&token, _)) = self
                        .pending_copies
                        .iter()
                        .find(|(_, pending)| pending.accesses.iter().any(|(reserved, _, _)| *reserved == root))
                    {
                        return Err(KernelInitializationError::PendingCopyAccess { instruction: id, token });
                    }
                }
                self.states.remove(&root);
            }
        }
        Ok(())
    }

    /// Checks selected reads and records definite writes using the operation's optional mask input. Unknown
    /// masks require the complete selected view for reads and establish no definite write coverage, except for the two
    /// provably complete masked stores that [`Self::writes_whole_selection`] recognizes. Selections that are not
    /// statically known require the complete root for reads and establish no write coverage.
    ///
    /// Every read of the instruction is checked against the coverage that enters it, and its definite writes are
    /// published together only after all of its accesses are checked. An instruction may perform its accesses in any
    /// order (e.g., an extension that copies a reference into itself may write its destination before it reads its
    /// source), so a write of one access never initializes an element for another access of the same instruction.
    fn accesses<Extension: KernelExtension>(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        id: InstructionId,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
        mask: Option<AtomId>,
        deferred: bool,
    ) -> Result<(), KernelInitializationError> {
        let known_mask = mask
            .and_then(|mask| match &region.atoms()[mask.index()] {
                Atom::Constant(ArrayIrValue::Array(mask)) => Some(mask.elements::<bool>()),
                _ => None,
            })
            .transpose()?;
        let instruction = &region.instructions()[id.index()];
        let mut writes = Vec::new();
        for access in self
            .references
            .analysis()
            .analysis()
            .access_modes()
            .iter()
            .filter(|access| access.instruction() == id)
        {
            let value = ValueId::new(region.id(), instruction.inputs()[access.input_index()]);
            let (root, selection) = self.selection(id, access.input_index(), value, bindings)?;
            if self.pending_copies.contains_key(&root) {
                return Err(KernelInitializationError::InvalidCopyTokenAccess { instruction: id, token: root });
            }
            let writes_only = access.mode() == ReferenceAccessMode::Write
                || self.references.swap_lowering(id) == Some(KernelSwapLowering::Store);
            let axes = selection.clone().unwrap_or_else(|| self.whole(root));
            if mask.is_none() && !writes_only {
                self.unmasked_accesses.push((root, instruction.operation().name(), axes.clone()));
            }
            let addressing = ArrayAddressing::new(self.types[&root].clone())?;
            if selection.is_none() {
                // The selection is not statically known, so the access may select any element of its root, and a
                // mask cannot be related to root elements. Treating the access as one of the whole root is sound
                // because a read is admitted only when every element of the root is initialized, a write establishes
                // no coverage, and both the pending copy reservations and the masked window check of the launch see
                // the whole root.
                for range in addressing.ranges(&axes)? {
                    self.validate_reservations(root, &range.elements(), access.mode(), id)?;
                }
                if !writes_only && !contains(&self.states[&root], &(0..addressing.element_count())) {
                    return Err(KernelInitializationError::UnknownSelection {
                        instruction: id,
                        input_index: access.input_index(),
                    });
                }
                continue;
            }
            let complete = mask.is_some_and(|mask| {
                known_mask.is_none() && self.writes_whole_selection(region, id, mask, root, &axes, bindings)
            });
            let establishes_write = writes_only && (mask.is_none() || known_mask.is_some() || complete) && !deferred;
            let mut mask_offset = 0;
            for range in addressing.ranges(&axes)? {
                let range = range.elements();
                let selected = if let Some(mask) = &known_mask {
                    let end = mask_offset + range.len();
                    let selected = masked_ranges(range, &mask[mask_offset..end]);
                    mask_offset = end;
                    selected
                } else {
                    vec![range]
                };
                for range in selected {
                    self.validate_reservations(root, &range, access.mode(), id)?;
                    if !writes_only && !contains(&self.states[&root], &range) {
                        return Err(KernelInitializationError::UninitializedRead { instruction: id, root });
                    }
                    if establishes_write {
                        writes.push((root, range));
                    }
                }
            }
        }
        for (root, range) in writes {
            insert(self.states.get_mut(&root).unwrap(), range);
        }
        Ok(())
    }

    /// Resolves a reference input to its canonical allocation and the root coordinate slice axes that its path selects,
    /// which are [`None`] when the path has a dynamic or otherwise unsupported transform. Such a selection may select
    /// any element of its root.
    fn selection(
        &self,
        instruction: InstructionId,
        input_index: usize,
        value: ValueId,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(ReferenceRoot, Option<Vec<ArraySliceAxis>>), KernelInitializationError> {
        let original = self.references.analysis().analysis().root_of(value).unwrap();
        let root = bindings.get(&original).copied().unwrap_or(original);
        let axes = self.references.path(instruction, input_index).unwrap().root_slice_axes(&self.types[&root]);
        if !self.states.contains_key(&root) {
            return Err(KernelInitializationError::UnavailableReference { value });
        }
        Ok((root, axes))
    }

    /// Returns the root coordinate slice axes that select every element of `root`, which stand in for a selection that
    /// is not statically known.
    fn whole(&self, root: ReferenceRoot) -> Vec<ArraySliceAxis> {
        // Every recorded root type is addressable, so its shape is static.
        self.types[&root]
            .static_shape()
            .unwrap()
            .dimensions()
            .iter()
            .map(|&extent| ArraySliceAxis::new(0, extent, 1))
            .collect()
    }

    /// Rejects accesses to pending destinations and mutations of pending sources, including through masked views. A
    /// reservation whose selection is not statically known reserves its whole root.
    fn validate_reservations(
        &self,
        root: ReferenceRoot,
        range: &Range<usize>,
        mode: ReferenceAccessMode,
        instruction: InstructionId,
    ) -> Result<(), KernelInitializationError> {
        for (&token, copy) in &self.pending_copies {
            for (reserved_root, axes, reserved_mode) in &copy.accesses {
                let conflicts = *reserved_mode != ReferenceAccessMode::Read || mode != ReferenceAccessMode::Read;
                if *reserved_root == root && conflicts {
                    let addressing = ArrayAddressing::new(self.types[&root].clone())?;
                    let axes = axes.clone().unwrap_or_else(|| self.whole(root));
                    if addressing.ranges(&axes)?.any(|reserved| {
                        let reserved = reserved.elements();
                        range.start < reserved.end && reserved.start < range.end
                    }) {
                        return Err(KernelInitializationError::PendingCopyAccess { instruction, token });
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolves only this particular region attachment, keeping shared regions' callers independent.
    fn bindings(
        &self,
        instruction: InstructionId,
        region_index: usize,
        parent: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> BTreeMap<ReferenceRoot, ReferenceRoot> {
        self.references
            .analysis()
            .analysis()
            .region_input_bindings()
            .iter()
            .filter(|binding| binding.instruction() == instruction && binding.region_index() == region_index)
            .map(|binding| {
                let local = self.references.analysis().analysis().root_of(binding.input()).unwrap();
                (local, parent.get(&binding.root()).copied().unwrap_or(binding.root()))
            })
            .collect()
    }

    /// Returns the value of `predicate` when it is the result of a dimension comparison in `region` between a grid
    /// coordinate that is zero in this analysis case and a dimension constant, and [`None`] for every other predicate.
    fn known_predicate<Extension: KernelExtension>(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        predicate: AtomId,
    ) -> Option<bool> {
        let instruction = region.instructions().iter().find(|instruction| instruction.outputs() == [predicate])?;
        let KernelOperation::Portable(ArrayIrOperation::Compare(operation)) = instruction.operation() else {
            return None;
        };
        let [left, right] = *instruction.inputs() else {
            return None;
        };
        let zero = |atom: AtomId| self.zero_coordinates.contains(&ValueId::new(region.id(), atom));
        let constant = |atom: AtomId| match &region.atoms()[atom.index()] {
            Atom::Constant(ArrayIrValue::Dimension(value)) => Some(value.extent()),
            _ => None,
        };
        let (left, right) = if zero(left) {
            (0, constant(right)?)
        } else if zero(right) {
            (constant(left)?, 0)
        } else {
            return None;
        };
        Some(match operation.direction() {
            ComparisonDirection::Equal => left == right,
            ComparisonDirection::NotEqual => left != right,
            ComparisonDirection::LessThan => left < right,
            ComparisonDirection::LessThanOrEqual => left <= right,
            ComparisonDirection::GreaterThan => left > right,
            ComparisonDirection::GreaterThanOrEqual => left >= right,
        })
    }

    /// Returns whether the instruction at `id` is a [`MaskedStoreOperation`](crate::memory::MaskedStoreOperation)
    /// whose runtime `mask` provably selects every element of its statically known selection `axes` of `root`, except
    /// for padding elements that are never published. Only two forms qualify.
    ///
    ///   - The mask is the output of a [`WindowMaskOperation`](crate::memory::WindowMaskOperation) applied, earlier in
    ///     the same region, to the very reference value that the store writes through, and the store applies no
    ///     transforms, so its selection is that whole reference.
    ///   - An earlier masked store of the same region writes the same selection of the same root under the complement
    ///     of the mask, i.e., one of the two masks is the output of a logical `not`, earlier in the same region, whose
    ///     input is the other mask.
    ///
    /// Masks that merely derive from these forms through other operations, the validity of another reference, stores
    /// through views, and halves in different regions never qualify.
    fn writes_whole_selection<Extension: KernelExtension>(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        id: InstructionId,
        mask: AtomId,
        root: ReferenceRoot,
        axes: &[ArraySliceAxis],
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> bool {
        let earlier = &region.instructions()[..id.index()];
        let store = &region.instructions()[id.index()];
        let KernelOperation::MaskedStore(operation) = store.operation() else {
            return false;
        };
        let producer = |atom: AtomId| earlier.iter().find(|instruction| instruction.outputs().contains(&atom));

        // A store under the validity mask of the reference that it writes through writes every valid element of that
        // reference, and the elements that it leaves unwritten are padding of a masked parameter window, which is never
        // published. A later unmasked read of that padding is still rejected by the launch check of unmasked accesses,
        // and a later masked read has its mask intersected with the window validity by the interpreter and by every
        // compiler adapter, so crediting the whole reference never lets a body observe an unwritten element. Atoms are
        // local to their region, so the same atom denotes the same reference value in the store and in the
        // `window_mask`, and an empty transform path selects that whole reference.
        if operation.transforms().is_empty()
            && producer(mask).is_some_and(|instruction| {
                matches!(instruction.operation(), KernelOperation::WindowMask(_))
                    && instruction.inputs()[0] == store.inputs()[0]
            })
        {
            return true;
        }

        // Two stores under the masks `m` and `not(m)` together write every element of their common selection, because
        // the masked operations require both masks to have exactly the selected referent shape, `m` is one immutable
        // value, and every element is selected by exactly one of the two masks. Equal root slice axes select the same
        // elements in the same row-major order, since the masks force equal view shapes and the canonical transforms
        // only select ranges and remove unit axes. Both halves are instructions of this region, so both execute in
        // every invocation of it before the later one is credited, and no half can pair with a half in a nested or
        // enclosing region. The pairing is recomputed from the instruction list and bindings of the current
        // invocation, so a region that is analyzed more than once (e.g., a shared branch) carries no pending half from
        // one analysis into another.
        let negates = |negation: AtomId, value: AtomId| {
            producer(negation).is_some_and(|instruction| {
                matches!(
                    instruction.operation(),
                    KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Not(_))),
                ) && instruction.inputs()[0] == value
            })
        };
        earlier.iter().enumerate().any(|(index, instruction)| {
            if !matches!(instruction.operation(), KernelOperation::MaskedStore(_)) {
                return false;
            }
            let other = instruction.inputs()[2];
            if !negates(mask, other) && !negates(other, mask) {
                return false;
            }
            let value = ValueId::new(region.id(), instruction.inputs()[0]);
            let original = self.references.analysis().analysis().root_of(value).unwrap();
            bindings.get(&original).copied().unwrap_or(original) == root
                && self
                    .references
                    .path(InstructionId::new(region.id(), index), 0)
                    .and_then(|path| path.root_slice_axes(&self.types[&root]))
                    .is_some_and(|other| other == axes)
        })
    }
}

/// Canonical accesses reserved until a completion token is consumed.
struct PendingCopy {
    /// Root, root coordinate slice axes, and declared access mode for each outstanding access. Slice axes are [`None`]
    /// for a selection that is not statically known, which reserves its whole root.
    accesses: Vec<(ReferenceRoot, Option<Vec<ArraySliceAxis>>, ReferenceAccessMode)>,

    /// Completion protocol of the producer, which is [`None`] for a portable async copy.
    protocol: Option<KernelCompletionProtocol>,

    /// Whether a commit has observed the token.
    committed: bool,
}

/// Validates the structural memory declarations of every extension operation in `region` and in its attached regions
/// with [`validate_extension_declarations`]. These declarations do not depend on the grid point or on which
/// instructions an analysis case reaches, so they are validated once, including for an empty launch and inside a branch
/// that no analysis case takes. A region that is attached in several places is also validated once.
fn validate_extensions<Extension: KernelExtension>(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
) -> Result<(), KernelInitializationError> {
    fn visit<Extension: KernelExtension>(
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        visited: &mut BTreeSet<RegionId>,
    ) -> Result<(), KernelInitializationError> {
        for (index, instruction) in region.instructions().iter().enumerate() {
            if let KernelOperation::Extension(operation) = instruction.operation() {
                validate_extension_declarations(region, InstructionId::new(region.id(), index), operation)?;
            }
            for &nested in instruction.regions() {
                if visited.insert(nested) {
                    visit(region.with_id(nested).unwrap(), visited)?;
                }
            }
        }
        Ok(())
    }
    visit(region, &mut BTreeSet::from([region.id()]))
}

/// Validates that the extension `operation` at instruction `id` of `region` has no attached regions, that its
/// [`KernelExtension::memory_semantics`] agree with its canonical reference effects, its reference output aliases, and
/// the types of its declared allocation, and that the completion input of a commit or a wait references a scalar token
/// (and otherwise fails with [`KernelInitializationError::InvalidCopyToken`]), and returns those semantics with its
/// declared accesses in declaration order.
fn validate_extension_declarations<Extension: KernelExtension>(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    id: InstructionId,
    operation: &Extension,
) -> Result<(KernelExtensionMemory, Vec<(usize, ReferenceAccessMode)>), KernelInitializationError> {
    let instruction = &region.instructions()[id.index()];
    let invalid =
        |message: &str| KernelInitializationError::InvalidExtension { instruction: id, message: message.to_owned() };
    if !instruction.regions().is_empty() {
        return Err(invalid("nested extension regions are unsupported"));
    }
    let semantics = operation.memory_semantics().map_err(|error| invalid(&error.to_string()))?;
    let effects = operation.effects();
    let declarations = effects.reference_effects();
    let allocations = declarations
        .iter()
        .filter_map(|effect| match effect {
            ReferenceEffect::Allocate { output_index } => Some(*output_index),
            _ => None,
        })
        .collect::<Vec<_>>();
    let accesses = declarations
        .iter()
        .filter_map(|effect| match effect {
            ReferenceEffect::Access { input_index, mode } => Some((*input_index, *mode)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let aliases = (0..instruction.outputs().len())
        .any(|output_index| operation.reference_output_identity_input(output_index).is_some());
    let validate_completion_input = |input: AtomId| {
        let token = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)));
        if region.atoms()[input.index()].r#type().as_ref() == &token {
            Ok(())
        } else {
            Err(KernelInitializationError::InvalidCopyToken { value: ValueId::new(region.id(), input) })
        }
    };
    match semantics {
        KernelExtensionMemory::Synchronous => {
            if !allocations.is_empty() || accesses.iter().any(|(_, mode)| *mode == ReferenceAccessMode::Consume) {
                return Err(invalid("synchronous operations cannot allocate or consume references"));
            }
        }
        KernelExtensionMemory::Allocation { output_index }
        | KernelExtensionMemory::Asynchronous { completion_output_index: output_index } => {
            if allocations != [output_index] || aliases {
                return Err(invalid("classification requires exactly its declared allocation and no aliases"));
            }
            let asynchronous = matches!(semantics, KernelExtensionMemory::Asynchronous { .. });
            if (!asynchronous && !accesses.is_empty())
                || accesses.iter().any(|(_, mode)| *mode == ReferenceAccessMode::Consume)
            {
                return Err(invalid("allocation accesses do not match its initialization contract"));
            }
            let output =
                instruction.outputs().get(output_index).ok_or_else(|| invalid("allocation output is absent"))?;
            let r#type = region.atoms()[output.index()].r#type();
            let ArrayIrType::Reference(reference) = r#type.as_ref() else {
                return Err(invalid("allocation output is not a reference"));
            };
            let referent = reference.referent().clone();
            if asynchronous && referent != ArrayType::scalar(DataType::Token) {
                return Err(invalid("completion allocation must reference a scalar token"));
            }
            if !asynchronous && referent.data_type() == DataType::Token {
                return Err(invalid("ordinary allocations cannot create completion tokens"));
            }
            ArrayAddressing::new(referent)?;
        }
        KernelExtensionMemory::Commit { completion_input_index: input_index } => {
            if !allocations.is_empty() || accesses != [(input_index, ReferenceAccessMode::Read)] || aliases {
                return Err(invalid("commit must read exactly its declared completion reference"));
            }
            let input = instruction.inputs().get(input_index).ok_or_else(|| invalid("completion input is absent"))?;
            validate_completion_input(*input)?;
        }
        KernelExtensionMemory::Wait { completion_input_index: input_index }
        | KernelExtensionMemory::Release { input_index } => {
            if !allocations.is_empty() || accesses != [(input_index, ReferenceAccessMode::Consume)] || aliases {
                return Err(invalid("wait or release must consume exactly its declared reference"));
            }
            let input = instruction.inputs().get(input_index).ok_or_else(|| invalid("consumed input is absent"))?;
            if matches!(semantics, KernelExtensionMemory::Wait { .. }) {
                validate_completion_input(*input)?;
            }
        }
    }
    Ok((semantics, accesses))
}

/// Returns whether sorted, merged intervals contain every element of `selection`.
fn contains(intervals: &[Range<usize>], selection: &Range<usize>) -> bool {
    selection.is_empty() || intervals.iter().any(|range| range.start <= selection.start && range.end >= selection.end)
}

/// Adds one interval and coalesces touching selections without storing one bit per element.
fn insert(intervals: &mut Vec<Range<usize>>, selection: Range<usize>) {
    if selection.is_empty() {
        return;
    }
    intervals.push(selection);
    intervals.sort_by_key(|range| range.start);
    let mut index = 0;
    while index + 1 < intervals.len() {
        if intervals[index + 1].start <= intervals[index].end {
            let next = intervals.remove(index + 1);
            intervals[index].end = intervals[index].end.max(next.end);
        } else {
            index += 1;
        }
    }
}

/// Selects contiguous active runs from an interval of root elements using its corresponding logical mask entries.
fn masked_ranges(range: Range<usize>, mask: &[bool]) -> Vec<Range<usize>> {
    let mut selected = Vec::new();
    let mut start = None;
    for (index, &active) in mask.iter().enumerate() {
        if active {
            start.get_or_insert(range.start + index);
        } else if let Some(start) = start.take() {
            selected.push(start..range.start + index);
        }
    }
    if let Some(start) = start {
        selected.push(start..range.end);
    }
    selected
}

/// Retains only coverage proven by both branch states.
fn intersect_states(
    left: BTreeMap<ReferenceRoot, Vec<Range<usize>>>,
    right: &BTreeMap<ReferenceRoot, Vec<Range<usize>>>,
) -> BTreeMap<ReferenceRoot, Vec<Range<usize>>> {
    left.into_iter()
        .map(|(root, ranges)| {
            let mut intersection = Vec::new();
            for left in ranges {
                for right in right.get(&root).into_iter().flatten() {
                    insert(&mut intersection, left.start.max(right.start)..left.end.min(right.end));
                }
            }
            (root, intersection)
        })
        .collect()
}

/// Returns, in increasing order, the grid axes with an extent greater than 1 that the separable tiling `axes` of a
/// mapping leaves unused. Grid points that differ only along these axes select the same window.
fn repeated_axes(extents: &[usize], axes: &[Option<usize>]) -> Vec<usize> {
    extents
        .iter()
        .enumerate()
        .filter(|(axis, extent)| **extent > 1 && !axes.contains(&Some(*axis)))
        .map(|(axis, _)| axis)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayReferenceTransformIndex, CompareOperation, ConditionOperation, DimensionAddOperation, DimensionBounds,
        DimensionMulOperation, DimensionType, EagerContext, FlatProgram, NotOperation, Placeholder, ProgramBuilder,
        ReferenceAccessDescriptor, ReferenceAccessOperation, ReferenceAddUpdateOperation, ReferenceAnalysisError,
        ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSwapOperation, ReferenceViewAnalysisError, ReferenceWriteOperation, RegionId, RegionInterface,
        TypeError, WhileOperation,
    };

    use crate::calls::{KernelDefinition, KernelParameter};
    use crate::grids::{Grid, GridDimension, GridError};
    use crate::mappings::BlockMapping;
    use crate::memory::{
        AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation,
        WaitOperation, WindowMaskOperation,
    };
    use crate::operations::NoKernelExtension;
    use crate::operations::tests::TestExtension;
    use crate::validation::KernelValidationError;

    use super::*;

    /// Number of evaluations of the memory semantics of [`CountingExtension`]. Only one test uses that extension.
    static COUNTED_SEMANTICS: AtomicUsize = AtomicUsize::new(0);

    /// Synchronous extension without inputs, outputs, or effects whose memory semantics count their evaluations in
    /// [`COUNTED_SEMANTICS`], so that a test can count the structural validations of one extension instruction.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct CountingExtension;

    impl Operation for CountingExtension {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            "test_counting"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayIrType],
            _region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(vec![])
        }
    }

    impl ReferenceAccessOperation for CountingExtension {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            0
        }

        fn reference_access_descriptor(
            &self,
            _input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
            None
        }

        fn with_reference_access_transforms(
            &self,
            _input_index: usize,
            _transforms: Vec<ArrayReferenceTransform>,
        ) -> Result<Self, ProgramError> {
            let message = "counting extension has no reference accesses".to_owned();
            Err(ProgramError::UnsupportedOperation { message })
        }
    }

    impl KernelExtension for CountingExtension {
        fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
            COUNTED_SEMANTICS.fetch_add(1, Ordering::Relaxed);
            Ok(KernelExtensionMemory::Synchronous)
        }
    }

    /// Builds one static write-only vector parameter whose blocks advance by their logical size.
    fn call(extent: usize, block: usize, programs: usize) -> KernelCallOperation {
        let mut mapping = ProgramBuilder::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
        let coordinate = mapping.add_input(coordinate_type.clone().into());
        let size = DimensionValue::constant(block).unwrap();
        let constant = mapping.add_constant(ArrayIrValue::Dimension(size.clone()));
        let start = mapping
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, size.r#type().as_ref()).unwrap(),
                Vec::new(),
                vec![coordinate, constant],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![block],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(programs), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![extent]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    /// Selects one repeated scalar parameter, optionally retaining pure mapping work to exercise bounded enumeration.
    fn repeated_call(executions: &[GridExecution], separable: bool) -> KernelCallOperation {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        for _ in executions {
            let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
            let coordinate = mapping.add_input(coordinate_type.clone().into());
            if !separable {
                let zero = DimensionValue::constant(0).unwrap();
                let zero_id = mapping.add_constant(ArrayIrValue::Dimension(zero.clone()));
                mapping
                    .add_instruction(
                        DimensionAddOperation::new(&coordinate_type, zero.r#type().as_ref()).unwrap(),
                        vec![],
                        vec![coordinate, zero_id],
                        None,
                    )
                    .unwrap();
            }
        }
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder; executions.len()], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        KernelCallOperation::new(
            Grid::new(
                executions.iter().map(|&execution| GridDimension::new(Dimension::Static(2), execution)).collect(),
            )
            .unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, mapping)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    /// Builds canonical scalar accesses in their declared order for grid ordering and atomic conflict tests.
    fn repeated_body(
        call: &KernelCallOperation,
        modes: &[ReferenceAccessMode],
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        for coordinate in call.coordinate_types() {
            builder.add_input(coordinate.clone().into());
        }
        let update = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i32).unwrap()));
        for mode in modes {
            let operation = match mode {
                ReferenceAccessMode::Read => KernelOperation::from(ReferenceReadOperation::new()),
                ReferenceAccessMode::Write => KernelOperation::from(ReferenceWriteOperation::new()),
                ReferenceAccessMode::Accumulate => KernelOperation::from(ReferenceAddUpdateOperation::new()),
                ReferenceAccessMode::AtomicAccumulate => {
                    KernelOperation::from(ReferenceAtomicAddUpdateOperation::new())
                }
                _ => panic!("unsupported repeated scalar fixture access"),
            };
            let inputs = if *mode == ReferenceAccessMode::Read { vec![reference] } else { vec![reference, update] };
            builder.add_instruction(operation, vec![], inputs, None).unwrap();
        }
        builder.build(vec![], vec![Placeholder; call.body_input_types().len()], vec![]).unwrap()
    }

    /// Creates a constant of the block size matching the call's reference input.
    fn contents(call: &KernelCallOperation) -> ArrayIrValue<Array> {
        let size = call.parameters()[0].mapping().block_shape()[0];
        ArrayIrValue::Array(Array::new(ArrayType::new_static(DataType::I32, vec![size]), vec![0; size * 4]).unwrap())
    }

    /// Writes either the whole block, just its first element, or nothing.
    fn body(
        call: &KernelCallOperation,
        selection: Option<bool>,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<NoKernelExtension>>> {
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        if let Some(partial) = selection {
            let transforms = if partial {
                vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) }]
            } else {
                vec![]
            };
            let value = builder.add_constant(if partial {
                ArrayIrValue::Array(Array::scalar(0i32).unwrap())
            } else {
                contents(call)
            });
            builder
                .add_instruction(
                    ReferenceWriteOperation::new().with_transforms(transforms),
                    vec![],
                    vec![reference, value],
                    None,
                )
                .unwrap();
        }
        builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()
    }

    /// Builds one pending copy with an optional intervening reference access and completion.
    fn copy_body(
        call: &KernelCallOperation,
        access: Option<(bool, bool)>,
        wait: bool,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        if let Some((source_access, write)) = access {
            let reference = if source_access { source } else { output };
            if write {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, contents], None)
                    .unwrap();
            } else {
                builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap();
            }
        }
        if wait {
            builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        }
        builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()
    }

    /// Adds the synchronous `copy_into` test extension, which copies `source` into `destination` and takes its source
    /// as its first input, so that its canonical effects declare the read first, only when `source_first` is set. Each
    /// operand pairs a reference with an optional start, which selects the two elements of a vector reference that
    /// begin at it, while [`None`] selects the whole reference.
    fn add_copy_into(
        builder: &mut ProgramBuilder<ArrayIrValue<Array>, KernelOperation<TestExtension>>,
        source_first: bool,
        destination: (AtomId, Option<usize>),
        source: (AtomId, Option<usize>),
    ) {
        let transforms = |start: Option<usize>| {
            start
                .map(|start| ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(start, 2, 1)] })
                .into_iter()
                .collect()
        };
        let [first, second] = if source_first { [source, destination] } else { [destination, source] };
        let operation =
            TestExtension::CopyInto { source_first, transforms: [transforms(first.1), transforms(second.1)] };
        builder
            .add_instruction(KernelOperation::Extension(operation), vec![], vec![first.0, second.0], None)
            .unwrap();
    }

    /// Starts an asynchronous transfer of `[3, 5]` from an initialized allocation into the write-only output of `call`
    /// (e.g., `call(2, 2, 1)`), and then applies each of `completions` to its token in order. The producer is the
    /// portable async copy when `producer` is [`None`], and otherwise the test update of the protocol that
    /// `Some(strict)` selects.
    fn completion_body(
        call: &KernelCallOperation,
        producer: Option<bool>,
        completions: &[KernelOperation<TestExtension>],
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<TestExtension>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![3i32, 5]).unwrap()));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let (producer, inputs) = match producer {
            None => (KernelOperation::AsyncCopy(AsyncCopyOperation::new()), vec![source, output]),
            Some(strict) => {
                (KernelOperation::Extension(TestExtension::Update { strict }), vec![source, source, output])
            }
        };
        let token = builder.add_instruction(producer, vec![], inputs, None).unwrap()[0];
        for completion in completions {
            builder.add_instruction(completion.clone(), vec![], vec![token], None).unwrap();
        }
        builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()
    }

    /// Builds a write-only `i32[2, 2]` output whose row windows follow only the first grid axis and a read-only
    /// `i32[2, 2 * extent]` input whose windows follow the first and the last grid axis, where `extent` is the extent
    /// of the last axis. The first axis is parallel with extent 2 and `dimensions` lists the remaining axes.
    fn carried_call(dimensions: &[(usize, GridExecution)]) -> KernelCallOperation {
        let rank = dimensions.len() + 1;
        let extent = dimensions.last().unwrap().0;
        let mappings = [false, true].map(|input| {
            let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let coordinate_types = (0..rank)
                .map(|axis| {
                    DimensionType::new(format!("coordinate_{axis}"), DimensionBounds::non_negative(None).unwrap())
                })
                .collect::<Vec<_>>();
            let coordinates =
                coordinate_types.iter().map(|r#type| mapping.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let start = if input {
                let size = DimensionValue::constant(2).unwrap();
                let size_id = mapping.add_constant(ArrayIrValue::Dimension(size.clone()));
                mapping
                    .add_instruction(
                        DimensionMulOperation::new(&coordinate_types[rank - 1], size.r#type().as_ref()).unwrap(),
                        vec![],
                        vec![coordinates[rank - 1], size_id],
                        None,
                    )
                    .unwrap()[0]
            } else {
                mapping.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()))
            };
            BlockMapping::new(
                mapping.build(vec![coordinates[0], start], vec![Placeholder; rank], vec![Placeholder; 2]).unwrap(),
                vec![1, 2],
                BoundaryPolicy::InBounds,
            )
            .unwrap()
        });
        let [output, input] = mappings;
        let mut grid = vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)];
        grid.extend(
            dimensions
                .iter()
                .map(|&(extent, execution)| GridDimension::new(Dimension::Static(extent), execution)),
        );
        KernelCallOperation::new(
            Grid::new(grid).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2, 2]),
                    KernelParameterAccess::WriteOnly,
                    output,
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2, 2 * extent]),
                    KernelParameterAccess::ReadOnly,
                    input,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    /// Writes zeros to the output window of [`carried_call`] when the last grid coordinate equals `first`, writing only
    /// its first element when `partial` is set, and then accumulates the input window into the output window.
    fn carried_body(
        call: &KernelCallOperation,
        first: usize,
        partial: bool,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>> {
        let output_type = call.parameters()[0].body_type();
        let mut write = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = write.add_input(output_type.clone());
        let (transforms, zeros) = if partial {
            (
                vec![ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(0) }],
                Array::vector(vec![0i32]).unwrap(),
            )
        } else {
            (vec![], Array::new(ArrayType::new_static(DataType::I32, vec![1, 2]), vec![0; 8]).unwrap())
        };
        let zeros = write.add_constant(ArrayIrValue::Array(zeros));
        write
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(transforms),
                vec![],
                vec![reference, zeros],
                None,
            )
            .unwrap();
        let write = write
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let mut skip = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        skip.add_input(output_type.clone());
        let skip = skip
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(output_type);
        let input = builder.add_input(call.parameters()[1].body_type());
        let coordinates = call
            .coordinate_types()
            .iter()
            .map(|r#type| builder.add_input(ArrayIrType::Dimension(r#type.clone())))
            .collect::<Vec<_>>();
        let first = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(first).unwrap()));
        let predicate = builder
            .add_instruction(
                ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::Equal)),
                vec![],
                vec![*coordinates.last().unwrap(), first],
                None,
            )
            .unwrap()[0];
        let write = builder.import_region(write.entry_region_ref());
        let skip = builder.import_region(skip.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![write, skip],
                vec![predicate, output],
                None,
            )
            .unwrap();
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), vec![], vec![output, value], None)
            .unwrap();
        builder.build(vec![], vec![Placeholder; call.body_input_types().len()], vec![]).unwrap()
    }

    #[test]
    fn test_kernel_initialization_error_display() {
        let region = RegionId::new(0);
        let instruction = InstructionId::new(region, 2);
        let token = ReferenceRoot::Allocation { instruction: InstructionId::new(region, 1), output_index: 0 };
        let value = ValueId::new(region, AtomId::new(3));

        // Wrapped errors render their sources unchanged.
        let memory = KernelMemoryError::UnmaskedWindowAccess { operation: "reference_read" };
        assert_eq!(KernelInitializationError::Memory(memory.clone()).to_string(), memory.to_string());
        let call = KernelError::Grid(GridError::ProgramCountOverflow);
        assert_eq!(KernelInitializationError::Call(call.clone()).to_string(), call.to_string());
        let program = ProgramError::MalformedProgram("malformed kernel body".to_owned());
        assert_eq!(KernelInitializationError::Program(program.clone()).to_string(), program.to_string());
        let mapping = BlockMappingError::OutOfBounds { axis: 0, start: 2, limit: 4, extent: 3 };
        assert_eq!(KernelInitializationError::Mapping(mapping.clone()).to_string(), mapping.to_string());

        assert_eq!(
            KernelInitializationError::InvalidCopyToken { value }.to_string(),
            "reference ValueId { region: RegionId { index: 0 }, atom: AtomId { index: 3 } } is not a pending async \
            copy token from this region",
        );
        assert_eq!(
            KernelInitializationError::InvalidCopyTokenAccess { instruction, token }.to_string(),
            "kernel instruction ^0[2] accesses copy token allocation at ^0[1] output 0 without `wait`",
        );
        assert_eq!(
            KernelInitializationError::IncompatibleCompletion { instruction, token }.to_string(),
            "kernel instruction ^0[2] does not follow the completion protocol of pending copy allocation at ^0[1] \
            output 0",
        );
        assert_eq!(
            KernelInitializationError::PendingCopyAccess { instruction, token }.to_string(),
            "kernel instruction ^0[2] conflicts with pending copy allocation at ^0[1] output 0",
        );
        assert_eq!(
            KernelInitializationError::OverlappingCopy { instruction }.to_string(),
            "async copy at ^0[2] has overlapping source and destination views",
        );
        assert_eq!(
            KernelInitializationError::CrossRegionCopy { instruction }.to_string(),
            "pending async copies cannot cross the region boundary at ^0[2]",
        );
        assert_eq!(
            KernelInitializationError::UnwaitedCopy { token }.to_string(),
            "kernel region exits without waiting for async copy allocation at ^0[1] output 0",
        );
        assert_eq!(
            KernelInitializationError::UnboundedLoop { instruction }.to_string(),
            "kernel loop ^0[2] requires an iteration bound",
        );
        assert_eq!(
            KernelInitializationError::UnsupportedOperation { operation: "scan", instruction }.to_string(),
            "kernel initialization does not support operation `scan` at ^0[2]",
        );
        assert_eq!(
            KernelInitializationError::InvalidExtension {
                instruction,
                message: "allocation output is absent".to_owned(),
            }
            .to_string(),
            "invalid kernel extension memory contract at ^0[2]: allocation output is absent",
        );
        assert_eq!(
            KernelInitializationError::UnavailableReference { value }.to_string(),
            "kernel reference ValueId { region: RegionId { index: 0 }, atom: AtomId { index: 3 } } is not a live \
            allocation",
        );
        assert_eq!(
            KernelInitializationError::UnknownSelection { instruction, input_index: 1 }.to_string(),
            "kernel initialization cannot prove the selection of reference input 1 at ^0[2]",
        );
        assert_eq!(
            KernelInitializationError::UninitializedRead {
                instruction,
                root: ReferenceRoot::RegionInput { region, input_index: 0 },
            }
            .to_string(),
            "kernel instruction ^0[2] reads uninitialized elements of region ^0 input 0",
        );
        assert_eq!(
            KernelInitializationError::IncompleteBody { parameter: 1 }.to_string(),
            "kernel body does not definitely initialize write-only parameter 1",
        );
        assert_eq!(
            KernelInitializationError::UnsupportedLaunch { boundary: "grid extents" }.to_string(),
            "kernel initialization cannot statically resolve the grid extents of the launch",
        );
        assert_eq!(
            KernelInitializationError::QualificationLimit { programs: 4, maximum: 3 }.to_string(),
            "kernel qualification requires 4 programs, exceeding the limit 3",
        );
        assert_eq!(
            KernelInitializationError::OverlappingWindows { parameter: 0 }.to_string(),
            "kernel parameter 0 has overlapping mutable windows",
        );
        assert_eq!(
            KernelInitializationError::IncompleteOutput { parameter: 0 }.to_string(),
            "kernel grid does not cover write-only parameter 0",
        );
    }

    #[test]
    fn test_validate_kernel_initialization() {
        let call = call(8, 2, 4);
        let body = body(&call, Some(false));
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_partial_and_missing_writes() {
        let call = call(2, 2, 1);
        for selection in [Some(true), None] {
            let body = body(&call, selection);
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 64),
                Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_write_only_reads_after_writes() {
        // A write-only parameter may be read back and accumulated into once the body has definitely written it.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let zeros = builder.add_constant(contents(&call));
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, zeros], None)
            .unwrap();
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), vec![], vec![reference, value], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_write_only_reads_require_writes() {
        // Write-only parameters start uninitialized, so every reading access mode needs a preceding definite write of
        // the elements that it reads, and a write of only the first element does not cover a read of both.
        let call = call(2, 2, 1);
        let element = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) }];
        for (operation, partial) in [
            (KernelOperation::from(ReferenceReadOperation::new()), false),
            (KernelOperation::from(ReferenceReadOperation::new()), true),
            (KernelOperation::from(ReferenceAddUpdateOperation::new()), false),
            (KernelOperation::from(ReferenceAtomicAddUpdateOperation::new()), false),
            (KernelOperation::from(ReferenceSwapOperation::new()), false),
        ] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(contents(&call));
            if partial {
                let scalar = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
                builder
                    .add_instruction(
                        ReferenceWriteOperation::new().with_transforms(element.clone()),
                        vec![],
                        vec![reference, scalar],
                        None,
                    )
                    .unwrap();
            }
            let reads = matches!(operation, KernelOperation::Portable(ArrayIrOperation::ReferenceRead(_)));
            let inputs = if reads { vec![reference] } else { vec![reference, value] };
            let swap = matches!(operation, KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_)));
            let old = builder.add_instruction(operation, vec![], inputs, None).unwrap().first().copied();
            if swap {
                // A live old value makes the swap an exchange that reads the parameter.
                builder
                    .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, old.unwrap()], None)
                    .unwrap();
            }
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 64),
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, usize::from(partial)),
                    root: ReferenceRoot::RegionInput { region, input_index: 0 },
                }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_combines_element_writes_and_dead_swaps() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
        for index in 0..2 {
            let operation = ReferenceSwapOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Static(index),
            }]);
            builder.add_instruction(operation, vec![], vec![reference, value], None).unwrap();
        }
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_masked_transforms() {
        let call = call(2, 2, 1);
        // Stores the first element of the view starting at 1 under a constant mask and loads the first element of the
        // view starting at `read_start` under the same mask.
        let masked_body = |read_start: usize| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let scratch = builder
                .add_instruction(
                    KernelOperation::Scratch(
                        ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![4]), 4).unwrap(),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            let written = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }];
            let value = builder.add_constant(contents(&call));
            let mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()));
            builder
                .add_instruction(
                    MaskedStoreOperation::new().with_transforms(written),
                    vec![],
                    vec![scratch, value, mask],
                    None,
                )
                .unwrap();
            let read = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(read_start, 2, 1)] }];
            let loaded = builder
                .add_instruction(
                    MaskedLoadOperation::new().with_transforms(read),
                    vec![],
                    vec![scratch, mask, value],
                    None,
                )
                .unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, loaded], None).unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let unwritten = masked_body(0);
        let region = unwritten.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(unwritten.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 2),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );
        let written = masked_body(1);
        assert_eq!(validate_kernel_initialization(written.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_masked_stores_and_dead_swaps() {
        let call = call(2, 2, 1);
        // Writes the output through a masked store or a masked swap whose old value is unused, under a constant mask
        // that selects the second element only when `complete` is set.
        let masked_body = |swap: bool, complete: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(contents(&call));
            let mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![true, complete]).unwrap()));
            if swap {
                builder
                    .add_instruction(MaskedSwapOperation::new(), vec![], vec![output, value, mask, value], None)
                    .unwrap();
            } else {
                builder
                    .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                    .unwrap();
            }
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let partial_store = masked_body(false, false);
        assert_eq!(
            validate_kernel_initialization(partial_store.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
        let complete_store = masked_body(false, true);
        assert_eq!(validate_kernel_initialization(complete_store.entry_region_ref(), &call, 0), Ok(()));
        let partial_swap = masked_body(true, false);
        assert_eq!(
            validate_kernel_initialization(partial_swap.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
        let complete_swap = masked_body(true, true);
        assert_eq!(validate_kernel_initialization(complete_swap.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_unknown_mask_cannot_establish_writes() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let false_mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, false]).unwrap()));
        let mask = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new())),
                vec![],
                vec![false_mask],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 })
        );
    }

    #[test]
    fn test_validate_kernel_initialization_unknown_mask_requires_full_read_initialization() {
        let call = call(2, 2, 1);
        // Writes a scratch allocation under a constant mask that selects the second element only when `complete` is
        // set, and then loads it under a runtime mask.
        let masked_body = |complete: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let scratch = builder
                .add_instruction(
                    KernelOperation::Scratch(
                        ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            let value = builder.add_constant(contents(&call));
            let mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![true, complete]).unwrap()));
            builder
                .add_instruction(MaskedStoreOperation::new(), vec![], vec![scratch, value, mask], None)
                .unwrap();
            let false_mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, false]).unwrap()));
            let unknown = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new())),
                    vec![],
                    vec![false_mask],
                    None,
                )
                .unwrap()[0];
            let loaded = builder
                .add_instruction(MaskedLoadOperation::new(), vec![], vec![scratch, unknown, value], None)
                .unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, loaded], None).unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let partial = masked_body(false);
        let region = partial.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(partial.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 3),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );
        let complete = masked_body(true);
        assert_eq!(validate_kernel_initialization(complete.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_window_mask_store() {
        // A store under the validity mask of its own masked edge window writes every valid element, which completely
        // initializes the write-only output even though the mask is only known at runtime.
        let original = call(3, 2, 2);
        let mapping =
            BlockMapping::new(original.parameters()[0].mapping().program().clone(), vec![2], BoundaryPolicy::Masked)
                .unwrap();
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![3]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let mask = builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap()[0];
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_window_mask_store_rejects_other_masks() {
        // Only the validity mask of the stored reference itself is credited. A mask derived from it through other
        // operations and an unrelated runtime mask establish no coverage.
        let call = call(2, 2, 1);
        for derived in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(contents(&call));
            let mut mask = if derived {
                builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap()[0]
            } else {
                builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, false]).unwrap()))
            };
            for _ in 0..if derived { 2 } else { 1 } {
                mask = builder
                    .add_instruction(
                        ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new())),
                        vec![],
                        vec![mask],
                        None,
                    )
                    .unwrap()[0];
            }
            builder
                .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                .unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 0),
                Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_window_mask_store_rejects_other_references() {
        // The validity mask of another parameter's window does not describe the elements of the stored window.
        let original = call(2, 2, 1);
        let parameter = original.parameters()[0].clone();
        let call = KernelCallOperation::new(original.grid().clone(), vec![parameter.clone(), parameter]).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let first = builder.add_input(call.parameters()[0].body_type());
        let second = builder.add_input(call.parameters()[1].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let mask = builder.add_instruction(WindowMaskOperation, vec![], vec![second], None).unwrap()[0];
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![first, value, mask], None)
            .unwrap();
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![second, value, mask], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_window_mask_store_rejects_views() {
        // A store through a view is not credited, even when the view selects the whole window.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let mask = builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap()[0];
        builder
            .add_instruction(
                MaskedStoreOperation::new()
                    .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }]),
                vec![],
                vec![output, value, mask],
                None,
            )
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_complementary_masked_stores() {
        // Two stores under a runtime mask and its negation together write every element of their common selection, in
        // either order.
        let call = call(2, 2, 1);
        for complement_first in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(contents(&call));
            let constant = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()));
            let not = ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new()));
            let mask = builder.add_instruction(not.clone(), vec![], vec![constant], None).unwrap()[0];
            let complement = builder.add_instruction(not, vec![], vec![mask], None).unwrap()[0];
            let masks = if complement_first { [complement, mask] } else { [mask, complement] };
            for mask in masks {
                builder
                    .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                    .unwrap();
            }
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
        }
    }

    #[test]
    fn test_validate_kernel_initialization_complementary_masked_stores_reject_repeated_masks() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let constant = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()));
        let mask = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new())),
                vec![],
                vec![constant],
                None,
            )
            .unwrap()[0];
        for _ in 0..2 {
            builder
                .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                .unwrap();
        }
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_complementary_masked_stores_reject_different_selections() {
        let call = call(4, 4, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![0i32, 0]).unwrap()));
        let constant = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()));
        let not = ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new()));
        let mask = builder.add_instruction(not.clone(), vec![], vec![constant], None).unwrap()[0];
        let complement = builder.add_instruction(not, vec![], vec![mask], None).unwrap()[0];
        for (start, mask) in [(0, mask), (2, complement)] {
            builder
                .add_instruction(
                    MaskedStoreOperation::new().with_transforms(vec![ArrayReferenceTransform::Slice {
                        axes: vec![ArraySliceAxis::new(start, 2, 1)],
                    }]),
                    vec![],
                    vec![output, value, mask],
                    None,
                )
                .unwrap();
        }
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_complementary_masked_stores_reject_split_regions() {
        // One half in the enclosing region and the other inside a branch never pair up, even though every branch
        // contains the same half.
        let call = call(2, 2, 1);
        let not = ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new()));
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = branch.add_input(call.parameters()[0].body_type());
        let mask = branch.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::Boolean, vec![2])));
        let value = branch.add_input(contents(&call).r#type().into_owned());
        let complement = branch.add_instruction(not.clone(), vec![], vec![mask], None).unwrap()[0];
        branch
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, complement], None)
            .unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let constant = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()));
        let mask = builder.add_instruction(not, vec![], vec![constant], None).unwrap()[0];
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
            .unwrap();
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, output, mask, value],
                None,
            )
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_dynamic_writes_establish_no_coverage() {
        // A dynamically indexed write is admitted but may select any element, so it covers none, whether or not a
        // static write of the other element precedes it.
        let call = call(2, 2, 1);
        for static_write in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
            let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i64).unwrap()));
            if static_write {
                let operation = ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]);
                builder.add_instruction(operation, vec![], vec![reference, value], None).unwrap();
            }
            let operation = ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
            builder.add_instruction(operation, vec![], vec![reference, value, index], None).unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 64),
                Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_dynamic_reads() {
        // Dynamically indexed reads of a read-only parameter and of a completely written scratch allocation need no
        // static selection, because every element that they may select is initialized.
        let original = call(2, 2, 1);
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                original.parameters()[0].clone(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2]),
                    KernelParameterAccess::ReadOnly,
                    original.parameters()[0].mapping().clone(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        let input = builder.add_input(call.parameters()[1].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i64).unwrap()));
        let dynamic = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let scratch = builder
            .add_instruction(
                ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                vec![],
                vec![],
                None,
            )
            .unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![scratch, value], None).unwrap();
        for reference in [input, scratch] {
            builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(dynamic.clone()),
                    vec![],
                    vec![reference, index],
                    None,
                )
                .unwrap();
        }
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_dynamic_reads_require_initialized_roots() {
        // A dynamically indexed read may select the element that the static write left uninitialized.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let scalar = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
        let scratch = builder
            .add_instruction(
                ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                vec![],
                vec![],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]),
                vec![],
                vec![scratch, scalar],
                None,
            )
            .unwrap();
        builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Dynamic,
                }]),
                vec![],
                vec![scratch, index],
                None,
            )
            .unwrap();
        let value = builder.add_constant(contents(&call));
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let result = validate_kernel_initialization(body.entry_region_ref(), &call, 64);
        assert_eq!(
            result,
            Err(KernelInitializationError::UnknownSelection {
                instruction: InstructionId::new(body.entry_region_ref().id(), 2),
                input_index: 0,
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "kernel initialization cannot prove the selection of reference input 0 at {}",
                InstructionId::new(body.entry_region_ref().id(), 2),
            ),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_dynamic_access_conflicts_with_pending_copy() {
        // A dynamic selection counts as its whole root, which the pending copy reserves as its destination.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let destination =
            builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token =
            builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, destination], None).unwrap()[0];
        builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Dynamic,
                }]),
                vec![],
                vec![destination, index],
                None,
            )
            .unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, contents], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::PendingCopyAccess {
                instruction: InstructionId::new(region.id(), 3),
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 2), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy() {
        let call = call(2, 2, 1);
        let body = copy_body(&call, None, true);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
        let body = copy_body(&call, Some((true, false)), true);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_reserves_references() {
        let call = call(2, 2, 1);
        for access in [(true, true), (false, true)] {
            let body = copy_body(&call, Some(access), true);
            let region = body.entry_region_ref();
            assert_eq!(
                validate_kernel_initialization(region, &call, 64),
                Err(KernelInitializationError::PendingCopyAccess {
                    instruction: InstructionId::new(region.id(), 2),
                    token: ReferenceRoot::Allocation {
                        instruction: InstructionId::new(region.id(), 1),
                        output_index: 0,
                    },
                }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_reserves_only_selected_views() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4, 5, 6]).unwrap()));
        let root = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let source = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }];
        let destination = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(2, 2, 1)] }];
        let independent = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(4, 2, 1)] }];
        let token = builder
            .add_instruction(
                AsyncCopyOperation::new()
                    .with_source_transforms(source)
                    .with_destination_transforms(destination.clone()),
                vec![],
                vec![root, root],
                None,
            )
            .unwrap()[0];
        let update = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![7i32, 8]).unwrap()));
        builder
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(independent),
                vec![],
                vec![root, update],
                None,
            )
            .unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let copied = builder
            .add_instruction(ReferenceReadOperation::new().with_transforms(destination), vec![], vec![root], None)
            .unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, copied], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_does_not_initialize_before_wait() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let destination = builder
            .add_instruction(
                ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                vec![],
                vec![],
                None,
            )
            .unwrap()[0];
        let token =
            builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, destination], None).unwrap()[0];
        builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![destination], None).unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, contents], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::PendingCopyAccess {
                instruction: InstructionId::new(region.id(), 3),
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 2), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_requires_wait() {
        let call = call(2, 2, 1);
        let body = copy_body(&call, None, false);
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::UnwaitedCopy {
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 1), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_rejects_overlapping_storage() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, source], None).unwrap()[0];
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, contents], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::OverlappingCopy { instruction: InstructionId::new(region.id(), 1) }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_rejects_generic_token_access() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        builder.add_instruction(ReferenceFreezeOperation::new(), vec![], vec![token], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::InvalidCopyTokenAccess {
                instruction: InstructionId::new(region.id(), 2),
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 1), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_wait_requires_copy_provenance() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let token =
            builder.add_constant(ArrayIrValue::Array(Array::new(ArrayType::scalar(DataType::Token), vec![]).unwrap()));
        let token = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![token], None).unwrap()[0];
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, contents], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::InvalidCopyToken { value: ValueId::new(region.id(), token) }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_window_mask_ignores_pending_copies() {
        // Taking the validity of a reserved destination or source accesses no storage, so it neither conflicts with the
        // pending copy nor requires the uninitialized destination to be written.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let contents = builder.add_constant(contents(&call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap();
        builder.add_instruction(WindowMaskOperation, vec![], vec![source], None).unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_requires_same_region_wait() {
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        let coordinate = builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        let first = builder.import_region(body(&call, Some(false)).entry_region_ref());
        let second = builder.import_region(body(&call, Some(false)).entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![first, second],
                vec![predicate, output, coordinate],
                None,
            )
            .unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::CrossRegionCopy { instruction: InstructionId::new(region.id(), 2) }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_across_independent_regions() {
        // A condition whose branches only access storage that the pending copy does not reserve may run before the
        // wait. A copy that a branch starts must also complete in that branch.
        let call = call(2, 2, 1);
        let branch_body = |nested_wait: bool| {
            let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let destination = branch.add_input(call.parameters()[0].body_type());
            let values = branch.add_constant(contents(&call));
            let source = branch.add_instruction(ReferenceNewOperation::new(), vec![], vec![values], None).unwrap()[0];
            let token =
                branch.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, destination], None).unwrap()[0];
            if nested_wait {
                branch.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
            }
            let branch = branch
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(call.coordinate_types()[0].clone().into());
            let contents = builder.add_constant(contents(&call));
            let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
            let source =
                builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
            let scratch = builder
                .add_instruction(
                    ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            let token =
                builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
            let branch = builder.import_region(branch.entry_region_ref());
            builder
                .add_instruction(
                    ArrayIrOperation::Condition(ConditionOperation::new()),
                    vec![branch, branch],
                    vec![predicate, scratch],
                    None,
                )
                .unwrap();
            builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let waited = branch_body(true);
        assert_eq!(validate_kernel_initialization(waited.entry_region_ref(), &call, 64), Ok(()));
        let unwaited = branch_body(false);
        let region = unwaited.entry_region_ref();
        let branch = region.instructions()[3].regions()[0];
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::UnwaitedCopy {
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(branch, 1), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_rejects_regions_reading_reservations() {
        // Reservations are checked at root granularity, so even a read of the copy source inside a branch is rejected.
        let call = call(2, 2, 1);
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = branch.add_input(call.parameters()[0].body_type());
        branch.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, source],
                None,
            )
            .unwrap();
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::CrossRegionCopy { instruction: InstructionId::new(region.id(), 2) }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_async_copy_rejects_nested_waits() {
        let call = call(2, 2, 1);
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let token = branch.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token))));
        branch.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder.add_instruction(AsyncCopyOperation::new(), vec![], vec![source, output], None).unwrap()[0];
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, token],
                None,
            )
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        // A branch may not consume a token that it borrows from the attaching region. The generic reference analysis
        // rejects that consumption before the initialization analysis, which would report the attaching condition.
        let region = body.entry_region_ref();
        let branch = region.instructions()[2].regions()[0];
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::Call(KernelError::Validation(KernelValidationError::from(
                ReferenceViewAnalysisError::Analysis(ReferenceAnalysisError::ConsumptionOutsideCreationScope {
                    operation: "wait",
                    instruction: InstructionId::new(branch, 0),
                    region: branch,
                    root: ReferenceRoot::RegionInput { region: branch, input_index: 0 },
                }),
            )))),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_scratch() {
        for extent in [0, 2] {
            let call = call(extent, extent, 1);
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let scratch = builder
                .add_instruction(
                    KernelOperation::Scratch(
                        ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![extent]), 4).unwrap(),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            if extent != 0 {
                let value = builder.add_constant(contents(&call));
                builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![scratch, value], None).unwrap();
            }
            let value =
                builder.add_instruction(ReferenceFreezeOperation::new(), vec![], vec![scratch], None).unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
        }
    }

    #[test]
    fn test_validate_kernel_initialization_scratch_rejects_missing_and_partial_writes() {
        let call = call(2, 2, 1);
        // Reads a scratch allocation that has not been written, or whose first element only has been written.
        let scratch_body = |partial: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let scratch = builder
                .add_instruction(
                    KernelOperation::Scratch(
                        ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            if partial {
                let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
                builder
                    .add_instruction(
                        ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Static(0),
                        }]),
                        vec![],
                        vec![scratch, value],
                        None,
                    )
                    .unwrap();
            }
            let value =
                builder.add_instruction(ReferenceFreezeOperation::new(), vec![], vec![scratch], None).unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let missing = scratch_body(false);
        let region = missing.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(missing.entry_region_ref(), &call, 64),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 1),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );
        let partial = scratch_body(true);
        let region = partial.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(partial.entry_region_ref(), &call, 64),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 2),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_extension_classification() {
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        let extension_body = |operation: TestExtension| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            builder.add_instruction(KernelOperation::Extension(operation), vec![], vec![], None).unwrap();
            builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![]).unwrap()
        };
        let synchronous = extension_body(TestExtension::Synchronous);
        assert_eq!(validate_kernel_initialization(synchronous.entry_region_ref(), &call, 1), Ok(()));
        let misclassified = extension_body(TestExtension::MisclassifiedAllocation);
        assert_eq!(
            validate_kernel_initialization(misclassified.entry_region_ref(), &call, 1),
            Err(KernelInitializationError::InvalidExtension {
                instruction: InstructionId::new(misclassified.entry_region_ref().id(), 0),
                message: "synchronous operations cannot allocate or consume references".to_owned(),
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_extension_classification_empty_launch() {
        // An empty launch runs no body, but its extension declarations are structural, so a misclassified extension
        // is rejected for an empty grid exactly as for a grid with one point.
        for programs in [1, 0] {
            let grid = Grid::new(vec![GridDimension::new(Dimension::Static(programs), GridExecution::Parallel)]);
            let call = KernelCallOperation::new(grid.unwrap(), vec![]).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let operation = KernelOperation::Extension(TestExtension::MisclassifiedAllocation);
            builder.add_instruction(operation, vec![], vec![], None).unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 1),
                Err(KernelInitializationError::InvalidExtension {
                    instruction: InstructionId::new(body.entry_region_ref().id(), 0),
                    message: "synchronous operations cannot allocate or consume references".to_owned(),
                }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_extension_classification_shared_regions() {
        // Nests eight conditions around one extension, where each condition attaches the region of the next one as
        // both of its branches. The extension is reachable along 256 paths, but its declarations are validated once.
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(0), GridExecution::Parallel)]);
        let call = KernelCallOperation::new(grid.unwrap(), vec![]).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<CountingExtension>>::new();
        builder
            .add_instruction(KernelOperation::Extension(CountingExtension), vec![], vec![], None)
            .unwrap();
        let mut body =
            builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![]).unwrap();
        for level in 0..8 {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<CountingExtension>>::new();
            if level == 7 {
                builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            }
            let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
            let branch = builder.import_region(body.entry_region_ref());
            builder
                .add_instruction(
                    ArrayIrOperation::Condition(ConditionOperation::new()),
                    vec![branch, branch],
                    vec![predicate],
                    None,
                )
                .unwrap();
            let inputs = if level == 7 { vec![Placeholder] } else { vec![] };
            body = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], inputs, vec![]).unwrap();
        }
        COUNTED_SEMANTICS.store(0, Ordering::Relaxed);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 1), Ok(()));
        assert_eq!(COUNTED_SEMANTICS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_validate_kernel_initialization_extension_classification_completion_tokens() {
        // Commits and waits on an `i32` scratch allocation. A completion input must reference a scalar token, which is
        // structural, so the body is rejected for an empty grid exactly as for a grid with one point.
        for completion in [TestExtension::Commit { strict: false }, TestExtension::Wait { strict: false }] {
            for programs in [1, 0] {
                let grid = Grid::new(vec![GridDimension::new(Dimension::Static(programs), GridExecution::Parallel)]);
                let call = KernelCallOperation::new(grid.unwrap(), vec![]).unwrap();
                let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
                builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
                let scratch = ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap();
                let scratch =
                    builder.add_instruction(KernelOperation::Scratch(scratch), vec![], vec![], None).unwrap()[0];
                let operation = KernelOperation::Extension(completion.clone());
                builder.add_instruction(operation, vec![], vec![scratch], None).unwrap();
                let body = builder
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                    .unwrap();
                let region = body.entry_region_ref().id();
                assert_eq!(
                    validate_kernel_initialization(body.entry_region_ref(), &call, 1),
                    Err(KernelInitializationError::InvalidCopyToken { value: ValueId::new(region, scratch) }),
                );
            }
        }
    }

    #[test]
    fn test_validate_kernel_initialization_reads_precede_instruction_writes() {
        // Binds the destination and the source of `copy_into` to the same uninitialized write-only scalar, so the copy
        // reads an uninitialized element whichever access it declares first.
        let mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()
            .build(vec![], Vec::<Placeholder>::new(), Vec::<Placeholder>::new())
            .unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::scalar(DataType::I32),
                    KernelParameterAccess::WriteOnly,
                    BlockMapping::new(mapping, vec![], BoundaryPolicy::InBounds).unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        for source_first in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            add_copy_into(&mut builder, source_first, (output, None), (output, None));
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 1),
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 0),
                    root: ReferenceRoot::RegionInput { region, input_index: 0 },
                }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_reads_precede_overlapping_instruction_writes() {
        // Initializes the first `written` elements of the write-only output and then copies its elements `1..3` into
        // its elements `2..4`, so the copy reads element 2 before it writes it.
        let call = call(4, 4, 1);
        let body = |written: usize, source_first: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let value = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![1i32; written]).unwrap()));
            let prefix = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, written, 1)] }];
            builder
                .add_instruction(
                    ReferenceWriteOperation::new().with_transforms(prefix),
                    vec![],
                    vec![output, value],
                    None,
                )
                .unwrap();
            add_copy_into(&mut builder, source_first, (output, Some(2)), (output, Some(1)));
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        for source_first in [false, true] {
            let uninitialized = body(2, source_first);
            let region = uninitialized.entry_region_ref().id();
            assert_eq!(
                validate_kernel_initialization(uninitialized.entry_region_ref(), &call, 1),
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 1),
                    root: ReferenceRoot::RegionInput { region, input_index: 0 },
                }),
            );

            // Once the source is initialized, the copy completes the output.
            let initialized = body(3, source_first);
            assert_eq!(validate_kernel_initialization(initialized.entry_region_ref(), &call, 1), Ok(()));
        }
    }

    #[test]
    fn test_validate_kernel_initialization_initialized_instruction_reads() {
        // Copies an initialized allocation into itself and then into the distinct write-only output.
        let call = call(4, 4, 1);
        for source_first in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let contents = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4]).unwrap()));
            let source =
                builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
            add_copy_into(&mut builder, source_first, (source, None), (source, None));
            add_copy_into(&mut builder, source_first, (output, None), (source, None));
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 1), Ok(()));
        }
    }

    #[test]
    fn test_validate_kernel_initialization_completion_protocols() {
        // Every token follows the protocol of its producer, and the host interpreter, which qualifies the body before
        // replaying it, reports the same outcome. The portable protocol and the default extension protocol admit any
        // number of commits, and the strict test protocol requires exactly one commit before the wait.
        let call = call(2, 2, 1);
        let wait = KernelOperation::<TestExtension>::Wait(WaitOperation);
        let commit = |strict| KernelOperation::Extension(TestExtension::Commit { strict });
        let extension_wait = |strict| KernelOperation::Extension(TestExtension::Wait { strict });
        let region = RegionId::new(0);
        let incompatible = |index| KernelInitializationError::IncompatibleCompletion {
            instruction: InstructionId::new(region, index),
            token: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 1), output_index: 0 },
        };
        let cases = [
            (None, vec![wait.clone()], Ok(())),
            (Some(false), vec![extension_wait(false)], Ok(())),
            (Some(false), vec![commit(false), extension_wait(false)], Ok(())),
            (Some(false), vec![commit(false), commit(false), extension_wait(false)], Ok(())),
            (Some(true), vec![commit(true), extension_wait(true)], Ok(())),
            (Some(true), vec![extension_wait(true)], Err(incompatible(2))),
            (Some(true), vec![commit(true), commit(true), extension_wait(true)], Err(incompatible(3))),
            (None, vec![extension_wait(false)], Err(incompatible(2))),
            (None, vec![commit(false), wait.clone()], Err(incompatible(2))),
            (Some(false), vec![wait.clone()], Err(incompatible(2))),
            (Some(false), vec![commit(false), wait.clone()], Err(incompatible(3))),
            (Some(true), vec![commit(false), extension_wait(false)], Err(incompatible(2))),
            (Some(true), vec![commit(true), extension_wait(false)], Err(incompatible(3))),
        ];
        for (producer, completions, expected) in cases {
            let body = completion_body(&call, producer, &completions);
            assert_eq!(body.entry_region_ref().id(), region);
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 1), expected);
            let definition = KernelDefinition::new(call.clone(), body).unwrap();
            match expected {
                Ok(()) => assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::vector(vec![3i32, 5]).unwrap()])),
                Err(expected) => assert_eq!(
                    definition.interpret(vec![], 1).unwrap_err().downcast_custom::<KernelInitializationError>(),
                    Some(&expected),
                ),
            }
        }

        // A wait consumes its token, so the program builder rejects any later use of it.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let contents = builder.add_constant(contents(&call));
        let source = builder.add_instruction(ReferenceNewOperation::new(), vec![], vec![contents], None).unwrap()[0];
        let token = builder
            .add_instruction(KernelOperation::AsyncCopy(AsyncCopyOperation::new()), vec![], vec![source, output], None)
            .unwrap()[0];
        builder.add_instruction(wait.clone(), vec![], vec![token], None).unwrap();
        assert_eq!(
            builder.add_instruction(wait, vec![], vec![token], None),
            Err(ProgramError::MalformedProgram(
                "`wait` consumes a reference whose alias family `wait` already consumed".to_owned(),
            )),
        );
        assert_eq!(
            builder.add_instruction(extension_wait(false), vec![], vec![token], None),
            Err(ProgramError::MalformedProgram(
                "`test_wait` consumes a reference whose alias family `wait` already consumed".to_owned(),
            )),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_extension_async_update() {
        // Waits for the update, copies its destination into the output, and releases the destination.
        fn finish(
            mut builder: ProgramBuilder<ArrayIrValue<Array>, KernelOperation<TestExtension>>,
            output: AtomId,
            destination: AtomId,
            token: AtomId,
        ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<TestExtension>>> {
            builder
                .add_instruction(
                    KernelOperation::Extension(TestExtension::Wait { strict: false }),
                    vec![],
                    vec![token],
                    None,
                )
                .unwrap();
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![destination], None).unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            builder
                .add_instruction(KernelOperation::Extension(TestExtension::Release), vec![], vec![destination], None)
                .unwrap();
            builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()
        }

        let call = call(1, 1, 1);
        // Starts an update from a source allocation into a destination allocation and commits it, writing the source
        // first only when `write_source` is set.
        let update = |write_source: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let r#type = ArrayType::new_static(DataType::I32, vec![1]);
            let source = builder
                .add_instruction(
                    KernelOperation::Extension(TestExtension::Allocate(r#type.clone())),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            let destination = builder
                .add_instruction(KernelOperation::Extension(TestExtension::Allocate(r#type)), vec![], vec![], None)
                .unwrap()[0];
            let value = builder.add_constant(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, vec![1]), &[3i32]).unwrap(),
            ));
            if write_source {
                builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![source, value], None).unwrap();
            }
            let token = builder
                .add_instruction(
                    KernelOperation::Extension(TestExtension::Update { strict: false }),
                    vec![],
                    vec![source, source, destination],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    KernelOperation::Extension(TestExtension::Commit { strict: false }),
                    vec![],
                    vec![token],
                    None,
                )
                .unwrap();
            (builder, output, source, destination, token)
        };

        // The destination is readable once the update has been waited on.
        let (builder, output, _, destination, token) = update(true);
        let body = finish(builder, output, destination, token);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 1), Ok(()));

        // The update reads its source when it starts.
        let (builder, output, _, destination, token) = update(false);
        let body = finish(builder, output, destination, token);
        let region = body.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 1),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 2),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );

        // A commit keeps the reservations, so reading the destination or releasing the source before the wait
        // conflicts with the pending update.
        let (mut builder, output, _, destination, token) = update(true);
        builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![destination], None).unwrap();
        let body = finish(builder, output, destination, token);
        let region = body.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 1),
            Err(KernelInitializationError::PendingCopyAccess {
                instruction: InstructionId::new(region, 5),
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 3), output_index: 0 },
            }),
        );
        let (mut builder, output, source, destination, token) = update(true);
        builder
            .add_instruction(KernelOperation::Extension(TestExtension::Release), vec![], vec![source], None)
            .unwrap();
        let body = finish(builder, output, destination, token);
        let region = body.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 1),
            Err(KernelInitializationError::PendingCopyAccess {
                instruction: InstructionId::new(region, 5),
                token: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 3), output_index: 0 },
            }),
        );

        // A wait or a release consumes the token, so a later commit is malformed.
        let commit = KernelOperation::Extension(TestExtension::Commit { strict: false });
        let (mut builder, _, _, _, token) = update(true);
        let wait = KernelOperation::Extension(TestExtension::Wait { strict: false });
        builder.add_instruction(wait, vec![], vec![token], None).unwrap();
        assert_eq!(
            builder.add_instruction(commit.clone(), vec![], vec![token], None),
            Err(ProgramError::MalformedProgram(
                "`test_commit` reads a reference whose alias family `test_wait` already consumed".to_owned(),
            )),
        );
        let (mut builder, _, _, _, token) = update(true);
        builder
            .add_instruction(KernelOperation::Extension(TestExtension::Release), vec![], vec![token], None)
            .unwrap();
        assert_eq!(
            builder.add_instruction(commit.clone(), vec![], vec![token], None),
            Err(ProgramError::MalformedProgram(
                "`test_commit` reads a reference whose alias family `test_release` already consumed".to_owned(),
            )),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_intersects_condition_branches() {
        let call = call(2, 2, 1);
        // The first branch writes the whole parameter, and the second branch writes it only when `complete` is set.
        let condition_body = |complete: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            let coordinate = builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
            let first = builder.import_region(body(&call, Some(false)).entry_region_ref());
            let second = builder.import_region(body(&call, complete.then_some(false)).entry_region_ref());
            builder
                .add_instruction(
                    ArrayIrOperation::Condition(ConditionOperation::new()),
                    vec![first, second],
                    vec![predicate, reference, coordinate],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let incomplete = condition_body(false);
        assert_eq!(
            validate_kernel_initialization(incomplete.entry_region_ref(), &call, 64),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
        let complete = condition_body(true);
        assert_eq!(validate_kernel_initialization(complete.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_folded_region_accesses() {
        let call = call(4, 4, 1);
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = branch.add_input(call.parameters()[0].body_type());
        let value = branch.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])));
        branch
            .add_instruction(
                ReferenceWriteOperation::new()
                    .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(2, 2, 1)] }]),
                vec![],
                vec![reference, value],
                None,
            )
            .unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![3i32, 7]).unwrap()));
        builder
            .add_instruction(
                ReferenceWriteOperation::new()
                    .with_transforms(vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }]),
                vec![],
                vec![reference, value],
                None,
            )
            .unwrap();
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_while_condition_and_zero_body_path() {
        let call = call(2, 2, 1);
        // The loop body always writes the whole parameter, and the condition writes it only when `condition_writes` is
        // set.
        let while_body = |condition_writes: bool, bounded: bool| {
            let mut condition = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = condition.add_input(call.parameters()[0].body_type());
            if condition_writes {
                let value = condition.add_constant(contents(&call));
                condition
                    .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
                    .unwrap();
            }
            let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(false).unwrap()));
            let condition = condition
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![predicate],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let mut loop_body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = loop_body.add_input(call.parameters()[0].body_type());
            let value = loop_body.add_constant(contents(&call));
            loop_body
                .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
                .unwrap();
            let loop_body = loop_body
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![reference],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let condition = builder.import_region(condition.entry_region_ref());
            let loop_body = builder.import_region(loop_body.entry_region_ref());
            builder
                .add_instruction(
                    ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(bounded.then_some(1)).unwrap()),
                    vec![condition, loop_body],
                    vec![reference],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };
        let body_writes = while_body(false, true);
        assert_eq!(
            validate_kernel_initialization(body_writes.entry_region_ref(), &call, 64),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
        let condition_writes = while_body(true, true);
        assert_eq!(validate_kernel_initialization(condition_writes.entry_region_ref(), &call, 64), Ok(()));
        let unbounded = while_body(true, false);
        assert_eq!(
            validate_kernel_initialization(unbounded.entry_region_ref(), &call, 64),
            Err(KernelInitializationError::UnboundedLoop {
                instruction: InstructionId::new(unbounded.entry_region_ref().id(), 0),
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_while_cannot_use_later_iteration_writes() {
        let call = call(2, 2, 1);
        let reference_type = call.parameters()[0].body_type();
        let mut condition = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        condition.add_input(reference_type.clone());
        let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![predicate],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut loop_body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = loop_body.add_input(reference_type.clone());
        let transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        loop_body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(transforms.clone()),
                vec![],
                vec![reference],
                None,
            )
            .unwrap();
        let update = loop_body.add_constant(ArrayIrValue::Array(Array::scalar(1i32).unwrap()));
        loop_body
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(transforms),
                vec![],
                vec![reference, update],
                None,
            )
            .unwrap();
        let loop_body = loop_body
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(reference_type);
        builder.add_input(call.coordinate_types()[0].clone().into());
        let scratch = builder
            .add_instruction(
                ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                vec![],
                vec![],
                None,
            )
            .unwrap()[0];
        let condition = builder.import_region(condition.entry_region_ref());
        let loop_body = builder.import_region(loop_body.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(Some(2)).unwrap()),
                vec![condition, loop_body],
                vec![scratch],
                None,
            )
            .unwrap();
        let update = builder.add_constant(contents(&call));
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, update], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let region = body.entry_region_ref();
        let loop_body = region.instructions()[1].regions()[1];
        assert_eq!(
            validate_kernel_initialization(region, &call, 64),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(loop_body, 0),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 0), output_index: 0 },
            })
        );
    }

    #[test]
    fn test_validate_kernel_initialization_coordinate_condition_without_carry_axis() {
        // The output window follows the only grid axis, so its coordinate is unknown and both branches are joined.
        let call = call(4, 2, 2);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        let coordinate = builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let predicate = builder
            .add_instruction(
                ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::Equal)),
                vec![],
                vec![coordinate, zero],
                None,
            )
            .unwrap()[0];
        let write = builder.import_region(body(&call, Some(false)).entry_region_ref());
        let skip = builder.import_region(body(&call, None).entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![write, skip],
                vec![predicate, reference, coordinate],
                None,
            )
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteBody { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_first_step_initialization() {
        // The write at `k == 0` initializes the output window that every later step along the sequential axis `k`
        // accumulates into, including when `k` has extent 1 and therefore no later step.
        let call = carried_call(&[(3, GridExecution::Sequential)]);
        let body = carried_body(&call, 0, false);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
        let call = carried_call(&[(1, GridExecution::Sequential)]);
        let body = carried_body(&call, 0, false);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_first_step_initialization_combines_carry_axes() {
        // The first output follows `l` and is carried along `k`, and the second output follows `k` and is carried along
        // `l`, so every combination of first and later steps along `k` and `l` must leave both outputs complete. Each
        // output is zeroed when the coordinate named by `firsts` is zero, which is only correct for its carry axis, and
        // otherwise the first step along `l` with a later step along `k` leaves the second output empty.
        let parameters = [1, 0]
            .map(|axis| {
                let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                let coordinates = (0..2)
                    .map(|axis| {
                        mapping.add_input(
                            DimensionType::new(
                                format!("coordinate_{axis}"),
                                DimensionBounds::non_negative(None).unwrap(),
                            )
                            .into(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mapping = BlockMapping::new(
                    mapping.build(vec![coordinates[axis]], vec![Placeholder; 2], vec![Placeholder]).unwrap(),
                    vec![1],
                    BoundaryPolicy::InBounds,
                )
                .unwrap();
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap()
            })
            .to_vec();
        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Sequential); 2]).unwrap(),
            parameters,
        )
        .unwrap();
        let zeroed_body = |firsts: [usize; 2]| {
            let mut write = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = write.add_input(call.parameters()[0].body_type());
            let zeros = write.add_constant(ArrayIrValue::Array(Array::vector(vec![0i32]).unwrap()));
            write.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, zeros], None).unwrap();
            let write = write
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            let mut skip = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            skip.add_input(call.parameters()[0].body_type());
            let skip = skip
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let outputs = call
                .parameters()
                .iter()
                .map(|parameter| builder.add_input(parameter.body_type()))
                .collect::<Vec<_>>();
            let coordinates = call
                .coordinate_types()
                .iter()
                .map(|r#type| builder.add_input(ArrayIrType::Dimension(r#type.clone())))
                .collect::<Vec<_>>();
            let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
            let write = builder.import_region(write.entry_region_ref());
            let skip = builder.import_region(skip.entry_region_ref());
            for (&output, first) in outputs.iter().zip(firsts) {
                let predicate = builder
                    .add_instruction(
                        ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::Equal)),
                        vec![],
                        vec![coordinates[first], zero],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(
                        ArrayIrOperation::Condition(ConditionOperation::new()),
                        vec![write, skip],
                        vec![predicate, output],
                        None,
                    )
                    .unwrap();
            }
            let update = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![1i32]).unwrap()));
            for &output in &outputs {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), vec![], vec![output, update], None)
                    .unwrap();
            }
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 4], vec![])
                .unwrap()
        };
        let carry_axes = zeroed_body([0, 1]);
        assert_eq!(validate_kernel_initialization(carry_axes.entry_region_ref(), &call, 0), Ok(()));
        let swapped_axes = zeroed_body([1, 0]);
        let region = swapped_axes.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(swapped_axes.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 5),
                root: ReferenceRoot::RegionInput { region, input_index: 1 },
            }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_first_step_initialization_requires_complete_first_step() {
        // The first step reads an empty window when the write happens at `k == 1` instead, or covers only part of it.
        let call = carried_call(&[(3, GridExecution::Sequential)]);
        let late = carried_body(&call, 1, false);
        let region = late.entry_region_ref().id();
        let expected = Err(KernelInitializationError::UninitializedRead {
            instruction: InstructionId::new(region, 3),
            root: ReferenceRoot::RegionInput { region, input_index: 0 },
        });
        assert_eq!(validate_kernel_initialization(late.entry_region_ref(), &call, 0), expected);
        let partial = carried_body(&call, 0, true);
        assert_eq!(validate_kernel_initialization(partial.entry_region_ref(), &call, 0), expected);
    }

    #[test]
    fn test_validate_kernel_initialization_first_step_initialization_requires_one_sequential_axis() {
        // A window repeated along a parallel axis or along two axes has no carry axis, so the conditional write is not
        // credited, and body qualification rejects the accumulation before launch qualification sees the overlap.
        for dimensions in
            [vec![(3, GridExecution::Parallel)], vec![(2, GridExecution::Sequential), (3, GridExecution::Sequential)]]
        {
            let call = carried_call(&dimensions);
            let body = carried_body(&call, 0, false);
            let region = body.entry_region_ref().id();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 0),
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 3),
                    root: ReferenceRoot::RegionInput { region, input_index: 0 },
                }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_empty_launch() {
        // The output is carried along `k`, so the late write is rejected once the launch has a point. An empty launch
        // runs no body, so its body accesses are not analyzed, but a nonempty write-only output is still uncovered.
        let nonempty = carried_call(&[(1, GridExecution::Parallel), (3, GridExecution::Sequential)]);
        let early = carried_body(&nonempty, 0, false);
        assert_eq!(validate_kernel_initialization(early.entry_region_ref(), &nonempty, 0), Ok(()));
        let late = carried_body(&nonempty, 1, false);
        let region = late.entry_region_ref().id();
        assert_eq!(
            validate_kernel_initialization(late.entry_region_ref(), &nonempty, 0),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 3),
                root: ReferenceRoot::RegionInput { region, input_index: 0 },
            }),
        );
        let empty = carried_call(&[(0, GridExecution::Parallel), (3, GridExecution::Sequential)]);
        let late = carried_body(&empty, 1, false);
        assert_eq!(
            validate_kernel_initialization(late.entry_region_ref(), &empty, 0),
            Err(KernelInitializationError::IncompleteOutput { parameter: 0 }),
        );

        // Reads an unwritten scratch allocation and leaves the empty write-only output alone.
        for programs in [1, 0] {
            let call = call(0, 0, programs);
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let scratch = builder
                .add_instruction(
                    KernelOperation::Scratch(
                        ScratchOperation::new(ArrayType::new_static(DataType::I32, vec![2]), 4).unwrap(),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            builder.add_instruction(ReferenceFreezeOperation::new(), vec![], vec![scratch], None).unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            let expected = if programs == 0 {
                Ok(())
            } else {
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 1),
                    root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
                })
            };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), expected);
        }
    }

    #[test]
    fn test_validate_kernel_initialization_empty_launch_carry_axes() {
        // Builds a grid whose first axis has extent `first` followed by `usize::BITS` sequential axes with extent 2,
        // and one write-only output for each of those axes, which is carried along it because its separable windows
        // follow every other axis with extent 2. Each window is the whole output, whose leading axis has a constant
        // start and extent 0, except for the first output when `nonempty` is set, whose leading axis has extent 1.
        let carried = usize::BITS as usize;
        let carried_axes_call = |first: usize, nonempty: bool| {
            let parameters = (0..carried)
                .map(|carry_axis| {
                    let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                    let coordinates = (0..=carried)
                        .map(|axis| {
                            mapping.add_input(
                                DimensionType::new(
                                    format!("coordinate_{axis}"),
                                    DimensionBounds::non_negative(None).unwrap(),
                                )
                                .into(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let zero = mapping.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
                    let starts = std::iter::once(zero)
                        .chain((1..=carried).filter(|&axis| axis != carry_axis + 1).map(|axis| coordinates[axis]))
                        .collect::<Vec<_>>();
                    let rank = starts.len();
                    let mut shape = vec![1; rank];
                    if carry_axis != 0 || !nonempty {
                        shape[0] = 0;
                    }
                    let mapping = BlockMapping::new(
                        mapping.build(starts, vec![Placeholder; carried + 1], vec![Placeholder; rank]).unwrap(),
                        shape.clone(),
                        BoundaryPolicy::InBounds,
                    )
                    .unwrap();
                    KernelParameter::new(
                        ArrayType::new_static(DataType::I32, shape),
                        KernelParameterAccess::WriteOnly,
                        mapping,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let mut grid = vec![GridDimension::new(Dimension::Static(first), GridExecution::Parallel)];
            grid.extend((0..carried).map(|_| GridDimension::new(Dimension::Static(2), GridExecution::Sequential)));
            let call = KernelCallOperation::new(Grid::new(grid).unwrap(), parameters).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            for parameter in call.parameters() {
                builder.add_input(parameter.body_type());
            }
            for coordinate in call.coordinate_types() {
                builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            }
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![],
                    vec![Placeholder; call.body_input_types().len()],
                    vec![],
                )
                .unwrap();
            (call, body)
        };
        let (call, body) = carried_axes_call(0, false);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
        let (call, body) = carried_axes_call(0, true);
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::IncompleteOutput { parameter: 0 }),
        );
        // A nonempty launch with that many carry axes has more points than a `usize` can count.
        let (call, body) = carried_axes_call(1, false);
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Call(KernelError::Grid(GridError::ProgramCountOverflow))),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_program_count_overflow() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        for _ in 0..2 {
            mapping.add_input(DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap()).into());
        }
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(usize::MAX / 2), GridExecution::Sequential); 2])
                .unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, mapping)
                    .unwrap(),
            ],
        )
        .unwrap();
        let body = repeated_body(&call, &[ReferenceAccessMode::Read]);
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Call(KernelError::Grid(GridError::ProgramCountOverflow))),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_enumeration_limit() {
        let mut mapping = ProgramBuilder::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
        let coordinate = mapping.add_input(coordinate_type.clone().into());
        let zero = DimensionValue::constant(0).unwrap();
        let constant = mapping.add_constant(ArrayIrValue::Dimension(zero.clone()));
        let start = mapping
            .add_instruction(
                DimensionAddOperation::new(&coordinate_type, zero.r#type().as_ref()).unwrap(),
                vec![],
                vec![coordinate, constant],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        assert_eq!(mapping.tiling_axes(), None);
        let nonempty = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![1]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let nonempty_body = body(&nonempty, Some(false));
        assert_eq!(
            validate_kernel_initialization(nonempty_body.entry_region_ref(), &nonempty, 0),
            Err(KernelInitializationError::QualificationLimit { programs: 1, maximum: 0 }),
        );
        let empty = call(0, 0, 0);
        let empty_body = body(&empty, None);
        assert_eq!(validate_kernel_initialization(empty_body.entry_region_ref(), &empty, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_separable_tiling_does_not_enumerate() {
        let call = call(1_000_000, 1, 1_000_000);
        let body = body(&call, Some(false));
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_separable_tiling_checks_maximal_bounds() {
        let call = call(3, 2, 2);
        let body = body(&call, Some(false));
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Mapping(BlockMappingError::OutOfBounds {
                axis: 0,
                start: 2,
                limit: 4,
                extent: 3,
            })),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_separable_and_repeated_axes() {
        // Writes an `i32[2, 2]` output whose windows follow two parallel grid axes, or follow one parallel grid axis
        // along both array axes when `repeated` is set.
        let writes = |repeated: bool| {
            let mut mapping = ProgramBuilder::new();
            let rank = if repeated { 1 } else { 2 };
            let inputs = (0..rank)
                .map(|axis| {
                    mapping.add_input(
                        DimensionType::new(format!("coordinate_{axis}"), DimensionBounds::non_negative(None).unwrap())
                            .into(),
                    )
                })
                .collect::<Vec<_>>();
            let mapping = BlockMapping::new(
                mapping
                    .build(
                        vec![inputs[0], inputs[if repeated { 0 } else { 1 }]],
                        vec![Placeholder; rank],
                        vec![Placeholder; 2],
                    )
                    .unwrap(),
                vec![1, 1],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            let call = KernelCallOperation::new(
                Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel); rank]).unwrap(),
                vec![
                    KernelParameter::new(
                        ArrayType::new_static(DataType::I32, vec![2, 2]),
                        KernelParameterAccess::WriteOnly,
                        mapping,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            for coordinate in call.coordinate_types() {
                builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            }
            let value = builder.add_constant(ArrayIrValue::Array(
                Array::new(ArrayType::new_static(DataType::I32, vec![1, 1]), vec![0; 4]).unwrap(),
            ));
            builder
                .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
                .unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![],
                    vec![Placeholder; rank + 1],
                    vec![],
                )
                .unwrap();
            (call, body)
        };
        let (separable_call, separable) = writes(false);
        assert_eq!(validate_kernel_initialization(separable.entry_region_ref(), &separable_call, 0), Ok(()));
        let (diagonal_call, diagonal) = writes(true);
        assert_eq!(
            validate_kernel_initialization(diagonal.entry_region_ref(), &diagonal_call, 0),
            Err(KernelInitializationError::QualificationLimit { programs: 2, maximum: 0 }),
        );
        assert_eq!(
            validate_kernel_initialization(diagonal.entry_region_ref(), &diagonal_call, 2),
            Err(KernelInitializationError::IncompleteOutput { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_requires_global_output_coverage() {
        for (extent, programs) in [(8, 3), (8, 0)] {
            let call = call(extent, 2, programs);
            let body = body(&call, Some(false));
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 64),
                Err(KernelInitializationError::IncompleteOutput { parameter: 0 }),
            );
        }
        let call = call(0, 0, 0);
        let body = body(&call, None);
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_masked_policy_and_overlapping_windows() {
        let original = call(4, 2, 2);
        // A masked parameter keeps the original advancing windows, and an unmasked parameter starts every window at 0.
        let modified = |masked: bool| {
            let mapping = original.parameters()[0].mapping();
            let program = if masked {
                mapping.program().clone()
            } else {
                let mut builder = ProgramBuilder::new();
                builder.add_input(mapping.program().input_types()[0].clone());
                let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
                builder.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap()
            };
            let mapping = BlockMapping::new(
                program,
                vec![2],
                if masked { BoundaryPolicy::Masked } else { BoundaryPolicy::InBounds },
            )
            .unwrap();
            KernelCallOperation::new(
                original.grid().clone(),
                vec![
                    KernelParameter::new(
                        ArrayType::new_static(DataType::I32, vec![4]),
                        KernelParameterAccess::WriteOnly,
                        mapping,
                    )
                    .unwrap(),
                ],
            )
            .unwrap()
        };
        let masked = modified(true);
        let masked_body = body(&masked, Some(false));
        assert_eq!(validate_kernel_initialization(masked_body.entry_region_ref(), &masked, 64), Ok(()));
        let overlapping = modified(false);
        let overlapping_body = body(&overlapping, Some(false));
        assert_eq!(
            validate_kernel_initialization(overlapping_body.entry_region_ref(), &overlapping, 64),
            Err(KernelInitializationError::OverlappingWindows { parameter: 0 }),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_ordered_repeated_writes() {
        for separable in [true, false] {
            let call = repeated_call(&[GridExecution::Sequential], separable);
            let writes = repeated_body(&call, &[ReferenceAccessMode::Write, ReferenceAccessMode::Write]);
            assert_eq!(validate_kernel_initialization(writes.entry_region_ref(), &call, 2), Ok(()));
            let accumulation = repeated_body(&call, &[ReferenceAccessMode::Accumulate]);
            assert_eq!(validate_kernel_initialization(accumulation.entry_region_ref(), &call, 2), Ok(()));
            let parallel = repeated_call(&[GridExecution::Parallel], separable);
            let writes = repeated_body(&parallel, &[ReferenceAccessMode::Write]);
            assert_eq!(
                validate_kernel_initialization(writes.entry_region_ref(), &parallel, 2),
                Err(KernelInitializationError::OverlappingWindows { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_parallel_atomic_accumulation() {
        for separable in [true, false] {
            let call = repeated_call(&[GridExecution::Parallel], separable);
            let atomic = repeated_body(&call, &[ReferenceAccessMode::AtomicAccumulate]);
            assert_eq!(validate_kernel_initialization(atomic.entry_region_ref(), &call, 2), Ok(()));
            let mixed = repeated_body(&call, &[ReferenceAccessMode::Read, ReferenceAccessMode::AtomicAccumulate]);
            assert_eq!(
                validate_kernel_initialization(mixed.entry_region_ref(), &call, 2),
                Err(KernelInitializationError::OverlappingWindows { parameter: 0 }),
            );
            // Accumulation that is not atomic through the same window of two unordered grid points is a race.
            let ordered = repeated_body(&call, &[ReferenceAccessMode::Accumulate]);
            assert_eq!(
                validate_kernel_initialization(ordered.entry_region_ref(), &call, 2),
                Err(KernelInitializationError::OverlappingWindows { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_parallel_disjoint_accumulation() {
        // Every parallel grid point accumulates into its own window of size 1, so no element has two unordered writers.
        let original = call(2, 1, 2);
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2]),
                    KernelParameterAccess::ReadWrite,
                    original.parameters()[0].mapping().clone(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let update = builder.add_constant(contents(&call));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), vec![], vec![reference, update], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_incomparable_sequential_coordinates() {
        for separable in [true, false] {
            let call = repeated_call(&[GridExecution::Sequential, GridExecution::Sequential], separable);
            let body = repeated_body(&call, &[ReferenceAccessMode::Write]);
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 4),
                Err(KernelInitializationError::OverlappingWindows { parameter: 0 }),
            );
        }
    }

    #[test]
    fn test_validate_kernel_initialization_edge_windows_require_masked_reads() {
        let original = call(3, 2, 2);
        let mapping =
            BlockMapping::new(original.parameters()[0].mapping().program().clone(), vec![2], BoundaryPolicy::Masked)
                .unwrap();
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![3]),
                    KernelParameterAccess::ReadWrite,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        // Reads the window, or swaps into it and writes the old value back only when `live` is set, either as a whole
        // or only at the `selected` element.
        let edge_body = |swap: bool, live: bool, selected: Option<usize>| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let transforms = selected
                .map(|index| {
                    vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(index) }]
                })
                .unwrap_or_default();
            if swap {
                let value = builder.add_constant(if selected.is_some() {
                    ArrayIrValue::Array(Array::scalar(0i32).unwrap())
                } else {
                    contents(&call)
                });
                let old = builder
                    .add_instruction(
                        ReferenceSwapOperation::new().with_transforms(transforms.clone()),
                        vec![],
                        vec![reference, value],
                        None,
                    )
                    .unwrap()[0];
                if live {
                    builder
                        .add_instruction(
                            ReferenceWriteOperation::new().with_transforms(transforms),
                            vec![],
                            vec![reference, old],
                            None,
                        )
                        .unwrap();
                }
            } else {
                builder
                    .add_instruction(
                        ReferenceReadOperation::new().with_transforms(transforms),
                        vec![],
                        vec![reference],
                        None,
                    )
                    .unwrap();
            }
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap()
        };

        // Only the first element of the edge window is valid in every grid point.
        let whole_read = edge_body(false, false, None);
        assert_eq!(
            validate_kernel_initialization(whole_read.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_read",
            })),
        );
        let valid_read = edge_body(false, false, Some(0));
        assert_eq!(validate_kernel_initialization(valid_read.entry_region_ref(), &call, 0), Ok(()));
        let padding_read = edge_body(false, false, Some(1));
        assert_eq!(
            validate_kernel_initialization(padding_read.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_read",
            })),
        );

        // A swap whose old value is unused lowers to a store, which never reads the padded window.
        let dead_swap = edge_body(true, false, None);
        assert_eq!(validate_kernel_initialization(dead_swap.entry_region_ref(), &call, 0), Ok(()));
        let live_swap = edge_body(true, true, None);
        assert_eq!(
            validate_kernel_initialization(live_swap.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_swap",
            })),
        );
        let valid_swap = edge_body(true, true, Some(0));
        assert_eq!(validate_kernel_initialization(valid_swap.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_edge_windows_reject_dynamic_reads() {
        // An unmasked dynamic read may select any element of its tile, so it is rejected exactly when some grid point's
        // tile has padding, i.e., for the extent 3 but not for the extent 4.
        let dynamic_read = |extent: usize| {
            let original = call(extent, 2, 2);
            let mapping = BlockMapping::new(
                original.parameters()[0].mapping().program().clone(),
                vec![2],
                BoundaryPolicy::Masked,
            )
            .unwrap();
            let call = KernelCallOperation::new(
                original.grid().clone(),
                vec![
                    KernelParameter::new(
                        ArrayType::new_static(DataType::I32, vec![extent]),
                        KernelParameterAccess::ReadWrite,
                        mapping,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
            builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Dynamic,
                    }]),
                    vec![],
                    vec![reference, index],
                    None,
                )
                .unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            (call, body)
        };
        let (padded_call, padded) = dynamic_read(3);
        assert_eq!(
            validate_kernel_initialization(padded.entry_region_ref(), &padded_call, 0),
            Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_read",
            })),
        );
        let (exact_call, exact) = dynamic_read(4);
        assert_eq!(validate_kernel_initialization(exact.entry_region_ref(), &exact_call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_write_only_edge_windows_require_masked_reads() {
        // Writing the whole tile initializes its padding only in private storage, so reading it back unmasked would
        // still expose padding as data.
        let original = call(3, 2, 2);
        let mapping =
            BlockMapping::new(original.parameters()[0].mapping().program().clone(), vec![2], BoundaryPolicy::Masked)
                .unwrap();
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![3]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
            .unwrap();
        builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            validate_kernel_initialization(body.entry_region_ref(), &call, 0),
            Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_read",
            })),
        );
    }

    #[test]
    fn test_validate_kernel_initialization_edge_windows_admit_masked_fallback() {
        let original = call(3, 2, 2);
        let mapping =
            BlockMapping::new(original.parameters()[0].mapping().program().clone(), vec![2], BoundaryPolicy::Masked)
                .unwrap();
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![3]),
                    KernelParameterAccess::ReadWrite,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let mask = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![true, true]).unwrap()));
        let other = builder.add_constant(contents(&call));
        builder
            .add_instruction(MaskedLoadOperation::new(), vec![], vec![reference, mask, other], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_validate_kernel_initialization_masked_dead_swap_is_store() {
        let original = call(3, 2, 2);
        let mapping =
            BlockMapping::new(original.parameters()[0].mapping().program().clone(), vec![2], BoundaryPolicy::Masked)
                .unwrap();
        let call = KernelCallOperation::new(
            original.grid().clone(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![3]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(contents(&call));
        builder
            .add_instruction(ReferenceSwapOperation::new(), vec![], vec![reference, value], None)
            .unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
    }

    #[test]
    fn test_contains() {
        assert!(contains(&[0..3, 5..8], &(1..3)));
        assert!(contains(&[], &(0..0)));
        assert!(!contains(&[0..3, 5..8], &(2..6)));
    }

    #[test]
    fn test_insert() {
        let mut intervals = vec![0..2, 4..6];
        insert(&mut intervals, 2..4);
        insert(&mut intervals, 1..3);
        insert(&mut intervals, 8..8);
        assert_eq!(intervals, vec![0..6]);
    }

    #[test]
    fn test_masked_ranges() {
        assert_eq!(masked_ranges(5..10, &[false, true, true, false, true]), vec![6..8, 9..10]);
        assert_eq!(masked_ranges(3..3, &[]), vec![]);
        assert_eq!(masked_ranges(2..4, &[false, false]), vec![]);
    }

    #[test]
    fn test_intersect_states() {
        let root = ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 };
        assert_eq!(
            intersect_states(BTreeMap::from([(root, vec![0..4, 6..10])]), &BTreeMap::from([(root, vec![2..8])])),
            BTreeMap::from([(root, vec![2..4, 6..8])]),
        );
    }
}
