//! Contains the host interpreter for qualified kernel calls, which executes kernels on host [`Array`]s through
//! canonical references and program replay.
//!
//! The host interpreter is the reference semantics of a kernel. It qualifies a [`KernelCallOperation`] together with
//! its attached body, allocates private reference storage, and then replays the verified body once per grid point.
//! It requires no device and no compiler, so it serves both as the executable specification against which compiled
//! kernels are checked and as a debugger for kernel bodies.
//!
//! # Qualification and Storage
//!
//! Scalar prefetch values are bound through canonical program specialization (i.e.,
//! [`KernelDefinition::specialize_prefetch`]) before qualification or allocation. Grid extents must already be
//! static, so the interpreter never chooses an unresolved dynamic launch extent. Qualification then proves
//! initialization, coverage, and disjointness before any storage is created, and it produces the validated reference
//! summary that replay uses to lower each swap whose old value is never used to a store.
//!
//! Input arrays remain immutable. Every invocation owns one private reference root per kernel parameter, which starts
//! from the input array of a read-only or read-write parameter and from uninitialized storage for a write-only
//! parameter. Only the completed roots of writable parameters become results, and a failure never publishes a
//! partially updated array.
//!
//! Uninitialized storage (i.e., write-only roots, the private tiles of masked windows, and scratch storage) is
//! allocated by [`Array::uninitialized`] with the checked storage size of its type. Its zero bytes are a valid encoding
//! of every data type, including types that cannot represent the number zero (e.g., [`DataType::F8E8M0FNU`]), so
//! allocation never requires a numeric zero. No execution observes these contents, because qualification rejects
//! every read that does not follow a definite write, the padding of a masked window is never read or published, and
//! a carried window reloads only the valid elements that the previous step published.
//!
//! # Grid Point Replay
//!
//! For each grid point, the interpreter evaluates every parameter mapping at the point's coordinates and builds the
//! window that the body receives for that parameter.
//!
//!   - An in-bounds window is a view of the root through the window's valid transform, so it aliases the root and
//!     every write to it updates the root directly.
//!   - A masked window is a private uninitialized tile whose type is the referent type of the parameter's body type.
//!     The tile loads the valid region of the root before the body runs and records which of its elements are valid.
//!     Unless its parameter is read-only, it publishes its valid region back to the root after the body completes, so
//!     private padding is never published.
//!
//! The body then receives these windows followed by the grid point's coordinates. Masked loads, stores, and swaps on
//! a masked window intersect their explicit mask with the validity of the elements that they select. Any other access
//! that reads a selection of a masked window containing invalid elements fails with
//! [`KernelMemoryError::UnmaskedWindowAccess`]. A swap whose old value is never used is lowered to a store by
//! qualification, so replay executes it as the equivalent `reference_write` or [`MaskedStoreOperation`], which reads
//! nothing. Its trace entry records a write, and its unused result is uninitialized storage that the NaN checks never
//! inspect. A [`WindowMaskOperation`](crate::memory::WindowMaskOperation) returns the validity of the elements that its
//! reference selects without accessing storage, and every element of a window without recorded validity (e.g., an
//! in-bounds window) is valid. A [`ScratchOperation`](crate::memory::ScratchOperation) allocates fresh uninitialized
//! storage on every binding.
//!
//! An [`AsyncCopyOperation`](crate::memory::AsyncCopyOperation) captures the initialized source elements when it is
//! issued and records a pending copy under a fresh completion token. The matching
//! [`WaitOperation`](crate::memory::WaitOperation) removes that pending copy and only then writes its destination. A
//! wait on an unknown or consumed token fails with [`KernelInterpretationError::UnknownCopyToken`], and a grid point
//! that completes with pending copies fails with [`KernelInterpretationError::PendingCopies`] before any of its
//! windows are published. The token of an asynchronous extension operation is pending under the
//! [`KernelExtension::completion_protocol`] of that operation until an extension wait of the same protocol consumes it,
//! while the interpretation rules of the extension perform its data movement. A commit or wait that does not follow
//! the protocol of the producer of its token, including a portable wait of an extension token and an extension commit
//! or wait of a portable copy, fails with [`KernelInterpretationError::IncompatibleCompletion`] before it runs, so its
//! token stays pending and no deferred write is published. Ordinary reference effects retain their existing semantics.
//! The host traversal chooses one admitted order for parallel atomic updates, and it does not promise bitwise agreement
//! across other valid floating-point orders. [`KernelDefinition::interpret_interleavings`] explores the other orders of
//! small kernels.
//!
//! Replay binds every operation through a private qualified context. That context records a trace entry when tracing,
//! charges one replay step, applies the optional NaN checks, and handles the kernel memory operations above. It
//! interprets every other operation through its canonical eager rule, and replays the regions that such a rule enters
//! through the same context, so nested operations undergo the same checks and every region entry also consumes a
//! replay step.
//!
//! # Entry Points
//!
//!   - [`KernelDefinition::interpret`] interprets a definition with no step limit, because kernel bodies admit only
//!     bounded loops. Its `maximum_programs` argument bounds only the grid enumeration of qualification, which
//!     evaluates the mapping of a parameter at every grid point when
//!     [`BlockMapping::tiling_axes`](crate::mappings::BlockMapping::tiling_axes) does not recognize it or when its
//!     recognized axes repeat a grid axis (e.g., the diagonal element starts `(i, i)`, which are recognized). Ordinary
//!     [`Program::interpret`](ryft_core::Program::interpret) of a `kernel_call` takes the same path with
//!     [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`].
//!   - [`KernelDefinition::interpret_with_limits`] caps both the number of grid points, whether or not qualification
//!     enumerates them, and the number of replay steps (i.e., operation bindings and region entries) across the whole
//!     invocation.
//!   - [`KernelDefinition::interpret_with_trace`] applies the caps of a [`KernelDebugOptions`], which its
//!     [`KernelDebugOptions::with_maximum_programs`], [`KernelDebugOptions::with_maximum_steps`], and
//!     [`KernelDebugOptions::with_nan_checks`] builder functions configure, and it records a [`KernelTraceEntry`] for
//!     every attempted operation binding, with a [`KernelTraceAccess`] for each of its reference accesses.
//!
//! Trace entries precede the effects of their operations and the resolution of their reference views, so the trace
//! retains the failing binding of a failed invocation, even when its view cannot be resolved (e.g., a dynamic index
//! into an empty axis). Such an entry lists the accesses resolved before the failure, and the binding still consumes
//! its replay step. To investigate a failure, retain the trace before propagating the execution result. Optional NaN
//! checks inspect ordinary operation inputs and outputs, including complex components and low precision
//! floating-point formats. They never read reference storage, so they never read uninitialized scratch merely to
//! inspect its contents. Precision and rounding remain those of the canonical operations, and diagnostics do not
//! substitute an execution model with wider precision.
//!
//! ```
//! use ryft_core::{Array, ProgramError};
//! use ryft_kernels::{KernelDebugOptions, KernelDefinition};
//!
//! # fn inspect(definition: &KernelDefinition, inputs: Vec<Array>) -> Result<(), ProgramError> {
//! let options = KernelDebugOptions::default().with_maximum_steps(10_000).with_nan_checks(true);
//! let mut trace = Vec::new();
//! let result = definition.interpret_with_trace(inputs, &options, &mut trace);
//! for entry in trace {
//!     eprintln!("{entry}");
//! }
//! let outputs = result?;
//! # Ok(())
//! # }
//! ```

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use num_complex::Complex;
use ryft_core::{
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReference, ArrayReferenceTransform,
    ArraySliceAxis, ArrayType, AttachedRegions, Context, DataType, DimensionValue, Domain, EagerContext,
    EagerInterpretationDriver, InstructionId, InterpretableOperation, InterpretationDriver, Operation, ProgramError,
    Provenance, ProvenanceScope, ProvenanceState, ReferenceAccessMode, ReferenceId, ReferenceWriteOperation,
    RegionDriver, RegionInterpreter, RegionRef, TypeError, Typed, TypedProgram, ValueResolution,
    validated_reference_access_descriptors,
};
use thiserror::Error;

use crate::calls::{KernelCallOperation, KernelDefinition};
use crate::initialization::{KernelInitializationError, validated_kernel_references};
use crate::mappings::BoundaryPolicy;
use crate::memory::{KernelMemoryError, MaskedStoreOperation, WINDOW_MASK_OPERATION_NAME};
use crate::operations::{
    KernelCompletionProtocol, KernelExtension, KernelExtensionMemory, KernelOperation, admits_completion,
};
use crate::validation::{KernelParameterAccess, KernelReferenceOperation, KernelReferenceSummary, KernelSwapLowering};

/// Failure of the host interpreter while it replays a qualified kernel body. The replay fails when an enabled NaN
/// check finds a NaN, when the asynchronous copy protocol is violated at runtime, or when a debug entry point exhausts
/// its replay step limit. Qualification failures are reported separately as [`KernelInitializationError`]s. The
/// debugging limits and checks never alter the device semantics of the kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelInterpretationError {
    /// Optional numerical checking found a NaN in an ordinary value. Reference storage is never read for this check.
    #[error(
        "nan in `{operation}` {position} {value} element {element} of type `{data_type}` at grid point {coordinate:?}"
    )]
    Nan {
        /// Canonical operation that produced or consumed the value.
        operation: &'static str,

        /// Whether the value is an operation `input` or `output`.
        position: &'static str,

        /// Position in that operation's input or output list.
        value: usize,

        /// Logical row-major element index. Either complex component may be NaN.
        element: usize,

        /// Original element type, before diagnostic inspection.
        data_type: DataType,

        /// Logical grid coordinate at the failing operation.
        coordinate: Vec<usize>,

        /// Original source provenance at the failing operation.
        provenance: Provenance,
    },

    /// A commit or wait did not identify a pending copy owned by this invocation.
    #[error("async copy completion token is unknown or already consumed")]
    UnknownCopyToken {
        /// Canonical allocation identity of the supplied completion token.
        token: ReferenceId,
    },

    /// A commit or wait does not follow the completion protocol of the operation that produced its pending token (i.e.,
    /// it belongs to another protocol, it waits before a commit that the protocol requires, or it repeats a commit
    /// that the protocol requires exactly once). The token stays pending and no deferred write is published.
    #[error("`{operation}` does not follow the completion protocol of the producer of its token")]
    IncompatibleCompletion {
        /// Canonical commit or wait operation.
        operation: &'static str,

        /// Canonical allocation identity of the supplied completion token.
        token: ReferenceId,
    },

    /// Defensive publication check for copies that have not completed successfully.
    #[error("kernel grid point {coordinate:?} has {count} copies without a wait")]
    PendingCopies {
        /// Number of incomplete copies in the program instance.
        count: usize,

        /// Logical grid coordinate whose output cannot be published.
        coordinate: Vec<usize>,
    },

    /// The invocation exhausted its budget of operation bindings and region entries before successful completion.
    #[error("kernel interpretation exceeded {maximum_steps} steps at grid point {coordinate:?} in `{operation}`")]
    StepLimit {
        /// Maximum replay steps requested for the complete invocation.
        maximum_steps: usize,

        /// Logical grid coordinate being interpreted when the limit was reached.
        coordinate: Vec<usize>,

        /// Canonical operation executing or entering an attached region.
        operation: &'static str,

        /// Existing program provenance at the failing step.
        provenance: Provenance,
    },
}

/// Default host replay budget of [`KernelDebugOptions`], counting operation bindings and region entries across the
/// entire grid. Counting region entries also bounds loops with constant conditions and empty bodies. Only the debug
/// entry points [`KernelDefinition::interpret_with_limits`] and [`KernelDefinition::interpret_with_trace`] apply a step
/// limit. [`KernelDefinition::interpret`], [`KernelCall`](crate::authoring::KernelCall) for host arrays, and ordinary
/// program interpretation of a `kernel_call` replay without one, because kernel bodies admit only bounded loops. This
/// limit is excluded from semantic identity.
pub const DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS: usize = 1_000_000;

/// Default grid size bound for host interpretation. Ordinary
/// [`Program::interpret`](ryft_core::programs::Program::interpret) of a `kernel_call` and
/// [`KernelCall`](crate::authoring::KernelCall) for host arrays use it as the maximum grid size that qualification
/// enumerates. Qualification enumerates the grid only for a parameter whose mapping
/// [`BlockMapping::tiling_axes`](crate::mappings::BlockMapping::tiling_axes) does not recognize or whose recognized
/// axes repeat a grid axis, so launches without such a parameter are not limited by it. [`KernelDebugOptions`] uses it
/// as a hard cap on grid points for [`KernelDefinition::interpret_with_trace`].
/// [`KernelDefinition::interpret`] and [`KernelDefinition::interpret_with_limits`] accept an explicit bound instead.
/// This is a host resource limit, not part of a kernel's semantic identity or a device launch limit.
pub const DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS: usize = 10_000;

/// Debugging choices that affect only host interpretation through [`KernelDefinition::interpret_with_trace`].
/// Mandatory bounds, race, and initialization validation cannot be disabled. Numerical checking observes ordinary
/// values without changing their data type, rounding, or accumulator behavior, which remain properties of canonical
/// operations. These options are excluded from kernel and compiler identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelDebugOptions {
    /// Maximum logical grid size admitted before storage is allocated.
    maximum_programs: usize,

    /// Maximum operation bindings and region entries across the invocation.
    maximum_steps: usize,

    /// Rejects NaNs in ordinary floating-point or complex operation inputs and outputs. Infinity is allowed.
    nan_checks: bool,
}

impl Default for KernelDebugOptions {
    fn default() -> Self {
        Self {
            maximum_programs: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
            maximum_steps: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
            nan_checks: false,
        }
    }
}

impl KernelDebugOptions {
    /// Returns these options with `maximum_programs` as the maximum logical grid size admitted before storage is
    /// allocated. A launch with more grid points fails before qualification, even when qualification would not
    /// enumerate them. The default is [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`].
    pub fn with_maximum_programs(mut self, maximum_programs: usize) -> Self {
        self.maximum_programs = maximum_programs;
        self
    }

    /// Returns these options with `maximum_steps` as the maximum number of operation bindings and region entries
    /// across the invocation. Zero permits an empty grid but rejects the first body entry of a nonempty launch. The
    /// default is [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS`].
    pub fn with_maximum_steps(mut self, maximum_steps: usize) -> Self {
        self.maximum_steps = maximum_steps;
        self
    }

    /// Returns these options with NaN checking enabled when `nan_checks` is `true` and disabled otherwise. While it is
    /// enabled, every operation binding inspects its ordinary floating-point and complex array inputs before the
    /// operation runs and its array outputs after it, and the interpretation fails with
    /// [`KernelInterpretationError::Nan`] at the first element that is NaN, or whose real or imaginary component is
    /// NaN. Infinities are allowed. Reference storage is never read for this check, and the check never changes data
    /// types, rounding, or accumulation. NaN checking is disabled by default.
    pub fn with_nan_checks(mut self, nan_checks: bool) -> Self {
        self.nan_checks = nan_checks;
        self
    }

    /// Returns the maximum logical grid size admitted before storage is allocated.
    pub fn maximum_programs(&self) -> usize {
        self.maximum_programs
    }

    /// Returns the maximum number of operation bindings and region entries across the invocation.
    pub fn maximum_steps(&self) -> usize {
        self.maximum_steps
    }

    /// Returns whether interpretation rejects NaNs in ordinary floating-point or complex operation inputs and outputs.
    pub fn nan_checks(&self) -> bool {
        self.nan_checks
    }
}

/// One attempted operation binding during host replay, which [`KernelDefinition::interpret_with_trace`] records.
/// Entries precede effects and the resolution of reference views, so the final entry is retained when its operation
/// fails, with the accesses whose views were resolved before the failure. Reference identities are numbered by first
/// observation within this invocation, independent of the allocation IDs of the process. Budget checks for region
/// entries do not create operation entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelTraceEntry {
    /// Logical program coordinate in the declared grid order.
    coordinate: Vec<usize>,

    /// Canonical operation name.
    operation: &'static str,

    /// Canonical provenance active at this binding.
    provenance: Provenance,

    /// Input reference accesses in the operation's declared effect order.
    accesses: Vec<KernelTraceAccess>,

    /// Identities of the copies pending before this operation starts, numbered within this invocation.
    pending_copies: Vec<usize>,
}

impl KernelTraceEntry {
    /// Returns the logical program coordinate in the declared grid order.
    pub fn coordinate(&self) -> &[usize] {
        &self.coordinate
    }

    /// Returns the canonical operation name.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the canonical provenance active at this binding.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Returns the input reference accesses in the operation's declared effect order.
    pub fn accesses(&self) -> &[KernelTraceAccess] {
        &self.accesses
    }

    /// Returns the identities of the copies pending before this operation starts, numbered within this invocation
    /// and in ascending order.
    pub fn pending_copies(&self) -> &[usize] {
        &self.pending_copies
    }
}

impl Display for KernelTraceEntry {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?} {}", self.coordinate, self.operation)?;
        for access in &self.accesses {
            write!(formatter, " input[{}]={}(root[{}]{})", access.input, access.mode, access.root, access.path)?;
        }
        write!(formatter, " pending={:?}", self.pending_copies)?;
        if !self.provenance.is_unknown() {
            write!(formatter, " at {}", self.provenance)?;
        }
        Ok(())
    }
}

/// A reference access in a [`KernelTraceEntry`], with its canonical transform path and the identity of its root within
/// this invocation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelTraceAccess {
    /// Input position of the accessed reference.
    input: usize,

    /// Exact canonical reference access mode.
    mode: ReferenceAccessMode,

    /// Allocation number assigned on first observation during this invocation.
    root: usize,

    /// Canonical transform path from this allocation to the selected elements, rendered as the [`Display`] form of
    /// each [`ArrayReferenceTransform`] prefixed with `/` (e.g., `/slice(axes=[0:2])`). An empty path selects the
    /// complete allocation.
    path: String,
}

impl KernelTraceAccess {
    /// Returns the input position of the accessed reference.
    pub fn input(&self) -> usize {
        self.input
    }

    /// Returns the exact canonical reference access mode.
    pub fn mode(&self) -> ReferenceAccessMode {
        self.mode
    }

    /// Returns the allocation number assigned on first observation during this invocation.
    pub fn root(&self) -> usize {
        self.root
    }

    /// Returns the canonical transform path from the allocation to the selected elements, rendered as the
    /// [`Display`] form of each [`ArrayReferenceTransform`] prefixed with `/` (e.g., `/slice(axes=[0:2])`). An empty
    /// path selects the complete allocation.
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl<Extension> KernelDefinition<Extension>
where
    EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    /// Interprets this definition on host arrays after checking initialization, output coverage, and races. The
    /// inputs and results follow the functional signature, so [`KernelParameterAccess::WriteOnly`] parameters consume
    /// no input and [`KernelParameterAccess::ReadOnly`] parameters produce no result. A failure never publishes a
    /// partially updated array or mutates an input. Qualification enumerates the grid only for a parameter whose
    /// mapping [`BlockMapping::tiling_axes`](crate::mappings::BlockMapping::tiling_axes) does not recognize or whose
    /// recognized axes repeat a grid axis. A recognized mapping can still require enumeration (e.g., the diagonal
    /// element starts `(i, i)`, for which `tiling_axes` returns `Some`). Grid points are not capped when no parameter
    /// requires enumeration, and replay has no step limit because kernel bodies admit only bounded loops. Use
    /// [`Self::interpret_with_limits`] to bound both.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order.
    ///   - `maximum_programs`: maximum grid size that qualification enumerates. Qualification rejects a larger grid
    ///     that it must enumerate before allocating reference storage or replaying a body, and it accepts a grid of any
    ///     size that it need not enumerate.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the inputs do not match the functional signature, when scalar prefetch
    /// specialization or qualification fails (e.g., with a [`KernelInitializationError`] for an uninitialized read, a
    /// dynamic grid extent, or a grid larger than `maximum_programs` that must be enumerated), or when replay fails.
    /// Replay fails when a canonical operation fails, when an access reads the padding of a masked window (i.e.,
    /// [`KernelMemoryError::UnmaskedWindowAccess`]), or when the asynchronous copy protocol is violated (i.e.,
    /// [`KernelInterpretationError::UnknownCopyToken`] or [`KernelInterpretationError::PendingCopies`]). Errors that
    /// this crate defines are wrapped with [`ProgramError::custom`] and recovered with
    /// [`ProgramError::downcast_custom`].
    pub fn interpret(&self, inputs: Vec<Array>, maximum_programs: usize) -> Result<Vec<Array>, ProgramError> {
        self.interpret_recorded(inputs, maximum_programs, None, None, None)
    }

    /// Interprets this definition with explicit host grid and replay limits. A replay step is an operation binding or
    /// region entry, so empty loop bodies still consume the budget. All grid points share one counter, and nested
    /// region replay preserves the original instruction provenance. Limit failures discard private output storage.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order.
    ///   - `maximum_programs`: maximum number of grid points allowed before qualification begins, even when
    ///     qualification would not enumerate them, and also the enumeration bound of that qualification.
    ///   - `maximum_steps`: maximum operation bindings and region entries across the invocation. Zero permits an
    ///     empty grid but rejects the first body entry of a nonempty launch.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::interpret`], a [`KernelInitializationError::QualificationLimit`] when the grid
    /// has more than `maximum_programs` points, and a [`KernelInterpretationError::StepLimit`] when replay exceeds
    /// `maximum_steps`.
    pub fn interpret_with_limits(
        &self,
        inputs: Vec<Array>,
        maximum_programs: usize,
        maximum_steps: usize,
    ) -> Result<Vec<Array>, ProgramError> {
        self.interpret_recorded(inputs, maximum_programs, Some(maximum_steps), None, None)
    }

    /// Interprets this definition with an operation trace retained on success or failure. Existing entries are cleared
    /// before qualification, so a qualification failure leaves an empty trace. The trace records attempted bindings,
    /// exact reference accesses, pending copy tokens, and original source provenance. No array contents are recorded.
    /// Host limits have the same meaning as in [`Self::interpret_with_limits`].
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order.
    ///   - `options`: host grid and replay limits, and whether NaN checks are enabled.
    ///   - `trace`: collector that receives a [`KernelTraceEntry`] for every attempted operation binding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::interpret_with_limits`] for the limits of `options`, and a
    /// [`KernelInterpretationError::Nan`] when NaN checks are enabled and an ordinary operation input or output
    /// contains a NaN.
    pub fn interpret_with_trace(
        &self,
        inputs: Vec<Array>,
        options: &KernelDebugOptions,
        trace: &mut Vec<KernelTraceEntry>,
    ) -> Result<Vec<Array>, ProgramError> {
        self.interpret_in_order(inputs, options, None, trace)
    }

    /// Shares ordinary replay with the bounded orders that the scheduler selects. Only the scheduler supplies an
    /// explicit order, together with the reference summary of the qualification that it performed once for every
    /// order, and it has already checked that the body is a flat region and generated every instruction occurrence
    /// exactly once. Each order still replays with fresh storage and runtime state.
    pub(super) fn interpret_in_order(
        &self,
        inputs: Vec<Array>,
        options: &KernelDebugOptions,
        order: Option<(&[usize], Rc<KernelReferenceSummary>)>,
        trace: &mut Vec<KernelTraceEntry>,
    ) -> Result<Vec<Array>, ProgramError> {
        trace.clear();
        let state = Rc::new(RefCell::new(KernelTraceState { nan_checks: options.nan_checks, ..Default::default() }));
        let result = self.interpret_recorded(
            inputs,
            options.maximum_programs,
            Some(options.maximum_steps),
            Some(state.clone()),
            order,
        );
        *trace = std::mem::take(&mut state.borrow_mut().entries);
        result
    }

    /// Runs the ordinary interpreter with optional debug limits and an optional trace collector local to the
    /// invocation.
    fn interpret_recorded(
        &self,
        inputs: Vec<Array>,
        maximum_programs: usize,
        maximum_steps: Option<usize>,
        trace: Option<Rc<RefCell<KernelTraceState>>>,
        order: Option<(&[usize], Rc<KernelReferenceSummary>)>,
    ) -> Result<Vec<Array>, ProgramError> {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation<Extension>>::new();
        let regions = AttachedRegions::from(vec![self.body().clone()]);
        let driver = EagerInterpretationDriver::new(&regions);
        self.operation()
            .interpret_with_budget(
                &context,
                &driver,
                &inputs.into_iter().map(ArrayIrValue::Array).collect::<Vec<_>>(),
                maximum_programs,
                maximum_steps,
                trace,
                order,
            )?
            .into_iter()
            .map(|value| match value {
                ArrayIrValue::Array(array) => Ok(array),
                _ => Err(ProgramError::MalformedProgram("kernel interpreter returned a non-array result".to_owned())),
            })
            .collect()
    }
}

impl<Extension> InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>
    for KernelCallOperation
where
    EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>(
        &self,
        context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.interpret_with_budget(
            context,
            driver,
            inputs,
            DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
            None,
            None,
            None,
        )
    }
}

impl KernelCallOperation {
    /// Qualifies and executes an attached body without relying on a previously constructed definition's summary.
    /// `maximum_programs` bounds grid enumeration in [`validated_kernel_references`]. The debug entry points supply
    /// `maximum_steps`, which also caps the number of grid points before qualification, while ordinary interpretation
    /// passes [`None`] and replays without either limit.
    fn interpret_with_budget<Extension, D>(
        &self,
        context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        driver: &D,
        inputs: &[ArrayIrValue<Array>],
        maximum_programs: usize,
        maximum_steps: Option<usize>,
        trace: Option<Rc<RefCell<KernelTraceState>>>,
        order: Option<(&[usize], Rc<KernelReferenceSummary>)>,
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError>
    where
        EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
                Type = ArrayIrType,
                Value = ArrayIrValue<Array>,
                Constant = ArrayIrValue<Array>,
                Operation = KernelOperation<Extension>,
            >,
        Extension:
            KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
        D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
    {
        let body = driver.region(0)?;
        let output_types = self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            std::slice::from_ref(body.interface()),
        )?;
        if !self.prefetch_types().is_empty() {
            let ordinary_count = inputs.len() - self.prefetch_types().len();
            let prefetched = inputs[ordinary_count..]
                .iter()
                .zip(self.prefetch_types())
                .map(|(value, r#type)| {
                    let ArrayIrValue::Array(array) = value else {
                        unreachable!("the validated prefetch signature contains only arrays");
                    };
                    Self::with_sharding_of(array.clone(), r#type)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let definition = KernelDefinition::new(self.clone(), body.to_program()).map_err(ProgramError::custom)?;
            let specialized = definition.specialize_prefetch(&prefetched).map_err(ProgramError::custom)?;
            let regions = AttachedRegions::from(vec![specialized.body().clone()]);
            let driver = EagerInterpretationDriver::new(&regions);
            return specialized.operation().interpret_with_budget(
                context,
                &driver,
                &inputs[..ordinary_count],
                maximum_programs,
                maximum_steps,
                trace,
                order,
            );
        }
        let extents = self
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.extent().value().ok_or_else(|| {
                    ProgramError::custom(KernelInitializationError::UnsupportedLaunch { boundary: "grid extents" })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let points = self.grid().points(&extents).map_err(ProgramError::custom)?;
        if maximum_steps.is_some() && points.len() > maximum_programs {
            return Err(ProgramError::custom(KernelInitializationError::QualificationLimit {
                programs: points.len(),
                maximum: maximum_programs,
            }));
        }
        let (order, references) = match order {
            Some((order, references)) => (Some(order), references),
            None => {
                let references =
                    validated_kernel_references(body, self, maximum_programs).map_err(ProgramError::custom)?;
                (None, Rc::new(references))
            }
        };
        // Kernel bodies admit only bounded loops, so ordinary interpretation needs no step limit and replays with a
        // budget that it cannot exhaust.
        let maximum_steps = maximum_steps.unwrap_or(usize::MAX);

        let mut input_values = inputs.iter();
        let roots = self
            .parameters()
            .iter()
            .map(|parameter| {
                let array = match parameter.access() {
                    KernelParameterAccess::ReadOnly | KernelParameterAccess::ReadWrite => {
                        let ArrayIrValue::Array(array) = input_values.next().unwrap() else {
                            unreachable!("the validated functional signature contains only arrays");
                        };
                        array.clone()
                    }
                    KernelParameterAccess::WriteOnly => Array::uninitialized(parameter.r#type().into_owned())?,
                };
                // The body sees the physical parameter type. A placed input differs from it only in its sharding and
                // therefore shares its storage.
                Ok(ArrayReference::new(Self::with_sharding_of(array, parameter.r#type().as_ref())?))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let remaining_steps = Rc::new(Cell::new(maximum_steps));
        let provenance = Rc::new(ProvenanceState::new());
        let mut scheduled = Vec::new();
        for coordinate in points {
            let mut validity = BTreeMap::new();
            let mut publications = Vec::new();
            let mut windows = self
                .parameters()
                .iter()
                .zip(&roots)
                .map(|(parameter, root)| {
                    let mapping_inputs = parameter
                        .mapping()
                        .program()
                        .input_types()
                        .iter()
                        .zip(&coordinate)
                        .map(|(r#type, &index)| {
                            let ArrayIrType::Dimension(r#type) = r#type else {
                                unreachable!("validated mappings have only dimension inputs");
                            };
                            DimensionValue::new(r#type.clone(), index).map_err(ProgramError::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let shape = parameter.r#type().static_shape().unwrap();
                    let window = parameter
                        .mapping()
                        .evaluate(&mapping_inputs, shape.dimensions())
                        .map_err(ProgramError::custom)?;
                    let source = root.with_transform(window.valid_transform().clone())?;
                    let reference = if parameter.mapping().boundary_policy() == BoundaryPolicy::Masked {
                        let ArrayIrType::Reference(body_type) = parameter.body_type() else { unreachable!() };
                        let tile_type = body_type.referent();
                        let tile = ArrayReference::new(Array::uninitialized(tile_type.clone())?);
                        let valid_shape = source.r#type().referent().static_shape().unwrap();
                        let selected = tile.with_transform(ArrayReferenceTransform::Slice {
                            axes: valid_shape
                                .dimensions()
                                .iter()
                                .map(|&extent| ArraySliceAxis::new(0, extent, 1))
                                .collect(),
                        })?;
                        // A carried write-only window holds what the previous step along its carry axis published,
                        // so every tile loads its valid elements. The root of a write-only parameter starts with the
                        // zero bytes of uninitialized storage, so the load is deterministic, and qualification admits
                        // no read of any other write-only tile before the body writes the elements that it reads.
                        Self::copy_window(&source, &selected)?;
                        let mask_type = tile_type.clone().with_data_type(DataType::Boolean);
                        let addressing = ArrayAddressing::new(mask_type.clone())?;
                        let mut index = vec![0; tile_type.rank()];
                        let mut elements = Vec::with_capacity(addressing.element_count());
                        for _ in 0..addressing.element_count() {
                            elements.push(
                                index.iter().zip(valid_shape.dimensions()).all(|(&index, &extent)| index < extent),
                            );
                            addressing.advance_index(&mut index);
                        }
                        validity.insert(tile.id(), ArrayReference::new(Array::from_elements(mask_type, &elements)?));
                        if parameter.access() != KernelParameterAccess::ReadOnly {
                            publications.push((selected, source));
                        }
                        tile
                    } else {
                        source
                    };
                    let value = ArrayIrValue::Reference(reference);
                    if value.r#type().as_ref() != &parameter.body_type() {
                        return Err(ProgramError::MalformedProgram(
                            "kernel window type differs from its declared canonical body type".to_owned(),
                        ));
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            windows.extend(
                self.coordinate_types()
                    .iter()
                    .zip(&coordinate)
                    .map(|(r#type, &index)| DimensionValue::new(r#type.clone(), index).map(ArrayIrValue::Dimension))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let qualified = QualifiedKernelContext {
                eager: *context,
                validity,
                references: references.clone(),
                instruction: Rc::new(Cell::new(None)),
                coordinate,
                maximum_steps,
                remaining_steps: remaining_steps.clone(),
                provenance: provenance.clone(),
                trace: trace.clone(),
                pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
            };
            if order.is_some() {
                scheduled.push(Some(RegionInterpreter::new(qualified, body, windows)?));
                continue;
            }
            qualified.validate_step(self.name())?;
            qualified.replay_region(body, windows)?;
            let pending = qualified.pending_copies.borrow().len();
            if pending != 0 {
                return Err(ProgramError::custom(KernelInterpretationError::PendingCopies {
                    count: pending,
                    coordinate: qualified.coordinate.clone(),
                }));
            }
            for (source, destination) in publications {
                Self::copy_window(&source, &destination)?;
            }
        }
        if let Some(order) = order {
            let mut started = vec![false; scheduled.len()];
            let mut positions = vec![0; scheduled.len()];
            for &program in order {
                let state = scheduled[program].take().unwrap();
                if !started[program] {
                    state.context().validate_step(self.name())?;
                    started[program] = true;
                }
                state.context().instruction.set(Some(InstructionId::new(body.id(), positions[program])));
                positions[program] += 1;
                scheduled[program] = Some(state.step()?);
            }
            for (program, state) in scheduled.into_iter().enumerate() {
                let state = state.unwrap();
                if !started[program] {
                    state.context().validate_step(self.name())?;
                }
                let pending = state.context().pending_copies.borrow().len();
                if pending != 0 {
                    return Err(ProgramError::custom(KernelInterpretationError::PendingCopies {
                        count: pending,
                        coordinate: state.context().coordinate.clone(),
                    }));
                }
                state.finish()?;
            }
        }
        self.parameters()
            .iter()
            .zip(roots)
            .filter(|(parameter, _)| parameter.access() != KernelParameterAccess::ReadOnly)
            .zip(&output_types)
            .map(|((_, root), r#type)| {
                let ArrayIrType::Array(r#type) = r#type else { unreachable!("kernel results are arrays") };
                Self::with_sharding_of(root.freeze()?, r#type).map(ArrayIrValue::Array)
            })
            .collect()
    }

    /// Returns `array` with the sharding of `type` over the same storage, which moves no data, because a host array
    /// stores the same bytes under every placement.
    fn with_sharding_of(array: Array, r#type: &ArrayType) -> Result<Array, ProgramError> {
        if array.r#type().sharding() == r#type.sharding() {
            return Ok(array);
        }
        let r#type = array
            .r#type()
            .into_owned()
            .with_sharding(r#type.sharding().cloned())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        Ok(Array::new_unchecked(r#type, array.shared_storage_bytes().clone()))
    }

    /// Copies only valid selected coordinates between a logical edge tile and the original full array. Physical layout
    /// may differ because the tile is private, but shape, data type, placement, and distributed dependencies must
    /// agree.
    fn copy_window(source: &ArrayReference<Array>, destination: &ArrayReference<Array>) -> Result<(), ProgramError> {
        let value = source.read()?;
        let destination_type = destination.r#type().referent().clone();
        if value.r#type().into_owned().with_layout(None) != destination_type.clone().with_layout(None) {
            return Err(
                TypeError::invalid("kernel edge tile and full array window have incompatible logical types").into()
            );
        }
        destination.write(Array::from_logical_bytes(destination_type, &value.logical_bytes())?)
    }
}

/// Mutable collection shared only by nested host replay contexts. It never enters semantic identity or compiled IR.
#[derive(Default)]
struct KernelTraceState {
    /// Attempted bindings collected in execution order.
    entries: Vec<KernelTraceEntry>,

    /// Stable local names for observed canonical allocation identities.
    roots: BTreeMap<ReferenceId, usize>,

    /// Optional numerical checks applied to bound ordinary values.
    nan_checks: bool,
}

/// Private replay authority created only after the entire attached body passes initialization qualification.
/// Portable eager rules remain unchanged, and their child region driver (i.e., [`QualifiedKernelDriver`]) retains
/// this authority across recursion.
#[derive(Clone)]
struct QualifiedKernelContext<Extension: Operation<Type = ArrayIrType>> {
    /// Canonical eager arithmetic and reference semantics.
    eager: EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,

    /// Valid coordinates of each tile. Private physical padding never counts as readable full array storage.
    validity: BTreeMap<ReferenceId, ArrayReference<Array>>,

    /// Validated summary of the attached body, which classifies every swap as a store or an exchange.
    references: Rc<KernelReferenceSummary>,

    /// Body instruction whose replay step is in progress. The first swap bound during that step takes it to look up
    /// the validated lowering, so a swap that a rule binds on its own behalf never inherits a store lowering.
    instruction: Rc<Cell<Option<InstructionId>>>,

    /// Logical program being replayed for diagnostics.
    coordinate: Vec<usize>,

    /// Host step limit of the whole invocation, retained for exact diagnostics.
    maximum_steps: usize,

    /// Remaining replay steps, shared across every grid point and nested region of this invocation.
    remaining_steps: Rc<Cell<usize>>,

    /// Canonical provenance composition during ordinary program replay, which restores the enclosing provenance even
    /// when a step fails.
    provenance: Rc<ProvenanceState>,

    /// Source snapshots and reserved destinations awaiting their canonical completion tokens.
    pending_copies: Rc<RefCell<BTreeMap<ReferenceId, PendingCopy>>>,

    /// Optional observer shared across all points and attached regions.
    trace: Option<Rc<RefCell<KernelTraceState>>>,
}

impl<Extension: Operation<Type = ArrayIrType>> QualifiedKernelContext<Extension> {
    /// Charges one bounded replay step before any instruction or region entry can perform effects, and fails with a
    /// [`KernelInterpretationError::StepLimit`] when the invocation has no steps left.
    fn validate_step(&self, operation: &'static str) -> Result<(), ProgramError> {
        let remaining = self.remaining_steps.get();
        if remaining == 0 {
            return Err(ProgramError::custom(KernelInterpretationError::StepLimit {
                maximum_steps: self.maximum_steps,
                coordinate: self.coordinate.clone(),
                operation,
                provenance: self.provenance.current(),
            }));
        }
        self.remaining_steps.set(remaining - 1);
        Ok(())
    }

    /// Validates that ordinary `values` contain no NaN when NaN checks are enabled. Only ordinary values are
    /// inspected, so reference initialization and reads retain their explicit operation semantics.
    fn validate_nans(
        &self,
        operation: &'static str,
        position: &'static str,
        values: &[ArrayIrValue<Array>],
    ) -> Result<(), ProgramError> {
        if !self.trace.as_ref().is_some_and(|trace| trace.borrow().nan_checks) {
            return Ok(());
        }
        for (value, array) in values.iter().enumerate() {
            let ArrayIrValue::Array(array) = array else {
                continue;
            };
            let data_type = array.r#type().data_type();
            if !data_type.is_floating_point() && !data_type.is_complex() {
                continue;
            }
            // Diagnostic widening must not inherit byte strides sized for a narrower element representation.
            let array =
                Array::from_logical_bytes(array.r#type().into_owned().with_layout(None), &array.logical_bytes())?;
            let element = if data_type.is_floating_point() {
                array.converted_to(DataType::F64)?.elements::<f64>()?.iter().position(|value| value.is_nan())
            } else if data_type.is_complex() {
                array
                    .converted_to(DataType::C128)?
                    .elements::<Complex<f64>>()?
                    .iter()
                    .position(|value| value.re.is_nan() || value.im.is_nan())
            } else {
                None
            };
            if let Some(element) = element {
                return Err(ProgramError::custom(KernelInterpretationError::Nan {
                    operation,
                    position,
                    value,
                    element,
                    data_type,
                    coordinate: self.coordinate.clone(),
                    provenance: self.provenance.current(),
                }));
            }
        }
        Ok(())
    }

    /// Projects a tile's validity through the same canonical transforms as the accessed reference. This includes
    /// indexed axes and nested slices. The mask has the same shape but owns independent immutable Boolean storage.
    fn window_validity(&self, reference: &ArrayReference<Array>) -> Result<Option<Array>, ProgramError> {
        let Some(validity) = self.validity.get(&reference.id()) else {
            return Ok(None);
        };
        let mut selected = validity.clone();
        for transform in reference.path().transforms() {
            selected = selected.with_transform(transform.clone())?;
        }
        selected.read().map(Some)
    }

    /// Applies a commit (when `commit` is set) or a wait of `operation` that follows `protocol` (i.e., [`None`] for the
    /// portable wait) to the pending completion of `token`, after checking that it follows the protocol of the
    /// producer of the token. A commit only records that it observed the token, and a wait removes and returns the
    /// pending completion, so that the caller can publish its deferred writes.
    fn complete(
        &self,
        operation: &'static str,
        token: &ArrayReference<Array>,
        protocol: Option<KernelCompletionProtocol>,
        commit: bool,
    ) -> Result<Option<PendingCopy>, ProgramError> {
        let mut pending_copies = self.pending_copies.borrow_mut();
        let pending = pending_copies
            .get_mut(&token.id())
            .ok_or_else(|| ProgramError::custom(KernelInterpretationError::UnknownCopyToken { token: token.id() }))?;
        if !admits_completion(pending.protocol, pending.committed, protocol, commit) {
            return Err(ProgramError::custom(KernelInterpretationError::IncompatibleCompletion {
                operation,
                token: token.id(),
            }));
        }
        if commit {
            pending.committed = true;
            return Ok(None);
        }
        Ok(pending_copies.remove(&token.id()))
    }

    /// Replays `region` one instruction at a time, recording each instruction so that [`Context::bind`] can apply
    /// its validated swap lowering. The enclosing instruction is restored afterwards, because a nested region runs
    /// within the step of the instruction that attaches it.
    fn replay_region(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        inputs: Vec<ArrayIrValue<Array>>,
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError>
    where
        Self: Context<
                Type = ArrayIrType,
                Value = ArrayIrValue<Array>,
                Constant = ArrayIrValue<Array>,
                Operation = KernelOperation<Extension>,
            >,
    {
        let enclosing = self.instruction.get();
        let mut interpreter = RegionInterpreter::new(self.clone(), region, inputs)?;
        for index in 0..region.instructions().len() {
            self.instruction.set(Some(InstructionId::new(region.id(), index)));
            interpreter = interpreter.step()?;
        }
        self.instruction.set(enclosing);
        interpreter.finish()
    }
}

impl<Extension: Operation<Type = ArrayIrType>> Domain for QualifiedKernelContext<Extension> {
    type Type = ArrayIrType;
    type Value = ArrayIrValue<Array>;
    type Constant = ArrayIrValue<Array>;
    type Operation = KernelOperation<Extension>;
}

impl<Extension> Context for QualifiedKernelContext<Extension>
where
    Self: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }

    fn bind<'r, O: Into<Self::Operation>, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
        &self,
        operation: O,
        regions: R,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>
    where
        Self::Constant: 'r,
        Self::Operation: 'r,
    {
        let operation = operation.into();
        let regions = regions.into();
        // The first swap bound during the replay step of a body instruction takes that instruction to look up its
        // validated lowering, so a swap that a rule binds on its own behalf never inherits a store lowering.
        let store = operation.swap_output_index().is_some()
            && self.instruction.take().and_then(|instruction| self.references.swap_lowering(instruction))
                == Some(KernelSwapLowering::Store);
        // Qualification and diagnostics inspect the same selection that ordinary access interpretation will apply.
        // Keep the original inputs for that interpretation so paths are never applied twice. Resolving a selection
        // can fail (e.g., for a dynamic index into an empty axis, or for a binding that a rule attempts with a
        // malformed reference access layout), and that failure is reported only after the attempted binding is traced
        // with the selections resolved before it and charged one replay step.
        let mut selected_references = BTreeMap::new();
        let (descriptors, mut selection) = match validated_reference_access_descriptors(&operation, inputs.len()) {
            Ok(descriptors) => (descriptors, Ok(())),
            Err(error) => (Vec::new(), Err(error)),
        };
        for (input_index, descriptor) in descriptors.iter().enumerate() {
            let (Some(descriptor), Some(ArrayIrValue::Reference(reference))) = (descriptor, inputs.get(input_index))
            else {
                continue;
            };
            match reference.with_transforms(descriptor.transforms(), &inputs[descriptor.bindings()]) {
                Ok(reference) => {
                    selected_references.insert(input_index, reference);
                }
                Err(error) => {
                    selection = Err(error);
                    break;
                }
            }
        }
        if let Some(trace) = &self.trace {
            let mut trace = trace.borrow_mut();
            let mut accesses = Vec::new();
            for (input, mode) in operation.effects().reference_accesses() {
                if let Some(reference) = selected_references.get(&input) {
                    let next = trace.roots.len();
                    let root = *trace.roots.entry(reference.id()).or_insert(next);
                    let path = reference.path().transforms().map(|transform| format!("/{transform}")).collect();
                    let mode = if store { ReferenceAccessMode::Write } else { mode };
                    accesses.push(KernelTraceAccess { input, mode, root, path });
                }
            }
            let mut pending_copies = Vec::new();
            for token in self.pending_copies.borrow().keys() {
                let next = trace.roots.len();
                pending_copies.push(*trace.roots.entry(*token).or_insert(next));
            }
            pending_copies.sort_unstable();
            trace.entries.push(KernelTraceEntry {
                coordinate: self.coordinate.clone(),
                operation: operation.name(),
                provenance: self.provenance.current(),
                accesses,
                pending_copies,
            });
        }
        self.validate_step(operation.name())?;
        selection?;
        self.validate_nans(operation.name(), "input", inputs)?;
        let execute = |operation: &KernelOperation<Extension>, inputs: &[ArrayIrValue<Array>]| {
            operation.validate_region_interfaces(&regions.region_interfaces())?;
            if let KernelOperation::Scratch(scratch) = operation {
                scratch.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &[],
                )?;
                let storage = Array::uninitialized(scratch.referent().clone())?;
                return Ok(vec![ArrayIrValue::Reference(ArrayReference::new(storage))]);
            }
            if let KernelOperation::WindowMask(window_mask) = operation {
                // The mask is the recorded validity of the tile that the reference views, restricted to that view,
                // and every element of a reference without recorded validity is valid. No storage is accessed.
                let output_types = window_mask.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &[],
                )?;
                let (ArrayIrValue::Reference(reference), [ArrayIrType::Array(output_type)]) =
                    (&inputs[0], output_types.as_slice())
                else {
                    unreachable!("`{WINDOW_MASK_OPERATION_NAME}` type inference admits only array references");
                };
                let mask = match self.window_validity(reference)? {
                    Some(validity) => Array::from_logical_bytes(output_type.clone(), &validity.logical_bytes())?,
                    None => {
                        let count = ArrayAddressing::new(output_type.clone())?.element_count();
                        Array::from_elements(output_type.clone(), &vec![true; count])?
                    }
                };
                return Ok(vec![ArrayIrValue::Array(mask)]);
            }
            let mask_index = match operation {
                KernelOperation::MaskedLoad(_) => Some(1),
                KernelOperation::MaskedStore(_) | KernelOperation::MaskedSwap(_) => Some(2),
                _ => None,
            };
            if let Some(mask_index) = mask_index {
                operation.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &regions.region_interfaces(),
                )?;
                let reference = selected_references.get(&0).unwrap();
                if let Some(validity) = self.window_validity(reference)? {
                    let ArrayIrValue::Array(mask) = &inputs[mask_index] else { unreachable!() };
                    let elements = mask
                        .elements::<bool>()?
                        .into_iter()
                        .zip(validity.elements::<bool>()?)
                        .map(|(requested, valid)| requested && valid)
                        .collect::<Vec<_>>();
                    let mut inputs = inputs.to_vec();
                    inputs[mask_index] =
                        ArrayIrValue::Array(Array::from_elements(mask.r#type().into_owned(), &elements)?);
                    return operation.interpret(
                        &self.eager,
                        &QualifiedKernelDriver { context: self, regions: &regions, operation: operation.name() },
                        &inputs,
                    );
                }
            } else {
                // Writes may fill private padding because only valid coordinates are published. An unmasked read or
                // read-modify-write must never expose physical padding as initialized full array data.
                for (input_index, mode) in operation.effects().reference_accesses() {
                    if mode != ReferenceAccessMode::Write
                        && let Some(reference) = selected_references.get(&input_index)
                        && let Some(validity) = self.window_validity(reference)?
                        && validity.elements::<bool>()?.iter().any(|valid| !valid)
                    {
                        return Err(ProgramError::custom(KernelMemoryError::UnmaskedWindowAccess {
                            operation: operation.name(),
                        }));
                    }
                }
            }
            match operation {
                KernelOperation::AsyncCopy(_) => {
                    operation.infer_output_types(
                        &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        &[],
                    )?;
                    let source = selected_references.get(&0).unwrap();
                    let destination = selected_references.get(&1).unwrap();
                    let source = source.read()?;
                    let token = ArrayReference::new(Array::new(ArrayType::scalar(DataType::Token), vec![])?);
                    self.pending_copies.borrow_mut().insert(
                        token.id(),
                        PendingCopy { protocol: None, committed: false, copy: Some((source, destination.clone())) },
                    );
                    Ok(vec![ArrayIrValue::Reference(token)])
                }
                KernelOperation::Wait(_) => {
                    operation.infer_output_types(
                        &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        &[],
                    )?;
                    let ArrayIrValue::Reference(token) = &inputs[0] else { unreachable!() };
                    let Some(PendingCopy { copy: Some((source, destination)), .. }) =
                        self.complete(operation.name(), token, None, false)?
                    else {
                        unreachable!("a portable wait completes only a portable copy");
                    };
                    let value =
                        Array::from_logical_bytes(destination.r#type().referent().clone(), &source.logical_bytes())?;
                    destination.write(value)?;
                    token.freeze()?;
                    Ok(vec![])
                }
                KernelOperation::Extension(extension) => {
                    let driver =
                        QualifiedKernelDriver { context: self, regions: &regions, operation: operation.name() };
                    let completion = |input_index: usize| match inputs.get(input_index) {
                        Some(ArrayIrValue::Reference(token)) => Ok(token),
                        _ => Err(ProgramError::MalformedProgram(format!(
                            "`{}` has no completion reference input {input_index}",
                            operation.name(),
                        ))),
                    };
                    match extension.memory_semantics()? {
                        KernelExtensionMemory::Asynchronous { completion_output_index } => {
                            let outputs = operation.interpret(&self.eager, &driver, inputs)?;
                            let Some(ArrayIrValue::Reference(token)) = outputs.get(completion_output_index) else {
                                return Err(ProgramError::MalformedProgram(format!(
                                    "`{}` has no completion reference output {completion_output_index}",
                                    operation.name(),
                                )));
                            };
                            self.pending_copies.borrow_mut().insert(
                                token.id(),
                                PendingCopy {
                                    protocol: Some(extension.completion_protocol()),
                                    committed: false,
                                    copy: None,
                                },
                            );
                            Ok(outputs)
                        }
                        KernelExtensionMemory::Commit { completion_input_index } => {
                            let token = completion(completion_input_index)?;
                            self.complete(operation.name(), token, Some(extension.completion_protocol()), true)?;
                            operation.interpret(&self.eager, &driver, inputs)
                        }
                        KernelExtensionMemory::Wait { completion_input_index } => {
                            let token = completion(completion_input_index)?;
                            self.complete(operation.name(), token, Some(extension.completion_protocol()), false)?;
                            operation.interpret(&self.eager, &driver, inputs)
                        }
                        _ => operation.interpret(&self.eager, &driver, inputs),
                    }
                }
                _ => operation.interpret(
                    &self.eager,
                    &QualifiedKernelDriver { context: self, regions: &regions, operation: operation.name() },
                    inputs,
                ),
            }
        };
        if !store {
            let outputs = execute(&operation, inputs)?;
            self.validate_nans(operation.name(), "output", &outputs)?;
            return Ok(outputs);
        }

        // A swap whose old value is never used executes as the equivalent store, so it never reads the reference and
        // its old value is never checked for NaNs. Its unused result is uninitialized storage of the old value type.
        let output_types = operation.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &regions.region_interfaces(),
        )?;
        match &operation {
            KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(swap)) => {
                let write = ReferenceWriteOperation::new().with_transforms(swap.transforms().to_vec());
                execute(&write.into(), inputs)?;
            }
            KernelOperation::MaskedSwap(swap) => {
                // The store takes the inputs of the swap without its `other` input.
                let store = MaskedStoreOperation::new().with_transforms(swap.transforms().to_vec());
                let inputs = inputs[..3].iter().chain(&inputs[4..]).cloned().collect::<Vec<_>>();
                execute(&store.into(), &inputs)?;
            }
            _ => unreachable!("only swaps have a validated store lowering"),
        }
        output_types
            .into_iter()
            .map(|r#type| {
                let ArrayIrType::Array(r#type) = r#type else {
                    unreachable!("swap type inference returns the array type of the old value");
                };
                Array::uninitialized(r#type).map(ArrayIrValue::Array)
            })
            .collect()
    }

    fn is_eager(&self) -> bool {
        true
    }

    fn provenance(&self) -> Provenance {
        self.provenance.current()
    }

    fn resolve(&self, value: &Self::Value) -> ValueResolution<Self::Constant> {
        ValueResolution::Constant(value.clone())
    }

    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.provenance.invoke_with_origin(origin, function)
    }

    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.provenance.invoke_with_scope(scope, function)
    }
}

/// One pending completion token, whose portable copy leaves its destination unchanged until the matching wait succeeds.
struct PendingCopy {
    /// Completion protocol of the producer, which is [`None`] for a portable async copy.
    protocol: Option<KernelCompletionProtocol>,

    /// Whether a commit has observed the token.
    committed: bool,

    /// Initialized source elements that a portable copy captured when it started, and its canonical destination view,
    /// which stays reserved until completion. The interpretation rules of an extension own the data movement of its
    /// asynchronous operations, so the token of an extension has no copy.
    copy: Option<(Array, ArrayReference<Array>)>,
}

/// Adapts unchanged eager higher-order rules to recursive replay in the private qualified context.
struct QualifiedKernelDriver<'r, Extension: Operation<Type = ArrayIrType>> {
    /// Qualification authority for this immutable invocation body.
    context: &'r QualifiedKernelContext<Extension>,

    /// Actual regions attached to the operation currently being interpreted.
    regions: &'r AttachedRegions<'r, ArrayIrValue<Array>, KernelOperation<Extension>>,

    /// Canonical operation whose region entry consumes a replay step.
    operation: &'static str,
}

impl<Extension: Operation<Type = ArrayIrType>> RegionDriver<ArrayIrValue<Array>, KernelOperation<Extension>>
    for QualifiedKernelDriver<'_, Extension>
{
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, ArrayIrValue<Array>, KernelOperation<Extension>>>
    where
        Extension: 'r,
    {
        self.regions.regions()
    }

    fn region_count(&self) -> usize {
        self.regions.region_count()
    }

    fn region(
        &self,
        index: usize,
    ) -> Result<RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>, ProgramError> {
        self.regions.region(index)
    }
}

impl<Extension> InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>
    for QualifiedKernelDriver<'_, Extension>
where
    EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    fn interpret_region(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        index: usize,
        inputs: Vec<ArrayIrValue<Array>>,
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.context.validate_step(self.operation)?;
        self.context.replay_region(self.region(index)?, inputs)
    }

    fn bind(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        operation: KernelOperation<Extension>,
        regions: AttachedRegions<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        // Operations that a rule binds on its own behalf are qualified exactly like the body's instructions, so they
        // consume replay steps and undergo the same reference and NaN checks.
        self.context.bind(operation, regions, inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrOperation, ArrayOperation, ArrayReferenceTransformIndex, CompareOperation,
        ComparisonDirection, ConditionOperation, Dimension, DimensionAddOperation, DimensionBounds,
        DimensionMulOperation, DimensionToScalarOperation, DimensionType, DotOperation, Layout, Memory, NotOperation,
        Placeholder, ProgramBuilder, ReduceOperation, ReductionKind, ReferenceAccessDescriptor,
        ReferenceAccessOperation, ReferenceAddUpdateOperation, ReferenceRead, ReferenceReadOperation, ReferenceRoot,
        ReferenceSwapOperation, ReferenceWrite, ReferenceWriteOperation, RegionInterface, ScaledDotOperation,
        StridedLayout, TracingContext, WhileOperation,
    };

    use crate::authoring::{MappingExpression, ParameterWindow, mapped_call, whole_array_parameter};
    use crate::calls::KernelParameter;
    use crate::grids::{Grid, GridDimension, GridExecution};
    use crate::mappings::BlockMapping;
    use crate::memory::{
        AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation,
        WaitOperation, WindowMaskOperation,
    };
    use crate::operations::NoKernelExtension;
    use crate::operations::tests::TestExtension;

    use super::*;

    /// Builds a copy or addition definition with blocks of one element over a scalar or vector.
    fn definition(vector: bool, addition: bool) -> KernelDefinition {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let outputs = if vector {
            vec![mapping.add_input(ArrayIrType::Dimension(DimensionType::new(
                "coordinate",
                DimensionBounds::non_negative(None).unwrap(),
            )))]
        } else {
            vec![]
        };
        let mapping = BlockMapping::new(
            mapping
                .build(outputs, vec![Placeholder; usize::from(vector)], vec![Placeholder; usize::from(vector)])
                .unwrap(),
            if vector { vec![1] } else { vec![] },
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let r#type = if vector { ArrayType::new_static(DataType::I32, [4]) } else { ArrayType::scalar(DataType::I32) };
        let mut parameters =
            vec![KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap()];
        if addition {
            parameters
                .push(KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap());
        }
        parameters.push(KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap());
        let operation = KernelCallOperation::new(
            Grid::new(if vector {
                vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]
            } else {
                vec![]
            })
            .unwrap(),
            parameters,
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _coordinates)| {
            let mut value = references[0].read()?;
            if addition {
                let second = references[1].read()?;
                value = references[0]
                    .context()
                    .bind(ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())), vec![], &[value, second])?
                    .remove(0);
            }
            references.last().unwrap().write(&value)
        })
        .unwrap()
    }

    /// Builds ordinary dimension multiplication for independent row-major tile coordinates.
    fn tiled_mapping(block_shape: Vec<usize>) -> BlockMapping {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mut starts = Vec::new();
        for &block in &block_shape {
            let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
            let coordinate = builder.add_input(coordinate_type.clone().into());
            let size = DimensionValue::constant(block).unwrap();
            let size_id = builder.add_constant(ArrayIrValue::Dimension(size.clone()));
            starts.push(
                builder
                    .add_instruction(
                        DimensionMulOperation::new(&coordinate_type, size.r#type().as_ref()).unwrap(),
                        vec![],
                        vec![coordinate, size_id],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let rank = block_shape.len();
        BlockMapping::new(
            builder.build(starts, vec![Placeholder; rank], vec![Placeholder; rank]).unwrap(),
            block_shape,
            BoundaryPolicy::Masked,
        )
        .unwrap()
    }

    /// Wraps canonical value operations with qualified kernel input/output references for conformance tests.
    fn value_definition(operation: ArrayOperation<Array>, input_types: &[ArrayType]) -> KernelDefinition {
        let output_types = operation.infer_output_types(input_types, &[]).unwrap();
        let mut parameters = input_types
            .iter()
            .map(|r#type| whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap())
            .collect::<Vec<_>>();
        parameters.extend(
            output_types
                .into_iter()
                .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap()),
        );
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        KernelDefinition::trace(call, |(references, _coordinates)| {
            let inputs = references[..input_types.len()]
                .iter()
                .map(|reference| reference.read())
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = references[0].context().bind(ArrayIrOperation::Array(operation), vec![], &inputs)?;
            for (reference, output) in references[input_types.len()..].iter().zip(outputs) {
                reference.write(&output)?;
            }
            Ok(())
        })
        .unwrap()
    }

    /// Builds a matrix product of `f32[m, k]` and `f32[k, n]` inputs with square tiles of size `block` over the grid
    /// `(i, j, k)`, where only `k` is sequential. The body zeroes its write-only output tile at `k == 0` and then
    /// accumulates one tile product into it at every `k`. Masked tiles use masked loads and stores throughout.
    fn carried_matmul_definition(shape: [usize; 3], block: usize, policy: BoundaryPolicy) -> KernelDefinition {
        let [m, k, n] = shape;
        let mapping = |axes: [usize; 2]| {
            let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let coordinate_types = (0..3)
                .map(|axis| {
                    DimensionType::new(format!("coordinate_{axis}"), DimensionBounds::non_negative(None).unwrap())
                })
                .collect::<Vec<_>>();
            let coordinates =
                coordinate_types.iter().map(|r#type| mapping.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let size = DimensionValue::constant(block).unwrap();
            let size_id = mapping.add_constant(ArrayIrValue::Dimension(size.clone()));
            let starts = axes
                .iter()
                .map(|&axis| {
                    mapping
                        .add_instruction(
                            DimensionMulOperation::new(&coordinate_types[axis], size.r#type().as_ref()).unwrap(),
                            vec![],
                            vec![coordinates[axis], size_id],
                            None,
                        )
                        .unwrap()[0]
                })
                .collect::<Vec<_>>();
            BlockMapping::new(
                mapping.build(starts, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap(),
                vec![block, block],
                policy,
            )
            .unwrap()
        };
        let operation = KernelCallOperation::new(
            Grid::new(vec![
                GridDimension::new(Dimension::Static(m.div_ceil(block)), GridExecution::Parallel),
                GridDimension::new(Dimension::Static(n.div_ceil(block)), GridExecution::Parallel),
                GridDimension::new(Dimension::Static(k.div_ceil(block)), GridExecution::Sequential),
            ])
            .unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::F32, [m, k]),
                    KernelParameterAccess::ReadOnly,
                    mapping([0, 2]),
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::F32, [k, n]),
                    KernelParameterAccess::ReadOnly,
                    mapping([2, 1]),
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::F32, [m, n]),
                    KernelParameterAccess::WriteOnly,
                    mapping([0, 1]),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let zeros = Array::matrix(block, block, vec![0f32; block * block]).unwrap();
        let branches = [true, false].map(|write| {
            let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = branch.add_input(operation.parameters()[2].body_type());
            if write {
                let zeros = branch.add_constant(ArrayIrValue::Array(zeros.clone()));
                branch
                    .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, zeros], None)
                    .unwrap();
            }
            branch
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap()
        });
        KernelDefinition::trace(operation, |(references, coordinates)| {
            let context = references[0].context();
            let zero = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
            let first = context
                .bind(
                    ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::Equal)),
                    vec![],
                    &[coordinates[2].clone(), zero],
                )?
                .remove(0);
            context.bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                branches.to_vec(),
                &[first, references[2].clone()],
            )?;
            // Masked tiles read their output tile through a masked load as well, because an ordinary accumulation
            // would read the padding of edge tiles.
            let masked = policy == BoundaryPolicy::Masked;
            let mask = context.lift(ArrayIrValue::Array(Array::matrix(block, block, vec![true; block * block])?))?;
            let other = context.lift(ArrayIrValue::Array(zeros.clone()))?;
            let mut tiles = references[..if masked { 3 } else { 2 }]
                .iter()
                .map(|reference| {
                    if masked {
                        let inputs = [reference.clone(), mask.clone(), other.clone()];
                        Ok(context.bind(MaskedLoadOperation::new(), vec![], &inputs)?.remove(0))
                    } else {
                        reference.read()
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let product = context
                .bind(ArrayIrOperation::from(ArrayOperation::Dot(DotOperation::matmul())), vec![], &tiles[..2])?
                .remove(0);
            if masked {
                let accumulator = tiles.pop().unwrap();
                let sum = context
                    .bind(
                        ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                        vec![],
                        &[accumulator, product],
                    )?
                    .remove(0);
                context.bind(MaskedStoreOperation::new(), vec![], &[references[2].clone(), sum, mask])?;
            } else {
                context.bind(ReferenceAddUpdateOperation::new(), vec![], &[references[2].clone(), product])?;
            }
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_kernel_interpretation_error_display() {
        assert_eq!(
            KernelInterpretationError::Nan {
                operation: "add",
                position: "input",
                value: 1,
                element: 2,
                data_type: DataType::F32,
                coordinate: vec![0, 1],
                provenance: Provenance::unknown(),
            }
            .to_string(),
            "nan in `add` input 1 element 2 of type `f32` at grid point [0, 1]",
        );
        let token = ArrayReference::new(Array::scalar(0i32).unwrap()).id();
        assert_eq!(
            KernelInterpretationError::UnknownCopyToken { token }.to_string(),
            "async copy completion token is unknown or already consumed",
        );
        assert_eq!(
            KernelInterpretationError::IncompatibleCompletion { operation: "test_wait", token }.to_string(),
            "`test_wait` does not follow the completion protocol of the producer of its token",
        );
        assert_eq!(
            KernelInterpretationError::PendingCopies { count: 2, coordinate: vec![3] }.to_string(),
            "kernel grid point [3] has 2 copies without a wait",
        );
        assert_eq!(
            KernelInterpretationError::StepLimit {
                maximum_steps: 5,
                coordinate: vec![1],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }
            .to_string(),
            "kernel interpretation exceeded 5 steps at grid point [1] in `reference_write`",
        );
    }

    #[test]
    fn test_kernel_debug_options_default() {
        assert_eq!(
            KernelDebugOptions::default(),
            KernelDebugOptions {
                maximum_programs: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
                maximum_steps: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
                nan_checks: false,
            },
        );
    }

    #[test]
    fn test_kernel_debug_options_with_maximum_programs() {
        let options = KernelDebugOptions::default().with_maximum_programs(3);
        assert_eq!(options.maximum_programs(), 3);
        assert_eq!(options.maximum_steps(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS);
        assert!(!options.nan_checks());

        // A grid with more points than the limit fails before qualification.
        let definition = definition(true, false);
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let mut trace = Vec::new();
        let error = definition.interpret_with_trace(vec![input.clone()], &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 4, maximum: 3 }),
        );
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &options.with_maximum_programs(4), &mut trace),
            Ok(vec![input]),
        );
    }

    #[test]
    fn test_kernel_debug_options_with_maximum_steps() {
        let options = KernelDebugOptions::default().with_maximum_steps(2);
        assert_eq!(options.maximum_programs(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS);
        assert_eq!(options.maximum_steps(), 2);
        assert!(!options.nan_checks());

        // The body entry and the read consume the two steps, so the write exceeds the limit.
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let mut trace = Vec::new();
        let error = definition.interpret_with_trace(vec![input.clone()], &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &options.with_maximum_steps(3), &mut trace),
            Ok(vec![input]),
        );
    }

    #[test]
    fn test_kernel_debug_options_with_nan_checks() {
        let options = KernelDebugOptions::default().with_nan_checks(true);
        assert_eq!(options.maximum_programs(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS);
        assert_eq!(options.maximum_steps(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS);
        assert!(options.nan_checks());
        assert_eq!(options.with_nan_checks(false), KernelDebugOptions::default());

        // Enabled checks reject a NaN input that disabled checks pass through unchanged.
        let definition = value_definition(
            ArrayOperation::Add(AddOperation::new()),
            &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
        );
        let inputs = vec![Array::scalar(f32::NAN).unwrap(), Array::scalar(1f32).unwrap()];
        let mut trace = Vec::new();
        let error = definition.interpret_with_trace(inputs.clone(), &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::Nan {
                operation: "reference_read",
                position: "output",
                value: 0,
                element: 0,
                data_type: DataType::F32,
                coordinate: vec![],
                provenance: Provenance::unknown(),
            }),
        );
        let output = definition.interpret_with_trace(inputs, &options.with_nan_checks(false), &mut trace).unwrap();
        assert!(output[0].elements::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn test_kernel_debug_options_maximum_programs() {
        assert_eq!(KernelDebugOptions::default().maximum_programs(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS);
        assert_eq!(KernelDebugOptions::default().with_maximum_programs(0).maximum_programs(), 0);
    }

    #[test]
    fn test_kernel_debug_options_maximum_steps() {
        assert_eq!(KernelDebugOptions::default().maximum_steps(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS);
        assert_eq!(KernelDebugOptions::default().with_maximum_steps(0).maximum_steps(), 0);
    }

    #[test]
    fn test_kernel_debug_options_nan_checks() {
        assert!(!KernelDebugOptions::default().nan_checks());
        assert!(KernelDebugOptions::default().with_nan_checks(true).nan_checks());
    }

    #[test]
    fn test_kernel_trace_entry_coordinate() {
        let entry = KernelTraceEntry {
            coordinate: vec![2, 3],
            operation: "reference_read",
            provenance: Provenance::unknown(),
            accesses: vec![],
            pending_copies: vec![],
        };
        assert_eq!(entry.coordinate(), &[2, 3]);
    }

    #[test]
    fn test_kernel_trace_entry_operation() {
        let entry = KernelTraceEntry {
            coordinate: vec![],
            operation: "reference_write",
            provenance: Provenance::unknown(),
            accesses: vec![],
            pending_copies: vec![],
        };
        assert_eq!(entry.operation(), "reference_write");
    }

    #[test]
    fn test_kernel_trace_entry_provenance() {
        let provenance = Provenance::scope(ProvenanceScope::new("layer"), Provenance::unknown());
        let entry = KernelTraceEntry {
            coordinate: vec![],
            operation: "reference_read",
            provenance: provenance.clone(),
            accesses: vec![],
            pending_copies: vec![],
        };
        assert_eq!(entry.provenance(), &provenance);
    }

    #[test]
    fn test_kernel_trace_entry_accesses() {
        let access = KernelTraceAccess {
            input: 1,
            mode: ReferenceAccessMode::Write,
            root: 0,
            path: "/slice(axes=[])".to_owned(),
        };
        let entry = KernelTraceEntry {
            coordinate: vec![],
            operation: "reference_write",
            provenance: Provenance::unknown(),
            accesses: vec![access.clone()],
            pending_copies: vec![],
        };
        assert_eq!(entry.accesses(), &[access]);
    }

    #[test]
    fn test_kernel_trace_entry_pending_copies() {
        let entry = KernelTraceEntry {
            coordinate: vec![],
            operation: "wait",
            provenance: Provenance::unknown(),
            accesses: vec![],
            pending_copies: vec![2, 5],
        };
        assert_eq!(entry.pending_copies(), &[2, 5]);
    }

    #[test]
    fn test_kernel_trace_entry_display() {
        let entry = KernelTraceEntry {
            coordinate: vec![2, 3],
            operation: "reference_read",
            provenance: Provenance::default(),
            accesses: vec![KernelTraceAccess {
                input: 0,
                mode: ReferenceAccessMode::Read,
                root: 1,
                path: "/slice(axes=[0:2])".to_owned(),
            }],
            pending_copies: vec![4],
        };
        assert_eq!(entry.to_string(), "[2, 3] reference_read input[0]=read(root[1]/slice(axes=[0:2])) pending=[4]");
    }

    #[test]
    fn test_kernel_trace_access_input() {
        let access = KernelTraceAccess { input: 3, mode: ReferenceAccessMode::Read, root: 0, path: String::new() };
        assert_eq!(access.input(), 3);
    }

    #[test]
    fn test_kernel_trace_access_mode() {
        let access = KernelTraceAccess { input: 0, mode: ReferenceAccessMode::Consume, root: 0, path: String::new() };
        assert_eq!(access.mode(), ReferenceAccessMode::Consume);
    }

    #[test]
    fn test_kernel_trace_access_root() {
        let access = KernelTraceAccess { input: 0, mode: ReferenceAccessMode::Read, root: 4, path: String::new() };
        assert_eq!(access.root(), 4);
    }

    #[test]
    fn test_kernel_trace_access_path() {
        let access = KernelTraceAccess {
            input: 0,
            mode: ReferenceAccessMode::Read,
            root: 0,
            path: "/slice(axes=[0:2])".to_owned(),
        };
        assert_eq!(access.path(), "/slice(axes=[0:2])");
    }

    #[test]
    fn test_kernel_definition_interpret() {
        let input = Array::scalar(17i32).unwrap();
        assert_eq!(definition(false, false).interpret(vec![input.clone()], 1), Ok(vec![input.clone()]));
        assert_eq!(input, Array::scalar(17i32).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_reduction() {
        let input = Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let definition = value_definition(
            ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
            &[input.r#type().into_owned()],
        );
        assert_eq!(definition.interpret(vec![input], 1), Ok(vec![Array::vector(vec![6i32, 15]).unwrap()]));
        assert_eq!(definition.simplified().unwrap().operation().output_types(), definition.operation().output_types());
    }

    #[test]
    fn test_kernel_definition_interpret_scaled_dot() {
        let inputs = vec![
            Array::matrix(1, 4, vec![1f32, 2., 3., 4.]).unwrap(),
            Array::matrix(4, 1, vec![1f32, 1., 1., 1.]).unwrap(),
            Array::matrix(1, 2, vec![2f32, 4.]).unwrap(),
            Array::matrix(2, 1, vec![1f32, 2.]).unwrap(),
        ];
        let definition = value_definition(
            ArrayOperation::ScaledDot(ScaledDotOperation::new(
                ScaledDotOperation::default_dimensions(2).unwrap(),
                DataType::F32,
                true,
                true,
            )),
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        );
        let expected = vec![Array::matrix(1, 1, vec![62f32]).unwrap()];
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(expected.clone()));
        assert_eq!(definition.simplified().unwrap().interpret(inputs, 1), Ok(expected));
    }

    #[test]
    fn test_kernel_definition_interpret_vector_add() {
        assert_eq!(
            definition(true, true).interpret(
                vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap(), Array::vector(vec![5i32, 6, 7, 8]).unwrap()],
                4,
            ),
            Ok(vec![Array::vector(vec![6i32, 8, 10, 12]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_parallel_accumulation() {
        // Every parallel grid point accumulates into its own window of one element.
        let mapping = definition(true, false).operation().parameters()[0].mapping().clone();
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::ReadWrite,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let update = context.lift(ArrayIrValue::Array(Array::vector(vec![10i32])?))?;
            context.bind(ReferenceAddUpdateOperation::new(), vec![], &[references[0].clone(), update])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap()], 4),
            Ok(vec![Array::vector(vec![11i32, 12, 13, 14]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_first_step_initialization() {
        // Every output tile is zeroed at `k == 0` and then accumulates along the sequential axis `k`.
        let definition = carried_matmul_definition([4, 4, 4], 2, BoundaryPolicy::InBounds);
        let left = Array::matrix(4, 4, (1..=16).map(|value| value as f32).collect()).unwrap();
        let right = Array::matrix(4, 4, (1..=16).rev().map(|value| value as f32).collect()).unwrap();
        let expected = Array::matrix(
            4,
            4,
            vec![80f32, 70., 60., 50., 240., 214., 188., 162., 400., 358., 316., 274., 560., 502., 444., 386.],
        )
        .unwrap();
        assert_eq!(definition.interpret(vec![left, right], 8), Ok(vec![expected]));

        // With a single step along `k`, its coordinate is always zero.
        let definition = carried_matmul_definition([2, 2, 2], 2, BoundaryPolicy::InBounds);
        let left = Array::matrix(2, 2, vec![1f32, 2., 3., 4.]).unwrap();
        let right = Array::matrix(2, 2, vec![5f32, 6., 7., 8.]).unwrap();
        let expected = Array::matrix(2, 2, vec![19f32, 22., 43., 50.]).unwrap();
        assert_eq!(definition.interpret(vec![left, right], 1), Ok(vec![expected]));
    }

    #[test]
    fn test_kernel_definition_interpret_first_step_initialization_masked_edges() {
        // Every step after the first loads the valid elements that the previous step published into its fresh tile.
        let definition = carried_matmul_definition([3, 4, 3], 2, BoundaryPolicy::Masked);
        let left = Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap();
        let right = Array::matrix(4, 3, (1..=12).map(|value| value as f32).collect()).unwrap();
        let expected = Array::matrix(3, 3, vec![70f32, 80., 90., 158., 184., 210., 246., 288., 330.]).unwrap();
        assert_eq!(definition.interpret(vec![left, right], 8), Ok(vec![expected]));
    }

    #[test]
    fn test_kernel_definition_interpret_coordinates() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = mapping.add_input(ArrayIrType::Dimension(DimensionType::new(
            "coordinate",
            DimensionBounds::non_negative(None).unwrap(),
        )));
        let mapping = BlockMapping::new(
            mapping.build(vec![coordinate], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I64, [4]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, coordinates)| {
            let context = coordinates[0].context();
            let value = context
                .bind(ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation), vec![], &coordinates)?
                .remove(0);
            context.bind(
                ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]),
                vec![],
                &[references[0].clone(), value],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 4), Ok(vec![Array::vector(vec![0i64, 1, 2, 3]).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_interpret_dynamic_index() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::I32, [4]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let index = references[1].read()?;
            let value = references[0]
                .context()
                .bind(
                    ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Dynamic,
                    }]),
                    vec![],
                    &[references[0].clone(), index],
                )?
                .remove(0);
            references[2].write(&value)
        })
        .unwrap();
        let inputs = vec![Array::vector(vec![10i32, 20, 30, 40]).unwrap(), Array::scalar(2i64).unwrap()];
        assert_eq!(definition.interpret(inputs, 1), Ok(vec![Array::scalar(30i32).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_interpret_budget() {
        // Adding zero to the coordinate hides the separable tiling proof, so qualification enumerates the grid.
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
        let coordinate = mapping.add_input(ArrayIrType::Dimension(coordinate_type.clone()));
        let zero = DimensionValue::constant(0).unwrap();
        let zero_id = mapping.add_constant(ArrayIrValue::Dimension(zero.clone()));
        let start = mapping
            .add_instruction(
                DimensionAddOperation::new(&coordinate_type, zero.r#type().as_ref()).unwrap(),
                vec![],
                vec![coordinate, zero_id],
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
        let r#type = ArrayType::new_static(DataType::I32, [2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[1].write(&references[0].read()?)
        })
        .unwrap();
        let input = Array::vector(vec![3i32, 5]).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 2), Ok(vec![input.clone()]));
        let error = definition.interpret(vec![input], 1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 2, maximum: 1 }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_budget_skips_tiled_proofs() {
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        assert_eq!(definition(true, false).interpret(vec![input.clone()], 0), Ok(vec![input]));
    }

    #[test]
    fn test_kernel_definition_interpret_budget_repeated_axes() {
        // Copies the diagonal of a 3 by 3 matrix into a 3 by 1 matrix. The diagonal element starts `(i, i)` are a
        // recognized tiling, but they repeat the grid axis, so qualification enumerates the 3 grid points.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(Dimension::Static(3), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [3, 3]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![1, 1],
                        boundary: BoundaryPolicy::InBounds,
                        starts: vec![MappingExpression::GridCoordinate(0), MappingExpression::GridCoordinate(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [3, 1]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![1, 1],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::GridCoordinate(0), MappingExpression::Constant(0)],
                    },
                ),
            ],
            vec![],
        )
        .unwrap();
        let parameters = operation.parameters();
        assert_eq!(parameters[0].mapping().tiling_axes(), Some(vec![Some(0), Some(0)]));
        assert_eq!(parameters[1].mapping().tiling_axes(), Some(vec![Some(0), None]));
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[1].write(&references[0].read()?)
        })
        .unwrap();

        // The grid is accepted at the bound and rejected below it.
        let input = Array::matrix(3, 3, vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(
            definition.interpret(vec![input.clone()], 3),
            Ok(vec![Array::matrix(3, 1, vec![1i32, 5, 9]).unwrap()]),
        );
        assert_eq!(
            definition.interpret(vec![input], 2),
            Err(ProgramError::custom(KernelInitializationError::QualificationLimit { programs: 3, maximum: 2 })),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_without_step_limit() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let error = definition.interpret_with_limits(vec![input.clone()], 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(vec![input]));
    }

    #[test]
    fn test_kernel_definition_interpret_scratch() {
        let operation = definition(false, false).operation().clone();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let scratch = references[0]
                .context()
                .bind(ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap(), vec![], &[])?
                .remove(0);
            scratch.write(&references[0].read()?)?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(23i32).unwrap()], 1),
            Ok(vec![Array::scalar(23i32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_write_only_read_back() {
        // The first output is written with zeros, accumulated into, and then read back into the second output.
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let zero = context.lift(ArrayIrValue::Array(Array::scalar(0i32)?))?;
            references[1].write(&zero)?;
            let input = references[0].read()?;
            context.bind(ReferenceAddUpdateOperation::new(), vec![], &[references[1].clone(), input])?;
            references[2].write(&references[1].read()?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(7i32).unwrap()], 1),
            Ok(vec![Array::scalar(7i32).unwrap(), Array::scalar(7i32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_nested_scratch() {
        let operation = definition(false, false).operation().clone();
        let branch = KernelDefinition::<NoKernelExtension>::trace(operation.clone(), |(references, _coordinates)| {
            let scratch = references[0]
                .context()
                .bind(ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap(), vec![], &[])?
                .remove(0);
            scratch.write(&references[0].read()?)?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let predicate = context.lift(ArrayIrValue::Array(Array::scalar(true).unwrap()))?;
            context.bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch.body().clone(), branch.body().clone()],
                &[predicate, references[0].clone(), references[1].clone()],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(31i32).unwrap()], 1),
            Ok(vec![Array::scalar(31i32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_storage_without_zero() {
        // `f8e8m0fnu` cannot represent zero, so its write-only output and scratch storage must not be zero filled.
        let r#type = ArrayType::new_static(DataType::F8E8M0FNU, [2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation.clone(), |(references, _)| {
            let scratch = references[0]
                .context()
                .bind(ScratchOperation::new(r#type.clone(), 1).unwrap(), vec![], &[])?
                .remove(0);
            scratch.write(&references[0].read()?)?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        let input = Array::new(r#type.clone(), vec![0x7f, 0x83]).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(vec![input.clone()]));
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &KernelDebugOptions::default(), &mut trace),
            Ok(vec![input.clone()]),
        );

        // Reading an element of the scratch storage that was never written still fails qualification.
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _)| {
            let context = references[0].context();
            let scratch = context.bind(ScratchOperation::new(r#type.clone(), 1).unwrap(), vec![], &[])?.remove(0);
            let value = Array::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![0x7f])?;
            let value = context.lift(ArrayIrValue::Array(value))?;
            let write = ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Static(0),
            }]);
            context.bind(write, vec![], &[scratch.clone(), value])?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        let region = definition.body().entry_region_ref().id();
        let error = definition.interpret(vec![input], 1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(region, 2),
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_async_copy() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = mapping.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            mapping.build(vec![zero], vec![], vec![Placeholder]).unwrap(),
            vec![2],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [2]);
        let output_type = input_type
            .clone()
            .with_memory(Memory::Host { pinned: false })
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(input_type, KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(output_type.clone(), KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let token = context
                .bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), references[1].clone()])?
                .remove(0);
            context.bind(WaitOperation, vec![], &[token])?;
            Ok(())
        })
        .unwrap();
        let input = Array::vector(vec![17i32, 29]).unwrap();
        let expected = vec![Array::from_elements(output_type, &[17i32, 29]).unwrap()];
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(expected.clone()));
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &KernelDebugOptions::default(), &mut trace),
            Ok(expected),
        );
        assert_eq!(trace.iter().map(|entry| entry.operation()).collect::<Vec<_>>(), vec!["async_copy", "wait"]);
        assert_eq!(trace[0].pending_copies(), Vec::<usize>::new());
        assert_eq!(trace[1].pending_copies(), vec![2]);
        assert_eq!(trace[1].accesses()[0].mode(), ReferenceAccessMode::Consume);
        assert_eq!(trace[1].accesses()[0].root(), 2);
        assert_eq!(input, Array::vector(vec![17i32, 29]).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_async_copy_across_condition() {
        // Double buffering, where a condition computes on other storage while the copy is pending.
        let vector = ArrayType::new_static(DataType::I32, [2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(vector.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(vector, KernelParameterAccess::WriteOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let branches = [5i32, 7].map(|value| {
            let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = branch.add_input(operation.parameters()[2].body_type());
            let value = branch.add_constant(ArrayIrValue::Array(Array::scalar(value).unwrap()));
            branch
                .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
                .unwrap();
            branch
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
                .unwrap()
        });
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let token = context
                .bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), references[1].clone()])?
                .remove(0);
            let predicate = context.lift(ArrayIrValue::Array(Array::scalar(true)?))?;
            context.bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                branches.to_vec(),
                &[predicate, references[2].clone()],
            )?;
            context.bind(WaitOperation, vec![], &[token])?;
            Ok(())
        })
        .unwrap();
        let input = Array::vector(vec![17i32, 29]).unwrap();
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &KernelDebugOptions::default(), &mut trace),
            Ok(vec![input, Array::scalar(5i32).unwrap()]),
        );
        // The branch replays with the pending copy that the body issued before entering it.
        assert_eq!(
            trace.iter().map(|entry| (entry.operation(), entry.pending_copies().to_vec())).collect::<Vec<_>>(),
            vec![("async_copy", vec![]), ("condition", vec![2]), ("reference_write", vec![2]), ("wait", vec![2])],
        );
    }

    #[test]
    fn test_kernel_definition_interpret_masked_edges() {
        let mapping = tiled_mapping(vec![2, 2]);
        let r#type = ArrayType::new_static(DataType::I32, [5, 3]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![
                GridDimension::new(Dimension::Static(3), GridExecution::Parallel),
                GridDimension::new(Dimension::Static(2), GridExecution::Parallel),
            ])
            .unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(r#type.clone(), KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::matrix(2, 2, vec![true; 4])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::matrix(2, 2, vec![-7i32; 4])?))?;
            let value = context
                .bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask.clone(), other])?
                .remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[1].clone(), value, mask])?;
            Ok(())
        })
        .unwrap();
        let input = Array::from_elements(r#type, &(0..15i32).collect::<Vec<_>>()).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 6), Ok(vec![input.clone()]));
        assert_eq!(input.elements::<i32>().unwrap(), (0..15i32).collect::<Vec<_>>());
    }

    #[test]
    fn test_kernel_definition_interpret_masked_edges_without_zero() {
        // The private edge tiles of the `f8e8m0fnu` input and of its write-only copy cannot be zero filled. The
        // explicit fill of the masked load is the only padding value that reaches the second output.
        let masked = tiled_mapping(vec![2]);
        let in_bounds = BlockMapping::new(masked.program().clone(), vec![2], BoundaryPolicy::InBounds).unwrap();
        let r#type = ArrayType::new_static(DataType::F8E8M0FNU, [3]);
        let padded_type = ArrayType::new_static(DataType::F8E8M0FNU, [4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, masked.clone()).unwrap(),
                KernelParameter::new(r#type.clone(), KernelParameterAccess::WriteOnly, masked).unwrap(),
                KernelParameter::new(padded_type.clone(), KernelParameterAccess::WriteOnly, in_bounds).unwrap(),
            ],
        )
        .unwrap();
        let tile_type = ArrayType::new_static(DataType::F8E8M0FNU, [2]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, true])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::new(tile_type.clone(), vec![0x81, 0x81])?))?;
            let value = context
                .bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask.clone(), other])?
                .remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[1].clone(), value.clone(), mask])?;
            references[2].write(&value)
        })
        .unwrap();
        let input = Array::new(r#type, vec![0x7f, 0x80, 0x83]).unwrap();
        assert_eq!(
            definition.interpret(vec![input.clone()], 2),
            Ok(vec![input, Array::new(padded_type, vec![0x7f, 0x80, 0x83, 0x81]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_window_mask() {
        // The first window is a masked edge window whose second grid point has one valid element, while every element
        // of the in-bounds output windows is valid. Each output records the mask of one window.
        let masked = tiled_mapping(vec![2]);
        let in_bounds = BlockMapping::new(masked.program().clone(), vec![2], BoundaryPolicy::InBounds).unwrap();
        let output_type = ArrayType::new_static(DataType::Boolean, [4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [3]),
                    KernelParameterAccess::ReadOnly,
                    masked,
                )
                .unwrap(),
                KernelParameter::new(output_type.clone(), KernelParameterAccess::WriteOnly, in_bounds.clone()).unwrap(),
                KernelParameter::new(output_type, KernelParameterAccess::WriteOnly, in_bounds).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let masked = context.bind(WindowMaskOperation, vec![], &[references[0].clone()])?.remove(0);
            references[1].write(&masked)?;
            let in_bounds = context.bind(WindowMaskOperation, vec![], &[references[2].clone()])?.remove(0);
            references[2].write(&in_bounds)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2),
            Ok(vec![
                Array::vector(vec![true, true, true, false]).unwrap(),
                Array::vector(vec![true, true, true, true]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_window_mask_store() {
        // A store under the validity mask of its own window initializes the write-only output, and only the valid
        // elements of the edge window are published.
        let mapping = tiled_mapping(vec![2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [3]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5i32, 7])?))?;
            let mask = context.bind(WindowMaskOperation, vec![], &[references[0].clone()])?.remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[0].clone(), value, mask])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 2), Ok(vec![Array::vector(vec![5i32, 7, 5]).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_interpret_complementary_masked_stores() {
        // Stores under a runtime mask and its negation together initialize the write-only output, in either order.
        let mapping =
            BlockMapping::new(tiled_mapping(vec![2]).program().clone(), vec![2], BoundaryPolicy::InBounds).unwrap();
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        for complement_first in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(operation.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(operation.coordinate_types()[0].clone()));
            let first = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![1i32, 2]).unwrap()));
            let second = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![3i32, 4]).unwrap()));
            let constant = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()));
            let not = ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new()));
            let mask = builder.add_instruction(not.clone(), vec![], vec![constant], None).unwrap()[0];
            let complement = builder.add_instruction(not, vec![], vec![mask], None).unwrap()[0];
            let stores = if complement_first {
                [(second, complement), (first, mask)]
            } else {
                [(first, mask), (second, complement)]
            };
            for (value, mask) in stores {
                builder
                    .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                    .unwrap();
            }
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let definition: KernelDefinition = KernelDefinition::new(operation.clone(), body).unwrap();
            assert_eq!(definition.interpret(vec![], 2), Ok(vec![Array::vector(vec![1i32, 4, 1, 4]).unwrap()]));
        }
    }

    #[test]
    fn test_kernel_definition_interpret_masked_other() {
        let mapping = tiled_mapping(vec![4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [2]),
                    KernelParameterAccess::ReadOnly,
                    mapping.clone(),
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![7i32; 4])?))?;
            let value =
                context.bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask, other])?.remove(0);
            references[1].write(&value)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2]).unwrap()], 1),
            Ok(vec![Array::vector(vec![1i32, 7, 7, 7]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_masked_swap_preserves_inactive_storage() {
        let mapping = tiled_mapping(vec![4]);
        let r#type = ArrayType::new_static(DataType::I32, [5]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadWrite, mapping.clone()).unwrap(),
                KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            let replacement = context.lift(ArrayIrValue::Array(Array::vector(vec![9i32, 10, 11, 12])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![-1i32; 4])?))?;
            let old = context
                .bind(MaskedSwapOperation::new(), vec![], &[references[0].clone(), replacement, mask, other])?
                .remove(0);
            references[1].write(&old)
        })
        .unwrap();
        let input = Array::vector(vec![1i32, 2, 3, 4, 5]).unwrap();
        let expected =
            vec![Array::vector(vec![9i32, 2, 11, 12, 9]).unwrap(), Array::vector(vec![1i32, -1, 3, 4, 5]).unwrap()];
        assert_eq!(definition.interpret(vec![input.clone()], 2), Ok(expected));
        assert_eq!(input.elements::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_kernel_definition_interpret_masked_dead_swap_is_store() {
        let mapping = tiled_mapping(vec![2]);
        let r#type = ArrayType::new_static(DataType::I32, [3]);
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        let operation = KernelCallOperation::new(
            grid.clone(),
            vec![KernelParameter::new(r#type.clone(), KernelParameterAccess::WriteOnly, mapping.clone()).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5i32, 7])?))?;
            context.bind(ReferenceSwapOperation::new(), vec![], &[references[0].clone(), value])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 2), Ok(vec![Array::vector(vec![5i32, 7, 5]).unwrap()]));

        // A swap whose old value is used reads the padded edge window, so qualification still rejects it.
        let operation = KernelCallOperation::new(
            grid,
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadWrite, mapping.clone()).unwrap(),
                KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5i32, 7])?))?;
            let old = context.bind(ReferenceSwapOperation::new(), vec![], &[references[0].clone(), value])?.remove(0);
            references[1].write(&old)
        })
        .unwrap();
        let error = definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                operation: "reference_swap",
            })),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_masked_copies() {
        /// A well-typed masked copy with one repeated element mask and independently initialized destination storage.
        #[derive(Debug)]
        struct MaskedCopyCase {
            /// Immutable source elements.
            input: Vec<i32>,

            /// Destination elements before the masked stores.
            initial: Vec<i32>,

            /// Element mask, whose nonzero length is also the logical block width.
            mask: Vec<bool>,
        }

        impl MaskedCopyCase {
            /// Generates a fixed bounded enumeration including empty arrays, partial tiles, and inactive masks.
            fn generate() -> Vec<Self> {
                let mut cases = Vec::new();
                for extent in 0..=5 {
                    for block in 1..=3 {
                        for bits in 0..(1usize << block) {
                            for pattern in 0..2 {
                                cases.push(Self {
                                    input: (0..extent).map(|index| (index - 2) * (pattern + 1)).collect(),
                                    initial: (0..extent).map(|index| 7 - index * (pattern + 1)).collect(),
                                    mask: (0..block).map(|element| bits & (1 << element) != 0).collect(),
                                });
                            }
                        }
                    }
                }
                cases
            }

            /// Computes expected immutable values without using kernel mappings, references, or array operations.
            fn expected(&self) -> Vec<i32> {
                self.input
                    .iter()
                    .zip(&self.initial)
                    .enumerate()
                    .map(|(index, (&input, &initial))| if self.mask[index % self.mask.len()] { input } else { initial })
                    .collect()
            }

            /// Constructs the same canonical typed kernel shape for each generated case.
            fn kernel(&self) -> KernelDefinition {
                let block = self.mask.len();
                let programs = self.input.len().div_ceil(block);
                let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                let coordinate_type =
                    DimensionType::new("coordinate", DimensionBounds::non_negative(Some(programs.max(1))).unwrap());
                let coordinate = mapping.add_input(coordinate_type.clone().into());
                let width = DimensionValue::constant(block).unwrap();
                let width_input = mapping.add_constant(ArrayIrValue::Dimension(width.clone()));
                let start = mapping
                    .add_instruction(
                        DimensionMulOperation::new(&coordinate_type, width.r#type().as_ref()).unwrap(),
                        vec![],
                        vec![coordinate, width_input],
                        None,
                    )
                    .unwrap()[0];
                let mapping = BlockMapping::new(
                    mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
                    vec![block],
                    BoundaryPolicy::Masked,
                )
                .unwrap();
                let r#type = ArrayType::new_static(DataType::I32, vec![self.input.len()]);
                let operation = KernelCallOperation::new(
                    Grid::new(vec![GridDimension::new(Dimension::Static(programs), GridExecution::Parallel)]).unwrap(),
                    vec![
                        KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                        KernelParameter::new(r#type, KernelParameterAccess::ReadWrite, mapping).unwrap(),
                    ],
                )
                .unwrap();
                KernelDefinition::trace(operation, |(references, _coordinates)| {
                    let context = references[0].context();
                    let mask = context.lift(ArrayIrValue::Array(Array::vector(self.mask.clone())?))?;
                    let other = context.lift(ArrayIrValue::Array(Array::vector(vec![0i32; block])?))?;
                    let value = context
                        .bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask.clone(), other])?
                        .remove(0);
                    context.bind(MaskedStoreOperation::new(), vec![], &[references[1].clone(), value, mask])?;
                    Ok(())
                })
                .unwrap()
            }
        }

        for case in MaskedCopyCase::generate() {
            let input = Array::vector(case.input.clone()).unwrap();
            let initial = Array::vector(case.initial.clone()).unwrap();
            assert_eq!(
                case.kernel().interpret(vec![input.clone(), initial.clone()], 5),
                Ok(vec![Array::vector(case.expected()).unwrap()]),
                "{case:?}",
            );
            assert_eq!(input.elements::<i32>().unwrap(), case.input, "{case:?}");
            assert_eq!(initial.elements::<i32>().unwrap(), case.initial, "{case:?}");
        }
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        assert_eq!(definition.interpret_with_limits(vec![input.clone()], 1, 3), Ok(vec![input.clone()]));
        let error = definition.interpret_with_limits(vec![input.clone()], 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(error.to_string(), "kernel interpretation exceeded 2 steps at grid point [] in `reference_write`");
        let error = definition.interpret_with_limits(vec![input.clone()], 0, 3).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 1, maximum: 0 }),
        );
        assert_eq!(input, Array::scalar(17i32).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_caps_tiled_grids() {
        let error = definition(true, false)
            .interpret_with_limits(
                vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap()],
                3,
                DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 4, maximum: 3 }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_shares_grid_budget() {
        let error = definition(true, false)
            .interpret_with_limits(vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap()], 4, 5)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 5,
                coordinate: vec![1],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_preserves_provenance() {
        let operation = definition(false, false).operation().clone();
        let scope = ProvenanceScope::new("copy output");
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let value = references[0].read()?;
            references[1].context().invoke_with_provenance_scope(scope.clone(), || references[1].write(&value))
        })
        .unwrap();
        let error = definition.interpret_with_limits(vec![Array::scalar(3i32).unwrap()], 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::scope(scope, Provenance::unknown()),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_counts_empty_loop_regions() {
        let mut condition = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![predicate], vec![], vec![Placeholder])
            .unwrap();
        let loop_body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![])
            .unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let condition = body.import_region(condition.entry_region_ref());
        let loop_body = body.import_region(loop_body.entry_region_ref());
        body.add_instruction(
            ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(100).unwrap()),
            vec![condition, loop_body],
            vec![],
            None,
        )
        .unwrap();
        let definition = KernelDefinition::new(
            KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
            body.build(vec![], vec![], vec![]).unwrap(),
        )
        .unwrap();
        let error = definition.interpret_with_limits(vec![], 1, 4).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 4,
                coordinate: vec![],
                operation: "while",
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions::default().with_maximum_programs(1).with_maximum_steps(3),
                &mut trace,
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            trace.iter().map(|entry| entry.operation()).collect::<Vec<_>>(),
            vec!["reference_read", "reference_write"],
        );
        assert_eq!(
            trace.iter().map(|entry| entry.coordinate().to_vec()).collect::<Vec<_>>(),
            vec![Vec::<usize>::new(), vec![]],
        );
        assert_eq!(trace[0].accesses()[0].mode(), ReferenceAccessMode::Read);
        assert_eq!(trace[0].accesses()[0].root(), 0);
        assert_eq!(trace[0].accesses()[0].path(), "/slice(axes=[])");
        assert_eq!(trace[1].accesses()[0].mode(), ReferenceAccessMode::Write);
        assert_eq!(trace[1].accesses()[0].root(), 1);
        assert_eq!(trace[0].pending_copies(), Vec::<usize>::new());
        let first = trace.clone();
        assert_eq!(
            definition.interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions::default().with_maximum_programs(1).with_maximum_steps(3),
                &mut trace,
            ),
            Ok(vec![input]),
        );
        assert_eq!(trace, first);
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan() {
        let definition = value_definition(
            ArrayOperation::Add(AddOperation::new()),
            &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
        );
        let inputs = vec![Array::scalar(f32::INFINITY).unwrap(), Array::scalar(f32::NEG_INFINITY).unwrap()];
        let mut trace = Vec::new();
        let options = KernelDebugOptions::default().with_nan_checks(true);
        let error = definition.interpret_with_trace(inputs.clone(), &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::Nan {
                operation: "add",
                position: "output",
                value: 0,
                element: 0,
                data_type: DataType::F32,
                coordinate: vec![],
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(error.to_string(), "nan in `add` output 0 element 0 of type `f32` at grid point []");
        assert_eq!(trace.last().unwrap().operation(), "add");
        let output = definition.interpret_with_trace(inputs, &KernelDebugOptions::default(), &mut trace).unwrap();
        assert!(output[0].elements::<f32>().unwrap()[0].is_nan());
        assert_eq!(trace.last().unwrap().operation(), "reference_write");
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan_input_formats() {
        for input in [
            Array::new(ArrayType::scalar(DataType::F8E4M3FN), vec![0x7f]).unwrap(),
            Array::scalar(Complex::new(0f64, f64::NAN)).unwrap(),
        ] {
            let data_type = input.r#type().data_type();
            let definition = value_definition(
                ArrayOperation::Add(AddOperation::new()),
                &[input.r#type().into_owned(), input.r#type().into_owned()],
            );
            let mut trace = Vec::new();
            let error = definition
                .interpret_with_trace(
                    vec![input.clone(), input],
                    &KernelDebugOptions::default().with_nan_checks(true),
                    &mut trace,
                )
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<KernelInterpretationError>(),
                Some(&KernelInterpretationError::Nan {
                    operation: "reference_read",
                    position: "output",
                    value: 0,
                    element: 0,
                    data_type,
                    coordinate: vec![],
                    provenance: Provenance::unknown(),
                }),
            );
            assert_eq!(trace.len(), 1);
        }
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan_skips_async_copy_storage() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let token = context
                .bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), references[1].clone()])?
                .remove(0);
            context.bind(WaitOperation, vec![], &[token])?;
            Ok(())
        })
        .unwrap();
        let mut trace = Vec::new();
        let options = KernelDebugOptions::default().with_nan_checks(true);
        let outputs = definition
            .interpret_with_trace(vec![Array::scalar(f32::NAN).unwrap()], &options, &mut trace)
            .unwrap();
        assert_eq!(outputs[0].elements::<f32>().unwrap()[0].to_bits(), f32::NAN.to_bits());
        assert_eq!(trace.iter().map(|entry| entry.operation()).collect::<Vec<_>>(), vec!["async_copy", "wait"]);
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan_skips_dead_swaps() {
        // Each program swaps a finite value into one element of a read-write vector whose first element is NaN. When
        // the old value is dead, the swap is a store that never reads the NaN, so ordinary, traced, and interleaved
        // execution agree. When the old value is copied to a second output, the swap reads it and reports the NaN.
        let mapping =
            BlockMapping::new(tiled_mapping(vec![1]).program().clone(), vec![1], BoundaryPolicy::InBounds).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        let swap_definition = |masked: bool, live: bool| {
            let mut parameters =
                vec![KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadWrite, mapping.clone()).unwrap()];
            if live {
                let output = KernelParameter::new(r#type.clone(), KernelParameterAccess::WriteOnly, mapping.clone());
                parameters.push(output.unwrap());
            }
            let operation = KernelCallOperation::new(grid.clone(), parameters).unwrap();
            KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
                let context = references[0].context();
                let value = context.lift(ArrayIrValue::Array(Array::vector(vec![3f32])?))?;
                let old = if masked {
                    let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true])?))?;
                    let other = context.lift(ArrayIrValue::Array(Array::vector(vec![0f32])?))?;
                    let inputs = [references[0].clone(), value, mask, other];
                    context.bind(MaskedSwapOperation::new(), vec![], &inputs)?.remove(0)
                } else {
                    context.bind(ReferenceSwapOperation::new(), vec![], &[references[0].clone(), value])?.remove(0)
                };
                if live { references[1].write(&old) } else { Ok(()) }
            })
            .unwrap()
        };
        let input = Array::vector(vec![f32::NAN, 1f32]).unwrap();
        let updated = Array::vector(vec![3f32, 3f32]).unwrap();
        let options = KernelDebugOptions::default().with_nan_checks(true);
        for (masked, operation) in [(false, "reference_swap"), (true, "masked_swap")] {
            // Dead swaps are traced as writes.
            let definition = swap_definition(masked, false);
            assert_eq!(definition.interpret(vec![input.clone()], 2), Ok(vec![updated.clone()]));
            let mut trace = Vec::new();
            assert_eq!(
                definition.interpret_with_trace(vec![input.clone()], &options, &mut trace),
                Ok(vec![updated.clone()]),
            );
            assert_eq!(
                trace
                    .iter()
                    .map(|entry| {
                        let modes = entry.accesses().iter().map(|access| access.mode()).collect::<Vec<_>>();
                        (entry.coordinate().to_vec(), entry.operation(), modes)
                    })
                    .collect::<Vec<_>>(),
                vec![
                    (vec![0], operation, vec![ReferenceAccessMode::Write]),
                    (vec![1], operation, vec![ReferenceAccessMode::Write]),
                ],
            );
            let executions = definition.interpret_interleavings(vec![input.clone()], &options, 2).unwrap();
            assert_eq!(executions.len(), 2);
            for execution in executions {
                assert_eq!(execution.result(), &Ok(vec![updated.clone()]));
                assert_eq!(execution.trace().len(), 2);
            }

            // A live old value is read, so the NaN check still reports it.
            let definition = swap_definition(masked, true);
            let outputs = definition.interpret(vec![input.clone()], 2).unwrap();
            assert_eq!(outputs[0], updated);
            assert_eq!(
                outputs[1].elements::<f32>().unwrap().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                vec![f32::NAN.to_bits(), 1f32.to_bits()],
            );
            let error = definition.interpret_with_trace(vec![input.clone()], &options, &mut trace).unwrap_err();
            assert_eq!(
                error.downcast_custom::<KernelInterpretationError>(),
                Some(&KernelInterpretationError::Nan {
                    operation,
                    position: "output",
                    value: 0,
                    element: 0,
                    data_type: DataType::F32,
                    coordinate: vec![0],
                    provenance: Provenance::unknown(),
                }),
            );
            assert_eq!(trace.last().unwrap().accesses()[0].mode(), ReferenceAccessMode::ReadWrite);
        }
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_failure() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let mut trace = Vec::new();
        let error = definition
            .interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions::default().with_maximum_programs(1).with_maximum_steps(2),
                &mut trace,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "kernel interpretation exceeded 2 steps at grid point [] in `reference_write`");
        assert_eq!(
            trace.iter().map(|entry| entry.operation()).collect::<Vec<_>>(),
            vec!["reference_read", "reference_write"],
        );
        let error = definition
            .interpret_with_trace(
                vec![input],
                &KernelDebugOptions::default().with_maximum_programs(0).with_maximum_steps(2),
                &mut trace,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "kernel qualification requires 1 programs, exceeding the limit 0");
        assert_eq!(trace, Vec::<KernelTraceEntry>::new());
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_failed_selection() {
        // The second read selects an element of `f32[0]` at a dynamic index that the first read loads, so resolving
        // its reference view fails after the attempted binding is traced and charged.
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, [0]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let scope = ProvenanceScope::new("select element");
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let index = references[1].read()?;
            let read = ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
            let value = context.invoke_with_provenance_scope(scope.clone(), || {
                Ok::<_, ProgramError>(context.bind(read, vec![], &[references[0].clone(), index])?.remove(0))
            })?;
            references[2].write(&value)
        })
        .unwrap();
        let inputs = vec![Array::vector(Vec::<f32>::new()).unwrap(), Array::scalar(0i64).unwrap()];
        let expected = ProgramError::from(TypeError::invalid("cannot dynamically index an empty reference axis"));
        assert_eq!(definition.interpret(inputs.clone(), 1), Err(expected.clone()));
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(inputs.clone(), &KernelDebugOptions::default(), &mut trace),
            Err(expected),
        );
        assert_eq!(
            trace
                .iter()
                .map(|entry| (entry.coordinate().to_vec(), entry.operation(), entry.provenance().clone()))
                .collect::<Vec<_>>(),
            vec![
                (vec![], "reference_read", Provenance::unknown()),
                (vec![], "reference_read", Provenance::scope(scope.clone(), Provenance::unknown())),
            ],
        );

        // The body entry, the first read, and the failing read each consume one replay step.
        let error = definition.interpret_with_limits(inputs, 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_read",
                provenance: Provenance::scope(scope, Provenance::unknown()),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_qualifies_operations_bound_by_rules() {
        // An operation that a rule binds through its interpretation driver is qualified like the instructions of the
        // body, so it is traced, it consumes a replay step, and its outputs undergo the NaN check.
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<TestExtension>::trace(call, |(references, _coordinates)| {
            let input = references[0].read()?;
            let output = references[0]
                .context()
                .bind(KernelOperation::Extension(TestExtension::Rebinding), vec![], &[input])?
                .remove(0);
            references[1].write(&output)
        })
        .unwrap();
        let mut trace = Vec::new();
        let options = KernelDebugOptions::default();
        assert_eq!(
            definition.interpret_with_trace(vec![Array::scalar(3.0f32).unwrap()], &options, &mut trace),
            Ok(vec![Array::scalar(0.0f32).unwrap()]),
        );
        assert_eq!(
            trace.iter().map(|entry| entry.operation()).collect::<Vec<_>>(),
            vec!["reference_read", "test_rebinding", "sub", "reference_write"],
        );

        // The body entry and every traced binding consume one replay step each, so a budget of three steps, which
        // admits the body entry, the read, and the extension, stops at the nested `sub`.
        let error = definition.interpret_with_limits(vec![Array::scalar(3.0f32).unwrap()], 1, 3).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 3,
                coordinate: vec![],
                operation: "sub",
                provenance: Provenance::unknown(),
            }),
        );

        // `inf - inf` is NaN, which the NaN check reports at the nested `sub`.
        let options = KernelDebugOptions::default().with_nan_checks(true);
        let inputs = vec![Array::scalar(f32::INFINITY).unwrap()];
        let error = definition.interpret_with_trace(inputs, &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::Nan {
                operation: "sub",
                position: "output",
                value: 0,
                element: 0,
                data_type: DataType::F32,
                coordinate: vec![],
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_malformed_rule_binding() {
        // Binds a `reference_read` without its reference input through its interpretation driver, which fails the
        // validation of its reference access layout.
        #[derive(Clone, Debug)]
        struct MalformedExtension;

        impl Operation for MalformedExtension {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "test_malformed"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(vec![])
            }
        }

        impl ReferenceAccessOperation for MalformedExtension {
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
                Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference accesses".to_owned(),
                })
            }
        }

        impl KernelExtension for MalformedExtension {
            fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
                Ok(KernelExtensionMemory::Synchronous)
            }
        }

        impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<MalformedExtension>>>
            for MalformedExtension
        {
            fn interpret<
                D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<MalformedExtension>>>,
            >(
                &self,
                context: &EagerContext<ArrayIrValue<Array>, KernelOperation<MalformedExtension>>,
                driver: &D,
                _inputs: &[ArrayIrValue<Array>],
            ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
                let operation = KernelOperation::Portable(ArrayIrOperation::from(ReferenceReadOperation::new()));
                driver.bind(context, operation, AttachedRegions::Empty, &[])
            }
        }

        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<MalformedExtension>>::new();
        builder
            .add_instruction(KernelOperation::Extension(MalformedExtension), vec![], vec![], None)
            .unwrap();
        let body = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![], &KernelDebugOptions::default(), &mut trace),
            Err(ProgramError::MalformedProgram(
                "operation `reference_read` declares reference access at input 0 but has only 0 inputs".to_owned(),
            )),
        );
        assert_eq!(
            trace.iter().map(KernelTraceEntry::operation).collect::<Vec<_>>(),
            vec!["test_malformed", "reference_read"],
        );
    }

    #[test]
    fn test_kernel_call_operation_interpret() {
        let definition = definition(true, true);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                inputs[0].context().bind(definition.operation().clone(), vec![definition.body().clone()], &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        let inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![5i32, 6, 7, 8]).unwrap()),
        ];
        let expected = vec![ArrayIrValue::Array(Array::vector(vec![6i32, 8, 10, 12]).unwrap())];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        let (_, replayed) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert_eq!(replayed.to_string(), program.to_string());
        assert_eq!(replayed.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_qualified_kernel_context_bind_async_completion() {
        // Construct the private context directly to verify the runtime defense independently of static rejection.
        let context = QualifiedKernelContext::<NoKernelExtension> {
            eager: EagerContext::new(),
            validity: BTreeMap::new(),
            references: Rc::new(definition(false, false).references().clone()),
            instruction: Rc::new(Cell::new(None)),
            coordinate: vec![],
            maximum_steps: 10,
            remaining_steps: Rc::new(Cell::new(10)),
            provenance: Rc::new(ProvenanceState::new()),
            trace: None,
            pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
        };
        let source = ArrayReference::new(Array::vector(vec![11i32, 13]).unwrap());
        let destination = ArrayReference::new(Array::vector(vec![0i32, 0]).unwrap());
        let token = context
            .bind(
                AsyncCopyOperation::new(),
                vec![],
                &[ArrayIrValue::Reference(source), ArrayIrValue::Reference(destination.clone())],
            )
            .unwrap()
            .remove(0);
        assert_eq!(destination.read(), Ok(Array::vector(vec![0i32, 0]).unwrap()));
        assert_eq!(context.bind(WaitOperation, vec![], std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(destination.read(), Ok(Array::vector(vec![11i32, 13]).unwrap()));
        let ArrayIrValue::Reference(reference) = &token else { unreachable!() };
        let error = context.bind(WaitOperation, vec![], std::slice::from_ref(&token)).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::UnknownCopyToken { token: reference.id() }),
        );
        assert_eq!(error.to_string(), "async copy completion token is unknown or already consumed");
    }

    #[test]
    fn test_qualified_kernel_context_bind_async_completion_transforms() {
        let context = QualifiedKernelContext::<NoKernelExtension> {
            eager: EagerContext::new(),
            validity: BTreeMap::new(),
            references: Rc::new(definition(false, false).references().clone()),
            instruction: Rc::new(Cell::new(None)),
            coordinate: vec![],
            maximum_steps: 10,
            remaining_steps: Rc::new(Cell::new(10)),
            provenance: Rc::new(ProvenanceState::new()),
            trace: None,
            pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
        };
        let source = ArrayReference::new(Array::matrix(2, 2, vec![1i32, 2, 11, 13]).unwrap());
        let destination = ArrayReference::new(Array::matrix(3, 2, vec![0i32; 6]).unwrap());
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }])
            .with_destination_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
        let token = context
            .bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(source),
                    ArrayIrValue::Reference(destination.clone()),
                    ArrayIrValue::Array(Array::scalar(1i64).unwrap()),
                    ArrayIrValue::Array(Array::scalar(2i64).unwrap()),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(destination.read(), Ok(Array::matrix(3, 2, vec![0i32; 6]).unwrap()));
        assert_eq!(context.bind(WaitOperation, vec![], &[token]), Ok(vec![]));
        assert_eq!(destination.read(), Ok(Array::matrix(3, 2, vec![0i32, 0, 0, 0, 11, 13]).unwrap()));
    }

    #[test]
    fn test_qualified_kernel_context_bind_completion_protocols() {
        // Construct the private context directly to verify the runtime defense independently of qualification, which
        // rejects the same pairings before replay. A commit or wait of another protocol leaves its token pending.
        let context = QualifiedKernelContext::<TestExtension> {
            eager: EagerContext::new(),
            validity: BTreeMap::new(),
            references: Rc::new(definition(false, false).references().clone()),
            instruction: Rc::new(Cell::new(None)),
            coordinate: vec![],
            maximum_steps: 100,
            remaining_steps: Rc::new(Cell::new(100)),
            provenance: Rc::new(ProvenanceState::new()),
            trace: None,
            pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
        };
        let bind = |operation: KernelOperation<TestExtension>, inputs: &[ArrayIrValue<Array>]| {
            context.bind(operation, vec![], inputs)
        };
        let error = |operation: KernelOperation<TestExtension>, token: &ArrayIrValue<Array>| {
            let error = bind(operation, std::slice::from_ref(token)).unwrap_err();
            error.downcast_custom::<KernelInterpretationError>().cloned()
        };
        let incompatible = |operation: &'static str, token: &ArrayIrValue<Array>| {
            let ArrayIrValue::Reference(token) = token else { unreachable!() };
            Some(KernelInterpretationError::IncompatibleCompletion { operation, token: token.id() })
        };
        let source = ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![3i32, 5]).unwrap()));
        let destination = ArrayReference::new(Array::vector(vec![0i32, 0]).unwrap());
        let wait = || KernelOperation::<TestExtension>::Wait(WaitOperation);
        let commit = |strict| KernelOperation::Extension(TestExtension::Commit { strict });
        let extension_wait = |strict| KernelOperation::Extension(TestExtension::Wait { strict });

        // An extension commit or wait cannot complete a portable copy, which the portable wait still completes.
        let token = bind(
            KernelOperation::AsyncCopy(AsyncCopyOperation::new()),
            &[source.clone(), ArrayIrValue::Reference(destination.clone())],
        )
        .unwrap()
        .remove(0);
        assert_eq!(error(commit(false), &token), incompatible("test_commit", &token));
        assert_eq!(error(extension_wait(false), &token), incompatible("test_wait", &token));
        assert_eq!(destination.read(), Ok(Array::vector(vec![0i32, 0]).unwrap()));
        assert_eq!(bind(wait(), std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(destination.read(), Ok(Array::vector(vec![3i32, 5]).unwrap()));

        // The portable wait cannot complete an extension token, which the extension wait still completes after any
        // number of commits, and a token cannot be completed again after its wait.
        let update = |strict| {
            let inputs = [source.clone(), source.clone(), ArrayIrValue::Reference(destination.clone())];
            bind(KernelOperation::Extension(TestExtension::Update { strict }), &inputs).unwrap().remove(0)
        };
        let token = update(false);
        assert_eq!(error(wait(), &token), incompatible("wait", &token));
        assert_eq!(bind(commit(false), std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(bind(commit(false), std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(bind(extension_wait(false), std::slice::from_ref(&token)), Ok(vec![]));
        let ArrayIrValue::Reference(reference) = &token else { unreachable!() };
        assert_eq!(
            error(extension_wait(false), &token),
            Some(KernelInterpretationError::UnknownCopyToken { token: reference.id() }),
        );

        // The strict protocol requires exactly one commit of its own protocol before the wait.
        let token = update(true);
        assert_eq!(error(extension_wait(true), &token), incompatible("test_wait", &token));
        assert_eq!(error(commit(false), &token), incompatible("test_commit", &token));
        assert_eq!(bind(commit(true), std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(error(commit(true), &token), incompatible("test_commit", &token));
        assert_eq!(error(extension_wait(false), &token), incompatible("test_wait", &token));
        assert_eq!(context.pending_copies.borrow().len(), 1);
        assert_eq!(bind(extension_wait(true), std::slice::from_ref(&token)), Ok(vec![]));
        assert_eq!(context.pending_copies.borrow().len(), 0);
    }

    #[test]
    fn test_qualified_kernel_context_bind_masked_views_project_validity() {
        let reference = ArrayReference::new(Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap());
        let validity = ArrayReference::new(Array::matrix(2, 2, vec![false, false, true, false]).unwrap());
        let context = QualifiedKernelContext::<NoKernelExtension> {
            eager: EagerContext::new(),
            validity: BTreeMap::from([(reference.id(), validity)]),
            references: Rc::new(definition(false, false).references().clone()),
            instruction: Rc::new(Cell::new(None)),
            coordinate: vec![],
            maximum_steps: 10,
            remaining_steps: Rc::new(Cell::new(10)),
            provenance: Rc::new(ProvenanceState::new()),
            trace: None,
            pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
        };
        let operation = MaskedLoadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Static(1),
        }]);
        assert_eq!(
            context.bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(reference),
                    ArrayIrValue::Array(Array::vector(vec![true, true]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![-1i32, -2]).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![3i32, -2]).unwrap())]),
        );
    }
}
