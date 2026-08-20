//! Static root, access, scope, and lifetime analysis for array references.
//!
//! [`ReferenceAnalysis`] is the validated program-level counterpart of operation-local
//! [`ReferenceOperationSemantics`](crate::programs::ReferenceOperationSemantics). Operation descriptors speak only in
//! input/output indices; this module resolves those indices to canonical region-relative roots, validates second-class
//! boundaries and consumption, and records explicit substitutions at nested-region call sites. Runtime holder identity
//! remains a separate invocation concern.

// TODO(eaplatanios): Review this module.
//  Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Display;

use serde::Serialize;
use thiserror::Error;

use crate::arrays::types::ir::ArrayIrType;
use crate::captures::CaptureConstant;
use crate::parameters::Parameterized;
use crate::programs::regions::reachable_region_mask;
use crate::programs::{
    Atom, AtomId, Effect, InstructionId, Operation, Program, ProgramError, ReferenceAccessMode,
    ReferenceOutputSemantics, RegionId, RegionRef, RegionRole, Type, Typed, Value, ValueId,
};

/// Static reference-analysis failure.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceAnalysisError {
    /// The declared lifted-capture prefix exceeds the entry input boundary.
    #[error("reference capture count {capture_count} exceeds entry input count {input_count}")]
    InvalidCaptureCount {
        /// Declared number of lifted captures.
        capture_count: usize,

        /// Actual entry input count.
        input_count: usize,
    },

    /// A reference-typed constant would hide runtime resource identity inside program storage.
    #[error("reference-typed constant `{atom}` in region `{region}` is not supported")]
    ReferenceConstant {
        /// Region containing the constant.
        region: RegionId,

        /// Constant atom.
        atom: AtomId,
    },

    /// A reference-typed constant in an attached region does not name a capture in its active lexical scope.
    #[error(
        "reference-typed capture constant `{atom}` in region `{region}` names capture {capture_index}, but its active \
         scope has {scope_input_count} addressable inputs"
    )]
    InvalidReferenceCapture {
        /// Region containing the constant.
        region: RegionId,

        /// Constant atom.
        atom: AtomId,

        /// Capture-table index stored by the constant.
        capture_index: usize,

        /// Number of addressable inputs in the active capture scope.
        scope_input_count: usize,
    },

    /// An operation declares a capture prefix longer than the attached region's input boundary.
    #[error(
        "operation `{operation}` at `{instruction}` declares {capture_count} capture inputs for attached region \
         #{region_index}, but that region has only {input_count} inputs"
    )]
    InvalidRegionCaptureCount {
        /// Instruction carrying the invalid attached-region scope.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region slot index.
        region_index: usize,

        /// Declared capture-prefix length.
        capture_count: usize,

        /// Attached region input count.
        input_count: usize,
    },

    /// A shared region containing reference capture constants is reachable from distinct lexical capture scopes.
    #[error(
        "reference capture constants in shared region `{region}` have ambiguous lexical scopes `{first_scope}` \
         ({first_scope_input_count} inputs) and `{second_scope}` ({second_scope_input_count} inputs)"
    )]
    AmbiguousReferenceCaptureScope {
        /// Shared region containing a capture constant.
        region: RegionId,

        /// First enclosing capture-scope root.
        first_scope: RegionId,

        /// Addressable input count of the first enclosing scope.
        first_scope_input_count: usize,

        /// Second enclosing capture-scope root.
        second_scope: RegionId,

        /// Addressable input count of the second enclosing scope.
        second_scope_input_count: usize,
    },

    /// A reference-typed attached-region capture does not match its lexical-scope input type.
    #[error(
        "reference-typed capture constant `{atom}` in region `{region}` has type `{capture_type}`, but scoped capture \
         {capture_index} has type `{input_type}`"
    )]
    ReferenceCaptureTypeMismatch {
        /// Region containing the constant.
        region: RegionId,

        /// Constant atom.
        atom: AtomId,

        /// Capture-table index stored by the constant.
        capture_index: usize,

        /// Constant's declared type.
        capture_type: String,

        /// Active lexical-scope input type.
        input_type: String,
    },

    /// Two capture constants in one region create distinct handles to the same lifted root.
    #[error(
        "reference capture constants `{first_atom}` and `{second_atom}` in region `{region}` both resolve to `{root}`"
    )]
    DuplicateReferenceCaptureAlias {
        /// Region containing both aliases.
        region: RegionId,

        /// First capture constant.
        first_atom: AtomId,

        /// Later capture constant.
        second_atom: AtomId,

        /// Duplicated lifted root.
        root: ReferenceRoot,
    },

    /// References are second-class and cannot cross a region's result boundary.
    #[error("reference output {output_index} of region `{region}` exposes root `{root}`")]
    ReferenceOutput {
        /// Region exposing the reference.
        region: RegionId,

        /// Output boundary position.
        output_index: usize,

        /// Root escaping through the boundary.
        root: ReferenceRoot,
    },

    /// Stateful reference semantics are missing the conservative ordered-state effect.
    #[error("reference semantics for `{operation}` at `{instruction}` require `OrderedState`")]
    MissingOrderedState {
        /// Instruction carrying the inconsistent semantics.
        instruction: InstructionId,

        /// Operation name.
        operation: String,
    },

    /// An operation descriptor names an input outside the instruction arity.
    #[error(
        "reference semantics for `{operation}` at `{instruction}` name input {input_index}, but the instruction has \
         {input_count} inputs"
    )]
    InvalidAccessIndex {
        /// Instruction carrying the invalid descriptor.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Invalid input position.
        input_index: usize,

        /// Actual instruction input count.
        input_count: usize,
    },

    /// An operation descriptor names an output outside the instruction arity.
    #[error(
        "reference semantics for `{operation}` at `{instruction}` name output {output_index}, but the instruction has \
         {output_count} outputs"
    )]
    InvalidOutputIndex {
        /// Instruction carrying the invalid descriptor.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Invalid output position.
        output_index: usize,

        /// Actual instruction output count.
        output_count: usize,
    },

    /// A descriptor access points at a non-reference operand.
    #[error(
        "reference semantics for `{operation}` at `{instruction}` classify non-reference input {input_index} of type \
         `{input_type}`"
    )]
    NonReferenceAccess {
        /// Instruction carrying the invalid descriptor.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Classified input position.
        input_index: usize,

        /// Actual input type.
        input_type: String,
    },

    /// A descriptor output classification points at a non-reference result.
    #[error(
        "reference semantics for `{operation}` at `{instruction}` classify non-reference output {output_index} of \
         type `{output_type}`"
    )]
    NonReferenceOutput {
        /// Instruction carrying the invalid descriptor.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Classified output position.
        output_index: usize,

        /// Actual output type.
        output_type: String,
    },

    /// A reference operand has no declared access or nested-region forwarding semantics.
    #[error(
        "reference input {input_index} of `{operation}` at `{instruction}` has no declared access or region-input \
         provenance"
    )]
    UnclassifiedInput {
        /// Instruction with the unclassified input.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Unclassified input position.
        input_index: usize,
    },

    /// A reference result has no root or alias classification.
    #[error("reference output {output_index} of `{operation}` at `{instruction}` has no root or alias classification")]
    UnclassifiedOutput {
        /// Instruction with the unclassified output.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Unclassified output position.
        output_index: usize,
    },

    /// An alias source has not resolved to one root.
    #[error("reference alias input {input_index} of `{operation}` at `{instruction}` does not resolve to a root")]
    UnresolvedAlias {
        /// Instruction defining the unresolved alias.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Alias-source input position.
        input_index: usize,
    },

    /// An alias changes its referent type.
    #[error(
        "reference alias output {output_index} of `{operation}` at `{instruction}` has type `{output_type}`, but its \
         source has type `{input_type}`"
    )]
    AliasTypeMismatch {
        /// Instruction defining the invalid alias.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Alias output position.
        output_index: usize,

        /// Source reference type.
        input_type: String,

        /// Alias result type.
        output_type: String,
    },

    /// A root or one of its aliases is used after consumption.
    #[error("reference input {input_index} of `{operation}` at `{instruction}` uses consumed root `{root}`")]
    UseAfterConsume {
        /// Instruction making the invalid use.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Input position making the invalid use.
        input_index: usize,

        /// Consumed root.
        root: ReferenceRoot,
    },

    /// Freeze/consume is restricted to the local allocation's defining handle.
    #[error(
        "`{operation}` at `{instruction}` can consume only a local root handle, but input {input_index} resolves to \
         `{root}`"
    )]
    InvalidConsume {
        /// Consuming instruction.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Consuming input position.
        input_index: usize,

        /// Root reached by the handle.
        root: ReferenceRoot,
    },

    /// A reference region input has no explicit direct source in the enclosing operation.
    #[error(
        "reference input {input_index} of attached region {region_index} on `{operation}` at `{instruction}` has no \
         direct parent-input provenance"
    )]
    UnsupportedRegionInput {
        /// Parent instruction attaching the region.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region position.
        region_index: usize,

        /// Input position in the attached region.
        input_index: usize,
    },

    /// A declared nested-region source is outside the parent instruction arity or not a reference.
    #[error(
        "reference input {input_index} of attached region {region_index} on `{operation}` at `{instruction}` maps to \
         invalid parent input {source_input_index}"
    )]
    InvalidRegionInputSource {
        /// Parent instruction attaching the region.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region position.
        region_index: usize,

        /// Input position in the attached region.
        input_index: usize,

        /// Invalid parent operand position.
        source_input_index: usize,
    },

    /// Two formal reference inputs in one region invocation resolve to the same caller root.
    #[error(
        "reference inputs {first_input_index} and {second_input_index} of attached region {region_index} on \
         `{operation}` at `{instruction}` both resolve to `{root}`"
    )]
    DuplicateRegionInputAlias {
        /// Parent instruction attaching the region.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region position.
        region_index: usize,

        /// First formal region input resolving to the root.
        first_input_index: usize,

        /// Later formal region input resolving to the same root.
        second_input_index: usize,

        /// Duplicated canonical caller root.
        root: ReferenceRoot,
    },

    /// A formal region input and a lexical capture resolve to the same caller root.
    #[error(
        "reference input {input_index} of attached region {region_index} on `{operation}` at `{instruction}` and \
         capture constant `{capture_atom}` in region `{capture_region}` both resolve to `{root}`"
    )]
    DuplicateRegionCaptureAlias {
        /// Parent instruction attaching the region.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region position.
        region_index: usize,

        /// Formal region input resolving to the root.
        input_index: usize,

        /// Region containing the colliding capture constant.
        capture_region: RegionId,

        /// Colliding capture constant.
        capture_atom: AtomId,

        /// Duplicated canonical caller root.
        root: ReferenceRoot,
    },

    /// Alternative nested result paths resolve one parent reference output to different roots.
    #[error(
        "reference output {output_index} of `{operation}` at `{instruction}` resolves to both `{first_root}` and \
         `{second_root}`"
    )]
    InconsistentRegionOutputRoots {
        /// Parent instruction receiving the inconsistent result.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Parent result position.
        output_index: usize,

        /// Root resolved from the first nested result path.
        first_root: ReferenceRoot,

        /// Root resolved from a later nested result path.
        second_root: ReferenceRoot,
    },

    /// A declared attached-region result source is absent, out of range, or not a resolved reference.
    #[error(
        "reference output {output_index} of `{operation}` at `{instruction}` has invalid attached-region source \
         {region_index} output {region_output_index}"
    )]
    InvalidRegionOutputSource {
        /// Parent instruction receiving the invalid source.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Parent reference output position.
        output_index: usize,

        /// Attached-region position named by the provenance.
        region_index: usize,

        /// Output position named within the attached region.
        region_output_index: usize,
    },

    /// Two structured outputs create parallel implicit handles to the same canonical root.
    #[error(
        "reference outputs {first_output_index} and {second_output_index} of `{operation}` at `{instruction}` both \
         forward root `{root}` without explicit alias semantics"
    )]
    DuplicateRegionOutputAlias {
        /// Parent instruction defining both outputs.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// First forwarded output position.
        first_output_index: usize,

        /// Later forwarded output position.
        second_output_index: usize,

        /// Duplicated canonical root.
        root: ReferenceRoot,
    },

    /// A loop carry result resolves to a different root than its zero-iteration input.
    #[error(
        "reference output {output_index} of `{operation}` at `{instruction}` resolves to `{output_root}`, but \
         fixed-point input {input_index} resolves to `{input_root}`"
    )]
    ReferenceCarryRootMismatch {
        /// Loop instruction defining the invalid carry result.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Reference output position.
        output_index: usize,

        /// Required zero-iteration input position.
        input_index: usize,

        /// Canonical entering root.
        input_root: ReferenceRoot,

        /// Canonical body result root.
        output_root: ReferenceRoot,
    },

    /// An attached-region access violates the owning operation's entering-reference policy.
    #[error(
        "reference root `{root}` entering region {region_index} of `{operation}` at `{instruction}` cannot be accessed \
         with mode `{mode:?}`"
    )]
    ForbiddenRegionInputAccess {
        /// Parent instruction attaching the region.
        instruction: InstructionId,

        /// Operation name.
        operation: String,

        /// Attached-region position.
        region_index: usize,

        /// Canonical caller root entering the region.
        root: ReferenceRoot,

        /// Forbidden access mode.
        mode: ReferenceAccessMode,
    },

    /// Reference state is not allowed in dormant transformation-rule regions.
    #[error("reference state in dormant rule region `{region}` is not supported")]
    RuleRegion {
        /// Dormant rule region containing reference state.
        region: RegionId,
    },
}

/// Canonical static root of a reference inside one source program.
///
/// Region inputs are parameterized roots: a nested region can be shared by several call sites, so its input denotes
/// whichever caller root [`ReferenceAnalysis::region_root_for_source`] records for the exact invocation. Allocations,
/// by contrast, are concrete program-local roots identified by their defining instruction and output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ReferenceRoot {
    /// Reference value entering one region invocation.
    RegionInput {
        /// Region owning the input.
        region: RegionId,

        /// Input position in the region boundary.
        input_index: usize,
    },

    /// Fresh root allocated by an instruction output.
    Allocation {
        /// Allocation instruction.
        instruction: InstructionId,

        /// Output position defining the root.
        output_index: usize,
    },
}

impl Display for ReferenceRoot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegionInput { region, input_index } => write!(formatter, "{region} input {input_index}"),
            Self::Allocation { instruction, output_index } => {
                write!(formatter, "{instruction} output {output_index}")
            }
        }
    }
}

/// Invocation source of one external reference root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReferenceSource {
    /// Capture lifted into the entry boundary before public arguments.
    Capture {
        /// Zero-based capture position in the lifted capture prefix.
        index: usize,
    },

    /// Public reference argument after the lifted capture prefix.
    PublicInput {
        /// Zero-based public input position, excluding lifted captures.
        index: usize,
    },
}

impl Display for ReferenceSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::PublicInput { index } => write!(formatter, "public input {index}"),
        }
    }
}

/// One external root expected at invocation time, in deterministic entry-boundary order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalReferenceRoot {
    /// Canonical root in the entry region.
    root: ReferenceRoot,

    /// Capture or public-input source supplying the runtime holder.
    source: ReferenceSource,
}

impl ExternalReferenceRoot {
    /// Returns the canonical entry-region root.
    #[inline]
    pub const fn root(&self) -> ReferenceRoot {
        self.root
    }

    /// Returns the invocation source supplying this root.
    #[inline]
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }
}

/// One validated operation-local reference access resolved to a canonical region-relative root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAccess {
    /// Instruction performing the access.
    instruction: InstructionId,

    /// Reference operand position on that instruction.
    input_index: usize,

    /// Canonical root resolved for the operand.
    root: ReferenceRoot,

    /// Declared semantic access mode.
    mode: ReferenceAccessMode,
}

impl ReferenceAccess {
    /// Returns the instruction performing this access.
    #[inline]
    pub const fn instruction(&self) -> InstructionId {
        self.instruction
    }

    /// Returns the reference operand position on the instruction.
    #[inline]
    pub const fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the canonical region-relative root accessed by the operand.
    #[inline]
    pub const fn root(&self) -> ReferenceRoot {
        self.root
    }

    /// Returns the semantic access mode.
    #[inline]
    pub const fn mode(&self) -> ReferenceAccessMode {
        self.mode
    }
}

/// One transitive may-access fact projected through nested region boundaries into an enclosing root namespace.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceTransitiveAccess {
    /// Canonical root after substitution into the summary owner's region.
    root: ReferenceRoot,

    /// Semantic access mode of the underlying operation-local access.
    mode: ReferenceAccessMode,
}

impl ReferenceTransitiveAccess {
    /// Returns the canonical root in the summary owner's region.
    #[inline]
    pub const fn root(&self) -> ReferenceRoot {
        self.root
    }

    /// Returns the semantic access mode.
    #[inline]
    pub const fn mode(&self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Substitution from a nested region's parameterized root to one caller root. The invoking instruction and attached
/// region are the analysis lookup keys and are not repeated here; the public
/// [`ReferenceAnalysis::region_root_for_source`] query resolves one substitution by its full invocation identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ReferenceRegionInputBinding {
    /// Parameterized root inside the attached region.
    region_root: ReferenceRoot,

    /// Parent instruction operand that supplies the root.
    source_input_index: usize,

    /// Canonical root resolved in the parent region.
    source_root: ReferenceRoot,
}

/// Analysis-local handle state associated with one reference-typed atom.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ReferenceHandle {
    /// Canonical region-relative resource root.
    root: ReferenceRoot,

    /// Whether this exact value is an unaliased root handle rather than a derived alias.
    is_root: bool,
}

/// Fully validated static reference analysis for one rooted region closure in a source arena.
///
/// Accesses, bindings, and external roots retain deterministic arena, instruction, and boundary order.
/// [`RegionId`], [`InstructionId`], and [`ValueId`] values in this artifact are meaningful only against the exact
/// source arena and root that were analyzed; the artifact does not retain or borrow either one.
#[derive(Debug)]
pub struct ReferenceAnalysis {
    /// Canonical root resolved for each reference atom, indexed by region then atom.
    roots: Vec<Vec<Option<ReferenceRoot>>>,

    /// Validated accesses in arena and instruction order.
    accesses: Vec<ReferenceAccess>,

    /// Nested-region root substitutions in caller instruction order.
    bindings: Vec<ReferenceRegionInputBinding>,

    /// Indexed lookup from one caller root to its formal root for a specific attached-region invocation.
    source_binding_indices: BTreeMap<(InstructionId, usize, ReferenceRoot), usize>,

    /// External roots in entry-boundary order.
    external_roots: Vec<ExternalReferenceRoot>,

    /// Transitive access summaries in arena and instruction order.
    instruction_summaries: Vec<Vec<ReferenceTransitiveAccess>>,

    /// Sparse lookup from source instruction position to [`instruction_summaries`](Self::instruction_summaries).
    instruction_summary_indices: Vec<Vec<Option<usize>>>,

    /// Transitive outward accesses indexed by source region.
    region_summaries: Vec<Vec<ReferenceTransitiveAccess>>,
}

impl ReferenceAnalysis {
    /// Returns the root resolved for `value`, or [`None`] when it is not reference-typed or lies outside the analyzed
    /// rooted closure.
    #[inline]
    pub fn root(&self, value: ValueId) -> Option<ReferenceRoot> {
        self.roots.get(value.region().index())?.get(value.atom().index()).copied().flatten()
    }

    /// Returns validated reference accesses in deterministic arena and instruction order.
    #[inline]
    pub fn accesses(&self) -> &[ReferenceAccess] {
        &self.accesses
    }

    /// Returns the formal attached-region root supplied by `source_root` at the attached-region invocation identified
    /// by `instruction` and `region_index`, or [`None`] when that exact invocation records no binding for the root.
    #[inline]
    pub fn region_root_for_source(
        &self,
        instruction: InstructionId,
        region_index: usize,
        source_root: ReferenceRoot,
    ) -> Option<ReferenceRoot> {
        self.source_binding_indices
            .get(&(instruction, region_index, source_root))
            .map(|index| self.bindings[*index].region_root)
    }

    /// Returns external roots in entry-boundary order.
    #[inline]
    pub fn external_roots(&self) -> &[ExternalReferenceRoot] {
        &self.external_roots
    }

    /// Returns the transitive may-access facts for `instruction`, with each access root resolved in the containing
    /// instruction's region namespace so discharge can widen a nested boundary without repeating root substitution.
    /// Returns [`None`] when the instruction has no reference access, lies outside the analyzed closure, or belongs
    /// to a reference-free closure handled by the empty-analysis fast path.
    pub fn instruction_summary(&self, instruction: InstructionId) -> Option<&[ReferenceTransitiveAccess]> {
        let summary = *self.instruction_summary_indices.get(instruction.region().index())?.get(instruction.index())?;
        summary.map(|index| self.instruction_summaries[index].as_slice())
    }

    /// Returns the transitive accesses that reach roots entering `region`, with every nested formal root substituted
    /// into `region`'s namespace. Local allocations are deliberately absent because they do not cross the boundary.
    /// A region without entering reference accesses — including one outside the analyzed closure or in a
    /// reference-free closure whose arena-sized tables exist — yields an empty slice; [`None`] is returned only for a
    /// region id outside the arena or under the empty-analysis fast path.
    pub(crate) fn region_summary(&self, region: RegionId) -> Option<&[ReferenceTransitiveAccess]> {
        self.region_summaries.get(region.index()).map(Vec::as_slice)
    }

    /// Returns whether the analyzed closure contains no reference-typed atoms and no reference-semantics operations,
    /// which is exactly the case handled by the empty-analysis fast path.
    #[inline]
    pub(crate) fn is_reference_free(&self) -> bool {
        self.roots.is_empty()
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Resolves and validates all reference roots and accesses in this program. Refer to the documentation of
    /// [`RegionRef::analyze_references`] for the analysis semantics; the program's entry region is the analysis root.
    #[inline]
    pub fn analyze_references(&self, capture_count: usize) -> Result<ReferenceAnalysis, ProgramError> {
        self.entry_region_ref().analyze_references(capture_count)
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: CaptureConstant<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Resolves references after entry captures have been lifted into the first `capture_count` inputs while allowing
    /// attached-region constants to retain lexical capture indices. Constants in ordinary nested control flow resolve
    /// to the corresponding entry root; constants in a call-owned capture scope resolve to that callee's leading
    /// inputs. Immediate reference constants and constants in the analyzed root region remain invalid.
    pub fn analyze_references_with_lifted_captures(
        &self,
        capture_count: usize,
    ) -> Result<ReferenceAnalysis, ProgramError> {
        self.entry_region_ref()
            .analyze_references_with_capture_indices(capture_count, CaptureConstant::capture_index)
    }
}

impl<V, O> RegionRef<'_, V, O>
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>,
{
    /// Resolves and validates all reference roots and accesses in this region's complete attached-region closure.
    ///
    /// Analysis is defined after capture lifting: `capture_count` root-region inputs form the capture prefix, and
    /// remaining root-region inputs are public arguments. Sibling arena regions outside this closure are ignored: this
    /// root can never reach them, so their reference state must not affect its legality. The returned artifact is
    /// arena-relative: its identifiers must be consumed only with this exact source arena.
    pub fn analyze_references(self, capture_count: usize) -> Result<ReferenceAnalysis, ProgramError> {
        self.analyze_references_with_capture_indices(capture_count, |_| None)
    }

    /// Implements reference analysis with an optional capture-table index projection for constants.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading root-region inputs that form the lifted-capture prefix.
    ///   - `capture_index`: Projects one constant value onto its capture-table index. Returning [`Some`] classifies a
    ///     reference-typed constant as a scoped capture handle resolved against the active lifted entry or
    ///     nested-call capture scope; returning [`None`] makes any reference-typed constant a hard
    ///     [`ReferenceAnalysisError::ReferenceConstant`] error, which is correct for ordinary value families.
    fn analyze_references_with_capture_indices<F>(
        self,
        capture_count: usize,
        capture_index: F,
    ) -> Result<ReferenceAnalysis, ProgramError>
    where
        F: Fn(&V) -> Option<usize>,
    {
        if capture_count > self.input_ids().len() {
            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidCaptureCount {
                capture_count,
                input_count: self.input_ids().len(),
            }));
        }

        let regions = self.arena();
        let included = reachable_region_mask(regions.len(), [self.id()], |id| &regions[id.index()]);

        let requires_reference_analysis = regions.iter().enumerate().any(|(region_index, region)| {
            included[region_index]
                && (region.atoms().iter().any(|atom| atom.r#type().is_reference())
                    || region
                        .instructions()
                        .iter()
                        .any(|instruction| !instruction.operation().reference_semantics().is_empty()))
        });
        if !requires_reference_analysis {
            return Ok(ReferenceAnalysis {
                roots: Vec::new(),
                accesses: Vec::new(),
                bindings: Vec::new(),
                source_binding_indices: BTreeMap::new(),
                external_roots: Vec::new(),
                instruction_summaries: Vec::new(),
                instruction_summary_indices: Vec::new(),
                region_summaries: Vec::new(),
            });
        }

        let entry = self.id();
        // Capture indices inherit the current lexical scope through ordinary control flow. A call-like operation can
        // establish a fresh scope whose leading attached-region inputs are that callee's lifted captures. Retaining
        // every reachable scope detects ambiguous sharing without path expansion.
        let mut capture_scopes = vec![BTreeSet::<(RegionId, usize)>::new(); regions.len()];
        let mut pending_capture_scopes = vec![(entry, (entry, capture_count))];
        while let Some((region, scope)) = pending_capture_scopes.pop() {
            if !capture_scopes[region.index()].insert(scope) {
                continue;
            }
            for (instruction_index, instruction) in regions[region.index()].instructions().iter().enumerate() {
                for (region_index, attached) in instruction.regions().iter().copied().enumerate() {
                    let attached_scope = match instruction.operation().region_capture_input_count(region_index) {
                        Some(capture_count) => {
                            let input_count = regions[attached.index()].input_ids().len();
                            if capture_count > input_count {
                                return Err(ProgramError::custom(ReferenceAnalysisError::InvalidRegionCaptureCount {
                                    instruction: InstructionId::new(region, instruction_index),
                                    operation: instruction.operation().name().to_string(),
                                    region_index,
                                    capture_count,
                                    input_count,
                                }));
                            }
                            (attached, capture_count)
                        }
                        None => scope,
                    };
                    pending_capture_scopes.push((attached, attached_scope));
                }
            }
        }
        let mut handles = regions.iter().map(|region| vec![None; region.atoms().len()]).collect::<Vec<_>>();
        let mut accesses = Vec::new();
        let mut bindings = Vec::new();
        let mut binding_indices = HashMap::new();
        let mut source_binding_indices = BTreeMap::new();
        let mut external_roots = Vec::new();
        let mut region_capture_roots = vec![Vec::<(ReferenceRoot, ValueId)>::new(); regions.len()];
        let mut pending_rule_regions = Vec::new();

        // Region roles are properties of incoming attachment edges. Conservatively reject a shared region if any
        // parent attaches it as a dormant rule, because ordinary reference analysis cannot make its legality depend on
        // which use site happened to invoke the shared body.
        for (region_index, region) in regions.iter().enumerate() {
            if !included[region_index] {
                continue;
            }
            for instruction in region.instructions() {
                for (slot_index, attached) in instruction.regions().iter().copied().enumerate() {
                    if instruction.operation().region_role(slot_index) == Some(RegionRole::Rule) {
                        pending_rule_regions.push(attached);
                    }
                }
            }
        }
        let rule_regions = reachable_region_mask(regions.len(), pending_rule_regions, |id| &regions[id.index()]);

        for (region_index, region) in regions.iter().enumerate() {
            if !included[region_index] {
                continue;
            }
            let region_id = RegionId::new(region_index);
            let mut live_roots = HashSet::new();

            for (input_index, atom) in region.input_ids().iter().copied().enumerate() {
                if region.atoms()[atom.index()].r#type().is_reference() {
                    let root = ReferenceRoot::RegionInput { region: region_id, input_index };
                    handles[region_index][atom.index()] = Some(ReferenceHandle { root, is_root: true });
                    live_roots.insert(root);
                }
            }

            for (atom_index, atom) in region.atoms().iter().enumerate() {
                let Atom::Constant(constant) = atom else {
                    continue;
                };
                let constant_type = constant.r#type();
                if !constant_type.is_reference() {
                    continue;
                }
                let atom = AtomId::new(atom_index);
                let Some(capture_index) = capture_index(constant) else {
                    return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceConstant {
                        region: region_id,
                        atom,
                    }));
                };
                let mut scopes = capture_scopes[region_index].iter().copied();
                let (scope, scope_input_count) = scopes.next().unwrap();
                if let Some((second_scope, second_scope_input_count)) = scopes.next() {
                    return Err(ProgramError::custom(ReferenceAnalysisError::AmbiguousReferenceCaptureScope {
                        region: region_id,
                        first_scope: scope,
                        first_scope_input_count: scope_input_count,
                        second_scope,
                        second_scope_input_count,
                    }));
                }
                if region_id == scope {
                    return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceConstant {
                        region: region_id,
                        atom,
                    }));
                }
                if capture_index >= scope_input_count {
                    return Err(ProgramError::custom(ReferenceAnalysisError::InvalidReferenceCapture {
                        region: region_id,
                        atom,
                        capture_index,
                        scope_input_count,
                    }));
                }
                let input_atom = regions[scope.index()].input_ids().get(capture_index).copied().ok_or_else(|| {
                    ProgramError::custom(ReferenceAnalysisError::InvalidReferenceCapture {
                        region: region_id,
                        atom,
                        capture_index,
                        scope_input_count,
                    })
                })?;
                let input_type = regions[scope.index()].atoms()[input_atom.index()].r#type();
                if constant_type.as_ref() != input_type.as_ref() {
                    return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceCaptureTypeMismatch {
                        region: region_id,
                        atom,
                        capture_index,
                        capture_type: constant_type.to_string(),
                        input_type: input_type.to_string(),
                    }));
                }
                let root = ReferenceRoot::RegionInput { region: scope, input_index: capture_index };
                if let Some((_, first)) =
                    region_capture_roots[region_index].iter().find(|(existing, _)| *existing == root)
                {
                    return Err(ProgramError::custom(ReferenceAnalysisError::DuplicateReferenceCaptureAlias {
                        region: region_id,
                        first_atom: first.atom(),
                        second_atom: atom,
                        root,
                    }));
                }
                region_capture_roots[region_index].push((root, ValueId::new(region_id, atom)));
                handles[region_index][atom_index] = Some(ReferenceHandle { root, is_root: true });
                live_roots.insert(root);
            }

            if rule_regions[region_index]
                && (handles[region_index].iter().any(Option::is_some)
                    || region
                        .instructions()
                        .iter()
                        .any(|instruction| !instruction.operation().reference_semantics().is_empty()))
            {
                return Err(ProgramError::custom(ReferenceAnalysisError::RuleRegion { region: region_id }));
            }

            for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                let instruction_id = InstructionId::new(region_id, instruction_index);
                let operation = instruction.operation();
                let semantics = operation.reference_semantics();
                let has_reference_boundary = instruction
                    .inputs()
                    .iter()
                    .chain(instruction.outputs())
                    .any(|atom| region.atoms()[atom.index()].r#type().is_reference())
                    || instruction.regions().iter().copied().any(|attached| {
                        let attached_region = &regions[attached.index()];
                        attached_region
                            .input_ids()
                            .iter()
                            .chain(attached_region.output_ids())
                            .copied()
                            .any(|atom| attached_region.atoms()[atom.index()].r#type().is_reference())
                    });
                if semantics.is_empty() && !has_reference_boundary {
                    continue;
                }
                let operation_name = operation.name();
                let requires_ordered_state =
                    semantics.outputs().iter().any(|output| matches!(output, ReferenceOutputSemantics::NewRoot { .. }))
                        || !semantics.accesses().is_empty();
                if requires_ordered_state && !operation.effects().contains(Effect::OrderedState) {
                    return Err(ProgramError::custom(ReferenceAnalysisError::MissingOrderedState {
                        instruction: instruction_id,
                        operation: operation_name.to_string(),
                    }));
                }
                let mut classified_inputs = vec![false; instruction.inputs().len()];
                let mut classified_outputs = vec![false; instruction.outputs().len()];

                for output in semantics.outputs().iter().copied() {
                    let output_index = output.output_index();
                    let Some(output_atom) = instruction.outputs().get(output_index).copied() else {
                        return Err(ProgramError::custom(ReferenceAnalysisError::InvalidOutputIndex {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            output_index,
                            output_count: instruction.outputs().len(),
                        }));
                    };
                    if !region.atoms()[output_atom.index()].r#type().is_reference() {
                        return Err(ProgramError::custom(ReferenceAnalysisError::NonReferenceOutput {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            output_index,
                            output_type: region.atoms()[output_atom.index()].r#type().to_string(),
                        }));
                    }
                    classified_outputs[output_index] = true;
                    let handle = match output {
                        ReferenceOutputSemantics::NewRoot { .. } => {
                            let root = ReferenceRoot::Allocation { instruction: instruction_id, output_index };
                            live_roots.insert(root);
                            ReferenceHandle { root, is_root: true }
                        }
                        ReferenceOutputSemantics::Alias { input_index, .. } => {
                            let Some(input_atom) = instruction.inputs().get(input_index).copied() else {
                                return Err(ProgramError::custom(ReferenceAnalysisError::InvalidAccessIndex {
                                    instruction: instruction_id,
                                    operation: operation_name.to_string(),
                                    input_index,
                                    input_count: instruction.inputs().len(),
                                }));
                            };
                            let Some(source) = handles[region_index][input_atom.index()] else {
                                return Err(ProgramError::custom(ReferenceAnalysisError::UnresolvedAlias {
                                    instruction: instruction_id,
                                    operation: operation_name.to_string(),
                                    input_index,
                                }));
                            };
                            if !live_roots.contains(&source.root) {
                                return Err(ProgramError::custom(ReferenceAnalysisError::UseAfterConsume {
                                    instruction: instruction_id,
                                    operation: operation_name.to_string(),
                                    input_index,
                                    root: source.root,
                                }));
                            }
                            let input_type = region.atoms()[input_atom.index()].r#type();
                            let output_type = region.atoms()[output_atom.index()].r#type();
                            if input_type.as_ref() != output_type.as_ref() {
                                return Err(ProgramError::custom(ReferenceAnalysisError::AliasTypeMismatch {
                                    instruction: instruction_id,
                                    operation: operation_name.to_string(),
                                    output_index,
                                    input_type: input_type.to_string(),
                                    output_type: output_type.to_string(),
                                }));
                            }
                            classified_inputs[input_index] = true;
                            ReferenceHandle { root: source.root, is_root: false }
                        }
                    };
                    handles[region_index][output_atom.index()] = Some(handle);
                }

                for access in semantics.accesses().iter().copied() {
                    let input_index = access.input_index();
                    let Some(input_atom) = instruction.inputs().get(input_index).copied() else {
                        return Err(ProgramError::custom(ReferenceAnalysisError::InvalidAccessIndex {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            input_index,
                            input_count: instruction.inputs().len(),
                        }));
                    };
                    let input_type = region.atoms()[input_atom.index()].r#type();
                    if !input_type.is_reference() {
                        return Err(ProgramError::custom(ReferenceAnalysisError::NonReferenceAccess {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            input_index,
                            input_type: input_type.to_string(),
                        }));
                    }
                    // Region inputs are initialized before replay, and every preceding reference output must have
                    // declared root/alias semantics before this instruction can be reached.
                    let handle = handles[region_index][input_atom.index()].unwrap();
                    if !live_roots.contains(&handle.root) {
                        return Err(ProgramError::custom(ReferenceAnalysisError::UseAfterConsume {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            input_index,
                            root: handle.root,
                        }));
                    }
                    if access.mode() == ReferenceAccessMode::Consume {
                        // Consumption requires the unaliased allocation handle in its creation region. Region inputs,
                        // capture handles, aliases, and allocations owned by another region are all rejected.
                        if !handle.is_root
                            || !matches!(
                                handle.root,
                                ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region_id,
                            )
                        {
                            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidConsume {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                input_index,
                                root: handle.root,
                            }));
                        }
                        live_roots.remove(&handle.root);
                    }
                    classified_inputs[input_index] = true;
                    accesses.push(ReferenceAccess {
                        instruction: instruction_id,
                        input_index,
                        root: handle.root,
                        mode: access.mode(),
                    });
                }

                for (attached_index, attached) in instruction.regions().iter().copied().enumerate() {
                    let attached_region = &regions[attached.index()];
                    let mut source_roots = HashMap::new();
                    for (region_input_index, region_input_atom) in
                        attached_region.input_ids().iter().copied().enumerate()
                    {
                        if !attached_region.atoms()[region_input_atom.index()].r#type().is_reference() {
                            continue;
                        }
                        let Some(source_input_index) =
                            operation.input_region_provenance(attached_index, region_input_index)
                        else {
                            return Err(ProgramError::custom(ReferenceAnalysisError::UnsupportedRegionInput {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                            }));
                        };
                        let invalid_region_input_source = || {
                            ProgramError::custom(ReferenceAnalysisError::InvalidRegionInputSource {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                                source_input_index,
                            })
                        };
                        let source_atom = instruction
                            .inputs()
                            .get(source_input_index)
                            .copied()
                            .ok_or_else(invalid_region_input_source)?;
                        let source_handle =
                            handles[region_index][source_atom.index()].ok_or_else(invalid_region_input_source)?;
                        if !live_roots.contains(&source_handle.root) {
                            return Err(ProgramError::custom(ReferenceAnalysisError::UseAfterConsume {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                input_index: source_input_index,
                                root: source_handle.root,
                            }));
                        }
                        let source_type = region.atoms()[source_atom.index()].r#type();
                        let region_input_type = attached_region.atoms()[region_input_atom.index()].r#type();
                        if source_type.as_ref() != region_input_type.as_ref() {
                            return Err(invalid_region_input_source());
                        }
                        if let Some(first_input_index) = source_roots.insert(source_handle.root, region_input_index) {
                            return Err(ProgramError::custom(ReferenceAnalysisError::DuplicateRegionInputAlias {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                first_input_index,
                                second_input_index: region_input_index,
                                root: source_handle.root,
                            }));
                        }
                        if let Some((_, capture)) =
                            region_capture_roots[attached.index()].iter().find(|(root, _)| *root == source_handle.root)
                        {
                            return Err(ProgramError::custom(ReferenceAnalysisError::DuplicateRegionCaptureAlias {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                                capture_region: capture.region(),
                                capture_atom: capture.atom(),
                                root: source_handle.root,
                            }));
                        }
                        classified_inputs[source_input_index] = true;
                        let binding = ReferenceRegionInputBinding {
                            region_root: ReferenceRoot::RegionInput {
                                region: attached,
                                input_index: region_input_index,
                            },
                            source_input_index,
                            source_root: source_handle.root,
                        };
                        binding_indices.insert((instruction_id, attached_index, binding.region_root), bindings.len());
                        source_binding_indices
                            .insert((instruction_id, attached_index, binding.source_root), bindings.len());
                        bindings.push(binding);
                    }
                }

                // Reference-valued results of attached regions may cross only a declared operation result path. The
                // nested handle is translated through the binding recorded for this exact invocation, so a shared
                // region can resolve to a different caller root at another instruction without conflating them.
                let mut forwarded_region_outputs = HashSet::new();
                let mut forwarded_output_roots = HashMap::new();
                for (output_index, output_atom) in instruction.outputs().iter().copied().enumerate() {
                    if !region.atoms()[output_atom.index()].r#type().is_reference() || classified_outputs[output_index]
                    {
                        continue;
                    }
                    let provenance = operation.output_region_provenance(output_index);
                    let mut output_root = None;
                    let mut output_is_root = true;
                    for source in provenance {
                        let invalid_region_output_source = || {
                            ProgramError::custom(ReferenceAnalysisError::InvalidRegionOutputSource {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                output_index,
                                region_index: source.region_index,
                                region_output_index: source.output_index,
                            })
                        };
                        let attached = instruction
                            .regions()
                            .get(source.region_index)
                            .copied()
                            .ok_or_else(invalid_region_output_source)?;
                        let attached_region = &regions[attached.index()];
                        let source_atom = attached_region
                            .output_ids()
                            .get(source.output_index)
                            .copied()
                            .ok_or_else(invalid_region_output_source)?;
                        let source_handle =
                            handles[attached.index()][source_atom.index()].ok_or_else(invalid_region_output_source)?;
                        forwarded_region_outputs.insert((source.region_index, source.output_index));
                        let (translated_root, translated_is_root) = match source_handle.root {
                            ReferenceRoot::RegionInput { region: source_region, .. } if source_region == attached => {
                                let binding = binding_indices
                                    .get(&(instruction_id, source.region_index, source_handle.root))
                                    .map(|index| &bindings[*index])
                                    .ok_or_else(|| {
                                        ProgramError::MalformedProgram(format!(
                                            "reference result of attached region {} has no caller-root binding",
                                            source.region_index,
                                        ))
                                    })?;
                                let source_atom = instruction.inputs()[binding.source_input_index];
                                let caller_handle = handles[region_index][source_atom.index()].ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "reference result of attached region {} maps to an unresolved caller input",
                                        source.region_index,
                                    ))
                                })?;
                                (binding.source_root, caller_handle.is_root && source_handle.is_root)
                            }
                            root => (root, source_handle.is_root),
                        };
                        if let Some(first_root) = output_root
                            && first_root != translated_root
                        {
                            return Err(ProgramError::custom(ReferenceAnalysisError::InconsistentRegionOutputRoots {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                output_index,
                                first_root,
                                second_root: translated_root,
                            }));
                        }
                        output_root = Some(translated_root);
                        output_is_root &= translated_is_root;
                    }
                    if let Some(root) = output_root {
                        if let Some(first_output_index) = forwarded_output_roots.insert(root, output_index) {
                            return Err(ProgramError::custom(ReferenceAnalysisError::DuplicateRegionOutputAlias {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                first_output_index,
                                second_output_index: output_index,
                                root,
                            }));
                        }
                        if let Some(input_index) = operation.reference_output_identity_input(output_index) {
                            let input_atom = instruction.inputs().get(input_index).copied().ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference output {output_index} of `{operation_name}` requires out-of-range input \
                                     {input_index}",
                                ))
                            })?;
                            let input_root = handles[region_index][input_atom.index()]
                                .ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "reference output {output_index} of `{operation_name}` requires non-reference \
                                         input {input_index}",
                                    ))
                                })?
                                .root;
                            if input_root != root {
                                return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceCarryRootMismatch {
                                    instruction: instruction_id,
                                    operation: operation_name.to_string(),
                                    output_index,
                                    input_index,
                                    input_root,
                                    output_root: root,
                                }));
                            }
                        }
                        // Whole-root handles forwarded through a structured computation remain the root handle in the
                        // caller. Phase 8 view aliases will carry their own handle-local view descriptor and must not
                        // take this root-only path.
                        handles[region_index][output_atom.index()] =
                            Some(ReferenceHandle { root, is_root: output_is_root });
                        classified_outputs[output_index] = true;
                    }
                }

                for (attached_index, attached) in instruction.regions().iter().copied().enumerate() {
                    for (output_index, output) in regions[attached.index()].output_ids().iter().copied().enumerate() {
                        if regions[attached.index()].atoms()[output.index()].r#type().is_reference()
                            && !forwarded_region_outputs.contains(&(attached_index, output_index))
                        {
                            return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceOutput {
                                region: attached,
                                output_index,
                                root: handles[attached.index()][output.index()].unwrap().root,
                            }));
                        }
                    }
                }

                for (input_index, input) in instruction.inputs().iter().copied().enumerate() {
                    if region.atoms()[input.index()].r#type().is_reference() && !classified_inputs[input_index] {
                        return Err(ProgramError::custom(ReferenceAnalysisError::UnclassifiedInput {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            input_index,
                        }));
                    }
                }
                for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
                    if region.atoms()[output.index()].r#type().is_reference() && !classified_outputs[output_index] {
                        return Err(ProgramError::custom(ReferenceAnalysisError::UnclassifiedOutput {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            output_index,
                        }));
                    }
                }
            }

            for (output_index, output) in region.output_ids().iter().copied().enumerate() {
                if region.atoms()[output.index()].r#type().is_reference() {
                    let root = handles[region_index][output.index()].unwrap().root;
                    if region_id == entry || matches!(root, ReferenceRoot::Allocation { .. }) {
                        return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceOutput {
                            region: region_id,
                            output_index,
                            root,
                        }));
                    }
                }
            }

            let nested_captures = region
                .instructions()
                .iter()
                .flat_map(|instruction| instruction.regions().iter().copied())
                .flat_map(|attached| region_capture_roots[attached.index()].iter().copied())
                .collect::<Vec<_>>();
            for capture in nested_captures {
                if region_capture_roots[region_index].iter().all(|existing| existing.0 != capture.0) {
                    region_capture_roots[region_index].push(capture);
                }
            }
        }

        // Build summaries only after every handle and invocation binding has been validated. Arena order is
        // topological, so each attached region's outward summary is complete before its parent instruction is
        // visited. Nested formal roots are substituted exactly through the analysis bindings above; local allocation
        // roots are filtered from the outward region summary because they never enter the caller.
        let mut accesses_by_instruction = regions
            .iter()
            .map(|region| vec![Vec::<ReferenceAccess>::new(); region.instructions().len()])
            .collect::<Vec<_>>();
        for access in &accesses {
            accesses_by_instruction[access.instruction.region().index()][access.instruction.index()].push(*access);
        }
        let mut region_summaries = vec![Vec::<ReferenceTransitiveAccess>::new(); regions.len()];
        let mut region_summary_sets = vec![HashSet::<ReferenceTransitiveAccess>::new(); regions.len()];
        let mut instruction_summaries = Vec::new();
        let mut instruction_summary_indices =
            regions.iter().map(|region| vec![None; region.instructions().len()]).collect::<Vec<_>>();
        for (region_index, region) in regions.iter().enumerate() {
            if !included[region_index] {
                continue;
            }
            let region_id = RegionId::new(region_index);
            for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                let instruction_id = InstructionId::new(region_id, instruction_index);
                let mut summary = accesses_by_instruction[region_index][instruction_index]
                    .iter()
                    .map(|source| ReferenceTransitiveAccess { root: source.root, mode: source.mode })
                    .collect::<Vec<_>>();
                let mut summary_set = summary.iter().copied().collect::<HashSet<_>>();
                for (attached_index, attached) in instruction.regions().iter().copied().enumerate() {
                    for access in &region_summaries[attached.index()] {
                        let root = match access.root {
                            ReferenceRoot::RegionInput { region: source_region, .. } if source_region == attached => {
                                binding_indices
                                    .get(&(instruction_id, attached_index, access.root))
                                    .map(|index| bindings[*index].source_root)
                                    .ok_or_else(|| {
                                        ProgramError::MalformedProgram(format!(
                                            "reference access in attached region {attached_index} has no caller-root \
                                             binding",
                                        ))
                                    })?
                            }
                            root => root,
                        };
                        if !instruction
                            .operation()
                            .allows_reference_access_through_region_input(attached_index, access.mode())
                        {
                            return Err(ProgramError::custom(ReferenceAnalysisError::ForbiddenRegionInputAccess {
                                instruction: instruction_id,
                                operation: instruction.operation().name().to_string(),
                                region_index: attached_index,
                                root,
                                mode: access.mode(),
                            }));
                        }
                        let access = ReferenceTransitiveAccess { root, mode: access.mode };
                        if summary_set.insert(access) {
                            summary.push(access);
                        }
                    }
                }
                for access in
                    summary.iter().copied().filter(|access| matches!(access.root, ReferenceRoot::RegionInput { .. }))
                {
                    if region_summary_sets[region_index].insert(access) {
                        region_summaries[region_index].push(access);
                    }
                }
                if !summary.is_empty() {
                    instruction_summary_indices[region_index][instruction_index] = Some(instruction_summaries.len());
                    instruction_summaries.push(summary);
                }
            }
        }

        for (input_index, input) in self.input_ids().iter().copied().enumerate() {
            let input_type = self.atoms()[input.index()].r#type();
            if !input_type.is_reference() {
                continue;
            }
            external_roots.push(ExternalReferenceRoot {
                root: handles[entry.index()][input.index()].unwrap().root,
                source: if input_index < capture_count {
                    ReferenceSource::Capture { index: input_index }
                } else {
                    ReferenceSource::PublicInput { index: input_index - capture_count }
                },
            });
        }

        let roots = handles
            .into_iter()
            .map(|region| region.into_iter().map(|handle| handle.map(|handle| handle.root)).collect())
            .collect();
        Ok(ReferenceAnalysis {
            roots,
            accesses,
            bindings,
            source_binding_indices,
            external_roots,
            instruction_summaries,
            instruction_summary_indices,
            region_summaries,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::captures::CaptureReference;
    use crate::operations::{
        ConditionOperation, FreezeReferenceOperation, NewReferenceOperation, ReferenceReadOperation,
        ReferenceSwapOperation, ScanOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        Effects, OutputRegionProvenance, ProgramBuilder, ReferenceInputAccess, ReferenceOperationSemantics,
        ReferenceType, RegionArena, RegionInterface, RegionSlot, TypeError,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type Capture = CaptureReference<ArrayIrType>;
    type CaptureArray = CaptureReference<ArrayType>;
    type CaptureOperation = ArrayIrOperation<CaptureArray>;

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    /// Malformed reference operation descriptors used to pin each static-analysis diagnostic.
    #[derive(Clone, Debug)]
    enum MalformedReferenceOperation {
        /// Leaves a reference input unclassified.
        Unclassified,

        /// Produces a reference output without root or alias semantics.
        UnclassifiedOutput,

        /// Describes an access beyond the operand list.
        InvalidAccess,

        /// Describes a root beyond the result list.
        InvalidOutput,

        /// Describes a reference access on an array operand.
        NonReferenceAccess,

        /// Describes a new root on an array result.
        BadRoot,

        /// Describes an alias whose result type differs from its source type.
        BadAlias,

        /// Produces a valid fresh reference root.
        Root,

        /// Produces a valid alias of a reference operand.
        Alias,

        /// Reads a valid reference operand.
        Read,

        /// Consumes a valid reference operand.
        Consume,

        /// Describes reference state without declaring the ordered-state effect.
        MissingEffect,

        /// Forwards an operand into a computation region.
        ComputationCarrier,

        /// Calls a computation region and forwards its outputs to the caller.
        CallCarrier,

        /// Establishes a capture scope longer than the attached region input boundary.
        InvalidCaptureScopeCarrier,

        /// Declares an out-of-range attached-region source for a reference output.
        InvalidOutputSourceCarrier,

        /// Forwards one attached-region reference output through two implicit handles.
        DuplicateOutputCarrier,

        /// Reports an invalid source operand for a region input.
        InvalidSourceCarrier,

        /// Forwards an operand into a dormant rule region.
        RuleCarrier,
    }

    impl Display for MalformedReferenceOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for MalformedReferenceOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Unclassified => "unclassified_reference",
                Self::UnclassifiedOutput => "unclassified_reference_output",
                Self::InvalidAccess => "invalid_reference_access",
                Self::InvalidOutput => "invalid_reference_output",
                Self::NonReferenceAccess => "non_reference_access",
                Self::BadRoot => "bad_reference_root",
                Self::BadAlias => "bad_reference_alias",
                Self::Root => "test_reference_root",
                Self::Alias => "test_reference_alias",
                Self::Read => "test_reference_read",
                Self::Consume => "test_reference_consume",
                Self::MissingEffect => "test_missing_reference_effect",
                Self::ComputationCarrier => "test_computation_carrier",
                Self::CallCarrier => "test_call_carrier",
                Self::InvalidCaptureScopeCarrier => "test_invalid_capture_scope_carrier",
                Self::InvalidOutputSourceCarrier => "test_invalid_output_source_carrier",
                Self::DuplicateOutputCarrier => "test_duplicate_output_carrier",
                Self::InvalidSourceCarrier => "test_invalid_source_carrier",
                Self::RuleCarrier => "test_rule_carrier",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::BadRoot | Self::Read => vec![scalar_type().into()],
                Self::Root | Self::MissingEffect | Self::InvalidOutput | Self::UnclassifiedOutput => {
                    vec![ReferenceType::new(scalar_type()).into()]
                }
                Self::Alias => vec![input_types[0].clone()],
                Self::CallCarrier | Self::InvalidCaptureScopeCarrier => region_interfaces[0].output_types().to_vec(),
                Self::InvalidOutputSourceCarrier => vec![region_interfaces[0].output_types()[0].clone()],
                Self::DuplicateOutputCarrier => vec![region_interfaces[0].output_types()[0].clone(); 2],
                Self::BadAlias => vec![ReferenceType::new(ArrayType::scalar(DataType::F64)).into()],
                Self::Consume => vec![scalar_type().into()],
                Self::Unclassified
                | Self::InvalidAccess
                | Self::NonReferenceAccess
                | Self::ComputationCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => Vec::new(),
            })
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                    if region_index == 0 =>
                {
                    Some(input_index)
                }
                Self::InvalidSourceCarrier if region_index == 0 => Some(1),
                _ => None,
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::CallCarrier | Self::InvalidCaptureScopeCarrier => {
                    vec![OutputRegionProvenance { region_index: 0, output_index }]
                }
                Self::InvalidOutputSourceCarrier => {
                    vec![OutputRegionProvenance { region_index: 1, output_index: 0 }]
                }
                Self::DuplicateOutputCarrier => vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
                _ => Vec::new(),
            }
        }

        fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
            match self {
                Self::CallCarrier if region_index == 0 => Some(1),
                Self::InvalidCaptureScopeCarrier if region_index == 0 => Some(2),
                _ => None,
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            Cow::Owned(match self {
                Self::Unclassified | Self::UnclassifiedOutput => ReferenceOperationSemantics::default(),
                Self::InvalidAccess => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(1, ReferenceAccessMode::Read)],
                ),
                Self::NonReferenceAccess => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ),
                Self::BadRoot => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::InvalidOutput => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 1 }],
                    Vec::new(),
                ),
                Self::Root => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::MissingEffect => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::Alias | Self::BadAlias => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0 }],
                    Vec::new(),
                ),
                Self::Read => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ),
                Self::Consume => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Consume)],
                ),
                Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => ReferenceOperationSemantics::default(),
            })
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier => const { &[RegionSlot::computation("body")] },
                Self::RuleCarrier => const { &[RegionSlot::rule("rule")] },
                _ => &[],
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::InvalidAccess
                | Self::NonReferenceAccess
                | Self::BadRoot
                | Self::InvalidOutput
                | Self::Root
                | Self::Read
                | Self::Consume => Effects::single(Effect::OrderedState),
                Self::Unclassified
                | Self::UnclassifiedOutput
                | Self::BadAlias
                | Self::Alias
                | Self::MissingEffect
                | Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => Effects::PURE,
            }
        }
    }

    #[test]
    fn test_reference_analysis_resolves_local_roots_and_consumption() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let analysis = program.analyze_references(0).unwrap();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(RegionId::new(0), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(RegionId::new(0), reference)), Some(root));
        assert_eq!(
            analysis.accesses(),
            &[
                ReferenceAccess {
                    instruction: InstructionId::new(RegionId::new(0), 1),
                    input_index: 0,
                    root,
                    mode: ReferenceAccessMode::Read,
                },
                ReferenceAccess {
                    instruction: InstructionId::new(RegionId::new(0), 2),
                    input_index: 0,
                    root,
                    mode: ReferenceAccessMode::Consume,
                },
            ],
        );
        assert!(analysis.external_roots().is_empty());

        // Every alias in the family observes root consumption, even when the invalid use is a plain read.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, read], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "reference_read".to_string(),
                input_index: 0,
                root,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_classifies_external_roots() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let captured = builder.add_input(reference_type.clone().into());
        let public = builder.add_input(reference_type.into());
        let captured_value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![captured]).unwrap()[0];
        let public_value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![public]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![captured_value, public_value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = program.analyze_references(1).unwrap();
        assert_eq!(
            analysis.external_roots(),
            &[
                ExternalReferenceRoot {
                    root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
                    source: ReferenceSource::Capture { index: 0 },
                },
                ExternalReferenceRoot {
                    root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 1 },
                    source: ReferenceSource::PublicInput { index: 0 },
                },
            ],
        );

        // External roots are never consumable by user code in the initial lifecycle model.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(scalar_type()).into());
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![external]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "freeze_reference".to_string(),
                input_index: 0,
                root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_region_reference_analysis_ignores_unreachable_sibling_regions() {
        let mut valid_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let array = valid_builder.add_input(scalar_type().into());
        let valid = valid_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![array], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut invalid_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = invalid_builder.add_input(ReferenceType::new(scalar_type()).into());
        let invalid = invalid_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let arena = RegionArena::from_regions(vec![
            valid.entry_region_ref().region().clone(),
            invalid.entry_region_ref().region().clone(),
        ])
        .unwrap();
        let valid = RegionRef::new(&arena, RegionId::new(0)).unwrap();
        let invalid = RegionRef::new(&arena, RegionId::new(1)).unwrap();
        // The reference-free root sees an arena whose sibling is illegal, yet its own closure still takes the
        // reference-free fast path; analyzing that sibling as the root reports the illegal escape.
        assert!(valid.analyze_references(0).unwrap().is_reference_free());
        let error = invalid.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: RegionId::new(1),
                output_index: 0,
                root: ReferenceRoot::RegionInput { region: RegionId::new(1), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_substitutes_condition_region_inputs() {
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = branch_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_references(0).unwrap();
        let instruction = InstructionId::new(program.entry(), 0);
        let source_root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let branch_root = ReferenceRoot::RegionInput { region: branch, input_index: 0 };
        assert_eq!(
            analysis.bindings,
            &[
                ReferenceRegionInputBinding { region_root: branch_root, source_input_index: 1, source_root },
                ReferenceRegionInputBinding { region_root: branch_root, source_input_index: 1, source_root },
            ],
        );
        assert_eq!(analysis.accesses()[0].root(), branch_root);

        // Both branch invocations of the shared region substitute the same caller root, and a root that never crosses
        // the boundary has no formal counterpart in either invocation.
        assert_eq!(analysis.region_root_for_source(instruction, 0, source_root), Some(branch_root));
        assert_eq!(analysis.region_root_for_source(instruction, 1, source_root), Some(branch_root));
        assert_eq!(analysis.region_root_for_source(instruction, 0, branch_root), None);
    }

    #[test]
    fn test_reference_analysis_rejects_inconsistent_condition_reference_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut first_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = first_builder.add_input(reference_type.clone().into());
        first_builder.add_input(reference_type.clone().into());
        let first_branch = first_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut second_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        second_builder.add_input(reference_type.clone().into());
        let second = second_builder.add_input(reference_type.clone().into());
        let second_branch = second_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first_branch = builder.import_region(first_branch.entry_region_ref());
        let second_branch = builder.import_region(second_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![first_branch, second_branch],
                vec![predicate, first, second],
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InconsistentRegionOutputRoots {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "condition".to_string(),
                output_index: 0,
                first_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
                second_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 2 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_invalid_or_duplicated_structured_reference_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let make_body = || {
            let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        // A declared attached-region source for a reference output must name an existing region and output.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let body = builder.import_region(make_body().entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidOutputSourceCarrier, vec![body], vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidRegionOutputSource {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_invalid_output_source_carrier".to_string(),
                output_index: 0,
                region_index: 1,
                region_output_index: 0,
            }),
        );

        // Two caller outputs cannot forward one attached-region reference output, because that would hand the caller
        // two unaliased handles to the same root.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let body = builder.import_region(make_body().entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::DuplicateOutputCarrier, vec![body], vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionOutputAlias {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_duplicate_output_carrier".to_string(),
                first_output_index: 0,
                second_output_index: 1,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_resolves_attached_lifted_capture_constants() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let captured = branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = branch_builder
            .add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![captured])
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();

        let ordinary_error = program.analyze_references(1).unwrap_err();
        assert_eq!(
            ordinary_error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceConstant { region: branch, atom: AtomId::new(0) }),
        );

        let analysis = program.analyze_references_with_lifted_captures(1).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(branch, captured)), Some(root));
        assert_eq!(analysis.accesses()[0].root(), root);
        assert_eq!(
            analysis.region_summary(branch).unwrap(),
            &[ReferenceTransitiveAccess { root, mode: ReferenceAccessMode::Read }],
        );
        assert_eq!(
            analysis.instruction_summary(InstructionId::new(program.entry(), 0)).unwrap(),
            &[ReferenceTransitiveAccess { root, mode: ReferenceAccessMode::Read }],
        );
    }

    #[test]
    fn test_reference_analysis_deduplicates_shared_diamond_summaries() {
        let reference_type = ReferenceType::new(scalar_type());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut leaf_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = leaf_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = leaf_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let mut shared = leaf_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        // Each level attaches the same preceding region twice. A path-sensitive flattening would construct 2^64
        // identical access facts, while the canonical summary retains one fact per root/mode/source triple.
        for _ in 0..64 {
            let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
            let branch = builder.import_region(shared.entry_region_ref());
            let predicate = builder.add_constant(Capture::new(1, predicate_type.clone().into()));
            let value = builder
                .add_instruction(
                    ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                    vec![branch, branch],
                    vec![predicate],
                )
                .unwrap()[0];
            shared = builder.build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder]).unwrap();
        }

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(shared.entry_region_ref());
        builder.add_input(reference_type.into());
        let predicate = builder.add_input(predicate_type.into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![branch, branch],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_references_with_lifted_captures(2).unwrap();
        let instruction = InstructionId::new(program.entry(), 0);
        assert_eq!(analysis.instruction_summary(instruction).unwrap().len(), 1);
        assert_eq!(analysis.region_summary(program.entry()).unwrap().len(), 1);
    }

    #[test]
    fn test_reference_analysis_rejects_capture_index_beyond_the_lifted_prefix() {
        // A capture constant addresses its active lexical scope, so an index past that scope's addressable inputs is a
        // precise diagnostic rather than a silent out-of-range read.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(1, reference_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidReferenceCapture {
                region: branch,
                atom: AtomId::new(0),
                capture_index: 1,
                scope_input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_constant_type_mismatch() {
        // The constant and the lifted boundary it resolves against must declare exactly the same reference type, so a
        // differing referent element type cannot be silently reinterpreted.
        let reference_type = ReferenceType::new(scalar_type());
        let mismatched_type = ReferenceType::new(ArrayType::scalar(DataType::F64));
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(0, mismatched_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCaptureTypeMismatch {
                region: branch,
                atom: AtomId::new(0),
                capture_index: 0,
                capture_type: ArrayIrType::Reference(mismatched_type).to_string(),
                input_type: ArrayIrType::Reference(reference_type).to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_duplicate_capture_constant_aliases() {
        // Two constants naming one captured root would create parallel unaliased handles inside a single invocation,
        // which the lifecycle model forbids.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateReferenceCaptureAlias {
                region: branch,
                first_atom: AtomId::new(0),
                second_atom: AtomId::new(1),
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_aliasing_a_formal_region_argument() {
        // A branch that receives a root as a formal argument cannot also reach that same root through a lexical capture
        // constant, because the two handles would alias one resource inside one invocation.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        branch_builder.add_input(reference_type.clone().into());
        let captured = branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = branch_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![captured]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionCaptureAlias {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "condition".to_string(),
                region_index: 0,
                input_index: 0,
                capture_region: branch,
                capture_atom: captured,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_call_capture_scope_longer_than_its_region_boundary() {
        // A call-like operation establishes its callee's capture scope from that region's leading inputs, so a declared
        // prefix longer than the region boundary is rejected while the scope map is still being built.
        let reference_type = ReferenceType::new(scalar_type());
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        callee_builder.add_input(reference_type.clone().into());
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidCaptureScopeCarrier, vec![callee], vec![reference])
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidRegionCaptureCount {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_invalid_capture_scope_carrier".to_string(),
                region_index: 0,
                capture_count: 2,
                input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_ambiguous_shared_capture_scopes() {
        // A shared region reachable both through a scope-establishing call and through ordinary control flow would have
        // its capture constants resolve against two different lexical scopes. Resolving them would make legality
        // path-sensitive, so the shared region is rejected outright.
        let reference_type = ReferenceType::new(scalar_type());
        let mut shared_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        shared_builder.add_input(reference_type.clone().into());
        shared_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let shared = shared_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let shared = builder.import_region(shared.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![shared], vec![first])
            .unwrap();
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![shared], Vec::new())
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();

        let error = program.analyze_references_with_lifted_captures(2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::AmbiguousReferenceCaptureScope {
                region: shared,
                first_scope: shared,
                first_scope_input_count: 1,
                second_scope: program.entry(),
                second_scope_input_count: 2,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_propagates_while_and_scan_carry_roots() {
        let reference_type = ReferenceType::new(scalar_type());

        // A while loop's reference carry keeps the caller root, and the instruction summary reports the union of
        // the condition region's read and the body region's write in the caller's own namespace.
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation, Vec::new(), vec![condition_reference])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body_reference = body_builder.add_input(reference_type.clone().into());
        let replacement = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        body_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![body_reference, replacement])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![body_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        let final_reference =
            builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![reference]).unwrap()[0];
        let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![final_reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_references(0).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), final_reference)), Some(root));
        assert_eq!(
            analysis.instruction_summary(InstructionId::new(program.entry(), 0)).unwrap(),
            &[
                ReferenceTransitiveAccess { root, mode: ReferenceAccessMode::Read },
                ReferenceTransitiveAccess { root, mode: ReferenceAccessMode::Write },
            ],
        );

        // A scan carry behaves the same way, and a body that only reads produces exactly one summarized access.
        let mut scan_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let scan_reference = scan_body_builder.add_input(reference_type.clone().into());
        scan_body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![scan_reference]).unwrap();
        let scan_body = scan_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![scan_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let scan_body = builder.import_region(scan_body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let final_reference = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 2), vec![scan_body], vec![reference])
            .unwrap()[0];
        let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![final_reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_references(0).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), final_reference)), Some(root));
        assert_eq!(
            analysis.instruction_summary(InstructionId::new(program.entry(), 0)).unwrap(),
            &[ReferenceTransitiveAccess { root, mode: ReferenceAccessMode::Read }],
        );
    }

    #[test]
    fn test_reference_analysis_rejects_permuted_while_reference_carries() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A body that swaps the two carries has no consistent fixed point for either root and is rejected precisely.
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = body_builder.add_input(reference_type.clone().into());
        let second = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![first, second]).unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCarryRootMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "while".to_string(),
                output_index: 0,
                input_index: 0,
                input_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                output_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_permuted_scan_reference_carries() {
        let reference_type = ReferenceType::new(scalar_type());

        // A body that swaps the two carries has no consistent fixed point for either root and is rejected precisely.
        let mut scan_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = scan_body_builder.add_input(reference_type.clone().into());
        let second = scan_body_builder.add_input(reference_type.clone().into());
        let scan_body = scan_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(scan_body.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder
            .add_instruction(ScanOperation::<TestValue>::new(2, 0), vec![body], vec![first, second])
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCarryRootMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "scan".to_string(),
                output_index: 0,
                input_index: 0,
                input_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                output_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_mutation_of_while_condition_input() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        let replacement = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        condition_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![reference]).unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();

        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ForbiddenRegionInputAccess {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "while".to_string(),
                region_index: 0,
                root,
                mode: ReferenceAccessMode::Write,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_allows_condition_local_reference_mutation() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(scalar_type().into());
        let initial = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32)));
        let reference = condition_builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let replacement = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        condition_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(scalar_type().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(scalar_type().into());
        let final_state =
            builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![final_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_references(0).unwrap();
        assert!(analysis.external_roots().is_empty());
        assert!(analysis.instruction_summary(InstructionId::new(program.entry(), 0)).is_none());
    }

    #[test]
    fn test_reference_analysis_rejects_duplicate_roots_within_one_region_invocation() {
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = branch_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![first]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![branch, branch],
                vec![predicate, reference, reference],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let instruction = InstructionId::new(program.entry(), 0);
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionInputAlias {
                instruction,
                operation: "condition".to_string(),
                region_index: 0,
                first_input_index: 0,
                second_input_index: 1,
                root,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_scopes_branch_local_allocations_to_their_creating_region() {
        // A root allocated inside a shared branch belongs to that branch invocation, never to the caller, so it stays a
        // region-local allocation root and never reaches the external boundary.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = branch_builder.add_input(scalar_type().into());
        let reference = branch_builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let value = branch_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, initial])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_references(0).unwrap();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(branch, 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(branch, reference)), Some(root));
        assert_eq!(analysis.accesses()[0].root(), root);
        assert!(analysis.external_roots().is_empty());
        assert!(analysis.region_summary(branch).unwrap().is_empty());
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_an_inherited_branch_root() {
        // A branch that receives a caller root as a formal argument only borrows it, so consuming it inside the branch
        // is reported against the branch's own region-input root.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = branch_builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(branch, 0),
                operation: "freeze_reference".to_string(),
                input_index: 0,
                root: ReferenceRoot::RegionInput { region: branch, input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_roots_escaping_a_region_output() {
        // References are second class in every region, so a nested region cannot return a resolved capture root through
        // its own output boundary even though the constant itself is legal there.
        let reference_type = ReferenceType::new(scalar_type());
        let mut child_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let captured = child_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let child = child_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let child = builder.import_region(child.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![child], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: child,
                output_index: 0,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_constants_in_a_call_scope_root_region() {
        // A callee region is the root of its own capture scope, so a capture constant there would have to address that
        // region's own inputs; such a constant is rejected exactly like an immediate reference constant.
        let reference_type = ReferenceType::new(scalar_type());
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        callee_builder.add_input(reference_type.clone().into());
        let captured = callee_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = callee_builder
            .add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![captured])
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let value = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![callee], vec![reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceConstant { region: callee, atom: AtomId::new(1) }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_index_beyond_a_nested_call_scope() {
        // Inside a callee, capture indices address that callee's declared capture prefix rather than the entry
        // boundary, so the reported scope input count is the nested prefix length and not the caller's input count.
        let reference_type = ReferenceType::new(scalar_type());
        let mut child_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        child_builder.add_constant(Capture::new(1, reference_type.clone().into()));
        let child = child_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let child = callee_builder.import_region(child.entry_region_ref());
        callee_builder.add_input(reference_type.clone().into());
        callee_builder.add_input(reference_type.clone().into());
        callee_builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![child], Vec::new())
            .unwrap();
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![callee], vec![first, second])
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        let child = program.regions()[callee.index()].instructions()[0].regions()[0];
        let error = program.analyze_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidReferenceCapture {
                region: child,
                atom: AtomId::new(0),
                capture_index: 1,
                scope_input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_reference_state_in_nested_rule_closure() {
        let mut leaf_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = leaf_builder.add_input(scalar_type().into());
        leaf_builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap();
        let leaf = leaf_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let mut rule_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let leaf = rule_builder.import_region(leaf.entry_region_ref());
        rule_builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![leaf], Vec::new())
            .unwrap();
        let rule = rule_builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let rule = builder.import_region(rule.entry_region_ref());
        builder.add_instruction(MalformedReferenceOperation::RuleCarrier, vec![rule], Vec::new()).unwrap();
        let program = builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::RuleRegion { region: RegionId::new(0) }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_count_exceeding_the_entry_input_count() {
        // The declared capture prefix is validated against the entry boundary before any operation semantics are
        // inspected, so the malformed operation in this fixture is never reached.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::MissingEffect, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidCaptureCount { capture_count: 2, input_count: 1 }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_stateful_semantics_without_ordered_state() {
        // Stateful reference semantics require the conservative ordered-state effect so that reference work can never
        // be reordered as if it were pure.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::MissingEffect, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::MissingOrderedState {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_missing_reference_effect".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_unclassified_reference_input() {
        // Every reference operand must be classified by the operation's own semantics descriptor.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder
            .add_instruction(MalformedReferenceOperation::Unclassified, Vec::new(), vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UnclassifiedInput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "unclassified_reference".to_string(),
                input_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_out_of_range_access_index() {
        // A declared access must name an operand that the instruction actually has.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidAccess, Vec::new(), vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidAccessIndex {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "invalid_reference_access".to_string(),
                input_index: 1,
                input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_access_on_a_non_reference_input() {
        // A declared access must land on a reference-typed operand.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let array = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::NonReferenceAccess, Vec::new(), vec![array])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::NonReferenceAccess {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "non_reference_access".to_string(),
                input_index: 0,
                input_type: "f32[]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_fresh_root_on_a_non_reference_output() {
        // A declared fresh root must land on a reference-typed result.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let array = builder.add_input(scalar_type().into());
        let bad = builder.add_instruction(MalformedReferenceOperation::BadRoot, Vec::new(), vec![array]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![bad], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::NonReferenceOutput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "bad_reference_root".to_string(),
                output_index: 0,
                output_type: "f32[]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_out_of_range_root_output_index() {
        // A declared root must name a result that the instruction actually has.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidOutput, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidOutputIndex {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "invalid_reference_output".to_string(),
                output_index: 1,
                output_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_unclassified_reference_output() {
        // Every reference-typed result must be classified as either a fresh root or an alias.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::UnclassifiedOutput, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UnclassifiedOutput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "unclassified_reference_output".to_string(),
                output_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_alias_type_mismatch() {
        // An alias must preserve its source handle's exact reference type.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder.add_instruction(MalformedReferenceOperation::BadAlias, Vec::new(), vec![reference]).unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::AliasTypeMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "bad_reference_alias".to_string(),
                output_index: 0,
                input_type: "ref<f32[]>".to_string(),
                output_type: "ref<f64[]>".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_missing_or_invalid_input_region_provenance() {
        let mut body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        for (operation, expected) in [
            (
                MalformedReferenceOperation::ComputationCarrier,
                ReferenceAnalysisError::UnsupportedRegionInput {
                    instruction: InstructionId::new(RegionId::new(1), 0),
                    operation: "test_computation_carrier".to_string(),
                    region_index: 0,
                    input_index: 0,
                },
            ),
            (
                MalformedReferenceOperation::InvalidSourceCarrier,
                ReferenceAnalysisError::InvalidRegionInputSource {
                    instruction: InstructionId::new(RegionId::new(1), 0),
                    operation: "test_invalid_source_carrier".to_string(),
                    region_index: 0,
                    input_index: 0,
                    source_input_index: 1,
                },
            ),
        ] {
            let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
            let body = builder.import_region(body.entry_region_ref());
            let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
            builder.add_instruction(operation, vec![body], vec![reference]).unwrap();
            let program =
                builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
            let error = program.analyze_references(0).unwrap_err();
            assert_eq!(error.downcast_custom::<ReferenceAnalysisError>(), Some(&expected));
        }
    }

    #[test]
    fn test_reference_analysis_resolves_an_alias_to_its_source_root() {
        // An explicit alias resolves to its source's canonical root, so accesses through either handle name one root.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let value = builder.add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_references(0).unwrap();
        let canonical_root =
            ReferenceRoot::Allocation { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), root)), Some(canonical_root));
        assert_eq!(analysis.root(ValueId::new(program.entry(), alias)), Some(canonical_root));
        assert_eq!(analysis.accesses()[0].root(), canonical_root);
    }

    #[test]
    fn test_reference_analysis_rejects_reading_through_an_alias_of_a_consumed_root() {
        // Consuming the root invalidates every handle in its alias family, including reads through an alias.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let frozen = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let read = builder.add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, read], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 3),
                operation: "test_reference_read".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_a_root_twice() {
        // A root can be consumed at most once.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let first = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let second = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, second], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_through_an_explicit_alias() {
        // Only an unaliased root handle may be consumed; an explicit alias never grants that right.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let value = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_forwards_call_roots_without_granting_alias_consumption() {
        // A structured call preserves a directly forwarded root handle, but it must not turn an explicit alias back
        // into a consumable root handle.
        let mut direct_body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let direct_reference = direct_body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let direct_body = direct_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![direct_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let direct_body = builder.import_region(direct_body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let forwarded = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![direct_body], vec![root])
            .unwrap()[0];
        let value =
            builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![forwarded]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_references(0).unwrap();
        let forwarded_root =
            ReferenceRoot::Allocation { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), forwarded)), Some(forwarded_root));

        let mut alias_body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let alias_reference = alias_body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let alias = alias_body_builder
            .add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![alias_reference])
            .unwrap()[0];
        let alias_body = alias_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![alias], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let alias_body = builder.import_region(alias_body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let forwarded = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![alias_body], vec![root])
            .unwrap()[0];
        let value =
            builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![forwarded]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_a_local_allocation_escaping_the_root_region_output() {
        // A local allocation cannot escape through the analyzed root region's own output boundary.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![root], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: program.entry(),
                output_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }
}
