//! Contains machinery for generic [`Program`]-level reference analysis.
//!
//! [`ReferenceAnalysis`] resolves every reference-typed value in a [`Region`] closure to exactly one canonical
//! [`ReferenceRoot`], records the alias edges and accesses that connect values to roots, and validates the lifetime,
//! capture, and region-boundary rules of the reference model. It relies only on generic [`Operation`] hooks (i.e.,
//! [`Operation::effects`], [`Operation::input_region_provenance`], [`Operation::output_region_provenance`],
//! [`Operation::region_capture_input_count`], [`Operation::reference_output_identity_input`], and
//! [`Operation::allows_reference_access_through_region_input`]) and on [`Type::is_reference`], and
//! so it knows nothing about arrays, view descriptions, or any particular [`Value`] family.
//!
//! Transform rules, kernel boundary validation, diagnostics, and lowering obtain structural facts through
//! [`RegionRef::reference_analysis`], which retains the analysis in the region's transform cache. Reference discharge
//! uses the same traversal with the identities bound at the rewritten boundary: two formal inputs can denote the same
//! caller allocation, and inherited captures belong to the current discharge environment. Those summaries are computed
//! per attachment and are not stored in the structural cache. They also track references needed by replay without
//! inventing an access effect for materialization, and ignore constants replay never materializes.
//! [`Instruction`](crate::Instruction) construction uses the incremental alias and lifetime tracking in
//! [`ProgramBuilder`](crate::ProgramBuilder); canonical builder identity queries consult the retained analysis when
//! a reference is forwarded through a nested region. The eager [`Reference`](crate::Reference) runtime enforces
//! concrete lifetimes independently.
//!
//! # Roots and Namespaces
//!
//! A root is a reference-typed input (i.e., [`ReferenceRoot::RegionInput`]), an [`Instruction`](crate::Instruction)'s
//! fresh allocation (i.e., [`ReferenceRoot::Allocation`]), or an external constant in an open region analysis (i.e.,
//! [`ReferenceRoot::Constant`]). Every region's values resolve to roots in that region's own _namespace_ (i.e., its own
//! inputs, its own allocations, and the capture roots it inherits from an enclosing scope). The analysis never rewrites
//! a nested region's records into its parent's namespace. Instead, each attachment of a nested region records one
//! [`ReferenceRegionInputBinding`] per reference-typed region input, mapping that formal input to the caller root
//! it denotes, and the attaching instruction's [`ReferenceTransitiveAccess`] summary is expressed in the caller's
//! namespace after substituting those bindings and dropping the nested region's local allocations. Only
//! [`RegionRole::Computation`] regions are entered as a dormant [`RegionRole::Rule`] region (e.g., a derived
//! rematerialization or custom derivative rule) is an input to a later transform rather than an executed child of the
//! attaching instruction, its reference-typed inputs are bound by that transform rather than by the instruction's
//! operands, and the transform validates it separately, so the analysis neither enters it nor attributes its accesses
//! to the instruction (i.e., the same rule by which [`Effects`](crate::Effects) exclude [`RegionRole::Rule`] regions).
//!
//! # Capture Scopes
//!
//! A capture-lifted program names its captures through constants whose capture index refers to the _active capture
//! scope_. At the analyzed region, the scope is its first `capture_count` inputs. A nested region either inherits the
//! scope of the instruction attaching it or, when [`Operation::region_capture_input_count`] returns `Some(n)`,
//! establishes a fresh scope from its first `n` inputs. A reference-typed constant resolves to the root bound
//! at its capture position, so a capture root is the same root in every region that inherits the scope.
//!
//! [`RegionRef::reference_analysis_with_constants`] analyzes an open region before its inherited captures are lifted.
//! It assigns external constant roots to unresolved inherited capture indices and concrete reference allocations.
//! Explicitly rebound capture scopes remain strict, including an empty prefix. The cache separates this mode from
//! strict capture-lifted analysis so a successful open region analysis cannot hide a missing capture in a closed one.
//!
//! # Boundaries
//!
//! Complete-value handles cross region boundaries freely: a nested region input with
//! [`InputRegionProvenance::Forwarded`] provenance denotes the caller root of the named operation input,
//! and a forwarded region output denotes the root it carried in. A derived view (i.e., any alias chain
//! containing a [`ReferenceAliasKind::View`] edge) crosses only when the attaching operation declares
//! [`InputRegionProvenance::View`] provenance for that region input through [`Operation::input_region_provenance`].
//! The named operation input must be a complete-value handle, and the region input is recorded as a view of it through
//! a [`ReferenceAliasEdge`] at position [`ReferenceAliasPosition::RegionInput`], whose description comes from the value
//! family's [`region_input_view`](crate::ReferenceViewOperation::region_input_view) hook. Such an input may be accessed
//! inside the region, where its accesses are attributed to the whole caller root, but it can neither be consumed there
//! nor be forwarded out. The view edge records only attachment-independent facts (which region input is a view, and of
//! which operand position), because a region is analyzed once and shared by every instruction attaching it; the
//! caller-side source root lives on the per-attachment [`ReferenceRegionInputBinding`], and a shared region reached
//! with a different boundary shape (an input that is a view under one attachment and a complete handle under another)
//! is rejected with [`ReferenceAnalysisError::InconsistentBoundaryViews`]. No other view enters or leaves an attached
//! region; a region that needs one recreates it from the carried root. Reference-typed outputs of region-carrying
//! operations resolve through [`Operation::reference_output_identity_input`] (every provenance origin must return
//! exactly the constrained root) or through [`Operation::output_region_provenance`] (all origins must agree).
//! An origin rooted in an allocation local to the attached region is an escaping allocation and is rejected.
//!
//! # Lifetime Rules
//!
//! Consumption is a complete-value lifetime event that must go through a complete-value handle, is legal only in the
//! region that allocated the root, and no access may follow it in program order, including accesses through aliases or
//! through nested regions of later instructions. Entry region inputs and captures are borrowed from the caller by
//! default and cannot be consumed. Internal callers may explicitly transfer ownership of selected non-capture entry
//! inputs so they can be consumed in that region. Attached regions still cannot consume them. For example:
//!
//! ```text
//! lambda %0:ref<f32[]> .                   external root: region input 0 (source input 0)
//! let %1:f32[] = reference_read %0         read of region input 0
//!     %2:ref<f32[]> = reference_new %1     local root: allocation at instruction 1
//!     %3:ref<f32[]> = reference_index %2   view alias of the allocation
//!     reference_write %3 %1                write reaching the allocation through the view
//!     %4:f32[] = reference_freeze %2       consumes the allocation; consuming %3 instead would be rejected
//! in (%4)
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Display;
use std::rc::Rc;
use std::sync::Arc;

use thiserror::Error;

use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::effects::{ReferenceAccessMode, ReferenceAliasKind};
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::ReferenceSource;
use crate::programs::references::values::ReferenceId;
use crate::programs::regions::{InputRegionProvenance, Region, RegionId, RegionRef, RegionRole};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId};

/// Error produced by [`ReferenceAnalysis`] when a [`Region`] closure violates the reference model. Each variant
/// identifies the region or [`Instruction`](crate::Instruction) at fault and includes the details needed to explain
/// the violated rule without repeating the analysis. Conversion to [`ProgramError`] preserves this error through
/// [`ReferenceError::Analysis`](crate::ReferenceError::Analysis).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceAnalysisError {
    /// An operation's reference effects or region provenance declarations are missing or invalid. This includes
    /// undeclared reference inputs or outputs, positions outside the application's inputs, outputs, or regions,
    /// and reference classifications applied to non-reference values.
    #[error("operation `{operation}` at {instruction} has an invalid reference declaration: {message}")]
    InvalidReferenceDeclaration {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Description of the malformed declaration.
        message: String,
    },

    /// An operation uses an input as a reference, but that input resolves to no reference root.
    #[error(
        "operation `{operation}` at {instruction} uses input {input_index} as a reference but it resolves to no \
         reference root"
    )]
    UnresolvedReference {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Input position that failed to resolve.
        input_index: usize,
    },

    /// A region stores a reference-typed constant that names no capture, but the requested analysis requires
    /// references to enter through explicit inputs or captures rather than concrete constants.
    #[error(
        "region {region} stores reference-typed constant {atom} that names no capture; references enter a program \
         only through inputs and captures"
    )]
    InvalidReferenceConstant {
        /// Region storing the constant.
        region: RegionId,

        /// Atom of the constant.
        atom: AtomId,
    },

    /// A reference-typed capture constant names a capture position that the active capture scope does not bind to a
    /// reference. The position may be outside the scope or may bind a non-reference value.
    #[error(
        "reference-typed constant {atom} in region {region} names capture {capture_index}, which the active capture \
         scope of {capture_count} captures does not bind to a reference"
    )]
    InvalidReferenceCapture {
        /// Region storing the constant.
        region: RegionId,

        /// Atom of the constant.
        atom: AtomId,

        /// Capture position named by the constant.
        capture_index: usize,

        /// Number of capture positions in the active scope.
        capture_count: usize,
    },

    /// A capture scope claims more inputs than its region has, or a shared region is reached under two different
    /// capture scopes.
    #[error("region {region} has an invalid capture scope: {message}")]
    InvalidCaptureScope {
        /// Region whose scope is invalid.
        region: RegionId,

        /// Description of the invalid scope.
        message: String,
    },

    /// A shared region is reached by attachments that disagree on which of its reference-typed inputs are boundary
    /// views created by the attaching operation.
    #[error("region {region} has inconsistent boundary views: {message}")]
    InconsistentBoundaryViews {
        /// Shared region whose attachments disagree on which inputs are views.
        region: RegionId,

        /// Description of the conflicting boundary view declarations.
        message: String,
    },

    /// An attached region returns a reference root that disagrees with the root required for an operation output.
    /// The expected root comes from the operation's input-identity constraint, or from the first forwarded region
    /// output when several regions supply the same output.
    #[error(
        "operation `{operation}` at {instruction} requires output {output_index} to denote {expected}, but \
         region {region_index} returns {actual} at output {region_output_index}"
    )]
    ReferenceRootMismatch {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Operation output whose reference root must be consistent.
        output_index: usize,

        /// Position of the attached region returning the mismatched root.
        region_index: usize,

        /// Output of the attached region returning the mismatched root.
        region_output_index: usize,

        /// Root the output must preserve.
        expected: ReferenceRoot,

        /// Root the attached region returns.
        actual: ReferenceRoot,
    },

    /// An operation forwards a reference allocated inside one of its attached regions out of that region.
    #[error(
        "operation `{operation}` at {instruction} forwards output {output_index} from a reference allocated at \
         {allocation} inside its attached region {region_index}; a local allocation cannot escape its creation scope"
    )]
    EscapingLocalAllocation {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Forwarded output position.
        output_index: usize,

        /// Position of the attached region owning the allocation.
        region_index: usize,

        /// Instruction performing the escaping allocation.
        allocation: InstructionId,
    },

    /// A derived reference view enters or leaves an attached region.
    #[error(
        "operation `{operation}` at {instruction} moves a derived reference view across region {region_index} \
         {boundary} {index}; only complete-value handles cross region boundaries"
    )]
    ViewCrossesRegionBoundary {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Position of the attached region.
        region_index: usize,

        /// Boundary side crossed by the view (i.e., `"input"` or `"output"`).
        boundary: &'static str,

        /// Position on that boundary of the attached region.
        index: usize,
    },

    /// An attached region performs an access on an entering root that its operation does not permit.
    #[error(
        "operation `{operation}` at {instruction} does not allow region {region_index} to access {root}, which \
         enters the region from its parent, with mode `{mode}`"
    )]
    DisallowedRegionAccess {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Position of the attached region.
        region_index: usize,

        /// Entering root, in the namespace of the region containing the instruction.
        root: ReferenceRoot,

        /// Disallowed access mode.
        mode: ReferenceAccessMode,
    },

    /// A reference is consumed through a derived view rather than through a complete-value handle.
    #[error(
        "operation `{operation}` at {instruction} consumes a derived view of {root} through input {input_index}, but \
         consumption invalidates the complete alias family; consume the root handle instead"
    )]
    ConsumptionThroughView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Consumed input position.
        input_index: usize,

        /// Root of the consumed view.
        root: ReferenceRoot,
    },

    /// An external reference (i.e., an entry input or capture) is consumed.
    #[error(
        "operation `{operation}` at {instruction} consumes external reference {root} ({external_source}), which \
         its caller owns"
    )]
    ExternalReferenceConsumption {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Consumed external root.
        root: ReferenceRoot,

        /// Logical source of the external root. This field is not named `source` because `thiserror` would treat a
        /// field of that name as the error's cause.
        external_source: ReferenceSource,
    },

    /// A reference that entered a region from its parent is consumed inside that region.
    #[error(
        "operation `{operation}` at {instruction} consumes {root}, which entered region {region} from its parent; a \
         reference may only be consumed in the region that allocated it"
    )]
    ConsumptionOutsideCreationScope {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Region containing the consuming instruction.
        region: RegionId,

        /// Consumed root.
        root: ReferenceRoot,
    },

    /// A reference is accessed, directly or through a nested region, after being consumed.
    #[error(
        "operation `{operation}` at {instruction} accesses {root} after `{consumer_operation}` at {consumer} \
         consumed it"
    )]
    UseAfterConsume {
        /// Name of the accessing operation.
        operation: &'static str,

        /// Instruction performing the access.
        instruction: InstructionId,

        /// Consumed root.
        root: ReferenceRoot,

        /// Instruction that consumed the root.
        consumer: InstructionId,

        /// Name of the consuming operation.
        consumer_operation: &'static str,
    },
}

impl From<ReferenceAnalysisError> for ProgramError {
    #[inline]
    fn from(error: ReferenceAnalysisError) -> Self {
        ProgramError::Reference(error.into())
    }
}

/// Canonical reference root that a reference-typed value denotes that can be a [`Region`] input, an
/// [`Instruction`](crate::Instruction)'s fresh allocation, or an external constant when analyzing an open region
/// with [`RegionRef::reference_analysis_with_constants`]. [`ReferenceRoot`]s are region-relative meaning that their
/// namespace is the region containing their defining input, instruction, or constant. The derived ordering (i.e.,
/// region inputs, allocations, then constants, each by region and position) is deterministic and independent of any
/// hash map iteration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceRoot {
    /// Reference-typed input of a [`Region`].
    RegionInput {
        /// Region owning the input.
        region: RegionId,

        /// Position of the input in the region's input boundary.
        input_index: usize,
    },

    /// Fresh reference allocation performed by an [`Instruction`](crate::Instruction).
    Allocation {
        /// Allocating instruction.
        instruction: InstructionId,

        /// Output of the allocating instruction defining the root.
        output_index: usize,
    },

    /// External reference named by a constant while analyzing an open region before capture lifting. Constants
    /// naming the same inherited capture or concrete allocation share the first encountered representative.
    Constant {
        /// Representative constant in the analyzed computation closure.
        value: ValueId,
    },
}

impl ReferenceRoot {
    /// Returns the [`RegionId`] of the [`Region`] whose namespace owns this [`ReferenceRoot`].
    #[inline]
    pub fn region(self) -> RegionId {
        match self {
            Self::RegionInput { region, .. } => region,
            Self::Allocation { instruction, .. } => instruction.region(),
            Self::Constant { value } => value.region(),
        }
    }

    /// Translates this [`ReferenceRoot`] through an attached [`Region`]'s input bindings. The region's own inputs
    /// resolve to caller roots, while roots captured from enclosing scopes pass through unchanged. Allocations created
    /// inside the attached region are reported as local so callers can prevent them from escaping through region
    /// outputs. A boundary view input is bound to the complete root of the value it views, so accesses through the view
    /// are attributed to that whole root. This is conservative; consumers that need the viewed coordinates use the view
    /// descriptions instead.
    ///
    /// # Parameters
    ///
    ///   - `attached`: [`RegionId`] of the attached [`Region`] whose input bindings are being substituted.
    ///   - `entering`: Caller [`ReferenceRoot`] for each reference input of the attached region, or [`None`] for
    ///     non-reference inputs. A [`Traversal`] must have validated these bindings before calling this function.
    fn substitute(self, attached: RegionId, entering: &[Option<ReferenceRoot>]) -> ReferenceSubstitution {
        match self {
            // Every reference-typed input of the attached region is bound before the region is analyzed, so the
            // binding exists by construction.
            Self::RegionInput { region, input_index } if region == attached => {
                ReferenceSubstitution::Caller(entering[input_index].unwrap())
            }
            Self::RegionInput { .. } | Self::Constant { .. } => ReferenceSubstitution::Caller(self),
            Self::Allocation { instruction, .. } if instruction.region() == attached => {
                ReferenceSubstitution::Local(instruction)
            }
            Self::Allocation { .. } => ReferenceSubstitution::Caller(self),
        }
    }
}

impl Display for ReferenceRoot {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegionInput { region, input_index } => write!(formatter, "region {region} input {input_index}"),
            Self::Constant { value } => write!(formatter, "constant {} in region {}", value.atom(), value.region()),
            Self::Allocation { instruction, output_index } => {
                write!(formatter, "allocation at {instruction} output {output_index}")
            }
        }
    }
}

/// Result of substituting an attached [`Region`]'s input bindings into a [`ReferenceRoot`].
enum ReferenceSubstitution {
    /// [`ReferenceRoot`] visible to the attaching [`Instruction`](crate::Instruction), either supplied through
    /// a [`Region`] input or captured from an enclosing scope.
    Caller(ReferenceRoot),

    /// Allocation created inside the attached [`Region`] by the given [`Instruction`](crate::Instruction),
    /// which cannot escape through a region output.
    Local(InstructionId),
}

/// A reference access performed directly by an [`Instruction`](crate::Instruction), resolved to the canonical
/// [`ReferenceRoot`] in the namespace of the [`Region`] containing that instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAccess {
    /// [`InstructionId`] of the [`Instruction`](crate::Instruction) performing the access.
    instruction: InstructionId,

    /// Input index in the access [`Instruction`](crate::Instruction) that corresponds to the reference.
    input_index: usize,

    /// Canonical [`ReferenceRoot`] reached by this [`ReferenceAccess`].
    root: ReferenceRoot,

    /// [`ReferenceAccessMode`] of this [`ReferenceAccess`].
    mode: ReferenceAccessMode,
}

impl ReferenceAccess {
    /// Creates a new [`ReferenceAccess`].
    pub const fn new(
        instruction: InstructionId,
        input_index: usize,
        root: ReferenceRoot,
        mode: ReferenceAccessMode,
    ) -> Self {
        Self { instruction, input_index, root, mode }
    }

    /// Returns the [`InstructionId`] of the [`Instruction`](crate::Instruction) performing the access.
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the input index in the access [`Instruction`](crate::Instruction) that corresponds to the reference.
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the canonical [`ReferenceRoot`] reached by this [`ReferenceAccess`].
    pub const fn root(self) -> ReferenceRoot {
        self.root
    }

    /// Returns the [`ReferenceAccessMode`] of this [`ReferenceAccess`].
    pub const fn mode(self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Position at which the [`Instruction`](crate::Instruction) of a [`ReferenceAliasEdge`] defines the aliasing value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceAliasPosition {
    /// The aliasing value is the output of the [`Instruction`](crate::Instruction) at this index,
    /// whose description a [`ReferenceViewOperation`](crate::ReferenceViewOperation) reports through
    /// [`reference_view`](crate::ReferenceViewOperation::reference_view).
    Output(usize),

    /// The aliasing value is input `input_index` of the [`Region`] attached at `region_index` of the
    /// [`Instruction`](crate::Instruction). A boundary view the instruction's operation creates from the operation
    /// input named by the `View` variant of [`Operation::input_region_provenance`], whose description
    /// a [`ReferenceViewOperation`](crate::ReferenceViewOperation) reports through
    /// [`region_input_view`](crate::ReferenceViewOperation::region_input_view).
    RegionInput {
        /// Position of the attached region among the instruction's regions.
        region_index: usize,

        /// Reference-typed input of the attached region.
        input_index: usize,
    },
}

/// Reference alias edge that defines one reference-typed value from another reference-typed value (i.e., that defines
/// a view). Edges are recorded for [`ReferenceAlias`](crate::ReferenceAlias) outputs and for outputs constrained by
/// [`Operation::reference_output_identity_input`], which are identity edges from the constrained input, both of which
/// connect values of the same [`Region`], and for region inputs that the attaching operation creates as boundary views
/// (i.e., [`ReferenceAliasPosition::RegionInput`]), which connect a nested region input to an operand of the attaching
/// [`Instruction`](crate::Instruction) in the parent region. Narrowing is _transitive_ meaning that an identity alias
/// of a derived view still represents only that view, and so [`narrows`](Self::narrows) describes the complete chain
/// from the root to the aliasing value rather than only this edge's own kind.
///
/// A shared region is analyzed once, so the edge of a boundary view is attachment-independent in every respect except
/// its [`instruction`](Self::instruction) and [`source`](Self::source), which name the attaching instruction and its
/// operand of the attachment that first reached the region; the analysis rejects a shared region whose attachments
/// disagree on which inputs are boundary views. Consumers that need per-attachment data use
/// [`ReferenceAnalysis::region_input_bindings`], which records every attachment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAliasEdge {
    /// [`Instruction`](crate::Instruction) defining the aliasing value.
    instruction: InstructionId,

    /// [`ReferenceAliasPosition`] at which the defining [`Instruction`](crate::Instruction)
    /// defines the aliasing value.
    position: ReferenceAliasPosition,

    /// [`ValueId`] of the reference-typed value the alias that this [`ReferenceAliasEdge`]
    /// corresponds to is derived from.
    source: ValueId,

    /// [`ReferenceAliasKind`] of this [`ReferenceAliasEdge`].
    kind: ReferenceAliasKind,

    /// Refer to [`Self::narrows`] for information on this field.
    narrows: bool,
}

impl ReferenceAliasEdge {
    /// Creates a new [`ReferenceAliasEdge`].
    pub const fn new(
        instruction: InstructionId,
        position: ReferenceAliasPosition,
        source: ValueId,
        kind: ReferenceAliasKind,
        narrows: bool,
    ) -> Self {
        Self { instruction, position, source, kind, narrows }
    }

    /// Returns the [`Instruction`](crate::Instruction) defining the aliasing value.
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the [`ReferenceAliasPosition`] at which the defining [`Instruction`](crate::Instruction)
    /// defines the aliasing value.
    pub const fn position(self) -> ReferenceAliasPosition {
        self.position
    }

    /// Returns the [`ValueId`] of the reference-typed value the alias that this [`ReferenceAliasEdge`]
    /// corresponds to is derived from.
    pub const fn source(self) -> ValueId {
        self.source
    }

    /// Returns the [`ReferenceAliasKind`] of this [`ReferenceAliasEdge`].
    pub const fn kind(self) -> ReferenceAliasKind {
        self.kind
    }

    /// Returns whether this [`ReferenceAliasEdge`] or an earlier edge in the source's alias chain creates a view.
    /// This is `true` for identity edges from existing views, for example.
    pub const fn narrows(self) -> bool {
        self.narrows
    }
}

/// Binding of a reference-typed input of an attached [`Region`] to the caller root it denotes for a particular
/// attachment. A shared region attached by several [`Instruction`](crate::Instruction)s has one binding per attachment,
/// so nested records stay in the nested region's own namespace, and consumers substitute them through these bindings.
/// The binding also records whether [`Operation::input_region_provenance`] says that the input is a boundary view the
/// attaching operation creates from its input rather than a complete forwarded value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceRegionInputBinding {
    /// [`Instruction`](crate::Instruction) attaching the [`Region`].
    instruction: InstructionId,

    /// Position of the attached [`Region`] among the [`Instruction`](crate::Instruction)'s regions.
    region_index: usize,

    /// [`ValueId`] of the reference-typed input value of the attached [`Region`].
    input: ValueId,

    /// [`ReferenceRoot`] the input denotes, in the namespace of the [`Region`]
    /// containing the [`Instruction`](crate::Instruction).
    root: ReferenceRoot,

    /// Refer to [`Self::is_view`] for information on this field.
    is_view: bool,
}

impl ReferenceRegionInputBinding {
    /// Creates a new [`ReferenceRegionInputBinding`].
    pub const fn new(
        instruction: InstructionId,
        region_index: usize,
        input: ValueId,
        root: ReferenceRoot,
        is_view: bool,
    ) -> Self {
        Self { instruction, region_index, input, root, is_view }
    }

    /// Returns the [`Instruction`](crate::Instruction) attaching the [`Region`].
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the position of the attached [`Region`] among the [`Instruction`](crate::Instruction)'s regions.
    pub const fn region_index(self) -> usize {
        self.region_index
    }

    /// Returns the [`ValueId`] of the reference-typed input value of the attached [`Region`].
    pub const fn input(self) -> ValueId {
        self.input
    }

    /// Returns the [`ReferenceRoot`] the input denotes, in the namespace of the [`Region`]
    /// containing the [`Instruction`](crate::Instruction).
    pub const fn root(self) -> ReferenceRoot {
        self.root
    }

    /// Returns whether the input is a boundary view of its root that the attaching operation creates from its operand,
    /// as opposed to a forwarded complete value handle of that root.
    pub const fn is_view(self) -> bool {
        self.is_view
    }
}

/// Transitive [`ReferenceAccessMode`]s of one [`Instruction`](crate::Instruction). This contains the access modes
/// involved for each [`ReferenceRoot`], directly or anywhere inside its attached [`Region`] closure, expressed in the
/// namespace of the region containing the instruction. Nested region inputs are substituted through their bindings,
/// and allocations local to nested regions are dropped, so a caller sees exactly which of its own roots the instruction
/// touches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceTransitiveAccess {
    /// [`ReferenceAccessMode`]s performed on each [`ReferenceRoot`], in canonical root order.
    access_modes: BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>>,
}

impl ReferenceTransitiveAccess {
    /// Returns the [`ReferenceAccessMode`]s performed on each [`ReferenceRoot`], in canonical root order.
    #[inline]
    pub fn access_modes(&self) -> &BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>> {
        &self.access_modes
    }

    /// Returns the [`ReferenceAccessMode`]s performed on `root`, in [`ReferenceAccessMode`] declaration order.
    #[inline]
    pub fn access_modes_for(&self, root: ReferenceRoot) -> impl '_ + Iterator<Item = ReferenceAccessMode> {
        self.access_modes.get(&root).into_iter().flatten().copied()
    }

    /// Returns the accessed [`ReferenceRoot`]s, in canonical root order.
    #[inline]
    pub fn roots(&self) -> impl '_ + Iterator<Item = ReferenceRoot> {
        self.access_modes.keys().copied()
    }

    /// Returns whether `root` is accessed with any of the [`ReferenceAccessMode::Write`],
    /// [`ReferenceAccessMode::ReadWrite`], and [`ReferenceAccessMode::Accumulate`] modes.
    #[inline]
    pub fn is_mutated(&self, root: ReferenceRoot) -> bool {
        self.access_modes_for(root).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate
            )
        })
    }

    /// Returns whether the corresponding [`Instruction`](crate::Instruction) accesses no [`ReferenceRoot`] at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.access_modes.is_empty()
    }
}

// TODO(eaplatanios): Review from this point onwards.

/// Reference topology, access, and lifetime analysis of one [`Region`] computation closure. Transform rules, boundary
/// validation, diagnostics, and lowering share this analysis through [`RegionRef::reference_analysis`]. It runs when
/// requested and is retained in the region's transform cache; constructing a program does not require this full
/// traversal. Reference discharge shares the traversal, but supplies caller identities and computes an uncached
/// boundary summary instead of treating formal inputs as independent roots.
///
/// The analysis resolves every reference-typed value of the closure to exactly one canonical [`ReferenceRoot`] in the
/// namespace of the region containing it: the region's own reference-typed inputs, its own allocations, and the
/// capture roots it inherits from an enclosing capture scope (the first `capture_count` inputs of the analyzed region,
/// or the fresh prefix an operation declares through [`Operation::region_capture_input_count`]). Nested regions are
/// analyzed in their own namespace, shared regions exactly once; each attachment records one
/// [`ReferenceRegionInputBinding`] per reference-typed region input, and the attaching instruction's
/// [`ReferenceTransitiveAccess`] summary is expressed in the caller's namespace with nested-local allocations dropped.
///
/// Along the way it enforces the reference model: operation effect declarations and region hooks must be well-formed,
/// only complete-value handles cross region boundaries (a derived view neither enters nor leaves an attached region
/// unless the attaching operation itself creates it for a region input, as described in the module documentation), a
/// reference-typed output of a region-carrying operation must preserve its identity-constrained input root or be
/// forwarded consistently from region outputs that are not nested-local allocations, attached regions may only perform
/// the access modes their operation permits on entering roots, and consumption must go through a complete-value handle
/// in the root's allocating region with no later access in program order. External roots are borrowed by default and
/// cannot be consumed. Internal callers may explicitly transfer ownership of selected non-capture entry inputs; those
/// inputs may be consumed in the entry region, but never through a view or in an attached region.
///
/// Every accessor is deterministic: roots, values, and summaries are stored in ordered maps, and accesses and bindings
/// in program order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceAnalysis {
    /// Analyzed region returned by [`Self::region`].
    region: RegionId,

    /// Roots returned by [`Self::roots`], with their external sources, transitive access modes, and consumers.
    roots: BTreeMap<ReferenceRoot, RootRecord>,

    /// Values returned by [`Self::values`], with their roots, view status, and aliases.
    values: BTreeMap<ValueId, ValueRecord>,

    /// Refer to [`Self::accesses`].
    accesses: Vec<ReferenceAccess>,

    /// Refer to [`Self::region_input_bindings`].
    region_input_bindings: Vec<ReferenceRegionInputBinding>,

    /// Instruction summaries returned by [`Self::transitive_access`].
    transitive_accesses: BTreeMap<InstructionId, ReferenceTransitiveAccess>,

    /// Refer to [`Self::output_roots`].
    output_roots: Vec<Option<ReferenceRoot>>,
}

impl ReferenceAnalysis {
    /// Analyzes the computation closure of `region` in program order and returns its
    /// [`ReferenceAnalysis`]. Reference-typed inputs become [`ReferenceRoot::RegionInput`] roots, classified by
    /// [`ReferenceSource::from_flat_input_index`] relative to the capture prefix in `capture_scope`. Reference-typed
    /// constants resolve to capture positions through [`Value::capture_index`]. `resolve_constants` additionally
    /// permits concrete constants and unbound inherited captures to become external roots. Explicit nested capture
    /// scopes remain checked against their declared input prefixes.
    ///
    /// Attached [`RegionRole::Computation`] regions are analyzed recursively and shared regions are analyzed exactly
    /// once. Dormant [`RegionRole::Rule`] regions are skipped.
    ///
    /// # Parameters
    ///
    ///   - `region`: [`Region`] whose computation closure is analyzed.
    ///   - `capture_scope`: Number of leading inputs binding lifted captures, or `None` when the region inherits
    ///     captures from an unknown outer scope. An explicit zero keeps capture lookup strict.
    ///   - `resolve_constants`: Whether concrete reference constants can become external roots. Unbound inherited
    ///     capture constants may also become external roots when `capture_scope` is `None`.
    ///   - `consumable_inputs`: Entry input indices whose reference ownership is transferred to the analyzed region.
    ///     Captures remain borrowed even if their indices occur in this list.
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceAnalysisError`] naming the first violated rule in program order.
    fn new<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        capture_scope: Option<usize>,
        resolve_constants: bool,
        consumable_inputs: &[usize],
    ) -> Result<Self, ReferenceAnalysisError> {
        let capture_count = capture_scope.unwrap_or(0);
        let input_ids = region.input_ids();
        if capture_count > input_ids.len() {
            return Err(ReferenceAnalysisError::InvalidCaptureScope {
                region: region.id(),
                message: format!(
                    "the capture prefix of {} inputs exceeds the region's {} inputs",
                    capture_count,
                    input_ids.len(),
                ),
            });
        }
        let scope = input_ids[..capture_count]
            .iter()
            .enumerate()
            .map(|(input_index, input)| {
                region.atoms()[input.index()]
                    .r#type()
                    .is_reference()
                    .then_some(ReferenceRoot::RegionInput { region: region.id(), input_index })
            })
            .collect::<Rc<[Option<ReferenceRoot>]>>();
        let mut traversal = Traversal::new(region, capture_count, consumable_inputs);
        traversal.constant_scope = (resolve_constants && capture_scope.is_none()).then(|| Rc::clone(&scope));
        traversal.resolve_constants = resolve_constants;
        let summary = traversal.analyze_region(region, scope, vec![None; input_ids.len()].into(), None)?;
        traversal.analysis.output_roots =
            summary.outputs.into_iter().map(|output| output.map(|(root, _)| root)).collect();
        Ok(traversal.analysis)
    }

    /// Summarizes a region under supplied reference identities instead of assuming distinct formal inputs.
    /// This internal path shares structural validation with cached analysis, but analyzes each attachment separately:
    /// aliases and capture bindings may differ between callers. Unused constants are ignored, as in region replay.
    ///
    /// # Parameters
    ///
    ///   - `region`: Region whose computation closure is summarized.
    ///   - `inputs`: Root bound to each reference input, or `None` for a non-reference input. The caller validates
    ///     their count and types. Equal roots denote aliases, and all supplied roots represent borrowed references.
    ///   - `captures`: Active capture scope, already rebound to an explicit input prefix when the owner declares one.
    ///   - `capture_count`: Number of leading inputs in that explicit prefix, or zero for inherited captures. The
    ///     caller validates the prefix length. This preserves capture/input positions in borrowing diagnostics.
    ///
    /// Supplied roots must be region inputs or constants, never local allocations. Allocation roots are created by
    /// the traversal so that consumption and escape checks retain their defining region. The result is not cached
    /// and does not expose the per-value records, which can differ between attachments of a shared region.
    pub(super) fn summarize_boundary<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        inputs: &[Option<ReferenceRoot>],
        captures: &[Option<ReferenceRoot>],
        capture_count: usize,
    ) -> Result<RegionSummary, ReferenceAnalysisError> {
        let mut traversal = Traversal::new(region, capture_count, &[]);
        traversal.analyze_region(region, captures.into(), vec![None; inputs.len()].into(), Some(inputs))
    }

    /// Returns the [`RegionId`] of the analyzed region.
    #[inline]
    pub fn region(&self) -> RegionId {
        self.region
    }

    /// Returns every root of the closure, including roots of nested regions, in canonical root order.
    #[inline]
    pub fn roots(&self) -> impl Iterator<Item = ReferenceRoot> + '_ {
        self.roots.keys().copied()
    }

    /// Returns every reference-typed value of the closure, including values of nested regions, in canonical
    /// [`ValueId`] order.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.values.keys().copied()
    }

    /// Returns the root that the reference-typed `value` denotes, in the namespace of the region containing it, or
    /// [`None`] when `value` is not a reference-typed value of the closure.
    #[inline]
    pub fn root_of(&self, value: ValueId) -> Option<ReferenceRoot> {
        self.values.get(&value).map(|record| record.root)
    }

    /// Returns the logical external source of `root` when it is a reference-typed input of the analyzed region, and
    /// [`None`] for allocations, external constants, and inputs of nested regions.
    #[inline]
    pub fn external_source(&self, root: ReferenceRoot) -> Option<ReferenceSource> {
        self.roots.get(&root).and_then(|record| record.source)
    }

    /// Returns every direct access of the closure, in program order within each region.
    #[inline]
    pub fn accesses(&self) -> &[ReferenceAccess] {
        self.accesses.as_slice()
    }

    /// Returns every access mode performed on `root`, directly or transitively through nested regions, in
    /// [`ReferenceAccessMode`] declaration order.
    #[inline]
    pub fn access_modes(&self, root: ReferenceRoot) -> impl Iterator<Item = ReferenceAccessMode> + '_ {
        self.roots.get(&root).into_iter().flat_map(|record| record.modes.iter().copied())
    }

    /// Returns whether any statically reachable access writes, swaps, or accumulates into `root`. This is deliberately
    /// conservative across structured control flow: a write in either branch of a condition or in a loop body counts
    /// even when execution may never take that path.
    #[inline]
    pub fn is_mutated(&self, root: ReferenceRoot) -> bool {
        self.access_modes(root).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate
            )
        })
    }

    /// Returns the alias edge defining `value` from another reference-typed value, or [`None`] when `value` is a root
    /// handle, a capture constant, a forwarded region output, or not a reference-typed value of the closure.
    #[inline]
    pub fn alias(&self, value: ValueId) -> Option<ReferenceAliasEdge> {
        self.values.get(&value).and_then(|record| record.alias)
    }

    /// Returns whether `value` is a derived view of its root (i.e., whether its alias chain contains a
    /// [`ReferenceAliasKind::View`] edge). Root handles and unknown values are not views.
    #[inline]
    pub fn is_view(&self, value: ValueId) -> bool {
        self.values.get(&value).is_some_and(|record| record.narrows)
    }

    /// Returns every attached-region input binding of the closure, in program order of the attaching instructions.
    #[inline]
    pub fn region_input_bindings(&self) -> &[ReferenceRegionInputBinding] {
        self.region_input_bindings.as_slice()
    }

    /// Returns the transitive access summary of `instruction`, or [`None`] when the instruction accesses no root,
    /// directly or through its attached regions.
    #[inline]
    pub fn transitive_access(&self, instruction: InstructionId) -> Option<&ReferenceTransitiveAccess> {
        self.transitive_accesses.get(&instruction)
    }

    /// Returns the instruction that consumed `root`, or [`None`] when `root` is never consumed.
    #[inline]
    pub fn consumer(&self, root: ReferenceRoot) -> Option<InstructionId> {
        self.roots.get(&root).and_then(|record| record.consumer)
    }

    /// Returns the root denoted by each output of the analyzed region, with [`None`] for value outputs. The analysis
    /// does not judge these outputs: a kernel boundary rejects every reference output, while a program boundary may
    /// forward entering roots, so each consumer applies its own rule.
    #[inline]
    pub fn output_roots(&self) -> &[Option<ReferenceRoot>] {
        self.output_roots.as_slice()
    }
}

impl<'r, V: Value, O: Operation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Returns the [`ReferenceAnalysis`] of this [`Region`]'s closure, retained in the region's transform cache so
    /// that kernel validation and transform rules consulting the same closure share one analysis. Discharge shares
    /// the traversal through a separate uncached entry point that supplies its boundary identities.
    /// The analysis is a pure structural function of the closure and `capture_count`, because reference-typed capture
    /// constants resolve through [`Value::capture_index`], and it is keyed by the closure's region identifiers as
    /// well, so a topology-preserving import that renumbers regions derives its own entry instead of being served
    /// identifiers from another arena. Refer to [`ReferenceAnalysis`] for the analysis semantics and validation rules.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs of this region that originate in a lifted capture table.
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceAnalysisError`] naming the first violated rule in program order. A failed analysis is
    /// not retained.
    pub fn reference_analysis(self, capture_count: usize) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        self.reference_analysis_with_arguments(&ReferenceAnalysisTransformArguments::new(
            self,
            Vec::new(),
            Some(capture_count),
            false,
        ))
    }

    /// Analyzes an open computation region whose inherited captures have not been lifted into an input prefix.
    /// Uses the canonical traversal, preserving view, lifetime, and nested-boundary validation. External constants
    /// remain distinguishable from explicit inputs, which lets transforms propagate input activity correctly.
    /// Concrete allocations and inherited capture indices are canonicalized independently. Explicit capture scopes
    /// declared by nested operations still bind only their declared input prefix, including an explicitly empty scope.
    ///
    /// # Errors
    ///
    /// Returns the first reference-analysis error in program order. Unbound inherited captures are represented as
    /// [`ReferenceRoot::Constant`], while an invalid capture index inside an explicitly rebound scope remains an error.
    pub fn reference_analysis_with_constants(self) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        self.reference_analysis_with_capture_scope(None)
    }

    /// Runs constant-aware analysis with the explicit capture prefix declared by an attaching operation, or with
    /// inherited captures when the operation declares no prefix. Concrete reference constants remain external roots.
    pub(crate) fn reference_analysis_with_capture_scope(
        self,
        capture_scope: Option<usize>,
    ) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        let arguments = ReferenceAnalysisTransformArguments::new(self, Vec::new(), capture_scope, true);
        self.reference_analysis_with_arguments(&arguments)
    }

    /// Analyzes a region whose caller transfers ownership of the listed input references to it. Those inputs may
    /// be consumed directly in this region; consumption through a view or inside an attached region remains invalid.
    /// Captures remain borrowed even if listed in `consumable_inputs`. The ownership list participates in the cache
    /// key so this analysis cannot bypass validation for callers using the default borrowed-input contract.
    pub(crate) fn reference_analysis_with_consumable_inputs(
        self,
        capture_count: usize,
        consumable_inputs: Vec<usize>,
    ) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        let arguments = ReferenceAnalysisTransformArguments::new(self, consumable_inputs, Some(capture_count), false);
        self.reference_analysis_with_arguments(&arguments)
    }

    /// Returns the [`ReferenceAnalysis`] of this [`Region`]'s closure under the already-derived cache key `arguments`.
    /// Refer to the documentation of [`reference_analysis`](Self::reference_analysis) for the analysis and its cache
    /// identity. Overlays derived from the analysis under the same key (e.g., the retained
    /// [`ReferenceViewAnalysis`](crate::programs::references::ReferenceViewAnalysis)) call this so that the closure is
    /// walked once per derivation rather than once more to rebuild the key.
    pub(crate) fn reference_analysis_with_arguments(
        self,
        arguments: &ReferenceAnalysisTransformArguments,
    ) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        let artifact = self.transform::<ReferenceAnalysisTransform, _, ReferenceAnalysisError>(
            arguments.clone(),
            |region, arguments| {
                let analysis = ReferenceAnalysis::new(
                    region,
                    arguments.capture_scope,
                    arguments.resolve_constants,
                    &arguments.consumable_inputs,
                )?;
                Ok(TransformArtifact::new(Vec::new(), Arc::new(analysis)))
            },
        )?;
        let (programs, analysis) = artifact.into_parts();
        assert!(programs.is_empty(), "reference analysis transform retained a program");
        Ok(analysis)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Analyzes the references of this [`Program`]'s entry region closure through the retained analysis of its entry
    /// region. Refer to the documentation of [`RegionRef::reference_analysis`] for more information.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs that originate in a lifted capture table.
    #[inline]
    pub fn reference_analysis(&self, capture_count: usize) -> Result<Arc<ReferenceAnalysis>, ReferenceAnalysisError> {
        self.entry_region_ref().reference_analysis(capture_count)
    }
}

/// Per-root record of a [`ReferenceAnalysis`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RootRecord {
    /// Logical external source, present only for inputs of the analyzed region.
    source: Option<ReferenceSource>,

    /// Direct and transitive access modes performed on the root.
    modes: BTreeSet<ReferenceAccessMode>,

    /// Instruction that consumed the root, if any.
    consumer: Option<InstructionId>,
}

/// Resolution of one reference-typed value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ValueRecord {
    /// Root the value denotes.
    root: ReferenceRoot,

    /// Whether the value is a derived view of its root.
    narrows: bool,

    /// Alias edge defining the value, if it is an alias.
    alias: Option<ReferenceAliasEdge>,
}

/// Result of analyzing one region, with nested accesses and materialization requirements included. Structural
/// analysis uses the region's own namespace and retains this result; boundary analysis uses caller identities and
/// derives a fresh result for each attachment.
#[derive(Clone, Debug)]
pub(crate) struct RegionSummary {
    /// Capture scope the region was analyzed under: the root bound at each capture position, or [`None`] for a
    /// non-reference value.
    scope: Rc<[Option<ReferenceRoot>]>,

    /// Alias edge for each region input that the attaching instruction creates as a view, or [`None`] for a
    /// non-reference value or a forwarded complete-value handle. Only their shape (i.e., which inputs are views) is
    /// independent of the attaching instruction, so that is what later attachments of a shared region are checked
    /// against.
    boundary: Rc<[Option<ReferenceAliasEdge>]>,

    /// Direct and transitive access modes per root, including the region's local allocations.
    pub(super) accesses: BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>>,

    /// Roots needed when replay materializes instruction inputs or region outputs, including inherited captures.
    /// Collected only for boundary summaries; cached structural analysis does not use this set. Materialization
    /// alone is not a semantic reference access.
    pub(super) reached: BTreeSet<ReferenceRoot>,

    /// Root and narrowing of each region output, or [`None`] for value outputs.
    pub(super) outputs: Vec<Option<(ReferenceRoot, bool)>>,
}

/// One attachment of a nested region, as seen from the attaching instruction.
struct AttachedRegion {
    /// Attached region.
    id: RegionId,

    /// Caller root entering through each region input, or [`None`] for value inputs.
    entering: Vec<Option<ReferenceRoot>>,

    /// Root and narrowing of each region output, in the nested namespace.
    outputs: Vec<Option<(ReferenceRoot, bool)>>,
}

/// Mutable state of one [`ReferenceAnalysis::new`] traversal.
struct Traversal<'r, V: Value, O: Operation<Type = V::Type>> {
    /// Analyzed region, whose inputs are the external roots.
    entry: RegionRef<'r, V, O>,

    /// Number of leading analyzed-region inputs that originate in a lifted capture table.
    capture_count: usize,

    /// Entry input indices that may be consumed directly in the entry region, excluding lifted captures.
    consumable_inputs: Vec<usize>,

    /// Original inherited scope when constants can be resolved before capture lifting.
    constant_scope: Option<Rc<[Option<ReferenceRoot>]>>,

    /// Whether concrete reference constants can be resolved regardless of the active capture scope.
    resolve_constants: bool,

    /// Canonical representatives for inherited capture indices and concrete allocation identities.
    constant_roots: HashMap<(Option<usize>, Option<ReferenceId>), ReferenceRoot>,

    /// Analysis being accumulated.
    analysis: ReferenceAnalysis,

    /// Summaries of the regions analyzed so far, so shared regions are analyzed once.
    summaries: HashMap<RegionId, RegionSummary>,
}

impl<'r, V: Value, O: Operation<Type = V::Type>> Traversal<'r, V, O> {
    /// Creates an empty traversal. Entry input ownership is independent of whether boundary bindings are supplied.
    fn new(region: RegionRef<'r, V, O>, capture_count: usize, consumable_inputs: &[usize]) -> Self {
        Self {
            entry: region,
            capture_count,
            consumable_inputs: consumable_inputs.to_vec(),
            constant_scope: None,
            resolve_constants: false,
            constant_roots: HashMap::new(),
            analysis: ReferenceAnalysis {
                region: region.id(),
                roots: BTreeMap::new(),
                values: BTreeMap::new(),
                accesses: Vec::new(),
                region_input_bindings: Vec::new(),
                transitive_accesses: BTreeMap::new(),
                output_roots: Vec::new(),
            },
            summaries: HashMap::new(),
        }
    }

    /// Analyzes `region` under `scope` with the boundary views in `boundary` (one entry per region input). Structural
    /// analysis reuses a memoized summary; boundary analysis supplies `inputs` and visits every attachment separately
    /// because its caller identities and capture bindings may differ.
    fn analyze_region(
        &mut self,
        region: RegionRef<'r, V, O>,
        scope: Rc<[Option<ReferenceRoot>]>,
        boundary: Rc<[Option<ReferenceAliasEdge>]>,
        inputs: Option<&[Option<ReferenceRoot>]>,
    ) -> Result<RegionSummary, ReferenceAnalysisError> {
        let region_id = region.id();
        if inputs.is_none()
            && let Some(summary) = self.summaries.get(&region_id)
        {
            // An inherited open scope permits unresolved capture constants, whereas an explicitly rebound scope
            // does not, even when both contain the same roots (for example, two empty scopes). Reusing a summary
            // requires the same bindings and the same permission to resolve those constants as external roots.
            if summary.scope != scope
                || self
                    .constant_scope
                    .as_ref()
                    .is_some_and(|initial| Rc::ptr_eq(initial, &summary.scope) != Rc::ptr_eq(initial, &scope))
            {
                return Err(ReferenceAnalysisError::InvalidCaptureScope {
                    region: region_id,
                    message: "shared region is reached under two different capture scopes".to_string(),
                });
            }
            if summary.boundary.iter().map(Option::is_some).ne(boundary.iter().map(Option::is_some)) {
                return Err(ReferenceAnalysisError::InconsistentBoundaryViews {
                    region: region_id,
                    message: "shared region is reached with two different sets of boundary view inputs".to_string(),
                });
            }
            return Ok(summary.clone());
        }
        let is_entry = region_id == self.entry.id();
        let atoms = region.atoms();
        let is_reference = |atom: AtomId| atoms[atom.index()].r#type().is_reference();
        let value_id = |atom: AtomId| ValueId::new(region_id, atom);

        // Reference-typed inputs seed the region's own roots. Only the analyzed region's inputs are external, and an
        // input the attaching operation creates as a boundary view is a narrowing view alias of the operand it was
        // created from, so nothing inside the region can consume it or forward it out.
        for (input_index, input) in region.input_ids().iter().copied().enumerate() {
            if !is_reference(input) {
                continue;
            }
            let root = inputs.map_or(ReferenceRoot::RegionInput { region: region_id, input_index }, |inputs| {
                // Input bindings have been validated by the entry adapter or the attaching instruction.
                inputs[input_index].unwrap()
            });
            let source = is_entry.then(|| ReferenceSource::from_flat_input_index(input_index, self.capture_count));
            let alias = boundary[input_index];
            self.analysis.roots.entry(root).or_insert_with(|| RootRecord { source, ..RootRecord::default() });
            self.analysis.values.insert(value_id(input), ValueRecord { root, narrows: alias.is_some(), alias });
        }

        // Boundary summaries follow replay, which lifts constants only when an instruction input or a region output
        // uses them. Materialization is distinct from access effects: even an ignored argument has to be supplied.
        // Structural analysis validates all constants and needs no materialization scan.
        let materialized = if inputs.is_some() {
            region
                .instructions()
                .iter()
                .flat_map(|instruction| instruction.inputs().iter().copied())
                .chain(region.output_ids().iter().copied())
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };

        // Reference-typed constants resolve through the active capture scope. Materializing one is not an access.
        for (index, atom) in atoms.iter().enumerate() {
            let Atom::Constant(constant) = atom else {
                continue;
            };
            if !constant.r#type().is_reference() {
                continue;
            }
            let atom_id = AtomId::new(index);
            if inputs.is_some() && !materialized.contains(&atom_id) {
                continue;
            }
            let capture_index = constant.capture_index();
            let root = match capture_index.and_then(|index| scope.get(index).copied().flatten()) {
                Some(root) => root,
                None if self.resolve_constants
                    && (capture_index.is_none() && constant.reference_id().is_some()
                        || self.constant_scope.as_ref().is_some_and(|initial| Rc::ptr_eq(initial, &scope))
                            && capture_index.is_some()) =>
                {
                    let key = match capture_index {
                        Some(index) => (Some(index), None),
                        None => (None, constant.reference_id()),
                    };
                    let root =
                        *self.constant_roots.entry(key).or_insert(ReferenceRoot::Constant { value: value_id(atom_id) });
                    self.analysis.roots.entry(root).or_default();
                    root
                }
                None => {
                    let Some(capture_index) = capture_index else {
                        return Err(ReferenceAnalysisError::InvalidReferenceConstant {
                            region: region_id,
                            atom: atom_id,
                        });
                    };
                    return Err(ReferenceAnalysisError::InvalidReferenceCapture {
                        region: region_id,
                        atom: atom_id,
                        capture_index,
                        capture_count: scope.len(),
                    });
                }
            };
            self.analysis.values.insert(value_id(atom_id), ValueRecord { root, narrows: false, alias: None });
        }

        let mut summary = RegionSummary {
            scope: Rc::clone(&scope),
            boundary: Rc::clone(&boundary),
            accesses: BTreeMap::new(),
            reached: BTreeSet::new(),
            outputs: Vec::new(),
        };
        let mut consumed = BTreeMap::<ReferenceRoot, InstructionId>::new();
        for (index, instruction) in region.instructions().iter().enumerate() {
            let id = InstructionId::new(region_id, index);
            let operation = instruction.operation();
            let name = operation.name();
            let malformed = |message: String| ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: name,
                instruction: id,
                message,
            };
            let input_atom = |input_index: usize, role: &str| {
                instruction.inputs().get(input_index).copied().ok_or_else(|| {
                    malformed(format!(
                        "{role} input {input_index} is out of range for an application with {} inputs",
                        instruction.inputs().len(),
                    ))
                })
            };
            let use_after_consume =
                |root: ReferenceRoot, consumer: InstructionId| ReferenceAnalysisError::UseAfterConsume {
                    operation: name,
                    instruction: id,
                    root,
                    consumer,
                    consumer_operation: region.instructions()[consumer.index()].operation().name(),
                };

            // Direct accesses declared by the operation, in declaration order.
            let effects = operation.effects();
            for (input_index, mode) in effects.accesses() {
                let atom = input_atom(input_index, "accessed")?;
                let record = self.resolve(value_id(atom), name, id, input_index)?;
                let root = record.root;
                if let Some(consumer) = consumed.get(&root) {
                    return Err(use_after_consume(root, *consumer));
                }
                if mode.is_consuming() {
                    if record.narrows {
                        return Err(ReferenceAnalysisError::ConsumptionThroughView {
                            operation: name,
                            instruction: id,
                            input_index,
                            root,
                        });
                    }
                    self.validate_consumption(name, id, region_id, root)?;
                    consumed.insert(root, id);
                    self.analysis.roots.entry(root).or_default().consumer = Some(id);
                }
                self.analysis.accesses.push(ReferenceAccess { instruction: id, input_index, root, mode });
                self.record_mode(id, root, mode, &mut summary);
            }

            // Allocations and aliases declared by the operation.
            let classified_output = |output_index: usize| -> Result<AtomId, ReferenceAnalysisError> {
                let atom = instruction.outputs().get(output_index).copied().ok_or_else(|| {
                    malformed(format!(
                        "classified output {output_index} is out of range for an application with {} outputs",
                        instruction.outputs().len(),
                    ))
                })?;
                if !is_reference(atom) {
                    return Err(malformed(format!(
                        "classified output {output_index} has non-reference type `{}`",
                        atoms[atom.index()].r#type(),
                    )));
                }
                Ok(atom)
            };
            for output_index in effects.allocation_output_indices() {
                let atom = classified_output(output_index)?;
                let root = ReferenceRoot::Allocation { instruction: id, output_index };
                self.analysis.roots.insert(root, RootRecord::default());
                self.analysis.values.insert(value_id(atom), ValueRecord { root, narrows: false, alias: None });
            }
            for alias in effects.reference_aliases() {
                let (output_index, input_index, kind) = (alias.output_index(), alias.input_index(), alias.kind());
                let atom = classified_output(output_index)?;
                let source_atom = input_atom(input_index, "aliased")?;
                let source = self.resolve(value_id(source_atom), name, id, input_index)?;
                let narrows = kind == ReferenceAliasKind::View || source.narrows;
                let alias = ReferenceAliasEdge {
                    instruction: id,
                    position: ReferenceAliasPosition::Output(output_index),
                    source: value_id(source_atom),
                    kind,
                    narrows,
                };
                self.analysis
                    .values
                    .insert(value_id(atom), ValueRecord { root: source.root, narrows, alias: Some(alias) });
            }

            // Attached regions are entered through their declared input provenance and analyzed in their own
            // namespace. Their transitive accesses are then substituted into this region's namespace, validated
            // against the operation's region access policy and against earlier consumption, and folded into this
            // instruction's summary.
            let mut attached = Vec::with_capacity(instruction.regions().len());
            for (region_index, attached_id) in instruction.regions().iter().copied().enumerate() {
                // Dormant rule regions are inputs to later transforms rather than executed children of this
                // instruction, exactly as for effects: their reference-typed inputs are bound by the transform that
                // instantiates them rather than by this instruction's operands, so they are neither entered nor
                // folded into this instruction's summary. The placeholder keeps region indices aligned for output
                // provenance, which may only name computation regions.
                if operation.region_role(region_index) == Some(RegionRole::Rule) {
                    attached.push(AttachedRegion { id: attached_id, entering: Vec::new(), outputs: Vec::new() });
                    continue;
                }
                let nested = region
                    .with_id(attached_id)
                    .map_err(|error| malformed(format!("attached region {attached_id} cannot be resolved: {error}")))?;
                let nested_inputs = nested.input_ids();
                let nested_is_reference = |input: AtomId| nested.atoms()[input.index()].r#type().is_reference();
                let mut entering = Vec::with_capacity(nested_inputs.len());
                let mut boundary = Vec::with_capacity(nested_inputs.len());
                for (input_index, input) in nested_inputs.iter().copied().enumerate() {
                    if !nested_is_reference(input) {
                        entering.push(None);
                        boundary.push(None);
                        continue;
                    }

                    // A reference-typed region input is either a forwarded complete-value handle or a view the
                    // operation creates at the boundary from one of its operands. Either way the operand itself must
                    // be a complete-value handle, and the region input is bound to its root.
                    let (supplying_index, view) = match operation.input_region_provenance(region_index, input_index) {
                        None => {
                            return Err(malformed(format!(
                                "reference input {input_index} of region {region_index} has no declared supplying \
                                 input",
                            )));
                        }
                        Some(InputRegionProvenance::Forwarded { input_index }) => (input_index, false),
                        Some(InputRegionProvenance::View { input_index }) => (input_index, true),
                    };
                    let atom = input_atom(supplying_index, "region-supplying")?;
                    let record = self.resolve(value_id(atom), name, id, supplying_index)?;
                    if record.narrows {
                        return Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                            operation: name,
                            instruction: id,
                            region_index,
                            boundary: "input",
                            index: input_index,
                        });
                    }
                    self.analysis.region_input_bindings.push(ReferenceRegionInputBinding {
                        instruction: id,
                        region_index,
                        input: ValueId::new(attached_id, input),
                        root: record.root,
                        is_view: view,
                    });
                    entering.push(Some(record.root));
                    boundary.push(view.then(|| ReferenceAliasEdge {
                        instruction: id,
                        position: ReferenceAliasPosition::RegionInput { region_index, input_index },
                        source: value_id(atom),
                        kind: ReferenceAliasKind::View,
                        narrows: true,
                    }));
                }
                let nested_scope = match operation.region_capture_input_count(region_index) {
                    None => Rc::clone(&scope),
                    Some(count) => {
                        if count > nested_inputs.len() {
                            return Err(ReferenceAnalysisError::InvalidCaptureScope {
                                region: attached_id,
                                message: format!(
                                    "operation `{name}` at {id} declares a capture prefix of {count} inputs but the \
                                     region has {} inputs",
                                    nested_inputs.len(),
                                ),
                            });
                        }
                        nested_inputs[..count]
                            .iter()
                            .enumerate()
                            .map(|(input_index, input)| {
                                nested_is_reference(*input).then(|| {
                                    if inputs.is_some() {
                                        entering[input_index].unwrap()
                                    } else {
                                        ReferenceRoot::RegionInput { region: attached_id, input_index }
                                    }
                                })
                            })
                            .collect()
                    }
                };
                let nested_summary = self.analyze_region(
                    nested,
                    nested_scope,
                    boundary.into(),
                    inputs.is_some().then_some(entering.as_slice()),
                )?;
                for root in &nested_summary.reached {
                    if let ReferenceSubstitution::Caller(root) = root.substitute(attached_id, &entering) {
                        summary.reached.insert(root);
                    }
                }
                for (nested_root, modes) in &nested_summary.accesses {
                    let ReferenceSubstitution::Caller(root) = nested_root.substitute(attached_id, &entering) else {
                        continue;
                    };
                    for mode in modes.iter().copied() {
                        if !operation.allows_reference_access_through_region_input(region_index, mode) {
                            return Err(ReferenceAnalysisError::DisallowedRegionAccess {
                                operation: name,
                                instruction: id,
                                region_index,
                                root,
                                mode,
                            });
                        }
                        if let Some(consumer) = consumed.get(&root) {
                            return Err(use_after_consume(root, *consumer));
                        }
                        self.record_mode(id, root, mode, &mut summary);
                    }
                }
                attached.push(AttachedRegion { id: attached_id, entering, outputs: nested_summary.outputs });
            }

            // A reference-typed output that the operation did not classify preserves a root rather than defining
            // one. It resolves through the declared input identity when there is one, in which case every provenance
            // origin must return exactly that root, and otherwise through the region outputs it forwards, which must
            // all agree.
            for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
                if !is_reference(output) || self.analysis.values.contains_key(&value_id(output)) {
                    continue;
                }
                let provenance = operation.output_region_provenance(output_index);
                let mut record = match operation.reference_output_identity_input(output_index) {
                    Some(input_index) => {
                        let atom = input_atom(input_index, "identity-preserved")?;
                        let source = self.resolve(value_id(atom), name, id, input_index)?;
                        let alias = ReferenceAliasEdge {
                            instruction: id,
                            position: ReferenceAliasPosition::Output(output_index),
                            source: value_id(atom),
                            kind: ReferenceAliasKind::Identity,
                            narrows: source.narrows,
                        };
                        Some(ValueRecord { root: source.root, narrows: source.narrows, alias: Some(alias) })
                    }
                    None => None,
                };
                for origin in provenance {
                    // Validate the declared region and output indices before inspecting the forwarded reference.
                    let region = attached.get(origin.region_index).ok_or_else(|| {
                        malformed(format!(
                            "output {} forwards region {} output {}, but the application attaches {} regions",
                            output_index,
                            origin.region_index,
                            origin.output_index,
                            attached.len(),
                        ))
                    })?;
                    let forwarded = region.outputs.get(origin.output_index).ok_or_else(|| {
                        malformed(format!(
                            "output {} forwards region {} output {}, but that region has {} outputs",
                            output_index,
                            origin.region_index,
                            origin.output_index,
                            region.outputs.len(),
                        ))
                    })?;
                    let Some((root, narrows)) = *forwarded else {
                        return Err(malformed(format!(
                            "output {} forwards region {} output {}, which is not a reference",
                            output_index, origin.region_index, origin.output_index,
                        )));
                    };

                    // A forwarded view cannot cross the region boundary: its root alone does not describe
                    // which coordinates the output references.
                    if narrows {
                        return Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                            operation: name,
                            instruction: id,
                            region_index: origin.region_index,
                            boundary: "output",
                            index: origin.output_index,
                        });
                    }

                    // Resolve region inputs to their caller roots. An allocation created inside the attached
                    // region cannot escape through an output, but an enclosing allocation can be forwarded.
                    let root = match root.substitute(region.id, &region.entering) {
                        ReferenceSubstitution::Caller(root) => root,
                        ReferenceSubstitution::Local(allocation) => {
                            return Err(ReferenceAnalysisError::EscapingLocalAllocation {
                                operation: name,
                                instruction: id,
                                output_index,
                                region_index: origin.region_index,
                                allocation,
                            });
                        }
                    };

                    match record {
                        None => record = Some(ValueRecord { root, narrows: false, alias: None }),
                        Some(source) if source.root != root => {
                            return Err(ReferenceAnalysisError::ReferenceRootMismatch {
                                operation: name,
                                instruction: id,
                                output_index,
                                region_index: origin.region_index,
                                region_output_index: origin.output_index,
                                expected: source.root,
                                actual: root,
                            });
                        }
                        Some(_) => {}
                    }
                }
                let Some(record) = record else {
                    return Err(malformed(format!(
                        "reference output {output_index} has no declared allocation, alias, input identity, \
                         or forwarded region output",
                    )));
                };
                self.analysis.values.insert(value_id(output), record);
            }
        }

        summary.reached.extend(
            materialized
                .into_iter()
                .filter_map(|atom| self.analysis.values.get(&value_id(atom)).map(|record| record.root)),
        );

        // Every reference-typed atom of a sealed region is an input, a constant, or an instruction output, each of
        // which the traversal above either bound or rejected, so the lookup cannot fail here.
        summary.outputs = region
            .output_ids()
            .iter()
            .copied()
            .map(|output| {
                is_reference(output).then(|| {
                    let record = self.analysis.values[&value_id(output)];
                    (record.root, record.narrows)
                })
            })
            .collect();

        if inputs.is_none() {
            self.summaries.insert(region_id, summary.clone());
        }

        Ok(summary)
    }

    /// Returns the resolution of the reference-typed `value`, or the [`ReferenceAnalysisError::UnresolvedReference`]
    /// naming the instruction input that expected a reference.
    #[inline]
    fn resolve(
        &self,
        value: ValueId,
        operation: &'static str,
        instruction: InstructionId,
        input_index: usize,
    ) -> Result<ValueRecord, ReferenceAnalysisError> {
        self.analysis.values.get(&value).copied().ok_or(ReferenceAnalysisError::UnresolvedReference {
            operation,
            instruction,
            input_index,
        })
    }

    /// Rejects consumption outside an allocation's creation region, unless the entry boundary explicitly transfers
    /// ownership of that input to the entry region. This exception does not transfer ownership to attached regions.
    fn validate_consumption(
        &self,
        operation: &'static str,
        instruction: InstructionId,
        region: RegionId,
        root: ReferenceRoot,
    ) -> Result<(), ReferenceAnalysisError> {
        match root {
            ReferenceRoot::Allocation { instruction: allocation, .. } if allocation.region() == region => Ok(()),
            ReferenceRoot::RegionInput { region: root_region, input_index }
                if root_region == self.entry.id()
                    && region == root_region
                    && input_index >= self.capture_count
                    && self.consumable_inputs.contains(&input_index) =>
            {
                Ok(())
            }
            ReferenceRoot::RegionInput { region: root_region, input_index } if root_region == self.entry.id() => {
                Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                    operation,
                    instruction,
                    root,
                    external_source: ReferenceSource::from_flat_input_index(input_index, self.capture_count),
                })
            }
            _ => Err(ReferenceAnalysisError::ConsumptionOutsideCreationScope { operation, instruction, region, root }),
        }
    }

    /// Records that `instruction` performs `mode` on `root`, directly or transitively.
    fn record_mode(
        &mut self,
        instruction: InstructionId,
        root: ReferenceRoot,
        mode: ReferenceAccessMode,
        summary: &mut RegionSummary,
    ) {
        self.analysis.roots.entry(root).or_default().modes.insert(mode);
        self.analysis
            .transitive_accesses
            .entry(instruction)
            .or_default()
            .access_modes
            .entry(root)
            .or_default()
            .insert(mode);
        summary.accesses.entry(root).or_default().insert(mode);
    }
}

// TODO(eaplatanios): Review up to here.

/// [`Region`] [`Transform`] marker for retained [`ReferenceAnalysis`] artifacts.
struct ReferenceAnalysisTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for ReferenceAnalysisTransform {
    type Arguments = ReferenceAnalysisTransformArguments;
    type Artifact = TransformArtifact<V, O, Arc<ReferenceAnalysis>>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// Cache key for a [`ReferenceAnalysisTransform`] result. It includes the analyzed [`Region`]'s closure identifiers,
/// input ownership settings, and capture resolution settings, so requests with different analysis assumptions use
/// separate cache entries.
///
/// The region identifiers are necessary because a [`ReferenceAnalysis`] stores [`RegionId`], [`InstructionId`], and
/// [`ValueId`]s that refer to the analyzed program. Importing that program can preserve its computations and reuse its
/// transform cache while assigning different region identifiers. For example, if an attached region changes from `^0`
/// to `^3`, returning the original analysis would incorrectly report facts about `^0`. Including the identifiers
/// returned by [`RegionRef::region_ids_in_closure`] in their first-encounter order gives the imported program a
/// separate entry when those identifiers change.
///
/// This key is meaningful within its owning transform cache, not as a global identifier for a computation. Different
/// [`Program`]s can use the same region identifiers. Cache reuse is safe because imports share a cache only when they
/// preserve the complete computation closure, apart from renumbering; sealing a region with attached regions creates
/// a fresh cache. Together, these rules ensure that equal keys within one cache refer to the same computations and
/// recorded identifiers, allowing repeated requests to reuse the analysis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ReferenceAnalysisTransformArguments {
    /// Refer to the documentation of [`ReferenceAnalysisTransformArguments::new`] for information on this field.
    regions: Vec<RegionId>,

    /// Refer to the documentation of [`ReferenceAnalysisTransformArguments::new`] for information on this field.
    consumable_inputs: Vec<usize>,

    /// Refer to the documentation of [`ReferenceAnalysisTransformArguments::new`] for information on this field.
    capture_scope: Option<usize>,

    /// Refer to the documentation of [`ReferenceAnalysisTransformArguments::new`] for information on this field.
    resolve_constants: bool,
}

impl ReferenceAnalysisTransformArguments {
    /// Creates a [`ReferenceAnalysisTransformArguments`] key for the retained analysis of `region`'s closure under
    /// the supplied ownership and capture settings. The same key identifies analyses derived from these facts (e.g.,
    /// the retained [`ReferenceViewAnalysis`](crate::ReferenceViewAnalysis)), and so all of them share one cache
    /// identity and one revalidation rule.
    ///
    /// # Parameters
    ///
    ///   - `region`: Borrowed [`Region`] whose closure identifiers are included in the key.
    ///   - `consumable_inputs`: Entry input indices whose ownership is transferred to the region. Captures remain
    ///     borrowed even when their indices occur in this list.
    ///   - `capture_scope`: Number of leading inputs binding lifted captures, or `None` for inherited captures.
    ///     `Some(0)` declares an explicitly empty capture prefix.
    ///   - `resolve_constants`: Whether concrete reference constants may become external roots. Unbound inherited
    ///     capture constants may also become external roots when `capture_scope` is `None`.
    pub(super) fn new<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        consumable_inputs: Vec<usize>,
        capture_scope: Option<usize>,
        resolve_constants: bool,
    ) -> Self {
        Self { regions: region.region_ids_in_closure(), consumable_inputs, capture_scope, resolve_constants }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Display;

    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArraySliceAxis, ArrayType,
        DataType, ReferenceIndexOperation, ReferenceSliceOperation,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::EagerContext;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::{
        AddOperation, ConditionOperation, ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceWriteOperation,
        WhileOperation,
    };
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::ProgramError;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{
        EffectClasses, Effects, ReferenceAccessMode, ReferenceAlias, ReferenceAliasKind, ReferenceEffect,
    };
    use crate::programs::identities::NoIdentity;
    use crate::programs::instructions::{Instruction, InstructionId};
    use crate::programs::operations::Operation;
    use crate::programs::programs::Program;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{OutputRegionProvenance, RegionId, RegionInterface, RegionSlot};
    use crate::programs::types::{Type, TypeError};
    use crate::programs::values::ValueId;

    use super::*;

    /// Minimal generic type universe: opaque indexed values plus references over them.
    #[derive(Clone, Debug, PartialEq)]
    enum TestType {
        Value(u8),
        Reference(Box<ReferenceType<TestType>>),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Value(index) => write!(formatter, "value<{index}>"),
                Self::Reference(reference) => Display::fmt(reference, formatter),
            }
        }
    }

    impl Parameter for TestType {}

    impl Type for TestType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Constant payload of the generic universe. The analysis never materializes values, so the capture reference
    /// stand-in is all it needs.
    type TestValue = CaptureReference<TestType>;

    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    type TestArrayValue = ArrayIrValue<Array>;

    type TestArrayOperation = ArrayIrOperation<Array>;

    /// Minimal generic operation universe: the flat reference language, two call-like operations with inherited and
    /// fresh capture scopes, while-, condition-, and scan-like structured operations (the scan views its stacked
    /// reference operands per iteration at the body boundary), one region operation declaring no input provenance,
    /// and one operation with caller-supplied (possibly malformed) effect declarations.
    #[derive(Clone, Debug)]
    enum TestOperation {
        New,
        Read,
        Write,
        Swap,
        Accumulate,
        Consume,
        View,
        Identity,
        Call,
        CallWithCaptures(usize),
        While,
        Condition,
        Scan { carry_count: usize },
        Opaque,
        Malformed(Effects),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for TestOperation {
        type Type = TestType;

        fn name(&self) -> &'static str {
            match self {
                Self::New => "test.new",
                Self::Read => "test.read",
                Self::Write => "test.write",
                Self::Swap => "test.swap",
                Self::Accumulate => "test.accumulate",
                Self::Consume => "test.consume",
                Self::View => "test.view",
                Self::Identity => "test.identity",
                Self::Call => "test.call",
                Self::CallWithCaptures(_) => "test.call_with_captures",
                Self::While => "test.while",
                Self::Condition => "test.condition",
                Self::Scan { .. } => "test.scan",
                Self::Opaque => "test.opaque",
                Self::Malformed(_) => "test.malformed",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Call | Self::CallWithCaptures(_) => const { &[RegionSlot::computation("callee")] },
                Self::While => const { &[RegionSlot::computation("condition"), RegionSlot::computation("body")] },
                Self::Condition => const { &[RegionSlot::computation("true"), RegionSlot::computation("false")] },
                Self::Scan { .. } | Self::Opaque => const { &[RegionSlot::computation("body")] },
                _ => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[TestType],
            region_interfaces: &[RegionInterface<TestType>],
        ) -> Result<Vec<TestType>, TypeError> {
            let referent = |index: usize| match input_types.get(index) {
                Some(TestType::Reference(reference)) => Ok(reference.referent().clone()),
                _ => Err(TypeError::invalid(format!("`{}` expected a reference at input {index}", self.name()))),
            };
            match self {
                Self::New => Ok(vec![TestType::Reference(Box::new(ReferenceType::new(input_types[0].clone())))]),
                Self::Read | Self::Consume | Self::Swap => Ok(vec![referent(0)?]),
                Self::Write | Self::Accumulate => referent(0).map(|_| Vec::new()),
                Self::View | Self::Identity => referent(0).map(|_| vec![input_types[0].clone()]),
                Self::While => Ok(input_types.to_vec()),
                Self::Call | Self::CallWithCaptures(_) | Self::Condition | Self::Scan { .. } | Self::Opaque => {
                    Ok(region_interfaces[0].output_types().to_vec())
                }
                Self::Malformed(_) => Ok(Vec::new()),
            }
        }

        fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<InputRegionProvenance> {
            match self {
                Self::Call | Self::CallWithCaptures(_) | Self::While => {
                    Some(InputRegionProvenance::Forwarded { input_index })
                }
                Self::Condition => Some(InputRegionProvenance::Forwarded { input_index: input_index + 1 }),
                Self::Scan { carry_count } if input_index < *carry_count => {
                    Some(InputRegionProvenance::Forwarded { input_index })
                }
                Self::Scan { .. } => Some(InputRegionProvenance::View { input_index }),
                _ => None,
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Call | Self::CallWithCaptures(_) | Self::Scan { .. } => {
                    vec![OutputRegionProvenance { region_index: 0, output_index }]
                }
                Self::While => vec![OutputRegionProvenance { region_index: 1, output_index }],
                Self::Condition => vec![
                    OutputRegionProvenance { region_index: 0, output_index },
                    OutputRegionProvenance { region_index: 1, output_index },
                ],
                _ => Vec::new(),
            }
        }

        fn region_capture_input_count(&self, _region_index: usize) -> Option<usize> {
            match self {
                Self::CallWithCaptures(count) => Some(*count),
                _ => None,
            }
        }

        fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
            match self {
                Self::While => Some(output_index),
                Self::Scan { carry_count } => (output_index < *carry_count).then_some(output_index),
                _ => None,
            }
        }

        fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
            !matches!(self, Self::While) || region_index != 0 || mode == ReferenceAccessMode::Read
        }

        fn effects(&self) -> Cow<'_, Effects> {
            let access = |mode| {
                Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Access { input_index: 0, mode }], Vec::new())
                    .unwrap()
            };
            let alias =
                |kind| Effects::new(EffectClasses::NONE, Vec::new(), vec![ReferenceAlias::new(0, 0, kind)]).unwrap();
            let effects = match self {
                Self::New => {
                    Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new())
                        .unwrap()
                }
                Self::Read => access(ReferenceAccessMode::Read),
                Self::Write => access(ReferenceAccessMode::Write),
                Self::Swap => access(ReferenceAccessMode::ReadWrite),
                Self::Accumulate => access(ReferenceAccessMode::Accumulate),
                Self::Consume => access(ReferenceAccessMode::Consume),
                Self::View => alias(ReferenceAliasKind::View),
                Self::Identity => alias(ReferenceAliasKind::Identity),
                Self::Malformed(effects) => return Cow::Borrowed(effects),
                _ => return Cow::Borrowed(Effects::empty()),
            };
            Cow::Owned(effects)
        }
    }

    /// Returns the opaque value type with the provided index.
    fn value_type(index: u8) -> TestType {
        TestType::Value(index)
    }

    /// Returns a reference type over the opaque value type with the provided index.
    fn reference_type(index: u8) -> TestType {
        TestType::Reference(Box::new(ReferenceType::new(TestType::Value(index))))
    }

    /// Returns a reference-typed capture constant naming capture `index`.
    fn capture(index: usize, referent: u8) -> TestValue {
        CaptureReference::new(index, reference_type(referent))
    }

    /// Finalizes `builder` into a flat program returning `outputs`.
    fn build(builder: TestBuilder, outputs: Vec<AtomId>) -> TestProgram {
        let input_count = builder.input_ids().len();
        let output_count = outputs.len();
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
            .unwrap()
    }

    /// Returns the [`InstructionId`] of instruction `index` in region `region`.
    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    /// Returns the [`ValueId`] of atom `atom` in region `region`.
    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    /// Returns the [`ReferenceRoot`] of input `input_index` of region `region`.
    fn input_root(region: usize, input_index: usize) -> ReferenceRoot {
        ReferenceRoot::RegionInput { region: RegionId::new(region), input_index }
    }

    /// Returns the [`ReferenceRoot`] allocated by output `output_index` of instruction `index` in region `region`.
    fn allocation_root(region: usize, index: usize, output_index: usize) -> ReferenceRoot {
        ReferenceRoot::Allocation { instruction: id(region, index), output_index }
    }

    /// Returns the analysis fixture shared by the accessor tests. Region `^0` is a callee that writes and reads its
    /// reference input and reads capture `0` through a capture constant; region `^1` is the entry, whose inputs are
    /// a captured reference `A`, a public reference `B`, and a value, and which allocates `C`, derives a view and an
    /// identity alias of that view, accesses every root, calls the callee on `B`, and finally consumes `C`:
    ///
    /// ```text
    /// ^0: lambda %0:ref<value<1>>, %1:value<2> .
    ///     let test.write %0 %1
    ///         %2:value<1> = test.read %0
    ///         %3:ref<value<0>> = capture 0
    ///         %4:value<0> = test.read %3
    ///     in (%2)
    ///
    /// ^1: lambda %0:ref<value<0>>, %1:ref<value<1>>, %2:value<2> .
    ///     let %3:ref<value<2>> = test.new %2
    ///         %4:ref<value<2>> = test.view %3
    ///         %5:ref<value<2>> = test.identity %4
    ///         %6:value<0> = test.read %0
    ///         test.write %1 %2
    ///         test.accumulate %4 %2
    ///         %7:value<2> = test.swap %5 %2
    ///         %8:value<1> = test.call [^0] %1 %2
    ///         %9:value<2> = test.consume %3
    ///     in (%6, %1, %9)
    /// ```
    fn fixture() -> TestProgram {
        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(1));
        let payload = callee.add_input(value_type(2));
        callee.add_instruction(TestOperation::Write, Vec::new(), vec![reference, payload], None).unwrap();
        let read = callee.add_instruction(TestOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let captured = callee.add_constant(capture(0, 0));
        callee.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let callee = build(callee, vec![read]);

        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let a = builder.add_input(reference_type(0));
        let b = builder.add_input(reference_type(1));
        let payload = builder.add_input(value_type(2));
        let c = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![c], None).unwrap()[0];
        let identity = builder.add_instruction(TestOperation::Identity, Vec::new(), vec![view], None).unwrap()[0];
        let read = builder.add_instruction(TestOperation::Read, Vec::new(), vec![a], None).unwrap()[0];
        builder.add_instruction(TestOperation::Write, Vec::new(), vec![b, payload], None).unwrap();
        builder.add_instruction(TestOperation::Accumulate, Vec::new(), vec![view, payload], None).unwrap();
        builder.add_instruction(TestOperation::Swap, Vec::new(), vec![identity, payload], None).unwrap();
        builder.add_instruction(TestOperation::Call, vec![callee], vec![b, payload], None).unwrap();
        let consumed = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![c], None).unwrap()[0];
        build(builder, vec![read, b, consumed])
    }

    /// Returns the analysis of the [`fixture`] with one lifted capture.
    fn fixture_analysis() -> Arc<ReferenceAnalysis> {
        fixture().reference_analysis(1).unwrap()
    }

    /// Builds a `while`-like program over one reference carry and one value carry. The condition reads the carried
    /// reference (or writes it when `mutating_condition` is set), and the body accumulates into it.
    fn while_program(mutating_condition: bool) -> TestProgram {
        let mut condition = TestBuilder::new();
        let reference = condition.add_input(reference_type(0));
        let counter = condition.add_input(value_type(1));
        let operation = if mutating_condition { TestOperation::Write } else { TestOperation::Read };
        let inputs = if mutating_condition { vec![reference, counter] } else { vec![reference] };
        condition.add_instruction(operation, Vec::new(), inputs, None).unwrap();
        let condition = build(condition, vec![counter]);

        let mut body = TestBuilder::new();
        let reference = body.add_input(reference_type(0));
        let counter = body.add_input(value_type(1));
        body.add_instruction(TestOperation::Accumulate, Vec::new(), vec![reference, counter], None).unwrap();
        let body = build(body, vec![reference, counter]);

        let mut builder = TestBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type(0));
        let counter = builder.add_input(value_type(1));
        let outputs = builder
            .add_instruction(TestOperation::While, vec![condition, body], vec![reference, counter], None)
            .unwrap()
            .to_vec();
        build(builder, outputs)
    }

    /// Builds a `scan`-like program over one reference carry `%0:ref<value<0>>` and one stacked reference operand
    /// `%1:ref<value<1>>` that the scan views per iteration at the boundary of `body`, whose inputs are the carry and
    /// the per-iteration view. When `narrowed_operand` is set, the stacked operand is first narrowed through
    /// `test.view`, so the scan is instruction `^1[1]` instead of `^1[0]`.
    fn stacked_scan_program(body: TestProgram, narrowed_operand: bool) -> TestProgram {
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(reference_type(0));
        let mut stacked = builder.add_input(reference_type(1));
        if narrowed_operand {
            stacked = builder.add_instruction(TestOperation::View, Vec::new(), vec![stacked], None).unwrap()[0];
        }
        let outputs = builder
            .add_instruction(TestOperation::Scan { carry_count: 1 }, vec![body], vec![carry, stacked], None)
            .unwrap()
            .to_vec();
        build(builder, outputs)
    }

    /// Builds a scan body over a reference carry and a per-iteration reference view that reads the view and returns
    /// the carry.
    fn reading_scan_body() -> TestProgram {
        let mut body = TestBuilder::new();
        let carry = body.add_input(reference_type(0));
        let element = body.add_input(reference_type(1));
        body.add_instruction(TestOperation::Read, Vec::new(), vec![element], None).unwrap();
        build(body, vec![carry])
    }

    #[test]
    fn test_reference_analysis_error() {
        let cases = [
            (
                ReferenceAnalysisError::InvalidReferenceDeclaration {
                    operation: "test.malformed",
                    instruction: id(0, 1),
                    message: "accessed input 3 is out of range for an application with 1 inputs".to_string(),
                },
                "operation `test.malformed` at ^0[1] has an invalid reference declaration: accessed input 3 is out of \
                 range for an application with 1 inputs",
            ),
            (
                ReferenceAnalysisError::UnresolvedReference {
                    operation: "test.read",
                    instruction: id(0, 1),
                    input_index: 0,
                },
                "operation `test.read` at ^0[1] uses input 0 as a reference but it resolves to no reference root",
            ),
            (
                ReferenceAnalysisError::InvalidReferenceConstant { region: RegionId::new(2), atom: AtomId::new(3) },
                "region ^2 stores reference-typed constant %3 that names no capture; references enter a program only \
                 through inputs and captures",
            ),
            (
                ReferenceAnalysisError::InvalidReferenceCapture {
                    region: RegionId::new(2),
                    atom: AtomId::new(3),
                    capture_index: 4,
                    capture_count: 1,
                },
                "reference-typed constant %3 in region ^2 names capture 4, which the active capture scope of 1 \
                 captures does not bind to a reference",
            ),
            (
                ReferenceAnalysisError::InvalidCaptureScope {
                    region: RegionId::new(2),
                    message: "the capture prefix of 3 inputs exceeds the region's 1 inputs".to_string(),
                },
                "region ^2 has an invalid capture scope: the capture prefix of 3 inputs exceeds the region's 1 inputs",
            ),
            (
                ReferenceAnalysisError::InconsistentBoundaryViews {
                    region: RegionId::new(2),
                    message: "shared region is reached with two different sets of boundary view inputs".to_string(),
                },
                "region ^2 has inconsistent boundary views: shared region is reached with two different sets of \
                 boundary view inputs",
            ),
            (
                ReferenceAnalysisError::ReferenceRootMismatch {
                    operation: "test.while",
                    instruction: id(2, 0),
                    output_index: 0,
                    region_index: 1,
                    region_output_index: 0,
                    expected: input_root(2, 0),
                    actual: input_root(2, 1),
                },
                "operation `test.while` at ^2[0] requires output 0 to denote region ^2 input 0, but region 1 \
                 returns region ^2 input 1 at output 0",
            ),
            (
                ReferenceAnalysisError::EscapingLocalAllocation {
                    operation: "test.call",
                    instruction: id(1, 0),
                    output_index: 0,
                    region_index: 0,
                    allocation: id(0, 0),
                },
                "operation `test.call` at ^1[0] forwards output 0 from a reference allocated at ^0[0] inside its \
                 attached region 0; a local allocation cannot escape its creation scope",
            ),
            (
                ReferenceAnalysisError::ViewCrossesRegionBoundary {
                    operation: "test.call",
                    instruction: id(1, 1),
                    region_index: 0,
                    boundary: "input",
                    index: 0,
                },
                "operation `test.call` at ^1[1] moves a derived reference view across region 0 input 0; only \
                 complete-value handles cross region boundaries",
            ),
            (
                ReferenceAnalysisError::DisallowedRegionAccess {
                    operation: "test.while",
                    instruction: id(2, 0),
                    region_index: 0,
                    root: input_root(2, 0),
                    mode: ReferenceAccessMode::Write,
                },
                "operation `test.while` at ^2[0] does not allow region 0 to access region ^2 input 0, which enters \
                 the region from its parent, with mode `write`",
            ),
            (
                ReferenceAnalysisError::ConsumptionThroughView {
                    operation: "test.consume",
                    instruction: id(0, 2),
                    input_index: 0,
                    root: allocation_root(0, 0, 0),
                },
                "operation `test.consume` at ^0[2] consumes a derived view of allocation at ^0[0] output 0 through \
                 input 0, but consumption invalidates the complete alias family; consume the root handle instead",
            ),
            (
                ReferenceAnalysisError::ExternalReferenceConsumption {
                    operation: "test.consume",
                    instruction: id(0, 0),
                    root: input_root(0, 0),
                    external_source: ReferenceSource::Capture { index: 0 },
                },
                "operation `test.consume` at ^0[0] consumes external reference region ^0 input 0 (capture 0), which \
                 its caller owns",
            ),
            (
                ReferenceAnalysisError::ConsumptionOutsideCreationScope {
                    operation: "test.consume",
                    instruction: id(0, 0),
                    region: RegionId::new(0),
                    root: input_root(0, 0),
                },
                "operation `test.consume` at ^0[0] consumes region ^0 input 0, which entered region ^0 from its \
                 parent; a reference may only be consumed in the region that allocated it",
            ),
            (
                ReferenceAnalysisError::UseAfterConsume {
                    operation: "test.read",
                    instruction: id(0, 2),
                    root: allocation_root(0, 0, 0),
                    consumer: id(0, 1),
                    consumer_operation: "test.consume",
                },
                "operation `test.read` at ^0[2] accesses allocation at ^0[0] output 0 after `test.consume` at ^0[1] \
                 consumed it",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
            assert_eq!(
                ProgramError::from(error.clone()),
                ProgramError::Reference(crate::programs::references::ReferenceError::Analysis(Box::new(error.clone()))),
            );
            assert_eq!(error.clone(), error);
        }
    }

    #[test]
    fn test_reference_root() {
        let input = input_root(1, 2);
        let allocation = allocation_root(0, 3, 1);
        assert_eq!(input.to_string(), "region ^1 input 2");
        assert_eq!(allocation.to_string(), "allocation at ^0[3] output 1");
        assert_eq!(input.region(), RegionId::new(1));
        assert_eq!(allocation.region(), RegionId::new(0));
        assert_eq!(format!("{input:?}"), "RegionInput { region: RegionId { index: 1 }, input_index: 2 }");

        // Region inputs order before allocations, and both order by region and then by position, so a sorted
        // collection of roots is deterministic regardless of insertion order.
        let roots = BTreeSet::from([allocation, input, input_root(1, 0), input_root(0, 5), allocation_root(0, 1, 0)]);
        assert_eq!(
            roots.into_iter().collect::<Vec<_>>(),
            vec![input_root(0, 5), input_root(1, 0), input, allocation_root(0, 1, 0), allocation],
        );
        assert_ne!(input, allocation);
        assert_eq!(input, input_root(1, 2));
    }

    #[test]
    fn test_reference_access() {
        let access = ReferenceAccess::new(id(1, 3), 2, input_root(1, 0), ReferenceAccessMode::Accumulate);
        assert_eq!(access.instruction(), id(1, 3));
        assert_eq!(access.input_index(), 2);
        assert_eq!(access.root(), input_root(1, 0));
        assert_eq!(access.mode(), ReferenceAccessMode::Accumulate);
        assert_eq!(access, ReferenceAccess::new(id(1, 3), 2, input_root(1, 0), ReferenceAccessMode::Accumulate));
        assert_ne!(access, ReferenceAccess::new(id(1, 3), 2, input_root(1, 0), ReferenceAccessMode::Read));
    }

    #[test]
    fn test_reference_alias_edge() {
        let output = ReferenceAliasPosition::Output(2);
        let edge = ReferenceAliasEdge::new(id(1, 1), output, value(1, 3), ReferenceAliasKind::View, true);
        assert_eq!(edge.instruction(), id(1, 1));
        assert_eq!(edge.position(), output);
        assert_eq!(edge.source(), value(1, 3));
        assert_eq!(edge.kind(), ReferenceAliasKind::View);
        assert!(edge.narrows());
        assert_eq!(edge, ReferenceAliasEdge::new(id(1, 1), output, value(1, 3), ReferenceAliasKind::View, true));
        let other = ReferenceAliasPosition::Output(0);
        assert_ne!(edge, ReferenceAliasEdge::new(id(1, 1), other, value(1, 3), ReferenceAliasKind::View, true));
        assert_ne!(edge, ReferenceAliasEdge::new(id(1, 1), output, value(1, 3), ReferenceAliasKind::Identity, true));

        // A boundary view edge is defined at a region input of the attaching instruction rather than at an output.
        let region_input = ReferenceAliasPosition::RegionInput { region_index: 0, input_index: 1 };
        let boundary = ReferenceAliasEdge::new(id(1, 1), region_input, value(1, 3), ReferenceAliasKind::View, true);
        assert_eq!(boundary.position(), region_input);
        assert_ne!(edge, boundary);
    }

    #[test]
    fn test_reference_region_input_binding() {
        let binding = ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1), false);
        assert_eq!(binding.instruction(), id(1, 7));
        assert_eq!(binding.region_index(), 0);
        assert_eq!(binding.input(), value(0, 0));
        assert_eq!(binding.root(), input_root(1, 1));
        assert!(!binding.is_view());
        assert_eq!(binding, ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1), false));
        assert_ne!(binding, ReferenceRegionInputBinding::new(id(1, 7), 1, value(0, 0), input_root(1, 1), false));
        let view = ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1), true);
        assert!(view.is_view());
        assert_ne!(binding, view);
    }

    #[test]
    fn test_reference_transitive_access() {
        let analysis = fixture_analysis();
        let (a, b) = (input_root(1, 0), input_root(1, 1));

        // The call reaches `A` through the callee's capture constant and `B` through its bound input.
        let summary = analysis.transitive_access(id(1, 7)).unwrap();
        assert_eq!(
            summary.access_modes(),
            &BTreeMap::from([
                (a, BTreeSet::from([ReferenceAccessMode::Read])),
                (b, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write])),
            ]),
        );
        assert_eq!(summary.roots().collect::<Vec<_>>(), vec![a, b]);
        assert_eq!(summary.access_modes_for(a).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
        assert_eq!(
            summary.access_modes_for(b).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write]
        );
        assert_eq!(summary.access_modes_for(allocation_root(1, 0, 0)).count(), 0);
        assert!(!summary.is_mutated(a));
        assert!(summary.is_mutated(b));
        assert!(!summary.is_empty());
        assert!(ReferenceTransitiveAccess::default().is_empty());
        assert_eq!(summary.clone(), *summary);
    }

    #[test]
    fn test_reference_analysis_new() {
        let program = fixture();
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), Some(1), false, &[]).unwrap();
        let (a, b, c, k) = (input_root(1, 0), input_root(1, 1), allocation_root(1, 0, 0), input_root(0, 0));
        assert_eq!(analysis.region(), RegionId::new(1));
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![k, a, b, c]);
        assert_eq!(
            analysis.accesses(),
            &[
                ReferenceAccess::new(id(1, 3), 0, a, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(1, 4), 0, b, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(1, 5), 0, c, ReferenceAccessMode::Accumulate),
                ReferenceAccess::new(id(1, 6), 0, c, ReferenceAccessMode::ReadWrite),
                ReferenceAccess::new(id(0, 0), 0, k, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(0, 1), 0, k, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(0, 2), 0, a, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(1, 8), 0, c, ReferenceAccessMode::Consume),
            ],
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), b, false)]
        );
        assert_eq!(analysis.output_roots(), &[None, Some(b), None]);

        // A region without references analyzes to an empty artifact.
        let mut builder = TestBuilder::new();
        let input = builder.add_input(value_type(0));
        let program = build(builder, vec![input]);
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), Some(0), false, &[]).unwrap();
        assert_eq!(analysis.roots().count(), 0);
        assert_eq!(analysis.accesses(), &[]);
        assert_eq!(analysis.output_roots(), &[None]);
    }

    #[test]
    fn test_reference_analysis_new_resolves_capture_constants_through_inherited_scopes() {
        // Both condition branches inherit the entry scope, so a capture constant in either branch denotes the entry's
        // captured reference directly, without any region input binding.
        let make_branch = |operation: TestOperation| {
            let mut branch = TestBuilder::new();
            let payload = branch.add_input(value_type(1));
            let captured = branch.add_constant(capture(0, 0));
            let inputs =
                if matches!(operation, TestOperation::Write) { vec![captured, payload] } else { vec![captured] };
            branch.add_instruction(operation, Vec::new(), inputs, None).unwrap();
            build(branch, vec![payload])
        };
        let mut builder = TestBuilder::new();
        let true_branch = builder.import_region(make_branch(TestOperation::Read).entry_region_ref());
        let false_branch = builder.import_region(make_branch(TestOperation::Write).entry_region_ref());
        builder.add_input(reference_type(0));
        let predicate = builder.add_input(value_type(9));
        let payload = builder.add_input(value_type(1));
        let outputs = builder
            .add_instruction(TestOperation::Condition, vec![true_branch, false_branch], vec![predicate, payload], None)
            .unwrap()
            .to_vec();
        let program = build(builder, outputs);

        let analysis = program.reference_analysis(1).unwrap();
        let a = input_root(2, 0);
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![a]);
        assert_eq!(analysis.root_of(value(0, 1)), Some(a));
        assert_eq!(analysis.root_of(value(1, 1)), Some(a));
        assert_eq!(analysis.external_source(a), Some(ReferenceSource::Capture { index: 0 }));
        assert_eq!(analysis.region_input_bindings(), &[]);
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write]))]),
        );
        assert!(analysis.is_mutated(a));
        assert_eq!(analysis.root_of(value(2, 0)), Some(a));
        assert_eq!(analysis.root_of(value(2, 1)), None);
    }

    #[test]
    fn test_reference_analysis_new_establishes_fresh_capture_scopes_in_callees() {
        // The callee's first input is its own capture prefix, so its capture constant denotes the callee's region input
        // rather than anything in the entry scope, and only the region input binding connects it to the caller.
        let mut callee = TestBuilder::new();
        let captured_input = callee.add_input(reference_type(0));
        let payload = callee.add_input(value_type(1));
        let captured = callee.add_constant(capture(0, 0));
        callee.add_instruction(TestOperation::Write, Vec::new(), vec![captured, payload], None).unwrap();
        callee.add_instruction(TestOperation::Read, Vec::new(), vec![captured_input], None).unwrap();
        let callee = build(callee, vec![payload]);

        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let a = builder.add_input(reference_type(0));
        let payload = builder.add_input(value_type(1));
        let outputs = builder
            .add_instruction(TestOperation::CallWithCaptures(1), vec![callee], vec![a, payload], None)
            .unwrap()
            .to_vec();
        let program = build(builder, outputs);

        let analysis = program.reference_analysis(0).unwrap();
        let (a, k) = (input_root(1, 0), input_root(0, 0));
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![k, a]);
        assert_eq!(analysis.root_of(value(0, 2)), Some(k));
        assert_eq!(analysis.external_source(a), Some(ReferenceSource::Input { index: 0 }));
        assert_eq!(analysis.external_source(k), None);
        assert_eq!(
            analysis.region_input_bindings(),
            &[ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a, false)]
        );
        assert_eq!(
            analysis.accesses(),
            &[
                ReferenceAccess::new(id(0, 0), 0, k, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(0, 1), 0, k, ReferenceAccessMode::Read),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(1, 0)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write]))]),
        );
        assert_eq!(
            analysis.access_modes(a).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write]
        );
    }

    #[test]
    fn test_reference_analysis_new_substitutes_roots_through_while_and_scan() {
        // A while carry keeps its entering root through the identity constraint, is recorded as an identity alias of
        // the entering value, and its condition and body accesses fold into the loop's summary.
        let analysis = while_program(false).reference_analysis(0).unwrap();
        let a = input_root(2, 0);
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![input_root(0, 0), input_root(1, 0), a]);
        assert_eq!(analysis.root_of(value(2, 2)), Some(a));
        assert_eq!(
            analysis.alias(value(2, 2)),
            Some(ReferenceAliasEdge::new(
                id(2, 0),
                ReferenceAliasPosition::Output(0),
                value(2, 0),
                ReferenceAliasKind::Identity,
                false
            ))
        );
        assert!(!analysis.is_view(value(2, 2)));
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(2, 0), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(2, 0), 1, value(1, 0), a, false),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Accumulate]))]),
        );
        assert_eq!(analysis.output_roots(), &[Some(a), None]);

        // Only the scan's carry prefix forwards and preserves roots; the trailing body input is a sliced element.
        let mut body = TestBuilder::new();
        let carry = body.add_input(reference_type(0));
        let element = body.add_input(value_type(1));
        body.add_instruction(TestOperation::Write, Vec::new(), vec![carry, element], None).unwrap();
        let body = build(body, vec![carry, element]);
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type(0));
        let sequence = builder.add_input(value_type(1));
        let outputs = builder
            .add_instruction(TestOperation::Scan { carry_count: 1 }, vec![body], vec![reference, sequence], None)
            .unwrap()
            .to_vec();
        let program = build(builder, outputs);
        let analysis = program.reference_analysis(0).unwrap();
        let a = input_root(1, 0);
        assert_eq!(analysis.root_of(value(1, 2)), Some(a));
        assert_eq!(
            analysis.alias(value(1, 2)),
            Some(ReferenceAliasEdge::new(
                id(1, 0),
                ReferenceAliasPosition::Output(0),
                value(1, 0),
                ReferenceAliasKind::Identity,
                false
            ))
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a, false)]
        );
        assert_eq!(
            analysis.transitive_access(id(1, 0)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Write]))]),
        );
        assert_eq!(analysis.output_roots(), &[Some(a), None]);
    }

    #[test]
    fn test_reference_analysis_new_substitutes_roots_through_conditions() {
        // Both branches return their entering reference, so the condition output denotes the caller root through
        // provenance alone: it is a complete-value handle with no alias edge.
        let make_branch = |operation: TestOperation| {
            let mut branch = TestBuilder::new();
            let reference = branch.add_input(reference_type(0));
            let payload = branch.add_input(value_type(1));
            let inputs =
                if matches!(operation, TestOperation::Write) { vec![reference, payload] } else { vec![reference] };
            branch.add_instruction(operation, Vec::new(), inputs, None).unwrap();
            build(branch, vec![reference])
        };
        let mut builder = TestBuilder::new();
        let true_branch = builder.import_region(make_branch(TestOperation::Write).entry_region_ref());
        let false_branch = builder.import_region(make_branch(TestOperation::Read).entry_region_ref());
        let predicate = builder.add_input(value_type(9));
        let a = builder.add_input(reference_type(0));
        let payload = builder.add_input(value_type(1));
        let outputs = builder
            .add_instruction(
                TestOperation::Condition,
                vec![true_branch, false_branch],
                vec![predicate, a, payload],
                None,
            )
            .unwrap()
            .to_vec();
        let program = build(builder, outputs);

        let analysis = program.reference_analysis(0).unwrap();
        let a = input_root(2, 1);
        assert_eq!(analysis.root_of(value(2, 3)), Some(a));
        assert_eq!(analysis.alias(value(2, 3)), None);
        assert!(!analysis.is_view(value(2, 3)));
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(2, 0), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(2, 0), 1, value(1, 0), a, false),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write]))]),
        );
        assert_eq!(analysis.output_roots(), &[Some(a)]);
    }

    #[test]
    fn test_reference_analysis_new_keeps_nested_allocations_local() {
        // A callee may allocate, mutate, and consume its own reference. The allocation is a root of the closure with a
        // consumer, but it never reaches the caller's namespace, so the call has no transitive accesses.
        let mut callee = TestBuilder::new();
        let payload = callee.add_input(value_type(1));
        let local = callee.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        callee.add_instruction(TestOperation::Write, Vec::new(), vec![local, payload], None).unwrap();
        let frozen = callee.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap()[0];
        let callee = build(callee, vec![frozen]);

        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let payload = builder.add_input(value_type(1));
        let outputs = builder.add_instruction(TestOperation::Call, vec![callee], vec![payload], None).unwrap().to_vec();
        let program = build(builder, outputs);

        let analysis = program.reference_analysis(0).unwrap();
        let local = allocation_root(0, 0, 0);
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![local]);
        assert_eq!(analysis.consumer(local), Some(id(0, 2)));
        assert_eq!(
            analysis.access_modes(local).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Write, ReferenceAccessMode::Consume]
        );
        assert_eq!(analysis.transitive_access(id(1, 0)), None);
        assert_eq!(analysis.region_input_bindings(), &[]);
        assert_eq!(analysis.output_roots(), &[None]);
    }

    #[test]
    fn test_reference_analysis_new_analyzes_shared_regions_once() {
        // One region attached as both branches is analyzed once, so its accesses appear once, while each attachment
        // records its own binding.
        let mut branch = TestBuilder::new();
        let reference = branch.add_input(reference_type(0));
        branch.add_instruction(TestOperation::Read, Vec::new(), vec![reference], None).unwrap();
        let branch = build(branch, vec![reference]);

        let mut builder = TestBuilder::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(value_type(9));
        let a = builder.add_input(reference_type(0));
        let outputs = builder
            .add_instruction(TestOperation::Condition, vec![branch, branch], vec![predicate, a], None)
            .unwrap()
            .to_vec();
        let program = build(builder, outputs);

        let analysis = program.reference_analysis(0).unwrap();
        let a = input_root(1, 1);
        assert_eq!(
            analysis.accesses(),
            &[ReferenceAccess::new(id(0, 0), 0, input_root(0, 0), ReferenceAccessMode::Read)]
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(1, 0), 1, value(0, 0), a, false),
            ],
        );
        assert_eq!(analysis.output_roots(), &[Some(a)]);
    }

    #[test]
    fn test_reference_analysis_new_binds_boundary_views() {
        // The stacked operand enters the body as a view the scan creates at the boundary: the body input is its own
        // root in the body's namespace, is recorded as a view alias of the operand at a region-input position, and the
        // body's accesses through it are attributed to the whole caller root.
        let analysis = stacked_scan_program(reading_scan_body(), false).reference_analysis(0).unwrap();
        let (a, b) = (input_root(1, 0), input_root(1, 1));
        assert_eq!(analysis.root_of(value(0, 1)), Some(input_root(0, 1)));
        assert!(!analysis.is_view(value(0, 0)));
        assert!(analysis.is_view(value(0, 1)));
        assert_eq!(analysis.alias(value(0, 0)), None);
        assert_eq!(
            analysis.alias(value(0, 1)),
            Some(ReferenceAliasEdge::new(
                id(1, 0),
                ReferenceAliasPosition::RegionInput { region_index: 0, input_index: 1 },
                value(1, 1),
                ReferenceAliasKind::View,
                true,
            )),
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 1), b, true),
            ],
        );
        assert_eq!(
            analysis.accesses(),
            &[ReferenceAccess::new(id(0, 0), 0, input_root(0, 1), ReferenceAccessMode::Read)],
        );
        assert_eq!(
            analysis.transitive_access(id(1, 0)).unwrap().access_modes(),
            &BTreeMap::from([(b, BTreeSet::from([ReferenceAccessMode::Read]))]),
        );
        assert_eq!(analysis.access_modes(b).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
        assert!(analysis.access_modes(a).next().is_none());
        assert_eq!(analysis.output_roots(), &[Some(a)]);
    }

    #[test]
    fn test_reference_analysis_new_analyzes_shared_boundary_views_once() {
        // Two scans attaching one body with the same boundary shape share its analysis: the body input's edge names
        // the first attachment, while each attachment records its own view binding.
        let body = reading_scan_body();
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(reference_type(0));
        let stacked = builder.add_input(reference_type(1));
        let first = builder
            .add_instruction(TestOperation::Scan { carry_count: 1 }, vec![body], vec![carry, stacked], None)
            .unwrap()[0];
        let outputs = builder
            .add_instruction(TestOperation::Scan { carry_count: 1 }, vec![body], vec![first, stacked], None)
            .unwrap()
            .to_vec();
        let analysis = build(builder, outputs).reference_analysis(0).unwrap();
        let (a, b) = (input_root(1, 0), input_root(1, 1));
        assert_eq!(
            analysis.alias(value(0, 1)),
            Some(ReferenceAliasEdge::new(
                id(1, 0),
                ReferenceAliasPosition::RegionInput { region_index: 0, input_index: 1 },
                value(1, 1),
                ReferenceAliasKind::View,
                true,
            )),
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 1), b, true),
                ReferenceRegionInputBinding::new(id(1, 1), 0, value(0, 0), a, false),
                ReferenceRegionInputBinding::new(id(1, 1), 0, value(0, 1), b, true),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(1, 1)).unwrap().access_modes(),
            &BTreeMap::from([(b, BTreeSet::from([ReferenceAccessMode::Read]))]),
        );

        // The same body attached once with a boundary view and once with two forwarded handles has no single set of
        // input records that is correct for both attachments.
        let body = reading_scan_body();
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(reference_type(0));
        let stacked = builder.add_input(reference_type(1));
        builder
            .add_instruction(TestOperation::Scan { carry_count: 1 }, vec![body], vec![carry, stacked], None)
            .unwrap();
        builder.add_instruction(TestOperation::Call, vec![body], vec![carry, stacked], None).unwrap();
        assert!(matches!(
            build(builder, Vec::new()).reference_analysis(0),
            Err(ReferenceAnalysisError::InconsistentBoundaryViews { region, message })
                if region == RegionId::new(0)
                    && message == "shared region is reached with two different sets of boundary view inputs",
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_narrowed_boundary_view_sources() {
        // The operand a boundary view is created from must itself be a complete-value handle.
        assert!(matches!(
            stacked_scan_program(reading_scan_body(), true).reference_analysis(0),
            Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                operation: "test.scan",
                instruction,
                region_index: 0,
                boundary: "input",
                index: 1,
            }) if instruction == id(1, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_consumption_through_boundary_views() {
        let mut body = TestBuilder::new();
        let carry = body.add_input(reference_type(0));
        let element = body.add_input(reference_type(1));
        body.add_instruction(TestOperation::Consume, Vec::new(), vec![element], None).unwrap();
        let body = build(body, vec![carry]);
        assert!(matches!(
            stacked_scan_program(body, false).reference_analysis(0),
            Err(ReferenceAnalysisError::ConsumptionThroughView {
                operation: "test.consume",
                instruction,
                input_index: 0,
                root,
            }) if instruction == id(0, 0) && root == input_root(0, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_forwarding_boundary_views() {
        // The view created at the boundary stays inside the region: returning it is a view leaving the region.
        let mut body = TestBuilder::new();
        let carry = body.add_input(reference_type(0));
        let element = body.add_input(reference_type(1));
        let body = build(body, vec![carry, element]);
        assert!(matches!(
            stacked_scan_program(body, false).reference_analysis(0),
            Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                operation: "test.scan",
                instruction,
                region_index: 0,
                boundary: "output",
                index: 1,
            }) if instruction == id(1, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_malformed_effects() {
        // The checked builder path rejects out-of-range effect declarations itself, so the malformed applications are
        // assembled through the unchecked rebuild hatch.
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type(0));
        let effects = Effects::new(
            EffectClasses::NONE,
            vec![ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Read }],
            Vec::new(),
        )
        .unwrap();
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(effects),
            vec![reference],
            Vec::new(),
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: "test.malformed",
                instruction,
                message,
            })
                if instruction == id(0, 0)
                    && message == "accessed input 3 is out of range for an application with 1 inputs",
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let effects =
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 2 }], Vec::new()).unwrap();
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(effects),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: "test.malformed",
                instruction,
                message,
            })
                if instruction == id(0, 0)
                    && message == "classified output 2 is out of range for an application with 0 outputs",
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let output = builder.add_variable(value_type(1));
        let effects =
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], Vec::new()).unwrap();
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(effects),
            Vec::new(),
            vec![output],
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: "test.malformed",
                instruction,
                message,
            })
                if instruction == id(0, 0) && message == "classified output 0 has non-reference type `value<1>`",
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_unresolved_references() {
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(0));
        let output = builder.add_variable(value_type(0));
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Read,
            vec![payload],
            vec![output],
            Vec::new(),
        ));
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::UnresolvedReference { operation: "test.read", instruction, input_index: 0 })
                if instruction == id(0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_capture_constants_outside_an_empty_scope() {
        // Every constant of this family names a capture position through `Value::capture_index`, so a reference-typed
        // constant in a program that declares no capture prefix is out of scope rather than unresolvable.
        let mut builder = TestBuilder::new();
        let captured = builder.add_constant(capture(0, 0));
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            ReferenceAnalysis::new(program.entry_region_ref(), Some(0), false, &[]),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { region, atom, capture_index: 0, capture_count: 0 })
                if region == RegionId::new(0) && atom == AtomId::new(0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_invalid_reference_captures() {
        // A capture index past the scope is rejected, and so is one whose scope position binds a value.
        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let captured = builder.add_constant(capture(1, 0));
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(1),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { region, atom, capture_index: 1, capture_count: 1 })
                if region == RegionId::new(0) && atom == AtomId::new(1),
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(value_type(0));
        let captured = builder.add_constant(capture(0, 0));
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(1),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { region, atom, capture_index: 0, capture_count: 1 })
                if region == RegionId::new(0) && atom == AtomId::new(1),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_invalid_capture_scopes() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type(0));
        let program = build(builder, vec![reference]);
        assert!(matches!(
            program.reference_analysis(3),
            Err(ReferenceAnalysisError::InvalidCaptureScope { region, message })
                if region == RegionId::new(0)
                    && message == "the capture prefix of 3 inputs exceeds the region's 1 inputs",
        ));

        let mut callee = TestBuilder::new();
        let payload = callee.add_input(value_type(1));
        let callee = build(callee, vec![payload]);
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let payload = builder.add_input(value_type(1));
        // The checked builder path rejects an oversized capture prefix itself, so the malformed application is
        // assembled through the unchecked rebuild hatch.
        let output = builder.add_variable(value_type(1));
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::CallWithCaptures(2),
            vec![payload],
            vec![output],
            vec![callee],
        ));
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidCaptureScope { region, message })
                if region == RegionId::new(0)
                    && message == "operation `test.call_with_captures` at ^1[0] declares a capture prefix of 2 inputs \
                                   but the region has 1 inputs",
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_undeclared_region_input_provenance() {
        let mut body = TestBuilder::new();
        let reference = body.add_input(reference_type(0));
        body.add_instruction(TestOperation::Read, Vec::new(), vec![reference], None).unwrap();
        let body = build(body, Vec::new());
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type(0));
        // The checked builder path rejects the undeclared provenance itself, so the malformed application is assembled
        // through the unchecked rebuild hatch.
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Opaque,
            vec![reference],
            Vec::new(),
            vec![body],
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: "test.opaque",
                instruction,
                message,
            }) if instruction == id(1, 0)
                && message == "reference input 0 of region 0 has no declared supplying input",
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_undeclared_reference_outputs() {
        let mut body = TestBuilder::new();
        let payload = body.add_input(value_type(1));
        let local = body.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let body = build(body, vec![local]);
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let payload = builder.add_input(value_type(1));
        // The checked builder path rejects the undeclared reference output itself, so the malformed application is
        // assembled through the unchecked rebuild hatch.
        let output = builder.add_variable(reference_type(1));
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Opaque,
            vec![payload],
            vec![output],
            vec![body],
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceDeclaration {
                operation: "test.opaque",
                instruction,
                message,
            }) if instruction == id(1, 0)
                && message == "reference output 0 has no declared allocation, alias, input identity, or forwarded \
                               region output",
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_inconsistent_forwarded_roots() {
        let make_branch = |returned: usize| {
            let mut branch = TestBuilder::new();
            let first = branch.add_input(reference_type(0));
            let second = branch.add_input(reference_type(0));
            build(branch, vec![[first, second][returned]])
        };
        let mut builder = TestBuilder::new();
        let true_branch = builder.import_region(make_branch(0).entry_region_ref());
        let false_branch = builder.import_region(make_branch(1).entry_region_ref());
        let predicate = builder.add_input(value_type(9));
        let a = builder.add_input(reference_type(0));
        let b = builder.add_input(reference_type(0));
        builder
            .add_instruction(TestOperation::Condition, vec![true_branch, false_branch], vec![predicate, a, b], None)
            .unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ReferenceRootMismatch {
                operation: "test.condition",
                instruction,
                output_index: 0,
                region_index: 1,
                region_output_index: 0,
                expected,
                actual,
            }) if instruction == id(2, 0) && expected == input_root(2, 1) && actual == input_root(2, 2),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_preserved_reference_root_mismatch() {
        let mut condition = TestBuilder::new();
        let first = condition.add_input(reference_type(0));
        condition.add_input(reference_type(0));
        let predicate = condition.add_instruction(TestOperation::Read, Vec::new(), vec![first], None).unwrap()[0];
        let condition = build(condition, vec![predicate]);
        let mut body = TestBuilder::new();
        let first = body.add_input(reference_type(0));
        let second = body.add_input(reference_type(0));
        let body = build(body, vec![second, first]);
        let mut builder = TestBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let a = builder.add_input(reference_type(0));
        let b = builder.add_input(reference_type(0));
        builder.add_instruction(TestOperation::While, vec![condition, body], vec![a, b], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ReferenceRootMismatch {
                operation: "test.while",
                instruction,
                output_index: 0,
                region_index: 1,
                region_output_index: 0,
                expected,
                actual,
            }) if instruction == id(2, 0) && expected == input_root(2, 0) && actual == input_root(2, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_escaping_local_allocations() {
        let mut callee = TestBuilder::new();
        let payload = callee.add_input(value_type(1));
        let local = callee.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let callee = build(callee, vec![local]);
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let payload = builder.add_input(value_type(1));
        builder.add_instruction(TestOperation::Call, vec![callee], vec![payload], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::EscapingLocalAllocation {
                operation: "test.call",
                instruction,
                output_index: 0,
                region_index: 0,
                allocation,
            }) if instruction == id(1, 0) && allocation == id(0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_views_crossing_region_boundaries() {
        // A view may neither enter an attached region nor be forwarded out of one.
        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(0));
        let callee = build(callee, vec![reference]);
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let a = builder.add_input(reference_type(0));
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![a], None).unwrap()[0];
        builder.add_instruction(TestOperation::Call, vec![callee], vec![view], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                operation: "test.call",
                instruction,
                region_index: 0,
                boundary: "input",
                index: 0,
            }) if instruction == id(1, 1),
        ));

        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(0));
        let view = callee.add_instruction(TestOperation::View, Vec::new(), vec![reference], None).unwrap()[0];
        let callee = build(callee, vec![view]);
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let a = builder.add_input(reference_type(0));
        builder.add_instruction(TestOperation::Call, vec![callee], vec![a], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                operation: "test.call",
                instruction,
                region_index: 0,
                boundary: "output",
                index: 0,
            }) if instruction == id(1, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_disallowed_region_accesses() {
        assert!(matches!(
            while_program(true).reference_analysis(0),
            Err(ReferenceAnalysisError::DisallowedRegionAccess {
                operation: "test.while",
                instruction,
                region_index: 0,
                root,
                mode: ReferenceAccessMode::Write,
            }) if instruction == id(2, 0) && root == input_root(2, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_consumption_through_views() {
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(1));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![local], None).unwrap()[0];
        let output = builder.add_variable(value_type(1));
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Consume,
            vec![view],
            vec![output],
            Vec::new(),
        ));
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ConsumptionThroughView {
                operation: "test.consume",
                instruction,
                input_index: 0,
                root,
            }) if instruction == id(0, 2) && root == allocation_root(0, 0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_external_consumption() {
        let mut builder = TestBuilder::new();
        let a = builder.add_input(reference_type(0));
        let output = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![a], None).unwrap()[0];
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.reference_analysis(1),
            Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: "test.consume",
                instruction,
                root,
                external_source: ReferenceSource::Capture { index: 0 },
            }) if instruction == id(0, 0) && root == input_root(0, 0),
        ));
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: "test.consume",
                instruction,
                root,
                external_source: ReferenceSource::Input { index: 0 },
            }) if instruction == id(0, 0) && root == input_root(0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_consumption_outside_creation_scope() {
        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(0));
        let frozen = callee.add_instruction(TestOperation::Consume, Vec::new(), vec![reference], None).unwrap()[0];
        let callee = build(callee, vec![frozen]);
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let payload = builder.add_input(value_type(0));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        builder.add_instruction(TestOperation::Call, vec![callee], vec![local], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ConsumptionOutsideCreationScope {
                operation: "test.consume",
                instruction,
                region,
                root,
            }) if instruction == id(0, 0) && region == RegionId::new(0) && root == input_root(0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_use_after_consume() {
        // A direct access after consumption is rejected, as is an access performed by a nested region of a later
        // instruction. The checked builder path rejects the direct case itself, so it uses the unchecked hatch.
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(1));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![local], None).unwrap()[0];
        builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap();
        let output = builder.add_variable(value_type(1));
        builder.add_instruction_unchecked(Instruction::new(TestOperation::Read, vec![view], vec![output], Vec::new()));
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::UseAfterConsume {
                operation: "test.read",
                instruction,
                root,
                consumer,
                consumer_operation: "test.consume",
            }) if instruction == id(0, 3) && root == allocation_root(0, 0, 0) && consumer == id(0, 2),
        ));

        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(1));
        callee.add_instruction(TestOperation::Read, Vec::new(), vec![reference], None).unwrap();
        let callee = build(callee, Vec::new());
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let payload = builder.add_input(value_type(1));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap();
        builder.add_instruction(TestOperation::Call, vec![callee], vec![local], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::UseAfterConsume {
                operation: "test.call",
                instruction,
                root,
                consumer,
                consumer_operation: "test.consume",
            }) if instruction == id(1, 2) && root == allocation_root(1, 0, 0) && consumer == id(1, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_new_classifies_external_sources_over_array_programs() {
        // Over the production array universe: a captured matrix reference is narrowed twice, a public scalar reference
        // is written and forwarded through a condition whose branches both return it, and the sources split at the
        // capture prefix.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type: ArrayIrType = ReferenceType::new(scalar_type.clone()).into();
        let make_branch = || {
            let mut branch = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
            let reference = branch.add_input(reference_type.clone());
            branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            branch
                .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(
                    vec![reference],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
        let true_branch = builder.import_region(make_branch().entry_region_ref());
        let false_branch = builder.import_region(make_branch().entry_region_ref());
        let captured = builder.add_input(ReferenceType::new(matrix_type).into());
        let external = builder.add_input(reference_type);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let replacement = builder.add_input(scalar_type.into());
        let row = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(0, 3, 1)]),
                Vec::new(),
                vec![captured],
                None,
            )
            .unwrap()[0];
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![row], None).unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![external, replacement], None)
            .unwrap();
        let forwarded = builder
            .add_instruction(
                ConditionOperation::<TestArrayValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, external],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(
                vec![snapshot, forwarded],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), Some(1), false, &[]).unwrap();
        let (captured, external) = (input_root(2, 0), input_root(2, 1));
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![input_root(0, 0), input_root(1, 0), captured, external]);
        assert_eq!(analysis.external_source(captured), Some(ReferenceSource::Capture { index: 0 }));
        assert_eq!(analysis.external_source(external), Some(ReferenceSource::Input { index: 0 }));
        assert_eq!(analysis.external_source(input_root(0, 0)), None);
        assert_eq!(
            analysis.alias(value(2, 4)),
            Some(ReferenceAliasEdge::new(
                id(2, 0),
                ReferenceAliasPosition::Output(0),
                value(2, 0),
                ReferenceAliasKind::View,
                true
            ))
        );
        assert_eq!(
            analysis.alias(value(2, 5)),
            Some(ReferenceAliasEdge::new(
                id(2, 1),
                ReferenceAliasPosition::Output(0),
                value(2, 4),
                ReferenceAliasKind::View,
                true
            ))
        );
        assert!(analysis.is_view(value(2, 5)));
        assert_eq!(analysis.root_of(value(2, 5)), Some(captured));
        assert_eq!(analysis.root_of(value(2, 7)), Some(external));
        assert_eq!(analysis.alias(value(2, 7)), None);
        assert_eq!(
            analysis.accesses(),
            &[
                ReferenceAccess::new(id(2, 2), 0, captured, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(2, 3), 0, external, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(0, 0), 0, input_root(0, 0), ReferenceAccessMode::Read),
                ReferenceAccess::new(id(1, 0), 0, input_root(1, 0), ReferenceAccessMode::Read),
            ],
        );
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(2, 4), 0, value(0, 0), external, false),
                ReferenceRegionInputBinding::new(id(2, 4), 1, value(1, 0), external, false),
            ],
        );
        assert!(!analysis.is_mutated(captured));
        assert!(analysis.is_mutated(external));
        assert_eq!(analysis.output_roots(), &[None, Some(external)]);
    }

    #[test]
    fn test_reference_analysis_new_enforces_while_identity_and_policy_over_array_programs() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type: ArrayIrType = ReferenceType::new(scalar_type.clone()).into();
        let make_condition = |mutating: bool| {
            let mut condition = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
            let counter = condition.add_input(scalar_type.clone().into());
            let reference = condition.add_input(reference_type.clone());
            let limit = if mutating {
                condition
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, counter], None)
                    .unwrap();
                condition.add_constant(TestArrayValue::Array(Array::scalar(3.0f32)))
            } else {
                condition.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0]
            };
            let predicate = condition
                .add_instruction(
                    ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                    Vec::new(),
                    vec![counter, limit],
                    None,
                )
                .unwrap()[0];
            condition
                .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(
                    vec![predicate],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let make_body = || {
            let mut body = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
            let counter = body.add_input(scalar_type.clone().into());
            let reference = body.add_input(reference_type.clone());
            let step = body.add_constant(TestArrayValue::Array(Array::scalar(1.0f32)));
            let next = body
                .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![counter, step], None)
                .unwrap()[0];
            body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, step], None)
                .unwrap();
            body.build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(
                vec![next, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap()
        };
        let make_loop = |condition: Program<
            TestArrayValue,
            TestArrayOperation,
            Vec<TestArrayValue>,
            Vec<TestArrayValue>,
        >| {
            let mut builder = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
            let condition = builder.import_region(condition.entry_region_ref());
            let body = builder.import_region(make_body().entry_region_ref());
            let counter = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_input(reference_type.clone());
            let outputs = builder
                .add_instruction(
                    WhileOperation::<ArrayIrType>::new(),
                    vec![condition, body],
                    vec![counter, reference],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // A read-only condition and an accumulating body are accepted, and the carried reference keeps its identity.
        let program = make_loop(make_condition(false));
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), Some(0), false, &[]).unwrap();
        let reference = input_root(2, 1);
        assert_eq!(analysis.root_of(value(2, 3)), Some(reference));
        assert_eq!(
            analysis.alias(value(2, 3)),
            Some(ReferenceAliasEdge::new(
                id(2, 0),
                ReferenceAliasPosition::Output(1),
                value(2, 1),
                ReferenceAliasKind::Identity,
                false
            ))
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().access_modes(),
            &BTreeMap::from([(
                reference,
                BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Accumulate])
            )]),
        );
        assert_eq!(analysis.output_roots(), &[None, Some(reference)]);

        // The production `while` lets its condition write an entering reference, because reference discharge rotates
        // such a loop into do-while form, so the analysis succeeds and reports the write on the carried root beside the
        // body's accumulation.
        let program = make_loop(make_condition(true));
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), Some(0), false, &[]).unwrap();
        assert_eq!(analysis.root_of(value(2, 3)), Some(reference));
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().access_modes(),
            &BTreeMap::from([(
                reference,
                BTreeSet::from([ReferenceAccessMode::Write, ReferenceAccessMode::Accumulate])
            )]),
        );
        assert_eq!(analysis.output_roots(), &[None, Some(reference)]);

        // A body that exchanges two carried references violates the positional identity constraint.
        let mut condition = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
        condition.add_input(reference_type.clone());
        condition.add_input(reference_type.clone());
        let predicate = condition.add_constant(TestArrayValue::Array(Array::scalar(false)));
        let condition = condition
            .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
        let first = body.add_input(reference_type.clone());
        let second = body.add_input(reference_type.clone());
        let body = body
            .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(
                vec![second, first],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestArrayValue, TestArrayOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let first = builder.add_input(reference_type.clone());
        let second = builder.add_input(reference_type);
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![first, second], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestArrayValue>, Vec<TestArrayValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert!(matches!(
            ReferenceAnalysis::new(program.entry_region_ref(), Some(0), false, &[]),
            Err(ReferenceAnalysisError::ReferenceRootMismatch {
                operation: "while",
                instruction,
                output_index: 0,
                region_index: 1,
                region_output_index: 0,
                expected,
                actual,
            }) if instruction == id(2, 0) && expected == input_root(2, 0) && actual == input_root(2, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_summarize_boundary() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type(0));
        let payload = builder.add_input(value_type(0));
        let captured = builder.add_constant(capture(0, 0));
        builder.add_instruction(TestOperation::Write, Vec::new(), vec![reference, payload], None).unwrap();
        let program = build(builder, vec![captured]);
        let root = input_root(0, 0);
        let summary =
            ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[Some(root), None], &[Some(root)], 0)
                .unwrap();

        // The returned capture and the written input denote one caller allocation. Returning the capture adds no
        // read effect, but its identity must be available to replay and to output-boundary reconstruction.
        assert_eq!(summary.reached, BTreeSet::from([root]));
        assert_eq!(summary.accesses, BTreeMap::from([(root, BTreeSet::from([ReferenceAccessMode::Write]))]));
        assert_eq!(summary.outputs, vec![Some((root, false))]);
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_validates_bound_output_identities() {
        let mut first_branch = TestBuilder::new();
        let first = first_branch.add_input(reference_type(0));
        first_branch.add_input(reference_type(0));
        let first_branch = build(first_branch, vec![first]);
        let mut second_branch = TestBuilder::new();
        second_branch.add_input(reference_type(0));
        let second = second_branch.add_input(reference_type(0));
        let second_branch = build(second_branch, vec![second]);
        let mut builder = TestBuilder::new();
        let condition = builder.add_input(value_type(0));
        let first = builder.add_input(reference_type(0));
        let second = builder.add_input(reference_type(0));
        let first_branch = builder.import_region(first_branch.entry_region_ref());
        let second_branch = builder.import_region(second_branch.entry_region_ref());
        let output = builder
            .add_instruction(
                TestOperation::Condition,
                vec![first_branch, second_branch],
                vec![condition, first, second],
                None,
            )
            .unwrap()[0];
        let program = build(builder, vec![output]);
        let region = program.entry_region_ref();
        let first = ReferenceRoot::RegionInput { region: region.id(), input_index: 1 };
        let second = ReferenceRoot::RegionInput { region: region.id(), input_index: 2 };

        // The branches may forward different formal inputs only when the supplied identities agree. The ordinary
        // cached analysis still validates the program under independent formal inputs.
        let summary = ReferenceAnalysis::summarize_boundary(region, &[None, Some(first), Some(first)], &[], 0).unwrap();
        assert_eq!(summary.outputs, vec![Some((first, false))]);
        assert!(matches!(
            ReferenceAnalysis::summarize_boundary(region, &[None, Some(first), Some(second)], &[], 0),
            Err(ReferenceAnalysisError::ReferenceRootMismatch { expected, actual, .. })
                if expected == first && actual == second,
        ));
        assert!(matches!(region.reference_analysis(0), Err(ReferenceAnalysisError::ReferenceRootMismatch { .. })));
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_rebinds_shared_regions() {
        let mut callee = TestBuilder::new();
        callee.add_input(reference_type(0));
        let captured = callee.add_constant(capture(0, 0));
        callee.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let callee = build(callee, Vec::new());
        let mut builder = TestBuilder::new();
        let first = builder.add_input(reference_type(0));
        let second = builder.add_input(reference_type(0));
        let callee = builder.import_region(callee.entry_region_ref());
        builder
            .add_instruction(TestOperation::CallWithCaptures(1), vec![callee], vec![first], None)
            .unwrap();
        builder
            .add_instruction(TestOperation::CallWithCaptures(1), vec![callee], vec![second], None)
            .unwrap();
        let program = build(builder, Vec::new());
        let region = program.entry_region_ref();
        let first = ReferenceRoot::RegionInput { region: region.id(), input_index: 0 };
        let second = ReferenceRoot::RegionInput { region: region.id(), input_index: 1 };
        let summary = ReferenceAnalysis::summarize_boundary(region, &[Some(first), Some(second)], &[], 0).unwrap();

        // Both attachments execute the same callee with a different capture binding. Neither may reuse facts
        // specialized for the other attachment, and the enclosing summary must include both reads.
        assert_eq!(summary.reached, BTreeSet::from([first, second]));
        assert_eq!(
            summary.accesses,
            BTreeMap::from([
                (first, BTreeSet::from([ReferenceAccessMode::Read])),
                (second, BTreeSet::from([ReferenceAccessMode::Read])),
            ])
        );
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_ignores_unused_constants() {
        let mut builder = TestBuilder::new();
        builder.add_constant(capture(3, 0));
        let program = build(builder, Vec::new());
        let summary = ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[], &[], 0).unwrap();
        assert_eq!(summary.reached, BTreeSet::new());
        assert_eq!(summary.accesses, BTreeMap::new());
        assert_eq!(summary.outputs, Vec::new());

        // Boundary analysis follows replay materialization; the existing structural entry point remains strict.
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { capture_index: 3, capture_count: 0, .. })
        ));
        let mut builder = TestBuilder::new();
        let captured = builder.add_constant(capture(3, 0));
        let program = build(builder, vec![captured]);
        assert!(matches!(
            ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[], &[], 0),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { capture_index: 3, capture_count: 0, .. }),
        ));
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_preserves_local_capture_ownership() {
        let mut callee = TestBuilder::new();
        let captured = callee.add_constant(capture(0, 0));
        let callee = build(callee, vec![captured]);
        let mut outer = TestBuilder::new();
        outer.add_input(reference_type(0));
        let callee = outer.import_region(callee.entry_region_ref());
        let output = outer.add_instruction(TestOperation::Call, vec![callee], Vec::new(), None).unwrap()[0];
        let outer = build(outer, vec![output]);
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(0));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let outer = builder.import_region(outer.entry_region_ref());
        let output =
            builder.add_instruction(TestOperation::CallWithCaptures(1), vec![outer], vec![local], None).unwrap()[0];
        let program = build(builder, vec![output]);
        let region = program.entry_region_ref();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 0), output_index: 0 };

        // Both nested calls return the allocation created by their caller. Capturing it does not turn it into a
        // fresh allocation of either nested region, so the escape checks must allow both returns.
        let summary = ReferenceAnalysis::summarize_boundary(region, &[None], &[], 0).unwrap();
        assert_eq!(summary.outputs, vec![Some((root, false))]);
        assert_eq!(summary.accesses, BTreeMap::new());
        assert_eq!(region.reference_analysis(0).unwrap().output_roots(), &[Some(root)]);
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_rejects_use_after_consume() {
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(0));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap();
        let output = builder.add_variable(value_type(0));
        // Construct an invalid instruction explicitly: the checked builder would already reject this access.
        builder.add_instruction_unchecked(Instruction::new(TestOperation::Read, vec![local], vec![output], Vec::new()));
        let program = build(builder, vec![output]);
        assert!(matches!(
            ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[None], &[], 0),
            Err(ReferenceAnalysisError::UseAfterConsume {
                operation: "test.read", instruction, root, consumer, consumer_operation: "test.consume",
            }) if instruction == id(0, 2) && root == allocation_root(0, 0, 0) && consumer == id(0, 1),
        ));
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_rejects_consumption_through_views() {
        let mut builder = TestBuilder::new();
        let payload = builder.add_input(value_type(0));
        let local = builder.add_instruction(TestOperation::New, Vec::new(), vec![payload], None).unwrap()[0];
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![local], None).unwrap()[0];
        let output = builder.add_variable(value_type(0));
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Consume,
            vec![view],
            vec![output],
            Vec::new(),
        ));
        let program = build(builder, vec![output]);
        assert!(matches!(
            ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[None], &[], 0),
            Err(ReferenceAnalysisError::ConsumptionThroughView {
                operation: "test.consume", instruction, input_index: 0, root,
            }) if instruction == id(0, 2) && root == allocation_root(0, 0, 0),
        ));
    }

    #[test]
    fn test_reference_analysis_summarize_boundary_preserves_capture_diagnostics() {
        let mut builder = TestBuilder::new();
        let captured = builder.add_input(reference_type(0));
        let output = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![captured], None).unwrap()[0];
        let program = build(builder, vec![output]);
        let root = input_root(0, 0);
        assert!(matches!(
            ReferenceAnalysis::summarize_boundary(program.entry_region_ref(), &[Some(root)], &[Some(root)], 1),
            Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: "test.consume", instruction, root: actual,
                external_source: ReferenceSource::Capture { index: 0 },
            }) if instruction == id(0, 0) && actual == root,
        ));
    }

    #[test]
    fn test_reference_analysis_region() {
        assert_eq!(fixture_analysis().region(), RegionId::new(1));
    }

    #[test]
    fn test_reference_analysis_roots() {
        assert_eq!(
            fixture_analysis().roots().collect::<Vec<_>>(),
            vec![input_root(0, 0), input_root(1, 0), input_root(1, 1), allocation_root(1, 0, 0)],
        );
    }

    #[test]
    fn test_reference_analysis_values() {
        // Include aliases and inherited captures in nested regions, but exclude non-reference inputs and results.
        assert_eq!(
            fixture_analysis().values().collect::<Vec<_>>(),
            vec![value(0, 0), value(0, 3), value(1, 0), value(1, 1), value(1, 3), value(1, 4), value(1, 5)],
        );
    }

    #[test]
    fn test_reference_analysis_root_of() {
        let analysis = fixture_analysis();
        let c = allocation_root(1, 0, 0);
        assert_eq!(analysis.root_of(value(1, 0)), Some(input_root(1, 0)));
        assert_eq!(analysis.root_of(value(1, 1)), Some(input_root(1, 1)));
        assert_eq!(analysis.root_of(value(1, 2)), None);
        assert_eq!(analysis.root_of(value(1, 3)), Some(c));
        assert_eq!(analysis.root_of(value(1, 4)), Some(c));
        assert_eq!(analysis.root_of(value(1, 5)), Some(c));
        assert_eq!(analysis.root_of(value(1, 6)), None);
        assert_eq!(analysis.root_of(value(0, 0)), Some(input_root(0, 0)));
        assert_eq!(analysis.root_of(value(0, 3)), Some(input_root(1, 0)));
        assert_eq!(analysis.root_of(value(3, 0)), None);
    }

    #[test]
    fn test_reference_analysis_external_source() {
        let analysis = fixture_analysis();
        assert_eq!(analysis.external_source(input_root(1, 0)), Some(ReferenceSource::Capture { index: 0 }));
        assert_eq!(analysis.external_source(input_root(1, 1)), Some(ReferenceSource::Input { index: 0 }));
        assert_eq!(analysis.external_source(input_root(0, 0)), None);
        assert_eq!(analysis.external_source(allocation_root(1, 0, 0)), None);
        assert_eq!(analysis.external_source(input_root(1, 2)), None);
    }

    #[test]
    fn test_reference_analysis_accesses() {
        let (a, b, c, k) = (input_root(1, 0), input_root(1, 1), allocation_root(1, 0, 0), input_root(0, 0));
        assert_eq!(
            fixture_analysis().accesses(),
            &[
                ReferenceAccess::new(id(1, 3), 0, a, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(1, 4), 0, b, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(1, 5), 0, c, ReferenceAccessMode::Accumulate),
                ReferenceAccess::new(id(1, 6), 0, c, ReferenceAccessMode::ReadWrite),
                ReferenceAccess::new(id(0, 0), 0, k, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(0, 1), 0, k, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(0, 2), 0, a, ReferenceAccessMode::Read),
                ReferenceAccess::new(id(1, 8), 0, c, ReferenceAccessMode::Consume),
            ],
        );
    }

    #[test]
    fn test_reference_analysis_access_modes() {
        // Modes are transitive: `B` gains the callee's read through its binding, while the callee's own input root
        // records only what the callee does directly.
        let analysis = fixture_analysis();
        assert_eq!(analysis.access_modes(input_root(1, 0)).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
        assert_eq!(
            analysis.access_modes(input_root(1, 1)).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write],
        );
        assert_eq!(
            analysis.access_modes(allocation_root(1, 0, 0)).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::ReadWrite, ReferenceAccessMode::Accumulate, ReferenceAccessMode::Consume],
        );
        assert_eq!(
            analysis.access_modes(input_root(0, 0)).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write],
        );
        assert_eq!(analysis.access_modes(input_root(1, 2)).count(), 0);
    }

    #[test]
    fn test_reference_analysis_is_mutated() {
        let analysis = fixture_analysis();
        assert!(!analysis.is_mutated(input_root(1, 0)));
        assert!(analysis.is_mutated(input_root(1, 1)));
        assert!(analysis.is_mutated(allocation_root(1, 0, 0)));
        assert!(!analysis.is_mutated(input_root(1, 2)));
    }

    #[test]
    fn test_reference_analysis_alias() {
        let analysis = fixture_analysis();
        assert_eq!(analysis.alias(value(1, 3)), None);
        assert_eq!(
            analysis.alias(value(1, 4)),
            Some(ReferenceAliasEdge::new(
                id(1, 1),
                ReferenceAliasPosition::Output(0),
                value(1, 3),
                ReferenceAliasKind::View,
                true
            ))
        );
        assert_eq!(
            analysis.alias(value(1, 5)),
            Some(ReferenceAliasEdge::new(
                id(1, 2),
                ReferenceAliasPosition::Output(0),
                value(1, 4),
                ReferenceAliasKind::Identity,
                true
            ))
        );
        assert_eq!(analysis.alias(value(1, 0)), None);
        assert_eq!(analysis.alias(value(0, 3)), None);
        assert_eq!(analysis.alias(value(1, 2)), None);
    }

    #[test]
    fn test_reference_analysis_is_view() {
        let analysis = fixture_analysis();
        assert!(!analysis.is_view(value(1, 0)));
        assert!(!analysis.is_view(value(1, 3)));
        assert!(analysis.is_view(value(1, 4)));
        assert!(analysis.is_view(value(1, 5)));
        assert!(!analysis.is_view(value(1, 2)));
        assert!(!analysis.is_view(value(0, 3)));
    }

    #[test]
    fn test_reference_analysis_region_input_bindings() {
        assert_eq!(
            fixture_analysis().region_input_bindings(),
            &[ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1), false)],
        );
    }

    #[test]
    fn test_reference_analysis_transitive_access() {
        let analysis = fixture_analysis();
        let (a, b, c, k) = (input_root(1, 0), input_root(1, 1), allocation_root(1, 0, 0), input_root(0, 0));
        assert_eq!(analysis.transitive_access(id(1, 0)), None);
        assert_eq!(analysis.transitive_access(id(1, 1)), None);
        assert_eq!(
            analysis.transitive_access(id(1, 3)).unwrap().access_modes(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 6)).unwrap().access_modes(),
            &BTreeMap::from([(c, BTreeSet::from([ReferenceAccessMode::ReadWrite]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 7)).unwrap().access_modes(),
            &BTreeMap::from([
                (a, BTreeSet::from([ReferenceAccessMode::Read])),
                (b, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write])),
            ]),
        );
        assert_eq!(
            analysis.transitive_access(id(0, 0)).unwrap().access_modes(),
            &BTreeMap::from([(k, BTreeSet::from([ReferenceAccessMode::Write]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 8)).unwrap().access_modes(),
            &BTreeMap::from([(c, BTreeSet::from([ReferenceAccessMode::Consume]))]),
        );
        assert_eq!(analysis.transitive_access(id(1, 9)), None);
    }

    #[test]
    fn test_reference_analysis_consumer() {
        let analysis = fixture_analysis();
        assert_eq!(analysis.consumer(allocation_root(1, 0, 0)), Some(id(1, 8)));
        assert_eq!(analysis.consumer(input_root(1, 0)), None);
        assert_eq!(analysis.consumer(input_root(1, 1)), None);
        assert_eq!(analysis.consumer(input_root(1, 2)), None);
    }

    #[test]
    fn test_reference_analysis_output_roots() {
        assert_eq!(fixture_analysis().output_roots(), &[None, Some(input_root(1, 1)), None]);
    }

    #[test]
    fn test_program_reference_analysis() {
        let program = fixture();
        let analysis = program.reference_analysis(1).unwrap();
        let direct = ReferenceAnalysis::new(program.entry_region_ref(), Some(1), false, &[]).unwrap();
        assert_eq!(analysis.roots().collect::<Vec<_>>(), direct.roots().collect::<Vec<_>>());
        assert_eq!(analysis.accesses(), direct.accesses());
        assert_eq!(analysis.region_input_bindings(), direct.region_input_bindings());
        assert_eq!(analysis.output_roots(), direct.output_roots());
        assert_eq!(analysis.external_source(input_root(1, 0)), Some(ReferenceSource::Capture { index: 0 }));
        assert!(matches!(
            program.reference_analysis(4),
            Err(ReferenceAnalysisError::InvalidCaptureScope { region, message })
                if region == RegionId::new(1)
                    && message == "the capture prefix of 4 inputs exceeds the region's 3 inputs",
        ));
    }

    #[test]
    fn test_region_ref_reference_analysis() {
        let program = fixture();
        let retained = program.entry_region_ref().reference_analysis(1).unwrap();
        assert_eq!(*retained, ReferenceAnalysis::new(program.entry_region_ref(), Some(1), false, &[]).unwrap());

        // Repeated requests, including through the program-level accessor and through a clone that shares the region
        // arena, are served the retained artifact (under `debug_assertions` each hit also re-derives and compares the
        // analysis), while an independently built program derives its own.
        assert!(Arc::ptr_eq(&retained, &program.entry_region_ref().reference_analysis(1).unwrap()));
        assert!(Arc::ptr_eq(&retained, &program.reference_analysis(1).unwrap()));
        assert!(Arc::ptr_eq(&retained, &program.clone().reference_analysis(1).unwrap()));
        assert!(!Arc::ptr_eq(&retained, &fixture().reference_analysis(1).unwrap()));
    }

    #[test]
    fn test_region_ref_reference_analysis_separates_capture_scopes() {
        let program = fixture();
        let failure = program.reference_analysis(0).unwrap_err();
        assert!(matches!(failure, ReferenceAnalysisError::InvalidReferenceCapture { capture_count: 0, .. }));
        assert_eq!(program.reference_analysis(0).unwrap_err(), failure);
        assert_eq!(program.reference_analysis(1).unwrap().region(), program.entry_region_ref().id());
    }

    #[test]
    fn test_region_ref_reference_analysis_is_retained_for_a_value_family_without_captures() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(reference_type);
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![value],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let root = ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 };
        let retained = program.reference_analysis(0).unwrap();
        assert!(Arc::ptr_eq(&retained, &program.reference_analysis(0).unwrap()));
        assert_eq!(retained.roots().collect::<Vec<_>>(), vec![root]);
        assert_eq!(retained.access_modes(root).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
    }

    #[test]
    fn test_region_ref_reference_analysis_derives_a_fresh_entry_for_a_rebased_import() {
        // A closure-copying import keeps the source region's transform cache but renumbers the attached regions, so the
        // records of the retained analysis would name the wrong identifiers. The closure's region identifiers are part
        // of the cache key, which is what makes a renumbered copy derive its own analysis while a copy that keeps the
        // source identifiers is still served the retained one.
        let mut callee = TestBuilder::new();
        let reference = callee.add_input(reference_type(0));
        let read = callee.add_instruction(TestOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let callee = build(callee, vec![read]);
        let mut entry = TestBuilder::new();
        let callee = entry.import_region(callee.entry_region_ref());
        let reference = entry.add_input(reference_type(0));
        let read = entry.add_instruction(TestOperation::Call, vec![callee], vec![reference], None).unwrap()[0];
        let source = build(entry, vec![read]);
        let retained = source.entry_region_ref().reference_analysis(0).unwrap();

        let mut wrapper = TestBuilder::new();
        let same_identifiers = wrapper.import_region(source.entry_region_ref());
        let renumbered = wrapper.import_region(source.entry_region_ref());
        let reference = wrapper.add_input(reference_type(0));
        wrapper.add_instruction(TestOperation::Call, vec![same_identifiers], vec![reference], None).unwrap();
        wrapper.add_instruction(TestOperation::Call, vec![renumbered], vec![reference], None).unwrap();
        let wrapper = build(wrapper, Vec::new());
        assert_eq!(same_identifiers, source.entry_region_ref().id());
        assert_ne!(renumbered, source.entry_region_ref().id());

        let same = wrapper.region_ref(same_identifiers).unwrap().reference_analysis(0).unwrap();
        assert!(Arc::ptr_eq(&same, &retained));
        let derived = wrapper.region_ref(renumbered).unwrap().reference_analysis(0).unwrap();
        assert!(!Arc::ptr_eq(&derived, &retained));
        assert_eq!(derived.region(), renumbered);
        assert_eq!(
            *derived,
            ReferenceAnalysis::new(wrapper.region_ref(renumbered).unwrap(), Some(0), false, &[]).unwrap()
        );
        assert_ne!(derived.roots().collect::<Vec<_>>(), retained.roots().collect::<Vec<_>>());

        // The source program keeps its own retained artifact, because only the copies were imported.
        assert!(Arc::ptr_eq(&source.entry_region_ref().reference_analysis(0).unwrap(), &retained));
    }
    #[test]
    fn test_region_ref_reference_analysis_with_constants() {
        let mut builder = TestBuilder::new();
        let captured = builder.add_constant(capture(4, 0));
        let repeated = builder.add_constant(capture(4, 0));
        let view = builder.add_instruction(TestOperation::View, Vec::new(), vec![captured], None).unwrap()[0];
        let program = build(builder, vec![view]);
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_constants().unwrap();
        let root = ReferenceRoot::Constant { value: ValueId::new(region.id(), captured) };
        assert_eq!(analysis.root_of(ValueId::new(region.id(), repeated)), Some(root));
        assert_eq!(analysis.output_roots(), &[Some(root)]);
        assert!(analysis.is_view(ValueId::new(region.id(), view)));
        assert!(Arc::ptr_eq(&analysis, &region.reference_analysis_with_constants().unwrap()));
        assert!(matches!(
            region.reference_analysis(0),
            Err(ReferenceAnalysisError::InvalidReferenceCapture { capture_index: 4, capture_count: 0, .. })
        ));
    }

    #[test]
    fn test_region_ref_reference_analysis_with_constants_preserves_nested_view_validation() {
        let mut child_builder = TestBuilder::new();
        let captured = child_builder.add_constant(capture(7, 0));
        let view = child_builder.add_instruction(TestOperation::View, Vec::new(), vec![captured], None).unwrap()[0];
        let forwarded =
            child_builder.add_instruction(TestOperation::Identity, Vec::new(), vec![view], None).unwrap()[0];
        let child = build(child_builder, vec![forwarded]);
        let mut builder = TestBuilder::new();
        let child = builder.import_region(child.entry_region_ref());
        let output = builder.add_instruction(TestOperation::Call, vec![child], Vec::new(), None).unwrap()[0];
        let program = build(builder, vec![output]);
        let region = program.entry_region_ref();
        assert_eq!(
            region.reference_analysis_with_constants(),
            Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
                operation: "test.call",
                instruction: InstructionId::new(region.id(), 0),
                region_index: 0,
                boundary: "output",
                index: 0,
            })
        );
    }

    #[test]
    fn test_region_ref_reference_analysis_with_constants_keeps_explicit_capture_scopes_strict() {
        let mut child_builder = TestBuilder::new();
        child_builder.add_input(value_type(0));
        let captured = child_builder.add_constant(capture(0, 0));
        let read = child_builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap()[0];
        let child = build(child_builder, vec![read]);
        let mut builder = TestBuilder::new();
        builder.add_constant(capture(0, 0));
        let payload = builder.add_input(value_type(0));
        let child = builder.import_region(child.entry_region_ref());
        let output = builder
            .add_instruction(TestOperation::CallWithCaptures(1), vec![child], vec![payload], None)
            .unwrap()[0];
        let program = build(builder, vec![output]);
        assert_eq!(
            program.entry_region_ref().reference_analysis_with_constants(),
            Err(ReferenceAnalysisError::InvalidReferenceCapture {
                region: child,
                atom: AtomId::new(1),
                capture_index: 0,
                capture_count: 1
            })
        );
        let child_region = program.region_ref(child).unwrap();
        assert_eq!(
            child_region.reference_analysis_with_capture_scope(Some(1)),
            Err(ReferenceAnalysisError::InvalidReferenceCapture {
                region: child,
                atom: AtomId::new(1),
                capture_index: 0,
                capture_count: 1
            })
        );
        let analysis = child_region.reference_analysis_with_constants().unwrap();
        assert_eq!(
            analysis.root_of(ValueId::new(child, AtomId::new(1))),
            Some(ReferenceRoot::Constant { value: ValueId::new(child, AtomId::new(1)) })
        );
    }

    #[test]
    fn test_region_ref_reference_analysis_with_constants_distinguishes_open_and_empty_capture_scopes() {
        let mut child = TestBuilder::new();
        let captured = child.add_constant(capture(0, 0));
        let output = child.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap()[0];
        let child = build(child, vec![output]);
        let mut builder = TestBuilder::new();
        let child = builder.import_region(child.entry_region_ref());
        builder.add_instruction(TestOperation::Call, vec![child], Vec::new(), None).unwrap();
        builder.add_instruction(TestOperation::CallWithCaptures(0), vec![child], Vec::new(), None).unwrap();
        let program = build(builder, Vec::new());

        // An inherited open scope accepts unresolved captures; an explicitly empty capture boundary forbids them.
        // Reusing the first attachment's analysis for the second would erase that distinction.
        assert_eq!(
            program.entry_region_ref().reference_analysis_with_constants(),
            Err(ReferenceAnalysisError::InvalidCaptureScope {
                region: child,
                message: "shared region is reached under two different capture scopes".to_string(),
            }),
        );
    }

    #[test]
    fn test_region_ref_reference_analysis_with_constants_has_separate_cache_entries() {
        let mut builder = TestBuilder::new();
        let input = builder.add_input(reference_type(0));
        let program = build(builder, vec![input]);
        let region = program.entry_region_ref();
        let strict = region.reference_analysis(0).unwrap();
        let open = region.reference_analysis_with_constants().unwrap();
        assert_eq!(strict, open);
        assert!(!Arc::ptr_eq(&strict, &open));
        assert!(Arc::ptr_eq(&strict, &region.reference_analysis(0).unwrap()));
        assert!(Arc::ptr_eq(&open, &region.reference_analysis_with_constants().unwrap()));
    }

    #[test]
    fn test_region_ref_reference_analysis_with_constants_unifies_concrete_allocations() {
        // Unlike the array IR, this third-party constant family admits concrete reference handles. The canonical
        // analysis must therefore unify its runtime allocations independently of inherited capture indices.
        /// Concrete reference constant family used to test runtime allocation canonicalization.
        #[derive(Clone, Debug, PartialEq, Parameter)]
        struct ReferenceConstant(ArrayIrValue<Array>);

        impl Display for ReferenceConstant {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }

        impl Typed for ReferenceConstant {
            type Type = ArrayIrType;

            fn r#type(&self) -> Cow<'_, ArrayIrType> {
                self.0.r#type()
            }
        }

        impl Value for ReferenceConstant {
            type DispatchDomain = EagerContext<Self>;
            type ExecutionDomain = EagerContext<Self>;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                EagerContext::new()
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                EagerContext::new()
            }

            fn reference_id(&self) -> Option<ReferenceId> {
                self.0.reference_id()
            }
        }

        let reference = ArrayReference::new(Array::scalar(2.0_f32));
        let mut builder = ProgramBuilder::<ReferenceConstant, TestArrayOperation>::new();
        let first = builder.add_constant(ReferenceConstant(ArrayIrValue::Reference(reference.clone())));
        let second = builder.add_constant(ReferenceConstant(ArrayIrValue::Reference(reference)));
        let program = builder
            .build::<Vec<ReferenceConstant>, Vec<ReferenceConstant>>(
                vec![first, second],
                Vec::new(),
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_constants().unwrap();
        let root = ReferenceRoot::Constant { value: ValueId::new(region.id(), first) };
        assert_eq!(analysis.output_roots(), &[Some(root), Some(root)]);
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![root]);
        assert_eq!(
            region.reference_analysis_with_capture_scope(Some(0)).unwrap().output_roots(),
            &[Some(root), Some(root)],
        );
    }

    #[test]
    fn test_region_ref_reference_analysis_with_consumable_inputs() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type(0));
        let output = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![reference], None).unwrap()[0];
        let program = build(builder, vec![output]);
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_consumable_inputs(0, vec![0]).unwrap();
        assert_eq!(analysis.consumer(input_root(0, 0)), Some(id(0, 0)));
        assert!(Arc::ptr_eq(&analysis, &region.reference_analysis_with_consumable_inputs(0, vec![0]).unwrap()));

        // An ownership-aware cache entry must not satisfy a borrowed-input or capture-boundary request.
        assert!(matches!(
            region.reference_analysis(0),
            Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                external_source: ReferenceSource::Input { index: 0 },
                ..
            })
        ));
        assert!(matches!(
            region.reference_analysis_with_consumable_inputs(1, vec![0]),
            Err(ReferenceAnalysisError::ExternalReferenceConsumption {
                external_source: ReferenceSource::Capture { index: 0 },
                ..
            })
        ));

        // Ownership belongs to the analyzed entry region; passing its input to a child does not transfer it again.
        let mut builder = TestBuilder::new();
        let callee = builder.import_region(region);
        let reference = builder.add_input(reference_type(0));
        let output = builder.add_instruction(TestOperation::Call, vec![callee], vec![reference], None).unwrap()[0];
        let program = build(builder, vec![output]);
        assert!(matches!(
            program.entry_region_ref().reference_analysis_with_consumable_inputs(0, vec![0]),
            Err(ReferenceAnalysisError::ConsumptionOutsideCreationScope { operation: "test.consume", .. })
        ));
    }
}
