//! Generic program-level reference analysis.
//!
//! [`ReferenceAnalysis`] resolves every reference-typed value in a [`Region`](crate::Region) closure to exactly one
//! canonical [`ReferenceRoot`], records the alias edges and accesses that connect values to roots, and validates the
//! lifetime, capture, and region-boundary rules of the reference model. It relies only on the generic
//! [`Operation`] hooks ([`Operation::reference_semantics`], [`Operation::input_region_provenance`],
//! [`Operation::output_region_provenance`], [`Operation::region_capture_input_count`],
//! [`Operation::reference_output_identity_input`], and [`Operation::allows_reference_access_through_region_input`])
//! and on [`Type::is_reference`], so it knows nothing about arrays, views, or any particular value family.
//!
//! This analysis is the shared fact source for everything that reasons about references structurally: reference
//! discharge, kernel-boundary validation, diagnostics, lowering, and every transform rule that must know which root an
//! operand denotes, how it is accessed, and whether it is a derived view. Consumers obtain it through
//! [`RegionRef::reference_analysis`], which retains one analysis per region closure in the region's transform cache so
//! that all of them share a single derivation. It is not a standing lint that every program pays for: ordinary program
//! construction does not run it, and such programs remain validated by the construction-time alias tracking of
//! [`ProgramBuilder`](crate::ProgramBuilder) and by the eager [`Reference`](crate::Reference) runtime, neither of which
//! depends on this module.
//!
//! # Roots and Namespaces
//!
//! A root is either a reference-typed input of some region ([`ReferenceRoot::RegionInput`]) or an allocation performed
//! by some instruction ([`ReferenceRoot::Allocation`]). Every region's values resolve to roots in that region's own
//! *namespace*: its own inputs, its own allocations, and the capture roots it inherits from an enclosing scope. The
//! analysis never rewrites a nested region's records into its parent's namespace. Instead, each attachment of a nested
//! region records one [`ReferenceRegionInputBinding`] per reference-typed region input, mapping that formal input to
//! the caller root it denotes, and the attaching instruction's [`ReferenceTransitiveAccess`] summary is expressed in
//! the caller's namespace after substituting those bindings and dropping the nested region's local allocations.
//!
//! # Capture Scopes
//!
//! A capture-lifted program names its captures through constants whose capture index refers to the *active capture
//! scope*. At the analyzed region, the scope is its first `capture_count` inputs. A nested region either inherits the
//! scope of the instruction attaching it or, when [`Operation::region_capture_input_count`] returns `Some(n)`,
//! establishes a fresh scope from its first `n` inputs. A reference-typed constant resolves to the root bound at its
//! capture position, so a capture root is the same root in every region that inherits the scope.
//!
//! # Root-Only Boundaries
//!
//! Only complete-value handles cross a region boundary. A derived view (any alias chain containing a
//! [`ReferenceAliasKind::View`] edge) may neither enter an attached region nor be forwarded out of one; the region
//! that needs a view recreates it from the carried root. Reference-typed outputs of region-carrying operations resolve
//! through [`Operation::reference_output_identity_input`] (every provenance origin must return exactly the constrained
//! root) or through [`Operation::output_region_provenance`] (all origins must agree). An origin rooted in an allocation
//! local to the attached region is an escaping allocation and is rejected.
//!
//! # Lifetime Rules
//!
//! Consumption is a complete-value lifetime event: it must go through a complete-value handle, it is legal only in the
//! region that allocated the root, and no access may follow it in program order, including accesses through aliases or
//! through nested regions of later instructions. External roots (entry-region inputs and captures) are owned by the
//! caller and are never consumed. For example:
//!
//! ```text
//! lambda %0:ref<f32[]> .                 external root: region input 0 (source input 0)
//! let %1:f32[] = reference_read %0       read of region input 0
//!     %2:ref<f32[]> = reference_new %1   local root: allocation at instruction 1
//!     %3:ref<f32[]> = reference_index %2 view alias of the allocation
//!     reference_write %3 %1              write reaching the allocation through the view
//!     %4:f32[] = reference_freeze %2     consumes the allocation; consuming %3 instead would be rejected
//! in (%4)
//! ```

// TODO(eaplatanios): Review this module.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Display;
use std::rc::Rc;
use std::sync::Arc;

use thiserror::Error;

use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::ReferenceSource;
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceAliasKind, ReferenceOutput};
use crate::programs::regions::{OutputRegionProvenance, Region, RegionId, RegionRef};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId};

/// Error produced by [`ReferenceAnalysis`] when a [`Region`](crate::Region) closure violates the reference model.
/// Every variant names the operation and instruction (or region and atom) at fault together with the violated rule,
/// so consumers can surface it without re-deriving context.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReferenceAnalysisError {
    /// An operation's declared reference semantics or region hooks name inputs, outputs, or regions that the
    /// application does not have, or classify a non-reference output as a reference.
    #[error("operation `{operation}` at {instruction} declares malformed reference semantics: {message}")]
    MalformedSemantics {
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

    /// A region stores a reference-typed constant that names no capture.
    #[error(
        "region {region} stores reference-typed constant {atom} that names no capture; references enter a program \
             only through inputs and captures"
    )]
    ReferenceConstant {
        /// Region storing the constant.
        region: RegionId,

        /// Atom of the constant.
        atom: AtomId,
    },

    /// A reference-typed capture constant names a capture position that the active capture scope does not bind to a
    /// reference.
    #[error(
        "reference-typed constant {atom} in region {region} names capture {capture_index}, which the active capture \
             scope of {capture_count} captures does not bind to a reference"
    )]
    CaptureOutOfScope {
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

    /// An operation passes a reference into an attached region input without declaring which input supplies it.
    #[error(
        "operation `{operation}` at {instruction} passes a reference into region {region_index} input {input_index} \
             without declaring which input supplies it"
    )]
    UndeclaredRegionInputProvenance {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Position of the attached region.
        region_index: usize,

        /// Reference-typed input of the attached region.
        input_index: usize,
    },

    /// An operation produces a reference-typed output that it neither classifies, constrains, nor forwards.
    #[error(
        "operation `{operation}` at {instruction} produces a reference at output {output_index} without declaring \
             whether it allocates, aliases, preserves an input identity, or forwards a region output"
    )]
    UndeclaredReferenceOutput {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Reference-typed output position.
        output_index: usize,
    },

    /// An operation forwards an output from region outputs that denote different reference roots.
    #[error(
        "operation `{operation}` at {instruction} forwards output {output_index} from region outputs that denote \
             different reference roots, {first} and {other}"
    )]
    InconsistentForwardedRoots {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Forwarded output position.
        output_index: usize,

        /// Root denoted by the first provenance origin.
        first: ReferenceRoot,

        /// Disagreeing root denoted by a later provenance origin.
        other: ReferenceRoot,
    },

    /// An attached region returns a root other than the one an identity-constrained output must preserve.
    #[error(
        "operation `{operation}` at {instruction} constrains output {output_index} to preserve {expected}, but \
             region {region_index} returns {actual} at output {region_output_index}"
    )]
    FixedPointRootMismatch {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Identity-constrained output position.
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
    ConsumeThroughView {
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
    ConsumeExternal {
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
    ConsumeOutsideCreationScope {
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
        ProgramError::MalformedProgram(error.to_string())
    }
}

/// Canonical reference root that a reference-typed value denotes: either a reference-typed input of a
/// [`Region`](crate::Region) or a fresh allocation performed by an [`Instruction`](crate::Instruction). Roots are
/// region-relative: a root belongs to the namespace of the region whose input it is or whose instruction allocated it.
/// The derived ordering (region inputs before allocations, then by region and position) is deterministic and
/// independent of any hash-map iteration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceRoot {
    /// Reference-typed input of a [`Region`](crate::Region).
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
}

impl ReferenceRoot {
    /// Returns the [`RegionId`] of the [`Region`](crate::Region) whose namespace owns this [`ReferenceRoot`].
    #[inline]
    pub fn region(self) -> RegionId {
        match self {
            Self::RegionInput { region, .. } => region,
            Self::Allocation { instruction, .. } => instruction.region(),
        }
    }
}

impl Display for ReferenceRoot {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegionInput { region, input_index } => write!(formatter, "region {region} input {input_index}"),
            Self::Allocation { instruction, output_index } => {
                write!(formatter, "allocation at {instruction} output {output_index}")
            }
        }
    }
}

/// One reference access performed directly by an [`Instruction`](crate::Instruction), resolved to the canonical
/// [`ReferenceRoot`] in the namespace of the region containing that instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAccess {
    /// Instruction performing the access.
    instruction: InstructionId,

    /// Accessed input position of the instruction.
    input_index: usize,

    /// Canonical root reached by the access.
    root: ReferenceRoot,

    /// Mode of the access.
    mode: ReferenceAccessMode,
}

impl ReferenceAccess {
    /// Creates a new [`ReferenceAccess`].
    #[inline]
    pub const fn new(
        instruction: InstructionId,
        input_index: usize,
        root: ReferenceRoot,
        mode: ReferenceAccessMode,
    ) -> Self {
        Self { instruction, input_index, root, mode }
    }

    /// Returns the instruction performing the access.
    #[inline]
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the accessed input position of the instruction.
    #[inline]
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the canonical root reached by the access.
    #[inline]
    pub const fn root(self) -> ReferenceRoot {
        self.root
    }

    /// Returns the mode of the access.
    #[inline]
    pub const fn mode(self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Alias edge that defines one reference-typed value from another reference-typed value of the same region. Edges are
/// recorded for [`ReferenceOutput::Alias`] outputs and for outputs constrained by
/// [`Operation::reference_output_identity_input`], which are identity edges from the constrained input. Narrowing is
/// transitive: an identity alias of a derived view still represents only that view, so [`narrows`](Self::narrows)
/// describes the complete chain from the root to the aliasing value rather than only this edge's own kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAliasEdge {
    /// Instruction defining the aliasing value.
    instruction: InstructionId,

    /// Index of the aliasing instruction's output that defines the aliasing value.
    output_index: usize,

    /// Reference-typed value the alias is derived from.
    source: ValueId,

    /// Kind of this edge.
    kind: ReferenceAliasKind,

    /// Whether the aliasing value is a derived view of its root.
    narrows: bool,
}

impl ReferenceAliasEdge {
    /// Creates a new [`ReferenceAliasEdge`].
    #[inline]
    pub const fn new(
        instruction: InstructionId,
        output_index: usize,
        source: ValueId,
        kind: ReferenceAliasKind,
        narrows: bool,
    ) -> Self {
        Self { instruction, output_index, source, kind, narrows }
    }

    /// Returns the instruction defining the aliasing value.
    #[inline]
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the index of the aliasing instruction's output that defines the aliasing value, which is the output
    /// whose description a [`ReferenceViewOperation`](crate::programs::references::ReferenceViewOperation) reports
    /// for this edge.
    #[inline]
    pub const fn output_index(self) -> usize {
        self.output_index
    }

    /// Returns the reference-typed value the alias is derived from.
    #[inline]
    pub const fn source(self) -> ValueId {
        self.source
    }

    /// Returns the kind of this edge.
    #[inline]
    pub const fn kind(self) -> ReferenceAliasKind {
        self.kind
    }

    /// Returns whether the aliasing value is a derived view of its root, through this edge or an earlier one.
    #[inline]
    pub const fn narrows(self) -> bool {
        self.narrows
    }
}

/// Binding of one reference-typed input of an attached [`Region`](crate::Region) to the caller root it denotes for one
/// particular attachment. A shared region attached by several instructions has one binding per attachment, so nested
/// records stay in the nested region's own namespace and consumers substitute them through these bindings.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceRegionInputBinding {
    /// Instruction attaching the region.
    instruction: InstructionId,

    /// Position of the attached region among the instruction's regions.
    region_index: usize,

    /// Reference-typed input value of the attached region.
    input: ValueId,

    /// Root the input denotes, in the namespace of the region containing the instruction.
    root: ReferenceRoot,
}

impl ReferenceRegionInputBinding {
    /// Creates a new [`ReferenceRegionInputBinding`].
    #[inline]
    pub const fn new(instruction: InstructionId, region_index: usize, input: ValueId, root: ReferenceRoot) -> Self {
        Self { instruction, region_index, input, root }
    }

    /// Returns the instruction attaching the region.
    #[inline]
    pub const fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Returns the position of the attached region among the instruction's regions.
    #[inline]
    pub const fn region_index(self) -> usize {
        self.region_index
    }

    /// Returns the reference-typed input value of the attached region.
    #[inline]
    pub const fn input(self) -> ValueId {
        self.input
    }

    /// Returns the root the input denotes, in the namespace of the region containing the instruction.
    #[inline]
    pub const fn root(self) -> ReferenceRoot {
        self.root
    }
}

/// Transitive reference accesses of one [`Instruction`](crate::Instruction): the modes it performs on each root,
/// directly or anywhere inside its attached region closure, expressed in the namespace of the region containing the
/// instruction. Nested region inputs are substituted through their bindings and allocations local to nested regions
/// are dropped, so a caller sees exactly which of its own roots the instruction touches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceTransitiveAccess {
    /// Access modes per root, in canonical root order.
    accesses: BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>>,
}

impl ReferenceTransitiveAccess {
    /// Returns the access modes performed on each root, in canonical root order.
    #[inline]
    pub fn accesses(&self) -> &BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>> {
        &self.accesses
    }

    /// Returns the accessed roots, in canonical root order.
    #[inline]
    pub fn roots(&self) -> impl Iterator<Item = ReferenceRoot> + '_ {
        self.accesses.keys().copied()
    }

    /// Returns the access modes performed on `root`, in [`ReferenceAccessMode`] declaration order.
    #[inline]
    pub fn modes(&self, root: ReferenceRoot) -> impl Iterator<Item = ReferenceAccessMode> + '_ {
        self.accesses.get(&root).into_iter().flatten().copied()
    }

    /// Returns whether any access writes, swaps, or accumulates into `root`.
    #[inline]
    pub fn is_mutated(&self, root: ReferenceRoot) -> bool {
        self.modes(root).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate
            )
        })
    }

    /// Returns whether the instruction accesses no root at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.accesses.is_empty()
    }
}

/// Reference topology, access, and lifetime analysis of one [`Region`](crate::Region) closure. This is kernel-owned
/// validation infrastructure that consumers (e.g., kernel-boundary validation, diagnostics, and lowering) invoke
/// explicitly on the programs they own. It is not a standing lint that every program pays for.
///
/// The analysis resolves every reference-typed value of the closure to exactly one canonical [`ReferenceRoot`] in the
/// namespace of the region containing it: the region's own reference-typed inputs, its own allocations, and the
/// capture roots it inherits from an enclosing capture scope (the first `capture_count` inputs of the analyzed region,
/// or the fresh prefix an operation declares through [`Operation::region_capture_input_count`]). Nested regions are
/// analyzed in their own namespace, shared regions exactly once; each attachment records one
/// [`ReferenceRegionInputBinding`] per reference-typed region input, and the attaching instruction's
/// [`ReferenceTransitiveAccess`] summary is expressed in the caller's namespace with nested-local allocations dropped.
///
/// Along the way it enforces the reference model: operation semantics and region hooks must be well-formed, only
/// complete-value handles cross region boundaries (derived views neither enter nor leave attached regions), a
/// reference-typed output of a region-carrying operation must preserve its identity-constrained input root or be
/// forwarded consistently from region outputs that are not nested-local allocations, attached regions may only perform
/// the access modes their operation permits on entering roots, and consumption must go through a complete-value handle
/// in the root's allocating region with no later access in program order. External roots (the analyzed region's inputs,
/// classified by [`ReferenceSource`]) are never consumed.
///
/// Every accessor is deterministic: roots, values, and summaries are stored in ordered maps, and accesses and bindings
/// in program order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceAnalysis {
    /// Analyzed region.
    region: RegionId,

    /// Every root of the closure, in canonical root order, with its external source, transitive modes, and consumer.
    roots: BTreeMap<ReferenceRoot, RootRecord>,

    /// Resolution of every reference-typed value of the closure.
    values: BTreeMap<ValueId, ValueRecord>,

    /// Direct accesses, in program order within each region.
    accesses: Vec<ReferenceAccess>,

    /// Region input bindings, in program order of the attaching instructions.
    region_input_bindings: Vec<ReferenceRegionInputBinding>,

    /// Transitive access summaries of the instructions that reach at least one root.
    transitive_accesses: BTreeMap<InstructionId, ReferenceTransitiveAccess>,

    /// Root denoted by each output of the analyzed region, or [`None`] for value outputs.
    output_roots: Vec<Option<ReferenceRoot>>,
}

impl ReferenceAnalysis {
    /// Analyzes the complete closure of `region` in program order and returns the resulting [`ReferenceAnalysis`].
    ///
    /// Reference-typed inputs of `region` become [`ReferenceRoot::RegionInput`] roots classified by
    /// [`ReferenceSource::from_flat_input_index`] relative to `capture_count`, and the first `capture_count` inputs
    /// form the capture scope through which reference-typed constants of `region` and of every region inheriting that
    /// scope are resolved. Attached regions are analyzed recursively, shared regions exactly once.
    ///
    /// # Parameters
    ///
    ///   - `region`: Region whose closure is analyzed.
    ///   - `capture_count`: Number of leading inputs of `region` that originate in a lifted capture table.
    ///     Reference-typed constants are resolved to capture positions through [`Value::capture_index`].
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceAnalysisError`] naming the first violated rule in program order.
    pub fn new<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        capture_count: usize,
    ) -> Result<Self, ReferenceAnalysisError> {
        let input_ids = region.input_ids();
        if capture_count > input_ids.len() {
            return Err(ReferenceAnalysisError::InvalidCaptureScope {
                region: region.id(),
                message: format!(
                    "the capture prefix of {capture_count} inputs exceeds the region's {} inputs",
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
            .collect::<CaptureScope>();
        let mut traversal = Traversal {
            entry: region,
            capture_count,
            analysis: Self {
                region: region.id(),
                roots: BTreeMap::new(),
                values: BTreeMap::new(),
                accesses: Vec::new(),
                region_input_bindings: Vec::new(),
                transitive_accesses: BTreeMap::new(),
                output_roots: Vec::new(),
            },
            summaries: HashMap::new(),
        };
        let summary = traversal.analyze_region(region, scope)?;
        traversal.analysis.output_roots =
            summary.outputs.into_iter().map(|output| output.map(|(root, _)| root)).collect();
        Ok(traversal.analysis)
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
    /// [`None`] for allocations and for inputs of nested regions.
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

/// [`Region`] [`Transform`] marker for retained [`ReferenceAnalysis`] artifacts.
pub(crate) struct ReferenceAnalysisTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for ReferenceAnalysisTransform {
    type Arguments = ReferenceAnalysisTransformArguments;
    type Artifact = TransformArtifact<V, O, Arc<ReferenceAnalysis>>;

    const DEFAULT_CACHE_CAPACITY: usize = 2;
}

/// Argument key for one retained [`ReferenceAnalysisTransform`].
///
/// A [`ReferenceAnalysis`] records concrete [`RegionId`], [`InstructionId`], and [`ValueId`]s, while a region's
/// transform cache is deliberately shared across topology-preserving imports that renumber attached regions. The capture
/// count alone would therefore serve a rebased copy records that name the original arena's identifiers, so the key also
/// carries the closure's region identifiers in first-encounter structural order (refer to the documentation of
/// [`RegionRef::region_ids_in_closure`]). Equal keys then guarantee that every recorded identifier is still valid: a
/// rebased copy gets its own entry, and repeated analysis of an unmoved region hits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ReferenceAnalysisTransformArguments {
    /// Number of leading analyzed-region inputs that originate in a lifted capture table.
    capture_count: usize,

    /// Region identifiers of the analyzed closure in first-encounter structural order.
    regions: Vec<RegionId>,
}

impl ReferenceAnalysisTransformArguments {
    /// Creates the key of the retained analysis of `region`'s closure under `capture_count` lifted captures. The
    /// same key identifies every overlay derived from that analysis (e.g., the retained
    /// [`ReferenceViewAnalysis`](crate::programs::references::ReferenceViewAnalysis)), so all of them share one cache
    /// identity and one revalidation rule.
    pub(crate) fn new<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        capture_count: usize,
    ) -> Self {
        Self { capture_count, regions: region.region_ids_in_closure() }
    }

    /// Returns the number of leading analyzed-region inputs that originate in a lifted capture table.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.capture_count
    }
}

impl<'r, V: Value, O: Operation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Returns the [`ReferenceAnalysis`] of this [`Region`]'s closure, retained in the region's transform cache so
    /// that discharge, kernel validation, and every transform rule consulting the same closure share one analysis.
    /// The analysis is a pure structural function of the closure and `capture_count`, because reference-typed capture
    /// constants resolve through [`Value::capture_index`], and it is keyed by the closure's region identifiers as
    /// well, so a topology-preserving import that renumbers regions derives its own entry instead of being served
    /// identifiers from another arena. Refer to the documentation of [`ReferenceAnalysis::new`] for the analysis
    /// itself; that function remains the uncached path.
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
        let arguments = ReferenceAnalysisTransformArguments::new(self, capture_count);
        let artifact =
            self.transform::<ReferenceAnalysisTransform, _, ReferenceAnalysisError>(arguments, |region, arguments| {
                let analysis = ReferenceAnalysis::new(region, arguments.capture_count())?;
                Ok(TransformArtifact::new(Vec::new(), Arc::new(analysis)))
            })?;
        let (programs, analysis) = artifact.into_parts();
        assert!(programs.is_empty(), "reference analysis transform retained a program");
        Ok(analysis)
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

/// Active capture scope: the root bound at each capture position, or [`None`] where the position carries a value.
type CaptureScope = Rc<[Option<ReferenceRoot>]>;

/// Result of analyzing one region once, in that region's own namespace.
#[derive(Clone, Debug)]
struct RegionSummary {
    /// Capture scope the region was analyzed under.
    scope: CaptureScope,

    /// Direct and transitive access modes per root, including the region's local allocations.
    accesses: BTreeMap<ReferenceRoot, BTreeSet<ReferenceAccessMode>>,

    /// Root and narrowing of each region output, or [`None`] for value outputs.
    outputs: Vec<Option<(ReferenceRoot, bool)>>,
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

/// Nested-namespace root translated into the attaching instruction's namespace.
enum Substituted {
    /// The root denotes a caller root.
    Caller(ReferenceRoot),

    /// The root is an allocation local to the attached region, performed by the given instruction.
    Local(InstructionId),
}

/// Translates a root of an attached region's namespace into the attaching instruction's namespace: the attached
/// region's own inputs resolve through their bindings, capture roots of enclosing scopes pass through unchanged, and
/// allocations are local to the attached region or to one of its descendants.
fn substitute(root: ReferenceRoot, attached: RegionId, entering: &[Option<ReferenceRoot>]) -> Substituted {
    match root {
        // Every reference-typed input of the attached region is bound before the region is analyzed, so the binding
        // exists by construction.
        ReferenceRoot::RegionInput { region, input_index } if region == attached => {
            Substituted::Caller(entering[input_index].unwrap())
        }
        ReferenceRoot::RegionInput { .. } => Substituted::Caller(root),
        ReferenceRoot::Allocation { instruction, .. } => Substituted::Local(instruction),
    }
}

/// Resolves the caller root that provenance `origin` supplies to `output_index` of the attaching instruction.
fn forwarded_root(
    operation: &'static str,
    instruction: InstructionId,
    output_index: usize,
    origin: OutputRegionProvenance,
    attached: &[AttachedRegion],
) -> Result<ReferenceRoot, ReferenceAnalysisError> {
    let malformed = |message: String| ReferenceAnalysisError::MalformedSemantics { operation, instruction, message };
    let region = attached.get(origin.region_index).ok_or_else(|| {
        malformed(format!(
            "output {output_index} forwards region {} output {}, but the application attaches {} regions",
            origin.region_index,
            origin.output_index,
            attached.len(),
        ))
    })?;
    let forwarded = region.outputs.get(origin.output_index).ok_or_else(|| {
        malformed(format!(
            "output {output_index} forwards region {} output {}, but that region has {} outputs",
            origin.region_index,
            origin.output_index,
            region.outputs.len(),
        ))
    })?;
    let Some((root, narrows)) = *forwarded else {
        return Err(malformed(format!(
            "output {output_index} forwards region {} output {}, which is not a reference",
            origin.region_index, origin.output_index,
        )));
    };
    if narrows {
        return Err(ReferenceAnalysisError::ViewCrossesRegionBoundary {
            operation,
            instruction,
            region_index: origin.region_index,
            boundary: "output",
            index: origin.output_index,
        });
    }
    match substitute(root, region.id, &region.entering) {
        Substituted::Caller(root) => Ok(root),
        Substituted::Local(allocation) => Err(ReferenceAnalysisError::EscapingLocalAllocation {
            operation,
            instruction,
            output_index,
            region_index: origin.region_index,
            allocation,
        }),
    }
}

/// Mutable state of one [`ReferenceAnalysis::new`] traversal.
struct Traversal<'r, V: Value, O: Operation<Type = V::Type>> {
    /// Analyzed region, whose inputs are the external roots.
    entry: RegionRef<'r, V, O>,

    /// Number of leading analyzed-region inputs that originate in a lifted capture table.
    capture_count: usize,

    /// Analysis being accumulated.
    analysis: ReferenceAnalysis,

    /// Summaries of the regions analyzed so far, so shared regions are analyzed once.
    summaries: HashMap<RegionId, RegionSummary>,
}

impl<'r, V: Value, O: Operation<Type = V::Type>> Traversal<'r, V, O> {
    /// Analyzes `region` under `scope`, or returns its memoized summary when it was analyzed before.
    fn analyze_region(
        &mut self,
        region: RegionRef<'r, V, O>,
        scope: CaptureScope,
    ) -> Result<RegionSummary, ReferenceAnalysisError> {
        let region_id = region.id();
        if let Some(summary) = self.summaries.get(&region_id) {
            if summary.scope != scope {
                return Err(ReferenceAnalysisError::InvalidCaptureScope {
                    region: region_id,
                    message: "shared region is reached under two different capture scopes".to_string(),
                });
            }
            return Ok(summary.clone());
        }
        let is_entry = region_id == self.entry.id();
        let atoms = region.atoms();
        let is_reference = |atom: AtomId| atoms[atom.index()].r#type().is_reference();
        let value_id = |atom: AtomId| ValueId::new(region_id, atom);

        // Reference-typed inputs seed the region's own roots. Only the analyzed region's inputs are external.
        for (input_index, input) in region.input_ids().iter().copied().enumerate() {
            if !is_reference(input) {
                continue;
            }
            let root = ReferenceRoot::RegionInput { region: region_id, input_index };
            let source = is_entry.then(|| ReferenceSource::from_flat_input_index(input_index, self.capture_count));
            self.analysis.roots.insert(root, RootRecord { source, ..RootRecord::default() });
            self.analysis.values.insert(value_id(input), ValueRecord { root, narrows: false, alias: None });
        }

        // Reference-typed constants resolve through the active capture scope. Materializing one is not an access.
        for (index, atom) in atoms.iter().enumerate() {
            let Atom::Constant(constant) = atom else {
                continue;
            };
            if !constant.r#type().is_reference() {
                continue;
            }
            let atom_id = AtomId::new(index);
            let Some(capture_index) = constant.capture_index() else {
                return Err(ReferenceAnalysisError::ReferenceConstant { region: region_id, atom: atom_id });
            };
            let Some(root) = scope.get(capture_index).copied().flatten() else {
                return Err(ReferenceAnalysisError::CaptureOutOfScope {
                    region: region_id,
                    atom: atom_id,
                    capture_index,
                    capture_count: scope.len(),
                });
            };
            self.analysis.values.insert(value_id(atom_id), ValueRecord { root, narrows: false, alias: None });
        }

        let mut summary = RegionSummary { scope: Rc::clone(&scope), accesses: BTreeMap::new(), outputs: Vec::new() };
        let mut consumed = BTreeMap::<ReferenceRoot, InstructionId>::new();
        for (index, instruction) in region.instructions().iter().enumerate() {
            let id = InstructionId::new(region_id, index);
            let operation = instruction.operation();
            let name = operation.name();
            let malformed = |message: String| ReferenceAnalysisError::MalformedSemantics {
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
            let semantics = operation.reference_semantics();
            for access in semantics.inputs() {
                let input_index = access.input_index();
                let atom = input_atom(input_index, "accessed")?;
                let record = self.resolve(value_id(atom), name, id, input_index)?;
                let (root, mode) = (record.root, access.mode());
                if let Some(consumer) = consumed.get(&root) {
                    return Err(use_after_consume(root, *consumer));
                }
                if mode.is_consuming() {
                    if record.narrows {
                        return Err(ReferenceAnalysisError::ConsumeThroughView {
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
            for output in semantics.outputs() {
                let output_index = output.output_index();
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
                let record = match *output {
                    ReferenceOutput::Allocation { .. } => {
                        let root = ReferenceRoot::Allocation { instruction: id, output_index };
                        self.analysis.roots.insert(root, RootRecord::default());
                        ValueRecord { root, narrows: false, alias: None }
                    }
                    ReferenceOutput::Alias { input_index, kind, .. } => {
                        let source_atom = input_atom(input_index, "aliased")?;
                        let source = self.resolve(value_id(source_atom), name, id, input_index)?;
                        let narrows = kind == ReferenceAliasKind::View || source.narrows;
                        let alias = ReferenceAliasEdge {
                            instruction: id,
                            output_index,
                            source: value_id(source_atom),
                            kind,
                            narrows,
                        };
                        ValueRecord { root: source.root, narrows, alias: Some(alias) }
                    }
                };
                self.analysis.values.insert(value_id(atom), record);
            }

            // Attached regions are entered through their declared input provenance and analyzed in their own
            // namespace. Their transitive accesses are then substituted into this region's namespace, validated
            // against the operation's region access policy and against earlier consumption, and folded into this
            // instruction's summary.
            let mut attached = Vec::with_capacity(instruction.regions().len());
            for (region_index, attached_id) in instruction.regions().iter().copied().enumerate() {
                let nested = region
                    .with_id(attached_id)
                    .map_err(|error| malformed(format!("attached region {attached_id} cannot be resolved: {error}")))?;
                let nested_inputs = nested.input_ids();
                let nested_is_reference = |input: AtomId| nested.atoms()[input.index()].r#type().is_reference();
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
                                nested_is_reference(*input)
                                    .then_some(ReferenceRoot::RegionInput { region: attached_id, input_index })
                            })
                            .collect()
                    }
                };
                let mut entering = Vec::with_capacity(nested_inputs.len());
                for (input_index, input) in nested_inputs.iter().copied().enumerate() {
                    if !nested_is_reference(input) {
                        entering.push(None);
                        continue;
                    }
                    let Some(supplying_index) = operation.input_region_provenance(region_index, input_index) else {
                        return Err(ReferenceAnalysisError::UndeclaredRegionInputProvenance {
                            operation: name,
                            instruction: id,
                            region_index,
                            input_index,
                        });
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
                    });
                    entering.push(Some(record.root));
                }
                let nested_summary = self.analyze_region(nested, nested_scope)?;
                for (nested_root, modes) in &nested_summary.accesses {
                    let Substituted::Caller(root) = substitute(*nested_root, attached_id, &entering) else {
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
                let record = match operation.reference_output_identity_input(output_index) {
                    Some(input_index) => {
                        let atom = input_atom(input_index, "identity-preserved")?;
                        let source = self.resolve(value_id(atom), name, id, input_index)?;
                        for origin in provenance {
                            let actual = forwarded_root(name, id, output_index, origin, &attached)?;
                            if actual != source.root {
                                return Err(ReferenceAnalysisError::FixedPointRootMismatch {
                                    operation: name,
                                    instruction: id,
                                    output_index,
                                    region_index: origin.region_index,
                                    region_output_index: origin.output_index,
                                    expected: source.root,
                                    actual,
                                });
                            }
                        }
                        let alias = ReferenceAliasEdge {
                            instruction: id,
                            output_index,
                            source: value_id(atom),
                            kind: ReferenceAliasKind::Identity,
                            narrows: source.narrows,
                        };
                        ValueRecord { root: source.root, narrows: source.narrows, alias: Some(alias) }
                    }
                    None => {
                        let mut forwarded = None;
                        for origin in provenance {
                            let root = forwarded_root(name, id, output_index, origin, &attached)?;
                            match forwarded {
                                None => forwarded = Some(root),
                                Some(first) if first != root => {
                                    return Err(ReferenceAnalysisError::InconsistentForwardedRoots {
                                        operation: name,
                                        instruction: id,
                                        output_index,
                                        first,
                                        other: root,
                                    });
                                }
                                Some(_) => {}
                            }
                        }
                        let Some(root) = forwarded else {
                            return Err(ReferenceAnalysisError::UndeclaredReferenceOutput {
                                operation: name,
                                instruction: id,
                                output_index,
                            });
                        };
                        ValueRecord { root, narrows: false, alias: None }
                    }
                };
                self.analysis.values.insert(value_id(output), record);
            }
        }

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
        self.summaries.insert(region_id, summary.clone());
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

    /// Rejects consumption of `root` unless `region` is the region that allocated it.
    fn validate_consumption(
        &self,
        operation: &'static str,
        instruction: InstructionId,
        region: RegionId,
        root: ReferenceRoot,
    ) -> Result<(), ReferenceAnalysisError> {
        match root {
            ReferenceRoot::Allocation { instruction: allocation, .. } if allocation.region() == region => Ok(()),
            ReferenceRoot::RegionInput { region: root_region, input_index } if root_region == self.entry.id() => {
                Err(ReferenceAnalysisError::ConsumeExternal {
                    operation,
                    instruction,
                    root,
                    external_source: ReferenceSource::from_flat_input_index(input_index, self.capture_count),
                })
            }
            _ => Err(ReferenceAnalysisError::ConsumeOutsideCreationScope { operation, instruction, region, root }),
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
            .accesses
            .entry(root)
            .or_default()
            .insert(mode);
        summary.accesses.entry(root).or_default().insert(mode);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArraySliceAxis, ArrayType, DataType,
        ReferenceIndexOperation, ReferenceSliceOperation,
    };
    use crate::captures::CaptureReference;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::{AddOperation, ConditionOperation, WhileOperation};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::ProgramError;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::identities::NoIdentity;
    use crate::programs::instructions::{Instruction, InstructionId};
    use crate::programs::operations::Operation;
    use crate::programs::programs::Program;
    use crate::programs::references::operations::{
        ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::programs::references::semantics::{
        ReferenceAccessMode, ReferenceAliasKind, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput,
    };
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
    /// fresh capture scopes, while-, condition-, and scan-like structured operations, one region operation declaring
    /// no reference hooks, and one operation with caller-supplied (possibly malformed) semantics.
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
        Malformed(ReferenceOperationSemantics),
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

        fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::Call | Self::CallWithCaptures(_) | Self::While => Some(input_index),
                Self::Condition => Some(input_index + 1),
                Self::Scan { carry_count } => (input_index < *carry_count).then_some(input_index),
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

        fn effects(&self) -> Effects {
            match self {
                Self::Read | Self::Write | Self::Swap | Self::Accumulate | Self::Consume => {
                    Effects::single(Effect::OrderedState)
                }
                _ => Effects::PURE,
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            let access = |mode| ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, mode)], Vec::new());
            let alias = |kind| {
                ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceOutput::Alias { output_index: 0, input_index: 0, kind }],
                )
            };
            let semantics = match self {
                Self::New => {
                    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }])
                }
                Self::Read => access(ReferenceAccessMode::Read),
                Self::Write => access(ReferenceAccessMode::Write),
                Self::Swap => access(ReferenceAccessMode::ReadWrite),
                Self::Accumulate => access(ReferenceAccessMode::Accumulate),
                Self::Consume => access(ReferenceAccessMode::Consume),
                Self::View => alias(ReferenceAliasKind::View),
                Self::Identity => alias(ReferenceAliasKind::Identity),
                Self::Malformed(semantics) => return Cow::Borrowed(semantics),
                _ => return Cow::Borrowed(ReferenceOperationSemantics::empty()),
            };
            Cow::Owned(semantics)
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

    #[test]
    fn test_reference_analysis_error() {
        let cases = [
            (
                ReferenceAnalysisError::MalformedSemantics {
                    operation: "test.malformed",
                    instruction: id(0, 1),
                    message: "accessed input 3 is out of range for an application with 1 inputs".to_string(),
                },
                "operation `test.malformed` at ^0[1] declares malformed reference semantics: accessed input 3 is out \
                 of range for an application with 1 inputs",
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
                ReferenceAnalysisError::ReferenceConstant { region: RegionId::new(2), atom: AtomId::new(3) },
                "region ^2 stores reference-typed constant %3 that names no capture; references enter a program only \
                 through inputs and captures",
            ),
            (
                ReferenceAnalysisError::CaptureOutOfScope {
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
                ReferenceAnalysisError::UndeclaredRegionInputProvenance {
                    operation: "test.opaque",
                    instruction: id(1, 0),
                    region_index: 0,
                    input_index: 2,
                },
                "operation `test.opaque` at ^1[0] passes a reference into region 0 input 2 without declaring which \
                 input supplies it",
            ),
            (
                ReferenceAnalysisError::UndeclaredReferenceOutput {
                    operation: "test.opaque",
                    instruction: id(1, 0),
                    output_index: 1,
                },
                "operation `test.opaque` at ^1[0] produces a reference at output 1 without declaring whether it \
                 allocates, aliases, preserves an input identity, or forwards a region output",
            ),
            (
                ReferenceAnalysisError::InconsistentForwardedRoots {
                    operation: "test.condition",
                    instruction: id(2, 0),
                    output_index: 0,
                    first: input_root(2, 1),
                    other: input_root(2, 2),
                },
                "operation `test.condition` at ^2[0] forwards output 0 from region outputs that denote different \
                 reference roots, region ^2 input 1 and region ^2 input 2",
            ),
            (
                ReferenceAnalysisError::FixedPointRootMismatch {
                    operation: "test.while",
                    instruction: id(2, 0),
                    output_index: 0,
                    region_index: 1,
                    region_output_index: 0,
                    expected: input_root(2, 0),
                    actual: input_root(2, 1),
                },
                "operation `test.while` at ^2[0] constrains output 0 to preserve region ^2 input 0, but region 1 \
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
                ReferenceAnalysisError::ConsumeThroughView {
                    operation: "test.consume",
                    instruction: id(0, 2),
                    input_index: 0,
                    root: allocation_root(0, 0, 0),
                },
                "operation `test.consume` at ^0[2] consumes a derived view of allocation at ^0[0] output 0 through \
                 input 0, but consumption invalidates the complete alias family; consume the root handle instead",
            ),
            (
                ReferenceAnalysisError::ConsumeExternal {
                    operation: "test.consume",
                    instruction: id(0, 0),
                    root: input_root(0, 0),
                    external_source: ReferenceSource::Capture { index: 0 },
                },
                "operation `test.consume` at ^0[0] consumes external reference region ^0 input 0 (capture 0), which \
                 its caller owns",
            ),
            (
                ReferenceAnalysisError::ConsumeOutsideCreationScope {
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
            assert_eq!(ProgramError::from(error.clone()), ProgramError::MalformedProgram(expected.to_string()));
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
        let edge = ReferenceAliasEdge::new(id(1, 1), 2, value(1, 3), ReferenceAliasKind::View, true);
        assert_eq!(edge.instruction(), id(1, 1));
        assert_eq!(edge.output_index(), 2);
        assert_eq!(edge.source(), value(1, 3));
        assert_eq!(edge.kind(), ReferenceAliasKind::View);
        assert!(edge.narrows());
        assert_eq!(edge, ReferenceAliasEdge::new(id(1, 1), 2, value(1, 3), ReferenceAliasKind::View, true));
        assert_ne!(edge, ReferenceAliasEdge::new(id(1, 1), 0, value(1, 3), ReferenceAliasKind::View, true));
        assert_ne!(edge, ReferenceAliasEdge::new(id(1, 1), 2, value(1, 3), ReferenceAliasKind::Identity, true));
    }

    #[test]
    fn test_reference_region_input_binding() {
        let binding = ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1));
        assert_eq!(binding.instruction(), id(1, 7));
        assert_eq!(binding.region_index(), 0);
        assert_eq!(binding.input(), value(0, 0));
        assert_eq!(binding.root(), input_root(1, 1));
        assert_eq!(binding, ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1)));
        assert_ne!(binding, ReferenceRegionInputBinding::new(id(1, 7), 1, value(0, 0), input_root(1, 1)));
    }

    #[test]
    fn test_reference_transitive_access() {
        let analysis = fixture_analysis();
        let (a, b) = (input_root(1, 0), input_root(1, 1));

        // The call reaches `A` through the callee's capture constant and `B` through its bound input.
        let summary = analysis.transitive_access(id(1, 7)).unwrap();
        assert_eq!(
            summary.accesses(),
            &BTreeMap::from([
                (a, BTreeSet::from([ReferenceAccessMode::Read])),
                (b, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write])),
            ]),
        );
        assert_eq!(summary.roots().collect::<Vec<_>>(), vec![a, b]);
        assert_eq!(summary.modes(a).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
        assert_eq!(summary.modes(b).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write]);
        assert_eq!(summary.modes(allocation_root(1, 0, 0)).count(), 0);
        assert!(!summary.is_mutated(a));
        assert!(summary.is_mutated(b));
        assert!(!summary.is_empty());
        assert!(ReferenceTransitiveAccess::default().is_empty());
        assert_eq!(summary.clone(), *summary);
    }

    #[test]
    fn test_reference_analysis_new() {
        let program = fixture();
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), 1).unwrap();
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
        assert_eq!(analysis.region_input_bindings(), &[ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), b)]);
        assert_eq!(analysis.output_roots(), &[None, Some(b), None]);

        // A region without references analyzes to an empty artifact.
        let mut builder = TestBuilder::new();
        let input = builder.add_input(value_type(0));
        let program = build(builder, vec![input]);
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
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
            analysis.transitive_access(id(2, 0)).unwrap().accesses(),
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
        assert_eq!(analysis.region_input_bindings(), &[ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a)]);
        assert_eq!(
            analysis.accesses(),
            &[
                ReferenceAccess::new(id(0, 0), 0, k, ReferenceAccessMode::Write),
                ReferenceAccess::new(id(0, 1), 0, k, ReferenceAccessMode::Read),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(1, 0)).unwrap().accesses(),
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
            Some(ReferenceAliasEdge::new(id(2, 0), 0, value(2, 0), ReferenceAliasKind::Identity, false))
        );
        assert!(!analysis.is_view(value(2, 2)));
        assert_eq!(
            analysis.region_input_bindings(),
            &[
                ReferenceRegionInputBinding::new(id(2, 0), 0, value(0, 0), a),
                ReferenceRegionInputBinding::new(id(2, 0), 1, value(1, 0), a),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().accesses(),
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
            Some(ReferenceAliasEdge::new(id(1, 0), 0, value(1, 0), ReferenceAliasKind::Identity, false))
        );
        assert_eq!(analysis.region_input_bindings(), &[ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a)]);
        assert_eq!(
            analysis.transitive_access(id(1, 0)).unwrap().accesses(),
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
                ReferenceRegionInputBinding::new(id(2, 0), 0, value(0, 0), a),
                ReferenceRegionInputBinding::new(id(2, 0), 1, value(1, 0), a),
            ],
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().accesses(),
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
                ReferenceRegionInputBinding::new(id(1, 0), 0, value(0, 0), a),
                ReferenceRegionInputBinding::new(id(1, 0), 1, value(0, 0), a),
            ],
        );
        assert_eq!(analysis.output_roots(), &[Some(a)]);
    }

    #[test]
    fn test_reference_analysis_new_rejects_malformed_semantics() {
        // The checked builder path rejects out-of-range semantics itself, so the malformed applications are assembled
        // through the unchecked rebuild hatch.
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type(0));
        let semantics =
            ReferenceOperationSemantics::new(vec![ReferenceInput::new(3, ReferenceAccessMode::Read)], Vec::new());
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(semantics),
            vec![reference],
            Vec::new(),
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::MalformedSemantics { operation: "test.malformed", instruction, message })
                if instruction == id(0, 0)
                    && message == "accessed input 3 is out of range for an application with 1 inputs",
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let semantics =
            ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 2 }]);
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(semantics),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::MalformedSemantics { operation: "test.malformed", instruction, message })
                if instruction == id(0, 0)
                    && message == "classified output 2 is out of range for an application with 0 outputs",
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let output = builder.add_variable(value_type(1));
        let semantics =
            ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }]);
        builder.add_instruction_unchecked(Instruction::new(
            TestOperation::Malformed(semantics),
            Vec::new(),
            vec![output],
            Vec::new(),
        ));
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::MalformedSemantics { operation: "test.malformed", instruction, message })
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
            ReferenceAnalysis::new(program.entry_region_ref(), 0),
            Err(ReferenceAnalysisError::CaptureOutOfScope { region, atom, capture_index: 0, capture_count: 0 })
                if region == RegionId::new(0) && atom == AtomId::new(0),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_captures_out_of_scope() {
        // A capture index past the scope is rejected, and so is one whose scope position binds a value.
        let mut builder = TestBuilder::new();
        builder.add_input(reference_type(0));
        let captured = builder.add_constant(capture(1, 0));
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(1),
            Err(ReferenceAnalysisError::CaptureOutOfScope { region, atom, capture_index: 1, capture_count: 1 })
                if region == RegionId::new(0) && atom == AtomId::new(1),
        ));

        let mut builder = TestBuilder::new();
        builder.add_input(value_type(0));
        let captured = builder.add_constant(capture(0, 0));
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        let program = build(builder, Vec::new());
        assert!(matches!(
            program.reference_analysis(1),
            Err(ReferenceAnalysisError::CaptureOutOfScope { region, atom, capture_index: 0, capture_count: 1 })
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
            Err(ReferenceAnalysisError::UndeclaredRegionInputProvenance {
                operation: "test.opaque",
                instruction,
                region_index: 0,
                input_index: 0,
            }) if instruction == id(1, 0),
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
            Err(ReferenceAnalysisError::UndeclaredReferenceOutput {
                operation: "test.opaque",
                instruction,
                output_index: 0,
            }) if instruction == id(1, 0),
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
            Err(ReferenceAnalysisError::InconsistentForwardedRoots {
                operation: "test.condition",
                instruction,
                output_index: 0,
                first,
                other,
            }) if instruction == id(2, 0) && first == input_root(2, 1) && other == input_root(2, 2),
        ));
    }

    #[test]
    fn test_reference_analysis_new_rejects_fixed_point_root_mismatch() {
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
            Err(ReferenceAnalysisError::FixedPointRootMismatch {
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
            Err(ReferenceAnalysisError::ConsumeThroughView {
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
            Err(ReferenceAnalysisError::ConsumeExternal {
                operation: "test.consume",
                instruction,
                root,
                external_source: ReferenceSource::Capture { index: 0 },
            }) if instruction == id(0, 0) && root == input_root(0, 0),
        ));
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ConsumeExternal {
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
            Err(ReferenceAnalysisError::ConsumeOutsideCreationScope {
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

        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), 1).unwrap();
        let (captured, external) = (input_root(2, 0), input_root(2, 1));
        assert_eq!(analysis.roots().collect::<Vec<_>>(), vec![input_root(0, 0), input_root(1, 0), captured, external]);
        assert_eq!(analysis.external_source(captured), Some(ReferenceSource::Capture { index: 0 }));
        assert_eq!(analysis.external_source(external), Some(ReferenceSource::Input { index: 0 }));
        assert_eq!(analysis.external_source(input_root(0, 0)), None);
        assert_eq!(
            analysis.alias(value(2, 4)),
            Some(ReferenceAliasEdge::new(id(2, 0), 0, value(2, 0), ReferenceAliasKind::View, true))
        );
        assert_eq!(
            analysis.alias(value(2, 5)),
            Some(ReferenceAliasEdge::new(id(2, 1), 0, value(2, 4), ReferenceAliasKind::View, true))
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
                ReferenceRegionInputBinding::new(id(2, 4), 0, value(0, 0), external),
                ReferenceRegionInputBinding::new(id(2, 4), 1, value(1, 0), external),
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
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let reference = input_root(2, 1);
        assert_eq!(analysis.root_of(value(2, 3)), Some(reference));
        assert_eq!(
            analysis.alias(value(2, 3)),
            Some(ReferenceAliasEdge::new(id(2, 0), 1, value(2, 1), ReferenceAliasKind::Identity, false))
        );
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().accesses(),
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
        let analysis = ReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(analysis.root_of(value(2, 3)), Some(reference));
        assert_eq!(
            analysis.transitive_access(id(2, 0)).unwrap().accesses(),
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
            ReferenceAnalysis::new(program.entry_region_ref(), 0),
            Err(ReferenceAnalysisError::FixedPointRootMismatch {
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
            Some(ReferenceAliasEdge::new(id(1, 1), 0, value(1, 3), ReferenceAliasKind::View, true))
        );
        assert_eq!(
            analysis.alias(value(1, 5)),
            Some(ReferenceAliasEdge::new(id(1, 2), 0, value(1, 4), ReferenceAliasKind::Identity, true))
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
            &[ReferenceRegionInputBinding::new(id(1, 7), 0, value(0, 0), input_root(1, 1))],
        );
    }

    #[test]
    fn test_reference_analysis_transitive_access() {
        let analysis = fixture_analysis();
        let (a, b, c, k) = (input_root(1, 0), input_root(1, 1), allocation_root(1, 0, 0), input_root(0, 0));
        assert_eq!(analysis.transitive_access(id(1, 0)), None);
        assert_eq!(analysis.transitive_access(id(1, 1)), None);
        assert_eq!(
            analysis.transitive_access(id(1, 3)).unwrap().accesses(),
            &BTreeMap::from([(a, BTreeSet::from([ReferenceAccessMode::Read]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 6)).unwrap().accesses(),
            &BTreeMap::from([(c, BTreeSet::from([ReferenceAccessMode::ReadWrite]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 7)).unwrap().accesses(),
            &BTreeMap::from([
                (a, BTreeSet::from([ReferenceAccessMode::Read])),
                (b, BTreeSet::from([ReferenceAccessMode::Read, ReferenceAccessMode::Write])),
            ]),
        );
        assert_eq!(
            analysis.transitive_access(id(0, 0)).unwrap().accesses(),
            &BTreeMap::from([(k, BTreeSet::from([ReferenceAccessMode::Write]))]),
        );
        assert_eq!(
            analysis.transitive_access(id(1, 8)).unwrap().accesses(),
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
        let direct = ReferenceAnalysis::new(program.entry_region_ref(), 1).unwrap();
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
}
