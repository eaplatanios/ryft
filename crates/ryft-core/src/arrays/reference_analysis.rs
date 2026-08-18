//! Static root, access, scope, and lifetime analysis for array references.
//!
//! [`ReferenceAnalysis`] is the validated program-level counterpart of operation-local
//! [`ReferenceOperationSemantics`](crate::programs::ReferenceOperationSemantics). Operation descriptors speak only in
//! input/output indices; this module resolves those indices to canonical region-relative roots, validates second-class
//! boundaries and consumption, and records explicit substitutions at nested-region call sites. Runtime holder identity
//! remains a separate invocation concern.

// TODO(eaplatanios): Review this module. Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::collections::{HashMap, HashSet};
use std::fmt::Display;

use thiserror::Error;

use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::ir::ArrayIrType;
use crate::parameters::Parameterized;
use crate::programs::{
    Atom, AtomId, Effect, InstructionId, Operation, Program, ProgramError, ReferenceAccessMode,
    ReferenceOutputSemantics, ReferenceType, RegionId, RegionRef, RegionRole, Typed, Value, ValueId,
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
/// whichever caller root is recorded by the corresponding [`ReferenceRegionInputBinding`]. Allocations, by contrast,
/// are concrete program-local roots identified by their defining instruction and output.
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Substitution from a nested region's parameterized root to one caller root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceRegionInputBinding {
    /// Parent instruction invoking the region.
    instruction: InstructionId,

    /// Attached-region position in the operation's declared region slots.
    region_index: usize,

    /// Parameterized root inside the attached region.
    region_root: ReferenceRoot,

    /// Parent instruction operand that supplies the root.
    source_input_index: usize,

    /// Canonical root resolved in the parent region.
    source_root: ReferenceRoot,
}

impl ReferenceRegionInputBinding {
    /// Returns the parent instruction invoking the region.
    #[inline]
    pub const fn instruction(&self) -> InstructionId {
        self.instruction
    }

    /// Returns the attached-region position in the operation's region slots.
    #[inline]
    pub const fn region_index(&self) -> usize {
        self.region_index
    }

    /// Returns the parameterized root inside the attached region.
    #[inline]
    pub const fn region_root(&self) -> ReferenceRoot {
        self.region_root
    }

    /// Returns the parent instruction operand that supplies the nested input.
    #[inline]
    pub const fn source_input_index(&self) -> usize {
        self.source_input_index
    }

    /// Returns the canonical source root in the parent region.
    #[inline]
    pub const fn source_root(&self) -> ReferenceRoot {
        self.source_root
    }
}

/// Analysis-local handle state associated with one reference-typed atom.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ReferenceHandle {
    /// Canonical region-relative resource root.
    root: ReferenceRoot,

    /// Whether this handle was introduced as a fresh allocation in its defining region, as opposed to arriving as a
    /// formal region input or an explicit alias of another handle.
    is_allocation: bool,
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

    /// External roots in entry-boundary order.
    external_roots: Vec<ExternalReferenceRoot>,
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

    /// Returns nested-region root substitutions in deterministic caller order.
    #[inline]
    pub fn bindings(&self) -> &[ReferenceRegionInputBinding] {
        &self.bindings
    }

    /// Returns external roots in entry-boundary order.
    #[inline]
    pub fn external_roots(&self) -> &[ExternalReferenceRoot] {
        &self.external_roots
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
        if capture_count > self.input_ids().len() {
            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidCaptureCount {
                capture_count,
                input_count: self.input_ids().len(),
            }));
        }

        // Attachments form a Directed Acyclic Graph in which one shared region can be reachable through many paths,
        // so the closure mask is an iterative worklist with an arena-indexed visited set.
        let regions = self.arena();
        let mut included = vec![false; regions.len()];
        let mut pending = vec![self.id()];
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut included[id.index()], true) {
                continue;
            }
            for instruction in regions[id.index()].instructions() {
                pending.extend(instruction.regions().iter().copied());
            }
        }

        let mut requires_reference_analysis = false;
        'regions: for (region_index, region) in regions.iter().enumerate() {
            if !included[region_index] {
                continue;
            }
            if region.atoms().iter().any(|atom| reference_type(atom.r#type().as_ref()).is_some()) {
                requires_reference_analysis = true;
                break;
            }
            for instruction in region.instructions() {
                let semantics = instruction.operation().reference_semantics();
                if !semantics.outputs().is_empty() || !semantics.accesses().is_empty() {
                    requires_reference_analysis = true;
                    break 'regions;
                }
            }
        }
        if !requires_reference_analysis {
            return Ok(ReferenceAnalysis {
                roots: Vec::new(),
                accesses: Vec::new(),
                bindings: Vec::new(),
                external_roots: Vec::new(),
            });
        }

        let mut handles = regions.iter().map(|region| vec![None; region.atoms().len()]).collect::<Vec<_>>();
        let mut accesses = Vec::new();
        let mut bindings = Vec::new();
        let mut external_roots = Vec::new();
        let mut rule_regions = vec![false; regions.len()];
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
        while let Some(region) = pending_rule_regions.pop() {
            if std::mem::replace(&mut rule_regions[region.index()], true) {
                continue;
            }
            pending_rule_regions.extend(
                regions[region.index()]
                    .instructions()
                    .iter()
                    .flat_map(|instruction| instruction.regions().iter().copied()),
            );
        }

        for (region_index, region) in regions.iter().enumerate() {
            if !included[region_index] {
                continue;
            }
            let region_id = RegionId::new(region_index);
            let mut live_roots = HashSet::new();

            for (input_index, atom) in region.input_ids().iter().copied().enumerate() {
                if reference_type(region.atoms()[atom.index()].r#type().as_ref()).is_some() {
                    let root = ReferenceRoot::RegionInput { region: region_id, input_index };
                    handles[region_index][atom.index()] = Some(ReferenceHandle { root, is_allocation: false });
                    live_roots.insert(root);
                }
            }

            for (atom_index, atom) in region.atoms().iter().enumerate() {
                if matches!(atom, Atom::Constant(_)) && reference_type(atom.r#type().as_ref()).is_some() {
                    return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceConstant {
                        region: region_id,
                        atom: AtomId::new(atom_index),
                    }));
                }
            }

            if rule_regions[region_index]
                && (handles[region_index].iter().any(Option::is_some)
                    || region.instructions().iter().any(|instruction| {
                        let semantics = instruction.operation().reference_semantics();
                        !semantics.outputs().is_empty() || !semantics.accesses().is_empty()
                    }))
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
                    .any(|atom| reference_type(region.atoms()[atom.index()].r#type().as_ref()).is_some())
                    || instruction.regions().iter().copied().any(|attached| {
                        let attached_region = &regions[attached.index()];
                        attached_region.input_ids().iter().copied().any(|atom| {
                            reference_type(attached_region.atoms()[atom.index()].r#type().as_ref()).is_some()
                        })
                    });
                if semantics.outputs().is_empty() && semantics.accesses().is_empty() && !has_reference_boundary {
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
                    if reference_type(region.atoms()[output_atom.index()].r#type().as_ref()).is_none() {
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
                            ReferenceHandle { root, is_allocation: true }
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
                            ReferenceHandle { root: source.root, is_allocation: false }
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
                    if reference_type(input_type.as_ref()).is_none() {
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
                        // An allocation handle is always local to its defining region, so the allocation bit alone
                        // rejects region inputs, aliases, and any non-local consumption target.
                        if !handle.is_allocation {
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
                        if reference_type(attached_region.atoms()[region_input_atom.index()].r#type().as_ref())
                            .is_none()
                        {
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
                        let Some(source_atom) = instruction.inputs().get(source_input_index).copied() else {
                            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidRegionInputSource {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                                source_input_index,
                            }));
                        };
                        let Some(source_handle) = handles[region_index][source_atom.index()] else {
                            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidRegionInputSource {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                                source_input_index,
                            }));
                        };
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
                            return Err(ProgramError::custom(ReferenceAnalysisError::InvalidRegionInputSource {
                                instruction: instruction_id,
                                operation: operation_name.to_string(),
                                region_index: attached_index,
                                input_index: region_input_index,
                                source_input_index,
                            }));
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
                        classified_inputs[source_input_index] = true;
                        bindings.push(ReferenceRegionInputBinding {
                            instruction: instruction_id,
                            region_index: attached_index,
                            region_root: ReferenceRoot::RegionInput {
                                region: attached,
                                input_index: region_input_index,
                            },
                            source_input_index,
                            source_root: source_handle.root,
                        });
                    }
                }

                for (input_index, input) in instruction.inputs().iter().copied().enumerate() {
                    if reference_type(region.atoms()[input.index()].r#type().as_ref()).is_some()
                        && !classified_inputs[input_index]
                    {
                        return Err(ProgramError::custom(ReferenceAnalysisError::UnclassifiedInput {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            input_index,
                        }));
                    }
                }
                for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
                    if reference_type(region.atoms()[output.index()].r#type().as_ref()).is_some()
                        && !classified_outputs[output_index]
                    {
                        return Err(ProgramError::custom(ReferenceAnalysisError::UnclassifiedOutput {
                            instruction: instruction_id,
                            operation: operation_name.to_string(),
                            output_index,
                        }));
                    }
                }
            }

            for (output_index, output) in region.output_ids().iter().copied().enumerate() {
                if reference_type(region.atoms()[output.index()].r#type().as_ref()).is_some() {
                    return Err(ProgramError::custom(ReferenceAnalysisError::ReferenceOutput {
                        region: region_id,
                        output_index,
                        root: handles[region_index][output.index()].unwrap().root,
                    }));
                }
            }
        }

        let entry = self.id();
        for (input_index, input) in self.input_ids().iter().copied().enumerate() {
            let input_type = self.atoms()[input.index()].r#type();
            if reference_type(input_type.as_ref()).is_none() {
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
        Ok(ReferenceAnalysis { roots, accesses, bindings, external_roots })
    }
}

/// Projects an Array IR type onto its reference member when present.
#[inline]
fn reference_type(r#type: &ArrayIrType) -> Option<&ReferenceType<ArrayType>> {
    match r#type {
        ArrayIrType::Reference(reference) => Some(reference),
        ArrayIrType::Array(_) | ArrayIrType::Dimension(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::types::data::DataType;
    use crate::operations::{
        ConditionOperation, FreezeReferenceOperation, NewReferenceOperation, ReferenceReadOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        Effects, Operation, ProgramBuilder, ReferenceInputAccess, ReferenceOperationSemantics,
        ReferenceOutputSemantics, RegionArena, RegionInterface, RegionSlot, TypeError,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;

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
                Self::InvalidSourceCarrier => "test_invalid_source_carrier",
                Self::RuleCarrier => "test_rule_carrier",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            _region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::BadRoot | Self::Read => vec![scalar_type().into()],
                Self::Root | Self::MissingEffect | Self::InvalidOutput | Self::UnclassifiedOutput => {
                    vec![ReferenceType::new(scalar_type()).into()]
                }
                Self::Alias => vec![input_types[0].clone()],
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

        fn input_region_provenance(&self, region_index: usize, _input_index: usize) -> Option<usize> {
            (matches!(self, Self::InvalidSourceCarrier) && region_index == 0).then_some(1)
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
                Self::ComputationCarrier | Self::InvalidSourceCarrier | Self::RuleCarrier => {
                    ReferenceOperationSemantics::default()
                }
            })
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::ComputationCarrier | Self::InvalidSourceCarrier => const { &[RegionSlot::computation("body")] },
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
        assert!(valid.analyze_references(0).is_ok());
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
        let source_root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let branch_root = ReferenceRoot::RegionInput { region: branch, input_index: 0 };
        assert_eq!(
            analysis.bindings(),
            &[
                ReferenceRegionInputBinding {
                    instruction: InstructionId::new(program.entry(), 0),
                    region_index: 0,
                    region_root: branch_root,
                    source_input_index: 1,
                    source_root,
                },
                ReferenceRegionInputBinding {
                    instruction: InstructionId::new(program.entry(), 0),
                    region_index: 1,
                    region_root: branch_root,
                    source_input_index: 1,
                    source_root,
                },
            ],
        );
        assert_eq!(analysis.accesses()[0].root(), branch_root);
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
    fn test_reference_analysis_enforces_nested_scope_lifetimes() {
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
        assert!(program.analyze_references(0).is_ok());

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
    fn test_reference_analysis_rejects_malformed_operation_semantics_precisely() {
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
        let error = program.analyze_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::MissingOrderedState {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_missing_reference_effect".to_string(),
            }),
        );

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
    fn test_reference_analysis_resolves_aliases_and_rejects_escape_or_alias_consumption() {
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
                root: canonical_root,
            }),
        );

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
                root: canonical_root,
            }),
        );

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
                root: canonical_root,
            }),
        );

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
