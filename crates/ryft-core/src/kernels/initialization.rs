//! Definite initialization and static launch qualification for portable kernel bodies.
//!
//! Writes establish coverage only for their canonical root-relative selections. Branch joins intersect coverage;
//! loops preserve the zero-body path while including their mandatory first condition. This initial verifier rejects
//! dynamic transforms, extensions without checked memory contracts, dynamic launch shapes, and unsupported control
//! flow. Padded parameter windows admit ordinary reads only through views valid in every invocation; masked accesses
//! use explicit fallbacks.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use thiserror::Error;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform,
    ArrayReferenceTransformPath, ArraySliceAxis, ArrayType, DataType, Dimension, DimensionValue,
};
use crate::kernels::calls::{KernelCallOperation, KernelError};
use crate::kernels::grids::GridExecution;
use crate::kernels::mappings::{BlockMappingError, BoundaryPolicy};
use crate::kernels::memory::KernelMemoryError;
use crate::kernels::operations::{KernelExtension, KernelExtensionMemory, KernelOperation};
use crate::kernels::validation::{KernelParameterAccess, KernelReferenceSummary, KernelSwapLowering};
use crate::programs::{
    Atom, AtomId, InstructionId, Operation, ProgramError, ReferenceAccessMode, ReferenceEffect, ReferenceRoot,
    ReferenceType, ReferenceViewOverlap, RegionRef, Typed, ValueId,
};

/// A body or launch lacks a definite initialization or disjointness proof.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelInitializationError {
    /// An ordinary access selects invalid lanes of a padded parameter window.
    #[error(transparent)]
    Memory(#[from] KernelMemoryError),

    /// The call's type, reference boundary, or grid access contract rejected the body.
    #[error(transparent)]
    Call(#[from] KernelError),

    /// Canonical addressing or dimension binding failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// A mapping could not produce a valid operand window.
    #[error(transparent)]
    Mapping(#[from] BlockMappingError),

    /// A completion operand does not name a pending copy token from this region.
    #[error("reference {value:?} is not a pending async copy token from this region")]
    InvalidCopyToken { value: ValueId },

    /// Only a wait may access a pending copy's token resource.
    #[error("kernel instruction {instruction} accesses copy token {token:?} without `wait`")]
    InvalidCopyTokenAccess { instruction: InstructionId, token: ReferenceRoot },

    /// A memory access conflicts with an outstanding source or destination reservation.
    #[error("kernel instruction {instruction} conflicts with pending copy {token:?}")]
    PendingCopyAccess { instruction: InstructionId, token: ReferenceRoot },

    /// A copy source and destination overlap within one allocation.
    #[error("async copy at {instruction} has overlapping source and destination views")]
    OverlappingCopy { instruction: InstructionId },

    /// Pending resources require a same-region wait before entering another region.
    #[error("pending async copies cannot cross the region boundary at {instruction}")]
    CrossRegionCopy { instruction: InstructionId },

    /// A region exits while a copy still owns its reservations.
    #[error("kernel region exits without waiting for async copy {token:?}")]
    UnwaitedCopy { token: ReferenceRoot },

    /// An executable loop lacks a finite iteration bound.
    #[error("kernel loop `{instruction}` requires an iteration bound")]
    UnboundedLoop { instruction: InstructionId },

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
    InvalidExtension { instruction: InstructionId, message: String },

    /// An allocation has not been established or was already released.
    #[error("kernel reference {value:?} is not a live allocation")]
    UnavailableReference { value: ValueId },

    /// A reference access selects a view that cannot be resolved statically. Selections are per access site: the same
    /// root may be statically selected at one access and dynamically at another.
    #[error("kernel initialization cannot prove the selection of reference input {input_index} at {instruction}")]
    UnknownSelection {
        /// Accessing instruction.
        instruction: InstructionId,

        /// Position among the instruction's inputs of the reference input with a dynamic or unsupported transform path.
        input_index: usize,
    },

    /// An access reads coordinates not definitely initialized before that instruction.
    #[error("kernel instruction {instruction} reads uninitialized elements of {root:?}")]
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

    /// Static qualification cannot determine a runtime extent or mask.
    #[error("kernel initialization requires static unmasked launch metadata for {boundary}")]
    UnsupportedLaunch {
        /// Metadata whose runtime semantics need a later qualification rule.
        boundary: &'static str,
    },

    /// Exact static grid enumeration exceeds the caller's qualification budget.
    #[error("kernel qualification requires {programs} programs, exceeding the limit {maximum}")]
    QualificationLimit {
        /// Static grid size.
        programs: usize,
        /// Caller-provided enumeration limit.
        maximum: usize,
    },

    /// Two mutable windows may execute on the same operand coordinates.
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
/// The supplied region is validated against this exact call. Read-write inputs begin initialized; write-only outputs
/// begin empty. Masked windows preserve fixed logical tile types, with raw reads checked against actual valid
/// lanes and output publication clipped to the operand. Unresolved selections never establish complete writes. Static
/// separable tiling uses the canonical mapping program to check the maximal coordinate and proves coverage from
/// its axis geometry. Other mappings use bounded enumeration through the same interpreter.
/// Overlapping mutable windows require grid ordering or exclusively atomic accumulation. Atomic/non-atomic conflicts
/// remain races; deterministic host traversal does not provide ordering absent from the grid contract.
///
/// # Parameters
///
///   - `region`: actual attached body being qualified.
///   - `call`: its canonical grid, operand types, accesses, and mappings.
///   - `maximum_programs`: maximum grid size to enumerate when a mapping lacks a separable tiling proof. Such larger
///     launches are rejected before body traversal. This resource limit does not participate in kernel semantic
///     identity; interpreters separately enforce their execution limits even when no enumeration is needed here.
pub fn validate_kernel_initialization<Extension>(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    call: &KernelCallOperation,
    maximum_programs: usize,
) -> Result<(), KernelInitializationError>
where
    Extension: KernelExtension,
{
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
    let points = call.grid().points(&extents).map_err(ProgramError::custom)?;
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
    let mut initialization = Initialization {
        references: &references,
        states: BTreeMap::new(),
        types: BTreeMap::new(),
        unmasked_accesses: Vec::new(),
        pending_copies: BTreeMap::new(),
    };
    let mut bindings = BTreeMap::new();
    for (parameter, declaration) in call.parameters().iter().enumerate() {
        let root = ReferenceRoot::RegionInput { region: region.id(), input_index: parameter };
        let ArrayIrType::Reference(r#type) = declaration.body_type() else { unreachable!() };
        let r#type = r#type.referent().clone();
        let count = ArrayAddressing::new(r#type.clone())?.element_count();
        initialization.states.insert(
            root,
            if declaration.access() == KernelParameterAccess::WriteOnly || count == 0 {
                vec![]
            } else {
                vec![0..count]
            },
        );
        initialization.types.insert(root, r#type);
        bindings.insert(root, root);
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
            .ok_or(KernelInitializationError::UnsupportedLaunch { boundary: "operand shapes" })?;
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
                let repeated_axes = extents
                    .iter()
                    .enumerate()
                    .filter(|(axis, extent)| **extent > 1 && !axes.contains(&Some(*axis)))
                    .map(|(axis, _)| axis)
                    .collect::<Vec<_>>();
                // Equal windows differ only on unused axes. A single sequential axis totally orders those points;
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
    for (root, operation, axes) in &initialization.unmasked_accesses {
        if let Some(valid_shape) = valid_shapes.get(root) {
            if !axes.iter().any(|axis| axis.size() == 0)
                && axes.iter().zip(valid_shape).any(|(axis, &valid)| axis.start() + axis.size() > valid)
            {
                return Err(KernelMemoryError::UnmaskedWindowAccess { operation }.into());
            }
        }
    }

    Ok(())
}

/// Canonical accesses reserved until a completion token is consumed.
struct PendingCopy {
    /// Root, root-coordinate slice axes, and declared access mode for each outstanding access.
    accesses: Vec<(ReferenceRoot, Vec<ArraySliceAxis>, ReferenceAccessMode)>,
}

/// Ordered coverage state, keyed by canonical roots after attachment-specific input substitution.
struct Initialization<'a> {
    /// Canonical whole-closure access and view analysis.
    references: &'a KernelReferenceSummary,
    /// Definitely initialized logical intervals for every encountered root.
    states: BTreeMap<ReferenceRoot, Vec<Range<usize>>>,
    /// Static root array types used to interpret canonical selections.
    types: BTreeMap<ReferenceRoot, ArrayType>,
    /// Ordinary reads and read-write accesses checked against actual launch validity after body initialization.
    unmasked_accesses: Vec<(ReferenceRoot, &'static str, Vec<ArraySliceAxis>)>,
    /// Outstanding asynchronous access reservations keyed by their canonical completion allocation.
    pending_copies: BTreeMap<ReferenceRoot, PendingCopy>,
}

impl Initialization<'_> {
    /// Visits instructions in execution order; releases retire allocations and joins retain definite coverage.
    fn region<Extension: KernelExtension>(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(), KernelInitializationError> {
        for (index, instruction) in region.instructions().iter().enumerate() {
            let id = InstructionId::new(region.id(), index);
            if !instruction.regions().is_empty() && !self.pending_copies.is_empty() {
                return Err(KernelInitializationError::CrossRegionCopy { instruction: id });
            }
            if matches!(instruction.operation(), KernelOperation::AsyncCopy(_)) {
                let (source, source_axes) =
                    self.selection(id, 0, ValueId::new(region.id(), instruction.inputs()[0]), bindings)?;
                let (destination, destination_axes) =
                    self.selection(id, 1, ValueId::new(region.id(), instruction.inputs()[1]), bindings)?;
                if source == destination {
                    let source_path = ArrayReferenceTransformPath::<ValueId>::root()
                        .with_transform(ArrayReferenceTransform::Slice { axes: source_axes.clone() });
                    let destination_path = ArrayReferenceTransformPath::<ValueId>::root()
                        .with_transform(ArrayReferenceTransform::Slice { axes: destination_axes.clone() });
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
                let pending =
                    self.pending_copies.remove(&token).ok_or(KernelInitializationError::InvalidCopyToken { value })?;
                self.complete(pending)?;
                self.states.remove(&token);
                continue;
            }
            if let KernelOperation::Scratch(operation) = instruction.operation() {
                let root = ReferenceRoot::Allocation { instruction: id, output_index: 0 };
                self.states.insert(root, Vec::new());
                self.types.insert(root, operation.referent().clone());
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
                        let entering = self.states.clone();
                        let mut joined = None;
                        for (position, &nested) in instruction.regions().iter().enumerate() {
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
                        // The first condition always executes, including on the zero-body path. Existing lifetime
                        // validation forbids consuming entering roots; stores only add coverage, so validating the
                        // first body from this state suffices for subsequent iterations' initialization safety.
                        // This is the monotone fixed-point subset: local allocations reset at each region entry,
                        // entering roots never lose initialized elements, and reads are checked before each write.
                        // The zero-iteration path prevents crediting body-only writes after the loop.
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
        if let Some(&token) = self.pending_copies.keys().next() {
            return Err(KernelInitializationError::UnwaitedCopy { token });
        }
        Ok(())
    }

    /// Publishes deferred writes only after the corresponding completion operation.
    fn complete(&mut self, pending: PendingCopy) -> Result<(), KernelInitializationError> {
        for (root, axes, mode) in pending.accesses {
            if mode == ReferenceAccessMode::Read {
                continue;
            }
            let addressing = ArrayAddressing::new(self.types[&root].clone())?;
            let state = self.states.get_mut(&root).unwrap();
            for range in addressing.ranges(&axes)? {
                insert(state, range.elements());
            }
        }
        Ok(())
    }

    /// Applies extension timing to canonical reference effects without redefining access geometry.
    fn extension<Extension: KernelExtension>(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        id: InstructionId,
        operation: &Extension,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(), KernelInitializationError> {
        let instruction = &region.instructions()[id.index()];
        let invalid = |message: &str| KernelInitializationError::InvalidExtension {
            instruction: id,
            message: message.to_owned(),
        };
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
        match semantics {
            KernelExtensionMemory::Synchronous => {
                if !allocations.is_empty() || accesses.iter().any(|(_, mode)| *mode == ReferenceAccessMode::Consume) {
                    return Err(invalid("synchronous operations cannot allocate or consume references"));
                }
                self.accesses(region, id, bindings, None, false)?;
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
                ArrayAddressing::new(referent.clone())?;
                let root = ReferenceRoot::Allocation { instruction: id, output_index };
                if asynchronous {
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
                    self.pending_copies.insert(root, PendingCopy { accesses: reservations });
                }
                self.types.insert(root, referent);
                self.states.insert(root, Vec::new());
            }
            KernelExtensionMemory::Commit { completion_input_index: input_index } => {
                if !allocations.is_empty() || accesses != [(input_index, ReferenceAccessMode::Read)] || aliases {
                    return Err(invalid("commit must read exactly its declared completion reference"));
                }
                let input =
                    instruction.inputs().get(input_index).ok_or_else(|| invalid("completion input is absent"))?;
                let value = ValueId::new(region.id(), *input);
                let (root, _) = self.selection(id, input_index, value, bindings)?;
                if !matches!(root, ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region.id())
                    || !self.pending_copies.contains_key(&root)
                {
                    return Err(KernelInitializationError::InvalidCopyToken { value });
                }
            }
            KernelExtensionMemory::Wait { completion_input_index: input_index }
            | KernelExtensionMemory::Release { input_index } => {
                if !allocations.is_empty() || accesses != [(input_index, ReferenceAccessMode::Consume)] || aliases {
                    return Err(invalid("wait or release must consume exactly its declared reference"));
                }
                let input = instruction.inputs().get(input_index).ok_or_else(|| invalid("consumed input is absent"))?;
                let value = ValueId::new(region.id(), *input);
                let (root, _) = self.selection(id, input_index, value, bindings)?;
                if !matches!(root, ReferenceRoot::Allocation { instruction, .. } if instruction.region() == region.id())
                {
                    return Err(invalid("wait or release requires an allocation from the current region"));
                }
                if matches!(semantics, KernelExtensionMemory::Wait { .. }) {
                    let pending = self
                        .pending_copies
                        .remove(&root)
                        .ok_or(KernelInitializationError::InvalidCopyToken { value })?;
                    self.complete(pending)?;
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
    /// masks require the complete selected view for reads and cannot establish any definite write coverage.
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
        for access in self
            .references
            .analysis()
            .analysis()
            .access_modes()
            .iter()
            .filter(|access| access.instruction() == id)
        {
            let value = ValueId::new(region.id(), instruction.inputs()[access.input_index()]);
            let (root, axes) = self.selection(id, access.input_index(), value, bindings)?;
            if self.pending_copies.contains_key(&root) {
                return Err(KernelInitializationError::InvalidCopyTokenAccess { instruction: id, token: root });
            }
            if mask.is_none() && access.mode() != ReferenceAccessMode::Write {
                self.unmasked_accesses.push((root, instruction.operation().name(), axes.clone()));
            }
            let addressing = ArrayAddressing::new(self.types[&root].clone())?;
            let writes_only = access.mode() == ReferenceAccessMode::Write
                || self.references.swap_lowering(id) == Some(KernelSwapLowering::Store);
            let establishes_write = writes_only && !(mask.is_some() && known_mask.is_none()) && !deferred;
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
                    self.check_reservations(root, &range, access.mode(), id)?;
                    let state = self.states.get_mut(&root).unwrap();
                    if !writes_only && !contains(state, &range) {
                        return Err(KernelInitializationError::UninitializedRead { instruction: id, root });
                    }
                    if establishes_write {
                        insert(state, range);
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolves a reference input to its canonical allocation and the root-coordinate slice axes that its path selects.
    fn selection(
        &self,
        instruction: InstructionId,
        input_index: usize,
        value: ValueId,
        bindings: &BTreeMap<ReferenceRoot, ReferenceRoot>,
    ) -> Result<(ReferenceRoot, Vec<ArraySliceAxis>), KernelInitializationError> {
        let original = self.references.analysis().analysis().root_of(value).unwrap();
        let root = bindings.get(&original).copied().unwrap_or(original);
        let axes = self
            .references
            .path(instruction, input_index)
            .unwrap()
            .root_slice_axes(&self.types[&root])
            .ok_or(KernelInitializationError::UnknownSelection { instruction, input_index })?;
        if !self.states.contains_key(&root) {
            return Err(KernelInitializationError::UnavailableReference { value });
        }
        Ok((root, axes))
    }

    /// Rejects accesses to pending destinations and mutations of pending sources, including through masked views.
    fn check_reservations(
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
                    if addressing.ranges(axes)?.any(|reserved| {
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
}

/// Selects contiguous active runs from a root-relative interval using its corresponding logical mask entries.
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayOperation, ArrayReferenceTransformIndex, ArraySliceAxis, DataType, DimensionBounds, DimensionType,
    };
    use crate::contexts::EagerContext;
    use crate::kernels::calls::KernelParameter;
    use crate::kernels::grids::{Grid, GridDimension, GridExecution};
    use crate::kernels::mappings::BlockMapping;
    use crate::kernels::memory::{
        AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation,
        WaitOperation,
    };
    use crate::kernels::operations::NoKernelExtension;
    use crate::operations::{
        ConditionOperation, DimensionAddOperation, DimensionMulOperation, NotOperation, ReferenceAddUpdateOperation,
        ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSwapOperation, ReferenceWriteOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{FlatProgram, ProgramBuilder, ReferenceAccessDescriptor, ReferenceAccessOperation};

    use super::*;

    /// Multi-source asynchronous update fixture using canonical effects and reference types.
    #[derive(Clone, Debug)]
    enum MemoryExtension {
        Allocate(ArrayType),
        MisclassifiedAllocation,
        Synchronous,
        Update,
        Commit,
        Wait,
        Release,
    }

    impl Operation for MemoryExtension {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Allocate(_) => "test_allocate",
                Self::MisclassifiedAllocation => "test_misclassified_allocation",
                Self::Synchronous => "test_synchronous",
                Self::Update => "test_async_update",
                Self::Commit => "test_commit",
                Self::Wait => "test_wait",
                Self::Release => "test_release",
            }
        }

        fn infer_output_types(
            &self,
            _inputs: &[ArrayIrType],
            _regions: &[crate::programs::RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, crate::programs::TypeError> {
            Ok(match self {
                Self::Allocate(referent) => vec![ArrayIrType::Reference(ReferenceType::new(referent.clone()))],
                Self::Update => vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)))],
                Self::MisclassifiedAllocation => {
                    vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)))]
                }
                Self::Synchronous | Self::Commit | Self::Wait | Self::Release => vec![],
            })
        }

        fn effects(&self) -> std::borrow::Cow<'_, crate::programs::Effects> {
            use crate::programs::{EffectClasses, Effects};
            let declarations = match self {
                Self::Allocate(_) | Self::MisclassifiedAllocation => {
                    vec![ReferenceEffect::Allocate { output_index: 0 }]
                }
                Self::Synchronous => vec![],
                Self::Update => vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
                Self::Commit => vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
                Self::Wait | Self::Release => {
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }]
                }
            };
            std::borrow::Cow::Owned(Effects::new(EffectClasses::NONE, declarations).unwrap())
        }
    }

    impl ReferenceAccessOperation for MemoryExtension {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            match self {
                Self::Allocate(_) | Self::MisclassifiedAllocation | Self::Synchronous => 0,
                Self::Update => 3,
                Self::Commit | Self::Wait | Self::Release => 1,
            }
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
            self.effects().accesses().any(|(index, _)| index == input_index).then(|| {
                let count = self.base_input_count();
                ReferenceAccessDescriptor::new(&[], count..count)
            })
        }

        fn with_reference_access_transforms(
            &self,
            input_index: usize,
            transforms: Vec<Self::Transform>,
        ) -> Result<Self, ProgramError> {
            if transforms.is_empty() && self.reference_access_descriptor(input_index).is_some() {
                Ok(self.clone())
            } else {
                Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference transforms".to_owned(),
                })
            }
        }
    }

    impl KernelExtension for MemoryExtension {
        fn memory_semantics(&self) -> Result<KernelExtensionMemory, crate::programs::TypeError> {
            Ok(match self {
                Self::Allocate(_) => KernelExtensionMemory::Allocation { output_index: 0 },
                Self::MisclassifiedAllocation | Self::Synchronous => KernelExtensionMemory::Synchronous,
                Self::Update => KernelExtensionMemory::Asynchronous { completion_output_index: 0 },
                Self::Commit => KernelExtensionMemory::Commit { completion_input_index: 0 },
                Self::Wait => KernelExtensionMemory::Wait { completion_input_index: 0 },
                Self::Release => KernelExtensionMemory::Release { input_index: 0 },
            })
        }
    }

    #[test]
    fn test_validate_kernel_initialization_extension_async_update() {
        let call = call(1, 1, 1);
        for conflict in [
            None,
            Some("read"),
            Some("release"),
            Some("uninitialized source"),
            Some("commit after wait"),
            Some("commit after release"),
        ] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<MemoryExtension>>::new();
            let output = builder.add_input(call.parameters()[0].body_type());
            builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
            let r#type = ArrayType::new_static(DataType::I32, vec![1]);
            let source = builder
                .add_instruction(
                    KernelOperation::Extension(MemoryExtension::Allocate(r#type.clone())),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            let destination = builder
                .add_instruction(KernelOperation::Extension(MemoryExtension::Allocate(r#type)), vec![], vec![], None)
                .unwrap()[0];
            let value = builder.add_constant(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, vec![1]), &[3i32]).unwrap(),
            ));
            if conflict != Some("uninitialized source") {
                builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![source, value], None).unwrap();
            }
            let token = builder
                .add_instruction(
                    KernelOperation::Extension(MemoryExtension::Update),
                    vec![],
                    vec![source, source, destination],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(KernelOperation::Extension(MemoryExtension::Commit), vec![], vec![token], None)
                .unwrap();
            if conflict == Some("read") {
                builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![destination], None).unwrap();
            }
            if conflict == Some("release") {
                builder
                    .add_instruction(KernelOperation::Extension(MemoryExtension::Release), vec![], vec![source], None)
                    .unwrap();
            }
            let completion =
                if conflict == Some("commit after release") { MemoryExtension::Release } else { MemoryExtension::Wait };
            builder.add_instruction(KernelOperation::Extension(completion), vec![], vec![token], None).unwrap();
            if matches!(conflict, Some("commit after wait" | "commit after release")) {
                let consumer = if conflict == Some("commit after wait") { "test_wait" } else { "test_release" };
                assert_eq!(
                    builder.add_instruction(
                        KernelOperation::Extension(MemoryExtension::Commit),
                        vec![],
                        vec![token],
                        None
                    ),
                    Err(ProgramError::MalformedProgram(format!(
                        "`test_commit` reads a reference whose alias family `{consumer}` already consumed",
                    ))),
                );
                continue;
            }
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![destination], None).unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            builder
                .add_instruction(KernelOperation::Extension(MemoryExtension::Release), vec![], vec![destination], None)
                .unwrap();
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            let result = validate_kernel_initialization(body.entry_region_ref(), &call, 1);
            match conflict {
                None => assert_eq!(result, Ok(())),
                Some("uninitialized source") => assert_eq!(
                    result,
                    Err(KernelInitializationError::UninitializedRead {
                        instruction: InstructionId::new(region, 2),
                        root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
                    })
                ),
                _ => assert_eq!(
                    result,
                    Err(KernelInitializationError::PendingCopyAccess {
                        instruction: InstructionId::new(region, 5),
                        token: ReferenceRoot::Allocation {
                            instruction: InstructionId::new(region, 3),
                            output_index: 0
                        },
                    })
                ),
            }
        }
    }

    #[test]
    fn test_validate_kernel_initialization_extension_classification() {
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        for (operation, valid) in
            [(MemoryExtension::Synchronous, true), (MemoryExtension::MisclassifiedAllocation, false)]
        {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<MemoryExtension>>::new();
            builder.add_instruction(KernelOperation::Extension(operation), vec![], vec![], None).unwrap();
            let body =
                builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![]).unwrap();
            let result = validate_kernel_initialization(body.entry_region_ref(), &call, 1);
            if valid {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(
                    result,
                    Err(KernelInitializationError::InvalidExtension {
                        instruction: InstructionId::new(body.entry_region_ref().id(), 0),
                        message: "synchronous operations cannot allocate or consume references".to_owned(),
                    })
                );
            }
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

    /// Selects one repeated scalar operand, optionally retaining pure mapping work to exercise bounded enumeration.
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

    /// Builds canonical scalar accesses in their declared order for grid ordering and atomic-conflict tests.
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

    /// Creates a block-sized constant matching the call's reference input.
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
    fn test_validate_kernel_initialization_intersects_condition_branches() {
        let call = call(2, 2, 1);
        for complete in [false, true] {
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let expected =
                if complete { Ok(()) } else { Err(KernelInitializationError::IncompleteBody { parameter: 0 }) };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), expected);
        }
    }

    #[test]
    fn test_validate_kernel_initialization_while_condition_and_zero_body_path() {
        let call = call(2, 2, 1);
        for (condition_writes, bounded) in [(false, true), (true, true), (true, false)] {
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let expected = if !bounded {
                Err(KernelInitializationError::UnboundedLoop {
                    instruction: InstructionId::new(body.entry_region_ref().id(), 0),
                })
            } else if condition_writes {
                Ok(())
            } else {
                Err(KernelInitializationError::IncompleteBody { parameter: 0 })
            };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 64), expected);
        }
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
        for masked in [true, false] {
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
            let modified = KernelCallOperation::new(
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
            .unwrap();
            let body = body(&modified, Some(false));
            let expected =
                if masked { Ok(()) } else { Err(KernelInitializationError::OverlappingWindows { parameter: 0 }) };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &modified, 64), expected);
        }
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
    fn test_validate_kernel_initialization_rejects_dynamic_transforms() {
        // Selections are resolved per access site: the same parameter root is statically selected by the first
        // write, so only the dynamically indexed second write is reported.
        let call = call(2, 2, 1);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(ArrayIrType::Dimension(call.coordinate_types()[0].clone()));
        let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i32).unwrap()));
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i64).unwrap()));
        let operation = ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Static(0),
        }]);
        builder.add_instruction(operation, vec![], vec![reference, value], None).unwrap();
        let operation = ReferenceWriteOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Dynamic,
        }]);
        builder.add_instruction(operation, vec![], vec![reference, value, index], None).unwrap();
        let body = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let result = validate_kernel_initialization(body.entry_region_ref(), &call, 64);
        assert_eq!(
            result,
            Err(KernelInitializationError::UnknownSelection {
                instruction: InstructionId::new(body.entry_region_ref().id(), 1),
                input_index: 0,
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "kernel initialization cannot prove the selection of reference input 0 at {}",
                InstructionId::new(body.entry_region_ref().id(), 1),
            ),
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
        for partial in [false, true] {
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            assert_eq!(
                validate_kernel_initialization(body.entry_region_ref(), &call, 64),
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, if partial { 2 } else { 1 }),
                    root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
                }),
            );
        }
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
    fn test_validate_kernel_initialization_masked_transforms() {
        let call = call(2, 2, 1);
        for read_start in [0, 1] {
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            let expected = if read_start == 1 {
                Ok(())
            } else {
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 2),
                    root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
                })
            };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), expected);
        }
    }

    #[test]
    fn test_validate_kernel_initialization_masked_stores_and_dead_swaps() {
        let call = call(2, 2, 1);
        for swap in [false, true] {
            for complete in [false, true] {
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
                let body = builder
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                    .unwrap();
                let expected =
                    if complete { Ok(()) } else { Err(KernelInitializationError::IncompleteBody { parameter: 0 }) };
                assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), expected);
            }
        }
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
        for complete in [false, true] {
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let region = body.entry_region_ref().id();
            let expected = if complete {
                Ok(())
            } else {
                Err(KernelInitializationError::UninitializedRead {
                    instruction: InstructionId::new(region, 3),
                    root: ReferenceRoot::Allocation { instruction: InstructionId::new(region, 0), output_index: 0 },
                })
            };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), expected);
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
        for (swap, selected) in [(false, None), (false, Some(0)), (false, Some(1)), (true, None), (true, Some(0))] {
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
                builder
                    .add_instruction(
                        ReferenceSwapOperation::new().with_transforms(transforms),
                        vec![],
                        vec![reference, value],
                        None,
                    )
                    .unwrap();
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
            let body = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
                .unwrap();
            let expected = if selected == Some(0) {
                Ok(())
            } else {
                Err(KernelInitializationError::Memory(KernelMemoryError::UnmaskedWindowAccess {
                    operation: if swap { "reference_swap" } else { "reference_read" },
                }))
            };
            assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), expected);
        }
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
        for repeated in [false, true] {
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
            if repeated {
                assert_eq!(
                    validate_kernel_initialization(body.entry_region_ref(), &call, 0),
                    Err(KernelInitializationError::QualificationLimit { programs: 2, maximum: 0 }),
                );
                assert_eq!(
                    validate_kernel_initialization(body.entry_region_ref(), &call, 2),
                    Err(KernelInitializationError::IncompleteOutput { parameter: 0 }),
                );
            } else {
                assert_eq!(validate_kernel_initialization(body.entry_region_ref(), &call, 0), Ok(()));
            }
        }
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
            let ordered = repeated_body(&call, &[ReferenceAccessMode::Accumulate]);
            assert_eq!(
                validate_kernel_initialization(ordered.entry_region_ref(), &call, 2),
                Err(KernelInitializationError::Call(KernelError::DisallowedGridAccess {
                    parameter: 0,
                    mode: ReferenceAccessMode::Accumulate,
                })),
            );
        }
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
    fn test_masked_ranges() {
        assert_eq!(masked_ranges(5..10, &[false, true, true, false, true]), vec![6..8, 9..10]);
        assert_eq!(masked_ranges(3..3, &[]), vec![]);
        assert_eq!(masked_ranges(2..4, &[false, false]), vec![]);
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
    fn test_intersect_states() {
        let root = ReferenceRoot::RegionInput { region: crate::programs::RegionId::new(0), input_index: 0 };
        assert_eq!(
            intersect_states(BTreeMap::from([(root, vec![0..4, 6..10])]), &BTreeMap::from([(root, vec![2..8])])),
            BTreeMap::from([(root, vec![2..4, 6..8])]),
        );
    }
}
