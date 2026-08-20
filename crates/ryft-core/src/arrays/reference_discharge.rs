//! Backend-neutral discharge of array references and their views into explicit immutable array state.
//!
//! [`Program::discharge_references`] consumes a flat, capture-lifted array-IR program, validates its complete
//! reference language through [`ReferenceAnalysis`], and rewrites reference state into ordinary array Single Static
//! Assignment (SSA) values. Index and static unit-stride slice views lower through canonical slice, reshape, and
//! update-slice operations. Conditions, loops, scans, and calls receive explicit immutable root-state boundaries
//! derived from the same analysis artifact; derived views must be recreated inside each attached region. The result
//! keeps the original public outputs as a prefix and appends one hidden final-state output for every mutated external
//! root.
//!
//! Operations without a dedicated [`ReferenceDischargeRule`] conservatively reject reference state anywhere in their
//! attached-region closures — including state that is allocated, mutated, and consumed entirely inside the region.
//! Today that covers `shard_map`, rematerialization, linear-call, and custom-derivative carriers; supporting
//! region-local references there requires per-family rules and is deliberately out of scope for the initial array
//! reference feature.

// TODO(eaplatanios): Review this module.
//  Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

use serde::Serialize;

use crate::arrays::operations::{ReferenceDischargeOperation, ReferenceDischargeRule};
use crate::arrays::reference_analysis::{ReferenceAnalysis, ReferenceRoot, ReferenceSource};
use crate::arrays::reference_views::{ArrayReferenceView, ArrayReferenceViewTransform};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::captures::{CaptureConstant, ClosedProgram};
use crate::operations::{
    AddOperation, FreezeReferenceOperation, NewReferenceOperation, ReferenceAddUpdateOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReshapeOperation, SliceOperation, UpdateSliceOperation,
};
use crate::parameters::{Parameterized, Placeholder};
use crate::programs::{
    Atom, AtomId, Effect, Effects, Instruction, InstructionId, Operation, Program, ProgramBuilder, ProgramError,
    ReferenceAccessMode, ReferenceOperationSemantics, RegionId, RegionRef, Type, TypeError, Typed, Value, ValueId,
};

/// Logical binding recipe for one external reference root in a discharged program, in canonical entry-boundary
/// (external-state list) order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DischargedReferenceState {
    /// Capture or public argument that supplies the runtime reference holder.
    source: ReferenceSource,

    /// Flat input position that receives the holder's entering immutable array value.
    discharged_input_index: usize,

    /// Hidden flat output position containing the final state, or [`None`] for a read-only root.
    final_state_output_index: Option<usize>,
}

impl DischargedReferenceState {
    /// Reconstructs a persisted logical external-state binding recipe.
    #[doc(hidden)]
    pub const fn new(
        source: ReferenceSource,
        discharged_input_index: usize,
        final_state_output_index: Option<usize>,
    ) -> Self {
        Self { source, discharged_input_index, final_state_output_index }
    }

    /// Returns the capture or public argument that supplies the runtime reference holder.
    #[inline]
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }

    /// Returns the flat discharged-program input position receiving the entering state.
    #[inline]
    pub const fn discharged_input_index(&self) -> usize {
        self.discharged_input_index
    }

    /// Returns whether the program replaces or accumulates into this external state, in which case its final value is
    /// returned through the hidden output named by
    /// [`final_state_output_index`](Self::final_state_output_index).
    #[inline]
    pub const fn is_mutated(&self) -> bool {
        self.final_state_output_index.is_some()
    }

    /// Returns the hidden final-state output position, or [`None`] when the state is read-only.
    #[inline]
    pub const fn final_state_output_index(&self) -> Option<usize> {
        self.final_state_output_index
    }
}

/// Reference-free program and logical external-state metadata produced by reference discharge.
#[derive(Debug)]
pub struct DischargedReferenceProgram<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> {
    /// Reference-free flat program whose public outputs form a prefix of its complete outputs.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Number of public output leaves before hidden final-state outputs.
    public_output_count: usize,

    /// External reference binding recipes in canonical entry-boundary order.
    external_states: Vec<DischargedReferenceState>,
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> DischargedReferenceProgram<V, O> {
    /// Returns the reference-free flat program.
    #[inline]
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.program
    }

    /// Returns the number of public outputs at the front of the program's output boundary.
    #[inline]
    pub const fn public_output_count(&self) -> usize {
        self.public_output_count
    }

    /// Returns external reference binding recipes in canonical entry-boundary order.
    #[inline]
    pub fn external_states(&self) -> &[DischargedReferenceState] {
        self.external_states.as_slice()
    }

    /// Consumes this discharge artifact and returns its reference-free program, public-output prefix length, and
    /// logical external-state binding recipes.
    #[inline]
    pub(crate) fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, usize, Vec<DischargedReferenceState>) {
        (self.program, self.public_output_count, self.external_states)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    /// Consumes this flat program and discharges array references and their views into immutable array SSA.
    ///
    /// `capture_count` classifies the leading input prefix as capture sources and the remaining inputs as public
    /// arguments. This ordinary-value entry rejects reference-typed constants everywhere. A program produced by
    /// [`ClosedProgram::to_program_with_lifted_captures`] may retain capture-reference constants in attached regions
    /// and must instead use [`Program::discharge_references_with_lifted_captures`]. Reference inputs are replaced
    /// one-for-one by their referent array types, while mutated external roots receive hidden final-state outputs
    /// after the original public-output prefix. Analysis and every operation-family support check complete before
    /// replay begins.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    pub fn discharge_references(self, capture_count: usize) -> Result<DischargedReferenceProgram<V, O>, ProgramError> {
        let analysis = self.analyze_references(capture_count)?;
        validate_discharge_support(&self, &analysis)?;
        discharge_with_analysis(self, analysis)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    /// Discharges this program's local array references on behalf of the generic `transform`, rejecting every
    /// caller-owned external reference root and returning the reference-free program with its boundary unchanged.
    ///
    /// The rejection and the appended-output invariant live here so that every transform adapter shares one gate: an
    /// external root would require runtime holder plumbing the transforms do not have, and a discharge that appended
    /// hidden final-state outputs despite empty external metadata would otherwise silently widen the adapter's
    /// public output boundary instead of erroring.
    pub(crate) fn discharge_local_references(
        self,
        capture_count: usize,
        transform: &'static str,
    ) -> Result<Self, ProgramError> {
        let discharged = self.discharge_references(capture_count)?;
        if let Some(state) = discharged.external_states().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{transform} supports only local references, but the program uses external `{}`",
                    state.source(),
                ),
            });
        }
        let (program, public_output_count, _) = discharged.into_parts();
        if public_output_count != program.output_count() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge appended {} hidden output(s) to a program without external state",
                program.output_count() - public_output_count,
            )));
        }
        Ok(program)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: CaptureConstant<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    /// Consumes a program whose captures were lifted into its leading inputs while attached regions may still contain
    /// capture-reference constants naming that prefix.
    ///
    /// This is the program-level form of [`ClosedProgram::discharge_references`]. Capture constants resolve against
    /// the active lifted entry or nested-call capture scope during analysis; discharge then threads their immutable
    /// array state across every enclosing structured boundary.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    pub fn discharge_references_with_lifted_captures(
        self,
        capture_count: usize,
    ) -> Result<DischargedReferenceProgram<V, O>, ProgramError> {
        let analysis = self.analyze_references_with_lifted_captures(capture_count)?;
        validate_discharge_support(&self, &analysis)?;
        discharge_with_analysis(self, analysis)
    }
}

impl<Capture, V, O, Input, Output> ClosedProgram<Capture, V, O, Input, Output>
where
    Capture: Value<Type = ArrayIrType>,
    V: CaptureConstant<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Lifts this closed program's captures and discharges every reachable array reference.
    ///
    /// The returned logical metadata continues to identify capture slots separately from public inputs. Concrete
    /// capture values remain owned by this [`ClosedProgram`]; discharge never embeds their mutable contents into the
    /// derived program.
    pub fn discharge_references(&self) -> Result<DischargedReferenceProgram<V, O>, ProgramError> {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_with_lifted_captures(capture_count)
    }
}

/// Placement of synthesized state inputs and outputs around one source region boundary.
struct RegionDischargeLayout {
    /// Insertion position for synthesized state inputs among source inputs.
    input_insertion: usize,

    /// Synthesized state input roots in canonical order.
    input_roots: Vec<ReferenceRoot>,

    /// Insertion position for synthesized state outputs among source outputs.
    output_insertion: usize,

    /// Synthesized state output roots in canonical order.
    output_roots: Vec<ReferenceRoot>,
}

/// One discharged region plus the positions corresponding to its source and synthesized outputs.
struct DischargedRegion<V: Value<Type = ArrayIrType>, O: ReferenceDischargeOperation> {
    /// Owned reference-free region program.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Destination output position for each source output, in source order.
    source_output_positions: Vec<usize>,

    /// Destination output position for each synthesized root.
    state_output_positions: HashMap<ReferenceRoot, usize>,
}

/// Object-safe view of one canonical reference operation used as the validation oracle for a primitive discharge
/// rule. Each primitive rule must match its canonical core operation exactly: the canonical descriptor pins the
/// access classification, and the canonical regionless inference re-derives the recorded boundary types (including
/// referent equality and the fixed-shape restriction), so a third-party operation cannot drift from the semantics
/// the rewrite assumes without being rejected.
trait PrimitiveReferenceContract {
    /// Returns the canonical operation-local reference semantics.
    fn contract_semantics(&self) -> Cow<'_, ReferenceOperationSemantics>;

    /// Runs the canonical regionless type inference over the recorded operand types.
    fn contract_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError>;
}

impl<O: Operation<Type = ArrayIrType>> PrimitiveReferenceContract for O {
    fn contract_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        self.reference_semantics()
    }

    fn contract_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        self.infer_output_types(input_types, &[])
    }
}

/// Rejects unsupported stateful operations before destination construction begins.
fn validate_discharge_support<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    analysis: &ReferenceAnalysis,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    if analysis.is_reference_free()
        && program.effects().is_pure()
        && program.regions().iter().all(|region| {
            region.instructions().iter().all(|instruction| {
                instruction.operation().reference_discharge_rule() == ReferenceDischargeRule::Ordinary
            })
        })
    {
        return Ok(());
    }

    // Sealed program arenas place every attached descendant before its parent, so one ascending pass computes the
    // complete transitive reference flag for each region.
    let mut closure_contains_reference = vec![false; program.regions().len()];
    for (region_index, region) in program.regions().iter().enumerate() {
        closure_contains_reference[region_index] = region.atoms().iter().any(|atom| atom.r#type().is_reference())
            || region
                .instructions()
                .iter()
                .flat_map(|instruction| instruction.regions())
                .any(|attached| closure_contains_reference[attached.index()]);
    }

    for region_index in 0..program.regions().len() {
        let region = RegionId::new(region_index);
        for (instruction_index, instruction) in program.regions()[region_index].instructions().iter().enumerate() {
            let instruction_id = InstructionId::new(region, instruction_index);
            let rule = instruction.operation().reference_discharge_rule();
            let semantics = instruction.operation().reference_semantics();
            let boundary_types = |atoms: &[AtomId]| {
                atoms
                    .iter()
                    .map(|atom| program.regions()[region_index].atoms()[atom.index()].r#type().into_owned())
                    .collect::<Vec<_>>()
            };
            // A reference-typed loop or scan carry must preserve its input root positionally so the zero-iteration
            // result is exactly the entering state.
            let positional_reference_carries = |carry_count: usize| {
                instruction.outputs().iter().take(carry_count).enumerate().all(|(output_index, atom)| {
                    !program.regions()[region_index].atoms()[atom.index()].r#type().is_reference()
                        || instruction.operation().reference_output_identity_input(output_index) == Some(output_index)
                })
            };
            let matches_primitive = |canonical: &dyn PrimitiveReferenceContract| {
                instruction.regions().is_empty()
                    && *semantics == *canonical.contract_semantics()
                    && canonical.contract_output_types(&boundary_types(instruction.inputs()))
                        == Ok(boundary_types(instruction.outputs()))
            };
            // Appended-state alignment in the higher-order rewrite relies on each attached computation region
            // mirroring the parent operand list after a constant leading-operand offset, both in arity and in
            // positional input provenance; a permuted or truncated forwarding contract would silently mismap the
            // appended state operands instead of erroring.
            let positional_regions = |leading_operand_count: usize| {
                instruction.regions().iter().copied().enumerate().all(|(attached_index, attached)| {
                    let region_input_count = program.regions()[attached.index()].input_ids().len();
                    region_input_count + leading_operand_count == instruction.inputs().len()
                        && (0..region_input_count).all(|input_index| {
                            instruction.operation().input_region_provenance(attached_index, input_index)
                                == Some(input_index + leading_operand_count)
                        })
                })
            };
            // Replay zips the parent's outputs positionally with its result region's outputs, so every higher-order
            // rule must attach result regions with exactly the parent's output arity and positional output
            // provenance; a reordering or omitting contract would silently mis-wire the rewritten outputs instead of
            // erroring.
            let positional_outputs =
                |result_region_slots: &[usize]| {
                    result_region_slots.iter().copied().all(|slot| {
                        instruction.regions().get(slot).is_some_and(|attached| {
                            program.regions()[attached.index()].output_ids().len() == instruction.outputs().len()
                        })
                    }) && (0..instruction.outputs().len()).all(|output_index| {
                        let provenance = instruction.operation().output_region_provenance(output_index);
                        provenance.len() == result_region_slots.len()
                            && result_region_slots.iter().copied().zip(&provenance).all(|(slot, origin)| {
                                origin.region_index == slot && origin.output_index == output_index
                            })
                    })
                };
            let rule_is_valid = match rule {
                ReferenceDischargeRule::Ordinary => true,
                ReferenceDischargeRule::NewReference => matches_primitive(&NewReferenceOperation),
                ReferenceDischargeRule::View => {
                    instruction.inputs().len() == 1
                        && instruction.outputs().len() == 1
                        && instruction.regions().is_empty()
                        && instruction.operation().effects().is_pure()
                        && semantics.accesses().is_empty()
                        && matches!(
                            semantics.outputs(),
                            [crate::programs::ReferenceOutputSemantics::Alias {
                                output_index: 0,
                                input_index: 0,
                                kind: crate::programs::ReferenceAliasKind::View,
                            }]
                        )
                        && instruction.operation().reference_view_transform().is_some()
                }
                ReferenceDischargeRule::Read => matches_primitive(&ReferenceReadOperation),
                ReferenceDischargeRule::Swap => matches_primitive(&ReferenceSwapOperation),
                ReferenceDischargeRule::AddUpdate => matches_primitive(&ReferenceAddUpdateOperation),
                ReferenceDischargeRule::Freeze => matches_primitive(&FreezeReferenceOperation),
                ReferenceDischargeRule::Condition => {
                    !instruction.inputs().is_empty()
                        && instruction.regions().len() == 2
                        && semantics.is_empty()
                        && positional_regions(1)
                        && positional_outputs(&[0, 1])
                }
                ReferenceDischargeRule::While => {
                    instruction.regions().len() == 2
                        && instruction.inputs().len() == instruction.outputs().len()
                        && semantics.is_empty()
                        && positional_regions(0)
                        && positional_outputs(&[1])
                        && positional_reference_carries(instruction.outputs().len())
                }
                ReferenceDischargeRule::Scan { carry_count } => {
                    instruction.regions().len() == 1
                        && carry_count <= instruction.inputs().len()
                        && carry_count <= instruction.outputs().len()
                        && semantics.is_empty()
                        && instruction.regions().first().is_some_and(|attached| {
                            // A dynamic-length scan carries one trailing runtime-length operand beyond the body's
                            // inputs; a static scan matches the body arity exactly.
                            let region_input_count = program.regions()[attached.index()].input_ids().len();
                            instruction.inputs().len() == region_input_count
                                || instruction.inputs().len() == region_input_count + 1
                        })
                        && (0..carry_count).all(|input_index| {
                            instruction.operation().input_region_provenance(0, input_index) == Some(input_index)
                        })
                        && positional_outputs(&[0])
                        && positional_reference_carries(carry_count)
                }
                ReferenceDischargeRule::Call => {
                    instruction.regions().len() == 1
                        && semantics.is_empty()
                        && positional_regions(0)
                        && positional_outputs(&[0])
                }
            };
            if !rule_is_valid {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` reports an incompatible `{}` reference discharge rule",
                    instruction.operation().name(),
                    rule.name(),
                )));
            }
            let has_reference_boundary = instruction
                .inputs()
                .iter()
                .chain(instruction.outputs())
                .any(|atom| program.regions()[region_index].atoms()[atom.index()].r#type().is_reference())
                || instruction.regions().iter().any(|attached| closure_contains_reference[attached.index()]);
            if matches!(rule, ReferenceDischargeRule::Ordinary)
                && (!semantics.is_empty()
                    || has_reference_boundary
                    || analysis.instruction_summary(instruction_id).is_some())
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` carries reference state but has no reference discharge rule",
                        instruction.operation().name(),
                    ),
                });
            }
            let eliminated = matches!(
                rule,
                ReferenceDischargeRule::NewReference
                    | ReferenceDischargeRule::Read
                    | ReferenceDischargeRule::Swap
                    | ReferenceDischargeRule::AddUpdate
                    | ReferenceDischargeRule::Freeze,
            );
            if eliminated && instruction.operation().effects() != Effects::single(Effect::OrderedState) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` reports a `{}` reference discharge rule with effects that cannot be eliminated",
                        instruction.operation().name(),
                        rule.name(),
                    ),
                });
            }
            if !eliminated && instruction.operation().effects().contains(Effect::OrderedState) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` carries ordered state that reference discharge cannot eliminate",
                        instruction.operation().name(),
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Rewrites a program using the analysis artifact produced immediately before this call.
fn discharge_with_analysis<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    analysis: ReferenceAnalysis,
) -> Result<DischargedReferenceProgram<V, O>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let public_output_count = program.output_count();
    if analysis.is_reference_free() {
        verify_discharged_program(&program)?;
        return Ok(DischargedReferenceProgram { program, public_output_count, external_states: Vec::new() });
    }

    let entry = program.entry();
    let mutated_root_set = analysis
        .region_summary(entry)
        .into_iter()
        .flatten()
        .filter(|access| matches!(access.mode(), ReferenceAccessMode::Write | ReferenceAccessMode::Accumulate))
        .map(|access| access.root())
        .collect::<HashSet<_>>();
    let mutated_roots = analysis
        .external_roots()
        .iter()
        .filter(|external| mutated_root_set.contains(&external.root()))
        .map(|external| external.root())
        .collect::<Vec<_>>();
    let layout = RegionDischargeLayout {
        input_insertion: 0,
        input_roots: Vec::new(),
        output_insertion: public_output_count,
        output_roots: mutated_roots,
    };
    let discharged = discharge_region(program.entry_region_ref(), &analysis, &layout)?;

    let mut external_states = Vec::with_capacity(analysis.external_roots().len());
    for external in analysis.external_roots() {
        let ReferenceRoot::RegionInput { region, input_index } = external.root() else {
            return Err(ProgramError::MalformedProgram(
                "external reference root is not an entry-region input".to_string(),
            ));
        };
        if region != entry {
            return Err(ProgramError::MalformedProgram(
                "external reference root belongs to a non-entry region".to_string(),
            ));
        }
        let mutated = mutated_root_set.contains(&external.root());
        let final_state_output_index = if mutated {
            Some(*discharged.state_output_positions.get(&external.root()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "mutated external reference root `{}` has no discharged final-state output",
                    external.root(),
                ))
            })?)
        } else {
            None
        };
        external_states.push(DischargedReferenceState {
            source: external.source(),
            discharged_input_index: input_index,
            final_state_output_index,
        });
    }

    verify_discharged_program(&discharged.program)?;
    Ok(DischargedReferenceProgram { program: discharged.program, public_output_count, external_states })
}

/// Discharges one source region with the requested synthesized state boundary.
fn stage_reference_view_transform<V, O>(
    builder: &mut ProgramBuilder<V, O>,
    input: AtomId,
    transform: &ArrayReferenceViewTransform,
) -> Result<AtomId, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let input_type = builder.atoms()[input.index()].r#type();
    let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
    let shape = input_type.static_shape().ok_or_else(|| {
        ProgramError::MalformedProgram(format!("discharged reference view input type `{input_type}` is not static"))
    })?;
    match transform {
        ArrayReferenceViewTransform::Index { axis, index } => {
            let mut starts = vec![0; shape.rank()];
            starts[*axis] = *index;
            let mut limits = shape.dimensions().to_vec();
            limits[*axis] = index + 1;
            let sliced = builder.add_instruction(
                O::from_reference_slice(SliceOperation::new(starts, limits)),
                Vec::new(),
                vec![input],
            )?[0];
            let output_shape = Shape::new(
                shape
                    .dimensions()
                    .iter()
                    .enumerate()
                    .filter_map(|(candidate, size)| (candidate != *axis).then_some(Dimension::Static(*size)))
                    .collect(),
            );
            Ok(builder.add_instruction(
                O::from_reference_reshape(ReshapeOperation::new(output_shape)),
                Vec::new(),
                vec![sliced],
            )?[0])
        }
        ArrayReferenceViewTransform::Slice { axes } => {
            let starts = axes.iter().map(|axis| axis.start()).collect::<Vec<_>>();
            let limits = axes.iter().map(|axis| axis.start() + axis.size()).collect::<Vec<_>>();
            Ok(builder.add_instruction(
                O::from_reference_slice(SliceOperation::new(starts, limits)),
                Vec::new(),
                vec![input],
            )?[0])
        }
    }
}

/// Stages each root-to-view intermediate exactly once.
fn stage_reference_view_intermediates<V, O>(
    builder: &mut ProgramBuilder<V, O>,
    root: AtomId,
    view: &ArrayReferenceView,
) -> Result<Vec<AtomId>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let mut intermediates = Vec::with_capacity(view.transforms().len() + 1);
    intermediates.push(root);
    for transform in view.transforms() {
        intermediates.push(stage_reference_view_transform(builder, *intermediates.last().unwrap(), transform)?);
    }
    Ok(intermediates)
}

/// Reconstructs a root from already staged view intermediates and one replacement leaf.
fn stage_reference_view_reconstruction<V, O>(
    builder: &mut ProgramBuilder<V, O>,
    view: &ArrayReferenceView,
    intermediates: &[AtomId],
    replacement: AtomId,
) -> Result<AtomId, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let mut reconstructed = replacement;
    for transform_index in (0..view.transforms().len()).rev() {
        let parent = intermediates[transform_index];
        reconstructed = match &view.transforms()[transform_index] {
            ArrayReferenceViewTransform::Index { axis, index } => {
                let parent_type = builder.atoms()[parent.index()].r#type();
                let parent_type = <&ArrayType>::try_from(parent_type.as_ref())?;
                let parent_rank = parent_type.rank();
                let mut update_shape = parent_type.static_shape().unwrap().dimensions().to_vec();
                update_shape[*axis] = 1;
                let update_shape = Shape::new(update_shape.into_iter().map(Dimension::Static).collect());
                let update = builder.add_instruction(
                    O::from_reference_reshape(ReshapeOperation::new(update_shape)),
                    Vec::new(),
                    vec![reconstructed],
                )?[0];
                let mut starts = vec![0; parent_rank];
                starts[*axis] = *index;
                builder.add_instruction(
                    O::from_reference_update_slice(UpdateSliceOperation::new(starts)),
                    Vec::new(),
                    vec![parent, update],
                )?[0]
            }
            ArrayReferenceViewTransform::Slice { axes } => {
                let starts = axes.iter().map(|axis| axis.start()).collect::<Vec<_>>();
                builder.add_instruction(
                    O::from_reference_update_slice(UpdateSliceOperation::new(starts)),
                    Vec::new(),
                    vec![parent, reconstructed],
                )?[0]
            }
        };
    }
    Ok(reconstructed)
}

fn discharge_region<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ReferenceAnalysis,
    layout: &RegionDischargeLayout,
) -> Result<DischargedRegion<V, O>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    validate_layout(layout, source.input_ids().len(), source.output_ids().len())?;
    let mut builder = ProgramBuilder::<V, O>::new();
    let mut mapped_atoms = vec![None; source.atoms().len()];
    let mut current_states = HashMap::new();

    for source_index in 0..=source.input_ids().len() {
        if source_index == layout.input_insertion {
            for root in &layout.input_roots {
                let input = builder.add_input(ArrayIrType::Array(root_referent_type(source, *root)?));
                if current_states.insert(*root, input).is_some() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference root `{root}` occurs more than once in a discharged region input boundary",
                    )));
                }
            }
        }
        let Some(source_input) = source.input_ids().get(source_index).copied() else {
            continue;
        };
        let input_type = source.atoms()[source_input.index()].r#type();
        let input = match input_type.as_ref() {
            ArrayIrType::Reference(reference) => builder.add_input(ArrayIrType::Array(reference.referent().clone())),
            input_type => builder.add_input(input_type.clone()),
        };
        if input_type.is_reference() {
            let root = analyzed_input_root(analysis, source.id(), source_input)?;
            if current_states.insert(root, input).is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference root `{root}` occurs more than once in a discharged region input boundary",
                )));
            }
        } else {
            mapped_atoms[source_input.index()] = Some(input);
        }
    }

    for (atom_index, atom) in source.atoms().iter().enumerate() {
        if let Atom::Constant(value) = atom {
            if atom.r#type().is_reference() {
                continue;
            }
            mapped_atoms[atom_index] = Some(builder.add_constant(value.clone()));
        }
    }

    for (instruction_index, instruction) in source.instructions().iter().enumerate() {
        let instruction_id = InstructionId::new(source.id(), instruction_index);
        match instruction.operation().reference_discharge_rule() {
            ReferenceDischargeRule::NewReference => {
                let initializer =
                    mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[0])?;
                let root = analysis.root(ValueId::new(source.id(), instruction.outputs()[0])).ok_or_else(|| {
                    ProgramError::MalformedProgram("reference allocation has no analyzed root".to_string())
                })?;
                current_states.insert(root, initializer);
            }
            ReferenceDischargeRule::View => {}
            ReferenceDischargeRule::Read => {
                let root = analyzed_input_root(analysis, source.id(), instruction.inputs()[0])?;
                let state = current_state(&current_states, root)?;
                let view = analysis
                    .view(ValueId::new(source.id(), instruction.inputs()[0]))
                    .ok_or_else(|| ProgramError::MalformedProgram("reference read has no analyzed view".to_string()))?;
                let intermediates = stage_reference_view_intermediates(&mut builder, state, view)?;
                mapped_atoms[instruction.outputs()[0].index()] = Some(*intermediates.last().unwrap());
            }
            ReferenceDischargeRule::Swap => {
                let root = analyzed_input_root(analysis, source.id(), instruction.inputs()[0])?;
                let state = current_state(&current_states, root)?;
                let replacement =
                    mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[1])?;
                let view = analysis
                    .view(ValueId::new(source.id(), instruction.inputs()[0]))
                    .ok_or_else(|| ProgramError::MalformedProgram("reference swap has no analyzed view".to_string()))?;
                let intermediates = stage_reference_view_intermediates(&mut builder, state, view)?;
                let old = *intermediates.last().unwrap();
                let updated =
                    stage_reference_view_reconstruction(&mut builder, view, intermediates.as_slice(), replacement)?;
                mapped_atoms[instruction.outputs()[0].index()] = Some(old);
                current_states.insert(root, updated);
            }
            ReferenceDischargeRule::AddUpdate => {
                let root = analyzed_input_root(analysis, source.id(), instruction.inputs()[0])?;
                let state = current_state(&current_states, root)?;
                let update = mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[1])?;
                let view = analysis.view(ValueId::new(source.id(), instruction.inputs()[0])).ok_or_else(|| {
                    ProgramError::MalformedProgram("reference additive update has no analyzed view".to_string())
                })?;
                let intermediates = stage_reference_view_intermediates(&mut builder, state, view)?;
                let current_view = *intermediates.last().unwrap();
                let updated_view = builder.add_instruction(
                    AddOperation::<ArrayIrType>::new(),
                    Vec::new(),
                    vec![current_view, update],
                )?[0];
                let updated =
                    stage_reference_view_reconstruction(&mut builder, view, intermediates.as_slice(), updated_view)?;
                current_states.insert(root, updated);
            }
            ReferenceDischargeRule::Freeze => {
                let root = analyzed_input_root(analysis, source.id(), instruction.inputs()[0])?;
                let state = current_state(&current_states, root)?;
                current_states.remove(&root);
                mapped_atoms[instruction.outputs()[0].index()] = Some(state);
            }
            rule @ (ReferenceDischargeRule::Condition
            | ReferenceDischargeRule::While
            | ReferenceDischargeRule::Scan { .. }
            | ReferenceDischargeRule::Call) => discharge_higher_order_instruction(
                source,
                analysis,
                instruction_id,
                instruction,
                rule,
                &mut builder,
                &mut mapped_atoms,
                &mut current_states,
            )?,
            ReferenceDischargeRule::Ordinary => {
                let inputs = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input| mapped_value(source, analysis, &mapped_atoms, &current_states, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let regions = instruction
                    .regions()
                    .iter()
                    .copied()
                    .map(|region| Ok(builder.import_region(source.with_id(region)?)))
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let outputs = builder.add_instruction(instruction.operation().clone(), regions, inputs)?.to_vec();
                if instruction.outputs().len() != outputs.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "ordinary operation produced {} outputs during discharge but the source instruction has {}",
                        outputs.len(),
                        instruction.outputs().len(),
                    )));
                }
                for (source_output, output) in instruction.outputs().iter().copied().zip(outputs) {
                    map_source_output(source, analysis, source_output, output, &mut mapped_atoms, &mut current_states)?;
                }
            }
        }
    }

    let mut output_ids = Vec::with_capacity(source.output_ids().len() + layout.output_roots.len());
    let mut source_output_positions = Vec::with_capacity(source.output_ids().len());
    let mut state_output_positions = HashMap::new();
    for source_index in 0..=source.output_ids().len() {
        if source_index == layout.output_insertion {
            for root in &layout.output_roots {
                let position = output_ids.len();
                output_ids.push(current_state(&current_states, *root)?);
                if state_output_positions.insert(*root, position).is_some() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference root `{root}` occurs more than once in a discharged region output boundary",
                    )));
                }
            }
        }
        let Some(source_output) = source.output_ids().get(source_index).copied() else {
            continue;
        };
        source_output_positions.push(output_ids.len());
        output_ids.push(mapped_value(source, analysis, &mapped_atoms, &current_states, source_output)?);
    }

    let input_count = source.input_ids().len() + layout.input_roots.len();
    let output_count = output_ids.len();
    let program =
        builder.build::<Vec<V>, Vec<V>>(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
    Ok(DischargedRegion { program, source_output_positions, state_output_positions })
}

/// Rewrites one region-bearing operation and widens its attached computation boundaries with immutable state.
fn discharge_higher_order_instruction<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ReferenceAnalysis,
    instruction_id: InstructionId,
    instruction: &Instruction<O>,
    rule: ReferenceDischargeRule,
    builder: &mut ProgramBuilder<V, O>,
    mapped_atoms: &mut [Option<AtomId>],
    current_states: &mut HashMap<ReferenceRoot, AtomId>,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let explicit_input_roots = instruction
        .inputs()
        .iter()
        .copied()
        .filter(|input| source.atoms()[input.index()].r#type().is_reference())
        .map(|input| analyzed_input_root(analysis, source.id(), input))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let represented_output_roots = instruction
        .outputs()
        .iter()
        .copied()
        .filter(|output| source.atoms()[output.index()].r#type().is_reference())
        .map(|output| {
            analysis.root(ValueId::new(source.id(), output)).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "reference output {output} of `{}` has no analyzed root during discharge",
                    instruction.operation().name(),
                ))
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let state_roots =
        instruction_state_roots(analysis, instruction_id, instruction, &represented_output_roots, current_states)?;
    let added_input_roots =
        state_roots.iter().copied().filter(|root| !explicit_input_roots.contains(root)).collect::<Vec<_>>();
    let added_output_roots = state_roots
        .iter()
        .copied()
        .filter(|root| !represented_output_roots.contains(root))
        .collect::<Vec<_>>();

    let input_insertion = match rule {
        ReferenceDischargeRule::Scan { carry_count } => carry_count,
        _ => instruction.inputs().len(),
    };
    let operation = match rule {
        ReferenceDischargeRule::Scan { .. } => {
            instruction.operation().with_added_reference_scan_carries(added_input_roots.len())?
        }
        _ => instruction.operation().clone(),
    };
    let inputs = discharged_instruction_inputs(
        source,
        analysis,
        instruction.inputs(),
        input_insertion,
        &added_input_roots,
        mapped_atoms,
        current_states,
    )?;

    let mut discharged_regions = Vec::with_capacity(instruction.regions().len());
    for (region_index, region) in instruction.regions().iter().copied().enumerate() {
        let attached = source.with_id(region)?;
        let region_input_roots = added_input_roots
            .iter()
            .copied()
            .map(|root| attached_root(analysis, instruction_id, region_index, root))
            .collect::<Vec<_>>();
        // A while condition observes entering state without producing state outputs; every other higher-order region
        // returns the synthesized state after its source outputs, except scans, which group synthesized carries with
        // the declared carry prefix on both boundaries.
        let region_layout = if matches!(rule, ReferenceDischargeRule::While) && region_index == 0 {
            RegionDischargeLayout {
                input_insertion: attached.input_ids().len(),
                input_roots: region_input_roots,
                output_insertion: attached.output_ids().len(),
                output_roots: Vec::new(),
            }
        } else {
            let region_output_roots = added_output_roots
                .iter()
                .copied()
                .map(|root| attached_root(analysis, instruction_id, region_index, root))
                .collect::<Vec<_>>();
            let (input_insertion, output_insertion) = match rule {
                ReferenceDischargeRule::Scan { carry_count } => (carry_count, carry_count),
                _ => (attached.input_ids().len(), attached.output_ids().len()),
            };
            RegionDischargeLayout {
                input_insertion,
                input_roots: region_input_roots,
                output_insertion,
                output_roots: region_output_roots,
            }
        };
        discharged_regions.push(discharge_region(attached, analysis, &region_layout)?);
    }

    let result_region_index = usize::from(matches!(rule, ReferenceDischargeRule::While));
    let source_output_positions = discharged_regions[result_region_index].source_output_positions.clone();
    let state_output_positions = discharged_regions[result_region_index].state_output_positions.clone();
    let attached_ids = discharged_regions
        .into_iter()
        .map(|region| builder.import_program(region.program))
        .collect::<Vec<_>>();
    let outputs = builder.add_instruction(operation, attached_ids, inputs)?.to_vec();
    for (source_output, position) in instruction.outputs().iter().copied().zip(source_output_positions) {
        let output = *outputs.get(position).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "discharged `{}` output position {position} is out of range",
                instruction.operation().name(),
            ))
        })?;
        map_source_output(source, analysis, source_output, output, mapped_atoms, current_states)?;
    }
    for root in added_output_roots {
        let attached_root = match rule {
            ReferenceDischargeRule::While => attached_root(analysis, instruction_id, 1, root),
            _ => attached_root(analysis, instruction_id, 0, root),
        };
        let position = state_output_positions.get(&attached_root).copied().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "discharged `{}` region has no final-state output for root `{root}`",
                instruction.operation().name(),
            ))
        })?;
        let output = *outputs.get(position).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "discharged `{}` final-state output position {position} is out of range",
                instruction.operation().name(),
            ))
        })?;
        current_states.insert(root, output);
    }
    Ok(())
}

/// Validates synthesized boundary insertion positions and uniqueness before constructing a destination region.
fn validate_layout(
    layout: &RegionDischargeLayout,
    source_input_count: usize,
    source_output_count: usize,
) -> Result<(), ProgramError> {
    if layout.input_insertion > source_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "discharged region input insertion position {} exceeds source input count {source_input_count}",
            layout.input_insertion,
        )));
    }
    if layout.output_insertion > source_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "discharged region output insertion position {} exceeds source output count {source_output_count}",
            layout.output_insertion,
        )));
    }
    Ok(())
}

/// Returns the referent array type of one analyzed root in the source arena.
fn root_referent_type<V, O>(source: RegionRef<'_, V, O>, root: ReferenceRoot) -> Result<ArrayType, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    let r#type = match root {
        ReferenceRoot::RegionInput { region, input_index } => {
            let region = source.with_id(region)?;
            let atom = region.input_ids().get(input_index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!("reference root `{root}` names an out-of-range region input"))
            })?;
            region.atoms()[atom.index()].r#type()
        }
        ReferenceRoot::Allocation { instruction, output_index } => {
            let region = source.with_id(instruction.region())?;
            let instruction = region.instructions().get(instruction.index()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!("reference root `{root}` names an out-of-range instruction"))
            })?;
            let atom = instruction.outputs().get(output_index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!("reference root `{root}` names an out-of-range output"))
            })?;
            region.atoms()[atom.index()].r#type()
        }
    };
    let ArrayIrType::Reference(reference) = r#type.as_ref() else {
        return Err(ProgramError::MalformedProgram(format!(
            "reference root `{root}` has non-reference type `{}`",
            r#type.as_ref(),
        )));
    };
    Ok(reference.referent().clone())
}

/// Returns the canonical roots whose state crosses one higher-order instruction boundary.
fn instruction_state_roots<O: ReferenceDischargeOperation>(
    analysis: &ReferenceAnalysis,
    instruction_id: InstructionId,
    instruction: &Instruction<O>,
    represented_output_roots: &BTreeSet<ReferenceRoot>,
    current_states: &HashMap<ReferenceRoot, AtomId>,
) -> Result<Vec<ReferenceRoot>, ProgramError> {
    let mut roots = represented_output_roots.clone();
    if let Some(summary) = analysis.instruction_summary(instruction_id) {
        roots.extend(summary.iter().map(|access| access.root()));
    }
    roots
        .into_iter()
        .map(|root| {
            current_states.contains_key(&root).then_some(root).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "higher-order operation `{}` accesses root `{root}` without entering state",
                    instruction.operation().name(),
                ))
            })
        })
        .collect()
}

/// Returns the formal root used by one attached region for a caller root.
fn attached_root(
    analysis: &ReferenceAnalysis,
    instruction: InstructionId,
    region_index: usize,
    source_root: ReferenceRoot,
) -> ReferenceRoot {
    analysis.region_root_for_source(instruction, region_index, source_root).unwrap_or(source_root)
}

/// Maps an instruction's source operands and inserts synthesized state operands at `insertion`.
fn discharged_instruction_inputs<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ReferenceAnalysis,
    source_inputs: &[AtomId],
    insertion: usize,
    added_roots: &[ReferenceRoot],
    mapped_atoms: &[Option<AtomId>],
    current_states: &HashMap<ReferenceRoot, AtomId>,
) -> Result<Vec<AtomId>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    if insertion > source_inputs.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "discharged instruction input insertion position {insertion} exceeds source input count {}",
            source_inputs.len(),
        )));
    }
    let mut inputs = Vec::with_capacity(source_inputs.len() + added_roots.len());
    for source_index in 0..=source_inputs.len() {
        if source_index == insertion {
            for root in added_roots {
                inputs.push(current_state(current_states, *root)?);
            }
        }
        if let Some(source_input) = source_inputs.get(source_index).copied() {
            inputs.push(mapped_value(source, analysis, mapped_atoms, current_states, source_input)?);
        }
    }
    Ok(inputs)
}

/// Returns the discharged immutable value corresponding to one source atom.
fn mapped_value<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ReferenceAnalysis,
    mapped_atoms: &[Option<AtomId>],
    current_states: &HashMap<ReferenceRoot, AtomId>,
    atom: AtomId,
) -> Result<AtomId, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    if source.atoms()[atom.index()].r#type().is_reference() {
        current_state(current_states, analyzed_input_root(analysis, source.id(), atom)?)
    } else {
        mapped_atoms.get(atom.index()).copied().flatten().ok_or_else(|| {
            ProgramError::MalformedProgram(format!("atom {atom} has no ordinary value during reference discharge"))
        })
    }
}

/// Maps one source instruction output to its discharged ordinary value or current reference state.
fn map_source_output<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ReferenceAnalysis,
    source_output: AtomId,
    output: AtomId,
    mapped_atoms: &mut [Option<AtomId>],
    current_states: &mut HashMap<ReferenceRoot, AtomId>,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ReferenceDischargeOperation,
{
    if source.atoms()[source_output.index()].r#type().is_reference() {
        let root = analysis.root(ValueId::new(source.id(), source_output)).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "reference output {source_output} has no analyzed root during discharge",
            ))
        })?;
        current_states.insert(root, output);
    } else {
        mapped_atoms[source_output.index()] = Some(output);
    }
    Ok(())
}

/// Returns the canonical root analyzed for one reference operand.
fn analyzed_input_root(
    analysis: &ReferenceAnalysis,
    region: RegionId,
    input: AtomId,
) -> Result<ReferenceRoot, ProgramError> {
    analysis.root(ValueId::new(region, input)).ok_or_else(|| {
        ProgramError::MalformedProgram(format!("reference operand {input} has no analyzed root during discharge"))
    })
}

/// Returns the current immutable state value for `root`.
fn current_state(states: &HashMap<ReferenceRoot, AtomId>, root: ReferenceRoot) -> Result<AtomId, ProgramError> {
    states
        .get(&root)
        .copied()
        .ok_or_else(|| ProgramError::MalformedProgram(format!("reference root `{root}` has no live discharge state")))
}

/// Verifies the successful-discharge structural postcondition over the complete region arena.
fn verify_discharged_program<V: Value<Type = ArrayIrType>, O: ReferenceDischargeOperation>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
) -> Result<(), ProgramError> {
    for region in program.regions() {
        if region.atoms().iter().any(|atom| atom.r#type().is_reference()) {
            return Err(ProgramError::MalformedProgram(
                "reference discharge produced a reference-typed atom".to_string(),
            ));
        }
        for instruction in region.instructions() {
            let semantics = instruction.operation().reference_semantics();
            if !semantics.is_empty() {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge retained operation `{}`",
                    instruction.operation().name(),
                )));
            }
            if instruction.operation().effects().contains(Effect::OrderedState) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge retained ordered state in operation `{}`",
                    instruction.operation().name(),
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::reference_analysis::ReferenceAnalysisError;
    use crate::arrays::reference_views::ArrayReference;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable};
    use crate::captures::CaptureReference;
    use crate::operations::{
        ConditionOperation, ReferenceIndexOperation, ReferenceSliceOperation, ScanOperation, WhileOperation,
    };
    use crate::programs::{OutputRegionProvenance, ReferenceInputAccess, ReferenceType, RegionInterface, RegionSlot};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type Capture = CaptureReference<ArrayIrType>;
    type CaptureArray = CaptureReference<ArrayType>;
    type CaptureOperation = ArrayIrOperation<CaptureArray>;

    /// Test operation that deliberately reports behavior incompatible with the public discharge contract.
    #[derive(Clone, Debug)]
    enum MalformedDischargeOperation {
        /// Reports a new-root rule for an ordinary array identity operation.
        NewReference,

        /// Reports a while rule without the required fixed-point root contract.
        While,

        /// Reports an eliminated read rule while carrying an additional observable effect.
        MixedEffectRead,

        /// Reports an eliminated read rule whose recorded output type deviates from the canonical read inference.
        MismatchedReadTypes,

        /// Reports a retained condition rule while carrying unresolved ordered state itself.
        StatefulCondition,

        /// Reports a condition rule whose output provenance omits the second branch, breaking the positional
        /// output mapping the rewrite zips against.
        MisroutedConditionOutputs,
    }

    impl Operation for MalformedDischargeOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::NewReference => "malformed_new_reference_discharge",
                Self::While => "malformed_while_discharge",
                Self::MixedEffectRead => "mixed_effect_reference_read",
                Self::MismatchedReadTypes => "mismatched_read_types_discharge",
                Self::StatefulCondition => "stateful_condition_discharge",
                Self::MisroutedConditionOutputs => "misrouted_condition_outputs_discharge",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::NewReference => input_types.to_vec(),
                Self::MixedEffectRead => match &input_types[0] {
                    ArrayIrType::Reference(reference) => vec![reference.referent().clone().into()],
                    input => vec![input.clone()],
                },
                Self::MismatchedReadTypes => vec![ArrayType::new_static(DataType::F32, [2]).into()],
                Self::StatefulCondition | Self::MisroutedConditionOutputs => {
                    region_interfaces[0].output_types().to_vec()
                }
                Self::While => region_interfaces[1].output_types().to_vec(),
            })
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::NewReference | Self::MixedEffectRead | Self::MismatchedReadTypes => &[],
                Self::While | Self::StatefulCondition | Self::MisroutedConditionOutputs => {
                    const { &[RegionSlot::computation("condition"), RegionSlot::computation("body")] }
                }
            }
        }

        fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::While | Self::StatefulCondition => Some(input_index),
                Self::MisroutedConditionOutputs => Some(input_index + 1),
                _ => None,
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::NewReference | Self::MixedEffectRead | Self::MismatchedReadTypes => Vec::new(),
                Self::StatefulCondition => vec![
                    OutputRegionProvenance { region_index: 0, output_index },
                    OutputRegionProvenance { region_index: 1, output_index },
                ],
                Self::While => vec![OutputRegionProvenance { region_index: 1, output_index }],
                Self::MisroutedConditionOutputs => vec![OutputRegionProvenance { region_index: 0, output_index }],
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            if matches!(self, Self::MixedEffectRead | Self::MismatchedReadTypes) {
                Cow::Owned(ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ))
            } else {
                Cow::Borrowed(ReferenceOperationSemantics::empty())
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::MixedEffectRead => {
                    Effects::single(Effect::OrderedState).union(Effects::single(Effect::OrderedIo))
                }
                Self::MismatchedReadTypes | Self::StatefulCondition => Effects::single(Effect::OrderedState),
                Self::NewReference | Self::While | Self::MisroutedConditionOutputs => Effects::PURE,
            }
        }
    }

    impl crate::arrays::ArrayReferenceOperation for MalformedDischargeOperation {}

    impl ReferenceDischargeOperation for MalformedDischargeOperation {
        fn reference_discharge_rule(&self) -> ReferenceDischargeRule {
            match self {
                Self::NewReference => ReferenceDischargeRule::NewReference,
                Self::While => ReferenceDischargeRule::While,
                Self::MixedEffectRead | Self::MismatchedReadTypes => ReferenceDischargeRule::Read,
                Self::StatefulCondition | Self::MisroutedConditionOutputs => ReferenceDischargeRule::Condition,
            }
        }

        fn with_added_reference_scan_carries(&self, _additional_carry_count: usize) -> Result<Self, ProgramError> {
            Ok(self.clone())
        }

        fn from_reference_reshape(_operation: ReshapeOperation) -> Self {
            Self::NewReference
        }

        fn from_reference_slice(_operation: SliceOperation) -> Self {
            Self::NewReference
        }

        fn from_reference_update_slice(_operation: UpdateSliceOperation) -> Self {
            Self::NewReference
        }
    }

    impl From<AddOperation<ArrayIrType>> for MalformedDischargeOperation {
        fn from(_operation: AddOperation<ArrayIrType>) -> Self {
            Self::NewReference
        }
    }

    // Returns the scalar `f32` array type used by the discharge fixtures.
    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    // Wraps one scalar array as an array-IR value.
    fn scalar(value: f32) -> TestValue {
        TestValue::Array(Array::scalar(value))
    }

    // Wraps one Boolean scalar array as an array-IR value.
    fn boolean(value: bool) -> TestValue {
        TestValue::Array(Array::scalar(value))
    }

    // Wraps one vector array as an array-IR value.
    fn vector(values: Vec<f32>) -> TestValue {
        TestValue::Array(Array::vector(values))
    }

    #[test]
    fn test_reference_free_discharge_is_identity() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type().into());
        let output =
            builder.add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![input, input]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let source_rendering = source.to_string();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().to_string(), source_rendering);
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(3.0)]), Ok(vec![scalar(6.0)]));
    }

    #[test]
    fn test_discharge_rejects_new_reference_rule_without_allocation_types() {
        // An allocation rule must consume one array and produce a reference to exactly that referent type; an identity
        // operation over an array reports the rule without satisfying its type contract.
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let input = builder.add_input(scalar_type().into());
        let output =
            builder.add_instruction(MalformedDischargeOperation::NewReference, Vec::new(), vec![input]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `malformed_new_reference_discharge` reports an incompatible `new_reference` reference \
                 discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_while_rule_without_fixed_point_reference_carries() {
        // A while rule must declare equal input and output arity so every reference carry stays at its own position;
        // this operation drops the reference output entirely.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        condition_builder.add_input(reference_type.clone().into());
        let predicate = condition_builder.add_constant(boolean(true));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedDischargeOperation::While, vec![condition, body], vec![reference])
            .unwrap();
        let source =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `malformed_while_discharge` reports an incompatible `while` reference discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_eliminated_rule_carrying_additional_effects() {
        // Discharge deletes every primitive reference operation, so an eliminated rule may declare only the
        // ordered-state effect that discharge itself resolves.
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let output = builder
            .add_instruction(MalformedDischargeOperation::MixedEffectRead, Vec::new(), vec![reference])
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "`mixed_effect_reference_read` reports a `read` reference discharge rule with effects that \
                          cannot be eliminated"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_discharge_rejects_read_rule_with_non_canonical_boundary_types() {
        // The canonical-inference half of the primitive oracle must fire on its own: this descriptor matches the
        // canonical read exactly, so only re-running `reference_read`'s inference over the recorded boundary types can
        // catch the mismatched output shape.
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let output = builder
            .add_instruction(MalformedDischargeOperation::MismatchedReadTypes, Vec::new(), vec![reference])
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `mismatched_read_types_discharge` reports an incompatible `read` reference discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_condition_rule_with_misrouted_output_provenance() {
        // Replay zips the parent's outputs positionally with the first branch's outputs and relies on both branches
        // declaring the same positional provenance, so a condition rule whose provenance omits the second branch must
        // be rejected before any rewrite instead of silently mis-wiring outputs.
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
            let input = builder.add_input(scalar_type().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let first = builder.import_region(branch().entry_region_ref());
        let second = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(scalar_type().into());
        let value = builder.add_input(scalar_type().into());
        let output = builder
            .add_instruction(
                MalformedDischargeOperation::MisroutedConditionOutputs,
                vec![first, second],
                vec![predicate, value],
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `misrouted_condition_outputs_discharge` reports an incompatible `condition` reference \
                 discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_retained_rule_carrying_ordered_state() {
        // A retained higher-order rule keeps its operation in the destination program, so its own ordered state would
        // survive discharge and must be rejected instead.
        let branch = {
            let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
            let value = builder.add_constant(scalar(1.0));
            builder.build::<Vec<TestValue>, Vec<TestValue>>(vec![value], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let true_branch = builder.import_region(branch.entry_region_ref());
        let false_branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_constant(boolean(true));
        let output = builder
            .add_instruction(
                MalformedDischargeOperation::StatefulCondition,
                vec![true_branch, false_branch],
                vec![predicate],
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "`stateful_condition_discharge` carries ordered state that reference discharge cannot \
                          eliminate"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_straight_line_discharge_stages_explicit_immutable_state_threading() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let replacement = builder.add_input(scalar_type().into());
        let update = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let first_snapshot = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let swapped_snapshot =
            builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement]).unwrap()[0];
        assert!(
            builder
                .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
                .unwrap()
                .is_empty(),
        );
        let final_snapshot = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, swapped_snapshot, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // The allocation, read, swap, and freeze primitives all disappear: the initializer becomes the entering state,
        // the read and the swap both forward the previous state value, and only the accumulation stages real work.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = add %1 %2
                in (%0, %0, %3)"},
        );
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().effects(), Effects::PURE);
    }

    #[test]
    fn test_reference_view_discharge_matches_eager_composed_updates() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let pair_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let indexed =
            builder.add_instruction(ReferenceIndexOperation::new(0, 3), Vec::new(), vec![reference]).unwrap()[0];
        let indexed_snapshot = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![indexed]).unwrap()[0];
        let outer = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]),
                Vec::new(),
                vec![reference],
            )
            .unwrap()[0];
        let composed = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]), Vec::new(), vec![outer])
            .unwrap()[0];
        let old = builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![composed, replacement]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![composed, update]).unwrap();
        let final_snapshot = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![indexed_snapshot, old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let inputs = vec![vector(vec![1.0, 2.0, 3.0, 4.0]), vector(vec![10.0, 20.0]), vector(vec![1.0, 2.0])];
        let expected = source.interpret(inputs.clone()).unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected),);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2], %2:f32[2] .
                let %3:f32[1] = slice [start_indices=[3], limit_indices=[4]] %0
                    %4:f32[] = reshape [shape=[]] %3
                    %5:f32[3] = slice [start_indices=[1], limit_indices=[4]] %0
                    %6:f32[2] = slice [start_indices=[0], limit_indices=[2]] %5
                    %7:f32[3] = update_slice [start_indices=[0]] %5 %1
                    %8:f32[4] = update_slice [start_indices=[1]] %0 %7
                    %9:f32[3] = slice [start_indices=[1], limit_indices=[4]] %8
                    %10:f32[2] = slice [start_indices=[0], limit_indices=[2]] %9
                    %11:f32[2] = add %10 %2
                    %12:f32[3] = update_slice [start_indices=[0]] %9 %11
                    %13:f32[4] = update_slice [start_indices=[1]] %8 %12
                in (%4, %6, %13)"},
        );
    }

    #[test]
    fn test_indexed_mutation_discharge_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.clone().into());
        let update = builder.add_input(row_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let row = builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference]).unwrap()[0];
        let old = builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![row, replacement]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![row, update]).unwrap();
        let final_snapshot = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            vector(vec![10.0, 20.0, 30.0]),
            vector(vec![1.0, 2.0, 3.0]),
        ];
        let expected = vec![
            vector(vec![4.0, 5.0, 6.0]),
            TestValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0])),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[3], %2:f32[3] .
                let %3:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %0
                    %4:f32[3] = reshape [shape=[3]] %3
                    %5:f32[1, 3] = reshape [shape=[1, 3]] %1
                    %6:f32[2, 3] = update_slice [start_indices=[1, 0]] %0 %5
                    %7:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %6
                    %8:f32[3] = reshape [shape=[3]] %7
                    %9:f32[3] = add %8 %2
                    %10:f32[1, 3] = reshape [shape=[1, 3]] %9
                    %11:f32[2, 3] = update_slice [start_indices=[1, 0]] %6 %10
                in (%4, %11)"},
        );
    }

    #[test]
    fn test_generated_short_state_programs_match_eager_and_immutable_oracles() {
        /// One operation in a bounded generated state program.
        #[derive(Copy, Clone)]
        enum Step {
            /// Observe the current state.
            Read,

            /// Replace the current state with the shared replacement input.
            Swap,

            /// Add the shared update input to the current state.
            AddUpdate,
        }

        // Every bounded read/swap/accumulate sequence must agree with both an independent scalar oracle and the eager
        // reference interpreter, which pins the state-threading rewrite over all short primitive orderings at once.
        for length in 0usize..=3 {
            for code in 0..3usize.pow(length as u32) {
                let mut remainder = code;
                let steps = (0..length)
                    .map(|_| {
                        let step = match remainder % 3 {
                            0 => Step::Read,
                            1 => Step::Swap,
                            2 => Step::AddUpdate,
                            _ => unreachable!(),
                        };
                        remainder /= 3;
                        step
                    })
                    .collect::<Vec<_>>();
                let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                let initial = builder.add_input(scalar_type().into());
                let replacement = builder.add_input(scalar_type().into());
                let update = builder.add_input(scalar_type().into());
                let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
                let mut outputs = Vec::new();
                let mut oracle_state = 2.0f32;
                let mut oracle_outputs = Vec::new();
                for step in steps {
                    match step {
                        Step::Read => {
                            outputs.push(
                                builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()
                                    [0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                        }
                        Step::Swap => {
                            outputs.push(
                                builder
                                    .add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement])
                                    .unwrap()[0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                            oracle_state = 7.0;
                        }
                        Step::AddUpdate => {
                            builder
                                .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
                                .unwrap();
                            oracle_state += 3.0;
                        }
                    }
                }
                outputs
                    .push(builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0]);
                oracle_outputs.push(scalar(oracle_state));
                let output_count = outputs.len();
                let source = builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(
                        outputs,
                        vec![Placeholder; 3],
                        vec![Placeholder; output_count],
                    )
                    .unwrap();
                let inputs = vec![scalar(2.0), scalar(7.0), scalar(3.0)];
                let eager = source.clone().interpret(inputs.clone()).unwrap();
                let discharged = source.discharge_references(0).unwrap();
                let functional = discharged.program().interpret(inputs).unwrap();
                assert_eq!(eager, oracle_outputs);
                assert_eq!(functional, oracle_outputs);
            }
        }
    }

    #[test]
    fn test_external_discharge_uses_boundary_order_and_appends_only_mutated_state() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference_type = ReferenceType::new(scalar_type());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let replacement = builder.add_input(scalar_type().into());

        // Access the second root first so metadata order cannot accidentally follow access order.
        let second_snapshot = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![second]).unwrap()[0];
        let first_snapshot =
            builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![first, replacement]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![second_snapshot, first_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Discharge is deterministic: two independent runs over the same source agree on the rewritten program and on
        // the complete external-state metadata, including its serialized form.
        let repeated = source.clone().discharge_references(0).unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().to_string(), repeated.program().to_string());
        assert_eq!(discharged.external_states(), repeated.external_states());
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            serde_json::to_string(repeated.external_states()).unwrap(),
        );
        assert_eq!(discharged.public_output_count(), 2);

        // Metadata follows entry-boundary order rather than access order, and only the swapped first root receives a
        // hidden final-state output after the public prefix.
        assert_eq!(
            discharged.external_states(),
            &[
                DischargedReferenceState {
                    source: ReferenceSource::PublicInput { index: 0 },
                    discharged_input_index: 0,
                    final_state_output_index: Some(2),
                },
                DischargedReferenceState {
                    source: ReferenceSource::PublicInput { index: 1 },
                    discharged_input_index: 1,
                    final_state_output_index: None,
                },
            ],
        );
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            concat!(
                r#"[{"source":{"public_input":{"index":0}},"discharged_input_index":0,"#,
                r#""final_state_output_index":2},{"source":{"public_input":{"index":1}},"#,
                r#""discharged_input_index":1,"final_state_output_index":null}]"#,
            ),
        );
        assert_eq!(
            format!("{:?}", discharged.external_states()),
            concat!(
                "[DischargedReferenceState { source: PublicInput { index: 0 }, ",
                "discharged_input_index: 0, final_state_output_index: Some(2) }, ",
                "DischargedReferenceState { source: PublicInput { index: 1 }, ",
                "discharged_input_index: 1, final_state_output_index: None }]",
            ),
        );
        assert_eq!(
            discharged.program().interpret(vec![scalar(10.0), scalar(20.0), scalar(7.0)]),
            Ok(vec![scalar(20.0), scalar(10.0), scalar(7.0)]),
        );
    }

    #[test]
    fn test_discharge_local_references() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let external = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Every transform adapter shares this gate, so its rejection names the requesting transform verbatim
        // together with the caller-owned boundary source the program depends on. Public arguments and captures are
        // both external: neither boundary can supply the runtime holder that writing final state back would need.
        assert!(matches!(
            external.clone().discharge_local_references(0, "differentiation"),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "differentiation supports only local references, but the program uses external \
                    `public input 0`",
        ));
        assert!(matches!(
            external.discharge_local_references(1, "batching"),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "batching supports only local references, but the program uses external `capture 0`",
        ));

        // A program that allocates every root itself passes the gate with its boundary unchanged, because hidden
        // final-state outputs are appended only for external roots. The result is an ordinary pure array program.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, initial]).unwrap();
        let output = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let local = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let discharged = local.discharge_local_references(0, "rematerialization").unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert!(discharged.effects().is_pure());
        assert_eq!(discharged.interpret(vec![scalar(3.0)]), Ok(vec![scalar(6.0)]));
    }

    #[test]
    fn test_discharge_rejects_invalid_program_before_producing_an_artifact() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let invalid_read = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![frozen, invalid_read],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let entry = source.entry();
        let error = source.discharge_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(entry, 2),
                operation: "reference_read".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(entry, 0), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_condition_discharge_threads_identical_state_through_unequal_branch_accesses() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_input(scalar_type().into());
        let snapshot = true_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement])
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        false_builder.add_input(scalar_type().into());
        let snapshot = false_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(scalar_type().into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // Both branches receive the entering state and return their own final state after the source output, so the
        // writing branch returns its replacement while the reading branch returns the state unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));

        // The true branch swaps, so the public snapshot is the entering state and the final state is the replacement.
        assert_eq!(
            discharged.program().interpret(vec![boolean(true), scalar(10.0), scalar(7.0)]),
            Ok(vec![scalar(10.0), scalar(7.0)]),
        );

        // The false branch only reads, so the entering state is both the snapshot and the final state.
        assert_eq!(
            discharged.program().interpret(vec![boolean(false), scalar(10.0), scalar(7.0)]),
            Ok(vec![scalar(10.0), scalar(10.0)]),
        );
    }

    #[test]
    fn test_condition_discharge_orders_multiple_roots_by_parent_boundary() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        true_builder.add_input(scalar_type().into());
        let second_replacement = true_builder.add_input(scalar_type().into());
        let first_snapshot = true_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![first]).unwrap()[0];
        let second_snapshot = true_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![second, second_replacement])
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        let first_replacement = false_builder.add_input(scalar_type().into());
        false_builder.add_input(scalar_type().into());
        let first_snapshot = false_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![first, first_replacement])
            .unwrap()[0];
        let second_snapshot =
            false_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![second]).unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let first_replacement = builder.add_input(scalar_type().into());
        let second_replacement = builder.add_input(scalar_type().into());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second, first_replacement, second_replacement],
            )
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();

        // Both branches write a different root, so both roots cross the boundary; the appended final-state outputs
        // follow parent entry-boundary order rather than the order in which either branch happens to access them.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states().len(), 2);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::PublicInput { index: 1 });
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        assert_eq!(discharged.external_states()[1].source(), ReferenceSource::PublicInput { index: 2 });
        assert_eq!(discharged.external_states()[1].final_state_output_index(), Some(3));

        // The true branch swaps only the second root, leaving the first root's final state at its entering value.
        let inputs = vec![boolean(true), scalar(10.0), scalar(20.0), scalar(11.0), scalar(22.0)];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![scalar(10.0), scalar(20.0), scalar(10.0), scalar(22.0)]),
        );

        // The false branch swaps only the first root, which mirrors the same contract on the other position.
        let inputs = vec![boolean(false), scalar(10.0), scalar(20.0), scalar(11.0), scalar(22.0)];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![scalar(10.0), scalar(20.0), scalar(11.0), scalar(20.0)]),
        );
    }

    #[test]
    fn test_nested_condition_while_discharge_keeps_local_state_inside_its_creation_scope() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let loop_condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let loop_body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let loop_condition = true_builder.import_region(loop_condition.entry_region_ref());
        let loop_body = true_builder.import_region(loop_body.entry_region_ref());
        let initial = true_builder.add_input(scalar_type().into());
        let reference = true_builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference =
            true_builder.add_instruction(operation, vec![loop_condition, loop_body], vec![reference]).unwrap()[0];
        let value = true_builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let value = false_builder.add_input(scalar_type().into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The root is allocated and frozen inside the true branch, so no state crosses the entry boundary even though a
        // nested loop mutates it; the false branch never sees the root at all.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![boolean(true), scalar(3.0)]), Ok(vec![scalar(5.0)]));
        assert_eq!(discharged.program().interpret(vec![boolean(false), scalar(3.0)]), Ok(vec![scalar(3.0)]));
    }

    #[test]
    fn test_while_discharge_preserves_zero_iteration_and_threads_mutated_state() {
        let build = |condition_value: bool, iteration_bound: Option<usize>| {
            let reference_type = ReferenceType::new(scalar_type());
            let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = condition_builder.add_input(reference_type.clone().into());
            condition_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
            let condition = condition_builder.add_constant(boolean(condition_value));
            let condition = condition_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = body_builder.add_input(reference_type.clone().into());
            let update = body_builder.add_constant(scalar(1.0));
            body_builder
                .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
                .unwrap();
            let body = body_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let condition = builder.import_region(condition.entry_region_ref());
            let body = builder.import_region(body.entry_region_ref());
            let reference = builder.add_input(reference_type.into());
            let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
            let reference = builder.add_instruction(operation, vec![condition, body], vec![reference]).unwrap()[0];
            let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        // A predicate that is immediately false leaves the entering state untouched in both the public read and the
        // appended final state.
        let zero_iteration = build(false, None).discharge_references(0).unwrap();
        assert_eq!(zero_iteration.program().interpret(vec![scalar(2.0)]), Ok(vec![scalar(2.0), scalar(2.0)]));

        // The reference carry becomes an ordinary array carry: the condition region observes the state without
        // returning it, while the body returns the accumulated state in the carry position.
        let three_iterations = build(true, Some(3)).discharge_references(0).unwrap();
        assert_eq!(
            three_iterations.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = while [iteration_bound=3] %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const true
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%1, %1)"},
        );
        assert_eq!(three_iterations.program().interpret(vec![scalar(2.0)]), Ok(vec![scalar(5.0), scalar(5.0)]));
    }

    #[test]
    fn test_scan_discharge_keeps_state_carries_separate_from_stacked_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let value = body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let operation = ScanOperation::<TestValue>::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs = builder.add_instruction(operation, vec![body], vec![reference]).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value =
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![final_reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![final_value, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The synthesized state joins the declared carry prefix on both boundaries instead of being appended after the
        // stacked outputs, and every unrelated scan attribute survives the rewrite unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[3] = scan [carry_count=1, length=3, reverse=true, unroll=3] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%1, %2, %1)"},
        );
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let TestOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(3));
        assert!(scan.reverse());
        assert_eq!(scan.unroll(), 3);
        assert_eq!(
            discharged.program().interpret(vec![scalar(2.0)]),
            Ok(vec![scalar(5.0), vector(vec![5.0, 4.0, 3.0]), scalar(5.0)]),
        );
    }

    #[test]
    fn test_scan_discharge_preserves_zero_length_state_identity() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let value = body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let outputs =
            builder.add_instruction(ScanOperation::<TestValue>::new(1, 0), vec![body], vec![reference]).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value =
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![final_reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![final_value, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        assert_eq!(
            discharged.program().interpret(vec![scalar(2.0)]),
            Ok(vec![scalar(2.0), vector(Vec::new()), scalar(2.0)]),
        );
    }

    #[test]
    fn test_closed_program_discharge_resolves_reference_captures_inside_condition_regions() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = branch_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = branch_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![branch, branch],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed = ClosedProgram::new(program, vec![ArrayIrValue::Reference(reference)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[DischargedReferenceState {
                source: ReferenceSource::Capture { index: 0 },
                discharged_input_index: 0,
                final_state_output_index: None,
            }],
        );
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            concat!(
                r#"[{"source":{"capture":{"index":0}},"discharged_input_index":0,"#,
                r#""final_state_output_index":null}]"#,
            ),
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![scalar_type().into(), ArrayType::scalar(DataType::Boolean).into()],
        );
    }

    #[test]
    fn test_closed_program_discharge_resolves_transitively_nested_reference_captures() {
        let reference_type = ReferenceType::new(scalar_type());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut leaf_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = leaf_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = leaf_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let leaf = leaf_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut middle_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let leaf = middle_builder.import_region(leaf.entry_region_ref());
        let predicate = middle_builder.add_constant(Capture::new(1, predicate_type.clone().into()));
        let value = middle_builder
            .add_instruction(ConditionOperation::<ArrayIrValue<CaptureArray>>::new(), vec![leaf, leaf], vec![predicate])
            .unwrap()[0];
        let middle = middle_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let middle = builder.import_region(middle.entry_region_ref());
        let predicate = builder.add_constant(Capture::new(1, predicate_type.into()));
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![middle, middle],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder]).unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed = ClosedProgram::new(program, vec![ArrayIrValue::Reference(reference), boolean(true)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_states()[0].is_mutated());
    }

    #[test]
    fn test_closed_program_discharge_threads_reference_captures_through_while() {
        // A capture read only by the loop condition still needs a synthesized state input on both while regions, but no
        // final-state output because nothing writes it.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = condition_builder.add_constant(Capture::new(0, reference_type.into()));
        condition_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
        let predicate = condition_builder.add_constant(Capture::new(1, ArrayType::scalar(DataType::Boolean).into()));
        let condition = condition_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![predicate], Vec::new(), vec![Placeholder])
            .unwrap();
        let body = ProgramBuilder::<Capture, CaptureOperation>::new()
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], Vec::new())
            .unwrap();
        let while_program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed =
            ClosedProgram::new(while_program, vec![ArrayIrValue::Reference(concrete_reference), boolean(false)])
                .unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert!(matches!(
            discharged.program().entry_region_ref().instructions()[0].operation(),
            CaptureOperation::While(_),
        ));
    }

    #[test]
    fn test_closed_program_discharge_threads_reference_captures_through_scan() {
        // A capture read by a scan body becomes a synthesized carry in front of the declared carry prefix, which raises
        // the rewritten scan's carry count without disturbing its length, direction, or unroll factor.
        let reference_type = ReferenceType::new(scalar_type());
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32));
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrValue<CaptureArray>>::new(0, 2), vec![body], Vec::new())
            .unwrap()[0];
        let scan_program =
            builder.build::<Vec<Capture>, Vec<Capture>>(vec![values], Vec::new(), vec![Placeholder]).unwrap();
        let closed = ClosedProgram::new(scan_program, vec![ArrayIrValue::Reference(concrete_reference)]).unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.program().output_types(), vec![vector(vec![0.0, 0.0]).r#type().into_owned()]);
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let CaptureOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(2));
        assert!(!scan.reverse());
        assert_eq!(scan.unroll(), 1);
    }

    #[test]
    fn test_closed_program_discharge_threads_mutated_reference_capture_through_scan() {
        // A capture that a scan body accumulates into reaches that body only through a synthesized carry, which is the
        // most involved discharge path: the state must enter the scan ahead of the declared carry prefix, be updated
        // inside the body, leave through the matching synthesized carry output, and reach the hidden entry final-state
        // output after the public prefix. The capture value family carries no data, so the rendered program rather than
        // an interpretation pins the resulting state flow.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let update = body_builder.add_constant(Capture::new(1, scalar_type().into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let value = body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrValue<CaptureArray>>::new(0, 3), vec![body], Vec::new())
            .unwrap()[0];
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(vec![values], Vec::new(), vec![Placeholder]).unwrap();
        let closed = ClosedProgram::new(
            program,
            vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32))), scalar(1.0)],
        )
        .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(discharged.external_states()[0].discharged_input_index(), 0);
        assert!(discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const capture#1:f32[]
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%3, %2)"},
        );
    }

    #[test]
    fn test_condition_discharge_matches_eager_reference_execution() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(scalar(1.0));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let snapshot = true_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let replacement = false_builder.add_constant(scalar(9.0));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement])
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Each branch mutates the shared root differently, so the eager reference interpreter and the discharged
        // program must agree on the branch snapshot as well as on the state observed after the condition.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        for (predicate, expected) in [(true, vec![scalar(5.0), scalar(5.0)]), (false, vec![scalar(4.0), scalar(9.0)])] {
            let inputs = vec![boolean(predicate), scalar(4.0)];
            let eager = source.clone().interpret(inputs.clone()).unwrap();
            assert_eq!(eager, expected);
            assert_eq!(discharged.program().interpret(inputs), Ok(eager));
        }
    }

    #[test]
    fn test_condition_discharge_recreates_view_inside_region() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let view =
                builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference]).unwrap()[0];
            let update = builder.add_constant(scalar(1.0));
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![view, update]).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let output = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let true_inputs = vec![boolean(true), vector(vec![1.0, 2.0, 3.0])];
        let false_inputs = vec![boolean(false), vector(vec![1.0, 2.0, 3.0])];
        assert_eq!(source.clone().interpret(true_inputs.clone()), Ok(vec![vector(vec![1.0, 3.0, 3.0])]));
        assert_eq!(source.clone().interpret(false_inputs.clone()), Ok(vec![vector(vec![1.0, 2.0, 3.0])]));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(true_inputs), Ok(vec![vector(vec![1.0, 3.0, 3.0])]));
        assert_eq!(discharged.program().interpret(false_inputs), Ok(vec![vector(vec![1.0, 2.0, 3.0])]));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3] .
                let %2:f32[3] = condition %0 %1 [
                    true={
                        lambda %0:f32[3] .
                        let %1:f32[] = const 1.0
                            %2:f32[1] = slice [start_indices=[1], limit_indices=[2]] %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = add %3 %1
                            %5:f32[1] = reshape [shape=[1]] %4
                            %6:f32[3] = update_slice [start_indices=[1]] %0 %5
                        in (%6)
                    },
                    false={
                        lambda %0:f32[3] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_while_discharge_matches_hand_written_immutable_state_passing_loop() {
        // Eager interpretation cannot execute a reference-carrying `while` at all, because masked predicate selection
        // has no meaning for reference carries, so the oracle here is a hand-written immutable loop that threads the
        // same state through an ordinary array carry instead of an eager run of the reference program.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(2.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        oracle_condition_builder.add_input(scalar_type().into());
        let predicate = oracle_condition_builder.add_constant(boolean(true));
        let oracle_condition = oracle_condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = oracle_body_builder.add_input(scalar_type().into());
        let update = oracle_body_builder.add_constant(scalar(2.0));
        let updated = oracle_body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![state, update])
            .unwrap()[0];
        let oracle_body = oracle_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![updated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let oracle_condition = oracle_builder.import_region(oracle_condition.entry_region_ref());
        let oracle_body = oracle_builder.import_region(oracle_body.entry_region_ref());
        let state = oracle_builder.add_input(scalar_type().into());
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let final_state =
            oracle_builder.add_instruction(operation, vec![oracle_condition, oracle_body], vec![state]).unwrap()[0];
        let oracle = oracle_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![final_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The condition region observes the carried state while the body accumulates into it, so the discharged loop
        // must reproduce the immutable loop for every initial state.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        for initial in [0.0f32, 1.0, -4.5] {
            let expected = oracle.clone().interpret(vec![scalar(initial)]).unwrap();
            assert_eq!(discharged.program().interpret(vec![scalar(initial)]), Ok(expected));
        }
        assert_eq!(discharged.program().interpret(vec![scalar(1.0)]), Ok(vec![scalar(7.0)]));
    }

    #[test]
    fn test_scan_discharge_matches_eager_reference_execution() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(3.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let value = body_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let outputs =
            builder.add_instruction(ScanOperation::<TestValue>::new(1, 4), vec![body], vec![reference]).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![final_reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![frozen, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The declared reference carry and the accumulating body must agree with eager execution on both the stacked
        // per-iteration snapshots and the state observed after the scan.
        let eager = source.clone().interpret(vec![scalar(0.0)]).unwrap();
        assert_eq!(eager, vec![scalar(12.0), vector(vec![3.0, 6.0, 9.0, 12.0])]);
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(0.0)]), Ok(eager));
    }

    #[test]
    fn test_dynamic_length_scan_discharge_accepts_the_trailing_runtime_length_operand() {
        // A dynamic-length scan carries one runtime-length operand after the body's inputs, so the discharge
        // preflight must accept the one-past-body parent arity instead of rejecting the canonical dynamic form.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(9)).unwrap());
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(3.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let runtime_length = builder.add_input(DimensionType::new(length.clone()).into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let scanned = builder
            .add_instruction(
                ScanOperation::<TestValue>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![reference, runtime_length],
            )
            .unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![scanned]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let runtime_length = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(length), 4).unwrap());
        let eager = source.clone().interpret(vec![scalar(0.0), runtime_length.clone()]).unwrap();
        assert_eq!(eager, vec![scalar(12.0)]);
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(0.0), runtime_length]), Ok(eager));
    }
}
