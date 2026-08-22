//! Backend-neutral discharge of array references and their views into explicit immutable array state.
//!
//! [`Program::discharge_references`] consumes a flat, capture-lifted array-IR program, validates its complete
//! reference language through [`ArrayReferenceAnalysis`], and rewrites reference state into ordinary array SSA
//! values. Index and static unit-stride slice views lower through canonical slice, reshape, and
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
//!
//! # Transform Boundary
//!
//! Discharge is the only supported route from local mutable state into generic transforms. Partial evaluation,
//! batching, forward- and reverse-mode differentiation, and rematerialization first prove that every root is local,
//! discharge the complete program, and then transform the reference-free result. The resulting behavior is the same
//! as transforming an explicitly immutable state-passing program. External public or captured roots are rejected by
//! these adapters: automatic differentiation of caller-owned state, mapped/shared reference batching, and
//! externally stateful rematerialization have no implicit semantics. Custom-derivative rule regions reject reference
//! state independently rather than inheriting a derivative for mutation.
//!
//! Supported local control flow follows the same rule. Conditions receive the current root state in both branches;
//! while bodies and scan bodies return updated hidden carries; nested calls receive and return the state required by
//! their canonical root summaries. A while condition may read entering state but cannot mutate or consume it because
//! its Boolean-only boundary has nowhere to publish an update. Derived views do not cross any of these boundaries and
//! must be recreated from the root inside the attached region.
//!
//! ```text
//! local reference program
//!     -> validate roots, views, lifetime, and access order
//!     -> discharge to immutable array SSA
//!     -> partial evaluation / batching / AD / rematerialization
//!
//! external or captured reference program
//!     -> stateful compilation and execution, or a targeted transform rejection
//! ```
//!
//! Representative supported compositions are shown below. The value-level transform capabilities themselves reject
//! reference operations outright ("must be discharged before differentiation/batching"). First call
//! [`ReferenceDischarge::discharge_local_references`], then use the ordinary transform: [`Program::jvp`] or
//! [`Program::linearize`] for forward mode, [`Pullback`](crate::Pullback) obtained from the linearization for reverse
//! mode,
//! [`Program::batched_with_local_references`](crate::Program::batched_with_local_references) for batching,
//! [`Program::partially_evaluate`] for partial evaluation, and
//! [`Program::rematerialize_with_local_references`](crate::Program::rematerialize_with_local_references) for
//! rematerialization.
//!
//! ```text
//! condition(predicate,
//!     true  = || { state.add_update(true_update) },
//!     false = || { state.swap(false_replacement) })
//! while read(state) < limit { state.add_update(step) }
//! scan(inputs) { |input| state.add_update(input); read(state) }
//!     -> explicit immutable state carries at every attached-region boundary
//!
//! let program = program.discharge_local_references(capture_count, "differentiation")?;
//! program.jvp()?                                  // state = new_reference(x); state.add_update(x); freeze(state)
//! program.linearize()?.pullback()
//!     -> discharge local state -> differentiate the reference-free program
//!
//! program.batched_with_local_references(...)
//!     -> discharge local state -> batch independent immutable state-passing programs
//! ```
//!
//! A root that is allocated, mutated, and consumed inside one program is discharged into ordinary array SSA, so the
//! rewritten callable is pure: it reports no external state and keeps exactly its original public outputs.
//!
//! ```
//! use ryft_core::{
//!     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, FreezeReferenceOperation, NewReferenceOperation,
//!     Placeholder, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceDischarge,
//! };
//!
//! let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
//! let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial])?[0];
//! builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])?;
//! let total = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference])?[0];
//! let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
//!     vec![total],
//!     vec![Placeholder; 2],
//!     vec![Placeholder],
//! )?;
//!
//! let discharged = program.discharge_references(0)?;
//! assert_eq!(discharged.public_output_count(), 1);
//! assert_eq!(discharged.external_states(), &[]);
//! assert_eq!(
//!     discharged.program().interpret(vec![
//!         ArrayIrValue::Array(Array::scalar(1.0_f32)),
//!         ArrayIrValue::Array(Array::scalar(2.0_f32)),
//!     ])?,
//!     vec![ArrayIrValue::Array(Array::scalar(3.0_f32))],
//! );
//! # Ok::<(), ryft_core::ProgramError>(())
//! ```
//!
//! This module is arrays-owned deliberately. The generic program layer defines root/alias/access vocabulary, while
//! discharge needs array-specific view composition and canonical slice, reshape, and update-slice reconstruction.

// TODO(eaplatanios): Review this module.
//  Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::borrow::Cow;
use std::collections::HashMap;

use crate::arrays::operations::ArrayReferenceDischargeOperation;
use crate::arrays::reference_analysis::ArrayReferenceAnalysis;
use crate::arrays::reference_views::{ArrayReferenceView, ViewReadCarrier, ViewWriteCarrier};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::Shape;
use crate::arrays::types::ir::ArrayIrType;
use crate::captures::{CaptureConstant, ClosedProgram};
use crate::operations::{AddOperation, ReshapeOperation, SliceOperation, UpdateSliceOperation};
use crate::parameters::{Parameterized, Placeholder};
use crate::programs::{
    Atom, AtomId, Effect, Effects, FreezeReferenceOperation, Instruction, InstructionId, NewReferenceOperation,
    Operation, Program, ProgramBuilder, ProgramError, ReferenceAddUpdateOperation, ReferenceAliasKind,
    ReferenceDischarge, ReferenceDischargePlan, ReferenceDischargeResult, ReferenceDischargeRule,
    ReferenceInstructionDischargePlan, ReferenceOperationSemantics, ReferenceOutputSemantics, ReferenceReadOperation,
    ReferenceRegionDischargePlan, ReferenceRoot, ReferenceSwapOperation, RegionId, RegionRef, Type, TypeError, Typed,
    Value, ValueId,
};

impl<V, O> ReferenceDischarge for Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    type DischargedProgram = Self;

    fn discharge_references(self, capture_count: usize) -> Result<ReferenceDischargeResult<Self>, ProgramError> {
        // The ordinary-value entry rejects reference-typed constants. Capture-lifted programs use the specialized
        // inherent entry point below, while both routes share the same analysis-before-replay ordering.
        let analysis = self.analyze_array_references(capture_count)?;
        validate_discharge_support(&self, &analysis)?;
        discharge_with_analysis(self, analysis)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: CaptureConstant<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
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
    ) -> Result<ReferenceDischargeResult<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
        let analysis = self.analyze_array_references_with_lifted_captures(capture_count)?;
        validate_discharge_support(&self, &analysis)?;
        discharge_with_analysis(self, analysis)
    }
}

impl<Capture, V, O, Input, Output> ClosedProgram<Capture, V, O, Input, Output>
where
    Capture: Value<Type = ArrayIrType>,
    V: CaptureConstant<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Lifts this closed program's captures and discharges every reachable array reference.
    ///
    /// The returned logical metadata continues to identify capture slots separately from public inputs. Concrete
    /// capture values remain owned by this [`ClosedProgram`]; discharge never embeds their mutable contents into the
    /// derived program.
    pub fn discharge_references(
        &self,
    ) -> Result<ReferenceDischargeResult<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_with_lifted_captures(capture_count)
    }
}

/// Object-safe view of one canonical reference operation used as the validation oracle for a primitive discharge
/// rule. Each primitive rule must match its canonical core operation exactly: the canonical descriptor pins the
/// access classification, and the canonical regionless inference re-derives the recorded boundary types (including
/// referent equality), so a third-party operation cannot drift from the semantics the rewrite assumes without being
/// rejected.
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
    analysis: &ArrayReferenceAnalysis,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
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
                ReferenceDischargeRule::NewRoot => {
                    matches_primitive(&NewReferenceOperation::<ArrayType, ArrayIrType>::new())
                }
                ReferenceDischargeRule::Alias => {
                    instruction.inputs().len() == 1
                        && instruction.outputs().len() == 1
                        && instruction.regions().is_empty()
                        && instruction.operation().effects().is_pure()
                        && semantics.accesses().is_empty()
                        && matches!(
                            semantics.outputs(),
                            [ReferenceOutputSemantics::Alias {
                                output_index: 0,
                                input_index: 0,
                                kind: ReferenceAliasKind::View,
                            }]
                        )
                        && instruction.operation().reference_view_transform().is_some()
                }
                ReferenceDischargeRule::Read => {
                    matches_primitive(&ReferenceReadOperation::<ArrayType, ArrayIrType>::new())
                }
                ReferenceDischargeRule::Replace => {
                    matches_primitive(&ReferenceSwapOperation::<ArrayType, ArrayIrType>::new())
                }
                ReferenceDischargeRule::Accumulate => {
                    matches_primitive(&ReferenceAddUpdateOperation::<ArrayType, ArrayIrType>::new())
                }
                ReferenceDischargeRule::Consume => {
                    matches_primitive(&FreezeReferenceOperation::<ArrayType, ArrayIrType>::new())
                }
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
                ReferenceDischargeRule::NewRoot
                    | ReferenceDischargeRule::Read
                    | ReferenceDischargeRule::Replace
                    | ReferenceDischargeRule::Accumulate
                    | ReferenceDischargeRule::Consume,
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
    analysis: ArrayReferenceAnalysis,
) -> Result<ReferenceDischargeResult<Program<V, O, Vec<V>, Vec<V>>>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    let public_output_count = program.output_count();
    if analysis.is_reference_free() {
        verify_discharged_program(&program)?;
        let total_input_count = program.input_count();
        return ReferenceDischargeResult::new(
            program,
            total_input_count,
            public_output_count,
            public_output_count,
            Vec::new(),
        );
    }

    let plan = ReferenceDischargePlan::new(&program, analysis.analysis(), O::reference_discharge_rule)?;
    let discharged = discharge_region(program.entry_region_ref(), &analysis, plan.entry())?;
    verify_discharged_program(&discharged)?;
    let total_input_count = discharged.input_count();
    let total_output_count = discharged.output_count();
    ReferenceDischargeResult::new(
        discharged,
        total_input_count,
        total_output_count,
        plan.public_output_count(),
        plan.external_states().to_vec(),
    )
}

/// Staged view carrier that emits the canonical slice, reshape, and update-slice instructions into a program
/// builder, sharing the single [`ArrayReferenceView`] traversal with the eager value carrier.
struct StagedViewCarrier<'b, V: Value<Type = ArrayIrType>, O: ArrayReferenceDischargeOperation>(
    /// Destination builder receiving the staged view instructions.
    &'b mut ProgramBuilder<V, O>,
);

impl<V, O> ViewReadCarrier for StagedViewCarrier<'_, V, O>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    type Value = AtomId;

    fn array_type<'c>(&'c self, value: &'c AtomId) -> Result<Cow<'c, ArrayType>, ProgramError> {
        match self.0.atoms()[value.index()].r#type() {
            Cow::Borrowed(r#type) => Ok(Cow::Borrowed(<&ArrayType>::try_from(r#type)?)),
            Cow::Owned(r#type) => Ok(Cow::Owned(<&ArrayType>::try_from(&r#type)?.clone())),
        }
    }

    fn slice(&mut self, input: &AtomId, starts: Vec<usize>, limits: Vec<usize>) -> Result<AtomId, ProgramError> {
        Ok(self.0.add_instruction(
            O::from_reference_slice(SliceOperation::new(starts, limits)),
            Vec::new(),
            vec![*input],
        )?[0])
    }

    fn reshape(&mut self, input: &AtomId, shape: Shape) -> Result<AtomId, ProgramError> {
        Ok(self
            .0
            .add_instruction(O::from_reference_reshape(ReshapeOperation::new(shape)), Vec::new(), vec![*input])?[0])
    }
}

impl<V, O> ViewWriteCarrier for StagedViewCarrier<'_, V, O>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    fn update_slice(&mut self, target: &AtomId, update: &AtomId, starts: Vec<usize>) -> Result<AtomId, ProgramError> {
        Ok(self.0.add_instruction(
            O::from_reference_update_slice(UpdateSliceOperation::new(starts)),
            Vec::new(),
            vec![*target, *update],
        )?[0])
    }
}

/// Resolves one reference operand's analyzed root, entering state, and view, and stages the root-to-view
/// intermediate snapshots shared by the read, swap, and additive-update discharge rules.
///
/// # Parameters
///
///   - `builder`: Destination program builder receiving the staged view instructions.
///   - `source`: Source region owning `input`.
///   - `analysis`: Reference analysis artifact for the source arena.
///   - `current_states`: Discharged immutable state for every live root.
///   - `access`: Access-kind label used verbatim in the missing-view diagnostic.
///   - `input`: Source reference operand whose view is staged.
fn stage_view_access<'a, V, O>(
    builder: &mut ProgramBuilder<V, O>,
    source: RegionRef<'_, V, O>,
    analysis: &'a ArrayReferenceAnalysis,
    current_states: &HashMap<ReferenceRoot, AtomId>,
    access: &'static str,
    input: AtomId,
) -> Result<(ReferenceRoot, &'a ArrayReferenceView, Vec<AtomId>), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    let root = analyzed_input_root(analysis, source.id(), input)?;
    let state = current_state(current_states, root)?;
    let view = analysis
        .view(ValueId::new(source.id(), input))
        .ok_or_else(|| ProgramError::MalformedProgram(format!("reference {access} has no analyzed view")))?;
    let intermediates = view.intermediates_in(&mut StagedViewCarrier(builder), state)?;
    Ok((root, view, intermediates))
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
    O: ArrayReferenceDischargeOperation,
{
    view.reconstruct_in(&mut StagedViewCarrier(builder), intermediates, replacement)
}

/// Discharges one source region with the requested synthesized state boundary.
fn discharge_region<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ArrayReferenceAnalysis,
    plan: &ReferenceRegionDischargePlan,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    let layout = plan.layout();
    let mut builder = ProgramBuilder::<V, O>::new();
    let mut mapped_atoms = vec![None; source.atoms().len()];
    let mut current_states = HashMap::new();

    for source_index in 0..=source.input_ids().len() {
        if source_index == layout.input_insertion() {
            for root in layout.input_roots() {
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
            ReferenceDischargeRule::NewRoot => {
                let initializer =
                    mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[0])?;
                let root = analysis.root(ValueId::new(source.id(), instruction.outputs()[0])).ok_or_else(|| {
                    ProgramError::MalformedProgram("reference allocation has no analyzed root".to_string())
                })?;
                current_states.insert(root, initializer);
            }
            ReferenceDischargeRule::Alias => {}
            ReferenceDischargeRule::Read => {
                let (_, _, intermediates) = stage_view_access(
                    &mut builder,
                    source,
                    analysis,
                    &current_states,
                    "read",
                    instruction.inputs()[0],
                )?;
                mapped_atoms[instruction.outputs()[0].index()] = Some(*intermediates.last().unwrap());
            }
            ReferenceDischargeRule::Replace => {
                let replacement =
                    mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[1])?;
                let (root, view, intermediates) = stage_view_access(
                    &mut builder,
                    source,
                    analysis,
                    &current_states,
                    "swap",
                    instruction.inputs()[0],
                )?;
                let old = *intermediates.last().unwrap();
                let updated =
                    stage_reference_view_reconstruction(&mut builder, view, intermediates.as_slice(), replacement)?;
                mapped_atoms[instruction.outputs()[0].index()] = Some(old);
                current_states.insert(root, updated);
            }
            ReferenceDischargeRule::Accumulate => {
                let update = mapped_value(source, analysis, &mapped_atoms, &current_states, instruction.inputs()[1])?;
                let (root, view, intermediates) = stage_view_access(
                    &mut builder,
                    source,
                    analysis,
                    &current_states,
                    "additive update",
                    instruction.inputs()[0],
                )?;
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
            ReferenceDischargeRule::Consume => {
                let root = analyzed_input_root(analysis, source.id(), instruction.inputs()[0])?;
                let state = current_state(&current_states, root)?;
                current_states.remove(&root);
                mapped_atoms[instruction.outputs()[0].index()] = Some(state);
            }
            ReferenceDischargeRule::Condition
            | ReferenceDischargeRule::While
            | ReferenceDischargeRule::Scan { .. }
            | ReferenceDischargeRule::Call => discharge_higher_order_instruction(
                source,
                analysis,
                instruction,
                plan.instruction(instruction_id).unwrap(),
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

    let mut output_ids = Vec::with_capacity(source.output_ids().len() + layout.output_roots().len());
    for source_index in 0..=source.output_ids().len() {
        if source_index == layout.output_insertion() {
            for root in layout.output_roots() {
                output_ids.push(current_state(&current_states, *root)?);
            }
        }
        let Some(source_output) = source.output_ids().get(source_index).copied() else {
            continue;
        };
        output_ids.push(mapped_value(source, analysis, &mapped_atoms, &current_states, source_output)?);
    }

    let input_count = source.input_ids().len() + layout.input_roots().len();
    let output_count = output_ids.len();
    let program =
        builder.build::<Vec<V>, Vec<V>>(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
    Ok(program)
}

/// Rewrites one region-bearing operation and widens its attached computation boundaries with immutable state.
fn discharge_higher_order_instruction<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ArrayReferenceAnalysis,
    instruction: &Instruction<O>,
    plan: &ReferenceInstructionDischargePlan,
    builder: &mut ProgramBuilder<V, O>,
    mapped_atoms: &mut [Option<AtomId>],
    current_states: &mut HashMap<ReferenceRoot, AtomId>,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    let rule = plan.rule();
    let operation = match rule {
        ReferenceDischargeRule::Scan { .. } => {
            instruction.operation().with_added_reference_scan_carries(plan.added_input_roots().len())?
        }
        _ => instruction.operation().clone(),
    };
    let inputs = discharged_instruction_inputs(
        source,
        analysis,
        instruction.inputs(),
        plan.input_insertion(),
        plan.added_input_roots(),
        mapped_atoms,
        current_states,
    )?;

    let mut discharged_regions = Vec::with_capacity(instruction.regions().len());
    for (region_index, region) in instruction.regions().iter().copied().enumerate() {
        let attached = source.with_id(region)?;
        discharged_regions.push(discharge_region(attached, analysis, &plan.regions()[region_index])?);
    }

    let result_region = &plan.regions()[plan.result_region_index()];
    let attached_ids = discharged_regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>();
    let outputs = builder.add_instruction(operation, attached_ids, inputs)?.to_vec();
    for (source_output_index, source_output) in instruction.outputs().iter().copied().enumerate() {
        let position = result_region.layout().source_output_position(source_output_index);
        let output = *outputs.get(position).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "discharged `{}` output position {position} is out of range",
                instruction.operation().name(),
            ))
        })?;
        map_source_output(source, analysis, source_output, output, mapped_atoms, current_states)?;
    }
    for root in plan.added_output_roots() {
        let attached_root = match rule {
            ReferenceDischargeRule::While => attached_root(analysis, plan.instruction(), 1, *root),
            _ => attached_root(analysis, plan.instruction(), 0, *root),
        };
        let position = result_region.layout().state_output_position(attached_root).ok_or_else(|| {
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
        current_states.insert(*root, output);
    }
    Ok(())
}

/// Returns the referent array type of one analyzed root in the source arena.
fn root_referent_type<V, O>(source: RegionRef<'_, V, O>, root: ReferenceRoot) -> Result<ArrayType, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
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

/// Returns the formal root used by one attached region for a caller root.
fn attached_root(
    analysis: &ArrayReferenceAnalysis,
    instruction: InstructionId,
    region_index: usize,
    source_root: ReferenceRoot,
) -> ReferenceRoot {
    analysis.region_root_for_source(instruction, region_index, source_root).unwrap_or(source_root)
}

/// Maps an instruction's source operands and inserts synthesized state operands at `insertion`.
fn discharged_instruction_inputs<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ArrayReferenceAnalysis,
    source_inputs: &[AtomId],
    insertion: usize,
    added_roots: &[ReferenceRoot],
    mapped_atoms: &[Option<AtomId>],
    current_states: &HashMap<ReferenceRoot, AtomId>,
) -> Result<Vec<AtomId>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
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
    analysis: &ArrayReferenceAnalysis,
    mapped_atoms: &[Option<AtomId>],
    current_states: &HashMap<ReferenceRoot, AtomId>,
    atom: AtomId,
) -> Result<AtomId, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    if source.atoms()[atom.index()].r#type().is_reference() {
        current_state(current_states, analyzed_input_root(analysis, source.id(), atom)?)
    } else {
        mapped_atoms.get(atom.index()).copied().flatten().ok_or_else(|| {
            ProgramError::MalformedProgram(format!("atom `{atom}` has no ordinary value during reference discharge"))
        })
    }
}

/// Maps one source instruction output to its discharged ordinary value or current reference state.
fn map_source_output<V, O>(
    source: RegionRef<'_, V, O>,
    analysis: &ArrayReferenceAnalysis,
    source_output: AtomId,
    output: AtomId,
    mapped_atoms: &mut [Option<AtomId>],
    current_states: &mut HashMap<ReferenceRoot, AtomId>,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceDischargeOperation,
{
    if source.atoms()[source_output.index()].r#type().is_reference() {
        let root = analysis.root(ValueId::new(source.id(), source_output)).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "reference output `{source_output}` has no analyzed root during discharge",
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
    analysis: &ArrayReferenceAnalysis,
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
fn verify_discharged_program<V: Value<Type = ArrayIrType>, O: ArrayReferenceDischargeOperation>(
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
    use crate::arrays::operations::{
        ArrayIrOperation, ArrayReferenceOperation, ReferenceIndexOperation, ReferenceSliceOperation,
    };
    use crate::arrays::reference_views::{ArrayReference, ArrayReferenceViewTransform};
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::captures::CaptureReference;
    use crate::operations::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::programs::{
        OutputRegionProvenance, ReferenceAccessMode, ReferenceAnalysisError, ReferenceInputAccess, ReferenceSource,
        ReferenceStateBinding, ReferenceType, RegionInterface, RegionSlot,
    };

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

        /// Reports a view rule for a canonical view alias that consumes a second, non-view operand.
        View,

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
                Self::View => "malformed_view_discharge",
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
                // The canonical view inference keeps the declared alias type exact, so only the extra operand
                // deviates from the view contract.
                Self::View => {
                    ReferenceIndexOperation::new(0, 0).infer_output_types(&input_types[..1], region_interfaces)?
                }
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
                Self::NewReference | Self::View | Self::MixedEffectRead | Self::MismatchedReadTypes => &[],
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
                Self::NewReference | Self::View | Self::MixedEffectRead | Self::MismatchedReadTypes => Vec::new(),
                Self::StatefulCondition => vec![
                    OutputRegionProvenance { region_index: 0, output_index },
                    OutputRegionProvenance { region_index: 1, output_index },
                ],
                Self::While => vec![OutputRegionProvenance { region_index: 1, output_index }],
                Self::MisroutedConditionOutputs => vec![OutputRegionProvenance { region_index: 0, output_index }],
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            match self {
                Self::MixedEffectRead | Self::MismatchedReadTypes => Cow::Owned(ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                )),
                Self::View => Cow::Owned(ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::View,
                    }],
                    Vec::new(),
                )),
                _ => Cow::Borrowed(ReferenceOperationSemantics::empty()),
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::MixedEffectRead => {
                    Effects::single(Effect::OrderedState).union(Effects::single(Effect::OrderedIo))
                }
                Self::MismatchedReadTypes | Self::StatefulCondition => Effects::single(Effect::OrderedState),
                Self::NewReference | Self::View | Self::While | Self::MisroutedConditionOutputs => Effects::PURE,
            }
        }
    }

    impl ArrayReferenceOperation for MalformedDischargeOperation {
        fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
            matches!(self, Self::View).then_some(ArrayReferenceViewTransform::Index { axis: 0, index: 0 })
        }
    }

    impl ArrayReferenceDischargeOperation for MalformedDischargeOperation {
        fn reference_discharge_rule(&self) -> ReferenceDischargeRule {
            match self {
                Self::NewReference => ReferenceDischargeRule::NewRoot,
                Self::View => ReferenceDischargeRule::Alias,
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
                "operation `malformed_new_reference_discharge` reports an incompatible `new_root` reference \
                 discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_view_rule_with_a_second_operand() {
        // Discharge replays a view rule by staging exactly its declared transform over the single aliased handle, so
        // an operand that the transform cannot account for must be rejected instead of silently dropped.
        let mut builder = ProgramBuilder::<TestValue, MalformedDischargeOperation>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into());
        let extra = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedDischargeOperation::View, Vec::new(), vec![reference, extra])
            .unwrap();
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "operation `malformed_view_discharge` reports an incompatible `alias` reference discharge rule"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_rejects_while_rule_without_fixed_point_reference_carries() {
        // A while rule must forward every reference carry positionally. This fixture keeps matching input and output
        // arity but never overrides `reference_output_identity_input`, so the positional-carry validation rejects it.
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let first_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let swapped_snapshot = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement])
            .unwrap()[0];
        assert!(
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
                .unwrap()
                .is_empty(),
        );
        let final_snapshot =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let indexed =
            builder.add_instruction(ReferenceIndexOperation::new(0, 3), Vec::new(), vec![reference]).unwrap()[0];
        let indexed_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![indexed]).unwrap()[0];
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
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![composed, replacement])
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![composed, update])
            .unwrap();
        let final_snapshot =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let row = builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference]).unwrap()[0];
        let old =
            builder.add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![row, update]).unwrap();
        let final_snapshot =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
    fn test_composed_index_of_slice_swap_discharge_reconstructs_both_view_steps() {
        // Swapping through an index composed onto a slice must write back through both steps in reverse order, so the
        // discharged program reconstructs the sliced block from the squeezed row before writing it into the root.
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let block = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![reference],
            )
            .unwrap()[0];
        let row = builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![block]).unwrap()[0];
        let old =
            builder.add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement]).unwrap()[0];
        let final_snapshot =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            vector(vec![10.0, 20.0]),
        ];
        let expected = vec![
            vector(vec![7.0, 8.0]),
            TestValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 9.0])),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %6:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %5
                    %7:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %6
                in (%4, %7)"},
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
                let reference =
                    builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
                let mut outputs = Vec::new();
                let mut oracle_state = 2.0f32;
                let mut oracle_outputs = Vec::new();
                for step in steps {
                    match step {
                        Step::Read => {
                            outputs.push(
                                builder
                                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference])
                                    .unwrap()[0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                        }
                        Step::Swap => {
                            outputs.push(
                                builder
                                    .add_instruction(
                                        ReferenceSwapOperation::new(),
                                        Vec::new(),
                                        vec![reference, replacement],
                                    )
                                    .unwrap()[0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                            oracle_state = 7.0;
                        }
                        Step::AddUpdate => {
                            builder
                                .add_instruction(
                                    ReferenceAddUpdateOperation::new(),
                                    Vec::new(),
                                    vec![reference, update],
                                )
                                .unwrap();
                            oracle_state += 3.0;
                        }
                    }
                }
                outputs.push(
                    builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0],
                );
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
        let second_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second]).unwrap()[0];
        let first_snapshot = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, replacement])
            .unwrap()[0];
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
                ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 0 }, 0, Some(2)),
                ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 1 }, 1, None),
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
                "[ReferenceStateBinding { source: PublicInput { index: 0 }, ",
                "discharged_input_index: 0, final_state_output_index: Some(2) }, ",
                "ReferenceStateBinding { source: PublicInput { index: 1 }, ",
                "discharged_input_index: 1, final_state_output_index: None }]",
            ),
        );
        assert_eq!(
            discharged.program().interpret(vec![scalar(10.0), scalar(20.0), scalar(7.0)]),
            Ok(vec![scalar(20.0), scalar(10.0), scalar(7.0)]),
        );
    }

    #[test]
    fn test_array_reference_discharge_matches_generic_plan_layout_and_bindings() {
        let reference_type = ReferenceType::new(scalar_type());
        let make_branch = || {
            let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
            let replacement = builder.add_input(scalar_type().into());
            let captured = builder.add_constant(Capture::new(0, reference_type.clone().into()));
            let old = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![captured, replacement])
                .unwrap()[0];
            builder
                .build::<Vec<Capture>, Vec<Capture>>(vec![old], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        builder.add_input(reference_type.clone().into());
        let read_only = builder.add_input(reference_type.clone().into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let replacement = builder.add_input(scalar_type().into());
        let branch = builder.import_region(make_branch().entry_region_ref());
        let old = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![branch, branch],
                vec![predicate, replacement],
            )
            .unwrap()[0];
        let snapshot = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![read_only]).unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![old, snapshot], vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        let analysis = program.analyze_array_references_with_lifted_captures(1).unwrap();
        let plan =
            ReferenceDischargePlan::new(&program, analysis.analysis(), CaptureOperation::reference_discharge_rule)
                .unwrap();
        let captured_root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        let read_only_root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let expected_bindings = [
            ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, 0, Some(2)),
            ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 0 }, 1, None),
        ];

        assert_eq!(plan.public_output_count(), 2);
        assert_eq!(plan.external_states(), expected_bindings);
        assert_eq!(plan.entry().layout().input_insertion(), 0);
        assert_eq!(plan.entry().layout().input_roots(), &[]);
        assert_eq!(plan.entry().layout().output_insertion(), 2);
        assert_eq!(plan.entry().layout().output_roots(), &[captured_root]);
        assert_eq!(plan.entry().layout().source_output_position(0), 0);
        assert_eq!(plan.entry().layout().source_output_position(1), 1);
        assert_eq!(plan.entry().layout().state_output_position(captured_root), Some(2));
        assert_eq!(plan.entry().layout().state_output_position(read_only_root), None);
        let condition = plan.entry().instruction(InstructionId::new(program.entry(), 0)).unwrap();
        assert_eq!(condition.rule(), ReferenceDischargeRule::Condition);
        assert_eq!(condition.input_insertion(), 2);
        assert_eq!(condition.added_input_roots(), &[captured_root]);
        assert_eq!(condition.added_output_roots(), &[captured_root]);
        assert_eq!(condition.regions().len(), 2);
        for branch in condition.regions() {
            assert_eq!(branch.layout().input_insertion(), 1);
            assert_eq!(branch.layout().input_roots(), &[captured_root]);
            assert_eq!(branch.layout().output_insertion(), 1);
            assert_eq!(branch.layout().output_roots(), &[captured_root]);
        }

        let discharged = program.discharge_references_with_lifted_captures(1).unwrap();
        assert_eq!(discharged.public_output_count(), plan.public_output_count());
        assert_eq!(discharged.external_states(), plan.external_states());
        assert_eq!(discharged.external_states(), expected_bindings);
        assert_eq!(discharged.program().output_count(), 3);
        let entry = discharged.program().entry_region_ref();
        let condition = &entry.instructions()[0];
        assert_eq!(condition.inputs().len(), 3);
        assert_eq!(condition.outputs().len(), 2);
        for region in condition.regions() {
            let branch = entry.with_id(*region).unwrap();
            assert_eq!(branch.input_ids().len(), 2);
            assert_eq!(branch.output_ids().len(), 2);
        }
    }

    #[test]
    fn test_array_reference_discharge_local_references() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let external = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The shared local-only gate names the requesting transform together with the caller-owned boundary source.
        // Public arguments and captures are both external: neither boundary supplies the runtime holder needed for
        // final-state write-back.
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, initial])
            .unwrap();
        let output = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let invalid_read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement])
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        false_builder.add_input(scalar_type().into());
        let snapshot =
            false_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        let first_snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first]).unwrap()[0];
        let second_snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![second, second_replacement])
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
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, first_replacement])
            .unwrap()[0];
        let second_snapshot =
            false_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second]).unwrap()[0];
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
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference])
            .unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let loop_condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let loop_body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let loop_condition = true_builder.import_region(loop_condition.entry_region_ref());
        let loop_body = true_builder.import_region(loop_body.entry_region_ref());
        let initial = true_builder.add_input(scalar_type().into());
        let reference =
            true_builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference =
            true_builder.add_instruction(operation, vec![loop_condition, loop_body], vec![reference]).unwrap()[0];
        let value =
            true_builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            condition_builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference])
                .unwrap();
            let condition = condition_builder.add_constant(boolean(condition_value));
            let condition = condition_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = body_builder.add_input(reference_type.clone().into());
            let update = body_builder.add_constant(scalar(1.0));
            body_builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
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
            let value = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference]).unwrap()[0];
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
    fn test_scan_discharge_appends_the_synthesized_carry_after_the_declared_carry_prefix() {
        // A scan that already declares an ordinary carry pins the synthesized-state placement exactly: the state
        // operand joins the carry prefix behind every declared carry and ahead of the trailing stacked inputs, on the
        // parent operand list, the body boundary, and the rewritten carry count alike.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let carry = body_builder.add_input(scalar_type().into());
        let element = body_builder.add_input(scalar_type().into());
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element])
            .unwrap();
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, element])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![next_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let final_carry = builder
            .add_instruction(
                ScanOperation::<ArrayIrValue<CaptureArray>>::new(1, 3),
                vec![body],
                vec![initial, elements],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let closed =
            ClosedProgram::new(program, vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32)))])
                .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3] .
                let %3:f32[], %4:f32[] = scan [carry_count=2, length=3, reverse=false] %1 %0 %2 [
                    body={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %2
                            %4:f32[] = add %0 %2
                        in (%4, %3)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(discharged.external_states()[0].discharged_input_index(), 0);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));
        let CaptureOperation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.length(), &Dimension::Static(3));
    }

    #[test]
    fn test_read_only_condition_discharge_adds_no_final_state_output() {
        // A closure that only reads an external root needs the state to enter both branches, but the root's value
        // never changes, so no branch gains a final-state result and the parent condition keeps exactly its public
        // outputs instead of carrying a dead state output.
        let reference_type = ReferenceType::new(scalar_type());
        let make_branch = || {
            let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = branch_builder.add_input(reference_type.clone().into());
            let snapshot =
                branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
            branch_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(make_branch().entry_region_ref());
        let false_branch = builder.import_region(make_branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.program().interpret(vec![boolean(true), scalar(4.0)]), Ok(vec![scalar(4.0)]));
        assert_eq!(discharged.program().interpret(vec![boolean(false), scalar(4.0)]), Ok(vec![scalar(4.0)]));
    }

    #[test]
    fn test_scan_discharge_preserves_zero_length_state_identity() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference]).unwrap()[0];
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
    fn test_call_discharge_widens_a_positional_callee_with_its_final_state() {
        /// Array-IR family extended with one positional call, mirroring how a backend attaches a compiled callee
        /// region, forwards its operands positionally, and reports its outputs positionally.
        #[derive(Clone, Debug)]
        enum CallingOperation {
            /// Native array-IR operation.
            Native(TestOperation),

            /// Positional call of one attached callee region.
            Call,
        }

        impl Operation for CallingOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                match self {
                    Self::Native(operation) => operation.name(),
                    Self::Call => "test_call",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                match self {
                    Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
                    Self::Call => Ok(region_interfaces[0].output_types().to_vec()),
                }
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                match self {
                    Self::Native(operation) => operation.region_slots(),
                    Self::Call => const { &[RegionSlot::computation("callee")] },
                }
            }

            fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
                match self {
                    Self::Native(operation) => operation.input_region_provenance(region_index, input_index),
                    Self::Call => Some(input_index),
                }
            }

            fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
                match self {
                    Self::Native(operation) => operation.output_region_provenance(output_index),
                    Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                }
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                match self {
                    Self::Native(operation) => operation.reference_semantics(),
                    Self::Call => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                }
            }

            fn effects(&self) -> Effects {
                match self {
                    Self::Native(operation) => operation.effects(),
                    Self::Call => Effects::PURE,
                }
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                match self {
                    Self::Native(operation) => operation.render(formatter, indentation),
                    Self::Call => formatter.write_str(self.name()),
                }
            }
        }

        impl ArrayReferenceOperation for CallingOperation {
            fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
                match self {
                    Self::Native(operation) => operation.reference_view_transform(),
                    Self::Call => None,
                }
            }
        }

        impl ArrayReferenceDischargeOperation for CallingOperation {
            fn reference_discharge_rule(&self) -> ReferenceDischargeRule {
                match self {
                    Self::Native(operation) => operation.reference_discharge_rule(),
                    Self::Call => ReferenceDischargeRule::Call,
                }
            }

            fn with_added_reference_scan_carries(&self, additional_carry_count: usize) -> Result<Self, ProgramError> {
                match self {
                    Self::Native(operation) => {
                        Ok(Self::Native(operation.with_added_reference_scan_carries(additional_carry_count)?))
                    }
                    Self::Call => Err(ProgramError::MalformedProgram(
                        "operation `test_call` is not a scan and cannot carry discharged reference state".to_string(),
                    )),
                }
            }

            fn from_reference_reshape(operation: ReshapeOperation) -> Self {
                Self::Native(TestOperation::from_reference_reshape(operation))
            }

            fn from_reference_slice(operation: SliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_slice(operation))
            }

            fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_update_slice(operation))
            }
        }

        impl From<AddOperation<ArrayIrType>> for CallingOperation {
            fn from(operation: AddOperation<ArrayIrType>) -> Self {
                Self::Native(operation.into())
            }
        }

        // The callee mutates the root it receives and returns only the old snapshot, so its declared boundary hides
        // the final state that the call site needs after discharge.
        let mut callee_builder = ProgramBuilder::<TestValue, CallingOperation>::new();
        let reference = callee_builder.add_input(ReferenceType::new(scalar_type()).into());
        let replacement = callee_builder.add_input(scalar_type().into());
        let old = callee_builder
            .add_instruction(
                CallingOperation::Native(ReferenceSwapOperation::new().into()),
                Vec::new(),
                vec![reference, replacement],
            )
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![old], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, CallingOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let replacement = builder.add_input(scalar_type().into());
        let root = builder
            .add_instruction(CallingOperation::Native(NewReferenceOperation::new().into()), Vec::new(), vec![initial])
            .unwrap()[0];
        let old = builder.add_instruction(CallingOperation::Call, vec![callee], vec![root, replacement]).unwrap()[0];
        let final_snapshot = builder
            .add_instruction(CallingOperation::Native(FreezeReferenceOperation::new().into()), Vec::new(), vec![root])
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Discharge appends the callee's final state to its outputs and threads that result into the freeze, leaving
        // the call itself in place with its positional operand and output contract intact.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = test_call %0 %1 [
                    callee={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_closed_program_discharge_resolves_reference_captures_inside_condition_regions() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = branch_builder.add_constant(Capture::new(0, reference_type.into()));
        let value =
            branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            &[ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, 0, None)],
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
        let value =
            leaf_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference])
            .unwrap();
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
        // A capture read by a scan body becomes a synthesized carry appended after the declared carry prefix,
        // which raises the rewritten scan's carry count without disturbing its length, direction, or unroll factor.
        let reference_type = ReferenceType::new(scalar_type());
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32));
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        // A capture that a scan body accumulates into reaches that body only through a synthesized carry, which is
        // the most involved discharge path: the state enters the scan appended after the declared carry prefix, is
        // updated inside the body, leaves through the matching synthesized carry output, and reaches the hidden
        // entry final-state output after the public prefix. The capture value family carries no data, so the
        // rendered program rather than an interpretation pins the resulting state flow.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let update = body_builder.add_constant(Capture::new(1, scalar_type().into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let replacement = false_builder.add_constant(scalar(9.0));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement])
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![view, update]).unwrap();
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
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let output = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference])
            .unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(2.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
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
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let outputs =
            builder.add_instruction(ScanOperation::<TestValue>::new(1, 4), vec![body], vec![reference]).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let frozen =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![final_reference]).unwrap()[0];
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
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let runtime_length = builder.add_input(DimensionType::new(length.clone()).into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let scanned = builder
            .add_instruction(
                ScanOperation::<TestValue>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![reference, runtime_length],
            )
            .unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![scanned]).unwrap()[0];
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
