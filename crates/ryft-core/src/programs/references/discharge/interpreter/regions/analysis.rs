use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::macros::check_count;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::references::discharge::transform::{
    ReferenceDischargeAllocationId, ReferenceDischargeCaptureScope,
};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceOutput};
use crate::programs::regions::RegionRef;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

// TODO(eaplatanios): Review this module.

/// Transitive reference-access summary of one region closure, expressed in the caller allocations its boundary names.
///
/// This is the analysis a structured rule needs before it can size its state boundary, and it is computed entirely
/// from generic hooks: operation-local [`Operation::reference_semantics`], the input- and output-region provenance
/// hooks, reference-output identity, and recursive summaries of nested regions. Allocations allocated inside the
/// closure are deliberately absent: they belong to no caller and cross no boundary.
///
/// The summary separates reachability from semantic access. The reached set holds every caller allocation the
/// closure's replay must be able to resolve, including a capture constant that is only rematerialized and passed
/// along, and is what sizes the state boundary through
/// [`boundary_widening`](crate::programs::references::ReferenceDischargeContext::boundary_widening).
/// [`accessed_allocations`](Self::accessed_allocations) and [`access_modes`](Self::access_modes) hold only the
/// allocations the closure semantically accesses, which is what region access policies validate. Sizing a boundary
/// from the accessed allocations would under-thread merely-forwarded captures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceDischargeRegionSummary {
    /// Every caller allocation the closure must be able to resolve while replaying, whether or not it is semantically
    /// accessed.
    reached: BTreeSet<ReferenceDischargeAllocationId>,

    /// Every caller allocation the closure accesses, mapped to its exact non-consuming access modes.
    accesses: BTreeMap<ReferenceDischargeAllocationId, BTreeSet<ReferenceAccessMode>>,

    /// Refer to the documentation of [`Self::output_allocations`].
    output_allocations: Vec<Option<ReferenceDischargeAllocationId>>,
}

impl ReferenceDischargeRegionSummary {
    /// Summarizes the transitive reference accesses of the closure of `region`, attached at `region_index` to
    /// `operation`, in the terms of the caller allocations its boundary names.
    ///
    /// A structured rule needs this summary before it can size its state boundary: which allocations a region
    /// closure touches, and which of them it mutates, is exactly what decides how wide the rewritten operation must
    /// be. The summary is computed entirely from generic hooks, namely operation-local
    /// [`Operation::reference_semantics`], the region-provenance hooks, reference-output identity, and recursive
    /// summaries of nested regions, so a third-party structured operation needs no companion declaration surface to
    /// be summarized. Every access mode the closure performs is validated against
    /// [`Operation::allows_reference_access_through_region_input`] before the summary is returned, and each nested
    /// region is validated against its own operation while the offending region is still named.
    ///
    /// The traversal maps each reference-typed atom of the region onto the caller allocation it denotes, or onto
    /// [`None`] when the allocation was allocated inside the closure and therefore crosses no boundary. Nested
    /// regions are entered through [`Operation::input_region_provenance`], and a structured operation's
    /// reference-typed output is resolved either by [`Operation::reference_output_identity_input`], which states
    /// outright which input's allocation it preserves, or by [`Operation::output_region_provenance`], which names the
    /// region output it forwards.
    ///
    /// The region's own capture scope is computed from `captures` rather than supplied, because whether a region
    /// establishes a fresh capture prefix is stated by [`Operation::region_capture_input_count`]. A reference-typed
    /// constant is resolved through that scope and seeded exactly like a boundary position, because a capture-lifted
    /// program names its caller's references that way. That is what lets a structured rule discover that its closure
    /// reaches an allocation its inputs never named, and therefore what makes added state positions reachable.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation the region is attached to.
    ///   - `region_index`: Position of the region among that operation's attached regions.
    ///   - `region`: Region whose closure is summarized.
    ///   - `inputs`: Caller allocation denoted by each of the region's declared inputs, in boundary order, with
    ///     [`None`] wherever the position carries a value.
    ///   - `captures`: Capture scope of the region in which `operation` is applied.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the
    /// operation declares a capture prefix longer than the region's boundary, when a reference-typed nested boundary
    /// position declares no provenance to follow, when the closure reaches a reference that entered neither through
    /// its boundary nor through its capture scope, when the closure consumes a caller allocation, which no state
    /// boundary can express, or when an operation does not permit one of the exact access modes a closure performs
    /// through one of its regions.
    pub fn new<V: Value, O: Operation<Type = V::Type>, Owner: Operation>(
        operation: &Owner,
        region_index: usize,
        region: RegionRef<'_, V, O>,
        inputs: &[Option<ReferenceDischargeAllocationId>],
        captures: &ReferenceDischargeCaptureScope<V>,
    ) -> Result<Self, ProgramError> {
        let captures =
            captures.nested_scope(operation.region_capture_input_count(region_index), inputs, region.id())?;
        check_count!("input", inputs, region.input_ids().len(), ProgramError);
        let mut summary = Self::default();
        let is_reference = |atom: AtomId| region.atoms()[atom.index()].r#type().is_reference();
        let mut allocations = HashMap::<AtomId, Option<ReferenceDischargeAllocationId>>::new();
        for (input, allocation) in region.input_ids().iter().copied().zip(inputs) {
            if is_reference(input) {
                allocations.insert(input, *allocation);
            }
        }

        // A capture-scoped constant is seeded exactly like a boundary position. Materializing one makes its
        // allocation reachable during replay but is not itself a semantic reference read; actual accesses are
        // recorded from operation semantics below.
        let materialized_atoms = region
            .instructions()
            .iter()
            .flat_map(|instruction| instruction.inputs().iter().copied())
            .chain(region.output_ids().iter().copied())
            .collect::<HashSet<_>>();
        for (atom_index, atom) in region.atoms().iter().enumerate() {
            let atom_id = AtomId::new(atom_index);
            if let Atom::Constant(constant) = atom
                && constant.r#type().is_reference()
                && let Some(allocation) = captures.resolve(constant)
            {
                allocations.insert(atom_id, Some(allocation));
                if materialized_atoms.contains(&atom_id) {
                    summary.reached.insert(allocation);
                }
            }
        }
        let input_atom = |instruction: &Instruction<O>, index: usize, role: &str| {
            instruction.inputs().get(index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "operation `{}` names {role} input {index} but the application has {} inputs",
                    instruction.operation().name(),
                    instruction.inputs().len(),
                ))
            })
        };

        // A reference-typed atom the traversal never bound denotes a reference that entered this region neither
        // through its boundary nor through its capture scope. The environment has no allocation for it, so the
        // summary reports it here rather than dropping the access and letting the replay fail later for a reason that
        // no longer names the operation that performed it.
        let resolve = |allocations: &HashMap<AtomId, Option<ReferenceDischargeAllocationId>>,
                       atom: AtomId,
                       operation: &str| {
            match allocations.get(&atom) {
                Some(allocation) => Ok(*allocation),
                None if is_reference(atom) => Err(ProgramError::MalformedProgram(format!(
                    "operation `{operation}` reaches a reference that entered region `{}` neither through its boundary \
                         nor through its capture scope",
                    region.id(),
                ))),
                None => Ok(None),
            }
        };
        for instruction in region.instructions() {
            let operation = instruction.operation();
            let semantics = operation.reference_semantics();
            for access in semantics.inputs() {
                let accessed = input_atom(instruction, access.input_index(), "an accessed")?;
                if let Some(allocation) = resolve(&allocations, accessed, operation.name())? {
                    summary.record(allocation, access.mode(), operation.name())?;
                }
            }
            for output in semantics.outputs() {
                let defined = instruction.outputs().get(output.output_index()).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "operation `{}` classifies output {} but the application has {} outputs",
                        operation.name(),
                        output.output_index(),
                        instruction.outputs().len(),
                    ))
                })?;
                let allocation = match output {
                    ReferenceOutput::Allocation { .. } => None,
                    ReferenceOutput::Alias { input_index, .. } => {
                        resolve(&allocations, input_atom(instruction, *input_index, "an aliased")?, operation.name())?
                    }
                };
                allocations.insert(defined, allocation);
            }
            let mut attached_output_allocations = Vec::with_capacity(instruction.regions().len());
            for (region_index, attached) in instruction.regions().iter().copied().enumerate() {
                let attached = region.with_id(attached)?;
                let nested = attached
                    .input_ids()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(input_index, input)| {
                        if !attached.atoms()[input.index()].r#type().is_reference() {
                            return Ok(None);
                        }
                        let Some(supplying_index) = operation.input_region_provenance(region_index, input_index) else {
                            return Err(ProgramError::MalformedProgram(format!(
                                "operation `{}` passes a reference into region {region_index} input {input_index} \
                                 without declaring which input supplies it",
                                operation.name(),
                            )));
                        };
                        resolve(&allocations, input_atom(instruction, supplying_index, "a region")?, operation.name())
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // The nested closure is summarized on its own first, so that an operation restricting what its
                // regions may do to an entering allocation is held to that restriction here, where the offending
                // region is still named, rather than only indirectly when a rebuilt region contradicts the widening
                // it was given.
                let nested_summary = Self::new(operation, region_index, attached, nested.as_slice(), &captures)?;
                summary.merge(&nested_summary);
                attached_output_allocations.push(nested_summary.output_allocations);
            }

            // A reference-typed output of a region-carrying operation preserves an allocation rather than classifying
            // one, so it resolves through the generic hooks that state where it came from: an explicit input identity
            // when the operation declares one, and otherwise the region output it forwards.
            for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
                if !is_reference(output) || allocations.contains_key(&output) {
                    continue;
                }
                let preserved = match operation.reference_output_identity_input(output_index) {
                    Some(input_index) => {
                        resolve(&allocations, input_atom(instruction, input_index, "a preserved")?, operation.name())?
                    }
                    None => {
                        // Without an explicit input identity, the operation must name through its output-region
                        // provenance the attached-region outputs it forwards. The first origin establishes the
                        // forwarded allocation and every additional origin must denote the same allocation, as
                        // happens for corresponding outputs of condition branches.
                        let provenance = operation.output_region_provenance(output_index);
                        if provenance.is_empty() {
                            return Err(ProgramError::MalformedProgram(format!(
                                "operation `{}` produces a reference at output {output_index} without declaring which \
                                 input allocation it preserves or which region output it forwards",
                                operation.name(),
                            )));
                        }
                        let mut forwarded = None;
                        for (position, origin) in provenance.iter().enumerate() {
                            let allocation = attached_output_allocations
                                .get(origin.region_index)
                                .and_then(|allocations| allocations.get(origin.output_index).copied())
                                .ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "operation `{}` forwards output {output_index} from region {} output {}, \
                                         which it does not attach",
                                        operation.name(),
                                        origin.region_index,
                                        origin.output_index,
                                    ))
                                })?;
                            if position == 0 {
                                forwarded = allocation;
                            } else if forwarded != allocation {
                                return Err(ProgramError::MalformedProgram(format!(
                                    "operation `{}` forwards output {output_index} from regions that return \
                                     different reference allocations",
                                    operation.name(),
                                )));
                            }
                        }
                        forwarded
                    }
                };
                allocations.insert(output, preserved);
            }
        }
        summary.output_allocations = region
            .output_ids()
            .iter()
            .copied()
            .map(|output| if is_reference(output) { allocations.get(&output).copied().flatten() } else { None })
            .collect();
        summary.reached.extend(summary.output_allocations.iter().copied().flatten());

        // Every exact access mode the closure performs is held to the region access policy that the owning operation
        // declares for this region.
        for (allocation, modes) in &summary.accesses {
            for mode in modes {
                if !operation.allows_reference_access_through_region_input(region_index, *mode) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "operation `{}` does not allow region {region_index} to access {allocation} with mode `{mode}`",
                        operation.name(),
                    )));
                }
            }
        }
        Ok(summary)
    }

    /// Returns every caller allocation the closure must be able to resolve, in canonical allocation order.
    #[inline]
    pub(crate) fn reached_allocations(&self) -> impl Iterator<Item = ReferenceDischargeAllocationId> + '_ {
        self.reached.iter().copied()
    }

    /// Returns every caller allocation the closure accesses, in canonical allocation order.
    #[inline]
    pub fn accessed_allocations(&self) -> impl Iterator<Item = ReferenceDischargeAllocationId> + '_ {
        self.accesses.keys().copied()
    }

    /// Returns the exact access modes recorded for `allocation`, in [`ReferenceAccessMode`] declaration order.
    #[inline]
    pub fn access_modes(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> impl Iterator<Item = ReferenceAccessMode> + '_ {
        self.accesses.get(&allocation).into_iter().flatten().copied()
    }

    /// Returns the caller allocation each declared region output denotes, or [`None`] where the output is a value. A
    /// region that returns an allocation already publishes that allocation's final state at its own output position,
    /// so a rule that widens the boundary must not publish it a second time.
    #[inline]
    pub fn output_allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.output_allocations.as_slice()
    }

    /// Returns whether any statically reachable path through the closure writes or accumulates into `allocation`. An
    /// allocation the closure only reads is not mutated, which is the fact read-only pruning consults.
    ///
    /// This classification is intentionally conservative across structured control flow: a write in either branch or
    /// in a loop body marks the allocation as mutated even when one execution takes the other branch or performs zero
    /// iterations. Discharge therefore threads and publishes a hidden final state for every such allocation, and at
    /// runtime that state is simply unchanged when the mutating path does not execute.
    #[inline]
    pub fn is_mutated(&self, allocation: ReferenceDischargeAllocationId) -> bool {
        self.access_modes(allocation).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate,
            )
        })
    }

    /// Merges another closure's reached allocations and accesses into this summary, which is what an operation with
    /// several attached regions threads through one shared state boundary. An allocation that only one nested closure
    /// returns or rematerializes stays reachable, and therefore threaded, at the merged level. Declared output
    /// allocations belong to one region's own boundary rather than to the shared state, so this summary keeps its own.
    /// An operation whose regions must agree on them, such as a condition, has that agreement checked against the
    /// rebuilt regions themselves.
    pub fn merge(&mut self, other: &Self) {
        self.reached.extend(other.reached.iter().copied());
        for (allocation, modes) in &other.accesses {
            self.accesses.entry(*allocation).or_default().extend(modes.iter().copied());
        }
    }

    /// Records one access, or rejects a consuming access to a caller allocation.
    ///
    /// # Parameters
    ///
    ///   - `allocation`: Caller allocation being accessed.
    ///   - `mode`: Semantic mode of the access.
    ///   - `operation`: Name of the accessing operation, used in the consumption diagnostic.
    pub(super) fn record(
        &mut self,
        allocation: ReferenceDischargeAllocationId,
        mode: ReferenceAccessMode,
        operation: &str,
    ) -> Result<(), ProgramError> {
        // A consumed allocation has no successor, so no symmetric boundary and no final-state output can describe what
        // happened to it, and an allocation that survives as a reference fares no better: whether a region consumed it
        // can depend on which branch ran, which the caller's environment cannot represent.
        if mode.is_consuming() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {allocation} into a region that consumes it through `{operation}`",
            )));
        }
        self.reached.insert(allocation);
        self.accesses.entry(allocation).or_default().insert(mode);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::references::discharge::tests::*;
    use crate::programs::references::discharge::transform::{
        RecursiveReferenceDischargeDriver, ReferenceDischargeCaptureScope, ReferenceDischargeDriver,
        ReferenceDischargeRegionBoundary, ReferenceDischargeRegionStateInsertion,
    };
    use crate::programs::references::semantics::ReferenceAccessMode;
    use crate::programs::references::types::ReferenceType;

    use super::*;

    #[test]
    fn test_reference_discharge_region_summary_unions_exact_access_modes() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let mut left = ReferenceDischargeRegionSummary::default();
        left.record(allocation, ReferenceAccessMode::Read, "list.read").unwrap();
        left.record(allocation, ReferenceAccessMode::ReadWrite, "list.swap").unwrap();
        left.output_allocations = vec![Some(allocation)];
        let mut right = ReferenceDischargeRegionSummary::default();
        right.record(allocation, ReferenceAccessMode::Write, "list.write").unwrap();
        right.record(allocation, ReferenceAccessMode::Accumulate, "list.add_update").unwrap();
        right.output_allocations = vec![None];

        let mut merged = left;
        merged.merge(&right);
        assert_eq!(
            merged.access_modes(allocation).collect::<Vec<_>>(),
            vec![
                ReferenceAccessMode::Read,
                ReferenceAccessMode::Write,
                ReferenceAccessMode::ReadWrite,
                ReferenceAccessMode::Accumulate,
            ],
        );
        assert!(merged.is_mutated(allocation));
        assert_eq!(merged.output_allocations(), [Some(allocation)]);
    }

    #[test]
    fn test_reference_discharge_region_summary_validates_each_exact_access_mode() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let modes = [
            ReferenceAccessMode::Read,
            ReferenceAccessMode::Write,
            ReferenceAccessMode::ReadWrite,
            ReferenceAccessMode::Accumulate,
        ];

        for accessed in modes {
            let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
            let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
            let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
            match accessed {
                ReferenceAccessMode::Read => {
                    builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap();
                }
                ReferenceAccessMode::Write => {
                    builder
                        .add_instruction(ListOperation::Write, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::ReadWrite => {
                    builder
                        .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::Accumulate => {
                    builder
                        .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::Consume => unreachable!(),
            }
            let region = builder
                .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap();

            for allowed in modes {
                let result = context.region_summary(
                    &SingleModeRegionOperation(allowed),
                    0,
                    region.entry_region_ref(),
                    &[Some(allocation), None],
                );
                if allowed == accessed {
                    let summary = result.unwrap();
                    assert_eq!(summary.access_modes(allocation).collect::<Vec<_>>(), vec![accessed]);
                    assert_eq!(
                        summary.is_mutated(allocation),
                        matches!(
                            accessed,
                            ReferenceAccessMode::Write
                                | ReferenceAccessMode::ReadWrite
                                | ReferenceAccessMode::Accumulate,
                        ),
                    );
                } else {
                    assert_eq!(
                        result,
                        Err(ProgramError::MalformedProgram(format!(
                            "operation `test.single_mode_region` does not allow region 0 to access {allocation} with \
                             mode `{accessed}`",
                        ))),
                    );
                }
            }
        }

        // A nested call's swap remains `ReadWrite` at the outer policy boundary; permitting `Write` cannot admit it
        // through a lossy generic mutation fact.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, replacement], None)
            .unwrap();
        let region = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            context.region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                region.entry_region_ref(),
                &[Some(allocation), None],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "operation `test.single_mode_region` does not allow region 0 to access {allocation} with mode \
                 `read/write`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_region_summary_reports_transitive_accesses_and_output_allocations() {
        // A callee that replaces the state of the reference it receives, so the outer region's access to that
        // allocation is transitive rather than local.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_reference =
            callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![callee_reference, replacement], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The outer region reads the caller's allocation directly, replaces it through the callee, and separately
        // allocates, reads, and returns an allocation of its own.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let local = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![snapshot], None).unwrap()[0];
        let local_snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![local], None).unwrap()[0];
        let previous = builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, replacement], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![reference, local, snapshot, local_snapshot, previous],
                vec![Placeholder; 2],
                vec![Placeholder; 5],
            )
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let summary = context
            .region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(allocation), None])
            .unwrap();

        // The caller allocation is reported as mutated because the nested callee replaces it, while the region's own
        // allocation crosses no boundary and is therefore absent from the summary entirely.
        assert_eq!(summary.accessed_allocations().collect::<Vec<_>>(), vec![allocation]);
        assert_eq!(
            summary.access_modes(allocation).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::ReadWrite],
        );
        assert!(summary.is_mutated(allocation));

        // A declared output resolves to the caller allocation it denotes: the first output returns the allocation
        // itself, the second returns a region-local allocation, and the remaining three are values.
        assert_eq!(summary.output_allocations(), &[Some(allocation), None, None, None, None]);
    }

    #[test]
    fn test_reference_discharge_region_summary_rejects_a_closure_that_consumes_a_caller_allocation() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // A consumed allocation has no successor state, so no state boundary can describe what became of it. The
        // summary rejects that outright rather than letting the caller keep threading state that is no longer live.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        assert_eq!(
            context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(allocation)]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {allocation} into a region that consumes it through `list.freeze`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_threads_a_capture_scoped_allocation_a_nested_region_only_receives() {
        // A closure can reach a capture-scoped allocation without ever accessing it, by passing the constant into a
        // nested region that ignores it. The replay still materializes the constant, because something consumes it, so
        // the allocation has to be threaded even though no reference access records it. In particular, materializing
        // the capture must not invent a semantic read that the enclosing operation's region policy could reject.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let ignored = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let forwarded = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![forwarded], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(callee.entry_region_ref().input_ids().contains(&ignored));

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee = builder.import_program(callee);
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let value = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let forwarded =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![captured, value], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![forwarded], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the captured allocation").unwrap().allocation_id();
        let context = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));

        // The enclosing policy accepts writes only. Capture reachability still sizes the boundary, while the exact
        // access summary remains empty because neither closure semantically accesses the allocation.
        let summary = context
            .region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                program.entry_region_ref(),
                &[None],
            )
            .unwrap();
        assert_eq!(summary.accessed_allocations().collect::<Vec<_>>(), Vec::<ReferenceDischargeAllocationId>::new());
        assert_eq!(summary.access_modes(allocation).collect::<Vec<_>>(), Vec::<ReferenceAccessMode>::new());
        assert!(!summary.is_mutated(allocation));
        assert_eq!(
            context.boundary_widening(&summary, &BTreeSet::new()).unwrap().threaded(),
            &BTreeSet::from([allocation]),
        );

        // The rebuilt region therefore receives the allocation's entering state and hands it to its own callee.
        let regions = [program.clone()];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![None],
            ReferenceDischargeRegionStateInsertion::new(vec![allocation], 1),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
        );
        let result = driver.rebuild_region(&context, 0, &boundary).unwrap();
        assert_eq!(
            result.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2> = list.call %1 %0 [
                    callee={
                        lambda %0:list<2>, %1:list<2> .
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert!(result.mutated_allocations().is_empty());

        // The same reached capture remains an entering boundary allocation when partial discharge preserves it. It
        // leaves the state-threaded and published sets empty because it crosses as its destination reference instead.
        let preserved_context = ListDischargeContext::new(ListDestination::new());
        let preserved = preserved_context
            .bind_preserved(
                ReferenceType::new(ListType { length: 2 }),
                ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })),
            )
            .unwrap();
        let preserved_allocation =
            preserved.try_as_reference("the preserved captured allocation").unwrap().allocation_id();
        let preserved_context = preserved_context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(preserved_allocation)],
        ));
        let preserved_summary = preserved_context
            .region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                program.entry_region_ref(),
                &[None],
            )
            .unwrap();
        let widening = preserved_context.boundary_widening(&preserved_summary, &BTreeSet::new()).unwrap();
        assert_eq!(widening.threaded(), &BTreeSet::new());
        assert_eq!(widening.entering(), &[preserved_allocation]);
        assert_eq!(widening.published(), &[]);
    }
}
