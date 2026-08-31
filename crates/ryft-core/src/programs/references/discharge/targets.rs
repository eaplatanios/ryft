use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::rc::Rc;

use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::results::ReferenceSource;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

/// A caller-selectable [`Reference`](crate::Reference) target for partial reference discharge. A target needs an
/// identity that exists in the _source_ [`Program`], before any replay begins, so it cannot reuse the environment's
/// [`ReferenceDischargeAllocationId`](crate::ReferenceDischargeAllocationId)s. In particular, a nested region's formal
/// reference input is invocation-parameterized (the region may be invoked from several call sites) and so it names no
/// single caller-owned reference and is deliberately not selectable. Targets resolve internally to allocations once
/// discharge starts.
///
/// Targets are arena-relative in exactly the sense that every other reference artifact is (i.e., their coordinates are
/// meaningful only against the program they were enumerated from). Target validation rejects every kind mismatch, and
/// the arena-relativity contract carries the rest because a coordinate taken from a different arena that happens to
/// name a valid allocation here is indistinguishable in principle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceDischargeTarget {
    /// Entry-boundary allocation supplied by the caller as a lifted capture or a public reference argument.
    External(ReferenceSource),

    /// Interior allocation target, identified by the allocating [`Instruction`](crate::Instruction) and the output
    /// position that defines the fresh allocation.
    Internal {
        /// Allocating [`Instruction`](crate::Instruction).
        instruction: InstructionId,

        /// Output position defining the fresh allocation.
        output_index: usize,
    },
}

impl Display for ReferenceDischargeTarget {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External(source) => write!(formatter, "external {source}"),
            Self::Internal { instruction, output_index } => {
                write!(formatter, "internal allocation at `{instruction}` output {output_index}")
            }
        }
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Reference targets one discharge normalizes into immutable state, with every unselected allocation preserved.
///
/// Selecting everything is deliberately a state of its own rather than a set listing every target. A program's targets
/// are enumerated from its own arena while the requested targets are caller-supplied, so full discharge — exactly the
/// everything-selected case of the one rewrite — must be expressible without naming anything, and an allocation that
/// no target *can* name, such as one bound directly rather than replayed, must still be discharged by it.
#[derive(Clone, Debug)]
pub(super) struct ReferenceDischargeTargets {
    /// Selected targets, or [`None`] when every target is selected.
    targets: Option<Rc<BTreeSet<ReferenceDischargeTarget>>>,
}

impl ReferenceDischargeTargets {
    /// Returns the targets full discharge runs under, which select every reference.
    pub(super) const fn everything() -> Self {
        Self { targets: None }
    }

    /// Returns exactly `targets`, preserving every allocation they do not name.
    #[inline]
    pub(super) fn from_targets(targets: &[ReferenceDischargeTarget]) -> Self {
        Self { targets: Some(Rc::new(targets.iter().copied().collect())) }
    }

    /// Returns whether `target` is selected for discharge.
    #[inline]
    pub(super) fn selects(&self, target: ReferenceDischargeTarget) -> bool {
        self.targets.as_ref().is_none_or(|targets| targets.contains(&target))
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Returns every [`ReferenceDischargeTarget`] this program exposes to partial reference discharge, in canonical
    /// order: the entry-boundary externals in boundary order, followed by the interior allocations ordered by their
    /// arena coordinates.
    ///
    /// This is a deliberately lightweight query. It reads only the entry boundary types and the generic
    /// [`Operation::reference_semantics`] hook over the attached region closure, so it does not run the discharge
    /// rewrite or construct its environments, and callers can enumerate selectable targets without paying for either.
    /// Allocations inside nested regions are included because every allocating instruction defines a concrete local
    /// reference wherever it occurs.
    ///
    /// One class of enumerated target is inert: an allocation inside a closure that no operation ever replays, such as
    /// the dormant derivative rule region of a `custom_jvp`. Discharge rejects such a program outright, whichever way
    /// the target is selected, because how a reference boundary widens there has no defined meaning. The enumeration
    /// reports the target anyway rather than second-guessing the region roles, so that it stays a structural query.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table, used to
    ///     split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the program's input count.
    pub fn reference_discharge_targets(
        &self,
        capture_count: usize,
    ) -> Result<Vec<ReferenceDischargeTarget>, ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge target enumeration requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut targets = input_ids
            .iter()
            .enumerate()
            .filter(|(_, input)| entry.atoms()[input.index()].r#type().is_reference())
            .map(|(input_index, _)| {
                ReferenceDischargeTarget::External(ReferenceSource::from_flat_input_index(input_index, capture_count))
            })
            .collect::<Vec<_>>();
        let mut allocations = entry
            .instructions_in_closure()
            .flat_map(|(instruction_id, instruction)| {
                instruction
                    .operation()
                    .reference_semantics()
                    .allocation_output_indices()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(move |output_index| ReferenceDischargeTarget::Internal {
                        instruction: instruction_id,
                        output_index,
                    })
            })
            .collect::<Vec<_>>();

        // Closure traversal visits regions in an unspecified order, so allocation coordinates are sorted to make the
        // enumeration reproducible for callers that persist or compare target sets.
        allocations.sort_unstable();
        targets.append(&mut allocations);
        Ok(targets)
    }

    /// Validates caller-provided partial reference discharge targets against this program.
    ///
    /// Every named target must exist in this program, must name a reference-typed entry position or a genuine
    /// reference-allocating output, and must appear at most once. Duplication is checked across the complete target set
    /// first, because a repeated target is ambiguous whatever it names.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    ///   - `targets`: Targets selected for discharge, in caller-chosen order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the offending target when a target is duplicated, names an
    /// out-of-range or non-reference entry position, names an instruction that this program does not contain, names
    /// an operation that defines no reference allocation, or names an output position of an allocating operation that
    /// is not itself an allocation.
    pub(crate) fn validate_reference_discharge_targets(
        &self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<(), ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge target validation requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut seen = HashSet::with_capacity(targets.len());
        for target in targets {
            if !seen.insert(*target) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge targets contain {target} more than once",
                )));
            }
        }

        // Only the named instructions are resolved, so validating a small target set does not pay for the reference
        // semantics of every instruction in the closure.
        let instructions = entry.instructions_in_closure().collect::<HashMap<_, _>>();
        for target in targets {
            let invalid_target = || {
                ProgramError::MalformedProgram(format!(
                    "reference discharge targets include {target}, which is not selectable in this program",
                ))
            };
            match target {
                ReferenceDischargeTarget::External(source) => {
                    let input_index = source.flat_input_index(capture_count).map_err(|_| invalid_target())?;
                    let input = input_ids.get(input_index).ok_or_else(invalid_target)?;
                    if !entry.atoms()[input.index()].r#type().is_reference() {
                        return Err(invalid_target());
                    }
                }
                ReferenceDischargeTarget::Internal { instruction, output_index } => {
                    let instruction = instructions.get(instruction).ok_or_else(invalid_target)?;
                    let operation = instruction.operation();
                    let output_indices =
                        operation.reference_semantics().allocation_output_indices().collect::<Vec<_>>();
                    if !output_indices.contains(output_index) {
                        return Err(invalid_target());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::instructions::InstructionId;
    use crate::programs::references::discharge::tests::{
        TestOperation, TestType, TestValue, boundary_program, reference_type,
    };

    use super::*;

    fn target_validation_program() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let allocation =
            builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        let read = builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    #[test]
    fn test_reference_discharge_target_ordering_hashing_and_rendering() {
        let program = boundary_program(0, 0);
        let entry = program.entry();
        let external_capture = ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 });
        let external_input = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let first_internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 0 };
        let second_output =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 1 };
        let second_internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 1), output_index: 0 };

        assert_eq!(external_capture, external_capture);
        assert_ne!(external_capture, external_input);
        assert!(external_capture < external_input);
        assert!(external_input < first_internal);
        assert!(first_internal < second_output);
        assert!(second_output < second_internal);

        let targets = HashSet::from([external_capture, first_internal]);
        assert!(targets.contains(&external_capture));
        assert!(targets.contains(&first_internal));
        assert!(!targets.contains(&external_input));
        assert_eq!(external_capture.to_string(), "external capture 0");
        assert_eq!(first_internal.to_string(), "internal allocation at `^0[0]` output 0");
        assert_eq!(format!("{external_capture:?}"), "External(Capture { index: 0 })");
    }

    #[test]
    fn test_reference_discharge_targets_select_everything_or_only_requested_targets() {
        let program = boundary_program(0, 0);
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        let everything = ReferenceDischargeTargets::everything();
        assert!(everything.selects(external));
        assert!(everything.selects(internal));

        let selected = ReferenceDischargeTargets::from_targets(&[external, external]);
        assert_eq!(selected.targets.as_ref().unwrap().len(), 1);
        assert!(selected.selects(external));
        assert!(!selected.selects(internal));
        let cloned = selected.clone();
        assert!(cloned.selects(external));
        assert!(!cloned.selects(internal));

        let empty = ReferenceDischargeTargets::from_targets(&[]);
        assert!(!empty.selects(external));
        assert!(!empty.selects(internal));
    }

    #[test]
    fn test_program_reference_discharge_targets_classifies_external_sources() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type(0));
        builder.add_input(reference_type(1));
        let ordinary = builder.add_input(TestType::Value(0));
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![ordinary], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.reference_discharge_targets(0),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
            ]),
        );
        assert_eq!(
            program.reference_discharge_targets(1),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
            ]),
        );
        assert_eq!(
            program.reference_discharge_targets(2),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 1 }),
            ]),
        );
        assert_eq!(program.reference_discharge_targets(3), program.reference_discharge_targets(2));
    }

    #[test]
    fn test_program_reference_discharge_targets_enumerates_nested_allocations_once_in_canonical_order() {
        // A callee region that allocates its own local allocation, so that enumeration is exercised across the complete
        // attached region closure rather than the entry region alone.
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = callee_builder.add_input(TestType::Value(0));
        let allocation = callee_builder
            .add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None)
            .unwrap()[0];
        let frozen =
            callee_builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let local = builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        let called = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];

        // The same callee region is attached twice, so its interior allocation must be enumerated once rather than
        // once per invocation.
        let repeated = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![called, repeated, frozen],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Closure traversal order is unspecified, so internal targets must be sorted by arena coordinate. Importing
        // the same region at two call sites must not duplicate the allocation it contains.
        assert_eq!(
            program.reference_discharge_targets(0),
            Ok(vec![
                ReferenceDischargeTarget::Internal { instruction: InstructionId::new(callee, 0), output_index: 0 },
                ReferenceDischargeTarget::Internal {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            ]),
        );
    }

    #[test]
    fn test_program_reference_discharge_targets_returns_an_empty_set_for_a_reference_free_program() {
        assert_eq!(boundary_program(2, 1).reference_discharge_targets(1), Ok(Vec::new()));
    }

    #[test]
    fn test_program_reference_discharge_targets_rejects_an_oversized_capture_prefix() {
        assert_eq!(
            boundary_program(3, 0).reference_discharge_targets(4),
            Err(ProgramError::MalformedProgram(
                "reference discharge target enumeration requests 4 captures but the program has 3 inputs".to_string(),
            )),
        );
    }

    #[test]
    fn test_program_validate_reference_discharge_targets_accepts_empty_and_reordered_valid_sets() {
        let program = target_validation_program();
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        assert_eq!(program.validate_reference_discharge_targets(0, &[]), Ok(()));
        assert_eq!(program.validate_reference_discharge_targets(0, &[internal, external]), Ok(()));
    }

    #[test]
    fn test_program_validate_reference_discharge_targets_rejects_invalid_set_shape() {
        let program = target_validation_program();
        let entry = program.entry();
        let allocation =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 0 };

        assert_eq!(
            program.validate_reference_discharge_targets(3, &[]),
            Err(ProgramError::MalformedProgram(
                "reference discharge target validation requests 3 captures but the program has 2 inputs".to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(0, &[allocation, allocation]),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets contain internal allocation at `^0[0]` output 0 more than once"
                    .to_string(),
            )),
        );

        // Duplicate detection runs before kind validation because repetition is ambiguous regardless of what is named.
        let invalid = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 7 });
        assert_eq!(
            program.validate_reference_discharge_targets(0, &[invalid, invalid]),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets contain external input 7 more than once".to_string(),
            )),
        );
    }

    #[test]
    fn test_program_validate_reference_discharge_targets_rejects_invalid_external_targets() {
        let program = target_validation_program();

        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include external capture 0, which is not selectable in this program"
                    .to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 2 })],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include external input 2, which is not selectable in this program"
                    .to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 })],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include external input 1, which is not selectable in this program"
                    .to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: usize::MAX })],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge targets include external input {}, which is not selectable in this program",
                usize::MAX,
            ))),
        );
    }

    #[test]
    fn test_program_validate_reference_discharge_targets_rejects_invalid_internal_targets() {
        let program = target_validation_program();
        let entry = program.entry();
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 7), output_index: 0 }],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[7]` output 0, which is not selectable \
                 in this program"
                    .to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 1), output_index: 0 }],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[1]` output 0, which is not selectable \
                 in this program"
                    .to_string(),
            )),
        );
        assert_eq!(
            program.validate_reference_discharge_targets(
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 1 }],
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[0]` output 1, which is not selectable \
                 in this program"
                    .to_string(),
            )),
        );
    }
}
