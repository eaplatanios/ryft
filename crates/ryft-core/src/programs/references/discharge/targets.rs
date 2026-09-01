use std::collections::{BTreeSet, HashMap};
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
/// Targets are arena-relative in exactly the sense that every other reference artifact is: their instruction and
/// boundary identifiers are meaningful only in the program arena from which they were enumerated. Target validation
/// rejects every kind mismatch, and the arena-relativity contract carries the rest because a target from a different
/// arena that happens to name a valid allocation here is indistinguishable in principle.
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

/// Selection of reference allocations to discharge (i.e., a collection of [`ReferenceDischargeTarget`]s). A partial
/// discharge stores the set of [`ReferenceDischargeTarget`]s selected by the caller and preserves every other
/// allocation. A full discharge instead uses a distinct "everything" state because it must also discharge allocations
/// that callers cannot name with a target, such as allocations bound directly during replay.
#[derive(Clone, Debug)]
pub struct ReferenceDischargeTargets {
    /// Selected targets, or [`None`] when every target is selected.
    targets: Option<Rc<BTreeSet<ReferenceDischargeTarget>>>,
}

impl ReferenceDischargeTargets {
    /// Returns the [`ReferenceDischargeTargets`] that full discharge runs under, which select every reference.
    pub const fn everything() -> Self {
        Self { targets: None }
    }

    /// Validates `targets` against `program` and returns a selection that discharges exactly those targets. Every
    /// target must name a reference-typed entry position or a reference allocation defined by an instruction in
    /// `program`, and each target may appear only once. Every reference not named by the resulting selection is
    /// preserved.
    ///
    /// # Parameters
    ///
    ///   - `program`: Program whose reference allocations the targets select.
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    ///   - `targets`: Targets selected for discharge, in caller-chosen order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the offending target when a target is duplicated, names an
    /// out-of-range or non-reference entry position, names an instruction that the program does not contain, names an
    /// operation that defines no reference allocation, or names an output position that is not an allocation. It also
    /// returns an error when `capture_count` exceeds the program's input count.
    pub fn from_targets<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>(
        program: &Program<V, O, Input, Output>,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<Self, ProgramError> {
        let entry = program.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge target validation requests {} captures but the program has {} inputs",
                capture_count,
                input_ids.len(),
            )));
        }

        let mut selected = BTreeSet::new();
        for target in targets {
            if !selected.insert(*target) {
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
                    let output_indices =
                        instruction.operation().reference_semantics().allocation_output_indices().collect::<Vec<_>>();
                    if !output_indices.contains(output_index) {
                        return Err(invalid_target());
                    }
                }
            }
        }

        Ok(Self { targets: Some(Rc::new(selected)) })
    }

    /// Returns `true` if `target` is selected for discharge by this [`ReferenceDischargeTargets`] set.
    #[inline]
    pub fn selects(&self, target: ReferenceDischargeTarget) -> bool {
        self.targets.as_ref().is_none_or(|targets| targets.contains(&target))
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Returns every [`ReferenceDischargeTarget`] that this [`Program`] exposes to partial reference discharge, in a
    /// canonical ordering with the entry-boundary externals in boundary order, followed by the interior allocations
    /// ordered by instruction and output position. This is a deliberately lightweight query. It reads only the entry
    /// boundary types and the generic [`Operation::reference_semantics`] over the attached [`Region`](crate::Region)
    /// closure, so it does not run the discharge rewrite or construct its environments, and callers can enumerate
    /// selectable targets without paying for either. Allocations inside nested regions are included because every
    /// allocating [`Instruction`](crate::Instruction) defines a concrete local reference wherever it occurs.
    ///
    /// One class of enumerated targets is inert: an allocation inside a closure that no operation ever replays, such
    /// as the dormant derivative rule region of a [`CustomJvpOperation`](crate::CustomJvpOperation). Discharge rejects
    /// such a program outright, whichever way the target is selected, because how a reference boundary widens there has
    /// no defined meaning. The enumeration reports the target anyway rather than second-guessing the region roles, so
    /// that it stays a structural query.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table, used to
    ///     split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
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
                "reference discharge target enumeration requests {} captures but the program has {} inputs",
                capture_count,
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

        // Closure traversal visits regions in an unspecified order, so internal targets are sorted by instruction and
        // output position to make the enumeration reproducible for callers that persist or compare target sets.
        allocations.sort_unstable();
        targets.append(&mut allocations);
        Ok(targets)
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

    fn test_target_validation_program() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
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
        let program = test_target_validation_program();
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        let everything = ReferenceDischargeTargets::everything();
        assert!(everything.selects(external));
        assert!(everything.selects(internal));

        let selected = ReferenceDischargeTargets::from_targets(&program, 0, &[external]).unwrap();
        assert_eq!(selected.targets.as_ref().unwrap().len(), 1);
        assert!(selected.selects(external));
        assert!(!selected.selects(internal));
        let cloned = selected.clone();
        assert!(cloned.selects(external));
        assert!(!cloned.selects(internal));

        let empty = ReferenceDischargeTargets::from_targets(&program, 0, &[]).unwrap();
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

        // Closure traversal order is unspecified, so internal targets must be sorted by instruction and output
        // position. Importing the same region at two call sites must not duplicate the allocation it contains.
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
    fn test_reference_discharge_targets_construction_accepts_empty_and_reordered_valid_sets() {
        let program = test_target_validation_program();
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        let empty = ReferenceDischargeTargets::from_targets(&program, 0, &[]).unwrap();
        assert_eq!(empty.targets.as_ref().unwrap().len(), 0);

        let reordered = ReferenceDischargeTargets::from_targets(&program, 0, &[internal, external]).unwrap();
        assert_eq!(reordered.targets.as_ref().unwrap().len(), 2);
        assert!(reordered.selects(external));
        assert!(reordered.selects(internal));
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_set_shape() {
        let program = test_target_validation_program();
        let entry = program.entry();
        let allocation =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 0 };

        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 3, &[]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge target validation requests 3 captures but the program has 2 inputs".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 0, &[allocation, allocation]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets contain internal allocation at `^0[0]` output 0 more than once"
                    .to_string(),
            ),
        );

        // Duplicate detection runs before kind validation because repetition is ambiguous regardless of what is named.
        let invalid = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 7 });
        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 0, &[invalid, invalid]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets contain external input 7 more than once".to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_external_targets() {
        let program = test_target_validation_program();

        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external capture 0, which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 2 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 2, which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 1, which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: usize::MAX })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge targets include external input {}, which is not selectable in this program",
                usize::MAX,
            )),
        );
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_internal_targets() {
        let program = test_target_validation_program();
        let entry = program.entry();
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 7), output_index: 0 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[7]` output 0, which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 1), output_index: 0 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[1]` output 0, which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 1 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[0]` output 1, which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
    }
}
