use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::rc::Rc;

// TODO(eaplatanios): Review this module.

use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

use super::results::ReferenceSource;

/// One caller-selectable reference site for partial reference discharge.
///
/// Selection needs an identity that exists in the *source* program, before any replay begins, so it cannot reuse the
/// environment's [`ReferenceAllocationHandle`](crate::programs::references::ReferenceAllocationHandle)s. In particular, a nested
/// region's formal reference input is invocation-parameterized — the region may be invoked from several call sites —
/// so it names no single caller-owned reference and is deliberately not selectable. Sites resolve internally to
/// allocations once discharge starts.
///
/// Sites are arena-relative in exactly the sense that every other reference artifact is: their coordinates are
/// meaningful only against the program they were enumerated from. Site validation rejects every kind mismatch, and
/// the arena-relativity contract carries the rest, because a coordinate taken from a different arena that happens to
/// name a valid allocation here is indistinguishable in principle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ReferenceDischargeSite {
    /// Entry-boundary allocation supplied by the caller as a lifted capture or a public reference argument.
    External(ReferenceSource),

    /// Interior allocation site, identified by the allocating instruction and the output position that defines the
    /// fresh allocation.
    Allocation {
        /// Allocating instruction.
        instruction: InstructionId,

        /// Output position defining the fresh allocation.
        output_index: usize,
    },
}

// Sites exist to be named in diagnostics, so the rendering backticks the arena coordinate it embeds. That keeps every
// message that interpolates a whole site consistent with the reference-site diagnostics, which backtick coordinates.
impl Display for ReferenceDischargeSite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External(source) => write!(formatter, "external {source}"),
            Self::Allocation { instruction, output_index } => {
                write!(formatter, "allocation at `{instruction}` output {output_index}")
            }
        }
    }
}

/// Reference sites one discharge normalizes into immutable state, with every unselected allocation preserved.
///
/// Selecting everything is deliberately a state of its own rather than a set listing every site. A program's sites are
/// enumerated from its own arena while a selection is caller-supplied, so full discharge — which is exactly the
/// everything-selected case of the one rewrite — must be expressible without naming anything, and an allocation that
/// no site *can* name, such as one bound directly rather than replayed, must still be discharged by it.
#[derive(Clone, Debug)]
pub(super) struct ReferenceDischargeSelection {
    /// Selected sites, or [`None`] when every site is selected.
    sites: Option<Rc<BTreeSet<ReferenceDischargeSite>>>,
}

impl ReferenceDischargeSelection {
    /// Returns the selection full discharge runs under, which selects every site.
    #[inline]
    pub(super) const fn everything() -> Self {
        Self { sites: None }
    }

    /// Returns the selection naming exactly `sites`, which preserves every allocation they do not name.
    #[inline]
    pub(super) fn from_sites(sites: &[ReferenceDischargeSite]) -> Self {
        Self { sites: Some(Rc::new(sites.iter().copied().collect())) }
    }

    /// Returns whether `site` is selected for discharge.
    #[inline]
    pub(super) fn selects(&self, site: ReferenceDischargeSite) -> bool {
        self.sites.as_ref().is_none_or(|sites| sites.contains(&site))
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Returns every [`ReferenceDischargeSite`] this program exposes to partial reference discharge, in canonical
    /// order: the entry-boundary externals in boundary order, followed by the interior allocations ordered by their
    /// arena coordinates.
    ///
    /// This is a deliberately lightweight query. It reads only the entry boundary types and the generic
    /// [`Operation::reference_semantics`] hook over the attached region closure, so it does not run the discharge
    /// rewrite or construct its environments, and callers can enumerate selectable sites without paying for either.
    /// Allocations inside nested regions are included because every allocating instruction defines a concrete local
    /// reference wherever it occurs.
    ///
    /// One class of enumerated site is inert: an allocation inside a closure that no operation ever replays, such as
    /// the dormant derivative rule region of a `custom_jvp`. Discharge rejects such a program outright, whichever way
    /// the site is selected, because how a reference boundary widens there has no defined meaning. The enumeration
    /// reports the site anyway rather than second-guessing the region roles, so that it stays a structural query.
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
    pub fn reference_discharge_sites(&self, capture_count: usize) -> Result<Vec<ReferenceDischargeSite>, ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge site enumeration requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut sites = input_ids
            .iter()
            .enumerate()
            .filter(|(_, input)| entry.atoms()[input.index()].r#type().is_reference())
            .map(|(input_index, _)| {
                ReferenceDischargeSite::External(ReferenceSource::from_flat_input_index(input_index, capture_count))
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
                    .map(move |output_index| ReferenceDischargeSite::Allocation {
                        instruction: instruction_id,
                        output_index,
                    })
            })
            .collect::<Vec<_>>();

        // Closure traversal visits regions in an unspecified order, so allocation coordinates are sorted to make the
        // enumeration reproducible for callers that persist or compare selections.
        allocations.sort_unstable();
        sites.append(&mut allocations);
        Ok(sites)
    }

    /// Validates a caller-provided partial reference discharge selection against this program.
    ///
    /// Every named site must exist in this program, must name a reference-typed entry position or a genuine
    /// reference-allocating output, and must appear at most once. Duplication is checked across the complete selection
    /// first, because a repeated site is ambiguous whatever it names.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    ///   - `sites`: Sites selected for discharge, in caller-chosen order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the offending site when a site is duplicated, names an
    /// out-of-range or non-reference entry position, names an instruction that this program does not contain, names
    /// an operation that defines no reference allocation, or names an output position of an allocating operation that
    /// is not itself an allocation.
    pub(crate) fn validate_reference_discharge_sites(
        &self,
        capture_count: usize,
        sites: &[ReferenceDischargeSite],
    ) -> Result<(), ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge selection requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut seen = HashSet::with_capacity(sites.len());
        for site in sites {
            if !seen.insert(*site) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge selection names {site} more than once",
                )));
            }
        }

        // Only the named instructions are resolved, so validating a small selection does not pay for the reference
        // semantics of every instruction in the closure.
        let instructions = entry.instructions_in_closure().collect::<HashMap<_, _>>();
        for site in sites {
            let invalid_site = || {
                ProgramError::MalformedProgram(format!(
                    "reference discharge selection names {site}, which is not a selectable site in this program",
                ))
            };
            match site {
                ReferenceDischargeSite::External(source) => {
                    let input_index = source.flat_input_index(capture_count).map_err(|_| invalid_site())?;
                    let input = input_ids.get(input_index).ok_or_else(invalid_site)?;
                    if !entry.atoms()[input.index()].r#type().is_reference() {
                        return Err(invalid_site());
                    }
                }
                ReferenceDischargeSite::Allocation { instruction, output_index } => {
                    let instruction = instructions.get(instruction).ok_or_else(invalid_site)?;
                    let operation = instruction.operation();
                    let output_indices =
                        operation.reference_semantics().allocation_output_indices().collect::<Vec<_>>();
                    if !output_indices.contains(output_index) {
                        return Err(invalid_site());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;

    use crate::programs::instructions::InstructionId;

    use crate::programs::references::discharge::tests::*;

    use super::*;

    #[test]
    fn test_reference_discharge_sites_enumerate_externals_before_allocations() {
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
        let captured = builder.add_input(reference_type(0));
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let local = builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap();
        let called = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];

        // The same callee region is attached twice, so its interior allocation must be enumerated once rather than
        // once per invocation.
        let repeated = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![called, repeated, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Externals come first in entry-boundary order, split at the capture prefix, and the allocations follow in
        // arena-coordinate order, including the one inside the shared callee region.
        assert_eq!(
            program.reference_discharge_sites(1),
            Ok(vec![
                ReferenceDischargeSite::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::Allocation { instruction: InstructionId::new(callee, 0), output_index: 0 },
                ReferenceDischargeSite::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            ]),
        );

        // The capture prefix is the only thing that moves when it changes, and an oversized prefix is rejected.
        assert_eq!(
            program.reference_discharge_sites(0).unwrap()[..2],
            [
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        assert_eq!(
            program.reference_discharge_sites(4),
            Err(ProgramError::MalformedProgram(
                "reference discharge site enumeration requests 4 captures but the program has 3 inputs".to_string(),
            )),
        );

        // Every enumerated site validates, and sites render with their kind so diagnostics stay unambiguous.
        let sites = program.reference_discharge_sites(1).unwrap();
        assert_eq!(program.validate_reference_discharge_sites(1, sites.as_slice()), Ok(()));
        assert_eq!(sites[0].to_string(), "external capture 0");
        assert_eq!(sites[3].to_string(), "allocation at `^1[0]` output 0");
    }

    #[test]
    fn test_reference_discharge_site_validation_rejects_malformed_selections() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let allocation =
            builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        let read = builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let entry = program.entry();
        let allocation =
            ReferenceDischargeSite::Allocation { instruction: InstructionId::new(entry, 0), output_index: 0 };
        let reject = |sites: &[ReferenceDischargeSite]| {
            let ProgramError::MalformedProgram(message) =
                program.validate_reference_discharge_sites(0, sites).unwrap_err()
            else {
                panic!("reference discharge site validation must report a malformed program");
            };
            message
        };

        // A repeated site is rejected before any kind check, because a duplicate selection is ambiguous whatever it
        // names.
        assert_eq!(
            reject(&[allocation, allocation]),
            "reference discharge selection names allocation at `^0[0]` output 0 more than once",
        );

        // Every invalid site uses one deterministic diagnostic; validity is defined by the canonical enumerated set.
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Capture { index: 0 })]),
            "reference discharge selection names external capture 0, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: 2 })]),
            "reference discharge selection names external input 2, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 })]),
            "reference discharge selection names external input 1, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: usize::MAX })]),
            format!(
                "reference discharge selection names external input {}, which is not a selectable site in this \
                 program",
                usize::MAX,
            ),
        );

        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 7),
                output_index: 0,
            }]),
            "reference discharge selection names allocation at `^0[7]` output 0, which is not a selectable site in \
             this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 1),
                output_index: 0,
            }]),
            "reference discharge selection names allocation at `^0[1]` output 0, which is not a selectable site in \
             this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 0),
                output_index: 1,
            }]),
            "reference discharge selection names allocation at `^0[0]` output 1, which is not a selectable site in \
             this program",
        );
    }
}
