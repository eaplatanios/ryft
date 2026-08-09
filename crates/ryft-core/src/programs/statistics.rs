//! Contains backend-neutral structural statistics for [`Program`]s.
//!
//! [`Program::statistics`] is the structural-inspection entry point for programs. It reports stored program structure
//! rather than an execution trace: dormant rule regions are included, dead instructions count toward instruction and
//! operation counts, and no simplification is applied. The result is a [`ProgramStatistics`] value holding one
//! [`RegionStatistics`] node per region-arena entry, in ascending [`RegionId`](crate::programs::regions::RegionId)
//! order, together with [`AttachedRegionStatistics`] edges that record which instructions attach which regions. A
//! region shared by several instructions (or by several slots of one instruction) therefore appears exactly once as a
//! node and once per use as an edge, so the region graph is never expanded into a tree and no count is inflated by
//! sharing.

// TODO(eaplatanios): Review this module.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::parameters::Parameterized;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::RegionRole;
use crate::programs::values::Value;

/// Structural statistics of one [`Program`], as computed by [`Program::statistics`].
///
/// The `regions` graph mirrors the program's validated region arena exactly: statistics appear in ascending
/// [`RegionId`](crate::programs::regions::RegionId) order, descendant regions always precede the regions that attach
/// them, and the final entry is always the program's entry region. Shared regions appear once in `regions` while
/// appearing in multiple [`AttachedRegionStatistics`] edge lists, and edges reference their target nodes by index
/// into `regions`.
///
/// Aggregate accessors such as [`total_instruction_count`](Self::total_instruction_count) are derived on demand and
/// are not part of the serialized form. The serialized JSON contains exactly one field, `regions`, and consumers
/// recover the entry region as the last element of that array.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProgramStatistics {
    /// Per-region statistics in ascending region-arena order, with the entry region last.
    regions: Vec<RegionStatistics>,
}

impl ProgramStatistics {
    /// Returns the per-region statistics in ascending region-arena order, with the entry region last.
    #[inline]
    pub fn regions(&self) -> &[RegionStatistics] {
        &self.regions
    }

    /// Returns the number of unique regions in the program's region arena.
    #[inline]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Returns the index of the entry region inside [`regions`](Self::regions). [`Program`] validation guarantees
    /// that the entry region is the final region-arena entry, so this is always `region_count() - 1`.
    #[inline]
    pub fn entry_region_index(&self) -> usize {
        self.regions.len() - 1
    }

    /// Returns the statistics of the program's entry region. A valid [`Program`] always contains at least its entry
    /// region, so this cannot panic.
    #[inline]
    pub fn entry(&self) -> &RegionStatistics {
        self.regions.last().unwrap()
    }

    /// Returns the total number of instructions across all regions. A shared region contributes its instructions
    /// once, regardless of how many times it is attached.
    #[inline]
    pub fn total_instruction_count(&self) -> usize {
        self.regions.iter().map(RegionStatistics::instruction_count).sum()
    }

    /// Returns the total number of constant atoms across all regions. A shared region contributes its constants
    /// once, regardless of how many times it is attached.
    #[inline]
    pub fn total_constant_count(&self) -> usize {
        self.regions.iter().map(RegionStatistics::constant_count).sum()
    }

    /// Returns the per-operation instruction counts merged across all regions, keyed by exact
    /// [`Operation::name`] strings. A shared region contributes its instructions once, regardless of how many times
    /// it is attached.
    pub fn total_operation_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut totals = BTreeMap::new();
        for region in &self.regions {
            for (name, count) in &region.operation_counts {
                *totals.entry(*name).or_insert(0) += count;
            }
        }
        totals
    }
}

/// Structural statistics of one region in a [`Program`]'s region arena. All statistics are region-local: instructions
/// and constants of attached regions are reported by those regions' own [`RegionStatistics`] nodes, and attached
/// regions contribute nothing to [`maximum_output_dependency_depth`](Self::maximum_output_dependency_depth).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RegionStatistics {
    /// Number of input atoms in the region's boundary.
    input_count: usize,

    /// Number of output atoms in the region's boundary.
    output_count: usize,

    /// Number of instructions in the region, including instructions whose outputs reach no region output.
    instruction_count: usize,

    /// Number of constant atoms stored in the region's atom table.
    constant_count: usize,

    /// Per-operation instruction counts, keyed by exact [`Operation::name`] strings.
    operation_counts: BTreeMap<&'static str, usize>,

    /// Maximum data-dependency depth over the region's outputs. Inputs and constants have depth zero, an instruction
    /// output has depth one plus the maximum depth of the instruction's inputs (so the outputs of a zero-input
    /// instruction have depth one), and the maximum is taken over region outputs only, so deeper dead work is
    /// excluded. A region without outputs, and a region output that is an input or constant atom, both yield depth
    /// zero. Attached regions contribute nothing: a `while` instruction is one step regardless of its body's depth.
    maximum_output_dependency_depth: usize,

    /// Attached-region edges recorded at their use sites, ordered by instruction index and then slot order.
    attached_regions: Vec<AttachedRegionStatistics>,
}

impl RegionStatistics {
    /// Returns the number of input atoms in the region's boundary.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Returns the number of output atoms in the region's boundary.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the number of instructions in the region, including instructions whose outputs reach no region output.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    /// Returns the number of constant atoms stored in the region's atom table.
    #[inline]
    pub fn constant_count(&self) -> usize {
        self.constant_count
    }

    /// Returns the per-operation instruction counts, keyed by exact [`Operation::name`] strings.
    #[inline]
    pub fn operation_counts(&self) -> &BTreeMap<&'static str, usize> {
        &self.operation_counts
    }

    /// Returns the maximum data-dependency depth over the region's outputs. Refer to the documentation of the
    /// corresponding field for the precise definition.
    #[inline]
    pub fn maximum_output_dependency_depth(&self) -> usize {
        self.maximum_output_dependency_depth
    }

    /// Returns the attached-region edges recorded at their use sites, ordered by instruction index and then slot
    /// order.
    #[inline]
    pub fn attached_regions(&self) -> &[AttachedRegionStatistics] {
        &self.attached_regions
    }
}

/// One attached-region edge in a [`RegionStatistics`] node: the use site at which an instruction attaches a region.
/// Edges are recorded per use, so a region attached several times — by several instructions or by several slots of
/// one instruction — produces several edges referencing the same [`region_index`](Self::region_index). Edge lists are
/// ordered rather than keyed by label, because slot-name uniqueness is a convention rather than an enforced
/// invariant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AttachedRegionStatistics {
    /// Index of the attaching instruction within its region's instruction list.
    instruction_index: usize,

    /// Exact [`Operation::name`] of the attaching instruction's operation.
    operation: &'static str,

    /// Name of the [`RegionSlot`](crate::programs::regions::RegionSlot) through which the region is attached.
    region_slot: &'static str,

    /// Semantic [`RegionRole`] of the slot through which the region is attached.
    region_role: RegionRole,

    /// Index of the attached region's [`RegionStatistics`] node inside [`ProgramStatistics::regions`].
    region_index: usize,
}

impl AttachedRegionStatistics {
    /// Returns the index of the attaching instruction within its region's instruction list.
    #[inline]
    pub fn instruction_index(&self) -> usize {
        self.instruction_index
    }

    /// Returns the exact [`Operation::name`] of the attaching instruction's operation.
    #[inline]
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the name of the [`RegionSlot`](crate::programs::regions::RegionSlot) through which the region is
    /// attached.
    #[inline]
    pub fn region_slot(&self) -> &'static str {
        self.region_slot
    }

    /// Returns the semantic [`RegionRole`] of the slot through which the region is attached.
    #[inline]
    pub fn region_role(&self) -> RegionRole {
        self.region_role
    }

    /// Returns the index of the attached region's [`RegionStatistics`] node inside [`ProgramStatistics::regions`].
    #[inline]
    pub fn region_index(&self) -> usize {
        self.region_index
    }

    /// Returns the fused `"{operation}.{region_slot}"` display label for this edge.
    #[inline]
    pub fn label(&self) -> String {
        format!("{}.{}", self.operation, self.region_slot)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Returns backend-neutral structural statistics for this [`Program`]. This is the structural-inspection entry
    /// point for programs: it reports stored program structure — including dormant rule regions and instructions
    /// whose outputs reach no region output — rather than an execution trace, and applies no simplification.
    ///
    /// The computation is infallible because [`Program`] construction already validated every atom and region
    /// reference. Refer to the documentation of [`ProgramStatistics`] and [`RegionStatistics`] for the precise
    /// semantics of the reported statistics.
    pub fn statistics(&self) -> ProgramStatistics {
        let regions = self
            .regions()
            .iter()
            .map(|region| {
                let mut operation_counts = BTreeMap::new();
                let mut attached_regions = Vec::new();
                // Inputs and constants keep the initial depth of zero; only instruction outputs are ever written.
                // The single forward pass relies on the validated topological instruction order that
                // `ProgramBuilder` establishes (i.e., producers precede consumers).
                let mut depth_by_atom = vec![0usize; region.atoms().len()];
                for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                    *operation_counts.entry(instruction.operation().name()).or_insert(0) += 1;
                    let input_depth =
                        instruction.inputs().iter().map(|input| depth_by_atom[input.index()]).max().unwrap_or(0);
                    for output in instruction.outputs().iter().copied() {
                        depth_by_atom[output.index()] = input_depth + 1;
                    }
                    // Region sealing guarantees one declared slot per attached region, so index directly.
                    let region_slots = instruction.operation().region_slots();
                    for (slot, attached) in instruction.regions().iter().copied().enumerate() {
                        attached_regions.push(AttachedRegionStatistics {
                            instruction_index,
                            operation: instruction.operation().name(),
                            region_slot: region_slots[slot].name,
                            region_role: region_slots[slot].role,
                            region_index: attached.index(),
                        });
                    }
                }
                RegionStatistics {
                    input_count: region.input_ids().len(),
                    output_count: region.output_ids().len(),
                    instruction_count: region.instructions().len(),
                    constant_count: region.atoms().iter().filter(|atom| atom.is_constant()).count(),
                    operation_counts,
                    maximum_output_dependency_depth: region
                        .output_ids()
                        .iter()
                        .map(|output| depth_by_atom[output.index()])
                        .max()
                        .unwrap_or(0),
                    attached_regions,
                }
            })
            .collect();
        ProgramStatistics { regions }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::DataType::F64;
    use crate::arrays::{Array, ArrayOperation, ArrayType};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::operations::constants::ConstantOperation;
    use crate::operations::{ADD_OPERATION_NAME, Sin};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::RegionSlot;
    use crate::tests::TestRegionOperation;

    use super::*;

    /// Builds a sealed single-input identity region and returns its program for importing into test builders.
    fn identity_region_program() -> Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Verifies exact entry-region statistics for a small traced scalar program.
    #[test]
    fn test_statistics_scalar_program_counts_and_depth() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (_, program): (Array, Program<Array, ArrayOperation<Array>, Array, Array>) = domain
            .interpret_and_trace(
                |x| {
                    let with_constant = x.clone() + x.context().constant(Array::scalar(1.0));
                    with_constant.sin()
                },
                Array::scalar(2.0),
            )
            .unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.region_count(), 1);
        assert_eq!(statistics.entry_region_index(), 0);
        assert_eq!(
            statistics.entry(),
            &RegionStatistics {
                input_count: 1,
                output_count: 1,
                instruction_count: 2,
                constant_count: 1,
                operation_counts: BTreeMap::from([(ADD_OPERATION_NAME, 1usize), ("sin", 1usize)]),
                maximum_output_dependency_depth: 2,
                attached_regions: Vec::new(),
            },
        );
    }

    /// Verifies that region outputs that are input or constant atoms have depth zero.
    #[test]
    fn test_statistics_input_and_constant_outputs_have_depth_zero() {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(F64));
        let constant = builder.add_constant(Array::scalar(1.0));
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![input, constant], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().input_count(), 1);
        assert_eq!(statistics.entry().output_count(), 2);
        assert_eq!(statistics.entry().instruction_count(), 0);
        assert_eq!(statistics.entry().constant_count(), 1);
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 0);
    }

    /// Verifies that the outputs of a used zero-input instruction have depth one.
    #[test]
    fn test_statistics_zero_input_instruction_output_has_depth_one() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let produced = builder.add_instruction(ConstantOperation::new(Array::scalar(1.0)), vec![], vec![]).unwrap()[0];
        let program: Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> =
            builder.build(vec![produced], vec![], vec![Placeholder]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().instruction_count(), 1);
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 1);
    }

    /// Verifies that a region without outputs has depth zero.
    #[test]
    fn test_statistics_zero_output_region_has_depth_zero() {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(F64));
        builder.add_instruction(TestRegionOperation::Effectful, vec![], vec![input]).unwrap();
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![], vec![Placeholder], vec![]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().output_count(), 0);
        assert_eq!(statistics.entry().instruction_count(), 1);
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 0);
    }

    /// Verifies that dead work deeper than every live output is excluded from the depth while still being counted in
    /// the instruction and operation counts.
    #[test]
    fn test_statistics_dead_work_is_excluded_from_depth_but_counted() {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(F64));
        let live = builder.add_instruction(TestRegionOperation::Add, vec![], vec![input, input]).unwrap()[0];
        let dead = builder.add_instruction(TestRegionOperation::Add, vec![], vec![live, live]).unwrap()[0];
        builder.add_instruction(TestRegionOperation::Add, vec![], vec![dead, dead]).unwrap();
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![live], vec![Placeholder], vec![Placeholder]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().instruction_count(), 3);
        assert_eq!(statistics.entry().operation_counts(), &BTreeMap::from([("add", 3usize)]));
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 1);
    }

    /// Verifies that duplicated region outputs affect only the output count and not the other statistics.
    #[test]
    fn test_statistics_duplicate_outputs_only_affect_output_count() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (_, program): ((Array, Array), Program<Array, ArrayOperation<Array>, Array, (Array, Array)>) = domain
            .interpret_and_trace(
                |x| {
                    let sine = x.sin()?;
                    Ok((sine.clone(), sine))
                },
                Array::scalar(2.0),
            )
            .unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().output_count(), 2);
        assert_eq!(statistics.entry().instruction_count(), 1);
        assert_eq!(statistics.entry().operation_counts(), &BTreeMap::from([("sin", 1usize)]));
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 1);
    }

    /// Verifies that a multi-output instruction assigns the same depth to every output.
    #[test]
    fn test_statistics_multi_output_instruction_outputs_share_depth() {
        // `TestRegionOperation::WithRegions` infers its outputs from the first attached region's output types, so a
        // two-output region yields a two-output instruction.
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(F64));
        let region_program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> = region_builder
            .build(vec![region_input, region_input], vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let body = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(F64));
        let outputs = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![body],
                vec![input],
            )
            .unwrap()
            .to_vec();
        assert_eq!(outputs.len(), 2);
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(outputs, vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.entry().output_count(), 2);
        assert_eq!(statistics.entry().maximum_output_dependency_depth(), 1);
    }

    /// Verifies deterministic arena ordering and exact attachment edges for a nested region graph.
    #[test]
    fn test_statistics_nested_region_graph_order_and_edges() {
        let leaf = identity_region_program();

        let mut middle_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested = middle_builder.import_region(leaf.entry_region_ref());
        let middle_input = middle_builder.add_input(ArrayType::scalar(F64));
        let middle_output = middle_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![nested],
                vec![middle_input],
            )
            .unwrap()[0];
        let middle: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            middle_builder.build(vec![middle_output], vec![Placeholder], vec![Placeholder]).unwrap();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let imported = builder.import_region(middle.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::rule("rule")] }),
                vec![imported],
                vec![input],
            )
            .unwrap()[0];
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // Descendants precede parents in arena order and the entry region is final: the leaf is region 0, the middle
        // region attaching it is region 1, and the entry attaching the middle region is region 2.
        let statistics = program.statistics();
        assert_eq!(statistics.region_count(), 3);
        assert_eq!(statistics.entry_region_index(), 2);
        assert_eq!(statistics.regions()[0].attached_regions(), &[]);
        assert_eq!(
            statistics.regions()[1].attached_regions(),
            &[AttachedRegionStatistics {
                instruction_index: 0,
                operation: "with_regions",
                region_slot: "body",
                region_role: RegionRole::Computation,
                region_index: 0,
            }],
        );
        assert_eq!(
            statistics.entry().attached_regions(),
            &[AttachedRegionStatistics {
                instruction_index: 0,
                operation: "with_regions",
                region_slot: "rule",
                region_role: RegionRole::Rule,
                region_index: 1,
            }],
        );
        assert_eq!(statistics.entry().attached_regions()[0].label(), "with_regions.rule");
    }

    /// Verifies that a region attached through two slots of one instruction yields one node and two edges.
    #[test]
    fn test_statistics_shared_region_yields_one_node_and_two_edges() {
        let leaf = identity_region_program();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let shared = builder.import_region(leaf.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![shared, shared],
                vec![input],
            )
            .unwrap()[0];
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let statistics = program.statistics();
        assert_eq!(statistics.region_count(), 2);
        assert_eq!(
            statistics.entry().attached_regions(),
            &[
                AttachedRegionStatistics {
                    instruction_index: 0,
                    operation: "with_regions",
                    region_slot: "first",
                    region_role: RegionRole::Computation,
                    region_index: 0,
                },
                AttachedRegionStatistics {
                    instruction_index: 0,
                    operation: "with_regions",
                    region_slot: "second",
                    region_role: RegionRole::Computation,
                    region_index: 0,
                },
            ],
        );
        // The shared region contributes its (zero) instructions once, so the totals count only the entry.
        assert_eq!(statistics.total_instruction_count(), 1);
    }

    /// Verifies the derived aggregate accessors on a multi-region program.
    #[test]
    fn test_statistics_aggregates_count_shared_regions_once() {
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(F64));
        let region_constant = region_builder.add_constant(Array::scalar(1.0));
        let region_output = region_builder
            .add_instruction(TestRegionOperation::Add, vec![], vec![region_input, region_constant])
            .unwrap()[0];
        let region_program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            region_builder.build(vec![region_output], vec![Placeholder], vec![Placeholder]).unwrap();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let shared = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(F64));
        let first = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![shared],
                vec![input],
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![shared],
                vec![first],
            )
            .unwrap()[0];
        let output = builder.add_instruction(TestRegionOperation::Add, vec![], vec![first, second]).unwrap()[0];
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // The shared region is attached twice but contributes its one instruction and one constant only once.
        let statistics = program.statistics();
        assert_eq!(statistics.region_count(), 2);
        assert_eq!(statistics.total_instruction_count(), 4);
        assert_eq!(statistics.total_constant_count(), 1);
        assert_eq!(statistics.total_operation_counts(), BTreeMap::from([("add", 2usize), ("with_regions", 2usize)]));
    }

    /// Verifies the exact serialized form, including field names, arena ordering, index-based edge references, and
    /// deterministic operation-count key order. Derived values such as region counts, entry indexes, labels, and
    /// totals must not appear in the serialized form.
    #[test]
    fn test_statistics_serialization() {
        let leaf = identity_region_program();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let body = builder.import_region(leaf.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(F64));
        let constant = builder.add_constant(Array::scalar(1.0));
        let added = builder.add_instruction(TestRegionOperation::Add, vec![], vec![input, constant]).unwrap()[0];
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![body],
                vec![added],
            )
            .unwrap()[0];
        let program: Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> =
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let serialized = serde_json::to_string_pretty(&program.statistics()).unwrap();
        assert_eq!(
            serialized,
            indoc! {r#"
                {
                  "regions": [
                    {
                      "input_count": 1,
                      "output_count": 1,
                      "instruction_count": 0,
                      "constant_count": 0,
                      "operation_counts": {},
                      "maximum_output_dependency_depth": 0,
                      "attached_regions": []
                    },
                    {
                      "input_count": 1,
                      "output_count": 1,
                      "instruction_count": 2,
                      "constant_count": 1,
                      "operation_counts": {
                        "add": 1,
                        "with_regions": 1
                      },
                      "maximum_output_dependency_depth": 2,
                      "attached_regions": [
                        {
                          "instruction_index": 1,
                          "operation": "with_regions",
                          "region_slot": "body",
                          "region_role": "computation",
                          "region_index": 0
                        }
                      ]
                    }
                  ]
                }"#}
            .trim_end(),
        );
    }
}
