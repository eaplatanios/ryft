use std::collections::BTreeSet;

use crate::contexts::Domain;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::values::Value;
use crate::tracing::TracingContext;

use super::super::ReferenceAllocationHandle;

// TODO(eaplatanios): Review this module.

/// of the same program universe, which seals into a program the rule attaches to its rewritten operation.
///
/// It is deliberately a fresh allocation trace rather than a nested trace of the live destination. A rebuilt region is a
/// self-contained artifact whose complete interface is its own boundary, so it must not close over any value of the
/// destination it will be attached in. Being an allocation trace is also what makes the type a fixed point of its own
/// construction — the destination of a destination is that same destination — which is what keeps the obligation that
/// this universe's operations discharge into it finite.
pub type ReferenceDischargeRegionDestination<C> = TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>;

/// One group of reference-related positions a rebuilt region gains: the allocations crossing there and the source
/// boundary position at which they are inserted.
///
/// A discharged allocation crosses as immutable state, while a preserved allocation crosses as its destination
/// reference.
///
/// Grouping the allocations with their insertion position keeps a boundary request's input and output groups from being
/// transposable: the two groups have the same shape, and passing them positionally compiled silently when swapped.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRegionStateInsertion {
    /// Allocations crossing at this group's positions, in canonical allocation order.
    allocations: Vec<ReferenceAllocationHandle>,

    /// Position in the source region's boundary at which the group is inserted.
    position: usize,
}

impl ReferenceRegionStateInsertion {
    /// Creates a boundary-position group inserting `allocations` at `position`.
    #[inline]
    pub fn new(allocations: Vec<ReferenceAllocationHandle>, position: usize) -> Self {
        Self { allocations, position }
    }
}

/// Symmetric widening facts one structured rule derives from a region summary through
/// [`state_widening`](crate::programs::references::ReferenceDischargeContext::state_widening).
///
/// The three sets state one algorithm every symmetric structured rewrite shares: the *threaded* allocations are the
/// discharged references crossing as immutable state, the *entering* allocations gain added positions because no
/// declared position already carries them, and the *published* allocations are the discharged references whose final
/// states the rebuilt regions must return. An entering preserved reference crosses in its added position as a
/// reference, so it belongs to neither the threaded nor the published set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceStateWidening {
    /// Every discharged reference the region closures reach, in canonical allocation order.
    pub(super) threaded: BTreeSet<ReferenceAllocationHandle>,

    /// Reached allocations gaining added boundary positions, in canonical allocation order. Discharged allocations
    /// cross as state and preserved allocations cross as references.
    pub(super) entering: Vec<ReferenceAllocationHandle>,

    /// Threaded allocations some closure mutates, in canonical allocation order.
    pub(super) published: Vec<ReferenceAllocationHandle>,
}

impl ReferenceStateWidening {
    /// Returns every discharged reference the region closures reach, in canonical allocation order.
    #[inline]
    pub fn threaded(&self) -> &BTreeSet<ReferenceAllocationHandle> {
        &self.threaded
    }

    /// Returns the reached allocations gaining added boundary positions, in canonical allocation order.
    #[inline]
    pub fn entering(&self) -> &[ReferenceAllocationHandle] {
        self.entering.as_slice()
    }

    /// Returns the threaded allocations some closure mutates, in canonical allocation order.
    #[inline]
    pub fn published(&self) -> &[ReferenceAllocationHandle] {
        self.published.as_slice()
    }
}

/// Boundary a structured reference discharge rule requests for one rebuilt region.
///
/// The rule owns the mapping from its own operands onto a region's declared inputs, because that mapping is part of
/// what the operation *is*. It therefore describes the declared input boundary itself, in region order, and names
/// separately the reference-related positions the rebuilt region gains: which allocations enter, which allocations it
/// must publish, and where each group is inserted.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRegionDischargeBoundary {
    /// Allocation entering at each declared source-region input position, or [`None`] for an ordinary value.
    declared_input_allocations: Vec<Option<ReferenceAllocationHandle>>,

    /// Length of the region's own leading capture prefix, from [`Operation::region_capture_input_count`], or [`None`]
    /// when the region inherits the capture scope of the region its operation is applied in.
    capture_input_count: Option<usize>,

    /// Allocations whose entering carrier the rebuilt region receives as added inputs, in canonical allocation order.
    /// Discharged references enter as immutable state; preserved references enter as their destination reference value.
    added_input_allocations: Vec<ReferenceAllocationHandle>,

    /// Position in the source region's input boundary at which the added inputs are inserted.
    input_insertion: usize,

    /// Allocations the rebuilt region publishes as added outputs, in canonical allocation order. Discharged
    /// allocations publish final state and preserved allocations publish their destination references.
    added_output_allocations: Vec<ReferenceAllocationHandle>,

    /// Position in the source region's output boundary at which the added outputs are inserted.
    output_insertion: usize,
}

impl ReferenceRegionDischargeBoundary {
    /// Creates a rebuilt-region boundary request.
    ///
    /// Added positions are described separately from the declared positions because only the declared positions are
    /// replayed: an added input exists in the rebuilt region's destination boundary and in the caller's operand list,
    /// but the source region's body never named it and therefore cannot consume it.
    ///
    /// The region's capture prefix is read off the operation rather than supplied, so that a rule cannot state one
    /// prefix here and let
    /// [`region_summary`](crate::programs::references::ReferenceDischargeContext::region_summary) derive a different
    /// one from the same hook. A rule therefore never reasons about captures at all.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation the region is attached to, whose
    ///     [`region_capture_input_count`](Operation::region_capture_input_count) states the region's own leading
    ///     capture prefix.
    ///   - `region_index`: Position of the region among that operation's attached regions.
    ///   - `declared_input_allocations`: Allocation entering at each declared boundary position, or [`None`] for an
    ///     ordinary value. Reference positions must come from
    ///     [`operand_allocation`](crate::programs::references::ReferenceDischargeContext::operand_allocation), which
    ///     validates that each operand carries the complete stored value rather than a derived view. Its length must
    ///     equal the source region's input count, because every declared position is rebuilt.
    ///   - `added_inputs`: Allocations whose entering state or preserved reference the rebuilt region receives as added
    ///     inputs, grouped with the source input position receiving them.
    ///   - `added_outputs`: Allocations the rebuilt region publishes as added outputs, grouped with the source output
    ///     position receiving them. Discharged allocations publish state and preserved allocations publish references.
    #[inline]
    pub fn new<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_allocations: Vec<Option<ReferenceAllocationHandle>>,
        added_inputs: ReferenceRegionStateInsertion,
        added_outputs: ReferenceRegionStateInsertion,
    ) -> Self {
        Self {
            declared_input_allocations,
            capture_input_count: operation.region_capture_input_count(region_index),
            added_input_allocations: added_inputs.allocations,
            input_insertion: added_inputs.position,
            added_output_allocations: added_outputs.allocations,
            output_insertion: added_outputs.position,
        }
    }

    /// Creates a rebuilt-region boundary request whose added inputs and outputs are the same allocations at the same
    /// position, which is the symmetric loop-carry shape `while` bodies and `scan` bodies thread.
    #[inline]
    pub fn symmetric<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_allocations: Vec<Option<ReferenceAllocationHandle>>,
        state: ReferenceRegionStateInsertion,
    ) -> Self {
        Self::new(operation, region_index, declared_input_allocations, state.clone(), state)
    }

    /// Returns the allocation entering at each declared boundary position, or [`None`] for an ordinary value.
    #[inline]
    pub(super) fn declared_input_allocations(&self) -> &[Option<ReferenceAllocationHandle>] {
        self.declared_input_allocations.as_slice()
    }

    /// Returns the region's own leading capture prefix, or [`None`] when it inherits its caller's capture scope.
    #[inline]
    pub(super) const fn capture_input_count(&self) -> Option<usize> {
        self.capture_input_count
    }

    /// Returns the allocations whose entering state or preserved reference the rebuilt region receives as added inputs.
    #[inline]
    pub(super) fn added_input_allocations(&self) -> &[ReferenceAllocationHandle] {
        self.added_input_allocations.as_slice()
    }

    /// Returns the source input position at which the added inputs are inserted.
    #[inline]
    pub(super) const fn input_insertion(&self) -> usize {
        self.input_insertion
    }

    /// Returns the allocations the rebuilt region publishes as added outputs.
    #[inline]
    pub(super) fn added_output_allocations(&self) -> &[ReferenceAllocationHandle] {
        self.added_output_allocations.as_slice()
    }

    /// Returns the source output position at which the added outputs are inserted.
    #[inline]
    pub(super) const fn output_insertion(&self) -> usize {
        self.output_insertion
    }
}

/// Sealed result of discharging one attached region against an isolated environment.
///
/// This is the transactional artifact of a structured rule's region fork, and it deliberately carries no values of
/// any kind. A reference handle produced inside the fork would keep addressing the fork's own abandoned environment,
/// and even a plain destination value is not detached under a staging destination, because it is itself a tracer
/// stamped with the fork's builder. Excluding both structurally is what makes the isolation a type-level fact rather
/// than a convention: the owning rule binds the rebuilt operation in its *own* context and merges the final states
/// from the outputs that binding produced.
#[derive(Debug)]
pub struct ReferenceRegionDischargeFork<V: Value, O: Operation<Type = V::Type>> {
    /// Rebuilt, discharged region program.
    pub(super) program: Program<V, O, Vec<V>, Vec<V>>,

    /// Allocation each *declared* region output denotes, or [`None`] for an ordinary output, in region-boundary order.
    pub(super) output_allocations: Vec<Option<ReferenceAllocationHandle>>,

    /// Threaded allocations the region's closure mutated, in canonical allocation order.
    pub(super) mutated_allocations: Vec<ReferenceAllocationHandle>,
}

impl<V: Value, O: Operation<Type = V::Type>> ReferenceRegionDischargeFork<V, O> {
    /// Returns the allocation each declared region output denotes, or [`None`] where the output is an ordinary value.
    #[inline]
    pub fn output_allocations(&self) -> &[Option<ReferenceAllocationHandle>] {
        self.output_allocations.as_slice()
    }

    /// Consumes this fork and returns the rebuilt region program.
    #[inline]
    pub fn into_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.program
    }

    /// Validates that this region's declared outputs denote exactly the allocations the widening that sized its boundary
    /// expected them to.
    ///
    /// The widening reads the declared output allocations from a *static* summary, and the boundary it sizes depends on
    /// them: an allocation a region already returns publishes its final state at that output's own position and must not be
    /// published a second time. This is where that prediction is held to what the replay actually produced, so a rule
    /// whose operation disagrees with its own generic hooks is reported instead of silently losing an update. It also
    /// makes the several regions of one operation agree with each other, because they are all checked against the one
    /// summary that sized their shared boundary.
    ///
    /// # Parameters
    ///
    ///   - `expected`: Allocation each declared output was predicted to denote, in region-boundary order.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this region's declared outputs denote different allocations.
    pub fn validate_predicted_output_allocations(
        &self,
        expected: &[Option<ReferenceAllocationHandle>],
        operation: &str,
    ) -> Result<(), ProgramError> {
        if self.output_allocations != expected {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` attaches a region whose outputs do not denote the references its state \
                 widening expected",
            )));
        }
        Ok(())
    }

    /// Validates that this region mutated no allocation the widening that sized its boundary did not publish.
    ///
    /// The boundary was sized from a summary computed before the region ran, so this is where the summary and the
    /// replay are held to each other. A mismatch means one of the generic hooks the summary follows under-reports what
    /// its operation does, and reporting it here is what keeps that from surfacing later as a lost update.
    ///
    /// # Parameters
    ///
    ///   - `published`: Allocations whose final state the widening decided this region publishes.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the first allocation this region mutated that `published` does not
    /// contain.
    pub fn validate_predicted_mutations(
        &self,
        published: &[ReferenceAllocationHandle],
        operation: &str,
    ) -> Result<(), ProgramError> {
        for allocation in &self.mutated_allocations {
            if !published.contains(allocation) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{operation}` mutated {allocation} in an attached region that its state widening did not \
                     predict",
                )));
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

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;

    use crate::programs::{
        RecursiveReferenceDischargeDriver, ReferenceDischargeDriver, ReferenceRegionDischargeBoundary,
        ReferenceRegionStateInsertion,
    };

    #[test]
    fn test_reference_region_discharge_fork_holds_the_replay_to_the_widening_that_sized_it() {
        // The boundary is sized from a summary computed before the region ran, so both validators exist to catch an
        // operation whose generic hooks disagree with what its closure actually does. Here the fork is produced
        // honestly and then held to deliberately wrong predictions, which is the shape a lying third-party family
        // would present.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous =
            builder.add_instruction(ListOperation::Swap, Vec::new(), vec![reference, update], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![previous, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the caller allocation").unwrap().allocation();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation), None],
            ReferenceRegionStateInsertion::new(Vec::new(), 2),
            ReferenceRegionStateInsertion::new(Vec::new(), 2),
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();

        // The region writes its entering allocation, so a widening that published nothing lost that update.
        assert_eq!(
            fork.validate_predicted_mutations(&[], "list.call"),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` mutated {allocation} in an attached region that its state widening did not predict",
            ))),
        );
        assert_eq!(fork.validate_predicted_mutations(&[allocation], "list.call"), Ok(()));

        // The region returns that allocation at its second output, so a widening that predicted an ordinary value there
        // would have published the allocation's final state twice.
        assert_eq!(fork.output_allocations(), &[None, Some(allocation)]);
        assert_eq!(
            fork.validate_predicted_output_allocations(&[None, None], "list.call"),
            Err(ProgramError::MalformedProgram(
                "operation `list.call` attaches a region whose outputs do not denote the references its state \
                 widening expected"
                    .to_string(),
            )),
        );
        assert_eq!(fork.validate_predicted_output_allocations(&[None, Some(allocation)], "list.call"), Ok(()));
    }
}
