use std::rc::Rc;

// TODO(eaplatanios): Review this module.

use crate::captures::CaptureConstant;
use crate::contexts::StagingContext;
use crate::parameters::Placeholder;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::values::Value;
use crate::tracing::TracingContext;

use super::interpreter::{
    RecursiveReferenceDischargeDriver, ReferenceCaptureScope, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargeValue, ReferenceDischargeableOperation, region_closure_touches_references,
};
use super::policies::ReferenceDischargePolicy;
use super::results::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, ReferenceDischargePayload, ReferenceDischargeResult,
    ReferenceSource,
};
use super::selection::{ReferenceDischargeSelection, ReferenceDischargeSite};

/// Program-level capability for normalizing references into explicit immutable state.
///
/// An implementation names its universe's [`ReferenceDischargePolicy`] and otherwise only forwards to the interpreter
/// entry point [`Program::discharge_references_with_policy`]. Generic transforms reach discharge through
/// [`discharge_local_references`](Self::discharge_local_references) and therefore neither name a policy nor inspect
/// family-specific alias metadata.
pub trait ReferenceDischarge: Sized {
    /// Reference-free program payload produced by this implementation.
    type DischargedProgram: ReferenceDischargePayload;

    /// Discharges every reference and returns the reference-free program plus its logical external-state bindings.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    fn discharge_references(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self::DischargedProgram>, ProgramError>;

    /// Discharges local references for `transform`, rejecting every caller-owned external allocation.
    ///
    /// The full-discharge implementation must prove that the result contains neither reference types nor unresolved
    /// ordered reference state. The checked result envelope ensures that an external-state-free result has no hidden
    /// output suffix, so this default returns the same program family with an unchanged public boundary.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    ///   - `transform`: Name used in diagnostics when caller-owned state prevents the transform.
    fn discharge_local_references(
        self,
        capture_count: usize,
        transform: &'static str,
    ) -> Result<Self::DischargedProgram, ProgramError> {
        let discharged = self.discharge_references(capture_count)?;
        if let Some(state) = discharged.external_states().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{transform} supports only local references, but the program uses external `{}`",
                    state.source(),
                ),
            });
        }
        let (program, _, _, _) = discharged.into_parts();
        Ok(program)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Discharges every reference this program touches by interpreting it in a [`ReferenceDischargeContext`] over a
    /// fresh trace of its own universe, returning the reference-free program together with its logical external-state
    /// bindings.
    ///
    /// This is the entry point production discharge runs through, and it owns the whole reference language: the
    /// primitives rewrite themselves, and a region-carrying operation either discharges through its own
    /// [`ReferenceDischargeableOperation`] rule — widening its boundaries with the state its regions touch — or, when
    /// nothing in its attached closure touches a reference, replays those regions verbatim. What survives as a
    /// rejection is a region closure that does touch a reference behind an operation with no rule of its own, and a
    /// reference that reaches a region neither through a boundary nor through a capture scope. A universe whose
    /// programs name their caller's references through capture constants uses
    /// [the capture-aware entry point](Self::discharge_references_with_lifted_captures_and_policy) instead, which is
    /// the same rewrite under a populated capture scope.
    ///
    /// Each source input keeps its position. A reference-typed input becomes an ordinary input carrying the referent's
    /// lifted type, which is the entering immutable state of the allocation that input denotes; every other input is
    /// replayed unchanged. The public outputs are exactly the source outputs, in order, and the final state of each
    /// *mutated* external allocation is appended after them as a hidden output in entry-boundary order. An allocation that the
    /// program only reads contributes no hidden output, so a read-only program keeps its original boundary exactly.
    ///
    /// The replay runs through [`ReferenceDischargeDriver::discharge_region`] rather than through
    /// [`interpret_in_context`](Self::interpret_in_context), because that is the path that threads each instruction's
    /// source coordinate into the rules, which is what makes an entry-region allocation identifiable.
    ///
    /// The rewritten payload is proven reference-free rather than assumed to be: the replay assembles a
    /// [`PartialReferenceDischargeResult`] and converts it through
    /// [`try_into_full`](PartialReferenceDischargeResult::try_into_full), so a rule that returned a reference-touching
    /// operation is reported here instead of surviving into a result whose contract says it cannot exist.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the program's input count, when an
    /// output still denotes a reference, when the program consumes an external allocation, whose state belongs to the
    /// caller, or when the rewritten payload fails the reference-freedom proof. Rule-level failures, including a
    /// use-after-consume and an access to an unbound allocation, propagate from the replay itself.
    pub fn discharge_references_with_policy<P>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeSelection::everything(),
        )?
        .try_into_full()
    }

    /// Discharges the references the caller *selected* and preserves every other one, returning the mixed program
    /// together with the logical external-state bindings of the allocations that became state.
    ///
    /// This is the same rewrite [`discharge_references_with_policy`](Self::discharge_references_with_policy) performs;
    /// full discharge is exactly its everything-selected case, and the two share one body. A selected allocation threads as
    /// immutable state in every respect described there. An unselected allocation instead *survives*: it keeps its
    /// reference-typed boundary position or its allocating instruction, every access to it replays verbatim as the
    /// reference operation the source performed, and a view derived from it replays its view operation too, so the
    /// rewritten program still denotes the same coordinates. Preserved references contribute no state input, no hidden
    /// final-state output, and no [`ExternalReferenceBinding`]: the payload's own boundary is where the caller sees them.
    ///
    /// This is what a kernel pipeline needs — normalize the pipeline's own state into explicit carries while the
    /// references a kernel body addresses stay references — and it is the reason the result envelope is
    /// [`PartialReferenceDischargeResult`], which proves nothing about reference freedom. A caller that expects the
    /// selection to have covered everything asks for the proof explicitly through
    /// [`try_into_full`](PartialReferenceDischargeResult::try_into_full).
    ///
    /// A preserved reference crosses a structured operation's region boundary the same way it crosses anything else: as the
    /// reference it already is, at its own declared operand position, exactly as the source passed it. It occupies no
    /// state carry, publishes no successor, and widens nothing, so a condition, loop, scan, or call can thread
    /// discharged state and surviving references side by side. What a preserved reference cannot become is *added* state
    /// that a rule synthesizes onto a rebuilt boundary, which is reported by name.
    ///
    /// Where a structured operation *declares* a reference-typed output — a loop carry, say — the rewritten operation
    /// still produces one, and it is deliberately left unused: the caller keeps the handle it already holds, because
    /// both denote the same allocation and one destination value per allocation is enough. A later full discharge of the same
    /// payload collapses that position into an ordinary state carry.
    ///
    /// A *capture-lifted* program has no partial form: this entry point recognizes no capture constant, so a
    /// reference-typed one is rejected where it is lifted, and
    /// [the capture-aware entry point](Self::discharge_references_with_lifted_captures_and_policy) remains
    /// full-discharge-only.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///   - `sites`: Reference sites to discharge, enumerated from this same program through
    ///     [`reference_discharge_sites`](Self::reference_discharge_sites). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `sites` does not validate against this program, when a rule
    /// synthesizes a preserved reference onto a rebuilt region's added state positions, and otherwise for every reason
    /// [`discharge_references_with_policy`](Self::discharge_references_with_policy) documents — with one deliberate
    /// exception. Consuming a *discharged* external allocation is still rejected, because a
    /// [`ExternalReferenceBinding`] cannot express a caller-owned reference that no longer denotes live state; consuming
    /// a *preserved* one is accepted, because the payload retains the consuming operation and the caller passes its
    /// reference handle to that operation directly.
    pub fn partially_discharge_references_with_policy<P>(
        self,
        capture_count: usize,
        sites: &[ReferenceDischargeSite],
    ) -> Result<PartialReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.validate_reference_discharge_sites(capture_count, sites)?;
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeSelection::from_sites(sites),
        )
    }

    /// Discharges every reference a *capture-lifted* program touches, resolving the capture-scoped reference
    /// constants its attached regions name their caller's references through.
    ///
    /// A capture-lifted program is one whose captures have been turned into a leading input prefix by
    /// [`ClosedProgram::to_program_with_lifted_captures`](crate::ClosedProgram::to_program_with_lifted_captures).
    /// Lifting rewrites the entry boundary, but an attached region keeps naming the same captures through
    /// [`CaptureReference`](crate::CaptureReference) constants, and those constants denote the very allocations the lifted
    /// prefix binds. This entry point therefore differs from
    /// [`discharge_references_with_policy`](Self::discharge_references_with_policy) in exactly one respect: it seeds
    /// the entry capture scope from that prefix, so a reference-typed capture constant resolves to the
    /// allocation its position already binds instead of being rejected as belonging to no allocation.
    ///
    /// Everything else — the boundary rewrite, the hidden final-state outputs, and the reference-freedom proof — is
    /// identical, and a program with no capture-scoped reference constant discharges to exactly the same result
    /// through either entry point.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Length of the lifted capture prefix, which is both the split between
    ///     [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions and the capture scope's own
    ///     length.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`discharge_references_with_policy`](Self::discharge_references_with_policy), and
    /// additionally reports a capture constant whose declared reference type disagrees with the allocation its position
    /// binds.
    pub fn discharge_references_with_lifted_captures_and_policy<P>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self>, ProgramError>
    where
        V: CaptureConstant,
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            CaptureConstant::capture_index,
            ReferenceDischargeSelection::everything(),
        )?
        .try_into_full()
    }

    /// Discharges the selected references of this program, recognizing capture constants through `capture_index`.
    ///
    /// This is the shared body of the program-level entry points, and the partial rewrite is the general one: full
    /// discharge is the everything-selected case, which is why the body always assembles the partial envelope and
    /// leaves the reference-freedom proof to the caller that promised it.
    ///
    /// The capture seam is a parameter rather than a bound because the interpreter deliberately serves constant
    /// families that are not capture-bearing at all, and it is a function pointer rather than a closure because the
    /// only two seams that ever reach it are "nothing is a capture" and [`CaptureConstant::capture_index`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    ///   - `capture_index`: Seam reporting the capture position a stored constant names.
    ///   - `selection`: Reference sites to discharge; every allocation the selection omits is preserved.
    ///
    /// # Errors
    ///
    /// Returns the errors the public entry points document, which is every error the replay can raise.
    fn discharge_references_with_capture_seam<P>(
        self,
        capture_count: usize,
        capture_index: fn(&V) -> Option<usize>,
        selection: ReferenceDischargeSelection,
    ) -> Result<PartialReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        let input_types = self.input_types();
        let input_count = input_types.len();
        if capture_count > input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge requests {capture_count} captures but the program has {input_count} inputs",
            )));
        }
        let public_output_count = self.output_count();

        // A program that touches no reference anywhere is already its own discharge, so it is returned untouched
        // rather than replayed into a fresh trace. This is not only cheaper on the two transform adapters that
        // discharge unconditionally: re-tracing would also renumber its atoms, drop its dead constants, and abandon
        // the region transform cache its regions carry, all for a rewrite that has nothing to rewrite.
        let entry = self.entry_region_ref();
        if !region_closure_touches_references(entry) {
            return PartialReferenceDischargeResult::new(self, capture_count, public_output_count, Vec::new());
        }

        // The block scopes the destination context, the discharge context, and every carrier, because recovering the
        // traced program below requires unique ownership of the shared builder and therefore that every other handle
        // to it has been released.
        let (builder, output_ids, external_states) = {
            let destination = TracingContext::<V, O>::new();
            let builder = destination.builder().clone();
            let context =
                ReferenceDischargeContext::<TracingContext<V, O>, P>::new_selecting(destination.clone(), selection);
            let mut inputs = Vec::with_capacity(input_count);
            let mut discharged_allocations = Vec::new();
            let mut capture_allocations = vec![None; capture_count];
            for (input_index, input_type) in input_types.into_iter().enumerate() {
                let Some(reference_type) = P::project_reference_type(&input_type) else {
                    inputs.push(ReferenceDischargeValue::Ordinary(destination.input(input_type)));
                    continue;
                };
                let source = ReferenceSource::from_flat_input_index(input_index, capture_count);
                let selected = context.selects_external(source);
                let carrier = if selected {
                    let state = destination.input(P::lift_referent_type(reference_type.referent().clone()));
                    context.allocate_discharged(reference_type, state)?
                } else {
                    // An unselected external allocation keeps its reference-typed boundary position exactly as the source
                    // declared it, so the caller still supplies the reference and every access to it replays verbatim.
                    context.bind_preserved(reference_type, destination.input(input_type))?
                };
                let allocation = carrier.expect_reference("an entry-boundary reference allocation")?.allocation();
                if selected {
                    discharged_allocations.push((source, allocation));
                }
                if input_index < capture_count {
                    capture_allocations[input_index] = Some(allocation);
                }
                inputs.push(carrier);
            }

            // The capture scope can only be established once the prefix has minted its allocations, and it is what lets a
            // nested region resolve the caller references it names through capture constants rather than through its
            // own boundary.
            let context = context.with_captures(ReferenceCaptureScope::new(capture_index, capture_allocations));

            let regions = [self];
            let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
            let outputs = driver.discharge_region(&context, 0, inputs)?;
            let mut output_ids = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| match output {
                    ReferenceDischargeValue::Ordinary(value) => value.atom_id(),

                    // A preserved reference survives in the rewritten program, so returning one returns its destination
                    // reference value. A discharged reference has no such value, because it became state. Returning an allocation
                    // is a use of it like any other, so its liveness is resolved against the environment rather than
                    // taken from the handle, which is what reports an allocation the program already consumed.
                    ReferenceDischargeValue::Reference(reference) => {
                        context.validate_live_allocation(reference.allocation())?;
                        reference
                            .preserved()
                            .ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference discharge expected an ordinary value for output {output_index} but \
                                     received {reference}",
                                ))
                            })?
                            .atom_id()
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            // A mutated external allocation publishes its final state as a hidden output; a read-only one publishes nothing,
            // which is what keeps a read-only program's boundary identical to its source boundary. A preserved
            // external allocation binds nothing at all: it never became state, so there is no state for a caller to supply
            // or to write back.
            let mut external_states = Vec::with_capacity(discharged_allocations.len());
            for (source, allocation) in discharged_allocations {
                if context.validate_live_allocation(allocation).is_err() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge consumed external {source}, whose state must remain owned by the caller",
                    )));
                }
                let output_index = if context.is_mutated(allocation)? {
                    output_ids.push(context.discharged_state(allocation)?.atom_id()?);
                    Some(output_ids.len() - 1)
                } else {
                    None
                };
                external_states.push(ExternalReferenceBinding::new(source, output_index));
            }
            (builder, output_ids, external_states)
        };

        let output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
        PartialReferenceDischargeResult::new(program, capture_count, public_output_count, external_states)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;

    use crate::programs::instructions::InstructionId;

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;

    use super::*;

    #[test]
    fn test_reference_discharge_local_references_calls_full_discharge_once_and_preserves_failure_precedence() {
        let calls = Rc::new(Cell::new(0));
        let local = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::Local };
        assert_eq!(local.discharge_local_references(0, "test transform"), Ok(TestPayload::new(7, 0, 0)));
        assert_eq!(calls.get(), 1);

        let external = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::External };
        assert_eq!(
            external.discharge_local_references(0, "test transform"),
            Err(ProgramError::UnsupportedOperation {
                message: "test transform supports only local references, but the program uses external \
                          `input 0`"
                    .to_string(),
            }),
        );
        assert_eq!(calls.get(), 2);

        let malformed = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::Malformed };
        assert_eq!(
            malformed.discharge_local_references(0, "test transform"),
            Err(ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            )),
        );
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn test_partial_reference_discharge_preserves_unselected_external_allocations() {
        // The kernel-pipeline shape, in a universe that mentions no arrays: one caller-owned allocation is selected and
        // becomes threaded state, while the other survives as a reference the rewritten program still accesses
        // through the very operations the source used.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![kernel], None).unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![pipeline, update], None).unwrap();
        builder.add_instruction(ListOperation::Swap, Vec::new(), vec![kernel, observed], None).unwrap();
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        assert_eq!(
            sites,
            vec![
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        let discharged = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &sites[..1]);
        let discharged = discharged.unwrap();

        // The selected allocation became an ordinary state input at its own boundary position and publishes its final state
        // as a hidden output; the preserved reference kept its reference type, so it binds nothing at all.
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>>, %2:list<2> .
                let %3:list<2> = list.read %1
                    %4:list<2> = list.select %0
                    %5:list<2> = list.add %4 %2
                    %6:list<2> = list.splice %0 %5
                    %7:list<2> = list.swap %1 %3
                in (%3, %6)"},
        );

        // The result deliberately proves nothing about reference freedom, and asking for the proof reports the
        // surviving reference rather than converting.
        assert_eq!(
            discharged.try_into_full().unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_preserves_an_unselected_allocation_site() {
        // An interior allocation is selectable in its own right, so a program can normalize its pipeline state while
        // the allocation a kernel body addresses is allocated, viewed, accessed, and consumed as a reference throughout.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![allocation], None)
            .unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Selecting nothing preserves the allocation, so the whole reference language survives: the view operation is
        // replayed too, and the derived handle consumes the reference that replay produced rather than re-deriving
        // the chain at the access.
        let discharged = source.clone().partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]);
        let discharged = discharged.unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<4>, %1:list<2> .
                let %2:ref<list<4>> = list.reference_new %0
                    %3:ref<list<2>> = list.slice %2
                    list.add_update %3 %1
                    %4:list<4> = list.freeze %2
                in (%4)"},
        );

        // Selecting the allocation instead discharges it, which is the everything-selected case and therefore has to
        // agree with full discharge exactly.
        let sites = source.reference_discharge_sites(0).unwrap();
        assert_eq!(
            sites,
            vec![ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(source.entry_region_ref().id(), 0),
                output_index: 0,
            }],
        );
        let selected = source
            .clone()
            .partially_discharge_references_with_policy::<ListReferenceDischarge>(0, sites.as_slice());
        let selected = selected.unwrap().try_into_full().unwrap();
        let full = source.discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
    }

    #[test]
    fn test_partial_reference_discharge_lets_a_program_consume_a_preserved_external_allocation() {
        // Full discharge rejects a program that consumes a caller-owned allocation, because a `ExternalReferenceBinding`
        // cannot describe reference state that no longer exists. A preserved reference has no binding to describe: the
        // payload keeps the consuming operation, and the caller passes its reference to that operation directly, so partial
        // discharge accepts what full discharge cannot express.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let preserved =
            source.clone().partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.freeze %0
                in (%1)"},
        );
        assert_eq!(
            source.discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose state must remain owned by the caller"
                    .to_string(),
            ),
        );

        // Returning the allocation afterwards is a use of it like any other, so the consumed allocation is reported at the output
        // that names it rather than published as a stale reference.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![frozen, external],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // An allocation rendering embeds the identity of the environment that minted it, which is process-global, so the
        // assertion pins everything around it.
        let error = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap_err();
        let ProgramError::MalformedProgram(message) = &error else {
            panic!("expected a malformed-program rejection but got {error:?}");
        };
        assert!(message.starts_with("reference discharge accessed consumed reference allocation "), "{message}");
        assert!(message.ends_with(":0"), "{message}");
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_allocation_beside_discharged_state() {
        // A structured boundary carries both kinds of allocation at once: a discharged carry crosses as immutable state and
        // is widened with a published successor, while a preserved carry crosses as the reference it already is, at
        // its own declared operand position, and widens nothing at all.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_state = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee_kernel = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed =
            callee_builder.add_instruction(ListOperation::Read, Vec::new(), vec![callee_kernel], None).unwrap()[0];
        callee_builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![callee_state, observed], None)
            .unwrap();
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee = builder.import_program(callee);
        let observed =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![pipeline, kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged =
            source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &sites[..1]).unwrap();

        // The selected allocation's entering state occupies its own operand position and its successor is appended as a
        // published output; the preserved reference's operand position still carries a reference, and the rebuilt callee
        // performs the read on it exactly as the source did.
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>> .
                let %2:list<2>, %3:list<2> = list.call %0 %1 [
                    callee={
                        lambda %0:list<2>, %1:ref<list<2>> .
                        let %2:list<2> = list.read %1
                            %3:list<2> = list.select %0
                            %4:list<2> = list.add %3 %2
                            %5:list<2> = list.splice %0 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_partial_reference_discharge_validates_its_selection_against_the_program() {
        // The selection is checked before anything is replayed, so a site this program does not expose is reported
        // against the program rather than surfacing later as an allocation that never appeared.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source
                .partially_discharge_references_with_policy::<ListReferenceDischarge>(
                    0,
                    &[ReferenceDischargeSite::External(ReferenceSource::Input { index: 3 })],
                )
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge selection names external input 3, which is not a selectable site in this program"
                    .to_string(),
            ),
        );
    }
}
