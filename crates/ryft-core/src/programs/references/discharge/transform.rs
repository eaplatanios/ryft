use std::rc::Rc;

use crate::captures::{CaptureConstant, ClosedProgram};
use crate::contexts::StagingContext;
use crate::parameters::{Parameterized, Placeholder};
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::interpreter::{
    RecursiveReferenceDischargeDriver, ReferenceCaptureScope, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargeValue, ReferenceDischargeableOperation,
};
use crate::programs::references::discharge::policies::{ReferenceDischargePolicy, ReferenceDischargeableType};
use crate::programs::references::discharge::results::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, ReferenceDischargeResult, ReferenceSource,
};
use crate::programs::references::discharge::targets::{ReferenceDischargeTarget, ReferenceDischargeTargets};
use crate::programs::references::types::ReferenceType;
use crate::programs::values::Value;
use crate::tracing::TracingContext;

// TODO(eaplatanios): Move `ReferenceDischargeContext` here.
// TODO(eaplatanios): Move `ReferenceDischargeEnvironmentId` here.
// TODO(eaplatanios): Move `ReferenceDischargeEnvironment` here.
// TODO(eaplatanios): Move `ReferenceAllocationEntry` here.
// TODO(eaplatanios): Move `ReferenceAllocationState` here.
// TODO(eaplatanios): Move `ReferenceDischargeAllocationId` here. Should this be renamed to `ReferenceAllocationId`?
// TODO(eaplatanios): Move `ReferenceCaptureScope` here.

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Rewrites every [`Reference`](crate::Reference) in this [`Program`] as explicit immutable state and returns the
    /// resulting reference-free program together with bindings for its external references.
    ///
    /// A reference-typed input keeps its position but becomes an ordinary input carrying the reference's initial state.
    /// Local reference allocations disappear. The source program's public outputs remain first and in the same order;
    /// the final state of each mutated external reference is appended as a hidden output in entry-boundary order.
    /// A read-only external reference adds no hidden output.
    ///
    /// Each operation defines how its reference effects are rewritten through [`ReferenceDischargeableOperation`].
    /// A structured operation must also thread through the reference state used by its attached
    /// [`Region`](crate::Region)s; a structured operation whose complete region closure is reference-free is replayed
    /// unchanged. The returned [`ReferenceDischargeResult`] proves that no reference type or reference operation
    /// remains anywhere in the rewritten region closure.
    ///
    /// Use [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// instead when the program was produced by lifting a [`ClosedProgram`]'s captures and attached regions may refer
    /// to those captures through capture constants.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the input count, when an external
    /// reference is consumed, when a reference reaches a region without a boundary or capture binding, when a
    /// structured operation has no rule for its reference-using regions, or when the rewritten program is not fully
    /// reference-free. Errors reported by individual operation rules, including use after consumption and access to
    /// an unbound reference, propagate unchanged.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites every [`Reference`](crate::Reference) in a capture-lifted [`Program`] as explicit immutable state and
    /// returns a proven reference-free result.
    ///
    /// A capture-lifted program has the captures of a [`ClosedProgram`] represented by a leading input prefix. Attached
    /// [`Region`](crate::Region)s may still name those captures through capture constants. This function resolves each
    /// reference-typed capture constant to the external reference bound at the corresponding prefix position.
    ///
    /// Apart from that capture resolution, this function has the same boundary rewrite, hidden final-state outputs,
    /// and reference-freedom guarantee as [`discharge_references`](Self::discharge_references). The two functions
    /// produce the same result when no attached region uses a reference-typed capture constant.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix. It determines both the capture scope and the
    ///     split between [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`discharge_references`](Self::discharge_references). It also returns
    /// [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            CaptureConstant::capture_index,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s as explicit immutable state while preserving
    /// every unselected reference.
    ///
    /// The selected references follow the same rewrite as [`discharge_references`](Self::discharge_references). An
    /// unselected reference keeps its reference-typed boundary position or allocation operation, and its accesses and
    /// derived views are replayed unchanged. It contributes no [`ExternalReferenceBinding`] or hidden final-state
    /// output because it never becomes explicit state.
    ///
    /// Preserved references can cross structured-[`Region`](crate::Region) boundaries beside discharged state. A
    /// declared reference position remains a reference position; when an attached region reaches a preserved reference
    /// only through an inherited capture, the rewrite adds a reference-typed position to keep the rebuilt region
    /// self-contained.
    ///
    /// The returned [`PartialReferenceDischargeResult`] may therefore still contain reference types and operations.
    /// If the selected targets were expected to cover every reference, convert it through
    /// [`ReferenceDischargeResult::try_from`] to validate and obtain the full-discharge guarantee.
    ///
    /// A capture-lifted program instead uses [`Self::partially_discharge_references_in_capture_lifted_program`], which
    /// performs the same target selection under a populated capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table,
    ///     used to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///   - `targets`: Reference targets to discharge, enumerated from this same program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `targets` does not belong to this program or otherwise violates
    /// the target-selection contract. It also returns the applicable errors documented by
    /// [`discharge_references`](Self::discharge_references), except that consuming a preserved external reference is
    /// allowed because its consuming operation remains in the rewritten program. Consuming a discharged external
    /// reference remains invalid because its state is still owned by the caller.
    #[inline]
    pub fn partially_discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, |_| None, targets)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s of a capture-lifted [`Program`] as explicit
    /// immutable state and preserves every unselected reference.
    ///
    /// This function combines the capture-constant resolution of
    /// [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// with the selection and partial-result behavior of
    /// [`partially_discharge_references`](Self::partially_discharge_references). The lifted capture prefix
    /// establishes the capture scope; only references named by `targets` become immutable state.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix.
    ///   - `targets`: Reference targets to discharge, enumerated from this same capture-lifted program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`partially_discharge_references`](Self::partially_discharge_references). It also
    /// returns [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn partially_discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, CaptureConstant::capture_index, targets)
    }

    /// Performs the shared partial-discharge rewrite for one validated target selection. `capture_index_of` resolves
    /// reference-typed capture constants when the input is capture-lifted; ordinary programs provide a resolver that
    /// matches no constant. The function always returns a [`PartialReferenceDischargeResult`]. Full-discharge entry
    /// points select every reference and then validate the result through [`ReferenceDischargeResult::try_from`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs that originated in the source program's capture table.
    ///   - `capture_index_of`: Function returning the capture position named by a stored constant.
    ///   - `targets`: Reference targets to discharge; every allocation they omit is preserved.
    ///
    /// # Errors
    ///
    /// Returns any validation or operation-rule error produced while rewriting the program.
    fn discharge_references_helper<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        capture_index_of: fn(&V) -> Option<usize>,
        targets: ReferenceDischargeTargets,
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let input_types = self.input_types();
        let input_count = input_types.len();
        if capture_count > input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge requests {capture_count} captures but the program has {input_count} inputs",
            )));
        }
        let output_count = self.output_count();

        // A program that touches no reference anywhere is already its own discharge, so it is returned untouched rather
        // than replayed into a fresh trace. This is not only cheaper on the two transform adapters that discharge
        // unconditionally: re-tracing would also renumber its atoms, drop its dead constants, and abandon the region
        // transform cache its regions carry, all for a rewrite that has nothing to rewrite.
        let entry = self.entry_region_ref();
        if !entry.contains_references_in_closure() {
            return PartialReferenceDischargeResult::new(self, capture_count, output_count, Vec::new());
        }

        // The block scopes the destination context, the discharge context, and every carrier, because recovering the
        // traced program below requires unique ownership of the shared builder and therefore that every other handle
        // to it has been released.
        let (builder, output_ids, external_reference_bindings) = {
            let destination = TracingContext::<V, O>::new();
            let builder = destination.builder().clone();
            let context = ReferenceDischargeContext::new_with_targets(destination.clone(), targets);
            let mut inputs = Vec::with_capacity(input_count);
            let mut discharged_allocations = Vec::new();
            let mut capture_allocations = vec![None; capture_count];
            for (input_index, input_type) in input_types.into_iter().enumerate() {
                let Ok(reference_type) = <&ReferenceType<P::Referent>>::try_from(&input_type) else {
                    inputs.push(ReferenceDischargeValue::Ordinary(destination.input(input_type)));
                    continue;
                };
                let reference_type = reference_type.clone();
                let source = ReferenceSource::from_flat_input_index(input_index, capture_count);
                let selected = context.selects_external(source);
                let carrier = if selected {
                    let state = destination.input(V::Type::from(reference_type.referent().clone()));
                    context.allocate_discharged(reference_type, state)?
                } else {
                    // An unselected external allocation keeps its reference-typed boundary position exactly as the
                    // source declared it, so the caller still supplies the reference, and every access to it replays
                    // verbatim.
                    context.bind_preserved(reference_type, destination.input(input_type))?
                };
                let allocation = carrier.expect_reference("an entry-boundary reference allocation")?.allocation_id();
                if selected {
                    discharged_allocations.push((source, allocation));
                }
                if input_index < capture_count {
                    capture_allocations[input_index] = Some(allocation);
                }
                inputs.push(carrier);
            }

            // The capture scope can only be established once the prefix has minted its allocations, and it is what lets
            // a nested region resolve the caller references it names through capture constants rather than through its
            // own boundary.
            let context = context.with_captures(ReferenceCaptureScope::new(capture_index_of, capture_allocations));

            let regions = [self];
            let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
            let outputs = driver.discharge_region(&context, 0, inputs)?;
            let mut output_ids = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| match output {
                    ReferenceDischargeValue::Ordinary(value) => value.atom_id(),
                    ReferenceDischargeValue::Reference(reference) => {
                        // A preserved reference survives in the rewritten program, so returning one returns its
                        // destination reference value. A discharged reference has no such value, because it became
                        // state. Returning an allocation is a use of it like any other, so its liveness is resolved
                        // against the environment rather than taken from the handle, which is what reports an
                        // allocation the program already consumed.
                        context.validate_live_allocation(reference.allocation_id())?;
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

            // A mutated external allocation publishes its final state as a hidden output. A read-only one publishes
            // nothing, which is what keeps a read-only program's boundary identical to its source boundary. A preserved
            // external allocation binds nothing at all (i.e., because it never became state, so there is no state for a
            // caller to supply or to write back).
            let mut external_reference_bindings = Vec::with_capacity(discharged_allocations.len());
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
                external_reference_bindings.push(ExternalReferenceBinding::new(source, output_index));
            }
            (builder, output_ids, external_reference_bindings)
        };

        let complete_output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program =
            builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; complete_output_count])?;
        PartialReferenceDischargeResult::new(program, capture_count, output_count, external_reference_bindings)
    }
}

impl<
    Capture: Value,
    V: CaptureConstant<Type = Capture::Type>,
    O: Operation<Type = Capture::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> ClosedProgram<Capture, V, O, Input, Output>
{
    /// Lifts this [`ClosedProgram`]'s captures into leading inputs and rewrites every [`Reference`](crate::Reference)
    /// as explicit immutable state.
    ///
    /// The returned [`ReferenceDischargeResult`] remains reference-free and records which leading inputs originated as
    /// captures rather than ordinary inputs. The concrete capture values remain owned by this closed program; their
    /// mutable contents are not embedded in the rewritten program.
    ///
    /// # Errors
    ///
    /// Returns errors produced while lifting the captures or performing capture-aware reference discharge.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        &self,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        Capture::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t Capture::Type>,
    {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_in_capture_lifted_program::<P>(capture_count)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::captures::CaptureReference;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::instructions::InstructionId;
    use crate::programs::references::discharge::tests::{
        ListIrType, ListIrValue, ListOperation, ListReferenceDischarge, ListType,
    };
    use crate::programs::references::types::ReferenceType;

    use super::*;

    /// Capture-constant family used by the capture-aware transform tests.
    type ListCapture = CaptureReference<ListIrType>;

    /// Closed list program used by the capture-aware transform tests.
    type ClosedListProgram = ClosedProgram<ListIrValue, ListCapture, ListOperation, Vec<ListCapture>, Vec<ListCapture>>;

    /// Builds a closed program whose attached region reads a reference solely through a capture constant.
    fn closed_list_program_with_nested_reference_capture() -> ClosedListProgram {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut callee_builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let captured_reference =
            callee_builder.add_constant(ListCapture::new(0, ListIrType::Reference(reference_type.clone())));
        let observed = callee_builder
            .add_instruction(ListOperation::Read, Vec::new(), vec![captured_reference], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let callee = builder.import_program(callee);
        let observed = builder.add_instruction(ListOperation::Call, vec![callee], Vec::new(), None).unwrap()[0];
        let source = builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();
        ClosedProgram::new(source, vec![ListIrValue::Reference(reference_type)]).unwrap()
    }

    #[test]
    fn test_program_discharge_references() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.capture_count(), 0);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2> = list.select %0
                    %3:list<2> = list.splice %0 %1
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_program_discharge_references_rejects_consumed_external_allocations() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose state must remain owned by the caller"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_discharge_references_in_capture_lifted_program() {
        // The attached region names the caller-owned reference only through a capture constant. Capture lifting moves
        // that reference into the entry input prefix while leaving the nested constant for the discharge transform to
        // resolve through its capture scope.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();

        let targets = lifted.reference_discharge_targets(1).unwrap();
        assert_eq!(targets, vec![ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })]);

        let discharged = lifted.discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1).unwrap();
        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_program_partially_discharge_references() {
        // The kernel-pipeline shape, in a universe that mentions no arrays: one caller-owned allocation is selected
        // and becomes threaded state, while the other survives as a reference the rewritten program still accesses
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

        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        let discharged = source.partially_discharge_references(0, &targets[..1]);
        let discharged = discharged.unwrap();

        // The selected allocation became an ordinary state input at its own boundary position and publishes its final
        // state as a hidden output; the preserved reference kept its reference type, so it binds nothing at all.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
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
            ReferenceDischargeResult::try_from(discharged).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_preserves_an_unselected_internal_target() {
        // An interior allocation is selectable in its own right, so a program can normalize its pipeline state while
        // the allocation a kernel body addresses is allocated, viewed, accessed, and consumed as a reference
        // throughout.
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
        let discharged = source.clone().partially_discharge_references(0, &[]);
        let discharged = discharged.unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
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
        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![ReferenceDischargeTarget::Internal {
                instruction: InstructionId::new(source.entry_region_ref().id(), 0),
                output_index: 0,
            }],
        );
        let selected = source.clone().partially_discharge_references(0, targets.as_slice());
        let selected = ReferenceDischargeResult::try_from(selected.unwrap()).unwrap();
        let full = source.discharge_references(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
    }

    #[test]
    fn test_program_partially_discharge_references_allows_consuming_a_preserved_external_allocation() {
        // A preserved reference has no external binding to describe: the program keeps the consuming operation, and
        // the caller passes its reference to that operation directly, so partial discharge preserves the source
        // program's consumption.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.freeze %0
                in (%1)"},
        );
        // Returning the allocation afterwards is a use of it like any other, so the consumed allocation is reported at
        // the output that names it rather than published as a stale reference.
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

        // An allocation rendering embeds the identity of the environment that minted it, which is process-global,
        // so the assertion pins everything around it.
        let error = source.partially_discharge_references(0, &[]).unwrap_err();
        let ProgramError::MalformedProgram(message) = &error else {
            panic!("expected a malformed-program rejection but got {error:?}");
        };
        assert!(message.starts_with("reference discharge accessed consumed reference allocation "), "{message}");
        assert!(message.ends_with(":0"), "{message}");
    }

    #[test]
    fn test_program_partially_discharge_references_threads_a_preserved_allocation_beside_discharged_state() {
        // A structured boundary carries both kinds of allocation at once: a discharged carry crosses as immutable
        // state and is widened with a published successor, while a preserved carry crosses as the reference it already
        // is, at its own declared operand position, and widens nothing at all.
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

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.partially_discharge_references(0, &targets[..1]).unwrap();

        // The selected allocation's entering state occupies its own operand position and its successor is appended as
        // a published output; the preserved reference's operand position still carries a reference, and the rebuilt
        // callee performs the read on it exactly as the source did.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
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
    fn test_program_partially_discharge_references_validates_targets_against_the_program() {
        // The targets are checked before anything is replayed, so a target this program does not expose is reported
        // against the program rather than surfacing later as an allocation that never appeared.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source
                .partially_discharge_references(
                    0,
                    &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 3 })],
                )
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 3, which is not selectable in this program"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_in_capture_lifted_program() {
        // Selecting nothing preserves the capture as a reference and explicitly threads it into the nested region.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();
        let discharged = lifted
            .partially_discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1, &[])
            .unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:ref<list<2>> .
                        let %1:list<2> = list.read %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_closed_program_discharge_references() {
        let closed = closed_list_program_with_nested_reference_capture();
        let discharged = closed.discharge_references::<ListReferenceDischarge>().unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }
}
