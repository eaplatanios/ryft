use crate::contexts::{Context, Domain};
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::regions::{EmptyRegionDriver, RegionDriver, RegionRef};
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

use super::super::policies::ReferenceDischargePolicy;
use super::regions::{ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork};
use crate::programs::references::types::ReferenceType;

use super::{ReferenceDischargeContext, ReferenceDischargeValue};

/// Replays one reference-free operation application verbatim over its rewritten operands.
///
/// This is the shared rule body for every operation that touches no reference: it is the discharge counterpart of the
/// ordinary interpretation path, and it is where the conversion seam from the payload into the destination's operation
/// family is spent. Because the operation is replayed rather than reinterpreted, the destination decides what
/// replaying means, so an eager destination executes it and a staging destination records it.
///
/// The precondition is *reference freedom*, not effect purity in the [`Effects`](crate::Effects) sense. An operation
/// with ordered or other effects replays here unchanged, because replaying it reproduces those effects in the
/// destination exactly as the source performed them; only a reference makes the rewrite the operation's own business.
///
/// A region-carrying application replays verbatim only when nothing in its attached closure touches a reference: its
/// regions are copied into the destination as they stand, which is exactly right for an operation whose regions
/// contain no state to thread. As soon as a reference does appear anywhere in that closure — or as an operand — the
/// application is rejected, because how a reference boundary widens is knowledge that belongs to the operation, and
/// such an operation must implement its own [`ReferenceDischargeableOperation`] rule.
///
/// # Parameters
///
///   - `operation`: Operation application being replayed.
///   - `context`: Active discharge context whose [`parent`](ReferenceDischargeContext::parent) binds the replay.
///   - `driver`: Application-scoped driver supplying any attached regions.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::UnsupportedOperation`] when a region-carrying application touches reference state, returns
/// [`ProgramError::MalformedProgram`] when a region-free application receives a live reference handle, and propagates
/// the destination's error from the replay itself.
pub fn discharge_reference_free_operation<C, P, O, D>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
    D: ReferenceDischargeDriver<C, P>,
{
    if driver.region_count() != 0 {
        let touches_references = inputs.iter().any(|input| matches!(input, ReferenceDischargeValue::Reference(_)))
            || driver.regions().any(region_closure_touches_references);
        if touches_references {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` carries reference state but has no reference discharge rule", operation.name()),
            });
        }
    }
    let regions = driver.regions().map(RegionRef::to_program).collect::<Vec<_>>();
    let values = inputs
        .iter()
        .enumerate()
        .map(|(input_index, input)| {
            input
                .expect_ordinary(&format!("an ordinary operand {input_index} of `{}`", operation.name()))
                .cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), regions, values.as_slice())?;
    Ok(outputs.into_iter().map(ReferenceDischargeValue::Ordinary).collect())
}

/// Returns whether `region` or any attached descendant contains a reference type or reference operation.
pub(in crate::programs::references::discharge) fn region_closure_touches_references<
    V: Value,
    O: Operation<Type = V::Type>,
>(
    region: RegionRef<'_, V, O>,
) -> bool {
    region.contains_atom_type_in_closure(Type::is_reference)
        || region
            .instructions_in_closure()
            .any(|(_, instruction)| !instruction.operation().reference_semantics().is_empty())
}

/// Replays one region-free, access-only application verbatim when every reference operand it accesses is preserved,
/// or returns [`None`] to hand the application to its own discharge rule.
///
/// This is the dispatch half of the preserved/discharged split: an operation whose reference semantics declare
/// accessed inputs and no reference outputs performs no rewrite over preserved roots — its honest rewrite is itself,
/// replayed through the destination — so the replay is owned by the dispatch path and rules see only discharged
/// accesses. Operations that declare reference outputs (allocations and view derivations) keep their own preserved
/// handling, because their outputs mint or derive handles. An application whose accessed reference operands mix
/// preserved and discharged roots still reaches its rule, which rejects it exactly as it would have rejected the
/// discharged half before this dispatch existed.
///
/// The operation's own inference is re-derived before the replay so that an operand-relationship mismatch is
/// reported with the same diagnostic a discharged access produces. Consuming accesses additionally unbind their
/// consumed roots after the replay so the environment stops handing them out.
pub(super) fn replay_preserved_access<C, P>(
    operation: &C::Operation,
    context: &ReferenceDischargeContext<C, P>,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Option<Vec<ReferenceDischargeValue<C, P>>>, ProgramError>
where
    C: Context,
    P: ReferenceDischargePolicy<C>,
{
    let semantics = operation.reference_semantics();
    if semantics.inputs().is_empty() || !semantics.outputs().is_empty() {
        return Ok(None);
    }
    let mut consumed = Vec::new();
    for access in semantics.inputs() {
        let Some(ReferenceDischargeValue::Reference(reference)) = inputs.get(access.input_index()) else {
            return Ok(None);
        };
        if reference.preserved().is_none() {
            return Ok(None);
        }
        if access.mode().is_consuming() {
            consumed.push(reference);
        }
    }
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    operation.infer_output_types(input_types.as_slice(), &[])?;
    let outputs = discharge_preserved_access(operation, context, inputs)?;
    for reference in consumed {
        context.unbind_preserved(reference)?;
    }
    Ok(Some(outputs))
}

/// Replays one access to a *preserved* root verbatim into the destination.
///
/// A preserved root survives partial reference discharge as an ordinary reference of the destination universe, so the
/// honest rewrite of an access to it is no rewrite at all: the operation is bound again, over the exact destination
/// reference value each handle denotes, and its results are the destination's own. The dispatch path owns the replay
/// for every region-free, access-only application, so access rules never call this themselves; it remains public for
/// a downstream rule that declares reference outputs and still wants to replay a preserved access it performs, which
/// is also why it takes no driver: an access carries no regions.
///
/// Each reference operand's *liveness* is checked against the environment rather than assumed from the handle, while
/// the destination value it contributes comes from the handle itself, which is the only place a derived view's exact
/// value lives. Replaying reproduces the source's own operation, which the destination is free to reject later, but a
/// use-after-consume is discharge's own invariant and belongs at the access that violates it.
///
/// A reference-typed result is rejected rather than wrapped. The environment would have no root for it, so it could
/// later cross a boundary or reach an access as an untracked value; an operation that derives a reference owns that
/// bookkeeping and must state it in its own rule, as the view primitives do.
///
/// # Parameters
///
///   - `operation`: Access being replayed.
///   - `context`: Active discharge context whose [`parent`](ReferenceDischargeContext::parent) binds the replay.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when an operand's root is no longer live, when an operand denotes a
/// discharged root, which has no destination reference value, or when a result is reference-typed, and propagates the
/// destination's error from the replay itself.
pub fn discharge_preserved_access<C, P, O>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
{
    let values = inputs
        .iter()
        .map(|input| match input {
            ReferenceDischargeValue::Ordinary(value) => Ok(value.clone()),
            ReferenceDischargeValue::Reference(reference) => match reference.preserved() {
                Some(value) => {
                    context.validate_live_root(reference.root())?;
                    Ok(value.clone())
                }
                None => Err(ProgramError::MalformedProgram(format!(
                    "reference discharge cannot replay `{}` over discharged {}, which has no destination reference \
                     value",
                    operation.name(),
                    reference.root(),
                ))),
            },
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), Vec::new(), values.as_slice())?;
    outputs
        .into_iter()
        .enumerate()
        .map(|(output_index, output)| {
            if let Some(r#type) = P::project_reference_type(output.r#type().as_ref()) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge replayed `{}` over a preserved root, but its output {output_index} is the \
                     reference `{type}`; an operation that derives a reference owns that root and needs a reference \
                     discharge rule of its own",
                    operation.name(),
                )));
            }
            Ok(ReferenceDischargeValue::Ordinary(output))
        })
        .collect()
}

/// Validates that one destination value denotes a reference of exactly `r#type`.
///
/// # Parameters
///
///   - `value`: Destination value offered as a preserved root's reference.
///   - `r#type`: Reference type the handle that will carry `value` exposes.
pub(super) fn validate_preserved_value<C: Domain, P: ReferenceDischargePolicy<C>>(
    value: &C::Value,
    r#type: &ReferenceType<P::Referent>,
) -> Result<(), ProgramError> {
    match P::project_reference_type(value.r#type().as_ref()) {
        Some(actual) if &actual == r#type => Ok(()),
        Some(actual) => Err(ProgramError::MalformedProgram(format!(
            "reference discharge preserved a root as `{actual}` but its handle exposes `{type}`",
        ))),
        None => Err(ProgramError::MalformedProgram(format!(
            "reference discharge preserved a root as `{}`, which is not a reference type",
            value.r#type(),
        ))),
    }
}

/// Validates that one immutable discharged state carries the lifted referent type of `r#type`.
pub(super) fn validate_discharged_value_type<C: Domain, P: ReferenceDischargePolicy<C>>(
    value: &C::Value,
    r#type: &ReferenceType<P::Referent>,
) -> Result<(), ProgramError> {
    let expected = P::lift_referent_type(r#type.referent().clone());
    if value.r#type().as_ref() != &expected {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge state has type `{}` but root `{}` requires `{expected}`",
            value.r#type(),
            r#type,
        )));
    }
    Ok(())
}

/// Provides one [`Operation`] application with its replay position and with recursive discharge of the
/// [`Region`](crate::Region)s attached to it.
///
/// [`RegionDriver`] supplies the structural region access, and this trait adds the three services that discharge
/// rules need on top of it. Region-free applications expose a region count of zero through the same contract.
pub trait ReferenceDischargeDriver<C: Domain, P: ReferenceDischargePolicy<C>>:
    RegionDriver<C::Constant, C::Operation>
{
    /// Returns the source coordinate of the operation application being discharged, or [`None`] when the application
    /// did not come from a replayed instruction.
    ///
    /// An allocation rule needs its own site to decide whether the caller selected it for discharge, so replaying a
    /// region through [`discharge_region`](Self::discharge_region) must supply the coordinate of every instruction it
    /// replays. Returning [`None`] declares the allocation unnameable by any
    /// [`ReferenceDischargeSite`](crate::programs::references::ReferenceDischargeSite) and therefore *always
    /// discharged*, silently ignoring
    /// the caller's partial selection. This method is deliberately required
    /// rather than defaulted: a replaying driver that forgot to forward its coordinate would otherwise disable
    /// partial discharge for its regions without any diagnostic.
    fn instruction(&self) -> Option<InstructionId>;

    /// Discharges the region at `index` over the provided carriers by re-entering the active discharge transform,
    /// binding the region's rewritten work directly into the destination program.
    ///
    /// The region is inlined under the *caller's* capture scope, which is correct for every region that inherits one.
    /// A region declaring its own leading capture prefix has to be rebuilt instead, through
    /// [`discharge_region_program`](Self::discharge_region_program), which establishes that prefix as the rebuilt
    /// region's own scope.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context whose environment the replayed region observes and mutates.
    ///   - `index`: Position of the attached region in operation-defined order.
    ///   - `inputs`: Carriers supplied to the region's boundary, in boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index` or when `inputs` does
    /// not describe the region's boundary, and propagates every failure the replayed rules raise.
    fn discharge_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>;

    /// Discharges the region at `index` against an *isolated* environment over a fresh destination of the same
    /// universe, and returns the sealed [`ReferenceRegionDischargeFork`] describing what that rebuilt region became.
    ///
    /// This is the transactional fork every structured rule builds on, and it is what
    /// [`discharge_region`](Self::discharge_region) is deliberately not: that service inlines a region's rewritten
    /// work into the live destination, which is right for an operation whose region is invoked in place and wrong for
    /// one whose region must survive as a region. The fork's environment contains exactly the roots `boundary` names,
    /// each entering as an ordinary value at its boundary position, so a region cannot reach a root its caller did not
    /// thread, and nothing it does can reach the caller's environment. The owning rule binds the rebuilt operation in
    /// its own context and merges the final states from the outputs of that binding.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context supplying the entering state, or the surviving reference, of every
    ///     root the boundary names.
    ///   - `index`: Position of the attached region in operation-defined order.
    ///   - `boundary`: Complete requested boundary of the rebuilt region.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index`, when `boundary` does
    /// not describe that region's declared boundary, when a root is threaded twice, when the region publishes a root
    /// its caller did not thread or publishes one through a derived view, and propagates every failure the rebuilt
    /// region's own rules raise.
    fn discharge_region_program(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError>;
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeDriver<C, P> for EmptyRegionDriver {
    // A region-free application replays no instruction, so its allocations carry no selectable coordinate.
    #[inline]
    fn instruction(&self) -> Option<InstructionId> {
        None
    }

    #[inline]
    fn discharge_region(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string()))
    }

    #[inline]
    fn discharge_region_program(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot rebuild a region".to_string()))
    }
}

/// Represents [`Operation`]s that can be discharged (i.e., rewritten so that the references they touch become
/// explicit immutable state).
///
/// The trait is parameterized by the destination [`Domain`] `C` that owns the rewritten values and by the
/// [`ReferenceDischargePolicy`] `P` naming the reference universe being discharged. Every rule receives the active
/// [`ReferenceDischargeContext`], which owns the root environment, plus a [`ReferenceDischargeDriver`] exposing the
/// application's replay position and attached regions.
///
/// Reference primitives implement their own rewrites: an allocation binds a fresh root, an access acts on the root's
/// current state through the policy's alias mechanics, and a freeze yields the current state and unbinds the root.
/// Structured operations implement their own boundary widening, because widening is a property of what the operation
/// does with its regions and therefore belongs to the operation. Everything else replays as-is over rewritten
/// operands. The system is consequently open over primitives: a third-party operation family participates by
/// implementing this trait, with no companion declaration surface beyond the generic
/// [`Operation::reference_semantics`] and region-provenance hooks it already implements.
///
/// Access rules see only *discharged* roots. When partial discharge preserves a root, the dispatch path replays every
/// region-free, access-only application over it verbatim through [`discharge_preserved_access`] before rule dispatch,
/// so an access rule never needs a preserved branch of its own. The exceptions own their preserved handling because
/// their outputs mint or derive handles: an allocation rule consults its replay position against the selection, and a
/// view rule derives through [`ReferenceDischargeContext::derive`], which replays the view over a preserved parent's
/// destination value.
///
/// `C` is bounded by [`Domain`] rather than [`Context`] for the same reason
/// [`InterpretableOperation`](crate::InterpretableOperation) is: the destination context's own binding contract is
/// established in terms of its operation family's rules, so reaching [`Context`] through this trait would make that
/// obligation recursive. Implementations bound `C` by the value and conversion capabilities their rewrite actually
/// uses, and higher-order rules request nested work through their driver rather than carrying a bound stating that
/// their own operation family is dischargeable, which is what keeps an operation enum's bound graph finite.
///
/// The super-trait is a plain [`Operation`] rather than `Operation<Type = C::Type>`, with the equality required per
/// function instead, matching [`BatchableOperation`](crate::BatchableOperation): the current trait solver cannot
/// discharge that projection equality at implementation heads whose context type is built from `Self`.
pub trait ReferenceDischargeableOperation<C: Domain, P: ReferenceDischargePolicy<C>>: Operation {
    /// Rewrites this operation application so that the references it touches become explicit immutable state, and
    /// returns the carriers its outputs produce.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context owning the root environment, through whose
    ///     [`parent`](ReferenceDischargeContext::parent) the rewritten work is bound.
    ///   - `driver`: Application-scoped driver exposing the replay position and any attached regions.
    ///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError`] when this application cannot be rewritten — because an operand is of the wrong kind,
    /// because the references its regions touch cannot be threaded through its boundary, or because the destination
    /// rejected the rewritten work.
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
    where
        Self: Operation<Type = C::Type>;
}

#[cfg(test)]
mod tests {

    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;

    use crate::programs::programs::ProgramRenderingMode;
    use crate::programs::provenance::{Provenance, ProvenanceScope};
    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::EmptyRegionDriver;

    use crate::programs::{RecursiveReferenceDischargeDriver, ReferenceDischargeDriver};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_reference_discharge_rules_unwrap_ordinary_carriers_and_reject_reference_handles() {
        // The operation's own discharge rule owns the unwrapping of ordinary carriers and the rejection diagnostic
        // for a live reference handle, so every operation-backed value capability inherits both without bespoke
        // delegation.
        let context = ListDischargeContext::new(ListDestination::new());
        let lhs = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 2]));
        let rhs = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![10, 20]));
        let sum = discharge_reference_free_operation(
            &ListOperation::Add,
            &context,
            &EmptyRegionDriver,
            &[lhs.clone(), rhs.clone()],
        )
        .unwrap();
        assert_eq!(sum, vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![11, 22]))]);

        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let root = allocated.expect_reference("the allocated root").unwrap().root();
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &EmptyRegionDriver, &[allocated, rhs]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an ordinary operand 0 of `list.add` but received {root} ref<list<2>>",
            ))),
        );

        // A region-free application replays no instruction, so an allocation rule that consults its replay position
        // sees `None` and treats the allocation as unconditionally discharged.
        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        let driver = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, None);
        ListOperation::ReferenceNew.discharge_references(&context, &driver, &[lhs]).unwrap();
        assert_eq!(OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone), vec![None]);
    }

    #[test]
    fn test_discharge_reference_free_operation_replays_reference_free_applications() {
        // The shared reference-free replay rule spends the conversion seam from the payload into the destination's
        // family, so an eager destination executes the replay and a staging destination would record it.
        let context = ListDischargeContext::new(ListDestination::new());
        let inputs = vec![
            ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![11, 22]))]),
        );

        // A region-carrying application whose closure touches a reference is rejected rather than replayed, because
        // how a reference boundary widens is knowledge that belongs to the operation. A reference-free closure instead
        // replays verbatim, which is what lets an operation whose regions hold no state keep this shared rule; here
        // the operation's own contract rejects the attachment, because `list.add` declares no region slots at all.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 1 })));
        let read = builder.add_instruction(ListOperation::Read, Vec::new(), vec![input], None).unwrap()[0];
        let stateful = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![read], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [stateful];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, inputs.as_slice()),
            Err(ProgramError::UnsupportedOperation {
                message: "`list.add` carries reference state but has no reference discharge rule".to_string(),
            }),
        );

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::List(ListType { length: 1 }));
        let reference_free = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [reference_free];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, inputs.as_slice()),
            Err(ProgramError::MalformedProgram(
                "operation `list.add` declares no region slots but 1 regions were attached".to_string(),
            )),
        );

        // An operation that does declare a region slot replays that region into the destination as it stands, which is
        // the whole rewrite for a region-carrying operation whose closure holds no state to thread.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_input = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let doubled = builder
            .add_instruction(ListOperation::Add, Vec::new(), vec![callee_input, callee_input], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [callee];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Call, &context, &driver, &inputs[..1]),
            Ok(vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![2, 4]))]),
        );
    }

    #[test]
    fn test_reference_discharge_replays_preserved_accesses_inside_their_source_provenance() {
        // The dispatch path replays a preserved-root access itself, before any rule runs, and that replay must still
        // happen inside the source instruction's recorded origin: provenance renders only under `WithProvenance`, so
        // no semantic rendering can catch an unwrapped replay dropping it.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let origin = |name: &str| Some(Provenance::scope(ProvenanceScope::new(name), Provenance::unknown()));
        builder
            .add_instruction(ListOperation::Write, Vec::new(), vec![reference, update], origin("probe_write"))
            .unwrap();
        let observed = builder
            .add_instruction(ListOperation::Read, Vec::new(), vec![reference], origin("probe_read"))
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let preserved = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap();
        let rendered = std::fmt::from_fn(|formatter| {
            preserved.program().render(formatter, 0, ProgramRenderingMode::WithProvenance)
        })
        .to_string();
        assert!(rendered.contains("; probe_write"), "write provenance lost:\n{rendered}");
        assert!(rendered.contains("; probe_read"), "read provenance lost:\n{rendered}");
    }

    #[test]
    fn test_reference_discharge_rules_thread_state_through_a_replayed_program() {
        // The program allocates one local root, narrows it to a composed view, accumulates into that view, replaces
        // it, adds the replaced and current selections, and finally freezes the whole root.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let root = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![root], None)
            .unwrap()[0];
        let update = builder.add_constant(ListIrValue::List(vec![10, 20]));
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let replacement = builder.add_constant(ListIrValue::List(vec![7, 8]));
        builder.add_instruction(ListOperation::Write, Vec::new(), vec![view, replacement], None).unwrap();
        let replaced =
            builder.add_instruction(ListOperation::Swap, Vec::new(), vec![view, replacement], None).unwrap()[0];
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![view], None).unwrap()[0];
        let total = builder.add_instruction(ListOperation::Add, Vec::new(), vec![replaced, snapshot], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![total, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        // Replaying the program through the region driver rewrites every reference primitive into ordinary state
        // threading, so the outputs are the values an eager reference execution would have produced.
        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        let context = ListDischargeContext::new(ListDestination::new());
        let input = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 2, 3, 4]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let outputs = driver.discharge_region(&context, 0, vec![input]).unwrap();
        assert_eq!(
            outputs,
            vec![
                ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![14, 16])),
                ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 7, 8, 4])),
            ],
        );

        // Every root the program created is gone once its `freeze` consumed it, so nothing leaks into the context.
        assert_eq!(context.live_roots(), Vec::new());

        // Replaying through the driver supplies every instruction's own source coordinate, which is what makes the
        // allocation selectable by a partial-discharge site.
        let observed = OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone);
        assert_eq!(observed.len(), 1);
        assert!(observed[0].is_some());
    }

    #[test]
    fn test_discharge_preserved_access_replays_one_access_verbatim_into_the_destination() {
        // The shared preserved replay consumes each handle's own destination reference value and binds the source's
        // own operation over it, which is what makes an access to a surviving root no rewrite at all. It runs against
        // a staging destination, because the eager destination of this universe declines to execute a reference
        // primitive and recording is what production discharge does anyway.
        let referent = ListType { length: 2 };
        let staging = TracingContext::<ListIrValue, ListOperation>::new();
        let builder = staging.builder().clone();
        let outputs = {
            let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(staging.clone());
            let preserved = context
                .bind_preserved(
                    ReferenceType::new(referent.clone()),
                    staging.input(ListIrType::Reference(ReferenceType::new(referent.clone()))),
                )
                .unwrap();
            let outputs =
                discharge_preserved_access(&ListOperation::Read, &context, std::slice::from_ref(&preserved)).unwrap();
            assert_eq!(outputs.len(), 1);
            vec![outputs[0].expect_ordinary("the replayed read result").unwrap().atom_id().unwrap()]
        };
        drop(staging);
        let program = Rc::try_unwrap(builder)
            .unwrap()
            .into_inner()
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.read %0
                in (%1)"},
        );

        // A replayed access that produces a reference would leave the environment without a root for it, so the
        // operation owning that root has to state its own rule instead.
        let staging = TracingContext::<ListIrValue, ListOperation>::new();
        let staged = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(staging.clone());
        let initial = ReferenceDischargeValue::Ordinary(staging.input(ListIrType::List(referent.clone())));
        assert_eq!(
            discharge_preserved_access(&ListOperation::ReferenceNew, &staged, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "reference discharge replayed `list.reference_new` over a preserved root, but its output 0 is the \
                 reference `ref<list<2>>`; an operation that derives a reference owns that root and needs a reference \
                 discharge rule of its own"
                    .to_string(),
            )),
        );

        // A discharged root has no destination reference value at all, so it cannot be replayed over.
        let context = ListDischargeContext::new(ListDestination::new());
        let discharged =
            context.allocate_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_root = discharged.expect_reference("the discharged root").unwrap().root();
        assert_eq!(
            discharge_preserved_access(&ListOperation::Read, &context, std::slice::from_ref(&discharged)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot replay `list.read` over discharged {discharged_root}, which has no \
                 destination reference value",
            ))),
        );
    }
}
