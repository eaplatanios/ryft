use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use crate::contexts::{Context, StagingContext};
use crate::macros::check_count;
use crate::parameters::Placeholder;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::regions::{RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::Typed;
use crate::programs::values::Value;

use crate::programs::references::types::ReferenceType;

use super::super::super::policies::ReferenceDischargePolicy;
use super::super::rules::{ReferenceDischargeDriver, discharge_preserved_access};
use super::super::{
    ReferenceAllocationHandle, ReferenceDischargeContext, ReferenceDischargeValue, ReferenceDischargeableOperation,
};
use super::analysis::nested_capture_scope;
use super::boundaries::{
    ReferenceDischargeRegionDestination, ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork,
};

// TODO(eaplatanios): Review this module.

/// [`ReferenceDischargeDriver`] scoped to one [`Operation`] application. It borrows the application's complete region
/// driver, which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees
/// without materializing a combined region collection.
pub struct RecursiveReferenceDischargeDriver<'r, D> {
    /// Application-scoped [`RegionDriver`].
    driver: &'r D,

    /// Source coordinate of the application, or [`None`] for an application that replays no instruction.
    instruction: Option<InstructionId>,
}

impl<'r, D> RecursiveReferenceDischargeDriver<'r, D> {
    /// Creates a new [`RecursiveReferenceDischargeDriver`].
    ///
    /// # Parameters
    ///
    ///   - `driver`: Application-scoped [`RegionDriver`] exposing the attached regions.
    ///   - `instruction`: Source coordinate of the application, or [`None`] for an application that replays no
    ///     instruction.
    #[inline]
    pub const fn new(driver: &'r D, instruction: Option<InstructionId>) -> Self {
        Self { driver, instruction }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveReferenceDischargeDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

// Recursive discharge replays the attached region one instruction at a time against the live environment, so an allocation
// created outside the region stays the same allocation inside it and the region's own allocations are ordinary new allocations.
// Constants lift into the destination through the parent, exactly as they do at the top level.
//
// The nested obligation is the one this crate's other structural transforms already carry: rebuilding a region needs
// this universe's operations to discharge into a fresh trace of the same universe as well as into the live
// destination. The requested reference type of a threaded allocation crosses that boundary, so the two policy
// instantiations must agree on their referent type system. Both obligations are stated here rather than on the
// per-operation rules on purpose. A rule that stated them would make the enum dispatcher's obligation graph circular,
// because the dispatcher's own predicate for a structured payload would then demand that the whole enum discharge
// into the destination whose dischargeability is what the graph is trying to establish.
impl<C, P, D> ReferenceDischargeDriver<C, P> for RecursiveReferenceDischargeDriver<'_, D>
where
    C: Context<
        Operation: ReferenceDischargeableOperation<C, P>
                       + ReferenceDischargeableOperation<ReferenceDischargeRegionDestination<C>, P>,
    >,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            ReferenceDischargeRegionDestination<C>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
    C::Type: From<<P as ReferenceDischargePolicy<C>>::Referent>
        + From<ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>,
    for<'t> &'t ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>: TryFrom<&'t C::Type>,
    D: RegionDriver<C::Constant, C::Operation>,
{
    #[inline]
    fn instruction(&self) -> Option<InstructionId> {
        self.instruction
    }

    #[inline]
    fn discharge_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_region_instructions(context, self.region(index)?, inputs)
    }

    #[inline]
    fn discharge_region_program(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError> {
        let region = self.region(index)?;
        type Destination<C> = ReferenceDischargeRegionDestination<C>;

        // Rebuild the source region in a fresh trace with a fresh allocation environment. The caller and fork can
        // communicate only through the boundary described below: neither side can accidentally retain a handle or
        // value belonging to the other environment.

        check_count!("input", boundary.declared_input_allocations(), region.input_ids().len(), ProgramError);
        let source_input_types = region.input_types();
        let source_input_count = source_input_types.len();
        let source_output_count = region.output_ids().len();
        if boundary.input_insertion() > source_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state inputs at {} but region `{}` declares {source_input_count} \
             inputs",
                boundary.input_insertion(),
                region.id(),
            )));
        }
        if boundary.output_insertion() > source_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state outputs at {} but region `{}` declares {source_output_count} \
             outputs",
                boundary.output_insertion(),
                region.id(),
            )));
        }

        // Added state may not land inside the region's own capture prefix: the rebuilt region keeps the prefix length its
        // operation declares, so a state input placed before the end of it would silently renumber the captures the
        // rebound operation still names.
        let capture_input_count = boundary.capture_input_count().unwrap_or(0);
        if boundary.input_insertion() < capture_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state inputs at {} but region `{}` declares a capture prefix of \
             {capture_input_count}",
                boundary.input_insertion(),
                region.id(),
            )));
        }

        // Every carrier, the fork's context, and the destination itself stay inside this block, because recovering the
        // rebuilt program below requires unique ownership of the destination's builder.
        let destination = Destination::<C>::new();
        let builder = destination.builder().clone();
        let (output_ids, output_allocations, mutated_allocations) = {
            // The fork inherits its caller's targets, because a target names a source coordinate that means the same thing
            // wherever the replay reaches it: an unselected allocation inside a rebuilt region survives there exactly as
            // it would have in the caller's own body.
            let fork = ReferenceDischargeContext::<Destination<C>, P>::new_with_targets(
                destination.clone(),
                context.targets.clone(),
            );

            let mut declared_allocations = BTreeSet::new();
            declared_allocations.extend(boundary.declared_input_allocations().iter().copied().flatten());
            let mut added_allocations = BTreeSet::new();
            for allocation in boundary.added_input_allocations() {
                if declared_allocations.contains(allocation) || !added_allocations.insert(*allocation) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge adds {allocation} to region `{}` more than once",
                        region.id(),
                    )));
                }
            }

            // Caller and fork allocations live in different environments, so explicit directional maps are the only
            // correspondence between them. Repeated declared positions may intentionally alias one caller allocation, while
            // synthesized state positions were already proven unique (and disjoint from the declared allocations) above. The
            // caller-to-fork map is ordered because the mutation-reconciliation loop below iterates it fallibly, and
            // diagnostics must not depend on hash order.
            let mut caller_to_fork = BTreeMap::<ReferenceAllocationHandle, ReferenceAllocationHandle>::new();
            let mut fork_to_caller = HashMap::<ReferenceAllocationHandle, ReferenceAllocationHandle>::new();
            let mut thread =
            |allocation: ReferenceAllocationHandle| -> Result<ReferenceDischargeValue<Destination<C>, P>, ProgramError> {
                let r#type = context.allocation_reference_type(allocation)?;
                let discharged = context.allocation_is_discharged(allocation)?;
                let input_type = if discharged {
                    <Destination<C> as crate::contexts::Domain>::Type::from(r#type.referent().clone())
                } else {
                    <Destination<C> as crate::contexts::Domain>::Type::from(r#type.clone())
                };
                let input = destination.input(input_type);
                if let Some(forked) = caller_to_fork.get(&allocation).copied() {
                    return fork.allocation_handle(forked);
                }
                let carrier = if discharged {
                    fork.allocate_discharged(r#type, input)?
                } else {
                    fork.bind_preserved(r#type, input)?
                };
                let forked = carrier.expect_reference("a threaded region allocation")?.allocation();
                caller_to_fork.insert(allocation, forked);
                fork_to_caller.insert(forked, allocation);
                Ok(carrier)
            };

            // Only the declared positions are replayed. An added input occupies a destination boundary position and a
            // caller operand position, but the source region's body never named it and so cannot consume it. A preserved
            // allocation occupies an added position only when an inherited capture is returned without a declared operand.
            let mut declared = Vec::with_capacity(source_input_count);
            for position in 0..=source_input_count {
                if position == boundary.input_insertion() {
                    for allocation in boundary.added_input_allocations() {
                        thread(*allocation)?;
                    }
                }
                let Some(allocation) = boundary.declared_input_allocations().get(position) else {
                    continue;
                };
                let source_type = &source_input_types[position];
                declared.push(match allocation {
                None => {
                    if <&ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>::try_from(source_type).is_ok() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge declares reference input {position} of region `{}` without an allocation",
                            region.id(),
                        )));
                    }
                    ReferenceDischargeValue::Ordinary(destination.input(source_type.clone()))
                }
                Some(allocation) => {
                    let Ok(source_reference_type) =
                        <&ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>::try_from(source_type)
                    else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge assigns {allocation} to ordinary input {position} of region `{}`",
                            region.id(),
                        )));
                    };
                    let allocation_type = context.allocation_reference_type(*allocation)?;
                    if &allocation_type != source_reference_type {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge assigns {allocation} of type `{allocation_type}` to input {position} of region \
                             `{}` with reference type `{source_reference_type}`",
                            region.id(),
                        )));
                    }
                    thread(*allocation)?
                }
            });
            }

            // The rebuilt region discharges under a scope naming only fork allocations, so the isolation the fork mints
            // holds for capture-scoped references too: a region declaring its own capture prefix reads that prefix off its
            // threaded declared inputs, and every other region inherits the caller's scope mapped onto the fork allocations
            // standing for its caller allocations. A caller allocation the boundary did not thread binds nothing.
            // Discharged capture accesses and outputs enter as state, while a preserved capture-scoped output enters as its
            // destination reference, so both states mint fork allocations before the inherited scope is established.
            let inherited = context.captures().with_allocations(
                context
                    .captures()
                    .allocations()
                    .iter()
                    .map(|allocation| allocation.and_then(|caller| caller_to_fork.get(&caller).copied()))
                    .collect(),
            );
            let fork_declared_allocations = declared
                .iter()
                .map(|input| match input {
                    ReferenceDischargeValue::Ordinary(_) => None,
                    ReferenceDischargeValue::Reference(reference) => Some(reference.allocation()),
                })
                .collect::<Vec<_>>();
            let fork = fork.with_captures(nested_capture_scope(
                boundary.capture_input_count(),
                fork_declared_allocations.as_slice(),
                &inherited,
                region.id(),
            )?);

            let outputs = discharge_region_instructions(&fork, region, declared)?;
            check_count!("output", outputs, source_output_count, ProgramError);

            let mut output_ids = Vec::with_capacity(source_output_count + boundary.added_output_allocations().len());
            let mut output_allocations = Vec::with_capacity(source_output_count);
            for position in 0..=source_output_count {
                if position == boundary.output_insertion() {
                    for allocation in boundary.added_output_allocations() {
                        let forked = caller_to_fork.get(allocation).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "reference discharge publishes {allocation} from region `{}` without threading it in",
                                region.id(),
                            ))
                        })?;
                        output_ids.push(fork.allocation_value(forked)?.atom_id()?);
                    }
                }
                let Some(output) = outputs.get(position) else {
                    continue;
                };
                match output {
                    ReferenceDischargeValue::Ordinary(value) => {
                        output_allocations.push(None);
                        output_ids.push(value.atom_id()?);
                    }
                    ReferenceDischargeValue::Reference(reference) => {
                        // A reference-typed region output publishes its allocation at that exact position — a discharged reference's
                        // current state, a preserved reference's own reference — and the owning rule maps it back onto the
                        // caller allocation through `output_allocations`. An allocation the caller did not thread has nowhere to be
                        // published, which is how a region-local allocation is stopped from escaping through the boundary.
                        let caller = fork_to_caller.get(&reference.allocation()).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "reference discharge cannot publish {} from region `{}`, whose caller did not thread \
                             that allocation",
                                reference.allocation(),
                                region.id(),
                            ))
                        })?;

                        // The published value denotes the complete stored value, so only a handle with complete-value provenance may
                        // cross. A view returned from a region has to be re-derived by whoever needs it, exactly as one
                        // passed into a region does.
                        let whole = fork.allocation_reference_type(reference.allocation())?;
                        if !reference.denotes_complete_value() {
                            return Err(ProgramError::MalformedProgram(format!(
                                "reference discharge cannot publish the derived view `{}` of {caller} from region `{}`, \
                             whose boundary carries the complete stored value `{whole}`",
                                reference.r#type(),
                                region.id(),
                            )));
                        }
                        output_allocations.push(Some(caller));
                        output_ids.push(match reference.preserved() {
                            Some(value) => value.atom_id()?,
                            None => fork.discharged_state(reference.allocation())?.atom_id()?,
                        });
                    }
                }
            }

            // Only threaded *state* can have been mutated. A preserved reference's writes replayed into the rebuilt region as
            // the operations the source performed, so there is no successor state for the caller to merge.
            let mut mutated_allocations = BTreeSet::new();
            for (caller, forked) in &caller_to_fork {
                if fork.allocation_is_discharged(*forked)? && fork.is_mutated(*forked)? {
                    mutated_allocations.insert(*caller);
                }
            }
            let mutated_allocations = mutated_allocations.into_iter().collect::<Vec<_>>();
            (output_ids, output_allocations, mutated_allocations)
        };
        drop(destination);

        let input_count = source_input_count + boundary.added_input_allocations().len();
        let output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
        Ok(ReferenceRegionDischargeFork { program, output_allocations, mutated_allocations })
    }
}

/// Replays one region's instructions against the live environment of `context`, binding their rewritten work through
/// the destination that context already owns.
///
/// This is the inlining replay both [`ReferenceDischargeDriver::discharge_region`] and the region fork use: the fork
/// differs only in which context it hands over, not in how a region's instructions are discharged.
///
/// # Parameters
///
///   - `context`: Active discharge context whose environment the replay observes and mutates.
///   - `region`: Source region being replayed.
///   - `inputs`: Carriers supplied to the region's boundary, in boundary order.
fn discharge_region_instructions<C, P>(
    context: &ReferenceDischargeContext<C, P>,
    region: RegionRef<'_, C::Constant, C::Operation>,
    inputs: Vec<ReferenceDischargeValue<C, P>>,
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<
        Operation: ReferenceDischargeableOperation<C, P>
                       + ReferenceDischargeableOperation<ReferenceDischargeRegionDestination<C>, P>,
    >,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            ReferenceDischargeRegionDestination<C>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
    C::Type: From<<P as ReferenceDischargePolicy<C>>::Referent>
        + From<ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>,
    for<'t> &'t ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>: TryFrom<&'t C::Type>,
{
    let mappings = RegionReplayMappings::new();
    let mut instruction_index = 0;
    region.interpret_with(
        inputs,
        |_, constant| lift_constant::<C, P>(context, constant.clone()),
        |instruction, instruction_inputs| {
            let position = InstructionId::new(region.id(), instruction_index);
            instruction_index += 1;
            // Run the complete rewrite of one application — the preserved-access replay included — inside the source
            // instruction's recorded origin, so every staged instruction records where it came from. Rules stage
            // their rewritten work through the destination parent, which is where the provenance state lives.
            context.parent().invoke_with_provenance_origin(instruction.provenance().clone(), || {
                if instruction.regions().is_empty() {
                    let operation = instruction.operation();
                    let semantics = operation.reference_semantics();

                    // A region-free operation that only accesses references can replay verbatim when every reference
                    // it accesses is preserved. Operations that do not access references need their ordinary rule,
                    // while operations that produce references need their own rule to register the resulting handles.
                    if !semantics.inputs().is_empty() && semantics.outputs().is_empty() {
                        let mut consumed = Vec::new();
                        let mut accesses_only_preserved_references = true;
                        for access in semantics.inputs() {
                            let Some(ReferenceDischargeValue::Reference(reference)) =
                                instruction_inputs.get(access.input_index())
                            else {
                                accesses_only_preserved_references = false;
                                break;
                            };
                            if reference.preserved().is_none() {
                                // Mixed preserved/discharged accesses belong to the operation's discharge rule, which
                                // can reject or rewrite them with full knowledge of the operation's semantics.
                                accesses_only_preserved_references = false;
                                break;
                            }
                            if access.mode().is_consuming() {
                                consumed.push(reference);
                            }
                        }

                        if accesses_only_preserved_references {
                            // Re-run inference over the carriers' current types before binding the unchanged operation.
                            // This preserves the operation's own operand-relationship diagnostics instead of allowing a
                            // destination binding failure to obscure them.
                            let input_types =
                                instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                            operation.infer_output_types(input_types.as_slice(), &[])?;
                            let outputs = discharge_preserved_access(operation, context, instruction_inputs)?;

                            // A consuming access invalidates its preserved handles only after replay succeeds. This
                            // keeps a failed destination bind from changing the discharge environment.
                            for reference in consumed {
                                context.unbind_preserved(reference)?;
                            }
                            return Ok(outputs);
                        }
                    }
                }
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &mappings)?;
                let driver = RecursiveReferenceDischargeDriver::new(&regions, Some(position));
                instruction.operation().discharge_references(context, &driver, instruction_inputs)
            })
        },
    )
}

/// Lifts one stored program constant into a discharge carrier.
///
/// A reference-typed constant resolves through the active [`ReferenceCaptureScope`], which is how a capture-lifted
/// program's nested regions name their caller's references: the constant denotes the allocation that capture position
/// already binds, so it yields that allocation's complete-value handle rather than a second allocation of its own.
///
/// A reference-typed constant that no scope resolves is rejected rather than lifted. Reference discharge threads allocations
/// through the environment it owns, and such a reference belongs to no allocation: it never entered through an input, a
/// capture binding, or an allocation, so nothing in the environment describes it. Wrapping it as an ordinary value
/// instead would let it survive into the destination and silently break the reference-freedom guarantee of
/// [`ReferenceDischargeResult`].
///
/// # Parameters
///
///   - `context`: Active discharge context, supplying both the capture scope and the destination that lifts the
///     constant.
///   - `constant`: Stored program constant being lifted.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when a reference-typed constant resolves to no allocation, or resolves to a
/// allocation whose reference type is not the one the constant declares, and propagates the destination's own lift error.
fn lift_constant<C: Context, P: ReferenceDischargePolicy<C>>(
    context: &ReferenceDischargeContext<C, P>,
    constant: C::Constant,
) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
where
    for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
{
    let constant_type = constant.r#type();
    if let Ok(r#type) = <&ReferenceType<P::Referent>>::try_from(constant_type.as_ref()) {
        let Some(allocation) = context.captures().resolve(&constant) else {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot lift a constant of reference type `{type}`; a reference enters a program \
                 through an input, a capture binding, or an allocation",
            )));
        };

        // A capture constant names the complete stored value its position binds, so a narrower declared type would silently
        // widen to the allocation's own value where the constant is used.
        let bound = context.allocation_reference_type(allocation)?;
        if r#type != &bound {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge resolved a capture constant of reference type `{type}` to {allocation}, which carries \
                 the reference type `{bound}`",
            )));
        }
        return context.allocation_handle(allocation);
    }
    Ok(ReferenceDischargeValue::Ordinary(context.parent().lift(constant)?))
}

#[cfg(test)]
mod tests {

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;

    use crate::programs::instructions::{Instruction, InstructionId};

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{EmptyRegionDriver, RegionId};

    use crate::programs::{
        RecursiveReferenceDischargeDriver, ReferenceDischargeDriver, ReferenceRegionDischargeBoundary,
        ReferenceRegionStateInsertion,
    };
    use crate::tracing::TracingContext;

    use super::super::super::ReferenceCaptureScope;
    use super::*;

    #[test]
    fn test_reference_discharge_rejects_reference_typed_constants() {
        // A reference stored as a program constant belongs to no allocation, so lifting it as an ordinary value would let it
        // survive into the destination and break the reference-freedom guarantee. Both lifting paths reject it.
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let stored = builder.add_constant(ListIrValue::Reference(reference_type.clone()));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![stored], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let rejection = ProgramError::MalformedProgram(
            "reference discharge cannot lift a constant of reference type `ref<list<2>>`; a reference enters a \
             program through an input, a capture binding, or an allocation"
                .to_string(),
        );
        assert_eq!(lift_constant(&context, ListIrValue::Reference(reference_type)).err(), Some(rejection.clone()));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(driver.discharge_region(&context, 0, Vec::new()), Err(rejection));
    }

    #[test]
    fn test_reference_discharge_lifts_a_capture_scoped_constant_as_its_bound_allocation() {
        // A capture-lifted program names its caller's references through constants, and such a constant denotes the
        // allocation that capture position already binds rather than a second allocation of its own.
        let pair = ReferenceType::new(ListType { length: 2 });
        let triple = ReferenceType::new(ListType { length: 3 });
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context.allocate_discharged(pair.clone(), ListIrValue::List(vec![1, 2])).unwrap();
        let allocation = allocated.expect_reference("the captured allocation").unwrap().allocation();
        let scoped = context
            .with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]));

        let lifted = lift_constant(&scoped, ListIrValue::Reference(pair.clone())).unwrap();
        let reference = lifted.expect_reference("the resolved capture").unwrap();
        assert_eq!(reference.allocation(), allocation);
        assert_eq!(reference.r#type(), &pair);
        assert_eq!(scoped.live_allocations(), vec![allocation]);

        // An ordinary constant is unaffected by the scope and lifts through the destination as usual.
        let ordinary = lift_constant(&scoped, ListIrValue::List(vec![3, 4])).unwrap();
        assert_eq!(ordinary, ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![3, 4])));

        // A capture position the scope does not bind keeps the ordinary reference-constant rejection.
        assert_eq!(
            lift_constant(&scoped, ListIrValue::Reference(triple.clone())).err(),
            Some(ProgramError::MalformedProgram(
                "reference discharge cannot lift a constant of reference type `ref<list<3>>`; a reference enters a \
                 program through an input, a capture binding, or an allocation"
                    .to_string(),
            )),
        );

        // A capture constant names the complete stored value its position binds, so a declared type the bound allocation does not
        // carry is reported rather than silently widened where the constant is used.
        let allocated = context.allocate_discharged(triple, ListIrValue::List(vec![1, 2, 3])).unwrap();
        let wider = allocated.expect_reference("the mismatched allocation").unwrap().allocation();
        let mismatched = scoped.with_captures(scoped.captures().with_allocations(vec![None, None, Some(wider)]));
        assert_eq!(
            lift_constant(&mismatched, ListIrValue::Reference(pair)).err(),
            Some(ProgramError::MalformedProgram(format!(
                "reference discharge resolved a capture constant of reference type `ref<list<2>>` to {wider}, which \
                 carries the reference type `ref<list<3>>`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_rebinds_the_capture_scope_in_fork_allocations() {
        // A region whose closure reaches a caller allocation through a capture constant declares no boundary position for
        // it, so the rule threads it as added state. The fork rebinds the caller's scope onto the fork allocation standing
        // for that caller allocation, which is what lets the rebuilt body resolve the very same constant against its own
        // isolated environment.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![captured], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the captured allocation").unwrap().allocation();
        let context = context
            .with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]));

        // The summary reports the capture-scoped access in caller-allocation terms, which is what sizes the boundary.
        let summary = context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[]).unwrap();
        assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![allocation]);
        assert!(!summary.is_mutated(allocation));
        assert_eq!(summary.output_allocations(), &[None]);

        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            ReferenceRegionStateInsertion::new(vec![allocation], 0),
            ReferenceRegionStateInsertion::new(Vec::new(), 0),
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.select %0
                in (%1)"},
        );
        assert_eq!(fork.output_allocations(), &[None]);
        assert_eq!(fork.mutated_allocations, []);

        // The caller environment is untouched: the fork read its own threaded copy of the state.
        assert_eq!(context.discharged_state(allocation), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_reference_discharge_drivers_report_their_replay_position() {
        let context = ListDischargeContext::new(ListDestination::new());
        let position = InstructionId::new(RegionId::new(0), 3);

        // A driver built without a source instruction reports none, and one built for a replayed instruction reports
        // exactly that coordinate.
        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::instruction(&EmptyRegionDriver),
            None,
        );
        let driver = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, Some(position));
        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::instruction(&driver),
            Some(position),
        );

        // Neither driver can serve a nested region, because neither has one.
        let no_regions = ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string());
        assert_eq!(EmptyRegionDriver.discharge_region(&context, 0, Vec::new()), Err(no_regions));
        assert_eq!(
            driver.discharge_region(&context, 0, Vec::new()),
            Err(ProgramError::MalformedProgram("region index 0 is out of range".to_string())),
        );
    }

    #[test]
    fn test_region_discharge_rejects_a_same_type_derived_allocation_output() {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(reference_type.clone()));
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 0, length: 2 }, Vec::new(), vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![view], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let allocation = allocated.expect_reference("the caller allocation").unwrap().allocation();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceRegionStateInsertion::new(Vec::new(), 1),
            ReferenceRegionStateInsertion::new(Vec::new(), 1),
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge cannot publish the derived view `ref<list<2>>` of {allocation} from region `{}`, \
                 whose boundary carries the complete stored value `ref<list<2>>`",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_isolates_the_caller_environment() {
        // A region that accumulates into the allocation it receives and returns that allocation unchanged, which is the shape a
        // structured rule threads state through.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_constant(ListIrValue::List(vec![10, 10]));
        builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, update], None)
            .unwrap();
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let destination = TracingContext::<ListIrValue, ListOperation>::new();
        let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(destination.clone());
        let state = destination.input(ListIrType::List(ListType { length: 2 }));
        let allocated = context.allocate_discharged(ReferenceType::new(ListType { length: 2 }), state).unwrap();
        let allocation = allocated.expect_reference("the caller allocation").unwrap().allocation();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceRegionStateInsertion::new(Vec::new(), 0),
            ReferenceRegionStateInsertion::new(Vec::new(), 1),
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();

        // The rebuilt region reports what it did in the caller's own terms, and the caller's environment is untouched:
        // the allocation is still unmutated and still holds the state it entered with.
        assert_eq!(fork.mutated_allocations, [allocation]);
        assert_eq!(fork.output_allocations(), &[Some(allocation)]);
        assert!(!context.is_mutated(allocation).unwrap());
        assert_eq!(context.discharged_state(allocation).unwrap().atom_id().unwrap(), AtomId::new(0));
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = const [10, 10]
                    %2:list<2> = list.select %0
                    %3:list<2> = list.add %2 %1
                    %4:list<2> = list.splice %0 %3
                in (%4)"},
        );

        // A replay that fails leaves the caller's environment exactly as it was and yields no values at all, because
        // the fork's result type carries none. The checked append rejects a read of a consumed family at
        // construction, so the failing program is assembled through the unchecked rebuild hatch.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![reference], None).unwrap();
        let failing = builder.add_variable(ListIrType::List(ListType { length: 2 }));
        builder.add_instruction_unchecked(Instruction::new(
            ListOperation::Read,
            vec![reference],
            vec![failing],
            Vec::new(),
        ));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![failing], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert!(matches!(
            driver.discharge_region_program(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.starts_with("reference discharge accessed consumed reference allocation "),
        ));
        assert!(!context.is_mutated(allocation).unwrap());
        assert_eq!(context.discharged_state(allocation).unwrap().atom_id().unwrap(), AtomId::new(0));
    }

    #[test]
    fn test_reference_discharge_region_program_rejects_duplicate_added_allocations() {
        let builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let program = builder.build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the added allocation").unwrap().allocation();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            ReferenceRegionStateInsertion::new(vec![allocation, allocation], 0),
            ReferenceRegionStateInsertion::new(Vec::new(), 0),
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {allocation} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_rejects_an_added_allocation_duplicating_a_declared_allocation() {
        // A repeated *declared* position deliberately aliases one caller allocation, but a synthesized state position must
        // never restate an allocation the boundary already declares: the rebuilt region would carry two boundary positions
        // for one state with no rule deciding which successor wins.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the declared allocation").unwrap().allocation();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceRegionStateInsertion::new(vec![allocation], 1),
            ReferenceRegionStateInsertion::new(Vec::new(), 0),
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {allocation} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_propagates_a_consumed_fork_allocation() {
        // This operation deliberately violates the generic contract: its summary claims no reference access, while
        // its discharge rule consumes the allocation. Fork sealing must report that consumed allocation instead of silently
        // omitting it from the mutation report.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen =
            builder.add_instruction(ListOperation::UnreportedFreeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
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
            vec![Some(allocation)],
            ReferenceRegionStateInsertion::new(Vec::new(), 1),
            ReferenceRegionStateInsertion::new(Vec::new(), 1),
        );

        assert!(matches!(
            driver.discharge_region_program(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.contains("reference discharge accessed consumed reference allocation"),
        ));
    }

    #[test]
    fn test_reference_discharge_region_program_inserts_added_state_at_its_boundary_position() {
        // Added state is what a region closure reaches without receiving it as a declared operand. No source construct
        // the interpreter currently accepts produces one — a reference reaches a region only through its boundary,
        // because a reference-typed constant is rejected outright — so the mechanics are exercised here directly,
        // against the boundary request a rule would make once capture-scoped references resolve.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let destination = TracingContext::<ListIrValue, ListOperation>::new();
        let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(destination.clone());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let accessed = context
            .allocate_discharged(reference_type.clone(), destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let accessed = accessed.expect_reference("the accessed allocation").unwrap().allocation();
        let carried = context
            .allocate_discharged(reference_type, destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let carried = carried.expect_reference("the carried allocation").unwrap().allocation();

        // The added input goes between the two declared inputs and the added output goes before the declared output,
        // which is the insertion arithmetic a scan's carry prefix depends on.
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![None, Some(accessed)],
            ReferenceRegionStateInsertion::new(vec![carried], 1),
            ReferenceRegionStateInsertion::new(vec![carried], 0),
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2>, %2:list<2> .
                let %3:list<2> = list.select %2
                    %4:list<2> = list.add %3 %0
                    %5:list<2> = list.splice %2 %4
                    %6:list<2> = list.select %5
                in (%1, %6)"},
        );

        // Only the allocation the closure actually reached is reported as mutated; the carried one passes through, which is
        // why a symmetric boundary can thread it without claiming the region wrote it.
        assert_eq!(fork.mutated_allocations, [accessed]);
        assert_eq!(fork.output_allocations(), &[None]);
    }
}
