use crate::contexts::Context;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::references::discharge::transform::{
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
};
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::RegionRef;
use crate::programs::types::Typed;

// TODO(eaplatanios): Review this module.

/// Replays one reference-free operation application verbatim over its rewritten operands.
///
/// This is the shared rule body for every operation that touches no reference: it is the discharge counterpart of the
/// standard interpretation path, and it is where the conversion seam from the payload into the destination's operation
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
/// such an operation must implement its own
/// [`ReferenceDischargeableOperation`](crate::programs::references::ReferenceDischargeableOperation) rule.
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
            || driver.regions().any(RegionRef::contains_references_in_closure);
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
            input.try_as_value(&format!("a value operand {input_index} of `{}`", operation.name())).cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), regions, values.as_slice())?;
    Ok(outputs.into_iter().map(ReferenceDischargeValue::Value).collect())
}

/// Replays one access to a *preserved* allocation verbatim into the destination.
///
/// A preserved reference survives partial reference discharge as a reference value of the destination universe, so the
/// honest rewrite of an access to it is no rewrite at all: the operation is bound again, over the exact destination
/// reference value each handle denotes, and its results are the destination's own. The dispatch path owns the replay
/// for every region-free, access-only application, so access rules never call this themselves; it remains public for
/// a downstream rule that declares reference outputs and still wants to replay a preserved access it performs, which
/// is also why it takes no driver: an access carries no regions.
///
/// Each reference operand's *liveness* is checked against the environment rather than assumed from the handle, while
/// the destination value it contributes comes from the reference itself, which is the only place a view's exact
/// value lives. Replaying reproduces the source's own operation, which the destination is free to reject later, but a
/// use-after-consume is discharge's own invariant and belongs at the access that violates it.
///
/// A reference-typed result is rejected rather than wrapped. The environment would have no allocation for it, so it could
/// later cross a boundary or reach an access as an untracked value; an operation that produces a reference owns that
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
/// Returns [`ProgramError::MalformedProgram`] when an operand's allocation is no longer live, when an operand denotes a
/// discharged reference, which has no destination reference value, or when a result is reference-typed, and propagates the
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
    for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
{
    let values = inputs
        .iter()
        .map(|input| match input {
            ReferenceDischargeValue::Value(value) => Ok(value.clone()),
            ReferenceDischargeValue::Reference(reference) => match reference.preserved() {
                Some(value) => {
                    // The handle retains its destination value after consumption, so consult the environment before
                    // replaying it to distinguish a live preserved reference from a stale handle.
                    context.allocation_entry(reference.allocation_id())?;
                    Ok(value.clone())
                }
                None => Err(ProgramError::MalformedProgram(format!(
                    "reference discharge cannot replay `{}` over discharged {}, which has no destination reference \
                     value",
                    operation.name(),
                    reference.allocation_id(),
                ))),
            },
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), Vec::new(), values.as_slice())?;
    outputs
        .into_iter()
        .enumerate()
        .map(|(output_index, output)| {
            let output_type = output.r#type();
            if let Ok(r#type) = <&ReferenceType<P::Referent>>::try_from(output_type.as_ref()) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge replayed `{}` over a preserved reference, but its output {output_index} is the \
                     reference `{type}`; an operation that produces a reference owns that allocation and needs a reference \
                     discharge rule of its own",
                    operation.name(),
                )));
            }
            Ok(ReferenceDischargeValue::Value(output))
        })
        .collect()
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

    use crate::programs::RecursiveReferenceDischargeDriver;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_discharge_reference_free_operation_replays_reference_free_applications() {
        // The shared reference-free replay rule spends the conversion seam from the payload into the destination's
        // family, so an eager destination executes the replay and a staging destination would record it.
        let context = ListDischargeContext::new(ListDestination::new());
        let inputs = vec![
            ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Value(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Value(ListIrValue::List(vec![11, 22]))]),
        );

        // This helper accepts only value carriers because an operation that receives a live reference owns its own
        // discharge rule.
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().allocation_id();
        assert_eq!(
            discharge_reference_free_operation(
                &ListOperation::Add,
                &context,
                &EmptyRegionDriver,
                &[allocated, inputs[1].clone()],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a value operand 0 of `list.add` but received {allocation} ref<list<2>>",
            ))),
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
            Ok(vec![ReferenceDischargeValue::Value(ListIrValue::List(vec![2, 4]))]),
        );
    }

    #[test]
    fn test_reference_discharge_replays_preserved_accesses_inside_their_source_provenance() {
        // The dispatch path replays a preserved-allocation access itself, before any rule runs, and that replay must still
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

        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        let rendered = std::fmt::from_fn(|formatter| {
            preserved.program().render(formatter, 0, ProgramRenderingMode::WithProvenance)
        })
        .to_string();
        assert!(rendered.contains("; probe_write"), "write provenance lost:\n{rendered}");
        assert!(rendered.contains("; probe_read"), "read provenance lost:\n{rendered}");
    }

    #[test]
    fn test_discharge_preserved_access_replays_one_access_verbatim_into_the_destination() {
        // The shared preserved replay consumes each handle's own destination reference value and binds the source's
        // own operation over it, which is what makes an access to a surviving allocation no rewrite at all. It runs against
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
            vec![outputs[0].try_as_value("the replayed read result").unwrap().atom_id().unwrap()]
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

        // A replayed access that produces a reference would leave the environment without an allocation for it, so the
        // operation owning that allocation has to state its own rule instead.
        let staging = TracingContext::<ListIrValue, ListOperation>::new();
        let staged = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(staging.clone());
        let initial = ReferenceDischargeValue::Value(staging.input(ListIrType::List(referent.clone())));
        assert_eq!(
            discharge_preserved_access(&ListOperation::ReferenceNew, &staged, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "reference discharge replayed `list.reference_new` over a preserved reference, but its output 0 is the \
                 reference `ref<list<2>>`; an operation that produces a reference owns that allocation and needs a reference \
                 discharge rule of its own"
                    .to_string(),
            )),
        );

        // A discharged reference has no destination reference value at all, so it cannot be replayed over.
        let context = ListDischargeContext::new(ListDestination::new());
        let discharged = context.bind_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_allocation = discharged.try_as_reference("the discharged reference").unwrap().allocation_id();
        assert_eq!(
            discharge_preserved_access(&ListOperation::Read, &context, std::slice::from_ref(&discharged)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot replay `list.read` over discharged {discharged_allocation}, which has no \
                 destination reference value",
            ))),
        );
    }
}
