use std::collections::BTreeSet;

use crate::contexts::Context;
use crate::macros::check_count;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;

use super::super::super::policies::ReferenceDischargePolicy;
use super::super::rules::ReferenceDischargeDriver;
use super::super::{ReferenceDischargeContext, ReferenceDischargeValue};
use super::ReferenceRegionSummary;
use super::boundaries::{ReferenceRegionDischargeBoundary, ReferenceRegionStateInsertion};

// TODO(eaplatanios): Review this module.

/// Rewrites one *positionally forwarding* region-carrying application so that the references its region closures
/// touch become explicit immutable state.
///
/// This is the shared rule body for the two structured shapes whose regions all mirror the operand list after a
/// constant leading offset and whose results are each region's own outputs: a condition, whose branches follow its
/// predicate, and a positional call, whose single callee follows nothing. Both widen the same way, so both reach it:
///
///   - the allocations every region closure touches are threaded in as operands appended after the declared ones, unless
///     they are already reference operands, in which case they thread at their own position;
///   - only the allocations some closure *mutates* are published back, as outputs appended after the declared ones. An allocation
///     the closures merely read needs no successor state, and pruning it is what keeps a read-only branch's boundary
///     identical to its source boundary;
///   - every attached region receives the identical state positions, so a rebuilt condition's branches keep agreeing
///     with each other. Only the capture prefix is read per region, because how many of a region's leading inputs are
///     its own captures is the operation's own per-region declaration.
///
/// # Parameters
///
///   - `operation`: Operation application being rewritten. It is replayed unchanged, because threading state past a
///     positional boundary changes only the boundary.
///   - `context`: Active discharge context owning the allocation environment.
///   - `driver`: Application-scoped driver supplying the attached regions.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///   - `leading_operand_count`: Number of leading operands that parameterize the operation itself rather than being
///     forwarded to its regions, which is one for a condition's predicate and zero for a positional call.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when the application has fewer operands than
/// `leading_operand_count`, when a leading operand is a live reference handle, when an attached region's boundary does
/// not forward the remaining operands positionally, when a reference operand names a derived view rather than a whole
/// allocation, when a region closure reaches an allocation that never entered the boundary or consumes one, when a region returns a
/// allocation its caller never threaded, when the attached regions disagree on which outputs denote references, or when a
/// region mutates an allocation the widening did not predict.
pub fn discharge_positional_region_operation<C, P, O, D>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
    leading_operand_count: usize,
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    C::Type: From<P::Referent>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
    D: ReferenceDischargeDriver<C, P>,
{
    let name = operation.name();
    if inputs.len() < leading_operand_count {
        return Err(ProgramError::MalformedProgram(format!(
            "operation `{name}` forwards its operands after {leading_operand_count} leading operands but the \
             application has {} operands",
            inputs.len(),
        )));
    }
    let (leading, forwarded) = inputs.split_at(leading_operand_count);
    for (index, input) in leading.iter().enumerate() {
        input.expect_ordinary(&format!("an ordinary leading operand {index} of `{name}`"))?;
    }
    let forwarded_allocations = forwarded
        .iter()
        .map(|operand| context.operand_allocation(operand, name))
        .collect::<Result<Vec<_>, _>>()?;

    // Every region forwards the same operands, so one summary of all of them decides one shared boundary. It is seeded
    // from the first region rather than from an empty summary, because merging keeps the receiver's declared output
    // allocations and an empty summary declares none.
    let region_count = driver.region_count();
    let mut summary: Option<ReferenceRegionSummary> = None;
    for index in 0..region_count {
        let region = driver.region(index)?;
        check_count!("input", region.input_ids(), forwarded.len(), ProgramError);
        let region_summary = context.region_summary(operation, index, region, forwarded_allocations.as_slice())?;
        summary = Some(match summary {
            Some(summary) => summary.merged(&region_summary),
            None => region_summary,
        });
    }
    let summary = summary.ok_or_else(|| {
        ProgramError::MalformedProgram(format!("operation `{name}` forwards its operands but attaches no regions"))
    })?;

    // A region that returns a discharged reference already publishes its final state at that output position, so only
    // a mutated state allocation absent from the declared outputs needs an appended output. Every reached allocation
    // absent from the operands gains an input: discharged captures cross as state, while preserved captures cross as
    // their destination references so the rebuilt region can bind its inherited capture scope.
    let represented = summary.output_allocations().iter().copied().flatten().collect::<BTreeSet<_>>();
    let threaded = context.threaded_state_allocations(&summary, name)?;
    let operand_allocations = forwarded_allocations.iter().copied().flatten().collect::<BTreeSet<_>>();
    let entering = summary.reached().filter(|allocation| !operand_allocations.contains(allocation)).collect::<Vec<_>>();
    let leaving = threaded
        .difference(&represented)
        .copied()
        .filter(|allocation| summary.is_mutated(*allocation))
        .collect::<Vec<_>>();

    // Every mutated allocation is published, whether through an appended output or through a declared reference output, and
    // that complete set is what the rebuilt regions are held to.
    let published = threaded.iter().copied().filter(|allocation| summary.is_mutated(*allocation)).collect::<Vec<_>>();

    let source_output_count = driver.region(0)?.output_ids().len();
    let declared_input_allocations = forwarded_allocations.clone();
    let mut regions = Vec::with_capacity(region_count);
    for index in 0..region_count {
        // Every region receives the same state positions, so a rebuilt condition's branches keep agreeing with each
        // other. Only the capture prefix is read per region, because it is the operation's own per-region declaration.
        let boundary = ReferenceRegionDischargeBoundary::new(
            operation,
            index,
            declared_input_allocations.clone(),
            ReferenceRegionStateInsertion::new(entering.clone(), forwarded.len()),
            ReferenceRegionStateInsertion::new(leaving.clone(), source_output_count),
        );
        let fork = driver.discharge_region_program(context, index, &boundary)?;
        fork.validate_predicted_mutations(published.as_slice(), name)?;
        fork.validate_predicted_output_allocations(summary.output_allocations(), name)?;
        regions.push(fork.into_program());
    }
    let output_allocations = summary.output_allocations();

    let mut operands = Vec::with_capacity(inputs.len() + entering.len());
    for input in inputs {
        operands.push(context.operand_value(input)?);
    }
    for allocation in &entering {
        operands.push(context.allocation_value(*allocation)?);
    }
    let outputs = context.parent().bind(operation.clone(), regions, operands.as_slice())?;
    check_count!("output", outputs, source_output_count + leaving.len(), ProgramError);

    // A declared output that denotes a reference is reported as the handle the caller already holds rather than as a
    // value. For a discharged reference that output carried its final state, which is merged back; for a preserved reference it
    // carried the reference itself, and there is nothing to merge. Appended outputs publish the remaining final
    // states.
    let mut results = Vec::with_capacity(source_output_count);
    for (position, output) in outputs.into_iter().enumerate() {
        if position >= source_output_count {
            context.set_discharged_state(leaving[position - source_output_count], output)?;
            continue;
        }
        match output_allocations[position] {
            Some(allocation) => {
                context.merge_boundary_state(&summary, &threaded, allocation, output)?;
                let forwarded = forwarded_allocations
                    .iter()
                    .position(|candidate| *candidate == Some(allocation))
                    .and_then(|position| forwarded.get(position).cloned());
                results.push(match forwarded {
                    Some(forwarded) => forwarded,
                    None => context.allocation_handle(allocation)?,
                });
            }
            None => results.push(ReferenceDischargeValue::Ordinary(output)),
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;

    use crate::programs::builders::ProgramBuilder;

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;

    use crate::programs::RecursiveReferenceDischargeDriver;

    use super::super::super::ReferenceCaptureScope;
    use super::*;

    #[test]
    fn test_reference_discharge_preserves_aliasing_between_repeated_declared_region_allocations() {
        // Both declared callee inputs denote one caller allocation. A write through the first must therefore be visible to
        // a read through the second even though the rebuilt boundary retains both declared positions.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let written = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_constant(ListIrValue::List(vec![7, 8]));
        callee_builder
            .add_instruction(ListOperation::Write, Vec::new(), vec![written, replacement], None)
            .unwrap();
        let snapshot =
            callee_builder.add_instruction(ListOperation::Read, Vec::new(), vec![observed], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee = builder.import_program(callee);
        let snapshot = builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Full discharge turns the shared allocation into state. The public snapshot and hidden final-state output both
        // observe the write, proving that the duplicate boundary position did not mint an independent fork allocation.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![ListIrValue::List(vec![1, 2])]),
            Ok(vec![ListIrValue::List(vec![7, 8]), ListIrValue::List(vec![7, 8])]),
        );

        // Partial discharge preserves the same alias as a reference. Both declared positions remain present in the
        // callee boundary, but its second input is unused and both accesses replay through the first canonical value.
        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.call %0 %0 [
                    callee={
                        lambda %0:ref<list<2>>, %1:ref<list<2>> .
                        let %2:list<2> = const [7, 8]
                            list.write %0 %2
                            %3:list<2> = list.read %0
                        in (%3)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_positional_region_discharge_recovers_a_returned_capture_scoped_allocation() {
        // This allocation reaches the region through its inherited capture scope, not through any forwarded operand. The
        // declared result must therefore be recovered from the context rather than from the empty operand list.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.expect_reference("the capture-scoped allocation").unwrap().allocation();
        let context = context
            .with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].expect_reference("the returned capture-scoped allocation").unwrap().allocation(),
            allocation
        );
        assert_eq!(
            context.read(results[0].expect_reference("the returned capture-scoped allocation").unwrap()),
            Ok(ListIrValue::List(vec![1, 2]),)
        );
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_positional_region_discharge_recovers_a_returned_preserved_capture_scoped_allocation() {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(reference_type.clone()));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let preserved = context.bind_preserved(reference_type, destination_reference.clone()).unwrap();
        let allocation = preserved.expect_reference("the preserved capture-scoped allocation").unwrap().allocation();
        let context = context
            .with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        let returned = results[0].expect_reference("the returned preserved capture-scoped allocation").unwrap();
        assert_eq!(returned.allocation(), allocation);
        assert_eq!(returned.preserved(), Some(&destination_reference));
        assert_eq!(context.operand_value(&results[0]), Ok(destination_reference));
    }

    #[test]
    fn test_reference_discharge_call_rule_threads_state_through_a_non_array_callee() {
        // The whole structured rewrite is universe-generic, so the prototype universe exercises it end to end: a
        // callee mutates the allocation it receives and returns only the previous snapshot, and discharge widens the call
        // with the final state the caller needs afterwards.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_reference =
            callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![callee_reference, update], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let previous =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![allocation, update], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2>, %3:list<2> = list.call %0 %1 [
                    callee={
                        lambda %0:list<2>, %1:list<2> .
                        let %2:list<2> = list.select %0
                            %3:list<2> = list.splice %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![ListIrValue::List(vec![1, 2]), ListIrValue::List(vec![7, 8])]),
            Ok(vec![ListIrValue::List(vec![1, 2]), ListIrValue::List(vec![7, 8])]),
        );
    }
}
