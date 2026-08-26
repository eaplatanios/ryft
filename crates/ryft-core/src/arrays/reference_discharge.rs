//! Backend-neutral discharge of array references and their views into explicit immutable array state.
//!
//! [`Program::discharge_references`] rewrites the reference state of an array-IR program into ordinary array SSA
//! values by interpreting it under [`ArrayReferenceDischarge`], this universe's [`ReferenceDischargePolicy`]. Index
//! and static unit-stride slice views lower through canonical slice, reshape, and update-slice operations.
//! Conditions, loops, scans, and calls widen their own boundaries with explicit immutable root state; derived views
//! must be recreated inside each attached region. The result keeps the original public outputs as a prefix and
//! appends one hidden final-state output for every mutated external root.
//!
//! An operation without a [`ReferenceDischargeableOperation`] rule of its own conservatively rejects reference state
//! anywhere in its attached-region closures — including state that is allocated, mutated, and consumed entirely
//! inside the region. Today that covers `shard_map`, rematerialization, linear-call, and custom-derivative carriers,
//! for each of which threading state through the region has no defined meaning rather than merely being
//! unimplemented.
//!
//! # Transform Boundary
//!
//! Discharge is the only supported route from local mutable state into generic transforms. Partial evaluation,
//! batching, forward- and reverse-mode differentiation, and rematerialization first prove that every root is local,
//! discharge the complete program, and then transform the reference-free result. The resulting behavior is the same
//! as transforming an explicitly immutable state-passing program. External public or captured roots are rejected by
//! these adapters: automatic differentiation of caller-owned state, mapped/shared reference batching, and
//! externally stateful rematerialization have no implicit semantics. Custom-derivative rule regions reject reference
//! state independently rather than inheriting a derivative for mutation.
//!
//! Supported local control flow follows the same rule. Conditions receive the current root state in both branches;
//! while bodies and scan bodies return updated hidden carries; nested calls receive and return the state required by
//! their canonical root summaries. A while condition may read entering state but cannot mutate it, because its
//! Boolean-only boundary has nowhere to publish an update. No region closure of any shape may consume a caller root,
//! because a consumed root has no successor state for a boundary to carry. Derived views do not cross any of these
//! boundaries and must be recreated from the root inside the attached region.
//!
//! Mutation summaries are conservative: a write in either condition branch or in a loop/scan body publishes hidden
//! final state and advances the external holder generation even when one execution takes the other branch, performs
//! zero loop iterations, or scans a zero-length axis. In those executions the published state equals the input state.
//!
//! ```text
//! local reference program
//!     -> discharge to immutable array SSA, rejecting misuse against the root it reached
//!     -> partial evaluation / batching / AD / rematerialization
//!
//! external or captured reference program
//!     -> stateful compilation and execution, or a targeted transform rejection
//! ```
//!
//! Representative supported compositions are shown below. The transforms themselves reject reference operations
//! outright ("must be discharged before differentiation/batching"). First call
//! [`ReferenceDischarge::discharge_local_references`], then use the ordinary transform: [`Program::jvp`] or
//! [`Program::linearize`] for forward mode, [`Pullback`](crate::Pullback) obtained from the linearization for reverse
//! mode, [`Program::batched_with_threaded_extent`](crate::Program::batched_with_threaded_extent) for batching,
//! [`Program::partially_evaluate`] for partial evaluation, and
//! [`Program::rematerialize_with_local_references`](crate::Program::rematerialize_with_local_references) for
//! rematerialization, which composes the discharge itself.
//!
//! ```text
//! condition(predicate,
//!     true  = || { state.add_update(true_update) },
//!     false = || { state.swap(false_replacement) })
//! while read(state) < limit { state.add_update(step) }
//! scan(inputs) { |input| state.add_update(input); read(state) }
//!     -> explicit immutable state carries at every attached-region boundary
//!
//! let program = program.discharge_local_references(capture_count, "differentiation")?;
//! program.jvp()?                                  // state = reference_new(x); state.add_update(x); freeze(state)
//! program.linearize()?.pullback()
//!     -> discharge local state -> differentiate the reference-free program
//!
//! program.discharge_local_references(capture_count, "batching")?.batched_with_threaded_extent(...)
//!     -> discharge local state -> batch independent immutable state-passing programs
//! ```
//!
//! A root that is allocated, mutated, and consumed inside one program is discharged into ordinary array SSA, so the
//! rewritten callable is reference-free: it reports no external state and keeps exactly its original public
//! outputs.
//!
//! ```
//! use ryft_core::{
//!     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, ReferenceFreezeOperation, ReferenceNewOperation,
//!     Placeholder, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceDischarge,
//! };
//!
//! let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
//! let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None)?[0];
//! builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)?;
//! let total = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)?[0];
//! let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
//!     vec![total],
//!     vec![Placeholder; 2],
//!     vec![Placeholder],
//! )?;
//!
//! let discharged = program.discharge_references(0)?;
//! assert_eq!(discharged.public_output_count(), 1);
//! assert_eq!(discharged.external_states(), &[]);
//! assert_eq!(
//!     discharged.program().interpret(vec![
//!         ArrayIrValue::Array(Array::scalar(1.0_f32)),
//!         ArrayIrValue::Array(Array::scalar(2.0_f32)),
//!     ])?,
//!     vec![ArrayIrValue::Array(Array::scalar(3.0_f32))],
//! );
//! # Ok::<(), ryft_core::ProgramError>(())
//! ```
//!
//! # Partial Discharge
//!
//! A caller that wants only *some* of a program's state made explicit names the sites to discharge and leaves the
//! rest alone. That is the shape the kernel pipeline needs: discharge the pipeline's own state, keep the references a
//! kernel body still accesses. Sites are enumerated from the program itself, so a caller selects by pointing at what
//! it can see rather than by reconstructing interpreter identities, and every root the selection omits survives in
//! the rewritten program as an ordinary reference whose accesses replay verbatim.
//!
//! ```
//! use ryft_core::{
//!     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, ReferenceFreezeOperation, ReferenceNewOperation,
//!     Placeholder, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceDischargeSite, ReferenceReadOperation,
//! };
//!
//! // Two independent roots: one accumulates and is frozen, the other is only read.
//! let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
//! let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
//! let accumulated = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None)?[0];
//! let observed = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![update], None)?[0];
//! builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![accumulated, update], None)?;
//! let total = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![accumulated], None)?[0];
//! let seen = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![observed], None)?[0];
//! let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
//!     vec![total, seen],
//!     vec![Placeholder; 2],
//!     vec![Placeholder; 2],
//! )?;
//!
//! // Both allocations are selectable, and discharging only one of them leaves the other a live reference — so the
//! // mixed result is not reference-free and refuses to convert into the full contract.
//! let sites = program.reference_discharge_sites(0)?;
//! assert_eq!(sites.len(), 2);
//! let partial = program.clone().partially_discharge_references(0, &sites[..1])?;
//! assert_eq!(partial.external_states(), &[]);
//! assert_eq!(partial.program().reference_discharge_sites(0)?.len(), 1);
//! assert!(partial.try_into_full().is_err());
//!
//! // Selecting everything discharges every root, and the same proof then succeeds.
//! let full = program.partially_discharge_references(0, &sites)?.try_into_full()?;
//! assert_eq!(full.public_output_count(), 2);
//! assert_eq!(full.program().reference_discharge_sites(0)?, Vec::new());
//! # Ok::<(), ryft_core::ProgramError>(())
//! ```
//!
//! This module is arrays-owned deliberately. The generic program layer defines the root, alias, and access
//! vocabulary and owns the interpreter itself, while this universe supplies the array-specific view composition and
//! the canonical slice, reshape, and update-slice reconstruction that composition lowers to.
//!
//! # Discharge Policy
//!
//! [`ArrayReferenceDischarge`] is this universe's [`ReferenceDischargePolicy`], and it is what
//! [`Program::discharge_references_with_policy`] threads: its referent is an [`ArrayType`]-typed array and its alias
//! is the composed [`ArrayReferenceView`] mapping a root to one handle's coordinates. Staged and eager reference
//! semantics cannot drift apart, because both reach their coordinates through the one [`ArrayReferenceView`]
//! traversal — the eager handles through a value carrier, the policy through a destination-context carrier.

// TODO(eaplatanios): Review this module.
//  Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::borrow::Cow;

use crate::arrays::operations::ArrayReferenceViewOperation;
use crate::arrays::reference_views::{ArrayReferenceView, ViewReadCarrier, ViewWriteCarrier};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::Shape;
use crate::arrays::types::ir::ArrayIrType;
use crate::captures::{CaptureConstant, ClosedProgram};
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::{AddOperation, ReshapeOperation, SliceOperation, UpdateSliceOperation};
use crate::parameters::Parameterized;
use crate::programs::{
    PartialReferenceDischargeResult, Program, ProgramError, ReferenceAccumulationPolicy, ReferenceDischarge,
    ReferenceDischargePolicy, ReferenceDischargeResult, ReferenceDischargeSite, ReferenceDischargeableOperation,
    ReferenceType, Typed, Value,
};
use crate::tracing::TracingContext;

impl<V, O> ReferenceDischarge for Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceViewOperation + ReferenceDischargeableOperation<TracingContext<V, O>, ArrayReferenceDischarge>,
{
    type DischargedProgram = Self;

    #[inline]
    fn discharge_references(self, capture_count: usize) -> Result<ReferenceDischargeResult<Self>, ProgramError> {
        self.discharge_references_with_policy::<ArrayReferenceDischarge>(capture_count)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceViewOperation + ReferenceDischargeableOperation<TracingContext<V, O>, ArrayReferenceDischarge>,
{
    /// Discharges the selected array reference sites and preserves every other one, returning the mixed program
    /// together with the external-state bindings of the roots that became state.
    ///
    /// This is the array universe's form of [`Program::partially_discharge_references_with_policy`], which documents
    /// the rewrite; [`ReferenceDischarge::discharge_references`] is its everything-selected case. A preserved array
    /// root keeps its reference-typed boundary position or its `reference_new` instruction, every access to it replays
    /// verbatim, and a view derived from it replays its `reference_index` or `reference_slice` step, so the surviving
    /// half of the program still denotes the same coordinates. Selected roots thread as immutable array state exactly
    /// as in full discharge, including the canonical slice, reshape, and update-slice reconstruction their views
    /// lower to.
    ///
    /// A preserved root crosses a condition, loop, scan, or call boundary as the reference it already is, at its own
    /// declared operand position, so the rewritten operation threads discharged state and surviving references side by
    /// side; only the discharged half widens. A preserved root may also be consumed, which full discharge rejects for
    /// a caller-owned root: the payload keeps the `reference_freeze`, so the caller hands its holder to that operation
    /// instead of to a state binding. A capture-lifted program has no partial form, and keeps using
    /// [`Program::discharge_references_with_lifted_captures`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    ///   - `sites`: Reference sites to discharge, enumerated from this same program through
    ///     [`Program::reference_discharge_sites`]. Every other root is preserved.
    #[inline]
    pub fn partially_discharge_references(
        self,
        capture_count: usize,
        sites: &[ReferenceDischargeSite],
    ) -> Result<PartialReferenceDischargeResult<Self>, ProgramError> {
        self.partially_discharge_references_with_policy::<ArrayReferenceDischarge>(capture_count, sites)
    }
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: CaptureConstant<Type = ArrayIrType>,
    O: ArrayReferenceViewOperation + ReferenceDischargeableOperation<TracingContext<V, O>, ArrayReferenceDischarge>,
{
    /// Consumes a program whose captures were lifted into its leading inputs while attached regions may still contain
    /// capture-reference constants naming that prefix.
    ///
    /// This is the program-level form of [`ClosedProgram::discharge_references`]. Capture constants resolve against
    /// the lifted entry prefix, or against the nested capture prefix of an enclosing call, through the discharge
    /// context's capture scope; discharge then threads their immutable array state across every enclosing
    /// structured boundary.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    #[inline]
    pub fn discharge_references_with_lifted_captures(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
        self.discharge_references_with_lifted_captures_and_policy::<ArrayReferenceDischarge>(capture_count)
    }
}

impl<Capture, V, O, Input, Output> ClosedProgram<Capture, V, O, Input, Output>
where
    Capture: Value<Type = ArrayIrType>,
    V: CaptureConstant<Type = ArrayIrType>,
    O: ArrayReferenceViewOperation + ReferenceDischargeableOperation<TracingContext<V, O>, ArrayReferenceDischarge>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Lifts this closed program's captures and discharges every reachable array reference.
    ///
    /// The returned logical metadata continues to identify capture slots separately from inputs. Concrete
    /// capture values remain owned by this [`ClosedProgram`]; discharge never embeds their mutable contents into the
    /// derived program.
    pub fn discharge_references(
        &self,
    ) -> Result<ReferenceDischargeResult<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_with_lifted_captures(capture_count)
    }
}

/// [`ReferenceDischargePolicy`] of the array reference universe.
///
/// An array reference's referent is an ordinary [`ArrayType`]-typed array, and the alias one flowing handle carries is
/// the composed [`ArrayReferenceView`] mapping its root to its own coordinates. Every access therefore reaches its
/// coordinates through the same view traversal the eager handles use, which is what keeps staged and eager reference
/// semantics from drifting apart: reading materializes the root-to-handle chain and takes its last snapshot, while a
/// replacement or an accumulation writes the new leaf back through that chain in reverse.
///
/// The destination is bounded by [`Context`] rather than [`Domain`](crate::Domain) because the view traversal binds
/// canonical slice, reshape, and update-slice operations into it. Those three have no value-level capability in the
/// composite array-IR universe (the capabilities are stated over [`ArrayType`]-typed values), so the policy reaches
/// them through the destination operation family's own construction seam.
#[derive(Copy, Clone, Debug)]
pub struct ArrayReferenceDischarge;

impl<C: Context<Type = ArrayIrType>> ReferenceDischargePolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceViewOperation,
{
    type Referent = ArrayType;
    type Alias = ArrayReferenceView;

    fn root_alias(_referent: &ArrayType) -> ArrayReferenceView {
        ArrayReferenceView::root()
    }

    fn lift_reference_type(r#type: ReferenceType<ArrayType>) -> ArrayIrType {
        ArrayIrType::Reference(r#type)
    }

    fn lift_referent_type(referent: ArrayType) -> ArrayIrType {
        ArrayIrType::Array(referent)
    }

    fn project_reference_type(r#type: &ArrayIrType) -> Option<ReferenceType<ArrayType>> {
        match r#type {
            ArrayIrType::Reference(reference) => Some(reference.clone()),
            ArrayIrType::Array(_) | ArrayIrType::Dimension(_) => None,
        }
    }

    fn read(context: &C, current: &C::Value, alias: &ArrayReferenceView) -> Result<C::Value, ProgramError> {
        let mut intermediates = alias.intermediates_in(&mut DestinationViewCarrier(context), current.clone())?;

        // The traversal always pushes the root itself first, so the chain is never empty and its last snapshot is the
        // value this handle selects.
        Ok(intermediates.pop().unwrap())
    }

    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceView,
    ) -> Result<C::Value, ProgramError> {
        alias.write_in(&mut DestinationViewCarrier(context), current.clone(), replacement)
    }

    fn replace(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceView,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        alias.swap_in(&mut DestinationViewCarrier(context), current.clone(), replacement)
    }
}

// Composite array-IR values deliberately expose no value-level addition: the composite family carries array payloads
// through `ArrayIrOperation::Array` and lifts the type-generic `AddOperation<ArrayIrType>` into that member instead,
// which is the same seam generic reverse mode uses to accumulate cotangents. Accumulation therefore binds the lifted
// addition through the destination, requiring nothing beyond the conversion the operation family already provides.
impl<C: Context<Type = ArrayIrType>> ReferenceAccumulationPolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceViewOperation + From<AddOperation<ArrayIrType>>,
{
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &ArrayReferenceView,
    ) -> Result<C::Value, ProgramError> {
        let mut carrier = DestinationViewCarrier(context);
        let intermediates = alias.intermediates_in(&mut carrier, current.clone())?;
        let selected = intermediates.last().unwrap().clone();
        let accumulated = carrier.bind(C::Operation::from(AddOperation::new()), &[&selected, &update])?;
        alias.reconstruct_in(&mut carrier, &intermediates[..alias.transforms().len()], accumulated)
    }
}

/// View carrier that binds the canonical slice, reshape, and update-slice operations of one array reference view into
/// a reference discharge destination, sharing the single [`ArrayReferenceView`] traversal with the eager value
/// carrier, which is what keeps staged and eager reference semantics from drifting apart.
struct DestinationViewCarrier<'c, C>(
    /// Destination context the staged view operations are bound through.
    &'c C,
);

impl<C: Context<Type = ArrayIrType>> ViewReadCarrier for DestinationViewCarrier<'_, C>
where
    C::Operation: ArrayReferenceViewOperation,
{
    type Value = C::Value;

    fn array_type<'c>(&'c self, value: &'c C::Value) -> Result<Cow<'c, ArrayType>, ProgramError> {
        match value.r#type() {
            Cow::Borrowed(r#type) => Ok(Cow::Borrowed(<&ArrayType>::try_from(r#type)?)),
            Cow::Owned(r#type) => Ok(Cow::Owned(<&ArrayType>::try_from(&r#type)?.clone())),
        }
    }

    fn slice(&mut self, input: &C::Value, starts: Vec<usize>, limits: Vec<usize>) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_slice(SliceOperation::new(starts, limits)), &[input])
    }

    fn reshape(&mut self, input: &C::Value, shape: Shape) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_reshape(ReshapeOperation::new(shape)), &[input])
    }
}

impl<C: Context<Type = ArrayIrType>> ViewWriteCarrier for DestinationViewCarrier<'_, C>
where
    C::Operation: ArrayReferenceViewOperation,
{
    fn update_slice(
        &mut self,
        target: &C::Value,
        update: &C::Value,
        starts: Vec<usize>,
    ) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_update_slice(UpdateSliceOperation::new(starts)), &[target, update])
    }
}

impl<C: Context<Type = ArrayIrType>> DestinationViewCarrier<'_, C> {
    /// Binds one single-result view operation into the destination and returns its result.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Destination-family operation to bind.
    ///   - `inputs`: Operands of the application, in operation-defined order.
    fn bind(&self, operation: C::Operation, inputs: &[&C::Value]) -> Result<C::Value, ProgramError> {
        let inputs = inputs.iter().map(|input| (*input).clone()).collect::<Vec<_>>();
        let mut outputs = self.0.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{
        ArrayIrOperation, ArrayOperation, ReferenceIndexOperation, ReferenceSliceOperation,
    };
    use crate::arrays::reference_views::{ArrayReference, ArrayReferenceView, ArrayReferenceViewTransform};
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::captures::CaptureReference;
    use crate::contexts::EagerContext;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{
        Effects, Instruction, InstructionId, Operation, OutputRegionProvenance, ProgramBuilder,
        ReferenceAddUpdateOperation, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargeValue,
        ReferenceDischargeableOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceOperationSemantics,
        ReferenceReadOperation, ReferenceSource, ReferenceStateBinding, ReferenceSwapOperation, ReferenceType,
        ReferenceWriteOperation, RegionInterface, RegionSlot, TypeError, discharge_positional_region_operation,
        discharge_reference_free_operation,
    };
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestDestination = TracingContext<TestValue, TestOperation>;
    type Capture = CaptureReference<ArrayIrType>;
    type CaptureArray = CaptureReference<ArrayType>;
    type CaptureOperation = ArrayIrOperation<CaptureArray>;

    // Returns the scalar `f32` array type used by the discharge fixtures.
    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    // Wraps one scalar array as an array-IR value.
    fn scalar(value: f32) -> TestValue {
        TestValue::Array(Array::scalar(value))
    }

    // Wraps one Boolean scalar array as an array-IR value.
    fn boolean(value: bool) -> TestValue {
        TestValue::Array(Array::scalar(value))
    }

    // Wraps one vector array as an array-IR value.
    fn vector(values: Vec<f32>) -> TestValue {
        TestValue::Array(Array::vector(values))
    }

    #[test]
    fn test_flat_reference_discharge_rewrites_the_complete_flat_language() {
        // Discharge is exercised over the complete flat reference language: allocation, read, write, swap, additive
        // update, freeze, and both composed view derivations, over a local root and over external roots. Every case
        // asserts the exact rewritten program before asserting behavior.
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pair_type = ArrayType::new_static(DataType::F32, [2]);

        // A local root allocated, viewed, mutated through the view, and frozen.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(vector_type.clone().into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 2), Vec::new(), vec![view], None).unwrap()[0];
        let element_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        let pair = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![view],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![pair, replacement], None)
            .unwrap();
        let replaced = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![pair, replacement], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pair, update], None)
            .unwrap();
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![element_snapshot, replaced, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let inputs = vec![vector(vec![1.0, 2.0, 3.0, 4.0]), vector(vec![10.0, 20.0]), vector(vec![1.0, 2.0])];
        let expected = source.clone().interpret(inputs.clone()).unwrap();

        // A local root leaves no external state behind, so the discharged boundary is exactly the source boundary,
        // and the rewritten program reproduces the eager reference execution.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));

        // Two external roots, one written and one only read, with the read reaching the second root first so that
        // boundary order cannot accidentally follow access order.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference_type = ReferenceType::new(scalar_type());
        let captured = builder.add_input(reference_type.clone().into());
        let public = builder.add_input(reference_type.into());
        let replacement = builder.add_input(scalar_type().into());
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![public], None).unwrap()[0];
        let swapped = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![captured, replacement], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, swapped], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        // The capture prefix splits the entry boundary, bindings follow that boundary rather than access order, and
        // only the mutated root receives a hidden final-state output after the public prefix.
        let discharged = source.discharge_references(1).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(
            discharged.external_states(),
            &[
                ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, 0, Some(2)),
                ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, 1, None),
            ],
        );
        assert_eq!(
            discharged.program().interpret(vec![scalar(10.0), scalar(20.0), scalar(7.0)]),
            Ok(vec![scalar(20.0), scalar(10.0), scalar(7.0)]),
        );

        // A program that only reads an external root keeps its source boundary exactly: the root enters as state and
        // publishes nothing, so no hidden output is appended.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(scalar_type()).into());
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, 0, None)],
        );

        // A reference-free program is its own discharge and is returned untouched.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type().into());
        let doubled = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let rendering = source.to_string();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().to_string(), rendering);
        assert_eq!(discharged.external_states(), &[]);
    }

    #[test]
    fn test_reference_discharge_omits_a_dead_constant() {
        // A program that touches references is replayed into a fresh trace through the shared program replay path,
        // which lifts only the constants something still consumes, so a constant nothing reads does not survive the
        // rewrite. That is what every other transform already does, and it is why the reference-free fast path above
        // exists: a program with nothing to rewrite keeps its atoms exactly.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let update = builder.add_constant(scalar(2.0));
        let unused = builder.add_constant(scalar(5.0));
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![root, update], None)
            .unwrap();
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_ne!(unused, update);

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 2.0
                    %2:f32[] = add %0 %1
                in (%2)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(1.0)]), Ok(vec![scalar(3.0)]));
    }

    #[test]
    fn test_reference_discharge_reports_environment_and_boundary_failures() {
        // Discharge catches at replay time what construction-time checking catches ahead of it: a root that a
        // `freeze` already consumed is reported against that exact root. The checked append rejects this program, so
        // the stale read is assembled through the unchecked rebuild hatch to prove discharge's own guard.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let stale = builder.add_variable(scalar_type().into());
        builder.add_instruction_unchecked(Instruction::new(
            ReferenceReadOperation::new().into(),
            vec![root],
            vec![stale],
            Vec::new(),
        ));
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, stale], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        // The minted root identity is process-local, so the assertion pins the diagnostic up to that coordinate.
        assert!(matches!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(0),
            Err(ProgramError::MalformedProgram(message))
                if message.starts_with("reference discharge accessed consumed reference root "),
        ));

        // A `freeze` through a derived view names no consumption at all, because consumption yields the whole root.
        // The eager handles and the checked append both reject the same program; discharge runs neither (the frozen
        // view is again assembled through the unchecked hatch), so it rejects rather than returning a whole-root
        // value under the view's narrower type.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [4]).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let frozen = builder.add_variable(ArrayType::new_static(DataType::F32, [2]).into());
        builder.add_instruction_unchecked(Instruction::new(
            ReferenceFreezeOperation::new().into(),
            vec![view],
            vec![frozen],
            Vec::new(),
        ));
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(0),
            Err(ProgramError::MalformedProgram(message)) if message.ends_with(
                "through the derived view `ref<f32[2]>`; consumption yields the whole root, whose referent is \
                 `f32[4]`",
            ),
        ));

        // A structured boundary carries whole-root state, so a derived view cannot cross one: passing it would widen
        // the view silently to the root's own value. The view has to be re-derived from the root inside the region.
        let view_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [3]));
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch_input = branch_builder.add_input(view_type.clone().into());
        let branch_snapshot = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![branch_input], None)
            .unwrap()[0];
        let branch = || {
            branch_builder
                .clone()
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![branch_snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [4]).into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let true_branch = builder.import_program(branch());
        let false_branch = builder.import_program(branch());
        let selected = builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![predicate, view], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![selected], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(0),
            Err(ProgramError::MalformedProgram(message)) if message.ends_with(
                "across a region boundary, which carries the whole root `ref<f32[4]>`; derive the view inside the \
                 region instead",
            ),
        ));

        // A caller-owned root belongs to the caller's holder, so consuming one is rejected by name rather than
        // silently dropping the caller's state.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(scalar_type()).into());
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose holder belongs to the caller".to_string(),
            ),
        );

        // An oversized capture prefix cannot describe the entry boundary it is meant to split.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type().into());
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(2).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge requests 2 captures but the program has 1 inputs".to_string(),
            ),
        );
    }

    #[test]
    fn test_array_reference_discharge_policy_stages_composed_view_accesses() {
        // The policy is the interpreter-side half of array reference discharge, so this test pins the exact
        // instruction sequence each of its three alias applications stages, over a composed index-of-slice view of a
        // 3x3 root. Each access materializes the root-to-handle chain against the state it observes, so the chain is
        // restaged per access rather than shared, and a replacement and an accumulation then write their new leaf
        // back through that chain in reverse.
        let alias = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 1 });
        let stage = |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
            let context = inputs[0].context().clone();
            let read = ArrayReferenceDischarge::read(&context, &inputs[0], &alias)?;
            let (previous, replaced) =
                ArrayReferenceDischarge::replace(&context, &inputs[0], inputs[1].clone(), &alias)?;
            let accumulated = ArrayReferenceDischarge::accumulate(&context, &replaced, inputs[1].clone(), &alias)?;
            Ok(vec![read, previous, replaced, accumulated])
        };
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            EagerContext::<TestValue, TestOperation>::trace(
                stage,
                vec![
                    ArrayIrType::Array(matrix_type.clone()),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %6:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %5
                    %7:f32[2] = reshape [shape=[2]] %6
                    %8:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %9:f32[2, 2] = update_slice [start_indices=[1, 0]] %5 %8
                    %10:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %9
                    %11:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %10
                    %12:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %11
                    %13:f32[2] = reshape [shape=[2]] %12
                    %14:f32[2] = add %13 %1
                    %15:f32[1, 2] = reshape [shape=[1, 2]] %14
                    %16:f32[2, 2] = update_slice [start_indices=[1, 0]] %11 %15
                    %17:f32[3, 3] = update_slice [start_indices=[1, 0]] %10 %16
                in (%4, %7, %10, %17)"},
        );

        // The lift and projection pair round-trips a reference type through the composite universe and classifies an
        // ordinary array type as not a reference, and an unviewed root's alias selects its complete referent.
        let reference_type = ReferenceType::new(matrix_type.clone());
        let lifted = <ArrayReferenceDischarge as ReferenceDischargePolicy<TestDestination>>::lift_reference_type(
            reference_type.clone(),
        );
        assert_eq!(lifted, ArrayIrType::Reference(reference_type.clone()));
        assert_eq!(
            <ArrayReferenceDischarge as ReferenceDischargePolicy<TestDestination>>::project_reference_type(&lifted),
            Some(reference_type),
        );
        assert_eq!(
            <ArrayReferenceDischarge as ReferenceDischargePolicy<TestDestination>>::lift_referent_type(
                matrix_type.clone(),
            ),
            ArrayIrType::Array(matrix_type.clone()),
        );
        assert_eq!(
            <ArrayReferenceDischarge as ReferenceDischargePolicy<TestDestination>>::project_reference_type(
                &ArrayIrType::Array(matrix_type.clone()),
            ),
            None,
        );
        assert!(
            <ArrayReferenceDischarge as ReferenceDischargePolicy<TestDestination>>::root_alias(&matrix_type).is_root(),
        );
    }

    #[test]
    fn test_array_reference_discharge_policy_write_skips_the_replaced_leaf_selection() {
        let alias = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 1 });
        let stage = |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
            let context = inputs[0].context().clone();
            Ok(vec![ArrayReferenceDischarge::write(&context, &inputs[0], inputs[1].clone(), &alias)?])
        };
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            EagerContext::<TestValue, TestOperation>::trace(
                stage,
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %4:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %3
                    %5:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %4
                in (%5)"},
        );
    }

    #[test]
    fn test_reference_free_discharge_is_identity() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type().into());
        let output = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let source_rendering = source.to_string();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().to_string(), source_rendering);
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(3.0)]), Ok(vec![scalar(6.0)]));
    }

    #[test]
    fn test_straight_line_discharge_stages_explicit_immutable_state_threading() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let replacement = builder.add_input(scalar_type().into());
        let update = builder.add_input(scalar_type().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let first_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let swapped_snapshot = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        assert!(
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap()
                .is_empty(),
        );
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, swapped_snapshot, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Allocation and every reference access disappear: the initializer becomes entering state, the read forwards
        // it, the write installs the replacement without producing a value, the swap forwards that replacement, and
        // only accumulation stages real work.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = add %1 %2
                in (%0, %1, %3)"},
        );
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().effects(), Effects::PURE);
    }

    #[test]
    fn test_reference_view_discharge_matches_eager_composed_updates() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let pair_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let indexed = builder
            .add_instruction(ReferenceIndexOperation::new(0, 3), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let indexed_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![indexed], None).unwrap()[0];
        let outer = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let composed = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![outer],
                None,
            )
            .unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![composed, replacement], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![composed, update], None)
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![indexed_snapshot, old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let inputs = vec![vector(vec![1.0, 2.0, 3.0, 4.0]), vector(vec![10.0, 20.0]), vector(vec![1.0, 2.0])];
        let expected = source.interpret(inputs.clone()).unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected),);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2], %2:f32[2] .
                let %3:f32[1] = slice [start_indices=[3], limit_indices=[4]] %0
                    %4:f32[] = reshape [shape=[]] %3
                    %5:f32[3] = slice [start_indices=[1], limit_indices=[4]] %0
                    %6:f32[2] = slice [start_indices=[0], limit_indices=[2]] %5
                    %7:f32[3] = update_slice [start_indices=[0]] %5 %1
                    %8:f32[4] = update_slice [start_indices=[1]] %0 %7
                    %9:f32[3] = slice [start_indices=[1], limit_indices=[4]] %8
                    %10:f32[2] = slice [start_indices=[0], limit_indices=[2]] %9
                    %11:f32[2] = add %10 %2
                    %12:f32[3] = update_slice [start_indices=[0]] %9 %11
                    %13:f32[4] = update_slice [start_indices=[1]] %8 %12
                in (%4, %6, %13)"},
        );
    }

    #[test]
    fn test_indexed_mutation_discharge_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.clone().into());
        let update = builder.add_input(row_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let row = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![row, update], None)
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            vector(vec![10.0, 20.0, 30.0]),
            vector(vec![1.0, 2.0, 3.0]),
        ];
        let expected = vec![
            vector(vec![4.0, 5.0, 6.0]),
            TestValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0])),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[3], %2:f32[3] .
                let %3:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %0
                    %4:f32[3] = reshape [shape=[3]] %3
                    %5:f32[1, 3] = reshape [shape=[1, 3]] %1
                    %6:f32[2, 3] = update_slice [start_indices=[1, 0]] %0 %5
                    %7:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %6
                    %8:f32[3] = reshape [shape=[3]] %7
                    %9:f32[3] = add %8 %2
                    %10:f32[1, 3] = reshape [shape=[1, 3]] %9
                    %11:f32[2, 3] = update_slice [start_indices=[1, 0]] %6 %10
                in (%4, %11)"},
        );
    }

    #[test]
    fn test_composed_index_of_slice_swap_discharge_reconstructs_both_view_steps() {
        // Swapping through an index composed onto a slice must write back through both steps in reverse order, so the
        // discharged program reconstructs the sliced block from the squeezed row before writing it into the root.
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let block = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let row =
            builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![block], None).unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement], None)
            .unwrap()[0];
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            vector(vec![10.0, 20.0]),
        ];
        let expected = vec![
            vector(vec![7.0, 8.0]),
            TestValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 9.0])),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %6:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %5
                    %7:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %6
                in (%4, %7)"},
        );
    }

    #[test]
    fn test_generated_short_state_programs_match_eager_and_immutable_oracles() {
        /// One operation in a bounded generated state program.
        #[derive(Copy, Clone)]
        enum Step {
            /// Observe the current state.
            Read,

            /// Replace the current state without observing it.
            Write,

            /// Replace the current state with the shared replacement input.
            Swap,

            /// Add the shared update input to the current state.
            AddUpdate,
        }

        // Every bounded read/write/swap/accumulate sequence must agree with both an independent scalar oracle and the
        // eager reference interpreter, which pins the state-threading rewrite over all short primitive orderings.
        for length in 0usize..=3 {
            for code in 0..4usize.pow(length as u32) {
                let mut remainder = code;
                let steps = (0..length)
                    .map(|_| {
                        let step = match remainder % 4 {
                            0 => Step::Read,
                            1 => Step::Write,
                            2 => Step::Swap,
                            3 => Step::AddUpdate,
                            _ => unreachable!(),
                        };
                        remainder /= 4;
                        step
                    })
                    .collect::<Vec<_>>();
                let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                let initial = builder.add_input(scalar_type().into());
                let replacement = builder.add_input(scalar_type().into());
                let update = builder.add_input(scalar_type().into());
                let reference =
                    builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
                let mut outputs = Vec::new();
                let mut oracle_state = 2.0f32;
                let mut oracle_outputs = Vec::new();
                for step in steps {
                    match step {
                        Step::Read => {
                            outputs.push(
                                builder
                                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                                    .unwrap()[0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                        }
                        Step::Write => {
                            builder
                                .add_instruction(
                                    ReferenceWriteOperation::new(),
                                    Vec::new(),
                                    vec![reference, replacement],
                                    None,
                                )
                                .unwrap();
                            oracle_state = 7.0;
                        }
                        Step::Swap => {
                            outputs.push(
                                builder
                                    .add_instruction(
                                        ReferenceSwapOperation::new(),
                                        Vec::new(),
                                        vec![reference, replacement],
                                        None,
                                    )
                                    .unwrap()[0],
                            );
                            oracle_outputs.push(scalar(oracle_state));
                            oracle_state = 7.0;
                        }
                        Step::AddUpdate => {
                            builder
                                .add_instruction(
                                    ReferenceAddUpdateOperation::new(),
                                    Vec::new(),
                                    vec![reference, update],
                                    None,
                                )
                                .unwrap();
                            oracle_state += 3.0;
                        }
                    }
                }
                outputs.push(
                    builder
                        .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
                        .unwrap()[0],
                );
                oracle_outputs.push(scalar(oracle_state));
                let output_count = outputs.len();
                let source = builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(
                        outputs,
                        vec![Placeholder; 3],
                        vec![Placeholder; output_count],
                    )
                    .unwrap();
                let inputs = vec![scalar(2.0), scalar(7.0), scalar(3.0)];
                let eager = source.clone().interpret(inputs.clone()).unwrap();

                // Every generated program is discharged and then checked against the eager and hand-written
                // immutable oracles.
                let discharged = source.discharge_references(0).unwrap();
                let functional = discharged.program().interpret(inputs).unwrap();
                assert_eq!(eager, oracle_outputs);
                assert_eq!(functional, oracle_outputs);
            }
        }
    }

    #[test]
    fn test_external_discharge_uses_boundary_order_and_appends_only_mutated_state() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference_type = ReferenceType::new(scalar_type());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let replacement = builder.add_input(scalar_type().into());

        // Access the second root first so metadata order cannot accidentally follow access order.
        let second_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let first_snapshot = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, replacement], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![second_snapshot, first_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Discharge is deterministic: two independent runs over the same source agree on the rewritten program and on
        // the complete external-state metadata, including its serialized form.
        let repeated = source.clone().discharge_references(0).unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().to_string(), repeated.program().to_string());
        assert_eq!(discharged.external_states(), repeated.external_states());
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            serde_json::to_string(repeated.external_states()).unwrap(),
        );
        assert_eq!(discharged.public_output_count(), 2);

        // Metadata follows entry-boundary order rather than access order, and only the swapped first root receives a
        // hidden final-state output after the public prefix.
        assert_eq!(
            discharged.external_states(),
            &[
                ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, 0, Some(2)),
                ReferenceStateBinding::new(ReferenceSource::Input { index: 1 }, 1, None),
            ],
        );
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            concat!(
                r#"[{"source":{"input":{"index":0}},"discharged_input_index":0,"#,
                r#""final_state_output_index":2},{"source":{"input":{"index":1}},"#,
                r#""discharged_input_index":1,"final_state_output_index":null}]"#,
            ),
        );
        assert_eq!(
            format!("{:?}", discharged.external_states()),
            concat!(
                "[ReferenceStateBinding { source: Input { index: 0 }, ",
                "discharged_input_index: 0, final_state_output_index: Some(2) }, ",
                "ReferenceStateBinding { source: Input { index: 1 }, ",
                "discharged_input_index: 1, final_state_output_index: None }]",
            ),
        );
        assert_eq!(
            discharged.program().interpret(vec![scalar(10.0), scalar(20.0), scalar(7.0)]),
            Ok(vec![scalar(20.0), scalar(10.0), scalar(7.0)]),
        );
    }

    #[test]
    fn test_partial_reference_discharge_keeps_kernel_references_beside_discharged_pipeline_state() {
        // The shape the kernel pipeline needs: the pipeline's own root is normalized into explicit array state while
        // the root a kernel body addresses stays a reference, including the view derived from it and the accesses
        // performed through that view.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let pipeline_initial = builder.add_input(scalar_type().into());
        let kernel_initial = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let step = builder.add_input(scalar_type().into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, step], None)
            .unwrap();
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![kernel], None).unwrap()[0];
        let previous = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![element, step], None)
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let kernel_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![previous, pipeline_final, kernel_final],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Both allocations are selectable in their own right, and the pipeline's is the only one selected.
        let entry = source.entry_region_ref().id();
        let sites = source.reference_discharge_sites(0).unwrap();
        assert_eq!(
            sites,
            vec![
                ReferenceDischargeSite::Allocation { instruction: InstructionId::new(entry, 0), output_index: 0 },
                ReferenceDischargeSite::Allocation { instruction: InstructionId::new(entry, 1), output_index: 0 },
            ],
        );
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();

        // The selected allocation disappeared into threaded array state, while the unselected one, its view, its swap,
        // and its freeze all survive as the reference operations the source performed. Neither root is caller-owned,
        // so the mixed program reports no external state and keeps exactly its source boundary.
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3], %2:f32[] .
                let %3:ref<f32[3]> = reference_new %1
                    %4:f32[] = add %0 %2
                    %5:ref<f32[]> = reference_index [axis=0, index=1] %3
                    %6:f32[] = reference_swap %5 %2
                    %7:f32[3] = reference_freeze %3
                in (%6, %4, %7)"},
        );

        // Eager reference semantics stay the oracle: the mixed program computes exactly what the source program does.
        let inputs = vec![scalar(10.0), vector(vec![1.0, 2.0, 3.0]), scalar(7.0)];
        let expected = vec![scalar(2.0), scalar(17.0), vector(vec![1.0, 7.0, 3.0])];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));

        // The mixed payload proves nothing about reference freedom, and asking for the proof reports the surviving
        // references instead of converting.
        assert_eq!(
            discharged.try_into_full().unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );

        // Selecting both allocations is the everything-selected case, so it must agree with full discharge exactly.
        let selected =
            source.clone().partially_discharge_references(0, sites.as_slice()).unwrap().try_into_full().unwrap();
        assert_eq!(selected.program().to_string(), source.discharge_references(0).unwrap().program().to_string());
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_root_through_condition_branches() {
        // A condition's shared state boundary carries both kinds of root: the selected one crosses as immutable state
        // and is widened with a published successor, while the preserved one crosses as the reference it already is,
        // at its own declared operand position, and is read inside each branch exactly as the source read it.
        let reference_type = ReferenceType::new(scalar_type());
        let branch = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch(true);
        let false_branch = branch(false);

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(scalar_type().into());
        let kernel_initial = builder.add_input(scalar_type().into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let kernel_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, pipeline_final, kernel_final],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %1 %3 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%2, %3)
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                            in (%2, %0)
                        },
                    ]
                    %6:f32[] = reference_freeze %3
                in (%4, %5, %6)"},
        );

        // Eager reference semantics stay the oracle on both sides of the rewrite.
        for (predicate, expected) in [(true, 13.0_f32), (false, 10.0)] {
            let inputs = vec![boolean(predicate), scalar(10.0), scalar(3.0)];
            let outputs = vec![scalar(3.0), scalar(expected), scalar(3.0)];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(outputs.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
        }
    }

    #[test]
    fn test_partial_reference_discharge_widens_nothing_for_a_preserved_root_a_region_writes() {
        // Read-only pruning and preservation meet here: the discharged root is only read, so it gains no appended
        // output, and the preserved root is *written* inside a branch, which still gains it nothing, because the write
        // replayed into the rebuilt branch as the operation the source performed. The condition therefore keeps its
        // source boundary exactly.
        let reference_type = ReferenceType::new(scalar_type());
        let branch = |writes: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
            if writes {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![kernel, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch(true);
        let false_branch = branch(false);

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(scalar_type().into());
        let kernel_initial = builder.add_input(scalar_type().into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let kernel_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, pipeline_final, kernel_final],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[] = condition %0 %1 %3 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let reference_write %1 %0
                            in (%0)
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            in (%0)
                        },
                    ]
                    %5:f32[] = reference_freeze %3
                in (%4, %1, %5)"},
        );

        for (predicate, kernel_final) in [(true, 10.0_f32), (false, 3.0)] {
            let inputs = vec![boolean(predicate), scalar(10.0), scalar(3.0)];
            let outputs = vec![scalar(10.0), scalar(10.0), scalar(kernel_final)];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(outputs.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
        }
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_root_through_nested_structured_boundaries() {
        // A rebuilt region is discharged against its own isolated environment, so a preserved root crossing two
        // boundaries is bound as a preserved root of the outer fork and then threaded again into the inner one. The
        // reference therefore reaches the innermost access as the caller's own, and the discharged root beside it is
        // widened independently at each level.
        let reference_type = ReferenceType::new(scalar_type());
        let inner = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let inner_true = inner(true);
        let inner_false = inner(false);

        let mut outer_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inner_true = outer_builder.import_region(inner_true.entry_region_ref());
        let inner_false = outer_builder.import_region(inner_false.entry_region_ref());
        let predicate = outer_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline = outer_builder.add_input(reference_type.clone().into());
        let kernel = outer_builder.add_input(reference_type.clone().into());
        let observed = outer_builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![inner_true, inner_false],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let outer = outer_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let outer_true = builder.import_region(outer.entry_region_ref());
        let outer_false = builder.import_region(outer.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(scalar_type().into());
        let kernel_initial = builder.add_input(scalar_type().into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![outer_true, outer_false],
                vec![predicate, predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, pipeline_final],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %0 %1 %3 [
                        true={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                        false={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                    ]
                in (%4, %5)"},
        );

        let inputs = vec![boolean(true), scalar(10.0), scalar(3.0)];
        let outputs = vec![scalar(3.0), scalar(13.0)];
        assert_eq!(source.interpret(inputs.clone()), Ok(outputs.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
    }

    #[test]
    fn test_partial_reference_discharge_selecting_nothing_is_the_identity_on_a_structured_program() {
        // Preserving every root is the opposite extreme from full discharge, and it must be the identity: every
        // access, every view, and every structured boundary replays exactly as the source declared it. This is the
        // sharpest statement of what "preserved" means, and it holds through a condition's attached regions.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(reference_type.clone().into());
        let replacement = branch_builder.add_input(scalar_type().into());
        let previous = branch_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(branch.entry_region_ref());
        let false_branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let replacement = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, root, replacement],
                None,
            )
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![previous, frozen], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let preserved = source.clone().partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.public_output_count(), 2);
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(preserved.program().to_string(), source.to_string());

        // Selecting the one site instead is full discharge, which is the other extreme of the same rewrite.
        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged =
            source.clone().partially_discharge_references(0, sites.as_slice()).unwrap().try_into_full().unwrap();
        assert_eq!(discharged.program().to_string(), source.discharge_references(0).unwrap().program().to_string());
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_carry_through_a_loop() {
        // A loop's boundaries stay symmetric with a preserved carry in them: the carry occupies its declared position
        // in the operand list, in both region boundaries, and in the output list, carrying a reference rather than
        // state, and the loop publishes no successor for it.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let counter = condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let observed = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![counter], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(scalar(3.0));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![observed, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let counter = body_builder.add_input(reference_type.clone().into());
        let step = body_builder.add_input(reference_type.clone().into());
        let increment =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![step], None).unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, increment], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![counter, step], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let counter_initial = builder.add_input(scalar_type().into());
        let step_initial = builder.add_input(scalar_type().into());
        let counter = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![counter_initial], None)
            .unwrap()[0];
        let step =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![step_initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(Some(8)).unwrap();
        let carried = builder.add_instruction(operation, vec![condition, body], vec![counter, step], None).unwrap();
        let (carried_counter, carried_step) = (carried[0], carried[1]);
        let total = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_counter], None)
            .unwrap()[0];
        let remaining = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_step], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![total, remaining], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[], %4:ref<f32[]> = while [iteration_bound=8] %0 %2 [
                        condition={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = const 3.0
                                %3:bool[] = compare [direction=LessThan] %0 %2
                            in (%3)
                        },
                        body={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%3, %1)
                        },
                    ]
                    %5:f32[] = reference_freeze %2
                in (%3, %5)"},
        );

        // The eager interpreter cannot carry a reference through a loop at all — its masked predicate selection needs
        // value semantics for every carry — so the mixed program is exactly as runnable as the source it came from,
        // and both report the same limitation. Running it needs a stateful domain, which is what preserving a
        // reference asks for in the first place.
        let inputs = vec![scalar(0.0), scalar(2.0)];
        let rejection = Err(ProgramError::UnsupportedOperation {
            message: "references must be discharged before while predicate selection".to_string(),
        });
        assert_eq!(source.interpret(inputs.clone()), rejection);
        assert_eq!(discharged.program().interpret(inputs), rejection);
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_carry_through_a_scan() {
        // A scan inserts its synthesized state carries immediately after the declared carry prefix, and a preserved
        // carry stays a declared carry: it keeps its position and its reference type on both boundaries.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let total = body_builder.add_input(reference_type.clone().into());
        let step = body_builder.add_input(reference_type.clone().into());
        let increment =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![step], None).unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![total, increment], None)
            .unwrap();
        let observed =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![total], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![total, step, observed],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let total_initial = builder.add_input(scalar_type().into());
        let step_initial = builder.add_input(scalar_type().into());
        let total = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![total_initial], None)
            .unwrap()[0];
        let step =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![step_initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<TestValue>::new(2, 3), vec![body], vec![total, step], None)
            .unwrap();
        let (carried_total, carried_step, stacked) = (outputs[0], outputs[1], outputs[2]);
        let final_total = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_total], None)
            .unwrap()[0];
        let final_step = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_step], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![final_total, stacked, final_step],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &sites[..1]).unwrap();
        assert_eq!(discharged.public_output_count(), 3);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[], %4:ref<f32[]>, %5:f32[3] = scan [carry_count=2, length=3, reverse=false] %0 %2 [
                        body={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%3, %1, %3)
                        },
                    ]
                    %6:f32[] = reference_freeze %2
                in (%3, %5, %6)"},
        );

        let inputs = vec![scalar(0.0), scalar(2.0)];
        let outputs = vec![scalar(6.0), vector(vec![2.0, 4.0, 6.0]), scalar(2.0)];
        assert_eq!(source.interpret(inputs.clone()), Ok(outputs.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
    }

    #[test]
    fn test_array_reference_discharge_local_references() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let external = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The shared local-only gate names the requesting transform together with the caller-owned boundary source.
        // Public arguments and captures are both external: neither boundary supplies the runtime holder needed for
        // final-state write-back.
        assert!(matches!(
            external.clone().discharge_local_references(0, "differentiation"),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "differentiation supports only local references, but the program uses external \
                    `input 0`",
        ));
        assert!(matches!(
            external.discharge_local_references(1, "batching"),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "batching supports only local references, but the program uses external `capture 0`",
        ));

        // A program that allocates every root itself passes the gate with its boundary unchanged, because hidden
        // final-state outputs are appended only for external roots. The result is an ordinary reference-free array
        // program.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, initial], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let local = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let discharged = local.discharge_local_references(0, "rematerialization").unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert!(discharged.effects().is_pure());
        assert_eq!(discharged.interpret(vec![scalar(3.0)]), Ok(vec![scalar(6.0)]));
    }

    #[test]
    fn test_condition_discharge_threads_identical_state_through_unequal_branch_accesses() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_input(scalar_type().into());
        true_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        false_builder.add_input(scalar_type().into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(scalar_type().into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // Both branches receive the entering state and return their own final state after the source output, so the
        // writing branch returns its replacement while the reading branch returns the state unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        in (%1, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));

        // The true branch writes and then reads, so both the public snapshot and final state are the replacement.
        assert_eq!(
            discharged.program().interpret(vec![boolean(true), scalar(10.0), scalar(7.0)]),
            Ok(vec![scalar(7.0), scalar(7.0)]),
        );

        // The false branch only reads, so the entering state is both the snapshot and the final state.
        assert_eq!(
            discharged.program().interpret(vec![boolean(false), scalar(10.0), scalar(7.0)]),
            Ok(vec![scalar(10.0), scalar(10.0)]),
        );
    }

    #[test]
    fn test_condition_discharge_orders_multiple_roots_by_parent_boundary() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        true_builder.add_input(scalar_type().into());
        let second_replacement = true_builder.add_input(scalar_type().into());
        let first_snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second_snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![second, second_replacement], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        let first_replacement = false_builder.add_input(scalar_type().into());
        false_builder.add_input(scalar_type().into());
        let first_snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, first_replacement], None)
            .unwrap()[0];
        let second_snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let first_replacement = builder.add_input(scalar_type().into());
        let second_replacement = builder.add_input(scalar_type().into());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second, first_replacement, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();

        // Both branches write a different root, so both roots cross the boundary; the appended final-state outputs
        // follow parent entry-boundary order rather than the order in which either branch happens to access them.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states().len(), 2);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Input { index: 1 });
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        assert_eq!(discharged.external_states()[1].source(), ReferenceSource::Input { index: 2 });
        assert_eq!(discharged.external_states()[1].final_state_output_index(), Some(3));

        // The true branch swaps only the second root, leaving the first root's final state at its entering value.
        let inputs = vec![boolean(true), scalar(10.0), scalar(20.0), scalar(11.0), scalar(22.0)];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![scalar(10.0), scalar(20.0), scalar(10.0), scalar(22.0)]),
        );

        // The false branch swaps only the first root, which mirrors the same contract on the other position.
        let inputs = vec![boolean(false), scalar(10.0), scalar(20.0), scalar(11.0), scalar(22.0)];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![scalar(10.0), scalar(20.0), scalar(11.0), scalar(20.0)]),
        );
    }

    #[test]
    fn test_condition_discharge_isolates_its_branches() {
        // Both branches accumulate a different amount into the same root and return the state they observe. If either
        // branch's staging leaked into the other's, the second branch would start from the first's successor state.
        let reference_type = ReferenceType::new(scalar_type());
        let branch = |amount: f32| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update = builder.add_constant(scalar(amount));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch(1.0));
        let false_branch = builder.import_program(branch(10.0));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, root],
                None,
            )
            .unwrap()[0];

        // The condition's outputs are bound in the *parent*, so a later parent instruction consumes them directly. A
        // value stamped with a branch's own destination builder would be rejected here instead of staged.
        let doubled = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![snapshot, snapshot], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = const 10.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                    %4:f32[] = add %2 %2
                in (%4, %3)"},
        );
        for (predicate, expected) in [(true, vec![scalar(6.0), scalar(3.0)]), (false, vec![scalar(24.0), scalar(12.0)])]
        {
            let inputs = vec![boolean(predicate), scalar(2.0)];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        }
    }

    #[test]
    fn test_condition_discharge_rejects_a_branch_local_allocation_that_escapes() {
        // Both branches allocate a root of their own and return it, so the condition's output denotes a reference its
        // caller never threaded in. Merging that output would hand the caller a handle into an environment that no
        // longer exists, so the rewrite rejects it instead.
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let initial = builder.add_input(scalar_type().into());
            let root =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![root], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch());
        let false_branch = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let escaped = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![escaped], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            source.discharge_references_with_policy::<ArrayReferenceDischarge>(0),
            Err(ProgramError::MalformedProgram(message))
                if message.ends_with("whose caller did not thread that root"),
        ));
    }

    #[test]
    fn test_nested_condition_while_discharge_keeps_local_state_inside_its_creation_scope() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let loop_condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let loop_body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let loop_condition = true_builder.import_region(loop_condition.entry_region_ref());
        let loop_body = true_builder.import_region(loop_body.entry_region_ref());
        let initial = true_builder.add_input(scalar_type().into());
        let reference =
            true_builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference = true_builder
            .add_instruction(operation, vec![loop_condition, loop_body], vec![reference], None)
            .unwrap()[0];
        let value = true_builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let value = false_builder.add_input(scalar_type().into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The root is allocated and frozen inside the true branch, so no state crosses the entry boundary even though a
        // nested loop mutates it; the false branch never sees the root at all.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![boolean(true), scalar(3.0)]), Ok(vec![scalar(5.0)]));
        assert_eq!(discharged.program().interpret(vec![boolean(false), scalar(3.0)]), Ok(vec![scalar(3.0)]));
    }

    #[test]
    fn test_while_discharge_preserves_zero_iteration_and_threads_mutated_state() {
        let build = |condition_value: bool, iteration_bound: Option<usize>| {
            let reference_type = ReferenceType::new(scalar_type());
            let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = condition_builder.add_input(reference_type.clone().into());
            condition_builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                .unwrap();
            let condition = condition_builder.add_constant(boolean(condition_value));
            let condition = condition_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = body_builder.add_input(reference_type.clone().into());
            let update = body_builder.add_constant(scalar(1.0));
            body_builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let body = body_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let condition = builder.import_region(condition.entry_region_ref());
            let body = builder.import_region(body.entry_region_ref());
            let reference = builder.add_input(reference_type.into());
            let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
            let reference =
                builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        // A predicate that is immediately false leaves the entering state untouched in both the public read and the
        // appended final state.
        let zero_iteration = build(false, None).discharge_references(0).unwrap();
        assert_eq!(zero_iteration.program().interpret(vec![scalar(2.0)]), Ok(vec![scalar(2.0), scalar(2.0)]));

        // The reference carry becomes an ordinary array carry: the condition region observes the state without
        // returning it, while the body returns the accumulated state in the carry position.
        let three_iterations = build(true, Some(3)).discharge_references(0).unwrap();
        assert_eq!(
            three_iterations.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = while [iteration_bound=3] %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const true
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%1, %1)"},
        );
        assert_eq!(three_iterations.program().interpret(vec![scalar(2.0)]), Ok(vec![scalar(5.0), scalar(5.0)]));
    }

    #[test]
    fn test_scan_discharge_keeps_state_carries_separate_from_stacked_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let operation = ScanOperation::<TestValue>::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs = builder.add_instruction(operation, vec![body], vec![reference], None).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![final_value, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The synthesized state joins the declared carry prefix on both boundaries instead of being appended after the
        // stacked outputs, and every unrelated scan attribute survives the rewrite unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[3] = scan [carry_count=1, length=3, reverse=true, unroll=3] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%1, %2, %1)"},
        );
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let TestOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(3));
        assert!(scan.reverse());
        assert_eq!(scan.unroll(), 3);
        assert_eq!(
            discharged.program().interpret(vec![scalar(2.0)]),
            Ok(vec![scalar(5.0), vector(vec![5.0, 4.0, 3.0]), scalar(5.0)]),
        );
    }

    #[test]
    fn test_scan_discharge_appends_the_synthesized_carry_after_the_declared_carry_prefix() {
        // A scan that already declares an ordinary carry pins the synthesized-state placement exactly: the state
        // operand joins the carry prefix behind every declared carry and ahead of the trailing stacked inputs, on the
        // parent operand list, the body boundary, and the rewritten carry count alike.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let carry = body_builder.add_input(scalar_type().into());
        let element = body_builder.add_input(scalar_type().into());
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element], None)
            .unwrap();
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, element], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![next_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let final_carry = builder
            .add_instruction(
                ScanOperation::<ArrayIrValue<CaptureArray>>::new(1, 3),
                vec![body],
                vec![initial, elements],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let closed =
            ClosedProgram::new(program, vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32)))])
                .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3] .
                let %3:f32[], %4:f32[] = scan [carry_count=2, length=3, reverse=false] %1 %0 %2 [
                    body={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %2
                            %4:f32[] = add %0 %2
                        in (%4, %3)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(discharged.external_states()[0].discharged_input_index(), 0);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));
        let CaptureOperation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.length(), &Dimension::Static(3));
    }

    #[test]
    fn test_read_only_loop_discharge_adds_no_final_state_output() {
        // A loop's boundaries stay symmetric — a carry position exists in the condition's and the body's boundaries or
        // in neither — but symmetry is a property of those boundaries, not a claim that the loop wrote what it carried.
        // Nothing here writes the external root, so it enters as state, rides the carry, and publishes no hidden final
        // state, which is what keeps its caller from writing an unchanged holder back.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let counter = condition_builder.add_input(scalar_type().into());
        let reference = condition_builder.add_input(reference_type.clone().into());
        let limit = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let counter = body_builder.add_input(scalar_type().into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let step = body_builder.add_constant(scalar(1.0));
        let next = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![next, reference], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let external = builder.add_input(reference_type.into());
        let counter = builder.add_input(scalar_type().into());
        let total = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![counter, external], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![total], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, 0, None)],
        );
        assert_eq!(discharged.program().interpret(vec![scalar(3.0), scalar(0.0)]), Ok(vec![scalar(3.0)]));
    }

    #[test]
    fn test_read_only_condition_discharge_adds_no_final_state_output() {
        // A closure that only reads an external root needs the state to enter both branches, but the root's value
        // never changes, so no branch gains a final-state result and the parent condition keeps exactly its public
        // outputs instead of carrying a dead state output.
        let reference_type = ReferenceType::new(scalar_type());
        let make_branch = || {
            let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = branch_builder.add_input(reference_type.clone().into());
            let snapshot = branch_builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                .unwrap()[0];
            branch_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(make_branch().entry_region_ref());
        let false_branch = builder.import_region(make_branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.program().interpret(vec![boolean(true), scalar(4.0)]), Ok(vec![scalar(4.0)]));
        assert_eq!(discharged.program().interpret(vec![boolean(false), scalar(4.0)]), Ok(vec![scalar(4.0)]));
    }

    #[test]
    fn test_scan_discharge_preserves_zero_length_state_identity() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(scalar(1.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 0), vec![body], vec![reference], None)
            .unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![final_value, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        assert_eq!(
            discharged.program().interpret(vec![scalar(2.0)]),
            Ok(vec![scalar(2.0), vector(Vec::new()), scalar(2.0)]),
        );
    }

    #[test]
    fn test_call_discharge_widens_a_positional_callee_with_its_final_state() {
        /// Array-IR family extended with one positional call, mirroring how a backend attaches a compiled callee
        /// region, forwards its operands positionally, and reports its outputs positionally.
        #[derive(Clone, Debug)]
        enum CallingOperation {
            /// Native array-IR operation.
            Native(TestOperation),

            /// Positional call of one attached callee region.
            Call,
        }

        impl Operation for CallingOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                match self {
                    Self::Native(operation) => operation.name(),
                    Self::Call => "test_call",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                match self {
                    Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
                    Self::Call => Ok(region_interfaces[0].output_types().to_vec()),
                }
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                match self {
                    Self::Native(operation) => operation.region_slots(),
                    Self::Call => const { &[RegionSlot::computation("callee")] },
                }
            }

            fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
                match self {
                    Self::Native(operation) => operation.input_region_provenance(region_index, input_index),
                    Self::Call => Some(input_index),
                }
            }

            fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
                match self {
                    Self::Native(operation) => operation.output_region_provenance(output_index),
                    Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                }
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                match self {
                    Self::Native(operation) => operation.reference_semantics(),
                    Self::Call => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                }
            }

            fn effects(&self) -> Effects {
                match self {
                    Self::Native(operation) => operation.effects(),
                    Self::Call => Effects::PURE,
                }
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                match self {
                    Self::Native(operation) => operation.render(formatter, indentation),
                    Self::Call => formatter.write_str(self.name()),
                }
            }
        }

        impl ArrayReferenceViewOperation for CallingOperation {
            fn from_reference_reshape(operation: ReshapeOperation) -> Self {
                Self::Native(TestOperation::from_reference_reshape(operation))
            }

            fn from_reference_slice(operation: SliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_slice(operation))
            }

            fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_update_slice(operation))
            }
        }

        impl From<AddOperation<ArrayIrType>> for CallingOperation {
            fn from(operation: AddOperation<ArrayIrType>) -> Self {
                Self::Native(operation.into())
            }
        }

        macro_rules! impl_calling_operation_from_reference_primitive {
            // Lifts one reference primitive into the calling family, which is the conversion seam a primitive rule
            // spends when it replays an access to a preserved root. A dispatch derive generates the same seam.
            ($payload:ident) => {
                impl From<$payload<ArrayType, ArrayIrType>> for CallingOperation {
                    fn from(operation: $payload<ArrayType, ArrayIrType>) -> Self {
                        Self::Native(operation.into())
                    }
                }
            };
        }

        impl_calling_operation_from_reference_primitive!(ReferenceNewOperation);
        impl_calling_operation_from_reference_primitive!(ReferenceReadOperation);
        impl_calling_operation_from_reference_primitive!(ReferenceWriteOperation);
        impl_calling_operation_from_reference_primitive!(ReferenceSwapOperation);
        impl_calling_operation_from_reference_primitive!(ReferenceAddUpdateOperation);
        impl_calling_operation_from_reference_primitive!(ReferenceFreezeOperation);

        // A third-party call-shaped family participates in structured discharge by reaching the shared positional
        // rewrite, exactly as the backend `jit_call` does, with no companion declaration surface beyond the generic
        // provenance hooks it already implements. Its reference primitives delegate to the universe-generic rules the
        // primitives own, and every other native payload replays as the enclosing enum, which is the same split a
        // dispatch derive generates.
        impl<C, P> ReferenceDischargeableOperation<C, P> for CallingOperation
        where
            C: Context<Type = ArrayIrType, Operation = CallingOperation>,
            P: ReferenceAccumulationPolicy<C>,
        {
            fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
                &self,
                context: &ReferenceDischargeContext<C, P>,
                driver: &D,
                inputs: &[ReferenceDischargeValue<C, P>],
            ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
                match self {
                    Self::Native(ArrayIrOperation::ReferenceNew(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(ArrayIrOperation::ReferenceRead(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(ArrayIrOperation::ReferenceWrite(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(ArrayIrOperation::ReferenceSwap(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(ArrayIrOperation::ReferenceAddUpdate(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(ArrayIrOperation::ReferenceFreeze(operation)) => {
                        operation.discharge_references(context, driver, inputs)
                    }
                    Self::Native(_) => discharge_reference_free_operation(self, context, driver, inputs),
                    Self::Call => discharge_positional_region_operation(self, context, driver, inputs, 0),
                }
            }
        }

        // The callee mutates the root it receives and returns only the old snapshot, so its declared boundary hides
        // the final state that the call site needs after discharge.
        let mut callee_builder = ProgramBuilder::<TestValue, CallingOperation>::new();
        let reference = callee_builder.add_input(ReferenceType::new(scalar_type()).into());
        let replacement = callee_builder.add_input(scalar_type().into());
        let old = callee_builder
            .add_instruction(
                CallingOperation::Native(ReferenceSwapOperation::new().into()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![old], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, CallingOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let replacement = builder.add_input(scalar_type().into());
        let root = builder
            .add_instruction(
                CallingOperation::Native(ReferenceNewOperation::new().into()),
                Vec::new(),
                vec![initial],
                None,
            )
            .unwrap()[0];
        let old = builder
            .add_instruction(CallingOperation::Call, vec![callee], vec![root, replacement], None)
            .unwrap()[0];
        let final_snapshot = builder
            .add_instruction(
                CallingOperation::Native(ReferenceFreezeOperation::new().into()),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Discharge appends the callee's final state to its outputs and threads that result into the freeze, leaving
        // the call itself in place with its positional operand and output contract intact.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = test_call %0 %1 [
                    callee={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_closed_program_discharge_resolves_reference_captures_inside_condition_regions() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = branch_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![branch, branch],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed = ClosedProgram::new(program, vec![ArrayIrValue::Reference(reference)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, 0, None)],
        );
        assert_eq!(
            serde_json::to_string(discharged.external_states()).unwrap(),
            concat!(
                r#"[{"source":{"capture":{"index":0}},"discharged_input_index":0,"#,
                r#""final_state_output_index":null}]"#,
            ),
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![scalar_type().into(), ArrayType::scalar(DataType::Boolean).into()],
        );
    }

    #[test]
    fn test_closed_program_discharge_resolves_transitively_nested_reference_captures() {
        let reference_type = ReferenceType::new(scalar_type());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut leaf_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = leaf_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = leaf_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let leaf = leaf_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut middle_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let leaf = middle_builder.import_region(leaf.entry_region_ref());
        let predicate = middle_builder.add_constant(Capture::new(1, predicate_type.clone().into()));
        let value = middle_builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![leaf, leaf],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let middle = middle_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let middle = builder.import_region(middle.entry_region_ref());
        let predicate = builder.add_constant(Capture::new(1, predicate_type.into()));
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![middle, middle],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder]).unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed = ClosedProgram::new(program, vec![ArrayIrValue::Reference(reference), boolean(true)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_states()[0].is_mutated());
    }

    #[test]
    fn test_closed_program_discharge_threads_reference_captures_through_while() {
        // A capture read only by the loop condition still needs a synthesized state input on both while regions, but no
        // final-state output because nothing writes it.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = condition_builder.add_constant(Capture::new(0, reference_type.into()));
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(Capture::new(1, ArrayType::scalar(DataType::Boolean).into()));
        let condition = condition_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![predicate], Vec::new(), vec![Placeholder])
            .unwrap();
        let body = ProgramBuilder::<Capture, CaptureOperation>::new()
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], Vec::new(), None)
            .unwrap();
        let while_program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32));
        let closed =
            ClosedProgram::new(while_program, vec![ArrayIrValue::Reference(concrete_reference), boolean(false)])
                .unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert!(matches!(
            discharged.program().entry_region_ref().instructions()[0].operation(),
            CaptureOperation::While(_),
        ));
    }

    #[test]
    fn test_closed_program_discharge_threads_reference_captures_through_scan() {
        // A capture read by a scan body becomes a synthesized carry appended after the declared carry prefix,
        // which raises the rewritten scan's carry count without disturbing its length, direction, or unroll factor.
        let reference_type = ReferenceType::new(scalar_type());
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32));
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrValue<CaptureArray>>::new(0, 2), vec![body], Vec::new(), None)
            .unwrap()[0];
        let scan_program =
            builder.build::<Vec<Capture>, Vec<Capture>>(vec![values], Vec::new(), vec![Placeholder]).unwrap();
        let closed = ClosedProgram::new(scan_program, vec![ArrayIrValue::Reference(concrete_reference)]).unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.program().output_types(), vec![vector(vec![0.0, 0.0]).r#type().into_owned()]);
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let CaptureOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(2));
        assert!(!scan.reverse());
        assert_eq!(scan.unroll(), 1);
    }

    #[test]
    fn test_closed_program_discharge_threads_mutated_reference_capture_through_scan() {
        // A capture that a scan body accumulates into reaches that body only through a synthesized carry, which is
        // the most involved discharge path: the state enters the scan appended after the declared carry prefix, is
        // updated inside the body, leaves through the matching synthesized carry output, and reaches the hidden
        // entry final-state output after the public prefix. The capture value family carries no data, so the
        // rendered program rather than an interpretation pins the resulting state flow.
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = body_builder.add_constant(Capture::new(0, reference_type.into()));
        let update = body_builder.add_constant(Capture::new(1, scalar_type().into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrValue<CaptureArray>>::new(0, 3), vec![body], Vec::new(), None)
            .unwrap()[0];
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(vec![values], Vec::new(), vec![Placeholder]).unwrap();
        let closed = ClosedProgram::new(
            program,
            vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32))), scalar(1.0)],
        )
        .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(discharged.external_states()[0].discharged_input_index(), 0);
        assert!(discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(1));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const capture#1:f32[]
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%3, %2)"},
        );
    }

    #[test]
    fn test_condition_discharge_matches_eager_reference_execution() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(scalar(1.0));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let replacement = false_builder.add_constant(scalar(9.0));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Each branch mutates the shared root differently, so the eager reference interpreter and the discharged
        // program must agree on the branch snapshot as well as on the state observed after the condition.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        for (predicate, expected) in [(true, vec![scalar(5.0), scalar(5.0)]), (false, vec![scalar(4.0), scalar(9.0)])] {
            let inputs = vec![boolean(predicate), scalar(4.0)];
            let eager = source.clone().interpret(inputs.clone()).unwrap();
            assert_eq!(eager, expected);
            assert_eq!(discharged.program().interpret(inputs), Ok(eager));
        }
    }

    #[test]
    fn test_condition_discharge_recreates_view_inside_region() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let view = builder
                .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
                .unwrap()[0];
            let update = builder.add_constant(scalar(1.0));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![view, update], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let true_inputs = vec![boolean(true), vector(vec![1.0, 2.0, 3.0])];
        let false_inputs = vec![boolean(false), vector(vec![1.0, 2.0, 3.0])];
        assert_eq!(source.clone().interpret(true_inputs.clone()), Ok(vec![vector(vec![1.0, 3.0, 3.0])]));
        assert_eq!(source.clone().interpret(false_inputs.clone()), Ok(vec![vector(vec![1.0, 2.0, 3.0])]));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(true_inputs), Ok(vec![vector(vec![1.0, 3.0, 3.0])]));
        assert_eq!(discharged.program().interpret(false_inputs), Ok(vec![vector(vec![1.0, 2.0, 3.0])]));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3] .
                let %2:f32[3] = condition %0 %1 [
                    true={
                        lambda %0:f32[3] .
                        let %1:f32[] = const 1.0
                            %2:f32[1] = slice [start_indices=[1], limit_indices=[2]] %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = add %3 %1
                            %5:f32[1] = reshape [shape=[1]] %4
                            %6:f32[3] = update_slice [start_indices=[1]] %0 %5
                        in (%6)
                    },
                    false={
                        lambda %0:f32[3] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_while_discharge_matches_hand_written_immutable_state_passing_loop() {
        // Eager interpretation cannot execute a reference-carrying `while` at all, because masked predicate selection
        // has no meaning for reference carries, so the oracle here is a hand-written immutable loop that threads the
        // same state through an ordinary array carry instead of an eager run of the reference program.
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(boolean(true));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(2.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        oracle_condition_builder.add_input(scalar_type().into());
        let predicate = oracle_condition_builder.add_constant(boolean(true));
        let oracle_condition = oracle_condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = oracle_body_builder.add_input(scalar_type().into());
        let update = oracle_body_builder.add_constant(scalar(2.0));
        let updated = oracle_body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![state, update], None)
            .unwrap()[0];
        let oracle_body = oracle_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![updated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let oracle_condition = oracle_builder.import_region(oracle_condition.entry_region_ref());
        let oracle_body = oracle_builder.import_region(oracle_body.entry_region_ref());
        let state = oracle_builder.add_input(scalar_type().into());
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let final_state = oracle_builder
            .add_instruction(operation, vec![oracle_condition, oracle_body], vec![state], None)
            .unwrap()[0];
        let oracle = oracle_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![final_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The condition region observes the carried state while the body accumulates into it, so the discharged loop
        // must reproduce the immutable loop for every initial state.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        for initial in [0.0f32, 1.0, -4.5] {
            let expected = oracle.clone().interpret(vec![scalar(initial)]).unwrap();
            assert_eq!(discharged.program().interpret(vec![scalar(initial)]), Ok(expected));
        }
        assert_eq!(discharged.program().interpret(vec![scalar(1.0)]), Ok(vec![scalar(7.0)]));
    }

    #[test]
    fn test_scan_discharge_matches_eager_reference_execution() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(3.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 4), vec![body], vec![reference], None)
            .unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![frozen, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The declared reference carry and the accumulating body must agree with eager execution on both the stacked
        // per-iteration snapshots and the state observed after the scan.
        let eager = source.clone().interpret(vec![scalar(0.0)]).unwrap();
        assert_eq!(eager, vec![scalar(12.0), vector(vec![3.0, 6.0, 9.0, 12.0])]);
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(0.0)]), Ok(eager));
    }

    #[test]
    fn test_dynamic_length_scan_discharge_accepts_the_trailing_runtime_length_operand() {
        // A dynamic-length scan carries one runtime-length operand after the body's inputs, so the scan discharge
        // rule's arity validation must accept the one-past-body parent arity instead of rejecting the canonical
        // dynamic form.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(9)).unwrap());
        let reference_type = ReferenceType::new(scalar_type());
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(scalar(3.0));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let runtime_length = builder.add_input(DimensionType::new(length.clone()).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let scanned = builder
            .add_instruction(
                ScanOperation::<TestValue>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![reference, runtime_length],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![scanned], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let runtime_length = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(length), 4).unwrap());
        let eager = source.clone().interpret(vec![scalar(0.0), runtime_length.clone()]).unwrap();
        assert_eq!(eager, vec![scalar(12.0)]);
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().interpret(vec![scalar(0.0), runtime_length]), Ok(eager));
    }
}
