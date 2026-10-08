//! Contains the `scan` control-flow operation: [`ScanOperation`], a shape-determined loop that threads `carry_count`
//! loop-carried values through an attached body [`Region`] while consuming one slice of every stacked array input and
//! producing one slice of every stacked output per iteration, together with its reference discharge, interpretation,
//! partial-evaluation, batching, forward-mode differentiation, and transposition rules. This is the analogue of [JAX's
//! `lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) (including `reverse` and the
//! lowering-only `unroll` factor) and lowers to a [StableHLO `while`](https://openxla.org/stablehlo/spec#while) loop
//! with counter-indexed slice reads and writes.

use std::collections::{BTreeSet, HashMap};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayOperation, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, DataType,
    Dimension, DimensionType, DimensionValue, DimensionVariable, MAX_DIMENSION_EXTENT, MeshAxisType, Shape,
    ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentDestinationKind, CotangentDestinations, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero::{DynamicZero, Zero, ZeroOperation};
use crate::operations::control_flow::{
    TemporalResidualOperation, TemporalResidualType, refine_output_types, region_input_mismatch,
    validate_output_identities,
};
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
use crate::operations::manipulation::slicing::{Slice, SliceOperation, UpdateSlice, UpdateSliceOperation};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::references::ReferenceNewOperation;
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::{
    Atom, AtomId, CalleeRegionDriver, InputRegionProvenance, MaybeZero, Operation, OperationBoundaryPruning,
    OperationFormatter, OperationProjection, OperationProvider, OutputRegionProvenance, Program, ProgramBuilder,
    ProgramError, ReferenceAccessOperation, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion,
    ReferenceDischargeRegionInput, ReferenceDischargeRegionOutput, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceRoot, ReferenceType, Region, RegionArena, RegionInterface,
    RegionLiveness, RegionRef, RegionSlot, Type, TypeError, TypeIdentityPosition, TypeIdentityRenaming, Typed, Value,
    ValueProjection, rewrite_reference_access_transforms, validated_reference_access_descriptors,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this since it is mostly vibe coded.

/// Canonical operation name for [`ScanOperation`].
pub const SCAN_OPERATION_NAME: &str = "scan";

/// [`Operation`] that applies a nested body [`Program`] a shape-determined number of times over loop-carried state,
/// consuming one slice of each stacked array input per iteration and stacking the body's per-iteration outputs.
/// Reference inputs supply complete roots so the body can select its own views. The shape-determined trip count lets
/// linearization save per-iteration residuals in stacks with known shapes and lets transposition process those stacks
/// in reverse visit order.
///
/// The body [`Program`] maps `[index, carry..., x_slice_or_reference...]` to `[carry..., y_slice...]`. Input zero is a
/// scalar [`I64`](DataType::I64) array containing the selected slice index. The next [`carry_count`](Self::carry_count)
/// inputs and the first `carry_count` outputs hold loop-carried state with matching types. Remaining array inputs
/// receive one slice of a stacked instruction input; remaining reference inputs receive the whole reference unchanged.
/// Remaining outputs contribute slices to stacked array results.
///
/// Operation inputs are `[carry..., stacked_xs...]` and outputs are `[final_carry..., stacked_ys...]`. The intrinsic
/// index is not an operation input or carry. Iteration `i` consumes slice `i` of every stacked array input and produces
/// slice `i` of every stacked output. With [`reverse`](Self::reverse), the body receives indices from `length - 1` down
/// to `0`: visit order changes but input/output slice pairing stays the same. Transposition flips that order and
/// obtains the index from the reversed loop, without storing a stack of indices.
///
/// The explicit [`length`](Self::length) also supports scans with no stacked inputs. Homogeneous [`ArrayType`] scans
/// require a static length. Composite [`ArrayIrType`] scans may use a dynamic dimension identity and consume its
/// matching first-class dimension value as a trailing runtime input. Every stacked leading axis must then have the
/// same runtime extent: the same dimension identity, or a static extent proved equal by singleton bounds. An unrelated
/// runtime length identity is admitted only when its bounds fix one exact extent and every stack has that extent.
///
/// Batching rejects bounded ragged inputs because the body's structurally batched boundary cannot retain their
/// per-item extent carriers. Shared dynamic dimensions remain supported by composite scans.
///
/// A composite body receiving `ref<[length, t]>` selects its current slice by attaching a leading dynamic index
/// transform to each access, with the body's index input as the first transform binding. Reference operations and their
/// transforms preserve that access path. The root parameter must not be consumed or frozen inside the body. Stacked
/// reference outputs remain unsupported: a scan cannot assemble a reference value from per-iteration handles. Discharge
/// can replace accesses confined to the current slice with ordinary stacked array inputs and updated slice outputs,
/// preserving slice-sized state without exposing a second reference-body interface.
///
/// The optional [`unroll`](Self::unroll) factor (attached via [`with_unroll`](Self::with_unroll)) is a
/// **lowering-only** attribute: interpretation and every transform rule (differentiation, transposition, batching)
/// ignore it semantically but preserve it on whatever scan they re-stage, while lowerings emit `unroll` body copies per
/// loop trip followed by the `length % unroll` remaining iterations, and no loop at all when `unroll` is at least a
/// static `length`.
///
/// The body computation is not part of this payload: it is a [`Region`] attached to the
/// [`Instruction`](crate::Instruction) applying the operation (the single [`region_slots`](Operation::region_slots)
/// slot `["body"]`), and semantic rules reach it through their driver-granted region access. Scans with owned bodies
/// supply the body [`Program`] through the region driver passed to [`Context::bind`]; [`Operation::infer_output_types`]
/// validates the body signature over the attached [`RegionInterface`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScanOperation<T: Type> {
    /// Number of loop-carried state leaves after the body index input and at the front of its outputs.
    pub(crate) carry_count: usize,

    /// Shape-determined trip count of this [`ScanOperation`].
    pub(crate) length: Dimension,

    /// Boolean indicating whether iterations visit the stacked slices in reverse order.
    pub(crate) reverse: bool,

    /// Lowering-only unroll factor: the number of body copies emitted per loop trip (`1` keeps one body per trip).
    pub(crate) unroll: usize,

    /// Type universe of the loop-carried state, stacked values, and attached body region.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> ScanOperation<T> {
    /// Creates a new [`ScanOperation`] with the provided carry count and shape-determined trip count, visiting
    /// iterations in increasing order (use [`Self::with_reverse`] to flip the visit order). The body [`Program`]
    /// mapping `[index, carry..., x_slice_or_reference...]` to `[carry..., y_slice...]` is supplied separately as the
    /// operation's attached region (via the region driver passed to [`Context::bind`]), and
    /// [`Operation::infer_output_types`] validates its signature against `carry_count` and `length`.
    ///
    /// # Parameters
    ///
    ///   - `carry_count`: Number of loop-carried state leaves, excluding the intrinsic body index input.
    ///   - `length`: Shape-determined trip count.
    #[inline]
    pub fn new<L: Into<Dimension>>(carry_count: usize, length: L) -> Self {
        Self { carry_count, length: length.into(), reverse: false, unroll: 1, marker: PhantomData }
    }

    /// Returns this [`ScanOperation`] with the slice visit order set to `reverse`. Reversal changes the order in which
    /// the body observes iteration indices, carries, and slices, while slice `i` of every stacked output still pairs
    /// with slice `i` of every stacked input.
    #[inline]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns this [`ScanOperation`] with the lowering unroll factor set to `unroll`, which must be at least `1`.
    /// Unrolling is lowering-only: interpretation and transform rules ignore the factor but preserve it on every scan
    /// they derive, while lowerings emit `unroll` consecutive body copies per loop trip and follow the loop with the
    /// `length % unroll` remaining iterations. A factor of at least a static [`length`](Self::length) unrolls the scan
    /// completely, with no loop at all (e.g., `usize::MAX` fully unrolls a scan of any static length).
    pub fn with_unroll(mut self, unroll: usize) -> Result<Self, ProgramError> {
        if unroll == 0 {
            return Err(TypeError::invalid(format!("`{SCAN_OPERATION_NAME}` unroll factor must be at least 1")).into());
        }
        self.unroll = unroll;
        Ok(self)
    }

    /// Returns the number of loop-carried state leaves of this [`ScanOperation`].
    #[inline]
    pub fn carry_count(&self) -> usize {
        self.carry_count
    }

    /// Returns the shape-determined trip count of this [`ScanOperation`].
    #[inline]
    pub fn length(&self) -> &Dimension {
        &self.length
    }

    /// Returns `true` when iterations of this [`ScanOperation`] visit the stacked slices in reverse order.
    #[inline]
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Returns the lowering-only unroll factor of this [`ScanOperation`] (the number of body copies emitted per loop
    /// trip; `1` when no unrolling was requested via [`with_unroll`](Self::with_unroll)).
    #[inline]
    pub fn unroll(&self) -> usize {
        self.unroll
    }

    /// Returns this [`ScanOperation`] with `additional_carry_count` extra carries appended after its existing carries,
    /// preserving its length, direction, and unroll factor. Reference discharge uses this to widen a scan with
    /// discharged reference state carries.
    fn with_added_carries(&self, additional_carry_count: usize) -> Result<Self, ProgramError> {
        let carry_count = self.carry_count.checked_add(additional_carry_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` carry count {} overflows when adding {additional_carry_count} discharged \
                 reference state carries",
                self.carry_count,
            ))
        })?;
        Ok(Self { carry_count, ..self.clone() })
    }
}

impl<T: ScanType> Display for ScanOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: ScanType> Operation for ScanOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        SCAN_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 1, TypeError);
        // Specialization refines a valid source body; it must not repair an invalid index or carry contract.
        T::validate_scan_body_type_signature(
            region_interfaces[0].input_types(),
            region_interfaces[0].output_types(),
            self.carry_count,
            &self.length,
        )?;
        let body_input_types = T::infer_scan_body_input_types(
            input_types,
            region_interfaces[0].input_types().len(),
            self.carry_count,
            &self.length,
        )?;
        let same_identities = region_interfaces[0]
            .input_types()
            .iter()
            .zip(&body_input_types)
            .all(|(declared, requested)| declared.identities().eq(requested.identities()));
        // The body is requested at the instantiated input types when their type identities differ from the declared
        // ones or when they strictly refine the declared ones (e.g., with a sharding that the actual inputs carry but
        // the body leaves unspecified), so that staging and differentiation specialize it and the programs derived
        // from it (e.g., its forward-mode derivative and its transposition) carry those refinements. Comparing type
        // identities explicitly also detects identity substitutions that are not directional refinements.
        let refines_declared_types = region_interfaces[0]
            .input_types()
            .iter()
            .zip(&body_input_types)
            .all(|(declared, requested)| declared.is_refined_by(requested))
            && region_interfaces[0].input_types() != body_input_types;
        if same_identities && !refines_declared_types {
            return Ok(vec![None]);
        }
        Ok(vec![Some(body_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("region", region_interfaces, 1, TypeError);
        validate_scan_length(&self.length)?;
        let output_types = T::infer_scan_output_types(
            region_interfaces[0].input_types(),
            region_interfaces[0].output_types(),
            self.carry_count,
            &self.length,
            input_types,
        )?;
        validate_output_identities(SCAN_OPERATION_NAME, input_types, output_types.as_slice())?;
        Ok(output_types)
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index != 0 {
            return InputRegionProvenance::None;
        }
        // The first input is generated by the loop. Reference instruction inputs, including whole stacks, enter
        // unchanged; any per-iteration reference view is an explicit instruction in the body.
        match input_index {
            0 => InputRegionProvenance::Local,
            index => InputRegionProvenance::Input { index: index - 1 },
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn prune_boundary(
        &self,
        input_count: usize,
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        let carry_count = self.carry_count;
        if used_outputs.len() < carry_count {
            return Err(ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` has {} outputs but carries {carry_count} values",
                used_outputs.len(),
            )));
        }

        // The body maps `[index, carry..., x...]` to `[carry..., y...]`, and each carry is kept or dropped as an input
        // and output pair. Keeping a carry output can make carry inputs live, which keeps the corresponding carry
        // outputs in turn, so the kept carries are the least fixed point that contains the used carry outputs (as in
        // JAX's `_scan_dce_rule`). Stacked inputs and outputs are pruned independently, while the index input and the
        // runtime length input that trails the stacked inputs of a scan with a dynamic length are always kept.
        let (used_carries, used_stacked_outputs) = used_outputs.split_at(carry_count);
        let mut kept_carries = used_carries.to_vec();
        let used_body_inputs = loop {
            let body_outputs = kept_carries.iter().chain(used_stacked_outputs).copied().collect::<Vec<_>>();
            let used_body_inputs = regions.used_region_inputs(0, &body_outputs)?;
            let mut changed = false;
            for (kept, used) in kept_carries.iter_mut().zip(used_body_inputs.iter().skip(1)) {
                changed |= *used && !*kept;
                *kept |= *used;
            }
            if !changed {
                break used_body_inputs;
            }
        };
        let stacked_count = used_body_inputs.len().checked_sub(1 + carry_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` body has fewer inputs than its index and {carry_count} carries",
            ))
        })?;
        let trailing_count = input_count.checked_sub(carry_count + stacked_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` has {input_count} inputs but its body takes {carry_count} carries and \
                 {stacked_count} stacked inputs",
            ))
        })?;
        let runtime_length_count = usize::from(self.length.variable().is_some());
        if trailing_count != runtime_length_count {
            return Err(ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` expects {runtime_length_count} trailing runtime length inputs but has \
                 {trailing_count}",
            )));
        }
        Ok(Some(OperationBoundaryPruning {
            operation: Self { carry_count: kept_carries.iter().filter(|kept| **kept).count(), ..self.clone() },
            kept_inputs: kept_carries
                .iter()
                .chain(&used_body_inputs[1 + carry_count..])
                .copied()
                .chain(std::iter::repeat_n(true, trailing_count))
                .collect(),
            kept_outputs: kept_carries.iter().chain(used_stacked_outputs).copied().collect(),
        }))
    }

    #[inline]
    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        // Only the leading carries preserve their input allocations positionally; stacked per-step outputs are fresh
        // values with no identity constraint.
        (output_index < self.carry_count).then_some(output_index)
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self { length: self.length.rename_type_identities(renaming), ..self.clone() })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCAN_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("carry_count", self.carry_count)?;
            operation.field("length", &self.length)?;
            operation.field("reverse", self.reverse)?;
            if self.unroll > 1 {
                operation.field("unroll", self.unroll)?;
            }
            Ok(())
        })
    }
}

// Scan owns iteration and state threading, while reference discharge decides how stacked reference state is selected
// and reconstructed. A reference stack whose every access selects the body's current row (a leading dynamic index
// transform bound to the body's index input) is replaced by slice-sized state: the leading index is removed from each
// access and its binding, keeping every instruction in place so that partial-discharge targets and capture scopes,
// which name source region, atom, and instruction identities, retain their meaning. When the body uses a complete root
// instead (for example, by passing it and the index into a nested call), the allocation state becomes a carry, which
// preserves arbitrary validated indexing without assuming that a nested input has the same runtime value as the scan
// index. Preserved reference stacks keep their complete handles and explicit indexing, and a zero-trip scan executes
// no selection and leaves the stacked state unchanged.
//
// Discharged state joins the leading carry prefix rather than following the declared inputs: the synthesized carries
// are inserted immediately after the source carries, in the parent input list, in the body's input boundary, and in
// both output boundaries, and the operation's carry count grows to match. Like `while`, a scan applies no read-only
// pruning, because a carry position exists in both boundaries or in neither. A carry that partial reference discharge
// *preserved* stays a declared carry: it keeps its position and its reference type on both boundaries, its accesses
// replay inside the body, and it publishes no successor. A preserved allocation reached only through an inherited
// capture gains the same kind of reference-typed carry so that the rebuilt body can bind that capture without turning
// it into state.
impl<C, P> ReferenceDischargeableOperation<C, P> for ScanOperation<ArrayIrType>
where
    C: Context<
            Type = ArrayIrType,
            Operation: ReferenceAccessOperation<Transform = ArrayReferenceTransform>
                           + From<DimensionSizeOperation>
                           + From<ScanOperation<ArrayIrType>>,
        > + DynamicZero<C::Value>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let name = self.name();
        self.validate_region_count(driver.region_count())?;
        let carry_count = self.carry_count();
        if inputs.len() < carry_count {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{name}` declares {carry_count} carries but the application has {} inputs",
                inputs.len(),
            )));
        }
        let (carry_inputs, stacked_inputs) = inputs.split_at(carry_count);
        let carries =
            carry_inputs.iter().map(|input| context.boundary_allocation(input)).collect::<Result<Vec<_>, _>>()?;

        let source_body = driver.region(0)?;
        let source_input_types = source_body.input_types();
        let source_stacked_input_end = source_input_types.len().checked_sub(1).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input"))
        })?;
        check_count!(
            "input",
            inputs,
            source_stacked_input_end + usize::from(self.length.variable().is_some()),
            ProgramError
        );
        if source_stacked_input_end < carry_count {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{name}` declares more carries than body inputs"
            )));
        }
        let mut body_allocations = vec![None];
        body_allocations.extend(
            inputs
                .iter()
                .take(source_input_types.len() - 1)
                .map(|input| context.boundary_allocation(input))
                .collect::<Result<Vec<_>, _>>()?,
        );

        // Discharged reference stacks can use slice-sized state when the body only selects its own current row.
        // Preserve source region, atom, and instruction identities: partial-discharge targets and capture scopes
        // name those identities. Removing the leading index from each access and its dynamic binding leaves every
        // instruction in place, keeping later allocation target identities unchanged.
        let mut normalized = None;
        let mut viewed_inputs = BTreeSet::new();
        let mut carried_inputs = Vec::new();
        for position in 1 + carry_count..source_input_types.len() {
            let Some(allocation) = body_allocations[position] else {
                continue;
            };
            if !context.is_allocation_discharged(allocation)? {
                continue;
            }
            if body_allocations
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != position && *candidate == Some(allocation))
            {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` cannot discharge overlapping stacked reference inputs as independent slices"
                )));
            }
            let root = source_body.input_ids()[position];
            let index = source_body.input_ids()[0];
            let reference = <&ReferenceType<ArrayType>>::try_from(&source_input_types[position])?;
            let selection = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
            let slice_type = selection.output_type(reference.referent())?;
            if source_body.output_ids().contains(&root) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` returns a stacked reference root from its body"
                )));
            }
            let mut selections = Vec::new();
            let mut slice_only = true;
            for (instruction_index, instruction) in source_body.instructions().iter().enumerate() {
                if !instruction.inputs().contains(&root) {
                    continue;
                }
                let effects = instruction.operation().effects();
                if !instruction.regions().is_empty()
                    || effects.classes().into_iter().any(|class| effects.declares(class))
                    || (0..instruction.outputs().len())
                        .any(|output| instruction.operation().reference_output_identity_input(output).is_some())
                {
                    slice_only = false;
                    break;
                }
                let descriptors =
                    validated_reference_access_descriptors(instruction.operation(), instruction.inputs().len())?;
                for (input_position, input) in instruction.inputs().iter().enumerate() {
                    if *input != root {
                        continue;
                    }
                    let Some(descriptor) = &descriptors[input_position] else {
                        slice_only = false;
                        break;
                    };
                    if !matches!(
                        descriptor.transforms().first(),
                        Some(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic })
                    ) || instruction.inputs().get(descriptor.bindings().start) != Some(&index)
                    {
                        slice_only = false;
                        break;
                    }
                    selections.push((instruction_index, input_position));
                }
                if !slice_only {
                    break;
                }
            }
            if !slice_only {
                carried_inputs.push(position);
                continue;
            }
            for (instruction_index, input_position) in selections {
                let normalized = normalized.get_or_insert_with(|| source_body.region().clone());
                let instruction = &normalized.instructions[instruction_index];

                // The selection pass above found a descriptor at this position, and rewriting another access of the
                // same instruction keeps this access's descriptor, so the lookup cannot fail.
                let descriptor =
                    validated_reference_access_descriptors(instruction.operation(), instruction.inputs().len())?
                        .into_iter()
                        .nth(input_position)
                        .flatten()
                        .unwrap();
                let bindings = instruction.inputs()[descriptor.bindings()].iter().skip(1).copied().collect();
                normalized.instructions[instruction_index] = rewrite_reference_access_transforms(
                    instruction,
                    input_position,
                    descriptor.transforms()[1..].to_vec(),
                    bindings,
                )?;
            }
            normalized.get_or_insert_with(|| source_body.region().clone()).atoms[root.index()] =
                Atom::Variable(ReferenceType::new(slice_type).into());
            viewed_inputs.insert(position);
        }
        let arena = normalized
            .map(|normalized| {
                let normalized =
                    Region::new(normalized.atoms, normalized.input_ids, normalized.output_ids, normalized.instructions);
                let mut regions =
                    source_body.arena().iter().take(source_body.id().index()).cloned().collect::<Vec<_>>();
                regions.push(normalized);
                RegionArena::from_regions(regions)
            })
            .transpose()?;
        let body = match &arena {
            Some(arena) => RegionRef::new(arena, source_body.id())?,
            None => source_body,
        };
        let summary = context.region_summary(self, 0, body, &body_allocations)?;

        // Reference carries must return the allocation they received, including when no iteration executes.
        if summary.output_allocations().len() < carry_count {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{name}` declares more carries than body outputs",
            )));
        }
        for (position, (returned, carry)) in
            summary.output_allocations()[..carry_count].iter().zip(&carries).enumerate()
        {
            if returned != carry {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` does not return carry {position} as the reference it entered with, so its \
                     `{SCAN_OPERATION_NAME}` state has no fixed point",
                )));
            }
        }
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let length = effective_scan_length(&self.length, &input_types, carry_count)?;
        // No reference access executes for zero trips, including accesses through declared carries or captures.
        // Elide bodies whose whole-root states would otherwise lower an unreachable index into an empty axis.
        let mut has_empty_root = false;
        for allocation in summary.reached_allocations() {
            let reference = context.allocation_reference(allocation)?;
            has_empty_root |= context.is_allocation_discharged(allocation)?
                && reference
                    .r#type()
                    .referent()
                    .shape()
                    .dimensions()
                    .iter()
                    .any(|dimension| dimension.bounds().extent() == Some(0));
        }
        if length.bounds().extent() == Some(0) && (!carried_inputs.is_empty() || has_empty_root) {
            // Keep the inferred dimension identities even when their bounds pin zero. Dynamic zeros consume the
            // corresponding runtime geometry, so this shortcut preserves the public region signature.
            let output_types = self.infer_output_types(&input_types, &[source_body.interface()])?;
            let boundary_values =
                inputs.iter().map(|input| context.boundary_value(input)).collect::<Result<Vec<_>, _>>()?;
            let has_output_geometry = output_types[carry_count..].iter().all(|output_type| {
                let ArrayIrType::Array(output_type) = output_type else { return false };
                output_type.shape().dimensions().iter().filter_map(Dimension::variable).all(|variable| {
                    boundary_values.iter().any(|value| match value.r#type().as_ref() {
                        ArrayIrType::Dimension(r#type) => r#type.variable() == variable,
                        ArrayIrType::Array(r#type) => {
                            r#type.shape().dimensions().iter().any(|dimension| dimension.variable() == Some(variable))
                        }
                        ArrayIrType::Reference(_) => false,
                    })
                })
            });
            // A preserved reference can be the only carrier of a dynamic slice extent. Keep the scan boundary in
            // that case: its body owns the geometry, and no iteration or reference read executes for zero trips.
            if has_output_geometry {
                let mut outputs = carry_inputs.to_vec();
                for output_type in &output_types[carry_count..] {
                    let ArrayIrType::Array(output_type) = output_type else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "operation `{name}` stacks the non-array output type `{output_type}`",
                        )));
                    };
                    let dimensions = output_type
                        .shape()
                        .dimensions()
                        .iter()
                        .filter_map(Dimension::variable)
                        .map(|variable| {
                            if let Some(input) = inputs.iter().find(|input| {
                                matches!(input.r#type().as_ref(), ArrayIrType::Dimension(r#type)
                                        if r#type.variable() == variable)
                            }) {
                                return context.boundary_value(input);
                            }
                            // An array also carries its runtime axis geometry. Reify that dimension when the boundary
                            // has no separate dimension leaf rather than dropping a valid symbolic output shape.
                            for input in inputs {
                                let value = context.boundary_value(input)?;
                                let input_type = value.r#type();
                                if let ArrayIrType::Array(array_type) = input_type.as_ref()
                                    && let Some(axis) = array_type
                                        .shape()
                                        .dimensions()
                                        .iter()
                                        .position(|dimension| dimension.variable() == Some(variable))
                                {
                                    let mut dimensions = context.parent().bind(
                                        DimensionSizeOperation::new(array_type, axis)?,
                                        Vec::new(),
                                        &[value],
                                    )?;
                                    check_count!("output", dimensions, 1, ProgramError);
                                    return Ok(dimensions.remove(0));
                                }
                            }
                            Err(ProgramError::UnsupportedOperation {
                                message: format!(
                                    "zero-trip `{SCAN_OPERATION_NAME}` output `{output_type}` has no input geometry \
                                     for dimension `{variable}`",
                                ),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    outputs
                        .push(ReferenceDischargeValue::Value(context.parent().dynamic_zero(output_type, &dimensions)?));
                }
                return Ok(outputs);
            }
        }
        let carried_allocations =
            carried_inputs.iter().map(|&position| body_allocations[position].unwrap()).collect::<Vec<_>>();
        let declared = body_allocations.iter().copied().flatten().collect::<BTreeSet<_>>();
        let widening = context.boundary_widening(&summary, &declared)?;
        let entering = widening.entering().to_vec();
        let declared_inputs = body_allocations
            .iter()
            .enumerate()
            .map(|(position, allocation)| match *allocation {
                None => ReferenceDischargeRegionInput::Value,
                Some(allocation) if viewed_inputs.contains(&position) => {
                    ReferenceDischargeRegionInput::View(allocation)
                }
                Some(allocation) => ReferenceDischargeRegionInput::Allocation(allocation),
            })
            .collect::<Vec<_>>();

        let view_outputs = body_allocations
            .iter()
            .enumerate()
            .skip(1 + carry_count)
            .filter_map(|(position, allocation)| {
                allocation
                    .filter(|allocation| viewed_inputs.contains(&position) && widening.published().contains(allocation))
                    .map(|_| ReferenceDischargeRegionOutput::View(position))
            })
            .collect();
        let mut state_allocations = entering.clone();
        state_allocations.extend(&carried_allocations);
        let state = ReferenceDischargeRegionBoundaryInsertion::new(state_allocations, carry_count);
        let boundary = ReferenceDischargeRegionBoundary::new(
            self,
            0,
            declared_inputs,
            ReferenceDischargeRegionBoundaryInsertion::new(entering.clone(), 1 + carry_count),
            [
                state.into(),
                ReferenceDischargeRegionBoundaryInsertion::new(view_outputs, driver.region(0)?.output_ids().len()),
            ],
        );
        let result = driver.rebuild_region(context, body, &boundary)?;
        result.validate_predicted_mutations(widening.published(), name)?;
        result.validate_predicted_output_allocations(summary.output_allocations(), name)?;

        let source_output_count = result.output_allocations().len();

        // A stacked reference input contributes its allocation's current state when discharged and its destination
        // reference when preserved, exactly like a carry does.
        let mut discharged_inputs = Vec::with_capacity(inputs.len() + entering.len());
        for input in carry_inputs {
            discharged_inputs.push(context.boundary_value(input)?);
        }
        for allocation in &entering {
            discharged_inputs.push(
                context
                    .boundary_value(&ReferenceDischargeValue::Reference(context.allocation_reference(*allocation)?))?,
            );
        }
        for &position in &carried_inputs {
            discharged_inputs.push(context.boundary_value(&inputs[position - 1])?);
        }
        for (index, input) in stacked_inputs.iter().enumerate() {
            if !carried_inputs.contains(&(1 + carry_count + index)) {
                discharged_inputs.push(context.boundary_value(input)?);
            }
        }
        let published_views = boundary
            .added_outputs()
            .iter()
            .flat_map(|group| group.sources())
            .filter_map(|output| match output {
                ReferenceDischargeRegionOutput::View(position) => Some(*position),
                ReferenceDischargeRegionOutput::Allocation(_) => None,
            })
            .collect::<Vec<_>>();
        let mut program = result.into_program();
        if !carried_inputs.is_empty() {
            // Move whole-root states into the carry prefix. Their final states were inserted alongside the other added
            // carries above; ordinary stacked inputs and optimized per-row states retain their order.
            let prefix = 1 + carry_count + entering.len();
            let moved = carried_inputs.iter().map(|position| position + entering.len()).collect::<Vec<_>>();
            let input_order = (0..prefix)
                .chain(moved.iter().copied())
                .chain((prefix..program.input_types().len()).filter(|position| !moved.contains(position)))
                .collect::<Vec<_>>();
            let output_order = (0..program.output_count()).collect::<Vec<_>>();
            program = reorder_program_boundary(&program, &input_order, &output_order)?;
        }
        let outputs = context.parent().bind(
            self.with_added_carries(entering.len() + carried_inputs.len())?,
            vec![program],
            discharged_inputs.as_slice(),
        )?;
        let published_offset = source_output_count + entering.len() + carried_inputs.len();
        check_count!("output", outputs, published_offset + published_views.len(), ProgramError);

        let mut results = Vec::with_capacity(source_output_count);
        for (position, output) in outputs.into_iter().enumerate() {
            if position < carry_count {
                match carries[position] {
                    Some(allocation) => {
                        context.merge_boundary_state(&summary, &widening, allocation, output)?;
                        results.push(carry_inputs[position].clone());
                    }
                    None => results.push(ReferenceDischargeValue::Value(output)),
                }
            } else if position < carry_count + entering.len() {
                let allocation = entering[position - carry_count];
                context.merge_boundary_state(&summary, &widening, allocation, output)?;
            } else if position < carry_count + entering.len() + carried_inputs.len() {
                let allocation = carried_allocations[position - carry_count - entering.len()];
                context.set_discharged_state(allocation, output, summary.is_mutated(allocation))?;
            } else if position < published_offset {
                results.push(ReferenceDischargeValue::Value(output));
            } else {
                // The appended stacked outputs are the final per-iteration states of the published views, in declared
                // input order. Their stacked type is the allocation's referent type, so each installs its allocation's
                // successor state directly; a published view always names an allocation, by construction of the
                // boundary above.
                let allocation = body_allocations[published_views[position - published_offset]].unwrap();
                context.set_discharged_state(allocation, output, true)?;
            }
        }
        Ok(results)
    }
}

impl<C> InterpretableOperation<C> for ScanOperation<ArrayType>
where
    C: Domain<Type = ArrayType> + Zero<C::Value> + Fill<i64, C::Value>,
    C::Value: Slice + UpdateSlice + Reshape,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        validate_scan_length(&self.length)?;
        let length = self.length.value().ok_or_else(|| ProgramError::UnsupportedOperation {
            message: format!(
                "cannot eagerly interpret homogeneous array `{SCAN_OPERATION_NAME}` with dynamic length `{}` \
                 without an explicit first-class dimension input",
                self.length,
            ),
        })?;
        self.validate_region_count(driver.region_count())?;
        let body = driver.region(0)?;
        let body_input_count = body.input_types().len().checked_sub(1).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input"))
        })?;
        check_count!("input", inputs, body_input_count, ProgramError);
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[body.interface()])?;
        // Iteration `iteration` consumes slice `iteration` of every stacked input and writes slice `iteration` of every
        // stacked output. With `reverse`, the iterations are visited from `length - 1` down to `0`, which changes the
        // visit order but not this pairing. Output stacks are allocated from the body's slice types up front, so a
        // zero-trip scan still returns correctly shaped (empty) stacks.
        let (carries, stacks) = inputs.split_at(self.carry_count);
        let mut carries = carries.to_vec();
        let mut accumulators = driver.region(0)?.interface().output_types()[self.carry_count..]
            .iter()
            .map(|slice_type| context.zero(&stacked_scan_type(slice_type, length)))
            .collect::<Result<Vec<_>, _>>()?;
        for visit in 0..length {
            let iteration = if self.reverse { length - 1 - visit } else { visit };
            let mut iteration_inputs = vec![context.fill(&ArrayType::scalar(DataType::I64), iteration as i64)?];
            iteration_inputs.extend(carries.iter().cloned());
            for stack in stacks {
                iteration_inputs.push(read_scan_iteration(stack, iteration)?);
            }
            let mut iteration_outputs = driver.interpret_region(context, 0, iteration_inputs)?;
            check_count!("output", iteration_outputs, self.carry_count + accumulators.len(), ProgramError);
            let stacked_outputs = iteration_outputs.split_off(self.carry_count);
            carries = iteration_outputs;
            for (accumulator, output) in accumulators.iter_mut().zip(stacked_outputs) {
                *accumulator = write_scan_iteration(accumulator.clone(), iteration, output)?;
            }
        }
        carries.extend(accumulators);
        Ok(carries)
    }
}

// Composite scans interpret their array stacks through the homogeneous array [`EagerContext`], pass reference stacks
// to the body as whole roots (the body selects its own per-iteration view with the explicit index), and take a dynamic
// trip count from the trailing first-class dimension input.
impl<A, O> InterpretableOperation<EagerContext<ArrayIrValue<A>, O>> for ScanOperation<ArrayIrType>
where
    A: Reshape + Slice + UpdateSlice + Value<Type = ArrayType>,
    O: Operation<Type = ArrayIrType>,
    EagerContext<A, ArrayOperation<A>>: Fill<i64, A> + Zero<A>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<A>, O>>>(
        &self,
        context: &EagerContext<ArrayIrValue<A>, O>,
        driver: &D,
        inputs: &[ArrayIrValue<A>],
    ) -> Result<Vec<ArrayIrValue<A>>, ProgramError> {
        validate_scan_length(self.length())?;
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        self.validate_region_count(driver.region_count())?;
        let body = driver.region(0)?;
        let body_input_types = body.input_types();
        let body_input_count = body_input_types.len().checked_sub(1).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input"))
        })?;
        check_count!("input", inputs, body_input_count + usize::from(length.variable().is_some()), ProgramError);
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let (inputs, length) = match length {
            Dimension::Static(length) => (inputs, *length),
            Dimension::Dynamic(_) => {
                let (runtime_length, scan_inputs) =
                    inputs.split_last().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
                let runtime_length = <ArrayIrValue<A> as ValueProjection<DimensionType>>::projected(runtime_length)?;
                // Concrete arrays expose static shapes even when the program declares symbolic axes. Specialize
                // the length to its actual extent before checking those shapes, so eager reads cannot truncate or
                // over-read a stack while valid symbolic programs still execute with their concrete inputs.
                let mut concrete_input_types = input_types.clone();
                *concrete_input_types.last_mut().unwrap() =
                    DimensionValue::constant(runtime_length.extent())?.r#type().into_owned().into();
                validate_scan_runtime_length(length, &concrete_input_types, carry_count, scan_inputs.len())?;
                (scan_inputs, runtime_length.extent())
            }
        };
        // The stacked outputs are allocated at the types that the actual inputs refine the body's declared types to
        // (refer to `refine_output_types`), and their leading axis is the resolved trip count. Only the refinement is
        // computed here: eager reference values may refine their declared reference types, which staged type
        // inference requires to match exactly.
        let expected_input_types =
            composite_scan_boundary_types(ScanBoundarySide::Input, &body_input_types[1..], carry_count, self.length())?;
        // Validate the declared primitive contract before refining it with concrete eager values. References may
        // refine their declared referents eagerly, but that must not bypass the intrinsic index or carry fixed point.
        let mut declared_input_types = expected_input_types.clone();
        if let Some(variable) = self.length().variable() {
            declared_input_types.push(DimensionType::from(variable.clone()).into());
        }
        self.infer_output_types(&declared_input_types, &[body.interface()])?;
        if body_input_types.iter().any(Type::is_reference) || body.contains_reference_accesses_in_closure() {
            // Validate borrowed-root ownership even for zero trips. Resolve the body's abstract output roots against
            // the actual entering handles so aliased inputs remain valid while unrelated roots cannot change slots.
            let analysis = body.reference_analysis_with_configuration(None, true, &[])?;
            for (position, input) in inputs.iter().take(carry_count).enumerate() {
                let ArrayIrValue::Reference(reference) = input else { continue };
                let same_root = match analysis.output_roots()[position] {
                    Some(ReferenceRoot::RegionInput { region, input_index }) if region == body.id() => {
                        matches!(input_index.checked_sub(1).and_then(|index| inputs.get(index)),
                            Some(ArrayIrValue::Reference(output)) if output.id() == reference.id())
                    }
                    _ => false,
                };
                if !same_root {
                    return Err(ProgramError::MalformedProgram(format!(
                        "operation `{SCAN_OPERATION_NAME}` does not return carry {position} as the reference it \
                         entered with, so its `{SCAN_OPERATION_NAME}` state has no fixed point",
                    )));
                }
            }
        }
        let declared_output_types =
            composite_scan_boundary_types(ScanBoundarySide::Output, &body.output_types(), carry_count, self.length())?;
        let stacked_output_types = refine_output_types(
            expected_input_types.as_slice(),
            &input_types[..expected_input_types.len()],
            declared_output_types.as_slice(),
            |index| (index < carry_count).then_some(index),
        )?[carry_count..]
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let (initial_carries, stacks) = inputs.split_at(carry_count);
        let mut carries = initial_carries.to_vec();
        let array_context = EagerContext::<A, ArrayOperation<A>>::new();
        let mut accumulators = stacked_output_types
            .iter()
            .map(|r#type| {
                let dimensions = r#type
                    .shape()
                    .dimensions()
                    .iter()
                    .enumerate()
                    .map(|(axis, dimension)| match dimension {
                        _ if axis == 0 => Ok(Dimension::Static(length)),
                        Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                        Dimension::Dynamic(variable) => inputs
                            .iter()
                            .find_map(|input| match input {
                                ArrayIrValue::Dimension(value) if value.r#type().variable() == variable => {
                                    Some(Dimension::Static(value.extent()))
                                }
                                _ => None,
                            })
                            .ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "cannot eagerly allocate `{}` output `{}` because its dynamic dimension `{}` is \
                                     not supplied as a first-class `{}` input",
                                    SCAN_OPERATION_NAME, r#type, variable, SCAN_OPERATION_NAME,
                                ))
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                array_context.zero(&r#type.clone().with_shape(Shape::new(dimensions)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for visit in 0..length {
            let iteration = if reverse { length - 1 - visit } else { visit };
            let mut iteration_inputs =
                vec![ArrayIrValue::Array(array_context.fill(&ArrayType::scalar(DataType::I64), iteration as i64)?)];
            iteration_inputs.extend(carries.iter().cloned());
            iteration_inputs.extend(
                stacks
                    .iter()
                    .map(|stack| match stack {
                        // The body selects its own view with the explicit index; passing the root preserves
                        // shared storage when more than one body value aliases the same allocation.
                        ArrayIrValue::Reference(_) => Ok(stack.clone()),
                        _ => Ok(ArrayIrValue::Array(read_scan_iteration(
                            <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(stack)?,
                            iteration,
                        )?)),
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?,
            );
            let mut iteration_outputs = driver.interpret_region(context, 0, iteration_inputs)?;
            check_count!("output", iteration_outputs, carry_count + stacked_output_types.len(), ProgramError);
            let iteration_outputs_to_stack = iteration_outputs.split_off(carry_count);
            carries = iteration_outputs;
            for (accumulator, value) in accumulators.iter_mut().zip(iteration_outputs_to_stack) {
                let value = <ArrayIrValue<A> as ValueProjection<ArrayType>>::into_projected(value)?;
                *accumulator = write_scan_iteration(accumulator.clone(), iteration, value)?;
            }
        }
        carries.extend(accumulators.into_iter().map(ArrayIrValue::Array));
        Ok(carries)
    }
}

// Partial evaluation proceeds in three stages:
//
//   1. A carry is *loop-invariant-known* iff the body passes it through unchanged and its initial value is known and
//      resolves to a constant in the known-side context. The body is then partially evaluated once with those carries
//      bound to their initial values, through the driver's region access rather than through
//      [`Program::partially_evaluate`](crate::Program::partially_evaluate) directly, so the rule carries no semantic
//      bounds on the operation family. Invariance is structural rather than decided by comparing values, because value
//      equality cannot distinguish values that the body changes without changing their equality class (e.g., `-0.0`
//      and `0.0`). A known initial value that does not resolve to a constant is never a candidate, because it cannot
//      be embedded into the rebuilt body, and skipping it also keeps the probe from folding symbolic known work into a
//      live staging context.
//   2. Every loop-invariant-known carry is dropped from the residual scan: its uses fold into the rebuilt body as
//      inline constants (as do the other known values that the body closes over, so the residual scan needs no
//      captures), and its final value is its known initial value. Only eager known-side contexts use this probe;
//      staging contexts use the knownness split directly so disposable probes cannot append live outer instructions.
//   3. Remaining *time-varying* known work (known carries that change across iterations and known stacked inputs) is
//      split off by `split_scan_by_knownness` into a *known scan* bound in the known-side context and an *unknown scan*
//      left in the residual program (refer to that function for the full recipe).
//
// Scans whose lengths admit zero and bodies with effects or deferred work skip the invariance probes, which could run
// body work (or its effects) that no iteration performs. Non-reference carries that the body passes through unchanged
// are forwarded from their inputs, and a scan where nothing folds residualizes unchanged.
impl<T, V, O, C> PartiallyEvaluatableOperation<C> for ScanOperation<T>
where
    T: ScanType + TemporalResidualType,
    V: Value<Type = T>,
    C: Context<Type = T, Constant = V, Operation = O>,
    O: Operation<Type = T> + From<ScanOperation<T>> + TemporalResidualOperation<T>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The rule requests all nested-computation work through its region access (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        let regions = driver.regions().collect::<Vec<_>>();
        self.validate_region_count(regions.len())?;
        let body = regions[0];
        let carry_count = self.carry_count;
        let body_input_types = body.input_types();
        T::validate_scan_body_type_signature(&body_input_types, &body.output_types(), carry_count, &self.length)?;
        let runtime_length_count = usize::from(self.length.variable().is_some());
        check_count!("input", inputs, body_input_types.len() - 1 + runtime_length_count, ProgramError);
        let mut input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        if context.parent().is_eager() {
            for (actual, expected) in input_types.iter_mut().zip(&body_input_types[1..]) {
                if actual.is_reference() && expected.is_refined_by(actual) {
                    *actual = expected.clone();
                }
            }
        }
        T::validate_scan_input_type_signature(&body_input_types, carry_count, &self.length, &input_types)?;
        match T::validate_scan_runtime_geometry(&self.length, carry_count, &input_types) {
            Ok(()) => self.infer_output_types(&input_types, &[body.interface()])?,
            // Eager nominal dimensions and concrete stacks can require value-level geometry validation. Retain the
            // original scan so specialization cannot erase that validation by forwarding every output.
            Err(_) => return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs),
        };

        // When every input is known the whole scan folds by binding it in the known-side context; defer to that
        // default behavior.
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        }

        // The trailing runtime length input of a dynamic-length scan is neither a carry nor a stacked input, so it does
        // not make any per-iteration work known.
        let scanned_inputs = &inputs[..body_input_types.len() - 1];
        let residualize = || context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        let split = || {
            split_scan_by_knownness(context, self, body, inputs, |input_known| {
                driver.partition_program(context, body, input_known)
            })
        };
        let mut outputs = 'evaluation: {
            // A scan whose length admits zero may run no iteration. Probing its body or hoisting invariant work could
            // execute (or stage) body work and surface errors for iterations that never run, so it residualizes
            // unchanged instead. Positive static lengths and positive dynamic bounds still permit specialization;
            // known runtime dimensions retain their declared bounds, which conservatively govern this decision too.
            if self.length.bounds().lower() == 0 {
                break 'evaluation residualize()?;
            }

            // The invariance probe below folds the body through the *live* known-side context. For an effectful body
            // the probe would execute (eager) or stage (staging) the body's effects once more, so effectful bodies skip
            // it entirely: the known-ness split's probes run through fresh, discarded contexts and remain safe (see the
            // effect placement contract on `PartialEvaluationContext::fold_or_residualize`). Every reference operation
            // is `OrderedState`, so a body touching references is never pure and the probe can never execute a
            // reference operation or change the active context's effect-ordering state. A body with deferred work
            // skips the probe for the same reason, because the probe would residualize its obligation again.
            if !body.effects().classes().is_empty() || body.effects().has_deferred_work() {
                if scanned_inputs.iter().any(PartialEvaluationValue::is_known) {
                    break 'evaluation split()?;
                }
                break 'evaluation residualize()?;
            }

            // A carry is loop-invariant-known when the body passes it through unchanged and its initial value is known
            // and resolves to a constant in the known-side context, so that it can be embedded into the rebuilt body as
            // a constant. Invariance is structural rather than decided by comparing values, because value equality
            // cannot distinguish values that the body changes without changing their equality class (e.g., a body that
            // negates a carry of `0.0` returns `-0.0`, which compares equal to `0.0`). Skipping symbolic known values
            // also keeps the probe below from folding symbolic known work into a live staging context. Even
            // constant-resolved staging inputs use the knownness split: a disposable live-context probe would append
            // instructions that its rebuilt body cannot embed as constants and would subsequently discard.
            let invariant = (0..carry_count)
                .map(|index| {
                    context.parent().is_eager()
                        && body.output_ids()[index] == body.input_ids()[index + 1]
                        && inputs[index].as_known().is_some_and(|value| context.parent().resolve(value).is_constant())
                })
                .collect::<Vec<bool>>();
            if invariant.iter().all(|folded| !folded) {
                if scanned_inputs.iter().any(PartialEvaluationValue::is_known) {
                    break 'evaluation split()?;
                }
                break 'evaluation residualize()?;
            }
            let mut body_knowledge = vec![PartialValue::Unknown(body_input_types[0].clone())];
            for index in 0..carry_count {
                body_knowledge.push(match (invariant[index], inputs[index].as_known()) {
                    (true, Some(value)) => PartialValue::Known(value.clone()),
                    _ => PartialValue::Unknown(body_input_types[index + 1].clone()),
                });
            }
            body_knowledge.extend(body_input_types[1 + carry_count..].iter().cloned().map(PartialValue::Unknown));
            let body_evaluation = driver.partially_evaluate_program(context, body, &body_knowledge)?;

            // Beyond the invariants, the remaining knowledge may still contain *time-varying* known work: known
            // non-invariant carry inits or known stacked inputs. Those cannot fold once, but they can ride a *known
            // scan* that runs per iteration, so after the invariant rewrite below the known-ness split takes over.
            let time_varying_known = (0..carry_count).any(|index| inputs[index].is_known() && !invariant[index])
                || scanned_inputs[carry_count..].iter().any(PartialEvaluationValue::is_known);

            // The rebuild below embeds the probe's known values as inline body constants, which is only possible when
            // they all resolve to constants. If a context represents a derived known value symbolically, defer to the
            // knownness split when time-varying known work remains and residualize unchanged otherwise.
            if !context.all_knowns_are_constants(&body_evaluation) {
                if time_varying_known {
                    break 'evaluation split()?;
                }
                break 'evaluation residualize()?;
            }

            // A loop-invariant-known carry holds its init value on every iteration and after the last one, so it is
            // dropped from the residual scan: its uses fold into the body as constants and its final value is its
            // known init. The residual body is rebuilt over `[index, kept_carry..., x_slice...]`.
            let mut builder = ProgramBuilder::<V, O>::new();
            let body_input_atoms = body_input_types
                .iter()
                .enumerate()
                .map(|(position, input_type)| {
                    let folded = (1..=carry_count).contains(&position) && invariant[position - 1];
                    (!folded).then(|| builder.add_input(input_type.clone()))
                })
                .collect::<Vec<_>>();

            // Feed the residual body program's inputs in its own input order. A surviving unknown body input is a
            // non-invariant carry or a scanned element and maps to the matching body input atom; a known residual (a
            // folded invariant carry value or another value the body closed over) is rebuilt as an inline constant by
            // recovering its staged payload through the known-side context.
            let mut residual_body_inputs = Vec::with_capacity(body_evaluation.inputs.len());
            for residual_input in body_evaluation.inputs.iter() {
                match residual_input {
                    PartialEvaluationInput::Unknown(body_input) => {
                        residual_body_inputs.push(body_input_atoms[*body_input].ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{SCAN_OPERATION_NAME}` partial evaluation folded carry {} but its residual body \
                                 still reads it",
                                body_input - 1,
                            ))
                        })?)
                    }
                    PartialEvaluationInput::Known(value) => {
                        residual_body_inputs.push(builder.add_constant(context.known_constant(value)?))
                    }
                }
            }
            let spliced_outputs = builder.splice_program(&body_evaluation.program, &residual_body_inputs)?;

            // Assemble the residual body outputs as `[kept_next_carry..., y_slice...]`: a folded output becomes an
            // inline constant, and an unknown output reads the spliced residual program's corresponding output.
            let body_output_atoms = (0..body.output_types().len())
                .filter(|&output_index| output_index >= carry_count || !invariant[output_index])
                .map(|output_index| match &body_evaluation.outputs[output_index] {
                    PartialEvaluationOutput::Known(value) => Ok(builder.add_constant(context.known_constant(value)?)),
                    PartialEvaluationOutput::Unknown(index) => Ok(spliced_outputs[*index]),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let residual_input_count = body_input_atoms.iter().flatten().count();
            let residual_output_count = body_output_atoms.len();
            let residual_body = builder.build::<Vec<V>, Vec<V>>(
                body_output_atoms,
                vec![Placeholder; residual_input_count],
                vec![Placeholder; residual_output_count],
            )?;
            let kept_carry_count = invariant.iter().filter(|&&folded| !folded).count();
            let scan = ScanOperation::<T>::new(kept_carry_count, self.length.clone())
                .with_reverse(self.reverse)
                .with_unroll(self.unroll)?;
            let scan_inputs = inputs
                .iter()
                .enumerate()
                .filter(|(index, _)| *index >= carry_count || !invariant[*index])
                .map(|(_, input)| input.clone())
                .collect::<Vec<_>>();

            // With time-varying known work remaining, the known-ness split of the invariant-folded scan finishes the
            // job. Otherwise, the residual scan runs over the original inputs without the folded carries, unless no
            // output survives, in which case the pure residual scan would be dead.
            let mut residual_outputs = if time_varying_known {
                split_scan_by_knownness(
                    context,
                    &scan,
                    residual_body.entry_region_ref(),
                    &scan_inputs,
                    |input_known| driver.partition_program(context, residual_body.entry_region_ref(), input_known),
                )?
            } else if residual_output_count == 0 {
                Vec::new()
            } else {
                context.fold_or_residualize(O::from(scan), vec![residual_body], &scan_inputs)?
            }
            .into_iter();
            (0..body.output_types().len())
                .map(|index| {
                    if index < carry_count && invariant[index] {
                        inputs[index].clone()
                    } else {
                        residual_outputs.next().unwrap()
                    }
                })
                .collect()
        };

        // Forwarding keeps a known initial value of a passed-through carry known.
        forward_scan_carries(body, carry_count, inputs, &mut outputs);
        Ok(outputs)
    }
}

// Under a *staging* parent, a scan is batched *structurally*, staging one batched scan into the enclosing trace (the
// shape of JAX's `_scan_batching_rule`), so that the size of the batched program stays independent of the trip count:
//
//   1. Every batched initial carry is realigned to batch axis 0, and every stacked input whose batch axis would
//      displace the leading scan dimension is realigned to batch axis 1, so per-iteration slices keep their batch
//      placement when the leading scan dimension is dropped.
//   2. The body is batched at `[replicated_index, carry_axes..., slice_axes...]`. Its carry axes reach a fixed point:
//      a scan's carry types are loop-invariant, so a replicated carry whose next value is batched *becomes* batched,
//      and the rule widens that carry's input axis and batches the body again until the body is axis-invariant (JAX's
//      `carry_bat` fixed point, which converges after at most `carry_count + 1` passes because every pass that does not
//      converge widens at least one carry). The body's outputs are then instantiated at the joined axes
//      ([`ProgramBatchingOutputAxesPolicy::AlignEachTo`], i.e., JAX's `instantiate=carry_bat`), reusing the program
//      of the stabilizing pass when its natural axes already are those joined axes.
//   3. Widened initial carries gain their batch axis through staged broadcasts, and one [`ScanOperation`] over the
//      batched body is bound into the parent with the same carry count, length, `reverse`, and (lowering-only) `unroll`
//      factor. Final carries come back at the carry axes, and stacked outputs at their per-iteration axes shifted right
//      by the new leading scan dimension. The staged stacked outputs carry the scan's inferred output types, which
//      never inherit optional sharding metadata from the inputs, so sharding propagation resolves it (refer to the
//      documentation of [`ScanType::infer_scan_output_types`]).
//
// Under an *eager* parent, the scan loop is instead replayed per iteration through `batch_scan_with_interpreter`, with
// each body instruction re-entering the batching rules of the operation family against the same active context. Its
// packed stacked accumulators retain per-item placement metadata exactly, and constants lift and stacked-output
// accumulators are seeded (via the parent's [`Zero`]) through `context.parent()`. In both cases, the outputs of
// non-reference carries that the body passes through unchanged are forwarded from their inputs.
impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for ScanOperation<ArrayType>
where
    C: Context<Type = ArrayType> + Zero<<C as Domain>::Value> + Fill<i64, <C as Domain>::Value>,
    <C as Domain>::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    C::Operation: OperationProvider<ArrayType, ZeroOperation<ArrayType>, Operation = C::Operation>
        + From<BroadcastOperation>
        + From<TransposeOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshapeOperation>
        + From<ScanOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Structural body batching takes mapped axes without ragged extents, and eager slicing rebuilds dense
        // carriers. Neither path may treat a ragged storage bound as the logical per-item extent.
        ArrayBatch::reject_ragged_inputs(self, inputs)?;
        self.validate_region_count(driver.region_count())?;
        let body = driver.region(0)?;
        let body_input_count = body.input_types().len().checked_sub(1).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input"))
        })?;
        check_count!("input", inputs, body_input_count, ProgramError);
        let input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
        // A rewritten batched scan must not legitimize a malformed original index, carry boundary, or stack shape.
        self.infer_output_types(&input_types, &[body.interface()])?;
        let carry_count = self.carry_count();
        if !context.parent().is_eager() {
            let body_output_count = body.output_types().len();

            // Realign batched carries to batch axis 0 and batched stacks off the leading scan dimension, so the
            // fixed point below only ever distinguishes replicated from batched-at-0 carries and every
            // per-iteration slice keeps its batch placement when the leading scan dimension is dropped.
            let mut carries =
                inputs[..carry_count].iter().map(|input| input.move_axis(0)).collect::<Result<Vec<_>, _>>()?;
            let stacks = inputs[carry_count..]
                .iter()
                .map(
                    |input| if input.batch_axis_position() == Some(0) { input.move_axis(1) } else { Ok(input.clone()) },
                )
                .collect::<Result<Vec<_>, _>>()?;
            let mut carry_axes = carries.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();
            let slice_axes =
                stacks.iter().map(|stack| scan_iteration_batch_axis(stack.batch_axis())).collect::<Vec<_>>();

            // Iterate the carry batch axes to a fixed point. Each pass discovers the body's natural output axes and
            // every pass that does not converge widens at least one replicated carry, so at most `carry_count + 1`
            // passes run. The pass that widens nothing determines the stacked outputs' per-iteration axes.
            let (stabilized_body, output_slice_axes) = loop {
                let mut iteration_axes = vec![BatchAxis::replicated()];
                iteration_axes.extend(carry_axes.iter().copied());
                iteration_axes.extend(slice_axes.iter().copied());
                let candidate = driver.batch_program(
                    context,
                    body,
                    iteration_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?;
                check_count!("output", candidate.output_axes(), body_output_count, ProgramError);
                let mut widened = false;
                for (carry_axis, output_axis) in carry_axes.iter_mut().zip(candidate.output_axes()) {
                    if carry_axis.is_replicated() && !output_axis.is_replicated() {
                        *carry_axis = BatchAxis::new(0);
                        widened = true;
                    }
                }
                if !widened {
                    let output_slice_axes = candidate.output_axes()[carry_count..].to_vec();
                    break (candidate, output_slice_axes);
                }
            };

            // Instantiate the body's outputs at the joined axes so its next-carry outputs align with its carry
            // inputs across iterations. The stabilizing pass already used these input axes, so when its discovered
            // (normalized) output axes equal the joined targets it *is* the aligned program and is kept as-is.
            let mut iteration_axes = vec![BatchAxis::replicated()];
            iteration_axes.extend(carry_axes.iter().copied());
            iteration_axes.extend(slice_axes.iter().copied());
            let mut target_axes = carry_axes.clone();
            target_axes.extend(output_slice_axes.iter().copied());
            let batched_body = context.align_batched_program_outputs(
                driver,
                body,
                iteration_axes.as_slice(),
                stabilized_body,
                target_axes.as_slice(),
            )?;

            // Widen the parent carry inits whose elements became batched (their batch axis is materialized through
            // a staged broadcast) and stage one batched scan over the batched body.
            for (carry, carry_axis) in carries.iter_mut().zip(carry_axes.iter()) {
                if !carry_axis.is_replicated() && carry.batch_axis().is_replicated() {
                    *carry = carry.broadcast(0, P::axis_size(context)?, context.axis_sharding().clone())?;
                }
            }
            let batched_scan = ScanOperation::<ArrayType>::new(carry_count, self.length())
                .with_reverse(self.reverse())
                .with_unroll(self.unroll())?;
            let mut values = carries.iter().map(|carry| carry.value().clone()).collect::<Vec<_>>();
            values.extend(stacks.iter().map(|stack| stack.value().clone()));
            let outputs = context.parent().bind(batched_scan, vec![batched_body], &values)?;
            check_count!("output", outputs, carry_count + output_slice_axes.len(), ProgramError);

            // Final carries come back at the carry axes; each stacked output gains the leading scan dimension,
            // shifting its per-iteration batch axis right by one.
            let mut output_axes = carry_axes;
            output_axes.extend(output_slice_axes.iter().map(|axis| match axis.axis() {
                Some(axis) => BatchAxis::new(axis.value() + 1),
                None => BatchAxis::replicated(),
            }));
            let mut outputs = outputs
                .into_iter()
                .zip(output_axes)
                .map(|(output, axis)| ArrayBatch::new(output, axis))
                .collect::<Result<Vec<_>, _>>()?;
            forward_scan_carries(body, carry_count, inputs, &mut outputs);
            return Ok(outputs.into());
        }

        if self.length().value() == Some(0) {
            // No iteration executes, but batching the body structurally still determines which per-iteration outputs
            // are mapped and where their packed batch dimensions live. Stacked inputs lose their per-item leading
            // scan dimension before entering the body, so their batch axes must be adjusted in the same way as an
            // actual iteration slice.
            let mut iteration_input_axes = vec![BatchAxis::replicated()];
            iteration_input_axes.extend(inputs[..self.carry_count()].iter().map(ArrayBatch::batch_axis));
            iteration_input_axes
                .extend(inputs[self.carry_count()..].iter().map(|input| scan_iteration_batch_axis(input.batch_axis())));
            let (batched_body, output_axes) = driver
                .batch_program(
                    context,
                    body,
                    iteration_input_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?
                .into_parts();
            let output_types = batched_body.output_types();
            check_count!("output", output_axes, output_types.len(), ProgramError);
            if output_types.len() < self.carry_count() {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` body has {} outputs but carry count is {}",
                    output_types.len(),
                    self.carry_count(),
                ))
                .into());
            }

            // A zero-length scan returns its initial carries unchanged. Its stacked outputs are empty arrays whose
            // packed element types and batch axes come from the structurally batched body. Inserting the leading
            // scan dimension shifts every mapped output axis right by one while preserving its placement metadata.
            let mut outputs = inputs[..self.carry_count()].to_vec();
            for (output_type, output_axis) in
                output_types.into_iter().zip(output_axes.into_iter()).skip(self.carry_count())
            {
                let stacked_type = output_type.with_inserted_dimension(0, Dimension::Static(0))?;
                let stacked_axis = match output_axis.axis() {
                    Some(axis) => BatchAxis::new(axis.value() + 1),
                    None => BatchAxis::replicated(),
                };
                let stacked_value = context.parent().zero(&stacked_type)?;
                outputs.push(ArrayBatch::new(stacked_value, stacked_axis)?);
            }
            return Ok(outputs.into());
        }

        let stacked_output_count = body.output_types().len() - self.carry_count();
        let length = self.length().value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: format!(
                "eager homogeneous `{SCAN_OPERATION_NAME}` batching requires a concrete trip count but got `{}`",
                self.length(),
            ),
        })?;
        let mut outputs = batch_scan_with_interpreter(
            self.carry_count(),
            length,
            self.reverse(),
            stacked_output_count,
            inputs,
            |stacked_type| context.parent().zero(stacked_type),
            |batch, axis| driver.align_batch_axis(context, batch, axis),
            |iteration, mut iteration_inputs| {
                let index = context.parent().fill(&ArrayType::scalar(DataType::I64), iteration as i64)?;
                iteration_inputs.insert(0, ArrayBatch::replicated(index));
                driver.batch_region(context, 0, iteration_inputs)
            },
        )?;
        forward_scan_carries(body, self.carry_count(), inputs, &mut outputs);
        Ok(outputs.into())
    }
}

// The rule carries the mapped extent as leading replicated state in the transformed scan. Array carries use the
// same monotonic mapped-axis fixed point as homogeneous scans, while first-class dimension carries remain
// replicated. Stacked inputs are arrays or references and stacked outputs are arrays, never first-class dimensions,
// because one shared dimension value cannot represent a different stacked extent for each batch item. A reference
// stack keeps the batch axis fixed by its referent (which must lie behind the leading scan axis). The body receives
// the whole packed root, and each access adjusts its folded transforms through `batch_reference_transforms`.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for ScanOperation<ArrayIrType>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<ScanOperation<ArrayIrType>>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // Structural body batching receives mapped axes without per-item ragged extent carriers, so reject those
        // inputs before body transformation or parent binding can lose their logical geometry.
        ArrayIrBatch::reject_ragged_inputs(self, inputs)?;
        self.validate_region_count(driver.region_count())?;
        let body = driver.region(0)?;
        let body_input_types = body.input_types();
        let body_input_count = body_input_types.len().checked_sub(1).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input"))
        })?;
        let runtime_length_count = usize::from(self.length().variable().is_some());
        check_count!("input", inputs, body_input_count + runtime_length_count, ProgramError);
        ArrayIrType::validate_scan_body_type_signature(
            &body_input_types,
            &body.output_types(),
            self.carry_count(),
            self.length(),
        )?;
        let mut input_types = inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>();
        if context.parent().is_eager() {
            // Eager reference handles may refine their declared referents. Validate their declared contract after
            // checking this refinement, while ordinary values retain their actual logical types for validation.
            let expected_input_types = composite_scan_boundary_types(
                ScanBoundarySide::Input,
                &body_input_types[1..],
                self.carry_count(),
                self.length(),
            )?;
            for (actual, expected) in input_types.iter_mut().zip(expected_input_types) {
                if actual.is_reference() && expected.is_refined_by(actual) {
                    *actual = expected;
                }
            }
        }
        ArrayIrType::validate_scan_input_type_signature(
            &body_input_types,
            self.carry_count(),
            self.length(),
            &input_types,
        )?;
        if context.parent().is_eager() && self.length().variable().is_some() {
            let runtime_length = inputs.last().unwrap();
            runtime_length.validate_replicated_dimension()?;
            if let Some(runtime_length) = context.parent().resolve(runtime_length.value()).into_constant() {
                let runtime_length = <C::Constant as ValueProjection<DimensionType>>::into_projected(runtime_length)?;
                // Eager packed arrays have concrete shapes even when their shared runtime length retains a nominal
                // identity. Validate those shapes against its concrete value before body batching can read a stack.
                *input_types.last_mut().unwrap() = DimensionValue::constant(runtime_length.extent())
                    .map_err(TypeError::from)?
                    .r#type()
                    .into_owned()
                    .into();
            }
        }
        self.infer_output_types(&input_types, &[body.interface()])?;
        let (scan_inputs, runtime_length) = if self.length().variable().is_some() {
            let Some((runtime_length, scan_inputs)) = inputs.split_last() else {
                return Err(ProgramError::InvalidInputCount { expected: body.input_types().len(), actual: 0 }.into());
            };
            runtime_length.validate_replicated_dimension()?;
            (scan_inputs, Some(runtime_length))
        } else {
            (inputs, None)
        };
        check_count!("input", scan_inputs, body.input_types().len() - 1, ProgramError);
        let carry_count = self.carry_count();

        // Canonicalize mapped array carries to the leading axis. Dimension carries remain replicated, and a reference
        // carry keeps the batch axis fixed by its referent, since shared storage cannot be moved.
        let mut carries = scan_inputs[..carry_count]
            .iter()
            .cloned()
            .map(|input| match input.unbatched_type() {
                ArrayIrType::Array(_) if !input.batch_axis().is_replicated() => {
                    driver.align_batch_axis(context, input, Axis::from(0))
                }
                ArrayIrType::Array(_) | ArrayIrType::Reference(_) => Ok(input),
                ArrayIrType::Dimension(_) => {
                    input.validate_replicated_dimension()?;
                    Ok(input)
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        // Mapped array stacks move their batch axis behind the leading scan axis. Reference roots retain
        // their batch axis because storage cannot be moved; each folded access handles indexing inside
        // the body instead of requiring a special view at the region boundary.
        let (stacks, slice_axes): (Vec<_>, Vec<_>) = scan_inputs[carry_count..]
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, input)| -> Result<_, BatchingError> {
                if matches!(input.unbatched_type(), ArrayIrType::Reference(_)) {
                    // The scan contract still requires the root's leading axis to be its scan axis. Shared
                    // reference storage cannot be moved to restore that axis when batching places another first.
                    if input.batch_axis_position() == Some(0) {
                        return Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "`{SCAN_OPERATION_NAME}` batching found the reference-typed stacked input at position \
                                 {} batched on its scan axis; a reference stack keeps its batch axis and must be \
                                 batched at an axis behind its leading scan axis",
                                carry_count + position,
                            ),
                        });
                    }
                    // Reference roots enter the body unchanged. The explicit indexing operation inside the
                    // body adjusts its selected axis under batching, so the boundary preserves this batch axis.
                    let axis = input.batch_axis();
                    return Ok((input, axis));
                }
                <&ArrayType>::try_from(&input.unbatched_type())?;
                let stack = if input.batch_axis_position() == Some(0) {
                    driver.align_batch_axis(context, input, Axis::from(1))?
                } else {
                    input
                };
                let slice_axis = scan_iteration_batch_axis(stack.batch_axis());
                Ok((stack, slice_axis))
            })
            .collect::<Result<_, _>>()?;
        let mut carry_axes = carries.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();

        // Iterate carry axes to a fixed point. A first-class dimension cannot widen because composite batching does
        // not admit mapped dimension values, and a reference carry is never widened: its axis is fixed by the input, so
        // the body must return it exactly as it entered.
        let (stabilized_body, output_slice_axes) = loop {
            let iteration_axes = std::iter::once(BatchAxis::replicated())
                .chain(carry_axes.iter().chain(slice_axes.iter()).copied())
                .collect::<Vec<_>>();
            let candidate = driver.batch_program(
                context,
                body,
                iteration_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", candidate.output_axes(), body.output_types().len(), ProgramError);
            let mut widened = false;
            for (index, (carry_axis, output_axis)) in
                carry_axes.iter_mut().zip(candidate.output_axes().iter()).enumerate()
            {
                let widens = carry_axis.is_replicated() && !output_axis.is_replicated();
                match scan_inputs[index].unbatched_type() {
                    ArrayIrType::Reference(_) => {
                        validate_reference_carry_axis(SCAN_OPERATION_NAME, index, *carry_axis, *output_axis)?;
                    }
                    ArrayIrType::Dimension(r#type) if widens => {
                        return Err(BatchingError::MappedDimension { r#type: Box::new(r#type), axis: *output_axis });
                    }
                    ArrayIrType::Array(_) if widens => {
                        *carry_axis = BatchAxis::new(0);
                        widened = true;
                    }
                    _ => {}
                }
            }
            if !widened {
                let output_slice_axes = candidate.output_axes()[carry_count..].to_vec();
                break (candidate, output_slice_axes);
            }
        };

        // The stabilizing pass already used these input axes, so when its discovered (normalized) output axes equal
        // the joined targets it *is* the aligned program and is kept as-is instead of being rebuilt.
        let iteration_axes = std::iter::once(BatchAxis::replicated())
            .chain(carry_axes.iter().chain(slice_axes.iter()).copied())
            .collect::<Vec<_>>();
        let target_axes = carry_axes.iter().chain(output_slice_axes.iter()).copied().collect::<Vec<_>>();
        let batched_body = context.align_batched_program_outputs(
            driver,
            body,
            iteration_axes.as_slice(),
            stabilized_body,
            target_axes.as_slice(),
        )?;
        for (carry, axis) in carries.iter_mut().zip(carry_axes.iter()) {
            if !axis.is_replicated() && carry.batch_axis().is_replicated() {
                *carry = driver.align_batch_axis(context, carry.clone(), Axis::from(0))?;
            }
        }

        // Composite batching prepends its extent parameter. The scan's intrinsic index must stay first;
        // the extent becomes an ordinary replicated carry immediately after it.
        let mut input_order = (0..batched_body.input_ids().len()).collect::<Vec<_>>();
        input_order.swap(0, 1);
        let output_order = (0..batched_body.output_ids().len()).collect::<Vec<_>>();
        let batched_body = reorder_program_boundary(&batched_body, &input_order, &output_order)?;

        let batched_scan = ScanOperation::<ArrayIrType>::new(carry_count + 1, self.length())
            .with_reverse(self.reverse())
            .with_unroll(self.unroll())?;
        let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
        packed_inputs.push(context.axis_extent().clone());
        packed_inputs.extend(carries.iter().map(|carry| carry.value().clone()));
        packed_inputs.extend(stacks.iter().map(|stack| stack.value().clone()));
        packed_inputs.extend(runtime_length.map(|runtime_length| runtime_length.value().clone()));
        let mut outputs = context.parent().bind(batched_scan, vec![batched_body], packed_inputs.as_slice())?;
        check_count!("output", outputs, 1 + carry_count + output_slice_axes.len(), ProgramError);
        outputs.remove(0);
        let mut output_axes = carry_axes;
        output_axes.extend(output_slice_axes.iter().map(|axis| match axis.axis() {
            Some(axis) => BatchAxis::new(axis.value() + 1),
            None => BatchAxis::replicated(),
        }));
        let mut outputs = outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayIrBatch::new(output, axis))
            .collect::<Result<Vec<_>, _>>()?;
        forward_scan_carries(body, carry_count, inputs, &mut outputs);
        Ok(outputs.into())
    }
}

// Shared-destination JVP stages one fused scan without residual stacks. Separated destinations linearize the body and
// reuse the partition reconstruction of partial evaluation, which stacks varying residuals and hoists invariant ones.
impl<T, C> DifferentiableOperation<C> for ScanOperation<T>
where
    T: DifferentiableType + ScanType + TemporalResidualType,
    C: Context<Type = T> + Zero<C::Value>,
    C::Operation:
        ResidualZeroProvider<T, Operation = C::Operation> + From<ScanOperation<T>> + TemporalResidualOperation<T>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();

        // The fused body is compact: it carries a tangent input exactly for the body inputs whose tangent is live and a
        // tangent output exactly for the body outputs whose tangent is live. Plumbing references, zero-space inputs,
        // and structural-zero tangents are not live. A carry whose initial tangent is a structural zero can still
        // acquire a live tangent from the other tangents on a later iteration, in which case it needs a tangent slot
        // from the first iteration on, so carry liveness is a monotone fixed point over the body (bounded by the carry
        // count, because every round that does not converge enlivens at least one carry). A tangent output of the
        // derived body that depends on no tangent input is a structural zero: such carry outputs are not live after
        // convergence, and such scanned outputs leave the scan as symbolic zeros instead of materialized zero stacks.
        // A reference carry's tangent can only come from its input, so it never becomes live through iteration.
        let regions = driver.regions().collect::<Vec<_>>();
        self.validate_region_count(regions.len())?;
        let body = regions[0];
        let body_input_types = body.input_types();
        let body_output_types = body.output_types();
        T::validate_scan_body_type_signature(&body_input_types, &body_output_types, carry_count, &length)?;
        let body_input_count = body_input_types.len();
        let body_output_count = body_output_types.len();
        let runtime_length_count = usize::from(length.variable().is_some());
        check_count!("input", inputs, body_input_count - 1 + runtime_length_count, ProgramError);
        let mut input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
        if context.primal().is_eager() {
            for (actual, expected) in input_types.iter_mut().zip(&body_input_types[1..]) {
                if actual.is_reference() && expected.is_refined_by(actual) {
                    *actual = expected.clone();
                }
            }
        }
        T::validate_scan_input_type_signature(&body_input_types, carry_count, &length, &input_types)?;
        let preserve_known_scan = match T::validate_scan_runtime_geometry(&length, carry_count, &input_types) {
            Ok(()) => {
                self.infer_output_types(&input_types, &[body.interface()])?;
                false
            }
            Err(_) if context.primal().is_eager() => true,
            Err(error) => return Err(error.into()),
        };
        let (body_inputs, runtime_length_inputs) = inputs.split_at(body_input_count - 1);
        let shared_destinations = std::ptr::eq(context.primal(), context.tangent());
        let mut input_has_tangent = std::iter::once(false)
            .chain(body_inputs.iter().map(|input| input.is_tangent_active() && !input.tangent().is_zero()))
            .collect::<Vec<_>>();
        let (input_indices, jvp_body, output_has_tangent) = loop {
            let input_indices = input_has_tangent
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| active.then_some(index))
                .collect::<Vec<_>>();
            let output_has_slot = body.tangent_output_mask(&input_indices)?;
            let jvp_body = driver.jvp_program(body, &input_indices)?;
            let tangent_inputs = (body_input_count..jvp_body.input_count()).collect::<Vec<_>>();
            let mut output_depends = jvp_body.output_dependence(&tangent_inputs)?.into_iter().skip(body_output_count);
            let output_has_tangent = output_has_slot
                .iter()
                .map(|&has_slot| has_slot && output_depends.next().unwrap())
                .collect::<Vec<_>>();
            let mut converged = true;
            for index in 0..carry_count {
                if output_has_tangent[index] && !input_has_tangent[index + 1] {
                    input_has_tangent[index + 1] = true;
                    converged = false;
                }
            }
            if converged {
                // Carry tangent slots must pair across the boundary, so a live carry keeps its tangent output even when
                // the body happens to reset it to a value that does not depend on any tangent.
                let output_has_tangent = (0..body_output_count)
                    .map(
                        |index| {
                            if index < carry_count { input_has_tangent[index + 1] } else { output_has_tangent[index] }
                        },
                    )
                    .collect::<Vec<_>>();
                let kept_outputs = (0..body_output_count)
                    .chain(
                        (0..body_output_count)
                            .filter(|&index| output_has_slot[index])
                            .enumerate()
                            .filter(|(_, index)| output_has_tangent[*index])
                            .map(|(slot, _)| body_output_count + slot),
                    )
                    .collect::<Vec<_>>();
                break (input_indices, jvp_body.with_outputs(&kept_outputs)?, output_has_tangent);
            }
        };
        let live_carry_count = input_has_tangent[1..1 + carry_count].iter().filter(|&&live| live).count();
        let live_input_count = input_indices.len();
        let live_output_count = output_has_tangent.iter().filter(|&&live| live).count();
        let input_order = live_scan_signature_permutation(&input_has_tangent, carry_count + 1)?;
        let output_order = live_scan_signature_permutation(&output_has_tangent, carry_count)?;
        let mut fused_inputs = body_inputs.iter().map(|input| (input.primal().clone(), true)).collect::<Vec<_>>();
        for (input, &active) in body_inputs.iter().zip(&input_has_tangent[1..]) {
            if active {
                let primal = context.primal_to_tangent(input.primal().clone())?;
                let tangent = C::Operation::materialize_zero_from_residual_sources(
                    context.tangent(),
                    input.tangent().clone(),
                    std::iter::once(&primal),
                )?;
                fused_inputs.push((tangent, false));
            }
        }
        let mut scan_inputs =
            input_order.iter().skip(1).map(|&index| fused_inputs[index - 1].clone()).collect::<Vec<_>>();
        scan_inputs.extend(runtime_length_inputs.iter().map(|input| (input.primal().clone(), true)));
        let fused_scan = ScanOperation::<T>::new(carry_count + live_carry_count, length)
            .with_reverse(reverse)
            .with_unroll(unroll)?;
        let outputs = if shared_destinations {
            let fused_body = reorder_program_boundary(&jvp_body, &input_order, &output_order)?;
            let scan_input_values = scan_inputs.iter().map(|(value, _)| value.clone()).collect::<Vec<_>>();
            context
                .primal()
                .bind(C::Operation::from(fused_scan), vec![fused_body], &scan_input_values)?
                .into_iter()
                .map(|value| (value, false))
                .collect::<Vec<_>>()
        } else {
            let (primal_program, tangent_program, residual_count) =
                driver.linearize_program(body, &input_indices)?.into_parts();
            // The linearized tangent program returns one tangent per output tangent slot, so the structural zeros that
            // the fixed point identified are projected out exactly as for the fused body.
            let tangent_slots = body.tangent_output_mask(&input_indices)?;
            let kept_tangent_outputs = (0..body_output_count)
                .filter(|&index| tangent_slots[index])
                .enumerate()
                .filter(|(_, index)| output_has_tangent[*index])
                .map(|(slot, _)| slot)
                .collect::<Vec<_>>();
            let tangent_program = tangent_program.with_outputs(&kept_tangent_outputs)?;
            let mut fused_input_types = body.input_types();
            fused_input_types.extend(tangent_program.input_types().into_iter().take(live_input_count));
            let reordered_input_types =
                input_order.iter().map(|&index| fused_input_types[index].clone()).collect::<Vec<_>>();
            let input_known = input_order.iter().map(|&index| index < body_input_count).collect::<Vec<_>>();
            let known_input_indices =
                input_known.iter().enumerate().filter_map(|(index, &known)| known.then_some(index)).collect();
            let residual_inputs = (0..live_input_count)
                .map(|index| {
                    PartialEvaluationInput::Unknown(
                        input_order.iter().position(|&source| source == body_input_count + index).unwrap(),
                    )
                })
                .chain((0..residual_count).map(PartialEvaluationInput::Known))
                .collect();
            let partition_outputs = output_order
                .iter()
                .map(|&index| {
                    if index < body_output_count {
                        PartialEvaluationOutput::Known(index)
                    } else {
                        PartialEvaluationOutput::Unknown(index - body_output_count)
                    }
                })
                .collect();
            let partition = PartitionedProgram::from_parts(
                Arc::unwrap_or_clone(primal_program),
                tangent_program,
                known_input_indices,
                residual_inputs,
                partition_outputs,
            );
            reconstruct_partitioned_scan(
                &fused_scan,
                &reordered_input_types,
                body_output_count + live_output_count,
                &scan_inputs,
                &input_known,
                &input_known[1..1 + carry_count + live_carry_count],
                preserve_known_scan,
                partition,
                |constant| Ok((context.primal().lift(constant)?, true)),
                |operation, programs, inputs| {
                    let inputs = inputs.iter().map(|(value, _)| value.clone()).collect::<Vec<_>>();
                    Ok(context
                        .primal()
                        .bind(operation, programs, &inputs)?
                        .into_iter()
                        .map(|value| (value, true))
                        .collect())
                },
                |operation, programs, inputs| {
                    let inputs = inputs
                        .iter()
                        .map(|(value, known)| {
                            if *known {
                                context.primal_to_tangent(value.clone()).map_err(ProgramError::from)
                            } else {
                                Ok(value.clone())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(context
                        .tangent()
                        .bind(operation, programs, &inputs)?
                        .into_iter()
                        .map(|value| (value, false))
                        .collect())
                },
            )?
            .ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCAN_OPERATION_NAME}` linearization cannot store the residual boundary required by its body, \
                     because a reference would cross from its known iterations to its tangent iterations; pass the \
                     reference as a differentiated input or discharge the references first",
                ),
            })?
        };
        let mut original_outputs = vec![None; body_output_count + live_output_count];
        for (&index, (value, _)) in output_order.iter().zip(outputs) {
            original_outputs[index] = Some(value);
        }
        let mut tangent_outputs = original_outputs.split_off(body_output_count).into_iter();
        let mut outputs = original_outputs
            .into_iter()
            .zip(output_has_tangent)
            .map(|(primal, active)| {
                let primal = primal.unwrap();
                if active {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap().unwrap())
                } else {
                    DifferentiationDual::new_with_zero_tangent(primal)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        forward_scan_carries(body, carry_count, inputs, &mut outputs);
        Ok(outputs)
    }
}

// The array universe has no reference types, so no input carries a cotangent reference and every cotangent is
// returned as a value.
impl<V, O> TransposableOperation<V, O> for ScanOperation<ArrayType>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + ResidualZeroProvider<ArrayType, Operation = O>
        + From<AddOperation<ArrayType>>
        + From<ScanOperation<ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents = CotangentDestinations::without_references(std::iter::repeat_n(true, inputs.len()));
        let contributions = transpose_primal_scan(self, context, driver, inputs, outputs, &cotangents)?;
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

// Reference carries and stacks accumulate through the enclosing context's cotangent references, which are resolved
// (and allocated on first use when their state cotangent is live) before the primal rule threads them positionally
// through the reversed scan. Ordinary carry cotangents drive earlier iterations and scanned cotangents have element
// geometry inside the body, so they are all returned as values before the accumulators apply demand or gradient
// buffers.
impl<V, O> TransposableOperation<V, O> for ScanOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>
        + ResidualZeroProvider<ArrayIrType, Operation = O>
        + From<AddOperation<ArrayIrType>>
        + From<ScanOperation<ArrayIrType>>
        + From<ReferenceNewOperation<ArrayType, ArrayIrType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents = context.cotangent_destinations(driver, inputs, &[])?;
        let contributions = transpose_primal_scan(self, context, driver, inputs, outputs, &cotangents)?;
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

/// Type-family semantics for [`ScanOperation`].
///
/// [`ArrayType`] stacks per-iteration arrays along a static leading axis. [`ArrayIrType`] additionally permits
/// references and first-class dimensions in carry positions and a dynamic length supplied by a trailing dimension
/// input. Both families require the body's first input to be a scalar [`I64`](DataType::I64) slice index.
///
/// Trailing array inputs enter as slices. A reference to a statically shaped array stack enters unchanged, and the body
/// creates any per-iteration view explicitly. Stacked outputs must be arrays: neither references nor first-class
/// dimensions have a stacked value representation. Scan lengths are [`Dimension`]s, so every implementing family uses
/// [`DimensionVariable`]s as its type identities. A dynamic length and every stacked leading axis must provably have
/// equal extents; a static stack whose extent only lies within the length's bounds is insufficient.
pub trait ScanType: Type<Identity = DimensionVariable> {
    /// Validates the intrinsic index and carry contracts of a declared iteration body independently of the values
    /// that bind it. Transform rules use this before rewriting or eliminating a scan so specialization cannot erase
    /// an invalid source signature. Homogeneous bodies require static array types; composite bodies additionally
    /// validate reference and dimension boundary roles.
    ///
    /// # Parameters
    ///
    ///   - `body_input_types`: Declared body inputs, including the intrinsic scalar `i64` index.
    ///   - `body_output_types`: Declared body outputs, with carry outputs before stacked-output slices.
    ///   - `carry_count`: Number of matching carry positions on both boundaries.
    ///   - `length`: Declared trip-count geometry, validated independently of a runtime value.
    ///
    /// # Errors
    ///
    /// Returns an error when the index, carry counts or types, length, or family-specific boundary roles are invalid.
    fn validate_scan_body_type_signature(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError>;

    /// Validates the operation inputs against the declared iteration inputs, including the type and identity of a
    /// trailing runtime length. This checks ordinary type refinement independently of whether the runtime length and
    /// the stacked axes are provably equal; use [`Self::validate_scan_runtime_geometry`] for that additional contract.
    /// Callers must first validate the body signature. Eager callers may replace a reference input with its declared
    /// reference type after checking that the actual referent refines it, as eager handles preserve runtime ownership.
    ///
    /// # Parameters
    ///
    ///   - `body_input_types`: Validated body inputs, including the intrinsic index.
    ///   - `carry_count`: Number of leading carried inputs.
    ///   - `length`: Declared trip count.
    ///   - `input_types`: Actual operation input types, including a trailing runtime length for dynamic scans.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched input counts, ordinary types, or runtime length types.
    fn validate_scan_input_type_signature(
        body_input_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<(), TypeError>;

    /// Validates that every stacked leading axis is provably equal to the trailing runtime length. Ordinary input
    /// types must already satisfy [`Self::validate_scan_input_type_signature`]. An eager value can have a nominal
    /// dimension type and concretely sized stacks whose equality is only established when interpreting the scan;
    /// transform rules must retain that runtime validation when this static proof fails.
    ///
    /// # Parameters
    ///
    ///   - `length`: Declared trip count.
    ///   - `carry_count`: Number of leading carried inputs.
    ///   - `input_types`: Actual operation input types, including the trailing runtime length when dynamic.
    ///
    /// # Errors
    ///
    /// Returns an error when runtime length and stacked leading dimensions are not provably equal.
    fn validate_scan_runtime_geometry(
        length: &Dimension,
        carry_count: usize,
        input_types: &[Self],
    ) -> Result<(), TypeError>;

    /// Derives the instantiated body input types, `[index, carry..., x_slice_or_reference...]`, from this scan's
    /// operation input types.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Operation input types, `[carry..., stacked_x...]`, followed by the runtime length input when
    ///     `length` is dynamic.
    ///   - `body_input_count`: Number of inputs of the attached body, including its intrinsic index input.
    ///   - `carry_count`: Number of loop-carried state leaves.
    ///   - `length`: Declared scan length.
    fn infer_scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError>;

    /// Validates a scan body signature and this scan's operation input types against it, and returns the operation
    /// output types, `[carry..., stacked_y...]`.
    ///
    /// The expected operation input types are derived from the body signature. Declared stacked array inputs omit
    /// optional layout and sharding metadata, while carries and reference stacks retain their body-declared types.
    /// Actual `input_types` may carry more precise metadata, such as the normalized
    /// [`Sharding`](crate::arrays::Sharding)s that concrete backend array types carry. Validation therefore uses the
    /// directional declared-vs-actual [`Type::is_refined_by`] relation instead of strict type equality, except for
    /// reference inputs, which must match their declared types exactly because their allocations keep their types when
    /// references are discharged. The output types are the carry output types of the body followed by its stacked
    /// output types, each stacked along a new leading scan axis, so they carry the shardings that the body declares
    /// (e.g., after staging specialized the body to sharded inputs) and leave unspecified the ones that it does not.
    /// Implementations may further refine them by the static extents that the inputs establish for dynamic dimensions
    /// (e.g., an `f32[3]` carry input for a carry that the body declares as `f32[rows]`), except where a carry defines
    /// the dimension's identity and may therefore change it across iterations.
    ///
    /// # Parameters
    ///
    ///   - `body_input_types`: Body input types, `[index, carry..., x_slice_or_reference...]`.
    ///   - `body_output_types`: Body output types, `[carry..., y_slice...]`.
    ///   - `carry_count`: Number of loop-carried state leaves.
    ///   - `length`: Declared scan length.
    ///   - `input_types`: Operation input types, `[carry..., stacked_x...]`, followed by the runtime length input when
    ///     `length` is dynamic.
    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError>;
}

impl ScanType for ArrayType {
    fn validate_scan_body_type_signature(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        validate_scan_length(length)?;
        if length.variable().is_some() {
            return Err(TypeError::invalid(format!(
                "homogeneous array `{SCAN_OPERATION_NAME}` requires a static length but got `{length}`; use a \
                 composite `{SCAN_OPERATION_NAME}` with a trailing first-class dimension input for a dynamic trip \
                 count",
            )));
        }
        if body_input_types.first() != Some(&ArrayType::scalar(DataType::I64)) {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` body input 0 must be a scalar `i64` slice index",
            )));
        }
        let body_input_types = &body_input_types[1..];
        if carry_count > body_input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds the body input count {}",
                body_input_types.len(),
            )));
        }
        if carry_count > body_output_types.len() {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds the body output count {}",
                body_output_types.len(),
            )));
        }
        check_types!(@same, format!("`{SCAN_OPERATION_NAME}` body carry"), [
            &body_input_types[..carry_count],
            &body_output_types[..carry_count],
        ]);
        for (index, input_type) in body_input_types.iter().enumerate() {
            validate_static_scan_type("input", index + 1, input_type)?;
        }
        for (index, output_type) in body_output_types.iter().enumerate() {
            validate_static_scan_type("output", index, output_type)?;
        }
        Ok(())
    }

    fn infer_scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError> {
        let stacked_input_end = body_input_count
            .checked_sub(1)
            .ok_or_else(|| TypeError::invalid(format!("`{SCAN_OPERATION_NAME}` body must have a slice index input")))?;
        check_count!("input", input_types, stacked_input_end, TypeError);
        if carry_count > stacked_input_end {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds the body input count {stacked_input_end}",
            )));
        }
        let mut body_input_types = vec![ArrayType::scalar(DataType::I64)];
        body_input_types.extend_from_slice(&input_types[..carry_count]);
        for (index, input_type) in input_types[carry_count..].iter().enumerate() {
            let index = carry_count + index;
            if input_type.rank() == 0 {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked input {index} must have rank at least 1",
                )));
            }
            let (slice_type, leading_dimension) = scan_slice_type(input_type, 0)?;
            if !length.is_refined_by(&leading_dimension) {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked input {index} must have leading dimension `{length}` but has type \
                     `{input_type}`",
                )));
            }
            body_input_types.push(slice_type);
        }
        Ok(body_input_types)
    }

    fn validate_scan_input_type_signature(
        body_input_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<(), TypeError> {
        let body_input_types = &body_input_types[1..];
        let mut expected_input_types = body_input_types[..carry_count].to_vec();
        expected_input_types.extend(
            body_input_types[carry_count..]
                .iter()
                .map(|slice_type| declared_stacked_scan_input_type(slice_type, length)),
        );
        check_count!("input", input_types, expected_input_types.len(), TypeError);
        for (index, (expected, actual)) in expected_input_types.iter().zip(input_types).enumerate() {
            if let Some(relation) = region_input_mismatch(expected, actual) {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` input {index} has type `{actual}`, which {relation} its expected type \
                     `{expected}`",
                )));
            }
            if index >= carry_count {
                scan_slice_type(actual, 0)?;
            }
        }
        Ok(())
    }

    fn validate_scan_runtime_geometry(
        _length: &Dimension,
        _carry_count: usize,
        _input_types: &[Self],
    ) -> Result<(), TypeError> {
        Ok(())
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        Self::validate_scan_body_type_signature(body_input_types, body_output_types, carry_count, length)?;
        Self::validate_scan_input_type_signature(body_input_types, carry_count, length, input_types)?;
        let mut output_types = body_output_types[..carry_count].to_vec();
        output_types
            .extend(body_output_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
        Ok(output_types)
    }
}

impl ScanType for ArrayIrType {
    fn validate_scan_body_type_signature(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        validate_scan_length(length)?;
        if body_input_types.first() != Some(&Self::Array(ArrayType::scalar(DataType::I64))) {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` body input 0 must be a scalar `i64` slice index",
            )));
        }
        let body_input_types = &body_input_types[1..];
        composite_scan_boundary_types(ScanBoundarySide::Input, body_input_types, carry_count, length)?;
        composite_scan_boundary_types(ScanBoundarySide::Output, body_output_types, carry_count, length)?;
        check_types!(@same, format!("`{SCAN_OPERATION_NAME}` body carry"), [
            &body_input_types[..carry_count],
            &body_output_types[..carry_count],
        ]);
        Ok(())
    }

    fn infer_scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError> {
        let runtime_length_count = usize::from(length.variable().is_some());
        let stacked_input_end = body_input_count
            .checked_sub(1)
            .ok_or_else(|| TypeError::invalid(format!("`{SCAN_OPERATION_NAME}` body must have a slice index input")))?;
        check_count!("input", input_types, stacked_input_end + runtime_length_count, TypeError);
        if carry_count > stacked_input_end {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds the body input count {stacked_input_end}",
            )));
        }
        let mut body_input_types = vec![Self::Array(ArrayType::scalar(DataType::I64))];
        body_input_types.extend_from_slice(&input_types[..carry_count]);
        for (index, r#type) in input_types[carry_count..stacked_input_end].iter().enumerate() {
            // Arrays enter as slices; references enter as whole roots so the body can form its own indexed view.
            let stacked_type = match r#type {
                Self::Array(r#type) => r#type,
                Self::Reference(reference) => reference.referent(),
                Self::Dimension(_) => {
                    return Err(TypeError::invalid(format!(
                        "`{SCAN_OPERATION_NAME}` stacked input {} must be an array or a reference but got `{type}`",
                        carry_count + index,
                        r#type = r#type,
                    )));
                }
            };
            if stacked_type.rank() == 0 {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked input {} must have rank at least 1",
                    carry_count + index,
                )));
            }
            let (slice_type, leading_dimension) = scan_slice_type(stacked_type, 0)?;
            if !length.is_refined_by(&leading_dimension) {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked input {} must have leading dimension `{length}` but has \
                     type `{type}`",
                    carry_count + index,
                    r#type = r#type,
                )));
            }
            body_input_types.push(match r#type {
                Self::Reference(_) => r#type.clone(),
                _ => Self::Array(slice_type),
            });
        }
        validate_scan_runtime_length(length, input_types, carry_count, stacked_input_end)?;
        Ok(body_input_types)
    }

    fn validate_scan_input_type_signature(
        body_input_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<(), TypeError> {
        let body_input_types = &body_input_types[1..];
        let expected_input_types =
            composite_scan_boundary_types(ScanBoundarySide::Input, body_input_types, carry_count, length)?;
        let runtime_length_count = usize::from(length.variable().is_some());
        check_count!("input", input_types, expected_input_types.len() + runtime_length_count, TypeError);
        for (index, (expected, actual)) in
            expected_input_types.iter().zip(&input_types[..expected_input_types.len()]).enumerate()
        {
            if let Some(relation) = region_input_mismatch(expected, actual) {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` input {index} has type `{actual}`, which {relation} its expected type \
                     `{expected}`",
                )));
            }
            if index >= carry_count
                && let ArrayIrType::Array(actual) = actual
            {
                scan_slice_type(actual, 0)?;
            }
        }
        validate_scan_runtime_length_type(length, input_types)?;
        Ok(())
    }

    fn validate_scan_runtime_geometry(
        length: &Dimension,
        carry_count: usize,
        input_types: &[Self],
    ) -> Result<(), TypeError> {
        let stacked_input_end = input_types.len() - usize::from(length.variable().is_some());
        validate_scan_runtime_length(length, input_types, carry_count, stacked_input_end).map(|_| ())
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        Self::validate_scan_body_type_signature(body_input_types, body_output_types, carry_count, length)?;
        Self::validate_scan_input_type_signature(body_input_types, carry_count, length, input_types)?;
        let expected_input_types =
            composite_scan_boundary_types(ScanBoundarySide::Input, &body_input_types[1..], carry_count, length)?;
        let output_types =
            composite_scan_boundary_types(ScanBoundarySide::Output, body_output_types, carry_count, length)?;
        // A runtime length input that only refines the declared bounds fixes the trip count to one exact extent, so the
        // stacked boundary axes are inferred at that extent instead of at the still-symbolic declared length. Leaving
        // them symbolic would type a concretely sized result as an independent runtime extent.
        let output_types =
            match validate_scan_runtime_length(length, input_types, carry_count, expected_input_types.len())? {
                Some(extent) => composite_scan_boundary_types(
                    ScanBoundarySide::Output,
                    body_output_types,
                    carry_count,
                    &Dimension::Static(extent),
                )?,
                None => output_types,
            };
        // The facts that the inputs establish then refine the outputs (refer to `refine_output_types`). This is a
        // separate step from the trip-count refinement above: the stacked outputs' leading axis describes the trip
        // count that is fixed before the first iteration, so it stays refined even when a first-class dimension carry
        // shares the length's identity and keeps every carry that refers to it symbolic. Only the carries preserve
        // the identities of reference inputs.
        refine_output_types(
            expected_input_types.as_slice(),
            &input_types[..expected_input_types.len()],
            output_types.as_slice(),
            |index| (index < carry_count).then_some(index),
        )
    }
}

/// Returns the per-iteration slice type of a stacked scan value of type `stacked_type` whose scan axis is `axis`,
/// together with the scan dimension. A placement over [`Auto`](MeshAxisType::Auto) mesh axes on the scan axis is only a
/// hint for the compiler, so the slice drops it: each iteration reads a single position of that axis, and the compiler
/// remains free to place the slice. A scan axis sharded over [`Manual`](MeshAxisType::Manual) mesh axes makes those
/// axes vary across the slice (refer to [`ArrayType::without_dimension`]), while one sharded over
/// [`Explicit`](MeshAxisType::Explicit) mesh axes is rejected, because only an explicit reshard can remove an explicit
/// placement.
fn scan_slice_type(stacked_type: &ArrayType, axis: usize) -> Result<(ArrayType, Dimension), TypeError> {
    let mut stacked_type = stacked_type.clone();
    if let Some(sharding) = stacked_type.sharding()
        && let Some(ShardingDimension::Sharded(axis_names)) = sharding.dimensions().get(axis)
        && axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
    {
        let axis_names = axis_names
            .iter()
            .filter(|name| sharding.mesh().axis_type(name) != Some(MeshAxisType::Auto))
            .cloned()
            .collect::<Vec<_>>();
        let mut dimensions = sharding.dimensions().to_vec();
        dimensions[axis] =
            if axis_names.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axis_names) };
        let sharding = sharding.with_dimensions(dimensions).map_err(|error| TypeError::invalid(error.to_string()))?;
        stacked_type = stacked_type.with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?;
    }
    stacked_type.without_dimension(axis).map_err(|error| {
        TypeError::invalid(format!(
            "`{SCAN_OPERATION_NAME}` cannot slice a stacked value of type `{stacked_type}` along its scan axis \
             ({error}); reshard it so that its scan axis is not sharded over explicit mesh axes",
        ))
    })
}

/// Validates that every dimension of `r#type` is static, reporting a precise error that names the scan `role` (for
/// example, `input 1` or `output 0`) when one is not. Homogeneous scans require static body array types, and composite
/// scans require static reference stacks so their per-iteration access paths have statically known root shapes.
/// Composite array slices may instead retain dynamic dimensions.
fn validate_static_scan_type(role: &str, index: usize, r#type: &ArrayType) -> Result<(), TypeError> {
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if dimension.value().is_none() {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` body {role} {index} must have a fully static type but axis {axis} of `{type}` \
                 has size `{dimension}`",
                r#type = r#type,
            )));
        }
    }
    Ok(())
}

/// Checks that a static scan length fits the scalar `i64` loop counter. Dynamic lengths receive the same bound when
/// their runtime [`DimensionValue`] is created, so symbolic bounds need no additional restriction here.
fn validate_scan_length(length: &Dimension) -> Result<(), TypeError> {
    if let Some(length) = length.value()
        && length > MAX_DIMENSION_EXTENT
    {
        return Err(TypeError::invalid(format!(
            "`{SCAN_OPERATION_NAME}` length {length} exceeds the maximum supported extent {MAX_DIMENSION_EXTENT}",
        )));
    }
    Ok(())
}

/// Returns the stacked variant of a scan body slice type, prepending `length` to its shape. The stacked type preserves
/// the slice's memory placement and sharding, with the stacked dimension replicated (i.e., it is the inverse of slicing
/// a stacked input with [`ArrayType::without_dimension`]), but carries no layout. This is the type of the stacks that
/// scans produce (e.g., their stacked outputs and the residual stacks of their derivatives).
pub(crate) fn stacked_scan_type<L: Into<Dimension>>(slice_type: &ArrayType, length: L) -> ArrayType {
    // Inserting a replicated dimension into a valid sharding cannot fail.
    slice_type.with_inserted_dimension(0, length.into()).unwrap()
}

/// Returns the declared type of a stacked scan input whose slices have type `slice_type`, which is the
/// [`stacked_scan_type`] of `slice_type` without its optional sharding metadata. Scan input validation compares it
/// against actual input types with [`Type::is_refined_by`], so stacked inputs are validated by their data types,
/// shapes, and memory placements alone.
fn declared_stacked_scan_input_type(slice_type: &ArrayType, length: &Dimension) -> ArrayType {
    let mut dimensions = Vec::with_capacity(slice_type.rank() + 1);
    dimensions.push(length.clone());
    dimensions.extend(slice_type.shape().dimensions().iter().cloned());
    ArrayType::new(slice_type.data_type(), Shape::new(dimensions)).with_memory(slice_type.memory())
}

/// Side of a composite scan's body signature used to derive its operation boundary. Reference stacks are admitted only
/// as inputs; a body cannot return a reference to be stacked.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ScanBoundarySide {
    Input,
    Output,
}

impl Display for ScanBoundarySide {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Input => "input",
            Self::Output => "output",
        })
    }
}

/// Derives operation types from a composite body's carry and trailing types, excluding its intrinsic index input.
/// Carries are unchanged and trailing arrays gain a leading scan axis. A trailing input reference already names the
/// whole stack and keeps its type; a trailing reference output is rejected.
fn composite_scan_boundary_types(
    side: ScanBoundarySide,
    body_types: &[ArrayIrType],
    carry_count: usize,
    length: &Dimension,
) -> Result<Vec<ArrayIrType>, TypeError> {
    if carry_count > body_types.len() {
        return Err(TypeError::invalid(format!(
            "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds the body {side} count {}",
            body_types.len(),
        )));
    }
    let mut boundary_types = body_types[..carry_count].to_vec();
    for (index, r#type) in body_types[carry_count..].iter().enumerate() {
        let index = carry_count + index;
        boundary_types.push(match (r#type, side) {
            (ArrayIrType::Array(r#type), ScanBoundarySide::Input) => {
                ArrayIrType::Array(declared_stacked_scan_input_type(r#type, length))
            }
            (ArrayIrType::Array(r#type), ScanBoundarySide::Output) => {
                ArrayIrType::Array(stacked_scan_type(r#type, length))
            }
            (ArrayIrType::Reference(reference), ScanBoundarySide::Input) => {
                validate_static_scan_type("input", index + 1, reference.referent())?;
                if reference.referent().rank() == 0 || !length.is_refined_by(&reference.referent().dimension(0)) {
                    return Err(TypeError::invalid(format!(
                        "`{SCAN_OPERATION_NAME}` reference input {index} must have leading dimension `{length}`",
                    )));
                }
                r#type.clone()
            }
            (ArrayIrType::Reference(_), ScanBoundarySide::Output) => {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked body output {index} must be an array but got `{type}`; a body \
                     cannot return a per-iteration reference view",
                    r#type = r#type,
                )));
            }
            (ArrayIrType::Dimension(_), _) => {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` stacked body {side} {} must be an array but got `{type}`",
                    index + usize::from(side == ScanBoundarySide::Input),
                    r#type = r#type,
                )));
            }
        });
    }
    Ok(boundary_types)
}

/// Validates the runtime dimension's identity or singleton refinement independently of stacked-axis equality.
fn validate_scan_runtime_length_type(
    length: &Dimension,
    input_types: &[ArrayIrType],
) -> Result<Option<usize>, TypeError> {
    let Some(variable) = length.variable() else {
        return Ok(None);
    };
    let runtime_length_type = <&DimensionType>::try_from(input_types.last().unwrap())?;
    if runtime_length_type.variable() == variable {
        return Ok(None);
    }
    let extent = runtime_length_type
        .extent()
        .filter(|_| DimensionType::from(variable.clone()).is_refined_by(runtime_length_type))
        .ok_or_else(|| {
            TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` runtime length input has type `{runtime_length_type}` but \
                 `{SCAN_OPERATION_NAME}` length requires `{variable}`",
            ))
        })?;
    Ok(Some(extent))
}

/// Validates the trailing runtime length input of a dynamic-length composite scan against the declared `length` and the
/// actual stacked input types, and returns the concrete trip count when the runtime length input only refines
/// `length`'s bounds instead of carrying its nominal identity.
///
/// A runtime length input that carries `length`'s own [`DimensionVariable`] defines exactly the runtime extent that
/// every stacked axis typed `length` has. Every stacked input must therefore have that same identity or an extent
/// provably equal to `length`, and [`None`] is returned. A static stack with an extent merely inside `length`'s bounds
/// cannot satisfy this contract, because the runtime length could differ from that extent. A runtime length input of an
/// unrelated identity is admissible only when its bounds pin exactly one extent inside `length`'s bounds. The trip
/// count is then that extent, which agrees with the stacked inputs only if each of them is itself
/// refined to the very same extent: a stacked axis left symbolic would be read `extent` times regardless of its actual
/// runtime size, over-reading the stacks that turn out shorter and silently truncating the longer ones.
///
/// This is the single definition of the runtime-length safety rule. Eager interpretation specializes the runtime
/// length input to its concrete extent before validating actual input types, since eager arrays have static shapes.
///
/// # Parameters
///
///   - `length`: Declared scan length.
///   - `input_types`: All scan input types, whose last entry is the runtime length input when `length` is dynamic.
///   - `carry_count`: Number of leading loop-carried inputs, which are not stacked.
///   - `stacked_input_end`: Exclusive end of the stacked input range, i.e., the index of the runtime length input.
fn validate_scan_runtime_length(
    length: &Dimension,
    input_types: &[ArrayIrType],
    carry_count: usize,
    stacked_input_end: usize,
) -> Result<Option<usize>, TypeError> {
    let Some(variable) = length.variable() else {
        return Ok(None);
    };
    let runtime_length_type = <&DimensionType>::try_from(input_types.last().unwrap())?;
    if runtime_length_type.variable() == variable {
        for (index, r#type) in input_types[carry_count..stacked_input_end].iter().enumerate() {
            let stacked_type = match r#type {
                ArrayIrType::Array(r#type) => Some(r#type),
                ArrayIrType::Reference(reference) => Some(reference.referent()),
                ArrayIrType::Dimension(_) => None,
            };
            if !stacked_type.is_some_and(|r#type| r#type.rank() > 0 && length.has_equal_extents(&r#type.dimension(0))) {
                return Err(TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` runtime length input has type `{runtime_length_type}` but stacked \
                     input {} has type `{type}` whose leading dimension is not provably equal to `{length}`",
                    carry_count + index,
                    r#type = r#type,
                )));
            }
        }
        return Ok(None);
    }
    let extent = validate_scan_runtime_length_type(length, input_types)?.unwrap();
    for (index, r#type) in input_types[carry_count..stacked_input_end].iter().enumerate() {
        let stacked_type = match r#type {
            ArrayIrType::Array(r#type) => Some(r#type),
            ArrayIrType::Reference(reference) => Some(reference.referent()),
            ArrayIrType::Dimension(_) => None,
        };
        let leading_extent = match stacked_type {
            Some(r#type) if r#type.rank() > 0 => r#type.dimension(0).bounds().extent(),
            _ => None,
        };
        if leading_extent != Some(extent) {
            return Err(TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` runtime length input has type `{runtime_length_type}` but stacked \
                 input {} has type `{type}` whose leading dimension is not refined to extent {extent}",
                carry_count + index,
                r#type = r#type,
            )));
        }
    }
    Ok(Some(extent))
}

/// Returns the [`Dimension`] that every stacked value of a composite scan is actually laid out over: the declared
/// `length`, unless the runtime length input only refines the declared bounds to one exact extent (refer to the
/// documentation of [`validate_scan_runtime_length`]), in which case it is that static extent.
fn effective_scan_length(
    length: &Dimension,
    input_types: &[ArrayIrType],
    carry_count: usize,
) -> Result<Dimension, TypeError> {
    // The trailing input is the runtime length input whenever the declared length is dynamic, so the stacked inputs
    // end one before it.
    let stacked_input_end = input_types.len() - usize::from(length.variable().is_some());
    Ok(match validate_scan_runtime_length(length, input_types, carry_count, stacked_input_end)? {
        Some(extent) => Dimension::Static(extent),
        None => length.clone(),
    })
}

/// Replaces the output of every non-reference carry that `body` passes through unchanged with that carry's input. Such
/// a carry ends with its initial value regardless of the trip count, so forwarding the input lets later work depend on
/// it instead of on the scan, keeps a known initial value known under partial evaluation, and preserves a symbolic-zero
/// tangent under differentiation. Reference carries are never forwarded, because their later uses must stay ordered
/// after the effects of the scan.
fn forward_scan_carries<V: Value, O: Operation<Type = V::Type>, Carry: Clone>(
    body: RegionRef<'_, V, O>,
    carry_count: usize,
    inputs: &[Carry],
    outputs: &mut [Carry],
) {
    let body_input_types = body.input_types();
    for index in 0..carry_count {
        if body.output_ids()[index] == body.input_ids()[index + 1] && !body_input_types[index + 1].is_reference() {
            outputs[index] = inputs[index].clone();
        }
    }
}

/// Extracts slice `iteration` of a stacked value along its leading axis and drops that axis.
///
/// The slice bounds and the squeezed shape are derived from the stacked value's own type, which must be fully static
/// with a leading axis of extent greater than `iteration` (guaranteed for stacked scan values by construction).
pub(crate) fn read_scan_iteration<V>(stack: &V, iteration: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    let stack_type = stack.r#type().into_owned();
    let (slice_type, _) = scan_slice_type(&stack_type, 0)?;
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                TypeError::invalid(format!(
                    "`{SCAN_OPERATION_NAME}` iteration extraction requires a static stacked type but got `{stack_type}`"
                ))
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = iteration;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = iteration + 1;
    let unit_strides = vec![1; dimensions.len()];
    let iteration_value = stack.slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    iteration_value.reshape_with_output_sharding(
        Shape::new(dimensions[1..].iter().map(|&dimension| Dimension::Static(dimension)).collect()),
        slice_type.sharding().cloned(),
    )
}

/// Writes `value` as slice `iteration` of `accumulator` along its leading axis, prepending a unit axis to `value`
/// first.
pub(crate) fn write_scan_iteration<V>(accumulator: V, iteration: usize, value: V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + UpdateSlice + Reshape,
{
    let value_type = value.r#type().into_owned();
    let mut dimensions = Vec::with_capacity(value_type.rank() + 1);
    dimensions.push(Dimension::Static(1));
    dimensions.extend(value_type.shape().dimensions().iter().cloned());
    let expanded = value.reshape(Shape::new(dimensions))?;
    let mut start_indices = vec![0; value_type.rank() + 1];
    start_indices[0] = iteration;
    accumulator.update_slice(&expanded, start_indices.as_slice())
}

/// Splits `scan` into a *known scan* bound in the enclosing known-side context and an *unknown scan* emitted into the
/// residual program, using a fixed point over carry knownness that preserves time-varying known computations.
///
/// A carry stays known iff the body computes its next value from known values alone (its init must be known); known
/// stacked inputs are known throughout. Unlike the loop-invariance rewrite (which folds a passed-through carry once and
/// therefore requires a constant-resolved init), known-ness needs no constant resolution, so symbolic known inits
/// (tracers into a live outer trace) participate fully. Each fixed-point round splits the body through a **fresh**
/// staging context whose inputs stand in for the known body inputs, so no probe or split work can leak into the
/// caller's context.
///
/// From the converged split, the *known scan*'s body maps the known carries and known stacked slices to the known
/// next-carries, the known per-iteration outputs, and the **residual edges** — every known per-iteration value the
/// unknown side consumes — which the known scan *stacks* over the scan length as extra scanned outputs. The known scan
/// is bound whole into the enclosing known-side context over the original known inputs (interpreting it under an eager
/// context and staging it into the outer program under a staging one). The *unknown scan*'s body consumes the unknown
/// carries, the unknown stacked slices, and one slice of each stacked edge per iteration; a known body next-carry
/// belonging to an *unknown* carry (one whose value the unknown side threads) is instantiated as one more residual edge
/// that the unknown body passes through. A known scan with no carries or instructions that only returns its input
/// slices at a static length is replaced by the corresponding stacked inputs. An effectful unknown body still produces
/// a zero-output residual scan when every boundary result belongs to the known side. If the known side has no outputs,
/// residual edges, or effects, the original scan residualizes unchanged through the default rule.
///
/// # Parameters
///
///   - `context`: Partial-evaluation context that binds known work and records the residual scan.
///   - `scan`: Source scan, whose length, traversal order, and lowering attributes are preserved.
///   - `body`: Validated iteration body, including its intrinsic index input.
///   - `inputs`: Source operation inputs, followed by its runtime length when the length is dynamic.
///   - `partition_region`: Driver request that partitions `body` for each carry-knownness fixed-point mask.
fn split_scan_by_knownness<V, O, C, PartitionRegion>(
    context: &PartialEvaluationContext<C>,
    scan: &ScanOperation<V::Type>,
    body: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<C::Value>],
    mut partition_region: PartitionRegion,
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value<Type: ScanType + TemporalResidualType>,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<ScanOperation<V::Type>> + TemporalResidualOperation<V::Type>,
    PartitionRegion: FnMut(&[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>,
{
    let carry_count = scan.carry_count;
    let body_input_types = body.input_types();
    let body_output_count = body.output_types().len();
    let runtime_length_count = usize::from(scan.length.variable().is_some());
    check_count!("input", inputs, body_input_types.len() - 1 + runtime_length_count, ProgramError);
    let (body_inputs, _) = inputs.split_at(body_input_types.len() - 1);
    let input_known = std::iter::once(true)
        .chain(body_inputs.iter().map(PartialEvaluationValue::is_known))
        .collect::<Vec<bool>>();

    // Fixed point over carry known-ness, each round partitioning the borrowed body through a fresh staging context.
    // Unlike the body's derived forward-mode and transposed programs, a partition is not retained by the body region's
    // transform cache: it carries known outputs that are values of the live parent context, so the body and the
    // known-ness mask alone do not determine it.
    let mut partition_body = |carry_known: &[bool]| -> Result<PartitionedProgram<V, O>, ProgramError> {
        let body_known = (0..body_input_types.len())
            .map(|index| {
                if index == 0 {
                    true
                } else if index <= carry_count {
                    carry_known[index - 1]
                } else {
                    input_known[index]
                }
            })
            .collect::<Vec<bool>>();
        partition_region(body_known.as_slice())
    };

    let mut carry_known = input_known[1..1 + carry_count].to_vec();
    let partition = loop {
        let partition = partition_body(&carry_known)?;
        let refined = (0..carry_count)
            .map(|index| {
                carry_known[index] && matches!(partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
            })
            .collect::<Vec<bool>>();
        if refined == carry_known {
            break partition;
        }
        carry_known = refined;
    };

    // The partition records the effect constraints of its construction contract. Ordinary specialization keeps
    // global ordering; separately invoked residual work can exchange only independent reference resources.
    if partition.has_effect_ordering_conflicts() {
        return context.fold_or_residualize(O::from(scan.clone()), vec![body.to_program()], inputs);
    }
    if let Some(outputs) = reconstruct_partitioned_scan(
        scan,
        &body_input_types,
        body_output_count,
        inputs,
        &input_known,
        &carry_known,
        false,
        partition,
        |constant| Ok(context.known_value(context.parent().lift(constant)?)),
        |operation, programs, inputs| context.fold_or_residualize(operation, programs, inputs),
        |operation, programs, inputs| context.residualize(operation, programs, inputs),
    )? {
        return Ok(outputs);
    }
    context.fold_or_residualize(O::from(scan.clone()), vec![body.to_program()], inputs)
}

/// Source of a loop-invariant value that the unknown scan of a partitioned scan threads as a carry that its body passes
/// through unchanged.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ScanInvariantSource {
    /// Initial value of the known carry at this index, which the body passes through unchanged.
    Carry(usize),

    /// Value of this feeder atom of the known program, which [`hoist_scan_invariants`] computes once before the scan.
    /// Feeders are keyed by atom rather than by residual edge, because a partition can expose the same value through
    /// several edges, and the unknown scan must thread it once.
    Hoisted(AtomId),
}

/// Computes the loop-invariant `feeders` of a partitioned scan once, before the scan, by replaying the pure,
/// region-free instructions of `known_program` that produce them (`invariant_instructions` marks the candidates) in the
/// known-side context, and returns the values of every atom computed this way, keyed by atom.
///
/// # Parameters
///
///   - `known_program`: Known side of the partitioned body.
///   - `invariant_instructions`: Whether each instruction of `known_program` is pure, region-free, and only consumes
///     loop-invariant atoms.
///   - `passed_through_carry_inputs`: Body input atoms of the known carries that the body passes through unchanged,
///     paired with their carry indices, whose values are the initial values of those carries in `body_inputs`.
///   - `body_inputs`: Scan inputs, `[carry..., stacked_x...]`, in the known-side representation for known inputs.
///   - `feeders`: Atoms of `known_program` to compute.
///   - `lift_known`: Lifts a constant of `known_program` into the known-side representation.
///   - `bind_known`: Binds one operation in the known-side context.
fn hoist_scan_invariants<V, O, Input, LiftKnown, KnownBind>(
    known_program: &Program<V, O, Vec<V>, Vec<V>>,
    invariant_instructions: &[bool],
    passed_through_carry_inputs: &[(AtomId, usize)],
    body_inputs: &[Input],
    feeders: impl Iterator<Item = AtomId>,
    lift_known: &mut LiftKnown,
    bind_known: &mut KnownBind,
) -> Result<HashMap<AtomId, Input>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Clone,
    LiftKnown: FnMut(V) -> Result<Input, ProgramError>,
    KnownBind: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
{
    let region = known_program.entry_region_ref();
    let mut values = passed_through_carry_inputs
        .iter()
        .map(|(atom, index)| (*atom, body_inputs[*index].clone()))
        .collect::<HashMap<_, _>>();

    // Only the invariant instructions that the requested feeders transitively need are replayed.
    let mut needed = vec![false; region.atoms().len()];
    for feeder in feeders {
        needed[feeder.index()] = true;
    }
    let mut replayed = vec![false; region.instructions().len()];
    for (position, instruction) in region.instructions().iter().enumerate().rev() {
        if invariant_instructions[position] && instruction.outputs().iter().any(|output| needed[output.index()]) {
            replayed[position] = true;
            for input in instruction.inputs() {
                needed[input.index()] = true;
            }
        }
    }
    for (atom, needed) in needed.iter().enumerate() {
        if let (true, Some(constant)) = (*needed, region.atoms()[atom].as_constant()) {
            values.insert(AtomId::new(atom), lift_known(constant.clone())?);
        }
    }
    for (instruction, _) in region.instructions().iter().zip(&replayed).filter(|(_, replayed)| **replayed) {
        let inputs = instruction.inputs().iter().map(|input| values[input].clone()).collect::<Vec<_>>();
        let outputs = bind_known(instruction.operation().clone(), Vec::new(), &inputs)?;
        check_count!("output", outputs, instruction.outputs().len(), ProgramError);
        values.extend(instruction.outputs().iter().copied().zip(outputs));
    }
    Ok(values)
}

/// Rebuilds two scans from a partitioned iteration body, storing each varying residual once per iteration. Returns
/// `None` when a reference cannot cross the resulting boundary or the split has no known work.
///
/// Computed invariant residuals are hoisted only when the declared length guarantees an iteration. Direct input
/// forwarding never evaluates body work and remains valid when the length admits zero.
///
/// # Parameters
///
///   - `scan`: Source scan, whose geometry and lowering attributes both derived scans retain.
///   - `body_input_types`: Full iteration input signature, including the intrinsic index.
///   - `body_output_count`: Number of original iteration outputs before adding residual storage.
///   - `inputs`: Source operation values in the binding destinations' shared representation.
///   - `input_known`: Knownness of the original body inputs, including the known intrinsic index.
///   - `carry_known`: Converged knownness of the initial and next carry values.
///   - `preserve_known_scan`: Retains the known scan and its full input boundary when runtime geometry requires eager
///     value validation before any computed invariant work is hoisted.
///   - `partition`: Known and residual body programs with their compact boundary descriptors.
///   - `lift_known`: Lifts a body constant into the known binding destination.
///   - `bind_known`: Binds the known scan and any hoisted invariant work.
///   - `bind_residual`: Binds the residual scan, preserving deferred work and effects.
// These arguments keep source validation, boundary wiring, and the two binding destinations explicit.
#[allow(clippy::too_many_arguments)]
fn reconstruct_partitioned_scan<V, O, Input, LiftKnown, KnownBind, ResidualBind>(
    scan: &ScanOperation<V::Type>,
    body_input_types: &[V::Type],
    body_output_count: usize,
    inputs: &[Input],
    input_known: &[bool],
    carry_known: &[bool],
    preserve_known_scan: bool,
    partition: PartitionedProgram<V, O>,
    mut lift_known: LiftKnown,
    mut bind_known: KnownBind,
    mut bind_residual: ResidualBind,
) -> Result<Option<Vec<Input>>, ProgramError>
where
    V: Value<Type: ScanType + TemporalResidualType>,
    O: Operation<Type = V::Type> + From<ScanOperation<V::Type>> + TemporalResidualOperation<V::Type>,
    Input: Clone,
    LiftKnown: FnMut(V) -> Result<Input, ProgramError>,
    KnownBind: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
    ResidualBind: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
{
    let carry_count = scan.carry_count;
    let (body_inputs, runtime_length_inputs) = inputs.split_at(body_input_types.len() - 1);
    let contains_reference_constants =
        [partition.known_program(), partition.residual_program()].into_iter().any(|program| {
            program.entry_region_ref().computation_regions().any(|region| {
                region.atoms().iter().any(|atom| atom.as_constant().is_some() && atom.r#type().is_reference())
            })
        });
    if partition.known_reference_inputs().next().is_some() || contains_reference_constants {
        return Ok(None);
    }
    let (known_program, residual_program, known_input_indices, residual_inputs, partition_outputs) =
        partition.into_parts();
    check_count!("output", partition_outputs, body_output_count, ProgramError);

    let expected_known_input_indices = (0..body_input_types.len())
        .filter(|&index| {
            if index == 0 {
                true
            } else if index <= carry_count {
                carry_known[index - 1]
            } else {
                input_known[index]
            }
        })
        .collect::<Vec<_>>();
    if known_input_indices != expected_known_input_indices {
        return Err(ProgramError::MalformedProgram(format!(
            "`{SCAN_OPERATION_NAME}` body partition reported known input indices {known_input_indices:?} but expected \
             {expected_known_input_indices:?}",
        )));
    }
    check_count!("input", residual_program.input_ids(), residual_inputs.len(), ProgramError);

    let known_result_count = partition_outputs
        .iter()
        .filter(|output| matches!(output, PartialEvaluationOutput::Known(_)))
        .count();
    let feeder_edge_count =
        residual_inputs.iter().filter(|input| matches!(input, PartialEvaluationInput::Known(_))).count();
    check_count!("output", known_program.output_ids(), known_result_count + feeder_edge_count, ProgramError);
    let known_program_output_types = known_program.output_types();

    // Assemble the known body's outputs: the known next-carries, then the known per-iteration outputs, then the
    // residual edges (the known feeders the unknown side consumes, plus the instantiated known next-carries of
    // unknown carries). Absolute positions into this list equal positions into the known scan's outputs, because the
    // known scan's outputs are its final carries followed by its stacked per-iteration outputs in the same order.
    let mut known_program_output_indices = Vec::with_capacity(known_program.output_ids().len());
    let mut known_program_output_edges = Vec::with_capacity(known_program.output_ids().len());
    let mut known_carry_output_positions = vec![None; carry_count];
    for index in 0..carry_count {
        if carry_known[index] {
            match &partition_outputs[index] {
                PartialEvaluationOutput::Known(output) => {
                    known_carry_output_positions[index] = Some(known_program_output_indices.len());
                    known_program_output_indices.push(*output);
                    known_program_output_edges.push(None);
                }
                PartialEvaluationOutput::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` known-ness fixed point converged with an unknown next value for a \
                         known carry",
                    )));
                }
            }
        }
    }
    let mut known_y_output_positions = vec![None; body_output_count - carry_count];
    for (position, output) in partition_outputs[carry_count..].iter().enumerate() {
        if let PartialEvaluationOutput::Known(output) = output {
            known_y_output_positions[position] = Some(known_program_output_indices.len());
            known_program_output_indices.push(*output);
            known_program_output_edges.push(None);
        }
    }
    // Atoms of the known program that hold the same value on every iteration: constants, the inputs of the known
    // carries that the body passes through unchanged, and the outputs of pure, region-free instructions over such
    // atoms. A feeder among them is computed once before the scan and threaded through the unknown scan as an
    // invariant carry instead of being stacked per iteration (the counterpart of JAX hoisting loop-invariant
    // residuals out of `scan`).
    let known_region = known_program.entry_region_ref();
    let mut invariant_atoms = known_region.atoms().iter().map(|atom| atom.as_constant().is_some()).collect::<Vec<_>>();
    let mut passed_through_carry_inputs = Vec::new();
    let mut passed_through_carries = vec![false; carry_count];
    for index in 0..carry_count {
        let Some(position) = known_input_indices.iter().position(|&input| input == index + 1) else {
            continue;
        };
        let carry_input = known_program.input_ids()[position];
        if carry_known[index]
            && matches!(
                &partition_outputs[index],
                PartialEvaluationOutput::Known(carry) if known_program.output_ids()[*carry] == carry_input
            )
        {
            invariant_atoms[carry_input.index()] = true;
            passed_through_carry_inputs.push((carry_input, index));
            passed_through_carries[index] = true;
        }
    }
    let mut invariant_instructions = Vec::with_capacity(known_region.instructions().len());
    for instruction in known_region.instructions() {
        let effects = instruction.operation().effects();
        let invariant = instruction.regions().is_empty()
            && effects.is_pure()
            && !effects.summary().has_deferred_work()
            && instruction.inputs().iter().all(|input| invariant_atoms[input.index()]);
        if invariant {
            for output in instruction.outputs() {
                invariant_atoms[output.index()] = true;
            }
        }
        invariant_instructions.push(invariant);
    }

    let mut edge_types = Vec::new();
    let mut index_edges = Vec::new();
    let mut edge_invariant_sources = Vec::new();
    let mut feeder_edge_positions = Vec::with_capacity(residual_inputs.len());
    let mut edge_stacked_inputs = Vec::with_capacity(residual_inputs.len());
    for input in residual_inputs.iter() {
        match input {
            PartialEvaluationInput::Known(edge) => {
                if *edge != edge_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` body partition reported residual edge {edge} out of order",
                    )));
                }
                let output = known_result_count + edge;
                let output_type = known_program_output_types.get(output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` body partition residual edge {edge} has no known-program output",
                    ))
                })?;
                // A feeder that is the body input of a known carry that the body passes through unchanged holds that
                // carry's initial value on every iteration, so the unknown scan threads the initial value as an
                // invariant carry. Any other loop-invariant feeder is hoisted. Every remaining feeder, including the
                // next value of a carry that the body updates, varies across iterations and is stacked below.
                let feeder = known_program.output_ids()[output];
                let invariant_source = passed_through_carry_inputs
                    .iter()
                    .find(|(carry_input, _)| *carry_input == feeder)
                    .map(|(_, index)| ScanInvariantSource::Carry(*index))
                    .or_else(|| {
                        // A computed feeder may fail even when its inputs are invariant. Keep that work inside the
                        // body unless the declared trip count guarantees at least one iteration.
                        (scan.length.bounds().lower() > 0 && invariant_atoms[feeder.index()])
                            .then_some(ScanInvariantSource::Hoisted(feeder))
                    });
                // Each split scan supplies its own selected slice index. A direct index feeder can therefore
                // refer to the residual body's index input instead of allocating and storing an index stack.
                let is_index = known_program.output_ids()[output] == known_program.input_ids()[0];

                // A direct feeder of a known stacked slice can be supplied by the original stacked input, which holds
                // exactly the slices that the known scan would otherwise stack again, as long as the slices are stored
                // as themselves.
                let stacked_input = known_program
                    .input_ids()
                    .iter()
                    .position(|input| *input == known_program.output_ids()[output])
                    .map(|position| known_input_indices[position])
                    .filter(|&index| index > carry_count);
                let stacked_input = match stacked_input {
                    Some(index)
                        if O::residual_to_storage(output_type)?.is_none()
                            && O::residual_from_storage(output_type)?.is_none()
                            && output_type.temporal_storage_type()? == *output_type =>
                    {
                        Some(index - 1)
                    }
                    _ => None,
                };
                index_edges.push(is_index);
                edge_types.push(output_type.clone());
                edge_invariant_sources.push(invariant_source);
                edge_stacked_inputs.push(stacked_input);
                if invariant_source.is_some() || is_index || stacked_input.is_some() {
                    feeder_edge_positions.push(None);
                } else {
                    feeder_edge_positions.push(Some((*edge, known_program_output_indices.len())));
                    known_program_output_indices.push(output);
                    known_program_output_edges.push(Some(*edge));
                }
            }
            PartialEvaluationInput::Unknown(_) => feeder_edge_positions.push(None),
        }
    }
    let mut instantiated_edge_positions = vec![None; carry_count];
    for index in 0..carry_count {
        if !carry_known[index]
            && let PartialEvaluationOutput::Known(output) = &partition_outputs[index]
        {
            let output_type = known_program_output_types.get(*output).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` body partition output {index} references missing known-program output \
                     {output}",
                ))
            })?;
            instantiated_edge_positions[index] = Some((edge_types.len(), known_program_output_indices.len()));
            let edge = edge_types.len();
            index_edges.push(false);
            edge_types.push(output_type.clone());
            edge_invariant_sources.push(None);
            known_program_output_indices.push(*output);
            known_program_output_edges.push(Some(edge));
        }
    }
    let mut invariant_sources = Vec::new();
    let edge_invariant_carry_positions = edge_invariant_sources
        .iter()
        .map(|source| {
            source.map(|source| {
                invariant_sources.iter().position(|&candidate| candidate == source).unwrap_or_else(|| {
                    invariant_sources.push(source);
                    invariant_sources.len() - 1
                })
            })
        })
        .collect::<Vec<_>>();
    let hoists_edges = invariant_sources.iter().any(|source| matches!(source, ScanInvariantSource::Hoisted(_)));
    let edge_storage_types = edge_types
        .iter()
        .zip(&edge_invariant_carry_positions)
        .map(
            |(edge_type, invariant_position)| {
                if invariant_position.is_some() { Ok(edge_type.clone()) } else { edge_type.temporal_storage_type() }
            },
        )
        .collect::<Result<Vec<_>, TypeError>>()?;

    // An empty known side means the split folds nothing; residualize unchanged through the default rule. A known side
    // that only forwards stacked inputs or known carries that the body passes through, or that only hoists invariant
    // feeders, still splits, but it needs no known scan.
    // A known carry that the body passes through unchanged ends with its initial value, so the known scan is needed
    // only for its other outputs or for its effects.
    let passed_through_carry_count = passed_through_carries.iter().filter(|&&passed_through| passed_through).count();
    let needs_known_scan = preserve_known_scan
        || known_program_output_indices.len() > passed_through_carry_count
        || !known_program.effects().classes().is_empty()
        || known_program.effects().has_deferred_work();
    if !needs_known_scan
        && !hoists_edges
        && passed_through_carry_count == 0
        && edge_stacked_inputs.iter().all(Option::is_none)
    {
        return Ok(None);
    }

    // Bind the known scan into the enclosing known-side context over the original known inputs.
    let known_carry_count = carry_known.iter().filter(|&&known| known).count();
    let known_scan_inputs = known_input_indices
        .iter()
        .filter(|&&index| index != 0)
        .map(|&index| {
            body_inputs.get(index - 1).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` body partition references missing `{SCAN_OPERATION_NAME}` input {index}",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut known_scan_inputs = known_scan_inputs;
    known_scan_inputs.extend_from_slice(runtime_length_inputs);
    let mut known_body_builder = ProgramBuilder::<V, O>::new();
    let known_body_inputs = known_program
        .input_types()
        .into_iter()
        .map(|input_type| known_body_builder.add_input(input_type))
        .collect::<Vec<_>>();
    let known_program_outputs = known_body_builder.splice_program(&known_program, known_body_inputs.as_slice())?;
    let known_output_atoms = known_program_output_indices
        .iter()
        .zip(&known_program_output_edges)
        .map(|(&output, &edge)| -> Result<AtomId, ProgramError> {
            let output = known_program_outputs.get(output).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` body partition references missing known-program output {output}",
                ))
            })?;
            let Some(edge) = edge else { return Ok(output) };
            let Some(operation) = O::residual_to_storage(&edge_types[edge])? else { return Ok(output) };
            let converted = known_body_builder.add_instruction(operation, Vec::new(), vec![output], None)?;
            check_count!("output", converted, 1, ProgramError);
            Ok(converted[0])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let known_body = known_body_builder.build::<Vec<V>, Vec<V>>(
        known_output_atoms,
        vec![Placeholder; known_body_inputs.len()],
        vec![Placeholder; known_program_output_indices.len()],
    )?;

    // The known program also computes the hoisted feeders, which the known scan must not recompute on every
    // iteration, so the known body keeps only the work that its outputs or effects need (and its whole boundary).
    let known_body_inputs = known_body.input_ids().to_vec();
    let known_body_outputs = known_body.output_ids().to_vec();
    let (known_body, _) = known_body.into_filtered(&known_body_inputs, &known_body_outputs, &known_body_inputs)?;
    let known_scan = ScanOperation::<V::Type>::new(known_carry_count, scan.length.clone())
        .with_reverse(scan.reverse)
        .with_unroll(scan.unroll)?;
    // A coefficient-only scan can merely return its input slices. Stacking those slices restores the original stacked
    // inputs, including under reverse iteration, so rebuilding another loop would only hide their producers. Keep
    // runtime-length scans explicit because their outputs may refine the stacked inputs' symbolic extent types.
    let forwarded_outputs = (matches!(scan.length(), Dimension::Static(_))
        && known_carry_count == 0
        && known_body.instructions().is_empty())
    .then(|| {
        known_body
            .output_ids()
            .iter()
            .map(|output| {
                known_body
                    .input_ids()
                    .iter()
                    .position(|input| input == output)
                    .and_then(|index| index.checked_sub(1))
                    .map(|index| known_scan_inputs[index].clone())
            })
            .collect::<Option<Vec<_>>>()
    })
    .flatten();
    let known_outputs = match forwarded_outputs {
        Some(outputs) => outputs,
        None if !needs_known_scan => Vec::new(),
        None => bind_known(O::from(known_scan), vec![known_body], known_scan_inputs.as_slice())?,
    };

    // Assemble the unknown body over `[index, unknown carries..., unknown stacked slices..., edge slices...]`.
    // Splice the residual body over its unknown inputs and edge inputs, with instantiated known next-carries passed
    // through from their edge slices.
    let mut unknown_output_ordinals = vec![None; body_output_count];
    let mut residual_outputs = Vec::new();
    let needs_unknown_scan =
        partition_outputs.iter().any(|output| matches!(output, PartialEvaluationOutput::Unknown(_)))
            || (0..carry_count).any(|index| !carry_known[index])
            || !residual_program.effects().classes().is_empty()
            || residual_program.effects().has_deferred_work();
    if needs_unknown_scan {
        let mut builder = ProgramBuilder::<V, O>::new();
        let index_atom = builder.add_input(body_input_types[0].clone());
        let invariant_carry_atoms = invariant_sources
            .iter()
            .map(|source| {
                builder.add_input(match source {
                    ScanInvariantSource::Carry(index) => body_input_types[index + 1].clone(),
                    ScanInvariantSource::Hoisted(feeder) => known_region.atoms()[feeder.index()].r#type().into_owned(),
                })
            })
            .collect::<Vec<_>>();
        let mut unknown_body_input_atoms = vec![None; body_input_types.len()];
        unknown_body_input_atoms[0] = Some(index_atom);
        for (index, input_type) in body_input_types.iter().enumerate() {
            let known = if index == 0 {
                true
            } else if index <= carry_count {
                carry_known[index - 1]
            } else {
                input_known[index]
            };
            if !known {
                unknown_body_input_atoms[index] = Some(builder.add_input(input_type.clone()));
            }
        }
        let mut restored_identity_edges = Vec::new();
        let mut edge_input_atoms = Vec::with_capacity(edge_types.len());
        for (edge, edge_type) in edge_types.iter().enumerate() {
            if index_edges[edge] {
                edge_input_atoms.push(index_atom);
                continue;
            }
            if let Some(position) = edge_invariant_carry_positions[edge] {
                edge_input_atoms.push(invariant_carry_atoms[position]);
                continue;
            }
            let storage = builder.add_input(edge_storage_types[edge].clone());
            let Some(operation) = O::residual_from_storage(edge_type)? else {
                edge_input_atoms.push(storage);
                continue;
            };

            // A residual partition may expose the same identity-defining value through multiple feeder edges. Restore
            // that value once so the generated region forwards one SSA definition instead of redefining the nominal
            // identity for every use.
            let defines_identity =
                edge_type.identities().any(|(position, _)| position == TypeIdentityPosition::Definition);
            if defines_identity
                && let Some((_, restored)) =
                    restored_identity_edges.iter().find(|(restored_type, _)| restored_type == edge_type)
            {
                edge_input_atoms.push(*restored);
                continue;
            }
            let restored = builder.add_instruction(operation, Vec::new(), vec![storage], None)?;
            check_count!("output", restored, 1, ProgramError);
            edge_input_atoms.push(restored[0]);
            if defines_identity {
                restored_identity_edges.push((edge_type.clone(), restored[0]));
            }
        }

        let mut spliced_inputs = Vec::with_capacity(residual_inputs.len());
        for input in residual_inputs.iter() {
            match input {
                PartialEvaluationInput::Unknown(index) => {
                    spliced_inputs.push(unknown_body_input_atoms.get(*index).copied().flatten().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{SCAN_OPERATION_NAME}` known-ness split saw a residual feeder for a known body input"
                        ))
                    })?);
                }
                PartialEvaluationInput::Known(edge) => {
                    spliced_inputs.push(*edge_input_atoms.get(*edge).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{SCAN_OPERATION_NAME}` known-ness split lost a residual edge"
                        ))
                    })?)
                }
            }
        }
        let spliced_outputs = builder.splice_program(&residual_program, &spliced_inputs)?;

        let mut unknown_output_atoms = invariant_carry_atoms.clone();
        for index in 0..body_output_count {
            let owned_by_unknown_side = if index < carry_count {
                !carry_known[index]
            } else {
                matches!(&partition_outputs[index], PartialEvaluationOutput::Unknown(_))
            };
            if !owned_by_unknown_side {
                continue;
            }
            unknown_output_ordinals[index] = Some(unknown_output_atoms.len());
            match &partition_outputs[index] {
                PartialEvaluationOutput::Unknown(spliced) => unknown_output_atoms.push(spliced_outputs[*spliced]),
                PartialEvaluationOutput::Known(_) => {
                    let (edge, _) = instantiated_edge_positions[index].ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{SCAN_OPERATION_NAME}` known-ness split lost an instantiated carry edge"
                        ))
                    })?;
                    unknown_output_atoms.push(edge_input_atoms[edge]);
                }
            }
        }

        let unknown_body_input_count = invariant_carry_atoms.len()
            + unknown_body_input_atoms.iter().filter(|atom| atom.is_some()).count()
            + edge_invariant_carry_positions
                .iter()
                .zip(&index_edges)
                .filter(|(position, is_index)| position.is_none() && !**is_index)
                .count();
        let unknown_output_count = unknown_output_atoms.len();
        let unknown_body = builder.build::<Vec<V>, Vec<V>>(
            unknown_output_atoms,
            vec![Placeholder; unknown_body_input_count],
            vec![Placeholder; unknown_output_count],
        )?;
        let unknown_carry_count = invariant_carry_atoms.len() + carry_known.iter().filter(|&&known| !known).count();
        let unknown_scan = ScanOperation::<V::Type>::new(unknown_carry_count, scan.length.clone())
            .with_reverse(scan.reverse)
            .with_unroll(scan.unroll)?;

        // The unknown scan consumes the unknown original inputs followed by one stacked edge per residual edge, each
        // edge fed by the known scan's matching stacked output.
        let mut hoisted_values = hoist_scan_invariants(
            &known_program,
            &invariant_instructions,
            &passed_through_carry_inputs,
            body_inputs,
            invariant_sources.iter().filter_map(|source| match source {
                ScanInvariantSource::Carry(_) => None,
                ScanInvariantSource::Hoisted(feeder) => Some(*feeder),
            }),
            &mut lift_known,
            &mut bind_known,
        )?;
        let mut unknown_scan_inputs = invariant_sources
            .iter()
            .map(|source| match source {
                ScanInvariantSource::Carry(index) => body_inputs[*index].clone(),
                ScanInvariantSource::Hoisted(feeder) => hoisted_values.remove(feeder).unwrap(),
            })
            .collect::<Vec<_>>();
        for (index, input) in body_inputs.iter().enumerate() {
            let known = if index < carry_count { carry_known[index] } else { input_known[index + 1] };
            if !known {
                unknown_scan_inputs.push(input.clone());
            }
        }
        // Edges fed by the known scan take its stacked outputs, while edges that forward known stacked slices take the
        // original stacked inputs instead.
        let mut stacked_edges = Vec::new();
        for (edge, known_output_position) in
            feeder_edge_positions.iter().flatten().chain(instantiated_edge_positions.iter().flatten())
        {
            let input = known_outputs.get(*known_output_position).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` known-ness split known `{SCAN_OPERATION_NAME}` produced no output for a \
                     residual edge",
                ))
            })?;
            stacked_edges.push((*edge, input));
        }
        for (edge, stacked_input) in edge_stacked_inputs.iter().enumerate() {
            if let Some(input) = stacked_input {
                stacked_edges.push((edge, body_inputs[*input].clone()));
            }
        }
        stacked_edges.sort_by_key(|(edge, _)| *edge);
        unknown_scan_inputs.extend(stacked_edges.into_iter().map(|(_, input)| input));
        unknown_scan_inputs.extend_from_slice(runtime_length_inputs);
        residual_outputs = bind_residual(O::from(unknown_scan), vec![unknown_body], unknown_scan_inputs.as_slice())?;
    }

    // Reassemble the original scan's outputs from the two sides.
    (0..body_output_count)
        .map(|index| {
            if index < carry_count && passed_through_carries[index] {
                return Ok(body_inputs[index].clone());
            }
            let known_position = if index < carry_count {
                known_carry_output_positions[index]
            } else {
                known_y_output_positions[index - carry_count]
            };
            if let Some(position) = known_position {
                return known_outputs.get(position).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` known-ness split known `{SCAN_OPERATION_NAME}` produced no output \
                         for a known result",
                    ))
                });
            }
            let ordinal = unknown_output_ordinals[index].ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` known-ness split produced a result owned by neither side"
                ))
            })?;
            residual_outputs.get(ordinal).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` known-ness split unknown `{SCAN_OPERATION_NAME}` produced no output for a \
                     residual result",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Drives one eager batched scan loop over `[carry..., stacked_x...]` input batches, delegating each iteration's body
/// evaluation to `interpret_iteration`, allocating stacked output accumulators through `allocate_zero`, and reconciling
/// batch axes through `align_batch_axis`.
///
/// Per-iteration slices of the stacked inputs are read along their *per-item* leading axis (refer to
/// [`read_scan_iteration_batch`]) so the batch axis threads through untouched, and the per-iteration outputs are
/// stacked along a fresh leading axis, shifting each output's batch axis right by one. A carry that starts replicated
/// can become mapped after any iteration, so the same stacked output can be replicated on early iterations and mapped
/// on later ones. The first mapped iteration therefore fixes the output's batch axis, the replicated iterations stacked
/// before it are broadcast to that axis, and every later iteration is aligned to it. The visit order reverses when
/// `reverse` is `true` while output slice `i` stays aligned with input slice `i`, exactly like the unbatched scan loop.
/// `length` must be positive, because zero-length scans are batched structurally by the caller.
#[allow(clippy::too_many_arguments)]
fn batch_scan_with_interpreter<V, AllocateZeroFn, AlignBatchAxisFn, InterpretIterationFn>(
    carry_count: usize,
    length: usize,
    reverse: bool,
    stacked_output_count: usize,
    inputs: &[ArrayBatch<V>],
    mut allocate_zero: AllocateZeroFn,
    mut align_batch_axis: AlignBatchAxisFn,
    mut interpret_iteration: InterpretIterationFn,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + UpdateSlice + Reshape,
    AllocateZeroFn: FnMut(&ArrayType) -> Result<V, ProgramError>,
    AlignBatchAxisFn: FnMut(ArrayBatch<V>, Axis) -> Result<ArrayBatch<V>, BatchingError>,
    InterpretIterationFn: FnMut(usize, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let (initial_carries, stacks) = inputs.split_at(carry_count);
    let mut carries = initial_carries.to_vec();
    let mut accumulators = (0..stacked_output_count).map(|_| None).collect::<Vec<Option<ArrayBatch<V>>>>();
    for visit in 0..length {
        let iteration = if reverse { length - 1 - visit } else { visit };
        let mut iteration_inputs = carries.clone();
        for stack in stacks {
            iteration_inputs.push(read_scan_iteration_batch(stack, iteration)?);
        }
        let mut iteration_outputs = interpret_iteration(iteration, iteration_inputs)?;
        check_count!("output", iteration_outputs, carry_count + stacked_output_count, ProgramError);
        let stacked_outputs = iteration_outputs.split_off(carry_count);
        carries = iteration_outputs;
        for (slot, output) in accumulators.iter_mut().zip(stacked_outputs) {
            let (accumulator, output) = match slot.take() {
                None => {
                    // The packed accumulator retains the iteration value's mapped dimension placement, while the newly
                    // inserted leading scan dimension is replicated.
                    let output_type = output.r#type().into_owned();
                    let stacked_type = output_type.with_inserted_dimension(0, Dimension::Static(length))?;
                    let stacked_axis =
                        BatchAxis::from_optional_position(output.batch_axis_position().map(|axis| axis + 1));
                    (ArrayBatch::new(allocate_zero(&stacked_type)?, stacked_axis)?, output)
                }
                Some(accumulator) => match (accumulator.batch_axis_position(), output.batch_axis_position()) {
                    (Some(stacked_axis), Some(axis)) if stacked_axis == axis + 1 => (accumulator, output),
                    (Some(stacked_axis), _) => (accumulator, align_batch_axis(output, Axis::from(stacked_axis - 1))?),
                    (None, Some(axis)) => (align_batch_axis(accumulator, Axis::from(axis + 1))?, output),
                    (None, None) => (accumulator, output),
                },
            };
            let stacked_axis = accumulator.batch_axis();
            let output_type = output.r#type().into_owned();
            let mut dimensions = Vec::with_capacity(output_type.rank() + 1);
            dimensions.push(Dimension::Static(1));
            dimensions.extend(output_type.shape().dimensions().iter().cloned());
            let expanded = output.into_value().reshape(Shape::new(dimensions))?;
            let mut start_indices = vec![0; output_type.rank() + 1];
            start_indices[0] = iteration;
            let updated = accumulator.into_value().update_slice(&expanded, start_indices.as_slice())?;
            *slot = Some(ArrayBatch::new(updated, stacked_axis)?);
        }
    }

    // `length` is positive, so every iteration wrote every accumulator.
    carries.extend(accumulators.into_iter().map(Option::unwrap));
    Ok(carries)
}

/// Extracts slice `iteration` of a stacked batch along its *per-item* leading axis and drops that axis.
///
/// The per-item leading axis is the scan length axis: packed axis `1` when the batch axis sits at packed axis `0`, and
/// packed axis `0` otherwise. The iteration batch keeps the input's batch axis, decremented when it sat after the
/// dropped axis.
fn read_scan_iteration_batch<V>(stack: &ArrayBatch<V>, iteration: usize) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    let stack_axis = match stack.batch_axis_position() {
        Some(0) => 1,
        _ => 0,
    };
    let stack_type = stack.r#type().into_owned();
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{SCAN_OPERATION_NAME}` batching requires static stacked input types but got `{stack_type}`"
                    ),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[stack_axis] = iteration;
    let mut limit_indices = dimensions.clone();
    limit_indices[stack_axis] = iteration + 1;
    let unit_strides = vec![1; dimensions.len()];
    let iteration_value =
        stack
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    let iteration_dimensions = dimensions
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != stack_axis)
        .map(|(_, &dimension)| Dimension::Static(dimension))
        .collect::<Vec<_>>();
    let (slice_type, _) = scan_slice_type(&stack_type, stack_axis)?;
    let iteration_value = iteration_value
        .reshape_with_output_sharding(Shape::new(iteration_dimensions), slice_type.sharding().cloned())?;
    ArrayBatch::new(iteration_value, scan_iteration_batch_axis(stack.batch_axis()))
}

/// Maps a stacked scan input's packed batch axis to the corresponding per-iteration batch axis after removing the
/// per-item leading scan dimension.
fn scan_iteration_batch_axis(batch_axis: BatchAxis) -> BatchAxis {
    match batch_axis.axis() {
        Some(axis) if axis.value() == 0 => BatchAxis::new(0),
        Some(axis) => BatchAxis::new(axis.value() - 1),
        None => BatchAxis::replicated(),
    }
}

/// Requires a reference carry of a batched loop to leave its body carrying the batch axis it entered with. A
/// reference's batch axis is fixed by its referent, so the fixed-point iteration over carry axes can neither widen nor
/// move it, and a body that returns the carry at any other axis is rejected.
///
/// # Parameters
///
///   - `operation_name`: Name of the looping operation, used in the rejection diagnostic.
///   - `index`: Position of the reference carry among the loop's carries.
///   - `entering`: Batch axis the carry enters the body with.
///   - `returned`: Batch axis the batched body returns the carry with.
///
/// # Errors
///
/// Returns [`BatchingError::UnsupportedOperation`] when `returned` differs from `entering`.
pub(crate) fn validate_reference_carry_axis(
    operation_name: &str,
    index: usize,
    entering: BatchAxis,
    returned: BatchAxis,
) -> Result<(), BatchingError> {
    if entering == returned {
        return Ok(());
    }
    Err(BatchingError::UnsupportedOperation {
        message: format!(
            "`{operation_name}` reference carry {index} enters carrying `{entering}` but its body returns it carrying \
             `{returned}`; a reference carry cannot change its batch axis, so pass the reference as a batched input at \
             the axis the body produces",
        ),
    })
}

/// Returns the permutation that converts one side of a compact fused JVP body signature from JVP order
/// (`[primal_entries..., live(tangent_entries)...]`) into scan order, where carries lead the scanned entries on both
/// the primal and live tangent sides: `[primal_carries..., live(tangent_carries)..., primal_scanned...,
/// live(tangent_scanned)...]`. The `has_tangent` mask marks the primal entries whose tangent entry exists in the
/// compact signature; the position of the `k`-th live entry's tangent entry is the number of primal entries plus `k`.
fn live_scan_signature_permutation(has_tangent: &[bool], carry_count: usize) -> Result<Vec<usize>, ProgramError> {
    let entry_count = has_tangent.len();
    if carry_count > entry_count {
        return Err(ProgramError::MalformedProgram(format!(
            "`{SCAN_OPERATION_NAME}` carry count {carry_count} exceeds fused body signature size {entry_count}",
        )));
    }
    let tangent_positions = has_tangent
        .iter()
        .scan(entry_count, |next_position, &live| {
            let position = live.then_some(*next_position);
            *next_position += usize::from(live);
            Some(position)
        })
        .collect::<Vec<_>>();
    let mut permutation = Vec::with_capacity(entry_count + has_tangent.iter().filter(|&&live| live).count());
    permutation.extend(0..carry_count);
    permutation.extend(tangent_positions[..carry_count].iter().flatten());
    permutation.extend(carry_count..entry_count);
    permutation.extend(tangent_positions[carry_count..].iter().flatten());
    Ok(permutation)
}

/// Rebuilds `program` with a new public boundary order. `input_order` and `output_order` list old boundary positions in
/// the desired new order.
fn reorder_program_boundary<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_order: &[usize],
    output_order: &[usize],
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>,
{
    /// Validates a complete boundary permutation and maps each old position to its new position.
    fn inverse_order(order: &[usize], length: usize, label: &str) -> Result<Vec<usize>, ProgramError> {
        if order.len() != length {
            return Err(ProgramError::MalformedProgram(format!(
                "{label} permutation has length {} but boundary has length {length}",
                order.len(),
            )));
        }
        let mut inverse = vec![None; length];
        for (new_position, &old_position) in order.iter().enumerate() {
            let Some(slot) = inverse.get_mut(old_position) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references out-of-range position {old_position}",
                )));
            };
            if slot.is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references position {old_position} more than once",
                )));
            }
            *slot = Some(new_position);
        }
        // Equal lengths, in-range positions, and uniqueness prove that every position is present.
        Ok(inverse.into_iter().map(Option::unwrap).collect())
    }

    let input_types = program.input_types();
    let output_count = program.output_count();
    let inverse_input_order = inverse_order(input_order, input_types.len(), "input")?;
    inverse_order(output_order, output_count, "output")?;
    let reordered_input_types = input_order.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
    let mut builder = ProgramBuilder::new();
    let inputs = reordered_input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let original_inputs = inverse_input_order.iter().map(|&new_position| inputs[new_position]).collect::<Vec<_>>();
    let outputs = builder.splice_program(program, original_inputs.as_slice())?;
    let reordered_outputs = output_order.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
    builder.build(reordered_outputs, vec![Placeholder; input_order.len()], vec![Placeholder; output_order.len()])
}

/// Partition-aware transposition rule for a [`ScanOperation`] in a tangent program. The per-iteration residuals of a
/// linearized scan are ordinary *scanned inputs* (known residual stacks) and its loop-invariant residuals are known
/// carries that the body passes through unchanged, so the rule reads them from the pullback and threads them back
/// through a transposed scan with the same loop geometry.
///
/// The scan's instruction inputs correspond to the body's inputs after its intrinsic index as `[carries...,
/// scanned_inputs...]`. The reversed loop regenerates the index. Each instruction input is independently linear (a
/// tangent the reverse accumulates) or known (a residual stack the pullback reads). The forward typically marks the
/// carry-and-scanned tangents linear and the residual stacks known, but the linear inputs need not form a leading run:
/// loop-invariant residuals ride as known carries that the body passes through unchanged (e.g., the hoisted residuals
/// of a partitioned scan or the invariant residuals of a bounded `while`), so a known input can sit among the linear
/// carries. A known carry initializer whose subsequent values depend on linear inputs is promoted to internal linear
/// state through a dependence fixed point. This includes materialized zero tangent initializers: the recurrence needs
/// their cotangents, while the known initializer itself contributes no input cotangent. This rule therefore:
///
///   1. Transposes the body through its instruction-scoped driver under each
///      input's own linearity. The transposed body maps every body output's cotangent followed by the cotangent
///      reference of every live reference input and by every known body input's runtime value to every *linear* body
///      input's cotangent:
///      `[carry_output_cotangent..., y_slice_cotangent..., cotangent_reference..., known_input_value...] ->
///      [linear_input_cotangent...]`.
///   2. Restores the reversed scan's carry-output arity, which
///      [`Program::transpose_with_respect_to`](crate::Program::transpose_with_respect_to) erases for known
///      carries (a known carry is not a linear input, so it contributes no carry cotangent output). Each known carry's
///      actual residual value is inserted into the matching body input and passed through the matching carry output,
///      so the reversed body preserves one carry slot per forward carry without fabricating a temporal zero stack.
///   3. Re-stages a primal [`ScanOperation`] over the restored body with flipped [`reverse`](ScanOperation::reverse)
///      and the same carry count, length, and (lowering-only) unroll factor, over `[output_cotangents...,
///      scanned_cotangent_references_and_known_input_stacks...]` (in body order). Known carries remain carries; only
///      known scanned inputs consume residual stacks, and only live reference stacks consume stacked cotangent
///      references. A structural-zero stacked output cotangent that needs no runtime dimensions is not an input: the
///      reversed body constructs its per-iteration zero slice itself. Flipping `reverse` pairs cotangent iteration `i`
///      with residual stack iteration `i` exactly when the forward scan consumed them, making reverse mode through the
///      scan total with no array-reversal operation.
///
/// The returned cotangents place the reversed scan's carry cotangents at the carry-input positions, its scanned-output
/// cotangents at the linear scanned-input positions, and a structural [`MaybeZero::Zero`] at the known scanned-input
/// positions, which carry no cotangent. Known initializers promoted to internal linear state also receive a structural
/// zero after their reversed carry cotangents have been computed. The body recursion happens through the
/// instruction-scoped driver's transposition requests in the same operation family, so it introduces no recursive
/// [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal scan staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-input [`PartialValue`] knowledge, ordered as `[carries..., scanned_inputs...]` and excluding the
///     body's intrinsic index. A linear input is [`Unknown`](PartialValue::Unknown); a known input is
///     [`Known`](PartialValue::Known) of the residual-stack tracer the pullback reads.
///   - `outputs`: Symbolic cotangents for the scan's outputs.
///   - `cotangents`: Cotangent destinations of the inputs (refer to the documentation of
///     [`TranspositionContext::cotangent_destinations`]). A live (`Reference`-kind) reference carry is threaded through
///     the reversed scan as a carry at its own position: the reversed body receives its cotangent reference as that
///     carry's input and passes it back out by identity as that carry's output, so every reversed iteration accumulates
///     into and reads from one shared cotangent reference. A live reference *stack* (a linear reference-typed scanned
///     input whose body input is the whole root) is threaded as a scanned input of the reversed scan at its own
///     position: its cotangent reference is the enclosing context's whole stacked cotangent reference. The reversed
///     body selects the per-iteration cotangent view using its own index and accumulates into it in place, while the
///     reversed scan has no output for it. A dead (`Ignore`-kind) reference input has no slot in the transposed body
///     and is dropped from the reversed scan's inputs. A *known* reference stack is rejected, since a linear body that
///     reads a primal reference is residualized whole by the partial-evaluation split and never reaches a tangent
///     program.
fn transpose_primal_scan<T, V, O, D: TranspositionDriver<V, O>>(
    operation: &ScanOperation<T>,
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    cotangents: &CotangentDestinations<Tracer<TracingContext<V, O>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    T: DifferentiableType + ScanType,
    V: Value<Type = T>,
    O: Operation<Type = T> + ResidualZeroProvider<T, Operation = O> + From<ScanOperation<T>>,
{
    // Validate the source boundary before zero elimination or body rewriting can conceal a malformed scan.
    operation.validate_region_count(driver.region_count())?;
    let body = driver.region(0)?;
    let body_input_types = body.input_types();
    let body_input_count = body_input_types
        .len()
        .checked_sub(1)
        .ok_or_else(|| ProgramError::MalformedProgram(format!("`{SCAN_OPERATION_NAME}` body has no index input")))?;
    let carry_count = operation.carry_count();
    let length = operation.length();
    T::validate_scan_body_type_signature(&body_input_types, &body.output_types(), carry_count, length)?;
    let runtime_length_count = usize::from(length.variable().is_some());
    check_count!("input", inputs, body_input_count + runtime_length_count, ProgramError);
    check_count!("output", outputs, body.output_types().len(), ProgramError);
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    operation.infer_output_types(&input_types, &[body.interface()])?;

    // A scan with only zero output cotangents and no live reference carry is a zero linear map, so every input
    // cotangent is zero. A live reference carry keeps the rule live, because its accumulated state cotangent flows
    // through the reversed body even when no ordinary output cotangent does. A body with deferred work or observable
    // rule effects also keeps it live.
    check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
    if outputs.iter().all(MaybeZero::is_zero)
        && !cotangents.has_reference_state_destinations()
        && !body.must_transpose()
    {
        return inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                Ok(MaybeZero::Zero(input_type.cotangent()?))
            })
            .collect();
    }

    // Input layout is `[carries..., scanned_inputs...]`, matching the body inputs after its index, where each
    // instruction input is independently linear (a tangent the reverse must accumulate) or known (a residual stack the
    // pullback reads). Linear inputs need not form a leading run: loop-invariant residuals ride as known carries that
    // the body passes through unchanged, so a known input can sit among the linear carries. The leading `carry_count`
    // instruction inputs are the carries and the rest are scanned inputs.
    let (scan_inputs, runtime_length_inputs) = inputs.split_at(body_input_count);
    let mut input_linear = scan_inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();

    // A known initializer can seed a linear recurrence: differentiating only scanned inputs materializes the initial
    // carry tangent as a known zero, although later iterations carry live tangents. Promote carries reached from linear
    // inputs to internal linear state until their dependencies stabilize. A forwarded known carry stays a residual,
    // even when reference effects make the dependence analysis conservative. Reference carries retain their resolved
    // state destinations: a known reference cannot be promoted to a value destination. The body transpose below
    // validates that the promoted recurrence is linear, and its initializer receives no returned cotangent.
    let body_program = body.to_program();
    loop {
        let linear_inputs = input_linear
            .iter()
            .enumerate()
            .filter_map(|(index, &linear)| linear.then_some(index + 1))
            .collect::<Vec<_>>();
        let output_depends = body_program.output_dependence(&linear_inputs)?;
        let mut changed = false;
        for index in 0..carry_count {
            if !input_linear[index]
                && body.output_ids()[index] != body.input_ids()[index + 1]
                && output_depends[index]
                && !scan_inputs[index].r#type().is_reference()
                && !scan_inputs[index].r#type().cotangent()?.is_zero_space()
            {
                input_linear[index] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // The remaining known carries are residual coefficients, whose forward values must pass through unchanged. A
    // changing coefficient cannot be reconstructed from its initializer during the reverse: linearization instead
    // supplies its per-iteration values as stacked residuals. Reject hand-built bodies that violate this boundary.
    if let Some(index) =
        (0..carry_count).find(|&index| !input_linear[index] && body.output_ids()[index] != body.input_ids()[index + 1])
    {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{SCAN_OPERATION_NAME}` transposition requires known carry {index} to pass through the body \
                 unchanged; time-varying known values must be supplied as stacked inputs",
            ),
        }
        .into());
    }

    // A linear reference input is a reference carry or a reference stack (a reference-typed scanned input whose body
    // input is the whole root), and the enclosing context resolved its cotangent destination. A known reference stack
    // never reaches a tangent program: a primal reference read inside a linear body is a known feeder that
    // `split_scan_by_knownness` residualizes whole, so this rejection guards hand-built programs only.
    let destination_kinds = &cotangents.kinds()[..scan_inputs.len()];
    if let Some(index) = (carry_count..scan_inputs.len())
        .find(|&index| !input_linear[index] && scan_inputs[index].r#type().is_reference())
    {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{SCAN_OPERATION_NAME}` transposition received a known reference-typed scanned input at position \
                 {index}; reference stacks are linear or absent",
            ),
        }
        .into());
    }

    // Transpose the body with each input's own linearity, threading each live reference input as a `Reference`
    // destination of the transposed body and each dead one as an `Ignore` destination. The transposed body maps the
    // cotangent of every non-reference body output, followed by the cotangent reference of every live reference input
    // and every known body input's runtime value, to the cotangent of every *linear* body input other than the dead
    // reference inputs (a live reference input's cotangent output being its cotangent reference itself):
    // `[carry_output_cotangent..., y_slice_cotangent..., cotangent_reference..., known_input_value...] ->
    // [linear_input_cotangent...]`, in body order on each side. The body is transposed through its region's retained
    // transform cache, so a body shared by several programs is transposed once per selection of linear inputs and
    // repeated attachments of the result intern by `Arc` identity.
    let (input_indices, selected_destination_kinds): (Vec<_>, Vec<_>) = input_linear
        .iter()
        .zip(destination_kinds)
        .enumerate()
        .filter_map(|(index, (&linear, &kind))| linear.then_some((index + 1, kind)))
        .unzip();
    check_count!("output", outputs, body.output_types().len(), ProgramError);
    let mut transposed_body = driver.transpose_program(body, &input_indices, &selected_destination_kinds)?;

    // A stacked output whose cotangent is a structural zero receives a per-iteration zero slice inside the reversed
    // body instead of a materialized zero stack, whenever that zero needs no runtime dimensions. Zeros that do need
    // them are materialized outside the scan below, where the boundary names their dimensions.
    let body_output_types = body.output_types();
    let zero_stacked_outputs = (0..outputs.len())
        .map(|index| -> Result<bool, DifferentiationError> {
            let output_type = &body_output_types[index];
            Ok(index >= carry_count
                && outputs[index].is_zero()
                && !output_type.is_reference()
                && O::zero_residual_types(&output_type.cotangent()?).is_empty())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // A known carry is loop state, not a stacked input. Move its exposed known-value input into the matching carry slot
    // and pass it through as the matching body output, and likewise move each live reference carry's cotangent
    // reference input into its carry slot, while a dead reference carry, which has no slot in the transposed body, is
    // dropped from the reversed scan's carries. Linear carries retain their cotangent slots, while known scanned inputs
    // are threaded as trailing per-iteration slices and live reference stacks as whole roots in body order. The
    // reversed body creates views using its own index, just as the primal body does. This avoids fabricating a zero
    // stack for a known carry and lets first-class dimension carries define the identities referenced by dynamic
    // tangent-array carries. Threading is a per-attachment rewrite of the retained transposition rather than a property
    // of the body, so the shared artifact is rebuilt here instead of being retained in its threaded form. Every
    // transposed body exposes the original index as a known input. Put it first so the reversed scan regenerates the
    // selected slice index instead of consuming an index stack from the pullback.
    transposed_body = Arc::new(thread_scan_carries(
        transposed_body.as_ref().clone(),
        body_output_types.as_slice(),
        input_linear.as_slice(),
        destination_kinds,
        &zero_stacked_outputs,
        carry_count,
    )?);
    let retained_carry_count =
        (0..carry_count).filter(|&index| cotangents.kind(index) != CotangentDestinationKind::Ignore).count();

    let transposed = ScanOperation::<T>::new(retained_carry_count, length)
        .with_reverse(!operation.reverse())
        .with_unroll(operation.unroll())?;

    // Stage the reversed scan over `[carry_cotangents..., live_stacked_cotangents..., known_input_value_stacks...]`,
    // matching the transposed body's input order. The output cotangents are typed by the *scan operation's* outputs,
    // not the body's per-iteration outputs: the leading carries keep their per-iteration shape while each trailing
    // stacked output cotangent is stacked along the scan length. A dead output's structural-zero cotangent still
    // becomes a real input of the reversed scan, unless it is a stacked output whose per-iteration zero the reversed
    // body constructs itself. Its type alone cannot construct it when it references runtime identities, but the
    // boundary collectively names every such quantity, so the zero is assembled from the peers one identity at a time.
    // Carry cotangents keep the per-iteration geometry that live peer carries and known carries also carry, while a
    // dead stacked cotangent's scan-length-prefixed geometry is split across the boundary: the length identity rides
    // the runtime length input (a first-class dimension) and any known residual stack, while its inner extents ride the
    // carries and per-iteration peers. Live cotangent references also supply their referents' geometry when every
    // ordinary output cotangent is zero and only reference state keeps the reverse live.
    let dimension_sources = || {
        outputs
            .iter()
            .filter_map(MaybeZero::as_value)
            .chain(inputs.iter().filter_map(PartialValue::as_known))
            .chain(cotangents.references())
    };
    let mut reversed_inputs = Vec::with_capacity(outputs.len() + input_linear.len());
    let mut cotangent_references = cotangents.references().iter();
    for index in 0..carry_count {
        match cotangents.kind(index) {
            // The cotangent references are consumed in input order: the carries here and the stacks below.
            CotangentDestinationKind::Reference => reversed_inputs.push(cotangent_references.next().unwrap().clone()),
            CotangentDestinationKind::Ignore => {}
            CotangentDestinationKind::Return if input_linear[index] => {
                reversed_inputs.push(O::materialize_zero_from_residual_sources(
                    context,
                    outputs[index].clone(),
                    dimension_sources(),
                )?);
            }
            CotangentDestinationKind::Return => {
                reversed_inputs.push(scan_inputs[index].as_known().cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` transposition carry input {index} has no known residual value",
                    ))
                })?);
            }
        }
    }
    for index in (carry_count..outputs.len()).filter(|&index| !zero_stacked_outputs[index]) {
        let (cotangent, output_type) = (&outputs[index], &body_output_types[index]);
        if !output_type.is_reference() && !output_type.cotangent()?.is_zero_space() {
            reversed_inputs.push(O::materialize_zero_from_residual_sources(
                context,
                cotangent.clone(),
                dimension_sources(),
            )?);
        }
    }

    // Append one scanned input per known or live reference-typed scanned body input, in body order, matching the
    // threaded body's stacked segment. A known *scanned* input is a residual stack read from the pullback (known
    // carries were already placed in their carry slots above and therefore add no trailing input here); a known
    // intermediate without a pullback value is one the partial-evaluation split must never leave in a tangent program,
    // so its absence is malformed. A live reference stack contributes the enclosing context's whole stacked cotangent
    // reference, which the reversed body views using its own index just as the primal body does, and a dead one has no
    // slot in the transposed body and contributes nothing.
    for index in carry_count..scan_inputs.len() {
        match cotangents.kind(index) {
            CotangentDestinationKind::Reference => reversed_inputs.push(cotangent_references.next().unwrap().clone()),
            CotangentDestinationKind::Ignore => {}
            CotangentDestinationKind::Return if input_linear[index] => {}
            CotangentDestinationKind::Return => {
                let residual = scan_inputs[index].as_known().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{SCAN_OPERATION_NAME}` transposition input {index} has no known residual value"
                    ))
                })?;
                reversed_inputs.push(residual.clone());
            }
        }
    }
    for (index, input) in runtime_length_inputs.iter().enumerate() {
        reversed_inputs.push(input.as_known().cloned().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` transposition runtime length input {index} is not known"
            ))
        })?);
    }

    // The reversed scan outputs one carry cotangent per retained carry and one stacked scanned-output cotangent per
    // *linear* non-reference scanned input (a reference stack's cotangent lives in its cotangent reference).
    let linear_scanned_count = (carry_count..scan_inputs.len())
        .filter(|&index| input_linear[index] && cotangents.kind(index) == CotangentDestinationKind::Return)
        .count();
    let scan_cotangents = context.stage_operation(
        O::from(transposed),
        CalleeRegionDriver::new(std::slice::from_ref(&transposed_body)),
        reversed_inputs.as_slice(),
    )?;
    check_count!("output", scan_cotangents, retained_carry_count + linear_scanned_count, ProgramError);

    // Reassemble one cotangent per input. The reversed scan outputs `[carry_cotangent..., scanned_input_cotangent...]`,
    // the carry cotangents (including the re-inserted zeros for known carries, and excluding the dropped dead reference
    // carries) leading the scanned-input cotangents over the *linear* non-reference scanned inputs. Every carry input
    // precedes every scanned input, so a single sequential drain hands each retained carry input the next carry
    // cotangent and each linear non-reference scanned input the next scanned-input cotangent in turn; known scanned
    // inputs carry a structural zero (they are residual stacks, which carry no cotangent). Known initializers promoted
    // to internal linear state likewise discard their carry cotangents. A live reference carry's
    // output is its cotangent reference, whose contents were accumulated in place, a live reference stack accumulated
    // into the enclosing context's stacked cotangent reference and has no output, and a dead reference input has no
    // output either, so every reference input receives a structural zero.
    let mut scan_cotangents = scan_cotangents.into_iter();
    let mut input_cotangents = input_linear
        .iter()
        .zip(scan_inputs)
        .enumerate()
        .map(|(index, (&linear, input))| -> Result<_, ProgramError> {
            if index < carry_count {
                match cotangents.kind(index) {
                    CotangentDestinationKind::Return if input.is_unknown() => {
                        Ok(MaybeZero::Value(scan_cotangents.next().unwrap()))
                    }
                    CotangentDestinationKind::Return | CotangentDestinationKind::Reference => {
                        scan_cotangents.next();
                        Ok(MaybeZero::Zero(input.r#type().cotangent()?))
                    }
                    CotangentDestinationKind::Ignore => Ok(MaybeZero::Zero(input.r#type().cotangent()?)),
                }
            } else if linear && cotangents.kind(index) == CotangentDestinationKind::Return {
                Ok(MaybeZero::Value(scan_cotangents.next().unwrap()))
            } else {
                Ok(MaybeZero::Zero(input.r#type().cotangent()?))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    for input in runtime_length_inputs {
        input_cotangents.push(MaybeZero::Zero(input.r#type().cotangent()?));
    }
    Ok(input_cotangents)
}

/// Rebuilds a transposed scan body so that known carry values and live reference-carry cotangent references occupy and
/// pass through their carry slots. Live reference-stack cotangent references follow as whole roots and known stacked
/// values as per-iteration slices in body order. Dead (`Ignore`-kind) reference inputs, which have no slot in the
/// transposed body, are dropped. The transposed body exposes `[non-reference output cotangents..., live reference
/// cotangent references..., known input values...]` and returns `[linear carry cotangents..., linear scanned-input
/// cotangents...]` (a live reference input's cotangent output being its cotangent reference by identity). The reversed
/// scan body consumes `[index, retained carries..., scanned slices / roots...]` and produces `[retained carries...,
/// stacked slices...]` in body order, so the boundary is permuted, the known carries are threaded through, and the
/// identity outputs of the live reference stacks, which cannot be stacked and whose accumulation is visible through the
/// shared stacked cotangent reference, are projected out. The cotangent of every body output marked in
/// `zero_stacked_outputs` is a structural zero that needs no runtime dimensions: the reversed body constructs it as a
/// per-iteration zero slice instead of reading it, so the reversed scan receives no zero stack for it.
fn thread_scan_carries<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    body_output_types: &[V::Type],
    input_linear: &[bool],
    destination_kinds: &[CotangentDestinationKind],
    zero_stacked_outputs: &[bool],
    carry_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
{
    let retained_carries = (0..carry_count)
        .filter(|&index| destination_kinds[index] != CotangentDestinationKind::Ignore)
        .collect::<Vec<_>>();
    let retained_linear_carry_count = retained_carries.iter().filter(|&&index| input_linear[index]).count();
    let body_output_count = body_output_types.len();

    // Every non-reference body output owns one cotangent slot (zero-space ones included, which keeps the numbering
    // stable), of which only the nonzero-space slots are selected below. Reference outputs own no slot at all.
    let mut cotangent_slot_count = 0;
    let mut output_cotangent_positions = Vec::with_capacity(body_output_count);
    for output_type in body_output_types {
        if output_type.is_reference() {
            output_cotangent_positions.push(None);
            continue;
        }
        let position = cotangent_slot_count;
        cotangent_slot_count += 1;
        output_cotangent_positions.push((!output_type.cotangent()?.is_zero_space()).then_some(position));
    }

    // Replace the cotangent inputs of the zero stacked outputs with zeros inside the body. The replaced inputs keep
    // their positions but become dead, so that the boundary selection below drops them.
    let zeroed_positions = (0..body_output_count)
        .filter(|&index| zero_stacked_outputs[index])
        .filter_map(|index| output_cotangent_positions[index].take())
        .collect::<Vec<_>>();
    let program = if zeroed_positions.is_empty() {
        program
    } else {
        let input_types = program.input_types();
        let mut builder = ProgramBuilder::new();
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let mut spliced_inputs = inputs.clone();
        for &position in &zeroed_positions {
            let (operation, zero_inputs) =
                O::zero_operation_with_residuals::<AtomId>(input_types[position].clone(), &[])?;
            let zero = builder.add_instruction(operation, Vec::new(), zero_inputs, None)?;
            check_count!("output", zero, 1, ProgramError);
            spliced_inputs[position] = zero[0];
        }
        let outputs = builder.splice_program(&program, &spliced_inputs)?;
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count])?
    };
    let reference_input_positions = destination_kinds
        .iter()
        .scan(cotangent_slot_count, |position, &kind| {
            let live_reference = kind == CotangentDestinationKind::Reference;
            let result = live_reference.then_some(*position);
            *position += usize::from(live_reference);
            Some(result)
        })
        .collect::<Vec<_>>();
    let known_input_start = cotangent_slot_count + reference_input_positions.iter().flatten().count();
    let known_input_positions = input_linear
        .iter()
        .scan(known_input_start + 1, |position, &linear| {
            let result = (!linear).then_some(*position);
            *position += usize::from(!linear);
            Some(result)
        })
        .collect::<Vec<_>>();
    // The intrinsic index is the first known input of the original body, after cotangent inputs.
    let mut input_order = vec![known_input_start];
    for &index in &retained_carries {
        let position = if !input_linear[index] {
            known_input_positions[index]
        } else if destination_kinds[index] == CotangentDestinationKind::Reference {
            reference_input_positions[index]
        } else {
            output_cotangent_positions[index]
        };
        input_order.push(position.ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` transposition carry {index} has no boundary slot in the transposed body",
            ))
        })?);
    }
    input_order.extend(output_cotangent_positions[carry_count..body_output_count].iter().flatten().copied());
    // A scanned input is either linear (with a cotangent reference slot exactly when it is a live reference) or known,
    // so at most one of the two positions is present for it.
    for index in carry_count..input_linear.len() {
        input_order.extend(reference_input_positions[index].or(known_input_positions[index]));
    }

    // Zero-space output-cotangent inputs carry no information and cannot affect a well-formed transposed body. Project
    // them out instead of fabricating typed values merely to satisfy the old boundary while splicing. Keeping every
    // selected input alive preserves known carry slots even when the derivative body does not otherwise read them.
    let selected_inputs = input_order
        .iter()
        .map(|&index| {
            program.input_ids().get(index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` transposition boundary references missing input position {index}",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_ids = program.output_ids().to_vec();
    let selected_input_count = selected_inputs.len();
    let (program, live_inputs) =
        program.into_filtered(selected_inputs.as_slice(), output_ids.as_slice(), selected_inputs.as_slice())?;
    if !live_inputs.iter().copied().eq(0..selected_input_count) {
        return Err(ProgramError::MalformedProgram(format!(
            "`{SCAN_OPERATION_NAME}` transposition boundary projection dropped a retained input"
        )));
    }

    let input_types = program.input_types();
    let mut builder = ProgramBuilder::new();
    let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let mut outputs = builder.splice_program(&program, inputs.as_slice())?;
    check_count!("output", outputs, program.output_count(), ProgramError);
    let trailing_outputs = outputs.split_off(retained_linear_carry_count);
    let mut linear_carry_outputs = outputs.into_iter();
    let mut restored_outputs = Vec::with_capacity(retained_carries.len() + trailing_outputs.len());
    for (position, &carry_index) in retained_carries.iter().enumerate() {
        if input_linear[carry_index] {
            restored_outputs.push(linear_carry_outputs.next().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{SCAN_OPERATION_NAME}` transposition missing linear carry cotangent output {carry_index}",
                ))
            })?);
        } else {
            restored_outputs.push(inputs[position + 1]);
        }
    }
    // The trailing outputs are the cotangents of the linear scanned inputs other than the dead reference stacks, in
    // body order. A live reference stack's output is its per-iteration cotangent view returned by identity, which the
    // reversed scan cannot stack, so only the `Return`-kind cotangents are kept.
    let mut trailing_outputs = trailing_outputs.into_iter();
    for index in carry_count..input_linear.len() {
        if !input_linear[index] || destination_kinds[index] == CotangentDestinationKind::Ignore {
            continue;
        }
        let output = trailing_outputs.next().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` transposition missing linear scanned input cotangent output {index}",
            ))
        })?;
        if destination_kinds[index] == CotangentDestinationKind::Return {
            restored_outputs.push(output);
        }
    }
    if trailing_outputs.next().is_some() {
        return Err(ProgramError::MalformedProgram(format!(
            "`{SCAN_OPERATION_NAME}` transposition body returned more scanned input cotangents than it has linear \
             scanned inputs",
        )));
    }
    let output_count = restored_outputs.len();
    builder.build(restored_outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count])
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, ArraySliceAxis, DataType,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        RaggedAxis, Sharding, ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchingTracer, RecursiveBatchingDriver, batch};
    use crate::captures::{CaptureReference, CapturingContext, ClosedProgram};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::reverse::tests::{run_transposed_with_destinations, transposition_statistics};
    use crate::differentiation::{
        CotangentDestination, CotangentSeed, Differentiate, DifferentiationTracer, Linearization, LinearizationTracer,
        ReverseModeDifferentiate, differentiate_at,
    };
    use crate::interpretation::EagerInterpretationDriver;
    use crate::macros::{check_gradient, check_operation_batching};
    use crate::operations::arithmetic::{Add, AddOperation, DivOperation, MulOperation, NegOperation};
    use crate::operations::assertions::{AssertOperation, AssertionError};
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::tests::{CountingBatchingDriver, array, dimension, resolve_captures};
    use crate::operations::custom_functions::operations::CustomFunctionTransposeOperation;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::differentiation::stop_gradient::StopGradientOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::exponential::ExpOperation;
    use crate::operations::manipulation::memory::TransferToMemoryOperation;
    use crate::operations::manipulation::slicing::DynamicSliceOperation;
    use crate::operations::reductions::{ReduceOperation, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialTracer;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, Program, ProgramBuilder, ReferenceSource, ReferenceType, ReferenceView,
        RegionDriver,
    };
    use crate::tracing::{DomainTracingContext, NestedTracingContext, Tracer, TracingContext};

    use super::*;

    /// Operation family of the homogeneous array scans under test.
    type TestOperation = ArrayOperation<Array>;

    /// Homogeneous array program, used both for scan bodies and for the programs that apply scans.
    type TestProgram = Program<Array, TestOperation, Vec<Array>, Vec<Array>>;

    /// Eager homogeneous array context.
    type TestEagerContext = EagerContext<Array, TestOperation>;

    /// Linearization tracer over [`TestEagerContext`].
    type TestTracer = LinearizationTracer<TestEagerContext>;

    /// Homogeneous array scan operation.
    type TestScanOperation = ScanOperation<ArrayType>;

    /// Composite array value, which admits references and first-class dimensions.
    type TestIrValue = ArrayIrValue<Array>;

    /// Operation family of the composite scans under test.
    type TestIrOperation = ArrayIrOperation<Array>;

    /// Composite array program, used both for scan bodies and for the programs that apply scans.
    type TestIrProgram = Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>;

    /// Linearization tracer over the eager composite array context.
    type TestIrTracer = LinearizationTracer<EagerContext<TestIrValue, TestIrOperation>>;

    /// Captured composite value in the reference discharge fixtures.
    type DischargeCapture = CaptureReference<ArrayIrType>;

    /// Operation family used by captured composite discharge programs.
    type DischargeCaptureOperation = ArrayIrOperation<CaptureReference<ArrayType>>;

    /// Direct scan-rule driver that borrows its body and uses the public retained region transforms. This avoids
    /// upstream boundary specialization when testing the rule's own validation contract.
    struct ScanDifferentiationDriver<'o, V: Value, O: Operation<Type = V::Type>>(&'o Program<V, O, Vec<V>, Vec<V>>);

    impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for ScanDifferentiationDriver<'_, V, O> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
        where
            V: 'r,
            O: 'r,
        {
            std::iter::once(self.0.entry_region_ref())
        }
    }

    impl<C> DifferentiationDriver<C> for ScanDifferentiationDriver<'_, C::Constant, C::Operation>
    where
        C: Context<Type: DifferentiableType>,
        C::Operation: DifferentiableOperation<C>
            + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
            + ResidualZeroProvider<C::Type, Operation = C::Operation>,
    {
        fn jvp_program(
            &self,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_indices: &[usize],
        ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError>
        {
            region.jvp_shared(input_indices)
        }

        fn linearize_program(
            &self,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_indices: &[usize],
        ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError> {
            region.linearize_shared(input_indices)
        }

        fn partition_jvp_program(
            &self,
            _region: RegionRef<'_, C::Constant, C::Operation>,
            _input_known: &[bool],
            _required_known_outputs: &[usize],
        ) -> Result<PartitionedProgram<C::Constant, C::Operation>, DifferentiationError> {
            panic!("scan rules do not request fused-body partitioning")
        }

        fn bind_jvp_operation<P: DifferentiationPolicy<C>>(
            &self,
            _context: &DifferentiationContext<C, P>,
            _operation: &C::Operation,
            _programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
            _inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
            panic!("scan rules do not request nested-operation replay")
        }
    }

    /// Builds a scan body over scalar `f64` loop state: `build` receives the body's `i64` index input followed by
    /// `input_count` scalar `f64` inputs (carries first, then slices) and returns the body outputs.
    fn scalar_body(
        input_count: usize,
        build: impl FnOnce(&mut ProgramBuilder<Array, TestOperation>, &[AtomId]) -> Vec<AtomId>,
    ) -> TestProgram {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let mut inputs = vec![builder.add_input(ArrayType::scalar(DataType::I64))];
        inputs.extend((0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))));
        let outputs = build(&mut builder, &inputs);
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; input_count + 1], vec![Placeholder; output_count]).unwrap()
    }

    /// Builds a program over inputs of `input_types` that applies `operation` with the attached `body` to all of its
    /// inputs and returns all of the scan's outputs.
    fn scan_program(operation: TestScanOperation, body: TestProgram, input_types: Vec<ArrayType>) -> TestProgram {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let input_count = input_types.len();
        let inputs = input_types.into_iter().map(|input_type| builder.add_input(input_type)).collect();
        let outputs = builder.add_instruction(operation, vec![body], inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`: the new carry is
    /// the running product and each iteration also emits that product as a stacked output slice.
    fn product_body() -> TestProgram {
        product_body_with_type(ArrayType::scalar(DataType::F64))
    }

    /// Builds a cumulative-product body over `r#type` that maps `[carry, x]` to `[carry * x, carry * x]`.
    fn product_body_with_type(r#type: ArrayType) -> TestProgram {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64));
        let carry = builder.add_input(r#type.clone());
        let value = builder.add_input(r#type);
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, value], None).unwrap()[0];
        builder.build(vec![product, product], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap()
    }

    /// Builds a cumulative-product [`ScanOperation`] whose nesting depth matches `lengths`, returning the operation
    /// together with its body program. Each nested body scans its slice of the stacked input with the next length.
    fn product_scan_with_lengths(lengths: &[usize]) -> (TestScanOperation, TestProgram) {
        assert!(!lengths.is_empty());
        if lengths.len() == 1 {
            return (TestScanOperation::new(1, lengths[0]), product_body());
        }
        let (inner_scan, inner_body) = product_scan_with_lengths(&lengths[1..]);
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64));
        let inner_body = builder.import_program(inner_body);
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let values = builder.add_input(ArrayType::new_static(DataType::F64, lengths[1..].to_vec()));
        let outputs =
            builder.add_instruction(inner_scan, vec![inner_body], vec![carry, values], None).unwrap().to_vec();
        let body = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();
        (TestScanOperation::new(1, lengths[0]), body)
    }

    /// Binds `operation` with the attached `body` to `inputs` in their domain and returns all of its outputs.
    fn bind_scan<V: Value<Type = ArrayType>>(
        operation: TestScanOperation,
        body: TestProgram,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = TestOperation>,
    {
        inputs[0].domain().bind(TestOperation::Scan(operation), vec![body], inputs)
    }

    /// Applies the three-iteration cumulative-product scan to `initial` and `values` in their domain and
    /// returns its final carry.
    fn apply_product_scan<V: Value<Type = ArrayType>>(initial: V, values: V) -> Result<V, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = TestOperation>,
    {
        Ok(bind_scan(TestScanOperation::new(1, 3), product_body(), &[initial, values])?.remove(0))
    }

    /// Builds a three-carry shift body `[first, second, third, item] -> [second, third, item, first]`. Dependence on an
    /// unknown third carry reaches the second carry after one fixed-point pass and the first carry after two.
    fn shifting_carry_body() -> TestProgram {
        scalar_body(4, |_, inputs| vec![inputs[2], inputs[3], inputs[4], inputs[1]])
    }

    /// Builds a body for zero-length scan tests that maps `[carry, x]` to `[carry, carry, 7]`: its first
    /// stacked result follows the carry's batch axis and its second stacked result is a replicated constant.
    fn zero_length_body(carry_type: ArrayType, slice_type: ArrayType) -> TestProgram {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64));
        let carry = builder.add_input(carry_type);
        let _value = builder.add_input(slice_type.clone());
        let constant = builder.add_constant(Array::from_elements::<f64>(slice_type, &[7.0]).unwrap());
        builder.build(vec![carry, carry, constant], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap()
    }

    /// Batches `scan` through the public [`BatchingContext::bind`] path with `body` as an owned attached region.
    fn batch_scan(
        context: &BatchingContext<TestEagerContext, ArrayBatchingPolicy>,
        scan: TestScanOperation,
        body: TestProgram,
        inputs: Vec<ArrayBatch<Array>>,
    ) -> Vec<ArrayBatch<Array>> {
        let tracer_inputs =
            inputs.into_iter().map(|input| BatchingTracer::new(context.clone(), input)).collect::<Vec<_>>();
        context
            .bind(TestOperation::Scan(scan), [body], tracer_inputs.as_slice())
            .unwrap()
            .into_iter()
            .map(|output| output.batch().clone())
            .collect()
    }

    /// Partially evaluates `program` against `knowledge`, interprets the residual program over the unknown entries of
    /// `values`, and asserts that reassembling its outputs with the folded known outputs reproduces the interpretation
    /// of `program` over `values`.
    fn assert_partial_evaluation_preserves_semantics(
        program: &TestProgram,
        knowledge: &[PartialValue<Array>],
        values: &[Array],
    ) {
        let evaluation = program.partially_evaluate(knowledge).unwrap();
        let residual_inputs = evaluation
            .inputs
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Known(value) => value.clone(),
                PartialEvaluationInput::Unknown(index) => values[*index].clone(),
            })
            .collect::<Vec<_>>();
        let residual_outputs = evaluation.program.interpret(residual_inputs).unwrap();
        let outputs = evaluation
            .outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => value.clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, program.interpret(values.to_vec()).unwrap());
    }

    /// Builds a composite scan body over a stacked reference root that maps `[carry, stack: ref<f32[3]>]` to `[carry +
    /// stack[index]]` after accumulating the carry into `stack[index]`, so that both the carry and the referent depend
    /// on the order of iteration.
    fn stacked_reference_body() -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![stack, carry, index],
                None,
            )
            .unwrap();
        let current = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![stack, index],
                None,
            )
            .unwrap()[0];
        let next_carry = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, current], None)
            .unwrap()[0];
        builder.build(vec![next_carry], vec![Placeholder; 3], vec![Placeholder]).unwrap()
    }

    /// Builds a composite scan body over a stacked reference root that maps `[carry, stack: ref<f32[3]>]` to `[carry +
    /// stack[index]]` without mutating the stack.
    fn stack_reading_body() -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let current = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![stack, index],
                None,
            )
            .unwrap()[0];
        let next_carry = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, current], None)
            .unwrap()[0];
        builder.build(vec![next_carry], vec![Placeholder; 3], vec![Placeholder]).unwrap()
    }

    /// Builds `f(carry: f32[], elements: f32[3]) -> (final_carry: f32[], elements': f32[3])`, which allocates a stacked
    /// reference over `elements`, scans [`stacked_reference_body`] over it for three iterations, and freezes the
    /// mutated stack. With the running carry `c_0 = carry` and `c_{i+1} = 2 c_i + x_i`, the outputs are `final_carry =
    /// 8 carry + 4 x_0 + 2 x_1 + x_2` and `elements'_i = x_i + c_i`, both linear in the inputs.
    fn stacked_reference_program() -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(stacked_reference_body());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        builder.build(vec![final_carry, frozen], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Builds `f(elements: f32[length]) -> (squares: f32[length], elements': f32[length])`, which allocates a stacked
    /// reference over `elements` and scans a body that passes the whole root and its index into a nested branch. The
    /// branch optionally increments the selected row (when `mutates` is `true`) and returns the square of the selected
    /// row, so the scan carries the root whole instead of slicing it per iteration.
    fn nested_region_root_program(length: usize, mutates: bool) -> TestIrProgram {
        let reference_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [length]));
        let row_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let mut branch = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = branch.add_input(ArrayType::scalar(DataType::I64).into());
        let root = branch.add_input(reference_type.clone().into());
        if mutates {
            let increment = branch.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
            branch
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(row_transforms.clone()),
                    Vec::new(),
                    vec![root, increment, index],
                    None,
                )
                .unwrap();
        }
        let value = branch
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(row_transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let squared = branch
            .add_instruction(
                TestIrOperation::Array(TestOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![value, value],
                None,
            )
            .unwrap()[0];
        let branch = branch
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![squared], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(reference_type.into());
        let predicate = body.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let branch = body.import_program(branch);
        let value = body
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, index, root], None)
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [length]).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let body = builder.import_program(body);
        let squares = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, length), vec![body], vec![root], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        builder.build(vec![squares, frozen], vec![Placeholder], vec![Placeholder; 2]).unwrap()
    }

    /// Builds a program whose composite `scan` runs the unspecialized body `[c, x] -> [c + x, x * x]` over a
    /// `f64[rows]` carry and `f64[rows]` slices for two iterations while the program feeds it a static `f64[3]` carry
    /// and static `f64[2, 3]` stacked input, so the scan's outputs are refined by its inputs rather than by a re-typed
    /// body.
    fn refined_vector_scan_program() -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)]));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(vector_type.clone().into());
        let slice = body_builder.add_input(vector_type.into());
        let next = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, slice], None)
            .unwrap()[0];
        let squared = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![slice, slice],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next, squared],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs = vec![
            builder.add_input(ArrayType::new_static(DataType::F64, [3]).into()),
            builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]).into()),
        ];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 2), vec![body], inputs, None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Builds a composite scan body over an extent carry, a scalar numeric carry, and a scalar input slice, returning
    /// the extent unchanged, the product as the next numeric carry, and that product as its stacked output slice.
    fn product_scan_body(
        extent_type: DimensionType,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type));
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let item = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let product = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![carry, item],
                None,
            )
            .unwrap()[0];
        builder.build(vec![extent, product, product], vec![Placeholder; 4], vec![Placeholder; 3]).unwrap()
    }

    #[test]
    fn test_scan() {
        // Construction defaults to forward iteration without unrolling.
        let operation = TestScanOperation::new(1, 3);
        assert_eq!(operation.carry_count(), 1);
        assert_eq!(operation.length(), &Dimension::Static(3));
        assert!(!operation.reverse());
        assert_eq!(operation.unroll(), 1);

        // The operation identity, its single body region slot, and its region provenance: body input 0 is the
        // loop-generated index, every other body input is the matching instruction input, and every output is the
        // matching body output. Only the leading carries preserve their input's reference identity.
        assert_eq!(operation.name(), SCAN_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(operation.input_region_provenance(0, 0), InputRegionProvenance::Local);
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 0 });
        assert_eq!(operation.input_region_provenance(0, 2), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );
        assert_eq!(operation.reference_output_identity_input(0), Some(0));
        assert_eq!(operation.reference_output_identity_input(1), None);

        // `Display` renders every semantic field and the unroll factor only when it is greater than 1.
        assert_eq!(operation.to_string(), "scan [carry_count=1, length=3, reverse=false]");
        assert_eq!(
            TestScanOperation::new(2, 4).with_reverse(true).with_unroll(2).unwrap().to_string(),
            "scan [carry_count=2, length=4, reverse=true, unroll=2]",
        );

        assert_eq!(
            format!("{operation:?}"),
            "ScanOperation { carry_count: 1, length: Static(3), reverse: false, unroll: 1, \
             marker: PhantomData<fn() -> ryft_core::arrays::types::arrays::ArrayType> }",
        );
        assert_eq!(operation, operation.clone());
        let different_carry_count = TestScanOperation::new(2, 3);
        let different_length = TestScanOperation::new(1, 4);
        let different_direction = operation.clone().with_reverse(true);
        let different_unroll = operation.clone().with_unroll(2).unwrap();
        assert_ne!(operation, different_carry_count);
        assert_ne!(operation, different_length);
        assert_ne!(operation, different_direction);
        assert_ne!(operation, different_unroll);
        let operations = HashMap::from([
            (operation.clone(), "original"),
            (different_carry_count, "different carry count"),
            (different_length, "different length"),
            (different_direction, "different direction"),
            (different_unroll, "different unroll"),
        ]);
        assert_eq!(operations.get(&operation), Some(&"original"));
        assert_eq!(operations.get(&TestScanOperation::new(2, 3)), Some(&"different carry count"));
        assert_eq!(operations.get(&TestScanOperation::new(1, 4)), Some(&"different length"));
        assert_eq!(operations.get(&operation.clone().with_reverse(true)), Some(&"different direction"));
        assert_eq!(operations.get(&operation.clone().with_unroll(2).unwrap()), Some(&"different unroll"));

        // Staging a scan attaches the body program as the region of the staged instruction instead of running its
        // iterations over staged values, and program rendering shows that region with its declared slot name.
        let context = DomainTracingContext::<TestEagerContext>::new();
        let carry = context.input(ArrayType::scalar(DataType::F64));
        let values = context.input(ArrayType::new_static(DataType::F64, [3]));
        let outputs = context.stage_operation(operation, [product_body()], &[carry, values]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_with_reverse() {
        // Setting the visit order preserves every other field, including an explicit unroll factor.
        let operation = TestScanOperation::new(1, 3).with_unroll(2).unwrap();
        let reversed = operation.clone().with_reverse(true);
        assert!(reversed.reverse());
        assert_eq!(reversed.carry_count(), 1);
        assert_eq!(reversed.length(), &Dimension::Static(3));
        assert_eq!(reversed.unroll(), 2);
        assert_eq!(reversed.with_reverse(false), operation);
    }

    #[test]
    fn test_scan_with_unroll() {
        // The lowering-only unroll factor must be at least 1 but need not divide the length: lowerings run the
        // remaining iterations after the unrolled loop, and a factor of at least the length unrolls completely.
        assert_eq!(
            TestScanOperation::new(1, 3).with_unroll(0),
            Err(ProgramError::Type(TypeError::invalid("`scan` unroll factor must be at least 1"))),
        );
        assert_eq!(TestScanOperation::new(1, 4).with_unroll(1).map(|operation| operation.unroll()), Ok(1));
        assert_eq!(TestScanOperation::new(1, 4).with_unroll(2).map(|operation| operation.unroll()), Ok(2));
        assert_eq!(TestScanOperation::new(1, 4).with_unroll(3).map(|operation| operation.unroll()), Ok(3));
        assert_eq!(TestScanOperation::new(1, 4).with_unroll(5).map(|operation| operation.unroll()), Ok(5));
        assert_eq!(
            TestScanOperation::new(1, 4).with_unroll(usize::MAX).map(|operation| operation.unroll()),
            Ok(usize::MAX),
        );

        // A dynamic length admits any positive factor as well, because the remainder is computed at run time.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let dynamic = ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length)).with_unroll(4).unwrap();
        assert_eq!(dynamic.unroll(), 4);
        assert_eq!(dynamic.to_string(), "scan [carry_count=1, length=length, reverse=false, unroll=4]");

        // Unrolling preserves every other field.
        let unrolled = TestScanOperation::new(2, 5).with_reverse(true).with_unroll(2).unwrap();
        assert_eq!(unrolled.carry_count(), 2);
        assert_eq!(unrolled.length(), &Dimension::Static(5));
        assert!(unrolled.reverse());
        assert_eq!(unrolled.unroll(), 2);
    }

    #[test]
    fn test_scan_with_added_carries() {
        // Widening appends the requested carries and preserves every other field, including the visit order and the
        // lowering-only unroll factor.
        let operation = TestScanOperation::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let widened = operation.with_added_carries(2).unwrap();
        assert_eq!(widened.carry_count(), 3);
        assert_eq!(widened.length(), &Dimension::Static(3));
        assert!(widened.reverse());
        assert_eq!(widened.unroll(), 3);
        assert_eq!(operation.with_added_carries(0), Ok(operation.clone()));

        // An overflowing carry count is reported instead of wrapping.
        assert_eq!(
            operation.with_added_carries(usize::MAX),
            Err(ProgramError::MalformedProgram(format!(
                "`scan` carry count 1 overflows when adding {} discharged reference state carries",
                usize::MAX,
            ))),
        );
    }

    #[test]
    fn test_scan_type_inference() {
        // Type inference validates the body signature over the attached region interface together with the operation
        // inputs and returns the carry types followed by the stacked body output types.
        let index_type = ArrayType::scalar(DataType::I64);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let operation = TestScanOperation::new(1, 3);
        let interfaces = vec![product_body().interface()];

        // A well-formed body requests no specialized region inputs, and the outputs are the carry types followed by
        // the stacked variants of the body's per-iteration output types.
        assert_eq!(
            operation.infer_region_input_types(&[scalar_type.clone(), stacked_type.clone()], interfaces.as_slice()),
            Ok(vec![None]),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_type.clone(), stacked_type.clone()], interfaces.as_slice()),
            Ok(vec![scalar_type.clone(), stacked_type.clone()]),
        );

        // The operation requires exactly one body region and validates its inputs against the body signature.
        assert_eq!(
            operation.infer_output_types(&[scalar_type.clone(), stacked_type.clone()], &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), interfaces.as_slice()),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_type.clone(), scalar_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid(
                "`scan` input 1 has type `f64[]`, which does not refine its expected type `f64[3]`"
            )),
        );

        // The body signature must cover the carries on both of its boundaries, keep the carry types invariant, and use
        // fully static types.
        assert_eq!(
            TestScanOperation::new(3, 3)
                .infer_output_types(&[scalar_type.clone(), stacked_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("`scan` carry count 3 exceeds the body input count 2")),
        );
        assert_eq!(
            TestScanOperation::new(3, 3)
                .infer_region_input_types(&[scalar_type.clone(), stacked_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("`scan` carry count 3 exceeds the body input count 2")),
        );
        let single_output_body = RegionInterface::new(
            vec![index_type.clone(), scalar_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            TestScanOperation::new(2, 3)
                .infer_output_types(&[scalar_type.clone(), scalar_type.clone()], &[single_output_body]),
            Err(TypeError::invalid("`scan` carry count 2 exceeds the body output count 1")),
        );
        let mismatched_carry_body = RegionInterface::new(
            vec![index_type.clone(), scalar_type.clone()],
            vec![ArrayType::scalar(DataType::Boolean)],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), &[mismatched_carry_body]),
            Err(TypeError::invalid("`scan` body carry type signature mismatch: expected [f64[]] but got [bool[]]")),
        );
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        let dynamic_body = RegionInterface::new(
            vec![index_type, dynamic_type.clone()],
            vec![dynamic_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&dynamic_type), &[dynamic_body]),
            Err(TypeError::invalid(
                "`scan` body input 1 must have a fully static type but axis 0 of `f64[dynamic]` has size `dynamic`",
            )),
        );

        // The body index must be a scalar `i64`, and a homogeneous array scan requires a static length.
        let mistyped_index_body = RegionInterface::new(
            vec![ArrayType::scalar(DataType::I32), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), &[mistyped_index_body]),
            Err(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(
            TestScanOperation::new(1, Dimension::Dynamic(length))
                .infer_output_types(&[scalar_type, stacked_type], interfaces.as_slice()),
            Err(TypeError::invalid(
                "homogeneous array `scan` requires a static length but got `length`; use a composite `scan` with a \
                 trailing first-class dimension input for a dynamic trip count",
            )),
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_scan_type_inference_rejects_unrepresentable_length() {
        // A carry-only scan has no stacked array whose shape would otherwise expose an oversized length, so both type
        // inference and eager interpretation reject the length itself before narrowing the counter to `i64`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let body = RegionInterface::new(
            vec![ArrayType::scalar(DataType::I64), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        let operation = TestScanOperation::new(1, MAX_DIMENSION_EXTENT + 1);
        let expected = TypeError::invalid(format!(
            "`scan` length {} exceeds the maximum supported extent {MAX_DIMENSION_EXTENT}",
            MAX_DIMENSION_EXTENT + 1,
        ));
        assert_eq!(operation.infer_output_types(&[scalar_type], &[body]), Err(expected.clone()));
        assert_eq!(
            operation.interpret(&TestEagerContext::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::from(expected)),
        );
    }

    #[test]
    fn test_scan_type_inference_validates_the_declared_body_before_specialization() {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let carry_type = ArrayType::new(DataType::F64, Shape::new(vec![rows.into()]));
        let concrete_type = ArrayType::new_static(DataType::F64, [2]);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::Boolean).into());
        let carry = body.add_input(carry_type.clone().into());
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(1, 1)
                .infer_region_input_types(&[concrete_type.clone().into()], &[body.entry_region_ref().interface()],),
            Err(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );
        assert_eq!(
            TracingContext::<TestIrValue, TestIrOperation>::trace(
                |inputs: Vec<Tracer<TracingContext<TestIrValue, TestIrOperation>>>| {
                    inputs[0].domain().bind(ScanOperation::<ArrayIrType>::new(1, 1), vec![body], &inputs)
                },
                vec![ArrayIrType::Array(concrete_type.clone())],
            )
            .unwrap_err(),
            ProgramError::from(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );

        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        body.add_input(carry_type.into());
        let replacement = body.add_constant(array(Array::vector(vec![3.0f64, 4.0]).unwrap()));
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![replacement], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(1, 1)
                .infer_region_input_types(&[concrete_type.into()], &[body.entry_region_ref().interface()],),
            Err(
                TypeError::invalid("`scan` body carry type signature mismatch: expected [f64[rows]] but got [f64[2]]",)
            ),
        );
    }

    #[test]
    fn test_scan_type_inference_refines_input_types() {
        // Scan input validation compares the declared types derived from the body signature against the actual input
        // types with `Type::is_refined_by`, so actual types carrying optional metadata that the declared types leave
        // unspecified (e.g., the normalized shardings that every concrete backend array type carries) are accepted,
        // while data type and shape mismatches are still rejected. Slicing a stacked input along its scan axis follows
        // the axis types of the scan axis's sharding.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let operation = TestScanOperation::new(1, 3);
        let interfaces = vec![product_body().interface()];
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_carry_type = scalar_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let sharded_stacked_type = stacked_type
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();

        // The inferred output types stay declared (i.e., metadata-free) rather than inheriting the input shardings,
        // while the body is requested at the refined input types so that the programs derived from it carry the
        // shardings.
        assert_eq!(
            operation.infer_output_types(
                &[sharded_carry_type.clone(), sharded_stacked_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![scalar_type.clone(), stacked_type.clone()]),
        );
        // The scan axis of the stacked input is sharded over an `Auto` mesh axis, which is only a placement hint, so
        // each per-iteration slice drops it.
        assert_eq!(
            operation.infer_region_input_types(
                &[sharded_carry_type.clone(), sharded_stacked_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![Some(vec![
                ArrayType::scalar(DataType::I64),
                sharded_carry_type.clone(),
                scalar_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap(),
            ])]),
        );

        // A scan axis sharded over a `Manual` mesh axis makes the slices vary over that axis, while one sharded over an
        // `Explicit` mesh axis is rejected consistently by both type inference functions.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 3, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual_stacked_type = stacked_type
            .clone()
            .with_sharding(Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap())
            .unwrap();
        let manual_carry_type =
            scalar_type.clone().with_sharding(Sharding::replicated(manual_mesh.clone(), 0)).unwrap();
        let region_input_types = operation
            .infer_region_input_types(&[manual_carry_type.clone(), manual_stacked_type.clone()], interfaces.as_slice())
            .unwrap();
        let Some(region_input_types) = &region_input_types[0] else {
            panic!("expected the body to be requested at the refined input types");
        };
        assert_eq!(
            region_input_types[2].sharding().unwrap().varying_manual_axes().iter().collect::<Vec<_>>(),
            vec!["m"],
        );
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("e", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_stacked_type = stacked_type
            .clone()
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["e"])]).unwrap())
            .unwrap();
        let explicit_carry_type = scalar_type.clone().with_sharding(Sharding::replicated(explicit_mesh, 0)).unwrap();
        let explicit_inputs = [explicit_carry_type, explicit_stacked_type];
        let region_error = operation.infer_region_input_types(&explicit_inputs, interfaces.as_slice()).unwrap_err();
        let output_error = operation.infer_output_types(&explicit_inputs, interfaces.as_slice()).unwrap_err();
        assert_eq!(region_error, output_error);
        assert_eq!(
            region_error,
            TypeError::invalid(
                "`scan` cannot slice a stacked value of type `f64[3][sharding={mesh<['e'=3:explicit]>, \
                 [{'e'}]}]` along \
                 its scan axis (cannot remove dimension 0 because it is sharded over the non-manual mesh axis `e`); \
                 reshard it so that its scan axis is not sharded over explicit mesh axes",
            ),
        );

        // Data type and shape mismatches are still rejected with the declared-versus-actual framing.
        assert_eq!(
            operation
                .infer_output_types(&[ArrayType::scalar(DataType::F32), stacked_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("`scan` input 0 has type `f32[]`, which does not refine its expected type `f64[]`")),
        );
        assert_eq!(
            operation
                .infer_output_types(&[scalar_type, ArrayType::new_static(DataType::F64, [4])], interfaces.as_slice(),),
            Err(TypeError::invalid(
                "`scan` input 1 has type `f64[4]`, which does not refine its expected type `f64[3]`"
            )),
        );
    }

    #[test]
    fn test_scan_type_inference_staging_falls_back_to_declared_regions_when_specialization_drops_a_refinement() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // The body replaces its carry with an unsharded constant, so the body specialized at a sharded carry no longer
        // maps its carry to itself. Staging falls back to the declared body, and the carry keeps its declared type.
        let carry_type = ArrayType::scalar(DataType::F32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = carry_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        body_builder.add_input(carry_type.clone().into());
        let replacement = body_builder.add_constant(TestIrValue::Array(Array::scalar(1f32).unwrap()));
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![replacement], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let (output_type, _) = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                Ok(context
                    .bind(ScanOperation::<ArrayIrType>::new(1, 3), vec![body.clone()], std::slice::from_ref(&input))?
                    .remove(0))
            },
            ArrayIrType::Array(sharded_type),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(carry_type));
    }

    #[test]
    fn test_scan_type_inference_refines_carries_to_the_extents_their_inputs_establish() {
        // Initial carries only need to refine the carry types, and the carry outputs take the static extents that the
        // inputs establish for identities that no output defines.
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let carry_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])));
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let operation = ScanOperation::<ArrayIrType>::new(1, 2);
        let interfaces = vec![RegionInterface::new(
            vec![index_type.clone(), carry_type.clone()],
            vec![carry_type.clone()],
            EffectClasses::NONE,
        )];
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&static_type), interfaces.as_slice()),
            Ok(vec![static_type.clone()]),
        );

        // Carries that share an identity share its established extent, even when one of their inputs is dynamic.
        let operation = ScanOperation::<ArrayIrType>::new(2, 2);
        let interfaces = vec![RegionInterface::new(
            vec![index_type.clone(), carry_type.clone(), carry_type.clone()],
            vec![carry_type.clone(), carry_type.clone()],
            EffectClasses::NONE,
        )];
        assert_eq!(
            operation.infer_output_types(&[carry_type.clone(), static_type.clone()], interfaces.as_slice()),
            Ok(vec![static_type.clone(), static_type.clone()]),
        );

        // A reference carry must equal its carry type exactly, because its allocation keeps its type when references
        // are discharged.
        let ArrayIrType::Array(referent_type) = &carry_type else { unreachable!() };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let operation = ScanOperation::<ArrayIrType>::new(1, 2);
        let interfaces = vec![RegionInterface::new(
            vec![index_type.clone(), reference_type.clone()],
            vec![reference_type],
            EffectClasses::NONE,
        )];
        let ArrayIrType::Array(static_referent_type) = static_type else { unreachable!() };
        assert_eq!(
            operation.infer_output_types(
                &[ArrayIrType::Reference(ReferenceType::new(static_referent_type))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`scan` input 0 has type `ref<f32[3]>`, which does not equal its expected type `ref<f32[rows]>`",
            )),
        );

        // When a first-class dimension carry shares the identity of the scan length, the carries that refer to it stay
        // symbolic because that carry may change across iterations, while the stacked output's leading axis still
        // takes the trip count that the exact runtime length fixes before the first iteration.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension_carry_type = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let array_carry_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let operation = ScanOperation::<ArrayIrType>::new(2, Dimension::Dynamic(length));
        let interfaces = vec![RegionInterface::new(
            vec![index_type, dimension_carry_type.clone(), array_carry_type.clone()],
            vec![
                dimension_carry_type.clone(),
                array_carry_type.clone(),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
            ],
            EffectClasses::NONE,
        )];
        let four = DimensionType::new("four", DimensionBounds::new(4, Some(5)).unwrap());
        assert_eq!(
            operation.infer_output_types(
                &[
                    dimension_carry_type.clone(),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
                    four.into(),
                ],
                interfaces.as_slice(),
            ),
            Ok(vec![
                dimension_carry_type,
                array_carry_type,
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            ]),
        );
    }

    #[test]
    fn test_scan_type_inference_composite() {
        // A composite scan admits first-class dimension carries. The body is requested at the instantiated input types
        // exactly when the actual inputs carry type identities other than the declared ones.
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let slice_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(extent)]),
        ));
        let body_interface = RegionInterface::new(
            vec![index_type.clone(), dimension_type.clone(), slice_type.clone()],
            vec![dimension_type.clone(), slice_type],
            EffectClasses::NONE,
        );
        let operation = ScanOperation::<ArrayIrType>::new(1, 3);
        let input_types = vec![dimension_type.clone(), stacked_type.clone()];
        assert_eq!(
            operation.infer_region_input_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Ok(vec![None]),
        );
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(3, 3)
                .infer_region_input_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Err(TypeError::invalid("`scan` carry count 3 exceeds the body input count 2")),
        );
        assert_eq!(
            operation.infer_output_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Ok(vec![dimension_type.clone(), stacked_type]),
        );
        let other = DimensionVariable::new("other", DimensionBounds::positive(Some(8)).unwrap());
        let other_dimension_type = ArrayIrType::Dimension(DimensionType::from(other.clone()));
        let other_slice_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(other.clone())])));
        let other_stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(other)]),
        ));
        assert_eq!(
            operation.infer_region_input_types(
                &[other_dimension_type.clone(), other_stacked_type],
                std::slice::from_ref(&body_interface),
            ),
            Ok(vec![Some(vec![index_type.clone(), other_dimension_type, other_slice_type])]),
        );

        // A first-class dimension cannot be stacked.
        let invalid_body_interface = RegionInterface::new(
            vec![index_type.clone(), dimension_type.clone(), dimension_type.clone()],
            vec![dimension_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(&[dimension_type.clone(), dimension_type], &[invalid_body_interface]),
            Err(TypeError::invalid(
                "`scan` stacked body input 2 must be an array but got `dimension<extent ∈ [1, 8)>`"
            )),
        );

        // A fresh dimension produced by the body cannot replace the declared carry identity on each iteration.
        // Supporting such shape-varying state would require an explicit widening contract, so `scan` keeps its
        // loop-carried type invariant and reports the exact incompatible signatures.
        let carry = DimensionVariable::new("carry", DimensionBounds::positive(Some(8)).unwrap());
        let next = DimensionVariable::new("next", DimensionBounds::positive(Some(8)).unwrap());
        let carry_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(carry)])));
        let next_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(next)])));
        let shape_varying_body =
            RegionInterface::new(vec![index_type, carry_type.clone()], vec![next_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&carry_type), &[shape_varying_body]),
            Err(TypeError::invalid(
                "`scan` body carry type signature mismatch: expected [f32[carry]] but got [f32[next]]"
            )),
        );
    }

    #[test]
    fn test_scan_type_inference_composite_runtime_length() {
        // A dynamic runtime-length input must carry the scan length's nominal identity: an unrelated dynamic identity
        // with compatible bounds cannot redefine the stacked axis. A runtime length input whose bounds pin one exact
        // extent fixes the trip count to that extent and is admissible only when every stacked input is refined to the
        // same extent, because a stacked axis left symbolic would otherwise be read a fixed number of times regardless
        // of its independently determined runtime size. The accepted refinement also types the stacked outputs at the
        // concrete extent instead of at the still-symbolic declared length.
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let slice_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let body_interface = RegionInterface::new(
            vec![index_type, dimension_type.clone(), slice_type.clone()],
            vec![dimension_type.clone(), slice_type],
            EffectClasses::NONE,
        );
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(5)).unwrap());
        let unrelated = DimensionVariable::new("unrelated", DimensionBounds::positive(Some(5)).unwrap());
        let three = DimensionType::new("three", DimensionBounds::new(3, Some(4)).unwrap());
        let symbolic_stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(length.clone()), Dimension::Dynamic(extent.clone())]),
        ));
        let static_stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(extent)]),
        ));
        let operation = ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone()));
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type.clone(), symbolic_stacked_type.clone(), DimensionType::from(length).into()],
                std::slice::from_ref(&body_interface),
            ),
            Ok(vec![dimension_type.clone(), symbolic_stacked_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type.clone(), symbolic_stacked_type.clone(), DimensionType::from(unrelated).into()],
                std::slice::from_ref(&body_interface),
            ),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<unrelated ∈ [1, 5)>` but `scan` length \
                 requires `length`",
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type.clone(), symbolic_stacked_type, three.clone().into()],
                std::slice::from_ref(&body_interface),
            ),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<3>` but stacked input 1 has type \
                 `f32[length, extent]` \
                 whose leading dimension is not refined to extent 3",
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type.clone(), static_stacked_type.clone(), three.into()],
                &[body_interface],
            ),
            Ok(vec![dimension_type, static_stacked_type]),
        );
    }

    #[test]
    fn test_scan_type_inference_rejects_static_stacks_with_a_nominal_dynamic_length() {
        // Matching the runtime length's identity does not prove that a statically refined stack has that extent.
        // Reject before specializing or staging the body, including reference stacks and zero-output bodies.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let runtime_length = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let stack = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let interface = RegionInterface::new(
            vec![ArrayType::scalar(DataType::I64).into(), ArrayType::scalar(DataType::F32).into()],
            vec![],
            EffectClasses::NONE,
        );
        let operation = ScanOperation::<ArrayIrType>::new(0, Dimension::Dynamic(length.clone()));
        let expected = TypeError::invalid(
            "`scan` runtime length input has type `dimension<length ∈ [0, 8)>` but stacked input 0 has type \
             `f32[3]` whose leading dimension is not provably equal to `length`",
        );
        assert_eq!(
            operation.infer_output_types(&[stack.clone(), runtime_length.clone()], std::slice::from_ref(&interface)),
            Err(expected.clone()),
        );
        assert_eq!(
            operation.infer_region_input_types(&[stack, runtime_length.clone()], std::slice::from_ref(&interface)),
            Err(expected),
        );
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let reference_interface = RegionInterface::new(
            vec![ArrayType::scalar(DataType::I64).into(), reference.clone()],
            vec![],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(&[reference, runtime_length], &[reference_interface]),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<length ∈ [0, 8)>` but stacked input 0 has type \
                 `ref<f32[3]>` whose leading dimension is not provably equal to `length`",
            )),
        );

        // Singleton bounds prove equality even when the runtime input keeps its nominal identity.
        let exact_length = DimensionVariable::new("exact_length", DimensionBounds::new(3, Some(4)).unwrap());
        let operation = ScanOperation::<ArrayIrType>::new(0, Dimension::Dynamic(exact_length.clone()));
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new_static(DataType::F32, [3]).into(), DimensionType::from(exact_length).into(),],
                &[interface],
            ),
            Ok(vec![]),
        );
    }

    #[test]
    fn test_scan_type_inference_composite_stacked_references() {
        // A composite scan admits stacked reference inputs, which enter the body as whole roots whose leading axis is
        // the scan axis.
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        let slice_type = ArrayType::new_static(DataType::F32, [2]);
        let stacked_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let slice_reference = ArrayIrType::Reference(ReferenceType::new(slice_type.clone()));
        let stacked_reference = ArrayIrType::Reference(ReferenceType::new(stacked_type.clone()));
        let length = Dimension::Static(3);
        let operation = ScanOperation::<ArrayIrType>::new(0, 3);

        // References enter as whole roots after the intrinsic index, while array outputs still stack their slice type.
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(std::slice::from_ref(&stacked_reference), 2, 0, &length),
            Ok(vec![index_type.clone(), stacked_reference.clone()]),
        );
        let body_interface = RegionInterface::new(
            vec![index_type.clone(), stacked_reference.clone()],
            vec![ArrayIrType::Array(slice_type.clone())],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&stacked_reference),
                std::slice::from_ref(&body_interface),
            ),
            Ok(vec![None]),
        );
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&stacked_reference), std::slice::from_ref(&body_interface)),
            Ok(vec![ArrayIrType::Array(stacked_type.clone())]),
        );

        // The stacked referent must be a stack over the scan length, and a first-class dimension is neither an array
        // nor a reference.
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(
                &[ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 2])))],
                2,
                0,
                &length,
            ),
            Err(TypeError::invalid(
                "`scan` stacked input 0 must have leading dimension `3` but has type `ref<f32[4, 2]>`"
            )),
        );
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(
                &[ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)))],
                2,
                0,
                &length,
            ),
            Err(TypeError::invalid("`scan` stacked input 0 must have rank at least 1")),
        );
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(&[DimensionType::from(extent.clone()).into()], 2, 0, &length),
            Err(TypeError::invalid(
                "`scan` stacked input 0 must be an array or a reference but got `dimension<extent ∈ [1, 8)>`",
            )),
        );

        // The per-iteration view is an in-bounds static slice of the referent, so a dynamically shaped referent is
        // rejected even though a stacked array of the same shape would be admitted.
        let dynamic_reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent.clone())]),
        )));
        let dynamic_interface = RegionInterface::new(
            vec![index_type.clone(), dynamic_reference],
            vec![ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])))],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&stacked_reference), &[dynamic_interface]),
            Err(TypeError::invalid(
                "`scan` body input 1 must have a fully static type but axis 0 of `f32[extent]` has size `extent`",
            )),
        );

        // A body cannot return a per-iteration reference view, because no stacked reference value exists that the scan
        // could assemble from it.
        let returning_interface = RegionInterface::new(
            vec![index_type.clone(), stacked_reference.clone()],
            vec![slice_reference],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&stacked_reference), &[returning_interface]),
            Err(TypeError::invalid(
                "`scan` stacked body output 0 must be an array but got `ref<f32[2]>`; a body cannot return a \
                 per-iteration reference view",
            )),
        );

        // A runtime length input that pins one exact extent applies the same refinement rule to stacked references as
        // to stacked arrays.
        let dynamic_length = DimensionVariable::new("length", DimensionBounds::positive(Some(5)).unwrap());
        let three = DimensionType::new("three", DimensionBounds::new(3, Some(4)).unwrap());
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(
                &[stacked_reference.clone(), three.into()],
                2,
                0,
                &Dimension::Dynamic(dynamic_length.clone()),
            ),
            Ok(vec![index_type, stacked_reference]),
        );
        let symbolic_reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(dynamic_length.clone()), Dimension::Static(2)]),
        )));
        let four = DimensionType::new("four", DimensionBounds::new(4, Some(5)).unwrap());
        assert_eq!(
            ArrayIrType::infer_scan_body_input_types(
                &[symbolic_reference, four.into()],
                2,
                0,
                &Dimension::Dynamic(dynamic_length),
            ),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<4>` but stacked input 0 has type \
                 `ref<f32[length, 2]>` \
                 whose leading dimension is not refined to extent 4",
            )),
        );
    }

    #[test]
    fn test_scan_boundary_pruning() {
        // The body maps `[sum, square, unused, factor, angle]` to
        // `[sum + square, square * square, unused * unused, sum * factor, sin(angle)]`. Using only the final `sum` and
        // the first stacked output keeps the carry `square` too, because the next `sum` depends on it, while the carry
        // `unused`, the stacked input `angle`, and the second stacked output are dropped.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let body = scalar_body(5, |builder, inputs| {
            let [_, sum, square, unused, factor, angle]: [AtomId; 6] = inputs.try_into().unwrap();
            let next_sum =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, square], None).unwrap()[0];
            let next_square =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![square, square], None).unwrap()[0];
            let next_unused =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![unused, unused], None).unwrap()[0];
            let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sum, factor], None).unwrap()[0];
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![angle], None).unwrap()[0];
            vec![next_sum, next_square, next_unused, scaled, sine]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let carries = [(); 3].map(|_| builder.add_input(scalar_type.clone()));
        let stacked_inputs = [(); 2].map(|_| builder.add_input(stacked_type.clone()));
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(
                TestScanOperation::new(3, 3).with_reverse(true).with_unroll(2).unwrap(),
                vec![body],
                carries.iter().chain(&stacked_inputs).copied().collect(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0], outputs[3]], vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[3], %4:f64[3] .
                let %5:f64[], %6:f64[], %7:f64[3] = scan [carry_count=2, length=3, reverse=true, unroll=2] %0 %1 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = add %1 %2
                            %5:f64[] = mul %2 %2
                            %6:f64[] = mul %1 %3
                        in (%4, %5, %6)
                    },
                ]
                in (%5, %7)
            "}
            .trim_end(),
        );
        let inputs = vec![
            Array::scalar(1.0f64).unwrap(),
            Array::scalar(0.5f64).unwrap(),
            Array::scalar(2.0f64).unwrap(),
            Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap(),
            Array::vector(vec![4.0f64, 5.0, 6.0]).unwrap(),
        ];
        assert_eq!(pruned.interpret(inputs.clone()), program.interpret(inputs));
    }

    #[test]
    fn test_scan_boundary_pruning_zero_length() {
        // Pruning a zero-length scan follows the same liveness through its body even though no iteration executes:
        // the unused carry and stacked output are dropped, and the kept carry returns its initial value.
        let body = scalar_body(3, |builder, inputs| {
            let [_, sum, unused, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let next_sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, value], None).unwrap()[0];
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, value], None).unwrap()[0];
            vec![next_sum, unused, doubled]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let sum = builder.add_input(ArrayType::scalar(DataType::F64));
        let unused = builder.add_input(ArrayType::scalar(DataType::F64));
        let values = builder.add_input(ArrayType::new_static(DataType::F64, [0]));
        let final_sum = builder
            .add_instruction(TestScanOperation::new(2, 0), vec![body], vec![sum, unused, values], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![final_sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[0] .
                let %3:f64[] = scan [carry_count=1, length=0, reverse=false] %0 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        let inputs = vec![
            Array::scalar(1.0f64).unwrap(),
            Array::scalar(2.0f64).unwrap(),
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap(),
        ];
        assert_eq!(pruned.interpret(inputs.clone()), Ok(vec![Array::scalar(1.0f64).unwrap()]));
        assert_eq!(program.interpret(inputs), Ok(vec![Array::scalar(1.0f64).unwrap()]));
    }

    #[test]
    fn test_scan_boundary_pruning_keeps_the_runtime_length_input() {
        // A dynamic-length scan always keeps its trailing runtime length input, which is not a body input, while its
        // unused carry and stacked output are pruned as usual.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let stacked_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let sum = body.add_input(scalar_type.clone());
        let unused = body.add_input(scalar_type.clone());
        let value = body.add_input(scalar_type.clone());
        let next_sum = body
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![sum, value], None)
            .unwrap()[0];
        let doubled = body
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![value, value], None)
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_sum, unused, doubled],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let sum = builder.add_input(scalar_type.clone());
        let unused = builder.add_input(scalar_type);
        let values = builder.add_input(stacked_type);
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(2, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![sum, unused, values, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[0]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[length], %3:dimension<length ∈ [1, 8)> .
                let %4:f32[] = scan [carry_count=1, length=length, reverse=false] %0 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        let inputs = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length), 3).unwrap()),
        ];
        assert_eq!(pruned.interpret(inputs.clone()), Ok(vec![TestIrValue::Array(Array::scalar(7.0f32).unwrap())]));
        assert_eq!(program.interpret(inputs), Ok(vec![TestIrValue::Array(Array::scalar(7.0f32).unwrap())]));
    }

    #[test]
    fn test_scan_boundary_pruning_rejects_an_invalid_runtime_length_boundary() {
        /// Liveness of a body containing only its unused intrinsic index and no outputs.
        struct EmptyBody;

        impl RegionLiveness for EmptyBody {
            fn used_region_inputs(
                &mut self,
                region_index: usize,
                used_outputs: &[bool],
            ) -> Result<Vec<bool>, ProgramError> {
                assert_eq!(region_index, 0);
                assert_eq!(used_outputs, &[] as &[bool]);
                Ok(vec![false])
            }
        }

        // Static scans have no trailing operands; dynamic scans require exactly one runtime length even when the
        // body and the public output boundary are empty.
        assert_eq!(
            TestScanOperation::new(0, 3).prune_boundary(1, &[], &mut EmptyBody),
            Err(
                ProgramError::MalformedProgram("`scan` expects 0 trailing runtime length inputs but has 1".to_owned(),)
            ),
        );
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(0, Dimension::Dynamic(length)).prune_boundary(0, &[], &mut EmptyBody),
            Err(
                ProgramError::MalformedProgram("`scan` expects 1 trailing runtime length inputs but has 0".to_owned(),)
            ),
        );
    }

    #[test]
    fn test_scan_rename_type_identities() {
        // Renaming rewrites the identity of a dynamic length and preserves every other field, while a static length
        // has no identity to rename.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let renamed = DimensionVariable::new("renamed", DimensionBounds::positive(Some(8)).unwrap());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(length.clone(), renamed.clone()).unwrap();
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(2, Dimension::Dynamic(length))
                .with_reverse(true)
                .with_unroll(3)
                .unwrap()
                .rename_type_identities(&renaming),
            Ok(ScanOperation::<ArrayIrType>::new(2, Dimension::Dynamic(renamed))
                .with_reverse(true)
                .with_unroll(3)
                .unwrap()),
        );
        let operation = ScanOperation::<ArrayIrType>::new(1, 3);
        assert_eq!(operation.rename_type_identities(&renaming), Ok(operation.clone()));
    }

    #[test]
    fn test_scan_reference_discharge() {
        // Reference discharge replaces the reference state that a scan threads with ordinary array carries.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let reference_type = ReferenceType::new(scalar_type.clone());

        // A scan carries reference state in its leading carry prefix rather than after its declared inputs. The body
        // declares one ordinary carry, one reference carry, and one per-iteration slice of a stacked input, and the
        // rewrite keeps that split intact: the state joins the carry prefix on the parent input list, on the body's
        // input boundary, and on the body's output boundary, while the stacked input and the stacked output stay behind
        // the prefix on their own side.
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let reference = body_builder.add_input(reference_type.into());
        let element = body_builder.add_input(scalar_type.clone().into());
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element], None)
            .unwrap();
        let current = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, element], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_carry, reference, current],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();

        // The rewritten payload keeps every attribute that is not about state: a reversed, fully unrolled scan of the
        // same length stays exactly that, and only its carry count would move if an allocation reached the body without
        // being one of its declared carries.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial_carry = builder.add_input(scalar_type.clone().into());
        let initial_state = builder.add_input(scalar_type.into());
        let elements = builder.add_input(stacked_type.into());
        let reference = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial_state], None)
            .unwrap()[0];
        let operation = ScanOperation::<ArrayIrType>::new(2, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs = builder
            .add_instruction(operation, vec![body], vec![initial_carry, reference, elements], None)
            .unwrap();
        let final_carry = outputs[0];
        let final_reference = outputs[1];
        let stacked = outputs[2];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_carry, stacked, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let inputs = vec![
            TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = source.clone().interpret(inputs.clone()).unwrap();
        assert_eq!(
            expected,
            vec![
                TestIrValue::Array(Array::scalar(6.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![16.0f32, 15.0, 13.0]).unwrap()),
                TestIrValue::Array(Array::scalar(16.0f32).unwrap()),
            ],
        );

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3] .
                let %3:f32[], %4:f32[], %5:f32[3] = scan [carry_count=2, length=3, reverse=true, unroll=3] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[], %3:f32[] .
                        let %4:f32[] = add %2 %3
                            %5:f32[] = add %1 %3
                        in (%5, %4, %4)
                    },
                ]
                in (%3, %5, %4)"},
        );
        let TestIrOperation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.length(), &Dimension::Static(3));
        assert!(scan.reverse());
        assert_eq!(scan.unroll(), 3);

        // The allocation is local to the program, so nothing crosses the entry boundary and the rewritten program must
        // reproduce the eager reference execution output for output.
        assert_eq!(discharged.output_count(), 3);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_threads_a_preserved_carry_through_a_scan() {
        // A scan inserts its synthesized state carries immediately after the declared carry prefix, and a preserved
        // carry stays a declared carry: it keeps its position and its reference type on both boundaries.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
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
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![total, step, observed],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let total_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let step_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let total = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![total_initial], None)
            .unwrap()[0];
        let step =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![step_initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 3), vec![body], vec![total, step], None)
            .unwrap();
        let (carried_total, carried_step, stacked) = (outputs[0], outputs[1], outputs[2]);
        let final_total = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_total], None)
            .unwrap()[0];
        let final_step = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_step], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_total, stacked, final_step],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.output_count(), 3);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[], %4:ref<f32[]>, %5:f32[3] = scan [carry_count=2, length=3, reverse=false] %0 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[] = reference_read %2
                                %4:f32[] = add %1 %3
                            in (%4, %2, %4)
                        },
                    ]
                    %6:f32[] = reference_freeze %2
                in (%3, %5, %6)"},
        );

        let inputs = vec![
            TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ];
        let outputs = vec![
            TestIrValue::Array(Array::scalar(6.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![2.0f32, 4.0, 6.0]).unwrap()),
            TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ];
        assert_eq!(source.interpret(inputs.clone()), Ok(outputs.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
    }

    #[test]
    fn test_scan_reference_discharge_keeps_state_carries_separate_from_stacked_outputs() {
        // A body that accumulates into its reference carry and stacks the state it observes keeps the discharged state
        // carry in the declared carry prefix, separate from the stacked output, and the rewrite preserves the visit
        // order and unroll factor of the source scan.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, value],
                vec![Placeholder, Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let operation = ScanOperation::<ArrayIrType>::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs = builder.add_instruction(operation, vec![body], vec![reference], None).unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
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
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %1 %2
                        in (%3, %3)
                    },
                ]
                in (%1, %2, %1)"},
        );
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let TestIrOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(3));
        assert!(scan.reverse());
        assert_eq!(scan.unroll(), 3);
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar(2.0f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![5.0f32, 4.0, 3.0]).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_appends_the_synthesized_carry_after_the_declared_carry_prefix() {
        // A scan that already declares an ordinary carry pins the synthesized-state placement exactly: the state input
        // joins the carry prefix behind every declared carry and ahead of the trailing stacked inputs, on the parent
        // input list, the body boundary, and the rewritten carry count alike.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let element = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference = body_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element], None)
            .unwrap();
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, element], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![next_carry],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, elements], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![final_carry],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let closed = ClosedProgram::new(
            program,
            vec![TestIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap()))],
        )
        .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3] .
                let %3:f32[], %4:f32[] = scan [carry_count=2, length=3, reverse=false] %1 %0 %2 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[], %3:f32[] .
                        let %4:f32[] = add %2 %3
                            %5:f32[] = add %1 %3
                        in (%5, %4)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(
            discharged.external_reference_bindings()[0].source().flat_input_index(discharged.capture_count()),
            Ok(0)
        );
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        let DischargeCaptureOperation::Scan(scan) =
            discharged.program().entry_region_ref().instructions()[0].operation()
        else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.length(), &Dimension::Static(3));
    }

    #[test]
    fn test_scan_reference_discharge_threads_reference_captures_through_scan() {
        // A capture read by a scan body becomes a synthesized carry appended after the declared carry prefix, which
        // raises the rewritten scan's carry count without disturbing its length, direction, or unroll factor.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32).unwrap());
        let mut body_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 2), vec![body], Vec::new(), None)
            .unwrap()[0];
        let scan_program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![values], Vec::new(), vec![Placeholder])
            .unwrap();
        let closed = ClosedProgram::new(scan_program, vec![TestIrValue::Reference(concrete_reference)]).unwrap();
        assert_eq!(
            closed.program().to_string(),
            indoc! {"
                lambda  .
                let %0:f32[2] = scan [carry_count=0, length=2, reverse=false] [
                    body={
                        lambda %0:i64[] .
                        let %1:ref<f32[]> = const capture#0:ref<f32[]>
                            %2:f32[] = reference_read %1
                        in (%2)
                    },
                ]
                in (%0)"},
        );
        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[2] = scan [carry_count=1, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        in (%1, %1)
                    },
                ]
                in (%2)"},
        );
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(discharged.program().output_types(), vec![ArrayType::new_static(DataType::F32, [2]).into()]);
        let scan = discharged.program().entry_region_ref().instructions()[0].operation();
        let DischargeCaptureOperation::Scan(scan) = scan else {
            panic!("expected discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(2));
        assert!(!scan.reverse());
        assert_eq!(scan.unroll(), 1);
    }

    #[test]
    fn test_scan_reference_discharge_threads_mutated_reference_capture_through_scan() {
        // A capture that a scan body accumulates into reaches that body only through a synthesized carry, which is the
        // most involved discharge path: the state enters the scan appended after the declared carry prefix, is updated
        // inside the body, leaves through the matching synthesized carry output, and reaches the hidden entry
        // final-state output after the public prefix. The capture value family carries no data, so the rendered program
        // rather than an interpretation pins the resulting state flow.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        let update = body_builder.add_constant(DischargeCapture::new(1, ArrayType::scalar(DataType::F32).into()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let values = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 3), vec![body], Vec::new(), None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![values], Vec::new(), vec![Placeholder])
            .unwrap();
        let closed = ClosedProgram::new(
            program,
            vec![
                TestIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32).unwrap())),
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ],
        )
        .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(
            discharged.external_reference_bindings()[0].source().flat_input_index(discharged.capture_count()),
            Ok(0)
        );
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const capture#1:f32[]
                            %3:f32[] = add %1 %2
                        in (%3, %3)
                    },
                ]
                in (%3, %2)"},
        );
    }

    #[test]
    fn test_scan_reference_discharge_captures_a_lazy_view_root_and_dynamic_binding() {
        // A body traced as a nested region reads a captured root through a lazy view. The view is not a program
        // value, so the region captures only its root and dynamic index, and the access re-applies its transforms.
        let context = TracingContext::<DischargeCapture, TestIrOperation, TestIrValue>::new();
        let stack = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(2i32).unwrap());
        let (_, body) = NestedTracingContext::trace(
            context.clone(),
            |inputs: Vec<Tracer<_>>| {
                let body = inputs[0].context().clone();
                let root = StagingContext::constant(&body, body.capture(stack.clone())?);
                let index = StagingContext::constant(&body, body.capture(index.clone())?);
                ReferenceView::new(root)?.dynamic_index(0, &index)?.read()
            },
            vec![ArrayType::scalar(DataType::I64).into()],
        )
        .unwrap();
        assert_eq!(body.instructions().len(), 1);
        let access = &body.instructions()[0];
        assert_eq!(access.inputs().len(), 2);
        assert_eq!(access.operation().reference_access_descriptor(0).unwrap().bindings(), 1..2);
        assert_eq!(
            access.operation().reference_access_descriptor(0).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
        );
        let output = context.bind(ScanOperation::<ArrayIrType>::new(0, 2), vec![body], &[]).unwrap().remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        let captures = context.captures().borrow().clone();
        assert_eq!(captures, vec![stack, index.clone()]);

        // Discharge threads the captured root's state as an array input while the index capture stays an ordinary
        // value, and executing the result selects the element the captured index names on every iteration.
        let closed = ClosedProgram::new(program, captures.clone()).unwrap();
        assert_eq!(
            closed.program().to_string(),
            indoc! {"
                lambda  .
                let %0:f32[2] = scan [carry_count=0, length=2, reverse=false] [
                    body={
                        lambda %0:i64[] .
                        let %1:ref<f32[3]> = const capture#0:ref<f32[3]>
                            %2:i32[] = const capture#1:i32[]
                            %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %1 %2
                        in (%3)
                    },
                ]
                in (%0)"},
        );
        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:i32[] .
                let %2:f32[3], %3:f32[2] = scan [carry_count=1, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:i32[] = const capture#1:i32[]
                            %3:f32[1] = dynamic_slice [sizes=[1]] %1 %2
                            %4:f32[] = reshape [shape=[]] %3
                        in (%1, %4)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![ArrayType::new_static(DataType::F32, [3]).into(), ArrayType::scalar(DataType::I32).into()],
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(
            resolve_captures(discharged.program(), &captures).to_string(),
            indoc! {"
                lambda %0:f32[3], %1:i32[] .
                let %2:f32[3], %3:f32[2] = scan [carry_count=1, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:i32[] = const 2
                            %3:f32[1] = dynamic_slice [sizes=[1]] %1 %2
                            %4:f32[] = reshape [shape=[]] %3
                        in (%1, %4)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            resolve_captures(discharged.program(), &captures)
                .interpret(vec![TestIrValue::Array(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap()), index]),
            Ok(vec![TestIrValue::Array(Array::vector(vec![30.0f32, 30.0]).unwrap())]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_matches_eager_reference_execution() {
        // A scan whose body accumulates into a declared reference carry and stacks the observed state discharges into
        // an ordinary carry that agrees with eager reference execution on both the stacked per-iteration snapshots and
        // the state observed after the scan.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar(3.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, value],
                vec![Placeholder, Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 4), vec![body], vec![reference], None)
            .unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![frozen, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let eager = source.clone().interpret(vec![TestIrValue::Array(Array::scalar(0.0f32).unwrap())]).unwrap();
        assert_eq!(
            eager,
            vec![
                TestIrValue::Array(Array::scalar(12.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![3.0f32, 6.0, 9.0, 12.0]).unwrap()),
            ]
        );
        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                    %2:ref<f32[]>, %3:f32[4] = scan [carry_count=1, length=4, reverse=false] %1 [
                        body={
                            lambda %0:i64[], %1:ref<f32[]> .
                            let %2:f32[] = const 3.0
                                () = reference_add_update %1 %2
                                %3:f32[] = reference_read %1
                            in (%1, %3)
                        },
                    ]
                    %4:f32[] = reference_freeze %2
                in (%4, %3)"},
        );
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[4] = scan [carry_count=1, length=4, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const 3.0
                            %3:f32[] = add %1 %2
                        in (%3, %3)
                    },
                ]
                in (%1, %2)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(discharged.program().interpret(vec![TestIrValue::Array(Array::scalar(0.0f32).unwrap())]), Ok(eager));
    }

    #[test]
    fn test_scan_reference_discharge_matches_eager_reference_execution_for_stacked_reference() {
        // The body creates a dynamic view of its whole reference root using the explicit slice index. Discharge
        // must agree with eager execution on the carry, the stacked value output, and the allocation's final state,
        // which the rewrite publishes through an extra stacked output.
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [4])).into());
        let slice_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        body_builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(slice_transforms.clone()),
                Vec::new(),
                vec![stack, carry, index],
                None,
            )
            .unwrap();
        let current = body_builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(slice_transforms),
                Vec::new(),
                vec![stack, index],
                None,
            )
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![current, current], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![current, doubled],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [4]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 4), vec![body], vec![initial, stack], None)
            .unwrap();
        let (final_carry, doubled) = (outputs[0], outputs[1]);
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_carry, doubled, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let inputs = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
        ];
        let eager = source.clone().interpret(inputs.clone()).unwrap();
        assert_eq!(
            eager,
            vec![
                TestIrValue::Array(Array::scalar(11.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![4.0f32, 8.0, 14.0, 22.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 4.0, 7.0, 11.0]).unwrap()),
            ]
        );
        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[4] .
                let %2:ref<f32[4]> = reference_new %1
                    %3:f32[], %4:f32[4] = scan [carry_count=1, length=4, reverse=false] %0 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[4]> .
                            let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                                %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %4:f32[] = add %3 %3
                            in (%3, %4)
                        },
                    ]
                    %5:f32[4] = reference_freeze %2
                in (%3, %4, %5)"},
        );
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[4] .
                let %2:f32[], %3:f32[4], %4:f32[4] = scan [carry_count=1, length=4, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %2 %1
                            %4:f32[] = add %3 %3
                        in (%3, %4, %3)
                    },
                ]
                in (%2, %3, %4)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().output_types(),
            vec![
                ArrayType::scalar(DataType::F32).into(),
                ArrayType::new_static(DataType::F32, [4]).into(),
                ArrayType::new_static(DataType::F32, [4]).into(),
            ],
        );
        assert_eq!(discharged.program().interpret(inputs), Ok(eager));
    }

    #[test]
    fn test_scan_reference_discharge_replaces_stacked_reference_with_slices() {
        // The body only selects its current row of the stacked reference, so discharge rewrites the stack into an
        // ordinary stacked array input carrying the allocation's state and, because the body mutates its current row,
        // one stacked output holding the rows' final states, which becomes the allocation's successor state (refer to
        // `test_scan_reference_discharge_publishes_stacked_reference` for the boundary itself).
        let program = stacked_reference_program();
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %2 %1
                            %4:f32[] = add %1 %3
                        in (%4, %3)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        let inputs = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestIrValue::Array(Array::scalar(19.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_reads_stacked_reference_as_stacked_array() {
        // A stacked reference the body only reads through its per-iteration view becomes an ordinary stacked array
        // input carrying the allocation's state, and the scan gains no output for it: the frozen allocation is the
        // unchanged state that entered the scan.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let stack = body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let current = body_builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![stack, index],
                None,
            )
            .unwrap()[0];
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, current], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next_carry], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_carry, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%2, %1)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        let inputs = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestIrValue::Array(Array::scalar(7.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    // Mutated reference carries become array carries, while a mutated current-row view contributes one stacked output.
    // Reversed iteration preserves each updated row's original position in the stack.
    #[test]
    fn test_scan_reference_discharge_publishes_stacked_reference() {
        // The body reads the per-iteration view into a reference carry and writes the carry's new value back into the
        // view, so both the carry and the stacked allocation are mutated. The rewrite discharges the carry into the
        // carry prefix as usual and the stacked reference into a stacked array input plus one stacked output holding
        // the view's final states, which is the allocation's successor state (a reversed scan pins that the stacked
        // output is indexed by the iteration's position rather than by iteration order).
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let total = body_builder.add_input(ReferenceType::new(scalar_type.clone()).into());
        let stack = body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let current = body_builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(element_transforms.clone()),
                Vec::new(),
                vec![stack, index],
                None,
            )
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![total, current], None)
            .unwrap();
        let running =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![total], None).unwrap()[0];
        body_builder
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![stack, running, index],
                None,
            )
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![total], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let total = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let final_total = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, 3).with_reverse(true),
                vec![body],
                vec![total, stack],
                None,
            )
            .unwrap()[0];
        let frozen_total = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_total], None)
            .unwrap()[0];
        let frozen_stack =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![frozen_total, frozen_stack],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestIrValue::Array(Array::scalar(6.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![6.0f32, 5.0, 3.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(discharged.program().interpret(inputs.clone()), Ok(expected.clone()));

        // Under partial discharge that preserves the stacked allocation, the scan keeps its reference-typed stacked
        // input and the body replays its accesses through the per-iteration view, while the discharged carry still
        // joins the carry prefix as state.
        let targets = program.reference_discharge_targets(0).unwrap();
        assert_eq!(targets.len(), 2);
        let preserved = program.partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:ref<f32[3]> = reference_new %1
                    %3:f32[] = scan [carry_count=1, length=3, reverse=true] %0 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[3]> .
                            let %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %4:f32[] = add %1 %3
                                () = reference_write [transforms=[index(axis=0, index=dynamic)]] %2 %4 %0
                            in (%4)
                        },
                    ]
                    %4:f32[3] = reference_freeze %2
                in (%3, %4)"},
        );
        assert_eq!(preserved.program().interpret(inputs), Ok(expected));
    }

    // A zero-trip scan publishes unchanged reference state without constructing an invalid access to an empty root.
    #[test]
    fn test_scan_reference_discharge_publishes_zero_length_stacked_reference_unchanged() {
        // Mutation summaries are conservative: a body that writes its per-iteration view publishes the stacked
        // allocation's final state even for a zero-length scan, whose published state is then simply the entering one.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let stack = body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [0])).into());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        body_builder
            .add_instruction(
                ReferenceWriteOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![stack, carry, index],
                None,
            )
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [0]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 0), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_carry, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[0] .
                in (%0, %1)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(4.0f32).unwrap()),
                TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(4.0f32).unwrap()),
                TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_preserves_zero_length_state_identity() {
        // A zero-length scan never runs the body that would update its reference carry, so the discharged state carry
        // publishes the entering state unchanged as the allocation's final state.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, value],
                vec![Placeholder, Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 0), vec![body], vec![reference], None)
            .unwrap();
        let final_reference = outputs[0];
        let stacked_values = outputs[1];
        let final_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_value, stacked_values],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:ref<f32[]>, %2:f32[0] = scan [carry_count=1, length=0, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]> .
                        let %2:f32[] = const 1.0
                            () = reference_add_update %1 %2
                            %3:f32[] = reference_read %1
                        in (%1, %3)
                    },
                ]
                    %3:f32[] = reference_read %1
                in (%3, %2)"},
        );
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[0] = scan [carry_count=1, length=0, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %1 %2
                        in (%3, %3)
                    },
                ]
                in (%1, %2, %1)"},
        );
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar(2.0f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_builds_refined_zero_length_stacks() {
        // A zero-length scan whose stacked reference reaches a nested `condition` whole is a carried root, so discharge
        // builds its outputs without running or lowering its body. The empty stacked output takes the scan's inferred
        // output types: a static `f32[3]` carry input refines the stacked `f32[rows]` slices to the type `f32[0, 3]`.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [0]));
        let mut branch = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = branch.add_input(ArrayType::scalar(DataType::I64).into());
        let root = branch.add_input(reference_type.clone().into());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let element = branch
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let branch = branch
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![element], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        body_builder.add_input(scalar_type.clone().into());
        let vector = body_builder.add_input(vector_type.into());
        let stack = body_builder.add_input(reference_type.into());
        let predicate = body_builder.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let branch = body_builder.import_program(branch);
        let element = body_builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, index, stack], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![element, vector, vector],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let vector = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [0]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 0), vec![body], vec![initial, vector, stack], None)
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[1], outputs[2]],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            source.output_types(),
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [0, 3])),
            ],
        );

        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3], %2:f32[0] .
                let %3:ref<f32[0]> = reference_new %2
                    %4:f32[], %5:f32[3], %6:f32[0, 3] = scan [carry_count=2, length=0, reverse=false] %0 %1 %3 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:f32[rows], %3:ref<f32[0]> .
                            let %4:bool[] = const true
                                %5:f32[] = condition %4 %0 %3 [
                                    true=^0={
                                        lambda %0:i64[], %1:ref<f32[0]> .
                                        let %2:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] \
                %1 %0
                                        in (%2)
                                    },
                                    false=^0,
                                ]
                            in (%5, %2, %2)
                        },
                    ]
                in (%5, %6)"},
        );
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3], %2:f32[0] .
                let %3:f32[0, 3] = zero [type=f32[0, 3]]
                in (%1, %3)"},
        );
        let vector = Array::vector(vec![1f32, 2.0, 3.0]).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(4.0f32).unwrap()),
                TestIrValue::Array(vector.clone()),
                TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(vector),
                TestIrValue::Array(
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap(),
                ),
            ]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_dynamic_length_accepts_the_trailing_runtime_length_input() {
        // A dynamic-length scan carries one runtime-length instruction input after the body's inputs, so the scan
        // discharge rule's arity validation must accept the one-past-body parent arity instead of rejecting the
        // canonical dynamic form.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(9)).unwrap());
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar(3.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let scanned = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![reference, runtime_length],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![scanned], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let runtime_length = TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length), 4).unwrap());
        let eager = source
            .clone()
            .interpret(vec![TestIrValue::Array(Array::scalar(0.0f32).unwrap()), runtime_length.clone()])
            .unwrap();
        assert_eq!(eager, vec![TestIrValue::Array(Array::scalar(12.0f32).unwrap())]);
        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:f32[], %1:dimension<length ∈ [1, 9)> .
                let %2:ref<f32[]> = reference_new %0
                    %3:ref<f32[]> = scan [carry_count=1, length=length, reverse=false] %2 %1 [
                        body={
                            lambda %0:i64[], %1:ref<f32[]> .
                            let %2:f32[] = const 3.0
                                () = reference_add_update %1 %2
                            in (%1)
                        },
                    ]
                    %4:f32[] = reference_freeze %3
                in (%4)"},
        );
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:dimension<length ∈ [1, 9)> .
                let %2:f32[] = scan [carry_count=1, length=length, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const 3.0
                            %3:f32[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged
                .program()
                .interpret(vec![TestIrValue::Array(Array::scalar(0.0f32).unwrap()), runtime_length]),
            Ok(eager)
        );
    }

    #[test]
    fn test_scan_reference_discharge_preserves_zero_length_dimension_identities() {
        // Both the trip count and a slice dimension retain their nominal identities despite zero-trip elision.
        // The unreachable empty-root access is through a declared carry, not an optimized stacked reference.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(1)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let empty_type = ArrayType::new_static(DataType::F32, [0]);
        let reference_type = ReferenceType::new(empty_type.clone());
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(reference_type.into());
        let row_count = body.add_input(DimensionType::from(rows.clone()).into());
        let values = body.add_input(values_type.clone().into());
        body.add_instruction(
            ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]),
            Vec::new(),
            vec![root, index],
            None,
        )
        .unwrap();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![root, row_count, values, values],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let empty = builder.add_input(empty_type.into());
        let row_count = builder.add_input(DimensionType::from(rows.clone()).into());
        let values = builder.add_input(values_type.into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![empty], None).unwrap()[0];
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(3, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![root, row_count, values, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let final_root = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_root, outputs[1], outputs[2], outputs[3]],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.program().output_types(), program.output_types());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[0], %1:dimension<rows ∈ [1, 5)>, %2:f32[rows], %3:dimension<0> .
                let %4:f32[length, rows] = zero [type=f32[length, rows]] %3 %1
                in (%0, %1, %2, %4)"},
        );
        let inputs = vec![
            TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
            dimension(&DimensionType::from(rows), 2),
            TestIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap()),
            dimension(&DimensionType::from(length), 0),
        ];
        let mut expected = inputs[..3].to_vec();
        expected.push(TestIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [0, 2]), &[] as &[f32]).unwrap(),
        ));
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_recovers_zero_output_geometry_from_an_array_carry() {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into()]));
        let empty_type = ArrayType::new_static(DataType::F32, [0]);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(empty_type.clone()).into());
        let values = body.add_input(values_type.clone().into());
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![root, values, values],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let empty = builder.add_input(empty_type.into());
        let values = builder.add_input(values_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![empty], None).unwrap()[0];
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 0), vec![body], vec![root, values], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[1], outputs[2]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.program().output_types(), program.output_types());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[0], %1:f32[rows] .
                let %2:dimension<rows ∈ [1, 5)> = dimension_size [axis=0] %1
                    %3:f32[0, rows] = zero [type=f32[0, rows]] %2
                in (%1, %3)"},
        );
        let values = array(Array::vector(vec![3.0f32, 4.0]).unwrap());
        let inputs = vec![array(Array::vector(Vec::<f32>::new()).unwrap()), values.clone()];
        let expected = vec![
            values,
            array(Array::from_elements(ArrayType::new_static(DataType::F32, [0, 2]), &[] as &[f32]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_recovers_zero_output_geometry_from_a_reference_carry() {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let empty_type = ArrayType::new(DataType::F32, Shape::new(vec![0.into(), rows.clone().into()]));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(empty_type.clone()).into());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![root, value], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let empty = builder.add_input(empty_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![empty], None).unwrap()[0];
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 0), vec![body], vec![root], None)
            .unwrap()
            .to_vec();
        let final_root = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![final_root, outputs[1]],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.program().output_types(), program.output_types());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[0, rows] .
                let %1:dimension<rows ∈ [1, 5)> = dimension_size [axis=1] %0
                    %2:f32[0, 0, rows] = zero [type=f32[0, 0, rows]] %1
                in (%0, %2)"},
        );
        let empty = array(Array::from_elements(ArrayType::new_static(DataType::F32, [0, 2]), &[] as &[f32]).unwrap());
        let expected = vec![
            empty.clone(),
            array(Array::from_elements(ArrayType::new_static(DataType::F32, [0, 0, 2]), &[] as &[f32]).unwrap()),
        ];
        assert_eq!(program.interpret(vec![empty.clone()]), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(vec![empty]), Ok(expected));
    }

    // Replacing the body's dynamic selection with a slice input preserves the identities of subsequent allocations, so
    // discharge targets collected from the original program still select the intended internal allocation.
    #[test]
    fn test_scan_reference_discharge_preserves_zero_output_geometry_in_a_preserved_reference() {
        // Partial discharge must retain a zero-trip body boundary when its only dynamic geometry remains in a handle.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.into()]));
        let empty_type = ArrayType::new_static(DataType::F32, [0]);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let empty = body.add_input(ReferenceType::new(empty_type.clone()).into());
        let values = body.add_input(ReferenceType::new(values_type.clone()).into());
        let observed = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![values], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![empty, values, observed],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let empty = builder.add_input(empty_type.into());
        let values = builder.add_input(values_type.into());
        let empty = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![empty], None).unwrap()[0];
        let values = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![values], None).unwrap()[0];
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 0), vec![body], vec![empty, values], None)
            .unwrap()
            .to_vec();
        let empty = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let values = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[1]], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![empty, values, outputs[2]],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.program().output_types(), source.output_types());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[0], %1:f32[rows] .
                let %2:ref<f32[rows]> = reference_new %1
                    %3:f32[0], %4:ref<f32[rows]>, %5:f32[0, rows] = scan [carry_count=2, length=0, reverse=false] %0 \
                    %2 [
                        body={
                            lambda %0:i64[], %1:f32[0], %2:ref<f32[rows]> .
                            let %3:f32[rows] = reference_read %2
                            in (%1, %2, %3)
                        },
                    ]
                    %6:f32[rows] = reference_freeze %2
                in (%3, %6, %5)"},
        );
        let inputs = vec![
            TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
            TestIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap()),
        ];
        let expected = vec![
            inputs[0].clone(),
            inputs[1].clone(),
            TestIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [0, 2]), &[] as &[f32]).unwrap(),
            ),
        ];
        assert_eq!(source.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_preserves_internal_target_identity() {
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [3]));
        let root = body.add_input(root_type.clone().into());
        let row_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let value = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(row_transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        // This allocation's target follows the selection being normalized. Its instruction identity must survive.
        let local = body.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
        let output = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![local], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let root = builder.add_input(root_type.into());
        let output = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 3), vec![body], vec![root], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(targets.len(), 2);
        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:ref<f32[3]> .
                let %1:f32[3] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[3]> .
                        let %2:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %1 %0
                            %3:ref<f32[]> = reference_new %2
                            %4:f32[] = reference_read %3
                        in (%4)
                    },
                ]
                in (%1)"},
        );
        let discharged = source.partially_discharge_references(0, &targets).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        let values = TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(discharged.program().interpret(vec![values.clone()]), Ok(vec![values]));
    }

    #[test]
    fn test_scan_reference_discharge_normalizes_chained_paths_and_remaining_bindings() {
        // Remove only the leading row selection; the trailing dynamic column still selects within the row slice.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let index_type = ArrayType::scalar(DataType::I32);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let row = body.add_input(ArrayType::scalar(DataType::I64).into());
        let column = body.add_input(index_type.clone().into());
        let root = body.add_input(ReferenceType::new(matrix_type.clone()).into());
        let update = body.add_constant(TestIrValue::Array(Array::scalar(10.0f32).unwrap()));
        let transforms = vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ];
        body.add_instruction(
            ReferenceAddUpdateOperation::new().with_transforms(transforms.clone()),
            Vec::new(),
            vec![root, update, row, column],
            None,
        )
        .unwrap();
        let read = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![root, row, column],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![column, read], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let matrix = builder.add_input(matrix_type.into());
        let column = builder.add_input(index_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![matrix], None).unwrap()[0];
        let body = builder.import_program(body);
        let mut outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 2), vec![body], vec![column, root], None)
            .unwrap()
            .to_vec();
        outputs
            .push(builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0]);
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i32[] .
                let %2:ref<f32[2, 3]> = reference_new %0
                    %3:i32[], %4:f32[2] = scan [carry_count=1, length=2, reverse=false] %1 %2 [
                        body={
                            lambda %0:i64[], %1:i32[], %2:ref<f32[2, 3]> .
                            let %3:f32[] = const 10.0
                                () = reference_add_update [
                                    transforms=[index(axis=0, index=dynamic), slice(axes=[1:3]), index(axis=0, \
                index=dynamic)],
                                ] %2 %3 %0 %1
                                %4:f32[] = reference_read [
                                    transforms=[index(axis=0, index=dynamic), slice(axes=[1:3]), index(axis=0, \
                index=dynamic)],
                                ] %2 %0 %1
                            in (%1, %4)
                        },
                    ]
                    %5:f32[2, 3] = reference_freeze %2
                in (%3, %4, %5)"},
        );
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i32[] .
                let %2:i32[], %3:f32[2], %4:f32[2, 3] = scan [carry_count=1, length=2, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:i32[], %2:f32[3] .
                        let %3:f32[] = const 10.0
                            %4:f32[2] = slice [start_indices=[1], limits=[3]] %2
                            %5:f32[1] = dynamic_slice [sizes=[1]] %4 %1
                            %6:f32[] = reshape [shape=[]] %5
                            %7:f32[] = add %6 %3
                            %8:f32[1] = reshape [shape=[1]] %7
                            %9:f32[2] = dynamic_update_slice %4 %8 %1
                            %10:f32[3] = update_slice [start_indices=[1]] %2 %9
                            %11:f32[2] = slice [start_indices=[1], limits=[3]] %10
                            %12:f32[1] = dynamic_slice [sizes=[1]] %11 %1
                            %13:f32[] = reshape [shape=[]] %12
                        in (%1, %13, %10)
                    },
                ]
                in (%2, %3, %4)"},
        );
        let scan = discharged
            .program()
            .instructions()
            .iter()
            .find(|instruction| instruction.operation().name() == "scan")
            .unwrap();
        let body = discharged.program().region(scan.regions()[0]).unwrap();
        assert_eq!(
            body.input_types(),
            vec![
                ArrayType::scalar(DataType::I64).into(),
                ArrayType::scalar(DataType::I32).into(),
                ArrayType::new_static(DataType::F32, [3]).into(),
            ]
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        let inputs = vec![
            TestIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            TestIrValue::Array(Array::scalar(1i32).unwrap()),
        ];
        let expected = vec![
            TestIrValue::Array(Array::scalar(1i32).unwrap()),
            TestIrValue::Array(Array::vector(vec![13.0f32, 16.0]).unwrap()),
            TestIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 13.0, 4.0, 5.0, 16.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_carries_the_whole_root_for_mixed_whole_and_row_accesses() {
        // One access selects the current row and another reads the complete root, so per-row state cannot represent
        // the body. The root must be carried whole, and each iteration observes the rows updated so far.
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(vector_type.clone()).into());
        let one = body.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body.add_instruction(
            ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]),
            Vec::new(),
            vec![root, one, index],
            None,
        )
        .unwrap();
        let snapshot = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(vector_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.import_program(body);
        let snapshots = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 3), vec![body], vec![root], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![snapshots, frozen],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let inputs = vec![TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())];
        let expected = vec![
            TestIrValue::Array(Array::matrix(3, 3, vec![2.0f32, 2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 4.0]).unwrap()),
            TestIrValue::Array(Array::vector(vec![2.0f32, 3.0, 4.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3, 3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:f32[] = const 1.0
                            %3:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                            %4:f32[] = reshape [shape=[]] %3
                            %5:f32[] = add %4 %2
                            %6:f32[1] = reshape [shape=[1]] %5
                            %7:f32[3] = dynamic_update_slice %1 %6 %0
                        in (%7, %7)
                    },
                ]
                in (%2, %1)"},
        );
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_carries_the_whole_root_for_a_non_leading_iteration_index() {
        // The access indexes the iteration's column rather than its row, which a per-row slice cannot express, so the
        // root must be carried whole.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(matrix_type.clone()).into());
        let column_transforms =
            vec![ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic }];
        let update = body.add_constant(TestIrValue::Array(Array::vector(vec![10.0f32, 20.0]).unwrap()));
        body.add_instruction(
            ReferenceAddUpdateOperation::new().with_transforms(column_transforms.clone()),
            Vec::new(),
            vec![root, update, index],
            None,
        )
        .unwrap();
        let column = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(column_transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![column], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(matrix_type.into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.import_program(body);
        let columns = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 2), vec![body], vec![root], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![columns, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let inputs = vec![TestIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())];
        let expected = vec![
            TestIrValue::Array(Array::matrix(2, 2, vec![11.0f32, 24.0, 12.0, 25.0]).unwrap()),
            TestIrValue::Array(Array::matrix(2, 3, vec![11.0f32, 12.0, 3.0, 24.0, 25.0, 6.0]).unwrap()),
        ];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3], %2:f32[2, 2] = scan [carry_count=1, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[2, 3] .
                        let %2:f32[2] = const [10.0, 20.0]
                            %3:f32[2, 1] = dynamic_slice [sizes=[2, 1]] %1 %0 %0
                            %4:f32[2] = reshape [shape=[2]] %3
                            %5:f32[2] = add %4 %2
                            %6:f32[2, 1] = reshape [shape=[2, 1]] %5
                            %7:f32[2, 3] = dynamic_update_slice %1 %6 %0 %0
                            %8:f32[2, 1] = dynamic_slice [sizes=[2, 1]] %7 %0 %0
                            %9:f32[2] = reshape [shape=[2]] %8
                        in (%7, %9)
                    },
                ]
                in (%2, %1)"},
        );
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_reference_discharge_row_selection_requires_a_static_referent() {
        // Per-row discharge strips the leading row selection and retypes the root to its row without consulting the
        // row's shape. That is sound because the row selection only types statically shaped referents: an access that
        // selects a row of a stacked root with a dynamic trailing extent is rejected when it is built, and the scan
        // rule computes the row type through the same selection before it rewrites anything.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(4)).unwrap());
        let matrix_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(extent)]));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(matrix_type).into());
        let read = ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Dynamic,
        }]);
        assert!(matches!(
            body.add_instruction(read, Vec::new(), vec![root, index], None),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "reference indexing requires a static referent type but got `f32[2, extent]`",
        ));
    }

    #[test]
    fn test_scan_reference_discharge_threads_whole_roots_through_nested_regions() {
        // Passing the root and the index into a branch is valid even though the branch's view is not visible directly
        // in the scan body, so the root is carried whole. Read-only state remains unchanged, while mutations reach the
        // next iteration and the final freeze. In both cases the branch squares the selected row.
        let program = nested_region_root_program(3, false);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:ref<f32[3]> = reference_new %0
                    %2:f32[3] = scan [carry_count=0, length=3, reverse=false] %1 [
                        body={
                            lambda %0:i64[], %1:ref<f32[3]> .
                            let %2:bool[] = const true
                                %3:f32[] = condition %2 %0 %1 [
                                    true=^0={
                                        lambda %0:i64[], %1:ref<f32[3]> .
                                        let %2:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] \
                %1 %0
                                            %3:f32[] = mul %2 %2
                                        in (%3)
                                    },
                                    false=^0,
                                ]
                            in (%3)
                        },
                    ]
                    %3:f32[3] = reference_freeze %1
                in (%2, %3)"},
        );
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:bool[] = const true
                            %3:f32[] = condition %2 %0 %1 [
                                true={
                                    lambda %0:i64[], %1:f32[3] .
                                    let %2:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                                        %3:f32[] = reshape [shape=[]] %2
                                        %4:f32[] = mul %3 %3
                                    in (%4)
                                },
                                false={
                                    lambda %0:i64[], %1:f32[3] .
                                    let %2:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                                        %3:f32[] = reshape [shape=[]] %2
                                        %4:f32[] = mul %3 %3
                                    in (%4)
                                },
                            ]
                        in (%1, %3)
                    },
                ]
                in (%2, %1)"},
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        let input = TestIrValue::Array(Array::vector(vec![0.0f32, 1.0, 2.0]).unwrap());
        let expected = vec![
            TestIrValue::Array(Array::vector(vec![0.0f32, 1.0, 4.0]).unwrap()),
            TestIrValue::Array(Array::vector(vec![0.0f32, 1.0, 2.0]).unwrap()),
        ];
        assert_eq!(program.interpret(vec![input.clone()]), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(vec![input.clone()]), Ok(expected));

        // The nonlinear row computation needs saved primal values. Linearization retains only scalar per-iteration
        // coefficients, never a length-by-length history of the entire root carried by the fallback.
        let linearization = discharged.program().linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = mul %3 %3
                        in (%1, %4, %3)
                    },
                ]
                in (%2, %0, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[3] .
                let %2:f32[3], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[3], %2:f32[] .
                        let %3:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                            %4:f32[] = reshape [shape=[]] %3
                            %5:f32[] = mul %2 %4
                            %6:f32[] = mul %2 %4
                            %7:f32[] = add %5 %6
                        in (%1, %7)
                    },
                ]
                in (%3, %0)"},
        );
        assert_eq!(linearization.primal().output_types()[2..], vec![ArrayType::new_static(DataType::F32, [3]).into()]);
        let mut tangent_inputs = vec![TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap())];
        tangent_inputs.extend(
            linearization.primal().as_ref().clone().interpret(vec![input.clone()]).unwrap().into_iter().skip(2),
        );
        assert_eq!(
            linearization.tangent().as_ref().clone().interpret(tangent_inputs),
            Ok(vec![
                TestIrValue::Array(Array::vector(vec![0.0f32, 2.0, 4.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap()),
            ]),
        );

        let program = nested_region_root_program(3, true);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:ref<f32[3]> = reference_new %0
                    %2:f32[3] = scan [carry_count=0, length=3, reverse=false] %1 [
                        body={
                            lambda %0:i64[], %1:ref<f32[3]> .
                            let %2:bool[] = const true
                                %3:f32[] = condition %2 %0 %1 [
                                    true=^0={
                                        lambda %0:i64[], %1:ref<f32[3]> .
                                        let %2:f32[] = const 1.0
                                            () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] \
                %1 %2 %0
                                            %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] \
                %1 %0
                                            %4:f32[] = mul %3 %3
                                        in (%4)
                                    },
                                    false=^0,
                                ]
                            in (%3)
                        },
                    ]
                    %3:f32[3] = reference_freeze %1
                in (%2, %3)"},
        );
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:bool[] = const true
                            %3:f32[], %4:f32[3] = condition %2 %0 %1 [
                                true={
                                    lambda %0:i64[], %1:f32[3] .
                                    let %2:f32[] = const 1.0
                                        %3:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                                        %4:f32[] = reshape [shape=[]] %3
                                        %5:f32[] = add %4 %2
                                        %6:f32[1] = reshape [shape=[1]] %5
                                        %7:f32[3] = dynamic_update_slice %1 %6 %0
                                        %8:f32[1] = dynamic_slice [sizes=[1]] %7 %0
                                        %9:f32[] = reshape [shape=[]] %8
                                        %10:f32[] = mul %9 %9
                                    in (%10, %7)
                                },
                                false={
                                    lambda %0:i64[], %1:f32[3] .
                                    let %2:f32[] = const 1.0
                                        %3:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                                        %4:f32[] = reshape [shape=[]] %3
                                        %5:f32[] = add %4 %2
                                        %6:f32[1] = reshape [shape=[1]] %5
                                        %7:f32[3] = dynamic_update_slice %1 %6 %0
                                        %8:f32[1] = dynamic_slice [sizes=[1]] %7 %0
                                        %9:f32[] = reshape [shape=[]] %8
                                        %10:f32[] = mul %9 %9
                                    in (%10, %7)
                                },
                            ]
                        in (%4, %3)
                    },
                ]
                in (%2, %1)"},
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        let expected = vec![
            TestIrValue::Array(Array::vector(vec![1.0f32, 4.0, 9.0]).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        assert_eq!(program.interpret(vec![input.clone()]), Ok(expected.clone()));
        assert_eq!(discharged.program().interpret(vec![input.clone()]), Ok(expected));
        let linearization = discharged.program().linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[3] .
                        let %2:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = const 1.0
                            %5:f32[] = add %3 %4
                            %6:f32[1] = reshape [shape=[1]] %5
                            %7:f32[3] = dynamic_update_slice %1 %6 %0
                            %8:f32[1] = dynamic_slice [sizes=[1]] %7 %0
                            %9:f32[] = reshape [shape=[]] %8
                            %10:f32[] = mul %9 %9
                        in (%7, %10, %9)
                    },
                ]
                in (%2, %1, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[3] .
                let %2:f32[3], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[3], %2:f32[] .
                        let %3:f32[1] = dynamic_slice [sizes=[1]] %1 %0
                            %4:f32[] = reshape [shape=[]] %3
                            %5:f32[1] = reshape [shape=[1]] %4
                            %6:f32[3] = dynamic_update_slice %1 %5 %0
                            %7:f32[1] = dynamic_slice [sizes=[1]] %6 %0
                            %8:f32[] = reshape [shape=[]] %7
                            %9:f32[] = mul %2 %8
                            %10:f32[] = mul %2 %8
                            %11:f32[] = add %9 %10
                        in (%6, %11)
                    },
                ]
                in (%3, %2)"},
        );
        assert_eq!(linearization.primal().output_types()[2..], vec![ArrayType::new_static(DataType::F32, [3]).into()]);
        let mut tangent_inputs = vec![TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap())];
        tangent_inputs
            .extend(linearization.primal().as_ref().clone().interpret(vec![input]).unwrap().into_iter().skip(2));
        assert_eq!(
            linearization.tangent().as_ref().clone().interpret(tangent_inputs),
            Ok(vec![
                TestIrValue::Array(Array::vector(vec![2.0f32, 4.0, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_reference_discharge_threads_empty_whole_roots_through_nested_regions() {
        // A zero-length scan over an empty root carried whole executes no access to the root, whether the branch
        // would only read it or also mutate it, so both the outputs and their tangents are empty.
        let empty = TestIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap());
        for mutates in [false, true] {
            let program = nested_region_root_program(0, mutates);
            assert_eq!(
                program.to_string(),
                if mutates {
                    indoc! {"
                        lambda %0:f32[0] .
                        let %1:ref<f32[0]> = reference_new %0
                            %2:f32[0] = scan [carry_count=0, length=0, reverse=false] %1 [
                                body={
                                    lambda %0:i64[], %1:ref<f32[0]> .
                                    let %2:bool[] = const true
                                        %3:f32[] = condition %2 %0 %1 [
                                            true=^0={
                                                lambda %0:i64[], %1:ref<f32[0]> .
                                                let %2:f32[] = const 1.0
                                                    () = reference_add_update [transforms=[index(axis=0, \
                        index=dynamic)]] %1 %2 %0
                                                    %3:f32[] = reference_read [transforms=[index(axis=0, \
                        index=dynamic)]] %1 %0
                                                    %4:f32[] = mul %3 %3
                                                in (%4)
                                            },
                                            false=^0,
                                        ]
                                    in (%3)
                                },
                            ]
                            %3:f32[0] = reference_freeze %1
                        in (%2, %3)"}
                } else {
                    indoc! {"
                        lambda %0:f32[0] .
                        let %1:ref<f32[0]> = reference_new %0
                            %2:f32[0] = scan [carry_count=0, length=0, reverse=false] %1 [
                                body={
                                    lambda %0:i64[], %1:ref<f32[0]> .
                                    let %2:bool[] = const true
                                        %3:f32[] = condition %2 %0 %1 [
                                            true=^0={
                                                lambda %0:i64[], %1:ref<f32[0]> .
                                                let %2:f32[] = reference_read [transforms=[index(axis=0, \
                        index=dynamic)]] %1 %0
                                                    %3:f32[] = mul %2 %2
                                                in (%3)
                                            },
                                            false=^0,
                                        ]
                                    in (%3)
                                },
                            ]
                            %3:f32[0] = reference_freeze %1
                        in (%2, %3)"}
                },
            );
            let discharged = program.clone().discharge_references(0).unwrap();
            assert_eq!(
                discharged.program().to_string(),
                indoc! {"
                    lambda %0:f32[0] .
                    let %1:f32[0] = zero [type=f32[0]]
                    in (%1, %0)"},
            );
            assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
            assert_eq!(program.interpret(vec![empty.clone()]), Ok(vec![empty.clone(), empty.clone()]));
            assert_eq!(discharged.program().interpret(vec![empty.clone()]), Ok(vec![empty.clone(), empty.clone()]));
            let linearization = discharged.program().linearize().unwrap();
            assert_eq!(
                linearization.primal().to_string(),
                indoc! {"
                    lambda %0:f32[0] .
                    let %1:f32[0] = zero [type=f32[0]]
                    in (%1, %0)"},
            );
            assert_eq!(
                linearization.tangent().to_string(),
                indoc! {"
                    lambda %0:f32[0] .
                    let %1:f32[0] = zero [type=f32[0]]
                    in (%1, %0)"},
            );
            assert_eq!(linearization.primal().output_types()[2..], Vec::<ArrayIrType>::new());
            let mut tangent_inputs = vec![empty.clone()];
            tangent_inputs.extend(
                linearization.primal().as_ref().clone().interpret(vec![empty.clone()]).unwrap().into_iter().skip(2),
            );
            assert_eq!(
                linearization.tangent().as_ref().clone().interpret(tangent_inputs),
                Ok(vec![empty.clone(), empty.clone()]),
            );
        }
    }

    #[test]
    fn test_scan_reference_discharge_transfers_indices_to_root_memory() {
        // The loop provides its index in default memory. An explicit transfer makes it a valid dynamic index for
        // host-resident state; discharge must preserve that transfer even though this is not the direct-row pattern.
        let memory = Memory::Host { pinned: true };
        let array_type = ArrayType::new_static(DataType::F32, [3]).with_memory(memory);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let root = body.add_input(ReferenceType::new(array_type.clone()).into());
        let index = body
            .add_instruction(
                TestIrOperation::Array(TestOperation::TransferToMemory(TransferToMemoryOperation::new(memory))),
                Vec::new(),
                vec![index],
                None,
            )
            .unwrap()[0];
        let row_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let value = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(row_transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(array_type.clone().into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 3), vec![body], vec![root], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3]@Host[Pinned] .
                let %1:ref<f32[3]@Host[Pinned]> = reference_new %0
                    %2:f32[3]@Host[Pinned] = scan [carry_count=0, length=3, reverse=false] %1 [
                        body={
                            lambda %0:i64[], %1:ref<f32[3]@Host[Pinned]> .
                            let %2:i64[]@Host[Pinned] = transfer_to_memory [destination=Host[Pinned]] %0
                                %3:f32[]@Host[Pinned] = reference_read [transforms=[index(axis=0, index=dynamic)]] \
                %1 %2
                            in (%3)
                        },
                    ]
                in (%2)"},
        );
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3]@Host[Pinned] .
                let %1:f32[3]@Host[Pinned], %2:f32[3]@Host[Pinned] = scan [carry_count=1, length=3, reverse=false] \
                %0 [
                    body={
                        lambda %0:i64[], %1:f32[3]@Host[Pinned] .
                        let %2:i64[]@Host[Pinned] = transfer_to_memory [destination=Host[Pinned]] %0
                            %3:f32[1]@Host[Pinned] = dynamic_slice [sizes=[1]] %1 %2
                            %4:f32[]@Host[Pinned] = reshape [shape=[]] %3
                        in (%1, %4)
                    },
                ]
                in (%2)"},
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        let value = TestIrValue::Array(Array::from_elements::<f32>(array_type, &[1.0, 2.0, 3.0]).unwrap());
        assert_eq!(program.interpret(vec![value.clone()]), Ok(vec![value.clone()]));
        assert_eq!(discharged.program().interpret(vec![value.clone()]), Ok(vec![value]));
    }

    #[test]
    fn test_scan_reference_discharge_rejects_overlapping_stacked_references() {
        // A current-row view is region-local state inside the rebuilt body, so no other handle of its allocation may
        // reach the body through its boundary: a carry is a complete handle that always overlaps the view, and another
        // stacked input of the same allocation selects the same rows on every iteration.
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let row_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let whole = body.add_input(ReferenceType::new(stacked_type.clone()).into());
        let root = body.add_input(ReferenceType::new(stacked_type.clone()).into());
        let current = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(row_transforms.clone()),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![whole, current],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let elements = builder.add_input(stacked_type.clone().into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![stack, stack], None)
            .unwrap()
            .to_vec();
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![frozen, outputs[1]],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `scan` cannot discharge overlapping stacked reference inputs as \
                               independent slices",
        ));

        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let source = body.add_input(ReferenceType::new(stacked_type.clone()).into());
        let destination = body.add_input(ReferenceType::new(stacked_type.clone()).into());
        let current = body
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(row_transforms.clone()),
                Vec::new(),
                vec![source, index],
                None,
            )
            .unwrap()[0];
        body.add_instruction(
            ReferenceWriteOperation::new().with_transforms(row_transforms),
            Vec::new(),
            vec![destination, current, index],
            None,
        )
        .unwrap();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![current], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let elements = builder.add_input(stacked_type.into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let copies = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(0, 3), vec![body], vec![stack, stack], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![copies, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `scan` cannot discharge overlapping stacked reference inputs as \
                               independent slices",
        ));
    }

    #[test]
    fn test_scan_reference_discharge_rejects_a_capture_aliasing_a_stacked_reference() {
        // A capture is a complete handle that the body reaches without any boundary position. Here the capture-lifted
        // body reads the complete stack through capture 0 while the scan also passes that same capture as its stacked
        // reference input, and discharge rejects lifting the reference-typed capture constant.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stack_reference_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [3]));
        let mut body = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone().into());
        let _stack = body.add_input(stack_reference_type.clone().into());
        let captured = body.add_constant(DischargeCapture::new(0, stack_reference_type.clone().into()));
        body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured], None).unwrap();
        let body = body
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![carry], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let captured = builder.add_constant(DischargeCapture::new(0, stack_reference_type.into()));
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, captured], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![final_carry],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let closed = ClosedProgram::new(
            program,
            vec![TestIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()))],
        )
        .unwrap();
        assert!(matches!(
            closed.discharge_references(),
            Err(ProgramError::MalformedProgram(message))
                if message == "reference discharge cannot lift a constant of reference type `ref<f32[3]>`; a \
                               reference enters a program through an input, a capture binding, or an allocation",
        ));
    }

    #[test]
    fn test_scan_reference_discharge_composes_with_partial_evaluation() {
        // Discharge turns the scan's mutated reference carry into an ordinary carry before partial evaluation runs, so
        // an unknown boundary residualizes the whole three-iteration loop as a pure reference-free program.
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(array_type.clone());
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(array_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let evaluation = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .partially_evaluate(&[PartialValue::Unknown(array_type.into())])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = scan [carry_count=1, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.program().interpret(vec![TestIrValue::Array(Array::scalar(3.0f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(6.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_scan_interpretation() {
        // Eager interpretation runs the body once per iteration over the threaded carries and the current slices.
        let context = TestEagerContext::new();
        let initial = Array::scalar(1.0).unwrap();
        let values = Array::vector(vec![2.0, 3.0, 4.0]).unwrap();

        // Iteration `i` threads the carry while consuming slice `i` of every stacked input and producing slice `i` of
        // every stacked output: a cumulative product over `[2, 3, 4]` starting at `1` produces the final carry `24`
        // and the running products `[2, 6, 24]`. The lowering-only unroll factor does not change the result.
        let expected = vec![Array::scalar(24.0).unwrap(), Array::vector(vec![2.0, 6.0, 24.0]).unwrap()];
        assert_eq!(
            context.bind(
                TestOperation::Scan(TestScanOperation::new(1, 3)),
                vec![product_body()],
                &[initial.clone(), values.clone()],
            ),
            Ok(expected.clone()),
        );
        assert_eq!(
            context.bind(
                TestOperation::Scan(TestScanOperation::new(1, 3).with_unroll(2).unwrap()),
                vec![product_body()],
                &[initial.clone(), values.clone()],
            ),
            Ok(expected),
        );

        // A stacked input whose scan axis is sharded over an `Auto` mesh axis is sliced like an unsharded one, because
        // the per-iteration slices drop that placement hint.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_values = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [3])
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap(),
            &[2.0, 3.0, 4.0],
        )
        .unwrap();
        let outputs = context
            .bind(
                TestOperation::Scan(TestScanOperation::new(1, 3)),
                vec![product_body()],
                &[initial.clone(), sharded_values],
            )
            .unwrap();
        assert_eq!(outputs.iter().map(Array::to_f64s).collect::<Vec<_>>(), vec![vec![24.0], vec![2.0, 6.0, 24.0]]);

        // A reversed scan visits the slices from the back while keeping output slice `i` paired with input slice `i`:
        // the running products visit `4, 3, 2` and land in slots `2, 1, 0`.
        assert_eq!(
            context.bind(
                TestOperation::Scan(TestScanOperation::new(1, 3).with_reverse(true)),
                vec![product_body()],
                &[initial.clone(), values],
            ),
            Ok(vec![Array::scalar(24.0).unwrap(), Array::vector(vec![24.0, 12.0, 4.0]).unwrap()]),
        );

        // The body's index input holds the slice index of the current iteration in both visit orders, so stacking it
        // produces `[0, 1, 2]` either way.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let index = builder.add_input(ArrayType::scalar(DataType::I64));
        let index_body = builder.build(vec![index], vec![Placeholder], vec![Placeholder]).unwrap();
        for reverse in [false, true] {
            assert_eq!(
                context.bind(
                    TestOperation::Scan(TestScanOperation::new(0, 3).with_reverse(reverse)),
                    vec![index_body.clone()],
                    &[],
                ),
                Ok(vec![Array::vector(vec![0i64, 1, 2]).unwrap()]),
            );
        }

        // A carry-only scan with no stacked inputs or outputs applies its body `length` times.
        let doubling_body = scalar_body(1, |builder, inputs| {
            builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        assert_eq!(
            context.bind(TestOperation::Scan(TestScanOperation::new(1, 3)), vec![doubling_body], &[initial.clone()]),
            Ok(vec![Array::scalar(8.0).unwrap()]),
        );

        // A zero-length scan returns its initial carries and empty stacked outputs.
        let empty = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap();
        assert_eq!(
            context.bind(
                TestOperation::Scan(TestScanOperation::new(1, 0)),
                vec![product_body()],
                &[initial, empty.clone()],
            ),
            Ok(vec![Array::scalar(1.0).unwrap(), empty]),
        );

        // A homogeneous array scan has no runtime length input, so a dynamic length cannot be interpreted eagerly.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(
            TestScanOperation::new(1, Dimension::Dynamic(length)).interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot eagerly interpret homogeneous array `scan` with dynamic length `length` without an \
                          explicit first-class dimension input"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_scan_interpretation_validates_direct_rule_input_counts() {
        let bodies = [Arc::new(product_body())];
        let region_driver = CalleeRegionDriver::new(&bodies);
        let driver = EagerInterpretationDriver::new(&region_driver);
        assert_eq!(
            TestScanOperation::new(1, 0).interpret(&TestEagerContext::new(), &driver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry, carry], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let bodies = [Arc::new(body)];
        let region_driver = CalleeRegionDriver::new(&bodies);
        let driver = EagerInterpretationDriver::new(&region_driver);
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(1, 0).interpret(&context, &driver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length)).interpret(
                &context,
                &driver,
                &[array(Array::scalar(1.0f64).unwrap())],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
    }

    #[test]
    fn test_scan_interpretation_validates_the_composite_body_contract() {
        // Direct eager binding must reject a changing carry type even when the body would not execute.
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output = builder.add_constant(array(Array::scalar(true).unwrap()));
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let expected =
            Err(TypeError::invalid("`scan` body carry type signature mismatch: expected [f64[]] but got [bool[]]")
                .into());
        let carry = array(Array::scalar(1.0f64).unwrap());
        assert_eq!(
            context.bind(ScanOperation::<ArrayIrType>::new(1, 0), vec![body.clone()], &[carry.clone()]),
            expected
        );
        assert_eq!(context.bind(ScanOperation::<ArrayIrType>::new(1, 1), vec![body], &[carry]), expected);

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            context.bind(ScanOperation::<ArrayIrType>::new(0, 0), vec![body], &[]),
            Err(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index").into()),
        );
    }

    #[test]
    fn test_scan_interpretation_preserves_reference_carry_identities() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let first = builder.add_input(ReferenceType::new(scalar_type.clone()).into());
        let second = builder.add_input(ReferenceType::new(scalar_type).into());
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, first],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let first = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0f32).unwrap()));
        let second = TestIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32).unwrap()));
        let error = Err(ProgramError::MalformedProgram(
            "operation `scan` does not return carry 0 as the reference it entered with, so its `scan` state has \
             no fixed point"
                .to_owned(),
        ));
        assert_eq!(
            context.bind(ScanOperation::<ArrayIrType>::new(2, 0), vec![body.clone()], &[first.clone(), second.clone()]),
            error,
        );
        assert_eq!(
            context.bind(ScanOperation::<ArrayIrType>::new(2, 1), vec![body.clone()], &[first.clone(), second]),
            error,
        );
        // Equal allocation identities are safe even when the abstract body input positions differ.
        assert_eq!(
            context.bind(ScanOperation::<ArrayIrType>::new(2, 1), vec![body], &[first.clone(), first.clone()]),
            Ok(vec![first.clone(), first]),
        );
    }

    #[test]
    fn test_scan_interpretation_validates_concrete_dynamic_stack_lengths() {
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(5)).unwrap());
        let scalar = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar.clone().into());
        let value = body.add_input(scalar.clone().into());
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry, value], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let carry = builder.add_input(scalar.into());
        let stack = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![length.clone().into()])).into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                vec![carry, stack, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let carry = array(Array::scalar(7.0f64).unwrap());
        let stack = array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        assert_eq!(
            program.clone().interpret(vec![
                carry.clone(),
                stack.clone(),
                dimension(&DimensionType::from(length.clone()), 2)
            ]),
            Ok(vec![carry.clone(), stack.clone()]),
        );
        assert_eq!(
            program.clone().interpret(vec![
                carry.clone(),
                stack.clone(),
                dimension(&DimensionType::from(length.clone()), 3)
            ]),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<3>` but stacked input 1 has type `f64[2]` \
                 whose leading dimension is not refined to extent 3",
            )
            .into()),
        );
        assert_eq!(
            program.interpret(vec![carry, stack, dimension(&DimensionType::from(length), 1)]),
            Err(TypeError::invalid(
                "`scan` runtime length input has type `dimension<1>` but stacked input 1 has type `f64[2]` \
                 whose leading dimension is not refined to extent 1",
            )
            .into()),
        );
    }

    #[test]
    fn test_scan_interpretation_reports_first_iteration_errors_without_allocating_indices() {
        // The largest supported trip count must reach the first body's failure without first allocating one index
        // per iteration. Both visit orders report their actual first index, in both type universes.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let index = builder.add_input(ArrayType::scalar(DataType::I64));
        let condition = builder.add_constant(Array::scalar(false).unwrap());
        builder
            .add_instruction(
                AssertOperation::new("first iteration").with_labels(vec!["index".to_string()]),
                Vec::new(),
                vec![condition, index],
                None,
            )
            .unwrap();
        let body = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            body.to_string(),
            indoc! {r#"
                lambda %0:i64[] .
                let %1:bool[] = const false
                    () = assert [message="first iteration", labels=["index"]] %1 %0
                in ()"#},
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let condition = builder.add_constant(array(Array::scalar(false).unwrap()));
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("first iteration").with_labels(vec!["index".to_string()]),
                Vec::new(),
                vec![condition, index],
                None,
            )
            .unwrap();
        let composite_body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(composite_body.to_string(), body.to_string());
        for reverse in [false, true] {
            let expected = AssertionError::Failed {
                message: "first iteration".to_string(),
                observations: vec![(
                    "index".to_string(),
                    if reverse { MAX_DIMENSION_EXTENT - 1 } else { 0 }.to_string(),
                )],
            };
            let error = TestEagerContext::new()
                .bind(TestScanOperation::new(0, MAX_DIMENSION_EXTENT).with_reverse(reverse), vec![body.clone()], &[])
                .unwrap_err();
            assert_eq!(error.downcast_custom::<AssertionError>(), Some(&expected));
            let error = EagerContext::<TestIrValue, TestIrOperation>::new()
                .bind(
                    ScanOperation::<ArrayIrType>::new(0, MAX_DIMENSION_EXTENT).with_reverse(reverse),
                    vec![composite_body.clone()],
                    &[],
                )
                .unwrap_err();
            assert_eq!(error.downcast_custom::<AssertionError>(), Some(&expected));
        }
    }

    #[test]
    fn test_scan_interpretation_allocates_fresh_local_references_per_iteration() {
        // The body allocates a local reference from each item and accumulates the carry into it, so every iteration
        // must start from a fresh allocation. An allocation that persisted across iterations would accumulate into the
        // previous iteration's value instead of the current item, and the carries would diverge after the first item.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone().into());
        let item = body.add_input(scalar_type.clone().into());
        let reference = body.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![item], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, carry], None)
            .unwrap();
        let next = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next, next], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![carry, items], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 3.0, 4.0]).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(9.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 5.0, 9.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_interpretation_threads_stacked_reference_roots() {
        // The body receives the complete stacked root and selects its current element with an access whose dynamic
        // index is the body's index input, so reference analysis sees one root binding and no aliases.
        let program = stacked_reference_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:ref<f32[3]> = reference_new %1
                    %3:f32[] = scan [carry_count=1, length=3, reverse=false] %0 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[3]> .
                            let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                                %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %4:f32[] = add %1 %3
                            in (%4)
                        },
                    ]
                    %4:f32[3] = reference_freeze %2
                in (%3, %4)
            "}
            .trim_end(),
        );
        let analysis = program.entry_region_ref().reference_analysis(0).unwrap();
        assert_eq!(analysis.region_input_bindings().len(), 1);
        assert_eq!(analysis.values().filter_map(|value| analysis.alias(value)).count(), 0);

        // Eager interpretation supplies the index and the whole root to each access: with `c_{i+1} = 2 c_i + x_i`, the
        // carries are `1, 3, 8, 19` and the stack accumulates `x_i + c_i`.
        assert_eq!(
            program.interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(19.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_interpretation_composite_outputs_refined_by_array_inputs() {
        // The body stacks its `f32[rows]` carry. A static `f32[3]` carry input fixes `rows = 3` without any first-class
        // dimension input, so eager interpretation allocates the stacked output at the refined type `f32[2, 3]`.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(vector_type.into());
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, carry], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![doubled, carry],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let carry = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 2), vec![body], vec![carry], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.output_types(),
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
            ],
        );
        assert_eq!(
            program.interpret(vec![TestIrValue::Array(Array::vector(vec![1f32, 2.0, 3.0]).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::vector(vec![4f32, 8.0, 12.0]).unwrap()),
                TestIrValue::Array(
                    Array::from_elements::<f32>(
                        ArrayType::new_static(DataType::F32, [2, 3]),
                        &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                    )
                    .unwrap(),
                ),
            ]),
        );
    }

    #[test]
    fn test_scan_interpretation_composite_reference_carries_that_refine_their_declared_types() {
        // A program input declared as `ref<f32[rows]>` may be interpreted with a concrete `ref<f32[3]>`. Eager scan
        // interpretation accepts that reference carry and allocates the stacked reads of its referent at `f32[2, 3]`.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(reference_type.clone().into());
        let value =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry, value], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let stacked = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 2), vec![body], vec![reference], None)
            .unwrap()[1];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![stacked], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let reference = ArrayReference::new(Array::vector(vec![1f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            program.interpret(vec![TestIrValue::Reference(reference)]),
            Ok(vec![array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn test_scan_interpretation_composite_stacked_reference_inputs_as_per_iteration_transforms() {
        /// Builds a program over `[carry, stack]` that applies `operation` to a body which accumulates the carry into
        /// the per-iteration slice reference and then folds the updated slice into the carry, and that returns the
        /// final carry.
        fn scanned(
            operation: ScanOperation<ArrayIrType>,
            length: usize,
        ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
            let scalar_type = ArrayType::scalar(DataType::F32);
            let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = body_builder.add_input(scalar_type.clone().into());
            let root =
                body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [length])).into());
            let transforms =
                vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
            body_builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(transforms.clone()),
                    Vec::new(),
                    vec![root, carry, index],
                    None,
                )
                .unwrap();
            let current = body_builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(transforms),
                    Vec::new(),
                    vec![root, index],
                    None,
                )
                .unwrap()[0];
            let next_carry = body_builder
                .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, current], None)
                .unwrap()[0];
            let body = body_builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next_carry], vec![Placeholder; 3], vec![Placeholder])
                .unwrap();
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let body = builder.import_program(body);
            let initial = builder.add_input(scalar_type.into());
            let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [length])).into());
            let final_carry = builder.add_instruction(operation, vec![body], vec![initial, stack], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        }

        /// Evaluates the unrolled recurrence eagerly, indexing `stack` in the requested iteration order.
        fn unrolled(stack: &ArrayReference<Array>, iterations: &[usize]) -> Result<Vec<TestIrValue>, ProgramError> {
            let mut carry = Array::scalar(1.0f32).unwrap();
            for &iteration in iterations {
                let element = stack.with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(iteration),
                })?;
                element.add_update(&carry)?;
                carry = carry.add(&element.read()?)?;
            }
            Ok(vec![array(carry)])
        }

        // The body explicitly indexes the stacked reference on its leading axis, so the body
        // mutates the caller's referent in place exactly as the eager recurrence does, and the final carry observes
        // every updated slice.
        let scanned_stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let unrolled_stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let outputs = scanned(ScanOperation::new(1, 3), 3)
            .interpret(vec![array(Array::scalar(1.0f32).unwrap()), TestIrValue::Reference(scanned_stack.clone())]);
        assert_eq!(outputs, Ok(vec![array(Array::scalar(19.0f32).unwrap())]));
        assert_eq!(unrolled(&unrolled_stack, &[0, 1, 2]), outputs);
        assert_eq!(scanned_stack.read(), Ok(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()));
        assert_eq!(unrolled_stack.read(), scanned_stack.read());

        // A reversed scan visits the slices from the last to the first.
        let reversed_stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let unrolled_reversed_stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let outputs = scanned(ScanOperation::new(1, 3).with_reverse(true), 3)
            .interpret(vec![array(Array::scalar(1.0f32).unwrap()), TestIrValue::Reference(reversed_stack.clone())]);
        assert_eq!(outputs, Ok(vec![array(Array::scalar(25.0f32).unwrap())]));
        assert_eq!(unrolled(&unrolled_reversed_stack, &[2, 1, 0]), outputs);
        assert_eq!(reversed_stack.read(), Ok(Array::vector(vec![13.0f32, 7.0, 4.0]).unwrap()));
        assert_eq!(unrolled_reversed_stack.read(), reversed_stack.read());

        // A zero-length scan runs no iteration, so the carry passes through and the referent is never touched.
        let empty_stack = ArrayReference::new(Array::vector(Vec::<f32>::new()).unwrap());
        assert_eq!(
            scanned(ScanOperation::new(1, 0), 0)
                .interpret(vec![array(Array::scalar(1.0f32).unwrap()), TestIrValue::Reference(empty_stack.clone())]),
            Ok(vec![array(Array::scalar(1.0f32).unwrap())]),
        );
        assert_eq!(empty_stack.read(), Ok(Array::vector(Vec::<f32>::new()).unwrap()));
    }

    #[test]
    fn test_scan_partial_evaluation_rejects_an_invalid_body_before_forwarding_carries() {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::F64));
        let body = builder.build(vec![carry], vec![Placeholder; 3], vec![Placeholder; 1]).unwrap();
        let context = PartialEvaluationContext::new(TestEagerContext::new());
        let carry = context.lift(Array::scalar(1f64).unwrap()).unwrap();
        let stack =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::new_static(DataType::F64, [1]), 0));
        assert_eq!(
            context.bind(TestScanOperation::new(1, 1), vec![body], &[carry, stack]).unwrap_err(),
            ProgramError::Type(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_rejects_a_mismatched_known_carry_before_forwarding_it() {
        let body = scalar_body(2, |_, inputs| vec![inputs[1]]);
        let context = PartialEvaluationContext::new(TestEagerContext::new());
        let carry = context.lift(Array::scalar(true).unwrap()).unwrap();
        let stack =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::new_static(DataType::F64, [1]), 0));
        assert_eq!(
            context.bind(TestScanOperation::new(1, 1), vec![body], &[carry, stack]).unwrap_err(),
            ProgramError::Type(TypeError::invalid(
                "`scan` input 0 has type `bool[]`, which does not refine its expected type `f64[]`",
            )),
        );
    }

    #[test]
    fn test_scan_partial_evaluation() {
        // Partial evaluation folds a scan whose inputs are all known and residualizes a scan with no foldable known
        // work unchanged.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program =
            scan_program(TestScanOperation::new(1, 3), product_body(), vec![scalar_type.clone(), stacked_type.clone()]);

        // A scan whose inputs are all known folds completely into known outputs.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(1.0).unwrap()),
                PartialValue::Known(Array::vector(vec![2.0, 3.0, 4.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(24.0).unwrap()),
                PartialEvaluationOutput::Known(Array::vector(vec![2.0, 6.0, 24.0]).unwrap()),
            ],
        );
        assert_eq!(evaluation.inputs, vec![]);
        assert_eq!(evaluation.program.instructions().len(), 0);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda  .
                in ()
            "}
            .trim_end(),
        );

        // A scan whose inputs are all unknown residualizes unchanged.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar_type), PartialValue::Unknown(stacked_type.clone())])
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.inputs, vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Unknown(1)]);
        assert_eq!(evaluation.program.to_string(), program.to_string());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );

        // A known initial carry whose next value depends on unknown slices cannot fold, so the scan residualizes
        // unchanged over the known carry as a residual input.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(1.0).unwrap()),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), &[Array::vector(vec![2.0, 3.0, 4.0]).unwrap()]),
            Ok(vec![Array::scalar(24.0).unwrap(), Array::vector(vec![2.0, 6.0, 24.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_residualizes_zero_length_scans_without_probing_the_body() {
        // A zero-length scan runs no iteration, so partial evaluation must not probe or specialize its body. The
        // division by the known zero carry stays in the residual body; signed array division by zero would yield -1,
        // but no slice is evaluated here. The passed-through carry keeps its known initial value as its final value.
        let carry_type = ArrayType::scalar(DataType::I32);
        let stacked_type = ArrayType::new_static(DataType::I32, [0]);
        let mut body = ProgramBuilder::<Array, TestOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64));
        let carry = body.add_input(carry_type.clone());
        let value = body.add_input(carry_type.clone());
        let one = body.add_constant(Array::scalar(1i32).unwrap());
        let inverse = body.add_instruction(DivOperation::new(), Vec::new(), vec![one, carry], None).unwrap()[0];
        let scaled = body.add_instruction(MulOperation::new(), Vec::new(), vec![inverse, value], None).unwrap()[0];
        let body = body.build(vec![carry, scaled], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();
        let program = scan_program(TestScanOperation::new(1, 0), body, vec![carry_type, stacked_type.clone()]);
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0i32).unwrap()),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(0i32).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:i32[0], %1:i32[] .
                let %2:i32[], %3:i32[0] = scan [carry_count=1, length=0, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:i32[], %2:i32[] .
                        let %3:i32[] = const 1
                            %4:i32[] = div %3 %1
                            %5:i32[] = mul %4 %2
                        in (%1, %5)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_keeps_effectful_known_work_in_the_known_scan() {
        // A body that prints inside its known chain keeps the effect in the known scan of the known-ness split:
        // effectful bodies skip the live-context invariance probes and go straight to the split, whose fresh probe
        // contexts fold the all-known print into the known side. The known scan staged into the live outer trace owns
        // the print (running it once per iteration, all before the residual side, per the effect placement contract)
        // and keeps the visit order and unroll factor of the source scan, while the residual scan stays pure.
        //
        // The body maps `[accumulator, scale, x]` to `[accumulator + (print(scale) * scale) * x, scale, accumulator']`.
        let body = scalar_body(3, |builder, inputs| {
            let [_, accumulator, scale, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let printed =
                builder.add_instruction(PrintOperation::new("scale"), Vec::new(), vec![scale], None).unwrap()[0];
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scaled], None).unwrap()[0];
            vec![next, scale, next]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3).with_reverse(true).with_unroll(2).unwrap(),
            body,
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let outer = TracingContext::<Array, TestOperation>::new();
        let scale = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Unknown(scalar_type), PartialValue::Known(scale), PartialValue::Unknown(stacked_type)],
            )
            .unwrap();
        let outer_builder = outer.builder().borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let known_program = outer_builder
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                outer_builder.instructions()[0].outputs().to_vec(),
                vec![Placeholder],
                vec![Placeholder; outer_builder.instructions()[0].outputs().len()],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[3] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 [
                    body={
                        lambda %0:i64[], %1:f64[] .
                        let %2:f64[] = print [label=scale] %1
                            %3:f64[] = mul %2 %1
                        in (%1, %3)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert!(evaluation.program.effects().classes().is_empty());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3] .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %3 %2
                            %5:f64[] = add %1 %4
                        in (%5, %5)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_preserves_order_between_known_and_unknown_effects() {
        // Ordered effects on both sides of a partition retain their original per-iteration execution order: the body
        // prints its known carry and its unknown slice on every iteration, and splitting it would print `known` twice
        // before either `unknown` instead of alternating them, so the scan residualizes whole.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [2]);
        let body = scalar_body(2, |builder, inputs| {
            builder.add_instruction(PrintOperation::new("known"), Vec::new(), vec![inputs[1]], None).unwrap();
            let printed =
                builder.add_instruction(PrintOperation::new("unknown"), Vec::new(), vec![inputs[2]], None).unwrap()[0];
            vec![inputs[1], printed]
        });
        let program =
            scan_program(TestScanOperation::new(1, 2), body.clone(), vec![scalar_type.clone(), stacked_type.clone()]);

        // A staging parent receives no known-side scan or other speculative effects. The carry passes through the body
        // unchanged, so its final value is forwarded from its known initial value.
        let outer = TracingContext::<Array, TestOperation>::new();
        let carry = outer.input(scalar_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(carry.clone()), PartialValue::Unknown(stacked_type.clone())],
            )
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[] .
                let %2:f64[], %3:f64[2] = scan [carry_count=1, length=2, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = print [label=known] %1
                            %4:f64[] = print [label=unknown] %2
                        in (%1, %4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );

        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(outer.builder().borrow().instructions().len(), 0);
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(carry), PartialEvaluationOutput::Unknown(0)]
        );
        assert_eq!(evaluation.program.instructions().len(), 1);
        let scan = &evaluation.program.instructions()[0];
        assert_eq!(
            evaluation.program.region_ref(scan.regions()[0]).unwrap().to_program().to_string(),
            body.to_string()
        );

        // The same placement holds under an eager parent, so specialization cannot execute the known prints before
        // the residual scan runs, and applying the residual program retains the original numeric outputs.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(stacked_type.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[] .
                let %2:f64[], %3:f64[2] = scan [carry_count=1, length=2, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = print [label=known] %1
                            %4:f64[] = print [label=unknown] %2
                        in (%1, %4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(3.0).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(evaluation.program.instructions().len(), 1);
        let scan = &evaluation.program.instructions()[0];
        assert_eq!(
            evaluation.program.region_ref(scan.regions()[0]).unwrap().to_program().to_string(),
            body.to_string()
        );
        let items = Array::from_elements::<f64>(stacked_type, &[5.0, 7.0]).unwrap();
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), &[items.clone()]),
            Ok(vec![Array::scalar(3.0).unwrap(), items]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_preserves_zero_output_residual_effects() {
        // A split scan retains an effectful unknown body as a zero-output residual scan even when every boundary result
        // belongs to the known side: here the body prints each unknown slice and passes its known carry through.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let body = scalar_body(2, |builder, inputs| {
            builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![inputs[2]], None).unwrap();
            vec![inputs[1]]
        });
        assert!(body.partition(&[true, true, false]).unwrap().residual_program().effects().classes().is_ordered());
        let program = scan_program(TestScanOperation::new(1, 3), body, vec![scalar_type.clone(), stacked_type.clone()]);
        let outer = TracingContext::<Array, TestOperation>::new();
        let carry = outer.input(scalar_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(carry.clone()), PartialValue::Unknown(stacked_type)],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Known(carry)]);
        assert!(evaluation.program.effects().classes().is_ordered());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let () = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f64[] .
                        let %2:f64[] = print [label=x] %1
                        in ()
                    },
                ]
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_folds_loop_invariant_known_carry() {
        // A loop-invariant known carry holds its initial value on every iteration, so it is dropped from the residual
        // scan, its uses fold into the residual body as constants (here `scale * scale` folds to `4`), and its final
        // value is its known initial value. The rewritten scan keeps the visit order and unroll factor of the source
        // scan. The body maps `[accumulator, scale, x]` to
        // `[accumulator + (scale * scale) * x, scale, accumulator + (scale * scale) * x]`.
        let body = scalar_body(3, |builder, inputs| {
            let [_, accumulator, scale, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scaled], None).unwrap()[0];
            vec![next, scale, next]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3).with_reverse(true).with_unroll(2).unwrap(),
            body,
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type),
                PartialValue::Known(Array::scalar(2.0).unwrap()),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(Array::scalar(2.0).unwrap()),
                PartialEvaluationOutput::Unknown(1),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = const 4.0
                            %4:f64[] = mul %3 %2
                            %5:f64[] = add %1 %4
                        in (%5, %5)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );

        // The accumulator visits the slices from the back: `1 -> 1 + 4 * 7 = 29 -> 29 + 4 * 6 = 53 -> 53 + 4 * 5 = 73`,
        // and the stacked output keeps each running value in the slot of the slice that produced it.
        assert_eq!(
            evaluation.interpret(
                &TestEagerContext::new(),
                &[Array::scalar(1.0).unwrap(), Array::vector(vec![5.0, 6.0, 7.0]).unwrap()],
            ),
            Ok(vec![
                Array::scalar(73.0).unwrap(),
                Array::scalar(2.0).unwrap(),
                Array::vector(vec![73.0, 53.0, 29.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_propagates_unknownness_across_carry_dependencies() {
        // Equal known initial values make the first carry of the shifting body appear invariant in the first probe.
        // The second carry becomes unknown first, and only the next pass reveals that the first carry must also remain
        // residual.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [4]);
        let program = scan_program(
            TestScanOperation::new(3, 4),
            shifting_carry_body(),
            vec![scalar_type.clone(), scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0.0).unwrap()),
                PartialValue::Known(Array::scalar(0.0).unwrap()),
                PartialValue::Unknown(scalar_type),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[4], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[4] = scan [carry_count=3, length=4, reverse=false] %2 %3 %0 \
                 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        in (%2, %3, %4, %1)
                    },
                ]
                in (%4, %5, %6, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
                PartialEvaluationOutput::Unknown(2),
                PartialEvaluationOutput::Unknown(3),
            ],
        );
        let expected = vec![
            Array::scalar(2.0).unwrap(),
            Array::scalar(3.0).unwrap(),
            Array::scalar(4.0).unwrap(),
            Array::vector(vec![0.0, 0.0, 10.0, 1.0]).unwrap(),
        ];
        assert_eq!(
            program.interpret(vec![
                Array::scalar(0.0).unwrap(),
                Array::scalar(0.0).unwrap(),
                Array::scalar(10.0).unwrap(),
                Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ]),
            Ok(expected.clone()),
        );
        assert_eq!(
            evaluation.interpret(
                &TestEagerContext::new(),
                &[Array::scalar(10.0).unwrap(), Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
            ),
            Ok(expected),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_splits_time_varying_known_work() {
        // Body `[sum, x] -> [sum + x * x, x * x]` over an unknown sum and known stacked slices. The known scan computes
        // the stacked squares during partial evaluation, and they surface both as the folded stacked output and as the
        // residual edge feeding the unknown scan, which keeps the visit order and unroll factor of the source scan.
        let body = scalar_body(2, |builder, inputs| {
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[2], inputs[2]], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], squared], None).unwrap()[0];
            vec![next, squared]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(1, 3).with_reverse(true).with_unroll(2).unwrap(),
            body,
            vec![scalar_type.clone(), stacked_type],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type),
                PartialValue::Known(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(Array::vector(vec![1.0, 4.0, 9.0]).unwrap()),
            ],
        );
        assert_eq!(
            evaluation.inputs,
            vec![
                PartialEvaluationInput::Unknown(0),
                PartialEvaluationInput::Known(Array::vector(vec![1.0, 4.0, 9.0]).unwrap()),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %1 %2
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(24.0).unwrap(), Array::vector(vec![1.0, 4.0, 9.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_splits_symbolic_known_carries_under_staging() {
        // Under a staging known-side context, a symbolic known carry (a genuine outer tracer) participates in the
        // known-ness split. The body maps `[accumulator, scale, x]` to
        // `[accumulator + (scale * scale) * x, scale, accumulator + (scale * scale) * x]`, whose known carry `scale`
        // passes through unchanged. The loop-invariant `scale * scale` is therefore hoisted into the outer trace and
        // computed once instead of being stacked per iteration, no known scan is needed, and the final `scale` is
        // forwarded from its known initial value. The fixed-point probes run through fresh contexts, so the hoisted
        // multiplication is the only instruction that the outer trace gains.
        let body = scalar_body(3, |builder, inputs| {
            let [_, accumulator, scale, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scaled], None).unwrap()[0];
            vec![next, scale, next]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body,
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let outer = TracingContext::<Array, TestOperation>::new();
        let scale = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Unknown(scalar_type),
                    PartialValue::Known(scale.clone()),
                    PartialValue::Unknown(stacked_type),
                ],
            )
            .unwrap();
        let outer_builder = outer.builder().borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let hoisted = outer_builder
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                outer_builder.instructions()[0].outputs().to_vec(),
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            hoisted.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(scale),
                PartialEvaluationOutput::Unknown(1),
            ],
        );
        assert_eq!(evaluation.inputs.len(), 3);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[], %5:f64[3] = scan [carry_count=2, length=3, reverse=false] %2 %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %1 %3
                            %5:f64[] = add %2 %4
                        in (%1, %5, %5)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_composite_refines_outputs_of_unspecialized_bodies() {
        // Partially evaluating a scan whose refined outputs come from an unspecialized body keeps them refined, whether
        // the carry or the stacked input is known, and the residual program reproduces the original one. With known
        // slices, the known scan stacks the squares and the residual scan consumes its refined stacked edge.
        let program = refined_vector_scan_program();
        let carry_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let stacked_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3]));
        assert_eq!(program.output_types(), vec![carry_type.clone(), stacked_type.clone()]);
        let arguments = vec![
            array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 3]),
                    &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                )
                .unwrap(),
            ),
        ];
        let expected = vec![
            array(Array::vector(vec![4.0, 5.0, 6.0]).unwrap()),
            array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 3]),
                    &[1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
                )
                .unwrap(),
            ),
        ];
        for (partial_inputs, known_outputs) in [
            (
                vec![PartialValue::Known(arguments[0].clone()), PartialValue::Unknown(stacked_type.clone())],
                [false, false],
            ),
            (vec![PartialValue::Unknown(carry_type.clone()), PartialValue::Known(arguments[1].clone())], [false, true]),
        ] {
            let evaluation = program.partially_evaluate(&partial_inputs).unwrap();
            assert_eq!(
                evaluation.program.to_string(),
                if known_outputs[1] {
                    indoc! {"
                    lambda %0:f64[3], %1:f64[2, 3] .
                    let %2:f64[3] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[rows], %2:f64[rows] .
                            let %3:f64[rows] = add %1 %2
                            in (%3)
                        },
                    ]
                    in (%2)
                "}
                    .trim_end()
                } else {
                    indoc! {"
                    lambda %0:f64[2, 3], %1:f64[3] .
                    let %2:f64[3], %3:f64[2, 3] = scan [carry_count=1, length=2, reverse=false] %1 %0 [
                        body={
                            lambda %0:i64[], %1:f64[rows], %2:f64[rows] .
                            let %3:f64[rows] = add %1 %2
                                %4:f64[rows] = mul %2 %2
                            in (%3, %4)
                        },
                    ]
                    in (%2, %3)
                "}
                    .trim_end()
                },
            );
            assert_eq!(
                evaluation
                    .outputs
                    .iter()
                    .map(|output| matches!(output, PartialEvaluationOutput::Known(_)))
                    .collect::<Vec<_>>(),
                known_outputs,
            );
            assert!(evaluation.program.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
            let residual_arguments = evaluation
                .inputs
                .iter()
                .map(|input| match input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => arguments[*index].clone(),
                })
                .collect::<Vec<_>>();
            let residual_outputs = evaluation.program.interpret(residual_arguments).unwrap();
            let reassembled = evaluation
                .outputs
                .iter()
                .map(|output| match output {
                    PartialEvaluationOutput::Known(value) => value.clone(),
                    PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
                })
                .collect::<Vec<_>>();
            assert_eq!(reassembled, expected);
        }

        // Under a staging known-side context, symbolic known slices split the scan the same way: the known scan staged
        // into the outer trace and the residual scan both keep refined types.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Unknown(carry_type), PartialValue::Known(outer.input(stacked_type))],
            )
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[2, 3] .
                let %2:f64[3] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[rows], %2:f64[rows] .
                        let %3:f64[rows] = add %1 %2
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );

        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[2, 3] = scan [carry_count=0, length=2, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f64[3] .
                        let %2:f64[3] = mul %1 %1
                        in (%2)
                    },
                ]
                in ()
            "}
            .trim_end(),
        );
        assert!(
            outer
                .builder()
                .borrow()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), TestIrOperation::Scan(_)))
        );
        assert!(outer.builder().borrow().atoms().iter().all(|atom| atom.r#type().identities().next().is_none()));
        assert!(evaluation.program.atoms().iter().all(|atom| atom.r#type().identities().next().is_none()));
    }

    #[test]
    fn test_scan_partial_evaluation_reuses_computed_constants_under_staging() {
        // A constant-resolved outer tracer must not trigger a live invariance probe whose disposable multiplication
        // remains in the outer program. The knownness split instead computes one coefficient that the residual uses.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [2]);
        let body = scalar_body(2, |builder, inputs| {
            let [_, scale, value]: [AtomId; 3] = inputs.try_into().unwrap();
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            vec![scale, scaled]
        });
        let program = scan_program(TestScanOperation::new(1, 2), body, vec![scalar_type, stacked_type.clone()]);
        let outer = TracingContext::<Array, TestOperation>::new();
        let scale = outer.lift(Array::scalar(2.0).unwrap()).unwrap();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(scale.clone()), PartialValue::Unknown(stacked_type)],
            )
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[] .
                let %2:f64[], %3:f64[2] = scan [carry_count=1, length=2, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%1, %3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(scale), PartialEvaluationOutput::Unknown(0)]
        );
        let PartialEvaluationInput::Known(coefficient) = &evaluation.inputs[1] else { panic!("missing coefficient") };
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![coefficient.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = const 2.0
                    %1:f64[] = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(outer_program.interpret(Vec::new()), Ok(vec![Array::scalar(4.0).unwrap()]));
    }

    #[test]
    fn test_scan_partial_evaluation_hoists_loop_invariant_residuals_eagerly() {
        // An effectful body skips the invariance probes and goes straight to the known-ness split, even under an eager
        // known-side context. The body maps `[accumulator, scale, x]` to `[accumulator + (scale * scale) * print(x),
        // scale]`, whose known carry `scale` passes through unchanged, so the split computes the loop-invariant
        // `scale * scale` once before the scan and threads it through the unknown scan as a carry that its body passes
        // through, instead of stacking a copy of it per iteration.
        let body = scalar_body(3, |builder, inputs| {
            let [_, accumulator, scale, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let printed = builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![value], None).unwrap()[0];
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, printed], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scaled], None).unwrap()[0];
            vec![next, scale]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body,
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type),
                PartialValue::Known(Array::scalar(2.0).unwrap()),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Known(Array::scalar(2.0).unwrap())],
        );
        assert_eq!(
            evaluation.inputs,
            vec![
                PartialEvaluationInput::Unknown(0),
                PartialEvaluationInput::Unknown(2),
                PartialEvaluationInput::Known(Array::scalar(4.0).unwrap()),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[] = scan [carry_count=2, length=3, reverse=false] %2 %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = print [label=x] %3
                            %5:f64[] = mul %1 %4
                            %6:f64[] = add %2 %5
                        in (%1, %6)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.interpret(
                &TestEagerContext::new(),
                &[Array::scalar(1.0).unwrap(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
            ),
            Ok(vec![Array::scalar(25.0).unwrap(), Array::scalar(2.0).unwrap()]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_preserves_signed_zero_carries() {
        // Body `[carry, x] -> [-carry, x]` over a known carry and unknown stacked inputs. Negating `0.0` yields `-0.0`,
        // which compares equal to `0.0`, so a value-based invariance check would wrongly fold the carry to its initial
        // value. After an odd number of iterations, the final carry must be `-0.0`.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let _index = builder.add_input(ArrayType::scalar(DataType::I64));
            let carry = builder.add_input(scalar_type.clone());
            let input = builder.add_input(scalar_type.clone());
            let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![negated, input], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone());
        let values = builder.add_input(stacked_type.clone());
        let outputs = builder
            .add_instruction(
                ArrayOperation::Scan(ScanOperation::new(1, 3)),
                vec![body_region],
                vec![initial, values],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0.0f64).unwrap()),
                PartialValue::Unknown(stacked_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f64[] .
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let PartialEvaluationOutput::Known(final_carry) = &evaluation.outputs[0] else {
            panic!("expected the final carry to be known");
        };
        assert!(final_carry.to_f64s()[0].is_sign_negative());
        assert_eq!(final_carry.to_string(), "-0.0");
    }

    #[test]
    fn test_scan_partial_evaluation_stacks_time_varying_known_carry_feeders() {
        // Body `[carry, x] -> [carry + 1, (carry + 1) * x]` over a known carry and unknown stacked inputs. The unknown
        // side consumes the known carry's next value, which changes on every iteration, so the split must stack it per
        // iteration instead of threading the carry's initial value.
        let body = scalar_body(2, |builder, inputs| {
            let one = builder.add_constant(Array::scalar(1.0f64).unwrap());
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], one], None).unwrap()[0];
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![next, inputs[2]], None).unwrap()[0];
            vec![next, product]
        });
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(1, 3),
            body,
            vec![ArrayType::scalar(DataType::F64), stacked_type.clone()],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0f64).unwrap()),
                PartialValue::Unknown(stacked_type.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = scan [carry_count=0, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_partial_evaluation_preserves_semantics(
            &program,
            &[PartialValue::Known(Array::scalar(0.0f64).unwrap()), PartialValue::Unknown(stacked_type)],
            &[Array::scalar(0.0f64).unwrap(), Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap()],
        );
    }

    #[test]
    fn test_scan_partial_evaluation_forwards_known_stacked_carry_feeders() {
        // Body `[carry, x, z] -> [x, x * z]` over a known carry, known stacked `x`, and unknown stacked `z`. The known
        // slice `x` is both the carry's next value and a feeder of the unknown side, so the unknown scan must read the
        // original stacked `x` exactly once.
        let body = scalar_body(3, |builder, inputs| {
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[2], inputs[3]], None).unwrap()[0];
            vec![inputs[2], product]
        });
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(1, 3),
            body,
            vec![ArrayType::scalar(DataType::F64), stacked_type.clone(), stacked_type.clone()],
        );
        let values = Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0f64).unwrap()),
                PartialValue::Known(values.clone()),
                PartialValue::Unknown(stacked_type.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = scan [carry_count=0, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_partial_evaluation_preserves_semantics(
            &program,
            &[
                PartialValue::Known(Array::scalar(0.0f64).unwrap()),
                PartialValue::Known(values.clone()),
                PartialValue::Unknown(stacked_type),
            ],
            &[Array::scalar(0.0f64).unwrap(), values, Array::vector(vec![5.0f64, 6.0, 7.0]).unwrap()],
        );
    }

    #[test]
    fn test_scan_partial_evaluation_forwards_known_stacked_inputs() {
        // Body `[product, sum, x] -> [product * x, sum + x]` over an unknown carry `product`, a known carry `sum`, and
        // known stacked slices. The unknown side consumes the known slices directly, so the residual scan takes the
        // original stacked input instead of a copy that the known scan stacks again, while the known scan still
        // carries `sum`.
        let body = scalar_body(3, |builder, inputs| {
            let next_product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[3]], None).unwrap()[0];
            let next_sum =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[2], inputs[3]], None).unwrap()[0];
            vec![next_product, next_sum]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body,
            vec![scalar_type.clone(), scalar_type, ArrayType::new_static(DataType::F64, [3])],
        );
        assert_eq!(
            program.partition(&[false, true, true]).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[1, 2],
                    residual_inputs=[Unknown(0), Known(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f64[], %1:f64[3] .
                    let %2:f64[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[] .
                            let %3:f64[] = add %1 %2
                            in (%3)
                        },
                    ]
                    in (%2, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[3] .
                    let %2:f64[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[] .
                            let %3:f64[] = mul %1 %2
                            in (%3)
                        },
                    ]
                    in (%2)
                }
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_forwards_passed_through_carries() {
        // The body maps `[scale, accumulator, x]` to `[scale, accumulator + scale * x]`. Its carry `scale` passes
        // through unchanged, so the scan's final `scale` is forwarded from its input: an unknown input is returned as
        // the residual program's input itself rather than as a scan output.
        let body = scalar_body(3, |builder, inputs| {
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[3]], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[2], scaled], None).unwrap()[0];
            vec![inputs[1], next]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body,
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type.clone()),
                PartialValue::Unknown(scalar_type.clone()),
                PartialValue::Unknown(stacked_type.clone()),
            ])
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3] .
                let %3:f64[], %4:f64[] = scan [carry_count=2, length=3, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %1 %3
                            %5:f64[] = add %2 %4
                        in (%1, %5)
                    },
                ]
                in (%0, %4)
            "}
            .trim_end(),
        );

        // A known non-constant `scale` (an outer tracer, which the invariance probes skip) stays known through the
        // known-ness split, even though the known scan that the split would otherwise need is never staged.
        let outer = TracingContext::<Array, TestOperation>::new();
        let scale = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(scale.clone()),
                    PartialValue::Unknown(scalar_type),
                    PartialValue::Unknown(stacked_type),
                ],
            )
            .unwrap();
        assert_eq!(outer.builder().borrow().instructions().len(), 0);
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(scale), PartialEvaluationOutput::Unknown(0)]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[] = scan [carry_count=2, length=3, reverse=false] %2 %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %1 %3
                            %5:f64[] = add %2 %4
                        in (%1, %5)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_dynamic_length() {
        // A dynamic-length composite scan folds its loop-invariant known carry like a static one, and the rewritten
        // residual scan keeps the dynamic length and its trailing runtime length input. The body maps
        // `[scale, accumulator, x]` to `[scale, accumulator + scale * x, scale * x]`.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let stacked_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let length_type = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let scale = body.add_input(scalar_type.clone());
        let accumulator = body.add_input(scalar_type.clone());
        let value = body.add_input(scalar_type.clone());
        let scaled = body
            .add_instruction(
                TestIrOperation::Array(TestOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![scale, value],
                None,
            )
            .unwrap()[0];
        let next = body
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![accumulator, scaled], None)
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![scale, next, scaled],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs = [scalar_type.clone(), scalar_type.clone(), stacked_type.clone(), length_type.clone()]
            .map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(2, Dimension::Dynamic(length.clone())),
                vec![body],
                inputs.to_vec(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 4], vec![Placeholder; 3])
            .unwrap();
        let scale = TestIrValue::Array(Array::scalar(2.0f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(scale.clone()),
                PartialValue::Unknown(scalar_type),
                PartialValue::Unknown(stacked_type),
                PartialValue::Unknown(length_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(scale),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[length], %2:dimension<length ∈ [1, 8)> .
                let %3:f32[], %4:f32[length] = scan [carry_count=1, length=length, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = const 2.0
                            %4:f32[] = mul %3 %2
                            %5:f32[] = add %1 %4
                        in (%5, %4)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[
                    TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                    TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
                    TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length), 3).unwrap()),
                ],
            ),
            Ok(vec![
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(13.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 4.0, 6.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_partial_evaluation_dynamic_length_that_admits_zero() {
        // Neither a known zero trip count nor an unknown trip count that admits zero may evaluate the body's
        // division by a known zero carry during specialization. The body is never executed when the count is zero.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        let stacked_type =
            ArrayIrType::Array(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let length_type = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone());
        let value = body.add_input(scalar_type.clone());
        let one = body.add_constant(TestIrValue::Array(Array::scalar(1i32).unwrap()));
        let inverse = body
            .add_instruction(
                TestIrOperation::Array(TestOperation::Div(DivOperation::new())),
                Vec::new(),
                vec![one, carry],
                None,
            )
            .unwrap()[0];
        let scaled = body
            .add_instruction(
                TestIrOperation::Array(TestOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![inverse, value],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![carry, scaled],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs =
            [scalar_type, stacked_type.clone(), length_type.clone()].map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                inputs.to_vec(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let zero = TestIrValue::Array(Array::scalar(0i32).unwrap());
        let empty = TestIrValue::Array(Array::vector(Vec::<i32>::new()).unwrap());
        let runtime_length =
            TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length.clone()), 0).unwrap());
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(zero.clone()),
                PartialValue::Unknown(stacked_type.clone()),
                PartialValue::Known(runtime_length.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:i32[length], %1:i32[], %2:dimension<length ∈ [0, 8)> .
                let %3:i32[], %4:i32[length] = scan [carry_count=1, length=length, reverse=false] %1 %0 %2 [
                    body={
                        lambda %0:i64[], %1:i32[], %2:i32[] .
                        let %3:i32[] = const 1
                            %4:i32[] = div %3 %1
                            %5:i32[] = mul %4 %2
                        in (%1, %5)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(zero.clone()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&empty)), Ok(vec![zero.clone(), empty.clone()]));

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(zero.clone()),
                PartialValue::Unknown(stacked_type),
                PartialValue::Unknown(length_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:i32[length], %1:dimension<length ∈ [0, 8)>, %2:i32[] .
                let %3:i32[], %4:i32[length] = scan [carry_count=1, length=length, reverse=false] %2 %0 %1 [
                    body={
                        lambda %0:i64[], %1:i32[], %2:i32[] .
                        let %3:i32[] = const 1
                            %4:i32[] = div %3 %1
                            %5:i32[] = mul %4 %2
                        in (%1, %5)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(zero.clone()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(evaluation.interpret(&context, &[empty.clone(), runtime_length]), Ok(vec![zero, empty]));
    }

    #[test]
    fn test_scan_partial_evaluation_residualizes_reference_carry_whole() {
        // `f(r, values) = { for value in values { write(r, 1); add_update(r, value) }; read(r) }` with the reference
        // carried through the scan.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let value = body_builder.add_input(scalar_type.clone().into());
        let one = body_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let values = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let reference = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference, values], None)
            .unwrap()[0];
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Moving the known write into a separate scan would run all resets before the unknown updates. Keeping
        // each reset beside its update produces 1 + 3 = 4, rather than 1 + (1 + 2 + 3) = 7.
        let live = ArrayReference::new(Array::scalar(2.0f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(TestIrValue::Reference(live.clone())),
                PartialValue::Unknown(ArrayType::new_static(DataType::F32, [3]).into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:ref<f32[]> .
                let %2:ref<f32[]> = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[] .
                        let %3:f32[] = const 1.0
                            () = reference_write %1 %3
                            () = reference_add_update %1 %2
                        in (%1)
                    },
                ]
                    %3:f32[] = reference_read %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(live.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation
                .program()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["scan", "reference_read"],
        );
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())],
            ),
            Ok(vec![TestIrValue::Array(Array::scalar(4.0f32).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::scalar(4.0f32).unwrap()));
    }

    #[test]
    fn test_scan_partial_evaluation_residualizes_reference_stacks_whole() {
        // A reference stack crosses no known-ness split: whichever of the carry and the stack is known, the scan
        // residualizes whole and threads a known stack by identity.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let stack_type = ArrayIrType::from(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(stacked_reference_body());
        let initial = builder.add_input(scalar_type.clone());
        let stack = builder.add_input(stack_type.clone());
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Known stack, unknown carry: the stack is a reference-typed known feeder of the body, so the scan
        // residualizes whole and the residual program threads the live stack by identity.
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type.clone()),
                PartialValue::Known(TestIrValue::Reference(stack.clone())),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[3]> .
                let %2:f32[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:ref<f32[3]> .
                        let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                            %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                            %4:f32[] = add %1 %3
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.inputs(),
            &[PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(TestIrValue::Reference(stack.clone()))],
        );
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Reference(stack.clone()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(19.0f32).unwrap())]),
        );
        assert_eq!(stack.read(), Ok(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()));

        // Known carry, unknown stack: the carry depends on the unknown stack after one iteration, so nothing stays
        // known and the scan residualizes unchanged over the known carry residual and the unknown stack.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
                PartialValue::Unknown(stack_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:ref<f32[3]>, %1:f32[] .
                let %2:f32[] = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:ref<f32[3]> .
                        let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                            %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                            %4:f32[] = add %1 %3
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.inputs(),
            &[
                PartialEvaluationInput::Unknown(1),
                PartialEvaluationInput::Known(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
            ],
        );
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            evaluation.program().interpret(vec![
                TestIrValue::Reference(stack.clone()),
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(19.0f32).unwrap())]),
        );
        assert_eq!(stack.read(), Ok(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()));
    }

    #[test]
    fn test_scan_batching() {
        // Eager batching replays the scan loop per iteration with every body instruction batched, so each batch item
        // runs its own cumulative product. Stacked inputs mapped at axis 0 are read along their per-item leading axis
        // (packed axis 1), and stacked outputs gain the scan axis in front of their per-iteration batch axis.
        let regions = vec![product_body()];
        check_operation_batching!(
            @exact,
            context = TestEagerContext::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = TestScanOperation::new(1, 3),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [
                {
                    inputs = [
                        (@replicated, Array::scalar(1.0).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![24.0, 210.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(1.0).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![24.0, 210.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, 10.0]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![24.0, 2100.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![2.0, 50.0, 6.0, 300.0, 24.0, 2100.0]).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, 2.0]).unwrap()),
                        (@replicated, Array::vector(vec![2.0, 3.0, 4.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![24.0, 48.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![2.0, 4.0, 6.0, 12.0, 24.0, 48.0]).unwrap()),
                    ],
                },
            ],
        );

        // A reversed batched scan visits the slices from the back while keeping output slice `i` paired with input
        // slice `i`: the reversed cumulative product over `[2, 3, 4]` is `[24, 12, 4]` for each batch item.
        check_operation_batching!(
            @exact,
            context = TestEagerContext::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = TestScanOperation::new(1, 3).with_reverse(true),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [
                    (@replicated, Array::scalar(1.0).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 1.0, 1.0, 2.0]).unwrap()),
                ],
                outputs = [
                    (@mapped(axis = 0), Array::vector(vec![24.0, 2.0]).unwrap()),
                    (@mapped(axis = 1), Array::matrix(3, 2, vec![24.0, 2.0, 12.0, 2.0, 4.0, 2.0]).unwrap()),
                ],
            }],
        );
    }

    #[test]
    fn test_scan_batching_rejects_ragged_inputs_before_transforming_body() {
        // A ragged carry's per-item extent lives outside its packed type. Reject it before body discovery can
        // substitute the storage bound for that extent, including when no body iteration would execute.
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let carry = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f64, 900.0, 800.0, 4.0, 5.0, 700.0]).unwrap(), 0)
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i64, 2]).unwrap(), extent, vec![0])])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(1, 0).batch(&context, &EmptyRegionDriver, &[carry]).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "`scan` does not support bounded ragged dimension `extent` on input 0".to_string(),
            },
        );
    }

    #[test]
    fn test_scan_batching_composite_rejects_ragged_inputs_before_transforming_body() {
        // Structural composite batching receives mapped axes without ragged carriers. Rejection must precede both
        // body discovery and parent binding, so padded carry elements never become live logical elements.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let carry = ArrayIrBatch::new(
            array(Array::matrix(2, 3, vec![1.0f64, 900.0, 800.0, 4.0, 5.0, 700.0]).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, array(Array::vector(vec![1i64, 2]).unwrap()), extent, vec![0])])
        .unwrap();
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(1, 3).batch(&context, &EmptyRegionDriver, &[carry]).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "`scan` does not support bounded ragged dimension `extent` on input 0".to_string(),
            },
        );
    }

    #[test]
    fn test_scan_batching_validates_direct_body_and_input_contracts() {
        // A direct rule invocation must not repair the body's index or index past an excessive carry count, and
        // even a passed-through zero-trip carry must satisfy the original logical input contract.
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        for (index_type, carry_count, input, message) in [
            (
                DataType::Boolean,
                1,
                Array::scalar(1.0f64).unwrap(),
                "`scan` body input 0 must be a scalar `i64` slice index",
            ),
            (DataType::I64, 2, Array::scalar(1.0f64).unwrap(), "`scan` carry count 2 exceeds the body input count 1"),
            (
                DataType::I64,
                1,
                Array::scalar(true).unwrap(),
                "`scan` input 0 has type `bool[]`, which does not refine its expected type `f64[]`",
            ),
        ] {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            builder.add_input(ArrayType::scalar(index_type));
            let carry = builder.add_input(ArrayType::scalar(DataType::F64));
            let body = builder.build(vec![carry], vec![Placeholder; 2], vec![Placeholder]).unwrap();
            let regions = [body];
            let driver = RecursiveBatchingDriver::new(&regions);
            assert_eq!(
                TestScanOperation::new(carry_count, 0)
                    .batch(&context, &driver, &[ArrayBatch::replicated(input)])
                    .unwrap_err(),
                BatchingError::from(TypeError::invalid(message)),
            );
        }
        assert_eq!(
            TestScanOperation::new(0, 0).batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::from(ProgramError::MalformedProgram(
                "operation `scan` declares 1 region slots but 0 regions were attached".to_string(),
            )),
        );
    }

    #[test]
    fn test_scan_batching_composite_validates_direct_body_and_input_contracts() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        for (index_type, carry_count, input, message) in [
            (
                DataType::Boolean,
                1,
                Array::scalar(1.0f64).unwrap(),
                "`scan` body input 0 must be a scalar `i64` slice index",
            ),
            (DataType::I64, 2, Array::scalar(1.0f64).unwrap(), "`scan` carry count 2 exceeds the body input count 1"),
            (
                DataType::I64,
                1,
                Array::scalar(true).unwrap(),
                "`scan` input 0 has type `bool[]`, which does not refine its expected type `f64[]`",
            ),
        ] {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(ArrayType::scalar(index_type).into());
            let carry = builder.add_input(ArrayType::scalar(DataType::F64).into());
            let body = builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let regions = [body];
            let driver = RecursiveBatchingDriver::new(&regions);
            assert_eq!(
                ScanOperation::<ArrayIrType>::new(carry_count, 0)
                    .batch(&context, &driver, &[ArrayIrBatch::replicated(array(input))])
                    .unwrap_err(),
                BatchingError::from(TypeError::invalid(message)),
            );
        }
        assert_eq!(
            ScanOperation::<ArrayIrType>::new(0, 0).batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::from(ProgramError::MalformedProgram(
                "operation `scan` declares 1 region slots but 0 regions were attached".to_string(),
            )),
        );
    }

    #[test]
    fn test_scan_batching_preserves_trailing_stacked_input_axes() {
        // The mapped axis lies behind both the scan axis and an element axis. Removing each scan slice must shift
        // axis 2 to axis 1, and stacking the slices must restore axis 2 without moving the element dimension.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64));
        let value = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let body = builder.build(vec![value], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let regions = vec![body];
        let values = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2, 3, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            context = TestEagerContext::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = TestScanOperation::new(0, 2),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [(@mapped(axis = 2), values.clone())],
                outputs = [(@mapped(axis = 2), values)],
            }],
        );
    }

    #[test]
    fn test_scan_batching_propagates_cascading_carry_axes() {
        // Mapping the scanned input first widens the third carry, then the second, then the first. The structural
        // fixed point must propagate through all three carries before the stabilized body is staged.
        let program = scan_program(
            TestScanOperation::new(3, 3),
            shifting_carry_body(),
            vec![
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::new_static(DataType::F64, [3]),
            ],
        );
        let batched = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(1)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(
            batched.output_axes(),
            &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(1)],
        );
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[3, 2] .
                let %4:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %0
                    %5:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %1
                    %6:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %2
                    %7:f64[2], %8:f64[2], %9:f64[2], %10:f64[3, 2] = scan [carry_count=3, length=3, \
                     reverse=false] %4 %5 %6 %3 [
                        body={
                            lambda %0:i64[], %1:f64[2], %2:f64[2], %3:f64[2], %4:f64[2] .
                            in (%2, %3, %4, %1)
                        },
                    ]
                in (%7, %8, %9, %10)"},
        );
        assert_eq!(
            batched.interpret(vec![
                Array::scalar(1.0f64).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(3.0f64).unwrap(),
                Array::matrix(3, 2, vec![10.0f64, 40.0, 20.0, 50.0, 30.0, 60.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![10.0f64, 40.0]).unwrap(),
                Array::vector(vec![20.0f64, 50.0]).unwrap(),
                Array::vector(vec![30.0f64, 60.0]).unwrap(),
                Array::matrix(3, 2, vec![1.0f64, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_batching_reconciles_stacked_outputs_that_become_batched_eagerly() {
        // Body `[carry, x] -> [carry + x, carry]` over a replicated initial carry and stacked inputs mapped at axis 0.
        // The first iteration stacks the still-replicated carry while later iterations stack mapped carries, so the
        // eager loop must broadcast the earlier iterations of the stacked output instead of rejecting the change.
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, inputs[1]]
        });
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let initial = ArrayBatch::replicated(Array::scalar(1.0f64).unwrap());
        let values =
            ArrayBatch::new(Array::matrix(2, 3, vec![2.0f64, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(), BatchAxis::new(0))
                .unwrap();
        assert_eq!(
            batch_scan(&context, TestScanOperation::new(1, 3), body, vec![initial, values]),
            vec![
                ArrayBatch::new(Array::vector(vec![10.0f64, 19.0]).unwrap(), BatchAxis::new(0)).unwrap(),
                ArrayBatch::new(
                    Array::matrix(3, 2, vec![1.0f64, 1.0, 3.0, 6.0, 6.0, 12.0]).unwrap(),
                    BatchAxis::new(1)
                )
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_scan_batching_forwards_passed_through_carries() {
        // The body maps `[scale, accumulator]` to `[scale, accumulator + scale]` over `f64[3]` carries. Its carry
        // `scale` passes through unchanged, so the batched scan returns the input batch of `scale` itself, at its own
        // batch axis, both eagerly and when staging, where the structural rule otherwise realigns carries to axis 0.
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let mut body = ProgramBuilder::<Array, TestOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64));
        let scale = body.add_input(vector_type.clone());
        let accumulator = body.add_input(vector_type.clone());
        let next = body.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scale], None).unwrap()[0];
        let body = body.build(vec![scale, next], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let scale =
            ArrayBatch::new(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(1))
                .unwrap();
        let accumulator = ArrayBatch::replicated(Array::vector(vec![0.0, 0.0, 0.0]).unwrap());
        assert_eq!(
            batch_scan(&context, TestScanOperation::new(2, 2), body.clone(), vec![scale.clone(), accumulator]),
            vec![
                scale,
                ArrayBatch::new(Array::matrix(3, 2, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap(), BatchAxis::new(1))
                    .unwrap(),
            ],
        );

        let regions = vec![body];
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let scale_atom = builder.borrow_mut().add_input(ArrayType::new_static(DataType::F64, [3, 2]));
        let accumulator_atom = builder.borrow_mut().add_input(vector_type);
        let inputs = vec![
            ArrayBatch::new(parent.tracer(scale_atom, None), BatchAxis::new(1)).unwrap(),
            ArrayBatch::replicated(parent.tracer(accumulator_atom, None)),
        ];
        let context = BatchingContext::new(parent, 2);
        let outputs = TestScanOperation::new(2, 2)
            .batch(&context, &RecursiveBatchingDriver::new(&regions), inputs.as_slice())
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0], inputs[0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap(), outputs[1].value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 2], %1:f64[3] .
                let %2:f64[2, 3] = transpose [permutation=[1, 0]] %0
                    %3:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %1
                    %4:f64[2, 3], %5:f64[2, 3] = scan [carry_count=2, length=2, reverse=false] %2 %3 [
                        body={
                            lambda %0:i64[], %1:f64[2, 3], %2:f64[2, 3] .
                            let %3:f64[2, 3] = add %2 %1
                            in (%1, %3)
                        },
                    ]
                in (%0, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_batching_stages_one_batched_scan_under_tracing() {
        // Batching a scan under a staging parent stages exactly one batched scan, with the replicated carry widened
        // through a staged broadcast and the batched stacked input realigned off the leading scan dimension, instead of
        // unrolling the loop into per-iteration body copies. The batched scan keeps the visit order and unroll factor.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let values_atom = builder.borrow_mut().add_input(ArrayType::new_static(DataType::F64, [2, 3]));
        let carry = parent.tracer(carry_atom, None);
        let values = parent.tracer(values_atom, None);
        let (final_carry, stacked) = batch(
            |(carry, values)| {
                let mut outputs = carry.context().bind(
                    TestOperation::Scan(TestScanOperation::new(1, 3).with_reverse(true).with_unroll(2).unwrap()),
                    vec![product_body()],
                    &[carry.clone(), values.clone()],
                )?;
                Ok((outputs.remove(0), outputs.remove(0)))
            },
            (carry, values),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Vec<Array>>(
                vec![final_carry.atom_id().unwrap(), stacked.atom_id().unwrap()],
                (Placeholder, Placeholder),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[2, 3] .
                let %2:f64[3, 2] = transpose [permutation=[1, 0]] %1
                    %3:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %0
                    %4:f64[2], %5:f64[3, 2] = scan [carry_count=1, length=3, reverse=true, unroll=2] %3 %2 [
                        body={
                            lambda %0:i64[], %1:f64[2], %2:f64[2] .
                            let %3:f64[2] = mul %1 %2
                            in (%3, %3)
                        },
                    ]
                    %6:f64[2, 3] = transpose [permutation=[1, 0]] %5
                in (%4, %6)
            "}
            .trim_end(),
        );

        // Interpreting the staged program computes the per-item reversed cumulative products, with the replicated
        // carry broadcast across the batch.
        let values = Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        assert_eq!(
            program.interpret((Array::scalar(1.0).unwrap(), values)),
            Ok(vec![
                Array::vector(vec![24.0, 210.0]).unwrap(),
                Array::matrix(2, 3, vec![24.0, 12.0, 4.0, 210.0, 42.0, 7.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_batching_composite_refines_outputs_of_unspecialized_bodies() {
        // Batching a scan whose refined outputs come from an unspecialized body keeps its batched outputs refined. The
        // carry is mapped and the stacked input is replicated, so each item computes `[c + x[0] + x[1], x * x]`.
        let extent = DimensionValue::constant(2).unwrap();
        let batched = refined_vector_scan_program()
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f64[2, 3], %2:f64[2, 3] .
                let %3:dimension<2>, %4:f64[2, 3], %5:f64[2, 3] = scan [carry_count=2, length=2, reverse=false] %0 \
                %1 %2 [
                    body={
                        lambda %0:i64[], %1:dimension<2>, %2:f64[2, 3], %3:f64[3] .
                        let %4:dimension<3> = dimension_size [axis=1] %2
                            %5:f64[2, 3] = broadcast [output_axes=[1]] %3 %1 %4
                            %6:f64[2, 3] = add %2 %5
                            %7:f64[3] = mul %3 %3
                        in (%1, %6, %7)
                    },
                ]
                in (%0, %4, %5)"},
        );
        let matrix_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3]));
        assert_eq!(batched.output_types()[1..], [matrix_type.clone(), matrix_type]);
        let matrix = |values: &[f64]| {
            array(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), values).unwrap())
        };
        let outputs = batched
            .interpret(vec![
                TestIrValue::Dimension(extent),
                matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                matrix(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            ])
            .unwrap();
        assert_eq!(outputs[1], matrix(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        assert_eq!(outputs[2], matrix(&[1.0, 1.0, 1.0, 4.0, 4.0, 4.0]));
    }

    #[test]
    fn test_scan_batching_preserves_reverse_and_unroll_of_composite_dynamic_length_scans() {
        // The composite rule threads the mapped extent as a leading replicated carry and keeps the dynamic length, its
        // trailing runtime length input, the visit order, and the unroll factor of the source scan. The body maps
        // `[carry, x]` to `[carry + x, carry]`.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let stacked_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let length_type = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone());
        let value = body.add_input(scalar_type.clone());
        let next = body
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, value], None)
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next, carry], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs = [scalar_type, stacked_type, length_type].map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone()))
                    .with_reverse(true)
                    .with_unroll(2)
                    .unwrap(),
                vec![body],
                inputs.to_vec(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = program
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(1), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(1)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[], %2:f32[length, 2], %3:dimension<length ∈ [1, 8)> .
                let %4:f32[2] = broadcast [output_axes=[]] %1 %0
                    %5:dimension<2>, %6:f32[2], %7:f32[length, 2] = scan [carry_count=2, length=length, \
                     reverse=true, unroll=2] %0 %4 %2 %3 [
                        body={
                            lambda %0:i64[], %1:dimension<2>, %2:f32[2], %3:f32[2] .
                            let %4:f32[2] = add %2 %3
                            in (%1, %4, %2)
                        },
                    ]
                in (%0, %6, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
                TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length), 3).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Dimension(axis_extent),
                TestIrValue::Array(Array::vector(vec![7.0f32, 16.0]).unwrap()),
                TestIrValue::Array(Array::matrix(3, 2, vec![6.0f32, 12.0, 4.0, 7.0, 1.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_batching_composite_validates_concrete_eager_runtime_geometry() {
        // A nominal runtime length remains symbolic in its type, while packed eager arrays expose concrete shapes.
        // Resolve its value for validation so valid direct binding executes and unequal lengths cannot truncate or
        // over-read the packed stack before the parent's eager scan interpreter validates its transformed boundary.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(5)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let slice = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let next = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, slice], None)
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next, next], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let regions = [body];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = |extent| {
            vec![
                ArrayIrBatch::replicated(array(Array::scalar(0.0f32).unwrap())),
                ArrayIrBatch::new(
                    array(Array::matrix(3, 2, vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap()),
                    BatchAxis::new(1),
                )
                .unwrap(),
                ArrayIrBatch::replicated(TestIrValue::Dimension(
                    DimensionValue::new(DimensionType::from(length.clone()), extent).unwrap(),
                )),
            ]
        };
        let operation = ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone()));
        let outputs = operation.batch(&context, &driver, &inputs(3)).unwrap().into_parts().0;
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(array(Array::vector(vec![6.0f32, 60.0]).unwrap()), BatchAxis::new(0)).unwrap(),
                ArrayIrBatch::new(
                    array(Array::matrix(3, 2, vec![1.0f32, 10.0, 3.0, 30.0, 6.0, 60.0]).unwrap()),
                    BatchAxis::new(1),
                )
                .unwrap(),
            ]
        );
        for extent in [2, 4] {
            assert_eq!(
                operation.batch(&context, &driver, &inputs(extent)).unwrap_err(),
                BatchingError::from(TypeError::invalid(format!(
                    "`scan` runtime length input has type `dimension<{extent}>` but stacked input 1 has type \
                     `f32[3]` whose leading dimension is not refined to extent {extent}",
                ))),
            );
        }
    }

    #[test]
    fn test_scan_batching_reuses_the_stabilized_body_discovery_program() {
        // The structural rule iterates the body's carry axes to a fixed point with natural output axes and then
        // instantiates the body at the joined carry and stacked-slice axes. `AlignEachTo` stages axis movement only
        // where a natural axis differs from a mapped target, so when the stabilizing pass already discovered those
        // targets its program is the aligned body and is not rebuilt.
        let regions = vec![product_body()];

        // Both carries and stacked slices are batched from the start, and the stacked input already carries its batch
        // axis off the leading scan dimension, so the first pass widens nothing and its discovered axes already equal
        // the joined targets: exactly one structural pass.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(ArrayType::new_static(DataType::F64, [2]));
        let values_atom = builder.borrow_mut().add_input(ArrayType::new_static(DataType::F64, [3, 2]));
        let carry = parent.tracer(carry_atom, None);
        let values = parent.tracer(values_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![
            ArrayBatch::new(carry, BatchAxis::new(0)).unwrap(),
            ArrayBatch::new(values, BatchAxis::new(1)).unwrap(),
        ];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = TestScanOperation::new(1, 3).batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 1);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap(), outputs[1].value().atom_id().unwrap()],
                (Placeholder, Placeholder),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[3, 2] .
                let %2:f64[2], %3:f64[3, 2] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[2], %2:f64[2] .
                        let %3:f64[2] = mul %1 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %3)"},
        );
        let outputs = program
            .interpret((
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 1.0]).unwrap(),
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [3, 2]),
                    &[2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
                )
                .unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0, 210.0]);
        assert_eq!(outputs[1].to_f64s(), vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A replicated carry whose next-carry output is batched widens once, so the fixed point runs two natural
        // passes. The second (stabilizing) pass is still reused instead of being replayed a third time.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let values_atom = builder.borrow_mut().add_input(ArrayType::new_static(DataType::F64, [2, 3]));
        let carry = parent.tracer(carry_atom, None);
        let values = parent.tracer(values_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(carry), ArrayBatch::new(values, BatchAxis::new(0)).unwrap()];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = TestScanOperation::new(1, 3).batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
    }

    #[test]
    fn test_scan_batching_infers_zero_length_mapped_and_replicated_outputs_while_tracing() {
        // A zero-length scan batched under a staging parent still batches its body structurally: the carry keeps its
        // mapped axis, the stacked carry output is mapped behind the leading scan dimension, and the stacked constant
        // output stays replicated, for explicit and manual mesh axes alike.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(carry_sharding.clone()).unwrap();
            let stack_type =
                ArrayType::new_static(DataType::F64, [0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let parent = TracingContext::<Array, TestOperation>::new();
            let builder = parent.builder().clone();
            let carry_atom = builder.borrow_mut().add_input(carry_type.clone());
            let stack_atom = builder.borrow_mut().add_input(stack_type.clone());
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
            let carries = ArrayBatch::new(parent.tracer(carry_atom, None), BatchAxis::new(0)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(parent.tracer(stack_atom, None));
            // The body's boundary types derive from the carry's unbatched per-item type (like a traced-over-inputs
            // body would), so its metadata — including any varying-manual-axes marker — matches the actual carries.
            let logical_type = carries.unbatched_type();
            let tracer_inputs =
                [BatchingTracer::new(context.clone(), carries), BatchingTracer::new(context.clone(), stacked_inputs)];
            let outputs = context
                .bind(
                    TestOperation::Scan(TestScanOperation::new(1, 0)),
                    [zero_length_body(logical_type, scan_slice_type(&stack_type, 0).unwrap().0)],
                    &tracer_inputs,
                )
                .unwrap();
            let output_axes = outputs.iter().map(|output| output.batch().batch_axis()).collect::<Vec<_>>();
            let output_atoms =
                outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
            drop(outputs);
            drop(tracer_inputs);
            drop(context);
            drop(parent);

            let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
            let program = builder
                .build::<Vec<Array>, Vec<Array>>(
                    output_atoms,
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder, Placeholder],
                )
                .unwrap();
            let expected = if axis_type == MeshAxisType::Explicit {
                indoc! {"
                    lambda %0:f64[2][sharding={mesh<['x'=2:explicit]>, [{'x'}]}], \
                    %1:f64[0][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %2:f64[2][sharding={mesh<['x'=2:explicit]>, [{'x'}]}], %3:f64[0, \
                    2][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}], %4:f64[0][sharding={mesh<['x'=2:explicit]>, \
                    [{}]}] = scan [carry_count=1, length=0, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[2][sharding={mesh<['x'=2:explicit]>, [{'x'}]}], \
                            %2:f64[][sharding={mesh<['x'=2:explicit]>, []}] .
                            let %3:f64[][sharding={mesh<['x'=2:explicit]>, []}] = const 7.0
                            in (%1, %1, %3)
                        },
                    ]
                    in (%0, %3, %4)
                "}
            } else {
                indoc! {"
                    lambda %0:f64[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}], \
                    %1:f64[0][sharding={mesh<['x'=2:manual]>, [{}]}] .
                    let %2:f64[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}], %3:f64[0, \
                    2][sharding={mesh<['x'=2:manual]>, [{}, {'x'}], varying_manual={'x'}}], \
                    %4:f64[0][sharding={mesh<['x'=2:manual]>, [{}]}] = scan [carry_count=1, \
                    length=0, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[2][sharding={mesh<['x'=2:manual]>, [{'x'}], \
                            varying_manual={'x'}}], %2:f64[][sharding={mesh<['x'=2:manual]>, []}] .
                            let %3:f64[][sharding={mesh<['x'=2:manual]>, []}] = const 7.0
                            in (%1, %1, %3)
                        },
                    ]
                    in (%0, %3, %4)
                "}
            };
            assert_eq!(program.to_string(), expected.trim_end());
            let output_types = program.output_types();

            assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(1), BatchAxis::replicated()]);
            assert_eq!(output_types[0].shape().dimensions(), &[Dimension::Static(2)]);
            assert_eq!(output_types[0].sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(output_types[0].sharding().unwrap().varying_manual_axes(), carry_sharding.varying_manual_axes());
            // The staged batched scan's stacked outputs stack the per-iteration output types of its body, so they keep
            // the sharding of the batched slices with a replicated stacked dimension, as eager batching does (refer to
            // `test_scan_batching_preserves_stacked_output_batch_placement`).
            assert_eq!(output_types[1].shape().dimensions(), &[Dimension::Static(0), Dimension::Static(2)]);
            assert_eq!(
                output_types[1].sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(output_types[1].sharding().unwrap().varying_manual_axes(), carry_sharding.varying_manual_axes());
            assert_eq!(output_types[2].shape().dimensions(), &[Dimension::Static(0)]);
            assert_eq!(output_types[2].sharding().unwrap().dimensions(), &[ShardingDimension::replicated()]);
            assert!(output_types[2].sharding().unwrap().varying_manual_axes().is_empty());
        }
    }

    #[test]
    fn test_scan_batching_infers_zero_length_mapped_and_replicated_outputs_eagerly() {
        // A zero-length scan batched under an eager parent runs no iteration, but batching its body structurally still
        // determines which empty stacked outputs are mapped and where their packed batch dimensions and shardings live,
        // for explicit and manual mesh axes alike.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries =
                ArrayBatch::new(Array::from_elements::<f64>(carry_type, &[1.0, 2.0]).unwrap(), BatchAxis::new(0))
                    .unwrap();
            let stack_type =
                ArrayType::new_static(DataType::F64, [0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(Array::from_elements::<f64>(stack_type, &[]).unwrap());
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = batch_scan(
                &context,
                TestScanOperation::new(1, 0),
                zero_length_body(carries.unbatched_type(), logical_type),
                vec![carries, stacked_inputs],
            );

            assert_eq!(outputs.len(), 3);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(2)]);
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(
                outputs[0].r#type().sharding().unwrap().varying_manual_axes(),
                carry_sharding.varying_manual_axes(),
            );
            assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(0), Dimension::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().varying_manual_axes(),
                carry_sharding.varying_manual_axes(),
            );
            assert!(outputs[1].value().storage_bytes().is_empty());
            assert_eq!(outputs[2].batch_axis(), BatchAxis::replicated());
            assert_eq!(outputs[2].r#type().shape().dimensions(), &[Dimension::Static(0)]);
            assert_eq!(outputs[2].r#type().sharding().unwrap().dimensions(), &[ShardingDimension::replicated()],);
            assert!(outputs[2].r#type().sharding().unwrap().varying_manual_axes().is_empty());
            assert!(outputs[2].value().storage_bytes().is_empty());
        }
    }

    #[test]
    fn test_scan_batching_preserves_stacked_output_batch_placement() {
        // Eager batching keeps the sharding of a mapped carry and gives each stacked output the sharding of its mapped
        // per-iteration slices with a replicated leading scan dimension, for explicit and manual mesh axes alike.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            // Batch-axis replication is independent of variation across manual mesh shards. Both operands already
            // vary over the manual axis so the eager body can multiply them without entering a manual region.
            let logical_sharding = Sharding::replicated(mesh.clone(), 0)
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let logical_type = ArrayType::scalar(DataType::F64).with_sharding(logical_sharding).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries =
                ArrayBatch::new(Array::from_elements::<f64>(carry_type, &[1.0, 2.0]).unwrap(), BatchAxis::new(0))
                    .unwrap();
            let stack_sharding = Sharding::replicated(mesh, 1)
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let stack_type = ArrayType::new_static(DataType::F64, [3]).with_sharding(stack_sharding).unwrap();
            let stacked_inputs =
                ArrayBatch::replicated(Array::from_elements::<f64>(stack_type, &[2.0, 3.0, 4.0]).unwrap());
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64));
            let carry = builder.add_input(carries.unbatched_type());
            let value = builder.add_input(logical_type);
            let next = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, value], None).unwrap()[0];
            let body = builder.build(vec![next, next], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();
            let outputs = batch_scan(&context, TestScanOperation::new(1, 3), body, vec![carries, stacked_inputs]);

            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(
                outputs[0].r#type().sharding().unwrap().varying_manual_axes(),
                carry_sharding.varying_manual_axes(),
            );
            assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 48.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(3), Dimension::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 4.0, 6.0, 12.0, 24.0, 48.0]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().varying_manual_axes(),
                carry_sharding.varying_manual_axes(),
            );
        }
    }

    #[test]
    fn test_scan_batching_threads_reference_carries() {
        // A reference carry threads positionally with the batch axis of the allocation that produced it, and the body
        // accumulates each item's own elements into it, so batching the stateful scan directly agrees with batching
        // its discharged counterpart: the stacked snapshots gain the scan axis in front of the batch axis and the
        // frozen final state stays at the referent's axis.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.into());
        let element = body_builder.add_input(scalar_type.clone().into());
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element], None)
            .unwrap();
        let current = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, current],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference, elements], None)
            .unwrap();
        let final_reference = outputs[0];
        let stacked = outputs[1];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![stacked, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let axis_extent = DimensionValue::constant(2).unwrap();
        let extent_type = axis_extent.r#type().into_owned();
        let direct = source
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let discharged = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(direct.output_axes(), &[BatchAxis::new(1), BatchAxis::new(0)]);
        assert_eq!(discharged.output_axes(), direct.output_axes());
        let direct = direct.into_parts().0;
        let discharged = discharged.into_parts().0;
        assert_eq!(
            direct.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2], %2:f32[2, 3] .
                let %3:ref<f32[2]> = reference_new %1
                    %4:f32[3, 2] = transpose [permutation=[1, 0]] %2
                    %5:dimension<2>, %6:ref<f32[2]>, %7:f32[3, 2] = scan [carry_count=2, length=3, reverse=false] \
                %0 %3 %4 [
                        body={
                            lambda %0:i64[], %1:dimension<2>, %2:ref<f32[2]>, %3:f32[2] .
                            let () = reference_add_update %2 %3
                                %4:f32[2] = reference_read %2
                            in (%1, %2, %4)
                        },
                    ]
                    %8:f32[2] = reference_freeze %6
                in (%0, %7, %8)"},
        );
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2], %2:f32[2, 3] .
                let %3:f32[3, 2] = transpose [permutation=[1, 0]] %2
                    %4:dimension<2>, %5:f32[2], %6:f32[3, 2] = scan [carry_count=2, length=3, reverse=false] %0 %1 \
                %3 [
                        body={
                            lambda %0:i64[], %1:dimension<2>, %2:f32[2], %3:f32[2] .
                            let %4:f32[2] = add %2 %3
                            in (%1, %4, %4)
                        },
                    ]
                in (%0, %6, %5)"},
        );
        let inputs = vec![
            TestIrValue::Dimension(axis_extent.clone()),
            TestIrValue::Array(Array::vector(vec![10.0f32, 20.0]).unwrap()),
            TestIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestIrValue::Dimension(axis_extent),
            TestIrValue::Array(Array::matrix(3, 2, vec![11.0f32, 21.0, 13.0, 23.0, 16.0, 26.0]).unwrap()),
            TestIrValue::Array(Array::vector(vec![16.0f32, 26.0]).unwrap()),
        ];
        assert_eq!(direct.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_batching_threads_reference_stacks() {
        // Batching the elements behind the scan axis packs the stack as `ref<f32[3, 2]>` at axis 1, so each
        // per-iteration view is `ref<f32[2]>` at axis 0 and the batched carry accumulates into its own item.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let extent_type = axis_extent.r#type().into_owned();
        let batched = stacked_reference_program()
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(1)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(1)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2], %2:f32[3, 2] .
                let %3:ref<f32[3, 2]> = reference_new %2
                    %4:dimension<2>, %5:f32[2] = scan [carry_count=2, length=3, reverse=false] %0 %1 %3 [
                        body={
                            lambda %0:i64[], %1:dimension<2>, %2:f32[2], %3:ref<f32[3, 2]> .
                            let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %3 %2 %0
                                %4:f32[2] = reference_read [transforms=[index(axis=0, index=dynamic)]] %3 %0
                                %5:f32[2] = add %2 %4
                            in (%1, %5)
                        },
                    ]
                    %6:f32[3, 2] = reference_freeze %3
                in (%0, %5, %6)"},
        );
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                TestIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::vector(vec![19.0f32, 48.0]).unwrap()),
                TestIrValue::Array(Array::matrix(3, 2, vec![2.0f32, 6.0, 5.0, 13.0, 11.0, 27.0]).unwrap()),
            ]),
        );

        // A replicated stack stays replicated (only a reference-typed program input can be replicated, since an
        // allocation is always batched): every item reads the same per-iteration view into its own carry.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(stack_reading_body());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let reading = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let batched = reading
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            batched.into_parts().0.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                TestIrValue::Reference(stack.clone()),
            ]),
            Ok(vec![
                TestIrValue::Dimension(axis_extent),
                TestIrValue::Array(Array::vector(vec![7.0f32, 8.0]).unwrap()),
            ]),
        );
        assert_eq!(stack.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));

        // A stack batched on its scan axis would need the body view to index the second axis of the packed referent,
        // which the scan cannot express, and a reference cannot be realigned.
        assert!(matches!(
            stacked_reference_program().batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`scan` batching found the reference-typed stacked input at position 1 batched on its \
                               scan axis; a reference stack keeps its batch axis and must be batched at an axis behind \
                               its leading scan axis",
        ));
    }

    #[test]
    fn test_scan_differentiation_rejects_missing_body_index_and_carry_outputs() {
        let context = DifferentiationContext::<TestEagerContext>::new(TestEagerContext::new());
        let body = ProgramBuilder::<Array, TestOperation>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            TestScanOperation::new(0, 1).jvp(&context, &ScanDifferentiationDriver(&body), &[]).unwrap_err(),
            DifferentiationError::from(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );

        let body = scalar_body(1, |_, _| Vec::new());
        let carry = DifferentiationDual::new(Array::scalar(1f64).unwrap(), Array::scalar(1f64).unwrap()).unwrap();
        assert_eq!(
            TestScanOperation::new(1, 1).jvp(&context, &ScanDifferentiationDriver(&body), &[carry]).unwrap_err(),
            DifferentiationError::from(TypeError::invalid("`scan` carry count 1 exceeds the body output count 0")),
        );
    }

    #[test]
    fn test_scan_differentiation_rejects_a_mismatched_primal_carry() {
        let body = scalar_body(1, |_, inputs| vec![inputs[1]]);
        let context = DifferentiationContext::<TestEagerContext>::new(TestEagerContext::new());
        let carry = DifferentiationDual::new(Array::scalar(1f32).unwrap(), Array::scalar(1f32).unwrap()).unwrap();
        assert_eq!(
            TestScanOperation::new(1, 1).jvp(&context, &ScanDifferentiationDriver(&body), &[carry]).unwrap_err(),
            DifferentiationError::from(TypeError::invalid(
                "`scan` input 0 has type `f32[]`, which does not refine its expected type `f64[]`",
            )),
        );
    }

    #[test]
    fn test_scan_differentiation_composite_validates_nominal_runtime_geometry_before_forwarding_carries() {
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(ArrayType::scalar(DataType::F64).into());
        builder.add_input(ArrayType::scalar(DataType::F64).into());
        let body = builder.build(vec![carry], vec![Placeholder; 3], vec![Placeholder; 1]).unwrap();
        let operation = ScanOperation::new(1, Dimension::Dynamic(length.clone()));
        let primals = |extent| {
            vec![
                array(Array::scalar(2f64).unwrap()),
                array(Array::vector(vec![3f64, 4.0, 5.0]).unwrap()),
                dimension(&DimensionType::from(length.clone()), extent),
            ]
        };

        // The nominal dimension's concrete extent agrees with the concrete stack, although its type alone cannot
        // prove that equality. The unused stack has a structural-zero tangent, so only the known scan validates its
        // original geometry before returning the unchanged carry; the tangent body needs no stack coefficient.
        let (outputs, pushforward) = differentiate_at(primals(3))
            .linearize(|inputs: Vec<TestIrTracer>| {
                let context = inputs[0].context();
                let mut duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();
                duals[1] = DifferentiationDual::new_with_zero_tangent(inputs[1].primal().clone())?;
                Ok(operation
                    .jvp(context, &ScanDifferentiationDriver(&body), &duals)?
                    .into_iter()
                    .map(|output| DifferentiationTracer::new(output, context.clone()))
                    .collect::<Vec<_>>())
            })
            .unwrap();
        assert_eq!(outputs, vec![array(Array::scalar(2f64).unwrap())]);
        assert_eq!(
            pushforward.program().to_string(),
            "lambda %0:f64[], %1:f64[3], %2:dimension<length ∈ [1, 4)> .\nin (%0)",
        );

        let error = differentiate_at(primals(2))
            .linearize(|inputs: Vec<TestIrTracer>| {
                let context = inputs[0].context();
                let mut duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();
                duals[1] = DifferentiationDual::new_with_zero_tangent(inputs[1].primal().clone())?;
                Ok(operation
                    .jvp(context, &ScanDifferentiationDriver(&body), &duals)?
                    .into_iter()
                    .map(|output| DifferentiationTracer::new(output, context.clone()))
                    .collect::<Vec<_>>())
            })
            .err()
            .unwrap();
        assert_eq!(
            error,
            DifferentiationError::from(TypeError::invalid(
                "`scan` runtime length input has type `dimension<2>` but stacked input 1 has type \
                 `f64[3]` whose leading dimension is not refined to extent 2",
            )),
        );
    }

    #[test]
    fn test_scan_differentiation() {
        // Forward mode stages one fused scan whose carries and body boundaries pair every primal entry that has a live
        // tangent with that tangent.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(TestScanOperation::new(1, 3), product_body(), vec![scalar_type, stacked_type]);
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                let %4:f64[], %5:f64[], %6:f64[3], %7:f64[3] = scan [carry_count=2, length=3, \
                 reverse=false] %0 %2 %1 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %1 %3
                            %6:f64[] = mul %3 %2
                            %7:f64[] = mul %1 %4
                            %8:f64[] = add %6 %7
                        in (%5, %8, %5, %8)
                    },
                ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );

        // For the cumulative product over `[2, 3, 4]` starting at `1`, a unit tangent on the initial carry propagates
        // as `∂(initial · x_0 · x_1 · x_2)/∂initial = 24` on the final carry and as the running products on the stacked
        // outputs, while a unit tangent on `x_1` propagates as `initial · x_0 · x_2 = 8` on the final carry and as
        // `[0, 2, 8]` on the stacked outputs (`y_0` does not depend on `x_1`).
        let primals = (Array::scalar(1.0).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        /// Applies a cumulative-product scan and returns its final carry and stacked intermediate products.
        fn function<V: Value<Type = ArrayType>>((initial, values): (V, V)) -> Result<(V, V), ProgramError>
        where
            V::Domain: Context<Type = ArrayType, Constant = Array, Operation = TestOperation>,
        {
            let mut outputs = bind_scan(TestScanOperation::new(1, 3), product_body(), &[initial, values])?;
            let stacked = outputs.remove(1);
            Ok((outputs.remove(0), stacked))
        }
        let outputs = (Array::scalar(24.0).unwrap(), Array::vector(vec![2.0, 6.0, 24.0]).unwrap());
        assert_eq!(
            differentiate_at(primals.clone())
                .jvp((Array::scalar(1.0).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap()), function),
            Ok((outputs.clone(), (Array::scalar(24.0).unwrap(), Array::vector(vec![2.0, 6.0, 24.0]).unwrap()))),
        );
        assert_eq!(
            differentiate_at(primals.clone())
                .jvp((Array::scalar(0.0).unwrap(), Array::vector(vec![0.0, 1.0, 0.0]).unwrap()), function),
            Ok((outputs.clone(), (Array::scalar(8.0).unwrap(), Array::vector(vec![0.0, 2.0, 8.0]).unwrap()))),
        );

        // Linearization produces the same pushforward, and reverse mode transposes it: the gradient of the final carry
        // is `initial`'s cofactor `24` and the products of the other slices, `[12, 8, 6]`.
        let (linearized_outputs, pushforward) = differentiate_at(primals.clone()).linearize(function).unwrap();
        assert_eq!(linearized_outputs, outputs);
        assert_eq!(
            pushforward.apply((Array::scalar(1.0).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap())),
            Ok((Array::scalar(24.0).unwrap(), Array::vector(vec![2.0, 6.0, 24.0]).unwrap())),
        );
        let (final_carry, pullback) = differentiate_at(primals)
            .vjp(|(initial, values): (TestTracer, TestTracer)| apply_product_scan(initial, values))
            .unwrap();
        assert_eq!(final_carry, Array::scalar(24.0).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(1.0).unwrap()),
            Ok((Array::scalar(24.0).unwrap(), Array::vector(vec![12.0, 8.0, 6.0]).unwrap())),
        );

        // Finite differences independently check the derivatives through every slice of the nonlinear recurrence.
        check_gradient!(
            |values| {
                let initial = values.domain().lift(Array::scalar(1f64).unwrap())?;
                apply_product_scan(initial, values)
            },
            at = Array::vector(vec![2f64, 3.0, 4.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_scan_differentiation_flows_through_reverse_scans() {
        // Forward- and reverse-mode differentiation flow through a `reverse` scan: the visit order flips while slice
        // `i` of every stacked value stays paired with iteration `i`. With reverse visit order, the carry runs
        // `2 · 5 = 10 → 10 · 4 = 40 → 40 · 3 = 120`, with `y_i` still paired with `x_i`.
        let function = |(carry, values): (TestTracer, TestTracer)| {
            let outputs = bind_scan(TestScanOperation::new(1, 3).with_reverse(true), product_body(), &[carry, values])?;
            Ok((outputs[0].clone(), outputs[1].clone()))
        };
        let primals = (Array::scalar(2.0).unwrap(), Array::vector(vec![3.0, 4.0, 5.0]).unwrap());
        let (outputs, pushforward) = differentiate_at(primals.clone()).linearize(function).unwrap();
        assert_eq!(
            pushforward.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                let %4:f64[], %5:f64[3] = scan [carry_count=1, length=3, reverse=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %3 %1
                            %6:f64[] = mul %4 %2
                            %7:f64[] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(outputs, (Array::scalar(120.0).unwrap(), Array::vector(vec![120.0, 40.0, 10.0]).unwrap()));

        // A pure carry tangent scales by the running product of the slices consumed up to each visit.
        assert_eq!(
            pushforward.apply((Array::scalar(1.0).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap())),
            Ok((Array::scalar(60.0).unwrap(), Array::vector(vec![60.0, 20.0, 5.0]).unwrap())),
        );

        // `∂(2 · 3 · 4 · 5)/∂carry = 60` and `∂/∂x = [40, 30, 24]`.
        let (final_carry, pullback) = differentiate_at(primals)
            .vjp(|(carry, values): (TestTracer, TestTracer)| {
                Ok(bind_scan(TestScanOperation::new(1, 3).with_reverse(true), product_body(), &[carry, values])?
                    .remove(0))
            })
            .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                let %4:f64[], %5:f64[3] = scan [carry_count=1, length=3, reverse=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %3 %1
                            %6:f64[] = mul %4 %2
                            %7:f64[] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.transposed_program(&[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3] .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = zero [type=f64[]]
                            %5:f64[] = add %1 %4
                            %6:f64[] = mul %2 %5
                            %7:f64[] = mul %3 %5
                        in (%6, %7)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(final_carry, Array::scalar(120.0).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(1.0).unwrap()),
            Ok((Array::scalar(60.0).unwrap(), Array::vector(vec![40.0, 30.0, 24.0]).unwrap())),
        );
    }

    #[test]
    fn test_scan_differentiation_stages_one_fused_scan_without_residual_stacks() {
        // The fused forward-mode rule stages exactly one scan with doubled carries and no per-iteration residual stacks
        // (refer to `test_scan_differentiation`), so pure forward mode pays a single loop pass and no reverse-mode
        // storage. Residual stacks appear only when linearization separates the primal and tangent programs: its known
        // scan then stacks the per-iteration known-to-unknown edges that the tangent scan consumes.
        let program = scan_program(
            TestScanOperation::new(1, 3),
            product_body(),
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [3])],
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3], %4:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%3, %3, %1)
                    },
                ]
                in (%2, %3, %1, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                let %4:f64[], %5:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %3 %1
                            %6:f64[] = mul %4 %2
                            %7:f64[] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_differentiation_preserves_reverse_and_unroll() {
        // Every scan that forward mode stages keeps the visit order and unroll factor of the source scan: the fused
        // scan of the JVP and both the primal and the tangent scans of a linearization.
        let program = scan_program(
            TestScanOperation::new(1, 3).with_reverse(true).with_unroll(2).unwrap(),
            product_body(),
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [3])],
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                let %4:f64[], %5:f64[], %6:f64[3], %7:f64[3] = scan [carry_count=2, length=3, reverse=true, \
                 unroll=2] %0 %2 %1 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %1 %3
                            %6:f64[] = mul %3 %2
                            %7:f64[] = mul %1 %4
                            %8:f64[] = add %6 %7
                        in (%5, %8, %5, %8)
                    },
                ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3], %4:f64[3] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                        in (%3, %3, %1)
                    },
                ]
                in (%2, %3, %1, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                let %4:f64[], %5:f64[3] = scan [carry_count=1, length=3, reverse=true, unroll=2] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %3 %1
                            %6:f64[] = mul %4 %2
                            %7:f64[] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_differentiation_zero_length() {
        // A zero-length scan returns its initial carries and empty stacked outputs, so the carry tangents and
        // cotangents pass through unchanged and the stacked tangents and cotangents are empty.
        let empty = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap();
        let program = scan_program(
            TestScanOperation::new(1, 0),
            product_body(),
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [0])],
        );
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[0], %2:f64[], %3:f64[0] .
                let %4:f64[], %5:f64[], %6:f64[0], %7:f64[0] = scan [carry_count=2, length=0, \
                 reverse=false] %0 %2 %1 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %1 %3
                            %6:f64[] = mul %3 %2
                            %7:f64[] = mul %1 %4
                            %8:f64[] = add %6 %7
                        in (%5, %8, %5, %8)
                    },
                ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![Array::scalar(2.0).unwrap(), empty.clone(), Array::scalar(3.0).unwrap(), empty.clone()]),
            Ok(vec![Array::scalar(2.0).unwrap(), empty.clone(), Array::scalar(3.0).unwrap(), empty.clone()]),
        );
        let (final_carry, pullback) = differentiate_at((Array::scalar(2.0).unwrap(), empty.clone()))
            .vjp(|(initial, values): (TestTracer, TestTracer)| {
                Ok(bind_scan(TestScanOperation::new(1, 0), product_body(), &[initial, values])?.remove(0))
            })
            .unwrap();
        assert_eq!(final_carry, Array::scalar(2.0).unwrap());
        assert_eq!(pullback.apply(Array::scalar(5.0).unwrap()), Ok((Array::scalar(5.0).unwrap(), empty)));
    }

    #[test]
    fn test_scan_differentiation_does_not_hoist_work_from_zero_length_scans() {
        // A residual computed from an invariant carry still belongs to the body when no iteration can run.
        // Linearization must keep the multiplication inside the primal scan instead of evaluating it eagerly.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = scalar_body(2, |builder, inputs| {
            let [_, scale, value]: [AtomId; 3] = inputs.try_into().unwrap();
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            vec![scale, scaled]
        });
        let program = scan_program(
            TestScanOperation::new(1, 0),
            body,
            vec![scalar_type, ArrayType::new_static(DataType::F64, [0])],
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[0] .
                let %2:f64[], %3:f64[0], %4:f64[0] = scan [carry_count=1, length=0, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %1
                            %4:f64[] = mul %3 %2
                        in (%1, %4, %3)
                    },
                ]
                in (%0, %3, %0, %1, %4)
            "}
            .trim_end(),
        );
        let empty = Array::vector(Vec::<f64>::new()).unwrap();
        assert_eq!(
            linearization.primal().interpret(vec![Array::scalar(2.0).unwrap(), empty.clone()]),
            Ok(vec![Array::scalar(2.0).unwrap(), empty.clone(), Array::scalar(2.0).unwrap(), empty.clone(), empty]),
        );
    }

    #[test]
    fn test_scan_differentiation_dynamic_length() {
        // A dynamic-length composite scan differentiates through its fused scan like a static one, threading its
        // trailing runtime length input, which has no tangent. The body maps `[carry, x]` to `[carry * x, carry * x]`.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone());
        let value = body.add_input(scalar_type.clone());
        let product = body
            .add_instruction(
                TestIrOperation::Array(TestOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![carry, value],
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![product, product],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs = [
            scalar_type,
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]))),
            ArrayIrType::Dimension(DimensionType::from(length.clone())),
        ]
        .map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                inputs.to_vec(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[length], %2:dimension<length ∈ [1, 8)>, %3:f32[], %4:f32[length] .
                let %5:f32[], %6:f32[], %7:f32[length], %8:f32[length] = scan [carry_count=2, length=length, \
                 reverse=false] %0 %3 %1 %4 %2 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[], %3:f32[], %4:f32[] .
                        let %5:f32[] = mul %1 %3
                            %6:f32[] = mul %3 %2
                            %7:f32[] = mul %1 %4
                            %8:f32[] = add %6 %7
                        in (%5, %8, %5, %8)
                    },
                ]
                in (%5, %7, %6, %8)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 3.0, 4.0]).unwrap()),
                TestIrValue::Dimension(DimensionValue::new(DimensionType::from(length), 3).unwrap()),
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![0.0f32, 1.0, 0.0]).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(24.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 6.0, 24.0]).unwrap()),
                TestIrValue::Array(Array::scalar(32.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 8.0, 32.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_does_not_hoist_work_from_dynamic_lengths_that_admit_zero() {
        // The runtime count can be zero or positive without rebuilding the linearization. Computed coefficients
        // therefore stay in the primal body and emerge as residual stacks with the same runtime leading extent.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(4)).unwrap());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let stacked_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(length.clone())])));
        let length_type = ArrayIrType::Dimension(DimensionType::from(length.clone()));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let scale = builder.add_input(scalar_type.clone());
        let item = builder.add_input(scalar_type.clone());
        let squared = builder
            .add_instruction(
                TestIrOperation::Array(TestOperation::from(MulOperation::new())),
                Vec::new(),
                vec![scale, scale],
                None,
            )
            .unwrap()[0];
        let scaled = builder
            .add_instruction(
                TestIrOperation::Array(TestOperation::from(MulOperation::new())),
                Vec::new(),
                vec![squared, item],
                None,
            )
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![scale, scaled],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let inputs = [scalar_type, stacked_type, length_type.clone()].map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length.clone())),
                vec![body],
                inputs.to_vec(),
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[length], %2:dimension<length ∈ [0, 4)> .
                let %3:f64[], %4:f64[length], %5:f64[length] = scan [carry_count=1, length=length, reverse=false] \
                    %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %1
                            %4:f64[] = mul %3 %2
                        in (%1, %4, %3)
                    },
                ]
                in (%0, %4, %0, %1, %5, %2)
            "}
            .trim_end(),
        );
        let scale = array(Array::scalar(2.0f64).unwrap());
        let empty = array(Array::vector(Vec::<f64>::new()).unwrap());
        let zero_length = dimension(&DimensionType::from(length.clone()), 0);
        assert_eq!(
            linearization.primal().interpret(vec![scale.clone(), empty.clone(), zero_length.clone()]),
            Ok(vec![scale.clone(), empty.clone(), scale.clone(), empty.clone(), empty, zero_length]),
        );
        let items = array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let positive_length = dimension(&DimensionType::from(length), 2);
        assert_eq!(
            linearization.primal().interpret(vec![scale.clone(), items.clone(), positive_length.clone()]),
            Ok(vec![
                scale.clone(),
                array(Array::vector(vec![12.0f64, 16.0]).unwrap()),
                scale,
                items,
                array(Array::vector(vec![4.0f64, 4.0]).unwrap()),
                positive_length,
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_skips_tangent_slots_of_carries_with_zero_tangents() {
        // The body maps `[counter, x]` to `[counter + 1, counter * x]` and the counter starts at a constant, so its
        // tangent is a structural zero that the body never makes live. The fused scan therefore carries no tangent
        // for the counter, while the stacked output still receives the tangent `counter · ẋ`.
        let body = scalar_body(2, |builder, inputs| {
            let one = builder.add_constant(Array::scalar(1.0f64).unwrap());
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], one], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, scaled]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let values = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let counter = builder.add_constant(Array::scalar(0.0f64).unwrap());
        let outputs = builder
            .add_instruction(TestScanOperation::new(1, 3), vec![body], vec![counter, values], None)
            .unwrap()
            .to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder; 2]).unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[] = const 0.0
                    %3:f64[], %4:f64[3], %5:f64[3] = scan [carry_count=1, length=3, reverse=false] %2 %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = const 1.0
                                %5:f64[] = add %1 %4
                                %6:f64[] = mul %1 %2
                                %7:f64[] = mul %1 %3
                            in (%5, %6, %7)
                        },
                    ]
                    %6:f64[] = zero [type=f64[]]
                in (%3, %4, %6, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::vector(vec![2.0, 3.0, 4.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
            ]),
            Ok(vec![
                Array::scalar(3.0).unwrap(),
                Array::vector(vec![0.0, 3.0, 8.0]).unwrap(),
                Array::scalar(0.0).unwrap(),
                Array::vector(vec![0.0, 1.0, 2.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_does_not_materialize_zero_tangents_of_stacked_inputs() {
        // The stacked input of the cumulative-product scan is a constant, so its tangent is a structural zero. The
        // fused body receives no tangent for it, and no zero stack is materialized for the fused scan to consume.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(product_body());
        let initial = builder.add_input(ArrayType::scalar(DataType::F64));
        let values = builder.add_constant(Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        let outputs = builder
            .add_instruction(TestScanOperation::new(1, 3), vec![body], vec![initial, values], None)
            .unwrap()
            .to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder; 2]).unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[3] = const [2.0, 3.0, 4.0]
                    %3:f64[], %4:f64[], %5:f64[3], %6:f64[3] = scan [carry_count=2, length=3, reverse=false] %0 %1 %2 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = mul %1 %3
                                %5:f64[] = mul %3 %2
                            in (%4, %5, %4, %5)
                        },
                    ]
                in (%3, %5, %4, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![Array::scalar(1.0).unwrap(), Array::scalar(1.0).unwrap()]),
            Ok(vec![
                Array::scalar(24.0).unwrap(),
                Array::vector(vec![2.0, 6.0, 24.0]).unwrap(),
                Array::scalar(24.0).unwrap(),
                Array::vector(vec![2.0, 6.0, 24.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_enlivens_carries_whose_tangents_become_live() {
        // The body maps `[carry, x]` to `[carry + x, carry]` and the carry starts at a constant, so its initial tangent
        // is a structural zero. The next carry depends on the live tangent of `x`, so the carry acquires a live tangent
        // after the first iteration and needs a tangent slot (seeded with a zero) from the first iteration on.
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, inputs[1]]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let values = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let initial = builder.add_constant(Array::scalar(0.0f64).unwrap());
        let outputs = builder
            .add_instruction(TestScanOperation::new(1, 3), vec![body], vec![initial, values], None)
            .unwrap()
            .to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder; 2]).unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[] = const 0.0
                    %3:f64[] = zero [type=f64[]]
                    %4:f64[], %5:f64[], %6:f64[3], %7:f64[3] = scan [carry_count=2, length=3, \
                     reverse=false] %2 %3 %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                            let %5:f64[] = add %1 %3
                                %6:f64[] = add %2 %4
                            in (%5, %6, %1, %2)
                        },
                    ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
            ]),
            Ok(vec![
                Array::scalar(6.0).unwrap(),
                Array::vector(vec![0.0, 1.0, 3.0]).unwrap(),
                Array::scalar(3.0).unwrap(),
                Array::vector(vec![0.0, 1.0, 2.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_leaves_independent_stacked_output_tangents_symbolic() {
        // The body maps `[counter, carry, x]` to `[counter + 1, carry * x, counter]` and the counter starts at a
        // constant. The stacked counter output depends on no tangent input, so it leaves the fused scan as a symbolic
        // zero tangent, which is only materialized at the program boundary, instead of as a stacked fused output.
        let body = scalar_body(3, |builder, inputs| {
            let one = builder.add_constant(Array::scalar(1.0f64).unwrap());
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], one], None).unwrap()[0];
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[2], inputs[3]], None).unwrap()[0];
            vec![next, product, inputs[1]]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(ArrayType::scalar(DataType::F64));
        let values = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let counter = builder.add_constant(Array::scalar(0.0f64).unwrap());
        let outputs = builder
            .add_instruction(TestScanOperation::new(2, 3), vec![body], vec![counter, initial, values], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[1], outputs[2]], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                let %4:f64[] = const 0.0
                    %5:f64[], %6:f64[], %7:f64[], %8:f64[3] = scan [carry_count=3, length=3, \
                     reverse=false] %4 %0 %2 %1 %3 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                            let %6:f64[] = const 1.0
                                %7:f64[] = add %1 %6
                                %8:f64[] = mul %2 %4
                                %9:f64[] = mul %4 %3
                                %10:f64[] = mul %2 %5
                                %11:f64[] = add %9 %10
                            in (%7, %8, %11, %1)
                        },
                    ]
                    %9:f64[3] = zero [type=f64[3]]
                in (%6, %8, %7, %9)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::scalar(1.0).unwrap(),
                Array::vector(vec![2.0, 3.0, 4.0]).unwrap(),
                Array::scalar(1.0).unwrap(),
                Array::vector(vec![0.0, 0.0, 0.0]).unwrap(),
            ]),
            Ok(vec![
                Array::scalar(24.0).unwrap(),
                Array::vector(vec![0.0, 1.0, 2.0]).unwrap(),
                Array::scalar(24.0).unwrap(),
                Array::vector(vec![0.0, 0.0, 0.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_forwards_passed_through_carries() {
        // The body maps `[scale, accumulator, x]` to `[scale, accumulator + scale * x]`. Its carry `scale` passes
        // through unchanged, so the scan's final `scale` and its tangent are forwarded from the input dual: a live
        // tangent is the input tangent itself, and a structural-zero tangent (here of a constant `scale`) stays a
        // symbolic zero that is only materialized at the program boundary.
        let body = scalar_body(3, |builder, inputs| {
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[3]], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[2], scaled], None).unwrap()[0];
            vec![inputs[1], next]
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body.clone(),
            vec![scalar_type.clone(), scalar_type.clone(), stacked_type.clone()],
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3], %3:f64[], %4:f64[], %5:f64[3] .
                let %6:f64[], %7:f64[], %8:f64[], %9:f64[] = scan [carry_count=4, length=3, \
                 reverse=false] %0 %1 %3 %4 %2 %5 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[], %6:f64[] .
                        let %7:f64[] = mul %1 %5
                            %8:f64[] = mul %5 %3
                            %9:f64[] = mul %1 %6
                            %10:f64[] = add %8 %9
                            %11:f64[] = add %2 %7
                            %12:f64[] = add %4 %10
                        in (%1, %11, %3, %12)
                    },
                ]
                in (%0, %7, %3, %9)
            "}
            .trim_end(),
        );

        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let scale = builder.add_constant(Array::scalar(2.0f64).unwrap());
        let initial = builder.add_input(scalar_type);
        let values = builder.add_input(stacked_type);
        let outputs = builder
            .add_instruction(TestScanOperation::new(2, 3), vec![body], vec![scale, initial, values], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                let %4:f64[] = const 2.0
                    %5:f64[], %6:f64[], %7:f64[] = scan [carry_count=3, length=3, reverse=false] %4 %0 %2 %1 %3 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                            let %6:f64[] = mul %1 %4
                                %7:f64[] = add %2 %6
                                %8:f64[] = mul %1 %5
                                %9:f64[] = add %3 %8
                            in (%1, %7, %9)
                        },
                    ]
                    %8:f64[] = zero [type=f64[]]
                in (%4, %6, %8, %7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_differentiation_hoists_loop_invariant_residuals() {
        // The body maps `[accumulator, scale, x]` to `[accumulator + (scale * scale) * x, scale]`. Linearization
        // computes the loop-invariant residual `scale * scale` derived from the passed-through carry `scale` once,
        // outside the scan, and threads it through the tangent scan as a passed-through carry instead of stacking a
        // copy of it per iteration, while the stacked residual `x` is read from the original stacked input.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = scalar_body(3, |builder, inputs| {
            let [_, accumulator, scale, value]: [AtomId; 4] = inputs.try_into().unwrap();
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, value], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, scaled], None).unwrap()[0];
            vec![next, scale]
        });
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body.clone(),
            vec![scalar_type.clone(), scalar_type, ArrayType::new_static(DataType::F64, [3])],
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3] .
                let %3:f64[], %4:f64[] = scan [carry_count=2, length=3, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %2 %2
                            %5:f64[] = mul %4 %3
                            %6:f64[] = add %1 %5
                        in (%6, %2)
                    },
                ]
                    %5:f64[] = mul %1 %1
                in (%3, %1, %1, %5, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3], %3:f64[], %4:f64[], %5:f64[3] .
                let %6:f64[], %7:f64[], %8:f64[], %9:f64[] = scan [carry_count=4, length=3, \
                 reverse=false] %3 %4 %0 %1 %2 %5 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[], %6:f64[] .
                        let %7:f64[] = mul %1 %4
                            %8:f64[] = mul %1 %4
                            %9:f64[] = add %7 %8
                            %10:f64[] = mul %6 %9
                            %11:f64[] = mul %2 %5
                            %12:f64[] = add %10 %11
                            %13:f64[] = add %3 %12
                        in (%1, %2, %13, %4)
                    },
                ]
                in (%8, %1)
            "}
            .trim_end(),
        );

        // With `accumulator = 1`, `scale = 2`, and `x = [1, 2, 3]`, the final accumulator is `1 + 4 · 6 = 25`, whose
        // gradient is `1` for the accumulator, `2 · scale · Σx = 24` for `scale`, and `scale² = 4` for every slice.
        let (output, pullback) = differentiate_at((
            Array::scalar(1.0).unwrap(),
            Array::scalar(2.0).unwrap(),
            Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
        ))
        .vjp(|(accumulator, scale, values): (TestTracer, TestTracer, TestTracer)| {
            Ok(bind_scan(TestScanOperation::new(2, 3), body, &[accumulator, scale, values])?.remove(0))
        })
        .unwrap();
        assert_eq!(output, Array::scalar(25.0).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(1.0).unwrap()),
            Ok((
                Array::scalar(1.0).unwrap(),
                Array::scalar(24.0).unwrap(),
                Array::vector(vec![4.0, 4.0, 4.0]).unwrap()
            )),
        );
    }

    #[test]
    fn test_scan_differentiation_stacks_time_varying_carry_residuals() {
        // The derivative of `exp` is its own output, so linearizing the body `[carry] -> [exp(carry)]` produces
        // residuals that are the carry's next value on every iteration. Reverse mode must store them per iteration
        // instead of reusing the carry's initial value: starting from `0`, the carries are `1`, `e`, and `e^e`, and
        // the derivative of the final carry is their product.
        let body = scalar_body(1, |builder, inputs| {
            builder
                .add_instruction(ExpOperation::<ArrayType>::new(), Vec::new(), vec![inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        let e = std::f64::consts::E;
        let (output, pullback) = differentiate_at(Array::scalar(0.0f64).unwrap())
            .vjp(|initial: TestTracer| Ok(bind_scan(TestScanOperation::new(1, 3), body, &[initial])?.remove(0)))
            .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.transposed_program(&[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(output.to_f64s(), vec![e.exp()]);
        let cotangent = pullback.apply(Array::scalar(1.0f64).unwrap()).unwrap().to_f64s()[0];
        assert!((cotangent - e * e.exp()).abs() < 1e-9, "{cotangent}");
    }

    #[test]
    fn test_scan_differentiation_preserves_manual_variation_of_residual_stacks() {
        // The carry changes on every iteration. Linearization stores its prior values in a stack that must keep
        // the carry's manual variation, and reverse mode slices those varying coefficients with matching geometry.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let scalar_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let stacked_type = scalar_type.with_inserted_dimension(0, Dimension::Static(2)).unwrap();
        let body = product_body_with_type(scalar_type.clone());
        let (_, program) = TracingContext::<Array, TestOperation>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, TestOperation>>>| {
                inputs[0].domain().bind(TestOperation::Scan(TestScanOperation::new(1, 2)), vec![body], &inputs)
            },
            vec![scalar_type.clone(), stacked_type.clone()],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let linearization = program.to_flat_program().linearize().unwrap();
        assert_eq!(
            linearization.primal().output_types(),
            vec![scalar_type.clone(), stacked_type.clone(), stacked_type.clone(), stacked_type.clone()],
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                    %1:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %2:f64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                    %3:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %4:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                            %2:f64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                        let %3:f64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %1 %2
                        in (%3, %3, %1)
                    },
                ]
                in (%2, %3, %1, %4)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.output_types(), vec![scalar_type.clone(), stacked_type.clone()]);
        let primal_outputs = linearization
            .primal()
            .interpret(vec![
                Array::from_elements(scalar_type.clone(), &[2.0f64]).unwrap(),
                Array::from_elements(stacked_type.clone(), &[3.0f64, 4.0]).unwrap(),
            ])
            .unwrap();
        assert_eq!(
            primal_outputs,
            vec![
                Array::from_elements(scalar_type.clone(), &[24.0f64]).unwrap(),
                Array::from_elements(stacked_type.clone(), &[6.0f64, 24.0]).unwrap(),
                Array::from_elements(stacked_type.clone(), &[3.0f64, 4.0]).unwrap(),
                Array::from_elements(stacked_type.clone(), &[2.0f64, 6.0]).unwrap(),
            ],
        );
        assert_eq!(
            pullback.interpret(vec![
                Array::from_elements(scalar_type.clone(), &[1.0f64]).unwrap(),
                Array::from_elements(stacked_type.clone(), &[0.0f64, 0.0]).unwrap(),
                primal_outputs[2].clone(),
                primal_outputs[3].clone(),
            ]),
            Ok(vec![
                Array::from_elements(scalar_type, &[12.0f64]).unwrap(),
                Array::from_elements(stacked_type, &[8.0f64, 6.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_with_zero_space_key_carry() {
        // A scan whose carries mix a differentiable accumulator with a zero-differential-space element: here a `u64`
        // key, the shape of every keyed training loop. The body maps `[accumulator, key, x]` to
        // `[accumulator * x, key, accumulator * x]`. The compact fused-JVP contract omits the key's tangent slot on
        // both the carry and the output boundaries, and reverse mode returns a typed zero-space cotangent for the key
        // input at the public boundary.
        let mut body = ProgramBuilder::<Array, TestOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64));
        let accumulator = body.add_input(ArrayType::scalar(DataType::F64));
        let key = body.add_input(ArrayType::scalar(DataType::U64));
        let value = body.add_input(ArrayType::scalar(DataType::F64));
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![accumulator, value], None).unwrap()[0];
        let body = body.build(vec![product, key, product], vec![Placeholder; 4], vec![Placeholder; 3]).unwrap();
        let program = scan_program(
            TestScanOperation::new(2, 3),
            body.clone(),
            vec![
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::U64),
                ArrayType::new_static(DataType::F64, [3]),
            ],
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:u64[], %2:f64[3], %3:f64[], %4:f64[3] .
                let %5:f64[], %6:u64[], %7:f64[], %8:f64[3], %9:f64[3] = scan [carry_count=3, length=3, \
                 reverse=false] %0 %1 %3 %2 %4 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:u64[], %3:f64[], %4:f64[], %5:f64[] .
                        let %6:f64[] = mul %1 %4
                            %7:f64[] = mul %4 %3
                            %8:f64[] = mul %1 %5
                            %9:f64[] = add %7 %8
                        in (%6, %2, %9, %6, %9)
                    },
                ]
                in (%5, %1, %8, %7, %9)
            "}
            .trim_end(),
        );

        // Reverse mode through the same scan: the accumulator and slice cotangents match the keyless product scan,
        // while the key input receives a typed zero-space cotangent.
        let ((output, stacked), pullback) = differentiate_at((
            Array::scalar(1.0).unwrap(),
            Array::scalar(7u64).unwrap(),
            Array::vector(vec![2.0, 3.0, 4.0]).unwrap(),
        ))
        .vjp(|(accumulator, key, values): (TestTracer, TestTracer, TestTracer)| {
            let mut outputs = bind_scan(TestScanOperation::new(2, 3), body, &[accumulator, key, values])?;
            let stacked = outputs.remove(2);
            Ok((outputs.remove(0), stacked))
        })
        .unwrap();
        assert_eq!(output, Array::scalar(24.0).unwrap());
        assert_eq!(stacked, Array::vector(vec![2.0, 6.0, 24.0]).unwrap());
        assert_eq!(
            pullback.apply((Array::scalar(1.0).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap())),
            Ok((
                Array::scalar(24.0).unwrap(),
                Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap(),
                Array::vector(vec![12.0, 8.0, 6.0]).unwrap(),
            )),
        );
    }

    #[test]
    fn test_scan_differentiation_supports_nested_scans() {
        // Nested scans differentiate by recursively replaying the inner scan inside each outer scan iteration. The
        // final carry is the product of every element, and a unit tangent on the initial carry follows the same
        // cumulative-product path through every scan level. Three levels cover a middle scan whose body contains
        // another scan whose body also has scan-local residuals.
        let (scan, body) = product_scan_with_lengths(&[2, 3]);
        assert_eq!(
            differentiate_at((
                Array::scalar(1.0).unwrap(),
                Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()
            ))
            .jvp(
                (Array::scalar(1.0).unwrap(), Array::matrix(2, 3, vec![0.0; 6]).unwrap()),
                |(initial, values)| {
                    let outputs = bind_scan(scan, body, &[initial, values])?;
                    Ok((outputs[0].clone(), outputs[1].clone()))
                },
            ),
            Ok((
                (
                    Array::scalar(5040.0).unwrap(),
                    Array::matrix(2, 3, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]).unwrap()
                ),
                (
                    Array::scalar(5040.0).unwrap(),
                    Array::matrix(2, 3, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]).unwrap()
                ),
            )),
        );

        let (scan, body) = product_scan_with_lengths(&[2, 2, 2]);
        let values_type = ArrayType::new_static(DataType::F64, [2, 2, 2]);
        let products = Array::from_elements::<f64>(
            values_type.clone(),
            &[2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0],
        )
        .unwrap();
        assert_eq!(
            differentiate_at((
                Array::scalar(1.0).unwrap(),
                Array::from_elements::<f64>(values_type.clone(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(),
            ))
            .jvp(
                (Array::scalar(1.0).unwrap(), Array::from_elements::<f64>(values_type, &[0.0; 8]).unwrap()),
                |(initial, values)| {
                    let outputs = bind_scan(scan, body, &[initial, values])?;
                    Ok((outputs[0].clone(), outputs[1].clone()))
                },
            ),
            Ok(((Array::scalar(362880.0).unwrap(), products.clone()), (Array::scalar(362880.0).unwrap(), products))),
        );
    }

    #[test]
    fn test_scan_differentiation_computes_dense_jacobians_and_hessians() {
        // For `f(initial, values) = initial · Π values`, the forward- and reverse-mode Jacobians agree on
        // `[Π values, initial · Π_{j ≠ i} values_j]`. Same-variable second derivatives vanish, mixed derivatives with
        // `initial` are the products excluding the corresponding value, and mixed derivatives between values are
        // `initial` times the remaining value.
        let primals = (Array::scalar(1.0).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        let forward = differentiate_at(primals.clone())
            .jacobian_forward(|(initial, values)| apply_product_scan(initial, values))
            .unwrap();
        let reverse = differentiate_at(primals.clone())
            .jacobian_reverse(|(initial, values)| apply_product_scan(initial, values))
            .unwrap();
        for jacobian in [forward, reverse] {
            assert_eq!(
                jacobian.iter_blocks().map(|block| block.value().clone()).collect::<Vec<_>>(),
                vec![Array::scalar(24.0).unwrap(), Array::vector(vec![12.0, 8.0, 6.0]).unwrap()],
            );
        }
        let hessian = TestEagerContext::new()
            .differentiate_at(primals)
            .hessian(|(initial, values)| apply_product_scan(initial, values))
            .unwrap();
        assert_eq!(
            hessian.iter_blocks().map(|block| block.value().to_f64s()).collect::<Vec<_>>(),
            vec![
                vec![0.0],
                vec![12.0, 8.0, 6.0],
                vec![12.0, 8.0, 6.0],
                vec![
                    0.0, 4.0, 3.0, //
                    4.0, 0.0, 2.0, //
                    3.0, 2.0, 0.0, //
                ],
            ],
        );
    }

    #[test]
    fn test_scan_differentiation_stages_reusable_reversed_scan_pullback() {
        // Reverse mode stages one reversed scan over the stacked residuals of the forward scan, so the pullback is a
        // reusable program that maps any seed to the corresponding cotangents. Only the final carry is differentiated,
        // so the stacked output's cotangent is a structural zero, which the reversed body constructs as a per-iteration
        // zero slice instead of reading a materialized zero stack.
        let (output, pullback) = TestEagerContext::new()
            .vjp(
                |(initial, values), ()| apply_product_scan(initial, values),
                (Array::scalar(1.0).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap()),
                (),
            )
            .unwrap();
        assert_eq!(output, Array::scalar(24.0).unwrap());
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[3] .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, length=3, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = zero [type=f64[]]
                            %5:f64[] = add %1 %4
                            %6:f64[] = mul %2 %5
                            %7:f64[] = mul %3 %5
                        in (%6, %7)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let mut inputs = vec![Array::scalar(1.0).unwrap()];
        inputs.extend(residuals.iter().cloned());
        assert_eq!(
            pullback.interpret(inputs),
            Ok(vec![Array::scalar(24.0).unwrap(), Array::vector(vec![12.0, 8.0, 6.0]).unwrap()]),
        );
        let mut inputs = vec![Array::scalar(2.0).unwrap()];
        inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(inputs),
            Ok(vec![Array::scalar(48.0).unwrap(), Array::vector(vec![24.0, 16.0, 12.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_differentiation_threads_stacked_reference_inputs() {
        // Forward mode needs no scan-specific rule for stacked references: the tangent allocation of the stack is an
        // active stacked reference input of the fused scan, whose body indexes both the primal and the tangent stack
        // with the body's index. The function is linear in `(carry, elements)`, so the tangent outputs are the function
        // applied to the tangent inputs.
        let jvp = stacked_reference_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3], %2:f32[], %3:f32[3] .
                let %4:ref<f32[3]> = reference_new %1
                    %5:ref<f32[3]> = reference_new %3
                    %6:f32[], %7:f32[] = scan [carry_count=2, length=3, reverse=false] %0 %2 %4 %5 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:f32[], %3:ref<f32[3]>, %4:ref<f32[3]> .
                            let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %3 %1 %0
                                () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %4 %2 %0
                                %5:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %3 %0
                                %6:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %4 %0
                                %7:f32[] = add %1 %5
                                %8:f32[] = add %2 %6
                            in (%7, %8)
                        },
                    ]
                    %8:f32[3] = reference_freeze %4
                    %9:f32[3] = reference_freeze %5
                in (%6, %8, %7, %9)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![0.0f32, 1.0, 0.0]).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(19.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()),
                TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 3.0, 5.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_reuses_the_shared_body_transforms() {
        // The `scan` differentiation rules reach their body through the per-region transform cache, so several programs
        // attaching one shared body derive its fused forward-mode program once and its transposition once per linearity
        // mask, while staging exactly the programs that the uncached path stages from independently built copies of the
        // same body.
        /// Builds a program that scans the provided body over three slices and then applies `epilogue` sines to the
        /// final carry, so that programs sharing one body still have distinct derived programs.
        fn scanning_program(body: &Arc<TestProgram>, epilogue: usize) -> TestProgram {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let initial = builder.add_input(ArrayType::scalar(DataType::F64));
            let values = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
            let body_region = builder.intern_callee(body, None).unwrap();
            let mut value = builder
                .add_instruction(
                    ArrayOperation::Scan(ScanOperation::new(1, 3)),
                    vec![body_region],
                    vec![initial, values],
                    None,
                )
                .unwrap()[0];
            for _ in 0..epilogue {
                value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            }
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        }

        let body = Arc::new(product_body());
        let first = scanning_program(&body, 1).linearize().unwrap();
        let second = scanning_program(&body, 2).linearize().unwrap();
        assert_ne!(first.tangent().to_string(), second.tangent().to_string());

        // An independently built copy of the same body shares no retained transforms, so it exercises the uncached
        // path and pins that caching changed nothing about what is staged.
        let uncached = scanning_program(&Arc::new(product_body()), 1).linearize().unwrap();
        assert_eq!(first.primal().to_string(), uncached.primal().to_string());
        assert_eq!(first.tangent().to_string(), uncached.tangent().to_string());
        assert_eq!(first.residual_count(), uncached.residual_count());

        // Transposing the tangent program twice transposes its scan body once: the second pass is served from the
        // body region's retained transposition and produces the identical pullback.
        // Build a fresh outer transpose on both calls so each reaches the nested region's cache. These static
        // tangent types need no residual dimension mappings for zeros; trailing residual inputs remain known.
        let tangent_input_indices = (0..first.tangent().input_ids().len() - first.residual_count()).collect::<Vec<_>>();
        let pullback = first.tangent().entry_region_ref().transpose(&tangent_input_indices, &[], &[]).unwrap();
        let repeated = first.tangent().entry_region_ref().transpose(&tangent_input_indices, &[], &[]).unwrap();
        assert_eq!(pullback.to_string(), repeated.to_string());
        let tangent_scan = first
            .tangent()
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayOperation::Scan(_)))
            .unwrap();
        let statistics =
            transposition_statistics(first.tangent().region_ref(tangent_scan.regions()[0]).unwrap()).unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 1));
        assert_eq!(
            pullback.to_string(),
            uncached
                .tangent()
                .entry_region_ref()
                .transpose(
                    &(0..uncached.tangent().input_ids().len() - uncached.residual_count()).collect::<Vec<_>>(),
                    &[],
                    &[],
                )
                .unwrap()
                .to_string(),
        );
    }

    #[test]
    fn test_scan_differentiation_propagates_tangents_across_carry_dependencies() {
        // The shifting body moves the tangent of the third carry into the second and first carries over successive
        // iterations, so linearization must thread tangents through all three carry slots.
        let (outputs, pushforward) = differentiate_at(vec![
            Array::scalar(0.0).unwrap(),
            Array::scalar(0.0).unwrap(),
            Array::scalar(10.0).unwrap(),
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ])
        .linearize(|inputs: Vec<TestTracer>| {
            inputs[0].context().bind(
                ArrayOperation::Scan(ScanOperation::new(3, 4)),
                vec![shifting_carry_body()],
                &inputs,
            )
        })
        .unwrap();
        assert_eq!(
            pushforward.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[4] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[4] = scan [carry_count=3, length=4, reverse=false] %0 %1 %2 \
                 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        in (%2, %3, %4, %1)
                    },
                ]
                in (%4, %5, %6, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            outputs,
            vec![
                Array::scalar(2.0).unwrap(),
                Array::scalar(3.0).unwrap(),
                Array::scalar(4.0).unwrap(),
                Array::vector(vec![0.0, 0.0, 10.0, 1.0]).unwrap(),
            ],
        );

        // Tangents must travel through all three carry slots on every invocation; reusing the pushforward must
        // not reuse a prior invocation's final carries or its stacked history.
        let tangents = vec![
            Array::scalar(0.0).unwrap(),
            Array::scalar(0.0).unwrap(),
            Array::scalar(2.0).unwrap(),
            Array::vector(vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        ];
        let expected = vec![
            Array::scalar(1.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::vector(vec![0.0, 0.0, 2.0, 1.0]).unwrap(),
        ];
        assert_eq!(pushforward.apply(tangents.clone()), Ok(expected.clone()));
        assert_eq!(
            pushforward.apply(vec![
                Array::scalar(0.0).unwrap(),
                Array::scalar(0.0).unwrap(),
                Array::scalar(5.0).unwrap(),
                Array::vector(vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
            ]),
            Ok(vec![
                Array::scalar(3.0).unwrap(),
                Array::scalar(4.0).unwrap(),
                Array::scalar(5.0).unwrap(),
                Array::vector(vec![0.0, 0.0, 5.0, 2.0]).unwrap(),
            ]),
        );
        assert_eq!(pushforward.apply(tangents), Ok(expected));
    }

    #[test]
    fn test_scan_differentiation_regenerates_slice_indices() {
        // Each row selects its diagonal entry using the loop-provided index. The derivative needs that same
        // index, but both scans can regenerate it; linearization must not save an index stack as a residual.
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64));
        let row = body_builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let selected = body_builder
            .add_instruction(DynamicSliceOperation::new(vec![1]), Vec::new(), vec![row, index], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![selected], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let matrix = builder.add_input(ArrayType::new_static(DataType::F64, [3, 3]));
        let output = builder
            .add_instruction(ScanOperation::new(0, 3).with_reverse(true), vec![body], vec![matrix], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3, 3] .
                let %1:f64[3, 1] = scan [carry_count=0, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:f64[3] .
                        let %2:f64[1] = dynamic_slice [sizes=[1]] %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3, 3] .
                let %1:f64[3, 1] = scan [carry_count=0, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:f64[3] .
                        let %2:f64[1] = dynamic_slice [sizes=[1]] %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[3, 1] .
                let %1:f64[3, 3] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f64[1] .
                        let %2:f64[3] = zero [type=f64[3]]
                            %3:f64[3] = dynamic_update_slice %2 %1 %0
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization
                .pushforward()
                .as_ref()
                .clone()
                .interpret(vec![Array::matrix(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap()]),
            Ok(vec![Array::matrix(3, 1, vec![1.0, 5.0, 9.0]).unwrap()]),
        );
        // Reverse iteration retains slice pairing: the transposed selection writes each seed to its diagonal.
        assert_eq!(
            linearization.pullback().unwrap().interpret(vec![Array::matrix(3, 1, vec![2.0, 3.0, 4.0]).unwrap()]),
            Ok(vec![Array::matrix(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_differentiation_preserves_the_sharding_of_stacked_input_cotangents() {
        // The body is declared over unsharded types, while the carry and the stacked input carry replicated sharding
        // annotations. The cotangents of both inputs must have the types that the cotangents of those annotated inputs
        // have, rather than the declared types of the body.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let carry_type = ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let stack_type =
            ArrayType::new_static(DataType::F64, [3]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let initial = Array::from_elements::<f64>(carry_type.clone(), &[1.0]).unwrap();
        let values = Array::from_elements::<f64>(stack_type.clone(), &[2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = TestEagerContext::new()
            .vjp(|(initial, values), ()| apply_product_scan(initial, values), (initial, values), ())
            .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:f64[][sharding={mesh<['x'=1:auto]>, []}], %1:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}], \
                 %2:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}], %3:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %4:f64[][sharding={mesh<['x'=1:auto]>, []}], %5:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}] = scan \
                 [carry_count=1, length=3, reverse=false] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[][sharding={mesh<['x'=1:auto]>, []}], \
                 %2:f64[][sharding={mesh<['x'=1:auto]>, []}], %3:f64[][sharding={mesh<['x'=1:auto]>, []}], \
                 %4:f64[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %5:f64[][sharding={mesh<['x'=1:auto]>, []}] = mul %3 %1
                            %6:f64[][sharding={mesh<['x'=1:auto]>, []}] = mul %4 %2
                            %7:f64[][sharding={mesh<['x'=1:auto]>, []}] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.transposed_program(&[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[][sharding={mesh<['x'=1:auto]>, []}], %1:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}], \
                 %2:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %3:f64[][sharding={mesh<['x'=1:auto]>, []}], %4:f64[3][sharding={mesh<['x'=1:auto]>, [{}]}] = scan \
                 [carry_count=1, length=3, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[][sharding={mesh<['x'=1:auto]>, []}], \
                 %2:f64[][sharding={mesh<['x'=1:auto]>, []}], %3:f64[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %4:f64[][sharding={mesh<['x'=1:auto]>, []}] = zero \
                         [type=f64[][sharding={mesh<['x'=1:auto]>, \
                 []}]]
                            %5:f64[][sharding={mesh<['x'=1:auto]>, []}] = add %1 %4
                            %6:f64[][sharding={mesh<['x'=1:auto]>, []}] = mul %2 %5
                            %7:f64[][sharding={mesh<['x'=1:auto]>, []}] = mul %3 %5
                        in (%6, %7)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(output.to_f64s(), vec![24.0]);
        let seed = Array::from_elements::<f64>(carry_type.cotangent().unwrap(), &[1.0]).unwrap();
        let (initial_cotangent, values_cotangent) = pullback.apply(seed).unwrap();
        assert_eq!(initial_cotangent.r#type().as_ref(), &carry_type.cotangent().unwrap());
        assert_eq!(initial_cotangent.to_f64s(), vec![24.0]);
        assert_eq!(values_cotangent.r#type().as_ref(), &stack_type.cotangent().unwrap());
        assert_eq!(values_cotangent.to_f64s(), vec![12.0, 8.0, 6.0]);
    }

    // Differentiation threads primal and tangent reference carries independently, preserving each running state while
    // ordinary carries and stacked outputs propagate their derivatives.
    #[test]
    fn test_scan_differentiation_threads_reference_carries() {
        // The body accumulates each scanned element into a reference carry and reports the running state, while an
        // ordinary carry sums the elements: `f(x, s, xs) = (x + Σxs, [s + xs₁, s + xs₁ + xs₂, ...], s + Σxs)`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let reference = body_builder.add_input(reference_type.into());
        let element = body_builder.add_input(scalar_type.clone().into());
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, element], None)
            .unwrap();
        let current = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let next_carry = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, element], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_carry, reference, current],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial_carry = builder.add_input(scalar_type.clone().into());
        let initial_state = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let reference = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial_state], None)
            .unwrap()[0];
        let outputs = builder
            .add_instruction(
                ScanOperation::<ArrayIrType>::new(2, 3),
                vec![body],
                vec![initial_carry, reference, elements],
                None,
            )
            .unwrap()
            .to_vec();
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[1]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[0], outputs[2], frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // The reference carry keeps its position and gains a tangent reference carry beside it, so the fused scan
        // carries `[carry, reference, carry_tangent, reference_tangent]` over a body that updates and reads the primal
        // and tangent references side by side.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3], %3:f32[], %4:f32[], %5:f32[3] .
                let %6:ref<f32[]> = reference_new %1
                    %7:ref<f32[]> = reference_new %4
                    %8:f32[], %9:ref<f32[]>, %10:f32[], %11:ref<f32[]>, %12:f32[3], %13:f32[3] = scan [carry_count=4, \
                 length=3, reverse=false] %0 %6 %3 %7 %2 %5 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[]>, %3:f32[], %4:ref<f32[]>, %5:f32[], %6:f32[] .
                            let () = reference_add_update %2 %5
                                () = reference_add_update %4 %6
                                %7:f32[] = reference_read %2
                                %8:f32[] = reference_read %4
                                %9:f32[] = add %1 %5
                                %10:f32[] = add %3 %6
                            in (%9, %2, %10, %4, %7, %8)
                        },
                    ]
                    %14:f32[] = reference_freeze %9
                    %15:f32[] = reference_freeze %11
                in (%8, %12, %14, %10, %13, %15)
            "}
            .trim_end(),
        );
        let scan = &jvp.instructions()[2];
        assert!(matches!(scan.operation(), TestIrOperation::Scan(operation) if operation.carry_count() == 4));
        assert_eq!(
            jvp.region_ref(scan.regions()[0]).unwrap().to_program().to_string(),
            indoc! {"
                lambda %0:i64[], %1:f32[], %2:ref<f32[]>, %3:f32[], %4:ref<f32[]>, %5:f32[], %6:f32[] .
                let () = reference_add_update %2 %5
                    () = reference_add_update %4 %6
                    %7:f32[] = reference_read %2
                    %8:f32[] = reference_read %4
                    %9:f32[] = add %1 %5
                    %10:f32[] = add %3 %6
                in (%9, %2, %10, %4, %7, %8)
            "}
            .trim_end(),
        );

        let inputs = vec![
            TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![0.0f32, 0.0, 1.0]).unwrap()),
        ];
        let discharged_jvp = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            discharged_jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[3], %3:f32[], %4:f32[], %5:f32[3] .
                let %6:f32[], %7:f32[], %8:f32[], %9:f32[], %10:f32[3], %11:f32[3] = scan [carry_count=4, length=3, \
                 reverse=false] %0 %1 %3 %4 %2 %5 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[], %3:f32[], %4:f32[], %5:f32[], %6:f32[] .
                        let %7:f32[] = add %2 %5
                            %8:f32[] = add %4 %6
                            %9:f32[] = add %1 %5
                            %10:f32[] = add %3 %6
                        in (%9, %7, %10, %8, %7, %8)
                    },
                ]
                in (%6, %10, %7, %8, %11, %9)
            "}
            .trim_end(),
        );
        let expected = discharged_jvp.interpret(inputs.clone()).unwrap();
        assert_eq!(
            expected,
            vec![
                TestIrValue::Array(Array::scalar(6.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![11.0f32, 13.0, 16.0]).unwrap()),
                TestIrValue::Array(Array::scalar(16.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 2.0]).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ],
        );
        assert_eq!(jvp.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_scan_differentiation_preserves_inactive_reference_carries() {
        // Differentiating with respect to the ordinary carry only leaves the reference carry inactive: it is threaded
        // through the fused scan without a tangent, and its observed state has a zero tangent.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, carry], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, doubled],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let carry = builder.add_input(scalar_type.into());
        let body = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 3), vec![body], vec![reference, carry], None)
            .unwrap()
            .to_vec();
        let state =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[1], state],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let %3:ref<f32[]>, %4:f32[], %5:f32[] = scan [carry_count=3, length=3, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[], %3:f32[] .
                        let %4:f32[] = add %2 %2
                            %5:f32[] = add %3 %3
                        in (%1, %4, %5)
                    },
                ]
                    %6:f32[] = reference_read %3
                    %7:f32[] = zero [type=f32[]]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(jvp.input_ids().len(), 3);
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap())),
                TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(24.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(16.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ])
        );
    }

    #[test]
    fn test_scan_differentiation_accumulates_reference_carry_cotangents_into_destinations() {
        // `f(r, xs) = scan { add_update(r, x_i); y_i = read(r) }` accumulates the scanned elements into the carried
        // reference and reports the running state: `y_i = r + Σ_{j ≤ i} x_j`. Its pure equivalent is the prefix sum of
        // `xs` shifted by the initial state, whose gradient under unit output cotangents is `x̄_j = length - j` and
        // `r̄ = length`.
        let body = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let element = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![carry, element], None)
                .unwrap();
            let current =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![carry, current],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let function = move |(reference, elements): (TestIrTracer, TestIrTracer)| {
            let context = reference.context().clone();
            let mut outputs =
                context.bind(ScanOperation::<ArrayIrType>::new(1, 3), vec![body.clone()], &[reference, elements])?;
            Ok(outputs.remove(1))
        };
        let reference = ArrayReference::new(Array::scalar(1.0f32).unwrap());
        let (value, pullback) = differentiate_at((
            TestIrValue::Reference(reference.clone()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ))
        .vjp(function)
        .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[3] .
                let %2:ref<f32[]>, %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[] .
                        let () = reference_add_update %1 %2
                            %3:f32[] = reference_read %1
                        in (%1, %3)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.transposed_program(&[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:ref<f32[]> .
                let %2:ref<f32[]>, %3:f32[3] = scan [carry_count=1, length=3, reverse=true] %1 %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[] .
                        let () = reference_add_update %1 %2
                            %3:f32[] = reference_read %1
                        in (%1, %3)
                    },
                ]
                in (%1, %3)
            "}
            .trim_end(),
        );
        assert_eq!(value, TestIrValue::Array(Array::vector(vec![2.0f32, 4.0, 7.0]).unwrap()));
        assert_eq!(reference.read(), Ok(Array::scalar(7.0f32).unwrap()));

        // The destination starts at the zero post-state cotangent and ends holding the cotangent of the initial state.
        let destination = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap())),
                (
                    CotangentDestination::Reference(TestIrValue::Reference(destination.clone())),
                    CotangentDestination::Return,
                ),
            ),
            Ok((None, Some(TestIrValue::Array(Array::vector(vec![3.0f32, 2.0, 1.0]).unwrap())))),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_scan_differentiation_through_nonlinear_reference_bodies() {
        // Reverse mode differentiates scans whose bodies read or update a stacked reference nonlinearly, both when the
        // reference is a differentiated input and when the function captures it without differentiating it. Each body
        // reads `stack[index]` and multiplies the carry by it, optionally after accumulating the carry into that row.
        /// Builds a scan body that multiplies its carry by a reference row, optionally updating that row first.
        fn body(updates: bool) -> TestIrProgram {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
            let transforms =
                vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
            if updates {
                builder
                    .add_instruction(
                        ReferenceAddUpdateOperation::new().with_transforms(transforms.clone()),
                        Vec::new(),
                        vec![stack, carry, index],
                        None,
                    )
                    .unwrap();
            }
            let current = builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(transforms),
                    Vec::new(),
                    vec![stack, index],
                    None,
                )
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                    Vec::new(),
                    vec![carry, current],
                    None,
                )
                .unwrap()[0];
            builder.build(vec![product], vec![Placeholder; 3], vec![Placeholder]).unwrap()
        }
        let seed = || CotangentSeed::Value(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));

        // With the stack `[1, 2, 3]` as a differentiated input and updated in place, the carries are `2`, `8`, and
        // `88`, and the derivative of the final carry with respect to the initial carry is `3 * 6 * 19 = 342`.
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let (final_carry, pullback) = differentiate_at((
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Reference(stack.clone()),
        ))
        .vjp(|(carry, stack): (TestIrTracer, TestIrTracer)| {
            Ok(carry
                .context()
                .bind(TestIrOperation::Scan(ScanOperation::new(1, 3)), vec![body(true)], &[carry.clone(), stack])?
                .remove(0))
        })
        .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[3]>, %2:f32[3], %3:f32[3] .
                let %4:f32[] = scan [carry_count=1, length=3, reverse=false] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:ref<f32[3]>, %3:f32[], %4:f32[] .
                        let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                            %5:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                            %6:f32[] = mul %3 %1
                            %7:f32[] = mul %4 %5
                            %8:f32[] = add %6 %7
                        in (%8)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback
                .transposed_program(&[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore])
                .unwrap()
                .to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3], %2:f32[3] .
                let %3:f32[3] = zero [type=f32[3]]
                    %4:ref<f32[3]> = reference_new %3
                    %5:f32[] = scan [carry_count=1, length=3, reverse=true] %0 %4 %1 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[3]>, %3:f32[], %4:f32[] .
                            let %5:f32[] = mul %4 %1
                                () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %5 %0
                                %6:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %7:f32[] = mul %3 %1
                                %8:f32[] = add %7 %6
                            in (%8)
                        },
                    ]
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(final_carry, TestIrValue::Array(Array::scalar(88.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(seed(), (CotangentDestination::Return, CotangentDestination::Ignore)),
            Ok((Some(TestIrValue::Array(Array::scalar(342.0f32).unwrap())), None)),
        );

        // A captured stack that is only read carries no tangent, so the derivative of the final carry is the product
        // of its rows.
        let captured = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let (final_carry, pullback) = differentiate_at(TestIrValue::Array(Array::scalar(1.0f32).unwrap()))
            .vjp(|carry: TestIrTracer| {
                let stack = carry.context().lift(TestIrValue::Reference(captured.clone()))?;
                Ok(carry
                    .context()
                    .bind(TestIrOperation::Scan(ScanOperation::new(1, 3)), vec![body(false)], &[carry.clone(), stack])?
                    .remove(0))
            })
            .unwrap();
        assert_eq!(
            pullback.linear_program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.transposed_program(&[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(final_carry, TestIrValue::Array(Array::scalar(6.0f32).unwrap()));
        assert_eq!(
            pullback.apply(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
            Ok(TestIrValue::Array(Array::scalar(6.0f32).unwrap()))
        );
    }

    #[test]
    fn test_scan_differentiation_composite_pullback_shapes_a_dead_dynamic_carry_cotangent_from_a_live_peer() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let first = body_builder.add_input(ArrayIrType::Array(array_type.clone()));
        let second = body_builder.add_input(ArrayIrType::Array(array_type.clone()));
        let sum = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![sum, second], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let second = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(TestIrOperation::Scan(ScanOperation::new(2, 2)), vec![region], vec![first, second], None)
            .unwrap()
            .to_vec();
        // Keeping only the accumulating carry leaves the second carry dead, so the reversed scan needs a dynamic zero
        // cotangent for it. The live first-carry cotangent names the same runtime extent, which the transpose boundary
        // reads off it before staging the mixed dynamic zero.
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[extent] .
                let %1:dimension<extent \u{2208} [1, 8)> = dimension_size [axis=0] %0
                    %2:f64[extent] = zero [type=f64[extent]] %1
                    %3:f64[extent], %4:f64[extent] = scan [carry_count=2, length=2, reverse=true] %0 %2 [
                        body={
                            lambda %0:i64[], %1:f64[extent], %2:f64[extent] .
                            let %3:f64[extent] = add %2 %1
                            in (%1, %3)
                        },
                    ]
                in (%3, %4)"}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())]),
            Ok(vec![
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![2.0, 2.0, 2.0]).unwrap())
            ]),
        );
    }

    #[test]
    fn test_scan_differentiation_composite_refines_outputs_of_unspecialized_bodies() {
        // Differentiating a scan whose refined outputs come from an unspecialized body keeps the primal, tangent, and
        // cotangent types refined. The scan computes `[c + x[0] + x[1], x * x]`.
        let program = refined_vector_scan_program();
        let stacked = |values: &[f64]| {
            array(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), values).unwrap())
        };
        let carry = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let slices = stacked(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[2, 3], %2:f64[3], %3:f64[2, 3] .
                let %4:f64[3], %5:f64[3], %6:f64[2, 3], %7:f64[2, 3] = scan [carry_count=2, length=2, reverse=false] \
                 %0 \
                 %2 %1 %3 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                        let %5:f64[3] = add %1 %3
                            %6:f64[3] = add %2 %4
                            %7:f64[3] = mul %3 %3
                            %8:f64[3] = mul %3 %4
                            %9:f64[3] = mul %3 %4
                            %10:f64[3] = add %8 %9
                        in (%5, %6, %7, %10)
                    },
                ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );
        assert!(jvp.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        assert_eq!(
            jvp.interpret(vec![
                carry.clone(),
                slices.clone(),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                stacked(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            ]),
            Ok(vec![
                array(Array::vector(vec![4.0, 5.0, 6.0]).unwrap()),
                stacked(&[1.0, 1.0, 1.0, 4.0, 4.0, 4.0]),
                array(Array::vector(vec![2.0, 2.0, 2.0]).unwrap()),
                stacked(&[0.0, 0.0, 0.0, 4.0, 4.0, 4.0]),
            ]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[2, 3] .
                let %2:f64[3], %3:f64[2, 3] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:f64[3] .
                        let %3:f64[3] = add %1 %2
                            %4:f64[3] = mul %2 %2
                        in (%3, %4)
                    },
                ]
                in (%2, %3, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[2, 3], %2:f64[2, 3] .
                let %3:f64[3], %4:f64[2, 3] = scan [carry_count=1, length=2, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                        let %4:f64[3] = add %1 %2
                            %5:f64[3] = mul %3 %2
                            %6:f64[3] = mul %3 %2
                            %7:f64[3] = add %5 %6
                        in (%4, %7)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[2, 3], %2:f64[2, 3] .
                let %3:f64[3], %4:f64[2, 3] = scan [carry_count=1, length=2, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:f64[3], %3:f64[3] .
                        let %4:f64[3] = mul %3 %2
                            %5:f64[3] = mul %3 %2
                            %6:f64[3] = add %4 %5
                            %7:f64[3] = add %6 %1
                        in (%1, %7)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization.primal().interpret(vec![carry, slices]).unwrap();
        let residuals = primal_outputs.split_off(2);
        let pullback = linearization.pullback().unwrap();
        assert!(pullback.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        let mut pullback_inputs =
            vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()), stacked(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()), stacked(&[3.0, 3.0, 3.0, 5.0, 5.0, 5.0])]),
        );
    }

    #[test]
    fn test_scan_differentiation_composite_jvp_forwards_a_dynamic_length_and_dimension_carry() {
        let carry_extent_type = DimensionType::new("carry_extent", DimensionBounds::positive(Some(8)).unwrap());
        let length_variable = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let length_type = DimensionType::from(length_variable.clone());
        let length = Dimension::Dynamic(length_variable);
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(carry_extent_type.clone()));
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let values =
            builder.add_input(ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![length.clone()]))));
        let runtime_length = builder.add_input(ArrayIrType::Dimension(length_type.clone()));
        let body = product_scan_body(carry_extent_type.clone());
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                TestIrOperation::Scan(ScanOperation::new(2, length)),
                vec![region],
                vec![extent, carry, values, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 4], vec![Placeholder; 3]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<carry_extent ∈ [1, 8)>, %1:f64[], %2:f64[length], %3:dimension<length ∈ [1, 8)>, \
                 %4:f64[], %5:f64[length] .
                let %6:dimension<carry_extent ∈ [1, 8)>, %7:f64[], %8:f64[], %9:f64[length], %10:f64[length] = scan \
                 [carry_count=3, length=length, reverse=false] %0 %1 %4 %2 %5 %3 [
                    body={
                        lambda %0:i64[], %1:dimension<carry_extent ∈ [1, 8)>, %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                        let %6:f64[] = mul %2 %4
                            %7:f64[] = mul %4 %3
                            %8:f64[] = mul %2 %5
                            %9:f64[] = add %7 %8
                        in (%1, %6, %9, %6, %9)
                    },
                ]
                in (%0, %7, %9, %8, %10)
            "}
            .trim_end(),
        );
        assert_eq!(jvp.input_count(), 6);
        assert_eq!(jvp.output_count(), 5);
        let outputs = jvp
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0).unwrap()),
                array(Array::vector(vec![2.0, 3.0, 4.0]).unwrap()),
                dimension(&length_type, 3),
                array(Array::scalar(5.0).unwrap()),
                array(Array::vector(vec![0.5, 1.0, 1.5]).unwrap()),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], TestIrValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestIrValue::Array(value) if value.to_f64s() == vec![24.0]));
        assert!(matches!(&outputs[3], TestIrValue::Array(value) if value.to_f64s() == vec![143.0]));

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<carry_extent ∈ [1, 8)>, %1:f64[], %2:f64[length], %3:dimension<length ∈ [1, 8)> .
                let %4:dimension<carry_extent ∈ [1, 8)>, %5:f64[], %6:f64[length], %7:f64[length] = scan \
                 [carry_count=2, \
                 length=length, reverse=false] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:dimension<carry_extent ∈ [1, 8)>, %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %2 %3
                        in (%1, %4, %4, %2)
                    },
                ]
                in (%0, %5, %6, %2, %7, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[length], %2:f64[length], %3:f64[length], %4:dimension<length ∈ [1, 8)> .
                let %5:f64[], %6:f64[length] = scan [carry_count=1, length=length, reverse=false] %0 %1 %2 %3 %4 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %3 %1
                            %6:f64[] = mul %4 %2
                            %7:f64[] = add %5 %6
                        in (%7, %7)
                    },
                ]
                in (%5, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[length], %2:f64[length], %3:f64[length], %4:dimension<length ∈ [1, 8)> .
                let %5:f64[], %6:f64[length] = scan [carry_count=1, length=length, reverse=true] %0 %1 %2 %3 %4 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = add %1 %2
                            %6:f64[] = mul %3 %5
                            %7:f64[] = mul %4 %5
                        in (%6, %7)
                    },
                ]
                in (%5, %6)
            "}
            .trim_end(),
        );
        assert_eq!(linearization.residual_count(), 3);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0).unwrap()),
                array(Array::vector(vec![2.0, 3.0, 4.0]).unwrap()),
                dimension(&length_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs =
            vec![array(Array::scalar(1.0).unwrap()), array(Array::vector(vec![0.0, 0.0, 0.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(24.0).unwrap()), array(Array::vector(vec![12.0, 8.0, 6.0]).unwrap())]),
        );
    }

    #[test]
    fn test_scan_differentiation_composite_jvp_shapes_a_disconnected_dynamic_carry_tangent_from_its_primal() {
        // Body `[extent, carry, x] -> [extent, carry + x]` over a dynamically shaped carry whose initial value has its
        // tangent severed. The severed carry still receives a tangent carry slot because the body makes it active
        // through `x`, so its structurally zero initial tangent must be shaped from its own primal.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let stacked_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(extent_type.variable().clone())]),
        );
        let body = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let _index = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let carry = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let sum = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![carry, input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![extent, sum],
                    vec![Placeholder; 4],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let initial = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let values = builder.add_input(ArrayIrType::Array(stacked_type));
        let severed = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(StopGradientOperation::<ArrayType>::new())),
                Vec::new(),
                vec![initial],
                None,
            )
            .unwrap()[0];
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                TestIrOperation::Scan(ScanOperation::new(2, 3)),
                vec![region],
                vec![extent, severed, values],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(vec![outputs[1]], vec![Placeholder; 3], vec![Placeholder]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:f64[3, extent], %3:f64[extent], %4:f64[3, \
                 extent] .
                let %5:f64[extent] = stop_gradient %1
                    %6:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %5
                    %7:f64[extent] = zero [type=f64[extent]] %6
                    %8:dimension<extent ∈ [1, 8)>, %9:f64[extent], %10:f64[extent] = scan [carry_count=3, length=3, \
                 reverse=false] %0 %5 %7 %2 %4 [
                        body={
                            lambda %0:i64[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent], %3:f64[extent], \
                 %4:f64[extent], %5:f64[extent] .
                            let %6:f64[extent] = add %2 %4
                                %7:f64[extent] = add %3 %5
                            in (%1, %6, %7)
                        },
                    ]
                in (%9, %10)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                dimension(&extent_type, 2),
                array(Array::vector(vec![1.0, 2.0]).unwrap()),
                array(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                array(Array::vector(vec![7.0, 7.0]).unwrap()),
                array(Array::matrix(3, 2, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap()),
            ]),
            Ok(vec![array(Array::vector(vec![10.0, 14.0]).unwrap()), array(Array::vector(vec![4.5, 6.0]).unwrap()),]),
        );
    }

    #[test]
    fn test_scan_differentiation_composite_pullback_stacks_varying_dimension_residuals_through_scalar_gateways() {
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let state = body_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestIrOperation::from(DimensionFromScalarOperation::new(iteration_variable)),
                Vec::new(),
                vec![counter],
                None,
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestIrOperation::from(DynamicBroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
                None,
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
                None,
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1u64).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_state, next_counter, next_state],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(scalar_f64));
        let counter = builder.add_input(ArrayIrType::Array(scalar_u64));
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(TestIrOperation::Scan(ScanOperation::new(2, 2)), vec![region], vec![state, counter], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:u64[] .
                let %2:f64[], %3:u64[], %4:f64[2], %5:i64[2], %6:i64[2] = scan [carry_count=2, length=2, \
                 reverse=false] \
                 %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:u64[] .
                        let %3:dimension<iteration ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %2
                            %4:f64[iteration] = broadcast [output_axes=[]] %1 %3
                            %5:f64[] = reduce [kind=sum, axes=[0]] %4
                            %6:u64[] = const 1
                            %7:u64[] = add %2 %6
                            %8:i64[] = dimension_to_scalar %3
                            %9:dimension<iteration ∈ [1, 4)> = dimension_size [axis=0] %4
                            %10:i64[] = dimension_to_scalar %9
                        in (%5, %7, %5, %8, %10)
                    },
                ]
                in (%2, %3, %4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:i64[2] .
                let %2:f64[], %3:f64[2] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:i64[] .
                        let %3:dimension<iteration ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %2
                            %4:f64[iteration] = broadcast [output_axes=[]] %1 %3
                            %5:f64[] = linear_call [residual_count=1] %3 %4 [
                                forward={
                                    lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[iteration] .
                                    let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                                    in (%2)
                                },
                                transpose={
                                    lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[] .
                                    let %2:f64[iteration] = broadcast [output_axes=[]] %1 %0
                                    in (%2)
                                },
                            ]
                        in (%5, %5)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[2], %2:i64[2] .
                let %3:f64[] = scan [carry_count=1, length=2, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:i64[] .
                        let %4:dimension<iteration ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %3
                            %5:f64[] = add %1 %2
                            %6:f64[iteration] = linear_call [residual_count=1] %4 %5 [
                                forward={
                                    lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[] .
                                    let %2:f64[iteration] = broadcast [output_axes=[]] %1 %0
                                    in (%2)
                                },
                                transpose={
                                    lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[iteration] .
                                    let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                                    in (%2)
                                },
                            ]
                            %7:f64[] = reduce [kind=sum, axes=[0]] %6
                        in (%7)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        // Only the changing extent is needed by the linear body; the primal state is not a coefficient.
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0).unwrap()), array(Array::scalar(1u64).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs =
            vec![array(Array::scalar(1.0).unwrap()), array(Array::vector(vec![0.0, 0.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(2.0).unwrap())])
        );
    }

    #[test]
    fn test_scan_transposition_validates_the_original_contract_before_zero_elimination() {
        /// A detached source driver whose body must be validated before any body transposition can occur.
        struct Driver(TestProgram);

        impl RegionDriver<Array, TestOperation> for Driver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, TestOperation>>
            where
                Array: 'r,
                TestOperation: 'r,
            {
                std::iter::once(self.0.entry_region_ref())
            }
        }

        impl TranspositionDriver<Array, TestOperation> for Driver {
            fn transpose_program(
                &self,
                _region: RegionRef<'_, Array, TestOperation>,
                _input_indices: &[usize],
                _destination_kinds: &[CotangentDestinationKind],
            ) -> Result<Arc<TestProgram>, DifferentiationError> {
                panic!("invalid source scans must fail before body transposition")
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = |index_type| {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            builder.add_input(ArrayType::scalar(index_type));
            let carry = builder.add_input(scalar_type.clone());
            builder.build(vec![carry], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let cotangents = CotangentDestinations::without_references([true]);
        let mut context = TracingContext::<Array, TestOperation>::new();
        assert_eq!(
            transpose_primal_scan(
                &TestScanOperation::new(1, 1),
                &mut context,
                &Driver(body(DataType::Boolean)),
                &[PartialValue::Unknown(scalar_type.clone())],
                &[MaybeZero::Zero(scalar_type.clone())],
                &cotangents,
            )
            .unwrap_err(),
            DifferentiationError::from(TypeError::invalid("`scan` body input 0 must be a scalar `i64` slice index")),
        );
        assert_eq!(
            transpose_primal_scan(
                &TestScanOperation::new(1, 1),
                &mut context,
                &Driver(body(DataType::I64)),
                &[PartialValue::Unknown(scalar_type.clone())],
                &[],
                &cotangents,
            )
            .unwrap_err(),
            DifferentiationError::from(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        let boolean = context.constant(Array::scalar(true).unwrap());
        assert_eq!(
            transpose_primal_scan(
                &TestScanOperation::new(1, 1),
                &mut context,
                &Driver(body(DataType::I64)),
                &[PartialValue::Known(boolean)],
                &[MaybeZero::Zero(scalar_type)],
                &cotangents,
            )
            .unwrap_err(),
            DifferentiationError::from(TypeError::invalid(
                "`scan` input 0 has type `bool[]`, which does not refine its expected type `f64[]`",
            )),
        );
    }

    #[test]
    fn test_scan_transposition() {
        // The body maps `[carry, x]` to `[carry + x, carry]`, so the scan is linear in its carry and its stacked input.
        // Its transposition is one scan in the opposite visit order with the same length and unroll factor, over the
        // output cotangents, and with unit output cotangents, `x_j` receives `1` from the final carry plus `1` from
        // every later stacked output.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new_static(DataType::F64, [3]);
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, inputs[1]]
        });
        let program =
            scan_program(TestScanOperation::new(1, 3), body.clone(), vec![scalar_type.clone(), stacked_type.clone()]);
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %2 %1
                        in (%3, %1)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![Array::scalar(1.0).unwrap(), Array::vector(vec![1.0, 1.0, 1.0]).unwrap()]),
            Ok(vec![Array::scalar(4.0).unwrap(), Array::vector(vec![3.0, 2.0, 1.0]).unwrap()]),
        );

        // A reversed scan transposes into a forward scan that keeps the unroll factor, and `x_j` now receives `1` from
        // every earlier stacked output.
        let program = scan_program(
            TestScanOperation::new(1, 3).with_reverse(true).with_unroll(2).unwrap(),
            body,
            vec![scalar_type.clone(), stacked_type.clone()],
        );
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false, unroll=2] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %2 %1
                        in (%3, %1)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![Array::scalar(1.0).unwrap(), Array::vector(vec![1.0, 1.0, 1.0]).unwrap()]),
            Ok(vec![Array::scalar(4.0).unwrap(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]),
        );

        // A known carry that the body passes through unchanged (here `scale` in `[scale, carry, x] ->
        // [scale, scale * carry + x, carry]`) stays a carry of the transposed scan, seeded with its known value. With
        // `scale = 2`, the final carry is `8 c + 4 x_0 + 2 x_1 + x_2` and the stacked outputs are `[c, c_1, c_2]`.
        let body = scalar_body(3, |builder, inputs| {
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, inputs[3]], None).unwrap()[0];
            vec![inputs[1], next, inputs[2]]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let inputs = [scalar_type.clone(), scalar_type, stacked_type].map(|input_type| builder.add_input(input_type));
        let outputs = builder
            .add_instruction(TestScanOperation::new(2, 3), vec![body], inputs.to_vec(), None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[1], outputs[2]], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[], %5:f64[3] = scan [carry_count=2, length=3, reverse=true] %2 %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %1 %2
                            %5:f64[] = add %3 %4
                        in (%1, %5, %2)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![
                Array::scalar(1.0).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
                Array::scalar(2.0).unwrap(),
            ]),
            Ok(vec![Array::scalar(15.0).unwrap(), Array::vector(vec![7.0, 3.0, 1.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_transposition_zero_length() {
        // A zero-length scan is the identity on its carries, so its transposition passes the carry cotangent through
        // and gives the empty stacked input an empty cotangent.
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, inputs[1]]
        });
        let empty = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap();
        let program = scan_program(
            TestScanOperation::new(1, 0),
            body,
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [0])],
        );
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[0] .
                let %2:f64[], %3:f64[0] = scan [carry_count=1, length=0, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %2 %1
                        in (%3, %1)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![Array::scalar(3.0).unwrap(), empty.clone()]),
            Ok(vec![Array::scalar(3.0).unwrap(), empty]),
        );
    }

    #[test]
    fn test_scan_transposition_composes_with_batching() {
        // Batch a linear scan, then transpose its packed program. The reversed scan keeps scan slices paired with
        // their seeds, and restores the original input's leading batch axis after computing its scanned cotangent.
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, inputs[1]]
        });
        let program = scan_program(
            TestScanOperation::new(1, 3),
            body,
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [3])],
        );
        let batched = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(1)]);
        let transposed = batched.into_parts().0.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[3, 2] .
                let %2:f64[2], %3:f64[3, 2] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[2], %2:f64[2] .
                        let %3:f64[2] = add %2 %1
                        in (%3, %1)
                    },
                ]
                    %4:f64[2, 3] = transpose [permutation=[1, 0]] %3
                in (%2, %4)"},
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::vector(vec![1.0f64, 10.0]).unwrap(),
                Array::matrix(3, 2, vec![2.0f64, 20.0, 3.0, 30.0, 4.0, 40.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![10.0f64, 100.0]).unwrap(),
                Array::matrix(2, 3, vec![8.0f64, 5.0, 1.0, 80.0, 50.0, 10.0]).unwrap(),
            ]),
        );

        // Batch the pullback independently at the same physical seed axes. It must preserve the same input
        // cotangents, even though its natural scanned-output axis lies behind the scan axis.
        let batched_pullback = program
            .transpose_with_respect_to(&[0, 1], &[])
            .unwrap()
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(1)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched_pullback.output_axes(), &[BatchAxis::new(0), BatchAxis::new(1)]);
        let batched_pullback = batched_pullback.into_parts().0;
        assert_eq!(
            batched_pullback.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[3, 2] .
                let %2:f64[2], %3:f64[3, 2] = scan [carry_count=1, length=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[2], %2:f64[2] .
                        let %3:f64[2] = add %2 %1
                        in (%3, %1)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            batched_pullback.interpret(vec![
                Array::vector(vec![1.0f64, 10.0]).unwrap(),
                Array::matrix(3, 2, vec![2.0f64, 20.0, 3.0, 30.0, 4.0, 40.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![10.0f64, 100.0]).unwrap(),
                Array::matrix(3, 2, vec![8.0f64, 80.0, 5.0, 50.0, 1.0, 10.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_scan_transposition_preserves_zero_output_deferred_work() {
        // The scan has no public outputs or differentiated inputs. Its known reference is plumbing for a dormant
        // backward rule, whose update must execute once per reversed iteration despite all output cotangents being
        // structural zeros and no cotangent reference state being live.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F64)).into();
        let mut backward_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = backward_builder.add_input(reference_type.clone());
        let update = backward_builder.add_constant(array(Array::scalar(1.0f64).unwrap()));
        backward_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let backward = backward_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = body_builder.add_input(reference_type.clone());
        let backward_region = body_builder.import_program(backward);
        body_builder
            .add_instruction(
                TestIrOperation::CustomFunctionTranspose(CustomFunctionTransposeOperation::from_backward_region(
                    1,
                    Vec::new(),
                    Vec::new(),
                )),
                vec![backward_region],
                vec![reference],
                None,
            )
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(body.effects().classes().is_empty());
        assert!(body.effects().has_deferred_work());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.clone());
        let body_region = builder.import_program(body.clone());
        builder.add_instruction(ScanOperation::new(1, 3), vec![body_region], vec![reference], None).unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f64[]> .
                let %1:ref<f64[]> = scan [carry_count=1, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:ref<f64[]> .
                        let %2:f64[] = const 1.0
                            () = reference_add_update %1 %2
                        in (%1)
                    },
                ]
                in ()"},
        );
        assert!(!transposed.effects().has_deferred_work());
        let reference = ArrayReference::new(Array::scalar(4.0f64).unwrap());
        assert_eq!(transposed.interpret(vec![TestIrValue::Reference(reference.clone())]), Ok(Vec::new()));
        assert_eq!(reference.read(), Ok(Array::scalar(7.0f64).unwrap()));

        // A zero-trip scan may expose its dormant rule to transposition, but must never execute its update.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type);
        let body_region = builder.import_program(body);
        builder.add_instruction(ScanOperation::new(1, 0), vec![body_region], vec![reference], None).unwrap();
        let empty_program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let empty_transposed = empty_program.transpose_with_respect_to(&[], &[]).unwrap();
        assert_eq!(
            empty_transposed.to_string(),
            indoc! {"
                lambda %0:ref<f64[]> .
                let %1:ref<f64[]> = scan [carry_count=1, length=0, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:ref<f64[]> .
                        let %2:f64[] = const 1.0
                            () = reference_add_update %1 %2
                        in (%1)
                    },
                ]
                in ()"},
        );
        let reference = ArrayReference::new(Array::scalar(4.0f64).unwrap());
        assert_eq!(empty_transposed.interpret(vec![TestIrValue::Reference(reference.clone())]), Ok(Vec::new()));
        assert_eq!(reference.read(), Ok(Array::scalar(4.0f64).unwrap()));
    }

    #[test]
    fn test_scan_transposition_promotes_known_initializers_of_linear_carries() {
        // The scanned input reaches the third carry, then the second, then the first. All initializers are known, but
        // the reversed body needs their carry cotangents to propagate gradients through all three iterations.
        let program = scan_program(
            TestScanOperation::new(3, 3),
            shifting_carry_body(),
            vec![
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::new_static(DataType::F64, [3]),
            ],
        );
        let transposed = program.transpose_with_respect_to(&[3], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[3], %4:f64[], %5:f64[], %6:f64[] .
                let %7:f64[], %8:f64[], %9:f64[], %10:f64[3] = scan [carry_count=3, length=3, reverse=true] \
                 %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        in (%4, %1, %2, %3)
                    },
                ]
                in (%10)"},
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(11.0f64).unwrap(),
                Array::scalar(13.0f64).unwrap(),
                Array::scalar(17.0f64).unwrap(),
                Array::vector(vec![19.0f64, 23.0, 29.0]).unwrap(),
                Array::scalar(1.0f64).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(3.0f64).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![11.0f64, 13.0, 17.0]).unwrap()]),
        );
    }

    #[test]
    fn test_scan_transposition_rejects_nonlinear_promoted_carries() {
        // The known carry becomes dependent on the linear scanned input. Promoting its recurrence state cannot make
        // the stacked product linear: both factors now vary with the selected input and the body transpose rejects it.
        let body = scalar_body(2, |builder, inputs| {
            let next =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![next, product]
        });
        let program = scan_program(
            TestScanOperation::new(1, 3),
            body,
            vec![ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [3])],
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `mul` does not support transposition for input pattern \
                               [left = linear, right = linear]",
        ));
    }

    #[test]
    fn test_scan_transposition_rejects_time_varying_known_carries() {
        // Body `[known, linear] -> [known + 1, linear * known]`. The known carry changes on every iteration, so the
        // reversed scan cannot linearize each iteration at the carry's initial value and transposition is rejected.
        let body = scalar_body(2, |builder, inputs| {
            let one = builder.add_constant(Array::scalar(1.0f64).unwrap());
            let next_known =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[1], one], None).unwrap()[0];
            let next_linear =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[2], inputs[1]], None).unwrap()[0];
            vec![next_known, next_linear]
        });
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let body = builder.import_program(body);
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestScanOperation::new(2, 3), vec![body], vec![known, linear], None)
            .unwrap()[1];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`scan` transposition requires known carry 0 to pass through the body unchanged; \
                               time-varying known values must be supplied as stacked inputs",
        ));
    }

    #[test]
    fn test_scan_transposition_materializes_dead_dynamic_stacked_cotangent() {
        // A dead stacked output's cotangent has the scan-length-prefixed type `f64[length, k]`, which no single value
        // at the transposition boundary carries: the length rides the runtime length input and the inner extent rides
        // the per-iteration carry cotangent. Identity-directed materialization assembles the zero from both, so the
        // reversed scan gets a well-typed input instead of failing on an unconstructible nullary zero.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let extent = DimensionVariable::new("k", DimensionBounds::positive(Some(8)).unwrap());
        let item_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(length.clone()), Dimension::Dynamic(extent)]),
        ));

        // The body maps `[carry, x]` to `[carry + x, carry]`, so the scan produces a final carry and a stacked
        // per-iteration output. Both halves are linear in the inputs.
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(item_type.clone());
        let value = body_builder.add_input(item_type.clone());
        let sum = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, value], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![sum, carry], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        // Only the final carry is a program output, so the stacked output is dead and its cotangent is a structural
        // zero of the dynamically shaped stacked type.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let runtime_length = builder.add_input(ArrayIrType::Dimension(DimensionType::from(length.clone())));
        let initial_carry = builder.add_input(item_type);
        let stacked_input = builder.add_input(stacked_type.clone());
        let body_region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                TestIrOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![initial_carry, stacked_input, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        assert_eq!(builder.atoms()[outputs[1].index()].r#type().as_ref(), &stacked_type);
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[0]], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // Transposing with respect to the carry initializer and the stacked input reaches the dead stacked cotangent.
        // The pullback reads the inner extent off the live carry cotangent, reuses the runtime length input, and stages
        // the mixed dynamic zero over both.
        let pullback = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[k], %1:dimension<length \u{2208} [1, 8)> .
                let %2:dimension<k \u{2208} [1, 8)> = dimension_size [axis=0] %0
                    %3:f64[length, k] = zero [type=f64[length, k]] %1 %2
                    %4:f64[k], %5:f64[length, k] = scan [carry_count=1, length=length, reverse=true] %0 %3 %1 [
                        body={
                            lambda %0:i64[], %1:f64[k], %2:f64[k] .
                            let %3:f64[k] = add %2 %1
                            in (%3, %1)
                        },
                    ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scan_transposition_uses_reference_geometry_for_dead_dynamic_outputs() {
        // The stacked snapshots are dead, while the reference's state cotangent remains live. Its referent is the only
        // source of the dynamic extent needed to seed those snapshots with zeros in the reversed scan.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let reference_type = ArrayIrType::from(ReferenceType::new(array_type));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let reference = builder.add_input(reference_type.clone());
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, snapshot], None)
            .unwrap();
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type);
        builder.add_instruction(ScanOperation::new(1, 2), vec![body], vec![reference], None).unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[extent]> .
                let %1:f32[extent] = reference_read %0
                    %2:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %1
                    %3:f32[2, extent] = zero [type=f32[2, extent]] %2
                    %4:ref<f32[extent]> = scan [carry_count=1, length=2, reverse=true] %0 %3 [
                        body={
                            lambda %0:i64[], %1:ref<f32[extent]>, %2:f32[extent] .
                            let %3:f32[extent] = reference_read %1
                                %4:f32[extent] = add %2 %3
                                () = reference_add_update %1 %4
                            in (%1)
                        },
                    ]
                in (%0)"},
        );
        let destination = ArrayReference::new(Array::vector(vec![3f32, 5.0, 7.0]).unwrap());
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Reference(destination.clone())]),
            Ok(vec![TestIrValue::Reference(destination.clone())]),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![12f32, 20.0, 28.0]).unwrap()));
        let destination = ArrayReference::new(Array::vector(vec![3f32, 5.0]).unwrap());
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Reference(destination.clone())]),
            Ok(vec![TestIrValue::Reference(destination.clone())]),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![12f32, 20.0]).unwrap()));
    }

    #[test]
    fn test_scan_transposition_threads_reference_stack_cotangents() {
        // Transposing a scan over a stacked reference threads the stack's cotangent reference through the reversed
        // scan, whose body selects and accumulates into the current row with its own index.
        let program = stacked_reference_program();
        let pullback = program.linearize().unwrap().pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[3] = zero [type=f32[3]]
                    %3:ref<f32[3]> = reference_new %2
                    () = reference_add_update %3 %1
                    %4:f32[] = scan [carry_count=1, length=3, reverse=true] %0 %3 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[3]> .
                            let () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %2 %1 %0
                                %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %4:f32[] = add %1 %3
                            in (%4)
                        },
                    ]
                    %5:f32[3] = reference_freeze %3
                in (%4, %5)"},
        );

        // The function is linear, so the pullback of `(ȳ_carry, ȳ_elements)` is
        // `(8 ȳ_carry + ȳ_0 + 2 ȳ_1 + 4 ȳ_2, [4 ȳ_carry + ȳ_0 + ȳ_1 + 2 ȳ_2, 2 ȳ_carry + ȳ_1 + ȳ_2, ȳ_carry + ȳ_2])`,
        // and it agrees with the pullback of the same computation over hand-unrolled static views of the stack.
        let seeds = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap()),
        ];
        let cotangents = vec![
            TestIrValue::Array(Array::scalar(15.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![8.0f32, 4.0, 2.0]).unwrap()),
        ];
        assert_eq!(pullback.interpret(seeds.clone()), Ok(cotangents.clone()));
        let unrolled = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let mut carry = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
            let stack =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
            for iteration in 0..3 {
                let element_transforms = vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(iteration),
                }];
                builder
                    .add_instruction(
                        ReferenceAddUpdateOperation::new().with_transforms(element_transforms.clone()),
                        Vec::new(),
                        vec![stack, carry],
                        None,
                    )
                    .unwrap();
                let current = builder
                    .add_instruction(
                        ReferenceReadOperation::new().with_transforms(element_transforms.clone()),
                        Vec::new(),
                        vec![stack],
                        None,
                    )
                    .unwrap()[0];
                carry = builder
                    .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, current], None)
                    .unwrap()[0];
            }
            let frozen =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![stack], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![carry, frozen],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(unrolled.linearize().unwrap().pullback().unwrap().interpret(seeds), Ok(cotangents));

        // With the stack as a differentiated input, a `Reference` destination is the stacked cotangent reference that
        // the reversed scan consumes directly: it holds the cotangent of the final stack contents on entry and the
        // cotangent of the initial contents on return. An `Ignore` destination accumulates through an internal
        // stacked cotangent reference instead and returns the carry cotangent only.
        let body = stacked_reference_body();
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let (final_carry, pullback) = differentiate_at((
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Reference(stack.clone()),
        ))
        .vjp(|(carry, stack): (TestIrTracer, TestIrTracer)| {
            let mut outputs = carry.context().bind(
                TestIrOperation::Scan(ScanOperation::new(1, 3)),
                vec![body.clone()],
                &[carry.clone(), stack],
            )?;
            Ok(outputs.remove(0))
        })
        .unwrap();
        assert_eq!(final_carry, TestIrValue::Array(Array::scalar(19.0f32).unwrap()));
        assert_eq!(stack.read(), Ok(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()));
        let destination = ArrayReference::new(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
                (
                    CotangentDestination::Return,
                    CotangentDestination::Reference(TestIrValue::Reference(destination.clone()))
                ),
            ),
            Ok((Some(TestIrValue::Array(Array::scalar(15.0f32).unwrap())), None)),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![8.0f32, 4.0, 2.0]).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
                (CotangentDestination::Return, CotangentDestination::Ignore),
            ),
            Ok((Some(TestIrValue::Array(Array::scalar(8.0f32).unwrap())), None)),
        );
    }

    #[test]
    fn test_scan_transposition_of_reference_stack_matches_discharged_form() {
        // With the stack allocated and frozen inside the program, both forms map `(carry, elements)` to
        // `(final_carry, elements')`, so their pullbacks take the same seeds and must return the same cotangents.
        let program = stacked_reference_program();
        let discharged =
            program.clone().discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        let pullback = program.linearize().unwrap().pullback().unwrap();
        let discharged_pullback = discharged.linearize().unwrap().pullback().unwrap();
        let seeds = vec![
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        ];
        let cotangents = vec![
            TestIrValue::Array(Array::scalar(25.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![13.0f32, 7.0, 4.0]).unwrap()),
        ];
        assert_eq!(pullback.interpret(seeds.clone()), Ok(cotangents.clone()));
        assert_eq!(discharged_pullback.interpret(seeds), Ok(cotangents));

        // With the stack as a reference input, the discharged form takes the stack's entering state as an ordinary
        // `f32[3]` input and returns its final state as a hidden output appended after the carry. A `Reference`
        // destination of the undischarged pullback holds the cotangent of the final state on entry and the cotangent
        // of the entering state on return, so seeding the discharged pullback's hidden output with the destination's
        // initial contents must return the destination's final contents as the array cotangent of the stack input,
        // next to the same carry cotangent.
        let body = stacked_reference_body();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_region = builder.import_program(body.clone());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body_region], vec![initial, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %2 %1
                            %4:f32[] = add %1 %3
                        in (%4, %3)
                    },
                ]
                in (%2, %3)"},
        );
        let stack = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let (final_carry, pullback) = differentiate_at((
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
            TestIrValue::Reference(stack.clone()),
        ))
        .vjp(|(carry, stack): (TestIrTracer, TestIrTracer)| {
            let mut outputs = carry.context().bind(
                TestIrOperation::Scan(ScanOperation::new(1, 3)),
                vec![body.clone()],
                &[carry.clone(), stack],
            )?;
            Ok(outputs.remove(0))
        })
        .unwrap();
        assert_eq!(final_carry, TestIrValue::Array(Array::scalar(19.0f32).unwrap()));
        assert_eq!(stack.read(), Ok(Array::vector(vec![2.0f32, 5.0, 11.0]).unwrap()));
        let destination = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let (carry_cotangent, _) = pullback
            .apply_with_destinations(
                CotangentSeed::Value(TestIrValue::Array(Array::scalar(1.0f32).unwrap())),
                (
                    CotangentDestination::Return,
                    CotangentDestination::Reference(TestIrValue::Reference(destination.clone())),
                ),
            )
            .unwrap();
        let discharged_cotangents = discharged
            .program()
            .linearize()
            .unwrap()
            .pullback()
            .unwrap()
            .interpret(vec![
                TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            discharged_cotangents,
            vec![carry_cotangent.unwrap(), TestIrValue::Array(destination.read().unwrap())],
        );
        assert_eq!(
            discharged_cotangents,
            vec![
                TestIrValue::Array(Array::scalar(25.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![13.0f32, 7.0, 4.0]).unwrap()),
            ],
        );
    }

    #[test]
    fn test_scan_transposition_drops_dead_reference_stack() {
        // A reference stack that the scan only stores into, and that nothing reads afterwards, has a provably zero
        // state cotangent: the scan input takes the `Ignore` kind, the reversed scan drops the stack and its body's
        // store, and the elements receive a structural zero cotangent.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let index = body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let root = body_builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        body_builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![root, carry, index],
                None,
            )
            .unwrap();
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, carry], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![doubled], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let initial = builder.add_input(scalar_type.into());
        let elements = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let stack = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let pullback = program.linearize().unwrap().pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = scan [carry_count=1, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = add %1 %1
                        in (%2)
                    },
                ]
                    %2:f32[3] = zero [type=f32[3]]
                in (%1, %2)"},
        );
        assert_eq!(
            pullback.interpret(vec![TestIrValue::Array(Array::scalar(1.0f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(8.0f32).unwrap()),
                TestIrValue::Array(Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scan_transposition_rejects_known_reference_stack() {
        // A known reference stack cannot reach a tangent program (the partial-evaluation split residualizes a scan
        // whose known side feeds a reference whole), so the transposition rule rejects it rather than reading a primal
        // reference inside the reversed body.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(stack_reading_body());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let stack = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let final_carry = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![initial, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`scan` transposition received a known reference-typed scanned input at position 1; \
                               reference stacks are linear or absent",
        ));
    }

    #[test]
    fn test_scan_transposition_reference_carry_destinations() {
        // `scan { add_update(r, x_i); y_i = read(r) }` over a reference carry: `y_i = r + Σ_{j ≤ i} x_j`, so with unit
        // output cotangents `x̄_j = length - j` and the destination ends holding `Σ ȳ_i = length`. The reference carry
        // is threaded through the reversed scan positionally and the body's view of it accumulates in place.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let element = body_builder.add_input(scalar_type.clone());
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![carry, element], None)
            .unwrap();
        let current =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![carry, current],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let elements = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference, elements], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The forwarded reference output shares the input's accumulator and has no cotangent slot, so the transposed
        // program consumes `[ȳs, r̄]` and produces `[r̄, x̄s]`.
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:ref<f32[]> .
                let %2:ref<f32[]>, %3:f32[3] = scan [carry_count=1, length=3, reverse=true] %1 %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[] .
                        let () = reference_add_update %1 %2
                            %3:f32[] = reference_read %1
                        in (%1, %3)
                    },
                ]
                in (%1, %3)"},
        );

        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
            ],
        );
        let scan = transposed
            .instructions()
            .iter()
            .find(|instruction| instruction.operation().name() == "scan")
            .unwrap();
        assert!(matches!(
            scan.operation(),
            TestIrOperation::Scan(operation) if operation.carry_count() == 1 && operation.reverse(),
        ));
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap()],
                vec![Array::scalar(0.0f32).unwrap()],
                vec![],
            ),
            vec![Array::vector(vec![3.0f32, 2.0, 1.0]).unwrap(), Array::scalar(3.0f32).unwrap()],
        );

        // The same program with a view of the carry inside the body accumulates into the root's accumulator through
        // the view: `add_update(r[1], x_i)` over `r: ref<f32[2]>` gives `x̄_i = r̄[1]`.
        let vector_reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(vector_reference_type.clone());
        let element = body_builder.add_input(scalar_type.clone());
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        body_builder
            .add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(element_transforms),
                Vec::new(),
                vec![carry, element],
                None,
            )
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(vector_reference_type.clone());
        let elements = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference, elements], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]> .
                let %1:ref<f32[2]>, %2:f32[3] = scan [carry_count=1, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[2]> .
                        let %2:f32[] = reference_read [transforms=[index(axis=0, index=1)]] %1
                        in (%1, %2)
                    },
                ]
                in (%0, %2)"},
        );

        assert_eq!(transposed.input_types(), vec![vector_reference_type]);
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::vector(vec![10.0f32, 20.0]).unwrap()],
                vec![]
            ),
            vec![Array::vector(vec![20.0f32, 20.0, 20.0]).unwrap(), Array::vector(vec![10.0f32, 20.0]).unwrap()],
        );
    }

    #[test]
    fn test_scan_transposition_write_only_reference_carry_destinations() {
        // `scan { write(r, x_i); y_i = x_i }` only stores into its reference carry. Under an `Ignore` destination for
        // the reference no later instruction accumulated into its root and the body never reads it, so its state
        // cotangent is provably zero: the body is transposed with an `Ignore` destination as well, the dead carry is
        // dropped from the reversed scan, and the pullback stages no cotangent reference at all instead of allocating,
        // zeroing, and freezing a dead accumulator around the reversed scan.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let stack_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let element = body_builder.add_input(scalar_type.clone());
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![carry, element], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![carry, element],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let elements = builder.add_input(stack_type.clone());
        let outputs = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, 3), vec![body], vec![reference, elements], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3] = scan [carry_count=0, length=3, reverse=true] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()],
                vec![],
                vec![]
            ),
            vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()],
        );

        // Under a `Reference` destination the carry's state cotangent is live, so the cotangent reference is threaded
        // through the reversed scan as a carry and the body's store transposes against it.
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:ref<f32[]> .
                let %2:ref<f32[]>, %3:f32[3] = scan [carry_count=1, length=3, reverse=true] %1 %0 [
                    body={
                        lambda %0:i64[], %1:ref<f32[]>, %2:f32[] .
                        let %3:f32[] = zero [type=f32[]]
                            %4:f32[] = reference_swap %1 %3
                            %5:f32[] = add %2 %4
                        in (%1, %5)
                    },
                ]
                in (%1, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![],
            ),
            vec![Array::vector(vec![1.0f32, 2.0, 8.0]).unwrap(), Array::scalar(0.0f32).unwrap()],
        );
    }

    #[test]
    fn test_stacked_scan_type_preserves_memory() {
        // Stacking prepends the scan length to the slice type and keeps the slice's memory placement.
        let slice_type = ArrayType::new_static(DataType::F64, [3]).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            stacked_scan_type(&slice_type, 2),
            ArrayType::new_static(DataType::F64, [2, 3]).with_memory(Memory::Host { pinned: true }),
        );
    }

    #[test]
    fn test_forward_scan_carries() {
        // The body maps `[scale, accumulator, reference]` to `[scale, accumulator + scale, reference]`. Only the
        // non-reference carry that the body passes through unchanged is forwarded from its input: an updated carry
        // keeps its scan output, and a reference carry keeps its scan output so that its later uses stay ordered after
        // the effects of the scan.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let scale = body.add_input(scalar_type.clone());
        let accumulator = body.add_input(scalar_type);
        let reference = body.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let next = body
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![accumulator, scale], None)
            .unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![scale, next, reference],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut outputs = vec!["final scale", "final accumulator", "final reference"];
        forward_scan_carries(body.entry_region_ref(), 3, &["scale", "accumulator", "reference"], &mut outputs);
        assert_eq!(outputs, vec!["scale", "final accumulator", "final reference"]);

        // Only the leading carry positions are considered.
        let mut outputs = vec!["final scale", "final accumulator", "final reference"];
        forward_scan_carries(body.entry_region_ref(), 0, &["scale", "accumulator", "reference"], &mut outputs);
        assert_eq!(outputs, vec!["final scale", "final accumulator", "final reference"]);
    }

    #[test]
    fn test_read_scan_iteration() {
        // Reading an iteration selects the slice at that index along the leading axis and drops that axis.
        let stack = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(read_scan_iteration(&stack, 0), Ok(Array::vector(vec![1.0, 2.0]).unwrap()));
        assert_eq!(read_scan_iteration(&stack, 2), Ok(Array::vector(vec![5.0, 6.0]).unwrap()));
        let stack = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(read_scan_iteration(&stack, 1), Ok(Array::scalar(2.0).unwrap()));
        assert_eq!(
            read_scan_iteration(&Array::scalar(1.0f64).unwrap(), 0),
            Err(TypeError::invalid(
                "`scan` cannot slice a stacked value of type `f64[]` along its scan axis (cannot remove dimension at \
                 index 0 for rank-0 array type); reshard it so that its scan axis is not sharded over explicit mesh \
                 axes",
            )
            .into()),
        );
    }

    #[test]
    fn test_write_scan_iteration() {
        // Writing an iteration stores the value as the slice at that index along the leading axis.
        let accumulator = Array::matrix(3, 2, vec![0.0; 6]).unwrap();
        assert_eq!(
            write_scan_iteration(accumulator, 2, Array::vector(vec![5.0, 6.0]).unwrap()),
            Ok(Array::matrix(3, 2, vec![0.0, 0.0, 0.0, 0.0, 5.0, 6.0]).unwrap()),
        );
        let accumulator = Array::vector(vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            write_scan_iteration(accumulator, 0, Array::scalar(4.0).unwrap()),
            Ok(Array::vector(vec![4.0, 0.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_batch_scan_with_interpreter_reports_first_iteration_errors_without_allocating_indices() {
        // No stack needs allocation here: the first body invocation fails, including for a reverse visit.
        for reverse in [false, true] {
            let mut visited = Vec::new();
            let error = BatchingError::UnsupportedOperation { message: "first iteration".to_string() };
            assert_eq!(
                batch_scan_with_interpreter::<Array, _, _, _>(
                    0,
                    MAX_DIMENSION_EXTENT,
                    reverse,
                    0,
                    &[],
                    |_| unreachable!(),
                    |_, _| unreachable!(),
                    |iteration, _| {
                        visited.push(iteration);
                        Err(error.clone())
                    },
                ),
                Err(error),
            );
            assert_eq!(visited, vec![if reverse { MAX_DIMENSION_EXTENT - 1 } else { 0 }]);
        }
    }

    #[test]
    fn test_scan_iteration_batch_axis() {
        // Removing the per-item leading scan dimension shifts every packed batch axis behind it left by one, while a
        // batch axis in front of it stays in place.
        assert_eq!(scan_iteration_batch_axis(BatchAxis::replicated()), BatchAxis::replicated());
        assert_eq!(scan_iteration_batch_axis(BatchAxis::new(0)), BatchAxis::new(0));
        assert_eq!(scan_iteration_batch_axis(BatchAxis::new(1)), BatchAxis::new(0));
        assert_eq!(scan_iteration_batch_axis(BatchAxis::new(2)), BatchAxis::new(1));
    }

    #[test]
    fn test_validate_reference_carry_axis() {
        // A reference carry that leaves the body at the axis it entered with is accepted whether mapped or replicated,
        // while any change of axis is rejected because the referent fixes the axis.
        assert_eq!(validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(0)), Ok(()));
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::replicated(), BatchAxis::replicated()),
            Ok(()),
        );
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(1)),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`scan` reference carry 1 enters carrying `axis 0` but its body returns it carrying `axis 1`; a \
                          reference carry cannot change its batch axis, so pass the reference as a batched input at \
                          the axis the body produces"
                        .to_string(),
            }),
        );
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 0, BatchAxis::replicated(), BatchAxis::new(0)),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`scan` reference carry 0 enters carrying `replicated` but its body returns it carrying `axis \
                          0`; a reference carry cannot change its batch axis, so pass the reference as a batched input \
                          at the axis the body produces"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_live_scan_signature_permutation() {
        // The live tangent entries follow all primal entries in JVP order. Scan order interleaves them so that the
        // carries, followed by the live carry tangents, lead the scanned entries and their live tangents.
        assert_eq!(live_scan_signature_permutation(&[true, false, true, true], 2), Ok(vec![0, 1, 4, 2, 3, 5, 6]));
        assert_eq!(live_scan_signature_permutation(&[false, true, false], 1), Ok(vec![0, 1, 2, 3]));
        assert_eq!(live_scan_signature_permutation(&[false, false], 2), Ok(vec![0, 1]));
        assert_eq!(
            live_scan_signature_permutation(&[true, true], 3),
            Err(ProgramError::MalformedProgram("`scan` carry count 3 exceeds fused body signature size 2".to_string())),
        );
    }

    #[test]
    fn test_reorder_program_boundary() {
        // Reordering permutes both public boundaries without changing the computation.
        let program = scalar_body(2, |builder, inputs| {
            let quotient =
                builder.add_instruction(DivOperation::new(), Vec::new(), vec![inputs[1], inputs[2]], None).unwrap()[0];
            vec![quotient, inputs[0]]
        });
        let reordered = reorder_program_boundary(&program, &[2, 0, 1], &[1, 0]).unwrap();
        assert_eq!(
            reordered.to_string(),
            indoc! {"
                lambda %0:f64[], %1:i64[], %2:f64[] .
                let %3:f64[] = div %2 %0
                in (%1, %3)
            "}
            .trim_end(),
        );

        // Nullary programs reorder only their outputs.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let first = builder.add_constant(Array::scalar(1.0).unwrap());
        let second = builder.add_constant(Array::scalar(2.0).unwrap());
        let nullary = builder.build(vec![first, second], Vec::new(), vec![Placeholder; 2]).unwrap();
        assert_eq!(
            reorder_program_boundary(&nullary, &[], &[1, 0]).unwrap().interpret(vec![]),
            Ok(vec![Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap(),])
        );

        // Each order must be a permutation of its boundary.
        assert_eq!(
            reorder_program_boundary(&program, &[0, 1], &[0, 1]).map(|_| ()),
            Err(ProgramError::MalformedProgram("input permutation has length 2 but boundary has length 3".to_string())),
        );
        assert_eq!(
            reorder_program_boundary(&program, &[0, 1, 3], &[0, 1]).map(|_| ()),
            Err(ProgramError::MalformedProgram("input permutation references out-of-range position 3".to_string())),
        );
        assert_eq!(
            reorder_program_boundary(&program, &[0, 1, 2], &[1, 1]).map(|_| ()),
            Err(ProgramError::MalformedProgram("output permutation references position 1 more than once".to_string())),
        );
    }

    #[test]
    fn test_thread_scan_carries() {
        // The transposed body exposes `[key_cotangent, accumulator_cotangent, index, known_key]`. The zero-space key
        // cotangent is an unused boundary input, so threading erases its slot and passes the known key value through
        // its carry slot instead of constructing a dynamically shaped zero, while the index moves to the front.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayIrType::Array(ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let accumulator_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let _key_cotangent = builder.add_input(key_type.cotangent().unwrap());
        let accumulator_cotangent = builder.add_input(accumulator_type.cotangent().unwrap());
        let _index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let _known_key = builder.add_input(key_type.clone());
        let transposed = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![accumulator_cotangent],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        let threaded = thread_scan_carries(
            transposed,
            &[key_type, accumulator_type],
            &[false, true],
            &[CotangentDestinationKind::Return; 2],
            &[false, false],
            2,
        )
        .unwrap();
        assert_eq!(
            threaded.to_string(),
            indoc! {"
                lambda %0:i64[], %1:u64[extent], %2:f64[] .
                in (%1, %2)
            "}
            .trim_end(),
        );
    }
}
