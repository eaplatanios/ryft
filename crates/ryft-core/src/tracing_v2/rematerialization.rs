//! Gradient rematerialization / checkpointing — the analogue of JAX's
//! [`jax.checkpoint` / `jax.remat`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
//!
//! [`rematerialize`] wraps a function so that reverse-mode differentiation through it trades memory for compute:
//! instead of storing every linearization residual produced inside the wrapped region, only the region's inputs
//! (plus any residuals selected by a [`RematerializationPolicy`]) are saved, and everything else is recomputed from
//! them in the backward pass.
//!
//! # Derivation Pipeline
//!
//! Each [`Rematerialize::call`] traces the wrapped body, linearizes it (see
//! [`Program::linearize`](Program::linearize) — the linearization's primal sub-program computes the body
//! outputs followed by every demanded residual), and then derives the three programs of one staged
//! [`RematerializeOperation`] as pure graph rewrites of the linearization's sub-programs:
//!
//!   1. **Classification** builds one [`RematerializationCandidate`] per classifiable instruction-produced residual —
//!      lazily following each producing operation's
//!      [`output_region_provenance`](Operation::output_region_provenance) through its attached regions, so a policy
//!      such as [`DotsSaveable`] sees dots inside scan bodies and condition branches rather than the outer
//!      higher-order operations — and consults the policy exactly once for each such residual, memoizing the returned
//!      [`RematerializationDecision`]s. Residuals whose recompute slice would reach a non-pure instruction are
//!      force-saved in producer-topological order, so effects execute exactly once.
//!   2. The **forward** program is the linearization primal with a rewritten output boundary: the body outputs, the
//!      region inputs, and the policy-saved residuals (behind their staged [`ResidualStorage`] store operations for
//!      stored payloads, e.g. an offloading memory transfer).
//!   3. The **backward** and **tangent** programs relocate the linearization's transposed pullback and tangent
//!      sub-programs onto the `(inputs..., saved..., cotangents-or-tangents...)` boundary, replacing each residual
//!      feeder with its saved payload (behind its staged restore operation) or with a memoized *recompute slice*
//!      copied from the primal program.
//!
//! The built-in policies mirror JAX's `jax.checkpoint_policies` (the name-based members classify residuals by the
//! [`tag`](crate::operations::tag::Tag::tag) key carried by the producing
//! [`TagOperation`]), and closure-backed custom policies wrap in [`PolicyFn`] with full access to every operation that
//! may have produced the boundary residual. Policies are pure classifiers: they never stage operations or touch
//! program builders, and reusable policies state their operation requirements as derive-generated variant-projection
//! bounds (e.g., a dot policy requires `for<'a> &'a O: TryInto<&'a DotOperation>`), so configuring a policy whose
//! operation capability is absent is a compile-time error at the configuration site.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use thiserror::Error;

use crate::arrays::{ArrayType, Memory};
use crate::batching::{
    BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentBatchingPolicy, DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, ResidualZeroProvider, TransposableOperation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::{AddOperation, DotOperation, TagOperation, TransferToMemoryOperation, Zero};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{
    Atom, AtomId, InstructionId, Operation, OperationFormatter, Program, ProgramBuilder, ProgramError, Region,
    RegionId, RegionInterface, RegionSlot, Type, TypeError, Typed, Value, ValueId,
};
use crate::tracing::{DomainTracer, Trace, TracingContext};
use crate::tracing_v2::operands::{check_nondifferentiated_tangents_are_zero, split_nondifferentiated};

/// Higher-order operation used by checkpointing/rematerialization.
///
/// [`RematerializeOperation`] has the same primal/forward/backward structure as
/// [`CustomVjpOperation`](crate::tracing_v2::custom_derivatives::CustomVjpOperation), but it also carries
/// a derived tangent program. That extra program is not user-authored custom-VJP state: it is produced by
/// [`Rematerialize`] so forward-mode differentiation can replay the rematerialized pushforward while reverse mode
/// replays the rematerialized pullback.
///
/// The `prevent_cse` flag is likewise rematerialization-specific. Backends may lower it as an optimization barrier
/// around rematerialized tangent/pullback outputs so compiler common-subexpression elimination does not undo the
/// requested memory/computation tradeoff.
///
/// The leading [`nondifferentiated_count`](Self::nondifferentiated_count) operands parameterize the call without being
/// differentiated: the primal and forward regions receive them in their own leading positions, the backward and
/// tangent regions receive them ahead of the forward tail, and they receive neither a tangent nor a cotangent. This is
/// the same operand split [`LinearCallOperation`](crate::LinearCallOperation) draws with its residual count, and the
/// direct analogue of JAX's `nondiff_argnums`. Batching is its canonical producer: a policy that threads batching
/// state through a structurally batched region's boundary (e.g., a composite universe's first-class mapped extent)
/// reintroduces that state as additional leading nondifferentiated operands of the batched call.
///
/// The `T` parameter fixes the type universe of every attached region and the call boundary. Each concrete payload
/// therefore has exactly one [`Operation<Type = T>`](Operation) contract, while the rematerialization algorithm remains one
/// shared implementation for all type universes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RematerializeOperation<T: Type> {
    /// Backend lowering hint requesting an optimization barrier around rematerialized backward/tangent outputs.
    prevent_cse: bool,

    /// Number of leading operands that parameterize the call without being differentiated.
    nondifferentiated_count: usize,

    /// Type universe in which the attached rematerialization regions operate.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> Copy for RematerializeOperation<T> {}

impl<T: Type> RematerializeOperation<T> {
    /// Creates a rematerialization operation. The primal, forward, backward, and tangent [`Program`]s are supplied
    /// separately as the operation's attached regions (via the region driver passed to
    /// [`Context::bind`]) in the region order `["primal", "forward", "backward", "tangent"]`;
    /// [`Operation::infer_output_types`] validates the forward, backward, and tangent interfaces against the
    /// primal interface.
    #[inline]
    pub fn new() -> Self {
        Self { prevent_cse: false, nondifferentiated_count: 0, marker: PhantomData }
    }

    /// Sets the number of leading operands that parameterize this call without being differentiated. Refer to the
    /// documentation of [`RematerializeOperation`] for the resulting region interfaces.
    #[inline]
    pub fn with_nondifferentiated_count(mut self, nondifferentiated_count: usize) -> Self {
        self.nondifferentiated_count = nondifferentiated_count;
        self
    }

    /// Returns the number of leading operands that parameterize this call without being differentiated.
    #[inline]
    pub fn nondifferentiated_count(&self) -> usize {
        self.nondifferentiated_count
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`). Without a barrier, a compiler may common-subexpression-eliminate
    /// values recomputed by the backward or tangent program against the forward pass, silently restoring the memory
    /// cost the rematerialization was meant to avoid.
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        self.prevent_cse = prevent_cse;
        self
    }

    /// Returns whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier.
    #[inline]
    pub fn prevent_cse(&self) -> bool {
        self.prevent_cse
    }

    /// Splits `values` into the leading nondifferentiated group and the trailing differentiated group.
    #[inline]
    fn split_inputs<'v, V>(&self, values: &'v [V]) -> Result<(&'v [V], &'v [V]), TypeError> {
        split_nondifferentiated(self.name(), self.nondifferentiated_count, values)
    }

    /// Validates the rematerialization contract over the four attached region interfaces
    /// (`["primal", "forward", "backward", "tangent"]` region order) and returns the primal interface; refer to the
    /// documentation of [`RematerializeOperation::new`] for the contract.
    fn validated_interfaces<'i>(
        &self,
        region_interfaces: &'i [RegionInterface<T>],
    ) -> Result<&'i RegionInterface<T>, TypeError> {
        if region_interfaces.len() != 4 {
            return Err(TypeError::invalid(format!(
                "rematerialize expects 4 attached regions but got {}",
                region_interfaces.len(),
            )));
        }
        let primal_interface = &region_interfaces[0];
        let forward_interface = &region_interfaces[1];
        let backward_interface = &region_interfaces[2];
        let tangent_interface = &region_interfaces[3];
        let input_types = primal_interface.input_types();
        let output_types = primal_interface.output_types();
        let (nondifferentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        check_types!(@same, "rematerialize forward input", [input_types, forward_interface.input_types()]);
        let forward_output_types = forward_interface.output_types();
        if forward_output_types.len() < output_types.len() {
            return Err(TypeError::invalid(format!(
                "rematerialize forward must produce at least the {} primal output(s) but produced {} value(s)",
                output_types.len(),
                forward_output_types.len(),
            )));
        }
        check_types!(@same, "rematerialize forward output", [
            output_types,
            &forward_output_types[..output_types.len()],
        ]);
        let residual_types = &forward_output_types[output_types.len()..];
        let expected_backward_input_types: Vec<T> =
            nondifferentiated_types.iter().chain(residual_types).chain(output_types.iter()).cloned().collect();
        check_types!(@same, "rematerialize backward input", [
            &expected_backward_input_types,
            backward_interface.input_types(),
        ]);
        check_types!(@same, "rematerialize backward output", [
            differentiated_types,
            backward_interface.output_types(),
        ]);
        let expected_tangent_input_types: Vec<T> =
            nondifferentiated_types.iter().chain(residual_types).chain(differentiated_types).cloned().collect();
        check_types!(@same, "rematerialize tangent input", [
            &expected_tangent_input_types,
            tangent_interface.input_types(),
        ]);
        check_types!(@same, "rematerialize tangent output", [output_types, tangent_interface.output_types()]);
        Ok(primal_interface)
    }
}

impl<T: Type> Default for RematerializeOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type> Display for RematerializeOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation for RematerializeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const {
            &[
                RegionSlot::computation("primal"),
                RegionSlot::rule("forward"),
                RegionSlot::rule("backward"),
                RegionSlot::rule("tangent"),
            ]
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        if region_interfaces.len() != 4 {
            return Err(TypeError::invalid(format!(
                "rematerialize expects 4 attached regions but got {}",
                region_interfaces.len(),
            )));
        }
        let primal_interface = &region_interfaces[0];
        let forward_interface = &region_interfaces[1];
        let primal_renaming = T::derive_identity_renaming(primal_interface.input_types(), input_types)?;
        let primal_output_types = primal_interface
            .output_types()
            .iter()
            .map(|r#type| r#type.rename_identities(&primal_renaming))
            .collect::<Result<Vec<_>, _>>()?;
        let forward_renaming = T::derive_identity_renaming(forward_interface.input_types(), input_types)?;
        let forward_output_types = forward_interface
            .output_types()
            .iter()
            .map(|r#type| r#type.rename_identities(&forward_renaming))
            .collect::<Result<Vec<_>, _>>()?;
        if forward_output_types.len() < primal_output_types.len() {
            return Err(TypeError::invalid(format!(
                "rematerialize forward must produce at least the {} primal output(s) but produced {} value(s)",
                primal_output_types.len(),
                forward_output_types.len(),
            )));
        }
        let residual_types = &forward_output_types[primal_output_types.len()..];
        let (nondifferentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        let backward_input_types = nondifferentiated_types
            .iter()
            .chain(residual_types)
            .chain(primal_output_types.iter())
            .cloned()
            .collect::<Vec<_>>();
        let tangent_input_types =
            nondifferentiated_types.iter().chain(residual_types).chain(differentiated_types).cloned().collect();
        Ok(vec![
            Some(input_types.to_vec()),
            Some(input_types.to_vec()),
            Some(backward_input_types),
            Some(tangent_input_types),
        ])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let primal_interface = self.validated_interfaces(region_interfaces)?;
        check_types!(@same, "rematerialize input", [primal_interface.input_types(), input_types]);
        Ok(primal_interface.output_types().to_vec())
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // A call whose operands are all differentiated renders as a bare name, so the nondifferentiated split appears
        // in rendered programs exactly where it exists.
        let operation = OperationFormatter::new(formatter, indentation, self.name())?;
        if self.nondifferentiated_count == 0 {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("nondifferentiated_count", self.nondifferentiated_count))
    }
}

impl<C: Domain> InterpretableOperation<C> for RematerializeOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`RematerializeOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<C: Context> PartiallyEvaluatableOperation<C> for RematerializeOperation<C::Type> where
    C::Operation: From<RematerializeOperation<C::Type>>
{
}

/// Capture-free forward-mode (JVP) rule for [`RematerializeOperation`]: replays the derived forward and tangent
/// programs through the active context, staging their operations in the shared builder.
///
/// Both derived programs are ordinary primal-enum programs, so the rule replays them through
/// [`Program::interpret_in_context`](crate::Program::interpret_in_context):
///
///   1. The forward program maps `inputs -> (outputs..., forward_tail...)`, where the tail is the region inputs
///      followed by the policy-saved residuals. Replaying it on the dual primals yields the primal outputs and the
///      forward tail; the tail is split off after the primal outputs.
///   2. The tangent program maps `(nondifferentiated..., forward_tail..., differentiated_input_tangents...) ->
///      output_tangents`, exactly the leading nondifferentiated operands and the forward tail followed by the
///      differentiated inputs' tangents (per [`RematerializeOperation::new`]'s signature validation), so those leading
///      operands and the tail are passed ahead of the dual tangents and replayed to produce the output tangents. The
///      tangent program
///      recomputes any unsaved residuals from the tail internally, so no residual reconstruction is needed here.
///   3. Each primal output is paired with its staged output tangent into a [`DifferentiationDual`].
///
/// Because both replayed programs are straight-line primal-enum operations referencing the staged tracers directly,
/// the rule introduces no symbolic capture and the enclosing partial-evaluation split discovers the residual
/// operand edges structurally — so
/// this is a leaf rule needing no nested differentiation or linearization request, and reverse mode transposes the
/// replayed recompute-and-pushforward operations like any other straight-line
/// tangent program. The [`prevent_cse`](RematerializeOperation::prevent_cse) optimization-barrier hint is
/// dropped in the forward (it is a backend lowering hint with no value-level semantics).
impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for RematerializeOperation<C::Type>
where
    C::Operation: ResidualZeroProvider<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The attached regions are `["primal", "forward", "backward", "tangent"]`; the primal interface provides
        // the boundary types.
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let tangent_region = driver.region(3)?;
        let output_count = primal_region.output_types().len();
        check_count!("input", inputs, primal_region.input_types().len(), ProgramError);

        // Replay the forward region on the dual primals, recovering the primal outputs followed by the forward tail
        // (region inputs plus policy-saved residuals) that the tangent region consumes.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = forward_region.interpret_in_context(context, primal_operands)?;
        if forward_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "rematerialize forward region produced {} outputs which is fewer than its {output_count} \
                 primal output(s)",
                forward_outputs.len(),
            ))
            .into());
        }
        let forward_tail = forward_outputs.split_off(output_count);
        let primal_outputs = forward_outputs;
        let (nondifferentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;

        check_nondifferentiated_tangents_are_zero(self.name(), nondifferentiated_inputs)?;

        // Replay the tangent region on `(nondifferentiated..., forward_tail..., differentiated_input_tangents...)`,
        // yielding one output tangent per primal output.
        let mut tangent_operands =
            nondifferentiated_inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        tangent_operands.extend(forward_tail);

        // The rematerialize call takes every differentiated input tangent as a real operand, so materialize structural
        // zeros against their own primal, which names every runtime quantity a reference-bearing tangent type omits;
        // static inputs keep the nullary zero.
        for input in differentiated_inputs {
            tangent_operands.push(C::Operation::materialize_zero_from_residual_sources(
                context,
                input.tangent().clone(),
                std::iter::once(input.primal()),
            )?);
        }

        let tangent_outputs = tangent_region.interpret_in_context(context, tangent_operands)?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

crate::impl_non_transposable_operation!(<T> RematerializeOperation<T> where T: Type);

/// Batching rule for [`RematerializeOperation`]. The primal and forward regions receive the wrapper operands' existing
/// axes, forward-tail residuals retain their natural axes, and the tangent region receives the nondifferentiated
/// operands' axes, those residual axes, and the differentiated operands' tangent axes. Corresponding primal, forward,
/// and tangent outputs are reconciled to one axis. The backward region receives the nondifferentiated, residual, and
/// reconciled output-cotangent axes, and mapped cotangents for replicated primal inputs are summed back to
/// replication. Rebuilding all four regions keeps the rematerialization boundary and its `prevent_cse` policy intact
/// without imposing a wrapper-wide axis position.
///
/// The batching policy owns the boundary shape of its structurally batched programs.
/// [`BatchingPolicy::adapt_batched_program`](crate::BatchingPolicy::adapt_batched_program) adapts each batched
/// region back to the plain rematerialization region boundary, and any
/// [`BatchingPolicy::boundary_operands`](crate::BatchingPolicy::boundary_operands) (e.g., a composite program's
/// first-class mapped extent) become additional leading
/// [nondifferentiated](RematerializeOperation::nondifferentiated_count) operands of the batched call, which is
/// precisely the operand role those bookkeeping values play: every region consumes them and none of them carries a
/// derivative.
impl<T: DifferentiableType, C: Context<Type = T>, P: CotangentBatchingPolicy<C>> BatchableOperation<C, P>
    for RematerializeOperation<T>
where
    C::Operation: From<RematerializeOperation<T>>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (nondifferentiated_axes, differentiated_axes) = self.split_inputs(input_axes.as_slice())?;
        let differentiated_axes = differentiated_axes.to_vec();
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let backward_region = driver.region(2)?;
        let tangent_region = driver.region(3)?;

        let naturally_batched_primal = driver.batch_program(
            context,
            primal_region,
            input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let primal_output_axes = naturally_batched_primal.output_axes();
        let naturally_batched_forward = driver.batch_program(
            context,
            forward_region,
            input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let forward_output_axes = naturally_batched_forward.output_axes();
        if forward_output_axes.len() < primal_output_axes.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "batched rematerialize forward region produced {} outputs which is fewer than its {} primal outputs",
                forward_output_axes.len(),
                primal_output_axes.len(),
            ))
            .into());
        }
        let (forward_primal_output_axes, residual_axes) = forward_output_axes.split_at(primal_output_axes.len());
        let residual_axes = residual_axes.to_vec();

        // The tangent region consumes the leading nondifferentiated operands and the exact forward tail, followed by
        // one tangent per differentiated wrapper input. Its natural output axes participate in the same boundary
        // decision as the ordinary primal and forward-prefix outputs.
        let tangent_input_axes = nondifferentiated_axes
            .iter()
            .chain(&residual_axes)
            .chain(&differentiated_axes)
            .copied()
            .collect::<Vec<_>>();
        let naturally_batched_tangent = driver.batch_program(
            context,
            tangent_region,
            tangent_input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let tangent_output_axes = naturally_batched_tangent.output_axes();
        check_count!("output", tangent_output_axes, primal_output_axes.len(), ProgramError);

        let output_axes = primal_output_axes
            .iter()
            .copied()
            .zip(forward_primal_output_axes.iter().copied())
            .zip(tangent_output_axes.iter().copied())
            .map(|((primal, forward), tangent)| {
                [primal, forward, tangent].into_iter().find(|axis| !axis.is_replicated()).unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let primal = context.align_and_adapt_batched_program_outputs(
            driver,
            primal_region,
            input_axes.as_slice(),
            naturally_batched_primal,
            output_axes.as_slice(),
        )?;
        let forward_required_output_axes =
            output_axes.iter().copied().chain(residual_axes.iter().copied()).collect::<Vec<_>>();
        let forward = context.align_and_adapt_batched_program_outputs(
            driver,
            forward_region,
            input_axes.as_slice(),
            naturally_batched_forward,
            forward_required_output_axes.as_slice(),
        )?;
        let tangent = context.align_and_adapt_batched_program_outputs(
            driver,
            tangent_region,
            tangent_input_axes.as_slice(),
            naturally_batched_tangent,
            output_axes.as_slice(),
        )?;

        // The backward region maps `(nondifferentiated..., forward_tail..., output_cotangents...)` to the
        // differentiated inputs' cotangents. Its structurally movable axes are aligned during batching; adaptation
        // sums a mapped cotangent when the corresponding primal input was replicated.
        let backward_input_axes =
            nondifferentiated_axes.iter().chain(&residual_axes).chain(&output_axes).copied().collect::<Vec<_>>();
        let batched_backward = driver.batch_program(
            context,
            backward_region,
            backward_input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(differentiated_axes.clone()),
        )?;
        let (backward, backward_output_axes) =
            P::adapt_batched_program(batched_backward, Some(differentiated_axes.as_slice()), P::sum_mapped_cotangents)?
                .into_parts();
        if backward_output_axes != differentiated_axes {
            return Err(BatchingError::MisalignedBatchAxes {
                message: format!(
                    "batched rematerialize backward output axes {backward_output_axes:?} do not match its \
                     differentiated input axes {differentiated_axes:?}",
                ),
            });
        }

        let boundary_operands = P::boundary_operands(context.axis_extent());
        let nondifferentiated_count = self.nondifferentiated_count + boundary_operands.len();
        let mut packed_inputs = boundary_operands;
        packed_inputs.extend(inputs.iter().map(P::value).cloned());
        let outputs = context.parent().bind(
            self.with_nondifferentiated_count(nondifferentiated_count),
            vec![primal, forward, backward, tangent],
            packed_inputs.as_slice(),
        )?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        outputs.into_iter().zip(output_axes).map(|(output, axis)| P::batch(output, axis)).collect()
    }
}

/// Represents rematerialization-specific errors.
///
/// [`RematerializationError`] and [`ProgramError`] deliberately form a normalized conversion cycle, mirroring
/// [`BatchingError`]: converting to [`ProgramError`] unwraps a
/// [`Program`](Self::Program) variant back into the program error that it carries and wraps every other variant in
/// [`ProgramError::Custom`], while converting to [`RematerializationError`] unwraps a [`ProgramError::Custom`]
/// payload holding a [`RematerializationError`] and wraps every other program error in [`Program`](Self::Program).
/// Round trips therefore never nest one error type inside the other, and `?` re-types errors correctly at both
/// boundaries.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RematerializationError {
    /// A [`RematerializationPolicy`] rejected saving one residual, aborting the whole transformation. The engine
    /// enriches the policy-supplied [`RematerializationRejection`] with the possible producing operation names and
    /// the rendered logical residual type.
    #[error(
        "policy rejected saving the residual of type {residual_type} produced by one of {operation_names:?}: \
         {rejection}"
    )]
    Rejected {
        /// Names of the operations that may have produced the rejected residual, in semantic provenance order.
        operation_names: Vec<String>,

        /// Rendered logical type of the rejected residual.
        residual_type: String,

        /// Policy-supplied rejection.
        rejection: RematerializationRejection,
    },

    /// A [`ResidualStorage`] operation violated its contract (it was not unary, single-result, and pure, or its
    /// restoration did not reproduce the logical residual type exactly).
    #[error("invalid residual storage operation: {message}")]
    InvalidStorageOperation {
        /// Human-readable description of the violated storage requirement.
        message: String,
    },

    /// An operation reported invalid residual provenance, such as an invalid attached-region or region-output index.
    #[error("unsupported residual provenance: {message}")]
    UnsupportedProvenance {
        /// Human-readable description of the unsupported provenance shape.
        message: String,
    },

    /// A non-rematerialization program error that surfaced while deriving the rematerialized programs.
    #[error(transparent)]
    Program(ProgramError),
}

impl From<ProgramError> for RematerializationError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(rematerialization) = error.downcast_custom::<RematerializationError>() {
            rematerialization.clone()
        } else {
            RematerializationError::Program(error)
        }
    }
}

impl From<RematerializationError> for ProgramError {
    #[inline]
    fn from(error: RematerializationError) -> Self {
        match error {
            RematerializationError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// Describes one operation output that may have produced a rematerialization residual.
///
/// Each producer exposes the complete concrete operation rather than a closed list of attributes selected by
/// `ryft-core`, so custom policies can inspect backend-specific variants and attributes directly. The application
/// types belong to this nested producer and may differ from the type of the outer residual boundary.
#[derive(Debug)]
pub struct RematerializationProducer<'a, T: Type, O: Operation<Type = T>> {
    /// Complete operation that may have produced the residual.
    operation: &'a O,

    /// Index of the producing operation output represented by this candidate, local to that producer's own
    /// instruction (not the outer boundary output index).
    output_index: usize,

    /// Abstract operand types at the producer's application site.
    input_types: Vec<T>,

    /// Abstract result types at the producer's application site.
    output_types: Vec<T>,
}

impl<'a, T: Type, O: Operation<Type = T>> RematerializationProducer<'a, T, O> {
    /// Returns the complete operation that may have produced the residual.
    #[inline]
    pub fn operation(&self) -> &'a O {
        self.operation
    }

    /// Returns the index of the producing operation output represented by this candidate, local to that producer's
    /// own instruction.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns the abstract operand types at the producer's application site.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        self.input_types.as_slice()
    }

    /// Returns the abstract result types at the producer's application site.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        self.output_types.as_slice()
    }
}

/// Policy-facing description of one rematerialization residual, passed to [`RematerializationPolicy::classify`].
///
/// A candidate represents the single value crossing the rematerialization boundary and contains every operation
/// output that may have produced it. Most candidates have one producer. A condition result may have one producer per
/// branch, in semantic branch order. Policies inspect the complete producer set and return one placement decision
/// for the boundary value; storage is never applied independently inside individual branches.
#[derive(Debug)]
pub struct RematerializationCandidate<'a, T: Type, O: Operation<Type = T>> {
    /// Operations that may have produced the residual, in stable semantic provenance order.
    producers: Vec<RematerializationProducer<'a, T, O>>,

    /// Abstract type of the value crossing the rematerialization boundary.
    residual_type: T,
}

impl<'a, T: Type, O: Operation<Type = T>> RematerializationCandidate<'a, T, O> {
    /// Returns the operations that may have produced this residual, in stable semantic provenance order.
    #[inline]
    pub fn producers(&self) -> &[RematerializationProducer<'a, T, O>] {
        self.producers.as_slice()
    }

    /// Returns the abstract type of the boundary residual. This may differ from the nested producers' result types;
    /// for example, a scan body producer may have a scalar result while the outer scan residual is stacked.
    #[inline]
    pub fn residual_type(&self) -> &T {
        &self.residual_type
    }

    /// Builds the classification candidate for `residual_atom` by recursively following the producing operations'
    /// [`output_region_provenance`](Operation::output_region_provenance) through `program`'s attached regions.
    /// Returns `None` when every provenance path ends at a region input or constant, which policies never see and
    /// which rematerialization always recomputes. Repeated leaf values are deduplicated while preserving their first
    /// occurrence in semantic provenance order.
    ///
    /// # Parameters
    ///
    ///   - `program`: Primal sub-program whose trailing outputs are the residuals.
    ///   - `residual_atom`: The residual output atom whose producing instruction is classified.
    ///   - `logical_residual_type`: Type of the residual value crossing the rematerialization boundary.
    fn from_program_residual<V>(
        program: &'a Program<V, O, Vec<V>, Vec<V>>,
        residual_atom: AtomId,
        logical_residual_type: T,
    ) -> Result<Option<Self>, RematerializationError>
    where
        V: Value<Type = T>,
    {
        let mut producers = Vec::new();
        let mut producer_values = HashSet::new();
        Self::resolve_producers(
            program,
            ValueId::new(program.entry(), residual_atom),
            &mut producer_values,
            &mut producers,
        )?;
        Ok((!producers.is_empty()).then_some(Self { producers, residual_type: logical_residual_type }))
    }

    /// Appends the leaf producers reachable from `value` to `producers` in semantic provenance order.
    fn resolve_producers<V>(
        program: &'a Program<V, O, Vec<V>, Vec<V>>,
        value: ValueId,
        producer_values: &mut HashSet<ValueId>,
        producers: &mut Vec<RematerializationProducer<'a, T, O>>,
    ) -> Result<(), RematerializationError>
    where
        V: Value<Type = T>,
    {
        let Some(instruction_id) = program.producer(value)? else {
            return Ok(());
        };
        let instruction = program.instruction(instruction_id)?;
        let output_index = instruction.outputs().iter().position(|output| *output == value.atom()).unwrap();
        let provenance = instruction.operation().output_region_provenance(output_index);
        if provenance.is_empty() {
            if producer_values.insert(value) {
                let region = program.region(value.region())?;
                let atom_type = |id: &AtomId| region.atoms()[id.index()].r#type().into_owned();
                producers.push(RematerializationProducer {
                    operation: instruction.operation(),
                    output_index,
                    input_types: instruction.inputs().iter().map(atom_type).collect(),
                    output_types: instruction.outputs().iter().map(atom_type).collect(),
                });
            }
            return Ok(());
        }

        for origin in provenance {
            let region_id = instruction.regions().get(origin.region_index).copied().ok_or_else(|| {
                RematerializationError::UnsupportedProvenance {
                    message: format!(
                        "operation '{}' reported provenance into region index {} but the instruction carries {} \
                         attached regions",
                        instruction.operation().name(),
                        origin.region_index,
                        instruction.regions().len(),
                    ),
                }
            })?;
            let region = program
                .region(region_id)
                .map_err(|error| RematerializationError::UnsupportedProvenance { message: error.to_string() })?;
            let atom = region.output_ids().get(origin.output_index).copied().ok_or_else(|| {
                RematerializationError::UnsupportedProvenance {
                    message: format!(
                        "operation '{}' reported provenance selecting output {} from a region with {} outputs",
                        instruction.operation().name(),
                        origin.output_index,
                        region.output_ids().len(),
                    ),
                }
            })?;
            Self::resolve_producers(program, ValueId::new(region_id, atom), producer_values, producers)?;
        }
        Ok(())
    }
}

/// Placement decision for one residual, returned by [`RematerializationPolicy::classify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RematerializationDecision<S> {
    /// Recompute the residual in the tangent and backward programs from the region inputs and other saved values.
    Recompute,

    /// Save the residual as a forward-program output consumed directly by the tangent and backward programs.
    Save,

    /// Save the residual behind a reversible storage transformation: the storage's store operation is staged on the
    /// forward side as the value crosses the boundary, and its restore operation is staged on the consuming side
    /// before any use.
    SaveWith(S),
}

/// Kind of a [`RematerializationRejection`].
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RematerializationRejectionKind {
    /// The residual's producing operation cannot participate in rematerialization.
    UnsupportedOperation,

    /// Policy-specific rejection that does not fit a more precise kind.
    Other,
}

/// Policy-supplied rejection of a rematerialization candidate, returned as the error side of
/// [`RematerializationPolicy::classify`]. Any *evaluated* rejection short-circuits the whole transformation
/// (including through combinators such as [`SaveFromBothPolicies`], via `?`).
///
/// The rejection is deliberately not generic over the type system: the rematerialization engine enriches it with the
/// possible producing operation names and the rendered logical residual type when wrapping it into
/// [`RematerializationError::Rejected`], which keeps backend-defined rejection reasons extensible without making the
/// error surface generic.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RematerializationRejection {
    /// Kind of this rejection.
    kind: RematerializationRejectionKind,

    /// Optional policy-supplied detail message.
    message: Option<String>,
}

impl RematerializationRejection {
    /// Creates a rejection reporting that the producing operation cannot participate in rematerialization.
    #[inline]
    pub fn unsupported_operation() -> Self {
        Self { kind: RematerializationRejectionKind::UnsupportedOperation, message: None }
    }

    /// Creates a policy-specific rejection carrying `message` as its detail.
    #[inline]
    pub fn other(message: impl Into<String>) -> Self {
        Self { kind: RematerializationRejectionKind::Other, message: Some(message.into()) }
    }

    /// Returns the kind of this rejection.
    #[inline]
    pub fn kind(&self) -> RematerializationRejectionKind {
        self.kind
    }

    /// Returns the optional policy-supplied detail message.
    #[inline]
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }
}

impl Display for RematerializationRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.kind, &self.message) {
            (RematerializationRejectionKind::UnsupportedOperation, None) => {
                write!(formatter, "the producing operation cannot participate in rematerialization")
            }
            (RematerializationRejectionKind::UnsupportedOperation, Some(message)) => {
                write!(formatter, "the producing operation cannot participate in rematerialization ({message})")
            }
            (_, Some(message)) => write!(formatter, "{message}"),
            (_, None) => write!(formatter, "the policy rejected the residual"),
        }
    }
}

/// Selects how residual-producing operation outputs are made available to the tangent and backward programs of a
/// rematerialized region — the analogue of the policy argument of JAX's
/// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
///
/// A residual is a value captured during linearization as a coefficient of the staged linear (tangent) map — for
/// example, `cos(x)` for `sin`, or the operand values for `mul`. Saved residuals are emitted as extra outputs of the
/// rematerialized region's forward program and consumed directly by its backward and tangent programs; unsaved
/// residuals are recomputed there from the region inputs. Residuals that are region inputs or constants are never
/// presented to policies: the backward and tangent programs always receive the region inputs, and constants are
/// re-created in place.
///
/// Policies are pure classifiers and must be deterministic for the lifetime of the [`Rematerialize`]
/// wrapper that owns them: each classifiable instruction-produced residual is classified exactly once per derivation,
/// derivations are cached by input types, and a policy that answered differently across calls would silently disagree
/// with its cached derivations.
///
/// Each candidate describes one boundary residual and every operation that may have produced it. This collection has
/// more than one entry for values produced by different condition branches. Built-in predicate policies save when any
/// possible producer matches; custom policies receive the complete ordered collection and return one decision for the
/// outer boundary value.
///
/// Reversible storage behavior (e.g., offloading a saved residual to pinned host memory) is described by the
/// associated [`Storage`](Self::Storage) type and returned through
/// [`SaveWith`](RematerializationDecision::SaveWith); the rematerialization engine stages the store and restore
/// operations, so policies with storage capabilities bring their own operation-type bounds (e.g.,
/// `O: From<TransferToMemoryOperation>` for [`MemoryTransferStorage`]) without imposing them on plain
/// rematerialization.
pub trait RematerializationPolicy<T: Type, O: Operation<Type = T>> {
    /// Reversible storage transformation available to this policy through
    /// [`SaveWith`](RematerializationDecision::SaveWith).
    type Storage: ResidualStorage<T, O>;

    /// Classifies one boundary residual from all of its possible producing operation outputs, or rejects the
    /// transformation.
    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<Self::Storage>, RematerializationRejection>;
}

/// Reversible transformation applied to a saved residual.
///
/// The rematerialization engine stages [`store_operation`](Self::store_operation) on the forward side as the saved
/// value crosses the boundary (the saved payload's type is the staged operation's inferred output type) and
/// [`restore_operation`](Self::restore_operation) on the consuming side before any use. Both operations must be
/// unary, single-result, and pure, and the restore operation's inferred output type must equal the logical residual
/// type exactly; the engine validates those shape properties when staging.
///
/// **Reversibility is an implementor law**: restoration must reproduce the logical residual *value* expected by the
/// tangent and backward programs, up to physical representation (e.g., a different memory placement expressible in
/// `T`). This cannot be validated structurally; the built-in implementations are verified numerically by tests.
pub trait ResidualStorage<T: Type, O: Operation<Type = T>> {
    /// Returns the operation producing the stored payload from the logical residual.
    fn store_operation(&self, logical_type: &T) -> Result<O, RematerializationError>;

    /// Returns the operation restoring the logical residual from the stored payload.
    fn restore_operation(&self, stored_type: &T, logical_type: &T) -> Result<O, RematerializationError>;
}

/// Uninhabited storage for policies that never return [`SaveWith`](RematerializationDecision::SaveWith). Because no
/// value of this type can be constructed, `SaveWith(NoStorage)` cannot be formed and the trait methods are trivially
/// exhaustive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoStorage {}

impl<T: Type, O: Operation<Type = T>> ResidualStorage<T, O> for NoStorage {
    fn store_operation(&self, _logical_type: &T) -> Result<O, RematerializationError> {
        match *self {}
    }

    fn restore_operation(&self, _stored_type: &T, _logical_type: &T) -> Result<O, RematerializationError> {
        match *self {}
    }
}

/// Storage that parks a saved residual in another memory space between the forward and backward passes: the store
/// operation transfers the residual to [`destination`](Self::destination) right after it is produced, and the
/// restore operation transfers it back to the logical residual's own memory right before it is consumed. Backends
/// legalize the staged transfers into their native placement machinery (the XLA backend lowers them to the
/// device-placement annotations consumed by its host-offloading pipeline), so offloaded residuals do not occupy
/// device memory between the two passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryTransferStorage {
    /// Destination [`Memory`] the saved residual is parked in.
    destination: Memory,
}

impl MemoryTransferStorage {
    /// Creates a new [`MemoryTransferStorage`] parking saved residuals in `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the destination [`Memory`] the saved residual is parked in.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<O> ResidualStorage<ArrayType, O> for MemoryTransferStorage
where
    O: Operation<Type = ArrayType> + From<TransferToMemoryOperation>,
{
    fn store_operation(&self, _logical_type: &ArrayType) -> Result<O, RematerializationError> {
        Ok(O::from(TransferToMemoryOperation::new(self.destination)))
    }

    fn restore_operation(
        &self,
        _stored_type: &ArrayType,
        logical_type: &ArrayType,
    ) -> Result<O, RematerializationError> {
        Ok(O::from(TransferToMemoryOperation::new(logical_type.memory())))
    }
}

/// Storage combinator backing [`SaveFromBothPolicies`]: the first policy's storage actions ride as
/// [`Left`](Self::Left) and the second's as [`Right`](Self::Right). Storage actions are never merged — combinators
/// select exactly one side's action per residual.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EitherStorage<S1, S2> {
    /// Storage action supplied by the first composed policy.
    Left(S1),

    /// Storage action supplied by the second composed policy.
    Right(S2),
}

impl<T: Type, O: Operation<Type = T>, S1, S2> ResidualStorage<T, O> for EitherStorage<S1, S2>
where
    S1: ResidualStorage<T, O>,
    S2: ResidualStorage<T, O>,
{
    fn store_operation(&self, logical_type: &T) -> Result<O, RematerializationError> {
        match self {
            Self::Left(storage) => storage.store_operation(logical_type),
            Self::Right(storage) => storage.store_operation(logical_type),
        }
    }

    fn restore_operation(&self, stored_type: &T, logical_type: &T) -> Result<O, RematerializationError> {
        match self {
            Self::Left(storage) => storage.restore_operation(stored_type, logical_type),
            Self::Right(storage) => storage.restore_operation(stored_type, logical_type),
        }
    }
}

/// Projects the concrete `P` payload out of an operation-family value through the derive-generated borrowed
/// `TryFrom` conversions, returning [`None`] when the family value is another variant.
#[inline]
fn project<'a, O, P>(operation: &'a O) -> Option<&'a P>
where
    &'a O: TryInto<&'a P>,
{
    operation.try_into().ok()
}

/// Returns the [`tag`](crate::operations::tag::Tag::tag) key when `operation` is a
/// [`TagOperation`](crate::operations::tag::TagOperation), and [`None`] otherwise.
#[inline]
fn tag_key<'a, T: Type + 'a, O>(operation: &'a O) -> Option<&'a str>
where
    &'a O: TryInto<&'a TagOperation<T>>,
{
    project::<O, TagOperation<T>>(operation).map(TagOperation::key)
}

/// Returns whether `operation` is a dot contraction without batch dimensions.
#[inline]
fn is_unbatched_dot<'a, O>(operation: &'a O) -> bool
where
    &'a O: TryInto<&'a DotOperation>,
{
    project::<O, DotOperation>(operation).is_some_and(|dot| {
        dot.dimensions().lhs_batching_dimensions().is_empty() && dot.dimensions().rhs_batching_dimensions().is_empty()
    })
}

/// Save nothing beyond the region inputs; recompute every residual in the backward pass. This is the default policy,
/// matching JAX's `nothing_saveable` (the default of `jax.checkpoint`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NothingSaveable;

impl<T: Type, O: Operation<Type = T>> RematerializationPolicy<T, O> for NothingSaveable {
    type Storage = NoStorage;

    fn classify(
        &self,
        _candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(RematerializationDecision::Recompute)
    }
}

/// Save every instruction-produced residual, making the rematerialization inert: the backward pass recomputes
/// nothing. Matches JAX's `everything_saveable`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EverythingSaveable;

impl<T: Type, O: Operation<Type = T>> RematerializationPolicy<T, O> for EverythingSaveable {
    type Storage = NoStorage;

    fn classify(
        &self,
        _candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(RematerializationDecision::Save)
    }
}

/// Save residuals produced by dot contractions and recompute the rest. Matches JAX's `dots_saveable`. Available
/// exactly in the operation families that contain [`DotOperation`], so configuring it for a scalar domain is a
/// compile-time error.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsSaveable;

impl<O> RematerializationPolicy<ArrayType, O> for DotsSaveable
where
    O: Operation<Type = ArrayType>,
    for<'a> &'a O: TryInto<&'a DotOperation>,
{
    type Storage = NoStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, ArrayType, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(
            match candidate
                .producers()
                .iter()
                .any(|producer| project::<O, DotOperation>(producer.operation()).is_some())
            {
                true => RematerializationDecision::Save,
                false => RematerializationDecision::Recompute,
            },
        )
    }
}

/// Save residuals produced by dot contractions whose [`DotDimensionNumbers`](crate::operations::math::DotDimensionNumbers)
/// have no batch dimensions and recompute the rest. Batched contractions behave more like cheap elementwise work per
/// batch element, so saving only the unbatched ones targets the genuinely expensive matrix products. Matches JAX's
/// `dots_with_no_batch_dims_saveable`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsWithNoBatchDimsSaveable;

impl<O> RematerializationPolicy<ArrayType, O> for DotsWithNoBatchDimsSaveable
where
    O: Operation<Type = ArrayType>,
    for<'a> &'a O: TryInto<&'a DotOperation>,
{
    type Storage = NoStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, ArrayType, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(match candidate.producers().iter().any(|producer| is_unbatched_dot(producer.operation())) {
            true => RematerializationDecision::Save,
            false => RematerializationDecision::Recompute,
        })
    }
}

/// Save only residuals tagged with one of the provided [`tag`](crate::operations::tag::Tag::tag) keys and recompute
/// everything else. Matches JAX's `save_only_these_names`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveOnlyTheseNames {
    /// Names whose residuals are saved.
    names: Vec<String>,
}

impl SaveOnlyTheseNames {
    /// Creates a new [`SaveOnlyTheseNames`] policy saving residuals tagged with one of `names`.
    #[inline]
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }
}

impl<T: Type, O> RematerializationPolicy<T, O> for SaveOnlyTheseNames
where
    O: Operation<Type = T>,
    for<'a> &'a O: TryInto<&'a TagOperation<T>>,
{
    type Storage = NoStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(
            match candidate.producers().iter().any(|producer| {
                tag_key::<T, O>(producer.operation())
                    .is_some_and(|name| self.names.iter().any(|configured_name| configured_name == name))
            }) {
                true => RematerializationDecision::Save,
                false => RematerializationDecision::Recompute,
            },
        )
    }
}

/// Save every *named* residual except those tagged with one of the provided names; unnamed residuals are recomputed.
/// Matches JAX's `save_any_names_but_these`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveAnyNamesButThese {
    /// Names whose residuals are recomputed.
    names: Vec<String>,
}

impl SaveAnyNamesButThese {
    /// Creates a new [`SaveAnyNamesButThese`] policy saving every named residual except those tagged with `names`.
    #[inline]
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }
}

impl<T: Type, O> RematerializationPolicy<T, O> for SaveAnyNamesButThese
where
    O: Operation<Type = T>,
    for<'a> &'a O: TryInto<&'a TagOperation<T>>,
{
    type Storage = NoStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(
            match candidate.producers().iter().any(|producer| {
                tag_key::<T, O>(producer.operation())
                    .is_some_and(|name| !self.names.iter().any(|configured_name| configured_name == name))
            }) {
                true => RematerializationDecision::Save,
                false => RematerializationDecision::Recompute,
            },
        )
    }
}

/// Save every instruction-produced residual except those tagged with one of the provided names. Matches JAX's
/// `save_anything_except_these_names`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveAnythingExceptTheseNames {
    /// Names whose residuals are recomputed.
    names: Vec<String>,
}

impl SaveAnythingExceptTheseNames {
    /// Creates a new [`SaveAnythingExceptTheseNames`] policy saving everything except residuals tagged with `names`.
    #[inline]
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }
}

impl<T: Type, O> RematerializationPolicy<T, O> for SaveAnythingExceptTheseNames
where
    O: Operation<Type = T>,
    for<'a> &'a O: TryInto<&'a TagOperation<T>>,
{
    type Storage = NoStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<NoStorage>, RematerializationRejection> {
        Ok(
            match candidate.producers().iter().any(|producer| {
                !tag_key::<T, O>(producer.operation())
                    .is_some_and(|name| self.names.iter().any(|configured_name| configured_name == name))
            }) {
                true => RematerializationDecision::Save,
                false => RematerializationDecision::Recompute,
            },
        )
    }
}

/// Combines two policies with lazy first-wins placement, matching JAX's `save_from_both_policies` and the previous
/// offloading combinator's semantics: the first policy classifies first, and its rejection propagates; a
/// [`Save`](RematerializationDecision::Save) or [`SaveWith`](RematerializationDecision::SaveWith) from the first
/// policy returns immediately (the second policy is *not* evaluated, so its rejections cannot fire and it never gets
/// to re-place a residual the first policy saved); only a [`Recompute`](RematerializationDecision::Recompute) from
/// the first policy consults the second. Storage actions are never merged: the first policy's actions ride as
/// [`EitherStorage::Left`] and the second's as [`EitherStorage::Right`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveFromBothPolicies<P1, P2> {
    /// Policy classifying first; its non-[`Recompute`](RematerializationDecision::Recompute) placement wins.
    first: P1,

    /// Policy consulted only when [`first`](Self::first) recomputes.
    second: P2,
}

impl<P1, P2> SaveFromBothPolicies<P1, P2> {
    /// Creates a new [`SaveFromBothPolicies`] combinator over `first` and `second`.
    #[inline]
    pub fn new(first: P1, second: P2) -> Self {
        Self { first, second }
    }
}

impl<T: Type, O: Operation<Type = T>, P1, P2> RematerializationPolicy<T, O> for SaveFromBothPolicies<P1, P2>
where
    P1: RematerializationPolicy<T, O>,
    P2: RematerializationPolicy<T, O>,
{
    type Storage = EitherStorage<P1::Storage, P2::Storage>;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<Self::Storage>, RematerializationRejection> {
        Ok(match self.first.classify(candidate)? {
            RematerializationDecision::Save => RematerializationDecision::Save,
            RematerializationDecision::SaveWith(storage) => {
                RematerializationDecision::SaveWith(EitherStorage::Left(storage))
            }
            RematerializationDecision::Recompute => match self.second.classify(candidate)? {
                RematerializationDecision::Save => RematerializationDecision::Save,
                RematerializationDecision::SaveWith(storage) => {
                    RematerializationDecision::SaveWith(EitherStorage::Right(storage))
                }
                RematerializationDecision::Recompute => RematerializationDecision::Recompute,
            },
        })
    }
}

/// Saves residuals tagged with one of the `saveable` [`tag`](crate::operations::tag::Tag::tag) keys in place,
/// offloads residuals tagged with one of the `offloadable` names to `destination`, and recomputes everything else
/// (including unnamed residuals). Matches JAX's `save_and_offload_only_these_names`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAndOffloadOnlyTheseNames {
    /// Names whose residuals are saved in their own memory space.
    saveable: Vec<String>,

    /// Names whose residuals are saved behind a transfer into [`destination`](Self::destination).
    offloadable: Vec<String>,

    /// Destination [`Memory`] for the offloaded residuals.
    destination: Memory,
}

impl SaveAndOffloadOnlyTheseNames {
    /// Creates a new [`SaveAndOffloadOnlyTheseNames`] policy.
    ///
    /// # Parameters
    ///
    ///   - `saveable`: Names whose residuals are saved in their own memory space.
    ///   - `offloadable`: Names whose residuals are saved behind a transfer into `destination`.
    ///   - `destination`: Destination [`Memory`] for the offloaded residuals.
    #[inline]
    pub fn new(
        saveable: impl IntoIterator<Item = impl Into<String>>,
        offloadable: impl IntoIterator<Item = impl Into<String>>,
        destination: Memory,
    ) -> Self {
        Self {
            saveable: saveable.into_iter().map(Into::into).collect(),
            offloadable: offloadable.into_iter().map(Into::into).collect(),
            destination,
        }
    }
}

impl<O> RematerializationPolicy<ArrayType, O> for SaveAndOffloadOnlyTheseNames
where
    O: Operation<Type = ArrayType> + From<TransferToMemoryOperation>,
    for<'a> &'a O: TryInto<&'a TagOperation<ArrayType>>,
{
    type Storage = MemoryTransferStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, ArrayType, O>,
    ) -> Result<RematerializationDecision<MemoryTransferStorage>, RematerializationRejection> {
        if candidate.producers().iter().any(|producer| {
            tag_key::<ArrayType, O>(producer.operation())
                .is_some_and(|name| self.saveable.iter().any(|configured_name| configured_name == name))
        }) {
            Ok(RematerializationDecision::Save)
        } else if candidate.producers().iter().any(|producer| {
            tag_key::<ArrayType, O>(producer.operation())
                .is_some_and(|name| self.offloadable.iter().any(|configured_name| configured_name == name))
        }) {
            Ok(RematerializationDecision::SaveWith(MemoryTransferStorage::new(self.destination)))
        } else {
            Ok(RematerializationDecision::Recompute)
        }
    }
}

/// Offloads residuals produced by dot contractions whose
/// [`DotDimensionNumbers`](crate::operations::math::DotDimensionNumbers) have no batch dimensions to
/// `destination` and recomputes the rest. Matches JAX's `offload_dot_with_no_batch_dims`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffloadDotsWithNoBatchDims {
    /// Destination [`Memory`] for the offloaded residuals.
    destination: Memory,
}

impl OffloadDotsWithNoBatchDims {
    /// Creates a new [`OffloadDotsWithNoBatchDims`] policy offloading unbatched dot residuals to `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }
}

impl<O> RematerializationPolicy<ArrayType, O> for OffloadDotsWithNoBatchDims
where
    O: Operation<Type = ArrayType> + From<TransferToMemoryOperation>,
    for<'a> &'a O: TryInto<&'a DotOperation>,
{
    type Storage = MemoryTransferStorage;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, ArrayType, O>,
    ) -> Result<RematerializationDecision<MemoryTransferStorage>, RematerializationRejection> {
        Ok(match candidate.producers().iter().any(|producer| is_unbatched_dot(producer.operation())) {
            true => RematerializationDecision::SaveWith(MemoryTransferStorage::new(self.destination)),
            false => RematerializationDecision::Recompute,
        })
    }
}

/// Closure-backed custom policy — the analogue of passing an arbitrary callable as a JAX `jax.checkpoint` policy.
///
/// The wrapped callable classifies each candidate directly. The storage type parameter `S` is carried as phantom
/// state so that closures returning storage-free decisions and closures returning storage actions both wrap without
/// a trait object (and therefore without imposing `'static` on the closure).
pub struct PolicyFn<F, S = NoStorage> {
    /// Classification callable.
    function: F,

    /// Phantom marker pinning the storage type produced by the callable's decisions.
    marker: PhantomData<fn() -> S>,
}

impl<F, S> PolicyFn<F, S> {
    /// Creates a new [`PolicyFn`] wrapping `function` as a rematerialization policy.
    #[inline]
    pub fn new(function: F) -> Self {
        Self { function, marker: PhantomData }
    }
}

impl<F: Clone, S> Clone for PolicyFn<F, S> {
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone(), marker: PhantomData }
    }
}

impl<F, S> Debug for PolicyFn<F, S> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("PolicyFn(..)")
    }
}

impl<T: Type, O: Operation<Type = T>, S, F> RematerializationPolicy<T, O> for PolicyFn<F, S>
where
    S: ResidualStorage<T, O>,
    F: Fn(&RematerializationCandidate<'_, T, O>) -> Result<RematerializationDecision<S>, RematerializationRejection>,
{
    type Storage = S;

    fn classify(
        &self,
        candidate: &RematerializationCandidate<'_, T, O>,
    ) -> Result<RematerializationDecision<S>, RematerializationRejection> {
        (self.function)(candidate)
    }
}

/// Validates one [`ResidualStorage`] operation against its input type and returns its single output type.
fn validate_storage_operation<T: Type, O: Operation<Type = T>>(
    operation: &O,
    input_type: &T,
) -> Result<T, RematerializationError> {
    let operation_name = operation.name();
    if !operation.effects().is_pure() {
        return Err(RematerializationError::InvalidStorageOperation {
            message: format!("storage operation '{operation_name}' must be pure but declares effects"),
        });
    }
    let mut output_types = operation.infer_output_types(std::slice::from_ref(input_type), &[]).map_err(|error| {
        RematerializationError::InvalidStorageOperation {
            message: format!(
                "storage operation '{operation_name}' rejected its single input of type {input_type}: {error}",
            ),
        }
    })?;
    if output_types.len() != 1 {
        return Err(RematerializationError::InvalidStorageOperation {
            message: format!(
                "storage operation '{operation_name}' must produce exactly one output but produced {}",
                output_types.len(),
            ),
        });
    }
    Ok(output_types.remove(0))
}

/// Stages one [`ResidualStorage`] operation over one already-staged value in a program under assembly and returns its
/// output atom after validating the storage contract.
fn stage_storage_operation<V: Value, O: Operation<Type = V::Type>>(
    builder: &mut ProgramBuilder<V, O>,
    operation: O,
    input: AtomId,
) -> Result<AtomId, ProgramError> {
    let input_type = builder.atoms()[input.index()].r#type().into_owned();
    validate_storage_operation(&operation, &input_type)?;
    let operation_name = operation.name();
    let outputs = builder.add_instruction(operation, Vec::new(), vec![input]).map_err(|error| {
        RematerializationError::InvalidStorageOperation {
            message: format!("storage operation '{operation_name}' could not be staged: {error}"),
        }
    })?;
    if outputs.len() != 1 {
        return Err(RematerializationError::InvalidStorageOperation {
            message: format!(
                "storage operation '{operation_name}' must produce exactly one output but produced {}",
                outputs.len(),
            ),
        }
        .into());
    }
    Ok(outputs[0])
}

/// Validates that a storage restoration reproduces the logical residual type exactly.
fn validate_restored_residual_type<T: Type>(restored_type: &T, logical_type: &T) -> Result<(), RematerializationError> {
    if restored_type != logical_type {
        return Err(RematerializationError::InvalidStorageOperation {
            message: format!(
                "storage restoration produced type {restored_type} but the logical residual type is {logical_type}",
            ),
        });
    }
    Ok(())
}

/// Returns whether the transitive recompute slice rooted at `root` stays pure: whether it reaches only pure
/// instructions before terminating at region inputs, constants, or the current saved cuts.
///
/// `safe` memoizes only *positive* answers, which stay valid as the classification pass upgrades residuals — cuts
/// only ever grow, and adding a cut can never make a pure slice impure. Negative answers are recomputed per root, so
/// the producer-topological upgrade pass in [`Rematerialize::call`] is deterministic: once an earlier root upgrades
/// to a saved cut, every later root's slice legitimately terminates there.
fn residual_slice_is_pure<V: Value, O: Operation<Type = V::Type>>(
    primal: &Program<V, O, Vec<V>, Vec<V>>,
    instruction_by_output: &[Option<usize>],
    cuts: &HashSet<usize>,
    safe: &mut HashSet<usize>,
    root: AtomId,
) -> bool {
    let mut stack = vec![root.index()];
    let mut visited = HashSet::new();
    while let Some(index) = stack.pop() {
        if safe.contains(&index) || cuts.contains(&index) || !visited.insert(index) {
            continue;
        }
        match instruction_by_output[index] {
            // Region inputs and constants terminate every slice.
            None => {}
            Some(instruction_index) => {
                let instruction = &primal.instructions()[instruction_index];
                // Instruction-level effects include the recursively derived effects of attached regions, so an
                // effect inside a nested body (e.g., a print in a scan body) also forces the save.
                let effects =
                    primal.instruction_effects(InstructionId::new(primal.entry(), instruction_index)).unwrap();
                if !effects.is_pure() {
                    return false;
                }
                stack.extend(instruction.inputs().iter().map(|input| input.index()));
            }
        }
    }
    safe.extend(visited);
    true
}

/// Copies memoized recompute slices out of a linearization's primal program into a reconstruction program under
/// assembly.
///
/// The resolver keeps two deliberately separate destination maps over primal atoms: the immutable [`cuts`](Self::cuts)
/// map (seeded up front with the region inputs and, per saved residual, the restored saved payload) and the
/// [`replayed`](Self::replayed) map recording the outputs of copied producer instructions. Resolution always
/// consults the cuts first, so for a producer `(a, b)` with `a` saved and `b` recomputed, the replayed instruction
/// also produces a replayed `a`, but every dependency on `a` resolves to the restored saved input — never the
/// replayed sibling. Each producer instruction is copied at most once (all of its outputs are recorded together),
/// slices terminate at cuts, region inputs, and constants, and — because the classification pass force-saves any
/// residual whose slice would reach a non-pure instruction — copying a non-pure instruction is an internal error.
struct PrimalSliceResolver<'p, V: Value, O> {
    /// Linearization primal program the slices are copied from.
    primal: &'p Program<V, O, Vec<V>, Vec<V>>,

    /// Producing-instruction index per primal atom index.
    instruction_by_output: Vec<Option<usize>>,

    /// Immutable saved-cut map from primal atom index to its destination atom: region inputs and restored saved
    /// residuals. Cuts always win over replayed siblings.
    cuts: Vec<Option<AtomId>>,

    /// Replay-output map from primal atom index to the destination atom produced by its copied instruction.
    replayed: Vec<Option<AtomId>>,

    /// Mapping preserving attached-region sharing across every producer instruction copied from the primal.
    region_remapping: HashMap<RegionId, RegionId>,
}

impl<'p, V: Value, O: Operation<Type = V::Type>> PrimalSliceResolver<'p, V, O> {
    /// Creates a resolver over `primal` whose region inputs resolve to `region_inputs` in the destination program.
    fn new(primal: &'p Program<V, O, Vec<V>, Vec<V>>, region_inputs: &[AtomId]) -> Self {
        let mut cuts = vec![None; primal.atoms().len()];
        for (position, input) in primal.input_ids().iter().enumerate() {
            cuts[input.index()] = Some(region_inputs[position]);
        }
        Self {
            primal,
            instruction_by_output: primal.instruction_by_output(),
            cuts,
            replayed: vec![None; primal.atoms().len()],
            region_remapping: HashMap::new(),
        }
    }

    /// Seeds one saved cut: primal `atom` resolves to `destination` (the saved input, or the staged restore output
    /// for stored payloads) in every slice.
    fn seed_cut(&mut self, atom: AtomId, destination: AtomId) {
        self.cuts[atom.index()] = Some(destination);
    }

    /// Resolves one already-terminal primal atom (a cut, an already-replayed output, or a constant) into the
    /// destination program.
    fn leaf(&mut self, atom: AtomId, builder: &mut ProgramBuilder<V, O>) -> Result<AtomId, ProgramError> {
        if let Some(destination) = self.cuts[atom.index()].or(self.replayed[atom.index()]) {
            return Ok(destination);
        }
        if let Atom::Constant(value) = &self.primal.atoms()[atom.index()] {
            let destination = builder.add_constant(value.clone());
            self.replayed[atom.index()] = Some(destination);
            return Ok(destination);
        }
        Err(ProgramError::UnboundAtomId { id: atom })
    }

    /// Resolves the primal `atom` into the destination program, copying its memoized recompute slice (every needed
    /// producer instruction, in primal instruction order) when it is not already available.
    fn resolve(&mut self, atom: AtomId, builder: &mut ProgramBuilder<V, O>) -> Result<AtomId, ProgramError> {
        if let Some(destination) = self.cuts[atom.index()].or(self.replayed[atom.index()]) {
            return Ok(destination);
        }
        if self.primal.atoms()[atom.index()].is_constant() {
            return self.leaf(atom, builder);
        }

        // Gather the not-yet-copied producer instructions the slice needs. `BTreeSet` iteration then copies them in
        // primal instruction order, so every copied instruction's inputs are terminal by the time it is copied.
        let mut needed = BTreeSet::new();
        let mut stack = vec![atom];
        while let Some(current) = stack.pop() {
            if self.cuts[current.index()].is_some()
                || self.replayed[current.index()].is_some()
                || self.primal.atoms()[current.index()].is_constant()
            {
                continue;
            }
            let Some(instruction_index) = self.instruction_by_output[current.index()] else {
                return Err(ProgramError::UnboundAtomId { id: current });
            };
            if needed.insert(instruction_index) {
                stack.extend(self.primal.instructions()[instruction_index].inputs().iter().copied());
            }
        }
        for instruction_index in needed {
            let instruction = &self.primal.instructions()[instruction_index];
            // The classification pass force-saves residual roots whose slices reach non-pure instructions
            // (including effects inside attached regions), so a recompute slice can never legitimately copy one.
            let effects =
                self.primal.instruction_effects(InstructionId::new(self.primal.entry(), instruction_index)).unwrap();
            if !effects.is_pure() {
                return Err(ProgramError::MalformedProgram(format!(
                    "rematerialization attempted to recompute the non-pure operation '{}'",
                    instruction.operation().name(),
                )));
            }
            let inputs =
                instruction.inputs().iter().map(|input| self.leaf(*input, builder)).collect::<Result<Vec<_>, _>>()?;
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    Ok(builder
                        .import_region_with_remapping(self.primal.region_ref(*region)?, &mut self.region_remapping))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let outputs = builder.add_instruction(instruction.operation().clone(), regions, inputs)?.to_vec();
            for (source_output, destination_output) in instruction.outputs().iter().zip(outputs) {
                // The replayed sibling of a saved output is recorded here, but `cuts` win at resolution, so
                // dependencies on the saved output keep resolving to the restored saved input.
                self.replayed[source_output.index()] = Some(destination_output);
            }
        }
        self.leaf(atom, builder)
    }
}

/// Assembles one reconstruction-side program (the tangent or backward program of a rematerialized region) over the
/// boundary `(inputs..., saved..., lead...)` — where `lead` is the input tangents for the tangent program and the
/// output cotangents for the backward program — by relocating the linearization's linear `source` program
/// (`(lead..., residuals...) -> outputs`) onto that boundary. Each residual feeder resolves to its saved payload
/// (behind a staged restore operation for stored payloads, validated to reproduce the logical residual type exactly)
/// or to its memoized recompute slice copied from the primal program. Effectful `source` instructions are relocated
/// as-is (they are part of the linear map's own semantics); only recompute slices are restricted to pure
/// instructions.
fn assemble_reconstruction_program<V, O, S>(
    primal: &Program<V, O, Vec<V>, Vec<V>>,
    source: &Program<V, O, Vec<V>, Vec<V>>,
    lead_count: usize,
    input_types: &[V::Type],
    saved_types: &[V::Type],
    residual_atoms: &[AtomId],
    residual_types: &[V::Type],
    decisions: &[RematerializationDecision<S>],
    saved_indices: &[usize],
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>,
    S: ResidualStorage<V::Type, O>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
{
    let mut builder = ProgramBuilder::<V, O>::new();
    let region_inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
    let saved_inputs = saved_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
    let source_input_types = source.input_types();
    let lead_inputs = source_input_types[..lead_count]
        .iter()
        .map(|r#type| builder.add_input(r#type.clone()))
        .collect::<Vec<_>>();

    // Seed the saved cuts, staging each stored payload's restore operation before any consumer and validating that
    // restoration reproduces the logical residual type exactly.
    let mut resolver = PrimalSliceResolver::new(primal, region_inputs.as_slice());
    for (slot, &index) in saved_indices.iter().enumerate() {
        let destination = match &decisions[index] {
            RematerializationDecision::Recompute => continue,
            RematerializationDecision::Save => saved_inputs[slot],
            RematerializationDecision::SaveWith(storage) => {
                let operation = storage.restore_operation(&saved_types[slot], &residual_types[index])?;
                let restored = stage_storage_operation(&mut builder, operation, saved_inputs[slot])?;
                let restored_type = builder.atoms()[restored.index()].r#type().into_owned();
                validate_restored_residual_type(&restored_type, &residual_types[index])?;
                restored
            }
        };
        resolver.seed_cut(residual_atoms[index], destination);
    }

    // Relocate the source program onto the destination boundary. Residual feeders resolve lazily, so recompute
    // slices are copied only for residuals the source program actually demands.
    let mut source_input_positions = vec![None; source.atoms().len()];
    for (position, input) in source.input_ids().iter().enumerate() {
        source_input_positions[input.index()] = Some(position);
    }
    let mut relocation: Vec<Option<AtomId>> = vec![None; source.atoms().len()];
    let source_region_ids = source
        .instructions()
        .iter()
        .flat_map(|instruction| instruction.regions().iter().copied())
        .collect::<Vec<_>>();
    let source_regions = source_region_ids
        .iter()
        .map(|region| source.region_ref(*region))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let relocated_regions = builder.import_regions(source_regions.as_slice())?;
    let region_relocation = source_region_ids.into_iter().zip(relocated_regions).collect::<HashMap<_, _>>();
    let lookup = |atom: AtomId,
                  relocation: &mut Vec<Option<AtomId>>,
                  resolver: &mut PrimalSliceResolver<'_, V, O>,
                  builder: &mut ProgramBuilder<V, O>|
     -> Result<AtomId, ProgramError> {
        if let Some(destination) = relocation[atom.index()] {
            return Ok(destination);
        }
        let destination = match source_input_positions[atom.index()] {
            Some(position) if position < lead_count => lead_inputs[position],
            Some(position) => resolver.resolve(residual_atoms[position - lead_count], builder)?,
            None => match &source.atoms()[atom.index()] {
                Atom::Constant(value) => builder.add_constant(value.clone()),
                Atom::Variable(_) => return Err(ProgramError::UnboundAtomId { id: atom }),
            },
        };
        relocation[atom.index()] = Some(destination);
        Ok(destination)
    };
    for instruction in source.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| lookup(*input, &mut relocation, &mut resolver, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;
        let regions = instruction
            .regions()
            .iter()
            .map(|region| {
                region_relocation.get(region).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!("region {region} was not imported during relocation"))
                })
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let outputs = builder.add_instruction(instruction.operation().clone(), regions, inputs)?.to_vec();
        for (source_output, destination_output) in instruction.outputs().iter().zip(outputs) {
            relocation[source_output.index()] = Some(destination_output);
        }
    }
    let output_ids = source
        .output_ids()
        .iter()
        .map(|output| lookup(*output, &mut relocation, &mut resolver, &mut builder))
        .collect::<Result<Vec<_>, _>>()?;

    let input_count = input_types.len() + saved_types.len() + lead_count;
    let output_count = output_ids.len();
    builder
        .build::<Vec<V>, Vec<V>>(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
        .into_simplified()
}

/// Function whose reverse-mode differentiation rematerializes interior values instead of storing them — the
/// ergonomic analogue of JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html),
/// built by [`rematerialize`].
///
/// The wrapped body is stored as a plain closure over [`DomainTracer`]s and nothing is derived at construction
/// time: each [`call`](Self::call) reads the input types off its tracer arguments, traces the body, derives the
/// forward and backward programs symbolically (specialized to those types and to the configured
/// [`RematerializationPolicy`]), and stages one [`RematerializeOperation`]. The derived forward program returns the
/// body outputs followed by the region inputs and the policy-saved residual values; the derived backward program
/// recomputes the remaining residuals from those saved values and replays the transposed tangent map. Reverse-mode
/// differentiation through the staged call therefore stores exactly the saved values — nothing interior — and both
/// derived programs are pruned of unreachable instructions, so saved residuals are genuinely not recomputed.
///
/// Unlike user-authored custom VJPs, the expansion also carries a derived *tangent program*, so forward-mode
/// differentiation works through rematerialized calls — matching `jax.checkpoint`, which supports `jvp`.
/// Un-differentiated calls replay the lean primal program and pay for neither residual computation nor saving.
///
/// Each [`call`](Self::call) caches its derivation inside the wrapper keyed by the flat input types — the analogue
/// of JAX caching traced rules on `(function, avals)` — so repeated calls with equal input types stage the
/// previously derived operation without re-tracing anything. Replacing the policy clears the cache in the newly typed
/// wrapper, and changing `prevent_cse` clears it because that flag is embedded in each staged operation.
///
/// The policy is a type parameter (defaulting to [`NothingSaveable`]) and every classifiable instruction-produced
/// residual goes through its [`RematerializationPolicy`] implementation exactly once, with the returned decisions
/// memoized across the derivation passes. Capability-requiring policies — for example, offloading policies whose
/// [`MemoryTransferStorage`] stages memory transfers — bring their own operation-type bounds without imposing them on
/// plain rematerialization.
pub struct Rematerialize<D: Domain, B, IT, OT, P = NothingSaveable>
where
    OT: Parameterized<DomainTracer<D>>,
{
    /// Closure computing the region output tree from the region input tree.
    body: B,

    /// Policy selecting which linearization residuals are saved (possibly offloaded) instead of recomputed.
    policy: P,

    /// Whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier;
    /// see [`Self::with_prevent_cse`].
    prevent_cse: bool,

    /// Derivations already produced by [`call`](Self::call), keyed by the flat input types they were specialized
    /// to. Entries hold the staged operation together with the body's output-tree structure. The handful of
    /// distinct input signatures a wrapper sees makes a linear scan cheaper and simpler than hashing types.
    cache: RefCell<Vec<CachedDerivation<D, OT>>>,

    /// Phantom marker pinning the [`Domain`] and the input and output tracer-tree types named by the body's
    /// signature. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT)>,
}

/// One [`Rematerialize`] cache entry: the flat input types a derivation was specialized to, the derived
/// rematerialization operation together with its four region programs (in
/// `["primal", "forward", "backward", "tangent"]` region order), and the structure of the body's output tree.
type CachedDerivation<D, OT> = (
    Vec<<D as Domain>::Type>,
    RematerializeOperation<<D as Domain>::Type>,
    Vec<
        Program<
            <D as Domain>::Constant,
            <D as Domain>::Operation,
            Vec<<D as Domain>::Constant>,
            Vec<<D as Domain>::Constant>,
        >,
    >,
    <OT as Parameterized<DomainTracer<D>>>::ParameterStructure,
);

/// Creates a [`Rematerialize`] function from a body closure over the [`Domain`] `D`'s tracers, with the default
/// [`NothingSaveable`] policy. Use [`Rematerialize::with_policy`] to select a different policy. Refer to the
/// documentation of [`Rematerialize`] for the derivation and rematerialization semantics.
pub fn rematerialize<D, B, IT, OT>(body: B) -> Rematerialize<D, B, IT, OT>
where
    D: Domain,
    B: Fn(IT) -> Result<OT, ProgramError>,
    OT: Parameterized<DomainTracer<D>>,
{
    Rematerialize {
        body,
        policy: NothingSaveable,
        prevent_cse: true,
        cache: RefCell::new(Vec::new()),
        marker: PhantomData,
    }
}

impl<D, B, IT, OT, P> Rematerialize<D, B, IT, OT, P>
where
    D: Domain,
    OT: Parameterized<DomainTracer<D>>,
{
    /// Replaces this rematerialization's policy, re-typing the wrapper to the new policy type — any
    /// [`RematerializationPolicy`] over the domain's type system and operation family. Policies with capability
    /// requirements (for example, offloading policies whose storage stages memory transfers) carry their own
    /// operation-type bounds, so configuring one in a domain without that capability fails to compile here. The
    /// derivation cache starts empty under the new policy.
    #[inline]
    pub fn with_policy<P2>(self, policy: P2) -> Rematerialize<D, B, IT, OT, P2>
    where
        P2: RematerializationPolicy<D::Type, D::Operation>,
    {
        Rematerialize {
            body: self.body,
            policy,
            prevent_cse: self.prevent_cse,
            cache: RefCell::new(Vec::new()),
            marker: PhantomData,
        }
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`), preventing the compiler from common-subexpression-eliminating
    /// the recomputed values against the forward pass — which would silently restore the memory cost the
    /// rematerialization was meant to avoid. Enabled by default, mirroring `jax.checkpoint`'s `prevent_cse=True`;
    /// disable it when the rematerialized region is staged somewhere CSE cannot reach (for example, under
    /// `jax.checkpoint`'s documented `scan` carve-out) and the barrier would inhibit useful optimizations.
    /// Offloaded residuals are unaffected either way: they cross through another memory space, which the compiler
    /// cannot common-subexpression-eliminate against the forward pass.
    #[inline]
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        // The flag is baked into every cached derivation (the staged operation carries it), so cached entries derived
        // with the old flag must not be served after the flag changes.
        if self.prevent_cse != prevent_cse {
            self.cache.borrow_mut().clear();
        }
        self.prevent_cse = prevent_cse;
        self
    }
}

impl<D, B, IT, OT, P> Rematerialize<D, B, IT, OT, P>
where
    D: Context,
    B: Fn(IT) -> Result<OT, ProgramError>,
    P: RematerializationPolicy<D::Type, <D as Domain>::Operation>,
    IT: Parameterized<DomainTracer<D>>,
    IT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
    OT: Parameterized<DomainTracer<D>>,
    OT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
    IT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = IT::Family,
            To<DomainTracer<D>> = IT,
            To<<D as Domain>::Constant> = IT::To<<D as Domain>::Constant>,
        >,
    OT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = OT::Family,
            To<DomainTracer<D>> = OT,
            To<<D as Domain>::Constant> = OT::To<<D as Domain>::Constant>,
        >,
    <D as Domain>::Operation: From<RematerializeOperation<<D as Domain>::Type>>
        + ResidualZeroProvider<D::Type>
        + From<AddOperation<D::Type>>
        + TransposableOperation<<D as Domain>::Constant, <D as Domain>::Operation>
        + DifferentiableOperation<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>
        + DifferentiableOperation<
            PartialEvaluationContext<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>,
        > + PartiallyEvaluatableOperation<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>,
    <D as Domain>::Type: DifferentiableType,
    Vec<<D as Domain>::Constant>: Parameterized<<D as Domain>::Constant, ParameterStructure = Vec<Placeholder>>,
{
    /// Stages this rematerialized function on the provided tracer inputs and returns its outputs, deriving the
    /// forward/backward programs specialized to the inputs' types. Reverse-mode differentiation of the staged call
    /// stores only the region inputs plus the policy-saved residuals and recomputes everything else.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        <D as Domain>::Type: DifferentiableType,
        V: Value<Type = D::Type>,
        V::DispatchDomain:
            Context<Type = D::Type, Constant = <D as Domain>::Constant, Operation = <D as Domain>::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_tracers = Vec::new();
        let structured_input_types = input
            .map_parameters(|tracer| {
                let r#type = tracer.r#type().into_owned();
                input_tracers.push(tracer);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_tracers.first() else {
            return Err(TypeError::invalid("rematerialization requires at least one input".to_string()).into());
        };
        let input_types = structured_input_types.parameters().cloned().collect::<Vec<_>>();

        // Stage a previously cached derivation when one exists for these input types, without re-tracing anything.
        let cached =
            self.cache.borrow().iter().find(|(cached_input_types, ..)| *cached_input_types == input_types).map(
                |(_, operation, operation_regions, output_structure)| {
                    (*operation, operation_regions.clone(), output_structure.clone())
                },
            );
        if let Some((operation, operation_regions, output_structure)) = cached {
            let operation = <D as Domain>::Operation::from(operation);
            let context = first.dispatch_domain();
            let outputs = context.bind(operation, operation_regions, &input_tracers)?;
            return Ok(Parameterized::from_parameters(output_structure, outputs)?);
        }

        let (structured_output_types, primal) = D::trace(|xs| (self.body)(xs), structured_input_types.clone())?;
        let primal = primal.to_flat_program();
        let output_types = structured_output_types.parameters().cloned().collect::<Vec<_>>();
        let output_count = output_types.len();
        let input_count = input_types.len();

        // Build the capture-free linearization of the body once. Its primal sub-program computes the body
        // outputs followed by every linearization residual (its trailing `residual_count` outputs), and its tangent
        // sub-program is the linear tangent map over `[input_tangents..., residuals...]`. The three derived programs
        // below all replay these two sub-programs, so the residual order is fixed once here and shared across them.
        let linearization = primal.linearize()?;
        let residual_count = linearization.residual_count();
        let residual_atoms = linearization.primal().output_ids()[output_count..].to_vec();
        let residual_types = linearization.primal().output_types().split_off(output_count);

        // Classify each instruction-produced residual exactly once from the provenance recovered from the primal
        // sub-program (the operation defining the residual atom, looked through nested provenance), memoizing the
        // returned decisions — storage values included — so the derivation passes below all reuse them. Residuals that
        // are region inputs, constants, or explicitly absent have no classifiable producer, so
        // `from_program_residual` returns `None` and they are never saved — the backward program always receives the
        // region inputs and recomputes everything else, exactly as before. A policy rejection aborts the whole
        // transformation.
        let mut decisions = Vec::with_capacity(residual_count);
        for index in 0..residual_count {
            let candidate = RematerializationCandidate::from_program_residual(
                linearization.primal(),
                residual_atoms[index],
                residual_types[index].clone(),
            )?;
            decisions.push(match candidate {
                None => RematerializationDecision::Recompute,
                Some(candidate) => self.policy.classify(&candidate).map_err(|rejection| {
                    ProgramError::from(RematerializationError::Rejected {
                        operation_names: candidate
                            .producers()
                            .iter()
                            .map(|producer| producer.operation().name().to_string())
                            .collect(),
                        residual_type: candidate.residual_type().to_string(),
                        rejection,
                    })
                })?,
            });
        }
        // Force-save upgrades, in producer-topological order: a `Recompute` residual whose recompute slice would
        // reach a non-pure instruction before terminating at a saved cut, a region input, or a constant is upgraded
        // to `Save` — recompute slices may only ever copy pure instructions, so the effect executes exactly once, in
        // the forward program. Roots are processed in ascending producing-instruction order, so once an earlier root
        // upgrades, every later root's slice legitimately terminates at the new cut (later producers cannot be
        // ancestors of earlier roots), making the pass deterministic.
        let instruction_by_output = linearization.primal().instruction_by_output();
        let mut cuts = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| !matches!(decision, RematerializationDecision::Recompute))
            .map(|(index, _)| residual_atoms[index].index())
            .collect::<HashSet<_>>();
        let mut recompute_roots = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| matches!(decision, RematerializationDecision::Recompute))
            .filter_map(|(index, _)| {
                instruction_by_output[residual_atoms[index].index()].map(|producer| (producer, index))
            })
            .collect::<Vec<_>>();
        recompute_roots.sort_unstable();
        let mut safe = HashSet::new();
        for (_, index) in recompute_roots {
            let root = residual_atoms[index];
            if !residual_slice_is_pure(linearization.primal(), &instruction_by_output, &cuts, &mut safe, root) {
                decisions[index] = RematerializationDecision::Save;
                cuts.insert(root.index());
            }
        }

        let saved_indices = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| !matches!(decision, RematerializationDecision::Recompute))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let saved_count = saved_indices.len();

        // Assemble the forward program directly from the primal sub-program: it is the primal with a rewritten
        // output boundary — the body outputs, then the region inputs, then the policy-saved residuals (behind their
        // staged store operations for stored payloads, whose inferred output types become the saved types, so an
        // offloaded residual naturally carries its destination memory). `into_simplified` prunes the producers of
        // dropped (recomputed) residual outputs; effectful instructions stay rooted and therefore execute exactly
        // once, in the forward program.
        let (forward, saved_types) = {
            let source = linearization.primal();
            let mut atoms = source.atoms().to_vec();
            let mut instructions = source.instructions().to_vec();
            let mut output_ids = source.output_ids()[..output_count].to_vec();
            output_ids.extend(source.input_ids().iter().copied());
            let mut saved_types = Vec::with_capacity(saved_count);
            for &index in saved_indices.iter() {
                match &decisions[index] {
                    RematerializationDecision::Recompute => continue,
                    RematerializationDecision::Save => {
                        saved_types.push(residual_types[index].clone());
                        output_ids.push(residual_atoms[index]);
                    }
                    RematerializationDecision::SaveWith(storage) => {
                        let operation = storage.store_operation(&residual_types[index])?;
                        let stored_type = validate_storage_operation(&operation, &residual_types[index])?;
                        let stored = AtomId::new(atoms.len());
                        atoms.push(Atom::Variable(stored_type.clone()));
                        instructions.push(crate::programs::Instruction::new(
                            operation,
                            vec![residual_atoms[index]],
                            vec![stored],
                            Vec::new(),
                        ));
                        saved_types.push(stored_type);
                        output_ids.push(stored);
                    }
                }
            }
            let output_structure = vec![Placeholder; output_ids.len()];
            // The rewritten boundary replaces only the entry region; the attached regions of copied instructions
            // stay valid because the rest of the arena is carried over verbatim (the entry region's identifier is
            // assigned last, so the copied instructions' region ids are unchanged).
            let mut regions = source.regions().iter().cloned().collect::<Vec<_>>();
            regions[source.entry().index()] =
                Region { atoms, input_ids: source.input_ids().to_vec(), output_ids, instructions };
            let forward = Program::new(vec![Placeholder; input_count], output_structure, regions, source.entry())?;
            (forward.into_simplified()?, saved_types)
        };

        // Assemble the backward program `(inputs..., saved..., output_cotangents...) -> input_cotangents` by
        // relocating the transposed pullback `(output_cotangents..., residuals...)` onto that boundary: saved
        // residual feeders map to the saved inputs (behind staged restore operations for stored payloads) and
        // recomputed feeders to memoized recompute slices copied from the primal program.
        let pullback = linearization.pullback()?;
        let backward = assemble_reconstruction_program(
            linearization.primal(),
            &pullback,
            output_count,
            input_types.as_slice(),
            saved_types.as_slice(),
            residual_atoms.as_slice(),
            residual_types.as_slice(),
            decisions.as_slice(),
            saved_indices.as_slice(),
        )?;

        // Assemble the tangent program `(inputs..., saved..., input_tangents...) -> output_tangents` the same way
        // from the linearization's tangent sub-program `(input_tangents..., residuals...)`, so that forward-mode
        // differentiation works through the rematerialized call (JAX's `jax.checkpoint` also supports `jvp`).
        let tangent = assemble_reconstruction_program(
            linearization.primal(),
            linearization.tangent(),
            input_count,
            input_types.as_slice(),
            saved_types.as_slice(),
            residual_atoms.as_slice(),
            residual_types.as_slice(),
            decisions.as_slice(),
            saved_indices.as_slice(),
        )?;

        let operation = RematerializeOperation::new().with_prevent_cse(self.prevent_cse);
        let operation_regions = vec![primal, forward, backward, tangent];
        let output_structure = structured_output_types.parameter_structure();
        self.cache
            .borrow_mut()
            .push((input_types, operation, operation_regions.clone(), output_structure.clone()));
        let context = first.dispatch_domain();
        let outputs = context.bind(<D as Domain>::Operation::from(operation), operation_regions, &input_tracers)?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use std::fmt::Debug;
    use std::rc::Rc;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Dimension, Memory, Shape, ShardingDimension};
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::operations::{Cos, Dot, DotDimensionNumbers, ScanOperation, Sin, Tag};
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::RegionRole;

    use super::*;

    /// Shorthand for the policy contract over the `Array` array domain used throughout these tests.
    trait TestPolicy: RematerializationPolicy<ArrayType, ArrayOperation<Array>> + Clone + Debug {}

    impl<P: RematerializationPolicy<ArrayType, ArrayOperation<Array>> + Clone + Debug> TestPolicy for P {}

    /// One staged [`RematerializeOperation`] together with its materialized region programs (in
    /// `["primal", "forward", "backward", "tangent"]` region order), exposing the same accessors the payload-owned
    /// operation used to so the structural assertions below stay readable.
    struct StagedRematerialization {
        regions: Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
    }

    impl StagedRematerialization {
        fn forward(&self) -> &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            &self.regions[1]
        }

        fn backward(&self) -> &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            &self.regions[2]
        }

        fn tangent(&self) -> &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            &self.regions[3]
        }
    }

    /// Traces `function.call` over one `input_type` input and returns the staged [`RematerializeOperation`]
    /// together with its materialized region programs.
    fn staged_operation<B, P>(
        function: &Rematerialize<
            EagerContext<Array, ArrayOperation<Array>>,
            B,
            DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            P,
        >,
        input_type: ArrayType,
    ) -> StagedRematerialization
    where
        B: Fn(
            DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError>,
        P: TestPolicy,
    {
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), input_type).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let regions = instruction
            .regions()
            .iter()
            .map(|region| program.region_ref(*region).unwrap().to_program())
            .collect::<Vec<_>>();
        StagedRematerialization { regions }
    }

    /// Computes `f(x) = u * sin(u)` with `u = x · x`, whose linearization residuals span all three policy classes:
    /// `u` is produced by a dot, `sin(u)` by a sine, and the sine rule's `cos(u)` factor by a cosine.
    fn dot_sine<V>(x: V) -> V
    where
        V: Clone + Sin + Dot + std::ops::Mul<Output = V>,
    {
        let u = x.dot(&x, &DotDimensionNumbers::inner_product());
        u.clone() * u.sin().unwrap()
    }

    /// [`dot_sine`] in the closure shape consumed by [`rematerialization`].
    fn dot_sine_body(
        input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        Ok(dot_sine(input))
    }

    /// Reference gradient of [`dot_sine_body`]: `∇f(x) = (sin(u) + u * cos(u)) * 2x` with `u = x · x`.
    fn dot_sine_gradient(x: &[f64]) -> Vec<f64> {
        let u: f64 = x.iter().map(|value| value * value).sum();
        x.iter().map(|value| (u.sin() + u * u.cos()) * 2.0 * value).collect()
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(size)]))
    }

    #[test]
    fn test_rematerialize_effects_only_include_the_primal_region() {
        let operation = RematerializeOperation::<ArrayType>::new();
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert!((1..4).all(|index| operation.region_role(index) == Some(RegionRole::Rule)));
    }

    #[test]
    fn test_rematerialization_matches_the_unrematerialized_gradient_under_every_policy() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (direct_value, direct_gradient) = domain.value_and_gradient(|x| dot_sine(x), input.clone()).unwrap();
        fn check(policy: impl TestPolicy, input: &Array, direct_value: &Array, expected_gradient: &[f64]) {
            let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body).with_policy(policy);
            let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            assert_abs_diff_eq!(value.to_f64s()[0], direct_value.to_f64s()[0], epsilon = 1e-9);
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
            }
        }
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(direct_gradient.to_f64s()[index], *expected, epsilon = 1e-9);
        }
        check(NothingSaveable, &input, &direct_value, &expected_gradient);
        check(EverythingSaveable, &input, &direct_value, &expected_gradient);
        check(DotsSaveable, &input, &direct_value, &expected_gradient);
    }

    /// Candidate classification follows stacked residual edges through `scan` boundaries to the body instructions
    /// that produce them, so structural policies see loop-interior operations: `DotsSaveable` saves exactly the
    /// per-iteration dot stack of a loop body (one more forward output than `NothingSaveable`), and every policy
    /// still produces the reference gradient (unsaved loop residuals recompute through the replayed known scan).
    #[test]
    fn test_rematerialization_policies_classify_residuals_inside_scan_bodies() {
        // Loop body `[c, x] -> [c * (x · x)]` over three two-element rows: `f(c0) = c0 * Π |xᵢ|²`.
        let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let squared_norms: Vec<f64> = rows.iter().map(|row| row.iter().map(|value| value * value).sum()).collect();
        let expected_gradient: f64 = squared_norms.iter().product();

        let body = {
            use crate::parameters::Placeholder;
            use crate::programs::ProgramBuilder;

            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let carry = builder.add_input(ArrayType::scalar(DataType::F64));
            let row = builder.add_input(vector_type(2));
            let dot = builder
                .add_instruction(
                    crate::operations::math::DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    vec![row, row],
                )
                .unwrap()[0];
            let next = builder
                .add_instruction(crate::operations::math::MulOperation::new(), Vec::new(), vec![carry, dot])
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder; 2], vec![Placeholder; 1])
                .unwrap()
        };
        let stacked = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
            rows.iter().flatten().copied().collect(),
        );
        let scan_body = |carry: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let context = carry.context().clone();
            let xs = StagingContext::constant(&context, stacked.clone());
            let scan = ScanOperation::new(1, 3);
            let outputs =
                context.stage_operation(ArrayOperation::Scan(scan), vec![body.clone()], &[carry, xs])?;
            Ok(outputs.into_iter().next().unwrap())
        };

        fn check(
            policy: impl TestPolicy,
            scan_body: impl Clone
            + Fn(
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )
                -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError>,
            expected_gradient: f64,
        ) -> usize {
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(scan_body).with_policy(policy);
            let operation = staged_operation(&function, ArrayType::scalar(DataType::F64));
            let forward_output_count = operation.forward().output_types().len();

            let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
                .value_and_gradient(|carry| function.call(carry).unwrap(), Array::scalar(2.0))
                .unwrap();
            assert_abs_diff_eq!(value.to_f64s()[0], 2.0 * expected_gradient, epsilon = 1e-9);
            assert_abs_diff_eq!(gradient.to_f64s()[0], expected_gradient, epsilon = 1e-9);
            forward_output_count
        }
        let baseline = check(NothingSaveable, scan_body.clone(), expected_gradient);
        let with_dots = check(DotsSaveable, scan_body, expected_gradient);
        // `DotsSaveable` saves exactly the stacked per-iteration dot outputs (one `[3]`-shaped residual) beyond the
        // `NothingSaveable` baseline of region output plus region input.
        assert_eq!(with_dots, baseline + 1);
    }

    #[test]
    fn test_condition_outputs_expose_all_possible_branch_producers() {
        use crate::operations::{ConditionOperation, DotOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        let vector_type = vector_type(2);
        let branch = |tag_output| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let lhs = builder.add_input(vector_type.clone());
            let rhs = builder.add_input(vector_type.clone());
            let dot = builder
                .add_instruction(
                    ArrayOperation::Dot(DotOperation::new(DotDimensionNumbers::inner_product())),
                    Vec::new(),
                    vec![lhs, rhs],
                )
                .unwrap()[0];
            let output = if tag_output {
                builder
                    .add_instruction(crate::operations::tag::TagOperation::new("false"), Vec::new(), vec![dot])
                    .unwrap()[0]
            } else {
                dot
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let lhs = builder.add_input(vector_type.clone());
        let rhs = builder.add_input(vector_type.clone());
        let condition_regions = vec![branch(false), branch(true)];
        let condition = ConditionOperation::new();
        let regions = condition_regions
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let output = builder
            .add_instruction(ArrayOperation::Condition(condition), regions, vec![predicate, lhs, rhs])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let candidate =
            RematerializationCandidate::from_program_residual(&program, output, ArrayType::scalar(DataType::F64))
                .unwrap()
                .unwrap();
        assert_eq!(candidate.producers().len(), 2);
        assert!(matches!(candidate.producers()[0].operation(), ArrayOperation::Dot(_)));
        assert!(matches!(candidate.producers()[1].operation(), ArrayOperation::Tag(_)));
        assert_eq!(
            candidate.producers().iter().map(|producer| producer.output_index()).collect::<Vec<_>>(),
            vec![0, 0],
        );

        let policy_calls = Rc::new(RefCell::new(0));
        let recorded_names = Rc::new(RefCell::new(Vec::new()));
        let recorded_calls = policy_calls.clone();
        let recorded_producers = recorded_names.clone();
        let policy =
            PolicyFn::new(move |candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
                *recorded_calls.borrow_mut() += 1;
                recorded_producers
                    .borrow_mut()
                    .extend(candidate.producers().iter().map(|producer| producer.operation().name().to_string()));
                Ok::<_, RematerializationRejection>(RematerializationDecision::<NoStorage>::Save)
            });
        assert_eq!(policy.classify(&candidate), Ok(RematerializationDecision::Save));
        assert_eq!(*policy_calls.borrow(), 1, "one boundary residual receives one policy decision");
        assert_eq!(recorded_names.borrow().as_slice(), &["dot", "tag"]);

        let storage_policy = SaveAndOffloadOnlyTheseNames::new(["false"], ["false"], PINNED_HOST);
        assert_eq!(
            storage_policy.classify(&candidate),
            Ok(RematerializationDecision::Save),
            "candidate-wide in-place saving takes precedence over offloading",
        );

        // Reusing one attached region in both condition slots reaches the same program-local value twice. The
        // resolver retains its first semantic occurrence and does not expose duplicate producers to policies.
        let mut shared_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let shared_region = shared_builder.import_region(condition_regions[0].entry_region_ref());
        let shared_predicate = shared_builder.add_input(ArrayType::scalar(DataType::Boolean));
        let shared_lhs = shared_builder.add_input(vector_type.clone());
        let shared_rhs = shared_builder.add_input(vector_type.clone());
        let shared_output = shared_builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![shared_region, shared_region],
                vec![shared_predicate, shared_lhs, shared_rhs],
            )
            .unwrap()[0];
        let shared_program = shared_builder
            .build::<Vec<Array>, Vec<Array>>(vec![shared_output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let shared_candidate = RematerializationCandidate::from_program_residual(
            &shared_program,
            shared_output,
            ArrayType::scalar(DataType::F64),
        )
        .unwrap()
        .unwrap();
        assert_eq!(shared_candidate.producers().len(), 1);

        // A constant-producing branch contributes no instruction producer, while the other branch's dot remains
        // visible. The boundary is classifiable whenever at least one possible path has an instruction producer.
        let constant_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            let output = builder.add_constant(Array::scalar(0.0));
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut mixed_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let dot_region = mixed_builder.import_region(condition_regions[0].entry_region_ref());
        let constant_region = mixed_builder.import_region(constant_branch.entry_region_ref());
        let mixed_predicate = mixed_builder.add_input(ArrayType::scalar(DataType::Boolean));
        let mixed_lhs = mixed_builder.add_input(vector_type.clone());
        let mixed_rhs = mixed_builder.add_input(vector_type);
        let mixed_output = mixed_builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![dot_region, constant_region],
                vec![mixed_predicate, mixed_lhs, mixed_rhs],
            )
            .unwrap()[0];
        let mixed_program = mixed_builder
            .build::<Vec<Array>, Vec<Array>>(vec![mixed_output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mixed_candidate = RematerializationCandidate::from_program_residual(
            &mixed_program,
            mixed_output,
            ArrayType::scalar(DataType::F64),
        )
        .unwrap()
        .unwrap();
        assert_eq!(mixed_candidate.producers().len(), 1);
        assert!(matches!(mixed_candidate.producers()[0].operation(), ArrayOperation::Dot(_)));
    }

    #[test]
    fn test_dots_saveable_saves_conditional_dot_residuals_for_both_predicates() {
        use crate::operations::{ConditionOperation, CosOperation, MulOperation, SinOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        /// Builds `x -> (x · x) * trig(x · x)` for one condition branch.
        fn branch(cosine: bool) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(vector_type(2));
            let dot = builder
                .add_instruction(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    vec![input, input],
                )
                .unwrap()[0];
            let trigonometric = if cosine {
                builder.add_instruction(CosOperation::new(), Vec::new(), vec![dot]).unwrap()[0]
            } else {
                builder.add_instruction(SinOperation::new(), Vec::new(), vec![dot]).unwrap()[0]
            };
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![dot, trigonometric]).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        /// Derives and executes a rematerialized constant-predicate condition under `policy`.
        fn check(policy: impl TestPolicy, predicate: bool) -> (usize, f64, Vec<f64>) {
            let true_branch = branch(false);
            let false_branch = branch(true);
            let body = move |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let context = input.dispatch_domain();
                let predicate =
                    context.lift(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![f64::from(predicate)]))?;
                let mut outputs = context.bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![true_branch.clone(), false_branch.clone()],
                    &[predicate, input],
                )?;
                Ok(outputs.remove(0))
            };
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy);
            let forward_output_count = staged_operation(&function, vector_type(2)).forward().output_types().len();
            let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
                .value_and_gradient(
                    |input| function.call(input).unwrap(),
                    Array::from_f64s(vector_type(2), vec![0.5, 1.5]),
                )
                .unwrap();
            (forward_output_count, value.to_f64s()[0], gradient.to_f64s())
        }

        let input = [0.5, 1.5];
        let dot = input.iter().map(|value| value * value).sum::<f64>();
        let true_derivative = dot.sin() + dot * dot.cos();
        let false_derivative = dot.cos() - dot * dot.sin();

        let (true_baseline, true_value, true_gradient) = check(NothingSaveable, true);
        let (true_saved, _, _) = check(DotsSaveable, true);
        assert_eq!(true_saved, true_baseline + 1, "the condition residual is one outer boundary value");
        assert_abs_diff_eq!(true_value, dot * dot.sin(), epsilon = 1e-9);
        for (index, value) in input.iter().enumerate() {
            assert_abs_diff_eq!(true_gradient[index], true_derivative * 2.0 * value, epsilon = 1e-9);
        }

        let (false_baseline, false_value, false_gradient) = check(NothingSaveable, false);
        let (false_saved, _, _) = check(DotsSaveable, false);
        assert_eq!(false_saved, false_baseline + 1, "the condition residual is one outer boundary value");
        assert_abs_diff_eq!(false_value, dot * dot.cos(), epsilon = 1e-9);
        for (index, value) in input.iter().enumerate() {
            assert_abs_diff_eq!(false_gradient[index], false_derivative * 2.0 * value, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rematerialization_policies_control_the_saved_residuals() {
        // `dot_sine_body` has one output and one input, and three instruction-produced residuals: the dot output
        // `u`, the sine output `sin(u)`, and the sine rule's `cos(u)` factor. The forward program therefore outputs
        // 2 values under `NothingSaveable` (output + input), 3 under `DotsSaveable` (+`u`), and 5 under
        // `EverythingSaveable`; and the backward program shrinks as more residuals are saved instead of recomputed.
        fn counts(policy: impl TestPolicy) -> (usize, usize) {
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body).with_policy(policy);
            let operation = staged_operation(&function, vector_type(2));
            (operation.forward().output_types().len(), operation.backward().instructions().len())
        }
        let cases = [counts(NothingSaveable), counts(DotsSaveable), counts(EverythingSaveable)];
        let forward_output_counts = cases.iter().map(|(forward, _)| *forward).collect::<Vec<_>>();
        let backward_instruction_counts = cases.iter().map(|(_, backward)| *backward).collect::<Vec<_>>();
        assert_eq!(forward_output_counts, vec![2, 3, 5]);
        // Saving everything prunes the whole recomputation graph from the backward program. Saving only the dot
        // output does not shrink it here because the unsaved `sin(u)` and `cos(u)` residuals still recompute from
        // `u`, keeping the dot instruction reachable; the saved value only short-circuits the factor use itself.
        assert!(
            backward_instruction_counts[0] >= backward_instruction_counts[1]
                && backward_instruction_counts[1] > backward_instruction_counts[2],
            "saving more residuals should never grow the backward program and saving everything should shrink it, \
             but instruction counts were {backward_instruction_counts:?}",
        );
    }

    #[test]
    fn test_tag_is_transparent_to_differentiation() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (primal, tangent) =
            domain.jvp(|x| Ok((x.clone() * x).tag("square")), Array::scalar(2.0), Array::scalar(1.0)).unwrap();
        assert_eq!(primal, Array::scalar(4.0));
        assert_eq!(tangent, Array::scalar(4.0));
        let (value, gradient) =
            domain.value_and_gradient(|x| (x.clone() * x).tag("square"), Array::scalar(3.0)).unwrap();
        assert_eq!(value, Array::scalar(9.0));
        assert_eq!(gradient, Array::scalar(6.0));
    }

    #[test]
    fn test_name_based_rematerialization_policies_classify_tagged_residuals() {
        // `f(x) = u * sin(u)` with `u = (x · x).tag("u")`: the tagged dot output is one of the three
        // instruction-produced residuals (`u`, `sin(u)`, and the sine rule's `cos(u)` factor), so name-based
        // policies can select it (or its complement) by tag.
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
            Ok(u.clone() * u.sin()?)
        }
        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        // Forward output counts: 2 base outputs (output + input), plus the residuals each policy saves.
        fn check(policy: impl TestPolicy, expected_forward_outputs: usize, input: &Array, expected: &[f64]) {
            let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy.clone());
            let operation = staged_operation(&function, vector_type(2));
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Every policy preserves the gradient; only the save/recompute split changes.
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            for (index, expected) in expected.iter().enumerate() {
                assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
            }
        }
        check(SaveOnlyTheseNames::new(["u"]), 3, &input, &expected_gradient);
        check(SaveOnlyTheseNames::new(["other"]), 2, &input, &expected_gradient);
        check(SaveAnyNamesButThese::new(["u"]), 2, &input, &expected_gradient);
        check(SaveAnyNamesButThese::new(["other"]), 3, &input, &expected_gradient);
        check(SaveAnythingExceptTheseNames::new(["u"]), 4, &input, &expected_gradient);
        check(SaveAnythingExceptTheseNames::new(["other"]), 5, &input, &expected_gradient);
    }

    #[test]
    fn test_jvp_through_rematerialization_uses_the_derived_tangent_program() {
        // Unlike user-authored custom VJPs (which reject forward mode, matching JAX), rematerialized calls carry a
        // derived tangent program, so `jvp` works through them — matching `jax.checkpoint`.
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body);
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x| function.call(x),
                Array::from_f64s(vector_type(2), vec![0.5, 1.5]),
                Array::from_f64s(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        // f(x) = u * sin(u) with u = x · x; the tangent against seed e_0 is the first gradient component.
        let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
        assert_abs_diff_eq!(primal.to_f64s()[0], u * u.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], dot_sine_gradient(&[0.5, 1.5])[0], epsilon = 1e-9);
    }

    #[test]
    fn test_rematerialization_preserves_custom_vjp_semantics_and_keeps_the_boundary_opaque() {
        use crate::tracing_v2::custom_derivatives::custom_vjp;

        // The custom backward rule triples the true gradient (expressed through addition to avoid constant lifting),
        // so a matching gradient proves the user-authored rule — not the true derivative — governs reverse mode
        // through the rematerialized region.
        let custom = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            move |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| custom.call(x),
        )
        .with_policy(EverythingSaveable);
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The custom-VJP boundary stays opaque to the policy: the rematerialized primal program preserves the
        // custom_vjp call intact, and even `EverythingSaveable` saves only the residual the user's forward rule
        // declares (`cos(x)`) — never values from inside the user-owned backward program — so the forward program
        // outputs exactly the body output, the region input, and that one residual.
        let scalar_type = ArrayType::new(DataType::F64, Shape::new(Vec::new()));
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), scalar_type).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let primal = program.region_ref(instruction.regions()[0]).unwrap();
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        assert!(
            primal
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::CustomVjp(_))),
            "the rematerialized primal program should preserve the custom_vjp call",
        );
        assert_eq!(forward.output_types().len(), 3);
    }

    #[test]
    fn test_prevent_cse_is_carried_on_the_staged_rematerialize_operation() {
        // `prevent_cse` defaults to enabled (JAX parity) and is carried on the staged operation as a backend
        // lowering hint; user-authored custom VJPs (constructed directly) leave it disabled.
        for (function, expected) in [
            (rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body), true),
            (
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body)
                    .with_prevent_cse(false),
                false,
            ),
        ] {
            let (_, program) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(operation.prevent_cse(), expected);
        }
    }

    #[test]
    fn test_rematerialize_remains_opaque_to_partial_evaluation() {
        let input_type = vector_type(2);
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body);
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), input_type.clone()).unwrap();
        let program = program.to_flat_program();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(input_type)]).unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = evaluation.program.instructions()[0].operation() else {
            panic!("partial evaluation should preserve the rematerialize boundary");
        };
        assert!(operation.prevent_cse());
    }

    /// Under a *staging* known-side context, a mixed rematerialized call stays fully opaque: nothing folds across
    /// the boundary (no intermediate crosses it as a residual edge), nothing is staged into the live outer trace,
    /// and the whole call residualizes with the symbolic known input as a residual-input feeder. This is precisely
    /// the memory profile rematerialization asks for — the residual side recomputes from the saved *inputs* instead
    /// of storing intermediates — so the conservative default rule is also the semantically correct online behavior;
    /// finer-grained save-versus-recompute choices belong to the policy-driven structural split (see
    /// `.tasks/plan_partition_policies.md`).
    #[test]
    fn test_rematerialize_remains_opaque_to_partial_evaluation_under_staging() {
        use crate::partial::PartialEvaluationInput;
        use crate::tracing::TracingContext;

        let input_type = vector_type(2);
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |(a, x): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| { Ok((a * x.clone()).sin()?.dot(&x, &DotDimensionNumbers::inner_product())) },
        );
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs| function.call(inputs),
            (input_type.clone(), input_type.clone()),
        )
        .unwrap();
        let program = program.to_flat_program();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let known = outer.input(input_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(input_type)])
            .unwrap();

        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = evaluation.program.instructions()[0].operation() else {
            panic!("staged partial evaluation should preserve the rematerialize boundary");
        };
        assert!(operation.prevent_cse());
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
    }

    #[test]
    fn test_nested_rematerialization_matches_the_unrematerialized_gradient() {
        // The analogue of JAX's sqrt-schedule idiom: rematerialized regions nest, with each level storing only its
        // own region inputs. Differentiating the outer call replays the inner call's backward program inside the
        // outer backward derivation, which interprets the inner (transposed) rematerialize call over tracers.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let inner = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        // f(x) = Σᵢ sin(xᵢ²) xᵢ, so ∂f/∂xⱼ = sin(xⱼ²) + 2 xⱼ² cos(xⱼ²).
        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_gradient = [0.5f64, 1.5].map(|x| (x * x).sin() + 2.0 * x * x * (x * x).cos());
        let (value, gradient) = domain.value_and_gradient(|x| outer.call(x).unwrap(), input).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], expected_value, epsilon = 1e-9);
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nested_rematerialization_preserves_the_nested_call_structure_and_residual_accounting() {
        let inner = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| outer.call(x), vector_type(2)).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let primal = program.region_ref(instruction.regions()[0]).unwrap();
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        // The outer primal program preserves the inner rematerialized call instead of inlining its body.
        assert!(
            primal
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::Rematerialize(_))),
            "the outer primal program should contain the inner rematerialized call",
        );
        // `NothingSaveable` everywhere: the outer forward program outputs only the body output plus the region
        // input, storing no interior residuals — in particular nothing produced inside the inner region.
        assert_eq!(forward.output_types().len(), 2);
    }

    #[test]
    fn test_jvp_through_nested_rematerialization_uses_the_derived_tangent_programs() {
        // Forward mode through nested rematerialized calls exercises the un-transposed rematerialize call replay over
        // tracers: deriving the outer tangent program interprets the inner call's tangent program inside the outer
        // tangent trace.
        let inner = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x| outer.call(x),
                Array::from_f64s(vector_type(2), vec![0.5, 1.5]),
                Array::from_f64s(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_tangent = {
            let x = 0.5f64;
            (x * x).sin() + 2.0 * x * x * (x * x).cos()
        };
        assert_abs_diff_eq!(primal.to_f64s()[0], expected_value, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], expected_tangent, epsilon = 1e-9);
    }

    #[test]
    fn test_nested_rank_zero_rematerialization_matches_the_unrematerialized_gradient() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let inner = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let y = inner.call(x.clone())?;
                Ok(y * x)
            },
        );
        // f(x) = sin(x²) x, so f'(x) = sin(x²) + 2 x² cos(x²).
        let (value, gradient) = domain.value_and_gradient(|x| outer.call(x).unwrap(), Array::scalar(0.7)).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.49f64.sin() * 0.7, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 0.49f64.sin() + 2.0 * 0.49 * 0.49f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_rematerialization_survives_batching_with_preserved_residual_structure() {
        use crate::batching::Batch;

        // Batching a rematerialized call preserves it around batched programs instead of inlining the primal, so
        // the memory-saving structure survives `vmap`: the staged program holds exactly one rematerialize call whose
        // batched forward program still stores only the body output and the region input plus the policy-saved
        // residuals — each now carrying the batch axis.
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        fn check(policy: impl TestPolicy, expected_forward_outputs: usize, matrix_type: &ArrayType) {
            let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body)
                .with_policy(policy.clone());
            let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
                |x| {
                    let context = x.context().clone();
                    Batch::batch(&context, |item| function.call(item), x, BatchAxis::new(0), BatchAxis::new(0), None)
                        .map_err(ProgramError::from)
                },
                matrix_type.clone(),
            )
            .unwrap();
            assert_eq!(program.instructions().len(), 1, "unexpected batched program shape for policy {policy:?}");
            let instruction = &program.instructions()[0];
            assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
            let forward = program.region_ref(instruction.regions()[1]).unwrap();
            let forward_output_types = forward.output_types();
            assert_eq!(
                forward_output_types.len(),
                expected_forward_outputs,
                "unexpected batched forward output count for policy {policy:?}",
            );
            // Every batched forward output carries the batch axis at position 0.
            for output_type in &forward_output_types {
                assert_eq!(
                    output_type.shape().dimensions().first().cloned(),
                    Some(Dimension::Static(2)),
                    "batched forward outputs should carry the batch axis for policy {policy:?}",
                );
            }
        }
        check(NothingSaveable, 2, &matrix_type);
        check(DotsSaveable, 3, &matrix_type);
        check(EverythingSaveable, 5, &matrix_type);
    }

    #[test]
    fn test_rematerialization_batching_preserves_nonzero_natural_axes_across_all_regions() {
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| x.sin(),
        );
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(3)).unwrap();
        let program = program.to_flat_program();

        // Each logical item is a length-3 vector, packed column-wise with the mapped dimension at axis 1. The primal,
        // forward tail, tangent, and backward computations are all elementwise, so none needs to move that axis.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        assert_eq!(batched.instructions().len(), 1);
        let instruction = &batched.instructions()[0];
        let ArrayOperation::Rematerialize(operation) = instruction.operation() else {
            panic!("batching must preserve the rematerialization wrapper");
        };
        assert!(operation.prevent_cse());
        for region in instruction.regions() {
            let region = batched.region_ref(*region).unwrap();
            assert!(
                region
                    .input_types()
                    .iter()
                    .chain(region.output_types().iter())
                    .all(|r#type| r#type.rank() == 2 && r#type.shape().dimensions()[1] == Dimension::Static(2))
            );
        }

        let input = Array::matrix(3, 2, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let expected = Array::matrix(
            3,
            2,
            vec![0.0f64.sin(), 0.5f64.sin(), 1.0f64.sin(), 1.5f64.sin(), 2.0f64.sin(), 2.5f64.sin()],
        );
        assert_eq!(batched.interpret(vec![input]).unwrap(), vec![expected]);
    }

    #[test]
    fn test_rematerialized_gradients_are_correct_through_batching() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::operations::{Reduce, ReductionKind};

        // `grad(vmap(rematerialize(f)))`: the gradient flows through the preserved batched call's derived backward
        // program and matches the analytic per-item gradients.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let (value, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |item| function.call(item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::from_f64s(vector_type(2), vec![0.5, 1.0]),
            )
            .unwrap();
        // f(x) = Σᵢ sin(xᵢ²), so ∂f/∂xⱼ = 2 xⱼ cos(xⱼ²).
        assert_abs_diff_eq!(value.to_f64s()[0], 0.25f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 0.5 * 0.25f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[1], 2.0 * 1.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_call_caches_derivations_per_input_types() {
        use std::cell::Cell;

        // The body closure runs only while deriving (the primal trace; the remaining passes replay the traced
        // program), so the closure invocation count equals the number of derivations.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let trace_count = Rc::new(Cell::new(0));
        let counter = trace_count.clone();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            move |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                counter.set(counter.get() + 1);
                Ok((x.clone() * x).sin()?)
            },
        );

        // Two calls with equal input types derive once, both within one trace and across separate traces.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let first = function.call(x.clone())?;
                let second = function.call(x)?;
                Ok(first + second)
            },
            vector_type(2),
        )
        .unwrap();
        assert_eq!(trace_count.get(), 1);
        let rematerialize_count = program
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Rematerialize(_)))
            .count();
        assert_eq!(rematerialize_count, 2, "both calls should stage their own rematerialize instruction");
        EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        assert_eq!(trace_count.get(), 1);

        // A different input type re-derives.
        EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(3)).unwrap();
        assert_eq!(trace_count.get(), 2);

        // Cache hits still differentiate correctly: the second gradient call reuses the derivation staged by the
        // first one.
        let (_, first_gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(0.7)).unwrap();
        let derivations_after_first_gradient = trace_count.get();
        let (_, second_gradient) =
            domain.value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(0.7)).unwrap();
        assert_eq!(trace_count.get(), derivations_after_first_gradient);
        assert_abs_diff_eq!(first_gradient.to_f64s()[0], 2.0 * 0.7 * 0.49f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(second_gradient.to_f64s()[0], first_gradient.to_f64s()[0], epsilon = 1e-9);
    }

    #[test]
    fn test_dots_with_no_batch_dims_saveable_skips_batched_contractions() {
        // The body stages two dots: a batched per-row inner product `u = dot(x, x; batch=[0])` and an unbatched
        // inner product `v = u · u`. `DotsSaveable` saves both dot residuals while
        // `DotsWithNoBatchDimsSaveable` saves only the unbatched one, so the forward output counts differ by one
        // (2 base outputs = body output + region input).
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let batched = DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]);
            let u = x.dot(&x, &batched);
            let v = u.dot(&u, &DotDimensionNumbers::inner_product());
            Ok(v.clone() * v.sin()?)
        }
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        fn check(policy: impl TestPolicy, expected_forward_outputs: usize, matrix_type: &ArrayType) {
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy.clone());
            let operation = staged_operation(&function, matrix_type.clone());
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
        check(DotsSaveable, 4, &matrix_type);
        check(DotsWithNoBatchDimsSaveable, 3, &matrix_type);
    }

    #[test]
    fn test_save_from_both_policies_saves_the_union_of_both_policies() {
        // The body produces one dot residual `u`, one named residual `s`, and one unnamed `cos` residual; the
        // combinator saves the union of what its two members save.
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product());
            let s = u.sin()?.tag("s");
            Ok(u * s)
        }
        fn check(policy: impl TestPolicy, expected_forward_outputs: usize) {
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy.clone());
            let operation = staged_operation(&function, vector_type(2));
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
        check(DotsSaveable, 3);
        check(SaveOnlyTheseNames::new(["s"]), 3);
        check(SaveFromBothPolicies::new(DotsSaveable, SaveOnlyTheseNames::new(["s"])), 4);
    }

    #[test]
    fn test_custom_policies_classify_residuals_through_candidates() {
        // Custom policies see each residual's classification facts. The first policy reproduces the
        // `SaveFromBothPolicies` union from the test above through candidate queries; the second selects by
        // operation name; and both observe the residuals' staged types.
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product());
            let s = u.sin()?.tag("s");
            Ok(u * s)
        }
        // Custom policies see the complete producing operation and match their operation family directly, without
        // any centrally selected fact list.
        let dot_or_named =
            PolicyFn::new(|candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
                assert!(candidate.residual_type().shape().dimensions().is_empty());
                let saves = candidate.producers().iter().any(|producer| match producer.operation() {
                    ArrayOperation::Dot(_) => true,
                    ArrayOperation::Tag(operation) => operation.key() == "s",
                    _ => false,
                });
                Ok::<_, RematerializationRejection>(match saves {
                    true => RematerializationDecision::<NoStorage>::Save,
                    false => RematerializationDecision::Recompute,
                })
            });
        let cosines_only =
            PolicyFn::new(|candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
                Ok::<_, RematerializationRejection>(
                    match candidate.producers().iter().any(|producer| producer.operation().name() == "cos") {
                        true => RematerializationDecision::<NoStorage>::Save,
                        false => RematerializationDecision::Recompute,
                    },
                )
            });
        let expected_gradient = {
            // f(x) = u sin(u) with u = x · x, so ∇f(x) = (sin(u) + u cos(u)) · 2x.
            let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
            [0.5f64, 1.5].map(|value| (u.sin() + u * u.cos()) * 2.0 * value)
        };
        fn check(policy: impl TestPolicy, expected_forward_outputs: usize, expected_gradient: &[f64]) {
            let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
            let function =
                rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy.clone());
            let operation = staged_operation(&function, vector_type(2));
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Custom policies only change the save/recompute split, never the gradient.
            let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
            }
        }
        check(dot_or_named, 4, &expected_gradient);
        check(cosines_only, 3, &expected_gradient);
    }

    #[test]
    fn test_second_order_reverse_through_rematerialization_matches_the_analytic_second_derivative() {
        use crate::differentiation::HessianDifferentiate;

        // Second-order differentiation through a rematerialized call: the inner reverse pass replays the derived
        // backward program over tracers (inlining it into the gradient program), and the outer pass differentiates
        // the result. f(x) = sin(x²), so f''(x) = 2 cos(x²) - 4x² sin(x²).
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let hessian = domain.hessian(|x| function.call(x), Array::scalar(0.7)).unwrap();
        let block = hessian.iter_blocks().next().unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(
            block.value().to_f64s()[0],
            2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_rank_zero_second_order_through_rematerialization_matches_the_analytic_second_derivative() {
        // This composes through nested reverse transforms: the outer reverse pass differentiates a closure that takes
        // the rematerialized gradient on its nested tracing context.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let (gradient, second_derivative) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    context.gradient(|y| function.call(y).unwrap(), x).unwrap()
                },
                Array::scalar(0.7),
            )
            .unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            second_derivative.to_f64s()[0],
            2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(),
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_rematerialized_pullback_recovers_the_tangent_map() {
        // The pullback of a rematerialized call carries a derived tangent program. For a scalar function the input
        // cotangent at a unit output cotangent equals the tangent-map coefficient `f'(x)`, so seeding the
        // direct-transpose pullback at `[1.0 ++ residuals]` recovers the tangent map. f(x) = sin(x²), so the recovered
        // value is f'(0.7) = 2·0.7·cos(0.7²).
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let (_, pullback) = domain.vjp(|x| function.call(x), Array::scalar(0.7)).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let output = pullback.interpret(pullback_inputs).unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(output[0].to_f64s()[0], 2.0 * x * (x * x).cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_jacobian_reverse_through_rematerialization_uses_the_rematerializing_backward_program() {
        use crate::differentiation::jacobian_reverse;

        // The Jacobian of elementwise `sin(x * x)` is the diagonal matrix `diag(cos(x²) * 2x)`;
        // `jacobian_reverse` exercises the batched replay of the derived backward program.
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.clone() * x).sin()?),
        );
        let jacobian =
            jacobian_reverse(|x| function.call(x), Array::from_f64s(vector_type(2), vec![0.5, 1.0])).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_abs_diff_eq!(block.value().to_f64s()[0], 0.25f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[3], 1.0f64.cos() * 2.0, epsilon = 1e-9);
    }

    /// Canonical offload destination used by the offloading policy tests.
    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    /// [`dot_sine`] with the dot output tagged `"u"`, so name-based offloading policies can select it.
    fn tagged_dot_sine_body(
        x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
        Ok(u.clone() * u.sin()?)
    }

    /// Returns whether `program` stages any memory transfers.
    fn contains_memory_transfers(
        program: &crate::programs::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> bool {
        program
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().name() == "transfer_to_memory")
    }

    #[test]
    fn test_offloading_policies_park_saved_residuals_in_the_destination_memory() {
        // The tagged body has three instruction-produced residuals (`u`, `sin(u)`, and the sine rule's `cos(u)`
        // factor), so saving or offloading `u` always yields three forward outputs (body output + region input +
        // `u`). Offloaded residuals are emitted behind a staged transfer — the saved forward output carries the
        // destination memory, and the backward and tangent programs transfer it back before consuming it — while
        // residuals saved in place stay in their own memory with no transfers anywhere.
        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let offload_u = SaveAndOffloadOnlyTheseNames::new([] as [&str; 0], ["u"], PINNED_HOST);
        let save_u = SaveAndOffloadOnlyTheseNames::new(["u"], [] as [&str; 0], PINNED_HOST);
        // The first non-`Recompute` placement wins, so a save-side hit shields `u` from the offload side, while a
        // recompute-only first side defers to the offload side.
        let save_beats_offload = SaveFromBothPolicies::new(save_u.clone(), offload_u.clone());
        let offload_after_recompute = SaveFromBothPolicies::new(NothingSaveable, offload_u.clone());
        fn check(policy: impl TestPolicy, expected_memory: Memory, input: &Array, expected_gradient: &[f64]) {
            let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
            let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(tagged_dot_sine_body)
                .with_policy(policy.clone());
            let operation = staged_operation(&function, vector_type(2));
            let forward_output_types = operation.forward().output_types();
            assert_eq!(forward_output_types.len(), 3, "unexpected forward output count for policy {policy:?}");
            assert_eq!(forward_output_types[0].memory(), Memory::Device);
            assert_eq!(forward_output_types[1].memory(), Memory::Device);
            assert_eq!(
                forward_output_types[2].memory(),
                expected_memory,
                "unexpected saved-residual memory for policy {policy:?}",
            );
            // Transfers appear exactly when the policy offloads: once in the forward program (to the destination)
            // and once in each of the backward and tangent programs (back to the source).
            let expects_transfers = expected_memory != Memory::Device;
            assert_eq!(
                contains_memory_transfers(operation.forward()),
                expects_transfers,
                "unexpected forward transfers for policy {policy:?}",
            );
            assert_eq!(
                contains_memory_transfers(operation.backward()),
                expects_transfers,
                "unexpected backward transfers for policy {policy:?}",
            );
            assert_eq!(
                contains_memory_transfers(operation.tangent()),
                expects_transfers,
                "unexpected tangent transfers for policy {policy:?}",
            );
            // Offloading changes placement, never values: gradients match the direct computation.
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
            }
        }
        check(offload_u, PINNED_HOST, &input, &expected_gradient);
        check(save_u, Memory::Device, &input, &expected_gradient);
        check(save_beats_offload, Memory::Device, &input, &expected_gradient);
        check(offload_after_recompute, PINNED_HOST, &input, &expected_gradient);
    }

    #[test]
    fn test_offload_dots_with_no_batch_dims_offloads_unbatched_contractions() {
        // `dot_sine_body`'s only dot residual is the unbatched inner product `u`, so the policy offloads exactly
        // that residual — `DotsSaveable`'s split, with the saved value parked in pinned host memory.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body)
            .with_policy(OffloadDotsWithNoBatchDims::new(PINNED_HOST));
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        let forward_output_types = forward.output_types();
        assert_eq!(forward_output_types.len(), 3);
        assert_eq!(forward_output_types[2].memory(), PINNED_HOST);

        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
        let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
        assert_abs_diff_eq!(value.to_f64s()[0], u * u.sin(), epsilon = 1e-9);
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_custom_offloading_policies_mix_all_three_verdicts() {
        // `u` is saved in place, `v = sin(u)` is offloaded, and the sine rule's `cos(u)` factor is recomputed, so
        // the forward program emits four outputs whose final two are the device-resident `u` and the host-parked
        // `v`.
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
            let v = u.sin()?.tag("v");
            Ok(u * v)
        }
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let policy = PolicyFn::new(|candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
            let key = match candidate.producers()[0].operation() {
                ArrayOperation::Tag(operation) => Some(operation.key()),
                _ => None,
            };
            Ok::<_, RematerializationRejection>(match key {
                Some("u") => RematerializationDecision::Save,
                Some("v") => RematerializationDecision::SaveWith(MemoryTransferStorage::new(PINNED_HOST)),
                _ => RematerializationDecision::Recompute,
            })
        });
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body).with_policy(policy);
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        let forward_output_types = forward.output_types();
        assert_eq!(forward_output_types.len(), 4);
        // The two saved residuals are `u` (saved in place, device-resident) and `v` (offloaded to pinned host);
        // their relative order follows the linearization's residual enumeration, which the test does not pin.
        let saved_memories =
            forward_output_types[2..].iter().map(|output_type| output_type.memory()).collect::<Vec<_>>();
        assert_eq!(saved_memories.len(), 2);
        assert!(saved_memories.contains(&Memory::Device), "expected a device-resident saved residual");
        assert!(saved_memories.contains(&PINNED_HOST), "expected a host-parked saved residual");

        // f(x) = u sin(u) with u = x · x, so the gradient matches `dot_sine`'s.
        let input = Array::from_f64s(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.to_f64s()[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_offloaded_rematerialization_survives_batching_with_host_parked_saved_types() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::operations::{Reduce, ReductionKind};

        // `vmap` preserves the rematerialized call around batched programs, and the offloaded saved residual keeps
        // its host placement with the batch axis prepended to its shape.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(tagged_dot_sine_body)
            .with_policy(SaveAndOffloadOnlyTheseNames::new([] as [&str; 0], ["u"], PINNED_HOST));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let context = x.context().clone();
                Batch::batch(&context, |item| function.call(item), x, BatchAxis::new(0), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            matrix_type.clone(),
        )
        .unwrap();
        assert_eq!(program.instructions().len(), 1);
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        let forward_output_types = forward.output_types();
        assert_eq!(forward_output_types.len(), 3);
        let saved_type = &forward_output_types[2];
        assert_eq!(saved_type.shape().dimensions().first().cloned(), Some(Dimension::Static(2)));
        assert_eq!(saved_type.memory(), PINNED_HOST);

        // `grad(vmap(...))` through the offloaded call matches the analytic per-item gradients.
        let rows = [[0.5, 1.5, 1.0], [0.25, 0.75, 1.25]];
        let (_, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |item| function.call(item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::from_f64s(matrix_type, rows.as_flattened().to_vec()),
            )
            .unwrap();
        for (row, values) in rows.iter().enumerate() {
            let expected_row_gradient = dot_sine_gradient(values);
            for (column, expected) in expected_row_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.to_f64s()[row * 3 + column], *expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_policy_rejections_short_circuit_and_carry_the_enriched_candidate_facts() {
        // Any evaluated rejection aborts the whole transformation, and the engine enriches the policy-supplied
        // rejection with the producing operation's name and the rendered logical residual type.
        let policy =
            PolicyFn::new(
                |candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| match candidate
                    .producers()
                    .iter()
                    .any(|producer| matches!(producer.operation(), ArrayOperation::Cos(_)))
                {
                    true => Err(RematerializationRejection::other("cosines are not allowed here")),
                    false => Ok(RematerializationDecision::<NoStorage>::Recompute),
                },
            );
        let function =
            rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body).with_policy(policy);
        let error =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap_err();
        let rematerialization = error.downcast_custom::<RematerializationError>().cloned();
        assert_eq!(
            rematerialization,
            Some(RematerializationError::Rejected {
                operation_names: vec!["cos".to_string()],
                residual_type: "f64[]".to_string(),
                rejection: RematerializationRejection::other("cosines are not allowed here"),
            }),
        );
    }

    #[test]
    fn test_candidate_output_indices_are_producer_local_through_scan_provenance() {
        use crate::operations::ScanOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Loop body `[c, x] -> [c * (x · x), x · x]`: the body's *second* output (the per-iteration dot) is defined
        // by the dot instruction's *first* (and only) output. Classifying the scan's stacked output at scan output
        // index 1 must therefore report the dot operation with the producer-local `output_index` 0 — not the outer
        // scan output index — while `residual_type` stays the outer stacked type.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let row_type = vector_type(2);
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let carry = builder.add_input(scalar_type.clone());
            let row = builder.add_input(row_type.clone());
            let dot = builder
                .add_instruction(
                    crate::operations::math::DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    vec![row, row],
                )
                .unwrap()[0];
            let next = builder
                .add_instruction(crate::operations::math::MulOperation::new(), Vec::new(), vec![carry, dot])
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next, dot], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let stacked_rows_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let carry = builder.add_input(scalar_type);
        let rows = builder.add_input(stacked_rows_type);
        let scan = ScanOperation::new(1, 3);
        let body_region = builder.import_region(body.entry_region_ref());
        let scan_outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![carry, rows])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(scan_outputs.clone(), vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let candidate =
            RematerializationCandidate::from_program_residual(&program, scan_outputs[1], stacked_type.clone())
                .unwrap()
                .unwrap();
        assert_eq!(candidate.producers().len(), 1);
        let producer = &candidate.producers()[0];
        assert!(matches!(producer.operation(), ArrayOperation::Dot(_)));
        assert_eq!(producer.output_index(), 0, "the output index must be local to the nested dot instruction");
        assert_eq!(candidate.residual_type(), &stacked_type, "the residual type must stay the outer stacked type");
        assert_eq!(producer.output_types()[producer.output_index()], ArrayType::scalar(DataType::F64));
    }

    #[test]
    fn test_recompute_slices_copy_multi_output_producers_once_with_saved_cuts_winning() {
        use std::fmt::Display;

        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        /// Test-only pure operation producing two outputs from one input, so one producer can carry one saved and
        /// one recomputed residual at the same time.
        #[derive(Clone, Debug)]
        struct SplitOperation;

        impl Display for SplitOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for SplitOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "split"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone(), input_types[0].clone()])
            }
        }

        // Primal program `x -> (a, b) = split(x)` with both outputs as residuals.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, SplitOperation>::new();
        let input = builder.add_input(scalar_type.clone());
        let split_outputs = builder.add_instruction(SplitOperation, Vec::new(), vec![input]).unwrap().to_vec();
        let primal = builder
            .build::<Vec<Array>, Vec<Array>>(split_outputs.clone(), vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        // Destination: `a` is a saved cut behind a fresh input; `b` resolves through a recompute slice. The slice
        // copies the split instruction exactly once (producing a replayed sibling for `a`), and resolving `a`
        // afterwards must keep returning the saved cut, never the replayed sibling.
        let mut destination = ProgramBuilder::<Array, SplitOperation>::new();
        let region_input = destination.add_input(scalar_type.clone());
        let saved_input = destination.add_input(scalar_type);
        let mut resolver = PrimalSliceResolver::new(&primal, std::slice::from_ref(&region_input));
        resolver.seed_cut(split_outputs[0], saved_input);

        let recomputed_b = resolver.resolve(split_outputs[1], &mut destination).unwrap();
        let resolved_a = resolver.resolve(split_outputs[0], &mut destination).unwrap();
        assert_eq!(resolved_a, saved_input, "saved cuts must win over the replayed sibling");
        assert_ne!(recomputed_b, saved_input);
        assert_eq!(
            destination.instructions().len(),
            1,
            "the multi-output producer must be copied exactly once for both of its outputs",
        );
        // Resolving `b` again reuses the replayed output instead of copying the producer a second time.
        assert_eq!(resolver.resolve(split_outputs[1], &mut destination).unwrap(), recomputed_b);
        assert_eq!(destination.instructions().len(), 1);
    }

    #[test]
    fn test_invalid_output_region_provenance_is_rejected() {
        use std::fmt::Display;

        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        /// Test-only operation that returns caller-selected output-region provenance.
        #[derive(Copy, Clone, Debug)]
        struct InvalidOriginOperation {
            provenance: crate::programs::regions::OutputRegionProvenance,
        }

        impl Display for InvalidOriginOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for InvalidOriginOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "invalid_origin"
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                const { &[RegionSlot::computation("body")] }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone()])
            }

            fn output_region_provenance(
                &self,
                _output_index: usize,
            ) -> Vec<crate::programs::regions::OutputRegionProvenance> {
                vec![self.provenance]
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body_builder = ProgramBuilder::<Array, InvalidOriginOperation>::new();
        let body_input = body_builder.add_input(scalar_type.clone());
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![body_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, InvalidOriginOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                InvalidOriginOperation {
                    provenance: crate::programs::regions::OutputRegionProvenance { region_index: 0, output_index: 1 },
                },
                vec![body_region],
                vec![input],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(matches!(
            RematerializationCandidate::from_program_residual(&program, output, scalar_type.clone()),
            Err(RematerializationError::UnsupportedProvenance { message })
                if message.contains("selecting output 1"),
        ));

        let mut builder = ProgramBuilder::<Array, InvalidOriginOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                InvalidOriginOperation {
                    provenance: crate::programs::regions::OutputRegionProvenance { region_index: 1, output_index: 0 },
                },
                vec![body_region],
                vec![input],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert!(matches!(
            RematerializationCandidate::from_program_residual(&program, output, scalar_type),
            Err(RematerializationError::UnsupportedProvenance { message })
                if message.contains("region index 1"),
        ));
    }

    #[test]
    fn test_origin_landing_on_a_region_input_skips_classification() {
        use std::fmt::Display;

        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        /// Test-only operation whose provenance forwards its attached region's pass-through output, so the walk
        /// lands on a region input rather than a producing instruction.
        #[derive(Clone, Debug)]
        struct PassThroughOriginOperation;

        impl Display for PassThroughOriginOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for PassThroughOriginOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "pass_through_origin"
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                const { &[RegionSlot::computation("body")] }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone()])
            }

            fn output_region_provenance(
                &self,
                output_index: usize,
            ) -> Vec<crate::programs::regions::OutputRegionProvenance> {
                vec![crate::programs::regions::OutputRegionProvenance { region_index: 0, output_index }]
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body_builder = ProgramBuilder::<Array, PassThroughOriginOperation>::new();
        let body_input = body_builder.add_input(scalar_type.clone());
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![body_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, PassThroughOriginOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let input = builder.add_input(scalar_type.clone());
        let output = builder.add_instruction(PassThroughOriginOperation, vec![body_region], vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(
            RematerializationCandidate::from_program_residual(&program, output, scalar_type).unwrap().is_none(),
            "an origin landing on a region input must not produce a policy candidate",
        );
    }

    #[test]
    fn test_invalid_storage_operations_return_structured_errors() {
        use std::fmt::Display;

        use crate::programs::{Effect, Effects};

        /// Malformed storage operation shape exercised by this test.
        #[derive(Copy, Clone, Debug)]
        enum InvalidStorageTestOperation {
            /// Pure operation producing no outputs.
            ZeroOutputs,

            /// Pure operation producing two outputs.
            TwoOutputs,

            /// Operation rejecting the engine-provided single input.
            RejectsInput,

            /// Single-result operation declaring an observable effect.
            Effectful,
        }

        impl Display for InvalidStorageTestOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for InvalidStorageTestOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                match self {
                    Self::ZeroOutputs => "zero_outputs",
                    Self::TwoOutputs => "two_outputs",
                    Self::RejectsInput => "rejects_input",
                    Self::Effectful => "effectful",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                if matches!(self, Self::RejectsInput) {
                    return Err(TypeError::invalid("unsupported storage input".to_string()));
                }
                check_count!("input", input_types, 1, TypeError);
                Ok(match self {
                    Self::ZeroOutputs => vec![],
                    Self::TwoOutputs => vec![input_types[0].clone(), input_types[0].clone()],
                    Self::RejectsInput => unreachable!(),
                    Self::Effectful => vec![input_types[0].clone()],
                })
            }

            fn effects(&self) -> Effects {
                match self {
                    Self::Effectful => Effects::single(Effect::OrderedIo),
                    _ => Effects::PURE,
                }
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let error = |operation| validate_storage_operation(&operation, &scalar_type).unwrap_err();
        assert!(matches!(
            error(InvalidStorageTestOperation::ZeroOutputs),
            RematerializationError::InvalidStorageOperation { message }
                if message == "storage operation 'zero_outputs' must produce exactly one output but produced 0",
        ));
        assert!(matches!(
            error(InvalidStorageTestOperation::TwoOutputs),
            RematerializationError::InvalidStorageOperation { message }
                if message == "storage operation 'two_outputs' must produce exactly one output but produced 2",
        ));
        assert!(matches!(
            error(InvalidStorageTestOperation::RejectsInput),
            RematerializationError::InvalidStorageOperation { message }
                if message == format!(
                    "storage operation 'rejects_input' rejected its single input of type {scalar_type}: unsupported storage input",
                ),
        ));
        assert!(matches!(
            error(InvalidStorageTestOperation::Effectful),
            RematerializationError::InvalidStorageOperation { message }
                if message == "storage operation 'effectful' must be pure but declares effects",
        ));

        let restored_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]));
        let error = validate_restored_residual_type(&restored_type, &scalar_type).unwrap_err();
        assert!(matches!(
            error,
            RematerializationError::InvalidStorageOperation { message }
                if message == format!(
                    "storage restoration produced type {restored_type} but the logical residual type is {scalar_type}",
                ),
        ));
    }

    #[test]
    fn test_with_prevent_cse_invalidates_cached_derivations() {
        // Reconfiguring `prevent_cse` after a derivation was cached must not serve the stale cached operation, whose
        // staged flag was baked in at derivation time.
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(dot_sine_body);
        EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let function = function.with_prevent_cse(false);
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("rematerialization should stage a rematerialize call");
        };
        assert!(!operation.prevent_cse(), "the reconfigured flag must invalidate the cached derivation");
    }

    /// Returns whether `program` stages any `print` instructions.
    fn contains_print(
        program: &crate::programs::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> bool {
        program.instructions().iter().any(|instruction| instruction.operation().name() == "print")
    }

    #[test]
    fn test_effectful_residuals_are_force_saved_and_never_recomputed() {
        use crate::operations::Print;

        // The body prints its intermediate, so the linearized primal contains one ordered-I/O instruction whose
        // output is a residual. Recompute slices may only copy pure instructions, so the classification pass
        // force-saves that residual: the forward program retains the print (it executes exactly once, in the
        // forward pass), the backward and tangent programs consume the saved value and contain no print, and the
        // gradient is unchanged. Under the previous replay-based derivation the backward and tangent programs
        // retained a copy of the print and re-executed it per interpretation (behavioral fix).
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let printed = (x.clone() * x.clone()).print("u");
            Ok(printed * x)
        }
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body);
        let operation = staged_operation(&function, ArrayType::scalar(DataType::F64));
        assert!(contains_print(operation.forward()), "the forward program must retain the effectful instruction");
        assert!(!contains_print(operation.backward()), "the backward program must not replay the effect");
        assert!(!contains_print(operation.tangent()), "the tangent program must not replay the effect");
        // With `NothingSaveable`, the print output is the only saved residual: outputs are the body output, the
        // region input, and the force-saved printed value.
        assert_eq!(operation.forward().output_types().len(), 3);

        // f(x) = x² · x = x³, so f'(x) = 3x².
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(2.0))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 8.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 12.0, epsilon = 1e-9);
    }

    #[test]
    fn test_effect_force_saving_is_topological_so_later_residuals_recompute_from_upgraded_cuts() {
        use crate::operations::Print;

        // `p = print(x · x)` is force-saved (its slice is the non-pure print). `y = p * p` depends on `p`, but once
        // `p` is upgraded to a saved cut, `y`'s recompute slice terminates there and stays pure, so `y` is
        // recomputed — the producer-topological order makes later residuals see earlier upgrades. The forward
        // program therefore saves exactly one residual (`p`), and the backward program recomputes `y` from it.
        fn body(
            x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
        ) -> Result<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
            let printed = (x.clone() * x.clone()).print("p");
            let y = printed.clone() * printed;
            Ok(y * x)
        }
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(body);
        let operation = staged_operation(&function, ArrayType::scalar(DataType::F64));
        assert!(contains_print(operation.forward()));
        assert!(!contains_print(operation.backward()));
        assert!(!contains_print(operation.tangent()));
        // Saved: exactly the force-saved `p` (outputs = body output + region input + `p`); `y` recomputes.
        assert_eq!(operation.forward().output_types().len(), 3);

        // f(x) = (x²)² · x = x⁵, so f'(x) = 5x⁴.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(1.5))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 1.5f64.powi(5), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 5.0 * 1.5f64.powi(4), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_residual_candidates_expose_the_replayed_forward_producer() {
        use crate::tracing_v2::custom_derivatives::custom_vjp;

        // Phase 0 boundary pin: the custom-VJP *forward* program is replayed through the linearization, so the
        // declared residual's producing instruction is the replayed internal `cos` — not the opaque call — while the
        // user-owned backward program contributes no candidates at all.
        let custom = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let names = Rc::new(RefCell::new(Vec::new()));
        let recorded = names.clone();
        let policy =
            PolicyFn::new(move |candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
                recorded
                    .borrow_mut()
                    .extend(candidate.producers().iter().map(|producer| producer.operation().name().to_string()));
                Ok::<_, RematerializationRejection>(RematerializationDecision::<NoStorage>::Recompute)
            });
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            move |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| custom.call(x),
        )
        .with_policy(policy);
        EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), ArrayType::scalar(DataType::F64))
            .unwrap();
        assert_eq!(names.borrow().clone(), vec!["cos".to_string()]);
    }

    #[test]
    fn test_bounded_while_residual_candidates_classify_through_the_staged_loop() {
        use crate::operations::{ComparisonDirection, WhileOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Phase 0 boundary pin for bounded-while residual classification. The loop body doubles its carry, so the
        // bounded loop's residual stacks reach the boundary; this records which producing operations the policy
        // sees for them.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(scalar_type.clone());
            let bound = builder.add_constant(Array::scalar(8.0));
            let predicate = builder
                .add_instruction(
                    crate::operations::compare::CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![state, bound],
                )
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(scalar_type.clone());
            let doubled = builder
                .add_instruction(crate::operations::math::MulOperation::new(), Vec::new(), vec![state, state])
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let names = Rc::new(RefCell::new(Vec::new()));
        let recorded = names.clone();
        let policy =
            PolicyFn::new(move |candidate: &RematerializationCandidate<'_, ArrayType, ArrayOperation<Array>>| {
                recorded
                    .borrow_mut()
                    .extend(candidate.producers().iter().map(|producer| producer.operation().name().to_string()));
                Ok::<_, RematerializationRejection>(RematerializationDecision::<NoStorage>::Recompute)
            });
        let remat_condition = condition.clone();
        let remat_body = body.clone();
        let function = rematerialize::<EagerContext<Array, ArrayOperation<Array>>, _, _, _>(
            move |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let context = x.context().clone();
                let operation = WhileOperation::new().with_iteration_bound(3)?;
                let outputs = context.stage_operation(
                    ArrayOperation::While(operation),
                    vec![remat_condition.clone(), remat_body.clone()],
                    &[x],
                )?;
                Ok(outputs.into_iter().next().unwrap())
            },
        )
        .with_policy(policy);
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(1.1))
            .unwrap();
        // x -> x^2 -> x^4 -> x^8 (three squarings before the predicate 1.1^8 > 8 stops the loop... the exact
        // iteration count is loop-driven; the assertions below only require derivative consistency).
        let direct = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let operation = WhileOperation::new().with_iteration_bound(3).unwrap();
                    let outputs = context
                        .bind(ArrayOperation::While(operation), vec![condition.clone(), body.clone()], &[x])
                        .unwrap();
                    outputs.into_iter().next().unwrap()
                },
                Array::scalar(1.1),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], direct.0.to_f64s()[0], epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], direct.1.to_f64s()[0], epsilon = 1e-9);
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), ArrayType::scalar(DataType::F64))
                .unwrap();
        let instruction = &program.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::Rematerialize(_)));
        let forward = program.region_ref(instruction.regions()[1]).unwrap();
        // Pinned boundary behavior: a bounded while keeps its residual stacks internal to the staged loop, so no
        // while-produced residuals cross the rematerialization boundary and the policy is never consulted for them.
        // The forward program stores only the body output and the region input.
        assert!(names.borrow().is_empty(), "bounded-while loops contribute no residual candidates");
        assert_eq!(forward.output_types().len(), 2);
    }
}
