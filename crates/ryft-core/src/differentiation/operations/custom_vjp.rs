use std::fmt::Display;
use std::marker::PhantomData;

// TODO(eaplatanios): Review this module.

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::batching::CotangentBatchingPolicy;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types, impl_non_transposable_operation};
use crate::operations::Zero;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Effect, Operation, OperationFormatter, ProgramError, RegionInterface, RegionSlot, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, Trace};

use super::LinearCallOperation;

/// Canonical operation name for [`CustomVjpOperation`].
pub const CUSTOM_VJP_OPERATION_NAME: &str = "custom_vjp";

/// Higher-order [`Operation`] pairing a primal program with user-supplied forward/backward (VJP) programs — the direct
/// analogue of JAX's [`custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).
///
/// The three [`Program`](crate::Program)s are supplied as the operation's attached regions (via the region driver
/// passed to [`Context::bind`]) in the region order `["primal", "forward", "backward"]`, and
/// [`Operation::infer_output_types`] validates the interface contract between them: the forward region consumes the
/// primal inputs and produces the primal outputs followed by arbitrarily many residual values, and the backward region
/// consumes the leading non-differentiated operands, then those residuals, then one cotangent per primal output, and
/// produces one cotangent per *differentiated* primal input. Keeping the primal program separate from the forward
/// program means un-differentiated calls never pay for residual computation.
///
/// The leading [`non_differentiated_count`](Self::non_differentiated_count) operands parameterize the call without being
/// differentiated: the primal and forward regions receive them in their own leading positions, the backward region
/// receives them ahead of the residuals, and they receive no cotangent. This is the same operand split
/// [`LinearCallOperation`] draws with its residual count, and the direct analogue of JAX's `nondiff_argnums`. Batching
/// is its canonical producer: a policy that threads batching state through a structurally batched region's boundary
/// (e.g., a composite universe's first-class mapped extent) reintroduces that state as additional leading
/// non-differentiated operands of the batched call.
///
/// The transforms treat a staged call as follows: interpretation replays the primal region; partial evaluation folds a
/// call whose operands are all known and otherwise residualizes it unchanged; batching preserves the call around
/// axis-reconciled copies of all three regions so the custom derivative survives a `batch` applied *before*
/// differentiation; and differentiation replays the forward region for the primal outputs and residuals and stages a
/// transpose-only [`LinearCallOperation`] carrier for the output tangents, whose transpose replays the user backward
/// program, so reverse mode uses exactly the user-supplied gradient. Because that carrier rejects interpretation,
/// forward-mode differentiation of a staged call is rejected, matching JAX's reverse-mode-only `custom_vjp`
/// semantics. Refer to the documentation of [`custom_vjp`] for the full semantics and for when to reach for a custom
/// VJP.
///
/// This operation is deliberately non-transposable, which does not restrict reverse-mode differentiation. Reverse mode
/// linearizes first, and the `jvp` rule replaces the operation with the transpose-only carrier described above — the
/// analogue of JAX's `custom_lin` primitive — so the operation itself is gone from the tangent program long before
/// transposition runs. Transposition can therefore only reach the operation when transposing a raw, un-linearized
/// program directly, which JAX rejects for its `custom_vjp_call` primitive in exactly the same way.
///
/// The `T` parameter fixes the type universe of all attached regions and the call boundary, so each concrete payload
/// has exactly one [`Operation<Type = T>`](Operation) contract while the semantic and transform implementations remain
/// shared across differentiable type universes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomVjpOperation<T: DifferentiableType> {
    /// Number of leading operands that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Type universe in which this custom-VJP call is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: DifferentiableType> Copy for CustomVjpOperation<T> {}

impl<T: DifferentiableType> Default for CustomVjpOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DifferentiableType> CustomVjpOperation<T> {
    /// Creates a custom-VJP call operation whose attached regions operate on `T` values and whose operands are all
    /// differentiated.
    #[inline]
    pub const fn new() -> Self {
        Self { non_differentiated_count: 0, marker: PhantomData }
    }

    /// Sets the number of leading operands that parameterize this call without being differentiated. Refer to the
    /// documentation of [`CustomVjpOperation`] for the resulting region interfaces.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Returns the number of leading operands that parameterize this call without being differentiated.
    #[inline]
    pub fn non_differentiated_count(&self) -> usize {
        self.non_differentiated_count
    }

    /// Splits `values` into the leading non-differentiated group and the trailing differentiated group.
    #[inline]
    fn split_inputs<'v, V>(&self, values: &'v [V]) -> Result<(&'v [V], &'v [V]), TypeError> {
        let input_count = values.len();
        if self.non_differentiated_count > input_count {
            return Err(TypeError::invalid(format!(
                "{} non-differentiated operand count {} exceeds input count {}",
                self.name(),
                self.non_differentiated_count,
                input_count,
            )));
        }
        Ok(values.split_at(self.non_differentiated_count))
    }

    /// Validates the custom-VJP contract over the three attached region interfaces
    /// (`["primal", "forward", "backward"]` region order) and returns the primal interface; refer to the documentation
    /// of [`CustomVjpOperation`] for the contract.
    fn validated_interfaces<'i>(
        &self,
        region_interfaces: &'i [RegionInterface<T>],
    ) -> Result<&'i RegionInterface<T>, TypeError> {
        check_count!("region", region_interfaces, 3, TypeError);
        let primal_interface = &region_interfaces[0];
        let forward_interface = &region_interfaces[1];
        let backward_interface = &region_interfaces[2];
        let input_types = primal_interface.input_types();
        let output_types = primal_interface.output_types();
        let (non_differentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        check_types!(@same, format!("{CUSTOM_VJP_OPERATION_NAME} forward input"), [
            input_types,
            forward_interface.input_types(),
        ]);
        let forward_output_types = forward_interface.output_types();
        if forward_output_types.len() < output_types.len() {
            return Err(TypeError::invalid(format!(
                "{} forward must produce at least the {} primal output(s) but produced {} value(s)",
                CUSTOM_VJP_OPERATION_NAME,
                output_types.len(),
                forward_output_types.len(),
            )));
        }
        check_types!(@same, format!("{CUSTOM_VJP_OPERATION_NAME} forward output"), [
            output_types,
            &forward_output_types[..output_types.len()],
        ]);
        let residual_types = &forward_output_types[output_types.len()..];
        let output_cotangent_types = output_types
            .iter()
            .map(DifferentiableType::cotangent)
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let expected_backward_input_types: Vec<T> = non_differentiated_types
            .iter()
            .chain(residual_types)
            .cloned()
            .chain(output_cotangent_types)
            .collect();
        check_types!(@same, format!("{CUSTOM_VJP_OPERATION_NAME} backward input"), [
            &expected_backward_input_types,
            backward_interface.input_types(),
        ]);
        let expected_backward_output_types = differentiated_types
            .iter()
            .map(DifferentiableType::cotangent)
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        check_types!(@same, format!("{CUSTOM_VJP_OPERATION_NAME} backward output"), [
            &expected_backward_output_types,
            backward_interface.output_types(),
        ]);
        Ok(primal_interface)
    }
}

impl<T: DifferentiableType> Display for CustomVjpOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation for CustomVjpOperation<T> {
    // The operation carries three regions. Writing the leading non-differentiated operands as `p`, differentiated
    // operands as `x`, primal outputs as `y`, forward residuals as `r`, and cotangents using an overbar, their contracts
    // are
    //
    //   primal:   (p, x)    → y
    //   forward:  (p, x)    → (y, r)
    //   backward: (p, r, ȳ) → x̄.
    //
    // Inference renames the independently traced primal and forward identities into the call boundary, derives the
    // backward signature from their resulting `y` and `r` types, and validates that no cotangent is produced for `p`.
    // Keeping the split explicit distinguishes a parameter operand from a differentiable operand whose cotangent merely
    // evaluates to zero.
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_VJP_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 3, TypeError);
        let primal_interface = &region_interfaces[0];
        let forward_interface = &region_interfaces[1];

        // The primal and forward regions were traced independently, so each boundary owns its own formal identities.
        // Derive each region's caller-specific renaming from its input boundary before using its outputs to construct
        // the backward region's input signature.
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
                "{} forward must produce at least the {} primal output(s) but produced {} value(s)",
                CUSTOM_VJP_OPERATION_NAME,
                primal_output_types.len(),
                forward_output_types.len(),
            )));
        }
        let (non_differentiated_types, _) = self.split_inputs(input_types)?;
        let mut backward_input_types = non_differentiated_types.to_vec();
        backward_input_types.extend_from_slice(&forward_output_types[primal_output_types.len()..]);
        backward_input_types.extend(
            primal_output_types
                .iter()
                .map(DifferentiableType::cotangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?,
        );
        Ok(vec![Some(input_types.to_vec()), Some(input_types.to_vec()), Some(backward_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let primal_interface = self.validated_interfaces(region_interfaces)?;
        check_types!(@same, format!("{CUSTOM_VJP_OPERATION_NAME} input"), [
            primal_interface.input_types(),
            input_types,
        ]);
        Ok(primal_interface.output_types().to_vec())
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // A call whose operands are all differentiated renders as a bare name, so the non-differentiated split appears
        // in rendered programs exactly where it exists.
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_VJP_OPERATION_NAME)?;
        if self.non_differentiated_count == 0 {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("non_differentiated_count", self.non_differentiated_count))
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomVjpOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // An ordinary call computes only `f(p, x) = y`. The forward region additionally materializes residuals `r`
        // solely for reverse mode, so interpretation replays the lean primal region at slot 0.
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: DifferentiableType>> PartiallyEvaluatableOperation<C> for CustomVjpOperation<C::Type>
where
    C::Operation: From<CustomVjpOperation<C::Type>>,
{
    // The default partial-evaluation rule is the desired one: interpret the primal region when every operand is known;
    // otherwise residualize the complete custom-VJP call so its forward and backward regions remain attached for a
    // later reverse-mode transformation.
}

impl<T: DifferentiableType, C: Context<Type = T>, P: CotangentBatchingPolicy<C>> BatchableOperation<C, P>
    for CustomVjpOperation<T>
where
    C::Operation: From<CustomVjpOperation<T>>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Batch all three region contracts while retaining the opaque custom-VJP carrier:
        //
        //   primal:   (p, x)    → y
        //   forward:  (p, x)    → (y_fwd, r)
        //   backward: (p, r, ȳ) → x̄.
        //
        // Reconcile each `y` with its corresponding `y_fwd` so the wrapper exposes one physical output axis, but keep
        // every residual's naturally produced axis because residuals are internal edges between the forward and
        // backward rules. Batch the backward region with `(p, r, ȳ)` on those exact axes and align each `x̄`
        // with its corresponding differentiated input `x`. When a replicated `x` receives a mapped cotangent, the
        // batching policy sums that mapped axis, which is the transpose of broadcasting `x` across the batch.
        //
        // A batching policy may add runtime boundary operands such as a first-class mapped extent. Those values must
        // reach all three regions but have no cotangent, so prepend them to `p` and increase
        // `non_differentiated_count` after the regions have been adapted to their new boundaries.
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (non_differentiated_axes, differentiated_axes) = self.split_inputs(input_axes.as_slice())?;
        let differentiated_axes = differentiated_axes.to_vec();
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let backward_region = driver.region(2)?;

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
                "batched {} forward region produced {} outputs which is fewer than its {} primal outputs",
                CUSTOM_VJP_OPERATION_NAME,
                forward_output_axes.len(),
                primal_output_axes.len(),
            ))
            .into());
        }
        let (forward_primal_output_axes, residual_axes) = forward_output_axes.split_at(primal_output_axes.len());

        // The ordinary primal and the primal prefix of the forward rule must expose one physical wrapper boundary.
        // Residuals are internal to the derivative rule and retain the axes naturally produced by the forward region.
        let output_axes = primal_output_axes
            .iter()
            .copied()
            .zip(forward_primal_output_axes.iter().copied())
            .map(|(primal, forward)| {
                [primal, forward].into_iter().find(|axis| !axis.is_replicated()).unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let residual_axes = residual_axes.to_vec();
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

        // The backward rule maps `(non_differentiated..., residuals..., output_cotangents...)` to the differentiated
        // inputs' cotangents. Align mapped results to their primal input positions while they are live; adaptation
        // then sums the only non-structural mismatch, namely a mapped cotangent corresponding to a replicated primal
        // input.
        let backward_input_axes = non_differentiated_axes
            .iter()
            .chain(&residual_axes)
            .chain(&output_axes)
            .copied()
            .collect::<Vec<_>>();
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
                    "batched {CUSTOM_VJP_OPERATION_NAME} backward output axes {backward_output_axes:?} do not match its \
                 differentiated input axes {differentiated_axes:?}",
                ),
            });
        }

        let boundary_operands = P::boundary_operands(context.axis_extent());
        let non_differentiated_count = self.non_differentiated_count + boundary_operands.len();
        let mut packed_inputs = boundary_operands;
        packed_inputs.extend(inputs.iter().map(P::value).cloned());
        let outputs = context.parent().bind(
            self.with_non_differentiated_count(non_differentiated_count),
            vec![primal, forward, backward],
            packed_inputs.as_slice(),
        )?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for CustomVjpOperation<C::Type>
where
    C::Type: DifferentiableType,
    C::Operation: ResidualZeroProvider<C::Type> + From<LinearCallOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A custom VJP specifies the pullback `ȳ ↦ x̄`, not the pushforward `ẋ ↦ ẏ`. Replay
        //
        //   forward(p, x) = (y, r)
        //
        // to recover the primal outputs and residuals, then stage an opaque linear call representing the unknown map
        //
        //   L_(p,r): ẋ ↦ ẏ.
        //
        // `LinearCallOperation` knows only how to transpose that map: its transpose replays
        // `backward(p, r, ȳ) = x̄`. An eager forward-mode use attempts to execute `L_(p,r)` and is therefore
        // rejected, while reverse mode transposes it without execution. Passing `p` and `r` as the carrier's leading
        // residual operands keeps the path capture-free and exposes every dependency as an ordinary SSA edge.
        //
        // The attached regions are `["primal", "forward", "backward"]`; the primal interface provides the boundary
        // types.
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let backward_region = driver.region(2)?;

        // The forward and backward rule regions are interpreted directly rather than routed through the transform
        // rejections, so unresolved state inside either must be rejected before any of it executes.
        if forward_region.effects().union(backward_region.effects()).contains(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{CUSTOM_VJP_OPERATION_NAME}` rule regions must not contain unresolved state"),
            }
            .into());
        }

        let output_count = primal_region.output_types().len();
        check_count!("input", inputs, primal_region.input_types().len(), ProgramError);
        // Replay the forward region on the dual primals, recovering the primal outputs followed by the residuals.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = forward_region.interpret_in_context(context, primal_operands)?;
        if forward_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "{} forward region produced {} outputs which is fewer than its {} primal output(s)",
                CUSTOM_VJP_OPERATION_NAME,
                forward_outputs.len(),
                output_count,
            ))
            .into());
        }
        let residuals = forward_outputs.split_off(output_count);
        let primal_outputs = forward_outputs;
        let (non_differentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;

        if let Some(input) = non_differentiated_inputs
            .iter()
            .find(|input| !input.tangent().is_zero() && !input.tangent().r#type().is_zero_space())
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot propagate the nonzero tangent of type `{}` supplied for one of its {} leading \
                     non-differentiated operands, because its rule has no tangent slot for them",
                    self.name(),
                    input.tangent().r#type(),
                    non_differentiated_inputs.len(),
                ),
            }
            .into());
        }

        let input_tangent_types = differentiated_inputs
            .iter()
            .map(|input| input.primal().r#type().tangent())
            .collect::<Result<Vec<_>, _>>()?;
        let output_tangent_types =
            primal_outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?;

        // Stage one opaque carrier over `[non_differentiated..., residuals..., differentiated_input_tangents...]`,
        // producing the output tangents. The carrier rejects forward interpretation and transposes by replaying the
        // user's backward region, whose own inputs are exactly that leading residual group followed by the output
        // cotangents. The transpose-only carrier takes every differentiated input tangent as a real operand, so
        // materialize structural zeros against their own primal, which names every runtime quantity a
        // reference-bearing tangent type omits.
        let mut carrier_operands =
            non_differentiated_inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        carrier_operands.extend(residuals);
        // The carrier's leading non-tangent group is the non-differentiated operands followed by the residuals, and
        // both are passed through the residual-count slot: to the linear call they are alike operands that its
        // transpose forwards to the backward region rather than transposing.
        let leading_operand_count = carrier_operands.len();
        carrier_operands.extend(
            differentiated_inputs
                .iter()
                .map(|input| {
                    C::Operation::materialize_zero_from_residual_sources(
                        context,
                        input.tangent().clone(),
                        std::iter::once(input.primal()),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let carrier =
            LinearCallOperation::transpose_only(leading_operand_count, input_tangent_types, output_tangent_types);
        // Any context that must *execute* the carrier (an eager forward-mode pass, or a forward-mode pass over an
        // already staged carrier) rejects it as unsupported. Restate that rejection in `custom_vjp` vocabulary instead
        // of leaking the carrier's internals, matching the clear error JAX raises when forward-mode autodiff is
        // applied to a `custom_vjp` function.
        let output_tangents = context.bind(carrier, vec![backward_region.to_program()], &carrier_operands).map_err(
            |error| match error {
                ProgramError::UnsupportedOperation { .. } => ProgramError::UnsupportedOperation {
                    message: format!(
                        "cannot apply forward-mode differentiation to a {CUSTOM_VJP_OPERATION_NAME} call; it supports \
                         only reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
                    ),
                },
                error => error,
            },
        )?;
        check_count!("output", output_tangents, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(output_tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

// The raw carrier is intentionally non-transposable. Differentiation first replaces `f(p, x)` with the opaque linear
// map `L_(p,r): ẋ ↦ ẏ`; reverse mode transposes that `LinearCallOperation`, whose rule evaluates
// `backward(p, r, ȳ) = x̄`. Therefore only an invalid direct transpose of an un-linearized custom-VJP carrier
// can reach this rejection path.
impl_non_transposable_operation!(<T> CustomVjpOperation<T> where T: DifferentiableType);

/// Function with user-supplied forward/backward (VJP) rules, built by [`custom_vjp`]. It stores the primal, forward,
/// and backward closures together with a phantom marker pinning the tracer-tree types named by those closure
/// signatures; refer to the documentation of [`custom_vjp`] for the calling convention, the tracing semantics, and
/// when to reach for a custom VJP.
pub struct CustomVjp<Primal, Forward, Backward, Inputs, Outputs, Residuals> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: Primal,

    /// Closure computing `(outputs, residuals)` from the primal input tree.
    forward: Forward,

    /// Closure computing the input cotangent tree from `(residuals, output_cotangents)`.
    backward: Backward,

    /// Phantom marker pinning the input, output, and residual tracer-tree types named by the closure signatures. The
    /// [`Domain`] whose universe the rules are traced into is recovered from the values passed to [`CustomVjp::call`],
    /// and so the wrapper stores neither a domain value nor a domain type witness.
    marker: PhantomData<fn() -> (Inputs, Outputs, Residuals)>,
}

impl<Primal, Forward, Backward, Inputs, Outputs, Residuals>
    CustomVjp<Primal, Forward, Backward, Inputs, Outputs, Residuals>
where
    Primal: Fn(Inputs) -> Result<Outputs, ProgramError>,
    Forward: Fn(Inputs) -> Result<(Outputs, Residuals), ProgramError>,
    Backward: Fn(Residuals, Outputs) -> Result<Inputs, ProgramError>,
{
    /// Stages this custom-VJP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Reverse-mode differentiation of the staged
    /// call replays the backward rule on the forward rule's residuals instead of differentiating the primal body.
    ///
    /// The [`Domain`] `D` whose universe the three rule programs are traced into is the
    /// [`DispatchDomain`](Value::DispatchDomain) of the values this is called with, which is exactly the context the
    /// call is staged into. It is therefore recovered from `input` and never has to be named at a construction or call
    /// site, while the stored closures still pin the tracer trees that universe must produce.
    pub fn call<D, V, InputValues>(
        &self,
        input: InputValues,
    ) -> Result<<Outputs::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        D: Context<Type: DifferentiableType, Value = V>,
        V: Value<Type = D::Type, DispatchDomain = D>,
        D::Operation: From<CustomVjpOperation<D::Type>>,
        Inputs: Parameterized<DomainTracer<D>>,
        Inputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
        Outputs: Parameterized<DomainTracer<D>>,
        Outputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<V>,
        Residuals: Parameterized<DomainTracer<D>>,
        Residuals::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
        Inputs::To<D::Type>: Clone + Parameterized<D::Type, Family = Inputs::Family, To<DomainTracer<D>> = Inputs>,
        Outputs::To<D::Type>: Clone + Parameterized<D::Type, Family = Outputs::Family, To<DomainTracer<D>> = Outputs>,
        Residuals::To<D::Type>: Parameterized<D::Type, Family = Residuals::Family, To<DomainTracer<D>> = Residuals>,
        InputValues: Parameterized<V, Family = Inputs::Family, To<D::Type> = Inputs::To<D::Type>>,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_values.first() else {
            return Err(TypeError::invalid(format!("{CUSTOM_VJP_OPERATION_NAME} requires at least one input")).into());
        };
        let (output_types, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let (forward_output_types, forward) = D::trace(|xs| (self.forward)(xs), input_types.clone())?;
        let (_, residual_types) = forward_output_types;
        let output_cotangent_types = output_types.clone().try_map_parameters(|r#type| r#type.cotangent())?;
        let (_, backward) = D::trace(
            |(residuals, cotangents)| (self.backward)(residuals, cotangents),
            (residual_types, output_cotangent_types),
        )?;
        let operation = D::Operation::from(CustomVjpOperation::new());
        // Bind through whatever context the inputs flow, so `custom_vjp` composes under `vmap`/`jvp`.
        let context = first.dispatch_domain();
        let outputs = context.bind(
            operation,
            vec![primal.to_flat_program(), forward.to_flat_program(), backward.to_flat_program()],
            &input_values,
        )?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over trees of [`DomainTracer`]s. This
/// is the ergonomic analogue of JAX's
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair.
///
/// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The Vector-Jacobian Product (VJP), or
/// pullback, maps an output cotangent `ȳ` to the input cotangent `x̄ = J_f(x)ᵀ · ȳ`. The three closure arguments factor
/// that computation through a residual tree `r`:
///
/// ```text
/// primal:   x      ↦ y = f(x)
/// forward:  x      ↦ (y, r) = (f(x), r)
/// backward: (r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ
/// ```
///
/// Thus, `primal` implements `f` for ordinary evaluation. `forward` recomputes `y` and saves exactly the residual tree
/// `r` needed by the reverse rule. `backward` receives `r` and the output-cotangent tree `ȳ`, then returns the
/// input-cotangent tree `x̄`. Ryft validates that both occurrences of `y` agree, that `ȳ` is the cotangent of `y`, and
/// that `x̄` is the cotangent of `x` when it traces the closures.
///
/// # When to use
///
/// Reach for a custom VJP when only the *reverse* rule is natural, or when the function is not (efficiently)
/// forward-differentiable. Common cases:
///
///   - **Implicit differentiation** — differentiate through a solver, optimizer, or fixed point via the implicit
///     function theorem rather than unrolling its iterations.
///   - **Adjoint methods** — backpropagate through an ODE or PDE solve via the adjoint system instead of
///     differentiating the integrator's individual steps.
///   - **External or black-box calls** — supply the reverse rule for a custom kernel or a computation that does not
///     itself trace into `ryft` programs.
///   - **Numerical stability** — replace an unstable or wasteful automatically derived gradient with a hand-written
///     one.
///
/// A custom VJP is reverse-mode only: forward-mode differentiation of a staged call is rejected, and the current
/// transpose implementation also rejects transposing its generated pullback, so higher-order derivatives through a
/// custom VJP are not yet supported. When the function is forward-differentiable or must participate in higher-order
/// differentiation, use [`custom_jvp`](crate::differentiation::custom_jvp) instead.
///
/// # Calling convention
///
/// All three closures operate on [`Parameterized`] trees of [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so
/// `x`, `y`, and `r` may each be a single tracer, a tuple, or any other parameterized structure. This high-level API
/// does not expose JAX's `nondiff_argnums` calling convention. Static non-differentiated configuration should be
/// captured by all three closures. A dynamic typed value should remain an explicit input; preserve it as a residual in
/// `r` when `backward` needs it and return a zero cotangent for it in `x̄`. Transform-injected runtime metadata uses the
/// lower-level [`CustomVjpOperation::non_differentiated_count`] contract instead, because it must remain an SSA operand
/// while contributing no cotangent output.
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y` for ordinary evaluation.
///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse-mode residual production.
///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
///
/// # Tracing semantics
///
/// Nothing is traced at construction time: each [`CustomVjp::call`] recovers the tracing [`Domain`] from the values it
/// is called with, reads the input types off those arguments, traces the closures into programs specialized to those
/// types, validates the rule signatures, and stages one [`CustomVjpOperation`] into the caller's staging context —
/// mirroring how JAX traces rule functions into jaxprs lazily at transform time. The primal closure is kept separate
/// from the forward closure for efficiency rather than necessity: an un-differentiated call should not pay for residual
/// computation. Interpretation, batching, and backend lowering replay the lean primal program, and the
/// residual-producing forward program runs only under reverse-mode
/// differentiation. Callers that do not care about the distinction can pass the same body for both — accepting that the
/// residual outputs are dead code outside of differentiation — which mirrors the common JAX idiom of writing `f_fwd` as
/// `return f(x), residuals`.
pub fn custom_vjp<Primal, Forward, Backward, Inputs, Outputs, Residuals>(
    primal: Primal,
    forward: Forward,
    backward: Backward,
) -> CustomVjp<Primal, Forward, Backward, Inputs, Outputs, Residuals>
where
    Primal: Fn(Inputs) -> Result<Outputs, ProgramError>,
    Forward: Fn(Inputs) -> Result<(Outputs, Residuals), ProgramError>,
    Backward: Fn(Residuals, Outputs) -> Result<Inputs, ProgramError>,
{
    CustomVjp { primal, forward, backward, marker: PhantomData }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Dimension, Shape, ShardingDimension};
    use crate::batching::{Batch, BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        Differentiate, ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate, differentiate_at,
    };
    use crate::operations::{Cos, CosOperation, MulOperation, Reduce, ReductionKind, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{Effects, Program, ProgramBuilder, RegionRole};

    use super::*;

    /// Returns the canonical test array type with the provided dimensions.
    fn test_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(
            DataType::F64,
            Shape::new(dimensions.iter().map(|dimension| Dimension::Static(*dimension)).collect()),
        )
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent`, detectably
    /// different from the true gradient so tests can prove the custom rule is used.
    fn tripled_sin_backward_program(
        r#type: &ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(Array::scalar(3.0));
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns a custom-VJP call over `f(x) = sin(x)` together with its `["primal", "forward", "backward"]` regions,
    /// whose backward rule deliberately triples the true gradient.
    fn custom_vjp_sin(
        r#type: &ArrayType,
    ) -> (ArrayOperation<Array>, Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>) {
        (
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            vec![sin_program(r#type), sin_forward_program(r#type), tripled_sin_backward_program(r#type)],
        )
    }

    /// Builds one flat program that binds `operation` with `regions` to inputs of `input_types`.
    fn wrapped_call_program(
        input_types: Vec<ArrayType>,
        operation: ArrayOperation<Array>,
        regions: Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let region_ids =
            regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect::<Vec<_>>();
        let input_count = input_types.len();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, region_ids, inputs).unwrap().to_vec();
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }

    #[test]
    fn test_custom_vjp() {
        let scalar = test_type(&[]);
        let operation = CustomVjpOperation::<ArrayType>::new();

        // Verify the operation's identity, rendering, and region contract: the primal program is a computation region
        // followed by the user forward and backward rule regions.
        assert_eq!(operation.name(), CUSTOM_VJP_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "custom_vjp");
        assert_eq!(
            operation.region_slots(),
            &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")],
        );
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));
        assert_eq!(operation.region_role(2), Some(RegionRole::Rule));

        // The primal and forward regions receive the call inputs, and the backward region receives the forward
        // region's trailing residuals followed by one cotangent per primal output.
        let primal_interface = sin_program(&scalar).interface();
        let forward_interface = sin_forward_program(&scalar).interface();
        let backward_interface = tripled_sin_backward_program(&scalar).interface();
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Ok(vec![
                Some(vec![scalar.clone()]),
                Some(vec![scalar.clone()]),
                Some(vec![scalar.clone(), scalar.clone()]),
            ]),
        );

        // Rules that satisfy the interface contract make the call produce the primal outputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar),
                &[primal_interface, forward_interface, backward_interface],
            ),
            Ok(vec![scalar.clone()]),
        );

        // Inference maps the primal boundary through *differential* types, so the cotangent boundary of an
        // `f8e8m0fnu` primal is `f32` while the residual keeps its own storage type.
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(Vec::new()));
        let cotangent_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()));
        let differential_primal_interface =
            RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], Effects::PURE);
        let differential_forward_interface =
            RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone(), scalar.clone()], Effects::PURE);
        let differential_backward_interface =
            RegionInterface::new(vec![scalar.clone(), cotangent_type.clone()], vec![cotangent_type], Effects::PURE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[differential_primal_interface, differential_forward_interface, differential_backward_interface],
            ),
            Ok(vec![primal_type]),
        );

        // The backward interface must consume `(residuals..., output cotangents...)`; a single-input rule whose
        // signature cannot line up with the forward residuals is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar),
                &[
                    sin_program(&scalar).interface(),
                    sin_forward_program(&scalar).interface(),
                    sin_program(&scalar).interface(),
                ],
            ),
            Err(TypeError::invalid(
                "custom_vjp backward input type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_vjp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let (operation, operation_regions) = custom_vjp_sin(&scalar);

        // Interpretation replays the lean primal region only, so an un-differentiated call produces just the primal
        // output and never pays for the forward region's residual computation.
        let outputs = EagerContext::<Array, ArrayOperation<Array>>::new()
            .bind(operation, operation_regions, &[Array::scalar(2.0)])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_remains_opaque_to_partial_evaluation() {
        // A call with an unknown operand residualizes unchanged instead of inlining its primal region, which is what
        // keeps the custom rule attached to the residual program.
        let scalar = test_type(&[]);
        let (operation, operation_regions) = custom_vjp_sin(&scalar);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let region_ids = operation_regions
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(scalar.clone());
        let output = builder.add_instruction(operation, region_ids, vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar)]).unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomVjp(_)));
    }

    #[test]
    fn test_custom_vjp_survives_batching_and_governs_the_batched_gradient() {
        // The reverse-mode analogue of the custom-JVP batching test: the (deliberately tripled) custom backward rule
        // governs the gradient through the batched call — mirroring JAX's `vmap`-of-`custom_vjp` semantics.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::vector(vec![0.5, 1.0]))
            .value_and_gradient(|x| {
                let context = x.context().clone();
                let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                    &context,
                    |item| {
                        let (operation, operation_regions) = custom_vjp_sin(&test_type(&[]));
                        Ok(item
                            .context()
                            .bind(operation, operation_regions, &[item.clone()])?
                            .into_iter()
                            .next()
                            .unwrap())
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.5f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[1], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(2.0))
            .value_and_gradient(|x| {
                let (operation, operation_regions) = custom_vjp_sin(&test_type(&[]));
                x.context().bind(operation, operation_regions, &[x.clone()]).unwrap().into_iter().next().unwrap()
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_rejects_forward_mode() {
        // A `custom_vjp` supplies no forward tangent program, so the tangent carrier its `jvp` rule stages cannot be
        // executed. Forward mode must therefore fail with a user-facing custom-VJP error rather than leaking the
        // carrier's internal vocabulary, matching JAX's "can't apply forward-mode autodiff to a custom_vjp function".
        let result = EagerContext::<Array, ArrayOperation<Array>>::new().jvp(
            |x, ()| {
                let (operation, operation_regions) = custom_vjp_sin(&test_type(&[]));
                Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
            },
            Array::scalar(2.0),
            Array::scalar(1.0),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward-mode differentiation to a custom_vjp call; it supports only \
                               reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or \
                               `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_custom_vjp_supports_multiple_outputs() {
        // A two-output custom VJP exercises the forward region's output/residual split: its leading values are the
        // primal outputs and the rest are residuals, and the backward region consumes one cotangent per output. The
        // deliberately wrong rule scales the first output's contribution by 2 and the second's by 3, so seeding one
        // output cotangent at a time isolates each term of the custom backward.
        let function = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                Ok(((x.sin()?, x.cos()?), (x.cos()?, x.sin()?)))
            },
            |(cosine, sine): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            ),
             (first, second)| {
                let from_first = cosine * first;
                let from_second = sine * second;
                Ok(from_first.clone() + from_first + from_second.clone() + from_second.clone() + from_second)
            },
        );
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let ((sine, cosine), pullback) = domain.vjp(|x, ()| function.call(x), Array::scalar(0.5), ()).unwrap();
        assert_abs_diff_eq!(sine.to_f64s()[0], 0.5f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(cosine.to_f64s()[0], 0.5f64.cos(), epsilon = 1e-9);

        let first_cotangent = pullback.apply((Array::scalar(1.0), Array::scalar(0.0))).unwrap();
        assert_abs_diff_eq!(first_cotangent.to_f64s()[0], 2.0 * 0.5f64.cos(), epsilon = 1e-9);
        let second_cotangent = pullback.apply((Array::scalar(0.0), Array::scalar(1.0))).unwrap();
        assert_abs_diff_eq!(second_cotangent.to_f64s()[0], 3.0 * 0.5f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_pullback_applies_the_backward_program() {
        // First-order reverse mode through a user custom VJP applies the user-supplied backward program. The
        // reverse entry stages an opaque tangent carrier and the direct transpose replays the backward program forward
        // into the pullback, so seeding the pullback at `[cotangent ++ residuals]` recovers `residual * cotangent`. The
        // user backward defines the residual as `cos(x)`, so at `x = 0.7` and a unit cotangent the input cotangent is
        // `cos(0.7)`.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let (_, pullback) = domain.vjp(|x, ()| function.call(x), Array::scalar(0.7), ()).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_abs_diff_eq!(input_cotangents[0].to_f64s()[0], 0.7f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_governs_the_reverse_jacobian() {
        // `jacobian_reverse` interprets the pullback with batch-stacked cotangent bases, exercising the batched replay
        // of the custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the diagonal
        // matrix `diag(3 * cos(x))`.
        let vector = test_type(&[2]);
        let jacobian = differentiate_at(Array::from_f64s(vector, vec![0.5, 1.0]))
            .jacobian_reverse(|x| {
                let (operation, operation_regions) = custom_vjp_sin(&test_type(&[2]));
                Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_abs_diff_eq!(block.value().to_f64s()[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[3], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_governs_reverse_mode() {
        let function = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                // The deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the
                // true gradient (expressed through addition to avoid constant lifting).
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(2.0))
            .value_and_gradient(|x| function.call(x).unwrap())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_supports_structured_signatures_and_captured_configuration() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` (pytree) calling convention, and the
        // captured `triple` closure plays the role of a JAX `nondiff_argnums` argument: static configuration
        // visible to the rule closures without being differentiated or stored as a residual.
        let repeats = 3usize;
        let function = custom_vjp(
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
            move |(x, y), cotangent| {
                // The deliberately wrong rule repeats both cotangents `repeats` times via the captured count.
                let (base_x, base_y) = (y * cotangent.clone(), x * cotangent);
                let (mut scaled_x, mut scaled_y) = (base_x.clone(), base_y.clone());
                for _ in 1..repeats {
                    scaled_x = scaled_x + base_x.clone();
                    scaled_y = scaled_y + base_y.clone();
                }
                Ok((scaled_x, scaled_y))
            },
        );
        let (value, (gradient_x, gradient_y)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(2.0), Array::scalar(5.0)))
            .value_and_gradient(|(x, y)| function.call((x, y)).unwrap())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 10.0, epsilon = 1e-9);
        // The custom rule triples the true gradients `(y, x)`.
        assert_abs_diff_eq!(gradient_x.to_f64s()[0], 3.0 * 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient_y.to_f64s()[0], 3.0 * 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_supports_empty_residuals() {
        // A forward rule that saves nothing (`Residuals = ()`) exercises the zero-residual carrier path: the backward
        // rule depends only on the output cotangent, so the deliberately wrong `backward(cotangent) = 2 * cotangent`
        // makes the gradient the constant `2` instead of `cos(x)`.
        let function = custom_vjp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, ())),
            |(), cotangent: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(cotangent.clone() + cotangent),
        );
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(2.0))
            .value_and_gradient(|x| function.call(x).unwrap())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_batching_broadcasts_replicated_inputs() {
        // Mapping only the first input verifies that the replicated operand remains shared at the wrapper boundary
        // while operations inside its regions broadcast it only where per-item multiplication requires alignment.
        let function = custom_vjp(
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
            |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
        );
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(x, y)| function.call((x, y)),
                (Array::vector(vec![2.0, 3.0, 4.0]), Array::scalar(5.0)),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_custom_vjp_batching_preserves_residual_axes_and_sums_replicated_input_cotangents() {
        let scalar_type = test_type(&[]);
        let primal = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y]).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let forward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y]).unwrap()[0];
            builder.build(vec![output, x, y], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap()
        };
        let backward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let cotangent = builder.add_input(scalar_type.clone());
            let x_cotangent = builder.add_instruction(MulOperation::new(), Vec::new(), vec![y, cotangent]).unwrap()[0];
            let y_cotangent = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, cotangent]).unwrap()[0];
            builder.build(vec![x_cotangent, y_cotangent], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap()
        };
        let program = wrapped_call_program(
            vec![scalar_type.clone(), scalar_type],
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            vec![primal, forward, backward],
        );

        // `x` varies across the batch while `y` is shared. The forward residuals therefore carry axes `(0, None)`.
        // The backward rule naturally produces both cotangents mapped, but `y`'s cotangent must be summed back to the
        // replicated position rather than leaking a mapped axis through the wrapper contract.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        let instruction = &batched.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::CustomVjp(_)));
        let forward = batched.region_ref(instruction.regions()[1]).unwrap();
        assert_eq!(forward.output_types(), &[test_type(&[2]), test_type(&[2]), test_type(&[]),],);
        let backward = batched.region_ref(instruction.regions()[2]).unwrap();
        assert_eq!(backward.output_types(), &[test_type(&[2]), test_type(&[])]);
        assert!(
            backward
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::Reduce(_))),
            "the cotangent of the replicated input must be summed across the mapped axis",
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::scalar(5.0)]).unwrap(),
            vec![Array::vector(vec![10.0, 15.0])],
        );
    }
}
