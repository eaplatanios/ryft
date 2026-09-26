use std::fmt::Display;
use std::marker::PhantomData;

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentBatchingPolicy, DifferentiableOperation, DifferentiableType, DifferentiationContext,
    DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, check_types, impl_non_transposable_operation, impl_reference_dischargeable_operation,
};
use crate::operations::constants::zero::Zero;
use crate::operations::differentiation::custom_jvp::{
    validate_custom_derivative_reference_boundary, validate_custom_derivative_replay, validate_non_differentiated_count,
};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    InputRegionProvenance, Operation, OperationFormatter, OutputRegionProvenance, ProgramError, RegionInterface,
    RegionSlot, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, Trace};

/// Canonical operation name for [`CustomVjpOperation`].
pub const CUSTOM_VJP_OPERATION_NAME: &str = "custom_vjp";

/// Higher-order [`Operation`] that pairs a primal [`Program`](crate::programs::Program) with user-supplied forward and
/// backward (i.e., Vector-Jacobian Product or VJP) programs and that the [`custom_vjp`] function stages. Refer to the
/// documentation of that function for the semantics of custom VJPs, including their treatment of references, how each
/// transform handles a staged call, and when to reach for one.
///
/// The three programs are supplied as the operation's attached regions (i.e., via the
/// [`RegionDriver`](crate::RegionDriver) passed to [`Context::bind`]) in the region order `["primal", "forward",
/// "backward"]`. Writing the leading [`non_differentiated_count`](Self::non_differentiated_count) inputs as `p`,
/// the remaining _differentiated_ inputs as `x`, the primal outputs as `y`, the forward residuals as `r`, and
/// cotangents using an overbar, the region interfaces are:
///
///   - **Primal:**   `(p, x) → y`,
///   - **Forward:**  `(p, x) → (y, r)`, with arbitrarily many residuals following the primal outputs, and
///   - **Backward:** `(p, r, ȳ) → x̄`, with one cotangent per primal output and one cotangent per differentiated input.
///
/// [`Operation::infer_output_types`] validates that the attached regions realize exactly these interfaces, that only
/// `p` contains references, and that no output is a reference. A reference-typed residual must additionally have the
/// type of a distinct plumbing input in `p`, because the forward region may hand a plumbing reference to the backward
/// region only by forwarding that input by identity (which [`CustomVjp::call`] checks on the traced forward program).
/// Keeping `p` explicit while omitting its cotangent from the backward region distinguishes an input that parameterizes
/// the rules from an ordinary input whose cotangent merely evaluates to zero. This is the same input split that
/// [`LinearCallOperation`] draws with its residual count. Batching is a canonical producer of such inputs: a batching
/// policy that threads batching state through a structurally batched region's boundary (e.g., a composite universe's
/// first-class mapped extent) reintroduces that state as additional leading non-differentiated inputs of the batched
/// call.
///
/// The `T` parameter fixes the type universe of all attached regions and the call boundary, so each concrete payload
/// has exactly one [`Operation<Type = T>`](Operation) contract while the semantic and transform implementations remain
/// shared across differentiable type universes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomVjpOperation<T: DifferentiableType> {
    /// Number of leading inputs that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: DifferentiableType> CustomVjpOperation<T> {
    /// Creates a new [`CustomVjpOperation`] whose attached regions operate on `T` values and whose inputs are all
    /// differentiated.
    pub const fn new() -> Self {
        Self { non_differentiated_count: 0, marker: PhantomData }
    }

    /// Sets the number of leading inputs that parameterize this call without being differentiated. Refer to the
    /// documentation of [`CustomVjpOperation`] for the impact of this property on the interfaces of the attached
    /// regions.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Returns the number of leading inputs that parameterize this call without being differentiated.
    #[inline]
    pub fn non_differentiated_count(&self) -> usize {
        self.non_differentiated_count
    }

    /// Splits the provided input `values` into the leading non-differentiated group and the trailing differentiated
    /// group, based on the value of [`Self::non_differentiated_count`].
    #[inline]
    fn split_inputs<'v, V>(&self, values: &'v [V]) -> Result<(&'v [V], &'v [V]), TypeError> {
        validate_non_differentiated_count(self.name(), self.non_differentiated_count, values.len())?;
        Ok(values.split_at(self.non_differentiated_count))
    }

    /// Validates the custom Vector-Jacobian Product (VJP) contract over the three attached region interfaces (in the
    /// `["primal", "forward", "backward"]` region order) and returns the primal interface. Refer to the documentation
    /// of [`CustomVjpOperation`] for that contract.
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
        validate_custom_derivative_reference_boundary(
            CUSTOM_VJP_OPERATION_NAME,
            self.non_differentiated_count,
            input_types,
            output_types,
        )?;

        // A residual is an internal edge from the forward rule to the backward rule. A reference-typed residual can
        // only be a plumbing input forwarded by identity, because saving a snapshot of a reference is not a residual
        // the backward rule could mutate, so its type must be the type of one of the leading non-differentiated
        // inputs. Each such input can be forwarded at most once, because the backward rule's boundary rejects one
        // reference bound at two of its positions. The identity itself is a property of the forward program that its
        // tracing boundary checks.
        let mut forwarded = vec![false; non_differentiated_types.len()];
        for (index, residual_type) in residual_types.iter().enumerate().filter(|(_, r#type)| r#type.is_reference()) {
            let available = non_differentiated_types
                .iter()
                .zip(forwarded.iter_mut())
                .find(|(r#type, forwarded)| *r#type == residual_type && !**forwarded);
            match available {
                Some((_, forwarded)) => *forwarded = true,
                None if non_differentiated_types.contains(residual_type) => {
                    return Err(TypeError::invalid(format!(
                        "{CUSTOM_VJP_OPERATION_NAME} forward rule returns residual {index} of reference type \
                         `{residual_type}`, but every leading non-differentiated input of that type is already \
                         forwarded by an earlier residual",
                    )));
                }
                None => {
                    return Err(TypeError::invalid(format!(
                        "{CUSTOM_VJP_OPERATION_NAME} forward rule returns residual {index} of reference type \
                         `{residual_type}`, which matches none of its leading non-differentiated inputs",
                    )));
                }
            }
        }

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

impl<T: DifferentiableType> Copy for CustomVjpOperation<T> {}

impl<T: DifferentiableType> Default for CustomVjpOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DifferentiableType> Display for CustomVjpOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation for CustomVjpOperation<T> {
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
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        // The primal computation region receives every input at its own position. The forward and backward regions
        // are dormant rules that reference analysis does not enter, so they declare no provenance.
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        // Interpretation replays the primal region, whose outputs are the call's outputs one for one.
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // A call whose inputs are all differentiated renders as a bare name, so the non-differentiated split appears
        // in rendered programs exactly where it exists.
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_VJP_OPERATION_NAME)?;
        if self.non_differentiated_count == 0 {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("non_differentiated_count", self.non_differentiated_count))
    }
}

// Local reference lifecycles discharge inside each region while all user-declared numeric boundaries stay intact.
// External reference inputs still require explicit state threading that these derivative interfaces do not supply.
impl_reference_dischargeable_operation!(@local_reference <T> CustomVjpOperation<T> where T: DifferentiableType);

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomVjpOperation<C::Type> {
    #[inline]
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

// The default partial-evaluation rule is the desired one here where we interpret the primal region when every input
// is known and residualize the complete custom VJP call so its forward and backward regions remain attached for a
// later reverse-mode transformation.
impl<C: Context<Type: DifferentiableType, Operation: From<CustomVjpOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for CustomVjpOperation<C::Type>
{
}

impl<T: DifferentiableType, C: Context<Type = T, Operation: From<CustomVjpOperation<T>>>, P: CotangentBatchingPolicy<C>>
    BatchableOperation<C, P> for CustomVjpOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Batch all three region contracts while retaining the opaque custom VJP carrier:
        //
        //   - Primal:   (p, x)    → y
        //   - Forward:  (p, x)    → (y_fwd, r)
        //   - Backward: (p, r, ȳ) → x̄.
        //
        // Reconcile each `y` with its corresponding `y_fwd` so the wrapper exposes one physical output axis, but
        // keep every residual's naturally produced axis because residuals are internal edges between the forward
        // and backward rules. Batch the backward region with `(p, r, ȳ)` on those exact axes and align each `x̄`
        // with its corresponding differentiated input `x`. When a replicated `x` receives a mapped cotangent, the
        // batching policy sums that mapped axis, which is the transpose of broadcasting `x` across the batch.
        //
        // A batching policy may add runtime boundary inputs such as a first-class mapped extent. Those
        // values must reach all three regions but have no cotangent, so prepend them to `p` and increase
        // `non_differentiated_count` after the regions have been adapted to their new boundaries.
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (non_differentiated_axes, differentiated_axes) = self.split_inputs(input_axes.as_slice())?;
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
            ProgramBatchingOutputAxesPolicy::AlignEachTo(differentiated_axes.to_vec()),
        )?;
        let (backward, backward_output_axes) =
            P::adapt_batched_program(batched_backward, Some(differentiated_axes), P::sum_mapped_cotangents)?
                .into_parts();
        if backward_output_axes.as_slice() != differentiated_axes {
            return Err(BatchingError::MisalignedBatchAxes {
                message: format!(
                    "batched {CUSTOM_VJP_OPERATION_NAME} backward output axes {backward_output_axes:?} do not match \
                     its differentiated input axes {differentiated_axes:?}",
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

impl<
    C: Context<
            Type: DifferentiableType,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation> + From<LinearCallOperation<C::Type>>,
        > + Zero<C::Value>,
> DifferentiableOperation<C> for CustomVjpOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A custom VJP specifies the pullback `ȳ ↦ x̄`, not the pushforward `ẋ ↦ ẏ`. Replay:
        //
        //   forward(p, x) = (y, r)
        //
        // to recover the primal outputs and residuals, then stage an opaque linear call representing the unknown map:
        //
        //   L_(p,r): ẋ ↦ ẏ.
        //
        // `LinearCallOperation` knows only how to transpose that map: its transpose replays `backward(p, r, ȳ) = x̄`.
        // An eager forward-mode use attempts to execute `L_(p,r)` and is therefore rejected, while reverse mode
        // transposes it without execution. Passing `p` and `r` as the carrier's leading residual inputs keeps the path
        // capture-free and exposes every dependency as an ordinary Single Static Assignment (SSA) edge.
        //
        // The attached regions are `["primal", "forward", "backward"]` and the primal interface provides
        // the boundary types.
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let backward_region = driver.region(2)?;

        let primal_output_types = primal_region.output_types();
        let output_count = primal_output_types.len();
        let (non_differentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;

        // The forward and backward rule regions bypass ordinary differentiation dispatch: the forward region is
        // replayed directly and the backward region is retained for later transposition, so the replayed inputs are
        // validated as defense in depth (refer to the documentation of `validate_custom_derivative_replay`).
        validate_custom_derivative_replay(
            CUSTOM_VJP_OPERATION_NAME,
            self.non_differentiated_count,
            context.primal(),
            inputs,
            primal_output_types.as_slice(),
        )?;
        check_count!("input", inputs, primal_region.input_types().len(), ProgramError);

        // Replay the forward region on the dual primals, recovering the primal outputs followed by the residuals.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = forward_region.interpret_in_context(context.primal(), primal_inputs)?;
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

        let input_tangent_types = differentiated_inputs
            .iter()
            .map(|input| input.primal().r#type().tangent())
            .collect::<Result<Vec<_>, _>>()?;
        let output_tangent_types =
            primal_outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?;

        // Stage one opaque carrier over `[non_differentiated..., residuals..., differentiated_input_tangents...]`,
        // producing the output tangents. The carrier rejects forward interpretation and transposes by replaying the
        // user's backward region, whose own inputs are exactly that leading residual group followed by the output
        // cotangents. The transpose-only carrier takes every differentiated input tangent as a real input, so
        // materialize structural zeros against their own primal, which names every runtime quantity a
        // reference-bearing tangent type omits.
        let mut carrier_inputs =
            non_differentiated_inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        carrier_inputs.extend(residuals);
        let mut carrier_inputs = carrier_inputs
            .into_iter()
            .map(|value| context.primal_to_tangent(value))
            .collect::<Result<Vec<_>, _>>()?;

        // The carrier's leading non-tangent group is the non-differentiated inputs followed by the residuals, and both
        // are passed through the residual-count slot: to the linear call they are alike inputs that its transpose
        // forwards to the backward region rather than transposing.
        let leading_input_count = carrier_inputs.len();
        for input in differentiated_inputs {
            let source = context.primal_to_tangent(input.primal().clone())?;
            carrier_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                context.tangent(),
                input.tangent().clone(),
                std::iter::once(&source),
            )?);
        }
        let carrier =
            LinearCallOperation::transpose_only(leading_input_count, input_tangent_types, output_tangent_types);

        // Any context that must _execute_ the carrier (i.e., an eager forward-mode pass or a forward-mode pass over an
        // already staged carrier) rejects it as unsupported. Restate that rejection in `custom_vjp` vocabulary instead
        // of leaking the internals of the carrier.
        let output_tangents = context
            .tangent()
            .bind(carrier, vec![backward_region.to_program()], &carrier_inputs)
            .map_err(|error| match error {
                ProgramError::UnsupportedOperation { .. } => ProgramError::UnsupportedOperation {
                    message: format!(
                        "cannot apply forward-mode differentiation to a {CUSTOM_VJP_OPERATION_NAME} call; it supports \
                         only reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
                    ),
                },
                error => error,
            })?;
        check_count!("output", output_tangents, output_count, ProgramError);

        primal_outputs
            .into_iter()
            .zip(output_tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()
    }
}

// The raw carrier is intentionally non-transposable, which does not restrict reverse-mode differentiation. Reverse
// mode linearizes first, and the JVP rule replaces `f(p, x)` with the opaque linear map `L_(p,r): ẋ ↦ ẏ` (i.e., the
// analogue of JAX's `custom_lin` primitive), so reverse mode transposes that `LinearCallOperation`, whose rule
// evaluates `backward(p, r, ȳ) = x̄`. Therefore, only an invalid direct transpose of an un-linearized custom VJP call
// can reach this rejection path.
impl_non_transposable_operation!(<T> CustomVjpOperation<T> where T: DifferentiableType);

// TODO(eaplatanios): Review from here onwards.

/// Function with user-supplied forward and backward (i.e., VJP) rules, built by [`custom_vjp`]. It stores the primal,
/// forward, and backward closures together with a phantom marker pinning the tracer-tree types named by those closure
/// signatures. Refer to the documentation of the [`custom_vjp`] function for the calling convention, the tracing
/// semantics, and when to reach for a custom VJP.
pub struct CustomVjp<Input, Output, Residual, Primal, Forward, Backward> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: Primal,

    /// Closure computing `(outputs, residuals)` from the primal input tree.
    forward: Forward,

    /// Closure computing the input cotangent tree from `(residuals, output_cotangents)`.
    backward: Backward,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Phantom marker pinning the input, output, and residual tracer-tree types named by the closure signatures. The
    /// [`Context`] whose universe the rules are traced into is recovered from the values passed to
    /// [`CustomVjp::call`], and so the wrapper stores neither a context value nor a context type witness.
    marker: PhantomData<fn() -> (Input, Output, Residual)>,
}

impl<
    Input,
    Output,
    Residual,
    Primal: Fn(Input) -> Result<Output, ProgramError>,
    Forward: Fn(Input) -> Result<(Output, Residual), ProgramError>,
    Backward: Fn(Residual, Output) -> Result<Input, ProgramError>,
> CustomVjp<Input, Output, Residual, Primal, Forward, Backward>
{
    /// Declares the leading `non_differentiated_count` flattened leaves of the input tree as non-differentiated
    /// _plumbing_ inputs, which is the high-level counterpart of [`CustomVjpOperation::with_non_differentiated_count`].
    /// Refer to the documentation of the [`custom_vjp`] function for the semantics of non-differentiated inputs.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Stages this custom-VJP function on the provided tracer `input` tree and returns its output tree. Refer to the
    /// documentation of the [`custom_vjp`] function for the tracing semantics and for how the transforms treat the
    /// staged call.
    ///
    /// The [`Context`] `C` whose universe the three closures are traced into is the
    /// [`DispatchDomain`](Value::DispatchDomain) of the values in `input`, which is exactly the context the call is
    /// staged into. It is therefore never named at a construction or call site, while the stored closures still pin
    /// the tracer trees that this universe must produce.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves, when the non-differentiated count exceeds the number of
    /// input leaves, when tracing any of the closures fails, when the forward closure returns a reference-typed
    /// residual that is not a non-differentiated input forwarded by identity, or when the staged [`CustomVjpOperation`]
    /// rejects the traced programs (e.g., because the rule signatures do not match the primal signature or because the
    /// call violates the reference contract).
    pub fn call<C, V, InputValues>(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        C: Context<Type: DifferentiableType, Value = V>,
        V: Value<Type = C::Type, DispatchDomain = C>,
        C::Operation: From<CustomVjpOperation<C::Type>>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Residual: Parameterized<DomainTracer<C>>,
        Residual::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output::To<C::Type>: Clone + Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
        Residual::To<C::Type>: Parameterized<C::Type, Family = Residual::Family, To<DomainTracer<C>> = Residual>,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
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
        let non_differentiated_count = self.non_differentiated_count;
        validate_non_differentiated_count(CUSTOM_VJP_OPERATION_NAME, non_differentiated_count, input_values.len())?;
        let (output_types, primal) = C::trace(&self.primal, input_types.clone())?;
        let ((_, residual_types), forward) = C::trace(&self.forward, input_types.clone())?;
        // A reference-typed residual must be an input forwarded by identity rather than a computed value: a reference
        // allocated by the forward rule would reach the backward rule as a residual whose mutation nothing outside the
        // rule observes, so the traced forward program is checked here, where its atoms are visible, while the staged
        // operation checks the residual's type and that the forwarded input is a leading non-differentiated one.
        let output_count = output_types.parameters().count();
        if let Some((index, residual)) =
            forward.output_ids().iter().skip(output_count).enumerate().find(|(_, residual)| {
                forward.atoms()[residual.index()].r#type().is_reference() && !forward.input_ids().contains(residual)
            })
        {
            return Err(TypeError::invalid(format!(
                "{} forward rule returns residual {} of reference type `{}` that is not a leading non-differentiated \
                 input forwarded by identity",
                CUSTOM_VJP_OPERATION_NAME,
                index,
                forward.atoms()[residual.index()].r#type(),
            ))
            .into());
        }
        let output_cotangent_types = output_types.clone().try_map_parameters(|r#type| r#type.cotangent())?;
        // The backward region consumes `[non_differentiated..., residuals..., output_cotangents...]`. The closure
        // sees only the residuals and cotangents, so the leading non-differentiated inputs are declared as unused
        // region inputs (a plumbing value the backward rule needs is forwarded to it as a residual), and the
        // input-shaped cotangent tree the closure returns loses its leading non-differentiated leaves so that the
        // staged rule produces exactly one cotangent per differentiated input.
        let non_differentiated_types =
            input_types.parameters().take(non_differentiated_count).cloned().collect::<Vec<_>>();
        let (_, backward) = C::trace(
            |(_, residuals, cotangents): (Vec<DomainTracer<C>>, Residual, Output)| {
                let cotangents = (self.backward)(residuals, cotangents)?;
                Ok(cotangents.into_parameters().skip(non_differentiated_count).collect::<Vec<_>>())
            },
            (non_differentiated_types, residual_types, output_cotangent_types),
        )?;
        let operation =
            C::Operation::from(CustomVjpOperation::new().with_non_differentiated_count(non_differentiated_count));
        // The call binds through whatever context the input values flow through (e.g., a staging trace, a batching
        // context, or a differentiation context), so the batching or differentiation rule of the bound operation fires
        // and `custom_vjp` composes with those transforms.
        let context = first.dispatch_domain();
        let outputs = context.bind(
            operation,
            vec![primal.into_flat_program(), forward.into_flat_program(), backward.into_flat_program()],
            &input_values,
        )?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over trees of [`DomainTracer`]s. This
/// is the analogue of JAX's [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair.
///
/// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The Vector-Jacobian Product (VJP), or
/// pullback, maps an output cotangent `ȳ` to the input cotangent `x̄ = J_f(x)ᵀ · ȳ`. The three closure arguments
/// factor that computation through a residual tree `r`:
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
/// Reach for a custom VJP when only the _reverse_ rule is natural, or when the function is not (efficiently)
/// forward-differentiable. Common cases are:
///
///   - **Implicit differentiation:** differentiate through a solver, optimizer, or fixed point via the implicit
///     function theorem rather than unrolling its iterations.
///   - **Adjoint methods:** backpropagate through an ODE or PDE solve via the adjoint system instead of
///     differentiating the individual steps of the integrator.
///   - **External or black-box calls:** supply the reverse rule for a custom kernel or for a computation that does not
///     itself trace into Ryft programs.
///   - **Numerical stability:** replace an unstable or wasteful automatically derived gradient with a hand-written
///     one.
///
/// A custom VJP is reverse-mode only: forward-mode differentiation of a staged call is rejected, and the current
/// transpose implementation also rejects transposing its generated pullback, so higher-order derivatives through a
/// custom VJP are not yet supported. When the function is forward-differentiable or must participate in higher-order
/// differentiation, use [`custom_jvp`](fn@crate::operations::differentiation::custom_jvp) instead.
///
/// # Calling convention
///
/// All three closures operate on [`Parameterized`] trees of [`DomainTracer`]s (i.e., Ryft's analogue of JAX
/// pytrees), so `x`, `y`, and `r` may each be a single tracer, a tuple, or any other parameterized structure. Static
/// non-differentiated configuration should be captured by all three closures. A dynamic value should remain an
/// explicit input, either as a non-differentiated input (see below) or as an ordinary input that `forward` preserves
/// as a residual in `r` when `backward` needs it and for which `backward` returns a zero cotangent in `x̄`.
///
/// Because [`custom_vjp`] builds a reusable function before any input is known, the `primal` closure must annotate
/// the tracer type of its input (e.g., `|x: DomainTracer<C>| ...`), which then also fixes the input types of
/// `forward` and, through the outputs and residuals that `forward` returns, those of `backward`.
/// [`custom_derivative_at`](crate::custom_derivative_at) instead stages the same rule at a known input, which lets all
/// three closures infer their parameter types from that input.
///
/// # Non-differentiated inputs
///
/// [`CustomVjp::with_non_differentiated_count`] declares the leading flattened input leaves as _plumbing_ that
/// parameterizes the call without being differentiated, which is the analogue of JAX's `nondiff_argnums`. Plumbing
/// leaves reach `primal` and `forward` at their usual positions and receive no cotangent: `backward` keeps returning a
/// full `x̄` tree so that its signature mirrors the primal signature, but the leaves that it returns at plumbing
/// positions are ignored and never become outputs of the staged [`CustomVjpOperation`]. A plumbing value that
/// `backward` needs must be forwarded to it as a residual in `r`. Differentiating the call with a nonzero tangent for
/// a numeric plumbing input is rejected because the rules cannot propagate that tangent.
///
/// # References
///
/// A reference-typed input is accepted only as plumbing, where every closure that receives it may read or write it. An
/// active reference input is rejected, because the rule interfaces define no tangent or cotangent reference for it and
/// so user-supplied rules could not express its derivative. A live tangent reference supplied for a plumbing input by
/// an enclosing transform is left untouched by the call, since the rules declare no derivative through the state it
/// denotes. No output of `primal` may be a reference, because `backward` would then have to consume that output's
/// cotangent reference.
///
/// `forward` may return a plumbing reference inside `r`, in which case the reference itself (rather than a snapshot of
/// its contents) is forwarded to `backward`. Every reference-typed residual must be a distinct plumbing input
/// forwarded by identity, since a reference allocated by `forward` would reach `backward` as state whose mutation
/// nothing outside of the rules observes. This enables the _stash-gradients_ pattern: a `stash` reference enters as
/// plumbing, `forward` returns it as a residual, and `backward` writes the incoming `ȳ` into it before returning `x̄`.
/// The closures may also allocate and use local reference state, which executes like any other primitive operation
/// whenever the corresponding program is replayed. When the call is differentiated, no two reference inputs may bind
/// the same allocation.
///
/// # Tracing semantics
///
/// Nothing is traced at construction time. Each [`CustomVjp::call`] recovers the tracing [`Context`] from the values it
/// is called with, reads the input types off those values, traces the closures into programs specialized to those
/// types, validates the rule signatures, and stages one [`CustomVjpOperation`] into the context through which those
/// values flow. The primal closure is kept separate from the forward closure for efficiency rather than necessity: an
/// un-differentiated call should not pay for residual computation. Callers that do not care about the distinction can
/// pass the same body for both, accepting that the residual outputs are dead code outside of differentiation (e.g.,
/// by writing `forward` as `|x| Ok((f(x)?, residuals))`).
///
/// # Transform semantics
///
/// The transforms treat a staged call as follows:
///
///   - _interpretation_ and backend lowering replay the lean primal program only,
///   - _partial evaluation_ folds a call whose inputs are all known and otherwise residualizes it unchanged, so that
///     the rules stay attached for a later differentiation,
///   - _batching_ preserves the call around axis-reconciled batched copies of all three programs, so that the custom
///     derivative survives batching applied _before_ differentiation, and it sums the cotangents that the batched
///     backward program produces for replicated inputs over the batch axis, and
///   - _differentiation_ replays the forward program for the primal outputs and residuals and stages a transpose-only
///     linear call for the output tangents whose transpose replays the backward program, so reverse mode uses exactly
///     the user-supplied gradient. Because that linear call cannot be executed, forward-mode differentiation of a
///     staged call is rejected, and the staged call itself is never transposed.
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y` for ordinary evaluation.
///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse-mode residual production.
///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
pub fn custom_vjp<Input, Output, Residual, Primal, Forward, Backward>(
    primal: Primal,
    forward: Forward,
    backward: Backward,
) -> CustomVjp<Input, Output, Residual, Primal, Forward, Backward>
where
    Primal: Fn(Input) -> Result<Output, ProgramError>,
    Forward: Fn(Input) -> Result<(Output, Residual), ProgramError>,
    Backward: Fn(Residual, Output) -> Result<Input, ProgramError>,
{
    CustomVjp { primal, forward, backward, non_differentiated_count: 0, marker: PhantomData }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        ShardingDimension,
    };
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{CotangentDestination, CotangentSeed, differentiate_at};
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::differentiation::custom_jvp::tests::nested_custom_derivative_state_program;
    use crate::operations::differentiation::tests::{
        ReferenceRuleDifferentiationDriver, custom_derivative_call_program,
    };
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::operations::references::{
        ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead, ReferenceWrite,
    };
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        EffectClass, EffectClasses, FlatProgram, ProgramBuilder, ReferenceType, RegionRole, ValueProjection,
    };

    use super::*;

    /// Eager context whose values are arrays.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager composite context whose values may be arrays or references.
    type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer of the composite universe used by the plumbing-reference tests.
    type ArrayIrTracer = DomainTracer<ArrayIrContext>;

    /// Error message of the forward-mode rejection of staged custom-VJP calls.
    const FORWARD_MODE_REJECTION: &str = "cannot apply forward-mode differentiation to a custom_vjp call; it supports \
                                          only reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or \
                                          `jacobian_reverse`)";

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let residual = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder; 2]).unwrap()
    }

    /// Builds the deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent`, which is
    /// detectably different from the true gradient so that tests can prove that the custom rule is used.
    fn tripled_sin_backward_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(Array::scalar(3.0).unwrap());
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![three, residual], None).unwrap()[0];
        let gradient =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, cotangent], None).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    /// Returns a custom-VJP call over `f(x) = sin(x)` together with its `["primal", "forward", "backward"]` regions,
    /// whose backward rule deliberately triples the true gradient.
    fn custom_vjp_sin(r#type: &ArrayType) -> (ArrayOperation<Array>, Vec<FlatProgram<ArrayContext>>) {
        (
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            vec![sin_program(r#type), sin_forward_program(r#type), tripled_sin_backward_program(r#type)],
        )
    }

    #[test]
    fn test_custom_vjp() {
        let operation = CustomVjpOperation::<ArrayType>::new();
        assert_eq!(operation, CustomVjpOperation::default());
        assert_eq!(operation.name(), CUSTOM_VJP_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(format!("{operation}"), "custom_vjp");

        // The primal program is a computation region followed by the dormant forward and backward rule regions.
        assert_eq!(
            operation.region_slots(),
            &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")],
        );
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));
        assert_eq!(operation.region_role(2), Some(RegionRole::Rule));

        // The primal region receives every input at its own position and its outputs are the call's outputs, while
        // the dormant rule regions declare no input provenance.
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(operation.input_region_provenance(2, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
        );
    }

    #[test]
    fn test_custom_vjp_with_non_differentiated_count() {
        let operation = CustomVjpOperation::<ArrayType>::new().with_non_differentiated_count(1);
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "custom_vjp [non_differentiated_count=1]");

        // The primal and forward regions receive the non-differentiated input at its own position, and the backward
        // region receives it ahead of the residuals but produces no cotangent for it.
        let parameter_type = ArrayType::new_static(DataType::F64, [2]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input_types = vec![parameter_type.clone(), scalar_type.clone()];
        let backward_input_types = vec![parameter_type, scalar_type.clone(), scalar_type.clone()];
        let interfaces = [
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE),
            RegionInterface::new(
                input_types.clone(),
                vec![scalar_type.clone(), scalar_type.clone()],
                EffectClasses::NONE,
            ),
            RegionInterface::new(backward_input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE),
        ];
        assert_eq!(
            operation.infer_region_input_types(&input_types, &interfaces),
            Ok(vec![Some(input_types.clone()), Some(input_types.clone()), Some(backward_input_types)]),
        );
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Ok(vec![scalar_type]));
    }

    #[test]
    fn test_custom_vjp_type_inference() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = CustomVjpOperation::<ArrayType>::new();
        let primal_interface = sin_program(&scalar_type).interface();
        let forward_interface = sin_forward_program(&scalar_type).interface();
        let backward_interface = tripled_sin_backward_program(&scalar_type).interface();

        // The primal and forward regions receive the call inputs, the backward region receives the trailing forward
        // residuals followed by one cotangent per primal output, and rules that satisfy the interface contract make the
        // call produce the primal outputs.
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Ok(vec![
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone(), scalar_type.clone()]),
            ]),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Ok(vec![scalar_type.clone()]),
        );

        // Inference maps the primal boundary through _differential_ types, so the cotangent boundary of an
        // `f8e8m0fnu` primal is `f32` while the residual keeps its own storage type.
        let primal_type = ArrayType::scalar(DataType::F8E8M0FNU);
        let cotangent_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[
                    RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], EffectClasses::NONE),
                    RegionInterface::new(
                        vec![primal_type.clone()],
                        vec![primal_type.clone(), scalar_type.clone()],
                        EffectClasses::NONE,
                    ),
                    RegionInterface::new(
                        vec![scalar_type.clone(), cotangent_type.clone()],
                        vec![cotangent_type],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Ok(vec![primal_type]),
        );

        // The forward interface must be `inputs... → (outputs..., residuals...)`.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(vec![vector_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE),
                    backward_interface.clone(),
                ],
            ),
            Err(TypeError::invalid(
                "custom_vjp forward input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(vec![scalar_type.clone()], Vec::new(), EffectClasses::NONE),
                    backward_interface.clone(),
                ],
            ),
            Err(TypeError::invalid(
                "custom_vjp forward must produce at least the 1 primal output(s) but produced 0 value(s)".to_string(),
            )),
        );

        // The backward interface must be `(residuals..., output_cotangents...) → input_cotangents...`, so a
        // primal-shaped rule signature is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), primal_interface.clone()],
            ),
            Err(TypeError::invalid(
                "custom_vjp backward input type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
                    .to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    forward_interface.clone(),
                    RegionInterface::new(
                        vec![scalar_type.clone(), scalar_type.clone()],
                        Vec::new(),
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Err(TypeError::invalid(
                "custom_vjp backward output type signature mismatch: expected [f64[]] but got []".to_string(),
            )),
        );

        // The call inputs must match the primal region inputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&vector_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Err(TypeError::invalid(
                "custom_vjp input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // The call carries exactly three regions and at most as many non-differentiated inputs as it has inputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), std::slice::from_ref(&primal_interface)),
            Err(TypeError::invalid("expected 3 regions but got 1".to_string())),
        );
        assert_eq!(
            operation.with_non_differentiated_count(2).infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface, forward_interface, backward_interface],
            ),
            Err(TypeError::invalid("custom_vjp non-differentiated input count 2 exceeds input count 1".to_string())),
        );
    }

    #[test]
    fn test_custom_vjp_type_inference_references() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input_types = vec![reference_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);
        let forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), reference_type.clone()],
            EffectClasses::NONE,
        );
        let backward_interface = RegionInterface::new(
            vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        let interfaces = [primal_interface.clone(), forward_interface.clone(), backward_interface.clone()];

        // The stash-gradients boundary: a leading plumbing reference that the forward rule hands to the backward rule
        // as a residual of the same reference type is accepted.
        let operation = CustomVjpOperation::<ArrayIrType>::new().with_non_differentiated_count(1);
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Ok(vec![scalar_type.clone()]));
        assert_eq!(
            operation.infer_region_input_types(&input_types, &interfaces),
            Ok(vec![
                Some(input_types.clone()),
                Some(input_types.clone()),
                Some(vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()]),
            ]),
        );

        // A reference-typed residual that matches no plumbing input cannot be a forwarded plumbing reference.
        let other_reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let allocating_forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), other_reference_type.clone()],
            EffectClasses::NONE,
        );
        let allocating_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), other_reference_type, scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[primal_interface.clone(), allocating_forward_interface, allocating_backward_interface],
            ),
            Err(TypeError::invalid(
                "custom_vjp forward rule returns residual 0 of reference type `ref<f64[]>`, which matches none of its \
                 leading non-differentiated inputs"
                    .to_string(),
            )),
        );

        // One plumbing input can be forwarded at most once: a second residual of the same reference type cannot be a
        // forwarded plumbing input (the backward rule's boundary would bind one reference at two positions), so it is
        // rejected even though its type matches a leading non-differentiated input.
        let duplicating_forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), reference_type.clone(), reference_type.clone()],
            EffectClasses::NONE,
        );
        let duplicating_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), reference_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[primal_interface.clone(), duplicating_forward_interface, duplicating_backward_interface],
            ),
            Err(TypeError::invalid(
                "custom_vjp forward rule returns residual 1 of reference type `ref<f32[]>`, but every leading \
                 non-differentiated input of that type is already forwarded by an earlier residual"
                    .to_string(),
            )),
        );

        // A reference input in the differentiated segment would need tangent and cotangent references that the rules
        // cannot define, so it is rejected even when the rule interfaces declare them.
        let active_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone()],
            vec![reference_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            CustomVjpOperation::<ArrayIrType>::new()
                .infer_output_types(&input_types, &[primal_interface, forward_interface, active_backward_interface]),
            Err(TypeError::invalid(
                "custom_vjp accepts reference inputs only in its leading non-differentiated segment; move input 0 of \
                 type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );

        // No output may be a reference, not even a forwarded plumbing input, because the backward rule would then have
        // to consume its cotangent reference.
        let forwarding_primal_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_forward_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_backward_interface =
            RegionInterface::new(vec![reference_type.clone(), reference_type], vec![scalar_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[forwarding_primal_interface, forwarding_forward_interface, forwarding_backward_interface],
            ),
            Err(TypeError::invalid(
                "custom_vjp cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_vjp_reference_discharge() {
        // The primal is the identity, but the backward rule deliberately returns three times its cotangent. A local
        // lifecycle inside that dormant rule must disappear during discharge without replacing the custom derivative.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let identity = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let cotangent = backward.add_input(scalar_type.clone());
        let three = backward.add_constant(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        let scaled = backward
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![three, cotangent], None)
            .unwrap()[0];
        let reference =
            backward.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![scaled], None).unwrap()[0];
        let output = backward
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let backward = backward
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = custom_derivative_call_program(
            CustomVjpOperation::new(),
            vec![identity.clone(), identity, backward],
            vec![scalar_type],
        )
        .discharge_references(0)
        .unwrap()
        .into_program_without_external_references()
        .unwrap();
        assert_eq!(program.instructions()[0].regions().len(), 3);
        assert!(!program.entry_region_ref().contains_references_in_closure());
        let input = ArrayIrValue::Array(Array::scalar(5.0f32).unwrap());
        assert_eq!(program.interpret(vec![input.clone()]), Ok(vec![input.clone()]));
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())];
        cotangents.extend(primal_outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_custom_vjp_reference_discharge_rejects_external_references() {
        // A custom-VJP call threads a plumbing reference into its dormant forward and backward rules, whose
        // reference-typed inputs are bound by the transform that instantiates them and therefore declare no input
        // provenance. Summarizing a condition branch containing such a call skips those rules exactly as the reference
        // analysis does, so discharging the program reaches the call's own discharge rule, which reports that a
        // caller reference cannot cross the custom-VJP boundary, instead of failing on undeclared provenance.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let identity = |input_types: Vec<ArrayIrType>, output_positions: Vec<usize>| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            let outputs = output_positions.iter().map(|position| inputs[*position]).collect::<Vec<_>>();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    outputs,
                    vec![Placeholder; input_types.len()],
                    vec![Placeholder; output_positions.len()],
                )
                .unwrap()
        };
        let branch = custom_derivative_call_program(
            CustomVjpOperation::<ArrayIrType>::new().with_non_differentiated_count(1),
            vec![
                identity(vec![reference_type.clone(), scalar_type.clone()], vec![1]),
                identity(vec![reference_type.clone(), scalar_type.clone()], vec![1, 0]),
                identity(vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()], vec![2]),
            ],
            vec![reference_type.clone(), scalar_type.clone()],
        );
        let program = custom_derivative_call_program(
            ConditionOperation::new(),
            vec![branch.clone(), branch],
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)), reference_type, scalar_type],
        );
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`custom_vjp` does not thread external references through discharge, but input 0 is a \
                    reference; pass reference-free inputs or discharge external references first",
        ));
    }

    #[test]
    fn test_custom_vjp_interpretation() {
        // Interpretation replays the primal region only, so an un-differentiated call produces just the primal output
        // and never pays for the residual computation of the forward region.
        let (operation, regions) = custom_vjp_sin(&ArrayType::scalar(DataType::F64));
        assert_eq!(
            ArrayContext::new().bind(operation, regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_custom_vjp_partial_evaluation() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (operation, regions) = custom_vjp_sin(&scalar_type);
        let program = custom_derivative_call_program(operation, regions, vec![scalar_type.clone()]);

        // A call whose inputs are all known folds by interpreting its primal region.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(Array::scalar(2.0).unwrap())]).unwrap();
        assert!(matches!(
            &evaluation.outputs[0],
            PartialEvaluationOutput::Known(output) if output == &Array::scalar(2.0f64.sin()).unwrap(),
        ));

        // A call with an unknown input residualizes unchanged instead of inlining its primal region, which keeps the
        // custom rules attached to the residual program.
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_type)]).unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomVjp(_)));
    }

    #[test]
    fn test_custom_vjp_batching() {
        let output: Array = batch(
            |x| {
                let (operation, regions) = custom_vjp_sin(&ArrayType::scalar(DataType::F64));
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            },
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![0.5f64.sin(), 1.0f64.sin(), 1.5f64.sin()]).unwrap());
    }

    #[test]
    fn test_custom_vjp_batching_residual_axes_and_replicated_cotangents() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let primal = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y], None).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let forward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y], None).unwrap()[0];
            builder.build(vec![output, x, y], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap()
        };
        let backward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let cotangent = builder.add_input(scalar_type.clone());
            let x_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![y, cotangent], None).unwrap()[0];
            let y_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, cotangent], None).unwrap()[0];
            builder.build(vec![x_cotangent, y_cotangent], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap()
        };
        let program = custom_derivative_call_program(
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            vec![primal, forward, backward],
            vec![scalar_type.clone(), scalar_type.clone()],
        );

        // `x` varies across the batch while `y` is shared, so the forward residuals carry axes `(0, None)`. The
        // backward rule naturally produces both cotangents mapped, but the cotangent of `y` must be summed back to the
        // replicated position rather than leaking a mapped axis through the call boundary.
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
        assert_eq!(forward.output_types(), &[vector_type.clone(), vector_type.clone(), scalar_type.clone()]);
        let backward = batched.region_ref(instruction.regions()[2]).unwrap();
        assert_eq!(backward.output_types(), &[vector_type, scalar_type]);
        assert!(
            backward
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::Reduce(_))),
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![2.0, 3.0]).unwrap(), Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::vector(vec![10.0, 15.0]).unwrap()]),
        );
    }

    #[test]
    fn test_custom_vjp_batching_preserves_custom_derivative() {
        // Differentiating through a batched custom call must still use the deliberately tripled custom backward rule,
        // because batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let (operation, regions) = custom_vjp_sin(&ArrayType::scalar(DataType::F64));
                        Ok(item.context().bind(operation, regions, &[item.clone()])?.remove(0))
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin() + 1.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::vector(vec![3.0 * 0.5f64.cos(), 3.0 * 1.0f64.cos()]).unwrap());
    }

    #[test]
    fn test_custom_vjp_differentiation() {
        // The custom backward rule triples the true gradient, which proves that it governs reverse-mode
        // differentiation.
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(|x| {
                let (operation, regions) = custom_vjp_sin(&ArrayType::scalar(DataType::F64));
                x.context().bind(operation, regions, &[x.clone()]).unwrap().remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(2.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(3.0 * 2.0f64.cos()).unwrap());
    }

    #[test]
    fn test_custom_vjp_differentiation_rejects_forward_mode() {
        // A custom VJP supplies no tangent program, so the tangent carrier that its JVP rule stages cannot be executed.
        // Forward mode must therefore fail with a user-facing custom-VJP error rather than leaking the internal
        // vocabulary of the carrier.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let (operation, regions) = custom_vjp_sin(&ArrayType::scalar(DataType::F64));
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == FORWARD_MODE_REJECTION,
        ));
    }

    #[test]
    fn test_custom_vjp_differentiation_nested_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();

        // A forward rule may allocate and use local reference state inside a dormant nested rule: it is replayed
        // directly (the driver makes recursive differentiation an assertion failure), so forward mode reaches the
        // transpose-only carrier and fails with the custom-VJP forward-mode rejection rather than a state rejection.
        let identity = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let forward = nested_custom_derivative_state_program(&scalar_type, false);
        assert!(forward.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let driver = ReferenceRuleDifferentiationDriver { programs: vec![identity.clone(), forward, identity] };
        assert!(matches!(
            CustomVjpOperation::<ArrayIrType>::new().jvp(
                &DifferentiationContext::fused(ArrayIrContext::new()),
                &driver,
                std::slice::from_ref(&input),
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == FORWARD_MODE_REJECTION,
        ));
    }

    #[test]
    fn test_custom_vjp_differentiation_reverse_jacobian() {
        // `jacobian_reverse` interprets the pullback with batch-stacked cotangent bases, which exercises the batched
        // replay of the custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the
        // diagonal matrix `diag(3 * cos(x))`.
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let jacobian = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .jacobian_reverse(|x| {
                let (operation, regions) = custom_vjp_sin(&vector_type);
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![3.0 * 0.5f64.cos(), 0.0, 0.0, 3.0 * 1.0f64.cos()]);
    }

    #[test]
    fn test_custom_vjp_transposition() {
        // Differentiation replaces the call with a transpose-only linear call before transposition, so only a direct
        // transpose of an un-linearized call reaches the operation, which rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (operation, regions) = custom_vjp_sin(&scalar_type);
        let program = custom_derivative_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_vjp` is not transposable",
        ));
    }

    #[test]
    fn test_custom_vjp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The deliberately wrong
        // rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the true gradient (expressed through
        // addition to avoid constant lifting), which proves that the rule is in control.
        let function = custom_vjp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(3.0 * 2.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_vjp_call_with_non_differentiated_count() {
        // The stash-gradients pattern: the leading `stash` reference is plumbing, the forward rule forwards it as a
        // residual by identity rather than saving a snapshot, and the backward rule writes the incoming cotangent into
        // it before returning `x̄ = cos(x) · ȳ`. The cotangent tree that it returns is input-shaped, so its leading
        // plumbing leaf is ignored.
        let function = custom_vjp(
            |(_, x): (ArrayIrTracer, ArrayIrTracer)| {
                Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
            },
            |(stash, x)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (stash, x.cos()?.into_value())))
            },
            |(stash, cosine), cotangent| {
                stash.write(&cotangent)?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((stash, (cosine * cotangent).into_value()))
            },
        )
        .with_non_differentiated_count(1);
        let stash = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let (value, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
        ))
        .vjp(|(stash, x)| function.call((stash, x)))
        .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(0.5f32.sin()).unwrap()));

        // Linearization replays only the forward rule, which does not touch the stash.
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // The stash is a plumbing input, so its own cotangent is ignored, while applying the pullback replays the
        // backward rule: `x̄` is the custom gradient and the stash now holds the cotangent that was pulled back.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(0.5f32.cos() * 2.0).unwrap())))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(2.0f32).unwrap()));

        // Every application writes the stash anew.
        pullback
            .apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            )
            .unwrap();
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The non-differentiated count cannot exceed the number of input leaves.
        let function = custom_vjp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        )
        .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_vjp non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_vjp_call_reference_contract() {
        let input_types = (
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        );

        // A reference input that is not declared as plumbing is an active input, which the staged operation rejects.
        let function = custom_vjp(
            |(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x),
            |(stash, x)| Ok((x, stash)),
            |stash, cotangent| Ok((stash, cotangent)),
        );
        assert_eq!(
            ArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types.clone()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_vjp accepts reference inputs only in its leading non-differentiated segment; move input 0 of \
                 type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // A forward rule may forward the plumbing reference as a residual but not return a reference it allocated.
        let function = custom_vjp(
            |(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x),
            |(_, x)| Ok((x.clone(), x.reference_new()?)),
            |allocated, cotangent| Ok((allocated, cotangent)),
        )
        .with_non_differentiated_count(1);
        assert_eq!(
            ArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types.clone()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_vjp forward rule returns residual 0 of reference type `ref<f32[]>` that is not a leading \
                 non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // Every reference-typed residual is held to that rule, not only the first one: forwarding the plumbing
        // reference and then returning an allocated reference beside it is rejected at the allocated residual.
        let function = custom_vjp(
            |(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x),
            |(stash, x)| Ok((x.clone(), (stash, x.reference_new()?))),
            |(stash, _allocated), cotangent| Ok((stash, cotangent)),
        )
        .with_non_differentiated_count(1);
        assert_eq!(
            ArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types.clone()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_vjp forward rule returns residual 1 of reference type `ref<f32[]>` that is not a leading \
                 non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // No rule may return a reference as a primal output, not even a plumbing input forwarded by identity.
        let function = custom_vjp(
            |(stash, _): (ArrayIrTracer, ArrayIrTracer)| Ok(stash),
            |(stash, _)| Ok((stash, ())),
            |(), cotangent| Ok((cotangent.clone(), cotangent.read()?)),
        )
        .with_non_differentiated_count(1);
        assert_eq!(
            ArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_vjp cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_vjp_call_pullback() {
        // The reverse entry stages an opaque tangent carrier, and its direct transpose replays the backward program
        // into the pullback, so seeding the pullback at `[cotangent, residuals...]` recovers `residual * cotangent`.
        // The forward rule defines the residual as `cos(x)`, so at `x = 0.7` and a unit cotangent the input cotangent
        // is `cos(0.7)`.
        let function = custom_vjp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let (_, pullback) = differentiate_at(Array::scalar(0.7).unwrap()).vjp(|x| function.call(x)).unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::scalar(0.7f64.cos()).unwrap()]));
    }

    #[test]
    fn test_custom_vjp_call_multiple_outputs() {
        // A two-output custom VJP exercises the output/residual split of the forward region: its leading values are the
        // primal outputs and the rest are residuals, and the backward region consumes one cotangent per output. The
        // deliberately wrong rule scales the contribution of the first output by 2 and that of the second by 3, so
        // seeding one output cotangent at a time isolates each term of the custom backward rule.
        let function = custom_vjp(
            |x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?)),
            |x| Ok(((x.sin()?, x.cos()?), (x.cos()?, x.sin()?))),
            |(cosine, sine), (first, second)| {
                let from_first = cosine * first;
                let from_second = sine * second;
                Ok(from_first.clone() + from_first + from_second.clone() + from_second.clone() + from_second)
            },
        );
        let ((sine, cosine), pullback) =
            differentiate_at(Array::scalar(0.5).unwrap()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(sine, Array::scalar(0.5f64.sin()).unwrap());
        assert_eq!(cosine, Array::scalar(0.5f64.cos()).unwrap());
        assert_eq!(
            pullback.apply((Array::scalar(1.0).unwrap(), Array::scalar(0.0).unwrap())),
            Ok(Array::scalar(2.0 * 0.5f64.cos()).unwrap()),
        );
        assert_eq!(
            pullback.apply((Array::scalar(0.0).unwrap(), Array::scalar(1.0).unwrap())),
            Ok(Array::scalar(3.0 * 0.5f64.sin()).unwrap()),
        );
    }

    #[test]
    fn test_custom_vjp_call_structured_signatures() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` calling convention, and the captured `repeats`
        // count plays the role of static configuration that is visible to the rule closures without being
        // differentiated or stored as a residual.
        let repeats = 3usize;
        let function = custom_vjp(
            |(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y),
            |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
            move |(x, y), cotangent| {
                // The deliberately wrong rule repeats both cotangents `repeats` times.
                let (base_x, base_y) = (y * cotangent.clone(), x * cotangent);
                let (mut scaled_x, mut scaled_y) = (base_x.clone(), base_y.clone());
                for _ in 1..repeats {
                    scaled_x = scaled_x + base_x.clone();
                    scaled_y = scaled_y + base_y.clone();
                }
                Ok((scaled_x, scaled_y))
            },
        );

        // The custom rule triples the true gradients `(y, x)`.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap()))
                .value_and_gradient(|(x, y)| function.call((x, y)).unwrap()),
            Ok((Array::scalar(10.0).unwrap(), (Array::scalar(15.0).unwrap(), Array::scalar(6.0).unwrap()))),
        );
    }

    #[test]
    fn test_custom_vjp_call_empty_residuals() {
        // A forward rule that saves nothing (i.e., `Residuals = ()`) exercises the zero-residual carrier path: the
        // backward rule depends only on the output cotangent, so the deliberately wrong
        // `backward(cotangent) = 2 * cotangent` makes the gradient the constant `2` instead of `cos(x)`.
        let function = custom_vjp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x| Ok((x.sin()?, ())),
            |(), cotangent| Ok(cotangent.clone() + cotangent),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0).unwrap())),
        );
    }

    #[test]
    fn test_custom_vjp_call_batching_replicated_inputs() {
        // Mapping only the first input verifies that the replicated input remains shared at the call boundary while
        // operations inside its regions broadcast it only where per-item multiplication requires alignment.
        let function = custom_vjp(
            |(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y),
            |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
            |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
        );
        let output: Array = batch(
            |(x, y)| function.call((x, y)),
            (Array::vector(vec![2.0, 3.0, 4.0]).unwrap(), Array::scalar(5.0).unwrap()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![10.0, 15.0, 20.0]).unwrap());
    }

    #[test]
    fn test_custom_vjp_call_zero_space_boundaries() {
        // Token primals, residuals, and zero-space cotangents carry no payload, so the wrapper must pass them through
        // the traced forward and backward rules unchanged instead of demanding a dense cotangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function = custom_vjp(
            |token: DomainTracer<ArrayContext>| Ok(token),
            |token| Ok((token.clone(), token)),
            |_residual, cotangent| Ok(cotangent),
        );
        let (value, pullback) = differentiate_at(token.clone()).vjp(|token| function.call(token)).unwrap();
        assert_eq!(value, token);
        assert_eq!(pullback.apply(zero.clone()), Ok(zero));
    }
}
