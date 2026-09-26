use std::fmt::Display;
use std::marker::PhantomData;

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, ResidualZeroProvider,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, check_types, impl_non_transposable_operation, impl_reference_dischargeable_operation,
};
use crate::operations::constants::zero::Zero;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    InputRegionProvenance, Operation, OperationFormatter, OutputRegionProvenance, Program, ProgramError,
    ReferenceBoundary, RegionInterface, RegionSlot, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, Trace};

/// Canonical operation name for [`CustomJvpOperation`].
pub const CUSTOM_JVP_OPERATION_NAME: &str = "custom_jvp";

/// Higher-order [`Operation`] that pairs a primal [`Program`] with a user-supplied Jacobian-Vector Product (JVP)
/// [`Program`] and that the [`custom_jvp`] function stages. Refer to the documentation of that function for the
/// semantics of custom JVPs, including their treatment of references, how each transform handles a staged call,
/// and when to reach for one.
///
/// The two programs are supplied as the operation's attached regions (i.e., via the
/// [`RegionDriver`](crate::RegionDriver) passed to [`Context::bind`]) in the region order `["primal", "jvp"]`.
/// Writing the leading [`non_differentiated_count`](Self::non_differentiated_count) inputs as `p`, the remaining
/// _differentiated_ inputs as `x`, and the primal outputs as `y`, the region interfaces are:
///
///   - `primal`: `(p, x) → y`, and
///   - `jvp`: `(p, x, ẋ) → (y, ẏ)`, with one tangent per differentiated input and one tangent per primal output.
///
/// [`Operation::infer_output_types`] validates that the attached regions realize exactly these interfaces, that only
/// `p` contains references, and that no output is a reference. Keeping `p` explicit while omitting its tangent from
/// the JVP region distinguishes an input that parameterizes the rule from an ordinary input whose tangent merely
/// happens to be zero. Batching is a canonical producer of such inputs with a batching policy that threads batching
/// state through a structurally batched region's boundary (e.g., a composite universe's first-class mapped extent)
/// reintroducing that state as additional leading non-differentiated inputs of the batched call.
///
/// The `T` parameter fixes the type universe of both attached regions and the call boundary, so each concrete payload
/// has exactly one [`Operation<Type = T>`](Operation) contract while the semantic and transform implementations remain
/// shared across differentiable type universes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomJvpOperation<T: DifferentiableType> {
    /// Number of leading inputs that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: DifferentiableType> CustomJvpOperation<T> {
    /// Creates a new [`CustomJvpOperation`] whose attached regions operate on `T` values and whose inputs are all
    /// differentiated.
    pub const fn new() -> Self {
        Self { non_differentiated_count: 0, marker: PhantomData }
    }

    /// Sets the number of leading inputs that parameterize this call without being differentiated. Refer to the
    /// documentation of [`CustomJvpOperation`] for the impact of this property on the interfaces of the attached
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
}

impl<T: DifferentiableType> Copy for CustomJvpOperation<T> {}

impl<T: DifferentiableType> Default for CustomJvpOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DifferentiableType> Display for CustomJvpOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation for CustomJvpOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_JVP_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("primal"), RegionSlot::rule("jvp")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        let (_, differentiated_input_types) = self.split_inputs(input_types)?;
        let mut jvp_input_types = input_types.to_vec();
        jvp_input_types.extend(
            differentiated_input_types
                .iter()
                .map(DifferentiableType::tangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?,
        );
        Ok(vec![Some(input_types.to_vec()), Some(jvp_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        // Output inference is a standalone validation entry point, so it must validate the complete
        // post-instantiation region contract even when `infer_region_input_types` was not called first.
        check_count!("region", region_interfaces, 2, TypeError);
        let primal_interface = &region_interfaces[0];
        let jvp_interface = &region_interfaces[1];
        let primal_input_types = primal_interface.input_types();
        let primal_output_types = primal_interface.output_types();
        let (_, differentiated_input_types) = self.split_inputs(primal_input_types)?;
        let mut expected_jvp_input_types = primal_input_types.to_vec();
        expected_jvp_input_types.extend(
            differentiated_input_types
                .iter()
                .map(DifferentiableType::tangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?,
        );
        check_types!(@same, format!("{CUSTOM_JVP_OPERATION_NAME} rule input"), [
            &expected_jvp_input_types,
            jvp_interface.input_types(),
        ]);
        let mut expected_jvp_output_types = primal_output_types.to_vec();
        expected_jvp_output_types.extend(
            primal_output_types
                .iter()
                .map(DifferentiableType::tangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?,
        );
        check_types!(@same, format!("{CUSTOM_JVP_OPERATION_NAME} rule output"), [
            &expected_jvp_output_types,
            jvp_interface.output_types(),
        ]);
        check_types!(@same, format!("{CUSTOM_JVP_OPERATION_NAME} input"), [
            primal_interface.input_types(),
            input_types,
        ]);
        validate_custom_derivative_reference_boundary(
            CUSTOM_JVP_OPERATION_NAME,
            self.non_differentiated_count,
            primal_input_types,
            primal_output_types,
        )?;
        Ok(primal_output_types.to_vec())
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        // The primal computation region receives every input at its own position. The JVP region is a dormant rule
        // that reference analysis does not enter, so it declares no provenance.
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
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_JVP_OPERATION_NAME)?;
        if self.non_differentiated_count == 0 {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("non_differentiated_count", self.non_differentiated_count))
    }
}

// Local reference lifecycles discharge inside each region while all user-declared numeric boundaries stay intact.
// External reference inputs still require explicit state threading that these derivative interfaces do not supply.
impl_reference_dischargeable_operation!(@local_reference <T> CustomJvpOperation<T> where T: DifferentiableType);

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomJvpOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // An ordinary call computes only `f(p, x) = y`. Replaying the JVP region here would also compute `ẏ` and
        // would charge every non-differentiated execution for derivative work, so interpretation delegates solely to
        // the primal region at slot 0.
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

// The default partial-evaluation rule is the desired one here where we interpret the primal region when every input
// is known and residualize the complete custom-JVP call so its attached derivative rule remains available to later
// differentiation, otherwise.
impl<C: Context<Type: DifferentiableType, Operation: From<CustomJvpOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for CustomJvpOperation<C::Type>
{
}

impl<T: DifferentiableType, C: Context<Type = T, Operation: From<CustomJvpOperation<T>>>, P: BatchingPolicy<C>>
    BatchableOperation<C, P> for CustomJvpOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Batch the two region contracts without opening the custom derivative:
        //
        //   - Primal: (p, x)    → y
        //   - JVP:    (p, x, ẋ) → (y_jvp, ẏ).
        //
        // Each `ẋ` follows the batch axis of its corresponding `x`, while `p` has no tangent counterpart. The
        // ordinary primal, JVP-primal, and JVP-tangent computations may independently choose replicated or mapped
        // representations for the same logical output. Reconcile those three axes to one wrapper axis, align both
        // batched regions to it, and retain the custom-JVP carrier so differentiation performed after batching still
        // uses the user rule.
        //
        // A batching policy may add runtime boundary inputs such as a first-class mapped extent. Those values must
        // reach both regions but have no derivative, so prepend them to `p` and increase `non_differentiated_count`.
        // Region adaptation owns any corresponding boundary rewrites; this rule owns only the flat input split and
        // the agreement of output axes.
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (_, differentiated_axes) = self.split_inputs(input_axes.as_slice())?;
        let primal_region = driver.region(0)?;
        let jvp_region = driver.region(1)?;

        // Discover the axes produced by the ordinary primal computation without imposing a wrapper-wide layout.
        let naturally_batched_primal = driver.batch_program(
            context,
            primal_region,
            input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let primal_output_axes = naturally_batched_primal.output_axes();

        // The JVP region consumes `(primals..., differentiated_tangents...)`. A tangent has the same packed batch-axis
        // position as its corresponding primal input, so the region receives the outer input-axis signature followed
        // by its differentiated suffix.
        let jvp_input_axes = input_axes.iter().copied().chain(differentiated_axes.iter().copied()).collect::<Vec<_>>();
        let naturally_batched_jvp = driver.batch_program(
            context,
            jvp_region,
            jvp_input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let jvp_output_axes = naturally_batched_jvp.output_axes();
        check_count!("output", jvp_output_axes, 2 * primal_output_axes.len(), ProgramError);
        let (jvp_primal_output_axes, jvp_tangent_output_axes) = jvp_output_axes.split_at(primal_output_axes.len());

        // Corresponding primal and tangent results must have one packed type at the custom-JVP boundary. Prefer the
        // ordinary primal's mapped position, then the JVP primal's, then the tangent's; mapped always wins over
        // replicated so reconciliation never discards batch variation.
        let output_axes = primal_output_axes
            .iter()
            .copied()
            .zip(jvp_primal_output_axes.iter().copied())
            .zip(jvp_tangent_output_axes.iter().copied())
            .map(|((primal, jvp_primal), tangent)| {
                [primal, jvp_primal, tangent].into_iter().find(|axis| !axis.is_replicated()).unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let primal = context.align_and_adapt_batched_program_outputs(
            driver,
            primal_region,
            input_axes.as_slice(),
            naturally_batched_primal,
            output_axes.as_slice(),
        )?;
        let jvp_required_output_axes =
            output_axes.iter().copied().chain(output_axes.iter().copied()).collect::<Vec<_>>();
        let jvp = context.align_and_adapt_batched_program_outputs(
            driver,
            jvp_region,
            jvp_input_axes.as_slice(),
            naturally_batched_jvp,
            jvp_required_output_axes.as_slice(),
        )?;

        let boundary_operands = P::boundary_operands(context.axis_extent());
        let non_differentiated_count = self.non_differentiated_count + boundary_operands.len();
        let mut packed_inputs = boundary_operands;
        packed_inputs.extend(inputs.iter().map(P::value).cloned());
        let outputs = context.parent().bind(
            self.with_non_differentiated_count(non_differentiated_count),
            vec![primal, jvp],
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
    C: Context<Type: DifferentiableType, Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>>
        + Zero<C::Value>,
> DifferentiableOperation<C> for CustomJvpOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Apply the user-supplied pushforward directly. For `f(p, x) = y`, region 1 implements:
        //
        //   j(p, x, ẋ) = (f(p, x), (∂f/∂x)(p, x) · ẋ) = (y, ẏ).
        //
        // Feed every primal value, followed only by the differentiated inputs' tangents; `p` has no tangent slot in the
        // rule, so a nonzero tangent for a numeric `p` is rejected below. Replay stages the rule's ordinary primitive
        // operations directly in the active context, so it introduces no symbolic capture. Consequently, reverse mode
        // differentiation can transpose the resulting linear map in `ẋ` exactly like any other tangent program, and
        // no nested differentiation request or special reverse rule is needed here.
        let jvp_region = driver.region(1)?;
        let output_types = jvp_region.output_types();
        let output_count = output_types.len() / 2;
        let (_, differentiated_inputs) = self.split_inputs(inputs)?;

        // The rule region is replayed directly rather than differentiated, so the replayed inputs are validated as
        // defense in depth (refer to the documentation of `validate_custom_derivative_replay`).
        validate_custom_derivative_replay(
            CUSTOM_JVP_OPERATION_NAME,
            self.non_differentiated_count,
            context.primal(),
            inputs,
            &output_types[..output_count],
        )?;
        check_count!("input", jvp_region.input_types(), inputs.len() + differentiated_inputs.len(), ProgramError);

        // The JVP region consumes `(primals..., differentiated_input_tangents...)`, so feed every dual primal followed
        // by the differentiated duals' tangents.
        let mut jvp_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();

        // The user's JVP region takes every differentiated input tangent as a real region input, so materialize
        // structural zeros against their own primal, which names every runtime quantity a reference-bearing tangent
        // type omits; static inputs keep the nullary zero.
        for input in differentiated_inputs {
            let source = context.primal_to_tangent(input.primal().clone())?;
            jvp_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                context.tangent(),
                input.tangent().clone(),
                std::iter::once(&source),
            )?);
        }

        // A fused differentiation context stages primal and tangent work in the same context, so the rule replays
        // there directly. Otherwise, partition the rule into its known part, which depends only on the primal inputs
        // and computes the primal outputs, and its tangent part, and replay each part in its own context.
        let mut outputs = if std::ptr::eq(context.primal(), context.tangent()) {
            jvp_region.interpret_in_context(context.primal(), jvp_inputs)?
        } else {
            let mut known = vec![true; inputs.len()];
            known.resize(jvp_inputs.len(), false);
            let partition = driver.partition_jvp_program(jvp_region, &known, &(0..output_count).collect::<Vec<_>>())?;
            partition.interpret_in_context(context, &jvp_inputs, output_count)?
        };
        check_count!("output", outputs, 2 * output_count, ProgramError);
        let tangents = outputs.split_off(output_count);
        outputs
            .into_iter()
            .zip(tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()
    }
}

// The raw carrier is intentionally non-transposable. Differentiation first replaces `f(p, x)` with the ordinary
// primitive program computing the linear map `ẋ ↦ (∂f/∂x)(p, x) · ẋ`. Reverse mode differentiation transposes that
// replayed program, that has no `CustomJvpOperation` instructions. Therefore, only an invalid direct transpose of
// an un-linearized carrier can reach this rejection path.
impl_non_transposable_operation!(<T> CustomJvpOperation<T> where T: DifferentiableType);

/// Function with a user-supplied Jacobian-Vector Product (JVP) rule, built by [`custom_jvp`]. It stores the primal and
/// JVP closures together with a phantom marker pinning the tracer types named by those closure signatures. Refer to the
/// documentation of the [`custom_jvp`] function for the calling convention, the tracing semantics, and when to reach
/// for a custom JVP.
pub struct CustomJvp<Input, Output, Primal, Jvp> {
    /// Closure computing the primal output value from the primal input value.
    primal: Primal,

    /// Closure computing `(outputs, output_tangents)` from `(inputs, input_tangents)`.
    jvp: Jvp,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Phantom marker pinning the input and output tracer types named by the closure signatures. The [`Context`]
    /// whose universe the rules are traced into is recovered from the values passed to [`CustomJvp::call`], and so
    /// the wrapper stores neither a context value nor a context type witness.
    marker: PhantomData<fn() -> (Input, Output)>,
}

impl<
    Input,
    Output,
    Primal: Fn(Input) -> Result<Output, ProgramError>,
    Jvp: Fn(Input, Input) -> Result<(Output, Output), ProgramError>,
> CustomJvp<Input, Output, Primal, Jvp>
{
    /// Declares the leading `non_differentiated_count` flattened leaves of the input value as non-differentiated
    /// _plumbing_ inputs, which is the high-level counterpart of [`CustomJvpOperation::with_non_differentiated_count`].
    /// Refer to the documentation of the [`custom_jvp`] function for the semantics of non-differentiated inputs.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Stages this custom Jacobian-Vector Product (JVP) function on the provided tracer `input` value and returns its
    /// output value. Refer to the documentation of the [`custom_jvp`] function for the tracing semantics and for how
    /// the transforms treat the staged call.
    ///
    /// The [`Context`] `C` whose universe the two closures are traced into is the
    /// [`DispatchDomain`](Value::DispatchDomain) of the values in `input`, which is exactly the context the call is
    /// staged into. It is therefore never named at a construction or call site, while the stored closures still pin
    /// the tracer values that this universe must produce.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves, when the non-differentiated count exceeds the number
    /// of input leaves, when tracing either closure fails, when the JVP closure uses the tangent placeholder of a
    /// non-differentiated input, or when the staged [`CustomJvpOperation`] rejects the traced programs (e.g., because
    /// the JVP rule signature does not match the primal signature or because the call violates the reference
    /// contract).
    pub fn call<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: DifferentiableType, Value = V>,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        C::Operation: From<CustomJvpOperation<C::Type>>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
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
            return Err(TypeError::invalid(format!("{CUSTOM_JVP_OPERATION_NAME} requires at least one input")).into());
        };

        validate_non_differentiated_count(
            CUSTOM_JVP_OPERATION_NAME,
            self.non_differentiated_count,
            input_values.len(),
        )?;

        let (_, primal) = C::trace(&self.primal, input_types.clone())?;
        let input_tangent_types = input_types.clone().try_map_parameters(|r#type| r#type.tangent())?;
        let ((output_types, _), jvp) = C::trace(|(x, t)| (self.jvp)(x, t), (input_types, input_tangent_types))?;
        let jvp = without_non_differentiated_tangent_inputs(
            jvp.into_flat_program(),
            input_values.len(),
            self.non_differentiated_count,
        )?;
        let operation =
            C::Operation::from(CustomJvpOperation::new().with_non_differentiated_count(self.non_differentiated_count));

        // The call binds through whatever context the input values flow through (e.g., a staging trace, a batching
        // context, or a differentiation context), so the batching or differentiation rule of the bound operation fires
        // and `custom_jvp` composes with those transforms.
        let context = first.dispatch_domain();
        let outputs = context.bind(operation, vec![primal.into_flat_program(), jvp], &input_values)?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Creates a [`CustomJvp`] function from a primal closure and a Jacobian-Vector Product (JVP)
/// closure over values of [`DomainTracer`]s. This is the analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair.
///
/// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The two closure arguments implement:
///
/// ```text
/// Primal:     x      ↦ y = f(x)
/// JVP:        (x, ẋ) ↦ (y, ẏ) = (f(x), J_f(x) · ẋ)
/// ```
///
/// Thus, `primal` receives the input tracer value `x` and returns the output tracer value `y`. `jvp` receives `x`
/// and an input-tangent value `ẋ`, then returns the primal output `y` together with the Jacobian-vector product
/// `ẏ = J_f(x) · ẋ`, which must be linear in `ẋ`. The tangent values have the same parameter structures as their
/// corresponding primal values, and Ryft validates these structural and type relationships when it traces the closures.
///
/// # When to Use
///
/// Reach for a custom JVP when the function _is_ forward-differentiable but its automatically derived tangent is
/// numerically unstable or wasteful and you want to supply a stable, efficient one by hand. Classic cases are a
/// `log`-`sum`-`exp`, a softmax, or a normalization, where a handwritten tangent avoids the cancellation or redundant
/// work that the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse mode obtains
/// its gradient by transposing the supplied tangent map, so the one rule composes with forward mode, reverse mode, and
/// their higher-order combinations. Prefer it over [`custom_vjp`](fn@crate::custom_vjp) whenever the function is
/// naturally forward-differentiable, and use a custom VJP only when just the reverse rule is natural (e.g., for
/// implicit differentiation or adjoint solvers).
///
/// # Calling Convention
///
/// Both closures operate on [`Parameterized`] values of [`DomainTracer`]s (i.e., Ryft's analogue of JAX pytrees), so
/// `x` and `y` may each be a single tracer, a tuple, or any other parameterized structure. Static non-differentiated
/// configuration should be captured by both closures. A dynamic value should remain an explicit input, either as a
/// non-differentiated input (see below) or as an ordinary input whose tangent is present in `ẋ` and which a rule that
/// treats the value as a parameter ignores when constructing `ẏ`.
///
/// Because [`custom_jvp`] builds a reusable function before any input is known, the `primal` closure must annotate
/// the tracer type of its input (e.g., `|x: DomainTracer<C>| ...`), which then also fixes the input types of the
/// `jvp` closure. [`custom_derivative_at`](crate::custom_derivative_at) instead stages the same rule at a known input,
/// which lets both closures infer their parameter types from that input.
///
/// # Non-Differentiated Inputs
///
/// [`CustomJvp::with_non_differentiated_count`] declares the leading flattened input leaves as _plumbing_ that
/// parameterizes the call without being differentiated, which is the analogue of JAX's `nondiff_argnums`. Plumbing
/// leaves reach both closures at their usual positions, and the `jvp` closure keeps receiving a full `ẋ` value so that
/// its signature mirrors the primal signature. However, the tangent leaves of plumbing inputs are placeholders that
/// the rule must not use, because the staged [`CustomJvpOperation`] has no tangent slot for non-differentiated inputs.
/// A rule that consumes or returns such a placeholder is rejected when it is traced, and differentiating the call with
/// a nonzero tangent for a numeric plumbing input is rejected because the rule cannot propagate that tangent.
///
/// # References
///
/// A reference-typed input is accepted only as plumbing, where both closures receive it unchanged and may read or
/// write it. An active reference input is rejected, because the rule interface defines no tangent reference for it and
/// so a user-supplied rule could not express its derivative. A live tangent reference supplied for a plumbing input by
/// an enclosing transform is left untouched by the call, since the rule declares no derivative through the state it
/// denotes. No output may be a reference either, because the rule would then have to produce that output's tangent
/// reference. The closures may also allocate and use local reference state, which executes like any other primitive
/// operation whenever the corresponding program is replayed. When the call is differentiated, no two reference inputs
/// may bind the same allocation.
///
/// # Tracing Semantics
///
/// Nothing is traced at construction time. Each [`CustomJvp::call`] recovers the tracing [`Context`] from the values
/// it is called with, reads the input types off those values, traces both closures into programs specialized to those
/// types, validates the rule signature, and stages one [`CustomJvpOperation`] into the context through which those
/// values flow. The primal closure is kept separate from the JVP closure for efficiency rather than necessity: the JVP
/// rule computes both the outputs and their tangents, so deriving the primal from it would make every un-differentiated
/// call pay for tangent computation.
///
/// # Transform Semantics
///
/// The transforms treat a staged call as follows:
///
///   - _interpretation_ and backend lowering replay the lean primal program only,
///   - _partial evaluation_ folds a call whose inputs are all known and otherwise residualizes it unchanged, so that
///     the JVP rule stays attached for a later differentiation,
///   - _batching_ preserves the call around axis-reconciled batched copies of both programs, so that the custom
///     derivative survives batching applied _before_ differentiation, and
///   - _differentiation_ replays the JVP program instead of differentiating the primal body. The replayed rule consists
///     of ordinary primitive operations, so reverse mode transposes the linear map in `ẋ` that it computes, exactly
///     like any other tangent program, and the staged call itself is never transposed.
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y`.
///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
#[inline]
pub fn custom_jvp<
    Input,
    Output,
    Primal: Fn(Input) -> Result<Output, ProgramError>,
    Jvp: Fn(Input, Input) -> Result<(Output, Output), ProgramError>,
>(
    primal: Primal,
    jvp: Jvp,
) -> CustomJvp<Input, Output, Primal, Jvp> {
    CustomJvp { primal, jvp, non_differentiated_count: 0, marker: PhantomData }
}

/// Validates that the leading `non_differentiated_count` input positions of a custom derivative call named `name` fit
/// within its `input_count` inputs.
///
/// # Errors
///
/// Returns a [`TypeError`] when `non_differentiated_count` exceeds `input_count`.
pub(super) fn validate_non_differentiated_count(
    name: &str,
    non_differentiated_count: usize,
    input_count: usize,
) -> Result<(), TypeError> {
    if non_differentiated_count > input_count {
        return Err(TypeError::invalid(format!(
            "{name} non-differentiated input count {non_differentiated_count} exceeds input count {input_count}",
        )));
    }
    Ok(())
}

/// Validates the reference contract that the custom derivative operations share over one primal boundary.
/// A reference-typed input is accepted only in the leading `non_differentiated_count` positions, which correspond to
/// _plumbing_ that every attached rule region receives unchanged (the rule interfaces define no tangent or cotangent
/// slot for a reference, so an active reference input would have a derivative that no user-supplied rule can express).
/// No output may be a reference, because a rule region would then have to produce that output's tangent reference or
/// consume its cotangent reference, and a user-supplied rule can neither allocate nor receive one. Both operations
/// apply this contract during type inference, so it holds for every constructed call, and their forward-mode rules
/// apply it again to the replayed inputs as defense in depth.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `input_types`: Primal input types in input order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns a [`TypeError`] naming the first reference-typed input in the differentiated segment, or otherwise the
/// first reference-typed output.
pub(super) fn validate_custom_derivative_reference_boundary<T: Type>(
    name: &str,
    non_differentiated_count: usize,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    if let Some((index, r#type)) = input_types
        .iter()
        .enumerate()
        .skip(non_differentiated_count)
        .find(|(_, r#type)| r#type.is_reference())
    {
        return Err(TypeError::invalid(format!(
            "{name} accepts reference inputs only in its leading non-differentiated segment; move input {index} of \
             type `{type}` before the differentiated inputs",
        )));
    }
    if let Some((index, r#type)) = output_types.iter().enumerate().find(|(_, r#type)| r#type.is_reference()) {
        return Err(TypeError::invalid(format!(
            "{name} cannot return a reference, but output {index} has type `{type}`",
        )));
    }
    Ok(())
}

/// Validates the inputs before replaying a custom derivative rule. Rule regions bypass ordinary differentiation
/// dispatch, so replay checks the reference contract of [`validate_custom_derivative_reference_boundary`] that type
/// inference already enforces and uses [`ReferenceBoundary`] to reject aliased concrete or staged references.
/// Non-differentiated numeric inputs must have zero tangents, while non-differentiated references carry state whose
/// tangent the rule leaves untouched.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `context`: Context in which the rule is replayed, which resolves the inputs.
///   - `inputs`: Dual inputs of the replayed call, in input order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns the [`ProgramError`] of the first violated contract: the [`TypeError`] of
/// [`validate_custom_derivative_reference_boundary`], a reference boundary error, or an unsupported nonzero tangent.
pub(super) fn validate_custom_derivative_replay<C: Context<Type: DifferentiableType>>(
    name: &str,
    non_differentiated_count: usize,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
    output_types: &[C::Type],
) -> Result<(), ProgramError> {
    let primal_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
    validate_custom_derivative_reference_boundary(
        name,
        non_differentiated_count,
        primal_types.as_slice(),
        output_types,
    )?;
    ReferenceBoundary::new_for_differentiation(context, inputs.iter().map(DifferentiationDual::primal), [], [])?;
    if let Some(input) = inputs.iter().take(non_differentiated_count).find(|input| {
        !input.primal().r#type().is_reference()
            && !input.tangent().is_zero()
            && !input.tangent().r#type().is_zero_space()
    }) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{} cannot propagate the nonzero tangent of type `{}` supplied for one of its \
                 {} leading non-differentiated inputs, because its rule has no tangent slot for them",
                name,
                input.tangent().r#type(),
                non_differentiated_count,
            ),
        });
    }
    Ok(())
}

/// Removes the tangent inputs that a traced JVP rule declares for the leading `non_differentiated_count` inputs.
/// The rule closure receives one tangent per input so that its signature mirrors the primal signature, but a
/// non-differentiated input has no tangent slot in the [`CustomJvpOperation`] contract, so its tangent input is a
/// placeholder that the rule must ignore. The traced program has inputs `[inputs..., input_tangents...]`, and the
/// placeholders are the tangents at positions `input_count..input_count + non_differentiated_count`, which are
/// projected away so that the remaining boundary is exactly `[inputs..., differentiated_input_tangents...]`.
///
/// # Parameters
///
///   - `program`: Traced JVP rule program over `[inputs..., input_tangents...]`.
///   - `input_count`: Number of primal inputs.
///   - `non_differentiated_count`: Number of leading non-differentiated inputs whose tangent placeholders are removed.
///
/// # Errors
///
/// Returns a [`TypeError`] when the rule consumes or returns a placeholder tangent, and propagates program projection
/// errors otherwise.
fn without_non_differentiated_tangent_inputs<V: Value, O: Clone + Operation<Type = V::Type>>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    input_count: usize,
    non_differentiated_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
    if non_differentiated_count == 0 {
        return Ok(program);
    }
    let input_ids = program.input_ids();
    check_count!("input", input_ids, 2 * input_count, ProgramError);
    let (primal_ids, tangent_ids) = input_ids.split_at(input_count);
    let (placeholder_ids, differentiated_tangent_ids) = tangent_ids.split_at(non_differentiated_count);
    for (index, placeholder) in placeholder_ids.iter().enumerate() {
        // Attached regions are closed over their own inputs, so a use of an entry input is always a direct input or
        // output of the entry region.
        let used = program.output_ids().contains(placeholder)
            || program.instructions().iter().any(|instruction| instruction.inputs().contains(placeholder));
        if used {
            return Err(TypeError::invalid(format!(
                "{CUSTOM_JVP_OPERATION_NAME} rule uses the tangent of leading non-differentiated input {index}, which \
                 has no tangent slot because non-differentiated inputs parameterize the rule without being \
                 differentiated",
            ))
            .into());
        }
    }
    let kept_ids = primal_ids.iter().chain(differentiated_tangent_ids).copied().collect::<Vec<_>>();
    let (pruned, _) = program.filtered(kept_ids.as_slice(), program.output_ids(), kept_ids.as_slice())?;
    Ok(pruned)
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, DataType, ShardingDimension,
    };
    use crate::axes::AxisIndexOperation;
    use crate::batching::{
        BatchAxis, BatchingContext, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, batch,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::assertions::{AssertOperation, AssertionError};
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::differentiation::tests::{
        ReferenceRuleDifferentiationDriver, custom_derivative_call_program,
    };
    use crate::operations::dot::{Dot, DotDimensionNumbers};
    use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
        ReferenceReadOperation,
    };
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        EffectClass, EffectClasses, FlatProgram, MaybeZero, ProgramBuilder, ReferenceType, RegionRole,
    };

    use super::*;

    /// Eager context whose values are arrays.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager composite context whose values may be arrays or references.
    type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Builds a reference-free program representing the identity function over `r#type`.
    fn array_ir_identity_program(r#type: &ArrayIrType) -> FlatProgram<ArrayIrContext> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Pairs an identity primal program with a custom JVP program that allocates and reads local reference state.
    fn custom_jvp_regions_with_reference_state(r#type: &ArrayIrType) -> Vec<FlatProgram<ArrayIrContext>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output, tangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        vec![array_ir_identity_program(r#type), jvp_program]
    }

    /// Builds a custom-derivative rule whose nested custom JVP closure contains local reference state.
    pub(crate) fn nested_custom_derivative_state_program(
        scalar_type: &ArrayIrType,
        include_tangent_output: bool,
    ) -> FlatProgram<ArrayIrContext> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let regions = custom_jvp_regions_with_reference_state(scalar_type)
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(scalar_type.clone());
        let tangent = include_tangent_output.then(|| builder.add_input(scalar_type.clone()));
        let output = builder
            .add_instruction(CustomJvpOperation::<ArrayIrType>::new(), regions, vec![input], None)
            .unwrap()[0];
        let mut outputs = vec![output];
        if let Some(tangent) = tangent {
            outputs.push(tangent);
        }
        let input_count = usize::from(include_tangent_output) + 1;
        let output_count = outputs.len();
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
            .unwrap()
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `jvp(x, ẋ) = (sin(x), 2 * cos(x) * ẋ)`, which is detectably different from
    /// the true derivative so that tests can prove that the custom rule is used.
    fn doubled_sin_jvp_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let two = builder.add_constant(Array::scalar(2.0).unwrap());
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![two, cosine], None).unwrap()[0];
        let output_tangent =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, tangent], None).unwrap()[0];
        builder.build(vec![y, output_tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Builds the malformed rule `jvp(x, ẋ) = (sin(x), 1)`, whose tangent ignores `ẋ` and is therefore an affine
    /// constant rather than a linear tangent map.
    fn known_tangent_jvp_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let tangent = builder.add_constant(Array::scalar(1.0).unwrap());
        builder.build(vec![y, tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Returns a custom-JVP call over `f(x) = sin(x)` together with its `["primal", "jvp"]` regions, whose JVP rule
    /// deliberately doubles the true derivative.
    fn custom_jvp_sin(r#type: &ArrayType) -> (ArrayOperation<Array>, Vec<FlatProgram<ArrayContext>>) {
        (
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            vec![sin_program(r#type), doubled_sin_jvp_program(r#type)],
        )
    }

    /// Builds the square function and its fused rule, with a tangent accumulator allocated before the required primal
    /// output. The coefficient is pure known work even though the accumulator must be fresh for every pushforward
    /// call. The rule reads the accumulator through [`ReferenceFreezeOperation`] when `consume` is `true` and through
    /// [`ReferenceReadOperation`] otherwise.
    fn stateful_square_regions(consume: bool) -> Vec<FlatProgram<ArrayIrContext>> {
        let scalar: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = primal.add_input(scalar.clone());
        let output = primal
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let primal = primal.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = rule.add_input(scalar.clone());
        let tangent = rule.add_input(scalar);
        let coefficient = rule.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let zero = rule.add_constant(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()));
        let accumulator = rule.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let update = rule
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![coefficient, tangent], None)
            .unwrap()[0];
        rule.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![accumulator, update], None)
            .unwrap();
        let read = if consume {
            ArrayIrOperation::from(ReferenceFreezeOperation::new())
        } else {
            ArrayIrOperation::from(ReferenceReadOperation::new())
        };
        let derivative = rule.add_instruction(read, Vec::new(), vec![accumulator], None).unwrap()[0];
        let output = rule
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let rule = rule.build(vec![output, derivative], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![primal, rule]
    }

    /// Builds the plumbing-reference primal `f(counters..., x) = { counters += x; x }` and the JVP rule
    /// `jvp(counters..., x, ẋ) = { counters += x; (x, ẋ) }` over `counter_count` leading `ref<f32[]>` counters and
    /// an `f32[]` input, so that replaying either region is observable through the counters.
    fn counting_custom_jvp_regions(counter_count: usize) -> Vec<FlatProgram<ArrayIrContext>> {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let counters = (0..counter_count).map(|_| primal.add_input(reference_type.clone())).collect::<Vec<_>>();
        let x = primal.add_input(scalar_type.clone());
        for counter in counters {
            primal
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, x], None)
                .unwrap();
        }
        let primal = primal
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![x],
                vec![Placeholder; counter_count + 1],
                vec![Placeholder],
            )
            .unwrap();
        let mut jvp = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let counters = (0..counter_count).map(|_| jvp.add_input(reference_type.clone())).collect::<Vec<_>>();
        let x = jvp.add_input(scalar_type.clone());
        let tangent = jvp.add_input(scalar_type);
        for counter in counters {
            jvp.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, x], None).unwrap();
        }
        let jvp = jvp
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![x, tangent],
                vec![Placeholder; counter_count + 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        vec![primal, jvp]
    }

    #[test]
    fn test_custom_jvp() {
        let operation = CustomJvpOperation::<ArrayType>::new();
        assert_eq!(operation, CustomJvpOperation::default());
        assert_eq!(operation.name(), CUSTOM_JVP_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(format!("{operation}"), "custom_jvp");

        // The primal program is a computation region and the user JVP program is a dormant rule region, in that order.
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("primal"), RegionSlot::rule("jvp")]);
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));

        // The primal region receives every input at its own position and its outputs are the call's outputs, while
        // the dormant rule region declares no input provenance.
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
        );
    }

    #[test]
    fn test_custom_jvp_with_non_differentiated_count() {
        let operation = CustomJvpOperation::<ArrayType>::new().with_non_differentiated_count(1);
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "custom_jvp [non_differentiated_count=1]");

        // Both regions receive the non-differentiated input at its own position, but the JVP region receives a tangent
        // only for the differentiated input.
        let parameter_type = ArrayType::new_static(DataType::F64, [2]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input_types = vec![parameter_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);
        let jvp_input_types = vec![parameter_type, scalar_type.clone(), scalar_type.clone()];
        let jvp_interface = RegionInterface::new(
            jvp_input_types.clone(),
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_region_input_types(&input_types, &[primal_interface.clone(), jvp_interface.clone()]),
            Ok(vec![Some(input_types.clone()), Some(jvp_input_types)]),
        );
        assert_eq!(
            operation.infer_output_types(&input_types, &[primal_interface, jvp_interface]),
            Ok(vec![scalar_type]),
        );
    }

    #[test]
    fn test_custom_jvp_type_inference() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = CustomJvpOperation::<ArrayType>::new();
        let primal_interface = sin_program(&scalar_type).interface();
        let jvp_interface = doubled_sin_jvp_program(&scalar_type).interface();

        // The primal region receives the call inputs, the JVP region receives `(inputs..., input_tangents...)`, and a
        // rule that satisfies the interface contract makes the call produce the primal outputs.
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Ok(vec![Some(vec![scalar_type.clone()]), Some(vec![scalar_type.clone(), scalar_type.clone()])]),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Ok(vec![scalar_type.clone()]),
        );

        // Inference maps the primal boundary through _differential_ types rather than requiring the tangents to reuse
        // the primal storage type, so an `f8e8m0fnu` primal pairs with an `f32` tangent.
        let primal_type = ArrayType::scalar(DataType::F8E8M0FNU);
        let tangent_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[
                    RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], EffectClasses::NONE),
                    RegionInterface::new(
                        vec![primal_type.clone(), tangent_type.clone()],
                        vec![primal_type.clone(), tangent_type],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Ok(vec![primal_type]),
        );

        // The JVP interface must be `(inputs..., input_tangents...) → (outputs..., output_tangents...)`, so a
        // primal-shaped rule signature is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), primal_interface.clone()],
            ),
            Err(TypeError::invalid(
                "custom_jvp rule input type signature mismatch: expected [f64[], f64[]] but got [f64[]]".to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(
                        vec![scalar_type.clone(), scalar_type.clone()],
                        vec![scalar_type.clone()],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Err(TypeError::invalid(
                "custom_jvp rule output type signature mismatch: expected [f64[], f64[]] but got [f64[]]".to_string(),
            )),
        );

        // The call inputs must match the primal region inputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&vector_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Err(TypeError::invalid(
                "custom_jvp input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // The call carries exactly two regions and at most as many non-differentiated inputs as it has inputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), std::slice::from_ref(&primal_interface)),
            Err(TypeError::invalid("expected 2 regions but got 1".to_string())),
        );
        assert_eq!(
            operation
                .with_non_differentiated_count(2)
                .infer_region_input_types(std::slice::from_ref(&scalar_type), &[primal_interface, jvp_interface]),
            Err(TypeError::invalid("custom_jvp non-differentiated input count 2 exceeds input count 1".to_string())),
        );
    }

    #[test]
    fn test_custom_jvp_type_inference_references() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input_types = vec![reference_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);

        // A reference input in the leading non-differentiated segment is plumbing that both regions receive at its own
        // position, so the rule interface carries a tangent only for the differentiated input.
        let jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            CustomJvpOperation::<ArrayIrType>::new()
                .with_non_differentiated_count(1)
                .infer_output_types(&input_types, &[primal_interface.clone(), jvp_interface]),
            Ok(vec![scalar_type.clone()]),
        );

        // The same input in the differentiated segment would need a tangent reference that the rule cannot define, so
        // it is rejected even when the rule interface declares one.
        let active_jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            CustomJvpOperation::<ArrayIrType>::new()
                .infer_output_types(&input_types, &[primal_interface, active_jvp_interface]),
            Err(TypeError::invalid(
                "custom_jvp accepts reference inputs only in its leading non-differentiated segment; move input 0 of \
                 type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );

        // No output may be a reference, not even a forwarded plumbing input, because the rule would then have to
        // produce its tangent reference.
        let forwarding_primal_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), scalar_type],
            vec![reference_type.clone(), reference_type],
            EffectClasses::NONE,
        );
        assert_eq!(
            CustomJvpOperation::<ArrayIrType>::new()
                .with_non_differentiated_count(1)
                .infer_output_types(&input_types, &[forwarding_primal_interface, forwarding_jvp_interface]),
            Err(TypeError::invalid(
                "custom_jvp cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_jvp_reference_discharge() {
        for consume in [false, true] {
            let program = custom_derivative_call_program(
                CustomJvpOperation::new(),
                stateful_square_regions(consume),
                vec![ArrayType::scalar(DataType::F32).into()],
            );

            // Local rule state is discharged inside the rule region, while the declared JVP rule remains attached and
            // active.
            let discharged =
                program.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
            assert_eq!(discharged.instructions()[0].regions().len(), 2);
            assert!(!discharged.entry_region_ref().contains_references_in_closure());
            assert_eq!(
                discharged.interpret(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
                Ok(vec![ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())]),
            );
            assert_eq!(
                discharged.jvp().unwrap().interpret(vec![
                    ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
                ]),
            );
        }
    }

    #[test]
    fn test_custom_jvp_interpretation() {
        // Interpretation replays the primal region only, so an un-differentiated call never pays for the tangent
        // computation of the JVP region.
        let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
        assert_eq!(
            ArrayContext::new().bind(operation, regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_custom_jvp_partial_evaluation() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (operation, regions) = custom_jvp_sin(&scalar_type);
        let program = custom_derivative_call_program(operation, regions, vec![scalar_type.clone()]);

        // A call whose inputs are all known folds by interpreting its primal region.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(Array::scalar(2.0).unwrap())]).unwrap();
        assert!(matches!(
            &evaluation.outputs[0],
            PartialEvaluationOutput::Known(output) if output == &Array::scalar(2.0f64.sin()).unwrap(),
        ));

        // A call with an unknown input residualizes unchanged instead of inlining its primal region, which keeps the
        // custom rule attached to the residual program.
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_type)]).unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomJvp(_)));
    }

    #[test]
    fn test_custom_jvp_batching() {
        let output: Array = batch(
            |x| {
                let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
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
    fn test_custom_jvp_batching_natural_output_axes() {
        let vector_type = ArrayType::new_static(DataType::F64, [3]);

        // The primal has one mapped identity output, one naturally replicated constant output, and one replicated
        // constant whose JVP tangent is mapped. The third pair forces reconciliation to mapped without forcing the
        // independently replicated second pair to acquire a batch axis.
        let primal = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]).unwrap());
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]).unwrap());
            builder.build(vec![input, replicated, reconciled], vec![Placeholder], vec![Placeholder; 3]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let tangent = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]).unwrap());
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]).unwrap());
            let zero = builder.add_constant(Array::vector(vec![0.0, 0.0, 0.0]).unwrap());
            builder
                .build(
                    vec![input, replicated, reconciled, tangent, zero, tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 6],
                )
                .unwrap()
        };
        let program = custom_derivative_call_program(
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            vec![primal, jvp],
            vec![vector_type],
        );

        // Mapping the input at packed axis 1 preserves that position for varying outputs. The independent constant
        // remains replicated, while the third primal constant is broadcast only because its corresponding tangent
        // varies at axis 1. None of the attached regions transposes that natural axis to a wrapper-wide convention.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1), BatchAxis::replicated(), BatchAxis::new(1)]);
        assert_eq!(batched.instructions().len(), 1);
        let instruction = &batched.instructions()[0];
        assert!(matches!(instruction.operation(), ArrayOperation::CustomJvp(_)));
        assert!(instruction.regions().iter().all(|region| {
            batched
                .region_ref(*region)
                .unwrap()
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayOperation::Transpose(_)))
        }));

        let input = Array::matrix(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        assert_eq!(
            batched.interpret(vec![input.clone()]),
            Ok(vec![
                input,
                Array::vector(vec![4.0, 5.0, 6.0]).unwrap(),
                Array::matrix(3, 2, vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_jvp_batching_named_axis_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder.build(vec![index], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            builder.add_input(scalar_type);
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            let tangent = builder.add_constant(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
            builder.build(vec![index, tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
        };
        let regions = vec![primal, jvp];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(ArrayContext::new(), 3).with_axis_name("items".to_string());

        // No input carries a mapped axis, but `axis_index("items")` observes the active transform inside both regions
        // and naturally produces a mapped output. Batching must therefore inspect the regions rather than assuming
        // that all-replicated call inputs imply all-replicated call outputs.
        let outputs = CustomJvpOperation::new()
            .batch(&context, &driver, &[ArrayBatch::replicated(Array::scalar(1.0).unwrap())])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![0u64, 1, 2]).unwrap());
    }

    #[test]
    fn test_custom_jvp_batching_preserves_custom_derivative() {
        // Differentiating through a batched custom call must still use the deliberately doubled custom rule, because
        // batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
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
        assert_eq!(gradient, Array::vector(vec![2.0 * 0.5f64.cos(), 2.0 * 1.0f64.cos()]).unwrap());
    }

    #[test]
    fn test_custom_jvp_differentiation() {
        // The custom rule doubles the true derivative, which proves that it governs forward-mode differentiation.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            })
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f64.sin()).unwrap());
        assert_eq!(tangent, Array::scalar(2.0 * 2.0f64.cos()).unwrap());

        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = differentiate_at(Array::scalar(3.0).unwrap())
            .value_and_gradient(|x| {
                let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
                x.context().bind(operation, regions, &[x.clone()]).unwrap().remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(3.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(2.0 * 3.0f64.cos()).unwrap());
    }

    #[test]
    fn test_custom_jvp_differentiation_second_order() {
        // The JVP rule replays the user program as plain primitive operations, so the gradient program that it
        // produces is itself differentiable. The doubled rule makes the first derivative `2 cos(x)`, and so the second
        // derivative is `-2 sin(x)`.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|x| {
                differentiate_at(x)
                    .gradient(|y| {
                        let (operation, regions) = custom_jvp_sin(&ArrayType::scalar(DataType::F64));
                        y.context().bind(operation, regions, &[y.clone()]).unwrap().remove(0)
                    })
                    .unwrap()
            })
            .unwrap();
        assert_eq!(gradient, Array::scalar(2.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-2.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_custom_jvp_differentiation_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_jvp_regions_with_reference_state(&scalar_type);

        // A custom derivative rule may allocate and use local reference state: the rule is replayed directly when it
        // consumes the active input, so its state executes like any other primitive operation of the identity rule.
        assert_eq!(
            differentiate_at(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())).jvp(
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                |input| {
                    let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                    Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&input))?.remove(0))
                },
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );

        // A lifted input has a structural zero tangent. The attached rule contains references, so binding still
        // invokes it, and its identity tangent evaluates to zero.
        assert_eq!(
            differentiate_at(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())).jvp(
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                |input| {
                    let lifted = input.context().lift(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()))?;
                    let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                    Ok(input.context().bind(operation, regions, std::slice::from_ref(&lifted))?.remove(0))
                },
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()),
            )),
        );
    }

    #[test]
    fn test_custom_jvp_differentiation_staged_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let program = custom_derivative_call_program(
            ArrayIrOperation::CustomJvp(CustomJvpOperation::new()),
            custom_jvp_regions_with_reference_state(&scalar_type),
            vec![scalar_type],
        );

        // The entry region is pure because the state lives in the dormant rule region. Forward mode replays that rule
        // when it fires on the live tangent, so the fused program stages the rule's local allocation and read, which
        // execute like any other primitive operations of the identity rule.
        assert!(program.effects().classes().is_empty());
        let jvp = program.jvp().unwrap();
        assert!(jvp.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ];
        assert_eq!(jvp.interpret(inputs.clone()), Ok(inputs));
    }

    #[test]
    fn test_custom_jvp_differentiation_nested_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();

        // A rule region may allocate and use local reference state inside a dormant nested rule: the rule is replayed
        // directly (the driver makes recursive differentiation an assertion failure), so its state executes like any
        // other primitive operation and the identity rule yields the identity dual.
        let jvp = nested_custom_derivative_state_program(&scalar_type, true);
        assert!(jvp.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let driver =
            ReferenceRuleDifferentiationDriver { programs: vec![array_ir_identity_program(&scalar_type), jvp] };
        let outputs = CustomJvpOperation::<ArrayIrType>::new()
            .jvp(&DifferentiationContext::fused(ArrayIrContext::new()), &driver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(ArrayIrValue::Array(tangent)) if tangent == &Array::scalar(1.0f32).unwrap(),
        ));
    }

    #[test]
    fn test_custom_jvp_differentiation_plumbing_references() {
        let context = DifferentiationContext::fused(ArrayIrContext::new());
        let driver = ReferenceRuleDifferentiationDriver { programs: counting_custom_jvp_regions(1) };

        // The plumbing counter reaches the replayed rule as the same reference and is mutated by it. Its live tangent
        // reference is left untouched, because the rule declares no derivative through the state it denotes.
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let inputs = [
            DifferentiationDual::new(
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Reference(counter_tangent.clone()),
            )
            .unwrap(),
            DifferentiationDual::new(
                ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )
            .unwrap(),
        ];
        let outputs = CustomJvpOperation::<ArrayIrType>::new()
            .with_non_differentiated_count(1)
            .jvp(&context, &driver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(ArrayIrValue::Array(tangent)) if tangent == &Array::scalar(1.0f32).unwrap(),
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // A reference input that is also bound at another input position is rejected before either rule region runs.
        let aliased = [
            inputs[0].clone(),
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Reference(counter.clone())).unwrap(),
            inputs[1].clone(),
        ];
        let aliasing_driver = ReferenceRuleDifferentiationDriver { programs: counting_custom_jvp_regions(2) };
        assert!(matches!(
            CustomJvpOperation::<ArrayIrType>::new().with_non_differentiated_count(2).jvp(
                &context,
                &aliasing_driver,
                &aliased,
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "input 1 and input 0 bind the same reference allocation",
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The same reference in the differentiated segment is rejected by the replayed rule as well.
        assert!(matches!(
            CustomJvpOperation::<ArrayIrType>::new().jvp(&context, &driver, &inputs),
            Err(DifferentiationError::Program(ProgramError::Type(error)))
                if error == TypeError::invalid(
                    "custom_jvp accepts reference inputs only in its leading non-differentiated segment; move input \
                     0 of type `ref<f32[]>` before the differentiated inputs"
                        .to_string(),
                ),
        ));
    }

    #[test]
    fn test_custom_jvp_differentiation_dormant_assertions() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut primal = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = primal.add_input(scalar_type.clone());
        let primal = primal.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut rule = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = rule.add_input(scalar_type.clone());
        let tangent = rule.add_input(scalar_type.clone());
        let zero = rule.add_constant(Array::scalar(0.0f32).unwrap());
        let predicate = rule
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        rule.add_instruction(
            AssertOperation::new("custom derivative requires positive input").with_labels(vec!["input".to_owned()]),
            Vec::new(),
            vec![predicate, input],
            None,
        )
        .unwrap();
        let rule = rule
            .build::<Vec<Array>, Vec<Array>>(vec![input, tangent], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = custom_derivative_call_program(CustomJvpOperation::new(), vec![primal, rule], vec![scalar_type]);

        // Dormant derivative rules contribute no execution effects to the primal call and do not run when it is
        // interpreted.
        assert_eq!(program.effects().classes(), EffectClasses::NONE);
        assert_eq!(program.interpret(vec![Array::scalar(-1.0f32).unwrap()]), Ok(vec![Array::scalar(-1.0f32).unwrap()]));

        // Differentiation replays the rule, which stages its assertion into the differentiated program.
        let differentiated = program.jvp().unwrap();
        assert_eq!(differentiated.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(
            differentiated
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "assert")
                .count(),
            1,
        );
        assert_eq!(
            differentiated.interpret(vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()]),
            Ok(vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()]),
        );
        let error = differentiated
            .interpret(vec![Array::scalar(-1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "custom derivative requires positive input".to_owned(),
                observations: vec![("input".to_owned(), "-1".to_owned())],
            }),
        );
    }

    #[test]
    fn test_custom_jvp_differentiation_linearization_fresh_tangent_state() {
        for (consume, endpoint) in [(false, "reference_read"), (true, "reference_freeze")] {
            let program = custom_derivative_call_program(
                CustomJvpOperation::new(),
                stateful_square_regions(consume),
                vec![ArrayType::scalar(DataType::F32).into()],
            );

            // Linearization hoists the pure coefficient into the primal program as a residual, while the tangent
            // program allocates a fresh accumulator on every application.
            let linearization = program.linearize().unwrap();
            assert_eq!(linearization.residual_count(), 1);
            assert_eq!(
                linearization
                    .primal()
                    .instructions()
                    .iter()
                    .map(|instruction| instruction.operation().name())
                    .collect::<Vec<_>>(),
                vec!["mul", "add"],
            );
            assert_eq!(
                linearization
                    .tangent()
                    .instructions()
                    .iter()
                    .map(|instruction| instruction.operation().name())
                    .collect::<Vec<_>>(),
                vec!["reference_new", "mul", "reference_add_update", endpoint],
            );
            let primals =
                linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]).unwrap();
            assert_eq!(
                primals,
                vec![
                    ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
                ],
            );
            for (tangent, expected) in [(2.0f32, 12.0f32), (5.0, 30.0), (2.0, 12.0)] {
                assert_eq!(
                    linearization
                        .tangent()
                        .interpret(vec![ArrayIrValue::Array(Array::scalar(tangent).unwrap()), primals[1].clone()]),
                    Ok(vec![ArrayIrValue::Array(Array::scalar(expected).unwrap())]),
                );
            }
        }
    }

    #[test]
    fn test_custom_jvp_differentiation_linearization_stateful_scan() {
        let vector: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        let regions = stateful_square_regions(false)
            .into_iter()
            .map(|body| {
                // Inline the existing rule into the explicit scan body, retaining its instructions and effects.
                let mut indexed_body = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                indexed_body.add_input(ArrayType::scalar(DataType::I64).into());
                let body_inputs =
                    body.input_types().iter().map(|r#type| indexed_body.add_input(r#type.clone())).collect::<Vec<_>>();
                let outputs = indexed_body.splice_program(&body, &body_inputs).unwrap();
                let output_count = outputs.len();
                let body = indexed_body
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                        outputs,
                        vec![Placeholder; 1 + body_inputs.len()],
                        vec![Placeholder; output_count],
                    )
                    .unwrap();
                custom_derivative_call_program(
                    ScanOperation::new(0, 3),
                    vec![body],
                    vec![vector.clone(); body_inputs.len()],
                )
            })
            .collect::<Vec<_>>();
        let program = custom_derivative_call_program(CustomJvpOperation::new(), regions, vec![vector]);

        // The fused rule is partitioned inside the scan: the known coefficient becomes a stacked residual, while the
        // tangent scan keeps the accumulator lifecycle.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let primal_operations = linearization
            .primal()
            .entry_region_ref()
            .computation_regions()
            .flat_map(|region| region.instructions().iter().map(|instruction| instruction.operation().name()))
            .collect::<Vec<_>>();
        let tangent_operations = linearization
            .tangent()
            .entry_region_ref()
            .computation_regions()
            .flat_map(|region| region.instructions().iter().map(|instruction| instruction.operation().name()))
            .collect::<Vec<_>>();
        assert_eq!(primal_operations, vec!["scan", "mul", "add"]);
        assert_eq!(tangent_operations, vec!["scan", "reference_new", "mul", "reference_add_update", "reference_read"]);
        let primals = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())])
            .unwrap();
        assert_eq!(
            primals,
            vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 4.0, 9.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0f32, 4.0, 6.0]).unwrap()),
            ],
        );
        for (tangent, expected) in [(2.0f32, [4.0f32, 8.0, 12.0]), (5.0, [10.0, 20.0, 30.0]), (2.0, [4.0, 8.0, 12.0])] {
            assert_eq!(
                linearization
                    .tangent()
                    .interpret(vec![ArrayIrValue::Array(Array::vector(vec![tangent; 3]).unwrap()), primals[1].clone()]),
                Ok(vec![ArrayIrValue::Array(Array::vector(expected.to_vec()).unwrap())]),
            );
        }
    }

    #[test]
    fn test_custom_jvp_differentiation_linearization_nested_aliased_reference_carries() {
        for scanned_view in [false, true] {
            let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
            let referent_type =
                if scanned_view { ArrayType::new_static(DataType::F32, [2]) } else { ArrayType::scalar(DataType::F32) };
            let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type));
            let initial =
                if scanned_view { Array::vector(vec![0.0f32; 2]).unwrap() } else { Array::scalar(0.0f32).unwrap() };
            let expected_state =
                if scanned_view { Array::vector(vec![3.0f32; 2]).unwrap() } else { Array::scalar(6.0f32).unwrap() };
            let mut branch = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let reference = branch.add_input(reference_type.clone());
            let branch = branch
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![reference],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();

            // The body receives the same allocation through a known carry and an unknown formal. The latter is either
            // another carry or a per-iteration scalar view of the stacked reference. Each iteration writes through the
            // unknown formal, then reads through the known formal.
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let index = body.add_input(ArrayType::scalar(DataType::I64).into());
            let known_reference = body.add_input(reference_type.clone());
            let increment = body.add_input(scalar_type.clone());
            let unknown_root = body.add_input(reference_type.clone());
            let (transforms, inputs) = if scanned_view {
                (
                    vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
                    vec![unknown_root, increment, index],
                )
            } else {
                (Vec::new(), vec![unknown_root, increment])
            };
            body.add_instruction(
                ReferenceAddUpdateOperation::new().with_transforms(transforms),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap();
            let read = body
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![known_reference], None)
                .unwrap()[0];
            let read = if scanned_view {
                body.add_instruction(
                    ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                    Vec::new(),
                    vec![read],
                    None,
                )
                .unwrap()[0]
            } else {
                read
            };
            let body_outputs =
                if scanned_view { vec![known_reference, read] } else { vec![known_reference, read, unknown_root] };
            let carry_count = body_outputs.len();
            let body = body
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    body_outputs,
                    vec![Placeholder; 4],
                    vec![Placeholder; carry_count],
                )
                .unwrap();

            let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let reference = rule.add_input(reference_type.clone());
            let primal = rule.add_input(scalar_type.clone());
            let tangent = rule.add_input(scalar_type.clone());
            let zero = rule.add_constant(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()));
            let predicate = rule
                .add_instruction(
                    ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                    Vec::new(),
                    vec![tangent, zero],
                    None,
                )
                .unwrap()[0];
            let branch = rule.import_program(branch);
            let alias = rule
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, reference], None)
                .unwrap()[0];
            let body = rule.import_program(body);
            let outputs = rule
                .add_instruction(ScanOperation::new(carry_count, 2), vec![body], vec![reference, tangent, alias], None)
                .unwrap()
                .to_vec();
            let rule = rule
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![primal, outputs[1]],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap();
            let reference = ArrayReference::new(initial.clone());
            assert_eq!(
                rule.interpret(vec![
                    ArrayIrValue::Reference(reference.clone()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
                ]),
            );
            assert_eq!(reference.read(), Ok(expected_state.clone()));

            // The condition's predicate is tangent-dependent, making its forwarded reference unknown even though
            // canonical provenance still ties it to the known reference. Nested partitioning must retain that alias.
            let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            primal.add_input(reference_type.clone());
            let input = primal.add_input(scalar_type.clone());
            let primal = primal
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            let program = custom_derivative_call_program(
                CustomJvpOperation::new().with_non_differentiated_count(1),
                vec![primal, rule],
                vec![reference_type, scalar_type],
            );
            let linearization = program.entry_region_ref().linearize(&[1]).unwrap();
            let reference = ArrayReference::new(initial);
            let mut primals = linearization
                .primal()
                .interpret(vec![
                    ArrayIrValue::Reference(reference.clone()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ])
                .unwrap();
            assert_eq!(primals.remove(0), ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
            let mut tangents = vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())];
            tangents.extend(primals);
            assert_eq!(
                linearization.tangent().interpret(tangents),
                Ok(vec![ArrayIrValue::Array(Array::scalar(6.0f32).unwrap())]),
            );
            assert_eq!(reference.read(), Ok(expected_state));
        }
    }

    #[test]
    fn test_custom_jvp_differentiation_rejects_known_tangent_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomJvp(CustomJvpOperation::new());
        let regions = || vec![sin_program(&scalar_type), known_tangent_jvp_program(&scalar_type)];
        let expected = "linearization produced a known tangent output; differentiation rules must represent \
                        input-independent zero tangents structurally";

        // Program-level linearization rejects the malformed rule rather than silently replacing its constant tangent
        // with zero.
        let program = custom_derivative_call_program(operation.clone(), regions(), vec![scalar_type.clone()]);
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));

        // Value-level linearization enforces the same rule contract before exposing a reusable pushforward.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .linearize(|input| Ok(input.context().bind(operation, regions(), &[input.clone()])?.remove(0))),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));
    }

    #[test]
    fn test_custom_jvp_transposition() {
        // Differentiation replaces the call with its replayed rule before transposition, so only a direct transpose of
        // an un-linearized call reaches the operation, which rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (operation, regions) = custom_jvp_sin(&scalar_type);
        let program = custom_derivative_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_jvp` is not transposable",
        ));
    }

    #[test]
    fn test_custom_jvp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The deliberately wrong
        // rule `jvp(x, ẋ) = (sin(x), cos(x) * ẋ + cos(x) * ẋ)` doubles the true derivative (expressed through addition
        // to avoid constant lifting), which proves that the rule is in control.
        let function = custom_jvp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x, tangent| {
                let tangent = x.cos()? * tangent;
                Ok((x.sin()?, tangent.clone() + tangent))
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );

        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_eq!(
            differentiate_at(Array::scalar(3.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(3.0f64.sin()).unwrap(), Array::scalar(2.0 * 3.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_jvp_call_with_non_differentiated_count() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // The leading counter is plumbing: it reaches both closures at its usual position, and the rule closure keeps
        // receiving a full tangent value whose counter leaf is a placeholder that it leaves unused.
        let function = custom_jvp(
            |(counter, x): (DomainTracer<ArrayIrContext>, DomainTracer<ArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            },
            |(counter, x), (_, tangent)| {
                counter.add_update(&x)?;
                Ok((x, tangent))
            },
        )
        .with_non_differentiated_count(1);
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        assert_eq!(
            differentiate_at((
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ))
            .jvp(
                (
                    ArrayIrValue::Reference(counter_tangent.clone()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ),
                |(counter, x)| function.call((counter, x)),
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );

        // Forward mode replays only the rule region, which increments the counter once and never touches its tangent.
        assert_eq!(counter.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // A rule that uses the placeholder tangent of a plumbing input is rejected when it is traced, because the
        // staged rule has no tangent slot for it.
        let function = custom_jvp(
            |(counter, x): (DomainTracer<ArrayIrContext>, DomainTracer<ArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            },
            |(_, x), (counter_tangent, tangent)| {
                counter_tangent.add_update(&tangent)?;
                Ok((x, tangent))
            },
        )
        .with_non_differentiated_count(1);
        assert_eq!(
            ArrayIrContext::trace(
                |(counter, x)| function.call((counter, x)),
                (reference_type.clone(), scalar_type.clone()),
            )
            .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_jvp rule uses the tangent of leading non-differentiated input 0, which has no tangent slot \
                 because non-differentiated inputs parameterize the rule without being differentiated"
                    .to_string(),
            ))),
        );

        // Without the declaration, the reference is an active input, which the staged operation rejects.
        let function = custom_jvp(
            |(counter, x): (DomainTracer<ArrayIrContext>, DomainTracer<ArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            },
            |(_, x), (_, tangent)| Ok((x, tangent)),
        );
        assert_eq!(
            ArrayIrContext::trace(|(counter, x)| function.call((counter, x)), (reference_type, scalar_type))
                .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_jvp accepts reference inputs only in its leading non-differentiated segment; move input 0 of \
                 type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // The non-differentiated count cannot exceed the number of input leaves.
        let function =
            custom_jvp(|x: DomainTracer<ArrayContext>| Ok(x.sin()?), |x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
                .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_jvp non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_jvp_call_rule_signature_mismatch() {
        // Arity mismatches are compile-time errors under the structured signatures, but shape mismatches remain
        // runtime concerns: this rule produces a scalar tangent for a vector-valued function, so the traced JVP
        // program fails the signature validation that `CustomJvpOperation` performs at the call site.
        let function = custom_jvp(
            |x: DomainTracer<ArrayContext>| Ok(x.sin()?),
            |x, tangent| Ok((x.sin()?, tangent.dot(&tangent, &DotDimensionNumbers::inner_product())?)),
        );
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "custom_jvp rule output type signature mismatch: expected [f64[2], f64[2]] but got [f64[2], f64[]]"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_jvp_call_zero_space_boundaries() {
        // Token primals and zero-space tangents carry no payload, so the wrapper must pass them through the traced
        // rule unchanged instead of demanding a dense tangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function = custom_jvp(|token: DomainTracer<ArrayContext>| Ok(token), |token, tangent| Ok((token, tangent)));
        assert_eq!(differentiate_at(token.clone()).jvp(zero.clone(), |token| function.call(token)), Ok((token, zero)));
    }

    #[test]
    fn test_validate_non_differentiated_count() {
        assert_eq!(validate_non_differentiated_count("custom_jvp", 0, 0), Ok(()));
        assert_eq!(validate_non_differentiated_count("custom_jvp", 2, 2), Ok(()));
        assert_eq!(
            validate_non_differentiated_count("custom_jvp", 3, 2),
            Err(TypeError::invalid("custom_jvp non-differentiated input count 3 exceeds input count 2".to_string())),
        );
    }

    #[test]
    fn test_validate_custom_derivative_reference_boundary() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // References are accepted in the leading non-differentiated input segment.
        assert_eq!(
            validate_custom_derivative_reference_boundary(
                "custom_jvp",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Ok(()),
        );

        // References are rejected among the differentiated inputs and among the outputs.
        assert_eq!(
            validate_custom_derivative_reference_boundary(
                "custom_jvp",
                1,
                &[scalar_type.clone(), reference_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Err(TypeError::invalid(
                "custom_jvp accepts reference inputs only in its leading non-differentiated segment; move input 1 of \
                 type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );
        assert_eq!(
            validate_custom_derivative_reference_boundary(
                "custom_jvp",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                &[scalar_type, reference_type],
            ),
            Err(TypeError::invalid(
                "custom_jvp cannot return a reference, but output 1 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_validate_custom_derivative_replay() {
        // A non-differentiated numeric input with a nonzero tangent has no tangent slot in the rule.
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            validate_custom_derivative_replay("custom_jvp", 1, &ArrayIrContext::new(), &[input], &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "custom_jvp cannot propagate the nonzero tangent of type `f32[]` supplied for one of \
                    its 1 leading non-differentiated inputs, because its rule has no tangent slot for them",
        ));
    }

    #[test]
    fn test_validate_custom_derivative_replay_staged_references() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        ArrayIrContext::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let mut duals = inputs
                    .iter()
                    .cloned()
                    .map(DifferentiationDual::new_with_zero_tangent)
                    .collect::<Result<Vec<_>, _>>()?;
                assert_eq!(validate_custom_derivative_replay("custom_jvp", 2, context, &duals, &[]), Ok(()));

                // Replaying the same valid boundary is allowed, including a fresh root allocated inside the trace.
                let local = context.bind(ReferenceNewOperation::new(), vec![], &inputs[2..])?.remove(0);
                duals[1] = DifferentiationDual::new_with_zero_tangent(local)?;
                assert_eq!(validate_custom_derivative_replay("custom_vjp", 2, context, &duals, &[]), Ok(()));
                assert_eq!(validate_custom_derivative_replay("custom_vjp", 2, context, &duals, &[]), Ok(()));

                // An ordinary staged input must not hide two reference inputs naming the same allocation.
                duals[1] = duals[0].clone();
                assert!(matches!(
                    validate_custom_derivative_replay("custom_jvp", 2, context, &duals, &[]),
                    Err(ProgramError::InvalidArgument { message })
                        if message == "input 1 and input 0 bind the same reference allocation",
                ));
                Ok(vec![inputs[2].clone()])
            },
            vec![reference_type.clone(), reference_type, scalar_type],
        )
        .unwrap();
    }
}
