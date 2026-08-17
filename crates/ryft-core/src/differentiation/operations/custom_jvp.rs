use std::fmt::Display;
use std::marker::PhantomData;

// TODO(eaplatanios): Review this module.

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
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

/// Canonical operation name for [`CustomJvpOperation`].
pub const CUSTOM_JVP_OPERATION_NAME: &str = "custom_jvp";

/// Higher-order [`Operation`] pairing a primal program with a user-supplied Jacobian-Vector Product (JVP) program.
/// This is an analogue to JAX's [`custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
/// The two [`Program`](crate::Program)s are supplied as the operation's attached regions (i.e., via the
/// [`RegionDriver`](crate::RegionDriver) passed to [`Context::bind`]) in the region order `["primal", "jvp"]`, and
/// [`Operation::infer_output_types`] validates the interface contract between them: the JVP region's inputs are the
/// primal inputs followed by one tangent per _differentiated_ primal input, and its outputs are the primal outputs
/// followed by one tangent per primal output. Keeping the primal program separate from the JVP program means that
/// un-differentiated calls never pay for tangent computation.
///
/// The leading [`non_differentiated_count`](Self::non_differentiated_count) operands parameterize the call without
/// being differentiated. Every attached region receives them in the same leading positions, but they contribute no
/// tangent to the JVP region's input signature and receive no cotangent. Batching is its canonical producer as a
/// batching policy that threads batching state through a structurally batched region's boundary (e.g., a composite
/// universe's first-class mapped extent) reintroduces that state as additional leading non-differentiated operands
/// of the batched call.
///
/// The transforms treat a staged call as follows:
///
///   - _interpretation_ replays the primal region,
///   - _partial evaluation_ folds a call whose operands are all known and otherwise residualizes it unchanged,
///   - _batching_ preserves the call around axis-reconciled copies of both regions so that the custom derivative
///     survives batching applied _before_ differentiation, and
///   - _differentiation_ replays the user JVP region instead of differentiating the primal body, so the user-supplied
///     derivative governs both forward and reverse mode differentiation.
///
/// Refer to the documentation of [`custom_jvp`] for the full semantics and for when to reach for a custom JVP.
///
/// Note that this operation is deliberately non-transposable, which does not restrict reverse-mode differentiation.
/// Reverse mode differentiation linearizes first, and the JVP rule replays the user JVP program as plain primitive
/// operations, so the operation itself is gone from the tangent program long before transposition runs. This is also
/// (at least partially) why the JVP program must be linear in its tangent arguments. Transposition can therefore only
/// reach the operation when transposing a raw, un-linearized program directly, which is not supported.
///
/// The `T` parameter fixes the type universe of both attached regions and the call boundary, so each concrete payload
/// has exactly one [`Operation<Type = T>`](Operation) contract while the semantic and transform implementations remain
/// shared across differentiable type universes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomJvpOperation<T: DifferentiableType> {
    /// Number of leading operands that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Type universe in which this [`CustomJvpOperation`] is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: DifferentiableType> CustomJvpOperation<T> {
    /// Creates a new [`CustomJvpOperation`] whose attached regions operate on `T` values and whose operands are all
    /// participating in the differentiation transform.
    #[inline]
    pub const fn new() -> Self {
        Self { non_differentiated_count: 0, marker: PhantomData }
    }

    /// Sets the number of leading inputs/operands that parameterize this call without participating in the
    /// differentiation transform. Refer to the documentation of [`CustomJvpOperation`] for the impact of this property
    /// on the resulting region interfaces for the attached regions.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Returns the number of leading inputs/operands that parameterize this call without participating in the
    /// differentiation transform.
    #[inline]
    pub fn non_differentiated_count(&self) -> usize {
        self.non_differentiated_count
    }

    /// Splits the provided in put `values` into the leading non-differentiated group and the trailing differentiated
    /// group, based on the value of [`Self::non_differentiated_count`].
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

// TODO(eaplatanios): Review from here onwards.

impl<T: DifferentiableType> Operation for CustomJvpOperation<T> {
    // `CustomJvpOperation`s carry two regions with one shared primal boundary. Writing the leading non-differentiated
    // operands as `p`, the differentiated operands as `x`, and the primal outputs as `y`, their contracts are:
    //
    //   - Primal: (p, x)    → y
    //   - JVP:    (p, x, ẋ) → (y, ẏ).
    //
    // Type inference declares those region inputs independently of the concrete programs and then checks that the
    // supplied interfaces realize the declaration exactly. Keeping `p` explicit but omitting `ṗ` distinguishes an
    // operand that parameterizes the rule from an ordinary input whose tangent merely happens to be zero.

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
        // Output inference is a standalone validation entry point, so it must validate the complete post-instantiation
        // region contract even when `infer_region_input_types` was not called first.
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
        Ok(primal_output_types.to_vec())
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // A call whose operands are all differentiated renders as a bare name, so the non-differentiated split appears
        // in rendered programs exactly where it exists.
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_JVP_OPERATION_NAME)?;
        if self.non_differentiated_count == 0 {
            return Ok(());
        }
        operation.bracketed(|operation| operation.field("non_differentiated_count", self.non_differentiated_count))
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomJvpOperation<C::Type> {
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

impl<C: Context<Type: DifferentiableType>> PartiallyEvaluatableOperation<C> for CustomJvpOperation<C::Type>
where
    C::Operation: From<CustomJvpOperation<C::Type>>,
{
    // The default partial-evaluation rule is the desired one: interpret the primal region when every operand is known;
    // otherwise residualize the complete custom-JVP call so its attached derivative rule remains available to later
    // differentiation.
}

impl<T: DifferentiableType, C: Context<Type = T>, P: BatchingPolicy<C>> BatchableOperation<C, P>
    for CustomJvpOperation<T>
where
    C::Operation: From<CustomJvpOperation<T>>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Batch the two region contracts without opening the custom derivative:
        //
        //   primal: (p, x)    → y
        //   jvp:    (p, x, ẋ) → (y_jvp, ẏ).
        //
        // Each `ẋ` follows the batch axis of its corresponding `x`, while `p` has no tangent counterpart. The
        // ordinary primal, JVP-primal, and JVP-tangent computations may independently choose replicated or mapped
        // representations for the same logical output. Reconcile those three axes to one wrapper axis, align both
        // batched regions to it, and retain the custom-JVP carrier so differentiation performed after batching still
        // uses the user rule.
        //
        // A batching policy may add runtime boundary operands such as a first-class mapped extent. Those values must
        // reach both regions but have no derivative, so prepend them to `p` and increase `non_differentiated_count`.
        // Region adaptation owns any corresponding boundary rewrites; this rule owns only the flat operand split and
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

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C> for CustomJvpOperation<C::Type>
where
    C::Operation: ResidualZeroProvider<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Apply the user-supplied pushforward directly. For `f(p, x) = y`, region 1 implements
        //
        //   j(p, x, ẋ) = (f(p, x), (∂f/∂x)(p, x) · ẋ) = (y, ẏ).
        //
        // Feed every primal value, followed only by the differentiated inputs' tangents; a live tangent for `p` would
        // violate the declared non-differentiated boundary and is rejected below. Replay stages the rule's ordinary
        // primitive operations directly in the active context, so it introduces no symbolic capture. Consequently,
        // reverse mode can transpose the resulting linear map in `ẋ` exactly like any other tangent program, and
        // no nested differentiation request or special reverse rule is needed here.
        let jvp_region = driver.region(1)?;

        // The rule region is interpreted directly rather than routed through the transform rejections, so unresolved
        // state anywhere in its attached-region closure (including dormant nested rules) must be rejected here before
        // any of it can enter the differentiated program.
        if jvp_region.contains_effect_in_closure(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{CUSTOM_JVP_OPERATION_NAME}` rule regions must not contain unresolved state"),
            }
            .into());
        }

        let output_count = jvp_region.output_types().len() / 2;
        let (non_differentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;
        check_count!("input", jvp_region.input_types(), inputs.len() + differentiated_inputs.len(), ProgramError);

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

        // The JVP region consumes `(primals..., differentiated_input_tangents...)`, so feed every dual primal followed
        // by the differentiated duals' tangents.
        let mut jvp_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();

        // The user's JVP region takes every differentiated input tangent as a real region input, so materialize
        // structural zeros against their own primal, which names every runtime quantity a reference-bearing tangent
        // type omits; static inputs keep the nullary zero.
        for input in differentiated_inputs {
            jvp_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                context,
                input.tangent().clone(),
                std::iter::once(input.primal()),
            )?);
        }

        let mut outputs = jvp_region.interpret_in_context(context, jvp_inputs)?;
        check_count!("output", outputs, 2 * output_count, ProgramError);
        let tangents = outputs.split_off(output_count);
        Ok(outputs
            .into_iter()
            .zip(tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

// The raw carrier is intentionally non-transposable. Differentiation first replaces `f(p, x)` with the ordinary
// primitive program computing the linear map `ẋ ↦ (∂f/∂x)(p, x) · ẋ`; reverse mode transposes that replayed
// program, not `CustomJvpOperation`. Therefore only an invalid direct transpose of an un-linearized carrier can reach
// this rejection path.
impl_non_transposable_operation!(<T> CustomJvpOperation<T> where T: DifferentiableType);

/// Function with a user-supplied JVP rule, built by [`custom_jvp`]. It stores the primal and JVP closures together
/// with a phantom marker pinning the tracer-tree types named by those closure signatures; refer to the documentation
/// of [`custom_jvp`] for the calling convention, the tracing semantics, and when to reach for a custom JVP.
pub struct CustomJvp<Primal, Jvp, Inputs, Outputs> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: Primal,

    /// Closure computing `(outputs, output_tangents)` from `(inputs, input_tangents)`.
    jvp: Jvp,

    /// Phantom marker pinning the input and output tracer-tree types named by the closure signatures. The [`Domain`]
    /// whose universe the rules are traced into is recovered from the values passed to [`CustomJvp::call`], and so the
    /// wrapper stores neither a domain value nor a domain type witness.
    marker: PhantomData<fn() -> (Inputs, Outputs)>,
}

impl<Primal, Jvp, Inputs, Outputs> CustomJvp<Primal, Jvp, Inputs, Outputs>
where
    Primal: Fn(Inputs) -> Result<Outputs, ProgramError>,
    Jvp: Fn(Inputs, Inputs) -> Result<(Outputs, Outputs), ProgramError>,
{
    /// Stages this custom-JVP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Differentiation of the staged call replays the
    /// JVP rule instead of differentiating the primal body, in both forward and reverse mode.
    ///
    /// The [`Domain`] `D` whose universe the two rule programs are traced into is the
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
        D::Operation: From<CustomJvpOperation<D::Type>>,
        Inputs: Parameterized<DomainTracer<D>>,
        Inputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
        Outputs: Parameterized<DomainTracer<D>>,
        Outputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<V>,
        Inputs::To<D::Type>: Clone + Parameterized<D::Type, Family = Inputs::Family, To<DomainTracer<D>> = Inputs>,
        Outputs::To<D::Type>: Parameterized<D::Type, Family = Outputs::Family, To<DomainTracer<D>> = Outputs>,
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
            return Err(TypeError::invalid(format!("{CUSTOM_JVP_OPERATION_NAME} requires at least one input")).into());
        };
        let (_, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let input_tangent_types = input_types.clone().try_map_parameters(|r#type| r#type.tangent())?;
        let (output_types, jvp) = D::trace(|(x, t)| (self.jvp)(x, t), (input_types, input_tangent_types))?;
        let operation = D::Operation::from(CustomJvpOperation::new());
        // The call binds through whatever context the input values flow (a staged trace, a batching context, or a
        // JVP context), so `custom_jvp` composes under `vmap`/`jvp` — the batch/JVP rule of the bound operation fires.
        let context = first.dispatch_domain();
        let outputs = context.bind(operation, vec![primal.to_flat_program(), jvp.to_flat_program()], &input_values)?;
        let output_structure = output_types.0.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Creates a [`CustomJvp`] function from a primal closure and a Jacobian-Vector Product (JVP) closure over trees of
/// [`DomainTracer`]s. This is the ergonomic analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair.
///
/// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The two closure arguments implement
///
/// ```text
/// primal: x      ↦ y = f(x)
/// jvp:    (x, ẋ) ↦ (y, ẏ) = (f(x), J_f(x) · ẋ)
/// ```
///
/// Thus, `primal` receives the input tracer tree `x` and returns the output tracer tree `y`. `jvp` receives `x` and an
/// input-tangent tree `ẋ`, then returns the primal output `y` together with the Jacobian-vector product
/// `ẏ = J_f(x) · ẋ`. The tangent trees have the same parameter structures as their corresponding primal trees, and
/// Ryft validates these structural and type relationships when it traces the closures.
///
/// # When to use
///
/// Reach for a custom JVP when the function *is* forward-differentiable but its automatically derived tangent is
/// numerically unstable or wasteful and you want to supply a stable, efficient one by hand — classic cases are a
/// `log`-`sum`-`exp`, a softmax, or a normalization, where a hand-written tangent avoids the cancellation or redundant
/// work the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse mode obtains its
/// gradient by transposing the supplied tangent map, so the one rule composes with forward mode, reverse mode, and
/// their higher-order combinations. Prefer it over [`custom_vjp`](crate::differentiation::custom_vjp) whenever the
/// function is naturally forward-differentiable, and reach for [`custom_vjp`](crate::differentiation::custom_vjp) only
/// when just the reverse rule is natural (for example implicit differentiation or adjoint solvers).
///
/// # Calling convention
///
/// Both closures operate on [`Parameterized`] trees of [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so `x`
/// and `y` may each be a single tracer, a tuple, or any other parameterized structure. This high-level API does not
/// expose JAX's `nondiff_argnums` calling convention. Static non-differentiated configuration should be captured by
/// both closures. A dynamic typed value should remain an explicit input; its tangent is consequently present in `ẋ`,
/// and a rule that treats the value as a parameter ignores that tangent when constructing `ẏ`. Transform-injected
/// runtime metadata uses the lower-level
/// [`CustomJvpOperation::non_differentiated_count`] contract instead, because it must remain an SSA operand while
/// contributing no tangent slot.
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y`.
///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
///
/// # Tracing semantics
///
/// Nothing is traced at construction time: each [`CustomJvp::call`] recovers the tracing [`Domain`] from the values it
/// is called with, reads the input types off those arguments, traces both closures into programs specialized to those
/// types, validates the rule signature, and stages one [`CustomJvpOperation`] into the caller's staging context —
/// mirroring how JAX traces rule functions into jaxprs lazily at transform time. The primal closure is kept separate
/// from the JVP closure for efficiency rather than necessity: the JVP rule computes both the outputs and their
/// tangents, so deriving the primal from it would make every un-differentiated call pay for tangent computation.
/// Interpretation, batching, and backend lowering replay the lean primal program, and the JVP program runs only under
/// differentiation.
#[inline]
pub fn custom_jvp<Primal, Jvp, Inputs, Outputs>(primal: Primal, jvp: Jvp) -> CustomJvp<Primal, Jvp, Inputs, Outputs>
where
    Primal: Fn(Inputs) -> Result<Outputs, ProgramError>,
    Jvp: Fn(Inputs, Inputs) -> Result<(Outputs, Outputs), ProgramError>,
{
    CustomJvp { primal, jvp, marker: PhantomData }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayOperation, ArrayType, DataType, Dimension, Shape, ShardingDimension,
    };
    use crate::axes::AxisIndexOperation;
    use crate::batching::{
        Batch, BatchAxis, BatchingContext, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{Differentiate, ForwardModeDifferentiate, LinearizationTracer};
    use crate::operations::{
        Cos, CosOperation, Dot, DotDimensionNumbers, MulOperation, Reduce, ReductionKind, Sin, SinOperation,
    };
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

    /// Builds the deliberately wrong rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`, detectably different from the
    /// true derivative so tests can prove the custom rule is used.
    fn doubled_sin_jvp_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let dx = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x]).unwrap()[0];
        let two = builder.add_constant(Array::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Returns a custom-JVP call over `f(x) = sin(x)` together with its `["primal", "jvp"]` regions, whose JVP rule
    /// deliberately doubles the true derivative.
    fn custom_jvp_sin(
        r#type: &ArrayType,
    ) -> (ArrayOperation<Array>, Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>) {
        (
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            vec![sin_program(r#type), doubled_sin_jvp_program(r#type)],
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

    /// Builds the malformed rule `jvp(x, dx) = (sin(x), 1)`. Its tangent ignores `dx` and is therefore an affine
    /// constant rather than a linear tangent map.
    fn known_tangent_jvp_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x]).unwrap()[0];
        let tangent = builder.add_constant(Array::scalar(1.0));
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    #[test]
    fn test_custom_jvp() {
        let scalar = test_type(&[]);
        let operation = CustomJvpOperation::<ArrayType>::new();

        // Verify the operation's identity, rendering, and region contract: the primal program is a computation region
        // and the user JVP program is a rule region, in that order.
        assert_eq!(operation.name(), CUSTOM_JVP_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "custom_jvp");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("primal"), RegionSlot::rule("jvp")]);
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));

        // The primal region receives the call inputs and the JVP region receives `(inputs..., input tangents...)`.
        let primal_interface = sin_program(&scalar).interface();
        let jvp_interface = doubled_sin_jvp_program(&scalar).interface();
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Ok(vec![Some(vec![scalar.clone()]), Some(vec![scalar.clone(), scalar.clone()])]),
        );

        // A rule that satisfies the interface contract makes the call produce the primal outputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar), &[primal_interface, jvp_interface]),
            Ok(vec![scalar.clone()]),
        );

        // Inference maps the primal boundary through *differential* types rather than requiring the tangents to reuse
        // the primal storage type, so an `f8e8m0fnu` primal pairs with an `f32` tangent.
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(Vec::new()));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()));
        let differential_primal_interface =
            RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], Effects::PURE);
        let differential_jvp_interface = RegionInterface::new(
            vec![primal_type.clone(), tangent_type.clone()],
            vec![primal_type.clone(), tangent_type],
            Effects::PURE,
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[differential_primal_interface, differential_jvp_interface],
            ),
            Ok(vec![primal_type]),
        );

        // The JVP interface must be `(inputs..., input tangents...) → (outputs..., output tangents...)`; a
        // primal-shaped rule signature is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar),
                &[sin_program(&scalar).interface(), sin_program(&scalar).interface()],
            ),
            Err(TypeError::invalid(
                "custom_jvp rule input type signature mismatch: expected [f64[], f64[]] but got [f64[]]".to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_jvp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let (operation, operation_regions) = custom_jvp_sin(&scalar);

        // Interpretation replays the lean primal region only, so an un-differentiated call never pays for the JVP
        // program's tangent computation.
        let outputs = EagerContext::<Array, ArrayOperation<Array>>::new()
            .bind(operation, operation_regions, &[Array::scalar(2.0)])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_remains_opaque_to_partial_evaluation() {
        // A call with an unknown operand residualizes unchanged instead of inlining its primal region, which is what
        // keeps the custom rule attached to the residual program.
        let scalar = test_type(&[]);
        let (operation, operation_regions) = custom_jvp_sin(&scalar);
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
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomJvp(_)));
    }

    #[test]
    fn test_custom_jvp_batches_the_attached_regions() {
        let scalar = test_type(&[]);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| {
                    let (operation, operation_regions) = custom_jvp_sin(&scalar);
                    Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
                },
                Array::vector(vec![0.5, 1.0, 1.5]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        for (actual, input) in output.to_f64s().iter().zip([0.5f64, 1.0, 1.5]) {
            assert_abs_diff_eq!(*actual, input.sin(), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_custom_jvp_batching_preserves_and_reconciles_natural_output_axes() {
        let vector_type = test_type(&[3]);

        // The primal has one mapped identity output, one naturally replicated constant output, and one replicated
        // constant whose JVP tangent is mapped. The third pair forces reconciliation to mapped without forcing the
        // independently replicated second pair to acquire a batch axis.
        let primal = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]));
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]));
            builder.build(vec![input, replicated, reconciled], vec![Placeholder], vec![Placeholder; 3]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let tangent = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]));
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]));
            let zero = builder.add_constant(Array::vector(vec![0.0, 0.0, 0.0]));
            builder
                .build(
                    vec![input, replicated, reconciled, tangent, zero, tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 6],
                )
                .unwrap()
        };
        let program = wrapped_call_program(
            vec![vector_type],
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            vec![primal, jvp],
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

        let input = Array::matrix(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        assert_eq!(
            batched.interpret(vec![input.clone()]).unwrap(),
            vec![input, Array::vector(vec![4.0, 5.0, 6.0]), Array::matrix(3, 2, vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0]),],
        );
    }

    #[test]
    fn test_custom_jvp_batching_discovers_named_axis_outputs_with_replicated_inputs() {
        let scalar_type = test_type(&[]);
        let primal = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new())
                .unwrap()[0];
            builder.build(vec![index], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            builder.add_input(scalar_type);
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new())
                .unwrap()[0];
            let tangent = builder.add_constant(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
            builder.build(vec![index, tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
        };
        let regions = vec![primal, jvp];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("items".to_string());

        // No operand carries a mapped axis, but `axis_index("items")` observes the active transform inside both
        // regions and naturally produces a mapped output. Batching must therefore inspect the regions rather than
        // assuming that all-replicated wrapper inputs imply all-replicated wrapper outputs.
        let outputs = CustomJvpOperation::new()
            .batch(&context, &driver, &[ArrayBatch::replicated(Array::scalar(1.0))])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![0u64, 1, 2]));
    }

    #[test]
    fn test_custom_jvp_survives_batching_and_governs_the_batched_gradient() {
        // Differentiating *through* a batch of the custom call must still use the (deliberately doubled) custom
        // rule: batching preserves the call around batched programs instead of inlining the primal, so the
        // custom derivative survives `batch` — mirroring JAX's `vmap`-of-`custom_jvp` semantics.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::vector(vec![0.5, 1.0]))
            .value_and_gradient(|x| {
                let context = x.context().clone();
                let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                    &context,
                    |item| {
                        let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
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
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[1], 2.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_forward_mode() {
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x, ()| {
                    let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                    Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
                },
                Array::scalar(2.0),
                Array::scalar(1.0),
                (),
            )
            .unwrap();
        assert_abs_diff_eq!(primal.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom rule doubles the true derivative, proving it is in control.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(3.0))
            .value_and_gradient(|x| {
                let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                x.context().bind(operation, operation_regions, &[x.clone()]).unwrap().into_iter().next().unwrap()
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 3.0f64.sin(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_supports_second_order_differentiation() {
        // The JVP rule replays the user program as plain primitive operations, so the gradient program it produces is
        // itself differentiable and arbitrary-order differentiation composes (as it does through JAX's `custom_jvp`).
        // The doubled rule makes the first derivative `2 cos(x)`, so the second derivative is `-2 sin(x)`.
        let (gradient, second_derivative) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(0.7))
            .value_and_gradient(|x| {
                let context = x.context().clone();
                context
                    .differentiate_at(x)
                    .gradient(|y| {
                        let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                        y.context()
                            .bind(operation, operation_regions, &[y.clone()])
                            .unwrap()
                            .into_iter()
                            .next()
                            .unwrap()
                    })
                    .unwrap()
            })
            .unwrap();
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 0.7f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(second_derivative.to_f64s()[0], -2.0 * 0.7f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_rejects_known_tangent_outputs() {
        let r#type = test_type(&[]);
        let operation = ArrayOperation::CustomJvp(CustomJvpOperation::new());
        let operation_regions = || vec![sin_program(&r#type), known_tangent_jvp_program(&r#type)];
        let expected = "linearization produced a known tangent output; differentiation rules must represent \
                        input-independent zero tangents structurally";

        // Program-level direct linearization must reject the malformed rule rather than silently replacing its
        // constant tangent with zero.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let region_ids = operation_regions()
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(operation.clone(), region_ids, vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));

        // Value-level direct linearization enforces the same rule contract before exposing a reusable pushforward.
        let result = EagerContext::<Array, ArrayOperation<Array>>::new().linearize(
            |input, ()| {
                let mut outputs = input.context().bind(operation, operation_regions(), &[input.clone()])?;
                Ok(outputs.remove(0))
            },
            Array::scalar(2.0),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));
    }

    #[test]
    fn test_custom_jvp_wrapper_traces_closures_lazily() {
        // No manual programs: the wrapper traces the closures at the call site, specialized to the input types.
        let function = custom_jvp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, dx| {
                // The deliberately wrong rule `jvp(x, dx) = (sin(x), cos(x) * dx + cos(x) * dx)` doubles the true
                // derivative (expressed through addition to avoid constant lifting), proving the rule is in control.
                let tangent = x.cos()? * dx;
                Ok((x.sin()?, tangent.clone() + tangent))
            },
        );
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(|x, ()| function.call(x), Array::scalar(2.0), Array::scalar(1.0), ())
            .unwrap();
        assert_abs_diff_eq!(primal.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(3.0))
            .value_and_gradient(|x| function.call(x).unwrap())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 3.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_wrapper_surfaces_rule_signature_mismatches() {
        // Arity mismatches are compile-time errors under the structured signatures, but shape mismatches remain
        // runtime concerns: this rule produces a scalar tangent for a vector-valued function, so the traced JVP
        // program fails the signature validation that `CustomJvpOperation` performs at the call site.
        let function = custom_jvp(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, dx| {
                Ok((x.sin()?, dx.dot(&dx, &DotDimensionNumbers::inner_product())))
            },
        );
        let error =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), test_type(&[2])).unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "custom_jvp rule output type signature mismatch: expected [f64[2], f64[2]] but got [f64[2], f64[]]"
                    .to_string(),
            )),
        );
    }
}
