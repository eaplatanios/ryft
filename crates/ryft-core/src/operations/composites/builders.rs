use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    Batch, BatchAxis, BatchAxisSpecification, BatchableType, BatchingError, BatchingLevelExtent, BatchingPolicy,
    BatchingTracer, RecursiveBatchingPolicy, batch,
};
use crate::contexts::Context;
use crate::differentiation::{CotangentAccumulator, CotangentBatchingPolicy, DifferentiableType};
use crate::operations::composites::configurations::{
    AccumulatingVjp, BroadcastInvariantBatching, BroadcastSensitiveBatching, CompositeBatching, CompositeJvp,
    CompositeVjp, DefaultBatching, DefaultJvp, DefaultVjp, ExplicitJvp, ExplicitVjp, JvpFromPrimal, SequentialBatching,
    SymbolicZeroJvp, SymbolicZeroVjp,
};
use crate::operations::composites::operations::{COMPOSITE_OPERATION_NAME, CompositeOperation};
use crate::operations::composites::rules::{CustomRuleDefinition, CustomRuleReference, CustomRuleRegistration};
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::programs::{MaybeZero, ProgramError, Type, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

#[cfg(doc)]
use crate::batching::DiagonalBatchingPolicy;

#[cfg(doc)]
use crate::differentiation::TranspositionContext;

#[cfg(doc)]
use crate::operations::control_flow::scan::{ScanOperation, ScanType};

#[cfg(doc)]
use crate::programs::Program;

/// Collects the [`Context`] bounds required by [`Composite::call`], so that wrappers can reuse them. Implemented for
/// every context that satisfies these bounds. The call's input, output, and rule configuration bounds remain separate.
pub trait CompositeCallContext:
    NamedAxes
    + Context<
        Type: Eq
                  + Hash
                  + Send
                  + Sync
                  + Type<Identity: Send + Sync>
                  + DifferentiableType
                  + BatchableType<
            Policy: RecursiveBatchingPolicy<TracingContext<Self::Constant, Self::Operation>>
                        + CotangentBatchingPolicy<TracingContext<Self::Constant, Self::Operation>>,
        >,
        Constant: 'static + Send + Sync,
        Operation: 'static + Send + Sync + From<CompositeOperation<Self::Constant, Self::Operation>>,
    >
{
}

impl<C> CompositeCallContext for C where
    C: NamedAxes
        + Context<
            Type: Eq
                      + Hash
                      + Send
                      + Sync
                      + Type<Identity: Send + Sync>
                      + DifferentiableType
                      + BatchableType<
                Policy: RecursiveBatchingPolicy<TracingContext<C::Constant, C::Operation>>
                            + CotangentBatchingPolicy<TracingContext<C::Constant, C::Operation>>,
            >,
            Constant: 'static + Send + Sync,
            Operation: 'static + Send + Sync + From<CompositeOperation<C::Constant, C::Operation>>,
        >
{
}

// TODO(eaplatanios): Review from here onwards.

/// Retained definitions that one [`Composite`] registered, one per operation family, call structure (i.e., input and
/// output parameter structure), and visible named axes. The registration handles live as long as the composite, so
/// every call staged by the composite shares the specialization caches of its structure and named axes until the
/// composite is dropped.
///
/// Entries erase their constant, operation, and parameter structure types behind [`Any`], which requires those types
/// to be `'static`. Registrations must also be [`Send`] so this registry can be shared through its mutex.
#[derive(Default)]
struct CompositeRegistrations {
    /// Registrations, each a [`CompositeRegistration`] of some family and structure types.
    entries: Mutex<Vec<Box<dyn Any + Send>>>,
}

/// Retained definition that a [`Composite`] registered for one operation family and call structure.
struct CompositeRegistration<V: Typed + Parameter, O, InputStructure, OutputStructure> {
    /// Structure of the calls' inputs.
    input_structure: InputStructure,

    /// Structure of the calls' outputs.
    output_structure: OutputStructure,

    /// Named axes visible where the calls are made, with which the definition traces its rules.
    named_axes: Vec<(String, NamedAxis)>,

    /// Registration handle, which owns the definition's specialization caches.
    registration: CustomRuleRegistration<V, O>,
}

impl CompositeRegistrations {
    /// Returns a reference to the definition registered for the family `(V, O)`, the provided call structure, and the
    /// provided named axes, registering the definition returned by `register_fn` if there is none yet.
    fn get_or_register<V, O, InputStructure, OutputStructure, F>(
        &self,
        input_structure: &InputStructure,
        output_structure: &OutputStructure,
        named_axes: &[(String, NamedAxis)],
        register_fn: F,
    ) -> CustomRuleReference<V, O>
    where
        V: 'static + Typed<Type: Eq + Hash> + Parameter,
        O: 'static,
        InputStructure: 'static + Clone + PartialEq + Send,
        OutputStructure: 'static + Clone + PartialEq + Send,
        CustomRuleRegistration<V, O>: Send,
        F: FnOnce() -> CustomRuleDefinition<V, O>,
    {
        let mut entries = self.entries.lock().expect("composite registration mutex is poisoned");
        let existing = entries.iter().find_map(|entry| {
            entry
                .downcast_ref::<CompositeRegistration<V, O, InputStructure, OutputStructure>>()
                .filter(|entry| {
                    &entry.input_structure == input_structure
                        && &entry.output_structure == output_structure
                        && entry.named_axes == named_axes
                })
                .map(|entry| entry.registration.reference())
        });
        existing.unwrap_or_else(|| {
            let registration = CustomRuleRegistration::new(register_fn());
            let reference = registration.reference();
            entries.push(Box::new(CompositeRegistration {
                input_structure: input_structure.clone(),
                output_structure: output_structure.clone(),
                named_axes: named_axes.to_vec(),
                registration,
            }));
            reference
        })
    }
}

/// Function with custom derivative or batching rules, built by [`composite`]. It stores a primal closure together with
/// an optional forward mode rule (refer to [`Self::with_jvp`] and [`Self::with_jvp_from_primal`]), optional
/// reverse mode rules (refer to [`Self::with_vjp`]), and an optional batching rule (refer to [`Self::with_batching`]),
/// and each [`call`](Self::call) stages one [`CompositeOperation`] with retained rules over the traced primal. Without
/// a batching rule, a batched call maps every output at axis 0, even an output that does not vary across the batch,
/// unless [`Self::with_batched_output_axes`] declares the batch axes of its outputs.
///
/// The rules are retained callbacks rather than traced programs: a call traces only its primal, and each rule is traced
/// on the first derivative request of each specialization (e.g., each distinct input signature or batching level) and
/// cached, so a program that is never differentiated never traces its rules. Consequently, a malformed rule is reported
/// when it is first needed rather than when the function is called. The primal and the rules must be `'static`,
/// [`Send`], and [`Sync`], because the function stores them type-erased (i.e., as `Arc<dyn Fn(...) + Send + Sync>`)
/// and staged programs retain the rules. Type erasure lets the type of a configured function be named by its structural
/// types alone (e.g., as [`JvpComposite`] or [`VjpComposite`]), so that it can be stored in a struct field. Retaining
/// the closure types would buy nothing measurable, because a call only traces its primal and each rule is traced once
/// per specialization and cached.
///
/// The function owns the specialization caches of its calls: calls that share an operation family and a call
/// structure (i.e., input and output parameter structure) share one registered definition and its caches for as long
/// as the function lives. Once the function is dropped, the derivative requests of calls that it staged trace their
/// rules again on every request. The rules of a call cannot discharge reference state, so differentiating a call whose
/// reference state was discharged is rejected when its rule programs contain reference state (refer to
/// [`CustomRuleDefinition::with_reference_discharge`]).
///
/// The forward mode configuration is part of the type (i.e., `Jvp` is [`DefaultJvp`], [`ExplicitJvp`],
/// [`SymbolicZeroJvp`], or [`JvpFromPrimal`]), so an explicit forward mode rule and a forward mode rule derived from
/// the primal cannot both be configured, whatever the order of the configuration calls. Refer to the documentation of
/// the [`composite`] function for the rule selection of every configuration.
pub struct Composite<Input, Output, Jvp = DefaultJvp, Vjp = DefaultVjp, Batching = DefaultBatching> {
    /// Closure computing the primal output value from the primal input value. The [`Context`] whose universe the
    /// primal and the rules are traced into is recovered from the values passed to [`Composite::call`], and so the
    /// function stores neither a context value nor a context type witness.
    primal: Arc<dyn Fn(Input) -> Result<Output, ProgramError> + Send + Sync>,

    /// Forward mode configuration.
    jvp: Jvp,

    /// Reverse mode configuration.
    vjp: Vjp,

    /// Batching configuration.
    batching: Batching,

    /// Batch axes of the flattened output leaves of calls batched without a custom batching rule, which default to
    /// mapping every output leaf at axis 0 (refer to [`Composite::with_batched_output_axes`]).
    batched_output_axes: Option<Vec<BatchAxis>>,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Name of the function, used in rendering and diagnostics.
    name: Cow<'static, str>,

    /// Definitions registered by the calls of this function.
    registrations: CompositeRegistrations,
}

/// [`Composite`] with an explicit forward mode rule (refer to [`Composite::with_jvp`]), which governs both forward and
/// reverse mode differentiation, and with the default reverse mode and batching configurations. It names the type of
/// such a function by its input and output types alone (e.g., for storing it in a struct field).
pub type JvpComposite<Input, Output> = Composite<Input, Output, ExplicitJvp<Input, Output>>;

/// [`Composite`] with explicit reverse mode forward and backward rules (refer to [`Composite::with_vjp`]) whose
/// residuals have type `Residual`, and with the default forward mode and batching configurations (i.e., forward mode
/// rejects its calls). It names the type of such a function by its input, output, and residual types alone (e.g., for
/// storing it in a struct field).
pub type VjpComposite<Input, Output, Residual> =
    Composite<Input, Output, DefaultJvp, ExplicitVjp<Input, Output, Residual>>;

/// [`Composite`] with a custom batching rule (refer to [`Composite::with_batching`]) whose leaves are `Tracer`s, and
/// with the default forward mode and reverse mode configurations (i.e., derivatives are derived from the primal). It
/// names the type of such a function by its tracer, input, and output types alone (e.g., for storing it in a struct
/// field).
pub type BatchedComposite<Tracer, Input, Output> =
    Composite<Input, Output, DefaultJvp, DefaultVjp, BroadcastInvariantBatching<Tracer, Input, Output>>;

impl<V: CustomCall> Composite<Vec<V>, Vec<V>> {
    /// Creates a [`Composite`], without any custom rule, whose primal calls the foreign kernel described by `operation`
    /// (i.e., it stages a [`CustomCallOperation`] whose outputs have the operation's declared output types). Foreign
    /// kernels are opaque, so a custom call rejects differentiation, and this constructor is how a differentiable
    /// foreign kernel is defined: its configuration functions add the derivative rules (e.g., [`Self::with_vjp`]),
    /// which may themselves call foreign kernels. Without derivative rules, differentiating the function differentiates
    /// its primal, which reaches the custom call and reports its error. This is the analogue of wrapping JAX's
    /// [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html) with
    /// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) or
    /// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
    ///
    /// The function takes and returns vectors of values, and it is named after the operation's target, which labels its
    /// staged calls. Interpretation and backend lowering execute the kernel, while derivative requests replay the
    /// rules. The values must be arrays (i.e., implement [`CustomCall`]), so heterogeneous contexts (e.g., array IR
    /// contexts) call the function through their [`ProjectedContext`](crate::ProjectedContext) onto arrays, and every
    /// call requires at least one input, whose context the kernel call dispatches through.
    pub fn from_custom_call(operation: CustomCallOperation) -> Self {
        let name = operation.target_name().to_owned();
        composite(move |inputs: Vec<V>| CustomCall::custom_call(&operation, &inputs)).with_name(name)
    }
}

impl<Input, Output, Jvp, Vjp, Batching> Composite<Input, Output, Jvp, Vjp, Batching> {
    /// Returns this function with the leading `non_differentiated_count` flattened input leaves treated as
    /// non-differentiated _plumbing_ inputs. Refer to the documentation of the [`composite`] function for the semantics
    /// of non-differentiated inputs. Calls staged before this reconfiguration keep their original rules. The rules
    /// receive plumbing references by identity and cannot thread their state explicitly, so the reference state of a
    /// program that calls this function with a reference input cannot be discharged (refer to
    /// [`Program::discharge_references`]).
    ///
    /// # Calling Pattern
    ///
    /// A function that wraps a composite with plumbing inputs usually prepares its inputs before the call:
    ///
    ///   - **Plumbing First:** The plumbing leaves are a prefix of the flattened input leaves, so a wrapper whose own
    ///     signature takes them elsewhere reorders them to the front of the call's input.
    ///   - **Stopped Gradients:** A call that receives a non-zero tangent for a numeric plumbing input rejects it
    ///     rather than silently dropping it, because its rules declare no derivative with respect to that input. This
    ///     deliberately diverges from JAX, which stops the gradients of `nondiff_argnums` implicitly. A wrapper whose
    ///     plumbing inputs may depend on differentiated values therefore applies
    ///     [`stop_gradient`](crate::StopGradient::stop_gradient) to them first, which states the zero derivative
    ///     explicitly.
    ///   - **Aligned Manual Variation:** Inside manual sharding (e.g., in the body of a `shard_map`), a wrapper aligns
    ///     the manual variation of its inputs before the call (refer to
    ///     [`align_manual_variation`](crate::ManualVariationAlignment::align_manual_variation)), so that the transposes
    ///     of those alignments, which precede the call's boundary, sum the cotangent contributions of invariant inputs
    ///     exactly once. The call cannot align its inputs itself, because alignment
    ///     is a capability of the input values rather than of every context in which a composite can be called.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayOperation, DomainTracer, EagerContext, ProgramError, StopGradient, composite,
    /// #     differentiate_at,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
    ///
    /// // `(scale, x) ↦ scale · x`, whose leading `scale` leaf is plumbing, so the JVP rule ignores its tangent.
    /// let scaled = composite(|(scale, x): (Tracer, Tracer)| Ok(scale * x))
    ///     .with_jvp(|(scale, x), (_, tangent)| Ok((scale.clone() * x, scale * tangent)))
    ///     .with_non_differentiated_count(1);
    ///
    /// // Differentiating with respect to `scale` as well requires stopping its gradient before the call.
    /// let inputs = (Array::scalar(3.0f64)?, Array::scalar(2.0f64)?);
    /// let (scale_gradient, x_gradient) =
    ///     differentiate_at(inputs).gradient(|(scale, x)| scaled.call((scale.stop_gradient()?, x)))?;
    /// assert_eq!(scale_gradient, Array::scalar(0.0f64)?);
    /// assert_eq!(x_gradient, Array::scalar(3.0f64)?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_non_differentiated_count(self, non_differentiated_count: usize) -> Self {
        let Self { primal, jvp, vjp, batching, batched_output_axes, name, .. } = self;
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }

    /// Returns this function with the provided name, which labels its staged calls in rendering and diagnostics.
    /// Calls staged before this reconfiguration keep their original name.
    #[inline]
    pub fn with_name<N: Into<Cow<'static, str>>>(self, name: N) -> Self {
        let Self { primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, .. } = self;
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name.into())
    }

    /// Returns this function with the provided configurations and without registered definitions, because the
    /// definitions registered by earlier calls were configured with the previous configurations. The calls staged
    /// earlier keep their definitions, but since this drops the registrations that own their specialization caches,
    /// their derivative requests trace their rules again on every request (refer to [`Composite`]).
    fn with_configuration<NewJvp, NewVjp, NewBatching>(
        primal: Arc<dyn Fn(Input) -> Result<Output, ProgramError> + Send + Sync>,
        jvp: NewJvp,
        vjp: NewVjp,
        batching: NewBatching,
        batched_output_axes: Option<Vec<BatchAxis>>,
        non_differentiated_count: usize,
        name: Cow<'static, str>,
    ) -> Composite<Input, Output, NewJvp, NewVjp, NewBatching> {
        Composite {
            primal,
            jvp,
            vjp,
            batching,
            batched_output_axes,
            non_differentiated_count,
            name,
            registrations: CompositeRegistrations::default(),
        }
    }
}

impl<Input, Output, Vjp, Batching> Composite<Input, Output, DefaultJvp, Vjp, Batching> {
    /// Returns this function with the provided Jacobian-Vector Product (JVP) rule, which governs forward mode
    /// differentiation and, unless reverse mode rules are configured as well, reverse mode differentiation. This is the
    /// analogue of JAX's [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
    /// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair.
    ///
    /// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The rule implements:
    ///
    /// ```text
    /// JVP:        (x, ẋ) ↦ (y, ẏ) = (f(x), J_f(x) · ẋ)
    /// ```
    ///
    /// Thus, `jvp` receives the input value `x` and an input-tangent value `ẋ`, then returns the primal output `y`
    /// together with the Jacobian-vector product `ẏ = J_f(x) · ẋ`, which must be linear in `ẋ`. The tangent values have
    /// the same parameter structures as their corresponding primal values, and Ryft validates these structural and type
    /// relationships when it traces the rule. The rule keeps receiving a full `ẋ` value when the function has
    /// non-differentiated inputs, so that its signature mirrors the primal signature, but the tangent leaves of those
    /// inputs are placeholders that it must not use, because the staged [`CompositeOperation`] has no tangent slot
    /// for them. A rule that consumes or returns such a placeholder is rejected when it is first traced.
    ///
    /// # When to Use
    ///
    /// Reach for a custom JVP when the function _is_ forward-differentiable but its automatically derived tangent is
    /// numerically unstable or wasteful and you want to supply a stable, efficient one by hand. Classic cases are a
    /// `log`-`sum`-`exp`, a softmax, or a normalization, where a handwritten tangent avoids the cancellation or
    /// redundant work that the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse
    /// mode obtains its gradient by transposing the supplied tangent map, so the one rule composes with forward mode,
    /// reverse mode, and their higher-order combinations. Prefer it over [`Self::with_vjp`] whenever the function is
    /// naturally forward-differentiable, and use reverse mode rules only when just the reverse rule is natural (e.g.,
    /// for implicit differentiation or adjoint solvers).
    ///
    /// # Transform Semantics
    ///
    /// Differentiation replays the rule program instead of differentiating the primal. The replayed rule consists of
    /// ordinary primitive operations, so reverse mode transposes the linear map in `ẋ` that it computes, exactly like
    /// any other tangent program, and the staged call itself is never transposed. Tangent conversions in the rule must
    /// therefore be transposable: a rule that converts a tangent to another type representation (e.g., `f32[n]` to
    /// `f32[2]`) uses [`cast_unchecked`](crate::Cast::cast_unchecked) with witnesses that carry the identities that the
    /// conversion drops (e.g., the corresponding primal input) and leaves the check to the primal conversion, because a
    /// checked conversion of a tangent checks a linear value, which reverse mode cannot transpose when no known value
    /// carries the extents of the resulting cotangent. Higher-order differentiation differentiates those replayed
    /// operations, so every combination of forward and reverse mode applies to the resulting derivatives, including
    /// differentiation of a pullback with respect to its cotangent seeds and transposing a pullback again (which
    /// restores the tangent map). The primal is kept separate from the rule for efficiency rather than necessity: the
    /// rule computes both the outputs and their tangents, so deriving the primal from it would make every
    /// un-differentiated call pay for tangent computation.
    ///
    /// There is no analogue of JAX's [`defjvps`][defjvps], which takes one closure per argument and sums their
    /// contributions. Inputs here are arbitrary [`Parameterized`] structures rather than positional arguments, so a
    /// closure per input leaf would have to be indexed by flattened leaf position, which discards the structure that
    /// the rule otherwise receives, and one rule over the structured tangent `ẋ` already expresses the same derivative
    /// by summing the per-leaf contributions itself. A rule that should skip the contributions of inactive leaves, as
    /// `defjvps` does for the arguments that are not differentiated, can use [`Self::with_symbolic_zero_jvp`] instead.
    ///
    /// [defjvps]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvps.html
    ///
    /// # Parameters
    ///
    ///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
    #[inline]
    pub fn with_jvp<Jvp: 'static + Fn(Input, Input) -> Result<(Output, Output), ProgramError> + Send + Sync>(
        self,
        jvp: Jvp,
    ) -> Composite<Input, Output, ExplicitJvp<Input, Output>, Vjp, Batching> {
        let Self { primal, vjp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        let jvp = ExplicitJvp { jvp: Arc::new(jvp) };
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }

    /// Returns this function with a forward mode rule derived from its primal: forward mode differentiates the primal,
    /// reusable linearization partitions that derivative once, and reverse mode transposes it unless reverse mode rules
    /// are configured as well. A function without derivative rules already derives its forward mode rule from its
    /// primal, so this configuration is needed only to pair reverse mode rules with the derived forward mode rule,
    /// which would otherwise make forward mode reject the calls.
    #[inline]
    pub fn with_jvp_from_primal(self) -> Composite<Input, Output, JvpFromPrimal, Vjp, Batching> {
        let Self { primal, vjp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        Self::with_configuration(
            primal,
            JvpFromPrimal,
            vjp,
            batching,
            batched_output_axes,
            non_differentiated_count,
            name,
        )
    }

    /// Returns this function with the provided Jacobian-Vector Product (JVP) rule `(x, ẋ) ↦ (y, ẏ)`, which is used as
    /// the rule of [`Self::with_jvp`] except that it receives `ẋ` and returns `ẏ` with [`MaybeZero`] leaves (i.e., as
    /// `Input::To<MaybeZero<Tracer>>` and `Output::To<MaybeZero<Tracer>>` values, such as
    /// `(MaybeZero<DomainTracer<C>>, MaybeZero<DomainTracer<C>>)` for an input `(DomainTracer<C>, DomainTracer<C>)`).
    /// The leaves of structurally zero input tangents (e.g., those of inputs that are not being differentiated, and
    /// those of every non-differentiated input) are [`MaybeZero::Zero`]s, which lets the rule skip the work that they
    /// would otherwise require, and the rule returns a [`MaybeZero::Zero`] of the output's tangent type for an output
    /// whose tangent it knows to be zero (e.g., an output that does not depend on the active inputs), which then stays
    /// a structural zero for the transforms that consume the tangent (e.g., reverse mode transposes no work for it).
    /// Each pattern of structurally zero input tangents is a separate specialization of the rule. This is the analogue
    /// of JAX's `defjvp(..., symbolic_zeros=True)`, including its symbolic-zero output tangents.
    ///
    /// `Tracer` must usually be annotated, because it cannot be inferred from `Input` (a [`Parameterized`] structure
    /// is generic over its leaf type): annotate the tangent parameter of the closure (e.g., `tangent:
    /// MaybeZero<Tracer>`) or pass `Tracer` explicitly. A `Tracer` that differs from the tracers of the context in
    /// which the function is called is rejected at compile time by [`Self::call`] rather than by this function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayOperation, Cos, DomainTracer, EagerContext, MaybeZero, ProgramError, Sin, Typed, composite,
    /// #     differentiate_at,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
    ///
    /// // `x ↦ (sin(x), x)` whose rule declares the tangent of its second output to be a structural zero (e.g., because
    /// // that output only reports the point at which the first one was evaluated).
    /// let function = composite(|x: Tracer| Ok((x.sin()?, x))).with_symbolic_zero_jvp(
    ///     |x: Tracer, tangent: MaybeZero<Tracer>| {
    ///         let sine_tangent = match tangent {
    ///             MaybeZero::Value(tangent) => MaybeZero::Value(x.cos()? * tangent),
    ///             MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type),
    ///         };
    ///         let zero = MaybeZero::Zero(x.r#type().into_owned());
    ///         Ok(((x.sin()?, x), (sine_tangent, zero)))
    ///     },
    /// );
    /// let gradient = differentiate_at(Array::scalar(0.5f64)?).gradient(|x| {
    ///     let (sine, point) = function.call(x)?;
    ///     Ok(sine + point)
    /// })?;
    /// assert_eq!(gradient, Array::scalar(0.5f64.cos())?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_symbolic_zero_jvp<Tracer, Jvp>(
        self,
        jvp: Jvp,
    ) -> Composite<Input, Output, SymbolicZeroJvp<Tracer, Input, Output>, Vjp, Batching>
    where
        Tracer: Typed + Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Jvp: 'static
            + Fn(Input, Input::To<MaybeZero<Tracer>>) -> Result<(Output, Output::To<MaybeZero<Tracer>>), ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, vjp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        let jvp = SymbolicZeroJvp { jvp: Arc::new(jvp) };
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }
}

impl<Input, Output, Jvp, Batching> Composite<Input, Output, Jvp, DefaultVjp, Batching> {
    /// Returns this function with the provided reverse mode forward and backward (i.e., Vector-Jacobian Product or VJP)
    /// rules, which govern reverse mode differentiation and take precedence over any forward mode rule for it. This is
    /// the analogue of JAX's [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
    /// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair.
    ///
    /// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The VJP, or pullback, maps an output
    /// cotangent `ȳ` to the input cotangent `x̄ = J_f(x)ᵀ · ȳ`. The two rules factor that computation through a
    /// residual value `r`:
    ///
    /// ```text
    /// Forward:      x      ↦ (y, r) = (f(x), r)
    /// Backward:     (r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ
    /// ```
    ///
    /// Thus, `forward` recomputes `y` and saves exactly the residual value `r` needed by the reverse rule, and
    /// `backward` receives `r` and the output-cotangent value `ȳ`, then returns the input-cotangent value `x̄`. When
    /// tracing, Ryft validates matching parameter structures and the corresponding primal, residual, and cotangent
    /// types. The caller must ensure that the primal and `forward` compute the same value of `y`; tracing cannot
    /// establish their numerical equivalence. A dynamic value that `backward` needs must be preserved as a residual in
    /// `r`, and `backward` returns a zero cotangent for an input that it treats as a parameter.
    ///
    /// The primal is kept separate from `forward` for efficiency rather than necessity: an un-differentiated call
    /// should not pay for residual computation. Callers that do not care about the distinction can pass the same body
    /// for both, accepting that the residual outputs are dead code outside of differentiation (e.g., by writing
    /// `forward` as `|x| Ok((f(x)?, residuals))`).
    ///
    /// # When to Use
    ///
    /// Reach for reverse mode rules when only the _reverse_ rule is natural, or when the function is not (efficiently)
    /// forward-differentiable. Common cases are:
    ///
    ///   - **Implicit Differentiation:** Differentiate through a solver, optimizer, or fixed point via the implicit
    ///     function theorem rather than unrolling its iterations.
    ///   - **Adjoint Methods:** Backpropagate through an Ordinary Differential Equation (ODE) or Partial Differential
    ///     Equation (PDE) solution via the adjoint system instead of differentiating the individual steps of the
    ///     integrator.
    ///   - **External or Black-Box Calls:** Supply the reverse rule for a custom kernel or for a computation that has
    ///     no automatic derivative (refer to [`Self::from_custom_call`]). The call must still trace as a Ryft operation
    ///     with interpretation and backend lowering support; the closures cannot execute arbitrary external code on
    ///     tracer values.
    ///   - **Numerical Stability:** Replace an unstable or wasteful automatically derived gradient with a handwritten
    ///     one.
    ///
    /// Without a forward mode rule, active forward mode differentiation and reusable forward linearization reject a
    /// call before executing its forward rule, because no pushforward rule was supplied. This rejection also occurs
    /// when constructing staged forward derivatives, rather than waiting for their execution. Calls with no active
    /// derivative inputs can still execute their primal directly. Add a forward mode rule (e.g., with
    /// [`Self::with_jvp`] or [`Self::with_jvp_from_primal`]) when callers need direct forward mode differentiation as
    /// well as reverse mode.
    ///
    /// # Non-Differentiated Inputs and References
    ///
    /// Non-differentiated inputs reach the primal and `forward` at their usual positions and receive no cotangent:
    /// `backward` keeps returning a full `x̄` value so that its signature mirrors the primal signature, but the leaves
    /// that it returns at non-differentiated positions are ignored, because the staged [`CompositeOperation`]
    /// produces no cotangents for them. A non-differentiated value that `backward` needs must be forwarded to it as a
    /// residual in `r`.
    ///
    /// `forward` may return a non-differentiated reference inside `r`, in which case the reference itself (rather
    /// than a snapshot of its contents) is forwarded to `backward`. Every reference-typed residual must be a distinct
    /// non-differentiated input forwarded by identity, which the first reverse mode derivative request validates when
    /// it traces `forward`. This interface supports externally owned mutable state, not privately allocated reference
    /// residuals; save immutable values when the backward rule needs a snapshot. This enables the _stash-gradients_
    /// pattern: a `stash` reference enters as a non-differentiated input, `forward` returns it as a residual, and
    /// `backward` writes the incoming `ȳ` into it before returning `x̄`.
    ///
    /// # Transform Semantics
    ///
    /// Reverse mode traces `forward` when it linearizes a call and `backward` when it transposes the resulting carrier.
    /// It replays the forward program for the primal outputs and residuals and stages a
    /// [`CompositeTransposeOperation`](crate::CompositeTransposeOperation) carrier whose transpose replays the backward
    /// program, so reverse mode uses exactly the user-supplied gradient, and the original call is never transposed. A
    /// call batched before it is differentiated batches both rule programs traced at the unbatched types and sums the
    /// cotangents that the batched backward program produces for replicated inputs over the batch axis. By default,
    /// the forward pass runs `forward` and keeps its residuals alive for the backward pass, while
    /// [`Composite::with_deferred_forward`] (i.e., the analogue of JAX's `optimize_remat` option) runs the primal in
    /// the forward pass instead and defers `forward` to the backward pass that needs its residuals.
    ///
    /// Higher-order differentiation differentiates the backward program's ordinary operations, because the pullback
    /// replays that program inline. Both forward and reverse mode apply to a pullback, with respect to the primal
    /// inputs (through the saved residuals) and to the cotangent seeds, even when the backward program is nonlinear in
    /// its seeds. This includes reverse over reverse and forward over reverse differentiation, but it does not supply a
    /// forward rule for a nested reverse-only call encountered along either path. Transposing a pullback again requires
    /// the backward program to be linear in its seeds and is rejected otherwise.
    ///
    /// # Parameters
    ///
    ///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse mode residual production.
    ///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
    #[inline]
    pub fn with_vjp<Residual, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> Composite<Input, Output, Jvp, ExplicitVjp<Input, Output, Residual>, Batching>
    where
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static + Fn(Residual, Output) -> Result<Input, ProgramError> + Send + Sync,
    {
        let Self { primal, jvp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        let vjp = ExplicitVjp { forward: Arc::new(forward), backward: Arc::new(backward), deferred_forward: false };
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }

    /// Returns this function with the provided reverse mode forward rule `x ↦ (y, r)` and backward rule `(r, ȳ) ↦ x̄`,
    /// which are used as the rules of [`Self::with_vjp`] except that the backward rule receives `ȳ` and returns `x̄`
    /// with [`MaybeZero`] leaves (i.e., as `Output::To<MaybeZero<Tracer>>` and `Input::To<MaybeZero<Tracer>>` values).
    /// The leaves of structural-zero cotangent seeds (e.g., those of unused outputs) are [`MaybeZero::Zero`]s, and the
    /// rule returns a [`MaybeZero::Zero`] of an input's cotangent type for an input whose cotangent it knows to be zero
    /// (e.g., an input that it treats as a parameter), which then contributes nothing to that input's cotangent and
    /// stays a structural zero instead of being materialized and accumulated. This is the analogue of JAX's
    /// `defvjp(..., symbolic_zeros=True)` together with a backward rule that returns `None` or a `SymbolicZero` for an
    /// input, except that the forward rule receives no per-input differentiation flags (JAX's `perturbed` flags).
    ///
    /// `Tracer` must usually be annotated, because it cannot be inferred from `Input` or `Output` (a [`Parameterized`]
    /// structure is generic over its leaf type): annotate the seed parameter of the backward closure (e.g., `seed:
    /// MaybeZero<Tracer>`) or pass `Tracer` explicitly. A `Tracer` that differs from the tracers of the context in
    /// which the function is called is rejected at compile time by [`Composite::call`] rather than by this function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayOperation, DomainTracer, EagerContext, MaybeZero, ProgramError, Typed, composite,
    /// #     differentiate_at,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
    ///
    /// // `(scale, x) ↦ scale · x`, whose backward rule treats `scale` as a parameter by returning a structural zero
    /// // for its cotangent instead of computing `x · ȳ`.
    /// let scaled = composite(|(scale, x): (Tracer, Tracer)| Ok(scale * x)).with_symbolic_zero_vjp(
    ///     |(scale, x): (Tracer, Tracer)| Ok((scale.clone() * x.clone(), (scale, x))),
    ///     |(scale, x): (Tracer, Tracer), seed: MaybeZero<Tracer>| {
    ///         let scale_cotangent = MaybeZero::Zero(scale.r#type().into_owned());
    ///         let x_cotangent = match seed {
    ///             MaybeZero::Value(seed) => MaybeZero::Value(scale * seed),
    ///             MaybeZero::Zero(_) => MaybeZero::Zero(x.r#type().into_owned()),
    ///         };
    ///         Ok((scale_cotangent, x_cotangent))
    ///     },
    /// );
    /// let inputs = (Array::scalar(3.0f64)?, Array::scalar(2.0f64)?);
    /// let (scale_gradient, x_gradient) = differentiate_at(inputs).gradient(|inputs| scaled.call(inputs))?;
    /// assert_eq!(scale_gradient, Array::scalar(0.0f64)?);
    /// assert_eq!(x_gradient, Array::scalar(3.0f64)?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_symbolic_zero_vjp<Tracer, Residual, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> Composite<Input, Output, Jvp, SymbolicZeroVjp<Tracer, Input, Output, Residual>, Batching>
    where
        Tracer: Typed + Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static
            + Fn(Residual, Output::To<MaybeZero<Tracer>>) -> Result<Input::To<MaybeZero<Tracer>>, ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        let vjp = SymbolicZeroVjp { forward: Arc::new(forward), backward: Arc::new(backward), deferred_forward: false };
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }

    /// Returns this function with the provided reverse mode forward rule `x ↦ (y, r)` and accumulating backward rule,
    /// which govern reverse mode differentiation as the rules of [`Self::with_vjp`] do, except that the backward rule
    /// submits each input cotangent to its [`CotangentAccumulator`] instead of returning it. One rule therefore serves
    /// every destination kind of an input's cotangent: it can update only the affected entries of a caller-provided
    /// buffer (refer to [`CotangentAccumulator::reference`]) instead of returning a full-sized cotangent, and it can
    /// skip the inputs whose cotangents are ignored (refer to [`CotangentAccumulator::is_needed`]). The rule is
    /// specialized once per pattern of destination kinds, and the buffers themselves are never part of a
    /// specialization.
    ///
    /// The backward rule is invoked while the call's carrier is transposed, with the [`TranspositionContext`], the
    /// residuals `r` (with the structure that the forward rule returned), the output cotangent seeds with
    /// [`MaybeZero`] leaves (i.e., as an `Output::To<MaybeZero<Tracer>>` value), and the accumulators with
    /// [`CotangentAccumulator`] leaves (i.e., as an `Input::To<CotangentAccumulator>` value; the accumulators of
    /// non-differentiated inputs are never needed). It returns a [`ProgramError`] like the other rules. An accumulator
    /// to which the rule submits nothing (or only [`MaybeZero::Zero`]s, which emit no work) leaves its input's
    /// cotangent a structural zero, so skipping an accumulator is how this rule declares a zero cotangent (i.e., the
    /// analogue of a JAX backward rule returning `None` for an input, as [`Self::with_symbolic_zero_vjp`] expresses
    /// with [`MaybeZero::Zero`]).
    ///
    /// `Tracer` and `Transposition` must usually be annotated, because neither can be inferred from `Input` or
    /// `Output` (a [`Parameterized`] structure is generic over its leaf type): annotate the transposition context and
    /// seed parameters of the backward closure (e.g., as `&mut TranspositionContext<..>` and `MaybeZero<Tracer>`) or
    /// pass them explicitly. Types that differ from those of the context in which the function is called are rejected
    /// at compile time by [`Composite::call`] rather than by this function.
    #[inline]
    pub fn with_accumulating_vjp<Tracer, Transposition, Residual, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> Composite<Input, Output, Jvp, AccumulatingVjp<Tracer, Transposition, Input, Output, Residual>, Batching>
    where
        Tracer: Typed + Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static
            + Fn(
                &mut Transposition,
                Residual,
                Output::To<MaybeZero<Tracer>>,
                Input::To<CotangentAccumulator>,
            ) -> Result<(), ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, batching, batched_output_axes, non_differentiated_count, name, .. } = self;
        let vjp = AccumulatingVjp { forward: Arc::new(forward), backward: Arc::new(backward), deferred_forward: false };
        Self::with_configuration(primal, jvp, vjp, batching, batched_output_axes, non_differentiated_count, name)
    }
}

impl<Input, Output, Jvp, Residual, Batching>
    Composite<Input, Output, Jvp, ExplicitVjp<Input, Output, Residual>, Batching>
{
    /// Returns this function with its reverse mode forward rule deferred to the backward pass, which is the analogue of
    /// JAX's `custom_vjp(..., optimize_remat=True)`. Reverse mode then computes the outputs of each call with the
    /// primal, rather than with `forward`, and keeps only the call's inputs alive for the backward pass, where it runs
    /// `forward` on them to recover the residuals `r` right before `backward` consumes them. This trades recomputing
    /// `forward` for the memory that its residuals would otherwise occupy between the two passes, and it pays off when
    /// the primal is cheaper than `forward` (e.g., when computing the residuals is the expensive part of `forward`).
    ///
    /// The swap applies wherever reverse mode partitions a call (e.g., in [`vjp`](crate::ReverseModeDifferentiate::vjp)
    /// and in rematerialized bodies, refer to [`rematerialize`](crate::rematerialize)), so a residual policy that saves
    /// everything still saves only the call's inputs, while one that saves nothing recomputes those inputs. Forward
    /// mode, batching, and interpretation are unaffected, and so is a call whose unused outputs were pruned (refer to
    /// [`CustomRuleDefinition::with_deferred_forward_rule`]). Higher-order differentiation differentiates the replayed
    /// forward and backward programs like the rest of the pullback.
    ///
    /// The same function is available for [`SymbolicZeroVjp`] and [`AccumulatingVjp`] configurations (i.e., after
    /// [`Composite::with_symbolic_zero_vjp`] or [`Composite::with_accumulating_vjp`]). Calls staged before this
    /// reconfiguration keep their original rules.
    ///
    /// # Errors
    ///
    /// `forward` runs in the backward pass, where its effects would execute out of order with the rest of the program,
    /// so reverse mode rejects a call whose `forward` has effects (e.g., a `print` or a reference update) with a
    /// [`ProgramError::UnsupportedOperation`] that names this function. Batching a call before differentiating it is
    /// supported, but batching a reverse mode derivative after it is taken and before it is transposed (e.g., batching
    /// a linearization) is rejected with a [`BatchingError::UnsupportedOperation`] that names this function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayOperation, Cos, DomainTracer, EagerContext, ProgramError, Sin, composite};
    /// # use ryft_core::differentiate_at;
    /// # fn main() -> Result<(), ProgramError> {
    /// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
    ///
    /// // `sin` whose forward rule saves `cos(x)` as its residual, which the deferral recomputes in the backward pass.
    /// let sine = composite(|x: Tracer| Ok(x.sin()?))
    ///     .with_vjp(|x: Tracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent))
    ///     .with_deferred_forward();
    /// let x = Array::scalar(0.5f64)?;
    /// let (value, pullback) = differentiate_at(x.clone()).vjp(|x| sine.call(x))?;
    /// assert_eq!(value, Array::scalar(0.5f64.sin())?);
    /// assert_eq!(pullback.residuals(), &[x]);
    /// assert_eq!(pullback.apply_materialized(Array::scalar(1.0f64)?)?, Array::scalar(0.5f64.cos())?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_deferred_forward(mut self) -> Self {
        self.vjp.deferred_forward = true;
        self.registrations = CompositeRegistrations::default();
        self
    }
}

impl<Tracer, Input, Output, Jvp, Residual, Batching>
    Composite<Input, Output, Jvp, SymbolicZeroVjp<Tracer, Input, Output, Residual>, Batching>
where
    Tracer: Typed + Parameter,
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Returns this function with its reverse mode forward rule deferred to the backward pass, exactly as for the
    /// rules of [`Composite::with_vjp`] (refer to the [`ExplicitVjp`] form of this function for the semantics).
    #[inline]
    pub fn with_deferred_forward(mut self) -> Self {
        self.vjp.deferred_forward = true;
        self.registrations = CompositeRegistrations::default();
        self
    }
}

impl<Tracer, Transposition, Input, Output, Jvp, Residual, Batching>
    Composite<Input, Output, Jvp, AccumulatingVjp<Tracer, Transposition, Input, Output, Residual>, Batching>
where
    Tracer: Typed + Parameter,
    Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Returns this function with its reverse mode forward rule deferred to the backward pass, exactly as for the
    /// rules of [`Composite::with_vjp`] (refer to the [`ExplicitVjp`] form of this function for the semantics).
    #[inline]
    pub fn with_deferred_forward(mut self) -> Self {
        self.vjp.deferred_forward = true;
        self.registrations = CompositeRegistrations::default();
        self
    }
}

impl<Input, Output, Jvp, Vjp> Composite<Input, Output, Jvp, Vjp, DefaultBatching> {
    /// Returns this function with the provided custom batching rule `(x, x_axes, extent) ↦ (y, y_axes)`, which batches
    /// its calls instead of structurally batching its primal (i.e., the analogue of JAX's
    /// [`custom_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html)). The rule
    /// receives the batched inputs (as values of the context in which the call is batched), their [`BatchAxis`]es
    /// (i.e., an `Input::To<BatchAxis>` value), and the extent of the batch axis (i.e., [`BatchingLevelExtent::Static`]
    /// for a host extent, or [`BatchingLevelExtent::Dynamic`] with a first-class extent value). It returns the batched
    /// outputs together with their batch axes (i.e., an `Output::To<BatchAxis>` value). A replicated input axis means
    /// that the input is the same for every batch item. Broadcasting a replicated input must preserve the rule's
    /// per-item numerical behavior.
    ///
    /// `Tracer` must usually be annotated, because it cannot be inferred from `Input` or `Output` (a [`Parameterized`]
    /// structure is generic over its leaf type): annotate the extent parameter of the closure (e.g., as
    /// `BatchingLevelExtent<Tracer>`) or pass `Tracer` explicitly. A `Tracer` that differs from the tracers of the
    /// context in which the function is called is rejected at compile time by [`Composite::call`] rather than by this
    /// function. [`BatchedComposite`] names the type of a function with only a batching rule.
    ///
    /// The rule is traced when a call is batched, once per batching level and input signature, and its program becomes
    /// the batched call's primal, so the rule takes precedence over structurally batching the primal (e.g., over the
    /// batching strategy of a foreign kernel called by [`Self::from_custom_call`]). Batching preserves the call and its
    /// derivative rules, and the rule must compute the batched primal consistently with them.
    ///
    /// # Derivatives
    ///
    /// How derivatives interact with the rule depends on the forward mode rule (refer to [`composite`] for the rule
    /// selection):
    ///
    ///   - **Derived from the primal** (i.e., without derivative rules, or with [`Self::with_jvp_from_primal`]): the
    ///     rule stays on the path of forward mode derivatives in either order of the two transforms, which is the
    ///     analogue of JAX's `custom_vmap_jvp`. Differentiating a batched call differentiates the rule's program, and
    ///     differentiating an unbatched call stages a derived call whose batching applies the derivative of the rule,
    ///     including when a forward mode Jacobian batches it or when a linearized pushforward is batched. The derived
    ///     rule computes the primal outputs with the rule applied at the batch axes of the primal inputs, exactly as
    ///     batching the call does, and the tangents with the derivative of that application, after aligning each
    ///     tangent to its primal (broadcasting or moving it), so the batched derivative has the batch axes of the
    ///     batched call. A replicated primal whose tangent is mapped is the exception: the rule cannot perturb a shared
    ///     input differently for each batch item, so the tangents come from a second application of the rule with that
    ///     primal broadcast. That gives the per-item derivatives of the batched primal for every rule that only
    ///     computes the batched primal for the batch axes that it receives, because such a rule computes the same
    ///     values whether the primal is broadcast or not. A rule whose result depends on its input axes beyond that
    ///     should use [`Self::with_broadcast_sensitive_batching`] instead, which recovers the exact per-item tangents
    ///     of such inputs from the outer product of their tangents' batch with the rule's batch (i.e., the diagonal of
    ///     the outer product that JAX returns for them when another input is mapped). When every primal is replicated
    ///     (e.g., in a forward mode Jacobian), it applies the rule with every input replicated and batches the rule's
    ///     derivative structurally, so the rule must also accept calls without mapped inputs. Linearization computes
    ///     the outputs with the call itself and recomputes the primal inside the staged pushforward, because the call
    ///     is opaque to partial evaluation, which is only valid for a primal without effects: a primal with effects
    ///     (e.g., one that updates or reads references) is linearized inline, so that its effects run once, and
    ///     batching that linearization's pushforward batches it structurally. Transposing a linearization transposes
    ///     the primal of its staged pushforward, so the rule does not apply to the transpose. Reverse mode therefore
    ///     inlines the derivative of the primal instead of staging a derived call.
    ///   - **Explicit** (i.e., [`Self::with_jvp`] or reverse mode rules): differentiating a batched call traces the
    ///     derivative rules at the unbatched types and batches them structurally, aligned to the batch axes that the
    ///     rule declared, and batching a derivative batches the explicit rule structurally, which is JAX's
    ///     `custom_jvp(custom_vmap(f))` nesting order. The other order nests two functions: an outer function with the
    ///     batching rule, whose primal calls an inner function with the derivative rule.
    #[inline]
    pub fn with_batching<Tracer, Rule>(
        self,
        rule: Rule,
    ) -> Composite<Input, Output, Jvp, Vjp, BroadcastInvariantBatching<Tracer, Input, Output>>
    where
        Tracer: Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Rule: 'static
            + Fn(
                Input,
                Input::To<BatchAxis>,
                BatchingLevelExtent<Tracer>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, vjp, non_differentiated_count, name, .. } = self;
        let batching = BroadcastInvariantBatching { rule: Arc::new(rule) };
        Self::with_configuration(primal, jvp, vjp, batching, None, non_differentiated_count, name)
    }

    /// Returns this function with the provided custom batching rule `(x, x_axes, extent) ↦ (y, y_axes)`, which is used
    /// as the rule of [`Self::with_batching`] except that its per-item numerical behavior may change when a replicated
    /// input is broadcast (e.g., a rule that applies a different approximation when an input is replicated). This only
    /// changes how derivatives derived from the primal are batched (refer to the derivatives section of
    /// [`Self::with_batching`]): when a replicated primal has a mapped tangent while another input is mapped, the
    /// rule is still applied once, at the primal inputs' own batch axes, and the per-item tangents of such inputs are
    /// recovered exactly from the outer product of their batch of tangents with the batch of the rule, instead of from
    /// a second application of the rule with that primal broadcast. The tangents are then exactly the per-item
    /// derivatives of the batched primal, which is also the diagonal of the outer product that JAX's `custom_vmap`
    /// returns for such inputs.
    ///
    /// The outer product costs one application of the rule's derivative with respect to those inputs per batch item,
    /// which is why a rule that only computes the batched primal (and therefore gives the same values when an input is
    /// broadcast) should use [`Self::with_batching`] instead. The family's batching policy must support taking
    /// diagonals (refer to [`DiagonalBatchingPolicy`]). `Tracer` must usually be annotated, as for
    /// [`Self::with_batching`].
    #[inline]
    pub fn with_broadcast_sensitive_batching<Tracer, Rule>(
        self,
        rule: Rule,
    ) -> Composite<Input, Output, Jvp, Vjp, BroadcastSensitiveBatching<Tracer, Input, Output>>
    where
        Tracer: Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Rule: 'static
            + Fn(
                Input,
                Input::To<BatchAxis>,
                BatchingLevelExtent<Tracer>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, vjp, non_differentiated_count, name, .. } = self;
        let batching = BroadcastSensitiveBatching { rule: Arc::new(rule) };
        Self::with_configuration(primal, jvp, vjp, batching, None, non_differentiated_count, name)
    }

    /// Returns this function with sequential batching, which batches its calls by looping over the batch items with a
    /// scan (refer to [`ScanOperation`]) whose body applies the unbatched call to one item at a time, instead of
    /// structurally batching its primal. This is the analogue of JAX's [`custom_batching.sequential_vmap`][sequential],
    /// and a convenience over [`Self::with_batching`] for primals that have no batched form (e.g., a foreign kernel
    /// called by [`Self::from_custom_call`] that processes one item at a time): the mapped inputs are moved to axis 0
    /// and sliced by the scan, the replicated inputs are passed to every iteration unchanged, and every output is
    /// mapped at axis 0. The batch extent must be statically known, including when represented by a first-class
    /// dimension, and the context's family must support scans (i.e., its types implement [`ScanType`] and its
    /// operations contain [`ScanOperation`]). The batch axis must be unnamed because the scan does not bind named
    /// collectives across its items for the primal or derivative rules.
    ///
    /// Unlike a custom batching rule, which replaces the call's primal while preserving the call, batching replaces
    /// the call with the scan, whose body calls the unbatched function. The function's derivative rules therefore keep
    /// governing every item: differentiating the batched call differentiates the scan, whose body applies the
    /// function's rules one item at a time, and a derivative derived from the primal that is batched after it is taken
    /// is batched sequentially as well (i.e., JAX's `custom_vmap_jvp` behavior for `sequential_vmap`). Batching the
    /// scan again at an enclosing batching level batches its body, whose call is again batched sequentially, so nested
    /// batching yields nested scans.
    ///
    /// [sequential]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.sequential_vmap.html
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     Array, ArrayOperation, BatchAxis, DomainTracer, EagerContext, ProgramError, Sin, batch, composite,
    /// # };
    /// # fn main() -> Result<(), ProgramError> {
    /// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
    ///
    /// // `x ↦ sin(x) · y` for one item at a time, with `y` shared by every batch item.
    /// let function = composite(|(x, y): (Tracer, Tracer)| Ok(x.sin()? * y)).with_sequential_batching();
    /// let outputs: Array = batch(
    ///     |inputs| function.call(inputs),
    ///     (Array::vector(vec![0.5f64, 1.0])?, Array::scalar(2.0f64)?),
    ///     (BatchAxis::new(0), BatchAxis::replicated()),
    ///     BatchAxis::new(0),
    ///     None,
    /// )?;
    /// assert_eq!(outputs, Array::vector(vec![2.0 * 0.5f64.sin(), 2.0 * 1.0f64.sin()])?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_sequential_batching(self) -> Composite<Input, Output, Jvp, Vjp, SequentialBatching> {
        let Self { primal, jvp, vjp, non_differentiated_count, name, .. } = self;
        Self::with_configuration(primal, jvp, vjp, SequentialBatching, None, non_differentiated_count, name)
    }

    /// Returns this function with the provided batch axes for the outputs of its batched calls, one per flattened
    /// output leaf. Without a custom batching rule, batching a call structurally batches its primal and maps every
    /// output at axis 0 by default, which is always valid but broadcasts outputs that do not vary across the batch. The
    /// declared axes apply at every batching level: the batched primal is aligned to them, and so is each derivative
    /// rule batched for the call. A custom batching rule declares the axes of its outputs itself, so configuring one
    /// (refer to [`Self::with_batching`]) discards these axes. Their count is validated when a call is batched.
    ///
    /// # Parameters
    ///
    ///   - `output_axes`: Batch axis of each flattened output leaf, in leaf order.
    #[inline]
    pub fn with_batched_output_axes(self, output_axes: Vec<BatchAxis>) -> Self {
        let Self { primal, jvp, vjp, batching, non_differentiated_count, name, .. } = self;
        Self::with_configuration(primal, jvp, vjp, batching, Some(output_axes), non_differentiated_count, name)
    }
}

impl<Input, Output, Jvp, Vjp, Batching> Composite<Input, Output, Jvp, Vjp, Batching> {
    /// Stages one call of this function on the provided `input` value and returns its output value. Refer to the
    /// documentation of the [`composite`] function for the tracing semantics and for how the transforms treat the
    /// staged call.
    ///
    /// The [`Context`] `C` whose universe the rules are traced into is the [`Domain`](Value::Domain)
    /// of the values in `input`, which is exactly the context the call is staged into. It is therefore never named at
    /// a construction or call site, while the stored closures still pin the tracers that this universe must produce.
    ///
    /// The primal and the rules are traced in fresh traces rather than in `C` itself, so each trace is seeded with the
    /// named axes in scope in `C` (refer to [`NamedAxes::named_axes`]). Code in the closures therefore resolves the
    /// axes of enclosing transforms (e.g., an `axis_index` or a collective over the axis of an enclosing `batch`) as it
    /// would if it were inlined. Rules traced under different bindings may differ, so calls under different named axes
    /// register separate rule definitions.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves, when the non-differentiated count exceeds the number of
    /// input leaves, when tracing the primal closure fails, or when the staged [`CompositeOperation`] rejects the
    /// call (e.g., because the call violates the reference contract). The rules are validated when they are first
    /// traced, so their errors are reported by the derivative requests that trace them.
    pub fn call<
        V: Value<Type = C::Type, Domain = C>,
        C: CompositeCallContext<Value = V>,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        CustomRuleDefinition<C::Constant, C::Operation>: CompositeJvp<Jvp, DomainTracer<C>, Input, Output>
            + CompositeVjp<Vjp, DomainTracer<C>, Input, Output>
            + CompositeBatching<Batching, DomainTracer<C>, Input, Output>,
        Input: Parameterized<DomainTracer<C>>,
        Input::ParameterStructure: 'static + PartialEq + Send,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::ParameterStructure: 'static + PartialEq + Send,
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
            return Err(TypeError::invalid(format!("`{}` requires at least one input", self.name)).into());
        };
        let input_count = input_values.len();
        if self.non_differentiated_count > input_count {
            return Err(TypeError::invalid(format!(
                "`{}` non-differentiated input count {} exceeds input count {input_count}",
                self.name, self.non_differentiated_count,
            ))
            .into());
        }

        // Only the primal is traced now. The rules are registered with the call structure and traced lazily. The
        // primal and the rules are traced in fresh traces, which are seeded with the named axes that are visible where
        // the function is called (e.g., the axis of an enclosing batching level), so that they resolve the same names
        // as the function would if it were inlined. Rules traced under different bindings may differ, so the named
        // axes are part of the registration key.
        let named_axes = first.domain().named_axes();
        let (output_types, primal) =
            DomainTracingContext::<C>::trace_with_named_axes(&*self.primal, input_types.clone(), named_axes.clone())?;
        let input_structure = input_types.parameter_structure();
        let output_structure = output_types.parameter_structure();
        let rules = self.registrations.get_or_register(&input_structure, &output_structure, &named_axes, || {
            let definition = CustomRuleDefinition::new(self.name.clone())
                .with_named_axes(named_axes.clone())
                .with_batching_support();
            let definition =
                definition.configure_jvp(&self.jvp, &input_structure, &output_structure, self.non_differentiated_count);
            let definition =
                definition.configure_vjp(&self.vjp, &input_structure, &output_structure, self.non_differentiated_count);

            // Every transform without a configured rule treats the primal as it treats any other function, so a
            // function without derivative rules differentiates its primal. A function with only reverse mode rules
            // keeps rejecting forward mode instead, because a derived pushforward would contradict its pullback.
            let definition = match definition.has_derivative_rules() {
                true => definition,
                false => definition.with_jvp_from_primal(),
            };
            let definition = match &self.batched_output_axes {
                Some(output_axes) => definition.with_batched_output_axes(output_axes.clone()),
                None => definition,
            };
            definition.configure_batching(&self.batching, &input_structure, &output_structure)
        });

        // The call binds through whatever context the input values flow through (e.g., a staging trace, a batching
        // context, or a differentiation context), so the batching or differentiation rule of the bound operation fires
        // and the function composes with those transforms.
        let operation = CompositeOperation::new(rules).with_non_differentiated_count(self.non_differentiated_count)?;
        let outputs = first.domain().bind(operation, vec![primal.into_flat_program()], input_values.as_slice())?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

impl<Input, Output, Jvp: Debug, Vjp: Debug, Batching: Debug> Debug for Composite<Input, Output, Jvp, Vjp, Batching> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures are opaque, so the configurations print their kinds only.
        formatter
            .debug_struct("Composite")
            .field("name", &self.name)
            .field("jvp", &self.jvp)
            .field("vjp", &self.vjp)
            .field("batching", &self.batching)
            .field("batched_output_axes", &self.batched_output_axes)
            .field("non_differentiated_count", &self.non_differentiated_count)
            .finish_non_exhaustive()
    }
}

/// Creates a [`Composite`] from a primal closure `x ↦ y = f(x)` over values of [`DomainTracer`]s, without any custom
/// rule. It is the one entry point for custom rules of every transform, and its configuration functions add them, where
/// each configuration is the analogue of one JAX entry point: [`Composite::with_jvp`] of
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html), [`Composite::with_vjp`] of
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html), and
/// [`Composite::with_batching`] of
/// [`jax.custom_batching.custom_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html),
/// and they combine in one registration:
///
///   - [`Composite::with_jvp`] adds a Jacobian-Vector Product (JVP) rule `(x, ẋ) ↦ (y, ẏ)`, and
///     [`Composite::with_symbolic_zero_jvp`] adds one that receives structural-zero tangents,
///   - [`Composite::with_jvp_from_primal`] derives the forward mode rule from the primal instead,
///   - [`Composite::with_vjp`] adds reverse mode forward and backward rules `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`,
///     [`Composite::with_symbolic_zero_vjp`] adds ones whose backward rule receives structural-zero seeds, and
///     [`Composite::with_accumulating_vjp`] adds ones whose backward rule submits its cotangents to their destinations
///     (e.g., caller-provided buffers) instead of returning them, and
///   - [`Composite::with_batching`] adds a custom batching rule, which batches calls instead of structurally batching
///     their primal, and [`Composite::with_sequential_batching`] batches calls by looping over their batch items.
///
/// [`Composite::from_custom_call`] creates a function whose primal calls a foreign kernel. Every transform without a
/// configured rule treats the primal as it treats any other function, so a function without derivative rules (e.g., one
/// with only a batching rule, which is JAX's bare `custom_vmap`) differentiates its primal. The one exception is a
/// function with only reverse mode rules, which rejects forward mode (as JAX's `custom_vjp` does), because a derived
/// pushforward would contradict its custom pullback. Each derivative mode selects exactly one rule:
///
/// | Configuration                          | Forward mode   | Reverse mode                          | Batching      |
/// |----------------------------------------|----------------|---------------------------------------|---------------|
/// | primal only                            | derived JVP    | transposed derived linearization      | structural    |
/// | `with_batching`                        | derived JVP    | transposed derived linearization      | batching rule |
/// | `with_jvp_from_primal`                 | derived JVP    | transposed derived linearization      | structural    |
/// | `with_jvp`                             | JVP rule       | transposed JVP rule linearization     | structural    |
/// | `with_vjp`                             | rejected       | forward and backward rules            | structural    |
/// | `with_jvp` and `with_vjp`              | JVP rule       | forward and backward rules            | structural    |
/// | `with_jvp_from_primal` and `with_vjp`  | derived JVP    | forward and backward rules            | structural    |
///
/// Any of the forms of a forward mode or reverse mode rule listed above takes the place of `with_jvp` or `with_vjp` in
/// this table, respectively, and `with_batching` combines with every configuration, replacing structural batching. The
/// user is responsible for the numerical consistency of the rules (e.g., that a forward rule computes the same outputs
/// as the primal, that a JVP rule and reverse mode rules describe the same derivative, or that a batching rule computes
/// the batched primal). Higher-order differentiation differentiates the selected rule programs rather than switching
/// back to the primal.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayOperation, BatchAxis, BatchingLevelExtent, Cos, DomainTracer, EagerContext, ProgramError, Sin,
/// #     batch, composite, differentiate_at,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
///
/// // `sin` with a JVP rule for forward mode and reverse mode rules that save `cos(x)` as their residual.
/// let sine = composite(|x: Tracer| Ok(x.sin()?))
///     .with_jvp(|x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
///     .with_vjp(|x| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
/// let (_, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| sine.call(x))?;
/// assert_eq!(tangent, Array::scalar(0.5f64.cos())?);
/// let gradient = differentiate_at(Array::scalar(0.5f64)?).gradient(|x| sine.call(x))?;
/// assert_eq!(gradient, Array::scalar(0.5f64.cos())?);
///
/// // A custom batching rule receives the batched input and its batch axis, and it declares the batch axis of the
/// // output (i.e., the output is mapped exactly when the input is).
/// let square = composite(|x: Tracer| Ok(x.clone() * x))
///     .with_batching(|x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.clone() * x, axis)));
/// let squares: Array = batch(
///     |x| square.call(x),
///     Array::vector(vec![1.0f64, 2.0, 3.0])?,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     None,
/// )?;
/// assert_eq!(squares, Array::vector(vec![1.0f64, 4.0, 9.0])?);
/// # Ok(())
/// # }
/// ```
///
/// # Calling Convention
///
/// All closures operate on [`Parameterized`] values of [`DomainTracer`]s (i.e., Ryft's analogue of JAX pytrees), so
/// inputs, outputs, and residuals may each be a single tracer, a tuple, or any other parameterized structure. Tangents
/// and cotangents have the parameter structures of their primal values. Because this function builds a reusable
/// function before any input is known, the primal closure must annotate the tracer type of its input (e.g.,
/// `|x: DomainTracer<C>| ...`), which then also fixes the input types of the rule closures. Static configuration
/// should be captured by the closures, while a dynamic value should remain an explicit input, either as a
/// non-differentiated input (see below) or as an ordinary input whose derivative the rules treat as zero.
///
/// # Non-Differentiated Inputs
///
/// [`Composite::with_non_differentiated_count`] declares the leading flattened input leaves as _plumbing_ that
/// parameterizes the call without being differentiated. These leaves remain dynamic inputs. Unlike JAX's
/// `nondiff_argnums`, this is a prefix of flattened leaves, not a set of static argument positions. Plumbing leaves
/// reach every closure at their usual positions, and the rules keep signatures that mirror the primal signature: the
/// tangent leaves that a JVP rule receives for them are placeholders that it must not use, and the cotangent leaves
/// that a backward rule returns for them are ignored (refer to [`Composite::with_jvp`] and [`Composite::with_vjp`]).
/// Differentiating a call with a non-zero tangent for a numeric plumbing input is rejected because the rules cannot
/// propagate that tangent (refer to [`Composite::with_non_differentiated_count`] for how callers prepare such inputs).
///
/// # References
///
/// A reference-typed input is accepted only as plumbing, where every closure that receives it may read or write it.
/// An active reference input is rejected, because the rule interfaces define no tangent or cotangent reference for it
/// and so user-supplied rules could not express its derivative. A live tangent reference supplied for a plumbing input
/// by an enclosing transform is left untouched by the call, since the rules declare no derivative through the state it
/// denotes. No output may be a reference, because the rules would then have to produce that output's tangent
/// reference or consume its cotangent reference. The closures may also allocate and use local reference state, which
/// executes like any other primitive operation whenever the corresponding program is replayed. When the call is
/// differentiated, no two reference inputs may bind the same allocation.
///
/// # Tracing Semantics
///
/// Nothing is traced at construction time. Each [`Composite::call`] recovers the tracing [`Context`] from the values
/// that it is called with, traces only the primal closure at their types, and stages one [`CompositeOperation`]
/// that retains the rules into the context through which those values flow. The rules are traced lazily, at the types
/// of each specialization (e.g., each distinct input signature): forward mode traces the JVP rule, reverse mode traces
/// the forward rule when it stages its reverse mode carrier and the backward rule when that carrier is transposed, and
/// batching traces the batching rule. Tracing a rule also validates its signature, so a malformed rule is reported by
/// the request that first needs it rather than by the call. The traced programs are cached with the function, and a
/// program that is never differentiated never traces its derivative rules (refer to [`Composite`] for the ownership of
/// that cache).
///
/// # Transform Semantics
///
/// The transforms treat a staged call as follows:
///
///   - _interpretation_ and backend lowering replay the lean primal program only,
///   - _partial evaluation_ folds a call when its inputs are known and effect-ordering constraints permit execution,
///     and it otherwise preserves the call and its retained rules for later differentiation,
///   - _batching_ batches the primal (with the batching rule, if there is one, and otherwise mapping every output at
///     axis 0 unless [`Composite::with_batched_output_axes`] declares other batch axes), preserves the call around the
///     batched primal program, and records the batching level, and a later derivative request batches the derivative
///     rule programs traced at the unbatched types, aligned to the batch axes of the batched call's outputs, so that
///     the custom derivative survives batching applied _before_ differentiation, while a batching rule survives
///     batching applied _after_ forward mode differentiation (refer to [`Composite::with_batching`]), and
///   - _differentiation_ replays the selected derivative rule programs instead of differentiating the primal (refer to
///     [`Composite::with_jvp`] and [`Composite::with_vjp`] for the semantics of each mode, including higher-order
///     differentiation), and
///   - _dead code elimination_ (i.e., [`Program::into_pruned`]) prunes the unused outputs of a call from its primal
///     while keeping every input, as JAX does for its custom derivative calls, and the call's rules are projected onto
///     its remaining outputs, except that a call whose forward mode rule is derived from its primal and that has a
///     custom batching rule stays whole.
///
/// # Differences from JAX
///
/// Beyond the transforms that JAX lacks (e.g., reverse mode through a custom batching rule), the following differences
/// are deliberate:
///
///   - **Partial Evaluation:** A call with unknown inputs is kept whole as a residual, together with its retained
///     rules, rather than having the work that depends only on its known inputs hoisted out of it, because its rules
///     are specialized for the complete call.
///   - **Foreign Kernels:** A function created by [`Composite::from_custom_call`] needs at least one input, whose
///     context the kernel call dispatches through.
///   - **Differentiation Flags:** The reverse mode forward rule receives no per-input differentiation flags (i.e.,
///     JAX's `perturbed` flags), while the backward rule can still skip structural-zero seeds and return
///     structural-zero cotangents (refer to [`Composite::with_symbolic_zero_vjp`]).
///   - **Rematerialization:** [`Composite::with_deferred_forward`], the analogue of JAX's `optimize_remat` option,
///     swaps the primal into the forward pass wherever reverse mode partitions a call, rather than only where dead code
///     elimination finds the forward rule's residuals unused (e.g., inside a rematerialized body), and it also supports
///     structural-zero and accumulating backward rules and higher-order differentiation, which JAX's option rejects.
///   - **Non-Differentiated Inputs:** Non-differentiated inputs are a prefix of flattened leaves, which must receive
///     stopped gradients, rather than static argument positions whose gradients are stopped implicitly (refer to
///     [`Composite::with_non_differentiated_count`]).
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y`.
#[inline]
pub fn composite<Input, Output, Primal: 'static + Fn(Input) -> Result<Output, ProgramError> + Send + Sync>(
    primal: Primal,
) -> Composite<Input, Output> {
    Composite {
        primal: Arc::new(primal),
        jvp: DefaultJvp,
        vjp: DefaultVjp,
        batching: DefaultBatching,
        batched_output_axes: None,
        non_differentiated_count: 0,
        name: Cow::Borrowed(COMPOSITE_OPERATION_NAME),
        registrations: CompositeRegistrations::default(),
    }
}

/// Checks on sample values whether the batching rule behind `function` depends on its input batch axes beyond computing
/// the batched function, which is what decides between [`Composite::with_batching`] and
/// [`Composite::with_broadcast_sensitive_batching`] (the former is exact only for rules that do not). The function is
/// batched once at `input_batch_axes`, and then once for each replicated input leaf with that leaf broadcast along the
/// batch instead. Every batched output is mapped at axis `0`, and the outputs of each broadcast batch are compared with
/// those of the first batch by `equivalent` (e.g., exact equality, or equality within a numerical tolerance for a rule
/// that uses a different but equivalent algorithm). This is the batching counterpart of
/// [`check_gradient!`](crate::check_gradient): it observes the rule only through the public [`batch`] transform.
///
/// Passing the check on samples does not prove that the rule is independent of its axes, but failing it proves that it
/// is not, in which case the function's derivatives batched with a replicated primal whose tangent is mapped require
/// [`Composite::with_broadcast_sensitive_batching`] to be the per-item derivatives of its batched primal.
///
/// # Parameters
///
///   - `function`: Function to batch, which typically calls a [`Composite`] with a custom batching rule.
///   - `input`: Sample input of `function`, with every mapped leaf packed along its batch axis.
///   - `input_batch_axes`: Batch axis of each input leaf, with the structure of `input`.
///   - `batch_axis`: Batch axis specification passed to every [`batch`] call. It must supply the batch extent when
///     every input leaf is replicated.
///   - `equivalent`: Comparison deciding whether two batched output leaves agree.
///
/// # Errors
///
/// Returns [`BatchingError::BroadcastInvarianceViolation`] for the first replicated input leaf whose broadcast changes
/// a batched output leaf. Failures during batching propagate as their existing [`BatchingError`] variants.
pub fn check_batching_rule_consistency<V, F, Input, Output, Specification, Equivalent>(
    function: F,
    input: Input,
    input_batch_axes: Input::To<BatchAxis>,
    batch_axis: Specification,
    equivalent: Equivalent,
) -> Result<(), BatchingError>
where
    V: Value<Domain: Batch>,
    F: Fn(Input::To<BatchingTracer<V::Domain, <V::Domain as Batch>::Policy>>) -> Result<Output, ProgramError>,
    Input: Parameterized<
            V,
            Family: ParameterizedFamily<BatchAxis>
                        + ParameterizedFamily<BatchingTracer<V::Domain, <V::Domain as Batch>::Policy>>,
        >,
    Output: Parameterized<
            BatchingTracer<V::Domain, <V::Domain as Batch>::Policy>,
            Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<V>,
        >,
    Specification: Into<BatchAxisSpecification<<<V::Domain as Batch>::Policy as BatchingPolicy<V::Domain>>::Extent>>,
    Equivalent: Fn(&V, &V) -> bool,
{
    let batch_axis = batch_axis.into();
    let structure = input.parameter_structure();
    let inputs = input.into_parameters().collect::<Vec<_>>();
    let input_axes = input_batch_axes.into_parameters().collect::<Vec<_>>();
    let batched = |inputs: Vec<V>, input_axes: Vec<BatchAxis>| -> Result<Vec<V>, BatchingError> {
        let input = Input::from_parameters(structure.clone(), inputs)?;
        let input_axes = Input::To::<BatchAxis>::from_parameters(structure.clone(), input_axes)?;
        let outputs = batch(|input| function(input), input, input_axes, BatchAxis::new(0), batch_axis.clone())?;
        Ok(outputs.into_parameters().collect())
    };
    let expected = batched(inputs.clone(), input_axes.clone())?;
    for (input, axis) in input_axes.iter().enumerate() {
        if !axis.is_replicated() {
            continue;
        }
        // The leaf is broadcast by a batch of the complete input that returns only that leaf, so that its extent is
        // inferred exactly as in the batches that it is compared with.
        let input_value = Input::from_parameters(structure.clone(), inputs.clone())?;
        let axes = Input::To::<BatchAxis>::from_parameters(structure.clone(), input_axes.clone())?;
        let broadcast: V = batch(
            |tracers: Input::To<BatchingTracer<V::Domain, <V::Domain as Batch>::Policy>>| {
                Ok(tracers.into_parameters().nth(input).unwrap())
            },
            input_value,
            axes,
            BatchAxis::new(0),
            batch_axis.clone(),
        )?;
        let mut broadcast_inputs = inputs.clone();
        broadcast_inputs[input] = broadcast;
        let mut broadcast_axes = input_axes.clone();
        broadcast_axes[input] = BatchAxis::new(0);
        let actual = batched(broadcast_inputs, broadcast_axes)?;
        if let Some(output) = expected.iter().zip(&actual).position(|(expected, actual)| !equivalent(expected, actual))
        {
            return Err(BatchingError::BroadcastInvarianceViolation { input, output });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArraySliceAxis, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, Shape,
        ShardingDimension,
    };
    use crate::axes::AxisError;
    use crate::batching::{
        BatchAxis, BatchAxisSpecification, BatchedProgram, BatchingError, ProgramBatchingOutputAxesPolicy, batch,
    };
    use crate::contexts::{Context, EagerContext, ProjectedContext};
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationError, DifferentiationRule,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::check_gradient;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::collectives::axis_index::AxisIndex;
    use crate::operations::collectives::parallel_reduce::ParallelReduce;
    use crate::operations::composites::rules::CustomRuleSource;
    use crate::operations::constants::zero::Zero;
    use crate::operations::custom_call::CustomCallBatching;
    use crate::operations::differentiation::stop_gradient::StopGradient;
    use crate::operations::dimensions::dimension_size::DimensionSize;
    use crate::operations::dot::{Dot, DotDimensionNumbers};
    use crate::operations::manipulation::broadcasting::DynamicBroadcast;
    use crate::operations::manipulation::conversions::{Cast, ConvertElementType};
    use crate::operations::manipulation::padding::PadOperation;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceNew, ReferenceRead, ReferenceWrite,
    };
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::parameters::Placeholder;
    use crate::programs::{FlatProgram, MaybeZero, ReferenceType, ValueProjection};
    use crate::tracing::Trace;

    use super::*;

    /// Eager context whose values are arrays.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager array IR context whose values may be arrays or references.
    type EagerArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer of [`EagerArrayIrContext`] values.
    type ArrayIrTracer = DomainTracer<EagerArrayIrContext>;

    /// Returns `sin` with a custom batching rule that computes `2 · sin(x)` instead (expressed through addition to
    /// avoid constant lifting), which makes visible whether a batched computation and each of its derivatives go
    /// through the rule. The rule maps its output exactly when its input is mapped.
    fn doubled_sine_when_batched()
    -> BatchedComposite<DomainTracer<ArrayContext>, DomainTracer<ArrayContext>, DomainTracer<ArrayContext>> {
        type Tracer = DomainTracer<ArrayContext>;
        composite(|x: Tracer| Ok(x.sin()?))
            .with_name("sine")
            .with_batching(|x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.sin()? + x.sin()?, axis)))
    }

    /// Returns `x · y` with a custom batching rule that computes `2 · x · y` instead (refer to
    /// [`doubled_sine_when_batched`]), whose output is mapped along the first mapped input axis.
    fn doubled_product_when_batched() -> BatchedComposite<
        DomainTracer<ArrayContext>,
        (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>),
        DomainTracer<ArrayContext>,
    > {
        type Tracer = DomainTracer<ArrayContext>;
        composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_name("product").with_batching(
            |(x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                let product = x * y;
                Ok((product.clone() + product, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        )
    }

    /// Returns the elementwise values `2 · sin(x)` and `2 · cos(x)` of `x`.
    fn doubled_sine_and_cosine(x: &[f64]) -> (Array, Array) {
        (
            Array::vector(x.iter().map(|x| 2.0 * f64::sin(*x)).collect()).unwrap(),
            Array::vector(x.iter().map(|x| 2.0 * f64::cos(*x)).collect()).unwrap(),
        )
    }

    #[test]
    fn test_composite_from_custom_call() {
        // The primal calls the foreign kernel `my_sin`, which the reference backend cannot execute, while the rules
        // are ordinary programs, so both differentiation modes execute without calling the kernel.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function = Composite::from_custom_call(CustomCallOperation::new("my_sin", vec![scalar_type.clone()]))
            .with_jvp(|inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
                Ok((vec![inputs[0].sin()?], vec![inputs[0].cos()? * tangents[0].clone()]))
            })
            .with_vjp(
                |inputs: Vec<Tracer>| Ok((vec![inputs[0].sin()?], inputs[0].cos()?)),
                |cosine, cotangents: Vec<Tracer>| Ok(vec![cosine * cotangents[0].clone()]),
            );
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"my_sin\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = custom_call [target=my_sin] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(2.0).unwrap()]),
            Err(ProgramError::UnsupportedOperation {
                message: "the reference array backend cannot execute the foreign kernel `my_sin`".to_string(),
            }),
        );
        assert_eq!(
            differentiate_at(vec![Array::scalar(2.0).unwrap()])
                .jvp(vec![Array::scalar(1.0).unwrap()], |inputs| function.call(inputs)),
            Ok((vec![Array::scalar(2.0f64.sin()).unwrap()], vec![Array::scalar(2.0f64.cos()).unwrap()])),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(vec![x]).map(|outputs| { outputs.into_iter().next().unwrap() })),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_composite_from_custom_call_batching() {
        // Batching a function built from a foreign kernel defers to the kernel's own strategy unless the function has a
        // custom batching rule, which takes precedence and here calls a batched kernel instead.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let axes = [BatchAxis::new(0)];
        let natural = ProgramBatchingOutputAxesPolicy::Natural;
        let operation = CustomCallOperation::new("my_sin", vec![scalar_type.clone()]);
        let jvp = |inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
            Ok((vec![inputs[0].sin()?], vec![inputs[0].cos()? * tangents[0].clone()]))
        };

        // The kernel rejects batching by default.
        let function = Composite::from_custom_call(operation.clone()).with_jvp(jvp);
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type.clone()]).unwrap();
        assert!(matches!(
            program.into_flat_program().batched(3, ShardingDimension::Replicated, &axes, natural.clone()),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `my_sin` has no batching rule for input 0 mapped at batch axis 0; invoke \
                               a kernel that understands the batch axis, or select an explicit batching behavior with \
                               `CustomCallOperation::with_batching`",
        ));

        // Its broadcasting strategy applies when the function has no batching rule.
        let function = Composite::from_custom_call(operation.clone().with_batching(CustomCallBatching::BroadcastAll))
            .with_jvp(jvp);
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type.clone()]).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &axes, natural.clone())
            .unwrap();
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = composite [name=\"my_sin\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = custom_call [target=my_sin, batching=broadcast_all] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // A batching rule takes precedence over the kernel's strategy, including under nested batching, and the batched
        // call keeps its JVP rule.
        let function = Composite::from_custom_call(operation).with_jvp(jvp).with_batching(
            |inputs: Vec<Tracer>, axes: Vec<BatchAxis>, _: BatchingLevelExtent<Tracer>| {
                let operation = CustomCallOperation::new("my_batched_sin", vec![inputs[0].r#type().into_owned()]);
                Ok((CustomCall::custom_call(&operation, &inputs)?, axes))
            },
        );
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type]).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &axes, natural.clone())
            .unwrap();
        let nested = batched.into_parts().0.batched(2, ShardingDimension::Replicated, &axes, natural).unwrap();
        let nested = nested.into_parts().0;
        assert_eq!(
            nested.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[2, 3] = composite [
                    name=\"my_sin\",
                    batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0]), (extent=2, input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[2, 3] .
                        let %1:f64[2, 3] = custom_call [target=my_batched_sin] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            nested.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[2, 3] .
                let %2:f64[2, 3] = sin %0
                    %3:f64[2, 3] = cos %0
                    %4:f64[2, 3] = mul %3 %1
                in (%2, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_from_custom_call_without_rules() {
        // Without derivative rules, differentiation differentiates the primal, which reaches the opaque custom call and
        // reports its own diagnostic in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function: Composite<Vec<Tracer>, Vec<Tracer>> =
            Composite::from_custom_call(CustomCallOperation::new("my_sin", vec![scalar_type]));
        let rejection = "custom call `my_sin` has no differentiation rule; call it through a `composite` with \
                         derivative rules to provide one";
        assert!(matches!(
            differentiate_at(vec![Array::scalar(2.0).unwrap()])
                .jvp(vec![Array::scalar(1.0).unwrap()], |inputs| function.call(inputs)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message })) if message == rejection,
        ));
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(vec![x]).map(|outputs| outputs.into_iter().next().unwrap())),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message })) if message == rejection,
        ));
    }

    #[test]
    fn test_composite_with_non_differentiated_count() {
        // The leading `scale` leaf is plumbing, so the JVP rule ignores its tangent. A stopped gradient states its zero
        // derivative, while an active tangent for it is rejected rather than silently dropped.
        type Tracer = DomainTracer<ArrayContext>;
        let scaled = composite(|(scale, x): (Tracer, Tracer)| Ok(scale * x))
            .with_jvp(|(scale, x), (_, tangent)| Ok((scale.clone() * x, scale * tangent)))
            .with_non_differentiated_count(1);
        let inputs = (Array::scalar(3.0f64).unwrap(), Array::scalar(2.0f64).unwrap());
        assert_eq!(
            differentiate_at(inputs.clone()).gradient(|(scale, x)| scaled.call((scale.stop_gradient()?, x))),
            Ok((Array::scalar(0.0f64).unwrap(), Array::scalar(3.0f64).unwrap())),
        );
        assert!(matches!(
            differentiate_at(inputs).gradient(|(scale, x)| scaled.call((scale, x))),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` cannot propagate the non-zero tangent of type `f64[]` supplied for one of \
                               its 1 leading non-differentiated inputs, because its rule has no tangent slot for them",
        ));
    }

    #[test]
    fn test_composite_with_name() {
        // Without a name, calls are labeled by the operation name. Renaming a function that was already called must not
        // reuse the definition registered by that call, which carries the previous name. A function with only
        // reverse mode rules rejects forward mode and names itself in the diagnostic.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?));
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"composite\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let function = function.with_name("sine");
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"sine\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // The name also labels the diagnostics of the call (e.g., the forward mode rejection of reverse-only rules).
        let function = function.with_vjp(|x| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward-mode differentiation to a `composite` call of `sine` that \
                               has only reverse-mode rules; it supports only reverse-mode differentiation (e.g., \
                               `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_composite_with_jvp() {
        // An explicit JVP rule governs forward mode and, by transposition, reverse mode. The rule doubles the true
        // derivative (expressed through addition to avoid constant lifting), which proves that the rule is in control.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp(|x, tangent| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        });
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0 * 2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_composite_with_jvp_zero_space_inputs() {
        // The tangent of an integer input belongs to a zero differential space, so it is always a structural zero,
        // which the specialization materializes for a rule that receives materialized tangents.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|(x, _): (Tracer, Tracer)| Ok(x.sin()?)).with_jvp(
            |(x, _): (Tracer, Tracer), (tangent, count_tangent): (Tracer, Tracer)| {
                assert_eq!(count_tangent.r#type().into_owned(), ArrayType::scalar(DataType::Zero));
                Ok((x.sin()?, x.cos()? * tangent))
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let count = x.context().lift(Array::scalar(3i32).unwrap())?;
                function.call((x, count))
            }),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_jvp_non_differentiated_inputs() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // The leading counter is plumbing: it reaches both closures at its usual position, and the rule closure keeps
        // receiving a full tangent value whose counter leaf is a placeholder that it leaves unused.
        let function =
            composite(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(counter, x), (_, tangent)| {
                counter.add_update(&x)?;
                Ok((x, tangent))
            })
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

        // A rule that uses the placeholder tangent of a plumbing input is rejected when it is first traced (i.e., by
        // the first derivative request), because the staged rule has no tangent slot for it.
        let function =
            composite(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(_, x), (counter_tangent, tangent)| {
                counter_tangent.add_update(&tangent)?;
                Ok((x, tangent))
            })
            .with_non_differentiated_count(1);
        assert_eq!(
            EagerArrayIrContext::trace(
                |(counter, x)| function.call((counter, x)),
                (reference_type.clone(), scalar_type.clone()),
            )
            .and_then(|(_, program)| {
                program.into_flat_program().jvp_with_respect_to(&[1]).map(|_| ()).map_err(ProgramError::from)
            }),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` rule uses the tangent of leading non-differentiated input 0, which has no tangent \
                 slot because non-differentiated inputs parameterize the rule without being differentiated"
                    .to_string(),
            ))),
        );

        // Returning the placeholder is also a use, even when no instruction consumes it.
        let function =
            composite(|(parameter, _): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(parameter))
                .with_jvp(|(parameter, _), (placeholder, _)| Ok((parameter, placeholder)))
                .with_non_differentiated_count(1);
        assert_eq!(
            ArrayContext::trace(
                |inputs| function.call(inputs),
                (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)),
            )
            .and_then(|(_, program)| {
                program.into_flat_program().jvp_with_respect_to(&[1]).map(|_| ()).map_err(ProgramError::from)
            }),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` rule uses the tangent of leading non-differentiated input 0, which has no tangent \
                 slot because non-differentiated inputs parameterize the rule without being differentiated"
                    .to_string(),
            ))),
        );

        // Without the declaration, the reference is an active input, which the staged operation rejects.
        let function =
            composite(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(_, x), (_, tangent)| Ok((x, tangent)));
        assert_eq!(
            EagerArrayIrContext::trace(|(counter, x)| function.call((counter, x)), (reference_type, scalar_type))
                .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // The non-differentiated count cannot exceed the number of input leaves.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_jvp(|x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
            .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_composite_with_jvp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The custom rule
        // `jvp(x, ẋ) = (sin(x), cos(x) * ẋ + cos(x) * ẋ)` doubles the mathematical derivative (expressed through
        // addition to avoid constant lifting), which proves that the rule is in control.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp(|x, tangent| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        });
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0f64.sin()).unwrap()));
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
    fn test_composite_with_jvp_structured_outputs() {
        // Distinct tuple leaves make flattening and reconstruction order observable.
        let function =
            composite(|x: DomainTracer<ArrayContext>| Ok((x.sin()?, vec![x.cos()?, x]))).with_jvp(|x, tangent| {
                Ok(((x.sin()?, vec![x.cos()?, x]), (tangent.clone(), vec![tangent.clone(), tangent])))
            });
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.interpret(Array::scalar(2.0).unwrap()),
            Ok((
                Array::scalar(2.0f64.sin()).unwrap(),
                vec![Array::scalar(2.0f64.cos()).unwrap(), Array::scalar(2.0).unwrap()]
            )),
        );
    }

    #[test]
    fn test_composite_with_jvp_tracing_errors() {
        // Empty collections cannot identify the context in which the call should execute.
        let function = composite(|inputs: Vec<DomainTracer<ArrayContext>>| Ok(inputs))
            .with_jvp(|inputs, tangents| Ok((inputs, tangents)));
        assert_eq!(
            ArrayContext::trace(|inputs| function.call(inputs), Vec::<ArrayType>::new()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("`composite` requires at least one input".to_string()))),
        );

        // Both closures propagate their original error without replacing it with a signature error.
        let function = composite(|_: DomainTracer<ArrayContext>| -> Result<DomainTracer<ArrayContext>, ProgramError> {
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() })
        })
        .with_jvp(|input, tangent| Ok((input, tangent)));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() }),
        );
        let function = composite(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_jvp(|_, _| Err(ProgramError::InvalidArgument { message: "jvp tracing failed".to_string() }));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::InvalidArgument { message: "jvp tracing failed".to_string() }),
        );
    }

    #[test]
    fn test_composite_with_jvp_rule_signature_mismatch() {
        // Fixed tuple structures are checked statically, but array shapes and collection lengths require runtime
        // validation. This rule produces a scalar tangent for a vector output, which fails signature validation when
        // the rule is first traced (i.e., by the first derivative request) rather than when the function is called.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_jvp(|x, tangent| Ok((x.sin()?, tangent.dot(&tangent, &DotDimensionNumbers::inner_product())?)));
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        assert_eq!(
            program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` JVP rule output type signature mismatch: expected [f64[2], \
                 f64[2]] but got [f64[2], f64[]]"
                    .to_string(),
            ))),
        );

        // Equal flattened leaf types cannot distinguish different nested vector lengths.
        let function =
            composite(|input: DomainTracer<ArrayContext>| Ok(vec![vec![input.clone(), input.clone()], vec![input]]))
                .with_jvp(|input, tangent| {
                    Ok((
                        vec![vec![input.clone()], vec![input.clone(), input]],
                        vec![vec![tangent.clone()], vec![tangent.clone(), tangent]],
                    ))
                });
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`composite` JVP rule output structure mismatch: expected {:?} but got {:?}",
                vec![vec![Placeholder; 2], vec![Placeholder]],
                vec![vec![Placeholder], vec![Placeholder; 2]],
            )))),
        );

        // The primal half may agree while the tangent half has the wrong structure.
        let function =
            composite(|input: DomainTracer<ArrayContext>| Ok(vec![vec![input.clone(), input.clone()], vec![input]]))
                .with_jvp(|input, tangent| {
                    Ok((
                        vec![vec![input.clone(), input.clone()], vec![input]],
                        vec![vec![tangent.clone()], vec![tangent.clone(), tangent]],
                    ))
                });
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`composite` JVP rule output tangent structure mismatch: expected {:?} but got {:?}",
                vec![vec![Placeholder; 2], vec![Placeholder]],
                vec![vec![Placeholder], vec![Placeholder; 2]],
            )))),
        );
    }

    #[test]
    fn test_composite_with_jvp_computed_dimension_outputs() {
        // The primal broadcasts `x` to the computed extent `n · n`, and the JVP rule recomputes that extent in its own
        // trace, which mints a fresh identity for it. The rule outputs therefore agree with the primal outputs only up
        // to a renaming of that identity, which is applied to the rule program.
        let broadcast = |dimension: &ArrayIrTracer, x: ArrayIrTracer| -> Result<_, ProgramError> {
            let dimension = ValueProjection::<DimensionType>::into_projected(dimension.clone())?;
            let extent = (dimension.clone() * dimension).into_value();
            Ok((extent.clone(), x.dynamic_broadcast(&[extent], &[])?))
        };
        let function = |shares_extent: bool| {
            composite(move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| broadcast(&dimension, x))
                .with_non_differentiated_count(1)
                .with_jvp(
                    move |(dimension, x): (ArrayIrTracer, ArrayIrTracer),
                          (_, tangent): (ArrayIrTracer, ArrayIrTracer)| {
                        let (extent, output) = broadcast(&dimension, x)?;
                        let extent_tangent = extent.context().zero(&extent.r#type().tangent()?)?;
                        let output_tangent = match shares_extent {
                            true => tangent.dynamic_broadcast(&[extent.clone()], &[])?,
                            false => broadcast(&dimension, tangent)?.1,
                        };
                        Ok(((extent, output), (extent_tangent, output_tangent)))
                    },
                )
        };
        let dimension = DimensionType::new("n", DimensionBounds::new(2, Some(5)).unwrap());
        let input_types = (ArrayIrType::from(dimension.clone()), ArrayIrType::from(ArrayType::scalar(DataType::F64)));
        let jvp = |shares_extent: bool| {
            let function = function(shares_extent);
            let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types.clone())?;
            program.into_flat_program().jvp_with_respect_to(&[1]).map_err(ProgramError::from)
        };
        let program = jvp(true).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [2, 5)>, %1:f64[], %2:f64[] .
                let %3:dimension<n * n ∈ [4, 17)> = dimension_mul %0 %0
                    %4:f64[n * n] = broadcast [output_axes=[]] %1 %3
                    %5:zero[] = zero [type=zero[]]
                    %6:f64[n * n] = broadcast [output_axes=[]] %2 %3
                in (%3, %4, %6)
            "}
            .trim_end(),
        );
        let outputs = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(dimension, 2).unwrap()),
                ArrayIrValue::Array(Array::scalar(3.0f64).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f64).unwrap()),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], ArrayIrValue::Dimension(extent) if extent.extent() == 4));
        assert_eq!(
            outputs[1..],
            [
                ArrayIrValue::Array(Array::vector(vec![3.0f64; 4]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f64; 4]).unwrap()),
            ],
        );

        // A rule that computes the extent of its output tangent separately establishes a second identity for it,
        // which the primal output and its tangent share, so its outputs are rejected.
        assert_eq!(
            jvp(false).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` JVP rule output type signature mismatch: expected \
                 [dimension<n * n ∈ [4, 17)>, f64[n * n], zero[], f64[n * n]] but got [dimension<n * n ∈ [4, 17)>, \
                 f64[n * n], zero[], f64[n * n]] with different type identities"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_composite_with_jvp_zero_space_boundaries() {
        // Token primals and zero-space tangents carry no payload, so the wrapper must pass them through the traced
        // rule unchanged instead of demanding a dense tangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function =
            composite(|token: DomainTracer<ArrayContext>| Ok(token)).with_jvp(|token, tangent| Ok((token, tangent)));
        assert_eq!(differentiate_at(token.clone()).jvp(zero.clone(), |token| function.call(token)), Ok((token, zero)));
    }

    #[test]
    fn test_composite_with_jvp_tangent_casts() {
        // The rule narrows `x: f64[n]` to `f64[2]` with a checked cast and converts the tangent to match. Reverse mode
        // transposes the replayed rule, so the cotangent must widen back to `f64[n]`. An unchecked tangent conversion
        // that names the primal input as its witness keeps `n` available to that transpose (as the derived rule does),
        // while a checked tangent conversion leaves a check on a linear value in the tangent program, which cannot be
        // transposed because no known value carries the extent of the cotangent that it would produce.
        type Tracer = DomainTracer<ArrayContext>;
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [2]);
        let program = |checked: bool| {
            let (primal_type, rule_type) = (static_type.clone(), static_type.clone());
            let function = composite(move |x: Tracer| x.cast(primal_type.clone(), &[])?.sin()).with_jvp(
                move |x: Tracer, tangent: Tracer| {
                    let y = x.cast(rule_type.clone(), &[])?;
                    let tangent = match checked {
                        true => tangent.cast(rule_type.clone(), &[])?,
                        false => tangent.cast_unchecked(rule_type.clone(), &[x])?,
                    };
                    Ok((y.sin()?, y.cos()? * tangent))
                },
            );
            let (_, program) = ArrayContext::trace(|x| function.call(x), symbolic_type.clone()).unwrap();
            program.into_flat_program()
        };
        let x = Array::vector(vec![1.0, 2.0]).unwrap();
        let tangent = Array::vector(vec![1.0, 0.5]).unwrap();
        let outputs = Ok(vec![
            Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap(),
            Array::vector(vec![1.0f64.cos(), 0.5 * 2.0f64.cos()]).unwrap(),
        ]);
        let mismatch = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f64[3]` to unrelated type `f64[2]`".to_string(),
        )));

        // Unchecked tangent conversion witnessed by the primal input.
        let unchecked = program(false);
        let jvp = unchecked.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n] .
                let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %3:f64[2] = cast [type=f64[2], inputs=[f64[n], f64[n]]] %1 %0
                    %4:f64[2] = sin %2
                    %5:f64[2] = cos %2
                    %6:f64[2] = mul %5 %3
                in (%4, %6)
            "}
            .trim_end(),
        );
        assert_eq!(jvp.interpret(vec![x.clone(), tangent.clone()]), outputs);
        let wide = Array::vector(vec![1.0; 3]).unwrap();
        assert_eq!(jvp.interpret(vec![wide.clone(), wide]), mismatch);
        let linearization = unchecked.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[n] .
                let %1:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %2:f64[2] = sin %1
                    %3:f64[2] = cos %1
                in (%2, %0, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n], %2:f64[2] .
                let %3:f64[2] = cast [type=f64[2], inputs=[f64[n], f64[n]]] %0 %1
                    %4:f64[2] = mul %2 %3
                in (%4)
            "}
            .trim_end(),
        );
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[n], %2:f64[2] .
                let %3:f64[2] = mul %2 %0
                    %4:f64[n] = cast [type=f64[n], inputs=[f64[2], f64[n]]] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
        let mut pullback_inputs = vec![tangent.clone()];
        pullback_inputs.extend(linearization.primal().interpret(vec![x.clone()]).unwrap().into_iter().skip(1));
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![1.0f64.cos(), 0.5 * 2.0f64.cos()]).unwrap()]),
        );
        check_gradient!(
            |x| {
                let output = unchecked.interpret_in_context(&x.domain(), vec![x])?.remove(0);
                output.reduce(&[0], ReductionKind::Sum)
            },
            at = x.clone(),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // Checked tangent conversion: forward mode checks the tangent as well, but reverse mode rejects the check.
        let checked = program(true);
        let jvp = checked.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n] .
                let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %3:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %1
                    %4:f64[2] = sin %2
                    %5:f64[2] = cos %2
                    %6:f64[2] = mul %5 %3
                in (%4, %6)
            "}
            .trim_end(),
        );
        assert_eq!(jvp.interpret(vec![x, tangent]), outputs);
        let linearization = checked.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[2] .
                let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %3:f64[2] = mul %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert!(matches!(
            linearization.tangent().transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot transpose a checked `cast` without the primal value its check needs; \
                               linearize the program first, or use `cast_unchecked` with witnesses for a tangent \
                               that an earlier check already proved",
        ));
    }

    #[test]
    fn test_composite_with_jvp_and_vjp() {
        // With both rule kinds, forward mode uses the JVP rule (which doubles the derivative) and reverse mode uses the
        // reverse mode rules (which triple it), whatever the order of the configuration calls.
        let doubled_jvp = |x: DomainTracer<ArrayContext>, tangent: DomainTracer<ArrayContext>| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        };
        let forward = |x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?));
        let tripled_backward = |cosine: DomainTracer<ArrayContext>, cotangent| {
            let gradient = cosine * cotangent;
            Ok(gradient.clone() + gradient.clone() + gradient)
        };
        let primal = |x: DomainTracer<ArrayContext>| Ok(x.sin()?);
        let jvp_first = composite(primal).with_jvp(doubled_jvp).with_vjp(forward, tripled_backward);
        let vjp_first = composite(primal).with_vjp(forward, tripled_backward).with_jvp(doubled_jvp);
        for (jvp, gradient) in [
            (
                differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| jvp_first.call(x)),
                differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| jvp_first.call(x)),
            ),
            (
                differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| vjp_first.call(x)),
                differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| vjp_first.call(x)),
            ),
        ] {
            assert_eq!(jvp, Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())));
            assert_eq!(gradient, Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()));
        }
    }

    #[test]
    fn test_composite_with_jvp_from_primal() {
        // Deriving the forward mode rule from the primal yields the true derivative in both modes, unless reverse mode
        // rules are configured as well, in which case they govern reverse mode (and triple the derivative here).
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp_from_primal();
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        let function = function.with_vjp(
            |x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?)),
            |cosine, cotangent| {
                let gradient = cosine * cotangent;
                Ok(gradient.clone() + gradient.clone() + gradient)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_composite_with_symbolic_zero_jvp() {
        // The product rule receives the structural zero of an input that is not being differentiated as a
        // `MaybeZero::Zero` leaf and skips its term, and each activity pattern is a separate specialization.
        type Tracer = DomainTracer<ArrayContext>;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let function = composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_symbolic_zero_jvp({
            let observed = observed.clone();
            move |(x, y): (Tracer, Tracer), tangents: (MaybeZero<Tracer>, MaybeZero<Tracer>)| {
                observed.lock().unwrap().push((tangents.0.is_zero(), tangents.1.is_zero()));
                let tangent = match tangents {
                    (MaybeZero::Value(dx), MaybeZero::Value(dy)) => MaybeZero::Value(dx * y.clone() + x.clone() * dy),
                    (MaybeZero::Value(dx), MaybeZero::Zero(_)) => MaybeZero::Value(dx * y.clone()),
                    (MaybeZero::Zero(_), MaybeZero::Value(dy)) => MaybeZero::Value(x.clone() * dy),
                    (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => MaybeZero::Zero(r#type),
                };
                Ok((x * y, tangent))
            }
        });
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let y = x.context().lift(Array::scalar(3.0).unwrap())?;
                function.call((x, y))
            }),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(3.0).unwrap())),
        );
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()))
                .jvp((Array::scalar(1.0).unwrap(), Array::scalar(1.0).unwrap()), |inputs| function.call(inputs)),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(5.0).unwrap())),
        );
        assert_eq!(*observed.lock().unwrap(), vec![(false, true), (false, false)]);
    }

    #[test]
    fn test_composite_with_symbolic_zero_jvp_zero_output_tangents() {
        // The JVP rule of `x ↦ (sin(x), x)` declares the tangent of its second output to be a structural zero (i.e.,
        // it treats that output as a constant). The zero is materialized at the boundary of the rule's specialization,
        // where replaying it recovers the structural zero, so it stays zero under forward mode, batching, reverse mode,
        // and second-order differentiation, which all differ from the derivative of the primal.
        type Tracer = DomainTracer<ArrayContext>;
        let function =
            composite(|x: Tracer| Ok((x.sin()?, x))).with_symbolic_zero_jvp(|x: Tracer, tangent: MaybeZero<Tracer>| {
                let sine_tangent = match tangent {
                    MaybeZero::Value(tangent) => MaybeZero::Value(x.cos()? * tangent),
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type),
                };
                let point_tangent = MaybeZero::Zero(x.r#type().into_owned());
                Ok(((x.sin()?, x), (sine_tangent, point_tangent)))
            });
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = cos %0
                    %3:f64[] = mul %2 %1
                    %4:f64[] = sin %0
                    %5:f64[] = zero [type=f64[]]
                    %6:f64[] = zero [type=f64[]]
                in (%4, %0, %3, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((
                (Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0).unwrap()),
                (Array::scalar(2.0f64.cos()).unwrap(), Array::scalar(0.0).unwrap()),
            )),
        );

        // Batching the derivative batches the materialized zero with the rest of the rule's specialization.
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 2],
                ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0); 4]),
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        assert_eq!(
            batched.interpret(vec![Array::vector(x.to_vec()).unwrap(), Array::vector(vec![1.0; 3]).unwrap()]),
            Ok(vec![
                Array::vector(x.iter().map(|x| f64::sin(*x)).collect()).unwrap(),
                Array::vector(x.to_vec()).unwrap(),
                Array::vector(x.iter().map(|x| f64::cos(*x)).collect()).unwrap(),
                Array::vector(vec![0.0; 3]).unwrap(),
            ]),
        );

        // Reverse mode transposes no work for the structural zero, so the second output contributes nothing to the
        // gradient, and second-order differentiation differentiates the replayed rule.
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| {
                differentiate_at(x)
                    .gradient(|x| {
                        let (sine, point) = function.call(x)?;
                        Ok::<_, ProgramError>(sine + point)
                    })
                    .unwrap()
            }),
            Ok((Array::scalar(2.0f64.cos()).unwrap(), Array::scalar(-(2.0f64.sin())).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_vjp() {
        // Reverse mode rules govern reverse mode, while forward mode rejects the call because no forward mode rule was
        // configured.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |cosine, cotangent| {
                let gradient = cosine * cotangent;
                Ok(gradient.clone() + gradient.clone() + gradient)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()),
        );
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward-mode differentiation to a `composite` call that has only \
                               reverse-mode rules; it supports only reverse-mode differentiation (e.g., `vjp`, \
                               `value_and_gradient`, or `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_composite_with_vjp_specialized_residual_structures() {
        // Specializations of one call structure may return differently structured residuals: here, one residual per
        // input dimension, so the scalar and vector specializations save zero and two residuals, respectively.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| {
                let cosine = x.cos()?;
                Ok((x.sin()?, (cosine.clone(), vec![cosine; x.r#type().rank()])))
            },
            |(cosine, _): (DomainTracer<ArrayContext>, Vec<DomainTracer<ArrayContext>>), cotangent| {
                Ok(cosine * cotangent)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        let (_, pullback) =
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(
            pullback.apply(Array::vector(vec![1.0f64, 1.0]).unwrap()),
            Ok(MaybeZero::Value(Array::vector(vec![1.0f64.cos(), 2.0f64.cos()]).unwrap())),
        );

        // Residual structures are recorded by their flat residual types, so two specializations whose residuals have
        // identical flat types but different structures are ambiguous. Here, the residual type is always `f64[]`.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| {
                let total = x.cos()?.reduce(&(0..x.r#type().rank()).collect::<Vec<_>>(), ReductionKind::Sum)?;
                let residuals = if x.r#type().rank() == 0 { vec![vec![total]] } else { vec![vec![], vec![total]] };
                Ok((x.sin()?, residuals))
            },
            |_: Vec<Vec<DomainTracer<ArrayContext>>>, cotangent| Ok(cotangent),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(1.0).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|x| function.call(x))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`composite` forward rule residual structure mismatch: expected {:?} but got {:?}",
                vec![vec![Placeholder]],
                vec![vec![], vec![Placeholder]],
            )))),
        );
    }

    #[test]
    fn test_composite_with_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        let function = composite(|x: ArrayIrTracer| Ok(x))
            .with_vjp(saved_extent, |_: ArrayIrTracer, cotangent: ArrayIrTracer| Ok(cotangent));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(MaybeZero::Value(seed)));
    }

    #[test]
    fn test_composite_with_vjp_computed_dimension_outputs() {
        // The primal broadcasts `x` to the computed extent `n · n`, and the forward rule recomputes that extent in its
        // own trace, which mints a fresh identity for it. The rule outputs therefore agree with the primal outputs
        // only up to a renaming of that identity, which is applied to the rule program, including its residuals.
        let broadcast = |dimension: &ArrayIrTracer, x: ArrayIrTracer| -> Result<_, ProgramError> {
            let dimension = ValueProjection::<DimensionType>::into_projected(dimension.clone())?;
            let extent = (dimension.clone() * dimension).into_value();
            Ok((extent.clone(), x.dynamic_broadcast(&[extent], &[])?))
        };
        let function = composite(move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| broadcast(&dimension, x))
            .with_vjp(
                move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| {
                    let (extent, output) = broadcast(&dimension, x)?;
                    Ok(((extent.clone(), output), (dimension, extent)))
                },
                |(dimension, _): (ArrayIrTracer, ArrayIrTracer), (_, cotangent): (ArrayIrTracer, ArrayIrTracer)| {
                    let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                    Ok((dimension, cotangent.reduce(&[0], ReductionKind::Sum)?.into_value()))
                },
            )
            .with_non_differentiated_count(1);
        let dimension = DimensionType::new("n", DimensionBounds::new(2, Some(5)).unwrap());
        let input_types = (ArrayIrType::from(dimension.clone()), ArrayIrType::from(ArrayType::scalar(DataType::F64)));
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types).unwrap();
        let linearization = program
            .into_flat_program()
            .entry_region_ref()
            .linearize_shared_for_rule(&[1], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [2, 5)>, %1:f64[] .
                let %2:dimension<n * n ∈ [4, 17)> = dimension_mul %0 %0
                    %3:f64[n * n] = broadcast [output_axes=[]] %1 %2
                    %4:dimension<n * n ∈ [4, 17)> = dimension_size [axis=0] %3
                in (%2, %3, %0, %2, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<n ∈ [2, 5)>, %2:dimension<n * n ∈ [4, 17)>, %3:dimension<n * n ∈ [4, 17)> .
                let %4:zero[], %5:f64[n * n] = composite_transpose [name=\"composite\", leading_input_count=4, seed_extent_count=1] %1 %1 %2 %3 %0
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[n * n], %1:dimension<n ∈ [2, 5)>, %2:dimension<n * n ∈ [4, 17)>, %3:dimension<n * n ∈ [4, 17)> .
                let %4:zero[] = zero [type=zero[]]
                    %5:f64[] = reduce [kind=sum, axes=[0]] %0
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_with_vjp_non_differentiated_inputs() {
        // The stash-gradients pattern: the leading `stash` reference is plumbing, the forward rule forwards it as a
        // residual by identity rather than saving a snapshot, and the backward rule writes the incoming cotangent into
        // it before returning `x̄ = cos(x) · ȳ`. The cotangent value that it returns is input-shaped, so its leading
        // plumbing leaf is ignored.
        let function = composite(|(_, x): (ArrayIrTracer, ArrayIrTracer)| {
            Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
        })
        .with_vjp(
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
            Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(0.5f32.cos() * 2.0).unwrap()))))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(2.0f32).unwrap()));

        // Every application writes the stash anew.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(0.5f32.cos() * 3.0).unwrap()))))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The non-differentiated count cannot exceed the number of input leaves.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_vjp(|x| Ok((x.sin()?, x.cos()?)), |residual, cotangent| Ok(residual * cotangent))
            .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_composite_with_vjp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The custom
        // rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the true gradient (expressed through
        // addition to avoid constant lifting), which proves that the rule is in control.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0f64.sin()).unwrap()));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(3.0 * 2.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_vjp_tracing_errors() {
        // An empty collection provides no value from which the call can recover its execution context.
        let function = composite(|inputs: Vec<DomainTracer<ArrayContext>>| Ok(inputs))
            .with_vjp(|inputs| Ok((inputs, ())), |(), cotangents| Ok(cotangents));
        assert_eq!(
            ArrayContext::trace(|inputs| function.call(inputs), Vec::<ArrayType>::new()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("`composite` requires at least one input".to_string()))),
        );

        // Each closure's original error must propagate rather than becoming a rule-interface diagnostic. The primal
        // closure is traced when the function is called, while the forward and backward closures are traced by the
        // first reverse mode derivative request.
        let function = composite(|_: DomainTracer<ArrayContext>| -> Result<DomainTracer<ArrayContext>, ProgramError> {
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() })
        })
        .with_vjp(|input| Ok((input, ())), |(), cotangent| Ok(cotangent));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() }),
        );
        let function = composite(|input: DomainTracer<ArrayContext>| Ok(input)).with_vjp(
            |_| -> Result<(DomainTracer<ArrayContext>, ()), ProgramError> {
                Err(ProgramError::InvalidArgument { message: "forward tracing failed".to_string() })
            },
            |(), cotangent| Ok(cotangent),
        );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::InvalidArgument { message: "forward tracing failed".to_string() }),
        );
        let function = composite(|input: DomainTracer<ArrayContext>| Ok(input)).with_vjp(
            |input| Ok((input, ())),
            |(), _| Err(ProgramError::InvalidArgument { message: "backward tracing failed".to_string() }),
        );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::InvalidArgument { message: "backward tracing failed".to_string() }),
        );
    }

    #[test]
    fn test_composite_with_vjp_rule_signature_mismatch() {
        // The closure signatures fix the parameter families, but array shapes still require validation after tracing,
        // which happens when the first reverse mode derivative request specializes the rules.
        let function = composite(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_vjp(|input| Ok((input.reduce(&[0], ReductionKind::Sum)?, ())), |(), cotangent| Ok(cotangent));
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` forward rule output type signature mismatch: expected [f64[2]] \
                 but got [f64[]]"
                    .to_string(),
            ))),
        );

        // A valid forward interface must not hide a backward rule that produces the wrong input-cotangent shape.
        let function = composite(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_vjp(|input| Ok((input, ())), |(), cotangent| Ok(cotangent.reduce(&[0], ReductionKind::Sum)?));
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` backward rule output type signature mismatch: expected [f64[2]] \
                 but got [f64[]]"
                    .to_string(),
            ))),
        );

        // The forward result must preserve nested collection lengths, not just the flattened types.
        let function =
            composite(|input: DomainTracer<ArrayContext>| Ok(vec![vec![input.clone(), input.clone()], vec![input]]))
                .with_vjp(
                    |input| Ok((vec![vec![input.clone()], vec![input.clone(), input]], ())),
                    |(), cotangents| Ok(cotangents[0][0].clone()),
                );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`composite` forward rule output structure mismatch: expected {:?} but got {:?}",
                vec![vec![Placeholder; 2], vec![Placeholder]],
                vec![vec![Placeholder], vec![Placeholder; 2]],
            )))),
        );

        // Check the full backward structure even when the first flattened leaf is non-differentiated. Reverse mode
        // linearizes with respect to the two differentiated leaves only and then transposes both tangent inputs.
        let function = composite(|input: Vec<Vec<DomainTracer<ArrayContext>>>| Ok(input[1][0].clone()))
            .with_vjp(
                |input| Ok((input[1][0].clone(), ())),
                |(), cotangent| Ok(vec![vec![cotangent.clone()], vec![cotangent.clone(), cotangent]]),
            )
            .with_non_differentiated_count(1);
        let (_, program) = ArrayContext::trace(
            |input| function.call(input),
            vec![vec![ArrayType::scalar(DataType::F64); 2], vec![ArrayType::scalar(DataType::F64)]],
        )
        .unwrap();
        assert_eq!(
            program
                .into_flat_program()
                .entry_region_ref()
                .linearize_shared_for_rule(&[1, 2], DifferentiationRule::JvpForTranspose, None)
                .and_then(|linearization| linearization.tangent().transpose_with_respect_to(&[0, 1], &[]))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`composite` backward rule cotangent structure mismatch: expected {:?} but got {:?}",
                vec![vec![Placeholder; 2], vec![Placeholder]],
                vec![vec![Placeholder], vec![Placeholder; 2]],
            )))),
        );
    }

    #[test]
    fn test_composite_with_vjp_reference_contract() {
        let input_types = (
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        );

        // A reference input that is not declared as plumbing is an active input, which the staged operation rejects.
        let function = composite(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(|(stash, x)| Ok((x, stash)), |stash, cotangent| Ok((stash, cotangent)));
        assert_eq!(
            EagerArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types.clone()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // A forward rule may forward the plumbing reference as a residual but not return a reference it allocated.
        // Residuals are only produced by the forward rule, which the first reverse mode derivative request traces.
        let inputs = (
            ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
            ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
        );
        let function = composite(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(|(_, x)| Ok((x.clone(), x.reference_new()?)), |allocated, cotangent| Ok((allocated, cotangent)))
            .with_non_differentiated_count(1);
        assert_eq!(
            differentiate_at(inputs.clone())
                .vjp(|(stash, x)| function.call((stash, x)))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` forward rule returns residual 0 of reference type `ref<f32[]>` that is not a \
                 leading non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // Every reference-typed residual is held to that rule, not only the first one: forwarding the plumbing
        // reference and then returning an allocated reference beside it is rejected at the allocated residual.
        let function = composite(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(
                |(stash, x)| Ok((x.clone(), (stash, x.reference_new()?))),
                |(stash, _allocated), cotangent| Ok((stash, cotangent)),
            )
            .with_non_differentiated_count(1);
        assert_eq!(
            differentiate_at(inputs.clone())
                .vjp(|(stash, x)| function.call((stash, x)))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` forward rule returns residual 1 of reference type `ref<f32[]>` that is not a \
                 leading non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // No rule may return a reference as a primal output, not even a plumbing input forwarded by identity.
        let function = composite(|(stash, _): (ArrayIrTracer, ArrayIrTracer)| Ok(stash))
            .with_vjp(|(stash, _)| Ok((stash, ())), |(), cotangent| Ok((cotangent.clone(), cotangent.read()?)))
            .with_non_differentiated_count(1);
        assert_eq!(
            EagerArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            ))),
        );

        // Two same-typed inputs do not permit forwarding the first one twice. Type multiplicity alone cannot detect
        // this alias, so the specialized forward rule checks which input each residual forwards.
        let function = composite(|(_, _, input): (ArrayIrTracer, ArrayIrTracer, ArrayIrTracer)| Ok(input))
            .with_vjp(
                |(first, _, input)| Ok((input, (first.clone(), first))),
                |(first, second), cotangent| Ok((first, second, cotangent)),
            )
            .with_non_differentiated_count(2);
        assert_eq!(
            differentiate_at((
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
                ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ))
            .vjp(|input| function.call(input))
            .and_then(|(_, pullback)| pullback.into_transposed_parts())
            .map(|_| ())
            .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`composite` forward rule returns residual 1 of reference type `ref<f32[]>` from an input \
                 already forwarded by an earlier residual",
            ))),
        );
    }

    #[test]
    fn test_composite_with_vjp_pullback() {
        // Extracting the stored pullback must preserve the custom backward rule and its residual inputs. The forward
        // rule saves `cos(x)`, and the backward rule returns `2 * residual * cotangent`, so a unit cotangent at
        // `x = 0.7` produces `2 * cos(0.7)` rather than the mathematical derivative.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok((residual.clone() + residual) * cotangent),
        );
        let (_, pullback) = differentiate_at(Array::scalar(0.7).unwrap()).vjp(|x| function.call(x)).unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::scalar(2.0 * 0.7f64.cos()).unwrap()]));
    }

    #[test]
    fn test_composite_with_vjp_cotangent_casts() {
        // The forward rule narrows `x: f64[n]` to `f64[2]` with a checked cast and saves `x` with the derivative, and
        // the backward rule widens the cotangent back to `f64[n]` with a checked cast that names `x` as its witness.
        // The staged pullback keeps both checks, the first one before the backward rule, in both orders of batching and
        // reverse mode differentiation.
        type Tracer = DomainTracer<ArrayContext>;
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [2]);
        let forward_type = static_type.clone();
        let function = composite(move |x: Tracer| x.cast(static_type.clone(), &[])?.sin()).with_vjp(
            move |x: Tracer| {
                let y = x.cast(forward_type.clone(), &[])?;
                Ok((y.sin()?, (x, y.cos()?)))
            },
            |(x, cosine): (Tracer, Tracer), cotangent: Tracer| (cosine * cotangent).cast(x.r#type().into_owned(), &[x]),
        );
        let (_, vjp) = ArrayContext::trace(
            |(x, seed): (Tracer, Tracer)| {
                let (y, pullback) = differentiate_at(x).vjp(|x| function.call(x))?;
                Ok((y, pullback.apply_materialized(seed)?))
            },
            (symbolic_type.clone(), ArrayType::new_static(DataType::F64, [2])),
        )
        .unwrap();
        let vjp = vjp.into_flat_program();
        assert_eq!(
            vjp.to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[2] .
                let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %3:f64[2] = sin %2
                    %4:f64[2] = cos %2
                    %5:f64[2] = mul %4 %1
                    %6:f64[n] = cast [type=f64[n], inputs=[f64[2], f64[n]], checked=true] %5 %0
                in (%3, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            vjp.interpret(vec![Array::vector(vec![1.0, 2.0]).unwrap(), Array::vector(vec![1.0, 0.5]).unwrap()]),
            Ok(vec![
                Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap(),
                Array::vector(vec![1.0f64.cos(), 0.5 * 2.0f64.cos()]).unwrap(),
            ]),
        );
        assert_eq!(
            vjp.interpret(vec![Array::vector(vec![1.0; 3]).unwrap(), Array::vector(vec![1.0, 0.5]).unwrap()]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f64[3]` to unrelated type `f64[2]`".to_string(),
            ))),
        );
        check_gradient!(
            |x| function.call(x)?.reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![1.0, 2.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // Differentiating then batching.
        let batched_vjp = vjp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched_vjp.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, 2] .
                let %2:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %0
                    %3:f64[3, 2] = cos %2
                    %4:f64[3, 2] = mul %3 %1
                    %5:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, 2], f64[3, n]], checked=true] %4 %0
                    %6:f64[3, 2] = sin %2
                in (%6, %5)
            "}
            .trim_end(),
        );
        let elements = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let x = Array::matrix(3, 2, elements.to_vec()).unwrap();
        let seed = Array::matrix(3, 2, vec![1.0; 6]).unwrap();
        let outputs = Ok(vec![
            Array::matrix(3, 2, elements.map(f64::sin).to_vec()).unwrap(),
            Array::matrix(3, 2, elements.map(f64::cos).to_vec()).unwrap(),
        ]);
        assert_eq!(batched_vjp.interpret(vec![x.clone(), seed.clone()]), outputs);

        // Batching then differentiating.
        let batched_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), n.simplified_dimension()]));
        let (_, vjp_of_batched) = ArrayContext::trace(
            |(x, seed): (Tracer, Tracer)| {
                let (y, pullback) = differentiate_at(x)
                    .vjp(|x| Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), None)?))?;
                Ok((y, pullback.apply_materialized(seed)?))
            },
            (batched_type, ArrayType::new_static(DataType::F64, [3, 2])),
        )
        .unwrap();
        let vjp_of_batched = vjp_of_batched.into_flat_program();
        assert_eq!(
            vjp_of_batched.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, 2] .
                let %2:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %0
                    %3:f64[3, 2] = sin %2
                    %4:f64[3, 2] = cos %2
                    %5:f64[3, 2] = mul %4 %1
                    %6:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, 2], f64[3, n]], checked=true] %5 %0
                in (%3, %6)
            "}
            .trim_end(),
        );
        assert_eq!(vjp_of_batched.interpret(vec![x, seed]), outputs);
    }

    #[test]
    fn test_composite_with_vjp_multiple_outputs() {
        // A two-output custom VJP exercises the output/residual split of the forward region: its leading values are the
        // primal outputs and the rest are residuals, and the backward region consumes one cotangent per output. The
        // custom rule scales the contribution of the first output by 2 and that of the second by 3, so
        // seeding one output cotangent at a time isolates each term of the custom backward rule.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?))).with_vjp(
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
            Ok(MaybeZero::Value(Array::scalar(2.0 * 0.5f64.cos()).unwrap())),
        );
        assert_eq!(
            pullback.apply((Array::scalar(0.0).unwrap(), Array::scalar(1.0).unwrap())),
            Ok(MaybeZero::Value(Array::scalar(3.0 * 0.5f64.sin()).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_vjp_structured_signatures() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` calling convention, and the captured `repeats`
        // count plays the role of static configuration that is visible to the rule closures without being
        // differentiated or stored as a residual.
        let repeats = 3usize;
        let function = composite(|(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y))
            .with_vjp(
                |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
                move |(x, y), cotangent| {
                    // The custom rule repeats both cotangents `repeats` times.
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
    fn test_composite_with_vjp_empty_residuals() {
        // A forward rule that saves nothing (i.e., `Residuals = ()`) exercises the zero-residual carrier path: the
        // backward rule depends only on the output cotangent, so the custom rule
        // `backward(cotangent) = 2 * cotangent` makes the gradient the constant `2` instead of `cos(x)`.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_vjp(|x| Ok((x.sin()?, ())), |(), cotangent| Ok(cotangent.clone() + cotangent));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_vjp_batching_replicated_inputs() {
        // Mapping only the first input verifies that the replicated input remains shared at the call boundary while
        // operations inside its regions broadcast it only where per-item multiplication requires alignment.
        let function = composite(|(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y))
            .with_vjp(
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
    fn test_composite_with_vjp_zero_space_boundaries() {
        // Token primals, residuals, and zero-space cotangents carry no payload, so the wrapper must pass them through
        // the traced forward and backward rules unchanged instead of demanding a dense cotangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function = composite(|token: DomainTracer<ArrayContext>| Ok(token))
            .with_vjp(|token| Ok((token.clone(), token)), |_residual, cotangent| Ok(cotangent));
        let (value, pullback) = differentiate_at(token.clone()).vjp(|token| function.call(token)).unwrap();
        assert_eq!(value, token);
        assert_eq!(pullback.apply(zero.clone()), Ok(MaybeZero::Zero(zero.r#type().into_owned())));
    }

    #[test]
    fn test_composite_with_vjp_effectful_backward_destinations() {
        // This is one custom backward closure with an observable reference effect and an intentionally nonlinear
        // seed formula. Destination specialization must preserve the closure's execution and additive semantics.
        let stash = ArrayReference::new(Array::scalar(-1.0f32).unwrap());
        let function = composite(|(_, value): (ArrayIrTracer, ArrayIrTracer)| Ok(value))
            .with_vjp(
                |(stash, value)| Ok((value, stash)),
                |stash: ArrayIrTracer, seed: ArrayIrTracer| {
                    stash.write(&seed)?;
                    let contribution = seed
                        .context()
                        .bind(
                            ArrayOperation::<Array>::Mul(MulOperation::new()),
                            Vec::new(),
                            &[seed.clone(), seed.clone()],
                        )?
                        .remove(0);
                    Ok((stash, contribution))
                },
            )
            .with_non_differentiated_count(1);
        let (_, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ))
        .vjp(|input| function.call(input))
        .unwrap();
        let first = ArrayReference::new(Array::scalar(10.0f32).unwrap());
        let second = ArrayReference::new(Array::scalar(20.0f32).unwrap());
        let seed = ArrayIrValue::Array(Array::scalar(3.0f32).unwrap());
        let destinations = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Reference];
        let retained = pullback.transposed_program(&destinations).unwrap().into_program();
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (CotangentDestination::Ignore, CotangentDestination::Reference(ArrayIrValue::Reference(first.clone()))),
            ),
            Ok((None, None)),
        );
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (
                    CotangentDestination::Ignore,
                    CotangentDestination::Reference(ArrayIrValue::Reference(second.clone()))
                ),
            ),
            Ok((None, None)),
        );
        assert_eq!(second.read(), Ok(Array::scalar(29.0f32).unwrap()));
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert!(Arc::ptr_eq(&retained, &pullback.transposed_program(&destinations).unwrap().into_program()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()))))),
        );

        // A numerical zero is still a live seed. Dropping every returned gradient must not erase the stash write.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Ignore),
            ),
            Ok((None, None)),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f32).unwrap()));
    }

    #[test]
    fn test_composite_with_symbolic_zero_vjp() {
        // The backward rule of `x ↦ (sin(x), cos(x))` receives the structural-zero seed of the unused second output as
        // a `MaybeZero::Zero` leaf and skips its term.
        type Tracer = DomainTracer<ArrayContext>;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let function = composite(|x: Tracer| Ok((x.sin()?, x.cos()?))).with_symbolic_zero_vjp(
            |x| Ok(((x.sin()?, x.cos()?), (x.cos()?, x.sin()?))),
            {
                let observed = observed.clone();
                move |(cosine, sine): (Tracer, Tracer), seeds: (MaybeZero<Tracer>, MaybeZero<Tracer>)| {
                    observed.lock().unwrap().push((seeds.0.is_zero(), seeds.1.is_zero()));
                    Ok(match seeds {
                        (MaybeZero::Value(sine_seed), MaybeZero::Value(cosine_seed)) => {
                            MaybeZero::Value(cosine * sine_seed - sine * cosine_seed)
                        }
                        (MaybeZero::Value(sine_seed), MaybeZero::Zero(_)) => MaybeZero::Value(cosine * sine_seed),
                        (MaybeZero::Zero(_), MaybeZero::Value(cosine_seed)) => MaybeZero::Value(-(sine * cosine_seed)),
                        (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => MaybeZero::Zero(r#type),
                    })
                }
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x).map(|(sine, _)| sine)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(x).map(|(sine, cosine)| sine + cosine)),
            Ok(Array::scalar(2.0f64.cos() - 2.0f64.sin()).unwrap()),
        );
        assert_eq!(*observed.lock().unwrap(), vec![(false, true), (false, false)]);
    }

    #[test]
    fn test_composite_with_symbolic_zero_vjp_zero_cotangents() {
        // The backward rule of `(scale, x) ↦ scale · x · x` treats `scale` as a parameter by returning a structural
        // zero for its cotangent. The zero is never computed: the pullback returns it as a structural zero, the
        // transposed program of a batched call has no output for it, and second-order differentiation differentiates
        // the backward rule's ordinary operations through its residuals.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|(scale, x): (Tracer, Tracer)| Ok(scale * x.clone() * x)).with_symbolic_zero_vjp(
            |(scale, x): (Tracer, Tracer)| Ok((scale.clone() * x.clone() * x.clone(), (scale, x))),
            |(scale, x): (Tracer, Tracer), seed: MaybeZero<Tracer>| {
                let scale_cotangent = MaybeZero::Zero(scale.r#type().into_owned());
                let x_cotangent = match seed {
                    MaybeZero::Value(seed) => MaybeZero::Value((scale.clone() + scale) * x * seed),
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type),
                };
                Ok((scale_cotangent, x_cotangent))
            },
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        let inputs = (Array::scalar(3.0).unwrap(), Array::scalar(2.0).unwrap());
        let (_, pullback) = differentiate_at(inputs.clone()).vjp(|inputs| function.call(inputs)).unwrap();
        assert_eq!(
            pullback.apply(Array::scalar(1.0).unwrap()),
            Ok((MaybeZero::Zero(scalar_type.clone()), MaybeZero::Value(Array::scalar(12.0).unwrap()))),
        );
        assert_eq!(
            differentiate_at(inputs.clone()).gradient(|inputs| function.call(inputs)),
            Ok((Array::scalar(0.0).unwrap(), Array::scalar(12.0).unwrap())),
        );

        // A call batched before it is differentiated batches the backward specialization, which keeps the structural
        // zero, so the pullback materializes it only at its own boundary.
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
            )
            .unwrap()
            .into_parts()
            .0;
        let linearization = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let pullback = linearization.tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[] = add %1 %1
                    %4:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %3
                    %5:f64[3] = mul %4 %2
                    %6:f64[3] = mul %5 %0
                    %7:f64[] = zero [type=f64[]]
                in (%7, %6)
            "}
            .trim_end(),
        );
        let residuals = linearization
            .primal()
            .interpret(vec![Array::scalar(3.0).unwrap(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()])
            .unwrap();
        let mut pullback_inputs = vec![Array::vector(vec![1.0; 3]).unwrap()];
        pullback_inputs.extend(residuals.into_iter().skip(1));
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::vector(vec![6.0, 12.0, 18.0]).unwrap()]),
        );

        // The second derivatives `∂²f/∂x∂scale = 2 x` and `∂²f/∂x² = 2 scale` come from the operations of the backward
        // rule.
        assert_eq!(
            differentiate_at(inputs).value_and_gradient(|inputs| {
                let (_, x_gradient) = differentiate_at(inputs).gradient(|inputs| function.call(inputs)).unwrap();
                x_gradient
            }),
            Ok((Array::scalar(12.0).unwrap(), (Array::scalar(4.0).unwrap(), Array::scalar(6.0).unwrap()))),
        );
    }

    #[test]
    fn test_composite_with_symbolic_zero_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        let function = composite(|x: ArrayIrTracer| Ok(x)).with_symbolic_zero_vjp(
            saved_extent,
            |_: ArrayIrTracer, seed: MaybeZero<ArrayIrTracer>| match seed {
                MaybeZero::Value(seed) => Ok(MaybeZero::Value(seed)),
                MaybeZero::Zero(_) => {
                    Err(ProgramError::InvalidArgument { message: "unexpected zero seed".to_string() })
                }
            },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(MaybeZero::Value(seed)));
    }

    #[test]
    fn test_composite_with_symbolic_zero_vjp_dead_output() {
        // Without residuals or non-differentiated inputs, the backward rule of a call whose output is dead would
        // receive neither leading inputs nor a non-zero seed. The adapter still invokes it in the existing trace, so
        // it can return a structural zero without recovering a context from a live input.
        type Tracer = DomainTracer<ArrayContext>;
        let invocations = Arc::new(AtomicUsize::new(0));
        let function = composite(|x: Tracer| Ok(x.sin()?)).with_symbolic_zero_vjp(|x: Tracer| Ok((x.sin()?, ())), {
            let invocations = invocations.clone();
            move |(): (), seed: MaybeZero<Tracer>| {
                invocations.fetch_add(1, Ordering::SeqCst);
                match seed {
                    MaybeZero::Value(seed) => Ok(MaybeZero::Value(seed)),
                    MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(r#type)),
                }
            }
        });
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| {
                let _dead = function.call(x.clone())?;
                Ok(x)
            }),
            Ok(Array::scalar(1.0).unwrap()),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_composite_with_accumulating_vjp() {
        // `(x, y) ↦ x[1] + y[1]` over `f64[4]` inputs. One accumulating backward rule serves every destination: it adds
        // the seed to only the affected entry of a caller buffer (without a full-sized temporary) and pads it into a
        // full cotangent for a returned destination. It is specialized once per pattern of destination kinds.
        type Transposition = TranspositionContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let element = |x: &ArrayIrTracer| -> Result<ArrayIrTracer, ProgramError> {
            Ok(ValueProjection::<ArrayType>::into_projected(x.clone())?.slice(&[1], &[2usize], &[1])?.into_value())
        };
        let add = |left: ArrayIrTracer, right: ArrayIrTracer| -> Result<ArrayIrTracer, ProgramError> {
            let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Add(AddOperation::new()));
            Ok(left.context().bind(operation, Vec::new(), &[left.clone(), right])?.remove(0))
        };
        let invocations = Arc::new(AtomicUsize::new(0));
        let function = composite(move |(x, y): (ArrayIrTracer, ArrayIrTracer)| add(element(&x)?, element(&y)?))
            .with_accumulating_vjp(
                move |(x, y): (ArrayIrTracer, ArrayIrTracer)| Ok((add(element(&x)?, element(&y)?)?, ())),
                {
                    let invocations = invocations.clone();
                    move |context: &mut Transposition,
                          (): (),
                          seed: MaybeZero<ArrayIrTracer>,
                          accumulators: (CotangentAccumulator, CotangentAccumulator)| {
                        invocations.fetch_add(1, Ordering::SeqCst);
                        let MaybeZero::Value(seed) = seed else {
                            return Ok(());
                        };
                        for accumulator in [accumulators.0, accumulators.1] {
                            if !accumulator.is_needed() {
                                continue;
                            }
                            if let Some(buffer) = accumulator.reference(context)? {
                                let operation = ReferenceAddUpdateOperation::new().with_transforms(vec![
                                    ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1)] },
                                ]);
                                context.bind(ArrayIrOperation::from(operation), Vec::new(), &[buffer, seed.clone()])?;
                            } else {
                                let zero = context.lift(ArrayIrValue::Array(Array::scalar(0.0f64)?))?;
                                let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Pad(
                                    PadOperation::new(vec![1], vec![2], vec![0])?,
                                ));
                                let cotangent = context.bind(operation, Vec::new(), &[seed.clone(), zero])?.remove(0);
                                accumulator.accumulate(context, MaybeZero::Value(cotangent))?;
                            }
                        }
                        Ok(())
                    }
                },
            );
        let vector = |values: [f64; 4]| ArrayIrValue::Array(Array::vector(values.to_vec()).unwrap());
        let seed = || CotangentSeed::Value(ArrayIrValue::Array(Array::vector(vec![5.0f64]).unwrap()));
        let (value, pullback) = differentiate_at((vector([1.0, 2.0, 3.0, 4.0]), vector([10.0, 20.0, 30.0, 40.0])))
            .vjp(|inputs| function.call(inputs))
            .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::vector(vec![22.0f64]).unwrap()));

        // Mixed destinations: a caller buffer receives only its affected entry, and a returned cotangent is padded.
        let buffer = ArrayReference::new(Array::vector(vec![1.0f64; 4]).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
                    CotangentDestination::Return
                ),
            ),
            Ok((None, Some(MaybeZero::Value(vector([0.0, 5.0, 0.0, 0.0]))))),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![1.0f64, 6.0, 1.0, 1.0]).unwrap()));
        assert_eq!(invocations.load(Ordering::SeqCst), 1);

        // Reapplying the same destination pattern reuses its specialization, while another pattern specializes again.
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
                    CotangentDestination::Return
                ),
            ),
            Ok((None, Some(MaybeZero::Value(vector([0.0, 5.0, 0.0, 0.0]))))),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![1.0f64, 11.0, 1.0, 1.0]).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(seed(), (CotangentDestination::Ignore, CotangentDestination::Return)),
            Ok((None, Some(MaybeZero::Value(vector([0.0, 5.0, 0.0, 0.0]))))),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 2);

        // Repeated inputs: both accumulators of `x ↦ f(x, x)` add to the same caller buffer.
        let (_, pullback) =
            differentiate_at(vector([1.0, 2.0, 3.0, 4.0])).vjp(|x| function.call((x.clone(), x))).unwrap();
        let buffer = ArrayReference::new(Array::vector(vec![0.0f64; 4]).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
            ),
            Ok(None),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![0.0f64, 10.0, 0.0, 0.0]).unwrap()));
    }

    #[test]
    fn test_composite_with_accumulating_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        type Transposition = TranspositionContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let function = composite(|x: ArrayIrTracer| Ok(x)).with_accumulating_vjp(
            saved_extent,
            |context: &mut Transposition,
             _: ArrayIrTracer,
             seed: MaybeZero<ArrayIrTracer>,
             accumulator: CotangentAccumulator| Ok(accumulator.accumulate(context, seed)?),
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(MaybeZero::Value(seed)));
    }

    #[test]
    fn test_composite_with_deferred_forward() {
        // `sin` whose forward rule saves `cos(x)`. Deferring the forward rule changes what the pullback keeps alive,
        // from that residual to the input `x`, but not the value or the gradient, for every form of the reverse mode
        // rules.
        type Tracer = DomainTracer<ArrayContext>;
        type Transposition = TranspositionContext<Array, ArrayOperation<Array>>;
        let x = Array::scalar(0.5f64).unwrap();
        let value = Array::scalar(0.5f64.sin()).unwrap();
        let seed = Array::scalar(2.0f64).unwrap();
        let gradient = Array::scalar(2.0 * 0.5f64.cos()).unwrap();
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_name("sine")
            .with_vjp(|x: Tracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
        let (output, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(output, value);
        assert_eq!(pullback.residuals(), &[Array::scalar(0.5f64.cos()).unwrap()]);
        assert_eq!(pullback.apply_materialized(seed.clone()), Ok(gradient.clone()));

        let function = function.with_deferred_forward();
        assert_eq!(
            format!("{function:?}"),
            "Composite { name: \"sine\", jvp: DefaultJvp, vjp: ExplicitVjp { deferred_forward: true, .. }, \
             batching: DefaultBatching, batched_output_axes: None, non_differentiated_count: 0, .. }",
        );
        let (output, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(output, value);
        assert_eq!(pullback.residuals(), std::slice::from_ref(&x));
        assert_eq!(pullback.apply_materialized(seed.clone()), Ok(gradient.clone()));

        // A backward rule that receives structural-zero seeds.
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_symbolic_zero_vjp(
                |x: Tracer| Ok((x.sin()?, x.cos()?)),
                |cosine: Tracer, seed: MaybeZero<Tracer>| match seed {
                    MaybeZero::Value(seed) => Ok(MaybeZero::Value(cosine * seed)),
                    MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(r#type)),
                },
            )
            .with_deferred_forward();
        let (output, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(output, value);
        assert_eq!(pullback.residuals(), std::slice::from_ref(&x));
        assert_eq!(pullback.apply_materialized(seed.clone()), Ok(gradient.clone()));

        // A backward rule that submits its cotangent to its accumulator.
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_accumulating_vjp(
                |x: Tracer| Ok((x.sin()?, x.cos()?)),
                |context: &mut Transposition,
                 cosine: Tracer,
                 seed: MaybeZero<Tracer>,
                 accumulator: CotangentAccumulator| {
                    if let MaybeZero::Value(seed) = seed {
                        accumulator.accumulate(context, MaybeZero::Value(cosine * seed))?;
                    }
                    Ok(())
                },
            )
            .with_deferred_forward();
        let (output, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(output, value);
        assert_eq!(pullback.residuals(), std::slice::from_ref(&x));
        assert_eq!(pullback.apply_materialized(seed), Ok(gradient));
    }

    #[test]
    fn test_composite_with_batching() {
        // The rule of `(x, y) ↦ x · y` computes `y · x`, which shows that it takes precedence over structurally
        // batching the primal, and declares that the output follows the mapped input. It is traced once per batching
        // level and input signature, which the extents that it records show.
        type Tracer = DomainTracer<ArrayContext>;
        let extents = Arc::new(Mutex::new(Vec::new()));
        let function = composite(|(x, y): (Tracer, Tracer)| Ok(x * y))
            .with_jvp(|(x, y): (Tracer, Tracer), (dx, dy): (Tracer, Tracer)| {
                let tangent = dx * y.clone() + x.clone() * dy;
                Ok((x * y, tangent.clone() + tangent))
            })
            .with_batching({
                let extents = extents.clone();
                move |(x, y): (Tracer, Tracer), axes: (BatchAxis, BatchAxis), extent: BatchingLevelExtent<Tracer>| {
                    let BatchingLevelExtent::Static(extent) = extent else {
                        return Err(ProgramError::InvalidArgument { message: "unexpected dynamic extent".to_string() });
                    };
                    extents.lock().unwrap().push(extent);
                    Ok((y * x, if axes.0 == BatchAxis::replicated() { axes.1 } else { axes.0 }))
                }
            });
        assert_eq!(
            batch(
                |(x, y)| function.call((x, y)),
                (Array::vector(vec![2.0, 3.0, 4.0]).unwrap(), Array::scalar(5.0).unwrap()),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![10.0, 15.0, 20.0]).unwrap()),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3]);

        // The staged batched call shares that trace, because it has the same level and input signature.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(
                3,
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
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[3] = composite [
                    name=\"composite\",
                    batching=[(extent=3, input_axes=[axis 0, replicated], output_axes=[axis 0])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3], %1:f64[] .
                        let %2:f64[3] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3]);

        // Batching preserves the call's derivative rules: the JVP rule, which doubles the derivative, governs the
        // differentiation of the batched call.
        let x = Array::vector(vec![2.0, 3.0, 4.0]).unwrap();
        let (y, x_tangent, y_tangent) =
            (Array::scalar(5.0).unwrap(), Array::vector(vec![1.0; 3]).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(
            batched.jvp().unwrap().interpret(vec![x, y, x_tangent, y_tangent]),
            Ok(vec![Array::vector(vec![10.0, 15.0, 20.0]).unwrap(), Array::vector(vec![14.0, 16.0, 18.0]).unwrap()]),
        );

        // Differentiating with respect to one input of the batched call specializes the JVP rule for that activity
        // pattern, materializing the other input's structural-zero tangent inside the batched specialization.
        assert_eq!(
            batched.jvp_with_respect_to(&[0]).unwrap().interpret(vec![
                Array::vector(vec![2.0, 3.0, 4.0]).unwrap(),
                Array::scalar(5.0).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![10.0, 15.0, 20.0]).unwrap(), Array::vector(vec![10.0; 3]).unwrap()]),
        );

        // Nested batching applies the rule again at the new level.
        let nested = batched
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            nested.interpret(vec![
                Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
                Array::scalar(5.0).unwrap(),
            ]),
            Ok(vec![Array::matrix(2, 3, vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0]).unwrap()]),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3, 2]);
    }

    #[test]
    fn test_composite_with_batching_references() {
        // The primal `(counter, x) ↦ x` adds `x` into its plumbing counter. Structurally batching it with a replicated
        // counter and a mapped `x` is rejected, because a replicated reference cannot receive a batched value, while
        // the batching rule adds the sum of the batch into the counter instead.
        let primal = |(counter, x): (ArrayIrTracer, ArrayIrTracer)| {
            counter.add_update(&x)?;
            Ok(x)
        };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let input_types = (reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let extent = DimensionValue::constant(3).unwrap();
        let axes = [BatchAxis::replicated(), BatchAxis::new(0)];
        let batched = |program: FlatProgram<EagerArrayIrContext>| {
            program.batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &axes,
                ProgramBatchingOutputAxesPolicy::Natural,
            )
        };
        let function = composite(primal).with_non_differentiated_count(1);
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types.clone()).unwrap();
        assert!(matches!(
            batched(program.into_flat_program()),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_add_update` cannot store a batched value into an unbatched reference; pass \
                               the reference as a batched input instead",
        ));

        let function = composite(primal).with_non_differentiated_count(1).with_batching(
            |(counter, x): (ArrayIrTracer, ArrayIrTracer),
             (_, x_axis): (BatchAxis, BatchAxis),
             _: BatchingLevelExtent<ArrayIrTracer>| {
                let sum = ValueProjection::<ArrayType>::into_projected(x.clone())?.reduce(&[0], ReductionKind::Sum)?;
                counter.add_update(&sum.into_value())?;
                Ok((x, x_axis))
            },
        );
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types).unwrap();
        let batched = batched(program.into_flat_program()).unwrap().into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:ref<f32[]>, %2:f32[3] .
                let %3:f32[3] = composite [
                    name=\"composite\",
                    non_differentiated_count=2,
                    batching=[(extent=dimension<3>, input_axes=[replicated, axis 0], output_axes=[axis 0])],
                ] %0 %1 %2 [
                    primal={
                        lambda %0:dimension<3>, %1:ref<f32[]>, %2:f32[3] .
                        let %3:f32[] = reduce [kind=sum, axes=[0]] %2
                            () = reference_add_update %1 %3
                        in (%2)
                    },
                ]
                in (%0, %3)
            "}
            .trim_end(),
        );
        let counter = ArrayReference::new(Array::scalar(1.0f32).unwrap());
        assert_eq!(
            batched.interpret(vec![
                ArrayIrValue::Dimension(extent.clone()),
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Dimension(extent),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(7.0f32).unwrap()));
    }

    #[test]
    fn test_composite_with_batching_casts() {
        // The primal narrows `x: f64[n]` to `f64[2]` with a checked cast before `sin`, while the batching rule applies
        // `sin` first and then narrows the batched value with a checked cast of its own, so that its program is
        // recognizable. The rule's check survives batching, differentiation of the batched call (which differentiates
        // the rule's program), and batching of the derived forward mode call (which applies the derivative of the
        // rule).
        type Tracer = DomainTracer<ArrayContext>;
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [2]);
        let function = composite(move |x: Tracer| x.cast(static_type.clone(), &[])?.sin()).with_batching(
            |x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| {
                let mut dimensions = x.r#type().shape().dimensions().to_vec();
                *dimensions.last_mut().unwrap() = Dimension::Static(2);
                let output_type = x.r#type().into_owned().with_shape(Shape::new(dimensions));
                Ok((x.sin()?.cast(output_type, &[])?, axis))
            },
        );
        let (_, program) = ArrayContext::trace(|x| function.call(x), symbolic_type).unwrap();
        let program = program.into_flat_program();
        let batched = |program: &FlatProgram<ArrayContext>, input_count: usize| {
            program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &vec![BatchAxis::new(0); input_count],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
        };
        let elements = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let x = Array::matrix(3, 2, elements.to_vec()).unwrap();
        let tangent = Array::matrix(3, 2, vec![1.0; 6]).unwrap();
        let wide = Array::matrix(3, 3, vec![1.0; 9]).unwrap();
        let sine = Array::matrix(3, 2, elements.map(f64::sin).to_vec()).unwrap();
        let cosine = Array::matrix(3, 2, elements.map(f64::cos).to_vec()).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f64[3, 3]` to unrelated type `f64[3, 2]`".to_string(),
        )));

        let batched_program = batched(&program, 1);
        assert_eq!(
            batched_program.to_string(),
            indoc! {"
                lambda %0:f64[3, n] .
                let %1:f64[3, 2] = composite [
                    name=\"composite\",
                    batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[3, n] .
                        let %1:f64[3, n] = sin %0
                            %2:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(batched_program.interpret(vec![x.clone()]), Ok(vec![sine.clone()]));
        assert_eq!(batched_program.interpret(vec![wide.clone()]), mismatch);

        // Batching then differentiating.
        let jvp_of_batched = batched_program.jvp().unwrap();
        assert_eq!(
            jvp_of_batched.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, n] .
                let %2:f64[3, n] = sin %0
                    %3:f64[3, n] = cos %0
                    %4:f64[3, n] = mul %3 %1
                    %5:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %2
                    %6:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, n], f64[3, 2]]] %2 %5
                    %7:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n], f64[3, n]]] %4 %6
                in (%5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(jvp_of_batched.interpret(vec![x.clone(), tangent.clone()]), Ok(vec![sine.clone(), cosine.clone()]),);
        assert_eq!(jvp_of_batched.interpret(vec![wide.clone(), wide.clone()]), mismatch);

        // Differentiating then batching: the derived forward mode call keeps the batching rule, whose derivative its
        // batching applies.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n] .
                let %2:f64[2], %3:f64[2] = composite [name=\"jvp(composite)\"] %0 %1 [
                    primal={
                        lambda %0:f64[n], %1:f64[n] .
                        let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                            %3:f64[n] = cast [type=f64[n], inputs=[f64[n], f64[2]]] %0 %2
                            %4:f64[2] = cast [type=f64[2], inputs=[f64[n], f64[n]]] %1 %3
                            %5:f64[2] = sin %2
                            %6:f64[2] = cos %2
                            %7:f64[2] = mul %6 %4
                        in (%5, %7)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        let batched_jvp = batched(&jvp, 2);
        assert_eq!(
            batched_jvp.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, n] .
                let %2:f64[3, 2], %3:f64[3, 2] = composite [
                    name=\"jvp(composite)\",
                    batching=[(extent=3, input_axes=[axis 0, axis 0], output_axes=[axis 0, axis 0])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3, n], %1:f64[3, n] .
                        let %2:f64[3, n] = sin %0
                            %3:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %2
                            %4:f64[3, n] = cos %0
                            %5:f64[3, n] = mul %4 %1
                            %6:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, n], f64[3, 2]]] %2 %3
                            %7:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n], f64[3, n]]] %5 %6
                        in (%3, %7)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(batched_jvp.interpret(vec![x, tangent]), Ok(vec![sine, cosine]));
        assert_eq!(batched_jvp.interpret(vec![wide.clone(), wide]), mismatch);
    }

    #[test]
    fn test_composite_with_batching_differentiation() {
        // A function with only a batching rule (i.e., bare `custom_vmap`) differentiates its primal in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_batching(|x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.sin()?, axis)));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_composite_with_batching_batch_then_differentiate() {
        // Batching the call applies its batching rule, so differentiating the batched call differentiates the rule's
        // program in both modes (i.e., `d(2 · sin(x)) = 2 · cos(x) · ẋ`).
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let jvp = batched.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = sin %0
                    %3:f64[3] = cos %0
                    %4:f64[3] = mul %3 %1
                    %5:f64[3] = sin %0
                    %6:f64[3] = cos %0
                    %7:f64[3] = mul %6 %1
                    %8:f64[3] = add %2 %5
                    %9:f64[3] = add %4 %7
                in (%8, %9)
            "}
            .trim_end(),
        );
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        assert_eq!(
            jvp.interpret(vec![Array::vector(x.to_vec()).unwrap(), ones.clone()]),
            Ok(vec![doubled_sine, doubled_cosine.clone()]),
        );
        let reverse = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let mut pullback_inputs = vec![ones];
        pullback_inputs
            .extend(reverse.primal().interpret(vec![Array::vector(x.to_vec()).unwrap()]).unwrap().into_iter().skip(1));
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![doubled_cosine]));
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch() {
        // Differentiating an unbatched call stages a derived call whose primal region is the derivative of the call's
        // primal region, and batching it applies the derivative of the batching rule, whatever the order in which the
        // two transforms are applied.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = composite [name=\"jvp(sine)\"] %0 %1 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        let (batched, output_axes) = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3], %3:f64[3] = composite [
                    name=\"jvp(sine)\",
                    batching=[(extent=3, input_axes=[axis 0, axis 0], output_axes=[axis 0, axis 0])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = sin %0
                            %3:f64[3] = sin %0
                            %4:f64[3] = add %2 %3
                            %5:f64[3] = cos %0
                            %6:f64[3] = mul %5 %1
                            %7:f64[3] = cos %0
                            %8:f64[3] = mul %7 %1
                            %9:f64[3] = add %6 %8
                        in (%4, %9)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(x.to_vec()).unwrap(), ones.clone()]),
            Ok(vec![doubled_sine.clone(), doubled_cosine.clone()]),
        );

        // Batching an eager derivative stages the derived call in the batching context directly.
        assert_eq!(
            batch(
                |(x, tangent)| Ok(differentiate_at(x).jvp(tangent, |x| function.call(x))?),
                (Array::vector(x.to_vec()).unwrap(), ones),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            ),
            Ok((doubled_sine, doubled_cosine)),
        );

        // Unbatched derivatives do not apply the batching rule.
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(0.5f64.sin()).unwrap(), Array::scalar(0.5f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_jacobian_forward() {
        // Linearization computes the outputs with the call itself and stages a pushforward call on its tangent side, so
        // the forward mode Jacobian (which batches that pushforward over its tangents) applies the rule's derivative.
        let function = doubled_sine_when_batched();
        let x = [0.5, 1.0, 1.5];
        let (_, doubled_cosine) = doubled_sine_and_cosine(&x);
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [3])).unwrap();
        let linearization = program.into_flat_program().linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = composite [name=\"sine\"] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = sin %0
                        in (%1)
                    },
                ]
                in (%1, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = composite [name=\"pushforward(sine)\"] %1 %0 [
                    primal={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = cos %0
                            %3:f64[3] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let batched = linearization
            .tangent()
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let tangents = Array::matrix(2, 3, vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5]).unwrap();
        let residual = Array::vector(x.to_vec()).unwrap();
        let doubled_cosine = doubled_cosine.to_f64s();
        assert_eq!(
            batched.interpret(vec![tangents, residual.clone()]),
            Ok(vec![
                Array::matrix(
                    2,
                    3,
                    doubled_cosine
                        .iter()
                        .chain(&doubled_cosine)
                        .enumerate()
                        .map(|(index, value)| { if index < 3 { *value } else { 0.5 * value } })
                        .collect()
                )
                .unwrap(),
            ]),
        );
        let jacobian = differentiate_at(residual).jacobian_forward(|x| function.call(x)).unwrap();
        assert_eq!(
            jacobian.values()[0],
            Array::matrix(
                3,
                3,
                (0..9).map(|index| if index % 4 == 0 { doubled_cosine[index / 4] } else { 0.0 }).collect()
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_mixed_axes() {
        // Every combination of mapped and replicated primals and tangents batches through the rule's derivative. The
        // first two broadcast the replicated side of a pair with one mapped side, and the last two batch the derivative
        // of the rule applied to replicated inputs structurally, because no input is mapped on both sides.
        let function = doubled_product_when_batched();
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let mapped = BatchAxis::new(0);
        let replicated = BatchAxis::replicated();
        let values = |axis: BatchAxis, value: [f64; 3]| match axis.is_replicated() {
            true => (Array::scalar(value[0]).unwrap(), vec![value[0]; 3]),
            false => (Array::vector(value.to_vec()).unwrap(), value.to_vec()),
        };
        for axes in [
            [mapped, mapped, mapped, mapped],
            [mapped, mapped, mapped, replicated],
            [mapped, replicated, mapped, mapped],
            [replicated, replicated, mapped, mapped],
            [mapped, replicated, replicated, mapped],
        ] {
            let (x, x_values) = values(axes[0], [1.0, 2.0, 3.0]);
            let (y, y_values) = values(axes[1], [4.0, 5.0, 6.0]);
            let (x_tangent, x_tangent_values) = values(axes[2], [1.0, 0.5, 0.25]);
            let (y_tangent, y_tangent_values) = values(axes[3], [2.0, 3.0, 4.0]);
            let (batched, output_axes) = jvp
                .batched(3, ShardingDimension::Replicated, &axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts();
            // The output is replicated exactly when both primals are, and its tangent is always mapped.
            let output_axis = if axes[0].is_replicated() && axes[1].is_replicated() { replicated } else { mapped };
            assert_eq!(output_axes, vec![output_axis, mapped], "{axes:?}");
            let expected_outputs = (0..3).map(|index| 2.0 * x_values[index] * y_values[index]).collect::<Vec<_>>();
            let expected_outputs = match output_axis.is_replicated() {
                true => Array::scalar(expected_outputs[0]).unwrap(),
                false => Array::vector(expected_outputs).unwrap(),
            };
            let expected_tangents = (0..3)
                .map(|index| {
                    2.0 * (x_tangent_values[index] * y_values[index] + x_values[index] * y_tangent_values[index])
                })
                .collect();
            assert_eq!(
                batched.interpret(vec![x, y, x_tangent, y_tangent]),
                Ok(vec![expected_outputs, Array::vector(expected_tangents).unwrap()]),
                "{axes:?}",
            );
        }
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_flag_sensitive_rule() {
        // The rule computes `2xy` when both inputs are mapped and `3xy` otherwise. The primal outputs of a batched
        // derivative always come from the rule applied at the primal inputs' batch axes, so they are those of the
        // batched call in both cases below.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_batching(
            |(x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                let product = x * y;
                let scale = if x_axis.is_replicated() || y_axis.is_replicated() { 3.0 } else { 2.0 };
                let scale = product.context().lift(Array::scalar(scale)?)?;
                Ok((product * scale, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let program = program.into_flat_program();
        let batched = |program: &FlatProgram<ArrayContext>, axes: &[BatchAxis], inputs: Vec<Array>| {
            let (batched, output_axes) = program
                .batched(3, ShardingDimension::Replicated, axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts();
            (batched.interpret(inputs).unwrap(), output_axes)
        };
        let jvp = program.jvp().unwrap();
        let mapped = BatchAxis::new(0);
        let replicated = BatchAxis::replicated();

        // For a mapped `x` and a replicated `y` whose tangent is mapped, the batched call returns `3xy`. The rule
        // cannot perturb the shared `y` differently for each batch item, so the tangent comes from the rule applied
        // with `y` broadcast, which is `2(ẋy + xẏ)`. JAX instead returns the outer product of the two batches, whose
        // diagonal is `3(ẋy + xẏ)` (i.e., `[18, 24, 39]` here). The two coincide for rules that do not depend on their
        // input axes (e.g., in `test_composite_with_batching_differentiate_then_batch_mixed_axes`).
        let x = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let y = Array::scalar(4.0f64).unwrap();
        let primal = Array::vector(vec![12.0f64, 24.0, 36.0]).unwrap();
        assert_eq!(
            batched(&program, &[mapped, replicated], vec![x.clone(), y.clone()]),
            (vec![primal.clone()], vec![mapped]),
        );
        assert_eq!(
            batched(
                &jvp,
                &[mapped, replicated, mapped, mapped],
                vec![
                    x,
                    y,
                    Array::vector(vec![1.0f64, 0.5, 0.25]).unwrap(),
                    Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap(),
                ],
            ),
            (vec![primal, Array::vector(vec![12.0f64, 16.0, 26.0]).unwrap()], vec![mapped; 2]),
        );

        // For mapped primals with replicated tangents, the rule is applied once, with both inputs mapped, so the
        // batched call and the batched derivative return `2xy`, and the derivative returns `2(ẋy + xẏ)`.
        let x = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let y = Array::vector(vec![4.0f64, 5.0, 6.0]).unwrap();
        let primal = Array::vector(vec![8.0f64, 20.0, 36.0]).unwrap();
        assert_eq!(
            batched(&program, &[mapped, mapped], vec![x.clone(), y.clone()]),
            (vec![primal.clone()], vec![mapped]),
        );
        assert_eq!(
            batched(
                &jvp,
                &[mapped, mapped, replicated, replicated],
                vec![x, y, Array::scalar(1.0f64).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            (vec![primal, Array::vector(vec![12.0f64, 18.0, 24.0]).unwrap()], vec![mapped; 2]),
        );

        // A broadcast sensitive configuration of the same rule applies it once, at the primal axes, in the first case
        // too, and recovers the per-item tangents of the replicated `y` from the outer product of its batch of tangents
        // with the rule's batch, so they are exactly the per-item derivatives `3(ẋy + xẏ)` of the batched primal `3xy`.
        let function = composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_broadcast_sensitive_batching(
            |(x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                let product = x * y;
                let scale = if x_axis.is_replicated() || y_axis.is_replicated() { 3.0 } else { 2.0 };
                let scale = product.context().lift(Array::scalar(scale)?)?;
                Ok((product * scale, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        );
        let (_, program) = ArrayContext::trace(
            |inputs| function.call(inputs),
            (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            batched(
                &jvp,
                &[mapped, replicated, mapped, mapped],
                vec![
                    Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap(),
                    Array::scalar(4.0f64).unwrap(),
                    Array::vector(vec![1.0f64, 0.5, 0.25]).unwrap(),
                    Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap(),
                ],
            ),
            (
                vec![
                    Array::vector(vec![12.0f64, 24.0, 36.0]).unwrap(),
                    Array::vector(vec![18.0f64, 24.0, 39.0]).unwrap(),
                ],
                vec![mapped; 2],
            ),
        );
        assert_eq!(
            batched(
                &jvp,
                &[mapped, mapped, replicated, replicated],
                vec![
                    Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap(),
                    Array::vector(vec![4.0f64, 5.0, 6.0]).unwrap(),
                    Array::scalar(1.0f64).unwrap(),
                    Array::scalar(2.0f64).unwrap(),
                ],
            ),
            (
                vec![
                    Array::vector(vec![8.0f64, 20.0, 36.0]).unwrap(),
                    Array::vector(vec![12.0f64, 18.0, 24.0]).unwrap(),
                ],
                vec![mapped; 2],
            ),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_non_differentiated_inputs() {
        // A derived call receives no tangents for non-differentiated inputs, which keep their batch axes when the rule
        // is applied.
        let function = doubled_product_when_batched().with_non_differentiated_count(1);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let jvp = program.into_flat_program().jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[] = composite [name=\"jvp(product)\", non_differentiated_count=1] %0 %1 %2 [
                    primal={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %0 %1
                            %4:f64[] = mul %0 %2
                        in (%3, %4)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.interpret(vec![
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![4.0, 5.0, 6.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![8.0, 20.0, 36.0]).unwrap(), Array::vector(vec![2.0, 4.0, 6.0]).unwrap(),]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_zero_space_inputs() {
        // Zero-space inputs (e.g., integers) have no tangents, so a derived call receives none for them, while the rule
        // still receives them with their batch axes.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|(x, _): (Tracer, Tracer)| Ok(x.sin()?)).with_name("sine").with_batching(
            |(x, _): (Tracer, Tracer), (axis, _): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                Ok((x.sin()? + x.sin()?, axis))
            },
        );
        let (_, program) = ArrayContext::trace(
            |inputs| function.call(inputs),
            (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::I32)),
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:i32[], %2:f64[] .
                let %3:f64[], %4:f64[] = composite [name=\"jvp(sine)\"] %0 %1 %2 [
                    primal={
                        lambda %0:f64[], %1:i32[], %2:f64[] .
                        let %3:f64[] = sin %0
                            %4:f64[] = cos %0
                            %5:f64[] = mul %4 %2
                        in (%3, %5)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 3],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                Array::vector(vec![1i32, 2, 3]).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
            ]),
            Ok(vec![doubled_sine, doubled_cosine]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_non_batchable_primal() {
        // The primal's derivative calls a foreign kernel, which cannot be batched (nor executed by the reference
        // backend), so batching a derivative succeeds only through the derivative of the batching rule, in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let kernel =
            Composite::from_custom_call(CustomCallOperation::new("my_sin", vec![ArrayType::scalar(DataType::F64)]))
                .with_jvp(|inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
                    let kernel = CustomCallOperation::new("my_sin", vec![ArrayType::scalar(DataType::F64)]);
                    Ok((CustomCall::custom_call(&kernel, &inputs)?, vec![inputs[0].cos()? * tangents[0].clone()]))
                });
        let function = composite(move |x: Tracer| Ok(kernel.call(vec![x])?.remove(0))).with_name("sine").with_batching(
            |x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.sin()? + x.sin()?, axis)),
        );
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        for axes in [[BatchAxis::new(0), BatchAxis::new(0)], [BatchAxis::replicated(), BatchAxis::new(0)]] {
            let batched = jvp
                .batched(3, ShardingDimension::Replicated, &axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts()
                .0;
            let expected = match axes[0].is_replicated() {
                true => vec![
                    Array::scalar(2.0 * 0.5f64.sin()).unwrap(),
                    Array::vector(vec![2.0 * 0.5f64.cos(), 1.0 * 0.5f64.cos(), 0.5 * 0.5f64.cos()]).unwrap(),
                ],
                false => vec![doubled_sine.clone(), doubled_cosine.clone()],
            };
            let primal = match axes[0].is_replicated() {
                true => Array::scalar(0.5).unwrap(),
                false => Array::vector(x.to_vec()).unwrap(),
            };
            let tangents = match axes[0].is_replicated() {
                true => Array::vector(vec![1.0, 0.5, 0.25]).unwrap(),
                false => Array::vector(vec![1.0; 3]).unwrap(),
            };
            assert_eq!(batched.interpret(vec![primal, tangents]), Ok(expected), "{axes:?}");
        }
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_projection() {
        // Projections of array IR contexts lift the regions of the member operations that they bind, so a function
        // written against a projection stages its derived call in the array IR family.
        type Tracer = DomainTracer<ProjectedContext<EagerArrayIrContext, ArrayType>>;
        let function = composite(|x: Tracer| Ok(x.sin()?)).with_batching(
            |x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.sin()? + x.sin()?, axis)),
        );
        let (_, program) = EagerArrayIrContext::trace(
            |x: ArrayIrTracer| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                let (value, tangent) = differentiate_at(x.clone()).jvp(x, |x| function.call(x))?;
                Ok::<_, ProgramError>((value.into_value(), tangent.into_value()))
            },
            ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = composite [name=\"jvp(composite)\"] %0 %0 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_output_axes() {
        // The derived batching rule keeps the batch axes that the rule declares, including nonzero ones, and moves a
        // tangent that is mapped along another axis than its primal to its primal's axis.
        let function = doubled_sine_when_batched();
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let (batched, output_axes) = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1), BatchAxis::new(1)]);
        let x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let tangents = Array::matrix(3, 2, vec![1.0; 6]).unwrap();
        assert_eq!(
            batched.interpret(vec![Array::matrix(2, 3, x.to_vec()).unwrap(), tangents]),
            Ok(vec![
                Array::matrix(2, 3, doubled_sine.to_f64s()).unwrap(),
                Array::matrix(2, 3, doubled_cosine.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_dynamic_extent() {
        // A first-class batch extent reaches the derived batching rule as the level's boundary input, which the source
        // rule and the batched derivative both consume.
        type Tracer = DomainTracer<EagerArrayIrContext>;
        let sine = |x: Tracer| ValueProjection::<ArrayType>::into_projected(x)?.sin();
        let function = composite(move |x: Tracer| Ok(sine(x)?.into_value())).with_name("sine").with_batching(
            move |x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| {
                Ok(((sine(x.clone())? + sine(x)?).into_value(), axis))
            },
        );
        let (_, program) =
            EagerArrayIrContext::trace(|x| function.call(x), ArrayIrType::Array(ArrayType::scalar(DataType::F64)))
                .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let (batched, output_axes) = jvp
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            batched.interpret(vec![
                ArrayIrValue::Dimension(extent.clone()),
                ArrayIrValue::Array(Array::vector(x.to_vec()).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0; 3]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Dimension(extent),
                ArrayIrValue::Array(doubled_sine),
                ArrayIrValue::Array(doubled_cosine),
            ]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_shared_derivation() {
        // Calls whose primal regions differ share one derived definition, while each keeps its own primal region.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(
            |(x, y)| Ok((function.call(x)?, function.call(y)?)),
            (ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [2])),
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[2], %2:f64[], %3:f64[2] .
                let %4:f64[], %5:f64[] = composite [name=\"jvp(sine)\"] %0 %2 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                    %6:f64[2], %7:f64[2] = composite [name=\"jvp(sine)\"] %1 %3 [
                        primal={
                            lambda %0:f64[2], %1:f64[2] .
                            let %2:f64[2] = sin %0
                                %3:f64[2] = cos %0
                                %4:f64[2] = mul %3 %1
                            in (%2, %4)
                        },
                    ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );

        // The rendering names the rules of both calls but cannot show that they share one derived definition, which
        // only their rule references record.
        let derived_rules = jvp
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Composite(operation) => operation.rules().unwrap().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(derived_rules.len(), 2);
        assert_eq!(derived_rules[0], derived_rules[1]);
        assert_ne!(jvp.instructions()[0].regions()[0], jvp.instructions()[1].regions()[0]);
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let y = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine_x, doubled_cosine_x) = doubled_sine_and_cosine(&x);
        let (doubled_sine_y, doubled_cosine_y) = doubled_sine_and_cosine(&y);
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                Array::matrix(3, 2, y.to_vec()).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
                Array::matrix(3, 2, vec![1.0; 6]).unwrap(),
            ]),
            Ok(vec![
                doubled_sine_x,
                Array::matrix(3, 2, doubled_sine_y.to_f64s()).unwrap(),
                doubled_cosine_x,
                Array::matrix(3, 2, doubled_cosine_y.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_effectful_primal() {
        // A pushforward that recomputed this primal would repeat its reference update and read the updated counter, so
        // linearization inlines the derivative of an effectful primal instead: the effect runs once, when the known
        // side runs, and every pushforward sees the coefficient that the known side observed.
        type Tracer = DomainTracer<EagerArrayIrContext>;
        let scaled_by_incremented_counter = |(counter, x): (Tracer, Tracer)| {
            let one = counter.context().lift(ArrayIrValue::Array(Array::scalar(1.0)?))?;
            counter.add_update(&one)?;
            let scale = ValueProjection::<ArrayType>::into_projected(counter.read()?)?;
            Ok::<_, ProgramError>((scale * ValueProjection::<ArrayType>::into_projected(x)?).into_value())
        };
        let function = composite(scaled_by_incremented_counter).with_non_differentiated_count(1).with_batching(
            move |inputs: (Tracer, Tracer), (_, axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                Ok((scaled_by_incremented_counter(inputs)?, axis))
            },
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let (_, program) = EagerArrayIrContext::trace(
            |inputs| function.call(inputs),
            (reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        )
        .unwrap();
        let linearization = program.into_flat_program().linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:f64[] .
                let %2:f64[] = const 1.0
                    () = reference_add_update %0 %2
                    %3:f64[] = reference_read %0
                    %4:f64[] = mul %3 %1
                in (%4, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let counter = ArrayReference::new(Array::scalar(0.0).unwrap());
        let outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Reference(counter.clone()), ArrayIrValue::Array(Array::scalar(2.0).unwrap())])
            .unwrap();
        assert_eq!(counter.read(), Ok(Array::scalar(1.0).unwrap()));
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())];
        tangent_inputs.extend(outputs.into_iter().skip(1));
        for _ in 0..2 {
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs.clone()),
                Ok(vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())]),
            );
        }
        assert_eq!(counter.read(), Ok(Array::scalar(1.0).unwrap()));
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_higher_order() {
        // A derived call is differentiated like its source, so higher-order derivatives keep the rule on their batching
        // path. Nested batching applies the derived rule at every level, where the source rule batches the source
        // function at the batched types (i.e., the rule is not composed with itself), as for the source call.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let second_order = program.into_flat_program().jvp().unwrap().jvp().unwrap();
        assert_eq!(
            second_order.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[] = composite [name=\"jvp(jvp(sine))\"] %0 %1 %2 %3 [
                    primal={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %0
                            %5:f64[] = cos %0
                            %6:f64[] = mul %5 %2
                            %7:f64[] = cos %0
                            %8:f64[] = sin %0
                            %9:f64[] = mul %8 %2
                            %10:f64[] = neg %9
                            %11:f64[] = mul %7 %1
                            %12:f64[] = mul %1 %10
                            %13:f64[] = mul %7 %3
                            %14:f64[] = add %12 %13
                        in (%4, %11, %6, %14)
                    },
                ]
                in (%4, %5, %6, %7)
            "}
            .trim_end(),
        );
        let batched = second_order
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let negated_doubled_sine = Array::vector(x.iter().map(|x| -2.0 * f64::sin(*x)).collect()).unwrap();
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                ones.clone(),
                ones.clone(),
                Array::vector(vec![0.0; 3]).unwrap()
            ]),
            Ok(vec![doubled_sine, doubled_cosine.clone(), doubled_cosine, negated_doubled_sine]),
        );

        let nested = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64))
            .unwrap()
            .1
            .into_flat_program()
            .jvp()
            .unwrap()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 2],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 2],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            nested
                .interpret(vec![Array::matrix(2, 3, x.to_vec()).unwrap(), Array::matrix(2, 3, vec![1.0; 6]).unwrap()]),
            Ok(vec![
                Array::matrix(2, 3, doubled_sine.to_f64s()).unwrap(),
                Array::matrix(2, 3, doubled_cosine.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_composite_with_batching_differentiate_then_batch_reverse_mode() {
        // Reverse mode inlines the derivative of the primal instead of staging a derived call, and a function with
        // reverse mode rules keeps using them.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let reverse = program
            .into_flat_program()
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(0.5f64.cos()).unwrap()),
        );

        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_jvp_from_primal()
            .with_vjp(
                |x| Ok((x.sin()?, x.cos()?)),
                |cosine, cotangent| Ok(cosine.clone() * cotangent.clone() + cosine * cotangent),
            )
            .with_batching(|x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| {
                Ok((x.sin()? + x.sin()?, axis))
            });
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0 * 0.5f64.cos()).unwrap()),
        );
        assert_eq!(
            batch(
                |(x, tangent)| Ok(differentiate_at(x).jvp(tangent, |x| function.call(x))?),
                (Array::vector(vec![0.5, 1.0]).unwrap(), Array::vector(vec![1.0, 1.0]).unwrap()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            ),
            Ok(doubled_sine_and_cosine(&[0.5, 1.0])),
        );
    }

    #[test]
    fn test_composite_with_broadcast_sensitive_batching() {
        // The rule of `(x, y, z) ↦ (xy, yz)` computes the batched products for every batch axis pattern, so it does not
        // depend on its axes, and the broadcast sensitive configuration must give exactly the per-item derivatives
        // that the plain configuration gives. With `x` mapped and `y` and `z` replicated, the mapped tangent `ẏ` (and,
        // in the second case, `ż`) of a replicated primal is recovered from an outer product, and the tangent of the
        // replicated output `yz` sums the broadcast tangent `yż` with the per-item tangents `ẏᵢz`.
        type Tracer = DomainTracer<ArrayContext>;
        let primal = |(x, y, z): (Tracer, Tracer, Tracer)| Ok((x * y.clone(), y * z));
        let rule = |(x, y, z): (Tracer, Tracer, Tracer),
                    (x_axis, y_axis, z_axis): (BatchAxis, BatchAxis, BatchAxis),
                    _: BatchingLevelExtent<Tracer>| {
            let mapped = |left: BatchAxis, right: BatchAxis| if left.is_replicated() { right } else { left };
            Ok(((x * y.clone(), y * z), (mapped(x_axis, y_axis), mapped(y_axis, z_axis))))
        };
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input_types = (scalar_type.clone(), scalar_type.clone(), scalar_type);
        let broadcast_sensitive = composite(primal).with_broadcast_sensitive_batching(rule);
        let (_, program) = ArrayContext::trace(|inputs| broadcast_sensitive.call(inputs), input_types.clone()).unwrap();
        let program = program.into_flat_program();
        let plain = composite(primal).with_batching(rule);
        let (_, plain_program) = ArrayContext::trace(|inputs| plain.call(inputs), input_types).unwrap();
        let plain_program = plain_program.into_flat_program();
        let batched = |program: &FlatProgram<ArrayContext>, axes: &[BatchAxis], inputs: Vec<Array>| {
            let (batched, output_axes) = program
                .batched(3, ShardingDimension::Replicated, axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts();
            let outputs = batched.interpret(inputs).unwrap();
            (batched, outputs, output_axes)
        };
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let scalar = |value: f64| Array::scalar(value).unwrap();
        let primals = vec![vector(vec![1.0, 2.0, 3.0]), scalar(4.0), scalar(5.0)];

        // Only `ẏ` belongs to a replicated primal: `ẋy + xẏ = [6, 8, 13]` and `ẏz + yż = [34, 39, 44]`.
        let axes = [mapped, replicated, replicated, mapped, mapped, replicated];
        let mut inputs = primals.clone();
        inputs.extend([vector(vec![1.0, 0.5, 0.25]), vector(vec![2.0, 3.0, 4.0]), scalar(6.0)]);
        let jvp = program.jvp().unwrap();
        let (batched_jvp, outputs, output_axes) = batched(&jvp, &axes, inputs.clone());
        let expected = vec![
            vector(vec![4.0, 8.0, 12.0]),
            scalar(20.0),
            vector(vec![6.0, 8.0, 13.0]),
            vector(vec![34.0, 39.0, 44.0]),
        ];
        assert_eq!(outputs, expected);
        assert_eq!(output_axes, vec![mapped, replicated, mapped, mapped]);
        assert_eq!(batched(&plain_program.jvp().unwrap(), &axes, inputs).1, expected);
        assert_eq!(batched_jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[], %3:f64[3], %4:f64[3], %5:f64[] .
                let %6:f64[3], %7:f64[], %8:f64[3], %9:f64[3] = composite [
                    name=\"jvp(composite)\",
                    batching=[(extent=3, input_axes=[axis 0, replicated, replicated, axis 0, axis 0, replicated], output_axes=[axis 0, replicated, axis 0, axis 0])],
                ] %0 %1 %2 %3 %4 %5 [
                    primal={
                        lambda %0:f64[3], %1:f64[], %2:f64[], %3:f64[3], %4:f64[3], %5:f64[] .
                        let %6:f64[3] = mul %0 %1
                            %7:f64[] = mul %1 %2
                            %8:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                            %9:f64[3] = mul %8 %3
                            %10:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[1]] %0
                            %11:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[0]] %4
                            %12:f64[3, 3] = mul %10 %11
                            %13:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                            %14:f64[3] = gather [
                                dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                                slice_sizes=[1, 1],
                            ] %12 %13
                            %15:f64[3] = add %9 %14
                            %16:f64[] = mul %1 %5
                            %17:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %16
                            %18:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %2
                            %19:f64[3] = mul %18 %4
                            %20:f64[3] = add %17 %19
                        in (%6, %7, %15, %20)
                    },
                ]
                in (%6, %7, %8, %9)
            "}
            .trim_end(),
        );

        // Both `ẏ` and `ż` belong to replicated primals: `ẏz + yż = [34, 43, 52]` has no following contribution.
        let axes = [mapped, replicated, replicated, mapped, mapped, mapped];
        let mut inputs = primals.clone();
        inputs.extend([vector(vec![1.0, 0.5, 0.25]), vector(vec![2.0, 3.0, 4.0]), vector(vec![6.0, 7.0, 8.0])]);
        let expected = vec![
            vector(vec![4.0, 8.0, 12.0]),
            scalar(20.0),
            vector(vec![6.0, 8.0, 13.0]),
            vector(vec![34.0, 43.0, 52.0]),
        ];
        assert_eq!(batched(&jvp, &axes, inputs.clone()).1, expected);
        assert_eq!(batched(&plain_program.jvp().unwrap(), &axes, inputs).1, expected);

        // Linearization stages a pushforward call, whose batching takes the same exact path when its residual primals
        // have the axes above, and structural batching when they are replicated (e.g., in a forward mode Jacobian).
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                let %6:f64[], %7:f64[] = composite [name=\"pushforward(composite)\"] %3 %4 %5 %0 %1 %2 [
                    primal={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                        let %6:f64[] = mul %1 %3
                            %7:f64[] = mul %0 %4
                            %8:f64[] = add %6 %7
                            %9:f64[] = mul %2 %4
                            %10:f64[] = mul %1 %5
                            %11:f64[] = add %9 %10
                        in (%8, %11)
                    },
                ]
                in (%6, %7)
            "}
            .trim_end(),
        );
        let tangent = linearization.tangent();
        let tangents = vec![vector(vec![1.0, 0.5, 0.25]), vector(vec![2.0, 3.0, 4.0]), scalar(6.0)];
        let mut inputs = tangents.clone();
        inputs.extend(primals.clone());
        let (_, outputs, output_axes) =
            batched(tangent, &[mapped, mapped, replicated, mapped, replicated, replicated], inputs);
        assert_eq!(outputs, vec![vector(vec![6.0, 8.0, 13.0]), vector(vec![34.0, 39.0, 44.0])]);
        assert_eq!(output_axes, vec![mapped; 2]);

        // A forward mode Jacobian at `(x, y, z) = (2, 4, 5)` batches the pushforward over the basis tangents with
        // replicated primals, which applies the rule once and batches its derivative structurally.
        let mut inputs = vec![vector(vec![1.0, 0.0, 0.0]), vector(vec![0.0, 1.0, 0.0]), vector(vec![0.0, 0.0, 1.0])];
        inputs.extend([scalar(2.0), scalar(4.0), scalar(5.0)]);
        let (_, outputs, output_axes) =
            batched(tangent, &[mapped, mapped, mapped, replicated, replicated, replicated], inputs);
        assert_eq!(outputs, vec![vector(vec![4.0, 2.0, 0.0]), vector(vec![0.0, 5.0, 4.0])]);
        assert_eq!(output_axes, vec![mapped; 2]);
    }

    #[test]
    fn test_composite_with_broadcast_sensitive_batching_dynamic_extent() {
        // The rule of `y ↦ 3y` maps its output itself: it computes `3y` broadcast along the batch when `y` is replicated
        // and `2y` otherwise. With a replicated `y` whose tangent is mapped along a first-class extent, the plain
        // configuration applies the rule with `y` broadcast, which gives `2ẏ`, while the broadcast sensitive
        // configuration takes the diagonal of the outer product over that extent, which gives the per-item
        // derivatives `3ẏ` of the batched primal.
        type Tracer = DomainTracer<EagerArrayIrContext>;
        let add = |left: &Tracer, right: &Tracer| -> Result<Tracer, ProgramError> {
            let left = ValueProjection::<ArrayType>::into_projected(left.clone())?;
            Ok((left + ValueProjection::<ArrayType>::into_projected(right.clone())?).into_value())
        };
        let primal = move |y: Tracer| add(&add(&y, &y)?, &y);
        let rule = move |y: Tracer, axis: BatchAxis, extent: BatchingLevelExtent<Tracer>| match (extent, axis.axis()) {
            (BatchingLevelExtent::Dynamic(extent), None) => {
                Ok((primal(y)?.dynamic_broadcast(&[extent], &[])?, BatchAxis::new(0)))
            }
            (_, Some(_)) => Ok((add(&y, &y)?, axis)),
            (BatchingLevelExtent::Static(_), None) => {
                Err(ProgramError::InvalidArgument { message: "the test rule expects a dynamic extent".to_string() })
            }
        };
        let extent_type = DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let extent = DimensionValue::new(extent_type.clone(), 3).unwrap();
        let batched_jvp = |program: FlatProgram<EagerArrayIrContext>| {
            program
                .jvp()
                .unwrap()
                .batched_with_threaded_extent(
                    extent_type.clone(),
                    ShardingDimension::Replicated,
                    &[BatchAxis::replicated(), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
        };
        let inputs = vec![
            ArrayIrValue::Dimension(extent.clone()),
            ArrayIrValue::Array(Array::scalar(4.0f64).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
        ];
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let outputs = |tangent: Vec<f64>| {
            Ok(vec![
                ArrayIrValue::Dimension(extent.clone()),
                ArrayIrValue::Array(Array::vector(vec![12.0f64; 3]).unwrap()),
                ArrayIrValue::Array(Array::vector(tangent).unwrap()),
            ])
        };

        let function = composite(primal).with_batching(rule);
        let (_, program) = EagerArrayIrContext::trace(|y| function.call(y), scalar_type.clone()).unwrap();
        let (batched, output_axes) = batched_jvp(program.into_flat_program());
        assert_eq!(output_axes, vec![BatchAxis::new(0); 2]);
        assert_eq!(batched.interpret(inputs.clone()), outputs(vec![2.0, 4.0, 6.0]));

        let function = composite(primal).with_broadcast_sensitive_batching(rule);
        let (_, program) = EagerArrayIrContext::trace(|y| function.call(y), scalar_type).unwrap();
        let (batched, output_axes) = batched_jvp(program.into_flat_program());
        assert_eq!(output_axes, vec![BatchAxis::new(0); 2]);
        assert_eq!(batched.interpret(inputs), outputs(vec![3.0, 6.0, 9.0]));
        assert_eq!(batched.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [1, 8)>, %1:f64[], %2:f64[n] .
                let %3:f64[n], %4:f64[n] = composite [
                    name=\"jvp(composite)\",
                    non_differentiated_count=1,
                    batching=[(extent=dimension<n ∈ [1, 8)>, input_axes=[replicated, axis 0], output_axes=[axis 0, axis 0])],
                ] %0 %1 %2 [
                    primal={
                        lambda %0:dimension<n ∈ [1, 8)>, %1:f64[], %2:f64[n] .
                        let %3:f64[] = add %1 %1
                            %4:f64[] = add %3 %1
                            %5:f64[n] = broadcast [output_axes=[]] %4 %0
                            %6:f64[n] = add %2 %2
                            %7:f64[n] = add %6 %2
                            %8:f64[n, n] = broadcast [output_axes=[0]] %7 %0 %0
                            %9:dimension<n ∈ [1, 8)> = dimension_size [axis=1] %8
                            %10:i64[n, 1] = iota [type=i64[n, 1], dimension=0] %9
                            %11:f64[n] = gather [
                                dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                                slice_sizes=[1, 1],
                            ] %8 %10
                        in (%5, %11)
                    },
                ]
                in (%0, %3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_with_broadcast_sensitive_batching_nested_levels() {
        // A derived call batched at an earlier level reaches the exact path with that level's batch axes on its inputs.
        // The rule of `(x, y) ↦ xy` does not depend on its axes, so the per-item derivative `ẋy + xẏ` (with `y` mapped
        // only at the inner level) is also what the plain configuration computes.
        type Tracer = DomainTracer<ArrayContext>;
        let primal = |(x, y): (Tracer, Tracer)| Ok(x * y);
        let rule =
            |(x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                Ok((x * y, if x_axis.is_replicated() { y_axis } else { x_axis }))
            };
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mapped = BatchAxis::new(0);
        let batched_jvp = |program: FlatProgram<ArrayContext>| {
            let (inner, _) = program
                .jvp()
                .unwrap()
                .batched(2, ShardingDimension::Replicated, &[mapped; 4], ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts();
            inner
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[mapped, BatchAxis::replicated(), mapped, mapped],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
        };
        let x = Array::matrix(3, 2, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::vector(vec![10.0f64, 20.0]).unwrap();
        let x_tangent = Array::matrix(3, 2, vec![1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y_tangent = Array::matrix(3, 2, vec![2.0f64, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let expected = vec![
            Array::matrix(3, 2, vec![10.0f64, 40.0, 30.0, 80.0, 50.0, 120.0]).unwrap(),
            Array::matrix(3, 2, vec![12.0f64, 6.0, 12.0, 40.0, 40.0, 62.0]).unwrap(),
        ];
        let inputs = vec![x, y, x_tangent, y_tangent];

        let function = composite(primal).with_batching(rule);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type.clone())).unwrap();
        let (batched, output_axes) = batched_jvp(program.into_flat_program());
        assert_eq!(output_axes, vec![mapped; 2]);
        assert_eq!(batched.interpret(inputs.clone()), Ok(expected.clone()));

        let function = composite(primal).with_broadcast_sensitive_batching(rule);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let (batched, output_axes) = batched_jvp(program.into_flat_program());
        assert_eq!(output_axes, vec![mapped; 2]);
        assert_eq!(batched.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_composite_with_broadcast_sensitive_batching_independent_outputs() {
        // In `(x, y) ↦ (x · x, x · y)` with only `ẏ` active, `ẏ` belongs to the replicated `y`, so it reaches the
        // tangent-only contribution. The output `x · x` does not depend on `y`, so that contribution is a zero that is
        // replicated along the second level, which keeps the batch axis of the mapped output without a diagonal.
        type Tracer = DomainTracer<ArrayContext>;
        let primal = |(x, y): (Tracer, Tracer)| Ok((x.clone() * x.clone(), x * y));
        let rule =
            |(x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis), _: BatchingLevelExtent<Tracer>| {
                let product_axis = if x_axis.is_replicated() { y_axis } else { x_axis };
                Ok(((x.clone() * x.clone(), x * y), (x_axis, product_axis)))
            };
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function = composite(primal).with_broadcast_sensitive_batching(rule);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let (batched, output_axes) = program
            .into_flat_program()
            .jvp_with_respect_to(&[1])
            .unwrap()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();

        // With `x = [1, 2, 3]`, `y = 4`, and `ẏ = [2, 3, 4]`, the tangents are `0` and `xẏ = [2, 6, 12]`. The
        // derivative of the call's primal does not depend on `ẏ` for `x · x`, so the tangent of that output is a
        // replicated zero.
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let scalar = |value: f64| Array::scalar(value).unwrap();
        assert_eq!(output_axes, vec![mapped, mapped, replicated, mapped]);
        assert_eq!(
            batched.interpret(vec![vector(vec![1.0, 2.0, 3.0]), scalar(4.0), vector(vec![2.0, 3.0, 4.0])]),
            Ok(vec![
                vector(vec![1.0, 4.0, 9.0]),
                vector(vec![4.0, 8.0, 12.0]),
                scalar(0.0),
                vector(vec![2.0, 6.0, 12.0]),
            ]),
        );
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[3], %4:f64[3], %5:f64[3], %6:f64[3] = composite [
                    name=\"jvp(composite)\",
                    batching=[(extent=3, input_axes=[axis 0, replicated, axis 0], \
                        output_axes=[axis 0, axis 0, axis 0, axis 0])],
                ] %0 %1 %2 [
                    primal={
                        lambda %0:f64[3], %1:f64[], %2:f64[3] .
                        let %3:f64[3] = mul %0 %0
                            %4:f64[3] = mul %0 %1
                            %5:f64[3] = zero [type=f64[3]]
                            %6:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[1]] %0
                            %7:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[0]] %2
                            %8:f64[3, 3] = mul %6 %7
                            %9:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                            %10:f64[3] = gather [
                                dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                                slice_sizes=[1, 1],
                            ] %8 %9
                        in (%3, %4, %5, %10)
                    },
                ]
                    %7:f64[] = zero [type=f64[]]
                in (%3, %4, %7, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_with_sequential_batching() {
        // Sequential batching loops over the batch items with a scan whose body calls the unbatched function, which
        // agrees with structurally batching the same primal, including for an input mapped along an axis other than 0
        // and for a replicated input, which becomes an invariant carry of the scan.
        type Tracer = DomainTracer<ArrayContext>;
        let primal = |(x, y): (Tracer, Tracer)| Ok(x.sin()? * y);
        let sequential = composite(primal).with_sequential_batching().with_name("sine");
        let structural = composite(primal).with_name("sine");
        let inputs =
            (Array::matrix(2, 3, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap(), Array::vector(vec![2.0, -1.0]).unwrap());
        let axes = (BatchAxis::new(1), BatchAxis::replicated());
        let expected: Array =
            batch(|inputs| structural.call(inputs), inputs.clone(), axes, BatchAxis::new(0), None).unwrap();
        let actual: Array = batch(|inputs| sequential.call(inputs), inputs, axes, BatchAxis::new(0), None).unwrap();
        assert_eq!(actual, expected);

        // The batched program replaces the call with the scan, whose body applies the call to one item.
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let (_, program) =
            ArrayContext::trace(|inputs| sequential.call(inputs), (vector_type.clone(), vector_type)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[2] .
                let %2:f64[3, 2] = transpose [permutation=[1, 0]] %0
                    %3:f64[2], %4:f64[3, 2] = scan [carry_count=1, iteration_count=3, reverse=false] %1 %2 [
                        body={
                            lambda %0:i64[], %1:f64[2], %2:f64[2] .
                            let %3:f64[2] = composite [name=\"sine\"] %2 %1 [
                                primal={
                                    lambda %0:f64[2], %1:f64[2] .
                                    let %2:f64[2] = sin %0
                                        %3:f64[2] = mul %2 %1
                                    in (%3)
                                },
                            ]
                            in (%1, %3)
                        },
                    ]
                in (%4)
            "}
            .trim_end(),
        );

        // The derivatives of the batched call are those of the scan, whose body differentiates the call one item at a
        // time, and a derivative taken before batching is batched sequentially as well.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| sequential.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let program = program.into_flat_program();
        let x = [0.5, 1.0, 1.5];
        let inputs =
            vec![Array::vector(x.to_vec()).unwrap(), Array::scalar(2.0).unwrap(), Array::vector(vec![1.0; 3]).unwrap()];
        let expected_tangent = Array::vector(x.iter().map(|x| 2.0 * f64::cos(*x)).collect()).unwrap();
        let batch_then_differentiate = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0
            .jvp_with_respect_to(&[0])
            .unwrap();
        assert_eq!(batch_then_differentiate.interpret(inputs.clone()).unwrap()[1], expected_tangent);
        let differentiate_then_batch = program
            .jvp_with_respect_to(&[0])
            .unwrap()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            differentiate_then_batch.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[], %4:f64[3], %5:f64[3] = scan [carry_count=1, iteration_count=3, reverse=false] %1 %0 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[], %5:f64[] = composite [name=\"jvp(sine)\"] %2 %1 %3 [
                            primal={
                                lambda %0:f64[], %1:f64[], %2:f64[] .
                                let %3:f64[] = sin %0
                                    %4:f64[] = mul %3 %1
                                    %5:f64[] = cos %0
                                    %6:f64[] = mul %5 %2
                                    %7:f64[] = mul %1 %6
                                in (%4, %7)
                            },
                        ]
                        in (%1, %4, %5)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(differentiate_then_batch.interpret(inputs).unwrap()[1], expected_tangent);
    }

    #[test]
    fn test_composite_with_batched_output_axes() {
        // `(x, y) ↦ (x · y, y · y)` with `x` mapped and `y` replicated. By default, every output is mapped at axis 0,
        // so the replicated `y · y` is broadcast, while the declared batch axes keep it replicated.
        type Tracer = DomainTracer<ArrayContext>;
        let primal = |(x, y): (Tracer, Tracer)| Ok((x * y.clone(), y.clone() * y));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let batched = |function: &Composite<(Tracer, Tracer), (Tracer, Tracer)>| {
            let (_, program) =
                ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type.clone()))
                    .unwrap();
            program
                .into_flat_program()
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0), BatchAxis::replicated()],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
        };
        let inputs = vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap(), Array::scalar(4.0).unwrap()];
        let product = Array::vector(vec![4.0, 8.0, 12.0]).unwrap();

        let (program, output_axes) = batched(&composite(primal));
        assert_eq!(output_axes, vec![BatchAxis::new(0); 2]);
        assert_eq!(program.interpret(inputs.clone()), Ok(vec![product.clone(), Array::vector(vec![16.0; 3]).unwrap()]));

        let (program, output_axes) =
            batched(&composite(primal).with_batched_output_axes(vec![BatchAxis::new(0), BatchAxis::replicated()]));
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(program.interpret(inputs), Ok(vec![product, Array::scalar(16.0).unwrap()]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[3], %3:f64[] = composite [
                    name=\"composite\",
                    batching=[(extent=3, input_axes=[axis 0, replicated], output_axes=[axis 0, replicated])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3], %1:f64[] .
                        let %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                            %3:f64[3] = mul %0 %2
                            %4:f64[] = mul %1 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_call() {
        // Nothing but the primal is traced when the function is called. Calls that share a call structure share one
        // registered definition, whose rule is traced once per specialization and cached while the function lives.
        let traces = Arc::new(AtomicUsize::new(0));
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp({
            let traces = traces.clone();
            move |x, tangent| {
                traces.fetch_add(1, Ordering::SeqCst);
                Ok((x.sin()?, x.cos()? * tangent))
            }
        });
        let (_, program) =
            ArrayContext::trace(|x| function.call(function.call(x)?), ArrayType::scalar(DataType::F64)).unwrap();
        let program = program.into_flat_program();

        // Renderings name the rules of calls but not the definitions that they share, which only their rule
        // references record.
        let rule_ids = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Composite(operation) => operation.rules().unwrap().id(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(rule_ids.len(), 2);
        assert_eq!(rule_ids[0], rule_ids[1]);
        assert_eq!(traces.load(Ordering::SeqCst), 0);

        // The derivative of `sin(sin(x))` at `x = 2` is `cos(sin(2)) · cos(2)`.
        let jvp = |program: &FlatProgram<ArrayContext>| {
            program.jvp().unwrap().interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()])
        };
        let expected = Ok(vec![
            Array::scalar(2.0f64.sin().sin()).unwrap(),
            Array::scalar(2.0f64.sin().cos() * 2.0f64.cos()).unwrap(),
        ]);
        assert_eq!(jvp(&program), expected);
        assert_eq!(traces.load(Ordering::SeqCst), 1);
        assert_eq!(jvp(&program), expected);
        assert_eq!(traces.load(Ordering::SeqCst), 1);

        // A new input signature is a new specialization of the same definition.
        let (_, vector_program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        assert_eq!(
            vector_program
                .into_flat_program()
                .jvp()
                .unwrap()
                .interpret(vec![Array::vector(vec![2.0, 3.0]).unwrap(), Array::vector(vec![1.0, 1.0]).unwrap(),]),
            Ok(vec![
                Array::vector(vec![2.0f64.sin(), 3.0f64.sin()]).unwrap(),
                Array::vector(vec![2.0f64.cos(), 3.0f64.cos()]).unwrap(),
            ]),
        );
        assert_eq!(traces.load(Ordering::SeqCst), 2);

        // Once the function is dropped, the calls that it staged trace their rule again on every derivative request.
        drop(function);
        assert_eq!(jvp(&program), expected);
        assert_eq!(traces.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_composite_call_structures() {
        // Each call structure registers its own definition, because the rules are adapted to the structured closures.
        let function =
            composite(|xs: Vec<DomainTracer<ArrayContext>>| Ok(xs)).with_jvp(|xs, tangents| Ok((xs, tangents)));
        let (_, program) = ArrayContext::trace(
            |x: DomainTracer<ArrayContext>| {
                let single = function.call(vec![x.clone()])?;
                let pair = function.call(vec![x.clone(), x.clone()])?;
                let other_single = function.call(vec![x])?;
                Ok((single, pair, other_single))
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();

        // Renderings name the rules of calls but not the definitions that they share, which only their rule
        // references record.
        let rule_ids = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Composite(operation) => operation.rules().unwrap().id(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(rule_ids.len(), 3);
        assert_ne!(rule_ids[0], rule_ids[1]);
        assert_eq!(rule_ids[0], rule_ids[2]);
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let pair = function.call(vec![x.clone(), x])?;
                Ok(pair[0].clone() + pair[1].clone())
            }),
            Ok((Array::scalar(4.0).unwrap(), Array::scalar(2.0).unwrap())),
        );
    }

    #[test]
    fn test_composite_call_named_axes() {
        // The primal and the rules are traced in fresh traces that are seeded with the named axes visible where the
        // function is called, so a function called under `batch` resolves the batch axis `items` in its primal
        // (`x ↦ x · i` at batch item `i`), its JVP rule (`(x, ẋ) ↦ (x · i, ẋ · i)`), its VJP rules (which save `i`
        // and pull back `ȳ ↦ ȳ · i`), and its named collectives (`x ↦ Σᵢ xᵢ`).
        type Tracer = DomainTracer<ArrayContext>;
        let index = |x: &Tracer| -> Result<Tracer, ProgramError> {
            x.context().axis_index("items")?.convert_element_type(DataType::F64)
        };
        let scaled = move |x: Tracer| -> Result<Tracer, ProgramError> { Ok(x.clone() * index(&x)?) };
        let inputs = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let ones = Array::vector(vec![1.0f64; 3]).unwrap();
        let items = || BatchAxisSpecification::named("items");

        // Outside of `batch`, nothing binds `items`.
        let function = composite(scaled);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "items".to_string() })),
        );
        assert_eq!(
            batch(|x| function.call(x), inputs.clone(), BatchAxis::new(0), BatchAxis::new(0), items()),
            Ok(Array::vector(vec![0.0f64, 2.0, 6.0]).unwrap()),
        );
        let (_, program) = ArrayContext::trace(
            |x| Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?),
            ArrayType::new_static(DataType::F64, [3]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = composite [
                    name=\"composite\",
                    batching=[(extent=3, axis_name=\"items\", input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:u64[3] = iota [type=u64[3], dimension=0]
                            %2:f64[3] = convert_element_type [data_type=f64] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        let function = composite(scaled)
            .with_jvp(move |x: Tracer, tangent: Tracer| Ok((scaled(x.clone())?, tangent * index(&x)?)));
        assert_eq!(
            differentiate_at(inputs.clone()).jvp(ones.clone(), |x| {
                Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?)
            }),
            Ok((Array::vector(vec![0.0f64, 2.0, 6.0]).unwrap(), Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap())),
        );

        let function = composite(scaled).with_vjp(
            move |x: Tracer| Ok((scaled(x.clone())?, index(&x)?)),
            |index: Tracer, cotangent: Tracer| Ok(cotangent * index),
        );
        assert_eq!(
            differentiate_at(inputs.clone()).gradient(|x| {
                batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?
                    .reduce(&[0], ReductionKind::Sum)
            }),
            Ok(Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap()),
        );

        let function = composite(|x: Tracer| x.parallel_reduce(ReductionKind::Sum, "items"));
        assert_eq!(
            batch(|x| function.call(x), inputs, BatchAxis::new(0), BatchAxis::new(0), items()),
            Ok(Array::vector(vec![6.0f64; 3]).unwrap()),
        );
    }

    #[test]
    fn test_composite_debug() {
        // Debug output names each configuration kind and the function, but not the closures.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok(x.sin()?)).with_name("sine");
        assert_eq!(
            format!("{function:?}"),
            "Composite { name: \"sine\", jvp: DefaultJvp, vjp: DefaultVjp, batching: DefaultBatching, \
             batched_output_axes: None, non_differentiated_count: 0, .. }",
        );
        let function = function
            .with_jvp(|x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
            .with_vjp(|x| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent))
            .with_batching(|x: Tracer, axis: BatchAxis, _: BatchingLevelExtent<Tracer>| Ok((x.sin()?, axis)));
        assert_eq!(
            format!("{function:?}"),
            "Composite { name: \"sine\", jvp: ExplicitJvp { .. }, vjp: ExplicitVjp { .. }, \
             batching: BroadcastInvariantBatching { .. }, batched_output_axes: None, non_differentiated_count: 0, .. }",
        );
    }

    #[test]
    fn test_composite() {
        // A function without custom rules differentiates its primal in both modes and to higher orders, and it batches
        // its primal structurally.
        let function = composite(|x: DomainTracer<ArrayContext>| Ok(x.sin()?));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| differentiate_at(x).gradient(|x| function.call(x)).unwrap()),
            Ok(Array::scalar(-2.0f64.sin()).unwrap()),
        );
        assert_eq!(
            batch(
                |x| function.call(x),
                Array::vector(vec![1.0, 2.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None
            ),
            Ok(Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap()),
        );
    }

    #[test]
    fn test_composite_checked_casts() {
        // A checked cast in the primal of a function without derivative rules keeps its runtime check in every
        // transform. The derived forward mode rule converts the tangent behind the primal check and retains the source
        // `f64[n]` for the extent that the converted tangent no longer carries, so that reverse mode can transpose the
        // conversion back to `f64[n]` with the retained source as its witness.
        type Tracer = DomainTracer<ArrayContext>;
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [2]);
        let function = composite(move |x: Tracer| x.cast(static_type.clone(), &[])?.sin());
        let (_, program) = ArrayContext::trace(|x| function.call(x), symbolic_type).unwrap();
        let program = program.into_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[n] .
                let %1:f64[2] = composite [name=\"composite\"] %0 [
                    primal={
                        lambda %0:f64[n] .
                        let %1:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                            %2:f64[2] = sin %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let x = Array::vector(vec![1.0, 2.0]).unwrap();
        let wide_x = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f64[3]` to unrelated type `f64[2]`".to_string(),
        )));
        assert_eq!(
            program.interpret(vec![x.clone()]),
            Ok(vec![Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap()]),
        );
        assert_eq!(program.interpret(vec![wide_x.clone()]), mismatch);

        // Forward mode inlines the primal and re-casts the source behind the primal check before converting the
        // tangent, so the tangent conversion never runs ahead of the check.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n] .
                let %2:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %3:f64[n] = cast [type=f64[n], inputs=[f64[n], f64[2]]] %0 %2
                    %4:f64[2] = cast [type=f64[2], inputs=[f64[n], f64[n]]] %1 %3
                    %5:f64[2] = sin %2
                    %6:f64[2] = cos %2
                    %7:f64[2] = mul %6 %4
                in (%5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![x.clone(), Array::vector(vec![1.0, 0.5]).unwrap()]),
            Ok(vec![
                Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap(),
                Array::vector(vec![1.0f64.cos(), 0.5 * 2.0f64.cos()]).unwrap(),
            ]),
        );
        assert_eq!(jvp.interpret(vec![wide_x.clone(), Array::vector(vec![1.0; 3]).unwrap()]), mismatch);

        // Linearization runs the check in the primal program and keeps the source as a residual of the unchecked
        // tangent conversion, whose transpose widens the cotangent back to `f64[n]` with that residual as its witness.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[n] .
                let %1:f64[2] = cast [type=f64[2], inputs=[f64[n]], checked=true] %0
                    %2:f64[2] = sin %1
                    %3:f64[2] = cos %1
                in (%2, %0, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[n], %2:f64[2] .
                let %3:f64[2] = cast [type=f64[2], inputs=[f64[n], f64[n]]] %0 %1
                    %4:f64[2] = mul %2 %3
                in (%4)
            "}
            .trim_end(),
        );
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[n], %2:f64[2] .
                let %3:f64[2] = mul %2 %0
                    %4:f64[n] = cast [type=f64[n], inputs=[f64[2], f64[n]]] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
        let mut pullback_inputs = vec![Array::vector(vec![1.0, 0.5]).unwrap()];
        pullback_inputs.extend(linearization.primal().interpret(vec![x.clone()]).unwrap().into_iter().skip(1));
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![1.0f64.cos(), 0.5 * 2.0f64.cos()]).unwrap()]),
        );
        assert_eq!(linearization.primal().interpret(vec![wide_x]), mismatch);
        check_gradient!(
            |x| {
                let output = program.interpret_in_context(&x.domain(), vec![x])?.remove(0);
                output.reduce(&[0], ReductionKind::Sum)
            },
            at = x,
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_composite_checked_casts_batching() {
        // Batching lifts the declared cast types by the batch axis and keeps the check, and both orders of batching and
        // differentiation stage the same checked primal conversion and unchecked tangent conversion.
        type Tracer = DomainTracer<ArrayContext>;
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [2]);
        let function = composite(move |x: Tracer| x.cast(static_type.clone(), &[])?.sin());
        let (_, program) = ArrayContext::trace(|x| function.call(x), symbolic_type).unwrap();
        let program = program.into_flat_program();
        let batched = |program: &FlatProgram<ArrayContext>, input_count: usize| {
            program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &vec![BatchAxis::new(0); input_count],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
        };
        let elements = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let x = Array::matrix(3, 2, elements.to_vec()).unwrap();
        let tangent = Array::matrix(3, 2, vec![1.0; 6]).unwrap();
        let (wide_x, wide_tangent) =
            (Array::matrix(3, 3, vec![1.0; 9]).unwrap(), Array::matrix(3, 3, vec![1.0; 9]).unwrap());
        let mismatch = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f64[3, 3]` to unrelated type `f64[3, 2]`".to_string(),
        )));
        let sine = Array::matrix(3, 2, elements.map(f64::sin).to_vec()).unwrap();
        let cosine = Array::matrix(3, 2, elements.map(f64::cos).to_vec()).unwrap();

        let batched_program = batched(&program, 1);
        assert_eq!(
            batched_program.to_string(),
            indoc! {"
                lambda %0:f64[3, n] .
                let %1:f64[3, 2] = composite [
                    name=\"composite\",
                    batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[3, n] .
                        let %1:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %0
                            %2:f64[3, 2] = sin %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(batched_program.interpret(vec![x.clone()]), Ok(vec![sine.clone()]));
        assert_eq!(batched_program.interpret(vec![wide_x.clone()]), mismatch);

        // Batching then differentiating.
        let jvp_of_batched = batched_program.jvp().unwrap();
        assert_eq!(
            jvp_of_batched.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, n] .
                let %2:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %0
                    %3:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, n], f64[3, 2]]] %0 %2
                    %4:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n], f64[3, n]]] %1 %3
                    %5:f64[3, 2] = sin %2
                    %6:f64[3, 2] = cos %2
                    %7:f64[3, 2] = mul %6 %4
                in (%5, %7)
            "}
            .trim_end(),
        );
        assert_eq!(jvp_of_batched.interpret(vec![x.clone(), tangent.clone()]), Ok(vec![sine.clone(), cosine.clone()]));
        assert_eq!(jvp_of_batched.interpret(vec![wide_x.clone(), wide_tangent.clone()]), mismatch);

        // Differentiating then batching.
        let batched_jvp = batched(&program.jvp().unwrap(), 2);
        assert_eq!(
            batched_jvp.to_string(),
            indoc! {"
                lambda %0:f64[3, n], %1:f64[3, n] .
                let %2:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n]], checked=true] %0
                    %3:f64[3, 2] = sin %2
                    %4:f64[3, 2] = cos %2
                    %5:f64[3, n] = cast [type=f64[3, n], inputs=[f64[3, n], f64[3, 2]]] %0 %2
                    %6:f64[3, 2] = cast [type=f64[3, 2], inputs=[f64[3, n], f64[3, n]]] %1 %5
                    %7:f64[3, 2] = mul %4 %6
                in (%3, %7)
            "}
            .trim_end(),
        );
        assert_eq!(batched_jvp.interpret(vec![x, tangent]), Ok(vec![sine, cosine]));
        assert_eq!(batched_jvp.interpret(vec![wide_x, wide_tangent]), mismatch);
    }

    #[test]
    fn test_check_batching_rule_consistency() {
        // A rule that computes `2xy` when both inputs are mapped and `xy` otherwise changes its batched output when the
        // replicated `y` is broadcast, while a rule that computes `xy` for every axis pattern does not.
        type Tracer = DomainTracer<ArrayContext>;
        let scaled_product = |inconsistent: bool| {
            composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_batching(
                move |(x, y): (Tracer, Tracer),
                      (x_axis, y_axis): (BatchAxis, BatchAxis),
                      _: BatchingLevelExtent<Tracer>| {
                    let product = x * y;
                    let scale = match inconsistent && !x_axis.is_replicated() && !y_axis.is_replicated() {
                        true => 2.0,
                        false => 1.0,
                    };
                    let scale = product.context().lift(Array::scalar(scale)?)?;
                    Ok((product * scale, if x_axis.is_replicated() { y_axis } else { x_axis }))
                },
            )
        };
        let input = (Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap(), Array::scalar(4.0f64).unwrap());
        let input_axes = (BatchAxis::new(0), BatchAxis::replicated());
        let check = |inconsistent: bool| {
            let function = scaled_product(inconsistent);
            check_batching_rule_consistency(
                |inputs| function.call(inputs),
                input.clone(),
                input_axes,
                None,
                |expected: &Array, actual: &Array| expected == actual,
            )
        };
        assert_eq!(check(false), Ok(()));
        assert_eq!(check(true), Err(BatchingError::BroadcastInvarianceViolation { input: 1, output: 0 }));

        // Every input of a fully replicated batch is broadcast in turn, which requires an explicit extent.
        let function = scaled_product(true);
        assert_eq!(
            check_batching_rule_consistency(
                |inputs| function.call(inputs),
                (Array::scalar(1.0f64).unwrap(), Array::scalar(4.0f64).unwrap()),
                (BatchAxis::replicated(), BatchAxis::replicated()),
                3,
                |expected: &Array, actual: &Array| expected == actual,
            ),
            Ok(()),
        );
    }

    #[test]
    fn test_check_batching_rule_consistency_batching_error() {
        assert_eq!(
            check_batching_rule_consistency(
                |input| Ok(input),
                Array::scalar(1f64).unwrap(),
                BatchAxis::replicated(),
                None,
                |expected: &Array, actual: &Array| expected == actual,
            ),
            Err(BatchingError::EmptyBatch),
        );
    }
}
