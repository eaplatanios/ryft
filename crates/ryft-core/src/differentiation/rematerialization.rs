//! Rematerialization (i.e., gradient checkpointing), which trades computation for memory under differentiation.
//! A rematerialized function computes what its body computes, but differentiation saves only the values of the body
//! that its [`ResidualPolicy`] selects and recomputes the others from the saved values when the derivative computation
//! needs them, instead of keeping every value of the body alive until then. This is the Ryft analogue of JAX's
//! [`jax.checkpoint`/`jax.remat`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
//!
//! [`rematerialize`] creates a rematerialized function from a closure, [`RematerializedFunction::with_policy`]
//! selects which values it saves, and [`RematerializedFunction::call`] stages one call of the function as a
//! [`RematerializeOperation`] whose body is the traced closure. Nothing is derived when the function is called: the
//! transforms derive the derivatives of the body when they need them. Linearization and reverse mode differentiation
//! split the derivative of the body into the work that runs up front, which computes the outputs and the values that
//! the policy saves, and a _differentiated_ call that recomputes everything else from the saved values when the
//! backward computation runs. [`saved_residuals`] reports which values a function saves.
//!
//! The built-in [`ResidualPolicy`]s of [`rematerialize`] are Ryft analogues of the
//! [JAX checkpoint policies](https://docs.jax.dev/en/latest/gradient-checkpointing.html#list-of-policies)
//! together with [`RematerializationPolicyFn`] for policies defined by closures and [`MemoryTransferStorage`]
//! for policies that offload the residuals that they save.
//!
//! Every built-in policy recognizes the producers of a residual by their payload operations (i.e.,
//! [`ResidualProducer::payload`](crate::ResidualProducer::payload)) rather than by their operation family, so the same
//! policy works for every family, including composite and backend families that hold array operations through projected
//! members. A residual that several producers may produce (e.g., the corresponding outputs of the two branches of a
//! `condition` operation) matches a policy when any of its producers does.
//!
//! The built-in policies are generic over the type universe and declare their instantiations for the array universes
//! [`ArrayType`] and [`ArrayIrType`] (refer to [`ResidualPolicy::native_instantiations`]), so that promoting a staged
//! rematerialization from one of these universes to the other re-instantiates its policy rather than projecting the
//! types of its candidates.
//!
//! # Examples
//!
//! ## Gradients
//!
//! A rematerialized function is differentiated like any other function. Its closure annotates its tracer input, which
//! determines the type universe that the function is traced in. By default, it saves nothing but its inputs, so the
//! backward computation of `x ↦ sin(x · x)` recomputes the dot product and its cosine from `x`, while saving dot
//! products saves the dot product and recomputes only the cosine:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, Dot, DotsSavable, DotDimensionNumbers, ProgramError, Sin,
//! #     ResidualSource, SavedResidual, TracingContext, differentiate_at, rematerialize, saved_residuals,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! let sine_of_dot = |x: Tracer| Ok(x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?);
//! let function = rematerialize(sine_of_dot);
//! let gradient = differentiate_at(Array::vector(vec![0.1f64, 0.2, 0.3])?).gradient(|x| function.call(x))?;
//! let cosine = 0.14f64.cos();
//! assert_eq!(gradient, Array::vector(vec![0.2 * cosine, 0.4 * cosine, 0.6 * cosine])?);
//!
//! let vector_type = ArrayType::new_static(DataType::F64, [3]);
//! let input = SavedResidual::new(vector_type.clone(), ResidualSource::Input { index: 0 });
//! assert_eq!(saved_residuals(|x: Tracer| function.call(x), vector_type.clone())?, vec![input.clone()]);
//! let function = rematerialize(sine_of_dot).with_policy(DotsSavable);
//! let dot = SavedResidual::new(ArrayType::scalar(DataType::F64), ResidualSource::Operation { name: "dot" });
//! assert_eq!(saved_residuals(|x: Tracer| function.call(x), vector_type)?, vec![input, dot]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Named Checkpoints
//!
//! [`Tag`](crate::Tag)ged values (the analogue of `jax.ad_checkpoint.checkpoint_name`) can be saved by name (e.g.,
//! with [`SaveOnlyTheseNames`]), which saves the tagged values whose names it lists and recomputes everything else:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, ProgramError, ResidualSource, SaveOnlyTheseNames, SavedResidual,
//! #     Sin, Tag, TracingContext, rematerialize, saved_residuals,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! let function = rematerialize(|x: Tracer| Ok((x.clone() * x).tag("square")?.sin()?.sin()?))
//!     .with_policy(SaveOnlyTheseNames::new(["square"]));
//! let scalar_type = ArrayType::scalar(DataType::F64);
//! assert_eq!(
//!     saved_residuals(|x: Tracer| function.call(x), scalar_type.clone())?,
//!     vec![
//!         SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 0 }),
//!         SavedResidual::new(scalar_type, ResidualSource::Tag { key: "square".to_owned() }),
//!     ],
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Policies
//!
//! [`RematerializationPolicyFn`] defines a policy through a closure that classifies each candidate residual (e.g., by
//! the payloads of the operations that may produce it; refer to
//! [`ResidualProducer::payload`](crate::ResidualProducer::payload)):
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, NoStorage, ProgramError, RematerializationPolicyFn,
//! #     ResidualDecision, ResidualRejection, ResidualSource, SavedResidual, Sin, SinOperation, TracingContext,
//! #     rematerialize, saved_residuals,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! // Saves the sines and recomputes everything else.
//! let policy = RematerializationPolicyFn::new::<ArrayType>(|candidate| {
//!     let producers = candidate.producers();
//!     let saved = producers.iter().any(|producer| producer.payload::<SinOperation<ArrayType>>().is_some());
//!     let decision = if saved { ResidualDecision::<NoStorage>::Save } else { ResidualDecision::Recompute };
//!     Ok::<_, ResidualRejection>(decision)
//! })
//! .with_name("save_sines");
//! let function = rematerialize(|x: Tracer| Ok(x.sin()?.sin()?)).with_policy(policy);
//! let scalar_type = ArrayType::scalar(DataType::F64);
//! assert_eq!(
//!     saved_residuals(|x: Tracer| function.call(x), scalar_type.clone())?,
//!     vec![
//!         SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 0 }),
//!         SavedResidual::new(scalar_type, ResidualSource::Operation { name: "sin" }),
//!     ],
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Offloading
//!
//! Policies can save values by offloading them through a [`ResidualStorage`] instead of keeping them in device memory.
//! For example, [`OffloadDotsWithNoBatchDimensions`] and [`SaveAndOffloadOnlyTheseNames`] transfer the values that
//! they offload to another [`Memory`] (e.g., pinned host memory) once they are computed and back before the backward
//! computation uses them (refer to [`MemoryTransferStorage`]):
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, Memory, ProgramError, ResidualSource,
//! #     SaveAndOffloadOnlyTheseNames, SavedResidual, Sin, Tag, TracingContext, rematerialize, saved_residuals,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! let host = Memory::Host { pinned: true };
//! let policy = SaveAndOffloadOnlyTheseNames::new(Vec::<String>::new(), ["square"], host)?;
//! let function = rematerialize(|x: Tracer| Ok((x.clone() * x).tag("square")?.sin()?.sin()?)).with_policy(policy);
//! let scalar_type = ArrayType::scalar(DataType::F64);
//! assert_eq!(
//!     saved_residuals(|x: Tracer| function.call(x), scalar_type.clone())?,
//!     vec![
//!         SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 0 }),
//!         SavedResidual::new(scalar_type.with_memory(host), ResidualSource::Tag { key: "square".to_owned() }),
//!     ],
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Optimization Barriers
//!
//! The differentiated call that recomputes the saved values places an optimization barrier on its inputs (refer to
//! [`RematerializedFunction::with_optimization_barrier`]), which keeps compilers from merging the recomputation with
//! the forward computation and thus from undoing the memory savings. The barrier can also keep compilers from
//! optimizing across it in other ways (e.g., from fusing operations), so it should be disabled when something else
//! already separates the recomputation from the forward computation. This is the case for a rematerialized function
//! that is called inside the body of a loop (e.g., a `scan` over the layers of a model), because the backward loop
//! recomputes each iteration after the forward loop has finished, which is also why
//! [JAX recommends](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) `prevent_cse=False` there:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ProgramError, RematerializationOptimizationBarrier, Sin, TracingContext, rematerialize,
//! # };
//! # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! // A layer that a `scan` over the layers of a model calls once per iteration.
//! let layer = rematerialize(|x: Tracer| Ok::<_, ProgramError>(x.sin()?.sin()?))
//!     .with_optimization_barrier(RematerializationOptimizationBarrier::None);
//! # let _ = layer;
//! ```
//!
//! The barrier can also be limited to some of the inputs with [`RematerializationOptimizationBarrier::Inputs`], whose
//! entries select the leaves of the input in [`Parameterized::parameters`] order. The values that differentiation saves
//! are always behind the barrier, and the unselected inputs stay free for compilers to optimize together with the
//! recomputation (e.g., the parameters of a layer, as opposed to the activations that it receives):
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, Mul, ProgramError, RematerializationOptimizationBarrier, Sin, TracingContext,
//! #     rematerialize,
//! # };
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! // Only the activation `x`, which is the first leaf of the input, goes through the barrier.
//! let layer = rematerialize(|(x, w): (Tracer, Tracer)| Ok::<_, ProgramError>(x.mul(&w)?.sin()?))
//!     .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true, false]));
//! # let _ = layer;
//! ```

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, Mutex};

use crate::arrays::{ArrayIrType, ArrayType, Memory};
use crate::axes::NamedAxes;
use crate::contexts::Context;
use crate::differentiation::DifferentiationRule;
use crate::differentiation::forward::DifferentiableOperation;
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::operations::{
    DotOperation, ReducePrecisionOperation, RematerializationOptimizationBarrier, RematerializeOperation, TagOperation,
    TransferToMemoryOperation,
};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{
    ErasedResidualStorage, NativeResidualPolicies, NoStorage, PartialEvaluationContext, PartiallyEvaluatableOperation,
    ResidualCandidate, ResidualDecision, ResidualPolicy, ResidualPolicyError, ResidualPolicyReference,
    ResidualRejection, ResidualStorage,
};
use crate::programs::{
    ErasedOperation, Operation, OperationPayloadProjection, ProgramError, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracingContext};

/// Rematerialized function, which [`rematerialize`] creates from a closure over [`DomainTracer`]s.
pub struct RematerializedFunction<Input, Output, Body, Policy = NothingSavable> {
    /// Closure that computes the body of the function.
    body: Body,

    /// Residual policy of the function together with its references in the type universes that were used so far.
    policy: Arc<PolicyReferences<Policy>>,

    /// Specifies the inputs of the staged calls on which backends place an optimization barrier when the calls
    /// are differentiated.
    optimization_barrier: RematerializationOptimizationBarrier,

    /// Input and output types of the closure.
    marker: PhantomData<fn() -> (Input, Output)>,
}

impl<Input, Output, Body, Policy> RematerializedFunction<Input, Output, Body, Policy> {
    /// Returns this [`RematerializedFunction`] with the provided residual policy, which decides which values of the
    /// body differentiation saves.
    #[inline]
    pub fn with_policy<NewPolicy>(self, policy: NewPolicy) -> RematerializedFunction<Input, Output, Body, NewPolicy> {
        RematerializedFunction {
            body: self.body,
            policy: Arc::new(PolicyReferences::new(policy)),
            optimization_barrier: self.optimization_barrier,
            marker: PhantomData,
        }
    }

    /// Sets the inputs on which backends place an optimization barrier when the staged calls of this
    /// [`RematerializedFunction`] are differentiated, which are [all](RematerializationOptimizationBarrier::All)
    /// by default. A [`RematerializationOptimizationBarrier::Inputs`] selection has one entry per leaf of the input
    /// of this function, in [`Parameterized::parameters`] order. This is the analogue of the `prevent_cse` parameter of
    /// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html), which can be disabled when
    /// the function is called in a loop body (e.g., of a `scan` operation), where the loop already keeps the
    /// recomputation from being merged with the original computation.
    #[inline]
    pub fn with_optimization_barrier(mut self, optimization_barrier: RematerializationOptimizationBarrier) -> Self {
        self.optimization_barrier = optimization_barrier;
        self
    }

    /// Stages one call of this [`RematerializedFunction`] on the provided `input` value and returns its output value.
    /// The [`Context`] `C` that the call is staged into is the [`Domain`](Value::Domain) of the values
    /// in `input`, so it is never named at a construction or call site. The body is traced in a fresh trace that is
    /// seeded with the [named axes](NamedAxes::named_axes) in scope in `C`, so that it resolves the axes of enclosing
    /// transforms as it would if it were inlined.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves (in which case [`call_in_context`](Self::call_in_context)
    /// must be used instead), when tracing the body fails, or when the staged [`RematerializeOperation`] rejects the
    /// call.
    pub fn call<
        V: Value<Type = C::Type, Domain = C>,
        C: Context<Type: 'static, Value = V, Operation: From<RematerializeOperation<C::Type>>> + NamedAxes,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        Body: Fn(Input) -> Result<Output, ProgramError>,
        Policy: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
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
            return Err(TypeError::invalid(
                "`rematerialize` requires at least one input to recover its context from; \
                 use `RematerializedFunction::call_in_context` for functions without inputs",
            )
            .into());
        };
        self.call_impl(&first.domain(), input_types, input_values.as_slice())
    }

    /// Stages one call of this function on the provided `input` value in the provided `context` and returns its output
    /// value. This is the explicit-context counterpart of [`call`](Self::call), which supports functions without
    /// inputs and contexts that are not the domain of the input values.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when tracing the body fails or when the staged [`RematerializeOperation`] rejects
    /// the call.
    pub fn call_in_context<
        C: Context<Type: 'static, Operation: From<RematerializeOperation<C::Type>>> + NamedAxes,
        InputValues: Parameterized<C::Value, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        context: &C,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        Body: Fn(Input) -> Result<Output, ProgramError>,
        Policy: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<C::Value>,
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
        self.call_impl(context, input_types, input_values.as_slice())
    }

    /// Traces the body at `input_types` and binds one call of it to `input_values` in `context`.
    fn call_impl<C: Context<Type: 'static, Operation: From<RematerializeOperation<C::Type>>> + NamedAxes>(
        &self,
        context: &C,
        input_types: Input::To<C::Type>,
        input_values: &[C::Value],
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        Body: Fn(Input) -> Result<Output, ProgramError>,
        Policy: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<C::Value>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        let (output_types, body) =
            DomainTracingContext::<C>::trace_with_named_axes(&self.body, input_types, context.named_axes())?;
        let output_structure = output_types.parameter_structure();
        let operation = RematerializeOperation::new(self.policy.reference::<C::Type>())
            .with_optimization_barrier(self.optimization_barrier.clone());
        let outputs = context.bind(operation, vec![body.into_flat_program()], input_values)?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

impl<Input, Output, Body: Clone, Policy> Clone for RematerializedFunction<Input, Output, Body, Policy> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            policy: self.policy.clone(),
            optimization_barrier: self.optimization_barrier.clone(),
            marker: PhantomData,
        }
    }
}

impl<Input, Output, Body, Policy: Debug> Debug for RematerializedFunction<Input, Output, Body, Policy> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RematerializedFunction")
            .field("policy", &self.policy.policy)
            .field("optimization_barrier", &self.optimization_barrier)
            .finish_non_exhaustive()
    }
}

/// Creates a [`RematerializedFunction`] from a closure `x ↦ y = f(x)` over [`DomainTracer`]s, which saves nothing under
/// differentiation (i.e., it uses the [`NothingSavable`] policy) and places an optimization barrier on its inputs when
/// it is differentiated. The closure must annotate the type of its tracer input, which determines the context that its
/// body is traced in, and nothing is traced until the function is called.
#[inline]
pub fn rematerialize<Input, Output, Body: Fn(Input) -> Result<Output, ProgramError>>(
    body: Body,
) -> RematerializedFunction<Input, Output, Body> {
    // `PolicyReferences` of the default `NothingSavable` policy, which every `RematerializedFunction` that does not
    // select a policy shares, so that all of their calls stage operations whose policies compare equal.
    static DEFAULT_POLICY_REFERENCES: LazyLock<Arc<PolicyReferences<NothingSavable>>> =
        LazyLock::new(|| Arc::new(PolicyReferences::new(NothingSavable)));
    RematerializedFunction {
        body,
        policy: DEFAULT_POLICY_REFERENCES.clone(),
        optimization_barrier: RematerializationOptimizationBarrier::All,
        marker: PhantomData,
    }
}

/// Source of a value that differentiation saves for the backward computation of a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResidualSource {
    /// Input of the function at the provided index among its flattened inputs.
    Input {
        /// Index of the input among the flattened inputs of the function.
        index: usize,
    },

    /// Constant defined in the function's body.
    Constant,

    /// Value tagged with the provided key. Refer to [`Tag`](crate::Tag) for more information on tagging.
    Tag {
        /// Key of the tag.
        key: String,
    },

    /// Output of an operation with the provided name.
    Operation {
        /// Name of the operation.
        name: &'static str,
    },
}

/// Value that differentiation saves for the backward computation of a function, together with its [`ResidualSource`],
/// which [`saved_residuals`] reports.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SavedResidual<T: Type> {
    /// Type of the saved value.
    r#type: T,

    /// Source of the saved value.
    source: ResidualSource,
}

impl<T: Type> SavedResidual<T> {
    /// Creates a new [`SavedResidual`] of the provided type with the provided source.
    #[inline]
    pub fn new(r#type: T, source: ResidualSource) -> Self {
        Self { r#type, source }
    }

    /// Returns the [`ResidualSource`] of the saved value.
    #[inline]
    pub fn source(&self) -> &ResidualSource {
        &self.source
    }
}

impl<T: Type> Display for SavedResidual<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            ResidualSource::Input { index } => write!(formatter, "{} from the input {index}", self.r#type),
            ResidualSource::Constant => write!(formatter, "{} from a constant", self.r#type),
            ResidualSource::Tag { key } => write!(formatter, "{} tagged `{key}`", self.r#type),
            ResidualSource::Operation { name } => write!(formatter, "{} output of `{name}`", self.r#type),
        }
    }
}

impl<T: Type> Typed for SavedResidual<T> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Returns the values that reverse mode differentiation of `function` at `input_types` saves for its backward
/// computation, in the order in which the backward computation receives them, which is the analogue of JAX's
/// [`jax.ad_checkpoint.saved_residuals`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#inspecting-residuals-with-jax-ad-checkpoint-print-saved-residuals).
/// Use it to check what a [`rematerialize`]d function and its residual policy save. The function is traced into a
/// [`Program`](crate::Program) like [`TracingContext::trace`] and the program is linearized, and each residual of the
/// linearization is reported with its [`ResidualSource`]. The program is linearized for reverse mode differentiation
/// (i.e., with the [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rules of its operations). The
/// source of a saved value looks through the operations that residual placement stages on it (i.e., the rounding of
/// narrow floating-point values and the store operations of [`MemoryTransferStorage`]), so that an offloaded value is
/// reported as the value that was offloaded. Such operations that the function applies itself are looked through as
/// well.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, Dot, DotsSavable, DotDimensionNumbers, ProgramError,
/// #     ResidualSource, SavedResidual, Sin, rematerialize, saved_residuals,
/// # };
/// # use ryft_core::TracingContext;
/// # fn main() -> Result<(), ProgramError> {
/// # type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
///
/// // `x ↦ sin(x · x)` saves its input and, because the policy saves dot products, the dot product as well,
/// // from which the backward computation recomputes the cosine.
/// let function = rematerialize(|x: Tracer| {
///     Ok(x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?)
/// }).with_policy(DotsSavable);
/// let residuals = saved_residuals(|x: Tracer| function.call(x), ArrayType::new_static(DataType::F64, [3]))?;
/// assert_eq!(
///     residuals,
///     vec![
///         SavedResidual::new(ArrayType::new_static(DataType::F64, [3]), ResidualSource::Input { index: 0 }),
///         SavedResidual::new(ArrayType::scalar(DataType::F64), ResidualSource::Operation { name: "dot" }),
///     ],
/// );
/// assert_eq!(residuals[1].to_string(), "f64[] output of `dot`");
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a [`ProgramError`] when tracing or linearizing the function fails.
pub fn saved_residuals<
    V: Value<Type: 'static + DifferentiableType>,
    O: Operation<Type = V::Type>
        + OperationPayloadProjection
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>,
    Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<TracingContext<V, O>>>>,
    Output: Parameterized<Tracer<TracingContext<V, O>>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    F: FnOnce(Input::To<Tracer<TracingContext<V, O>>>) -> Result<Output, ProgramError>,
>(
    function: F,
    input_types: Input,
) -> Result<Vec<SavedResidual<V::Type>>, ProgramError> {
    // The program is linearized with the `jvp_for_transpose` rules that reverse mode differentiation uses,
    // which can save different values than the `jvp` rules (e.g., for custom functions with distinct rules).
    let (_, program) = TracingContext::<V, O>::trace(function, input_types)?;
    let program = program.into_flat_program();
    let input_indices = (0..program.input_ids().len()).collect::<Vec<_>>();
    let linearization = program
        .entry_region_ref()
        .linearize_shared_for_rule(&input_indices, DifferentiationRule::JvpForTranspose)?;
    let primal = linearization.primal();
    let instruction_by_output = primal.instruction_by_output();
    let output_ids = primal.output_ids();
    output_ids[output_ids.len() - linearization.residual_count()..]
        .iter()
        .map(|residual| {
            let residual_type = primal.atoms()[residual.index()].r#type().into_owned();
            let mut atom = *residual;
            let source = loop {
                if let Some(index) = primal.input_ids().iter().position(|input| *input == atom) {
                    break ResidualSource::Input { index };
                }
                let Some(index) = instruction_by_output[atom.index()] else {
                    break ResidualSource::Constant;
                };
                let instruction = &primal.instructions()[index];
                let operation = instruction.operation();
                if let Some(key) = TagOperation::<V::Type>::key_of(operation) {
                    break ResidualSource::Tag { key: key.to_owned() };
                }
                if operation.projected_payload::<ReducePrecisionOperation<ArrayType>>().is_none()
                    && operation.projected_payload::<TransferToMemoryOperation>().is_none()
                {
                    break ResidualSource::Operation { name: operation.name() };
                }
                atom = instruction.inputs()[0];
            };
            Ok(SavedResidual::new(residual_type, source))
        })
        .collect()
}

/// Residual policy of a [`RematerializedFunction`] together with its [`ResidualPolicyReference`]s in the type universes
/// that its calls have used so far. [`ResidualPolicyReference`]s compare by the identity of the policy definition that
/// [`ResidualPolicyReference::new`] registers, so creating one per call would make calls of the same function stage
/// unequal operations. Clones of a [`RematerializedFunction`] share one [`PolicyReferences`], and therefore stage equal
/// operations too.
struct PolicyReferences<Policy> {
    /// Residual policy.
    policy: Policy,

    /// [`ResidualPolicyReference`] to `policy` in each type universe that was used so far, keyed by the [`TypeId`]
    /// of the universe.
    references: Mutex<Vec<(TypeId, Box<dyn Any + Send + Sync>)>>,
}

impl<Policy> PolicyReferences<Policy> {
    /// Creates new [`PolicyReferences`] for `policy` that hold no references yet.
    #[inline]
    fn new(policy: Policy) -> Self {
        Self { policy, references: Mutex::new(Vec::new()) }
    }

    /// Returns the [`ResidualPolicyReference`] to the policy in the type universe `T`, which is registered
    /// on first use.
    fn reference<T: 'static + Type>(&self) -> ResidualPolicyReference<T>
    where
        Policy: Clone + ResidualPolicy<T>,
    {
        let mut references = self.references.lock().expect("the residual policy references lock is poisoned");
        let reference = references
            .iter()
            .find(|(universe, _)| *universe == TypeId::of::<T>())
            .and_then(|(_, reference)| reference.downcast_ref::<ResidualPolicyReference<T>>());
        if let Some(reference) = reference {
            return reference.clone();
        }
        let reference = ResidualPolicyReference::new(self.policy.clone());
        references.push((TypeId::of::<T>(), Box::new(reference.clone())));
        reference
    }
}

/// Canonical policy name for [`NothingSavable`].
pub const NOTHING_SAVABLE_POLICY_NAME: &str = "nothing_savable";

/// [`ResidualPolicy`] that saves nothing, so that differentiation recomputes every residual from the inputs of the
/// rematerialized function. This is the default policy of [`rematerialize`]. This is the Ryft
/// analogue of JAX's
/// [`nothing_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.nothing_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NothingSavable;

impl<T: 'static + Type> ResidualPolicy<T> for NothingSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        NOTHING_SAVABLE_POLICY_NAME
    }

    #[inline]
    fn classify(
        &self,
        _candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        Ok(ResidualDecision::Recompute)
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default().with::<ArrayType, _>(*self).with::<ArrayIrType, _>(*self)
    }
}

/// Canonical policy name for [`EverythingSavable`].
pub const EVERYTHING_SAVABLE_POLICY_NAME: &str = "everything_savable";

/// [`ResidualPolicy`] that saves every residual, so that differentiation recomputes nothing. This is the Ryft analogue
/// of JAX's
/// [`everything_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.everything_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EverythingSavable;

impl<T: 'static + Type> ResidualPolicy<T> for EverythingSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        EVERYTHING_SAVABLE_POLICY_NAME
    }

    #[inline]
    fn classify(
        &self,
        _candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        Ok(ResidualDecision::Save)
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default().with::<ArrayType, _>(*self).with::<ArrayIrType, _>(*self)
    }
}

/// Canonical policy name for [`DotsSavable`].
pub const DOTS_SAVABLE_POLICY_NAME: &str = "dots_savable";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s produce and recomputes every other residual.
/// This is the Ryft analogue of JAX's
/// [`dots_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.dots_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsSavable;

impl<T: 'static + Type> ResidualPolicy<T> for DotsSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        DOTS_SAVABLE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some());
        Ok(match saved {
            true => ResidualDecision::Save,
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default().with::<ArrayType, _>(*self).with::<ArrayIrType, _>(*self)
    }
}

/// Canonical policy name for [`DotsWithNoBatchDimensionsSavable`].
pub const DOTS_WITH_NO_BATCH_DIMENSIONS_SAVABLE_POLICY_NAME: &str = "dots_with_no_batch_dimensions_savable";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s without batching dimensions (e.g., matrix
/// multiplications) produce and recomputes every other residual. This is the Ryft analogue of JAX's
/// [`dots_with_no_batch_dims_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.dots_with_no_batch_dims_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsWithNoBatchDimensionsSavable;

impl<T: 'static + Type> ResidualPolicy<T> for DotsWithNoBatchDimensionsSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        DOTS_WITH_NO_BATCH_DIMENSIONS_SAVABLE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().filter_map(|producer| producer.payload::<DotOperation>()).any(|dot| {
            let dimensions = dot.dimensions();
            dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
        });
        Ok(match saved {
            true => ResidualDecision::Save,
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default().with::<ArrayType, _>(*self).with::<ArrayIrType, _>(*self)
    }
}

/// Canonical policy name for [`OffloadDotsWithNoBatchDimensions`].
pub const OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME: &str = "offload_dots_with_no_batch_dimensions";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s without batching dimensions produce by offloading
/// them to the provided [`Memory`] and recomputes every other residual. This is the Ryft analogue of JAX's
/// [`offload_dot_with_no_batch_dims`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.offload_dot_with_no_batch_dims.html).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffloadDotsWithNoBatchDimensions {
    /// [`Memory`] that the saved residuals are offloaded to.
    destination: Memory,
}

impl OffloadDotsWithNoBatchDimensions {
    /// Creates a new [`OffloadDotsWithNoBatchDimensions`] policy that offloads the residuals that it saves to
    /// `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the [`Memory`] that the saved residuals are offloaded to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for OffloadDotsWithNoBatchDimensions
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    type Storage = MemoryTransferStorage;

    #[inline]
    fn name(&self) -> &str {
        OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<MemoryTransferStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().filter_map(|producer| producer.payload::<DotOperation>()).any(|dot| {
            let dimensions = dot.dimensions();
            dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
        });
        Ok(match saved {
            true => ResidualDecision::SaveWith(MemoryTransferStorage::new(self.destination)),
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default().with::<ArrayType, _>(*self).with::<ArrayIrType, _>(*self)
    }
}

/// Canonical policy name for [`SaveOnlyTheseNames`].
pub const SAVE_ONLY_THESE_NAMES_POLICY_NAME: &str = "save_only_these_names";

/// [`ResidualPolicy`] that saves the residuals that are tagged (using [`Tag`](crate::Tag)) with one of the provided
/// names and recomputes every other residual. This is the Ryft analogue of JAX's
/// [`save_only_these_names`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.save_only_these_names.html).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveOnlyTheseNames {
    /// Names of the tags whose values are saved.
    names: Vec<String>,
}

impl SaveOnlyTheseNames {
    /// Creates a new [`SaveOnlyTheseNames`] policy that saves the values that are tagged with one of `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveOnlyTheseNames {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ONLY_THESE_NAMES_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .any(|key| self.names.iter().any(|name| name == key));
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAnyNamesButThese`].
pub const SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME: &str = "save_any_names_but_these";

/// [`ResidualPolicy`] that saves the residuals that are tagged (using [`Tag`](crate::Tag)) with any name other than the
/// provided ones and recomputes every other residual, including untagged ones. This is the Ryft analogue of JAX's
/// [`save_any_names_but_these`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.save_any_names_but_these.html).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAnyNamesButThese {
    /// Names of the tags whose values are not saved.
    names: Vec<String>,
}

impl SaveAnyNamesButThese {
    /// Creates a new [`SaveAnyNamesButThese`] policy that saves the tagged values whose tag names are not in `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are not saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAnyNamesButThese {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .any(|key| !self.names.iter().any(|name| name == key));
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAnythingExceptTheseNames`].
pub const SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME: &str = "save_anything_except_these_names";

/// [`ResidualPolicy`] that saves every residual except the ones that are tagged (using [`Tag`](crate::Tag)) with
/// one of the provided names. Unlike [`SaveAnyNamesButThese`], it also saves untagged residuals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAnythingExceptTheseNames {
    /// Names of the tags whose values are not saved.
    names: Vec<String>,
}

impl SaveAnythingExceptTheseNames {
    /// Creates a new [`SaveAnythingExceptTheseNames`] policy that saves every value except the ones that are tagged
    /// with one of `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are not saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAnythingExceptTheseNames {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().any(|producer| {
            TagOperation::<T>::key_of(producer.operation()).is_none_or(|key| !self.names.iter().any(|name| name == key))
        });
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAndOffloadOnlyTheseNames`].
pub const SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME: &str = "save_and_offload_only_these_names";

/// [`ResidualPolicy`] that saves the residuals that are tagged (using [`Tag`](crate::Tag)) with one of the provided
/// savable names, offloads the ones that are tagged with one of the provided offloadable names to the provided
/// [`Memory`], and recomputes every other residual. This is the Ryft analogue of JAX's
/// [`save_and_offload_only_these_names`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.save_and_offload_only_these_names.html).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAndOffloadOnlyTheseNames {
    /// Names of the tags whose values are saved.
    savable_names: Vec<String>,

    /// Names of the tags whose values are offloaded.
    offloadable_names: Vec<String>,

    /// [`Memory`] that the offloaded values are transferred to.
    destination: Memory,
}

impl SaveAndOffloadOnlyTheseNames {
    /// Creates a new [`SaveAndOffloadOnlyTheseNames`] policy that saves the values that are tagged with one of
    /// `savable_names` and offloads the ones that are tagged with one of `offloadable_names` to `destination`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] when a name is both savable and offloadable, because a value cannot
    /// be both kept in place and offloaded.
    pub fn new<S: Into<String>, O: Into<String>>(
        savable_names: impl IntoIterator<Item = S>,
        offloadable_names: impl IntoIterator<Item = O>,
        destination: Memory,
    ) -> Result<Self, ProgramError> {
        let savable_names = savable_names.into_iter().map(Into::into).collect::<Vec<String>>();
        let offloadable_names = offloadable_names.into_iter().map(Into::into).collect::<Vec<String>>();
        let overlapping_names = savable_names
            .iter()
            .filter(|name| offloadable_names.contains(name))
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>();
        if !overlapping_names.is_empty() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "names {} cannot be both savable and offloadable by a `{}` policy",
                    overlapping_names.join(", "),
                    SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME,
                ),
            });
        }
        Ok(Self { savable_names, offloadable_names, destination })
    }

    /// Returns the names of the tags whose values are saved.
    #[inline]
    pub fn savable_names(&self) -> &[String] {
        self.savable_names.as_slice()
    }

    /// Returns the names of the tags whose values are offloaded.
    #[inline]
    pub fn offloadable_names(&self) -> &[String] {
        self.offloadable_names.as_slice()
    }

    /// Returns the [`Memory`] that the offloaded values are transferred to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAndOffloadOnlyTheseNames
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    type Storage = MemoryTransferStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<MemoryTransferStorage>, ResidualRejection> {
        let keys = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .collect::<Vec<_>>();
        let tagged_with = |names: &[String]| keys.iter().any(|key| names.iter().any(|name| name == key));
        Ok(if tagged_with(&self.savable_names) {
            ResidualDecision::Save
        } else if tagged_with(&self.offloadable_names) {
            ResidualDecision::SaveWith(MemoryTransferStorage::new(self.destination))
        } else {
            ResidualDecision::Recompute
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveFromBothPolicies`].
pub const SAVE_FROM_BOTH_POLICIES_POLICY_NAME: &str = "save_from_both_policies";

/// [`ResidualPolicy`] that combines two policies, saving each residual that either of them saves. The first policy
/// classifies each residual first, and the second policy is only consulted for the residuals that the first one
/// recomputes. A residual that the first policy saves through a [`ResidualStorage`] therefore keeps that storage,
/// which makes this a strict superset of JAX's policy, whose combination of two policies rejects offloading.
/// Rejections of either policy are returned as they are. This is the Ryft analogue of JAX's
/// [`save_from_both_policies`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.save_from_both_policies.html).
///
/// Unlike the other built-in policies, this policy declares no instantiations in other type universes, because its two
/// policies may be defined for one universe only (e.g., [`RematerializationPolicyFn`]s). Promoting a staged
/// rematerialization that uses it to another universe therefore projects the types of its candidates (refer to
/// [`ResidualPolicyReference::lift`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveFromBothPolicies<P1, P2> {
    /// Policy that classifies each residual first.
    first: P1,

    /// Policy that classifies the residuals that `first` recomputes.
    second: P2,
}

impl<P1, P2> SaveFromBothPolicies<P1, P2> {
    /// Creates a new [`SaveFromBothPolicies`] policy that saves each residual that `first` or `second` saves.
    #[inline]
    pub fn new(first: P1, second: P2) -> Self {
        Self { first, second }
    }

    /// Returns the policy that classifies each residual first.
    #[inline]
    pub fn first(&self) -> &P1 {
        &self.first
    }

    /// Returns the policy that classifies the residuals that [`first`](Self::first) recomputes.
    #[inline]
    pub fn second(&self) -> &P2 {
        &self.second
    }
}

impl<T: 'static + Type, P1: ResidualPolicy<T>, P2: ResidualPolicy<T>> ResidualPolicy<T>
    for SaveFromBothPolicies<P1, P2>
{
    type Storage = ErasedResidualStorage<T>;

    #[inline]
    fn name(&self) -> &str {
        SAVE_FROM_BOTH_POLICIES_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualRejection> {
        match self.first.classify(candidate)?.into_erased() {
            ResidualDecision::Recompute => Ok(self.second.classify(candidate)?.into_erased()),
            decision => Ok(decision),
        }
    }
}

/// Default name for [`RematerializationPolicyFn`].
pub const REMATERIALIZATION_POLICY_FN_POLICY_NAME: &str = "rematerialization_policy_fn";

/// [`ResidualPolicy`] whose classifier is the provided closure, for policies that the built-in ones cannot express.
/// The closure receives each candidate residual and returns its decision, possibly with a [`ResidualStorage`] of type
/// `S`, or a rejection that forbids every placement of the residual. Policies that need different storages for
/// different residuals can return [`ErasedResidualStorage`]s.
///
/// Use this adapter for closure-defined policies, including closures that capture runtime configuration. Implement
/// [`ResidualPolicy`] directly for a named, reusable policy type or for a policy that needs native instantiations in
/// multiple type universes.
///
/// A [`RematerializationPolicyFn`] is defined for the type universe of the candidates that its closure accepts and
/// declares no instantiations in other universes. Promoting a staged rematerialization that uses it to another universe
/// therefore projects the types of its candidates (refer to [`ResidualPolicyReference::lift`]).
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{ArrayType, NoStorage, RematerializationPolicyFn, ResidualDecision, ResidualRejection};
/// // Saves the residuals that have one producer and recomputes the ones that several producers may produce.
/// let policy = RematerializationPolicyFn::new::<ArrayType>(|candidate| {
///     Ok::<_, ResidualRejection>(match candidate.producers().len() {
///         1 => ResidualDecision::<NoStorage>::Save,
///         _ => ResidualDecision::Recompute,
///     })
/// })
/// .with_name("save_unique_producers");
/// # let _ = policy;
/// ```
pub struct RematerializationPolicyFn<F, S = NoStorage> {
    /// Name of this policy.
    name: Cow<'static, str>,

    /// Closure that classifies each candidate residual.
    function: F,

    /// Storage of the decisions that the closure returns.
    marker: PhantomData<fn() -> S>,
}

impl<F, S> RematerializationPolicyFn<F, S> {
    /// Creates a new [`RematerializationPolicyFn`] named `rematerialization_policy_fn` whose classifier is `function`,
    /// which classifies candidates of the type universe `T`.
    #[inline]
    pub fn new<T: 'static + Type>(function: F) -> Self
    where
        F: Fn(&ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection>,
    {
        Self { name: Cow::Borrowed(REMATERIALIZATION_POLICY_FN_POLICY_NAME), function, marker: PhantomData }
    }

    /// Returns this [`RematerializationPolicyFn`] with the provided name, which is used in diagnostics and in the
    /// rendering of the operations that carry the policy.
    #[inline]
    pub fn with_name<N: Into<Cow<'static, str>>>(mut self, name: N) -> Self {
        self.name = name.into();
        self
    }
}

impl<F: Clone, S> Clone for RematerializationPolicyFn<F, S> {
    #[inline]
    fn clone(&self) -> Self {
        Self { name: self.name.clone(), function: self.function.clone(), marker: PhantomData }
    }
}

impl<F, S> Debug for RematerializationPolicyFn<F, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RematerializationPolicyFn")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<
    T: 'static + Type,
    S: ResidualStorage<T>,
    F: 'static + Send + Sync + Fn(&ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection>,
> ResidualPolicy<T> for RematerializationPolicyFn<F, S>
{
    type Storage = S;

    #[inline]
    fn name(&self) -> &str {
        self.name.as_ref()
    }

    #[inline]
    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection> {
        (self.function)(candidate)
    }
}

/// [`ResidualStorage`] that offloads array residuals to the provided [`Memory`] with a [`TransferToMemoryOperation`],
/// and restores them by transferring them back to the memory of the residual. It supports every type universe whose
/// types project into [`ArrayType`] (e.g., [`ArrayType`] itself and [`ArrayIrType`]) and rejects residuals that are not
/// arrays (e.g., dimensions or references).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryTransferStorage {
    /// [`Memory`] that the residuals are offloaded to.
    destination: Memory,
}

impl MemoryTransferStorage {
    /// Creates a new [`MemoryTransferStorage`] that offloads residuals to `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the [`Memory`] that the residuals are offloaded to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualStorage<T> for MemoryTransferStorage
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    #[inline]
    fn name(&self) -> String {
        "memory_transfer".to_owned()
    }

    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        <&ArrayType>::try_from(residual_type).map_err(|_| ResidualPolicyError::UnsupportedStorage {
            storage: ResidualStorage::<T>::name(self),
            residual_type: residual_type.to_string(),
            message: "the residual is not an array".to_owned(),
        })?;
        Ok(vec![ErasedOperation::new(TransferToMemoryOperation::new(self.destination))])
    }

    fn restore_payloads(
        &self,
        _stored_type: &T,
        residual_type: &T,
    ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        let residual_array_type =
            <&ArrayType>::try_from(residual_type).map_err(|_| ResidualPolicyError::UnsupportedStorage {
                storage: ResidualStorage::<T>::name(self),
                residual_type: residual_type.to_string(),
                message: "the residual is not an array".to_owned(),
            })?;
        Ok(vec![ErasedOperation::new(TransferToMemoryOperation::new(residual_array_type.memory()))])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        DimensionBounds, DimensionType, Memory, ShardingDimension,
    };
    use crate::axes::AxisError;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchedProgram, ProgramBatchingOutputAxesPolicy, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::operations::{
        AxisIndex, Constant, ConvertElementType, Cos, DimensionSizeOperation, Dot, DotDimensionNumbers, Exp,
        MulOperation, ReducePrecision, ReferenceRead, Sin, SinOperation, Tag, custom_function,
    };
    use crate::partial::ResidualProducer;
    use crate::programs::{Program, ReferenceType};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestTracer = Tracer<TracingContext<Array, ArrayOperation<Array>>>;
    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrTracer = Tracer<TracingContext<TestIrValue, TestIrOperation>>;

    /// Traces `function` into a program over one `f64[]` scalar input.
    fn trace<F: Fn(TestTracer) -> Result<TestTracer, ProgramError>>(
        function: F,
    ) -> Program<Array, ArrayOperation<Array>, Array, Array> {
        TracingContext::<Array, ArrayOperation<Array>>::trace(function, ArrayType::scalar(DataType::F64))
            .unwrap()
            .1
    }

    /// Returns the [`RematerializeOperation`] of the instruction at `index` in `program`.
    fn operation(
        program: &Program<Array, ArrayOperation<Array>, Array, Array>,
        index: usize,
    ) -> RematerializeOperation<ArrayType> {
        match program.instructions()[index].operation() {
            ArrayOperation::Rematerialize(operation) => operation.clone(),
            operation => panic!("expected a `rematerialize` operation but got `{operation}`"),
        }
    }

    /// Returns a dot product that contracts the leading dimensions of its inputs and has no batching dimensions.
    fn dot() -> TestIrOperation {
        ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])))
            .into()
    }

    /// Returns a dot product that contracts the trailing dimensions of its inputs and batches their leading ones.
    fn batched_dot() -> TestIrOperation {
        ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0])))
            .into()
    }

    /// Returns a tag with the provided key.
    fn tag(key: &str) -> TestIrOperation {
        ArrayOperation::<Array>::Tag(TagOperation::new(key)).into()
    }

    /// Returns a sine, which is neither a dot product nor a tag.
    fn sine() -> TestIrOperation {
        ArrayOperation::<Array>::from(SinOperation::<ArrayType>::new()).into()
    }

    /// Returns a scalar candidate that output 0 of any of `operations` may produce. The built-in policies classify
    /// candidates by the payloads of their producers only, so the producers carry no input types.
    fn candidate(operations: &[TestIrOperation]) -> ResidualCandidate<'_, ArrayIrType> {
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let producers = operations
            .iter()
            .map(|operation| ResidualProducer::new(operation, 0, Vec::new(), vec![scalar_type.clone()]))
            .collect();
        ResidualCandidate::new(producers, scalar_type)
    }

    /// Classifies a dimension candidate, whose type does not project into [`ArrayType`], with a reference to `policy`
    /// that is lifted from [`ArrayType`] into [`ArrayIrType`]. A policy that declares a native [`ArrayIrType`]
    /// instantiation classifies the candidate with it, while any other policy projects the types of the candidate and
    /// cannot classify it.
    fn classify_lifted_dimension<P: ResidualPolicy<ArrayType>>(
        policy: P,
    ) -> Result<ResidualDecision<ErasedResidualStorage<ArrayIrType>>, ResidualPolicyError> {
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()));
        let dimension_size = TestIrOperation::DimensionSize(
            DimensionSizeOperation::new(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap(),
        );
        let producer = ResidualProducer::new(&dimension_size, 0, Vec::new(), vec![dimension_type.clone()]);
        ResidualPolicyReference::<ArrayType>::new(policy)
            .lift::<ArrayIrType>()
            .classify(&ResidualCandidate::new(vec![producer], dimension_type))
    }

    #[test]
    fn test_rematerialized_function_with_policy() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSavable);
        let program = trace(|x| function.call(x));
        assert_eq!(operation(&program, 0).policy().name(), DOTS_SAVABLE_POLICY_NAME);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize [policy=\"dots_savable\"] %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_rematerialized_function_with_optimization_barrier() {
        // Calls place an optimization barrier on all of their inputs by default, and the selection carries over to
        // every staged call.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?));
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::All,
        );
        let function = function.with_optimization_barrier(RematerializationOptimizationBarrier::None);
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::None,
        );
        let function = function.with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false]));
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::Inputs(vec![false]),
        );
    }

    #[test]
    fn test_rematerialized_function_call() {
        // Each call stages one `rematerialize` operation, whose body is the traced closure, into the domain of its
        // inputs. Repeated calls of one function stage equal operations because they share the reference to its policy.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?));
        let program = trace(|x| function.call(function.call(x)?));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                    %2:f64[] = rematerialize %1 [
                        body={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                            in (%1)
                        },
                    ]
                in (%2)"},
        );
        assert_eq!(operation(&program, 0), operation(&program, 1));
        assert_eq!(program.interpret(Array::scalar(0.5f64).unwrap()), Ok(Array::scalar(0.5f64.sin().sin()).unwrap()));
    }

    #[test]
    fn test_rematerialized_function_call_distinguishes_input_structures() {
        // Calls whose inputs have the same flattened types but different structures trace separate bodies.
        let function = rematerialize(|groups: Vec<Vec<TestTracer>>| {
            Ok(groups.into_iter().map(|group| group[0].clone()).collect::<Vec<_>>())
        });
        let scalar_type = ArrayType::scalar(DataType::F64);
        let two = Array::scalar(2.0f64).unwrap();
        let three = Array::scalar(3.0f64).unwrap();
        let (_, first) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |groups| function.call(groups),
            vec![vec![scalar_type.clone(), scalar_type.clone()]],
        )
        .unwrap();
        assert_eq!(first.interpret(vec![vec![two.clone(), three.clone()]]), Ok(vec![two.clone()]));
        let (_, second) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |groups| function.call(groups),
            vec![vec![scalar_type.clone()], vec![scalar_type]],
        )
        .unwrap();
        assert_eq!(second.interpret(vec![vec![two.clone()], vec![three.clone()]]), Ok(vec![two, three]));
    }

    #[test]
    fn test_rematerialized_function_call_saves_the_values_that_its_policy_selects() {
        // Saving dot products saves the dot product inside the recomputed computation of `x ↦ sin(x · x)`, rather than
        // only choosing among the values that ordinary linearization would save.
        let x = Array::vector(vec![0.1f64, 0.2]).unwrap();
        let sine_of_dot = |x: TestTracer| -> Result<TestTracer, ProgramError> {
            Ok(x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?)
        };
        let function = rematerialize(sine_of_dot);
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.residuals(), &[x.clone()]);
        let function = rematerialize(sine_of_dot).with_policy(DotsSavable);
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.residuals(), &[x, Array::scalar(0.1f64 * 0.1 + 0.2 * 0.2).unwrap()]);
    }

    #[test]
    fn test_rematerialized_function_call_saves_only_the_values_that_differentiation_needs() {
        // Saving everything for `x ↦ exp(x)` saves only `exp(x)`, as differentiating `exp` directly does, and
        // linearization saves the same values as reverse-mode differentiation, according to the policy.
        let x = Array::scalar(0.7f64).unwrap();
        let (_, direct) = differentiate_at(x.clone()).vjp(|x| Ok(x.exp()?)).unwrap();
        let function = rematerialize(|x: TestTracer| Ok(x.exp()?)).with_policy(EverythingSavable);
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.residuals(), direct.residuals());
        assert_eq!(pullback.residuals(), &[Array::scalar(0.7f64.exp()).unwrap()]);
        let function = rematerialize(|x: TestTracer| Ok((x.clone() * x).sin()?));
        let (_, pushforward) = differentiate_at(x.clone()).linearize(|x| function.call(x)).unwrap();
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pushforward.residuals(), &[x.clone()]);
        assert_eq!(pullback.residuals(), &[x]);
    }

    #[test]
    fn test_rematerialized_function_call_with_reference_inputs() {
        // A rematerialized call that reads a reference input batches, with the reference mapped like the value.
        let function = rematerialize(|(reference, x): (TestIrTracer, TestIrTracer)| {
            let value = reference.read()?;
            let context = x.context().clone();
            let product = ArrayOperation::<Array>::Mul(MulOperation::new());
            Ok(context.bind(ArrayIrOperation::Array(product), Vec::new(), &[value, x])?.remove(0))
        });
        let scalar_type = ArrayType::scalar(DataType::F32);
        let (_, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| function.call(inputs),
            (ArrayIrType::from(ReferenceType::new(scalar_type.clone())), ArrayIrType::from(scalar_type.clone())),
        )
        .unwrap();
        let extent = DimensionType::new("batch", DimensionBounds::new(2, Some(3)).unwrap());
        let (batched, _) = program
            .into_flat_program()
            .batched_with_threaded_extent(
                extent,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:ref<f32[2]>, %2:f32[2] .
                let %3:f32[2] = rematerialize %0 %1 %2 [
                    body={
                        lambda %0:dimension<2>, %1:ref<f32[2]>, %2:f32[2] .
                        let %3:f32[2] = reference_read %1
                            %4:f32[2] = mul %3 %2
                        in (%4)
                    },
                ]
                in (%0, %3)"},
        );
    }

    #[test]
    fn test_rematerialized_function_call_with_an_unused_reference_input() {
        // An unused reference input does not make reverse-mode differentiation save the values that the call
        // recomputes.
        let sine_of_square = |x: TestIrTracer| -> Result<TestIrTracer, ProgramError> {
            let context = x.context().clone();
            let square = ArrayIrOperation::Array(ArrayOperation::<Array>::Mul(MulOperation::new()));
            let square = context.bind(square, Vec::new(), &[x.clone(), x])?;
            let sine = ArrayIrOperation::Array(ArrayOperation::<Array>::Sin(SinOperation::new()));
            Ok(context.bind(sine, Vec::new(), &square)?.remove(0))
        };
        let x = TestIrValue::Array(Array::scalar(0.7f32).unwrap());
        let function = rematerialize(sine_of_square);
        let (_, without_reference) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        let function = rematerialize(move |(_reference, x): (TestIrTracer, TestIrTracer)| sine_of_square(x));
        let reference = TestIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap()));
        let (_, with_reference) = differentiate_at((reference, x.clone())).vjp(|inputs| function.call(inputs)).unwrap();
        assert_eq!(without_reference.residuals(), &[x.clone()]);
        assert_eq!(with_reference.residuals(), &[x]);
    }

    #[test]
    fn test_rematerialized_function_call_named_axes() {
        // The body is traced in a fresh trace that is seeded with the named axes in scope where the function is
        // called, so a call under `batch` resolves the batch axis `items` (`x ↦ x · i` at batch item `i`), while a call
        // outside of it cannot.
        let function = rematerialize(|x: TestTracer| {
            let index = x.context().axis_index("items")?.convert_element_type(DataType::F64)?;
            Ok(x * index)
        });
        let traced = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| function.call(x),
            ArrayType::scalar(DataType::F64),
        );
        assert_eq!(
            traced.map(|_| ()),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "items".to_string() })),
        );
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let items = BatchAxisSpecification::named("items");
                Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items)?)
            },
            ArrayType::new_static(DataType::F64, [3]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = rematerialize %0 [
                    body={
                        lambda %0:f64[3] .
                        let %1:u64[3] = iota [type=u64[3], dimension=0]
                            %2:f64[3] = convert_element_type [data_type=f64] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_rematerialized_function_call_body_error() {
        // An error that the body returns while it is traced is the error of the call.
        let function = rematerialize(|_: TestTracer| -> Result<TestTracer, ProgramError> {
            Err(ProgramError::InvalidArgument { message: "the body failed".to_string() })
        });
        let traced = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| function.call(x),
            ArrayType::scalar(DataType::F64),
        );
        assert_eq!(traced.map(|_| ()), Err(ProgramError::InvalidArgument { message: "the body failed".to_string() }));
    }

    #[test]
    fn test_rematerialized_function_call_without_inputs() {
        let function = rematerialize(|inputs: Vec<TestTracer>| Ok(inputs));
        assert_eq!(
            function.call(Vec::<TestTracer>::new()),
            Err(ProgramError::from(TypeError::invalid(
                "`rematerialize` requires at least one input to recover its context from; use \
                 `RematerializedFunction::call_in_context` for functions without inputs",
            ))),
        );
    }

    #[test]
    fn test_rematerialized_function_call_in_context() {
        // Eager arrays dispatch to a context whose operation family cannot represent `rematerialize`, so eager calls
        // name a context that can.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize(|inputs: Vec<TestTracer>| -> Result<Vec<TestTracer>, ProgramError> {
            inputs.into_iter().map(|x| Ok(x.sin()?)).collect()
        });
        assert_eq!(
            function.call_in_context(&context, vec![Array::scalar(0.5f64).unwrap()]),
            Ok(vec![Array::scalar(0.5f64.sin()).unwrap()]),
        );

        // Unlike `call`, `call_in_context` supports functions without inputs.
        assert_eq!(function.call_in_context(&context, Vec::<Array>::new()), Ok(Vec::new()));
    }

    #[test]
    fn test_rematerialized_function_call_in_context_type_universes() {
        // A function without inputs can be called in several type universes. Each universe gets its own reference to
        // the policy of the function, which its later calls reuse, so that repeated calls in one universe stage equal
        // operations.
        let function = rematerialize(|(): ()| Ok(()));
        let array_context = TracingContext::<Array, ArrayOperation<Array>>::new();
        function.call_in_context(&array_context, ()).unwrap();
        function.call_in_context(&array_context, ()).unwrap();
        let array_operations = array_context
            .builder()
            .borrow()
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Rematerialize(operation) => operation.clone(),
                operation => panic!("expected a `rematerialize` operation but got `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(array_operations.len(), 2);
        assert_eq!(array_operations[0], array_operations[1]);
        let ir_context = TracingContext::<TestIrValue, TestIrOperation>::new();
        function.call_in_context(&ir_context, ()).unwrap();
        function.call_in_context(&ir_context, ()).unwrap();
        let ir_operations = ir_context
            .builder()
            .borrow()
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayIrOperation::Rematerialize(operation) => operation.clone(),
                operation => panic!("expected a `rematerialize` operation but got `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(ir_operations.len(), 2);
        assert_eq!(ir_operations[0], ir_operations[1]);
        assert_ne!(ir_operations[0].policy().id(), array_operations[0].policy().id());
    }

    #[test]
    fn test_rematerialized_function_clone() {
        // Clones share the reference to the policy of the function, so their calls stage equal operations.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSavable);
        let clone = function.clone();
        let program = trace(|x| clone.call(function.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));

        // Selecting the same policy again defines a new policy, whose calls stage different operations.
        let other = function.clone().with_policy(DotsSavable);
        let program = trace(|x| other.call(function.call(x)?));
        assert_ne!(operation(&program, 0), operation(&program, 1));
    }

    #[test]
    fn test_rematerialized_function_debug() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None);
        assert_eq!(
            format!("{function:?}"),
            "RematerializedFunction { policy: NothingSavable, optimization_barrier: None, .. }",
        );
    }

    #[test]
    fn test_rematerialize() {
        // Functions that use the default policy share its reference, so their calls stage equal operations.
        let sine = rematerialize(|x: TestTracer| Ok(x.sin()?));
        let identity = rematerialize(|x: TestTracer| Ok(x));
        let program = trace(|x| identity.call(sine.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));
        assert_eq!(operation(&program, 0).policy().name(), NOTHING_SAVABLE_POLICY_NAME);
    }

    #[test]
    fn test_residual_source() {
        let sources = HashSet::from([
            ResidualSource::Input { index: 0 },
            ResidualSource::Constant,
            ResidualSource::Tag { key: "dot".to_owned() },
            ResidualSource::Operation { name: "cos" },
        ]);
        assert_eq!(sources.len(), 4);
        assert!(sources.contains(&ResidualSource::Tag { key: "dot".to_owned() }));
        assert!(!sources.contains(&ResidualSource::Input { index: 1 }));
        assert!(!sources.contains(&ResidualSource::Tag { key: "sin".to_owned() }));
        assert!(!sources.contains(&ResidualSource::Operation { name: "sin" }));
    }

    #[test]
    fn test_saved_residual() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let residual = SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 1 });
        assert_eq!(residual.r#type().as_ref(), &scalar_type);
        assert_eq!(residual.source(), &ResidualSource::Input { index: 1 });

        // Saved residuals compare and hash by their type and their source.
        let residuals = HashSet::from([residual.clone()]);
        assert!(residuals.contains(&residual));
        let other_source = SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 0 });
        let other_type = SavedResidual::new(ArrayType::scalar(DataType::F64), ResidualSource::Input { index: 1 });
        assert!(!residuals.contains(&other_source));
        assert!(!residuals.contains(&other_type));
    }

    #[test]
    fn test_saved_residual_display() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 1 }).to_string(),
            "f32[] from the input 1",
        );
        assert_eq!(
            SavedResidual::new(scalar_type.clone(), ResidualSource::Constant).to_string(),
            "f32[] from a constant",
        );
        assert_eq!(
            SavedResidual::new(scalar_type.clone(), ResidualSource::Tag { key: "dot".to_owned() }).to_string(),
            "f32[] tagged `dot`",
        );
        assert_eq!(
            SavedResidual::new(scalar_type, ResidualSource::Operation { name: "cos" }).to_string(),
            "f32[] output of `cos`",
        );
    }

    #[test]
    fn test_saved_residuals() {
        // `x ↦ sin(tag(x · x, "dot"))` saves its input under every policy, and the dot product whenever the policy
        // saves it, which is reported through its tag when the policy selects it by name.
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input = SavedResidual::new(vector_type.clone(), ResidualSource::Input { index: 0 });
        let sine_of_dot = |x: TestTracer| -> Result<TestTracer, ProgramError> {
            let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
            Ok(x.dot(&x, &dimensions)?.tag("dot")?.sin()?)
        };
        let function = rematerialize(sine_of_dot);
        assert_eq!(saved_residuals(|x: TestTracer| function.call(x), vector_type.clone()), Ok(vec![input.clone()]));
        let function = rematerialize(sine_of_dot).with_policy(DotsSavable);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), vector_type.clone()),
            Ok(vec![input.clone(), SavedResidual::new(scalar_type.clone(), ResidualSource::Operation { name: "dot" })]),
        );
        let function = rematerialize(sine_of_dot).with_policy(SaveOnlyTheseNames::new(["dot"]));
        let tagged = SavedResidual::new(scalar_type.clone(), ResidualSource::Tag { key: "dot".to_owned() });
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), vector_type.clone()),
            Ok(vec![input.clone(), tagged]),
        );

        // An offloaded value is reported as the value that was offloaded, with the type of its offloaded copy.
        let host = Memory::Host { pinned: true };
        let policy = SaveAndOffloadOnlyTheseNames::new(Vec::<String>::new(), ["dot"], host).unwrap();
        let function = rematerialize(sine_of_dot).with_policy(policy);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), vector_type),
            Ok(vec![
                input,
                SavedResidual::new(
                    scalar_type.clone().with_memory(host),
                    ResidualSource::Tag { key: "dot".to_owned() },
                ),
            ]),
        );

        // Reverse-mode rules decide what is saved: a custom function with only a reverse-mode rule saves its residual.
        let sine = custom_function(|x: TestTracer| Ok(x.sin()?))
            .with_vjp(|x: TestTracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
        let function = rematerialize(move |x: TestTracer| sine.call(x)).with_policy(EverythingSavable);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), scalar_type.clone()),
            Ok(vec![SavedResidual::new(scalar_type.clone(), ResidualSource::Operation { name: "cos" })]),
        );

        // A rounded value is reported as the value that was rounded.
        let function = rematerialize(|x: TestTracer| {
            let sine = x.sin()?;
            Ok(sine.clone() * sine)
        })
        .with_policy(EverythingSavable);
        let bf16_type = ArrayType::scalar(DataType::BF16);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), bf16_type.clone()),
            Ok(vec![
                SavedResidual::new(bf16_type.clone(), ResidualSource::Operation { name: "cos" }),
                SavedResidual::new(bf16_type, ResidualSource::Operation { name: "sin" }),
            ]),
        );

        // Inputs are reported at their positions among the flattened inputs of the function (`x · y` saves `y` for the
        // tangent of `x` before `x` for the tangent of `y`). Constants are re-created rather than saved, so a value is
        // reported as a constant only when it is computed from one by an operation that is looked through, such as the
        // rounding of `3` to `bf16` precision below.
        assert_eq!(
            saved_residuals(|(x, y): (TestTracer, TestTracer)| Ok(x * y), (scalar_type.clone(), scalar_type.clone())),
            Ok(vec![
                SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 1 }),
                SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 0 }),
            ]),
        );
        assert_eq!(
            saved_residuals(
                |x: TestTracer| {
                    let constant = x.context().constant(Array::scalar(3.0f64).unwrap())?.reduce_precision(8, 7)?;
                    Ok(x.sin()? * constant)
                },
                scalar_type.clone(),
            ),
            Ok(vec![
                SavedResidual::new(scalar_type.clone(), ResidualSource::Operation { name: "cos" }),
                SavedResidual::new(scalar_type.clone(), ResidualSource::Constant),
            ]),
        );

        // Errors that tracing the function raises are reported as they are.
        assert_eq!(
            saved_residuals(
                |_: TestTracer| -> Result<TestTracer, ProgramError> {
                    Err(ProgramError::InvalidArgument { message: "the function failed".to_string() })
                },
                scalar_type,
            ),
            Err(ProgramError::InvalidArgument { message: "the function failed".to_string() }),
        );
    }

    #[test]
    fn test_nothing_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&NothingSavable), NOTHING_SAVABLE_POLICY_NAME);
        assert_eq!(NothingSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(NothingSavable.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(NothingSavable), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_everything_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&EverythingSavable), EVERYTHING_SAVABLE_POLICY_NAME);
        assert_eq!(EverythingSavable.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save));
        assert_eq!(EverythingSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(EverythingSavable), Ok(ResidualDecision::Save)));
    }

    #[test]
    fn test_dots_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&DotsSavable), DOTS_SAVABLE_POLICY_NAME);
        assert_eq!(DotsSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(DotsSavable.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Save));
        assert_eq!(DotsSavable.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is saved when any of them is a dot product.
        assert_eq!(DotsSavable.classify(&candidate(&[sine(), dot()])), Ok(ResidualDecision::Save));

        // Dot products are recognized in the array operation family itself too.
        let dot = ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(
            vec![0],
            vec![0],
            vec![],
            vec![],
        )));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let producer = ResidualProducer::new(&dot, 0, Vec::new(), vec![scalar_type.clone()]);
        assert_eq!(
            DotsSavable.classify(&ResidualCandidate::new(vec![producer], scalar_type)),
            Ok(ResidualDecision::Save),
        );

        // Lifting a reference to the policy from `ArrayType` into `ArrayIrType` uses its native instantiation, which
        // classifies candidates whose types do not project into `ArrayType` instead of rejecting them.
        assert!(matches!(classify_lifted_dimension(DotsSavable), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_dots_with_no_batch_dimensions_savable() {
        let policy = DotsWithNoBatchDimensionsSavable;
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), DOTS_WITH_NO_BATCH_DIMENSIONS_SAVABLE_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is saved when any of them is a dot product without batching
        // dimensions.
        assert_eq!(policy.classify(&candidate(&[batched_dot(), dot()])), Ok(ResidualDecision::Save));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_offload_dots_with_no_batch_dimensions() {
        let host = Memory::Host { pinned: true };
        let policy = OffloadDotsWithNoBatchDimensions::new(host);
        assert_eq!(policy.destination(), host);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME);
        assert_eq!(
            policy.classify(&candidate(&[dot()])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
        assert_eq!(policy.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is offloaded when any of them is a dot product without
        // batching dimensions.
        assert_eq!(
            policy.classify(&candidate(&[batched_dot(), dot()])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );

        // Policies compare and hash by their destination.
        let policies = HashSet::from([policy]);
        assert!(policies.contains(&OffloadDotsWithNoBatchDimensions::new(host)));
        assert!(!policies.contains(&OffloadDotsWithNoBatchDimensions::new(Memory::Host { pinned: false })));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_save_only_these_names() {
        let policy = SaveOnlyTheseNames::new(["a", "b"]);
        assert_eq!(policy.names(), &["a".to_owned(), "b".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), SAVE_ONLY_THESE_NAMES_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("c")])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is saved when any of them is tagged with a saved name.
        assert_eq!(policy.classify(&candidate(&[tag("c"), tag("b")])), Ok(ResidualDecision::Save));

        // Policies compare and hash by their names.
        let policies = HashSet::from([policy.clone()]);
        assert!(policies.contains(&SaveOnlyTheseNames::new(["a", "b"])));
        assert!(!policies.contains(&SaveOnlyTheseNames::new(["a"])));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_save_any_names_but_these() {
        let policy = SaveAnyNamesButThese::new(["a"]);
        assert_eq!(policy.names(), &["a".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[tag("b")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));

        // Untagged residuals are recomputed.
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is saved when any of them is tagged with a name that is not
        // excluded.
        assert_eq!(policy.classify(&candidate(&[tag("a"), tag("b")])), Ok(ResidualDecision::Save));

        // Policies compare and hash by their names.
        let policies = HashSet::from([policy.clone()]);
        assert!(policies.contains(&SaveAnyNamesButThese::new(["a"])));
        assert!(!policies.contains(&SaveAnyNamesButThese::new(["b"])));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_save_anything_except_these_names() {
        let policy = SaveAnythingExceptTheseNames::new(["a"]);
        assert_eq!(policy.names(), &["a".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[tag("b")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));

        // Unlike `SaveAnyNamesButThese`, untagged residuals are saved, including ones that an excluded tag may also
        // produce.
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a"), sine()])), Ok(ResidualDecision::Save));

        // Policies compare and hash by their names.
        let policies = HashSet::from([policy.clone()]);
        assert!(policies.contains(&SaveAnythingExceptTheseNames::new(["a"])));
        assert!(!policies.contains(&SaveAnythingExceptTheseNames::new(["b"])));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Save)));
    }

    #[test]
    fn test_save_and_offload_only_these_names() {
        let host = Memory::Host { pinned: false };
        let policy = SaveAndOffloadOnlyTheseNames::new(["a"], ["b"], host).unwrap();
        assert_eq!(policy.savable_names(), &["a".to_owned()]);
        assert_eq!(policy.offloadable_names(), &["b".to_owned()]);
        assert_eq!(policy.destination(), host);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save));
        assert_eq!(
            policy.classify(&candidate(&[tag("b")])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
        assert_eq!(policy.classify(&candidate(&[tag("c")])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));

        // Saving takes precedence over offloading for residuals that both kinds of tags may produce.
        assert_eq!(policy.classify(&candidate(&[tag("b"), tag("a")])), Ok(ResidualDecision::Save));

        // Policies compare and hash by their names and their destination.
        let policies = HashSet::from([policy.clone()]);
        assert!(policies.contains(&SaveAndOffloadOnlyTheseNames::new(["a"], ["b"], host).unwrap()));
        assert!(!policies.contains(&SaveAndOffloadOnlyTheseNames::new(["b"], ["a"], host).unwrap()));
        assert!(!policies.contains(&SaveAndOffloadOnlyTheseNames::new(["a"], ["b"], Memory::Device).unwrap()));

        // Lifting a reference to the policy into `ArrayIrType` uses its native instantiation.
        assert!(matches!(classify_lifted_dimension(policy), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_save_and_offload_only_these_names_overlapping_names() {
        assert_eq!(
            SaveAndOffloadOnlyTheseNames::new(["a", "b", "c"], ["c", "a"], Memory::Device),
            Err(ProgramError::InvalidArgument {
                message: "names `a`, `c` cannot be both savable and offloadable by a \
                    `save_and_offload_only_these_names` policy"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_save_from_both_policies() {
        let host = Memory::Host { pinned: true };
        let policy =
            SaveFromBothPolicies::new(OffloadDotsWithNoBatchDimensions::new(host), SaveOnlyTheseNames::new(["a"]));
        assert_eq!(policy.first(), &OffloadDotsWithNoBatchDimensions::new(host));
        assert_eq!(policy.second(), &SaveOnlyTheseNames::new(["a"]));
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), SAVE_FROM_BOTH_POLICIES_POLICY_NAME);

        // The first policy decides first, keeping its storage (which still offloads to `host`), and the second one
        // decides what the first recomputes.
        let Ok(ResidualDecision::SaveWith(storage)) = policy.classify(&candidate(&[dot()])) else {
            panic!("expected an offloaded residual");
        };
        assert_eq!(storage.name(), "memory_transfer");
        let mut store = storage.store_payloads(&ArrayIrType::from(ArrayType::scalar(DataType::F64))).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.remove(0).downcast::<TransferToMemoryOperation>().unwrap().destination(), host);
        assert!(matches!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save)));
        assert!(matches!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute)));

        // Rejections of either policy are returned as they are. A rejection of the first policy is returned without
        // consulting the second one, and a residual that the first policy saves never reaches a rejecting second one.
        let rejecting = RematerializationPolicyFn::new::<ArrayIrType>(|_| {
            Err::<ResidualDecision<NoStorage>, _>(ResidualRejection::new("rejected"))
        });
        let policy = SaveFromBothPolicies::new(rejecting.clone(), EverythingSavable);
        assert_eq!(policy.classify(&candidate(&[sine()])).map(|_| ()), Err(ResidualRejection::new("rejected")));
        let policy = SaveFromBothPolicies::new(NothingSavable, rejecting.clone());
        assert_eq!(policy.classify(&candidate(&[sine()])).map(|_| ()), Err(ResidualRejection::new("rejected")));
        let policy = SaveFromBothPolicies::new(EverythingSavable, rejecting);
        assert!(matches!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save)));

        // The policy declares no native instantiations, so lifting a reference to it projects the types of each
        // candidate and cannot classify the ones that do not project.
        assert_eq!(
            classify_lifted_dimension(SaveFromBothPolicies::new(DotsSavable, NothingSavable)).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedProjection {
                policy: SAVE_FROM_BOTH_POLICIES_POLICY_NAME.to_owned(),
                position: "the output 0 of producer `dimension_size`".to_owned(),
                residual_type: DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()).to_string(),
            }),
        );
    }

    #[test]
    fn test_rematerialization_policy_fn() {
        // Saves the residuals that have one producer and recomputes the ones that several producers may produce.
        let policy = RematerializationPolicyFn::new::<ArrayIrType>(|candidate| {
            Ok::<_, ResidualRejection>(match candidate.producers().len() {
                1 => ResidualDecision::<NoStorage>::Save,
                _ => ResidualDecision::Recompute,
            })
        });
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), REMATERIALIZATION_POLICY_FN_POLICY_NAME);
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[sine(), dot()])), Ok(ResidualDecision::Recompute));

        // Names, either static or owned, apply to clones too, and the closure does not render.
        let policy = policy.with_name("save_unique_producers");
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy.clone()), "save_unique_producers");
        assert_eq!(format!("{policy:?}"), "RematerializationPolicyFn { name: \"save_unique_producers\", .. }");
        let policy = policy.with_name(format!("save_{}_producers", "single"));
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy.clone()), "save_single_producers");

        // Closures may return storages too.
        let host = Memory::Host { pinned: true };
        let policy = RematerializationPolicyFn::new::<ArrayIrType>(move |_| {
            Ok::<_, ResidualRejection>(ResidualDecision::SaveWith(MemoryTransferStorage::new(host)))
        });
        assert_eq!(
            policy.classify(&candidate(&[sine()])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
    }

    #[test]
    fn test_memory_transfer_storage() {
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()));
        let host = Memory::Host { pinned: true };
        let storage = MemoryTransferStorage::new(host);
        assert_eq!(storage.destination(), host);
        assert_eq!(ResidualStorage::<ArrayIrType>::name(&storage), "memory_transfer");

        // Storing transfers the residual to the destination, and restoring transfers it back to its own memory.
        let mut store = storage.store_payloads(&scalar_type).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.remove(0).downcast::<TransferToMemoryOperation>().unwrap().destination(), host);
        let stored_type = ArrayIrType::from(ArrayType::scalar(DataType::F64).with_memory(host));
        let mut restore = storage.restore_payloads(&stored_type, &scalar_type).unwrap();
        assert_eq!(restore.len(), 1);
        assert_eq!(restore.remove(0).downcast::<TransferToMemoryOperation>().unwrap().destination(), Memory::Device);

        // Residuals that are not arrays cannot be offloaded.
        let error = ResidualPolicyError::UnsupportedStorage {
            storage: "memory_transfer".to_owned(),
            residual_type: dimension_type.to_string(),
            message: "the residual is not an array".to_owned(),
        };
        assert_eq!(storage.store_payloads(&dimension_type).map(|_| ()), Err(error.clone()));
        assert_eq!(storage.restore_payloads(&dimension_type, &dimension_type).map(|_| ()), Err(error));

        // Storages compare and hash by their destination.
        let storages = HashSet::from([storage]);
        assert!(storages.contains(&MemoryTransferStorage::new(host)));
        assert!(!storages.contains(&MemoryTransferStorage::new(Memory::Device)));
    }
}
