//! Rematerialization (i.e., gradient checkpointing), which trades computation for memory under differentiation. A
//! rematerialized function computes what its body computes, but differentiation saves only the values of the body that
//! its [`ResidualPolicy`] selects and recomputes the others from the saved values when the derivative computation
//! needs them, instead of keeping every value of the body alive until then. This is the analogue of
//! [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) (also known as `jax.remat`).
//!
//! [`rematerialize`] creates a rematerialized function from a closure, [`Rematerialize::with_policy`] selects which
//! values it saves (refer to the [`policies`] module for the built-in policies), and [`Rematerialize::call`] stages one
//! call of the function as a [`RematerializeOperation`] whose body is the traced closure. Nothing is derived when the
//! function is called: the transforms derive the derivatives of the body when they need them. Linearization and
//! reverse-mode differentiation split the derivative of the body into the work that runs up front, which computes the
//! outputs and the values that the policy saves, and a _differentiated_ call that recomputes everything else from the
//! saved values when the backward computation runs. [`saved_residuals`] reports which values a function saves.
//!
//! # Examples
//!
//! ## Gradients
//!
//! A rematerialized function is differentiated like any other function. Its closure annotates its tracer input, which
//! determines the type universe that the function is traced in. By default it saves nothing but its inputs, so the
//! backward computation of `x ↦ sin(x · x)` recomputes the dot product and its cosine from `x`, while saving dot
//! products saves the dot product and recomputes only the cosine:
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::{
//! #     DotsSaveable, ResidualSource, SavedResidual, rematerialize, saved_residuals,
//! # };
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, Dot, DotDimensionNumbers, ProgramError, Sin, TracingContext,
//! #     differentiate_at,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
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
//! let function = rematerialize(sine_of_dot).with_policy(DotsSaveable);
//! let dot = SavedResidual::new(ArrayType::scalar(DataType::F64), ResidualSource::Operation { name: "dot" });
//! assert_eq!(saved_residuals(|x: Tracer| function.call(x), vector_type)?, vec![input, dot]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Named Checkpoints
//!
//! [`Tag`](crate::Tag)ged values (the analogue of `jax.ad_checkpoint.checkpoint_name`) can be saved by name, e.g.,
//! with [`SaveOnlyTheseNames`], which saves the tagged values whose names it lists and recomputes everything else:
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::{
//! #     ResidualSource, SaveOnlyTheseNames, SavedResidual, rematerialize, saved_residuals,
//! # };
//! # use ryft_core::{Array, ArrayOperation, ArrayType, DataType, ProgramError, Sin, Tag, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
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
//! [`PolicyFn`] defines a policy through a closure that classifies each candidate residual, e.g., by the payloads of
//! the operations that may produce it (refer to [`ResidualProducer::payload`](crate::ResidualProducer::payload)):
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::{
//! #     PolicyFn, ResidualSource, SavedResidual, rematerialize, saved_residuals,
//! # };
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, NoStorage, ProgramError, ResidualDecision, ResidualRejection, Sin,
//! #     SinOperation, TracingContext,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! // Saves the sines and recomputes everything else.
//! let policy = PolicyFn::new::<ArrayType>(|candidate| {
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
//! Policies can save values by offloading them through a [`ResidualStorage`](crate::ResidualStorage) instead of keeping
//! them in device memory. For example, [`OffloadDotsWithNoBatchDimensions`] and [`SaveAndOffloadOnlyTheseNames`]
//! transfer the values that they offload to another [`Memory`](crate::Memory) (e.g., pinned host memory) once they are
//! computed and back before the backward computation uses them (refer to [`MemoryTransferStorage`]):
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::{
//! #     ResidualSource, SaveAndOffloadOnlyTheseNames, SavedResidual, rematerialize, saved_residuals,
//! # };
//! # use ryft_core::{Array, ArrayOperation, ArrayType, DataType, Memory, ProgramError, Sin, Tag, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
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
//! [`Rematerialize::with_optimization_barrier`]), which keeps compilers from merging the recomputation with the forward
//! computation and thus from undoing the memory savings. The barrier can also keep compilers from optimizing across it
//! in other ways (e.g., from fusing operations), so it should be disabled when something else already separates the
//! recomputation from the forward computation. This is the case for a rematerialized function that is called inside
//! the body of a loop (e.g., a `scan` over the layers of a model), because the backward loop recomputes each iteration
//! after the forward loop has finished, which is also why
//! [JAX recommends](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) `prevent_cse=False` there:
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::rematerialize;
//! # use ryft_core::{Array, ArrayOperation, ProgramError, RematerializationOptimizationBarrier, Sin, TracingContext};
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
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
//! # use ryft_core::differentiation::rematerialization::rematerialize;
//! # use ryft_core::{
//! #     Array, ArrayOperation, Mul, ProgramError, RematerializationOptimizationBarrier, Sin, TracingContext,
//! # };
//! type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
//!
//! // Only the activation `x`, which is the first leaf of the input, goes through the barrier.
//! let layer = rematerialize(|(x, w): (Tracer, Tracer)| Ok::<_, ProgramError>(x.mul(&w)?.sin()?))
//!     .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true, false]));
//! # let _ = layer;
//! ```

// TODO(eaplatanios): Review this module.

pub mod policies;

use std::any::{Any, TypeId};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, Mutex};

use crate::arrays::ArrayType;
use crate::axes::NamedAxes;
use crate::contexts::Context;
use crate::differentiation::DifferentiationRule;
use crate::differentiation::forward::DifferentiableOperation;
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::operations::{
    ReducePrecisionOperation, RematerializationOptimizationBarrier, RematerializeOperation, TagOperation,
    TransferToMemoryOperation,
};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{
    PartialEvaluationContext, PartiallyEvaluatableOperation, ResidualPolicy, ResidualPolicyReference,
};
use crate::programs::{Operation, OperationPayloadProjection, ProgramError, Type, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracingContext};

pub use policies::{
    DOTS_SAVEABLE_POLICY_NAME, DOTS_WITH_NO_BATCH_DIMENSIONS_SAVEABLE_POLICY_NAME, DotsSaveable,
    DotsWithNoBatchDimensionsSaveable, EVERYTHING_SAVEABLE_POLICY_NAME, EverythingSaveable, MemoryTransferStorage,
    NOTHING_SAVEABLE_POLICY_NAME, NothingSaveable, OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME,
    OffloadDotsWithNoBatchDimensions, POLICY_FN_POLICY_NAME, PolicyFn, SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME,
    SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME, SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME,
    SAVE_FROM_BOTH_POLICIES_POLICY_NAME, SAVE_ONLY_THESE_NAMES_POLICY_NAME, SaveAndOffloadOnlyTheseNames,
    SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies, SaveOnlyTheseNames,
};

/// [`PolicyReferences`] of the default [`NothingSaveable`] policy, which every [`Rematerialize`] that does not select
/// a policy shares, so that all of their calls stage operations whose policies compare equal.
static DEFAULT_POLICY_REFERENCES: LazyLock<Arc<PolicyReferences<NothingSaveable>>> =
    LazyLock::new(|| Arc::new(PolicyReferences::new(NothingSaveable)));

/// Residual policy of a [`Rematerialize`] together with its [`ResidualPolicyReference`]s in the type universes that
/// its calls have used so far. [`ResidualPolicyReference`]s compare by the identity of the policy definition that
/// [`ResidualPolicyReference::new`] registers, so creating one per call would make calls of the same function stage
/// unequal operations. Clones of a [`Rematerialize`] share one [`PolicyReferences`], and therefore stage equal
/// operations too.
struct PolicyReferences<P> {
    /// Residual policy.
    policy: P,

    /// [`ResidualPolicyReference`] to `policy` in each type universe that was used so far, keyed by the [`TypeId`] of
    /// the universe.
    references: Mutex<Vec<(TypeId, Box<dyn Any + Send + Sync>)>>,
}

impl<P> PolicyReferences<P> {
    /// Creates new [`PolicyReferences`] for `policy` that hold no references yet.
    #[inline]
    fn new(policy: P) -> Self {
        Self { policy, references: Mutex::new(Vec::new()) }
    }

    /// Returns the [`ResidualPolicyReference`] to the policy in the type universe `T`, which is registered on first
    /// use.
    fn reference<T: 'static + Type>(&self) -> ResidualPolicyReference<T>
    where
        P: Clone + ResidualPolicy<T>,
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

/// Rematerialized function, which [`rematerialize`] creates from a closure over [`DomainTracer`]s. Refer to the
/// [module documentation](self) for more information.
pub struct Rematerialize<Input, Output, Body, P = NothingSaveable> {
    /// Closure that computes the body of the function.
    body: Body,

    /// Residual policy of the function together with its references in the type universes that were used so far.
    policy: Arc<PolicyReferences<P>>,

    /// Inputs of the staged calls on which backends place an optimization barrier when the calls are differentiated
    /// (refer to [`RematerializeOperation::optimization_barrier`]).
    optimization_barrier: RematerializationOptimizationBarrier,

    /// Input and output types of the closure.
    marker: PhantomData<fn() -> (Input, Output)>,
}

impl<Input, Output, Body, P> Rematerialize<Input, Output, Body, P> {
    /// Returns this function with the provided residual policy, which decides which values of the body differentiation
    /// saves (refer to the [`policies`] module for the built-in policies).
    #[inline]
    pub fn with_policy<Q>(self, policy: Q) -> Rematerialize<Input, Output, Body, Q> {
        Rematerialize {
            body: self.body,
            policy: Arc::new(PolicyReferences::new(policy)),
            optimization_barrier: self.optimization_barrier,
            marker: PhantomData,
        }
    }

    /// Sets the inputs on which backends place an optimization barrier when the staged calls of this function are
    /// differentiated, which are [all of them](RematerializationOptimizationBarrier::All) by default (refer to
    /// [`RematerializeOperation::optimization_barrier`]). A [`RematerializationOptimizationBarrier::Inputs`] selection
    /// has one entry per leaf of the input of this function, in [`Parameterized::parameters`] order. This is the
    /// analogue of the `prevent_cse` parameter of
    /// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html), which can be disabled when
    /// the function is called in a loop body (e.g., of a `scan`), where the loop already keeps the recomputation from
    /// being merged with the original computation.
    #[inline]
    pub fn with_optimization_barrier(mut self, optimization_barrier: RematerializationOptimizationBarrier) -> Self {
        self.optimization_barrier = optimization_barrier;
        self
    }

    /// Stages one call of this function on the provided `input` value and returns its output value. The [`Context`]
    /// `C` that the call is staged into is the [`DispatchDomain`](Value::DispatchDomain) of the values in `input`, so
    /// it is never named at a construction or call site. The body is traced in a fresh trace that is seeded with the
    /// named axes in scope in `C` (refer to [`NamedAxes::named_axes`]), so that it resolves the axes of enclosing
    /// transforms as it would if it were inlined.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves (in which case [`call_in_context`](Self::call_in_context)
    /// must be used instead), when tracing the body fails, or when the staged [`RematerializeOperation`] rejects the
    /// call.
    pub fn call<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: 'static, Value = V> + NamedAxes,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
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
                "`rematerialize` requires at least one input to recover its context from; use \
                 `Rematerialize::call_in_context` for functions without inputs",
            )
            .into());
        };
        self.stage(&first.dispatch_domain(), input_types, input_values.as_slice())
    }

    /// Stages one call of this function on the provided `input` value in the provided `context` and returns its output
    /// value. This is the explicit-context counterpart of [`call`](Self::call), which supports functions without
    /// inputs and contexts that are not the dispatch domain of the input values.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when tracing the body fails or when the staged [`RematerializeOperation`] rejects
    /// the call.
    pub fn call_in_context<
        C: Context<Type: 'static> + NamedAxes,
        InputValues: Parameterized<C::Value, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        context: &C,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
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
        self.stage(context, input_types, input_values.as_slice())
    }

    /// Traces the body at `input_types` and binds one call of it to `input_values` in `context`.
    fn stage<C: Context<Type: 'static> + NamedAxes>(
        &self,
        context: &C,
        input_types: Input::To<C::Type>,
        input_values: &[C::Value],
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
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

impl<Input, Output, Body: Clone, P> Clone for Rematerialize<Input, Output, Body, P> {
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

impl<Input, Output, Body, P: Debug> Debug for Rematerialize<Input, Output, Body, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Rematerialize")
            .field("policy", &self.policy.policy)
            .field("optimization_barrier", &self.optimization_barrier)
            .finish_non_exhaustive()
    }
}

/// Creates a [`Rematerialize`] function from a closure `x ↦ y = f(x)` over [`DomainTracer`]s, which saves nothing under
/// differentiation (i.e., it uses the [`NothingSaveable`] policy) and places an optimization barrier on its inputs when
/// it is differentiated. The closure must annotate the type of its tracer input, which determines the context that its
/// body is traced in, and nothing is traced until the function is called. Refer to the [module documentation](self)
/// for more information.
#[inline]
pub fn rematerialize<Input, Output, Body: Fn(Input) -> Result<Output, ProgramError>>(
    body: Body,
) -> Rematerialize<Input, Output, Body> {
    Rematerialize {
        body,
        policy: DEFAULT_POLICY_REFERENCES.clone(),
        optimization_barrier: RematerializationOptimizationBarrier::All,
        marker: PhantomData,
    }
}

/// Source of a value that differentiation saves for the backward computation of a function (refer to
/// [`saved_residuals`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResidualSource {
    /// Input of the function at the provided index among its flattened inputs.
    Input {
        /// Index of the input among the flattened inputs of the function.
        index: usize,
    },

    /// Constant of the function.
    Constant,

    /// Value tagged with the provided key (refer to [`Tag`](crate::Tag)).
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

/// Value that differentiation saves for the backward computation of a function, together with its
/// [`ResidualSource`], which [`saved_residuals`] reports. It renders like the entries that JAX's
/// [`print_saved_residuals`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#inspecting-residuals-with-jax-ad-checkpoint-print-saved-residuals)
/// prints (e.g., `f32[3] from the input 0` or ``f32[] tagged `dot` ``).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SavedResidual<T: Type> {
    /// Type of the saved value.
    residual_type: T,

    /// Source of the saved value.
    source: ResidualSource,
}

impl<T: Type> SavedResidual<T> {
    /// Creates a new [`SavedResidual`] of type `residual_type` with the provided source.
    #[inline]
    pub fn new(residual_type: T, source: ResidualSource) -> Self {
        Self { residual_type, source }
    }

    /// Returns the type of the saved value.
    #[inline]
    pub fn residual_type(&self) -> &T {
        &self.residual_type
    }

    /// Returns the source of the saved value.
    #[inline]
    pub fn source(&self) -> &ResidualSource {
        &self.source
    }
}

impl<T: Type> Display for SavedResidual<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            ResidualSource::Input { index } => write!(formatter, "{} from the input {index}", self.residual_type),
            ResidualSource::Constant => write!(formatter, "{} from a constant", self.residual_type),
            ResidualSource::Tag { key } => write!(formatter, "{} tagged `{key}`", self.residual_type),
            ResidualSource::Operation { name } => write!(formatter, "{} output of `{name}`", self.residual_type),
        }
    }
}

/// Returns the values that reverse-mode differentiation of `function` at `input_types` saves for its backward
/// computation, in the order in which the backward computation receives them, which is the analogue of JAX's
/// [`jax.ad_checkpoint.saved_residuals`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#inspecting-residuals-with-jax-ad-checkpoint-print-saved-residuals).
/// Use it to check what a [`rematerialize`]d function and its residual policy save. The function is traced into a
/// [`Program`](crate::Program) like [`TracingContext::trace`] and the program is linearized, and each residual of the
/// linearization is reported with its [`ResidualSource`]. The program is linearized for reverse-mode differentiation
/// (i.e., with the [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rules of its operations). The
/// source of a saved value looks through the operations that residual placement stages on it (i.e., the rounding of
/// narrow floating-point values and the store operations of [`MemoryTransferStorage`]), so that an offloaded value is
/// reported as the value that was offloaded. Such operations that the function applies itself are looked through as
/// well.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::differentiation::rematerialization::{
/// #     DotsSaveable, ResidualSource, SavedResidual, rematerialize, saved_residuals,
/// # };
/// # use ryft_core::{Array, ArrayOperation, ArrayType, DataType, Dot, DotDimensionNumbers, ProgramError, Sin};
/// # use ryft_core::TracingContext;
/// # fn main() -> Result<(), ProgramError> {
/// type Tracer = ryft_core::Tracer<TracingContext<Array, ArrayOperation<Array>>>;
///
/// // `x ↦ sin(x · x)` saves its input and, because the policy saves dot products, the dot product as well, from
/// // which the backward computation recomputes the cosine.
/// let function = rematerialize(|x: Tracer| {
///     Ok(x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?)
/// })
/// .with_policy(DotsSaveable);
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
pub fn saved_residuals<V, O, Input, Output, F>(
    function: F,
    input_types: Input,
) -> Result<Vec<SavedResidual<V::Type>>, ProgramError>
where
    V: Value<Type: 'static + DifferentiableType>,
    O: Operation<Type = V::Type>
        + OperationPayloadProjection
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>,
    F: FnOnce(Input::To<Tracer<TracingContext<V, O>>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<TracingContext<V, O>>>>,
    Output: Parameterized<Tracer<TracingContext<V, O>>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
{
    // The program is linearized with the `jvp_for_transpose` rules that reverse-mode differentiation uses, which can
    // save different values than the `jvp` rules (e.g., for custom functions with distinct rules).
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        DimensionBounds, DimensionType, Memory, ShardingDimension,
    };
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::operations::custom_function;
    use crate::operations::{Cos, Dot, DotDimensionNumbers, Exp, MulOperation, ReferenceRead, Sin, SinOperation, Tag};
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

    #[test]
    fn test_rematerialize_with_policy() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSaveable);
        let program = trace(|x| function.call(x));
        assert_eq!(operation(&program, 0).policy().name(), "dots_saveable");
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize [policy=\"dots_saveable\"] %0 [
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
    fn test_rematerialize_with_optimization_barrier() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?));
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::All
        );
        let function = function.with_optimization_barrier(RematerializationOptimizationBarrier::None);
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::None
        );
        let function = function.with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false]));
        assert_eq!(
            operation(&trace(|x| function.call(x)), 0).optimization_barrier(),
            &RematerializationOptimizationBarrier::Inputs(vec![false]),
        );
    }

    #[test]
    fn test_rematerialize_call() {
        // Each call stages one `rematerialize` operation, whose body is the traced closure, into the dispatch domain of
        // its inputs. Repeated calls of one function stage equal operations because they share the reference to its
        // policy.
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
        assert_eq!(program.interpret(Array::scalar(0.5f64).unwrap()), Ok(Array::scalar(0.5f64.sin().sin()).unwrap()),);
    }

    #[test]
    fn test_rematerialize_call_distinguishes_input_structures() {
        // Regression test for review finding R1: calls whose inputs have the same flattened types but different
        // structures trace separate bodies.
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
    fn test_rematerialize_call_saves_the_values_that_its_policy_selects() {
        // Regression test for review finding R3: saving dot products saves the dot product inside the recomputed
        // computation of `x ↦ sin(x · x)`, rather than only choosing among the values that ordinary linearization
        // would save.
        let x = Array::vector(vec![0.1f64, 0.2]).unwrap();
        let sine_of_dot = |x: TestTracer| -> Result<TestTracer, ProgramError> {
            Ok(x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?)
        };
        let function = rematerialize(sine_of_dot);
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.residuals(), &[x.clone()]);
        let function = rematerialize(sine_of_dot).with_policy(DotsSaveable);
        let (_, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.residuals(), &[x, Array::scalar(0.1f64 * 0.1 + 0.2 * 0.2).unwrap()]);
    }

    #[test]
    fn test_rematerialize_call_saves_only_the_values_that_differentiation_needs() {
        // Regression test for review findings R6 and R8: saving everything for `x ↦ exp(x)` saves only `exp(x)`, as
        // differentiating `exp` directly does, and linearization saves the same values as reverse-mode
        // differentiation, according to the policy.
        let x = Array::scalar(0.7f64).unwrap();
        let (_, direct) = differentiate_at(x.clone()).vjp(|x| Ok(x.exp()?)).unwrap();
        let function = rematerialize(|x: TestTracer| Ok(x.exp()?)).with_policy(EverythingSaveable);
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
    fn test_rematerialize_call_with_reference_inputs() {
        // Regression test for review finding R5: batching a rematerialized call that reads a reference input succeeds.
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
        let batched = program.into_flat_program().batched_with_threaded_extent(
            extent,
            ShardingDimension::Replicated,
            &[BatchAxis::new(0), BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        assert!(batched.is_ok());

        // Regression test for review finding R7: an unused reference input does not make reverse-mode differentiation
        // save the values that the call recomputes.
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
    fn test_rematerialize_call_without_inputs() {
        let function = rematerialize(|inputs: Vec<TestTracer>| Ok(inputs));
        assert_eq!(
            function.call(Vec::<TestTracer>::new()),
            Err(ProgramError::from(TypeError::invalid(
                "`rematerialize` requires at least one input to recover its context from; use \
                 `Rematerialize::call_in_context` for functions without inputs",
            ))),
        );
    }

    #[test]
    fn test_rematerialize_call_in_context() {
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
    fn test_rematerialize_clone() {
        // Clones share the reference to the policy of the function, so their calls stage equal operations.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSaveable);
        let clone = function.clone();
        let program = trace(|x| clone.call(function.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));

        // Selecting the same policy again defines a new policy, whose calls stage different operations.
        let other = function.clone().with_policy(DotsSaveable);
        let program = trace(|x| other.call(function.call(x)?));
        assert_ne!(operation(&program, 0), operation(&program, 1));
    }

    #[test]
    fn test_rematerialize_debug() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None);
        assert_eq!(
            format!("{function:?}"),
            "Rematerialize { policy: NothingSaveable, optimization_barrier: None, .. }",
        );
    }

    #[test]
    fn test_rematerialize() {
        // Functions that use the default policy share its reference, so their calls stage equal operations.
        let sine = rematerialize(|x: TestTracer| Ok(x.sin()?));
        let identity = rematerialize(|x: TestTracer| Ok(x));
        let program = trace(|x| identity.call(sine.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));
        assert_eq!(operation(&program, 0).policy().name(), "nothing_saveable");
    }

    #[test]
    fn test_saved_residual() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let residual = SavedResidual::new(scalar_type.clone(), ResidualSource::Input { index: 1 });
        assert_eq!(residual.residual_type(), &scalar_type);
        assert_eq!(residual.source(), &ResidualSource::Input { index: 1 });
        assert_eq!(residual.to_string(), "f32[] from the input 1");
        assert_eq!(
            SavedResidual::new(scalar_type.clone(), ResidualSource::Constant).to_string(),
            "f32[] from a constant"
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
        let function = rematerialize(sine_of_dot).with_policy(DotsSaveable);
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
                    ResidualSource::Tag { key: "dot".to_owned() }
                ),
            ]),
        );

        // Reverse-mode rules decide what is saved: a custom function with only a reverse-mode rule saves its residual.
        let sine = custom_function(|x: TestTracer| Ok(x.sin()?))
            .with_vjp(|x: TestTracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
        let function = rematerialize(move |x: TestTracer| sine.call(x)).with_policy(EverythingSaveable);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), scalar_type.clone()),
            Ok(vec![SavedResidual::new(scalar_type.clone(), ResidualSource::Operation { name: "cos" })]),
        );

        // A rounded value is reported as the value that was rounded.
        let function = rematerialize(|x: TestTracer| {
            let sine = x.sin()?;
            Ok(sine.clone() * sine)
        })
        .with_policy(EverythingSaveable);
        let bf16_type = ArrayType::scalar(DataType::BF16);
        assert_eq!(
            saved_residuals(|x: TestTracer| function.call(x), bf16_type.clone()),
            Ok(vec![
                SavedResidual::new(bf16_type.clone(), ResidualSource::Operation { name: "cos" }),
                SavedResidual::new(bf16_type, ResidualSource::Operation { name: "sin" }),
            ]),
        );
    }
}
