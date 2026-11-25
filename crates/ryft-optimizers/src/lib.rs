//! Composable gradient-based optimizers over [`Parameterized`] structures, in the style of
//! [Optax](https://optax.readthedocs.io/).
//!
//! An optimizer is a pure gradient transformation. [`Optimizer::initialize`] creates a state for a parameter structure,
//! and [`Optimizer::update`] maps gradients and that state to updates and a new state, without mutating anything. The
//! caller applies the updates with [`apply_updates`] and threads the new state into the next step. Because both the
//! state and the updates are ordinary [`Parameterized`] structures whose leaves are array values, the same optimizer
//! runs eagerly on host [`Array`](ryft_core::Array)s, on the tracers of differentiation and staging, and inside
//! compiled training steps, where its state crosses the compiled function boundary like any other input or output.
//!
//! # State
//!
//! The state of an optimizer is generic over the parameter structure that it was initialized for. Per-leaf statistics
//! (e.g., the moment estimates of [`ScaleByAdam`]) are stored as structures of the same type as the parameters, so the
//! leaves of a state are named by the paths of the parameters (e.g., `$.first_moments.weights` for the parameter at
//! `$.weights`), and checkpoints of the state stay readable when the parameters change order. A gradient structure
//! that does not match the structure of the state is rejected with [`OptimizerError::StructureMismatch`].
//!
//! Stateful optimizers that keep a step count (which most of them do) create it from the first leaf of the parameters,
//! so they need at least one leaf, and their [`Optimizer::initialize`] rejects a parameter structure without leaves
//! with [`OptimizerError::StructureMismatch`]. This also applies to an empty subset of the leaves that [`Masked`] or
//! [`Partition`] select for such an optimizer.
//!
//! # Building Optimizers
//!
//! Larger optimizers are chains of small transformations. The [`transformations`] module provides the rescalings of
//! the Adam family (e.g., [`ScaleByAdam`]), of root mean squares (e.g., [`ScaleByRms`]), and of factored statistics
//! (e.g., [`ScaleByFactoredRms`]), together with momentum ([`Trace`] and [`Ema`]), sign-based rescalings, norm-based
//! rescalings, clipping (e.g., [`ClipByGlobalNorm`]), constraints, step sizes, line searches (e.g.,
//! [`ScaleByZoomLinesearch`]), L-BFGS preconditioning ([`ScaleByLbfgs`]), scaling ([`Scale`] and [`ScaleBySchedule`]),
//! weight decay ([`AddDecayedWeights`]), and gradient noise ([`AddNoise`]). The [`combinators`] module composes
//! optimizers in order ([`Chain`]) and applies them to subsets of the leaves that are selected by their paths
//! ([`Masked`] and [`Partition`]). The [`wrappers`] module changes when or how another optimizer is applied (e.g.,
//! [`ApplyIfFinite`] skips steps whose gradients are not finite, [`MultiSteps`] accumulates gradients, and
//! [`InjectLearningRate`] keeps the learning rate in the state). The [`schedules`] module provides learning rate
//! [`Schedule`]s that are evaluated from the step count with array operations, and that can be joined with
//! [`JoinedSchedule`]. The [`optimizers`] module provides ready to use chains with named states, such as [`Sgd`],
//! [`Adam`], [`AdamW`], [`RmsProp`], [`Adafactor`], [`Lion`], [`Lbfgs`], [`Muon`], and [`Prodigy`], together with
//! [`Dimuon`], an unofficial reconstruction of the LoRA optimizer of Thinking Machines, whose formulation is
//! unpublished.
//!
//! Some transformations need more than the gradients, the state, and the parameters (e.g., line searches need the value
//! of the objective). [`UpdateArguments`] carries these optional arguments, which every combinator forwards unchanged
//! (or restricted to the leaves that it selects).
//!
//! # Numerics
//!
//! Statistics (i.e., moments and norms) are computed in `f32`, or in `f64` when the corresponding gradient leaf is
//! `f64`. [`Optimizer::initialize`] sees only the parameters, so it creates the statistics of `f64` parameters in
//! `f64`, and every update stores each statistic in the data type of the one that it replaces, so that states keep
//! their data types across updates (e.g., of compiled training steps). The global norm of [`ClipByGlobalNorm`] is
//! accumulated in `f64` when any leaf is `f64`. Gradients of narrower floating-point data types (e.g., `bf16`) are
//! converted up before they are accumulated, and every update is returned in the data type of its gradient leaf. Step
//! counts are rank-0 `i32` arrays, which stop increasing at `i32::MAX` instead of overflowing, as
//! [`optax.safe_increment`](https://optax.readthedocs.io/en/latest/api/utilities.html#optax.safe_increment) does.
//!
//! # Example
//!
//! The following example minimizes the sum of `(x - 3)²` over the elements `x` of a vector with [`AdamW`], without
//! weight decay.
//!
//! ```rust
//! use ryft_core::{Array, FillLike, Mul, Reduce, ReductionKind, Sub, differentiate_at};
//! use ryft_optimizers::{AdamW, Optimizer, UpdateArguments, apply_updates};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let optimizer = AdamW::new(0.1).with_weight_decay(0.0)?;
//! let mut parameters = Array::vector(vec![0.0f32, 1.0])?;
//! let mut state = optimizer.initialize(&parameters)?;
//! for _ in 0..200 {
//!     let (_, gradients) = differentiate_at(parameters.clone()).value_and_gradient(|parameters| {
//!         let residuals = Sub::sub(&parameters, &parameters.fill_like(3.0f32)?)?;
//!         Mul::mul(&residuals, &residuals)?.reduce(&[0], ReductionKind::Sum)
//!     })?;
//!     let (updates, next_state) = optimizer.update(gradients, state, &parameters, &UpdateArguments::new())?;
//!     parameters = apply_updates(parameters, updates)?;
//!     state = next_state;
//! }
//! assert!(parameters.elements::<f32>()?.iter().all(|parameter| (parameter - 3.0).abs() < 0.1));
//! # Ok(())
//! # }
//! ```

pub mod combinators;
mod macros;
pub mod optimizers;
pub mod schedules;
pub mod transformations;
pub mod wrappers;

pub use combinators::{
    Chain, Freeze, LeafPath, Masked, MaskedState, Partition, PartitionState, SelectedLeaves, SelectiveTransform,
};
pub use optimizers::{
    AdaBelief, AdaBeliefState, Adadelta, AdadeltaState, Adafactor, AdafactorState, Adagrad, AdagradState, Adam,
    AdamState, AdamW, AdamWState, Adamax, AdamaxState, Adan, AdanState, Amsgrad, AmsgradState, Dimuon, DimuonState,
    Fromage, Lamb, LambState, Lars, LarsState, Lbfgs, LbfgsState, Lion, LionState, Muon, MuonState, NoisySgd,
    NoisySgdState, NovoGrad, NovoGradState, OptimisticAdam, OptimisticAdamState, OptimisticGradientDescent,
    OptimisticGradientDescentState, PolyakSgd, PolyakSgdState, Prodigy, ProdigyState, Radam, RadamState, RmsProp,
    RmsPropState, Rprop, RpropState, Sgd, SgdState, SignSgd, SignSgdState, Sm3, Sm3State, Yogi, YogiState,
};
pub use schedules::{
    CosineDecaySchedule, CosineOneCycleSchedule, ExponentialDecaySchedule, Interpolation, JoinedSchedule,
    LinearOneCycleSchedule, LinearSchedule, PiecewiseConstantSchedule, PiecewiseInterpolateSchedule,
    PolynomialSchedule, Schedule, SgdrSchedule, WarmupCosineDecaySchedule, WarmupExponentialDecaySchedule,
};
pub use transformations::{
    AdaptiveGradClip, AddDecayedWeights, AddNoise, AddNoiseState, BacktrackingLinesearchInfo, Centralize, Clip,
    ClipByBlockRms, ClipByGlobalNorm, Ema, EmaState, InitialGuessStrategy, KeepParamsNonnegative, LionMode,
    MuonDimensionNumbers, MuonPreconditioning, NewtonSchulzCoefficients, NormalizeByUpdateNorm, PolyakVariant,
    ReduceOnPlateau, ReduceOnPlateauState, Scale, ScaleByAdadelta, ScaleByAdadeltaState, ScaleByAdam, ScaleByAdamState,
    ScaleByAdamax, ScaleByAdan, ScaleByAdanState, ScaleByAmsgrad, ScaleByAmsgradState, ScaleByBacktrackingLinesearch,
    ScaleByBacktrackingLinesearchState, ScaleByBelief, ScaleByBeliefState, ScaleByDimuon, ScaleByDimuonState,
    ScaleByFactoredRms, ScaleByFactoredRmsState, ScaleByLbfgs, ScaleByLbfgsState, ScaleByLion, ScaleByLionState,
    ScaleByMuon, ScaleByMuonState, ScaleByNovograd, ScaleByNovogradState, ScaleByOptimisticGradient,
    ScaleByOptimisticGradientState, ScaleByParamBlockNorm, ScaleByParamBlockRms, ScaleByPolyak, ScaleByRadam,
    ScaleByRms, ScaleByRmsState, ScaleByRprop, ScaleByRpropState, ScaleByRss, ScaleByRssState, ScaleBySchedule,
    ScaleByScheduleState, ScaleBySign, ScaleBySm3, ScaleBySm3State, ScaleByStddev, ScaleByStddevState,
    ScaleByTrustRatio, ScaleByYogi, ScaleByZoomLinesearch, ScaleByZoomLinesearchState, SetToZero, Trace, TraceState,
    ZeroNans, ZeroNansState, ZoomLinesearchInfo,
};
pub use wrappers::{
    ApplyEvery, ApplyEveryState, ApplyIfFinite, ApplyIfFiniteState, ConditionallyMask, ConditionallyMaskState,
    ConditionallyTransform, ConditionallyTransformState, InjectLearningRate, InjectLearningRateState, Lookahead,
    LookaheadParameters, LookaheadState, MultiSteps, MultiStepsState, ScheduleFree, ScheduleFreeAdamW, ScheduleFreeSgd,
    ScheduleFreeState, StepCondition, UpdateSkipCriterion,
};

use ryft_core::{
    Add, ArrayOperations, ArrayType, ComparisonDirection, DataType, Div, Fill, Mul, Parameter, ParameterError,
    ParameterPath, Parameterized, ProgramError, ReductionKind, Rem, Sub,
};
use thiserror::Error;

// TODO(eaplatanios): Review this module.

/// Represents errors raised while configuring optimizers or computing their updates. Errors of the array operations
/// that compute the updates and of the parameter structures that carry them are wrapped transparently.
#[derive(Error, Clone, Debug, PartialEq)]
pub enum OptimizerError {
    #[error(transparent)]
    Program(#[from] ProgramError),

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error("{message}")]
    StructureMismatch { message: String },

    #[error("{message}")]
    InvalidConfiguration { message: String },

    #[error("{message}")]
    MissingArgument { message: String },
}

/// Leaf value bound that every optimizer in this crate requires, so that generic code can state one bound instead of
/// the list of capabilities that optimizers use. It groups the array capabilities of [`ArrayOperations`] over
/// [`ArrayType`] with the ability of the value's [`Domain`](ryft_core::Value::Domain) to [`Fill`] rank-0 `i32`
/// arrays, which optimizers use to create step counts without an exemplar of the same shape. It is implemented
/// automatically for every value that satisfies these bounds (e.g., host [`Array`](ryft_core::Array)s, the tracers of
/// staging and differentiation over them, and the values that compiled functions of backend crates are traced with),
/// and so it must never be implemented manually.
pub trait OptimizerValue: ArrayOperations<Type = ArrayType, Domain: Fill<i32, Self>> {}

impl<P: ArrayOperations<Type = ArrayType>> OptimizerValue for P where P::Domain: Fill<i32, P> {}

/// Pure gradient transformation, which is the analogue of the Optax
/// [`GradientTransformationExtraArgs`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs).
/// [`Self::initialize`] creates the state of the transformation for a parameter structure, and [`Self::update`] maps
/// gradients, the current state, the current parameters, and optional [`UpdateArguments`] to updates and the next
/// state. Neither function mutates the optimizer or its inputs, so an optimizer is a description of an update rule
/// that can be shared, cloned, and reused for several parameter structures. Updates are not applied to the parameters
/// by the optimizer itself. The caller adds them with [`apply_updates`], which keeps the convention of Optax that
/// updates are added (i.e., a gradient descent step produces negative updates).
///
/// # State Threading
///
/// The state is threaded explicitly by the caller, which passes the state returned by one update to the next one. The
/// state is itself a [`Parameterized`] structure over the same leaf type as the parameters, so that it can cross the
/// boundary of a compiled training step as an input and an output, be differentiated around, and be checkpointed
/// leaf by leaf through [`Parameterized::named_parameters`]. Inside a compiled step, the whole update is traced into
/// array operations. In particular, optimizers never read values back to the host and express data dependent behavior
/// (e.g., skipping a step whose gradients are not finite) with staged operations such as selections.
///
/// # State Structure
///
/// The state type [`Self::State`] is generic over the parameter structure `S`, and per-leaf statistics are stored as
/// values of type `S` that have the structure of the parameters. The leaves of a state are therefore named by the
/// paths of the parameters, and a gradient structure that has a different structure than the state (e.g., a vector
/// with another length or a map with other keys) is rejected by [`Self::update`] with
/// [`OptimizerError::StructureMismatch`]. Gradients and parameters must have the same structure as each other, which
/// [`Self::update`] validates in the same way.
pub trait Optimizer<P: Parameter> {
    /// State threaded between updates for the parameter structure `S`. It is a [`Parameterized`] structure so that it
    /// can cross compiled function boundaries and be checkpointed by path. Stateless optimizers use the unit type
    /// `()`, which has no leaves, and stateful ones store their per-leaf statistics as values of type `S`.
    type State<S: Parameterized<P>>: Parameterized<P>;

    /// Creates the initial state of this optimizer for `parameters`. Only the types of the leaves (i.e., their shapes
    /// and data types) and the structure affect the state, and the leaf values are never read.
    ///
    /// # Errors
    ///
    /// Returns an [`OptimizerError`] when the state cannot be created for `parameters` (e.g., when a stateful optimizer
    /// needs at least one leaf to create its step count from), or when the array operations that create it fail.
    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError>;

    /// Transforms `gradients` into updates with the same structure and returns them together with the next state.
    ///
    /// # Parameters
    ///
    ///   - `gradients`: Gradients (or the updates of a previous transformation in a chain) whose structure matches
    ///     that of `parameters`.
    ///   - `state`: State returned by [`Self::initialize`] or by the previous update for the same structure.
    ///   - `parameters`: Current parameters, which some transformations read (e.g., weight decay).
    ///   - `arguments`: Optional arguments that some transformations need (e.g., the value of the objective), which
    ///     [`UpdateArguments::new`] leaves empty.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::StructureMismatch`] when the structures of `gradients`, `state`, and `parameters`
    /// differ, [`OptimizerError::MissingArgument`] when the optimizer needs an argument that `arguments` does not
    /// provide, and other [`OptimizerError`]s when the array operations that compute the updates fail.
    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError>;
}

/// Optional arguments of [`Optimizer::update`] beyond the gradients, the state, and the parameters, which are the
/// analogue of the keyword arguments that Optax passes to a
/// [`GradientTransformationExtraArgs`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs)
/// (i.e., `value`, `grad`, and `value_fn`). Optax differentiates `value_fn` itself where it needs gradients at other
/// parameters (e.g., in its zoom line search), which this crate cannot do for an arbitrary leaf type, so a caller
/// provides the value and the gradients of the objective at other parameters as one more function. Every argument is
/// borrowed and may be absent. Transformations that need an argument read it with [`Self::validate_value`],
/// [`Self::validate_gradients`], [`Self::validate_objective`], or [`Self::validate_objective_with_gradients`], which
/// return [`OptimizerError::MissingArgument`] when it is absent, and transformations that do not need it ignore it.
/// Combinators pass the same arguments to every optimizer that they apply.
pub struct UpdateArguments<'o, P, S> {
    /// Value of the objective at the current parameters, which is a rank-0 array.
    value: Option<&'o P>,

    /// Gradients of the objective at the current parameters, which may differ from the gradients that a transformation
    /// receives (e.g., in a chain, where a transformation receives the updates of the previous one).
    gradients: Option<&'o S>,

    /// Function that evaluates the objective at other parameters (e.g., for line searches) and returns a rank-0 array.
    objective: Option<&'o dyn Fn(&S) -> Result<P, OptimizerError>>,

    /// Function that evaluates the objective and its gradients at other parameters (e.g., for line searches that check
    /// the curvature condition) and returns a rank-0 array and gradients with the structure of the parameters.
    objective_with_gradients: Option<&'o dyn Fn(&S) -> Result<(P, S), OptimizerError>>,
}

impl<'o, P, S> UpdateArguments<'o, P, S> {
    /// Creates new [`UpdateArguments`] without any arguments, which is what optimizers that need no arguments are
    /// updated with.
    #[inline]
    pub fn new() -> Self {
        Self { value: None, gradients: None, objective: None, objective_with_gradients: None }
    }

    /// Returns these arguments with `value` as the value of the objective at the current parameters.
    #[inline]
    pub fn with_value(mut self, value: &'o P) -> Self {
        self.value = Some(value);
        self
    }

    /// Returns these arguments with `gradients` as the gradients of the objective at the current parameters.
    #[inline]
    pub fn with_gradients(mut self, gradients: &'o S) -> Self {
        self.gradients = Some(gradients);
        self
    }

    /// Returns these arguments with `objective` as the function that evaluates the objective at other parameters.
    #[inline]
    pub fn with_objective(mut self, objective: &'o dyn Fn(&S) -> Result<P, OptimizerError>) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Returns these arguments with `objective_with_gradients` as the function that evaluates the objective and its
    /// gradients at other parameters (e.g., the value and the gradients that `differentiate_at(...)
    /// .value_and_gradient(...)` of `ryft-core` returns).
    #[inline]
    pub fn with_objective_with_gradients(
        mut self,
        objective_with_gradients: &'o dyn Fn(&S) -> Result<(P, S), OptimizerError>,
    ) -> Self {
        self.objective_with_gradients = Some(objective_with_gradients);
        self
    }

    /// Returns the value of the objective at the current parameters, if it was provided.
    #[inline]
    pub fn value(&self) -> Option<&'o P> {
        self.value
    }

    /// Returns the gradients of the objective at the current parameters, if they were provided.
    #[inline]
    pub fn gradients(&self) -> Option<&'o S> {
        self.gradients
    }

    /// Returns the function that evaluates the objective at other parameters, if it was provided.
    #[inline]
    pub fn objective(&self) -> Option<&'o dyn Fn(&S) -> Result<P, OptimizerError>> {
        self.objective
    }

    /// Returns the function that evaluates the objective and its gradients at other parameters, if it was provided.
    #[inline]
    pub fn objective_with_gradients(&self) -> Option<&'o dyn Fn(&S) -> Result<(P, S), OptimizerError>> {
        self.objective_with_gradients
    }

    /// Returns the value of the objective at the current parameters, which `optimizer` requires.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::MissingArgument`] naming `optimizer` when the value was not provided.
    pub fn validate_value(&self, optimizer: &str) -> Result<&'o P, OptimizerError> {
        self.value.ok_or_else(|| missing_argument(optimizer, "value"))
    }

    /// Returns the gradients of the objective at the current parameters, which `optimizer` requires.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::MissingArgument`] naming `optimizer` when the gradients were not provided.
    pub fn validate_gradients(&self, optimizer: &str) -> Result<&'o S, OptimizerError> {
        self.gradients.ok_or_else(|| missing_argument(optimizer, "gradients"))
    }

    /// Returns the function that evaluates the objective at other parameters, which `optimizer` requires.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::MissingArgument`] naming `optimizer` when the function was not provided.
    pub fn validate_objective(
        &self,
        optimizer: &str,
    ) -> Result<&'o dyn Fn(&S) -> Result<P, OptimizerError>, OptimizerError> {
        self.objective.ok_or_else(|| missing_argument(optimizer, "objective"))
    }

    /// Returns the function that evaluates the objective and its gradients at other parameters, which `optimizer`
    /// requires.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::MissingArgument`] naming `optimizer` when the function was not provided.
    #[allow(clippy::type_complexity)]
    pub fn validate_objective_with_gradients(
        &self,
        optimizer: &str,
    ) -> Result<&'o dyn Fn(&S) -> Result<(P, S), OptimizerError>, OptimizerError> {
        self.objective_with_gradients.ok_or_else(|| missing_argument(optimizer, "objective_with_gradients"))
    }
}

impl<P, S> Default for UpdateArguments<'_, P, S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Adds `updates` to `parameters` leaf by leaf and returns the updated parameters, which is the analogue of
/// [`optax.apply_updates`](https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.apply_updates). Each
/// parameter and its update are added in the wider of their two data types (i.e., the data type of the update when the
/// data type of the parameter can be promoted to it, and the data type of the parameter otherwise), and the sum is
/// converted to the data type of the parameter, so that the result is rounded once. Updates computed in a wider data
/// type (e.g., `f32` updates of `bf16` parameters) therefore produce parameters of the original data type.
///
/// # Errors
///
/// Returns [`OptimizerError::StructureMismatch`] when `parameters` and `updates` have different structures, and other
/// [`OptimizerError`]s when an addition or a conversion fails.
pub fn apply_updates<P: OptimizerValue, S: Parameterized<P>>(parameters: S, updates: S) -> Result<S, OptimizerError> {
    validate_structure("updates", &parameters.parameter_paths().collect::<Vec<_>>(), &updates)?;
    let structure = parameters.parameter_structure();
    let parameters = parameters
        .into_parameters()
        .zip(updates.into_parameters())
        .map(|(parameter, update)| {
            let parameter_type = parameter.r#type().data_type();
            let update_type = update.r#type().data_type();
            let sum_type = if parameter_type.is_promotable_to(update_type) { update_type } else { parameter_type };
            let sum = Add::add(&convert_to(&parameter, sum_type)?, &convert_to(&update, sum_type)?)?;
            Ok(convert_to(&sum, parameter_type)?)
        })
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(structure, parameters)?)
}

/// Returns the moving average `α · new + (1 - α) · old` of the parameters `new_parameters` and their previous average
/// `old_parameters` with the step size `α`, leaf by leaf, which is the analogue of
/// [`optax.incremental_update`](
/// https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.incremental_update). It tracks an exponential
/// moving average of the parameters (i.e., Polyak averaging), for evaluation or for the target network of temporal
/// difference learning. The average is computed in the statistic data type of every leaf of `old_parameters` (refer
/// to the [crate documentation](crate#numerics)) and returned in its data type.
///
/// # Errors
///
/// Returns [`OptimizerError::StructureMismatch`] when `new_parameters` and `old_parameters` have different structures,
/// [`OptimizerError::InvalidConfiguration`] when `step_size` is not finite, and other [`OptimizerError`]s when the
/// array operations fail.
pub fn incremental_update<P: OptimizerValue, S: Parameterized<P>>(
    new_parameters: &S,
    old_parameters: S,
    step_size: f64,
) -> Result<S, OptimizerError> {
    validate_finite("step_size", step_size)?;
    validate_structure("old_parameters", &new_parameters.parameter_paths().collect::<Vec<_>>(), &old_parameters)?;
    let structure = old_parameters.parameter_structure();
    let averages = new_parameters
        .parameters()
        .zip(old_parameters.into_parameters())
        .map(|(new, old)| {
            let data_type = old.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let new = convert_to(new, statistic_type)?;
            let old = convert_to(&old, statistic_type)?;
            let average = Add::add(
                &Mul::mul(&new.fill_like(step_size)?, &new)?,
                &Mul::mul(&old.fill_like(1.0 - step_size)?, &old)?,
            )?;
            Ok(convert_to(&average, data_type)?)
        })
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(structure, averages)?)
}

/// Returns `new_parameters` when the rank-0 integer step count `steps` is a multiple of `update_period` and
/// `old_parameters` otherwise, which is the analogue of
/// [`optax.periodic_update`](https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.periodic_update). It
/// keeps a slow copy of the parameters that is refreshed every `update_period` steps (e.g., the target network of
/// temporal difference learning). The choice is made with array operations, so it is staged inside compiled training
/// steps.
///
/// # Errors
///
/// Returns [`OptimizerError::StructureMismatch`] when `new_parameters` and `old_parameters` have different structures,
/// [`OptimizerError::InvalidConfiguration`] when `update_period` is zero or greater than `i32::MAX`, and other
/// [`OptimizerError`]s when the array operations fail.
pub fn periodic_update<P: OptimizerValue, S: Parameterized<P>>(
    new_parameters: S,
    old_parameters: S,
    steps: &P,
    update_period: usize,
) -> Result<S, OptimizerError> {
    if update_period == 0 || i32::try_from(update_period).is_err() {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`update_period` must be in `[1, {}]` but got `{update_period}`", i32::MAX),
        });
    }
    validate_structure("old_parameters", &new_parameters.parameter_paths().collect::<Vec<_>>(), &old_parameters)?;
    let steps = convert_to(steps, DataType::I32)?;
    let remainder = Rem::rem(&steps, &steps.fill_like(update_period as i32)?)?;
    let update = remainder.compare(&remainder.fill_like(0i32)?, ComparisonDirection::Equal)?;
    let structure = new_parameters.parameter_structure();
    let parameters = new_parameters
        .into_parameters()
        .zip(old_parameters.into_parameters())
        .map(|(new, old)| Ok(update.select(&new, &old)?))
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(structure, parameters)?)
}

/// Returns the data type in which optimizers compute and store statistics of leaves of `data_type`, which is `f64` for
/// `f64` leaves and `f32` for all other leaves.
pub(crate) fn statistic_data_type(data_type: DataType) -> DataType {
    if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 }
}

/// Returns `value` converted to `data_type`, or a clone of `value` when it already has that data type, so that no
/// redundant conversion is staged.
pub(crate) fn convert_to<P: OptimizerValue>(value: &P, data_type: DataType) -> Result<P, ProgramError> {
    if value.r#type().data_type() == data_type { Ok(value.clone()) } else { value.convert_element_type(data_type) }
}

/// Returns the axes of `value` in increasing order, which reductions use to reduce `value` to a rank-0 array.
pub(crate) fn all_axes<P: OptimizerValue>(value: &P) -> Vec<usize> {
    (0..value.r#type().rank()).collect()
}

/// Returns a rank-0 `i32` zero step count in the domain of the first leaf of `parameters`, with the memory space and
/// mesh of that leaf. `optimizer` names the optimizer in the error that is returned when `parameters` has no leaves.
pub(crate) fn initial_count<P: OptimizerValue, S: Parameterized<P>>(
    optimizer: &str,
    parameters: &S,
) -> Result<P, OptimizerError> {
    let exemplar = parameters.parameters().next().ok_or_else(|| OptimizerError::StructureMismatch {
        message: format!("`{optimizer}` requires at least one parameter to create its step count"),
    })?;
    zero_count_like(exemplar)
}

/// Returns the step count `count` incremented by one, or `count` itself when it is already `i32::MAX`, which is the
/// analogue of [`optax.safe_increment`](
/// https://optax.readthedocs.io/en/latest/api/utilities.html#optax.safe_increment) for rank-0 `i32` counts. The
/// increment is selected with array operations, so it is staged inside compiled training steps.
pub(crate) fn increment_count<P: OptimizerValue>(count: &P) -> Result<P, OptimizerError> {
    // Signed integer additions wrap, so the wrapped sum at the maximum is computed but never selected.
    let maximum = count.fill_like(i32::MAX)?;
    let incremented = Add::add(count, &count.fill_like(1i32)?)?;
    Ok(count.compare(&maximum, ComparisonDirection::LessThan)?.select(&incremented, &maximum)?)
}

/// Validates that `value`, which the error names `name`, has the leaf paths `expected_paths` in that order, which is
/// how optimizers check that gradients, parameters, and per-leaf statistics have the same structure.
pub(crate) fn validate_structure<P: Parameter, S: Parameterized<P>>(
    name: &str,
    expected_paths: &[ParameterPath],
    value: &S,
) -> Result<(), OptimizerError> {
    validate_leaf_count(name, expected_paths.len(), value.parameter_count())?;
    for (index, (expected, actual)) in expected_paths.iter().zip(value.parameter_paths()).enumerate() {
        if expected != &actual {
            return Err(OptimizerError::StructureMismatch {
                message: format!("expected leaf {index} of `{name}` to have the path `{expected}` but got `{actual}`"),
            });
        }
    }
    Ok(())
}

/// Validates that the structure that `name` describes has the `expected` number of leaves, given that it has `actual`
/// leaves.
pub(crate) fn validate_leaf_count(name: &str, expected: usize, actual: usize) -> Result<(), OptimizerError> {
    if expected != actual {
        return Err(OptimizerError::StructureMismatch {
            message: format!("expected the leaf count of `{name}` to be {expected} but got {actual}"),
        });
    }
    Ok(())
}

/// Validates that hyperparameter `name` has a finite `value`.
pub(crate) fn validate_finite(name: &str, value: f64) -> Result<(), OptimizerError> {
    if !value.is_finite() {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be finite but got `{value}`"),
        });
    }
    Ok(())
}

/// Validates that the hyperparameter `name` has a finite and non-negative `value`.
pub(crate) fn validate_non_negative(name: &str, value: f64) -> Result<(), OptimizerError> {
    if !value.is_finite() || value < 0.0 {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be finite and non-negative but got `{value}`"),
        });
    }
    Ok(())
}

/// Validates that the hyperparameter `name` has a positive and finite `value`.
pub(crate) fn validate_positive(name: &str, value: f64) -> Result<(), OptimizerError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be positive and finite but got `{value}`"),
        });
    }
    Ok(())
}

/// Validates that the decay rate hyperparameter `name` has a `value` in `[0, 1)`.
pub(crate) fn validate_decay_rate(name: &str, value: f64) -> Result<(), OptimizerError> {
    if !(0.0..1.0).contains(&value) {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be in `[0, 1)` but got `{value}`"),
        });
    }
    Ok(())
}

/// Validates that the data type hyperparameter `name`, in which a statistic is stored, is a floating-point
/// `data_type`.
pub(crate) fn validate_floating_point(name: &str, data_type: DataType) -> Result<(), OptimizerError> {
    if !data_type.is_floating_point() {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be a floating-point data type but got `{data_type}`"),
        });
    }
    Ok(())
}

/// Returns a structure with the structure of `parameters` whose leaves have the shapes of the parameters and hold
/// `value`, in the data type `data_type` or, when it is [`None`], in the statistic data type of each parameter (refer
/// to [`statistic_data_type`]). Optimizers create their per-leaf statistics with it.
pub(crate) fn filled_statistics<P: OptimizerValue, S: Parameterized<P>>(
    parameters: &S,
    value: f64,
    data_type: Option<DataType>,
) -> Result<S, OptimizerError> {
    let statistics = parameters
        .parameters()
        .map(|parameter| {
            let data_type = data_type.unwrap_or_else(|| statistic_data_type(parameter.r#type().data_type()));
            Ok(convert_to(parameter, data_type)?.fill_like(value)?)
        })
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(parameters.parameter_structure(), statistics)?)
}

/// Returns the exponential moving average update `(1 - decay) · value + decay · moment`, which is the operation order
/// of `optax.tree.update_moment` (whose `value` is a gradient for first moments and a squared gradient for second
/// moments). Both arrays must have the same data type.
pub(crate) fn update_moment<P: OptimizerValue>(value: &P, moment: &P, decay: f64) -> Result<P, OptimizerError> {
    Ok(Add::add(&Mul::mul(&value.fill_like(1.0 - decay)?, value)?, &Mul::mul(&value.fill_like(decay)?, moment)?)?)
}

/// Returns the rank-0 bias correction `1 - decayᵗ` of exponential moving averages with `decay` after `t` updates,
/// where `step` is the rank-0 `f32` step count `t`, which is the denominator of `optax.tree.bias_correction`.
pub(crate) fn bias_correction<P: OptimizerValue>(step: &P, decay: f64) -> Result<P, OptimizerError> {
    Ok(Sub::sub(&step.fill_like(1f64)?, &step.fill_like(decay)?.pow(step)?)?)
}

/// Returns `moment` divided by the rank-0 bias `correction` (refer to [`bias_correction`]), which is converted to the
/// data type of `moment` first, as `optax.tree.bias_correction` does.
pub(crate) fn bias_corrected<P: OptimizerValue>(moment: &P, correction: &P) -> Result<P, OptimizerError> {
    Ok(Div::div(moment, &convert_to(correction, moment.r#type().data_type())?)?)
}

/// Returns the rank-0 inner product `Σᵢ ⟨leftᵢ, rightᵢ⟩` of the leaves `left` and `right` (which have the same
/// shapes in the same order), accumulated in `f32`, or in `f64` when any leaf is `f64`, which is
/// `optax.tree.real(optax.tree.vdot(left, right))` for real leaves. `name` names the inner product in the error that
/// is returned when there are no leaves.
pub(crate) fn inner_product<P: OptimizerValue>(name: &str, left: &[P], right: &[P]) -> Result<P, OptimizerError> {
    validate_leaf_count(name, left.len(), right.len())?;
    let statistic_type = if left.iter().chain(right).any(|leaf| leaf.r#type().data_type() == DataType::F64) {
        DataType::F64
    } else {
        DataType::F32
    };
    let mut product: Option<P> = None;
    for (left, right) in left.iter().zip(right) {
        let left = convert_to(left, statistic_type)?;
        let leaf_product =
            Mul::mul(&left, &convert_to(right, statistic_type)?)?.reduce(&all_axes(&left), ReductionKind::Sum)?;
        product = Some(match product {
            None => leaf_product,
            Some(product) => Add::add(&product, &leaf_product)?,
        });
    }
    product.ok_or_else(|| OptimizerError::StructureMismatch { message: format!("`{name}` requires at least one leaf") })
}

/// Returns a rank-0 `i32` zero in the domain of `exemplar`, with its memory space and mesh, which optimizers compare
/// with itself to create rank-0 boolean flags without another bound on the leaf values.
pub(crate) fn zero_count_like<P: OptimizerValue>(exemplar: &P) -> Result<P, OptimizerError> {
    let r#type = exemplar.r#type().scalar_like().map_err(ProgramError::from)?.with_data_type(DataType::I32);
    Ok(exemplar.domain().fill(&r#type, 0i32)?)
}

/// Returns the [`OptimizerError::MissingArgument`] error for the update argument `argument` that `optimizer` requires.
fn missing_argument(optimizer: &str, argument: &str) -> OptimizerError {
    OptimizerError::MissingArgument {
        message: format!("`{optimizer}` requires the `{argument}` update argument, which was not provided"),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{Array, FillLike, Reduce, Typed, bf16};

    use super::*;

    /// Returns the `f32` elements of the updates that `optimizer` computes from its initial state for each gradient of
    /// `gradients` in order, where the parameters are the single `f32` vector leaf `parameters`, which stays fixed,
    /// and every gradient is a vector leaf of the same length. Transformations are tested with it on the values that
    /// their update rules give for a few steps.
    pub(crate) fn vector_updates<O: Optimizer<Array>>(
        optimizer: &O,
        parameters: &[f32],
        gradients: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        let parameters = vec![Array::vector(parameters.to_vec()).unwrap()];
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for gradient in gradients {
            let gradients = vec![Array::vector(gradient.to_vec()).unwrap()];
            let (step_updates, next_state) =
                optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(step_updates[0].elements::<f32>().unwrap());
            state = next_state;
        }
        updates
    }

    /// Returns the data types of the leaves of the state that `optimizer` creates for the single `f64` vector leaf
    /// `[1, 2]`, and those of the state after one update with the `f32` gradients `[0.5, -0.25]`. Tests use it to
    /// check that states keep their data types across updates, which compiled training steps require.
    pub(crate) fn state_data_types<O: Optimizer<Array>>(optimizer: &O) -> (Vec<DataType>, Vec<DataType>) {
        let parameters = vec![Array::vector(vec![1.0f64, 2.0]).unwrap()];
        let gradients = vec![Array::vector(vec![0.5f32, -0.25]).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        let initial = state.parameters().map(|leaf| leaf.r#type().data_type()).collect();
        let (_, state) = optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        (initial, state.parameters().map(|leaf| leaf.r#type().data_type()).collect())
    }

    /// Returns the leaf paths of the state that `optimizer` creates for parameters with the leaves `bias` (a scalar)
    /// and `weights` (a vector of two elements), so that tests can check that statistics are named by the paths of
    /// the parameters.
    pub(crate) fn state_paths<O: Optimizer<Array>>(optimizer: &O) -> Vec<String> {
        let parameters = BTreeMap::from([
            ("bias", Array::scalar(1.0f32).unwrap()),
            ("weights", Array::vector(vec![2.0f32, 3.0]).unwrap()),
        ]);
        optimizer
            .initialize(&parameters)
            .unwrap()
            .named_parameters()
            .map(|(path, _)| path.to_string())
            .collect()
    }

    /// Stateless test transformation that reads every [`UpdateArguments`] field, which tests use to check that
    /// combinators and wrappers forward their arguments. The update of every leaf is the sum of the leaf of the
    /// argument gradients at the same position, the argument value, and the objective at the parameters, so every field
    /// affects the result. Leaves must be rank-0.
    pub(crate) struct ArgumentProbe;

    impl<P: OptimizerValue> Optimizer<P> for ArgumentProbe {
        type State<S: Parameterized<P>> = ();

        fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
            let _ = parameters;
            Ok(())
        }

        fn update<S: Parameterized<P>>(
            &self,
            gradients: S,
            state: Self::State<S>,
            parameters: &S,
            arguments: &UpdateArguments<'_, P, S>,
        ) -> Result<(S, Self::State<S>), OptimizerError> {
            let value = arguments.validate_value("ArgumentProbe")?;
            let argument_gradients = arguments.validate_gradients("ArgumentProbe")?;
            let objective = arguments.validate_objective("ArgumentProbe")?(parameters)?;
            let offset = Add::add(value, &objective)?;
            let structure = gradients.parameter_structure();
            let updates = argument_gradients
                .parameters()
                .map(|gradient| Ok(Add::add(gradient, &offset)?))
                .collect::<Result<Vec<_>, OptimizerError>>()?;
            Ok((S::from_parameters(structure, updates)?, state))
        }
    }

    /// Initial leaves of the parameters of the Optax parity tests (refer to [`parity_parameters`]), which are the
    /// vector `bias` with three elements followed by the row-major 2 × 3 matrix `weights`.
    pub(crate) const PARITY_PARAMETERS: [f32; 9] = [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.5, 0.75];

    /// Gradients of the four steps of the Optax parity tests, in the leaf order of [`PARITY_PARAMETERS`]. They differ
    /// between steps, include negative values, and most elements change sign between some pair of steps.
    pub(crate) const PARITY_GRADIENTS: [[f32; 9]; 4] = [
        [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
        [-0.05, 0.25, 0.1, 0.2, 0.3, -0.4, 0.1, -0.2, 0.35],
        [0.3, -0.1, -0.2, -0.15, 0.05, 0.45, 0.25, 0.4, -0.1],
        [0.2, 0.15, -0.35, 0.1, -0.3, 0.05, -0.45, 0.2, 0.3],
    ];

    /// Curvatures `c` of the quadratic objective `f(x) = Σᵢ ½ · cᵢ · (xᵢ - 1)²` of the Optax parity tests (refer to
    /// [`parity_quadratic`]), in the leaf order of [`PARITY_PARAMETERS`].
    pub(crate) const PARITY_CURVATURES: [f32; 9] = [1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

    /// Parameter structure of the Optax parity tests, whose leaves are `bias` (a vector) and `weights` (a matrix), in
    /// this order, which is also the order in which `jax.tree.leaves` visits the keys of the same Python `dict`.
    pub(crate) type ParityParameters = BTreeMap<&'static str, Array>;

    /// Returns the parity parameter structure whose nine leaf elements are `elements`, in the order of
    /// [`PARITY_PARAMETERS`].
    pub(crate) fn parity_parameters(elements: &[f32]) -> ParityParameters {
        assert_eq!(elements.len(), 9);
        BTreeMap::from([
            ("bias", Array::vector(elements[..3].to_vec()).unwrap()),
            ("weights", Array::matrix(2, 3, elements[3..].to_vec()).unwrap()),
        ])
    }

    /// Returns the `f32` elements of the leaves of `parameters` (a parity parameter structure), in the order of
    /// [`PARITY_PARAMETERS`].
    pub(crate) fn parity_elements(parameters: &ParityParameters) -> Vec<f32> {
        parameters.values().flat_map(|leaf| leaf.elements::<f32>().unwrap()).collect()
    }

    /// Returns the updates that `optimizer` computes for the [`PARITY_GRADIENTS`] in order, starting from its initial
    /// state for the [`PARITY_PARAMETERS`], where each update is applied with [`apply_updates`] before the next step
    /// (so that weight decay and other rules that read the parameters see them change). This is the loop of the Optax
    /// generator scripts, which call `optax.apply_updates` in the same way.
    pub(crate) fn parity_updates<O: Optimizer<Array>>(optimizer: &O) -> Vec<Vec<f32>> {
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for gradients in &PARITY_GRADIENTS {
            let (step_updates, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        updates
    }

    /// Returns the value and the gradients of the quadratic objective `f(x) = Σᵢ ½ · cᵢ · (xᵢ - 1)²` of the Optax
    /// parity tests at `parameters`, where `c` is [`PARITY_CURVATURES`].
    pub(crate) fn parity_quadratic(parameters: &ParityParameters) -> Result<(Array, ParityParameters), OptimizerError> {
        let curvatures = parity_parameters(&PARITY_CURVATURES);
        let mut value = Array::scalar(0.0f32)?;
        let mut gradients = Vec::new();
        for (leaf, curvature) in parameters.values().zip(curvatures.values()) {
            let residuals = Sub::sub(leaf, &leaf.fill_like(1.0f32)?)?;
            let leaf_gradients = Mul::mul(curvature, &residuals)?;
            let leaf_value = Mul::mul(&leaf_gradients, &residuals)?.reduce(&all_axes(leaf), ReductionKind::Sum)?;
            value = Add::add(&value, &Mul::mul(&leaf_value, &Array::scalar(0.5f32)?)?)?;
            gradients.push(leaf_gradients);
        }
        Ok((value, ParityParameters::from_parameters(parameters.parameter_structure(), gradients)?))
    }

    /// Returns the updates of `steps` steps of `optimizer` on the quadratic objective of [`parity_quadratic`] from
    /// the [`PARITY_PARAMETERS`], where every step passes the gradients at the current parameters, together with the
    /// value, the gradients, the objective, and the objective with gradients as [`UpdateArguments`], and applies the
    /// updates with [`apply_updates`].
    pub(crate) fn parity_quadratic_updates<O: Optimizer<Array>>(optimizer: &O, steps: usize) -> Vec<Vec<f32>> {
        let objective =
            |parameters: &ParityParameters| -> Result<Array, OptimizerError> { Ok(parity_quadratic(parameters)?.0) };
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for _ in 0..steps {
            let (value, gradients) = parity_quadratic(&parameters).unwrap();
            let arguments = UpdateArguments::new()
                .with_value(&value)
                .with_gradients(&gradients)
                .with_objective(&objective)
                .with_objective_with_gradients(&parity_quadratic);
            let (step_updates, next_state) =
                optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        updates
    }

    /// Default relative tolerance of the Optax parity tests, which allows for the different association of `f32`
    /// operations in XLA and in this crate over a few steps.
    pub(crate) const PARITY_RELATIVE_TOLERANCE: f32 = 1e-5;

    /// Absolute tolerance floor of the Optax parity tests, which applies to values near zero.
    pub(crate) const PARITY_ABSOLUTE_TOLERANCE: f32 = 1e-6;

    /// Asserts that every element of `actual` matches the element of `expected` (values returned by Optax) at the same
    /// position within [`PARITY_RELATIVE_TOLERANCE`] (refer to [`assert_parity_with_tolerance`]).
    #[track_caller]
    pub(crate) fn assert_parity<A: AsRef<[f32]>, E: AsRef<[f32]>>(actual: &[A], expected: &[E]) {
        assert_parity_with_tolerance(actual, expected, PARITY_RELATIVE_TOLERANCE);
    }

    /// Asserts that `actual` and `expected` (values returned by Optax) have the same shape and that every element of
    /// `actual` is within `relative_tolerance · |expected| + PARITY_ABSOLUTE_TOLERANCE` of the corresponding element
    /// of `expected`. Non-finite values must match exactly (with every `NaN` matching every other `NaN`).
    #[track_caller]
    pub(crate) fn assert_parity_with_tolerance<A: AsRef<[f32]>, E: AsRef<[f32]>>(
        actual: &[A],
        expected: &[E],
        relative_tolerance: f32,
    ) {
        assert_eq!(actual.len(), expected.len(), "expected {} rows but got {}", expected.len(), actual.len());
        for (row, (actual_row, expected_row)) in actual.iter().zip(expected).enumerate() {
            let (actual_row, expected_row) = (actual_row.as_ref(), expected_row.as_ref());
            assert_eq!(actual_row.len(), expected_row.len(), "row {row} has a different length");
            for (column, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
                let matches = if expected.is_finite() {
                    (actual - expected).abs() <= relative_tolerance * expected.abs() + PARITY_ABSOLUTE_TOLERANCE
                } else {
                    (expected.is_nan() && actual.is_nan()) || actual == expected
                };
                assert!(
                    matches,
                    "element {column} of row {row} differs from Optax\n  actual:   {actual_row:?}\n  expected: \
                     {expected_row:?}",
                );
            }
        }
    }

    #[test]
    fn test_update_arguments_new() {
        let arguments = UpdateArguments::<Array, Vec<Array>>::new();
        assert!(arguments.value().is_none());
        assert!(arguments.gradients().is_none());
        assert!(arguments.objective().is_none());
        assert!(arguments.objective_with_gradients().is_none());
        assert!(UpdateArguments::<Array, Vec<Array>>::default().value().is_none());
    }

    #[test]
    fn test_update_arguments_with_value() {
        let value = Array::scalar(2.5f32).unwrap();
        let arguments = UpdateArguments::<Array, Vec<Array>>::new().with_value(&value);
        assert_eq!(arguments.value(), Some(&value));
        assert_eq!(arguments.validate_value("Probe"), Ok(&value));
    }

    #[test]
    fn test_update_arguments_with_gradients() {
        let gradients = vec![Array::scalar(1.5f32).unwrap()];
        let arguments = UpdateArguments::<Array, Vec<Array>>::new().with_gradients(&gradients);
        assert_eq!(arguments.gradients(), Some(&gradients));
        assert_eq!(arguments.validate_gradients("Probe"), Ok(&gradients));
    }

    #[test]
    fn test_update_arguments_with_objective() {
        let objective = |parameters: &Vec<Array>| -> Result<Array, OptimizerError> { Ok(parameters[0].clone()) };
        let arguments = UpdateArguments::<Array, Vec<Array>>::new().with_objective(&objective);
        let parameters = vec![Array::scalar(3.0f32).unwrap()];
        assert_eq!(arguments.objective().unwrap()(&parameters), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(arguments.validate_objective("Probe").unwrap()(&parameters), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_update_arguments_with_objective_with_gradients() {
        let objective = |parameters: &Vec<Array>| -> Result<(Array, Vec<Array>), OptimizerError> {
            Ok((parameters[0].clone(), parameters.clone()))
        };
        let arguments = UpdateArguments::<Array, Vec<Array>>::new().with_objective_with_gradients(&objective);
        let parameters = vec![Array::scalar(3.0f32).unwrap()];
        assert_eq!(
            arguments.objective_with_gradients().unwrap()(&parameters),
            Ok((parameters[0].clone(), parameters.clone())),
        );
        assert_eq!(
            arguments.validate_objective_with_gradients("Probe").unwrap()(&parameters),
            Ok((parameters[0].clone(), parameters.clone())),
        );
    }

    #[test]
    fn test_update_arguments_validate_missing() {
        let arguments = UpdateArguments::<Array, Vec<Array>>::new();
        assert!(matches!(
            arguments.validate_value("Probe"),
            Err(OptimizerError::MissingArgument { message })
                if message == "`Probe` requires the `value` update argument, which was not provided",
        ));
        assert!(matches!(
            arguments.validate_gradients("Probe"),
            Err(OptimizerError::MissingArgument { message })
                if message == "`Probe` requires the `gradients` update argument, which was not provided",
        ));
        assert!(matches!(
            arguments.validate_objective("Probe"),
            Err(OptimizerError::MissingArgument { message })
                if message == "`Probe` requires the `objective` update argument, which was not provided",
        ));
        assert!(matches!(
            arguments.validate_objective_with_gradients("Probe"),
            Err(OptimizerError::MissingArgument { message })
                if message == "`Probe` requires the `objective_with_gradients` update argument, which was not provided",
        ));
    }

    #[test]
    fn test_apply_updates_optax_parity() {
        // Optax 9ecde04b: `optax.apply_updates(PARITY_PARAMETERS, PARITY_GRADIENTS[0])`.
        const EXPECTED: [f32; 9] = [0.6, -1.2, 2.3, 1.4, -0.6, 0.45, -2.3, 2.0, 0.14999998];
        let parameters = parity_parameters(&PARITY_PARAMETERS);
        let updated = apply_updates(parameters, parity_parameters(&PARITY_GRADIENTS[0])).unwrap();
        assert_parity(&[parity_elements(&updated)], &[EXPECTED]);
    }

    #[test]
    fn test_apply_updates() {
        // The sums are returned in the data types of the parameters.
        let parameters = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(1.5)).unwrap());
        let updates = (Array::vector(vec![0.5f32, -0.25]).unwrap(), Array::scalar(0.25f32).unwrap());
        let (first, second) = apply_updates(parameters.clone(), updates).unwrap();
        assert_eq!(first, Array::vector(vec![1.5f32, 1.75]).unwrap());
        assert_eq!(second, Array::scalar(bf16::from_f32(1.75)).unwrap());

        // A `bf16` parameter and an `f32` update are added in `f32` and rounded once. Rounding the update `2⁻⁸ + 2⁻²⁰`
        // to `bf16` first gives `2⁻⁸`, whose sum with `1` is a tie that rounds to `1`, while the `f32` sum rounds up.
        let parameters = vec![Array::scalar(bf16::from_f32(1.0)).unwrap()];
        let updates = vec![Array::scalar(2.0f32.powi(-8) + 2.0f32.powi(-20)).unwrap()];
        assert_eq!(apply_updates(parameters, updates), Ok(vec![Array::scalar(bf16::from_f32(1.0078125)).unwrap()]));

        // Structures with different numbers of leaves or different leaf paths are rejected.
        assert!(matches!(
            apply_updates(vec![Array::scalar(1.0f32).unwrap()], Vec::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `updates` to be 1 but got 0",
        ));
        assert!(matches!(
            apply_updates(
                BTreeMap::from([("a", Array::scalar(1.0f32).unwrap())]),
                BTreeMap::from([("b", Array::scalar(1.0f32).unwrap())]),
            ),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected leaf 0 of `updates` to have the path `$[\"a\"]` but got `$[\"b\"]`",
        ));
    }

    #[test]
    fn test_incremental_update() {
        // With `α = 0.25`, the averages are `0.25 · 4 + 0.75 · 0 = 1` and then `0.25 · 4 + 0.75 · 1 = 1.75`, and the
        // average of a `bf16` leaf stays `bf16`.
        let new = (Array::vector(vec![4.0f32, 8.0]).unwrap(), Array::scalar(bf16::from_f32(4.0)).unwrap());
        let old = (Array::vector(vec![0.0f32, 0.0]).unwrap(), Array::scalar(bf16::from_f32(0.0)).unwrap());
        let average = incremental_update(&new, old, 0.25).unwrap();
        assert_eq!(average, (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(1.0)).unwrap()));
        let average = incremental_update(&new, average, 0.25).unwrap();
        assert_eq!(average, (Array::vector(vec![1.75f32, 3.5]).unwrap(), Array::scalar(bf16::from_f32(1.75)).unwrap()));
        assert!(matches!(
            incremental_update(&vec![Array::scalar(1.0f32).unwrap()], Vec::new(), 0.5),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `old_parameters` to be 1 but got 0",
        ));
        assert!(matches!(
            incremental_update(&Vec::<Array>::new(), Vec::new(), f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`step_size` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_periodic_update() {
        // The new parameters replace the old ones at the steps `0` and `2` of the period `2` only.
        let new = vec![Array::scalar(1.0f32).unwrap()];
        let old = vec![Array::scalar(0.0f32).unwrap()];
        assert_eq!(periodic_update(new.clone(), old.clone(), &Array::scalar(0i32).unwrap(), 2), Ok(new.clone()));
        assert_eq!(periodic_update(new.clone(), old.clone(), &Array::scalar(1i32).unwrap(), 2), Ok(old.clone()));
        assert_eq!(periodic_update(new.clone(), old.clone(), &Array::scalar(2i32).unwrap(), 2), Ok(new.clone()));
        assert!(matches!(
            periodic_update(new, old, &Array::scalar(0i32).unwrap(), 0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`update_period` must be in `[1, 2147483647]` but got `0`",
        ));
    }

    #[test]
    fn test_incremental_update_optax_parity() {
        // Optax 9ecde04b: `optax.incremental_update(GRADIENTS[0], old, 0.25)` twice, starting from `PARAMETERS`.
        const EXPECTED: [[f32; 9]; 2] = [
            [0.4, -0.8, 1.575, 0.85, -0.4, 0.2375, -1.575, 1.25, 0.4125],
            [0.32500002, -0.65000004, 1.2562501, 0.7375001, -0.32500002, 0.22812499, -1.2562501, 1.0625, 0.15937498],
        ];
        let new = parity_parameters(&PARITY_GRADIENTS[0]);
        let first = incremental_update(&new, parity_parameters(&PARITY_PARAMETERS), 0.25).unwrap();
        let second = incremental_update(&new, first.clone(), 0.25).unwrap();
        assert_parity(&[parity_elements(&first), parity_elements(&second)], &EXPECTED);
    }

    #[test]
    fn test_periodic_update_optax_parity() {
        // Optax 9ecde04b: `optax.periodic_update(GRADIENTS[0], PARAMETERS, step, 3)` for the steps `0` to `6`.
        const EXPECTED: [[f32; 9]; 7] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.5, 0.75],
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.5, 0.75],
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.5, 0.75],
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.5, 0.75],
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
        ];
        let new = parity_parameters(&PARITY_GRADIENTS[0]);
        let old = parity_parameters(&PARITY_PARAMETERS);
        let updated = (0..7)
            .map(|step| {
                let step = Array::scalar(step).unwrap();
                parity_elements(&periodic_update(new.clone(), old.clone(), &step, 3).unwrap())
            })
            .collect::<Vec<_>>();
        assert_parity(&updated, &EXPECTED);
    }

    #[test]
    fn test_initial_count() {
        let parameters = (Array::vector(vec![1.0f64, 2.0]).unwrap(), Array::scalar(3.0f32).unwrap());
        assert_eq!(initial_count("Probe", &parameters), Ok(Array::scalar(0i32).unwrap()));
        assert!(matches!(
            initial_count("Probe", &Vec::<Array>::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "`Probe` requires at least one parameter to create its step count",
        ));
    }

    #[test]
    fn test_increment_count() {
        assert_eq!(increment_count(&Array::scalar(0i32).unwrap()), Ok(Array::scalar(1i32).unwrap()));
        assert_eq!(increment_count(&Array::scalar(i32::MAX - 1).unwrap()), Ok(Array::scalar(i32::MAX).unwrap()));

        // Counts saturate at `i32::MAX` instead of wrapping to `i32::MIN`.
        assert_eq!(increment_count(&Array::scalar(i32::MAX).unwrap()), Ok(Array::scalar(i32::MAX).unwrap()));
    }

    #[test]
    fn test_validate_structure() {
        let expected = vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()];
        let paths = expected.parameter_paths().collect::<Vec<_>>();
        assert_eq!(validate_structure("gradients", &paths, &expected), Ok(()));
        assert!(matches!(
            validate_structure("gradients", &paths, &vec![Array::scalar(1.0f32).unwrap()]),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `gradients` to be 2 but got 1",
        ));
    }
}
