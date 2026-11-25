//! Ready to use optimizers, which are chains of the transformations of the [`transformations`](crate::transformations)
//! module with named states, in the style of the
//! [Optax optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html).
//!
//! The optimizers are grouped in one submodule per family, and each one merges the Optax aliases that differ only by a
//! flag (e.g., [`Adam`] covers `adam`, `nadam`, `adamw`, and `nadamw`). Every optimizer documents the chain of
//! transformations that it applies, and its state is a named struct with one field per stateful transformation, so that
//! checkpoints refer to the state by stable paths that follow the parameter structure. Learning rates are
//! [`Schedule`](crate::Schedule)s (e.g., an `f64` for a constant learning rate), which are applied with
//! [`ScaleBySchedule::learning_rate`](crate::ScaleBySchedule::learning_rate) and so add a step count to the state.
//!
//!   - **Adam family:** [`Adam`], [`AdamW`], [`Amsgrad`], [`Adamax`], [`Radam`], [`Yogi`], [`AdaBelief`], [`Adan`],
//!     and [`OptimisticAdam`] in [`adam`].
//!   - **Factored statistics:** [`Adafactor`] and [`Sm3`] in [`factorized`].
//!   - **Layer-wise adaptation:** [`Lamb`], [`Lars`], and [`Fromage`] in [`layerwise`].
//!   - **Quasi-Newton methods:** [`Lbfgs`] in [`lbfgs`], with a line search.
//!   - **Matrix orthogonalization:** [`Muon`] in [`muon`] orthogonalizes the momentum of the matrix leaves with
//!     Newton–Schulz iterations and updates all other leaves with AdamW, and [`Dimuon`] orthogonalizes the row and
//!     column conditioned momentum of LoRA factors, with global gradient clipping and a learning rate. It is an
//!     unofficial reconstruction of the optimizer of Thinking Machines, whose formulation is unpublished.
//!   - **Learning rate estimation:** [`Prodigy`] in [`prodigy`] estimates its learning rate from the distance to the
//!     initial parameters.
//!   - **Root mean squares:** [`RmsProp`] (which covers centered RMSProp and momentum), [`Adagrad`], [`Adadelta`],
//!     and [`NovoGrad`] in [`rms`].
//!   - **Stochastic gradient descent:** [`Sgd`] (with optional momentum), [`NoisySgd`], [`OptimisticGradientDescent`],
//!     and [`PolyakSgd`] in [`sgd`].
//!   - **Signs:** [`SignSgd`] (which covers Signum), [`Lion`], and [`Rprop`] in [`sign`].

pub mod adam;
pub mod factorized;
pub mod layerwise;
pub mod lbfgs;
pub mod muon;
pub mod prodigy;
pub mod rms;
pub mod sgd;
pub mod sign;

pub use adam::{
    AdaBelief, AdaBeliefState, Adam, AdamState, AdamW, AdamWState, Adamax, AdamaxState, Adan, AdanState, Amsgrad,
    AmsgradState, OptimisticAdam, OptimisticAdamState, Radam, RadamState, Yogi, YogiState,
};

pub use factorized::{Adafactor, AdafactorState, Sm3, Sm3State};
pub use layerwise::{Fromage, Lamb, LambState, Lars, LarsState};
pub use lbfgs::{Lbfgs, LbfgsState};
pub use muon::{Dimuon, DimuonState, Muon, MuonState};
pub use prodigy::{Prodigy, ProdigyState};
pub use rms::{Adadelta, AdadeltaState, Adagrad, AdagradState, NovoGrad, NovoGradState, RmsProp, RmsPropState};
pub use sgd::{
    NoisySgd, NoisySgdState, OptimisticGradientDescent, OptimisticGradientDescentState, PolyakSgd, PolyakSgdState, Sgd,
    SgdState,
};
pub use sign::{Lion, LionState, Rprop, RpropState, SignSgd, SignSgdState};

use ryft_core::{ParameterPath, Parameterized};

use crate::{AddDecayedWeights, Optimizer, OptimizerError, OptimizerValue, UpdateArguments};

// TODO(eaplatanios): Review this module.

/// Optional weight decay of an optimizer, which is applied only once its rate is set. It keeps its mask separately from
/// whether it is enabled, so that a mask that is set before the rate is not lost.
#[derive(Clone, Debug)]
pub(crate) struct OptionalWeightDecay {
    /// Transformation that holds the rate and the mask of the weight decay.
    weight_decay: AddDecayedWeights,

    /// Whether the rate was set, which enables the weight decay.
    enabled: bool,
}

impl OptionalWeightDecay {
    /// Creates a new disabled [`OptionalWeightDecay`] without a mask.
    pub(crate) fn new() -> Self {
        Self { weight_decay: AddDecayedWeights::new(0.0).unwrap(), enabled: false }
    }

    /// Returns this weight decay enabled with the rate `rate`, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `rate` is not finite.
    pub(crate) fn with_rate(mut self, rate: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(rate)?;
        self.enabled = true;
        Ok(self)
    }

    /// Returns this weight decay with `mask` as the predicate that selects the leaves that decay, whether or not its
    /// rate was set.
    pub(crate) fn with_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`AddDecayedWeights`] transformation of this weight decay, if it is enabled.
    pub(crate) fn transformation(&self) -> Option<&AddDecayedWeights> {
        self.enabled.then_some(&self.weight_decay)
    }

    /// Returns `updates` with the decayed weights added, or `updates` itself when this weight decay is disabled.
    pub(crate) fn apply<P: OptimizerValue, S: Parameterized<P>>(
        &self,
        updates: S,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<S, OptimizerError> {
        match self.transformation() {
            Some(weight_decay) => Ok(weight_decay.update(updates, (), parameters, arguments)?.0),
            None => Ok(updates),
        }
    }
}
