//! Composable gradient transformations, which are the building blocks of optimizers, in the style of the
//! [Optax transformations](https://optax.readthedocs.io/en/latest/api/transformations.html).
//!
//! Each transformation implements [`Optimizer`](crate::Optimizer) and so maps gradients (or the updates of the
//! previous transformation in a chain) to updates, threading its own state. The transformations are grouped in one
//! submodule per family:
//!
//!   - **Adam family:** [`ScaleByAdam`] in [`adam`] rescales gradients by bias corrected estimates of their first and
//!     second moments, and [`ScaleByAmsgrad`], [`ScaleByAdamax`], [`ScaleByRadam`], [`ScaleByYogi`],
//!     [`ScaleByBelief`], and [`ScaleByAdan`] are its variants.
//!   - **Root mean squares:** [`ScaleByRms`], [`ScaleByStddev`], [`ScaleByRss`], [`ScaleByAdadelta`], and
//!     [`ScaleByNovograd`] in [`rms`] divide gradients by running averages or sums of their squares.
//!   - **Factored statistics:** [`ScaleByFactoredRms`] and [`ScaleBySm3`] in [`factorized`] keep second moment
//!     statistics per axis rather than per element.
//!   - **Momentum:** [`Trace`] and [`Ema`] in [`momentum`] accumulate decaying sums or averages of the updates.
//!   - **Signs:** [`ScaleBySign`], [`ScaleByLion`], and [`ScaleByRprop`] in [`sign`] keep only the signs of the
//!     gradients or of their moments.
//!   - **Norms:** [`ScaleByTrustRatio`], [`ScaleByParamBlockNorm`], [`ScaleByParamBlockRms`],
//!     [`NormalizeByUpdateNorm`], and [`Centralize`] in [`normalization`] rescale or center the updates by norms.
//!   - **Clipping:** [`ClipByGlobalNorm`], [`Clip`], [`ClipByBlockRms`], and [`AdaptiveGradClip`] in [`clipping`]
//!     bound the magnitude of the updates.
//!   - **Constraints:** [`ZeroNans`] and [`KeepParamsNonnegative`] in [`constraining`] replace NaNs and keep
//!     parameters non-negative.
//!   - **Matrix orthogonalization:** [`ScaleByMuon`] in [`muon`] orthogonalizes the momentum of parameter matrices with
//!     Newton–Schulz iterations, and [`ScaleByDimuon`] orthogonalizes the row and column conditioned momentum of LoRA
//!     factors, which is an unofficial reconstruction of the optimizer of Thinking Machines, whose formulation is
//!     unpublished.
//!   - **Plateaus:** [`ReduceOnPlateau`] in [`plateau`] scales the updates down when the value of the objective, which
//!     it reads from the [`UpdateArguments`](crate::UpdateArguments), stops improving.
//!   - **Step sizes:** [`ScaleByOptimisticGradient`] in [`optimistic`] extrapolates from the previous gradient, and
//!     [`ScaleByPolyak`] in [`polyak`] scales the updates by the Polyak step size, which needs the value of the
//!     objective.
//!   - **Scaling:** [`Scale`] in [`scaling`] multiplies the updates by a constant, [`ScaleBySchedule`] multiplies
//!     them by the value of a [`Schedule`](crate::Schedule) at the current step (e.g., a negative learning rate, which
//!     is `optax.scale_by_learning_rate`), and [`SetToZero`] replaces them with zeros.
//!   - **Additions:** [`AddDecayedWeights`] in [`adding`] adds a multiple of the parameters to the updates, which is
//!     the decoupled weight decay of AdamW, and [`AddNoise`] in [`noise`] adds annealed Gaussian noise.
//!
//! Transformations are composed with the combinators of the [`combinators`](crate::combinators) module. For example,
//! `Chain::new(ClipByGlobalNorm::new(1.0)?, ScaleByAdam::new()).then(Scale::new(-1e-3)?)` clips the gradients,
//! rescales them with Adam, and scales the result by a negative learning rate, which is the Optax chain
//! `optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-1e-3))`.
//!
//! Stateless transformations use `()` as their state. Stateful ones store a rank-0 `i32` step count and, where they
//! need them, per-leaf statistics as structures shaped like the parameters, as described in the documentation of
//! [`Optimizer`](crate::Optimizer). All of them follow the numerics contract of the
//! [crate documentation](crate#numerics).

pub mod adam;
pub mod adding;
pub mod clipping;
pub mod constraining;
pub mod factorized;
pub mod lbfgs;
pub mod linesearch;
pub mod momentum;
pub mod muon;
pub mod noise;
pub mod normalization;
pub mod optimistic;
pub mod plateau;
pub mod polyak;
pub mod rms;
pub mod scaling;
pub mod sign;

pub use adam::{
    ScaleByAdam, ScaleByAdamState, ScaleByAdamax, ScaleByAdan, ScaleByAdanState, ScaleByAmsgrad, ScaleByAmsgradState,
    ScaleByBelief, ScaleByBeliefState, ScaleByRadam, ScaleByYogi,
};
pub use adding::AddDecayedWeights;
pub use clipping::{AdaptiveGradClip, Clip, ClipByBlockRms, ClipByGlobalNorm};
pub use constraining::{KeepParamsNonnegative, ZeroNans, ZeroNansState};
pub use factorized::{ScaleByFactoredRms, ScaleByFactoredRmsState, ScaleBySm3, ScaleBySm3State};
pub use lbfgs::{ScaleByLbfgs, ScaleByLbfgsState};
pub use linesearch::{
    BacktrackingLinesearchInfo, InitialGuessStrategy, ScaleByBacktrackingLinesearch,
    ScaleByBacktrackingLinesearchState, ScaleByZoomLinesearch, ScaleByZoomLinesearchState, ZoomLinesearchInfo,
};
pub use momentum::{Ema, EmaState, Trace, TraceState};
pub use muon::{
    MuonDimensionNumbers, MuonPreconditioning, NewtonSchulzCoefficients, ScaleByDimuon, ScaleByDimuonState,
    ScaleByMuon, ScaleByMuonState,
};
pub use noise::{AddNoise, AddNoiseState};
pub use normalization::{
    Centralize, NormalizeByUpdateNorm, ScaleByParamBlockNorm, ScaleByParamBlockRms, ScaleByTrustRatio,
};
pub use optimistic::{ScaleByOptimisticGradient, ScaleByOptimisticGradientState};
pub use plateau::{ReduceOnPlateau, ReduceOnPlateauState};
pub use polyak::{PolyakVariant, ScaleByPolyak};
pub use rms::{
    ScaleByAdadelta, ScaleByAdadeltaState, ScaleByNovograd, ScaleByNovogradState, ScaleByRms, ScaleByRmsState,
    ScaleByRss, ScaleByRssState, ScaleByStddev, ScaleByStddevState,
};
pub use scaling::{Scale, ScaleBySchedule, ScaleByScheduleState, SetToZero};
pub use sign::{LionMode, ScaleByLion, ScaleByLionState, ScaleByRprop, ScaleByRpropState, ScaleBySign};

// TODO(eaplatanios): Review this module.
