//! Wrappers that change when or how another optimizer is applied, in the style of the
//! [Optax optimizer wrappers](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html).
//!
//!   - **Conditions:** [`ApplyIfFinite`] in [`conditionality`] skips the updates of another optimizer whose gradients
//!     are not finite, and [`ConditionallyTransform`] and [`ConditionallyMask`] apply another optimizer only at the
//!     steps that a [`StepCondition`] selects.
//!   - **Accumulation:** [`MultiSteps`] in [`accumulation`] accumulates gradients over several mini-steps before it
//!     applies another optimizer to their average, and [`ApplyEvery`] accumulates the updates and emits their sum
//!     every few steps.
//!   - **Lookahead:** [`Lookahead`] in [`lookahead`] takes steps with a fast optimizer and periodically moves a set of
//!     slow parameters towards the fast ones, over the pair of parameters of [`LookaheadParameters`].
//!   - **Schedule-free optimization:** [`ScheduleFree`] in [`schedule_free`] wraps a base optimizer, and
//!     [`ScheduleFreeSgd`] and [`ScheduleFreeAdamW`] are its SGD and AdamW variants.
//!     [`ScheduleFreeState::evaluation_parameters`] returns the parameters at which to evaluate the model.
//!   - **Injected hyperparameters:** [`InjectLearningRate`] in [`injection`] applies a learning rate that is stored in
//!     the state, where it can be read and changed between steps.
//!
//! Every data dependent decision is made with array operations (i.e., both outcomes are computed and one is selected),
//! so wrappers are staged rather than evaluated on the host inside compiled training steps. Wrappers pass their
//! [`UpdateArguments`](crate::UpdateArguments) to the optimizer that they wrap.

pub mod accumulation;
pub mod conditionality;
pub mod injection;
pub mod lookahead;
pub mod schedule_free;

pub use accumulation::{ApplyEvery, ApplyEveryState, MultiSteps, MultiStepsState, UpdateSkipCriterion};
pub use conditionality::{
    ApplyIfFinite, ApplyIfFiniteState, ConditionallyMask, ConditionallyMaskState, ConditionallyTransform,
    ConditionallyTransformState, StepCondition,
};
pub use injection::{InjectLearningRate, InjectLearningRateState};
pub use lookahead::{Lookahead, LookaheadParameters, LookaheadState};
pub use schedule_free::{ScheduleFree, ScheduleFreeAdamW, ScheduleFreeSgd, ScheduleFreeState};

use ryft_core::Parameterized;

use crate::{OptimizerError, OptimizerValue, convert_to, validate_leaf_count};

// TODO(eaplatanios): Review this module.

/// Returns the structure `selected` with every leaf replaced by the selection of that leaf and the leaf of `otherwise`
/// at the same position, under the rank-0 boolean `condition` (i.e., the leaf of `selected` where it holds and that of
/// `otherwise` where it does not). Every leaf of `selected` is converted to the data type of the leaf of `otherwise`
/// first, so that a selected state keeps the data types of the state that it replaces. `name` names the structure in
/// the error that is returned when the leaf counts of `selected` and `otherwise` differ. Wrappers use it to select
/// between the results of an inner update and the values that they keep when it is not applied.
fn select_structure<P: OptimizerValue, S: Parameterized<P>>(
    name: &str,
    condition: &P,
    selected: S,
    otherwise: Vec<P>,
) -> Result<S, OptimizerError> {
    validate_leaf_count(name, otherwise.len(), selected.parameter_count())?;
    let structure = selected.parameter_structure();
    let leaves = selected
        .into_parameters()
        .zip(otherwise)
        .map(|(selected, otherwise)| {
            Ok(condition.select(&convert_to(&selected, otherwise.r#type().data_type())?, &otherwise)?)
        })
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(structure, leaves)?)
}
