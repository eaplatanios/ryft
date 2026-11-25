//! Sequential composition of optimizers.

use ryft_core::{Parameter, Parameterized};

use crate::{Optimizer, OptimizerError, UpdateArguments};

// TODO(eaplatanios): Review this module.

/// Optimizer that applies two optimizers in order, passing the updates of the first one to the second one as its
/// gradients, which is the analogue of
/// [`optax.chain`](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.chain) for two
/// transformations. Its state is the pair of the states of the two optimizers, so the leaves of the state of the first
/// optimizer have paths that start with `$.0` and those of the second one have paths that start with `$.1`. Both
/// optimizers receive the same [`UpdateArguments`]. Longer chains nest pairs, which [`Self::then`] builds (e.g.,
/// `Chain::new(a, b).then(c)` applies `a`, `b`, and then `c`).
#[derive(Clone, Debug, PartialEq)]
pub struct Chain<A, B> {
    /// Optimizer that is applied first.
    first: A,

    /// Optimizer that is applied to the updates of the first optimizer.
    second: B,
}

impl<A, B> Chain<A, B> {
    /// Creates a new [`Chain`] that applies `first` and then `second`.
    #[inline]
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Returns a [`Chain`] that applies this chain and then `next`.
    #[inline]
    pub fn then<N>(self, next: N) -> Chain<Self, N> {
        Chain::new(self, next)
    }

    /// Returns the optimizer that this chain applies first.
    #[inline]
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Returns the optimizer that this chain applies to the updates of its first optimizer.
    #[inline]
    pub fn second(&self) -> &B {
        &self.second
    }
}

impl<P: Parameter, A: Optimizer<P>, B: Optimizer<P>> Optimizer<P> for Chain<A, B> {
    type State<S: Parameterized<P>> = (A::State<S>, B::State<S>);

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok((self.first.initialize(parameters)?, self.second.initialize(parameters)?))
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (first_state, second_state) = state;
        let (updates, first_state) = self.first.update(gradients, first_state, parameters, arguments)?;
        let (updates, second_state) = self.second.update(updates, second_state, parameters, arguments)?;
        Ok((updates, (first_state, second_state)))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Add, Array};

    use crate::tests::{ArgumentProbe, assert_parity, parity_updates};
    use crate::{ClipByGlobalNorm, Scale, ScaleByAdam, ScaleBySchedule};

    use super::*;

    #[test]
    fn test_chain_new() {
        let chain = Chain::new(Scale::new(2.0).unwrap(), ClipByGlobalNorm::new(1.0).unwrap());
        assert_eq!(chain.first(), &Scale::new(2.0).unwrap());
        assert_eq!(chain.second(), &ClipByGlobalNorm::new(1.0).unwrap());
    }

    #[test]
    fn test_chain_then() {
        let chain = Chain::new(Scale::new(2.0).unwrap(), Scale::new(3.0).unwrap()).then(Scale::new(4.0).unwrap());
        assert_eq!(chain.first().second(), &Scale::new(3.0).unwrap());
        assert_eq!(chain.second(), &Scale::new(4.0).unwrap());
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        let state = chain.initialize(&gradients).unwrap();
        assert_eq!(
            chain.update(gradients.clone(), state, &gradients, &UpdateArguments::new()).unwrap().0,
            vec![Array::scalar(24f32).unwrap()],
        );
    }

    #[test]
    fn test_chain_initialize() {
        let chain = Chain::new(ScaleBySchedule::new(1.0f64), Scale::new(2.0).unwrap());
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let (first_state, ()) = chain.initialize(&parameters).unwrap();
        assert_eq!(first_state.count(), &Array::scalar(0i32).unwrap());
    }

    #[test]
    fn test_chain_update() {
        // Scaling before clipping bounds the final norm, whereas clipping before scaling does not, so order matters.
        let gradients = vec![Array::vector(vec![3.0f32, 4.0]).unwrap()];
        let arguments = UpdateArguments::new();
        let scale_then_clip = Chain::new(Scale::new(2.0).unwrap(), ClipByGlobalNorm::new(1.0).unwrap());
        let clip_then_scale = Chain::new(ClipByGlobalNorm::new(1.0).unwrap(), Scale::new(2.0).unwrap());
        assert_eq!(
            scale_then_clip.update(gradients.clone(), ((), ()), &gradients, &arguments).unwrap(),
            (vec![Array::vector(vec![6.0f32 / 10.0, 8.0f32 / 10.0]).unwrap()], ((), ())),
        );
        assert_eq!(
            clip_then_scale.update(gradients.clone(), ((), ()), &gradients, &arguments).unwrap(),
            (vec![Array::vector(vec![2.0 * (3.0f32 / 5.0), 2.0 * (4.0f32 / 5.0)]).unwrap()], ((), ())),
        );
    }

    #[test]
    fn test_chain_update_arguments() {
        // The probe follows a scaling, so it receives the scaled updates as its gradients, but it computes its updates
        // from the arguments that the chain forwards to it: `10 + 0.5 + (1 + 2)` and `20 + 0.5 + (1 + 2)`.
        let chain = Chain::new(Scale::new(2.0).unwrap(), ArgumentProbe);
        let parameters = vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()];
        let gradients = vec![Array::scalar(5.0f32).unwrap(), Array::scalar(6.0f32).unwrap()];
        let value = Array::scalar(0.5f32).unwrap();
        let argument_gradients = vec![Array::scalar(10.0f32).unwrap(), Array::scalar(20.0f32).unwrap()];
        let objective = |parameters: &Vec<Array>| -> Result<Array, OptimizerError> {
            Ok(Add::add(&parameters[0], &parameters[1])?)
        };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&argument_gradients)
            .with_objective(&objective);
        let (updates, ((), ())) = chain.update(gradients.clone(), ((), ()), &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::scalar(13.5f32).unwrap(), Array::scalar(23.5f32).unwrap()]);

        // Each missing argument is reported by the transformation that requires it.
        let arguments = UpdateArguments::new().with_gradients(&argument_gradients).with_objective(&objective);
        assert!(matches!(
            chain.update(gradients.clone(), ((), ()), &parameters, &arguments),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ArgumentProbe` requires the `value` update argument, which was not provided",
        ));
        let arguments = UpdateArguments::new().with_value(&value).with_objective(&objective);
        assert!(matches!(
            chain.update(gradients.clone(), ((), ()), &parameters, &arguments),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ArgumentProbe` requires the `gradients` update argument, which was not provided",
        ));
        let arguments = UpdateArguments::new().with_value(&value).with_gradients(&argument_gradients);
        assert!(matches!(
            chain.update(gradients, ((), ()), &parameters, &arguments),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ArgumentProbe` requires the `objective` update argument, which was not provided",
        ));
    }

    #[test]
    fn test_chain_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.clip_by_global_norm(0.5), optax.scale_by_adam(), optax.scale(-0.1))`.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                -0.099999316,
                0.09999933,
                -0.09999932,
                -0.09999933,
                0.099999316,
                -0.09999933,
                0.09999932,
                -0.09999932,
                0.09999933,
            ],
            [
                -0.012274058,
                -0.031316966,
                -0.09216964,
                -0.09755343,
                -0.056847967,
                0.047509003,
                0.029293472,
                -0.022152413,
                0.0048859003,
            ],
            [
                -0.063851304,
                -0.003044958,
                -0.015720673,
                -0.04576727,
                -0.053128786,
                -0.019313602,
                -0.029164527,
                -0.05594021,
                0.0144913215,
            ],
            [
                -0.07518661,
                -0.026598925,
                0.033803128,
                -0.05034563,
                0.006372053,
                -0.020260513,
                0.030515,
                -0.061875086,
                -0.015267737,
            ],
        ];
        let optimizer =
            Chain::new(ClipByGlobalNorm::new(0.5).unwrap(), ScaleByAdam::new()).then(Scale::new(-0.1).unwrap());
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }
}
