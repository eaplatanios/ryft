use crate::operations::math::{Max, Min};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Value-level elementwise clamp capability, restricting each element of a value to the inclusive `[lower, upper]`
/// interval. [`Clamp`] is not a primitive operation: it is provided for every value that supports [`Max`] and
/// [`Min`] as the composition `max(lower, min(x, upper))`, which is exactly how
/// [StableHLO defines `clamp`](https://openxla.org/stablehlo/spec#clamp). The composition inherits the primitives'
/// semantics: operands promote to a common real numeric element type and broadcast, NaNs propagate, and the tangent
/// follows the clamped value (so gradients are `1` strictly inside the interval and `0` outside it).
pub trait Clamp: Sized {
    /// Clamps this value elementwise to the inclusive `[lower, upper]` interval, returning a
    /// [`ProgramError`] if something goes wrong.
    ///
    /// # Parameters
    ///
    ///   - `lower`: Inclusive elementwise lower bound.
    ///   - `upper`: Inclusive elementwise upper bound.
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError>;
}

impl<V: Max + Min> Clamp for V {
    #[inline]
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError> {
        self.min(upper)?.max(lower)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::differentiation::value_and_gradient;
    use crate::operations::constants::OneLike;

    use super::*;

    /// Clamps `x` elementwise to the `[-1, 1]` interval, staging the bounds from `x` itself so that the helper works
    /// for both eager values and tracers.
    fn clamp_to_unit_interval<V: Clone + Clamp + OneLike + std::ops::Neg<Output = V>>(x: V) -> Result<V, ProgramError> {
        let upper = x.one_like();
        let lower = -upper.clone();
        x.clamp(&lower, &upper)
    }

    #[test]
    fn test_clamp() {
        let lower = Array::scalar(-1.0f64);
        let upper = Array::scalar(1.0f64);
        assert_eq!(Array::scalar(0.5f64).clamp(&lower, &upper).unwrap(), Array::scalar(0.5f64));
        assert_eq!(Array::scalar(-2.5f64).clamp(&lower, &upper).unwrap(), Array::scalar(-1.0f64));
        assert_eq!(Array::scalar(2.5f64).clamp(&lower, &upper).unwrap(), Array::scalar(1.0f64));
        assert_eq!(Array::scalar(7i32).clamp(&Array::scalar(0i32), &Array::scalar(5i32)).unwrap(), Array::scalar(5i32),);

        assert_eq!(
            Array::vector(vec![-2.0, 0.5, 3.0]).clamp(&Array::scalar(-1.0), &Array::scalar(1.0)).unwrap(),
            Array::vector(vec![-1.0, 0.5, 1.0]),
        );
    }

    #[test]
    fn test_clamp_differentiation() {
        // The gradient follows the clamped value: `1` strictly inside the interval and `0` outside it.
        let (value, gradient) = value_and_gradient(clamp_to_unit_interval, Array::scalar(0.5)).unwrap();
        assert_eq!(value.to_f64s(), vec![0.5]);
        assert_eq!(gradient.to_f64s(), vec![1.0]);
        let (value, gradient) = value_and_gradient(clamp_to_unit_interval, Array::scalar(2.5)).unwrap();
        assert_eq!(value.to_f64s(), vec![1.0]);
        assert_eq!(gradient.to_f64s(), vec![0.0]);
        let (value, gradient) = value_and_gradient(clamp_to_unit_interval, Array::scalar(-2.5)).unwrap();
        assert_eq!(value.to_f64s(), vec![-1.0]);
        assert_eq!(gradient.to_f64s(), vec![0.0]);
    }
}
