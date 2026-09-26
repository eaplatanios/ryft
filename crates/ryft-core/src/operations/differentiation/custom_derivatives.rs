use crate::contexts::Context;
use crate::differentiation::DifferentiableType;
use crate::operations::differentiation::custom_jvp::{CustomJvpOperation, custom_jvp};
use crate::operations::differentiation::custom_vjp::{CustomVjpOperation, custom_vjp};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Value};
use crate::tracing::DomainTracer;

// TODO(eaplatanios): Review from here onwards.

/// Builder, returned by [`custom_derivative_at`], that stages a custom derivative rule at a known input tree. Because
/// the builder captures the input before the rule closures are written, those closures infer their tracer parameter
/// types from the input and need no type annotations.
/// [`with_non_differentiated_count`](Self::with_non_differentiated_count) configures the call, and the terminal
/// [`jvp`](Self::jvp) and [`vjp`](Self::vjp) functions stage it as a [`CustomJvpOperation`] or a
/// [`CustomVjpOperation`], respectively. Refer to the documentation of the [`custom_jvp`] and [`custom_vjp`] functions
/// for the semantics of the staged calls.
pub struct CustomDerivativeBuilder<Input> {
    /// Input tree at which the custom derivative rule is staged.
    input: Input,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,
}

impl<Input> CustomDerivativeBuilder<Input> {
    /// Declares the leading `non_differentiated_count` flattened leaves of the input tree as non-differentiated
    /// _plumbing_ inputs, which is the builder counterpart of
    /// [`CustomJvp::with_non_differentiated_count`](crate::CustomJvp::with_non_differentiated_count) and
    /// [`CustomVjp::with_non_differentiated_count`](crate::CustomVjp::with_non_differentiated_count). Refer to the
    /// documentation of the [`custom_jvp`] and [`custom_vjp`] functions for the semantics of non-differentiated
    /// inputs.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Stages a custom-JVP call at the input of this builder and returns its output tree. This is equivalent to
    /// `custom_jvp(primal, jvp).with_non_differentiated_count(count).call(input)`, except that the closures infer their
    /// parameter types from the input. Refer to the documentation of the [`custom_jvp`] function for the semantics of
    /// the staged call.
    ///
    /// # Parameters
    ///
    ///   - `primal`: Closure implementing `f(x) = y`.
    ///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
    ///
    /// # Errors
    ///
    /// Returns the [`ProgramError`]s described in the documentation of [`CustomJvp::call`](crate::CustomJvp::call).
    pub fn jvp<D, V, Outputs, Primal, Jvp>(
        self,
        primal: Primal,
        jvp: Jvp,
    ) -> Result<<Outputs::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        D: Context<Type: DifferentiableType, Value = V>,
        V: Value<Type = D::Type, DispatchDomain = D>,
        D::Operation: From<CustomJvpOperation<D::Type>>,
        Input: Parameterized<V>,
        Input::Family:
            ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<D>>,
        Input::To<DomainTracer<D>>:
            Parameterized<DomainTracer<D>, Family = Input::Family, To<D::Type> = Input::To<D::Type>>,
        Input::To<D::Type>:
            Clone + Parameterized<D::Type, Family = Input::Family, To<DomainTracer<D>> = Input::To<DomainTracer<D>>>,
        Outputs: Parameterized<DomainTracer<D>>,
        Outputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<V>,
        Outputs::To<D::Type>: Parameterized<D::Type, Family = Outputs::Family, To<DomainTracer<D>> = Outputs>,
        Primal: Fn(Input::To<DomainTracer<D>>) -> Result<Outputs, ProgramError>,
        Jvp: Fn(Input::To<DomainTracer<D>>, Input::To<DomainTracer<D>>) -> Result<(Outputs, Outputs), ProgramError>,
    {
        custom_jvp(primal, jvp)
            .with_non_differentiated_count(self.non_differentiated_count)
            .call(self.input)
    }

    /// Stages a custom-VJP call at the input of this builder and returns its output tree. This is equivalent to
    /// `custom_vjp(primal, forward, backward).with_non_differentiated_count(count).call(input)`, except that the
    /// closures infer their parameter types from the input and, for `backward`, from the residuals that `forward`
    /// returns. Refer to the documentation of the [`custom_vjp`] function for the semantics of the staged call.
    ///
    /// # Parameters
    ///
    ///   - `primal`: Closure implementing `f(x) = y` for ordinary evaluation.
    ///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse-mode residual production.
    ///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
    ///
    /// # Errors
    ///
    /// Returns the [`ProgramError`]s described in the documentation of [`CustomVjp::call`](crate::CustomVjp::call).
    pub fn vjp<D, V, Outputs, Residuals, Primal, Forward, Backward>(
        self,
        primal: Primal,
        forward: Forward,
        backward: Backward,
    ) -> Result<<Outputs::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        D: Context<Type: DifferentiableType, Value = V>,
        V: Value<Type = D::Type, DispatchDomain = D>,
        D::Operation: From<CustomVjpOperation<D::Type>>,
        Input: Parameterized<V>,
        Input::Family:
            ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<D>>,
        Input::To<DomainTracer<D>>:
            Parameterized<DomainTracer<D>, Family = Input::Family, To<D::Type> = Input::To<D::Type>>,
        Input::To<D::Type>:
            Clone + Parameterized<D::Type, Family = Input::Family, To<DomainTracer<D>> = Input::To<DomainTracer<D>>>,
        Outputs: Parameterized<DomainTracer<D>>,
        Outputs::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant> + ParameterizedFamily<V>,
        Outputs::To<D::Type>: Clone + Parameterized<D::Type, Family = Outputs::Family, To<DomainTracer<D>> = Outputs>,
        Residuals: Parameterized<DomainTracer<D>>,
        Residuals::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
        Residuals::To<D::Type>: Parameterized<D::Type, Family = Residuals::Family, To<DomainTracer<D>> = Residuals>,
        Primal: Fn(Input::To<DomainTracer<D>>) -> Result<Outputs, ProgramError>,
        Forward: Fn(Input::To<DomainTracer<D>>) -> Result<(Outputs, Residuals), ProgramError>,
        Backward: Fn(Residuals, Outputs) -> Result<Input::To<DomainTracer<D>>, ProgramError>,
    {
        custom_vjp(primal, forward, backward)
            .with_non_differentiated_count(self.non_differentiated_count)
            .call(self.input)
    }
}

/// Creates a [`CustomDerivativeBuilder`] that stages a custom derivative rule at `input`, which is the input-first
/// counterpart of the [`custom_jvp`] and [`custom_vjp`] functions. Those functions build a reusable function from rule
/// closures before any input is known, so their closures must annotate the tracer type of their input. This function
/// instead receives the input before the rule closures, so the closures infer all of their parameter types from it.
/// Prefer it when a rule is applied where it is defined, and prefer [`custom_jvp`] or [`custom_vjp`] when the same
/// rule is called at several sites.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Cos, ProgramError, Sin, custom_derivative_at, differentiate_at};
/// # fn main() -> Result<(), ProgramError> {
/// // A custom JVP rule for `sin` that doubles the true derivative, so that its effect is visible.
/// let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| {
///     custom_derivative_at(x).jvp(
///         |x| Ok(x.sin()?),
///         |x, tangent| {
///             let tangent = x.cos()? * tangent;
///             Ok((x.sin()?, tangent.clone() + tangent))
///         },
///     )
/// })?;
/// assert_eq!(value, Array::scalar(0.5f64.sin())?);
/// assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
///
/// // A custom VJP rule for `sin` that saves `cos(x)` as its residual and doubles the true gradient.
/// let gradient = differentiate_at(Array::scalar(0.5f64)?).gradient(|x| {
///     custom_derivative_at(x)
///         .vjp(
///             |x| Ok(x.sin()?),
///             |x| Ok((x.sin()?, x.cos()?)),
///             |cosine, cotangent| {
///                 let gradient = cosine * cotangent;
///                 Ok(gradient.clone() + gradient)
///             },
///         )
///         .unwrap()
/// })?;
/// assert_eq!(gradient, Array::scalar(2.0 * 0.5f64.cos())?);
/// # Ok(())
/// # }
/// ```
///
/// # Parameters
///
///   - `input`: [`Parameterized`] tree of values at which the custom derivative rule is staged.
#[inline]
pub fn custom_derivative_at<Input>(input: Input) -> CustomDerivativeBuilder<Input> {
    CustomDerivativeBuilder { input, non_differentiated_count: 0 }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrValue, ArrayReference};
    use crate::differentiation::{CotangentDestination, CotangentSeed, differentiate_at};
    use crate::operations::references::{ReferenceAddUpdate, ReferenceWrite};
    use crate::operations::trigonometric::{Cos, Sin};

    use super::*;

    #[test]
    fn test_custom_derivative_builder_with_non_differentiated_count() {
        // The leading counter is plumbing for a custom JVP rule: it reaches both closures at its usual position and the
        // rule leaves the tangent placeholder of the counter unused.
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
                |input| {
                    custom_derivative_at(input).with_non_differentiated_count(1).jvp(
                        |(counter, x)| {
                            counter.add_update(&x)?;
                            Ok(x)
                        },
                        |(counter, x), (_, tangent)| {
                            counter.add_update(&x)?;
                            Ok((x, tangent))
                        },
                    )
                },
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // The leading stash is plumbing for a custom VJP rule: the forward rule forwards it as a residual, and the
        // backward rule writes the incoming cotangent into it while its own cotangent leaf is ignored.
        let stash = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let (value, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ))
        .vjp(|input| {
            custom_derivative_at(input).with_non_differentiated_count(1).vjp(
                |(_, x)| Ok(x),
                |(stash, x)| Ok((x, stash)),
                |stash, cotangent| {
                    stash.write(&cotangent)?;
                    Ok((stash, cotangent))
                },
            )
        })
        .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_custom_derivative_builder_jvp() {
        // The deliberately wrong rule `jvp(x, ẋ) = (sin(x), 2 * cos(x) * ẋ)` doubles the true derivative, which proves
        // that it governs both forward- and reverse-mode differentiation.
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                custom_derivative_at(x).jvp(
                    |x| Ok(x.sin()?),
                    |x, tangent| {
                        let tangent = x.cos()? * tangent;
                        Ok((x.sin()?, tangent.clone() + tangent))
                    },
                )
            }),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(3.0).unwrap()).value_and_gradient(|x| {
                custom_derivative_at(x)
                    .jvp(
                        |x| Ok(x.sin()?),
                        |x, tangent| {
                            let tangent = x.cos()? * tangent;
                            Ok((x.sin()?, tangent.clone() + tangent))
                        },
                    )
                    .unwrap()
            }),
            Ok((Array::scalar(3.0f64.sin()).unwrap(), Array::scalar(2.0 * 3.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_derivative_builder_vjp() {
        // Tuple inputs and destructured tuple residuals infer their types from the input and from the forward rule.
        // The deliberately wrong rule doubles the true gradients `(y, x)`.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap())).value_and_gradient(
                |(x, y)| {
                    custom_derivative_at((x, y))
                        .vjp(
                            |(x, y)| Ok(x * y),
                            |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
                            |(x, y), cotangent| {
                                let x_cotangent = y * cotangent.clone();
                                let y_cotangent = x * cotangent;
                                Ok((x_cotangent.clone() + x_cotangent, y_cotangent.clone() + y_cotangent))
                            },
                        )
                        .unwrap()
                },
            ),
            Ok((Array::scalar(10.0).unwrap(), (Array::scalar(10.0).unwrap(), Array::scalar(4.0).unwrap()))),
        );
    }
}
