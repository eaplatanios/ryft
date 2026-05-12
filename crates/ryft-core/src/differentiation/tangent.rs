use std::borrow::Cow;
use std::convert::Infallible;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::operations::constants::{One, Zero, ZeroLike};
use crate::parameters::Parameter;
use crate::tracing::{Traceable, TracingError};
use crate::types::{Type, Typed};

/// [`Tangent`] produced when differentiating a primal value and which is the main value type that forward-mode tangent
/// [`Program`](crate::Program)s operate over.
///
/// In order to explain what a tangent is more formally, let us introduce some notation:
///
///   - `f: X -> Y` is a _differentiable map_.
///   - `x` is a point in the input space `X`.
///   - `T_x X` is the _tangent_ space of `X` at `x`; its elements are input perturbations or directions.
///   - `dot_x` is an input tangent in `T_x X`.
///   - `d f_x: T_x X -> T_{f(x)} Y` is the derivative of `f` at `x`, viewed as a _linear map_ that pushes input
///     tangents forward to output tangents.
///
/// Given an input tangent `dot_x`, forward-mode differentiation computes the output tangent `dot_y` in `T_{f(x)} Y` by
/// applying the derivative directly: `dot_y = d f_x(dot_x)`. In finite-dimensional coordinates, if `d f_x` is
/// represented by the Jacobian matrix `J_f(x)`, this is the Jacobian-vector product `dot_y = J_f(x) dot_x`.
///
/// In Ryft staged differentiation code, [`Tangent`]s are the values consumed and produced by linear _pushforward_
/// [`Program`](crate::Program)s. [`Tangent::Zero`] represents a structural zero tangent: it carries only abstract
/// [`Type`] metadata, so linear interpreters can propagate zero tangent spaces without materializing a concrete
/// payload. [`Tangent::Value`] carries an explicit tangent payload. That payload may still be a concrete value whose
/// numeric contents are all zero; the variant only records that the tangent is represented by a payload rather than
/// by the symbolic zero branch. Fully zero tangent spaces use `Tangent<T, Infallible>`, where [`Tangent::Value`] is
/// statically unconstructible.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub enum Tangent<T: Type, V: Traceable<T>> {
    /// [`Tangent`] value that is structurally known to be zero.
    Zero(T),

    /// [`Tangent`] value that is not structurally known to be zero.
    Value(V),
}

impl<T: Type, V: Traceable<T>> Tangent<T, V> {
    /// Creates a new [`Tangent::Zero`].
    #[inline]
    pub fn zero(r#type: T) -> Self {
        Self::Zero(r#type)
    }

    /// Creates a new [`Tangent::Value`].
    #[inline]
    pub fn value(value: V) -> Self {
        Self::Value(value)
    }

    /// Returns `true` if this is a [`Tangent::Zero`].
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// Returns the value stored in this [`Tangent`], if it is a [`Tangent::Value`], and `None` otherwise.
    #[inline]
    pub fn as_value(&self) -> Option<&V> {
        match self {
            Self::Zero(_) => None,
            Self::Value(value) => Some(value),
        }
    }
}

impl<T: Type, V: Traceable<T>> Display for Tangent<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(r#type) => write!(formatter, "Zero[{type}]", type = r#type),
            Self::Value(value) => Display::fmt(value, formatter),
        }
    }
}

impl<T: Type, V: Traceable<T>> Typed<T> for Tangent<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Zero(r#type) => Cow::Borrowed(r#type),
            Self::Value(value) => value.r#type(),
        }
    }
}

impl<T: Parameter + Type, V: Traceable<T>> Traceable<T> for Tangent<T, V> {}

impl<T: Type, V: Traceable<T>> Zero<T> for Tangent<T, V> {
    #[inline]
    fn zero(r#type: &T) -> Result<Self, TracingError> {
        Ok(Self::Zero(r#type.clone()))
    }
}

impl<T: Type, V: Traceable<T> + One<T>> One<T> for Tangent<T, V> {
    #[inline]
    fn one(r#type: &T) -> Result<Self, TracingError> {
        Ok(Self::Value(V::one(r#type)?))
    }
}

impl<T: Type, V: Traceable<T>> ZeroLike for Tangent<T, V> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::Zero(self.r#type().into_owned())
    }
}

// `Tangent<T, Infallible>` is the zero-only tangent representation described in the `Tangent` documentation:
// `Tangent::Value(Infallible)` cannot be constructed, but the generic enum still requires its payload type to
// satisfy the ordinary trace leaf value contracts. These implementations are vacuous because there is no `Infallible`
// value to inspect or print.
impl<T: Type> Typed<T> for Infallible {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match *self {}
    }
}

impl Parameter for Infallible {}

impl<T: Type> Traceable<T> for Infallible {}
