use crate::operations::Operation;
use crate::parameters::Parameter;
use crate::tracing::Traceable;
use crate::tracing::domains::ProgramTracer;
use crate::types::Type;

/// [`Cotangent`] produced when differentiating a [`Program`](crate::Program) and which is the main value type that
/// [_transposition_](crate::Program::transpose) operates over.
///
/// In order to explain what a cotangent is more formally, let us introduce some notation:
///
///   - `f: X -> Y` is a _differentiable map_.
///   - `x` is a point in the input space `X`.
///   - `T_x X` is the _tangent_ space of `X` at `x`; its elements are input perturbations or directions.
///   - `T_x^* X` is the _dual_ of `T_x X`; its elements are _cotangents_ (i.e., linear functionals) `T_x X -> R`.
///   - `d f_x: T_x X -> T_{f(x)} Y` is the derivative of `f` at `x`, viewed as a _linear map_ that pushes input
///     tangents forward to output tangents.
///
/// Given an output cotangent `bar_y` in `T_{f(x)}^* Y`, reverse-mode differentiation computes the input cotangent
/// `bar_x` in `T_x^* X` by applying the dual, or pullback, of the derivative: `(d f_x)^*: T_{f(x)}^* Y -> T_x^* X`.
/// Formally, `bar_x = (d f_x)^*(bar_y)` is defined by `bar_x(dot_x) = bar_y(d f_x(dot_x))` for every input tangent
/// `dot_x` in `T_x X`. In finite-dimensional coordinates, if `d f_x` is represented by the Jacobian matrix `J_f(x)`,
///  this is the vector-Jacobian product `bar_x = J_f(x)^T bar_y`.
///
/// In the [`transposition`](crate::differentiation::transposition) module, the derivative has already been staged as a
/// linear tangent pushforward [`Program`](crate::Program). Transposition builds the dual pullback program, and
/// [`Cotangent`] is the rule-boundary representation of one symbolic cotangent contribution during that construction.
/// [`Cotangent::Zero`] represents a structural zero: no atom is staged in the transpose builder because the current
/// instruction contributes nothing to that input cotangent. [`Cotangent::Staged`] carries an actual symbolic cotangent
/// [`Tracer`](crate::tracing::Tracer) in the active [`ProgramTracingContext`](crate::tracing::ProgramTracingContext).
pub enum Cotangent<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> {
    /// [`Cotangent`] value that is known to be zero, structurally, and thus has not corresponding staged atom.
    Zero,

    /// [`Cotangent`] value that is staged in a [`Program`](crate::Program) that is being traced.
    Staged(ProgramTracer<'domain, T, V, O>),
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> Cotangent<'domain, T, V, O> {
    /// Creates a new [`Cotangent::Zero`].
    #[inline]
    pub const fn zero() -> Self {
        Self::Zero
    }

    /// Creates a new [`Cotangent::Staged`].
    #[inline]
    pub const fn staged(cotangent: ProgramTracer<'domain, T, V, O>) -> Self {
        Self::Staged(cotangent)
    }

    /// Returns `true` if this is a [`Cotangent::Zero`].
    #[inline]
    pub const fn is_zero(&self) -> bool {
        matches!(self, Self::Zero)
    }

    /// Returns the [`ProgramTracer`] stored in this [`Cotangent`], if it is a [`Cotangent::Staged`],
    /// and `None` otherwise.
    #[inline]
    pub fn as_staged(&self) -> Option<&ProgramTracer<'domain, T, V, O>> {
        match self {
            Self::Zero => None,
            Self::Staged(cotangent) => Some(cotangent),
        }
    }
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> Clone for Cotangent<'domain, T, V, O> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Staged(cotangent) => Self::Staged(cotangent.clone()),
        }
    }
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> From<ProgramTracer<'domain, T, V, O>>
    for Cotangent<'domain, T, V, O>
{
    #[inline]
    fn from(cotangent: ProgramTracer<'domain, T, V, O>) -> Self {
        Self::staged(cotangent)
    }
}
