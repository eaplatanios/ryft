//! Capability bundles that aggregate the trait bounds repeatedly listed across the
//! [`InterpretableOperation`](crate::operations::InterpretableOperation),
//! [`TransposableOperation`](crate::differentiation::TransposableOperation), and
//! [`DifferentiableOperation`](crate::tracing_v2::DifferentiableOperation) implementations on the primitive operation
//! enums defined in [`primitive`](crate::tracing_v2::operations::primitive).
//!
//! The module exposes two flavors of bundle:
//!   - **Value-side bundles** ([`SupportsLinearArithmeticOperations`], [`SupportsArithmeticOperations`],
//!     [`SupportsTrigonometricOperations`], [`SupportsConstantOperations`], [`SupportsManipulationOperations`],
//!     [`SupportsLinearAlgebraOperations`], [`SupportsComparisonOperations`]) group the *value-type* requirements
//!     of a single operation category and are used as bounds on the value type `V` (or `D::Value` / `D::Tangent`).
//!   - **Operation-side bundles** ([`SupportsLinearScalarOperation`], [`SupportsLinearArrayOperation`]) group the
//!     corresponding *operation-type* requirements
//!     ([`SupportsZeroLike`](crate::operations::constants::SupportsZeroLike),
//!     [`SupportsNeg`](crate::operations::arithmetic::SupportsNeg),
//!     [`SupportsScale`](crate::operations::arithmetic::SupportsScale), etc.)
//!     and are used as bounds on `LinearOperationOf<D>` in the linearization rules of `jvp` and `transpose`.
//!
//! Each bundle has a blanket implementation, so consumers never implement them directly. The naming parallels the
//! operation-side `Supports*` traits already present in `ryft-core` (for example,
//! [`SupportsAdd`](crate::operations::arithmetic::SupportsAdd)).
//!
//! Bundles are deliberately orthogonal: each impl site composes only the categories its dispatcher actually
//! exercises. Single-trait bounds such as [`Fill<ArrayType, f64>`](crate::operations::constants::Fill),
//! [`Select`](crate::operations::control_flow::Select),
//! [`BooleanLike`](crate::operations::BooleanLike), and the bare
//! [`DotOps`](crate::tracing_v2::operations::matrix::DotOps) (without the captured-factor variants) are intentionally
//! not bundled — they are already one trait each and listing them inline keeps the bound list explicit at the call
//! site.

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::operations::arithmetic::{Scale, SupportsAdd, SupportsNeg, SupportsScale, SupportsSub};
use crate::operations::constants::{One, OneLike, SupportsZeroLike, Zero, ZeroLike};
use crate::operations::manipulation::{Broadcast, SupportsBroadcast, SupportsTranspose};
use crate::operations::trigonometric::{Cos, Sin};
use crate::programs::Value;
use crate::types::Type;

use super::dot::{LeftDot, RightDot, SupportsLeftDot, SupportsRightDot};
use super::matrix::DotOps;
use super::reduce::{Reduce, SupportsReduce};
use super::reshape::ReshapeOps;
use crate::operations::compare::Compare;
use crate::operations::manipulation::SupportsReshape;

/// Linear elementwise arithmetic primitives: addition, subtraction, negation, multiplication, and captured-factor
/// [`Scale`].
///
/// This bundle deliberately excludes [`Div`], which is not a linear map. See [`SupportsArithmeticOperations`] for
/// the ordinary (non-linear) extension that adds division.
///
/// The factor type parameter `F` of the captured-factor [`Scale`] defaults to `Self`; provide a distinct `F` when
/// the scaling factor lives in another value family, such as the parent context's constant type in the batching
/// rules for [`ArrayOperation`](super::primitive::ArrayOperation).
pub trait SupportsLinearArithmeticOperations<F = Self>:
    Sized + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Scale<F, Output = Self>
{
}

impl<F, V> SupportsLinearArithmeticOperations<F> for V where
    V: Add<Output = V> + Sub<Output = V> + Neg<Output = V> + Mul<Output = V> + Scale<F, Output = V>
{
}

/// Ordinary elementwise arithmetic primitives: [`SupportsLinearArithmeticOperations`] plus [`Div`].
///
/// [`Div`] is excluded from the linear bundle because division is not a linear map; this trait is the bundle to use
/// for ordinary (non-linear) dispatch paths such as [`ScalarOperation`](crate::operations::scalars::ScalarOperation)
/// and [`ArrayOperation`](super::primitive::ArrayOperation). The factor type parameter `F` follows
/// [`SupportsLinearArithmeticOperations`].
pub trait SupportsArithmeticOperations<F = Self>: SupportsLinearArithmeticOperations<F> + Div<Output = Self> {}

impl<F, V> SupportsArithmeticOperations<F> for V where V: SupportsLinearArithmeticOperations<F> + Div<Output = V> {}

/// Trigonometric primitives: [`Sin`] and [`Cos`].
pub trait SupportsTrigonometricOperations: Sin + Cos {}

impl<V> SupportsTrigonometricOperations for V where V: Sin + Cos {}

/// Type-parameterized and "like"-style constant primitives.
///
/// Composes [`Zero<T>`], [`One<T>`], [`ZeroLike`], and [`OneLike`]. The `f64`-keyed
/// [`Fill<ArrayType, f64>`](crate::operations::constants::Fill) primitive is intentionally not included here:
/// only the operation enums that include a `Fill` variant (notably [`ArrayOperation`](super::primitive::ArrayOperation)
/// and [`LinearArrayOperation`](super::primitive::LinearArrayOperation)) need it, and it is cleaner to list it
/// inline at those sites than to fragment this bundle.
pub trait SupportsConstantOperations<T: Type>: Zero<T> + One<T> + ZeroLike + OneLike {}

impl<T, V> SupportsConstantOperations<T> for V
where
    T: Type,
    V: Zero<T> + One<T> + ZeroLike + OneLike,
{
}

/// Shape-manipulation primitives: [`ReshapeOps`], [`Broadcast`], and [`Reduce`].
pub trait SupportsManipulationOperations: ReshapeOps + Broadcast<Output = Self> + Reduce {}

impl<V> SupportsManipulationOperations for V where V: ReshapeOps + Broadcast<Output = V> + Reduce {}

/// Linear-side linear-algebra primitives: the general [`DotOps`] plus the captured-factor [`LeftDot`] and
/// [`RightDot`] maps emitted by JVP rules of `Dot` and `Transpose`.
///
/// Ordinary (non-linear) operation enums that only require [`DotOps`] should list it as a single bound rather than pulling
/// in this bundle, to avoid spurious captured-factor requirements on the value type. The factor type parameter `F`
/// of the captured-factor maps follows [`SupportsLinearArithmeticOperations`].
pub trait SupportsLinearAlgebraOperations<F = Self>: DotOps + LeftDot<F> + RightDot<F> {}

impl<F, V> SupportsLinearAlgebraOperations<F> for V where V: DotOps + LeftDot<F> + RightDot<F> {}

/// Comparison and boolean-logical primitives: typed [`Compare`], the binary [`BitAnd`], [`BitOr`], and
/// [`BitXor`], and negation [`Not`].
pub trait SupportsComparisonOperations:
    Compare<Output = Self> + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Not<Output = Self>
{
}

impl<V> SupportsComparisonOperations for V where
    V: Compare<Output = V> + BitAnd<Output = V> + BitOr<Output = V> + BitXor<Output = V> + Not<Output = V>
{
}

/// Operation-type capabilities required for staging the linear scalar primitives during linearization (the `jvp` and
/// `transpose` rules) of the ordinary scalar operations.
///
/// Composes [`SupportsZeroLike`], [`SupportsNeg`], [`SupportsSub`], and the captured-factor [`SupportsScale`].
/// The factor type parameter `F` defaults to the tangent type `V`; provide a distinct `F` (typically the primal
/// value type from the source trace) when captured-factor scaling crosses the primal/tangent boundary.
pub trait SupportsLinearScalarOperation<T: Type, F: Value<T>>:
    SupportsAdd<T> + SupportsZeroLike<T> + SupportsNeg<T> + SupportsSub<T> + SupportsScale<T, F>
{
}

impl<T, F, C> SupportsLinearScalarOperation<T, F> for C
where
    T: Type,
    F: Value<T>,
    C: SupportsAdd<T> + SupportsZeroLike<T> + SupportsNeg<T> + SupportsSub<T> + SupportsScale<T, F>,
{
}

/// Operation-type capabilities required for staging the linear array primitives during linearization on `ArrayType`.
///
/// Extends [`SupportsLinearScalarOperation`] with the captured-factor dot maps ([`SupportsLeftDot`],
/// [`SupportsRightDot`]), [`SupportsTranspose`], and the array-shape manipulation primitives
/// ([`SupportsReshape`], [`SupportsBroadcast`], [`SupportsReduce`]).
pub trait SupportsLinearArrayOperation<T: Type, F: Value<T>>:
    SupportsLinearScalarOperation<T, F>
    + SupportsLeftDot<T, F>
    + SupportsRightDot<T, F>
    + SupportsTranspose<T>
    + SupportsReshape<T>
    + SupportsBroadcast<T>
    + SupportsReduce<T>
{
}

impl<T, F, C> SupportsLinearArrayOperation<T, F> for C
where
    T: Type,
    F: Value<T>,
    C: SupportsLinearScalarOperation<T, F>
        + SupportsLeftDot<T, F>
        + SupportsRightDot<T, F>
        + SupportsTranspose<T>
        + SupportsReshape<T>
        + SupportsBroadcast<T>
        + SupportsReduce<T>,
{
}
