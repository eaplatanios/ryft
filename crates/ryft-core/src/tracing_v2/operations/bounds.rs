//! Capability bundles that aggregate the trait bounds repeatedly listed across the
//! [`InterpretableOperation`](crate::operations::InterpretableOperation),
//! [`LinearOperation`](crate::differentiation::LinearOperation), and
//! [`DifferentiableOperation`](crate::tracing_v2::DifferentiableOperation) implementations on the primitive
//! carriers defined in [`super::primitive`].
//!
//! The module exposes two flavors of bundle:
//!   - **Value-side bundles** ([`SupportsLinearArithmeticOperations`], [`SupportsArithmeticOperations`],
//!     [`SupportsTrigonometricOperations`], [`SupportsConstantOperations`], [`SupportsManipulationOperations`],
//!     [`SupportsLinearAlgebraOperations`], [`SupportsComparisonOperations`]) group the *value-type* requirements
//!     of a single operation category and are used as bounds on the value type `V` (or `D::Value` / `D::Tangent`).
//!   - **Carrier-side bundles** ([`SupportsLinearScalarOperationCarrier`],
//!     [`SupportsLinearArrayOperationCarrier`]) group the corresponding *operation-carrier* requirements
//!     ([`SupportsZeroLike`], [`SupportsNeg`], [`SupportsScale`], etc.) and are used as bounds on
//!     `LinearOperationCarrier<D>` in the linearization rules of `jvp` and `transpose`.
//!
//! Each bundle has a blanket implementation, so consumers never implement them directly. The naming parallels the
//! carrier-side `Supports*` traits already present in `ryft-core` (for example,
//! [`SupportsAdd`](crate::operations::arithmetic::SupportsAdd)).
//!
//! Bundles are deliberately orthogonal: each impl site composes only the categories its dispatcher actually
//! exercises. Single-trait bounds such as [`ConstantLike<f64>`](crate::operations::constants::ConstantLike),
//! [`Select`](super::select::Select), [`ControlFlowValue`](super::control_flow::ControlFlowValue), and the bare
//! [`DotOps`](super::matrix::DotOps) (without the captured-factor variants) are intentionally not bundled — they are
//! already one trait each and listing them inline keeps the bound list explicit at the call site.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::operations::arithmetic::{Scale, SupportsAdd, SupportsNeg, SupportsScale, SupportsSub};
use crate::operations::constants::{One, OneLike, SupportsZeroLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::tracing::Traceable;
use crate::types::Type;

use super::broadcast::{BroadcastInDim, SupportsBroadcastInDim};
use super::compare::Compare;
use super::dot::{LeftDot, RightDot, SupportsLeftDot, SupportsRightDot};
use super::logical::{LogicalBinary, LogicalNot};
use super::matrix::DotOps;
use super::reduce::{Reduce, SupportsReduce};
use super::reshape::{ReshapeOps, SupportsReshape};
use super::transpose::SupportsTranspose;

/// Linear elementwise arithmetic primitives: addition, subtraction, negation, multiplication, and captured-factor
/// [`Scale`].
///
/// This bundle deliberately excludes [`Div`], which is not a linear map. See [`SupportsArithmeticOperations`] for
/// the ordinary (non-linear) extension that adds division.
pub trait SupportsLinearArithmeticOperations:
    Sized + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self> + Scale<Output = Self>
{
}

impl<V> SupportsLinearArithmeticOperations for V where
    V: Add<Output = V> + Sub<Output = V> + Neg<Output = V> + Mul<Output = V> + Scale<Output = V>
{
}

/// Ordinary elementwise arithmetic primitives: [`SupportsLinearArithmeticOperations`] plus [`Div`].
///
/// [`Div`] is excluded from the linear bundle because division is not a linear map; this trait is the bundle to use
/// for ordinary (non-linear) dispatch paths such as [`ScalarOperation`](super::scalars::ScalarOperation) and
/// [`ArrayOperation`](super::primitive::ArrayOperation).
pub trait SupportsArithmeticOperations: SupportsLinearArithmeticOperations + Div<Output = Self> {}

impl<V> SupportsArithmeticOperations for V where V: SupportsLinearArithmeticOperations + Div<Output = V> {}

/// Trigonometric primitives: [`Sin`] and [`Cos`].
pub trait SupportsTrigonometricOperations: Sin + Cos {}

impl<V> SupportsTrigonometricOperations for V where V: Sin + Cos {}

/// Type-parameterized and "like"-style constant primitives.
///
/// Composes [`Zero<T>`], [`One<T>`], [`ZeroLike`], and [`OneLike`]. The `f64`-keyed
/// [`ConstantLike<f64>`](crate::operations::constants::ConstantLike) primitive is intentionally not included here:
/// only the carriers that ship a `ConstantLike` variant (notably [`ArrayOperation`](super::primitive::ArrayOperation)
/// and [`LinearArrayOperation`](super::primitive::LinearArrayOperation)) need it, and it is cleaner to list it
/// inline at those sites than to fragment this bundle.
pub trait SupportsConstantOperations<T: Type>: Zero<T> + One<T> + ZeroLike + OneLike {}

impl<T, V> SupportsConstantOperations<T> for V
where
    T: Type,
    V: Zero<T> + One<T> + ZeroLike + OneLike,
{
}

/// Shape-manipulation primitives: [`ReshapeOps`], [`BroadcastInDim`], and [`Reduce`].
pub trait SupportsManipulationOperations: ReshapeOps + BroadcastInDim + Reduce {}

impl<V> SupportsManipulationOperations for V where V: ReshapeOps + BroadcastInDim + Reduce {}

/// Linear-side linear-algebra primitives: the general [`DotOps`] plus the captured-factor [`LeftDot`] and
/// [`RightDot`] maps emitted by JVP rules of `Dot` and `Transpose`.
///
/// Ordinary (non-linear) carriers that only require [`DotOps`] should list it as a single bound rather than pulling
/// in this bundle, to avoid spurious captured-factor requirements on the value type.
pub trait SupportsLinearAlgebraOperations: DotOps + LeftDot + RightDot {}

impl<V> SupportsLinearAlgebraOperations for V where V: DotOps + LeftDot + RightDot {}

/// Comparison and boolean-logical primitives: typed [`Compare`], binary [`LogicalBinary`], and logical negation
/// [`LogicalNot`].
pub trait SupportsComparisonOperations: Compare<Output = Self> + LogicalBinary + LogicalNot {}

impl<V> SupportsComparisonOperations for V where V: Compare<Output = V> + LogicalBinary + LogicalNot {}

/// Carrier capabilities required for staging the linear scalar primitives during linearization (the `jvp` and
/// `transpose` rules) of the ordinary scalar operations.
///
/// Composes [`SupportsZeroLike`], [`SupportsNeg`], [`SupportsSub`], and the captured-factor [`SupportsScale`].
/// The factor type parameter `F` defaults to the tangent type `V`; provide a distinct `F` (typically the primal
/// value type from the source trace) when captured-factor scaling crosses the primal/tangent boundary.
pub trait SupportsLinearScalarOperationCarrier<T: Type, V: Traceable<T>, F: Traceable<T> = V>:
    SupportsAdd<T, V> + SupportsZeroLike<T, V> + SupportsNeg<T, V> + SupportsSub<T, V> + SupportsScale<T, V, F>
{
}

impl<T, V, F, C> SupportsLinearScalarOperationCarrier<T, V, F> for C
where
    T: Type,
    V: Traceable<T>,
    F: Traceable<T>,
    C: SupportsAdd<T, V> + SupportsZeroLike<T, V> + SupportsNeg<T, V> + SupportsSub<T, V> + SupportsScale<T, V, F>,
{
}

/// Carrier capabilities required for staging the linear array primitives during linearization on `ArrayType`.
///
/// Extends [`SupportsLinearScalarOperationCarrier`] with the captured-factor dot maps ([`SupportsLeftDot`],
/// [`SupportsRightDot`]), [`SupportsTranspose`], and the array-shape manipulation primitives
/// ([`SupportsReshape`], [`SupportsBroadcastInDim`], [`SupportsReduce`]).
pub trait SupportsLinearArrayOperationCarrier<T: Type, V: Traceable<T>, F: Traceable<T> = V>:
    SupportsLinearScalarOperationCarrier<T, V, F>
    + SupportsLeftDot<T, V, F>
    + SupportsRightDot<T, V, F>
    + SupportsTranspose<T, V>
    + SupportsReshape<T, V>
    + SupportsBroadcastInDim<T, V>
    + SupportsReduce<T, V>
{
}

impl<T, V, F, C> SupportsLinearArrayOperationCarrier<T, V, F> for C
where
    T: Type,
    V: Traceable<T>,
    F: Traceable<T>,
    C: SupportsLinearScalarOperationCarrier<T, V, F>
        + SupportsLeftDot<T, V, F>
        + SupportsRightDot<T, V, F>
        + SupportsTranspose<T, V>
        + SupportsReshape<T, V>
        + SupportsBroadcastInDim<T, V>
        + SupportsReduce<T, V>,
{
}
