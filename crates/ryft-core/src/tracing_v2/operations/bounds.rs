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
//!     corresponding *operation-type* requirements as the standard per-variant conversions
//!     ([`From<AddOperation>`](crate::operations::arithmetic::AddOperation),
//!     [`From<NegOperation>`](crate::operations::arithmetic::NegOperation),
//!     [`From<ScaleOperation>`](crate::operations::arithmetic::ScaleOperation),
//!     [`From<ReshapeOperation>`](crate::operations::manipulation::ReshapeOperation), etc.). These bundles are used as
//!     bounds on `LinearOperationOf<D>` in the linearization rules of `jvp` and `transpose`.
//!
//! Each bundle has a blanket implementation, so consumers never implement them directly.
//!
//! Bundles are deliberately orthogonal: each impl site composes only the categories its dispatcher actually
//! exercises. Context-side bounds such as [`Fill<ArrayType, f64, V>`](crate::operations::constants::Fill),
//! [`Select`](crate::operations::control_flow::Select),
//! [`BooleanLike`](crate::operations::BooleanLike), and the bare
//! [`DotOps`](crate::tracing_v2::operations::dot::DotOps) (without the captured-factor variants) are intentionally
//! not bundled — they are already one trait each and listing them inline keeps the bound list explicit at the call
//! site.

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::operations::arithmetic::{AddOperation, NegOperation, ScaleOperation, SubOperation};
use crate::operations::constants::{OneLike, ZeroLike, ZeroLikeOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, Concatenate, DynamicSlice, DynamicUpdateSlice, Gather, Pad, PadOperation,
    ReshapeOperation, Scatter, Slice, SliceOperation, TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::trigonometric::{Cos, Sin};
use crate::payloads::Input;
use crate::programs::Value;
use crate::types::{ArrayType, Type};

use super::dot::{DotOps, LeftDotOperation, RightDotOperation};
use super::reduce::{Reduce, ReduceOperation};
use super::reshape::ReshapeOps;
use crate::operations::compare::Compare;
use crate::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};

/// Linear elementwise arithmetic primitives: addition, subtraction, negation, and multiplication.
///
/// This bundle deliberately excludes [`Div`], which is not a linear map. See [`SupportsArithmeticOperations`] for
/// the ordinary (non-linear) extension that adds division.
///
/// The factor type parameter is retained for API symmetry with the operation-side linear bundles, but factor scaling
/// itself is provided by the context-owned [`Scale`](crate::operations::arithmetic::Scale) capability at the impl sites
/// that interpret [`ScaleOperation`]s.
pub trait SupportsLinearArithmeticOperations<F = Self>:
    Sized + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self>
{
}

impl<F, V> SupportsLinearArithmeticOperations<F> for V where
    V: Add<Output = V> + Sub<Output = V> + Neg<Output = V> + Mul<Output = V>
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

/// Exemplar-derived constant primitives.
///
/// Composes [`ZeroLike`] and [`OneLike`]. Type-parameterized nullary primitives such as
/// [`Zero`](crate::operations::constants::Zero), [`One`](crate::operations::constants::One), and
/// [`Fill`](crate::operations::constants::Fill) are context-side capabilities, so impl sites that interpret nullary
/// operations list those bounds on `V::InterpretationContext` directly.
pub trait SupportsConstantOperations<T: Type>: ZeroLike + OneLike {}

impl<T, V> SupportsConstantOperations<T> for V
where
    T: Type,
    V: ZeroLike + OneLike,
{
}

/// Shape-manipulation and sharding-control primitives: [`ReshapeOps`], [`Broadcast`], [`Reduce`], [`Pad`],
/// [`Concatenate`], the slicing family ([`Slice`], [`UpdateSlice`], [`DynamicSlice`], [`DynamicUpdateSlice`]), and
/// the sharding-control [`Reshard`] and [`ConstrainSharding`].
pub trait SupportsManipulationOperations:
    ReshapeOps
    + Broadcast
    + Reduce
    + Pad
    + Concatenate
    + Slice
    + UpdateSlice
    + DynamicSlice
    + DynamicUpdateSlice
    + Gather
    + Scatter
    + Reshard
    + ConstrainSharding
{
}

impl<V> SupportsManipulationOperations for V where
    V: ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
{
}

/// Linear-side linear-algebra primitives: the general [`DotOps`].
///
/// Captured-factor dot maps are context-owned interpretation capabilities on
/// [`LeftDot`](crate::tracing_v2::operations::dot::LeftDot) and
/// [`RightDot`](crate::tracing_v2::operations::dot::RightDot), so impl sites that interpret
/// [`LeftDotOperation`]s or [`RightDotOperation`]s list those context bounds directly. The factor type parameter is
/// retained for API symmetry with the operation-side linear bundles.
pub trait SupportsLinearAlgebraOperations<F = Self>: DotOps {}

impl<F, V> SupportsLinearAlgebraOperations<F> for V where V: DotOps {}

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
/// Composes the per-variant conversions from [`AddOperation`], [`ZeroLikeOperation`], [`NegOperation`],
/// [`SubOperation`], and the captured-factor [`ScaleOperation`]. The factor type parameter `F` defaults to the tangent
/// type `V`; provide a distinct `F` (typically the primal value type from the source trace) when captured-factor
/// scaling crosses the primal/tangent boundary.
pub trait SupportsLinearScalarOperation<T: Type, F: Value<T>>:
    From<AddOperation>
    + From<ZeroLikeOperation>
    + From<NegOperation>
    + From<SubOperation>
    + From<ScaleOperation<T, F, Input>>
{
}

impl<T, F, C> SupportsLinearScalarOperation<T, F> for C
where
    T: Type,
    F: Value<T>,
    C: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<T, F, Input>>,
{
}

/// Operation-type capabilities required for staging the linear array primitives during linearization on `ArrayType`.
///
/// Extends [`SupportsLinearScalarOperation`] with the per-variant conversions for the captured-factor dot maps
/// ([`From<LeftDotOperation>`](crate::tracing_v2::operations::dot::LeftDotOperation),
/// [`From<RightDotOperation>`](crate::tracing_v2::operations::dot::RightDotOperation)),
/// [`From<TransposeOperation>`](crate::operations::manipulation::TransposeOperation), the array-shape manipulation
/// primitives ([`From<ReshapeOperation>`](crate::operations::manipulation::ReshapeOperation),
/// [`From<BroadcastOperation>`](crate::operations::manipulation::BroadcastOperation),
/// [`From<ReduceOperation>`](crate::tracing_v2::operations::reduce::ReduceOperation),
/// [`From<PadOperation>`](crate::operations::manipulation::PadOperation)), the statically indexed slicing pair
/// ([`From<SliceOperation>`](crate::operations::manipulation::SliceOperation),
/// [`From<UpdateSliceOperation>`](crate::operations::manipulation::UpdateSliceOperation)), and the sharding-control
/// conversions ([`From<ReshardOperation>`](crate::operations::sharding::ReshardOperation),
/// [`From<ShardingConstraintOperation>`](crate::operations::sharding::ShardingConstraintOperation)). The dynamically
/// indexed slicing primitives are not included because their linear forms capture start indices as factors; rules
/// that stage them list
/// [`From<LinearDynamicSliceOperation>`](crate::operations::manipulation::LinearDynamicSliceOperation) and
/// [`From<LinearDynamicUpdateSliceOperation>`](crate::operations::manipulation::LinearDynamicUpdateSliceOperation)
/// inline, mirroring [`From<LinearSelectOperation>`](crate::tracing_v2::operations::select::LinearSelectOperation).
pub trait SupportsLinearArrayOperation<F: Value<ArrayType>>:
    SupportsLinearScalarOperation<ArrayType, F>
    + From<LeftDotOperation<F, Input>>
    + From<RightDotOperation<F, Input>>
    + From<TransposeOperation>
    + From<ReshapeOperation>
    + From<BroadcastOperation>
    + From<ReduceOperation>
    + From<PadOperation>
    + From<SliceOperation>
    + From<UpdateSliceOperation>
    + From<ReshardOperation>
    + From<ShardingConstraintOperation>
{
}

impl<F, C> SupportsLinearArrayOperation<F> for C
where
    F: Value<ArrayType>,
    C: SupportsLinearScalarOperation<ArrayType, F>
        + From<LeftDotOperation<F, Input>>
        + From<RightDotOperation<F, Input>>
        + From<TransposeOperation>
        + From<ReshapeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<PadOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshardOperation>
        + From<ShardingConstraintOperation>,
{
}
