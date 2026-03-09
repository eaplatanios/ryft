//! Foundational leaf-value traits used by `tracing_v2`.
//!
//! The tracing prototype distinguishes between two related concepts:
//!
//! - a concrete leaf value such as `f64` or an `ndarray::Array2<f64>`;
//! - the structural metadata that tracing needs in order to stage programs without inspecting full values.
//!
//! The traits in this module capture that boundary. They are intentionally small so that future tensor-like leaf
//! types, including PJRT-backed arrays, can adopt the tracing machinery by implementing a compact set of behaviors.

use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
};

use ryft_macros::Parameter;

use crate::parameters::Parameter;

/// Abstract shape-and-dtype information for scalar leaves used by the prototype implementation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Parameter)]
pub enum ScalarAbstract {
    /// 32-bit floating-point scalar.
    F32,
    /// 64-bit floating-point scalar.
    F64,
}

impl Display for ScalarAbstract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarAbstract::F32 => write!(f, "f32[]"),
            ScalarAbstract::F64 => write!(f, "f64[]"),
        }
    }
}

/// Minimal floating-point surface used by the scalar tracing primitives.
///
/// Later backends can extend tracing to richer value types by implementing these operations on their leaf type.
pub trait FloatExt: Clone + Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> {
    /// Computes the elementwise sine.
    fn sin(self) -> Self;

    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
}

/// Trait implemented by indivisible tracing leaves.
///
/// `TraceLeaf` connects a concrete leaf value to its lightweight abstract value so that graphs can be staged and
/// validated without materializing every intermediate result.
pub trait TraceLeaf: Clone + Parameter {
    /// Abstract metadata used during tracing.
    type Abstract: Clone + Debug + Eq + PartialEq;

    /// Returns the abstract value corresponding to `self`.
    fn abstract_value(&self) -> Self::Abstract;
}

/// Creates a zero value with the same logical shape as `self`.
pub trait ZeroLike: Clone {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Creates a one value with the same logical shape as `self`.
pub trait OneLike: Clone {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

/// Convenience trait for leaves that fully participate in `tracing_v2`.
///
/// A `TraceValue` can be used as a staged program input/output leaf, as a primal value, and as a tangent/cotangent
/// constant where appropriate.
pub trait TraceValue: FloatExt + TraceLeaf + ZeroLike + 'static {}

impl<T> TraceValue for T where T: FloatExt + TraceLeaf + ZeroLike + 'static {}

impl FloatExt for f32 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceLeaf for f32 {
    type Abstract = ScalarAbstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        ScalarAbstract::F32
    }
}

impl ZeroLike for f32 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f32 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

impl FloatExt for f64 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceLeaf for f64 {
    type Abstract = ScalarAbstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        ScalarAbstract::F64
    }
}

impl ZeroLike for f64 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f64 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::test_support;

    use super::*;

    #[test]
    fn scalar_leaf_traits_report_expected_values() {
        assert_eq!(1.25f32.abstract_value(), ScalarAbstract::F32);
        assert_eq!(2.5f64.abstract_value(), ScalarAbstract::F64);
        assert_eq!(3.0f32.zero_like(), 0.0);
        assert_eq!(3.0f64.one_like(), 1.0);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(FloatExt::sin(angle), angle.sin());
        assert_eq!(FloatExt::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
