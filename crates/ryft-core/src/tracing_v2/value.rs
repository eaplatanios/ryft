//! Foundational leaf-value traits used by `tracing_v2`.
//!
//! The tracing prototype distinguishes between two related concepts:
//!
//! - a concrete leaf value such as `f64` or an `ndarray::Array2<f64>`;
//! - the [`ArrayType`](crate::types::ArrayType) metadata that tracing needs in order to stage programs without
//!   inspecting full values.
//!
//! The traits in this module capture that boundary. They are intentionally small so that future tensor-like leaf
//! types, including PJRT-backed arrays, can adopt the tracing machinery by implementing a compact set of behaviors.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    types::{ArrayType, Typed},
};

/// Minimal floating-point surface used by the scalar tracing primitives.
///
/// Later backends can extend tracing to richer value types by implementing these operations on their leaf type.
pub trait FloatExt: Clone + Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> {
    /// Computes the elementwise sine.
    fn sin(self) -> Self;

    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
}

/// Returns a zero value with the same structure as an existing value.
///
/// This is the universal "zero-like" capability: every type that participates in differentiation or batching can
/// produce a zero from an exemplar, including tracer wrappers that cannot be synthesized from abstract metadata alone.
pub trait ZeroLike: Clone {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Synthesizes a zero value from abstract [`ArrayType`] metadata alone.
///
/// Unlike [`ZeroLike`], which produces a zero from an existing exemplar value, this trait is implemented on the
/// _type-metadata_ side: `Self` is a type parameterized by [`ArrayType`] (e.g., [`ArrayType`] itself for scalar
/// leaves, or `Pair<ArrayType>` for compound structures), and [`Zero::zero`] produces the corresponding concrete value
/// by reparameterizing from [`ArrayType`] to `P`.
///
/// This trait is only meaningful for concrete leaf types and parameterized containers of them — tracer wrappers that
/// carry builder state cannot be synthesized from metadata.
pub trait Zero<P: Parameter>: Parameterized<ArrayType> {
    /// Constructs one zero value from the abstract metadata in `self`.
    fn zero(&self) -> Result<Self::To<P>, crate::tracing_v2::TraceError>
    where
        Self::Family: ParameterizedFamily<P>;
}

/// Returns a one value with the same structure as an existing value.
///
/// This mirrors [`ZeroLike`] for the multiplicative identity.
pub trait OneLike: Clone {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

/// Synthesizes a one value from abstract [`ArrayType`] metadata alone.
///
/// This mirrors [`Zero`] for the multiplicative identity. `Self` is a type parameterized by [`ArrayType`], and
/// [`One::one`] produces the corresponding concrete value by reparameterizing from [`ArrayType`] to `P`.
pub trait One<P: Parameter>: Parameterized<ArrayType> {
    /// Constructs one one-valued value from the abstract metadata in `self`.
    fn one(&self) -> Result<Self::To<P>, crate::tracing_v2::TraceError>
    where
        Self::Family: ParameterizedFamily<P>;
}

/// Marker trait for concrete tracing leaves that participate in eager transform dispatch.
///
/// Exact identity detection itself lives on [`TraceValue`]. This marker exists only to partition the blanket impls
/// for concrete leaf regimes from the corresponding traced-leaf impls for
/// [`JitTracer`](crate::tracing_v2::JitTracer).
pub trait ConcreteTraceValue: TraceValue {}

/// Convenience trait for stageable leaves used by `tracing_v2`.
///
/// [`TraceValue`] identifies leaf values that can appear in staged graphs and participate in abstract evaluation. It
/// ties each runtime leaf to the shared [`ArrayType`](crate::types::ArrayType) descriptor used by `tracing_v2`
/// via [`Typed`], while deliberately not implying eager numeric operations such as [`FloatExt`] or
/// differentiation-specific capabilities such as [`ZeroLike`]. Those requirements live on the primitive operations
/// and transforms that actually use them.
///
/// Leaf types that support exact algebraic identity detection should override [`TraceValue::is_zero`] and
/// [`TraceValue::is_one`]. The default implementations return `false`, which keeps purely abstract leaves valid while
/// opting them out of constant-identity simplification.
pub trait TraceValue: Clone + Parameter + Typed<ArrayType> + 'static {
    /// Returns `true` if every element of this value is exactly zero.
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    /// Returns `true` if every element of this value is exactly one.
    #[inline]
    fn is_one(&self) -> bool {
        false
    }
}

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

impl TraceValue for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl ConcreteTraceValue for f32 {}

impl ZeroLike for f32 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl Zero<f32> for ArrayType {
    #[inline]
    fn zero(&self) -> Result<f32, crate::tracing_v2::TraceError> {
        if *self == ArrayType::scalar(crate::types::DataType::F32) {
            Ok(0.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeZeroWitness {
                value_kind: std::any::type_name::<f32>(),
                abstract_value: self.clone(),
            })
        }
    }
}

impl OneLike for f32 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

impl One<f32> for ArrayType {
    #[inline]
    fn one(&self) -> Result<f32, crate::tracing_v2::TraceError> {
        if *self == ArrayType::scalar(crate::types::DataType::F32) {
            Ok(1.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeOneWitness {
                value_kind: std::any::type_name::<f32>(),
                abstract_value: self.clone(),
            })
        }
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

impl TraceValue for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl ConcreteTraceValue for f64 {}

impl ZeroLike for f64 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl Zero<f64> for ArrayType {
    #[inline]
    fn zero(&self) -> Result<f64, crate::tracing_v2::TraceError> {
        if *self == ArrayType::scalar(crate::types::DataType::F64) {
            Ok(0.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeZeroWitness {
                value_kind: std::any::type_name::<f64>(),
                abstract_value: self.clone(),
            })
        }
    }
}

impl OneLike for f64 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

impl One<f64> for ArrayType {
    #[inline]
    fn one(&self) -> Result<f64, crate::tracing_v2::TraceError> {
        if *self == ArrayType::scalar(crate::types::DataType::F64) {
            Ok(1.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeOneWitness {
                value_kind: std::any::type_name::<f64>(),
                abstract_value: self.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use ryft_macros::Parameterized;

    use crate::{
        parameters::Parameter,
        tracing_v2::TraceError,
        tracing_v2::test_support,
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::*;

    #[test]
    fn scalar_leaf_traits_report_expected_values() {
        assert_eq!(<f32 as Typed<ArrayType>>::tpe(&1.25f32), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::tpe(&2.5f64), ArrayType::scalar(DataType::F64));
        assert_eq!(ZeroLike::zero_like(&3.0f32), 0.0);
        assert_eq!(OneLike::one_like(&3.0f64), 1.0);
        assert_eq!(Zero::<f32>::zero(&ArrayType::scalar(DataType::F32)), Ok(0.0));
        assert_eq!(Zero::<f64>::zero(&ArrayType::scalar(DataType::F64)), Ok(0.0));
        assert_eq!(One::<f32>::one(&ArrayType::scalar(DataType::F32)), Ok(1.0));
        assert_eq!(One::<f64>::one(&ArrayType::scalar(DataType::F64)), Ok(1.0));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn parameterized_zero_and_one_accept_structured_array_types() {
        #[derive(Clone, Debug, PartialEq, Parameterized)]
        #[ryft(crate = "crate::parameters")]
        struct Pair<T: Parameter> {
            left: T,
            right: T,
        }

        impl ZeroLike for Pair<f32> {
            fn zero_like(&self) -> Self {
                Self { left: 0.0, right: 0.0 }
            }
        }

        impl Zero<f32> for Pair<ArrayType> {
            fn zero(&self) -> Result<Pair<f32>, TraceError> {
                Ok(Pair { left: Zero::<f32>::zero(&self.left)?, right: Zero::<f32>::zero(&self.right)? })
            }
        }

        impl OneLike for Pair<f32> {
            fn one_like(&self) -> Self {
                Self { left: 1.0, right: 1.0 }
            }
        }

        impl One<f32> for Pair<ArrayType> {
            fn one(&self) -> Result<Pair<f32>, TraceError> {
                Ok(Pair { left: One::<f32>::one(&self.left)?, right: One::<f32>::one(&self.right)? })
            }
        }

        let abstract_value = Pair { left: ArrayType::scalar(DataType::F32), right: ArrayType::scalar(DataType::F32) };
        assert_eq!(Zero::<f32>::zero(&abstract_value), Ok(Pair { left: 0.0, right: 0.0 }));
        assert_eq!(One::<f32>::one(&abstract_value), Ok(Pair { left: 1.0, right: 1.0 }));
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(FloatExt::sin(angle), angle.sin());
        assert_eq!(FloatExt::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
