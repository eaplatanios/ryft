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

/// Creates zero values either from abstract metadata or from an existing value.
///
/// The `zero(...)` constructor is used when staging must synthesize one representative witness from
/// [`Self::To<ArrayType>`](Parameterized::To) metadata alone.
pub trait Zero<P: Parameter = Self>: Clone + Parameterized<P> {
    /// Constructs one zero value compatible with `r#type`.
    fn zero(r#type: Self::To<ArrayType>) -> Result<Self, crate::tracing_v2::TraceError>
    where
        Self: Sized,
        Self::Family: ParameterizedFamily<ArrayType>;

    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Creates one values either from abstract metadata or from an existing value.
///
/// This mirrors [`Zero`], using [`Self::To<ArrayType>`](Parameterized::To) to preserve the parameter structure of the
/// abstract witness description.
pub trait One<P: Parameter = Self>: Clone + Parameterized<P> {
    /// Constructs one one-valued witness compatible with `r#type`.
    fn one(r#type: Self::To<ArrayType>) -> Result<Self, crate::tracing_v2::TraceError>
    where
        Self: Sized,
        Self::Family: ParameterizedFamily<ArrayType>;

    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
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
/// differentiation-specific capabilities such as [`Zero`]. Those requirements live on the primitive operations
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

impl Zero for f32 {
    #[inline]
    fn zero(r#type: ArrayType) -> Result<Self, crate::tracing_v2::TraceError> {
        if r#type == ArrayType::scalar(crate::types::DataType::F32) {
            Ok(0.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeZeroWitness {
                value_kind: std::any::type_name::<Self>(),
                abstract_value: r#type,
            })
        }
    }

    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl One for f32 {
    #[inline]
    fn one(r#type: ArrayType) -> Result<Self, crate::tracing_v2::TraceError> {
        if r#type == ArrayType::scalar(crate::types::DataType::F32) {
            Ok(1.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeOneWitness {
                value_kind: std::any::type_name::<Self>(),
                abstract_value: r#type,
            })
        }
    }

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

impl Zero for f64 {
    #[inline]
    fn zero(r#type: ArrayType) -> Result<Self, crate::tracing_v2::TraceError> {
        if r#type == ArrayType::scalar(crate::types::DataType::F64) {
            Ok(0.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeZeroWitness {
                value_kind: std::any::type_name::<Self>(),
                abstract_value: r#type,
            })
        }
    }

    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl One for f64 {
    #[inline]
    fn one(r#type: ArrayType) -> Result<Self, crate::tracing_v2::TraceError> {
        if r#type == ArrayType::scalar(crate::types::DataType::F64) {
            Ok(1.0)
        } else {
            Err(crate::tracing_v2::TraceError::CannotSynthesizeOneWitness {
                value_kind: std::any::type_name::<Self>(),
                abstract_value: r#type,
            })
        }
    }

    #[inline]
    fn one_like(&self) -> Self {
        1.0
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
        assert_eq!(3.0f32.zero_like(), 0.0);
        assert_eq!(3.0f64.one_like(), 1.0);
        assert_eq!(f32::zero(ArrayType::scalar(DataType::F32)), Ok(0.0));
        assert_eq!(f64::zero(ArrayType::scalar(DataType::F64)), Ok(0.0));
        assert_eq!(f32::one(ArrayType::scalar(DataType::F32)), Ok(1.0));
        assert_eq!(f64::one(ArrayType::scalar(DataType::F64)), Ok(1.0));
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

        impl Zero<f32> for Pair<f32> {
            fn zero(r#type: Self::To<ArrayType>) -> Result<Self, TraceError> {
                Ok(Self { left: f32::zero(r#type.left)?, right: f32::zero(r#type.right)? })
            }

            fn zero_like(&self) -> Self {
                Self { left: 0.0, right: 0.0 }
            }
        }

        impl One<f32> for Pair<f32> {
            fn one(r#type: Self::To<ArrayType>) -> Result<Self, TraceError> {
                Ok(Self { left: f32::one(r#type.left)?, right: f32::one(r#type.right)? })
            }

            fn one_like(&self) -> Self {
                Self { left: 1.0, right: 1.0 }
            }
        }

        let abstract_value = Pair { left: ArrayType::scalar(DataType::F32), right: ArrayType::scalar(DataType::F32) };
        assert_eq!(Pair::<f32>::zero(abstract_value.clone()), Ok(Pair { left: 0.0, right: 0.0 }));
        assert_eq!(Pair::<f32>::one(abstract_value), Ok(Pair { left: 1.0, right: 1.0 }));
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(FloatExt::sin(angle), angle.sin());
        assert_eq!(FloatExt::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
