pub mod constant_like;
pub mod one;
pub mod one_like;
pub mod zero;
pub mod zero_like;

pub use constant_like::*;
pub use one::*;
pub use one_like::*;
pub use zero::*;
pub use zero_like::*;

use half::{bf16, f16};

use crate::tracing::TracingError;
use crate::types::{DataType, TypeError};

macro_rules! impl_constants_for_scalar {
    ($ty:ty, $data_type:path, $zero:expr, $one:expr) => {
        impl ZeroLike for $ty {
            #[inline]
            fn zero_like(&self) -> Self {
                $zero
            }
        }

        impl OneLike for $ty {
            #[inline]
            fn one_like(&self) -> Self {
                $one
            }
        }

        impl Zero<DataType> for $ty {
            #[inline]
            fn zero(r#type: &DataType) -> Result<Self, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {} but got {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($zero)
            }
        }

        impl One<DataType> for $ty {
            #[inline]
            fn one(r#type: &DataType) -> Result<Self, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {} but got {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($one)
            }
        }
    };
}

impl_constants_for_scalar!(bool, DataType::Boolean, false, true);
impl_constants_for_scalar!(i8, DataType::I8, 0i8, 1i8);
impl_constants_for_scalar!(i16, DataType::I16, 0i16, 1i16);
impl_constants_for_scalar!(i32, DataType::I32, 0i32, 1i32);
impl_constants_for_scalar!(i64, DataType::I64, 0i64, 1i64);
impl_constants_for_scalar!(u8, DataType::U8, 0u8, 1u8);
impl_constants_for_scalar!(u16, DataType::U16, 0u16, 1u16);
impl_constants_for_scalar!(u32, DataType::U32, 0u32, 1u32);
impl_constants_for_scalar!(u64, DataType::U64, 0u64, 1u64);
impl_constants_for_scalar!(bf16, DataType::BF16, bf16::ZERO, bf16::ONE);
impl_constants_for_scalar!(f16, DataType::F16, f16::ZERO, f16::ONE);
impl_constants_for_scalar!(f32, DataType::F32, 0.0f32, 1.0f32);
impl_constants_for_scalar!(f64, DataType::F64, 0.0f64, 1.0f64);
