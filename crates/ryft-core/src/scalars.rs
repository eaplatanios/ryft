use std::borrow::Cow;
use std::marker::PhantomData;

use half::{bf16, f16};

use crate::contexts::Context;
use crate::domains::Domain;
use crate::operations::InterpretableOperation;
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::programs::{ProgramError, Value};
use crate::types::{DataType, Typed};

macro_rules! impl_typed_for_scalar {
    ($ty:ty, $data_type:path) => {
        impl Typed<DataType> for $ty {
            fn r#type(&self) -> Cow<'_, DataType> {
                Cow::Owned($data_type)
            }
        }
    };
}

impl_typed_for_scalar!(bool, DataType::Boolean);
impl_typed_for_scalar!(i8, DataType::I8);
impl_typed_for_scalar!(i16, DataType::I16);
impl_typed_for_scalar!(i32, DataType::I32);
impl_typed_for_scalar!(i64, DataType::I64);
impl_typed_for_scalar!(u8, DataType::U8);
impl_typed_for_scalar!(u16, DataType::U16);
impl_typed_for_scalar!(u32, DataType::U32);
impl_typed_for_scalar!(u64, DataType::U64);
impl_typed_for_scalar!(bf16, DataType::BF16);
impl_typed_for_scalar!(f16, DataType::F16);
impl_typed_for_scalar!(f32, DataType::F32);
impl_typed_for_scalar!(f64, DataType::F64);

// Scalars are valid values for their `DataType`, mirroring their `Typed<DataType>` implementations, and so each
// scalar type implements `Value<DataType>`. Eager scalar interpretation needs no staging context, and so their
// `InterpretationContext` is `()`.
macro_rules! impl_value_for_scalar {
    ($ty:ty) => {
        impl Value<DataType> for $ty {
            type InterpretationContext = ();

            #[inline]
            fn interpretation_context(&self) -> Option<Self::InterpretationContext> {
                Some(())
            }
        }
    };
}

impl_value_for_scalar!(bool);
impl_value_for_scalar!(i8);
impl_value_for_scalar!(i16);
impl_value_for_scalar!(i32);
impl_value_for_scalar!(i64);
impl_value_for_scalar!(u8);
impl_value_for_scalar!(u16);
impl_value_for_scalar!(u32);
impl_value_for_scalar!(u64);
impl_value_for_scalar!(bf16);
impl_value_for_scalar!(f16);
impl_value_for_scalar!(f32);
impl_value_for_scalar!(f64);

/// Stateless [`Domain`] that uses [`DataType`] for scalar metadata and Rust scalar values such as `f32` for runtime
/// values. [`ScalarDomain`] is the minimal scalar-only backend used throughout tests and examples in `ryft-core`. It
/// demonstrates the intended role of an eager [`Context`] in the smallest possible form. There are no device handles,
/// no mesh states, and no backend registries. There are just the built-in [`ScalarOperation`] variants plus
/// [`DataType`]-driven construction of scalar values.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarDomain<V> {
    /// [`PhantomData`] marker that ties this zero-sized [`ScalarDomain`] to its scalar value type.
    marker: PhantomData<fn() -> V>,
}

impl<V> ScalarDomain<V> {
    /// Creates a new [`ScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

/// Stateless linear [`Domain`] for scalar tangent and cotangent [`Program`](crate::Program)s. This is the linear
/// complement of [`ScalarDomain`]. They both use the same scalar type (i.e., [`DataType`]) and the same runtime scalar
/// values (i.e., `f32`, `f64`, etc.). They differ only in the operation type selected by [`Domain`]:
///
/// - [`ScalarDomain`] records ordinary scalar programs using [`ScalarOperation`].
/// - [`LinearScalarDomain`] records linear tangent and cotangent programs using [`LinearScalarOperation`].
///
/// This separate domain is needed because [`Domain::Operation`] is an associated type. Once [`ScalarDomain`] says
/// "ordinary scalar traces store [`ScalarOperation`] instructions", the same domain type cannot also say "linear scalar
/// traces store [`LinearScalarOperation`] instructions". Automatic differentiation therefore keeps a tiny companion
/// domain for linear [`Program`](crate::Program)s.
///
/// For example, tracing `f(x) = x * x` with [`ScalarDomain<f64>`] records an ordinary multiplication. Linearizing that
/// program at `x = 3.0` produces a tangent program equivalent to `δx -> 3.0 * δx + 3.0 * δx`. That tangent program is
/// stored with [`LinearScalarOperation`] instructions such as `scale` and `add`. [`LinearScalarDomain`] is what tells
/// the generic tracing machinery to use that linear operation type instead of the standard operation type.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScalarDomain<V> {
    /// [`PhantomData`] marker that ties this zero-sized [`LinearScalarDomain`] to its scalar value type.
    marker: PhantomData<fn() -> V>,
}

impl<V> LinearScalarDomain<V> {
    /// Creates a new [`LinearScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_domain_for_scalar {
    ($ty:ty) => {
        impl Domain for ScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
            type Constant = $ty;
            type Operation = ScalarOperation<$ty>;
        }

        impl Domain for LinearScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
            type Constant = $ty;
            type Operation = LinearScalarOperation<$ty>;
        }
    };
}

impl_domain_for_scalar!(bool);
impl_domain_for_scalar!(i8);
impl_domain_for_scalar!(i16);
impl_domain_for_scalar!(i32);
impl_domain_for_scalar!(i64);
impl_domain_for_scalar!(u8);
impl_domain_for_scalar!(u16);
impl_domain_for_scalar!(u32);
impl_domain_for_scalar!(u64);
impl_domain_for_scalar!(bf16);
impl_domain_for_scalar!(f16);
impl_domain_for_scalar!(f32);
impl_domain_for_scalar!(f64);

// Eager [`Context`] support is provided through the operation's interpretation, so it is available only for the
// floating-point element types whose [`ScalarOperation`]/[`LinearScalarOperation`] enums are fully interpretable. The
// integer and boolean scalar domains intentionally remain trace-only [`Domain`]s because their operation enums include
// variants (such as negation, division, and the trigonometric primitives) that those element types do not support.
// Because [`Domain::Value`] equals [`Domain::Constant`] here, these are eager [`Context`]s whose bind interprets.
impl<V: Clone + Value<DataType>> Context for ScalarDomain<V>
where
    Self: Domain<Value = V, Constant = V, Operation: InterpretableOperation<DataType, V>>,
    <V as Value<DataType>>::InterpretationContext: Default,
{
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind<O: Into<Self::Operation>>(
        &self,
        operation: O,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        operation.into().interpret(&mut <V as Value<DataType>>::InterpretationContext::default(), inputs)
    }
}

impl<V: Clone + Value<DataType>> Context for LinearScalarDomain<V>
where
    Self: Domain<Value = V, Constant = V, Operation: InterpretableOperation<DataType, V>>,
    <V as Value<DataType>>::InterpretationContext: Default,
{
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind<O: Into<Self::Operation>>(
        &self,
        operation: O,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        operation.into().interpret(&mut <V as Value<DataType>>::InterpretationContext::default(), inputs)
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::constants::{OneOperation, ZeroOperation};

    use super::*;

    #[test]
    fn test_scalar_domain() {
        // Both [`ScalarDomain`] and [`LinearScalarDomain`] are zero-sized tokens.
        assert_eq!(size_of::<ScalarDomain<bool>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<bf16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f64>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<bool>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<i8>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<i16>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<i32>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<i64>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<u8>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<u16>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<u32>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<u64>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<bf16>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<f16>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<f32>>(), 0);
        assert_eq!(size_of::<LinearScalarDomain<f64>>(), 0);

        // Both are eager `Context`s for floating-point element types and so, binding a nullary zero/one operation
        // interprets it directly over concrete values, yielding the corresponding scalar identity.
        assert_eq!(ScalarDomain::<f64>::new().bind(ZeroOperation::new(DataType::F64), &[]), Ok(vec![0.0]));
        assert_eq!(ScalarDomain::<f64>::default().bind(OneOperation::new(DataType::F64), &[]), Ok(vec![1.0]));
        assert_eq!(LinearScalarDomain::<f64>::new().bind(ZeroOperation::new(DataType::F64), &[]), Ok(vec![0.0]));
        assert_eq!(LinearScalarDomain::<f64>::default().bind(OneOperation::new(DataType::F64), &[]), Ok(vec![1.0]));
    }
}
