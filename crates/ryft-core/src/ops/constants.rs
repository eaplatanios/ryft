use half::{bf16, f16};
use paste::paste;

use crate::{
    differentiation::{Differentiable, JvpTracer},
    programs::ConstantExpression,
    tracing::Tracer,
    types::{ArrayType, Type, Typed},
};

/// TODO(eaplatanios): For tracing purposes (primarily) we want to support lazy evaluation of certain types of constant
///  expressions. Specifically, we want to have a notion of a `ConstantExpression` that can be appropriately converted
///  to a program when tracing. That is because we want to avoid materializing large values (or values that are
///  expensive to compute) and storing them as program constants. For example, `tensor.ones([5000, 5000])` should not
///  store a constant with 25M elements in the program.

// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct ConstantOp<Value> {
//     pub value: Value,
// }

// impl<Type, Value: Debug + Typed<Type>> Op<Type> for ConstantOp<Value> {
//     #[inline]
//     fn render(&self) -> String {
//         todo!()
//     }

//     #[inline]
//     fn input_count(&self) -> Option<usize> {
//         Some(0)
//     }

//     #[inline]
//     fn output_count(&self) -> Option<usize> {
//         Some(1)
//     }

//     #[inline]
//     fn infer_output_types(&self, input_types: &[&Type]) -> Result<Vec<Type>, ProgramError> {
//         if input_types.len() > 0 {
//             Err(ProgramError::InvalidInputCount { expected: 0, got: input_types.len() })
//         } else {
//             Ok(vec![self.value.tpe()])
//         }
//     }
// }

// impl<Type, Value: Clone + Debug + Typed<Type>> InterpretableOp<Type, Value> for ConstantOp<Value> {
//     #[inline]
//     fn interpret(&self, inputs: &[&Value]) -> Result<Vec<Value>, ProgramError> {
//         if inputs.len() > 0 {
//             Err(ProgramError::InvalidInputCount { expected: 0, got: inputs.len() })
//         } else {
//             Ok(vec![self.value.clone()])
//         }
//     }
// }
//
// impl<Value> Expression<ConstantOp<Value>> {
//     #[inline]
//     pub fn constant(id: AtomId, value: Value) -> Self {
//         Self { op: ConstantOp { value }, inputs: Vec::new(), outputs: vec![id] }
//     }
// }

pub trait Constant<V> {
    fn constant(value: V) -> Self;
}

impl Constant<f32> for f32 {
    fn constant(value: f32) -> Self {
        value
    }
}

impl<V: Constant<f32>, T: Zero> Constant<f32> for JvpTracer<V, T>
where
    V: Differentiable<T> + Typed<T::T>,
{
    fn constant(value: f32) -> Self {
        V::constant(value).into()
    }
}

pub trait Zero: Typed<Self::T> {
    type T: Type;

    fn zero(tpe: &Self::T) -> Self;
}

impl Zero for bool {
    type T = ArrayType;

    fn zero(_: &Self::T) -> Self {
        false
    }
}

macro_rules! zero_primitive_impl {
    ($ty:ty) => {
        paste! {
            impl Zero for $ty {
                type T = ArrayType;

                fn zero(_: &Self::T) -> Self {
                    [<0 $ty>]
                }
            }
        }
    };
}

zero_primitive_impl!(i8);
zero_primitive_impl!(i16);
zero_primitive_impl!(i32);
zero_primitive_impl!(i64);
zero_primitive_impl!(u8);
zero_primitive_impl!(u16);
zero_primitive_impl!(u32);
zero_primitive_impl!(u64);
zero_primitive_impl!(f32);
zero_primitive_impl!(f64);

impl Zero for bf16 {
    type T = ArrayType;

    fn zero(_: &Self::T) -> Self {
        bf16::ZERO
    }
}

impl Zero for f16 {
    type T = ArrayType;

    fn zero(_: &Self::T) -> Self {
        f16::ZERO
    }
}

// impl Zero for Placeholder {
//     type T = ArrayType;
//
//     fn zero(_: &Self::T) -> Self {
//         Self
//     }
// }

impl<V: Zero<T: Clone>, VT: Zero<T: Clone>> Zero for JvpTracer<V, VT> {
    type T = JvpTracer<V::T, VT::T>;

    fn zero(tpe: &Self::T) -> Self {
        Self { value: V::zero(&tpe.value), tangent: VT::zero(&tpe.tangent) }
    }
}

impl<T: Clone + Type, V: Zero<T = T>, O> Zero for Tracer<T, V, O> {
    type T = V::T;

    fn zero(tpe: &Self::T) -> Self {
        let value = V::zero(tpe);
        let tpe = value.tpe();
        Self::Constant(ConstantExpression::Value { tpe, value })
    }
}

pub trait One: Typed<Self::T> {
    type T: Type;

    fn one(tpe: &Self::T) -> Self;
}

impl One for bool {
    type T = ArrayType;

    fn one(_: &Self::T) -> Self {
        true
    }
}

macro_rules! one_primitive_impl {
    ($ty:ty) => {
        paste! {
            impl One for $ty {
                type T = ArrayType;

                fn one(_: &Self::T) -> Self {
                    [<1 $ty>]
                }
            }
        }
    };
}

one_primitive_impl!(i8);
one_primitive_impl!(i16);
one_primitive_impl!(i32);
one_primitive_impl!(i64);
one_primitive_impl!(u8);
one_primitive_impl!(u16);
one_primitive_impl!(u32);
one_primitive_impl!(u64);
one_primitive_impl!(f32);
one_primitive_impl!(f64);

impl One for bf16 {
    type T = ArrayType;

    fn one(_: &Self::T) -> Self {
        bf16::ZERO
    }
}

impl One for f16 {
    type T = ArrayType;

    fn one(_: &Self::T) -> Self {
        f16::ZERO
    }
}

// impl One for Placeholder {
//     type T = ArrayType;
//
//     fn one(_: &Self::T) -> Self {
//         Self
//     }
// }

impl<V: One<T: Clone>, VT: Zero<T: Clone>> One for JvpTracer<V, VT> {
    type T = JvpTracer<V::T, VT::T>;

    fn one(tpe: &Self::T) -> Self {
        Self { value: V::one(&tpe.value), tangent: VT::zero(&tpe.tangent) }
    }
}

impl<T: Clone + Type, V: One<T = T>, O> One for Tracer<T, V, O> {
    type T = T;

    fn one(tpe: &Self::T) -> Self {
        let value = V::one(tpe);
        let tpe = value.tpe();
        Self::Constant(ConstantExpression::Value { tpe, value })
    }
}
