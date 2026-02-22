use std::fmt::Display;

use half::{bf16, f16};

use crate::{
    differentiation::JvpTracer,
    programs::{Constant, ConstantExpression, Program, ProgramType},
    tracing::{Tracer, VariableTracer},
};

use super::array_type::{ArrayType, DataType};

// Our tracing module defines its own [DataType] and [Shape] types that ought to be decoupled from the corresponding
// types that may be used by different kinds of [Parameter]s. This enables us to support multiple different backends
// for just-in-time compilation and numerical computation, some of which may even support statically typed tensors
// with both their data type and shape being checked at compile time.

pub trait Type: Display {
    fn is_subtype_of(&self, other: &Self) -> bool;
}

impl<ValueType: Type, TangentType: Type> Type for JvpTracer<ValueType, TangentType> {
    fn is_subtype_of(&self, other: &Self) -> bool {
        self.value.is_subtype_of(&other.value) && self.tangent.is_subtype_of(&other.tangent)
    }
}

pub trait Typed<T> {
    fn tpe(&self) -> T;
}

macro_rules! impl_typed_for_scalar {
    ($ty:ty, $data_type:path) => {
        impl Typed<ArrayType> for $ty {
            fn tpe(&self) -> ArrayType {
                ArrayType::scalar($data_type)
            }
        }
    };
}

impl_typed_for_scalar!(bool, DataType::Boolean);
impl_typed_for_scalar!(i8, DataType::Int8);
impl_typed_for_scalar!(i16, DataType::Int16);
impl_typed_for_scalar!(i32, DataType::Int32);
impl_typed_for_scalar!(i64, DataType::Int64);
impl_typed_for_scalar!(u8, DataType::UnsignedInt8);
impl_typed_for_scalar!(u16, DataType::UnsignedInt16);
impl_typed_for_scalar!(u32, DataType::UnsignedInt32);
impl_typed_for_scalar!(u64, DataType::UnsignedInt64);
impl_typed_for_scalar!(bf16, DataType::BFloat16);
impl_typed_for_scalar!(f16, DataType::Float16);
impl_typed_for_scalar!(f32, DataType::Float32);
impl_typed_for_scalar!(f64, DataType::Float64);

impl<ValueType: Clone, TangentType: Clone, Value: Typed<ValueType>, Tangent: Typed<TangentType>>
    Typed<JvpTracer<ValueType, TangentType>> for JvpTracer<Value, Tangent>
{
    fn tpe(&self) -> JvpTracer<ValueType, TangentType> {
        JvpTracer { value: self.value.tpe(), tangent: self.tangent.tpe() }
    }
}

impl<T, V: Typed<T>> Typed<T> for Constant<V> {
    fn tpe(&self) -> T {
        self.value.tpe()
    }
}

impl<T: Clone, V, O> Typed<T> for ConstantExpression<T, V, O> {
    fn tpe(&self) -> T {
        match &self {
            ConstantExpression::Value { tpe, .. } => tpe.clone(),
            ConstantExpression::Expression { tpe, .. } => tpe.clone(),
        }
    }
}

impl<T: Clone, V, O> Typed<T> for VariableTracer<T, V, O> {
    fn tpe(&self) -> T {
        self.builder.borrow().atom_type(&self.id).unwrap().clone()
    }
}

impl<T: Clone, V: Typed<T>, O> Typed<T> for Tracer<T, V, O> {
    fn tpe(&self) -> T {
        match &self {
            Tracer::Constant(constant) => constant.tpe(),
            Tracer::Variable(variable) => variable.tpe(),
        }
    }
}

impl<T: Clone, V: Typed<T>, O> Typed<ProgramType<T>> for Program<T, V, O> {
    fn tpe(&self) -> ProgramType<T> {
        ProgramType {
            inputs: self.inputs.iter().map(|input| input.tpe().clone()).collect(),
            outputs: self.outputs.iter().map(|output| output.tpe().clone()).collect(),
        }
    }
}
