//! Legacy v0 typed-value helpers used by the differentiation stack.

use half::{bf16, f16};

use crate::{
    programs::{Constant, ConstantExpression, Program, ProgramType},
    tracing_v0::{Tracer, VariableTracer},
    types::data_type::DataType,
    types::{ArrayType, Type, Typed},
};

use super::JvpTracer;

// Our tracing module defines its own [DataType] and [Shape] types that ought to be decoupled from the corresponding
// types that may be used by different kinds of [Parameter]s. This enables us to support multiple different backends
// for just-in-time compilation and numerical computation, some of which may even support statically typed tensors
// with both their data type and shape being checked at compile time.

impl<ValueType: Type, TangentType: Type> Type for JvpTracer<ValueType, TangentType> {
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.value.is_compatible_with(&other.value) && self.tangent.is_compatible_with(&other.tangent)
    }
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

impl<ValueType: Clone + Type, TangentType: Clone + Type, Value: Typed<ValueType>, Tangent: Typed<TangentType>>
    Typed<JvpTracer<ValueType, TangentType>> for JvpTracer<Value, Tangent>
{
    fn tpe(&self) -> JvpTracer<ValueType, TangentType> {
        JvpTracer { value: self.value.tpe(), tangent: self.tangent.tpe() }
    }
}

impl<T: Type, V: Typed<T>> Typed<T> for Constant<V> {
    fn tpe(&self) -> T {
        self.value.tpe()
    }
}

impl<T: Clone + Type, V, O> Typed<T> for ConstantExpression<T, V, O> {
    fn tpe(&self) -> T {
        match &self {
            ConstantExpression::Value { tpe, .. } => tpe.clone(),
            ConstantExpression::Expression { tpe, .. } => tpe.clone(),
        }
    }
}

impl<T: Clone + Type, V, O> Typed<T> for VariableTracer<T, V, O> {
    fn tpe(&self) -> T {
        self.builder.borrow().atom_type(&self.id).unwrap().clone()
    }
}

impl<T: Clone + Type, V: Typed<T>, O> Typed<T> for Tracer<T, V, O> {
    fn tpe(&self) -> T {
        match &self {
            Tracer::Constant(constant) => constant.tpe(),
            Tracer::Variable(variable) => variable.tpe(),
        }
    }
}

impl<T: Clone + Type, V: Typed<T>, O> Typed<ProgramType<T>> for Program<T, V, O> {
    fn tpe(&self) -> ProgramType<T> {
        ProgramType {
            inputs: self.inputs.iter().map(|input| input.tpe().clone()).collect(),
            outputs: self.outputs.iter().map(|output| output.tpe().clone()).collect(),
        }
    }
}
