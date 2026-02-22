use std::{
    fmt::Display,
    ops::{Mul, Neg},
};

use crate::{
    differentiation::JvpTracer,
    parameters::Parameter,
    programs::{InterpretableOp, Op, ProgramError},
    tracing::{TraceableOp, Tracer},
    types::Typed,
};

// ======================================================= SIN =======================================================

pub trait Sin {
    fn sin(self) -> Self;
}

impl Sin for f32 {
    fn sin(self) -> Self {
        self.sin()
    }
}

impl Sin for f64 {
    fn sin(self) -> Self {
        self.sin()
    }
}

impl<V: Clone + Parameter + Sin + Cos + Mul<T, Output = T>, T> Sin for JvpTracer<V, T> {
    fn sin(self) -> Self {
        Self { value: self.value.clone().sin(), tangent: self.value.cos() * self.tangent }
    }
}

#[derive(Clone, Debug)]
pub struct SinOp;

impl Display for SinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sin")
    }
}

impl<T: Clone> Op<T> for SinOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        // TODO(eaplatanios): Assert non-empty.
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Clone, V: Clone + Sin> InterpretableOp<T, V> for SinOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        Ok(vec![inputs[0].clone().sin()])
    }
}

// TODO(eaplatanios): Update the other macros below with similar logic.
macro_rules! impl_sin_for_tracer {
        ($ty:path) => { impl_sin_for_tracer!($ty, value_type_bounds = []); };
        ($ty:path, value_type_bounds = [$($value_type_bounds:path),*]) => {
            impl<T: Clone, V: Clone + Display + Sin + Typed<T> $(+$value_type_bounds)*> Sin for Tracer<T, V, Box<dyn $ty>> {
                fn sin(self) -> Self {
                    let inputs = vec![&self];
                    let outputs = (Box::new(SinOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                    debug_assert_eq!(outputs.len(), 1);
                    outputs[0].clone()
                }
            }
        };
    }

// TODO(eaplatanios): We need to also add support for handling constants like `Tracer + Tensor`.
// TODO(eaplatanios): `Box<dyn Op<T>>` is way too generic. What can we do with this?
impl_sin_for_tracer!(Op<T>);
impl_sin_for_tracer!(InterpretableOp<T, V>, value_type_bounds = [Clone, Sin]);

// ======================================================= COS =======================================================

pub trait Cos {
    fn cos(self) -> Self;
}

impl Cos for f32 {
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for f64 {
    fn cos(self) -> Self {
        self.cos()
    }
}

impl<V: Clone + Parameter + Sin + Cos + Neg<Output = V> + Mul<T, Output = T>, T> Cos for JvpTracer<V, T> {
    fn cos(self) -> Self {
        Self { value: self.value.clone().cos(), tangent: -self.value.sin() * self.tangent }
    }
}

#[derive(Clone, Debug)]
pub struct CosOp;

impl Display for CosOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cos")
    }
}

impl<T: Clone> Op<T> for CosOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        // TODO(eaplatanios): Assert non-empty.
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Clone, V: Clone + Cos> InterpretableOp<T, V> for CosOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        Ok(vec![inputs[0].clone().cos()])
    }
}

// TODO(eaplatanios): Update the other macros below with similar logic.
macro_rules! impl_cos_for_tracer {
        ($ty:path) => { impl_cos_for_tracer!($ty, value_type_bounds = []); };
        ($ty:path, value_type_bounds = [$($value_type_bounds:path),*]) => {
            impl<T: Clone, V: Clone + Display + Cos + Typed<T> $(+$value_type_bounds)*> Cos for Tracer<T, V, Box<dyn $ty>> {
                fn cos(self) -> Self {
                    let inputs = vec![&self];
                    let outputs = (Box::new(CosOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                    debug_assert_eq!(outputs.len(), 1);
                    outputs[0].clone()
                }
            }
        };
    }

// TODO(eaplatanios): We need to also add support for handling constants like `Tracer + Tensor`.
// TODO(eaplatanios): `Box<dyn Op<T>>` is way too generic. What can we do with this?
impl_cos_for_tracer!(Op<T>);
impl_cos_for_tracer!(InterpretableOp<T, V>, value_type_bounds = [Clone, Sin]);
