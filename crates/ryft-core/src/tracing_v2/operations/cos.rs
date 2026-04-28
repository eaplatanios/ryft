use std::fmt::{Debug, Display};
use std::ops::Neg;

use half::{bf16, f16};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::DifferentiableEngine;
use crate::tracing_v2::engines::{StagingEngine, Tracer};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::sin::Sin;
use super::{DifferentiableOperation, InterpretableOperation, Operation, SupportsNeg, SupportsScale, unary_abstract};

/// Hidden carrier capability for staging the cosine primitive.
#[doc(hidden)]
pub trait SupportsCos<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the cosine primitive.
    fn cos_operation() -> Self;
}

/// Elementwise cosine capability.
///
/// This trait is implemented by concrete values and the transform-local wrappers that can represent
/// cosine symbolically.
pub trait Cos: Sized {
    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
}

impl Cos for f32 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for f64 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for bf16 {
    #[inline]
    fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }
}

impl Cos for f16 {
    #[inline]
    fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }
}

/// Elementwise cosine primitive.
///
/// [`CosOperation`] is stored in staged programs whenever cosine is traced explicitly, and its JVP rule is
/// reused by both forward-mode evaluation and higher-order traced transforms.
#[derive(Clone, Default)]
pub struct CosOperation;

impl Debug for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Cos")
    }
}

impl Display for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cos")
    }
}

impl Operation<ArrayType> for CosOperation {
    fn name(&self) -> &'static str {
        "cos"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![unary_abstract(input_types)?])
    }
}

impl<V: Typed<ArrayType> + Clone + Cos> InterpretableOperation<ArrayType, V> for CosOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<E> DifferentiableOperation<E> for CosOperation
where
    E: DifferentiableEngine + ?Sized,
    CosOperation: Operation<E::Type>,
    E::Value: Cos + Sin + Neg<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsNeg<E::Type, E::Value> + SupportsScale<E::Type, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
        let input = &inputs[0];
        let scaled = context
            .apply_operation(
                &[input.tangent],
                <E::LinearOperation as SupportsScale<E::Type, E::Value>>::scale_operation(input.primal.clone().sin()),
                1,
            )?
            .into_iter()
            .next()
            .expect("cos jvp scale should produce one tangent");
        let tangent = context
            .apply_operation(&[scaled], <E::LinearOperation as SupportsNeg<E::Type, E::Value>>::neg_operation(), 1)?
            .into_iter()
            .next()
            .expect("cos jvp neg should produce one tangent");
        Ok(vec![JvpTracer { primal: input.primal.clone().cos(), tangent }])
    }
}

impl<'engine, V: Traceable<ArrayType> + Cos, E> Cos for Tracer<'engine, E>
where
    E: StagingEngine<Type = ArrayType, Value = V> + ?Sized,
    E::Operation: SupportsCos<ArrayType, V>,
{
    #[inline]
    fn cos(self) -> Self {
        self.unary(E::Operation::cos_operation())
    }
}
