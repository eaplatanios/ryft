use std::fmt::Display;
use std::ops::Neg;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::sin::Sin;
use super::{SupportsNeg, SupportsScale};

/// Trait that represents [`Operation`] carrier types that support/include [`CosOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`CosOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsCos<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the cosine [`Operation`].
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
#[derive(Clone, Debug, Default)]
pub struct CosOperation;

impl Display for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl<T: Type> Operation<T> for CosOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "cos"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<V: Clone + Typed<ArrayType> + Cos> InterpretableOperation<ArrayType, V> for CosOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V: Clone + Typed<DataType> + Cos> InterpretableOperation<DataType, V> for CosOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<E> DifferentiableOperation<E> for CosOperation
where
    E: DifferentiableEngine,
    CosOperation: Operation<E::Type>,
    E::Value: Cos + Sin + Neg<Output = E::Value> + Differentiable<E::Type>,
    <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier:
        SupportsNeg<E::Type, E::Tangent> + SupportsScale<E::Type, E::Tangent, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let scaled_outputs = context.stage(
            <<E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier as SupportsScale<
                E::Type,
                E::Tangent,
                E::Value,
            >>::scale_operation(input.primal.clone().sin()),
            &[input.tangent],
        )?;
        check_count!("output", scaled_outputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier as SupportsNeg<
                E::Type,
                E::Tangent,
            >>::neg_operation(),
            &[scaled_outputs[0]],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: input.primal.clone().cos(), tangent: tangent_outputs[0] }])
    }
}

impl<'engine, E> Cos for Tracer<'engine, E>
where
    E: TracingEngine,
    E::Value: Cos,
    E::OperationCarrier: SupportsCos<E::Type, E::Value>,
{
    #[inline]
    fn cos(self) -> Self {
        self.unary(E::OperationCarrier::cos_operation())
    }
}
