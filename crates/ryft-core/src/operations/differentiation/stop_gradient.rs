use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::ProgramError;
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

// TODO(eaplatanios): Review this file.

/// Canonical operation name for [`StopGradientOperation`].
pub const STOP_GRADIENT_OPERATION_NAME: &'static str = "stop_gradient";

/// [`Operation`] that returns its input unchanged while severing gradient flow — the direct analogue of JAX's
/// [`lax.stop_gradient`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html).
///
/// Interpretation, batching, and backend lowering all treat this operation as the identity. Differentiation does not:
/// its JVP rule passes the primal through unchanged and replaces the tangent with a symbolic
/// [`Tangent::Zero`](crate::differentiation::Tangent), so no derivative flows through the marked value in either
/// forward or reverse mode. Because the rule never stages a linear operation, `stop_gradient` cannot appear in
/// pushforward programs and therefore needs no transpose rule.
#[derive(Clone, Debug, Default)]
pub struct StopGradientOperation;

impl Display for StopGradientOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(STOP_GRADIENT_OPERATION_NAME)
    }
}

impl Operation<DataType> for StopGradientOperation {
    #[inline]
    fn name(&self) -> &'static str {
        STOP_GRADIENT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl ElementwiseOperation for StopGradientOperation {
    #[inline]
    fn name(&self) -> &'static str {
        STOP_GRADIENT_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType>> InterpretableOperation<DataType, V> for StopGradientOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

impl<V: Clone + Typed<ArrayType>> InterpretableOperation<ArrayType, V> for StopGradientOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`StopGradientOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`StopGradientOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsStopGradient<T: Type> {
    /// Constructs an instance of [`StopGradientOperation`] for this [`Operation`] type.
    fn stop_gradient_operation() -> Self;
}

/// Value-level gradient-severing capability. [`StopGradient`] fills the same role for [`StopGradientOperation`] that
/// [`Sin`](crate::operations::trigonometric::Sin) fills for [`SinOperation`](crate::operations::trigonometric): on
/// concrete values it is the identity, while on traced values it stages a [`StopGradientOperation`] whose JVP severs
/// the tangent.
pub trait StopGradient: Sized {
    /// Returns this value unchanged while marking it as a constant for differentiation.
    fn stop_gradient(self) -> Self;
}

macro_rules! impl_stop_gradient_identity {
    ($($ty:ty),* $(,)?) => {
        $(
            impl StopGradient for $ty {
                #[inline]
                fn stop_gradient(self) -> Self {
                    self
                }
            }
        )*
    };
}

impl_stop_gradient_identity!(bf16, f16, f32, f64);

impl<C: StagingContext<Operation: SupportsStopGradient<C::Type>>> StopGradient for Tracer<C> {
    #[inline]
    fn stop_gradient(self) -> Self {
        self.unary(C::Operation::stop_gradient_operation())
    }
}
