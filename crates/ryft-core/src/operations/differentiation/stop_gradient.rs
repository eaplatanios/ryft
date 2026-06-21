use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`StopGradientOperation`].
pub const STOP_GRADIENT_OPERATION_NAME: &'static str = "stop_gradient";

// TODO(eaplatanios): Link to [`Pushforward`].
/// [`Operation`] that returns its input unchanged while severing gradient flow/propagation. Interpretation,
/// batching, and backend lowering all treat this operation as the identity function, but differentiation does not.
/// The Jacobian-Vector Product (JVP) rule of this operation passes the primal through unchanged and replaces the
/// tangent with a canonical staged [`ZeroOperation`](crate::ZeroOperation), so that no derivative flows through the
/// marked value in either forward or reverse automatic differentiation. Because the rule stages only that canonical
/// zero tangent, `stop_gradient` cannot appear in pushforward programs and therefore needs no (and has no)
/// [`TransposableOperation`](crate::TransposableOperation) implementation.
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

impl Operation<ArrayType> for StopGradientOperation {
    #[inline]
    fn name(&self) -> &'static str {
        STOP_GRADIENT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for StopGradientOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<T: Type, V: Clone + Value<T>> InterpretableOperation<T, V> for StopGradientOperation
where
    Self: Operation<T>,
{
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

/// Value-level gradient stopping capability. [`StopGradient`] fills the same role for [`StopGradientOperation`]
/// that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait StopGradient: Sized {
    /// Returns this value unchanged while marking it as a constant for differentiation purposes.
    fn stop_gradient(&self) -> Self;
}

macro_rules! impl_stop_gradient_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl StopGradient for $ty {
                #[inline]
                fn stop_gradient(&self) -> Self {
                    *self
                }
            }
        )*
    };
}

impl_stop_gradient_for_scalar!(bf16, f16, f32, f64);

impl<C: StagingContext<Operation: From<StopGradientOperation>>> StopGradient for Tracer<C> {
    #[inline]
    fn stop_gradient(&self) -> Self {
        self.unary(StopGradientOperation)
    }
}

// TODO(eaplatanios): Add unit tests.
