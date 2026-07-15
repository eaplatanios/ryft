use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::types::{ArrayType, DataType};

/// Canonical operation name for [`StopGradientOperation`].
pub const STOP_GRADIENT_OPERATION_NAME: &str = "stop_gradient";

// TODO(eaplatanios): Review this module.

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
    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
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
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for StopGradientOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain> InterpretableOperation<C> for StopGradientOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context> PartiallyEvaluatableOperation<C> for StopGradientOperation where
    C::Operation: From<StopGradientOperation>
{
}

/// Value-level gradient stopping capability. [`StopGradient`] fills the same role for [`StopGradientOperation`]
/// that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait StopGradient: Sized {
    /// Returns this value unchanged while marking it as a constant for differentiation purposes.
    fn stop_gradient(&self) -> Self;
}

/// Any context-carrying value stops gradients by binding a [`StopGradientOperation`] through its own context: a
/// staged tracer records the operation, while batching / JVP tracers apply their transform rules. The
/// `From<StopGradientOperation>` bound makes this blanket disjoint from the concrete eager value types (whose context
/// operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), which implement
/// [`StopGradient`] directly.
impl<V: Value> StopGradient for V
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<StopGradientOperation>,
{
    #[inline]
    fn stop_gradient(&self) -> Self {
        self.dispatch_domain()
            .bind(StopGradientOperation, Vec::new(), &[self.clone()])
            .expect("`stop_gradient` operation failed")
            .remove(0)
    }
}

// TODO(eaplatanios): Add unit tests.
