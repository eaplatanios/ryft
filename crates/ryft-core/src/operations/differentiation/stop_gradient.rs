use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
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
/// tangent with a structural zero, so that no derivative flows through the marked value in either forward or reverse
/// automatic differentiation. Because the rule stages only that zero tangent, `stop_gradient` can never appear on a
/// linear operand in a valid tangent program, and its
/// [`TransposableOperation`](crate::TransposableOperation) implementation reports an error.
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

impl_differentiable_elementwise_operation!(@non_differentiable StopGradientOperation);

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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{jvp, value_and_gradient};
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{Shape, Size};

    use super::*;

    #[test]
    fn test_stop_gradient() {
        let operation = StopGradientOperation;

        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, -2.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(1.0f64, -2.0))]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(2.0)]),
        );
    }

    #[test]
    fn test_stop_gradient_type_inference() {
        // Gradient stopping is the identity on every data type, including types the numeric operations reject.
        for input_type in [DataType::Token, DataType::Boolean, DataType::I32, DataType::C64] {
            check_operation_type_inference!(
                @elementwise @unary,
                operation = StopGradientOperation,
                cases = [{
                    input_data_types = [input_type],
                    output_data_types = [input_type],
                }],
            );
        }

        // Partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = StopGradientOperation,
            cases = [{
                input_types = [unreduced.clone()],
                output_types = [unreduced],
            }],
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = StopGradientOperation,
            cases = [{
                input_types = [reduced.clone()],
                output_types = [reduced],
            }],
        );
    }

    #[test]
    fn test_stop_gradient_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = StopGradientOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
            }],
        );
    }

    #[test]
    fn test_stop_gradient_composes_with_batching() {
        // Gradient stopping composes with batching: `x * stop_gradient(x)` batches like `x * x`.
        let output: Array = batch(
            |x| Ok(x.clone() * x.stop_gradient()),
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_stop_gradient_differentiation() {
        // The JVP passes the primal through and severs the tangent. This intentionally differs from the numerical
        // derivative of the identity primal function, so the finite-difference operation helper does not apply.
        let (primal, tangent) = jvp(|x| Ok(x.stop_gradient()), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_eq!(primal, 2.0);
        assert_eq!(tangent, 0.0);

        // The JAX documentation example: `f(x) = x * stop_gradient(x)` differentiates like `x * c` with `c` frozen
        // at the primal value, so `f'(x) = stop_gradient(x)`.
        let (value, gradient) = value_and_gradient(|x| x.clone() * x.stop_gradient(), Scalar::from(3.0)).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 3.0);

        // The staged tangent program replays the primal operation and stages no tangent computation: the severed
        // tangent output materializes as a canonical zero.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(StopGradientOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = stop_gradient %0
                    %3:f64[] = zero [type=f64[]]
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_stop_gradient_partial_evaluation() {
        check_operation_partial_evaluation!(operation = StopGradientOperation, inputs = [2.0], expected = 2.0,);
    }

    #[test]
    fn test_stop_gradient_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = StopGradientOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
