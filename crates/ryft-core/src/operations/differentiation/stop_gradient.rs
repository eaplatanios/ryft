use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, DataType};
use crate::batching::{BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::parameters::{Parameter, Parameterized};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{Operation, ProgramError, RegionInterface, Type, TypeError, Typed, Value};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`StopGradientOperation`].
pub const STOP_GRADIENT_OPERATION_NAME: &str = "stop_gradient";

/// [`Operation`] that returns its input unchanged while severing gradient flow/propagation. The `T` parameter fixes
/// the operation's type universe at construction time, so each zero-sized payload implements exactly one [`Operation`]
/// contract. Interpretation and backend lowering treat this operation as the identity function. Batching preserves
/// its mapped axis and rebinds the barrier through the parent transform. The Jacobian-Vector Product (JVP) rule passes
/// the primal through unchanged and replaces the tangent with a structural zero, so that no derivative flows through
/// the marked value in either forward or reverse automatic differentiation. Because the rule stages only that zero
/// tangent, `stop_gradient` can never appear on a linear operand in a valid tangent program, and its
/// [`TransposableOperation`](crate::TransposableOperation) implementation reports an error.
#[derive(Clone, Debug)]
pub struct StopGradientOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> StopGradientOperation<T> {
    /// Constructs a stop-gradient operation for the `T` type universe.
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Default for StopGradientOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type> Display for StopGradientOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(STOP_GRADIENT_OPERATION_NAME)
    }
}

impl Operation for StopGradientOperation<DataType> {
    type Type = DataType;

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

impl Operation for StopGradientOperation<ArrayType> {
    type Type = ArrayType;

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
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, C: Domain<Type = T>> InterpretableOperation<C> for StopGradientOperation<T>
where
    Self: Operation<Type = C::Type>,
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
impl<T: Type, C: Context<Type = T>> PartiallyEvaluatableOperation<C> for StopGradientOperation<T>
where
    C::Operation: From<StopGradientOperation<T>>,
    Self: Operation<Type = T>,
{
}

/// Batching preserves the operand's mapped axis while recursively rebinding the gradient barrier through the parent
/// [`Context`]. Rebinding is essential when the parent value is itself a differentiation or batching tracer: treating
/// the packed value as an ordinary interpreted identity would clone that tracer and silently expose its tangent to an
/// enclosing transform.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for StopGradientOperation<ArrayType>
where
    C::Operation: From<StopGradientOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let mut outputs = context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(input.value()))?;
        check_count!("output", outputs, 1, ProgramError);
        let output = outputs.remove(0);
        Ok(vec![ArrayBatch::new(output.r#type().into_owned(), output, input.batch_axis())?])
    }
}

impl_non_differentiable_operation!(<T> StopGradientOperation<T> where T: Type);
impl_non_transposable_operation!(<T> StopGradientOperation<T> where T: Type);

/// Value-level gradient stopping capability. [`StopGradient`] fills the same role for [`StopGradientOperation`]
/// that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait StopGradient: Sized {
    /// Returns this value unchanged while marking it as a constant for differentiation purposes.
    fn stop_gradient(&self) -> Self;
}

/// Stops gradient propagation through every leaf in `values` while preserving its exact [`Parameterized`] structure.
/// This is the structure-aware counterpart of [`StopGradient::stop_gradient`]. It accepts nested tuples, vectors, and
/// custom parameterized types, and returns an unchanged primal structure whose leaves are constants to every enclosing
/// differentiation transform.
pub fn stop_gradient<V: Parameter + StopGradient, Values: Parameterized<V>>(mut values: Values) -> Values {
    for value in values.parameters_mut() {
        *value = value.stop_gradient();
    }
    values
}

/// Any context-carrying value stops gradients by binding a [`StopGradientOperation`] through its own context: a
/// staged tracer records the operation, while batching / JVP tracers apply their transform rules. The
/// `From<StopGradientOperation<V::Type>>` bound makes this blanket disjoint from the concrete eager value types (whose
/// context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), which implement
/// [`StopGradient`] directly.
impl<V: Value> StopGradient for V
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<StopGradientOperation<V::Type>>,
{
    #[inline]
    fn stop_gradient(&self) -> Self {
        self.dispatch_domain()
            .bind(StopGradientOperation::new(), Vec::new(), &[self.clone()])
            .expect("`stop_gradient` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayOperation, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use crate::backends::Array;
    use crate::batching::{BatchAxis, BatchedProgram, ProgramBatchingOutputAxesPolicy, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{LinearizationTracer, gradient, jvp, value_and_gradient};
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    #[test]
    fn test_stop_gradient() {
        let array_operation = StopGradientOperation::<ArrayType>::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &array_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &array_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(Complex::new(1.0f64, -2.0))],
            ),
            Ok(vec![Array::scalar(Complex::new(1.0f64, -2.0))]),
        );

        let input = Array::vector(vec![1.0, -2.0]);
        assert_eq!(input.stop_gradient(), input);
    }

    #[test]
    fn test_stop_gradient_parameterized_structure() {
        let values = (
            Array::scalar(1.0f32),
            vec![Array::scalar(2_i32), Array::scalar(Complex::new(3.0f64, -4.0)), Array::scalar(true)],
        );
        assert_eq!(crate::stop_gradient(values.clone()), values);

        assert!(crate::stop_gradient(Vec::<Array>::new()).is_empty());
        assert_eq!(crate::stop_gradient::<Array, _>(()), ());

        let first_derivative = gradient(
            |input| {
                let stopped = crate::stop_gradient((input.clone(), vec![input.clone(), input]));
                stopped.0 + stopped.1[0].clone() + stopped.1[1].clone()
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_eq!(first_derivative, Array::scalar(0.0));

        let second_derivative = gradient(
            |input| {
                gradient(
                    |inner| {
                        let stopped = crate::stop_gradient((inner.clone(), vec![inner.clone(), inner]));
                        stopped.0 + stopped.1[0].clone() + stopped.1[1].clone()
                    },
                    input,
                )
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_eq!(second_derivative, Array::scalar(0.0));
    }

    #[test]
    fn test_stop_gradient_type_inference() {
        // Gradient stopping is the identity on every data type, including types the numeric operations reject.
        for input_type in [DataType::Token, DataType::Boolean, DataType::I32, DataType::C64] {
            let input_type = ArrayType::new(input_type, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
            check_operation_type_inference!(
                operation = StopGradientOperation::<ArrayType>::new(),
                cases = [{
                    input_types = [input_type.clone()],
                    output_types = [input_type],
                }],
            );
        }

        // Partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = StopGradientOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced.clone()],
                output_types = [unreduced],
            }],
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = StopGradientOperation::<ArrayType>::new(),
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
            operation = StopGradientOperation::<ArrayType>::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
            }],
        );

        // Program batching must preserve the barrier in the staged physical program so later differentiation still
        // sees it. The blanket elementwise batching rule would interpret the operation as an identity and erase it.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(StopGradientOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[2] = stop_gradient %0
                in (%1)
            "}
            .trim_end(),
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

        // When batching is nested inside forward mode, the batch rule must rebind the barrier through the surrounding
        // differentiation context instead of cloning the packed differentiation tracer.
        let (primal, tangent): (Array, Array) = jvp(
            |x| Ok(batch(|item| Ok(item.stop_gradient()), x, BatchAxis::new(0), BatchAxis::new(0), None)?),
            Array::vector(vec![2.0, 3.0]),
            Array::vector(vec![5.0, 7.0]),
        )
        .unwrap();
        assert_eq!(primal.to_f64s(), vec![2.0, 3.0]);
        assert_eq!(tangent.to_f64s(), vec![0.0, 0.0]);

        // Reverse mode exercises the same transform order. Each item differentiates as `x * c`, where the stopped
        // factor `c` is frozen at that item's primal value.
        let (value, gradient): (Array, Array) = value_and_gradient(
            |x| {
                let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |item| Ok(item.clone() * item.stop_gradient()),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?;
                Ok::<_, ProgramError>(mapped.reduce(&[0], ReductionKind::Sum))
            },
            Array::vector(vec![2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.to_f64s(), vec![13.0]);
        assert_eq!(gradient.to_f64s(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_stop_gradient_differentiation() {
        // The JVP passes the primal through and severs the tangent. This intentionally differs from the numerical
        // derivative of the identity primal function, so the finite-difference operation helper does not apply.
        let (primal, tangent) = jvp(|x| Ok(x.stop_gradient()), Array::scalar(2.0), Array::scalar(3.0)).unwrap();
        assert_eq!(primal, Array::scalar(2.0));
        assert_eq!(tangent, Array::scalar(0.0));

        // The JAX documentation example: `f(x) = x * stop_gradient(x)` differentiates like `x * c` with `c` frozen
        // at the primal value, so `f'(x) = stop_gradient(x)`.
        let (value, first_derivative) =
            value_and_gradient(|x| x.clone() * x.stop_gradient(), Array::scalar(3.0)).unwrap();
        assert_eq!(value, Array::scalar(9.0));
        assert_eq!(first_derivative, Array::scalar(3.0));

        // A stop-gradient barrier applies to every active differentiation level. The first derivative of
        // `x * stop_gradient(x)` is the frozen primal `x`, but an enclosing derivative cannot differentiate it again.
        let second_derivative =
            gradient(|x| gradient(|y| y.clone() * y.stop_gradient(), x), Array::scalar(3.0)).unwrap();
        assert_eq!(second_derivative, Array::scalar(0.0));

        // The staged tangent program replays the primal operation and stages no tangent computation: the severed
        // tangent output materializes as a canonical zero.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(StopGradientOperation::new(), Vec::new(), vec![input]).unwrap()[0];
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
        check_operation_partial_evaluation!(
            operation = StopGradientOperation::<ArrayType>::new(),
            inputs = [Array::scalar(2.0)],
            expected = Array::scalar(2.0),
        );
    }

    #[test]
    fn test_stop_gradient_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = StopGradientOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
